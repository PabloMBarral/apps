"""Tests de core.gases.ideal (Fase 9.1): el gas ideal entre dos estados."""

from __future__ import annotations

import json
import math
import pickle

import pytest
from scipy.optimize import brentq

from core.gases import ideal as ig

C = 273.15

# Tabla del vademecum §4.6 (CoolProp a 25 °C): M, R, c_p y c_v
VADEMECUM_TABLE = {
    "air": (28.97, 0.287, 1.005, 0.718),
    "Ar": (39.95, 0.208, 0.520, 0.312),
    "CO2": (44.01, 0.189, 0.844, 0.655),
    "He": (4.00, 2.077, 5.193, 3.116),
    "H2": (2.02, 4.124, 14.303, 10.179),
    "CH4": (16.04, 0.518, 2.226, 1.708),
    "N2": (28.01, 0.297, 1.040, 0.743),
    "O2": (32.00, 0.260, 0.918, 0.658),
    "H2O": (18.02, 0.462, 1.864, 1.403),
}


def _change(gas: str, s1: tuple, s2: tuple) -> ig.StateChange:
    return ig.solve_state_change(ig.StateChangeInputs(gas, ig.StatePair(*s1), ig.StatePair(*s2)))


# ---------------------------------------------------------------------
# El gas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("key", ig.VADEMECUM_GASES)
def test_constants_match_the_vademecum_table(key: str) -> None:
    g = ig.gas(key)
    M, R, cp, cv = VADEMECUM_TABLE[key]
    assert g.M * 1000 == pytest.approx(M, abs=0.01)
    assert g.R / 1000 == pytest.approx(R, abs=6e-4)
    assert g.cp_ref / 1000 == pytest.approx(cp, rel=1e-3)
    assert g.cv(ig.T_REF_K) / 1000 == pytest.approx(cv, rel=1.5e-3)


def test_every_gas_is_consistent() -> None:
    for g in ig.IDEAL_GASES.values():
        T = max(g.T_min_K, 300.0)
        assert g.cv(T) == pytest.approx(g.cp(T) - g.R)  # Mayer
        assert g.k(T) == pytest.approx(g.cp(T) / g.cv(T))
        assert g.h(ig.T_REF_K) == pytest.approx(0.0, abs=1e-6)
        assert g.u(ig.T_REF_K) == pytest.approx(0.0, abs=1e-6)
        assert g.s0(ig.T_REF_K) == pytest.approx(0.0, abs=1e-9)
        assert g.pr(ig.T_REF_K) == pytest.approx(1.0)
        assert g.vr(ig.T_REF_K) == pytest.approx(ig.T_REF_K)


@pytest.mark.parametrize("key", ["air", "CO2", "H2O", "H2", "CH4"])
@pytest.mark.parametrize("T", [350.0, 900.0, 1800.0])
def test_cp_is_the_derivative_of_h_and_s0(key: str, T: float) -> None:
    g = ig.gas(key)
    dT = 0.01
    assert (g.h(T + dT) - g.h(T - dT)) / (2 * dT) == pytest.approx(g.cp(T), rel=1e-6)
    assert (g.s0(T + dT) - g.s0(T - dT)) / (2 * dT) == pytest.approx(g.cp(T) / T, rel=1e-6)
    assert (g.u(T + dT) - g.u(T - dT)) / (2 * dT) == pytest.approx(g.cv(T), rel=1e-6)


def test_monatomic_gases() -> None:
    he, ar = ig.gas("He"), ig.gas("Ar")
    for T in (250.0, 1000.0, 4000.0):
        assert he.cp(T) == pytest.approx(2.5 * he.R, rel=1e-14)
        assert he.k(T) == pytest.approx(5.0 / 3.0)
        assert ar.cp(T) == pytest.approx(2.5 * ar.R, rel=1e-4)  # NASA


def test_relative_pressure_gives_the_isentropic_temperature() -> None:
    """Motor de auto (Çengel §7-9): de 22 °C con v₁/v₂ = 8, isoentrópica → 662,7 K."""
    air = ig.gas("air")
    T1 = 22.0 + C
    T2 = brentq(lambda T: air.vr(T) - air.vr(T1) / 8.0, 300.0, 1500.0)
    assert T2 == pytest.approx(662.7, abs=0.5)
    # y con presiones: p_r2/p_r1 = p₂/p₁
    p_ratio = 8.0 * T2 / T1
    assert air.pr(T2) / air.pr(T1) == pytest.approx(p_ratio, rel=1e-9)


def test_unknown_gas_and_temperature_range() -> None:
    with pytest.raises(ValueError, match="no está en la lista"):
        ig.gas("xenón")
    with pytest.raises(ValueError, match="fuera del rango de los polinomios NASA del propano"):
        ig.gas_state(ig.gas("C3H8"), p_Pa=1e5, T_K=250.0)
    with pytest.raises(ValueError, match="de 200 K"):
        ig.gas_state(ig.gas("air"), p_Pa=1e5, T_K=6500.0)


# ---------------------------------------------------------------------
# Estados
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "pair",
    [
        ig.StatePair("pT", 2e5, 400.0),
        ig.StatePair("pv", 2e5, 0.574),
        ig.StatePair("Tv", 400.0, 0.574),
    ],
)
def test_states_satisfy_pv_RT(pair: ig.StatePair) -> None:
    air = ig.gas("air")
    st = ig.state_from_pair(air, pair)
    assert st.p_Pa * st.v_m3_per_kg == pytest.approx(air.R * st.T_K, rel=1e-12)


def test_state_messages() -> None:
    air = ig.gas("air")
    with pytest.raises(ValueError, match="exactamente dos"):
        ig.gas_state(air, p_Pa=1e5, label="1")
    with pytest.raises(ValueError, match="exactamente dos"):
        ig.gas_state(air, p_Pa=1e5, T_K=300.0, v_m3_per_kg=0.8)
    with pytest.raises(ValueError, match="La presión del estado 2 tiene que ser positiva"):
        ig.gas_state(air, p_Pa=-1.0, T_K=300.0, label="2")
    with pytest.raises(ValueError, match="El volumen específico"):
        ig.gas_state(air, p_Pa=1e5, v_m3_per_kg=0.0)


def test_ideal_gas_check() -> None:
    z, phase = ig.ideal_gas_check(ig.gas("air"), 101_325.0, 300.0)
    assert z == pytest.approx(1.0, abs=1e-3)
    z, phase = ig.ideal_gas_check(ig.gas("H2O"), 101_325.0, 298.15)
    assert phase == "liquid"
    z, reason = ig.ideal_gas_check(ig.gas("CO2"), 101_325.0, 210.0)
    assert z is None and reason


# ---------------------------------------------------------------------
# Entre dos estados
# ---------------------------------------------------------------------


def test_air_compression_of_cengel() -> None:
    """Çengel §7-9: aire de 100 kPa y 17 °C a 600 kPa y 57 °C, Δs = −0,3842 kJ/(kg·K)."""
    r = _change("air", ("pT", 100e3, 17.0 + C), ("pT", 600e3, 57.0 + C))
    assert r.variable.ds / 1000 == pytest.approx(-0.3842, abs=5e-4)
    assert r.mean.ds / 1000 == pytest.approx(-0.3842, abs=5e-4)
    assert r.mean.cp / 1000 == pytest.approx(1.006, abs=1e-3)


def test_the_three_models() -> None:
    r = _change("air", ("pT", 101_325.0, 300.0), ("pT", 101_325.0, 1500.0))
    g = r.gas
    # constante a 25 °C
    assert r.constant.cp == pytest.approx(g.cp_ref)
    assert r.constant.dh == pytest.approx(g.cp_ref * 1200.0)
    assert r.constant.du == pytest.approx((g.cp_ref - g.R) * 1200.0)
    # a la temperatura media
    assert r.T_mean_K == pytest.approx(900.0)
    assert r.mean.cp == pytest.approx(g.cp(900.0))
    # variable: integrales exactas, Δu = Δh − R·ΔT
    assert r.variable.dh == pytest.approx(g.h(1500.0) - g.h(300.0))
    assert r.variable.du == pytest.approx(r.variable.dh - g.R * 1200.0)
    assert r.variable.ds == pytest.approx(g.s0(1500.0) - g.s0(300.0))
    # c_p constante subestima: el c_p del aire sube con T
    assert r.error("constant", "dh") == pytest.approx(-0.0975, abs=0.005)
    assert abs(r.error("mean", "dh")) < 0.01
    assert "9,8 % más bajo" in r.notes[0]


def test_isothermal_change() -> None:
    r = _change("N2", ("pT", 1e5, 400.0), ("pT", 5e5, 400.0))
    for c in r.changes:
        assert c.du == pytest.approx(0.0, abs=1e-9) and c.dh == pytest.approx(0.0, abs=1e-9)
        assert c.ds == pytest.approx(-r.gas.R * math.log(5.0))
    assert r.error("constant", "dh") is None
    assert r.error("constant", "ds") == pytest.approx(0.0, abs=1e-12)
    assert any("T₁ = T₂" in n for n in r.notes)


def test_helium_models_agree() -> None:
    r = _change("He", ("pT", 1e5, 300.0), ("pT", 1e5, 1000.0))
    assert r.constant.dh == pytest.approx(r.variable.dh)
    assert r.mean.ds == pytest.approx(r.variable.ds)
    assert "monoatómico" in r.notes[0]


def test_ideal_gas_notes() -> None:
    liquid = _change("H2O", ("pT", 101_325.0, 298.15), ("pT", 101_325.0, 400.0))
    assert any("es líquido" in n for n in liquid.notes)
    co2 = _change("CO2", ("pT", 6e6, 310.0), ("pT", 6e6, 400.0))
    assert co2.Z1 is not None and co2.Z1 < 0.95
    assert any("se aparta" in n and "Z = p·v/(R·T) = 0," in n for n in co2.notes)
    fine = _change("air", ("pT", 1e5, 300.0), ("pT", 2e5, 350.0))
    assert not any("se aparta" in n for n in fine.notes)


@pytest.mark.parametrize("name", list(ig.STATE_EXAMPLES))
def test_examples(name: str) -> None:
    r = ig.solve_state_change(ig.STATE_EXAMPLES[name].inputs)
    for c in r.changes:
        assert math.isfinite(c.dh) and math.isfinite(c.ds)
    data = ig.state_change_to_dict(r)
    json.dumps(data)
    assert data["variable"]["dh_J_per_kg"] == pytest.approx(r.variable.dh)
    assert pickle.loads(pickle.dumps(r)) == r
