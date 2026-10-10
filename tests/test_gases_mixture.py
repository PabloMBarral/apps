"""Tests de core.gases.mixture (Fase 9.1): mezclas de gases ideales y mezcla adiabática."""

from __future__ import annotations

import json
import math
import pickle

import pytest

from core.gases import ideal as ig
from core.gases import mixture as mx

C = 273.15
ATM = 101_325.0


def _mix(components: list[tuple[str, float]], basis: str, T: float = 300.0, p: float = ATM):
    return mx.solve_mixture(
        mx.MixtureInputs(tuple(mx.Component(k, a) for k, a in components), basis, T, p)
    )


# ---------------------------------------------------------------------
# Composición
# ---------------------------------------------------------------------


def test_cengel_composition() -> None:
    """Çengel §13-1: 3 kg de O₂, 5 kg de N₂ y 12 kg de CH₄."""
    r = _mix([("O2", 3.0), ("N2", 5.0), ("CH4", 12.0)], "mass")
    xs = [c.x for c in r.components]
    ys = [c.y for c in r.components]
    assert xs == pytest.approx([0.15, 0.25, 0.60])
    assert ys == pytest.approx([0.092, 0.175, 0.733], abs=6e-4)
    assert r.M * 1000 == pytest.approx(19.6, abs=0.05)
    assert r.R / 1000 == pytest.approx(0.424, abs=1e-3)
    assert r.m_kg == pytest.approx(20.0)


def test_air_from_its_composition() -> None:
    r = mx.solve_mixture(mx.COMPOSITION_EXAMPLES["El aire seco (fracciones molares)"].inputs)
    air = ig.gas("air")
    assert r.M == pytest.approx(air.M)
    assert r.cp == pytest.approx(air.cp(r.inputs.T_K))
    assert r.k == pytest.approx(1.400, abs=1e-3)


def test_bases_are_equivalent() -> None:
    by_mass = _mix([("CO2", 4.4), ("N2", 2.8)], "mass", 500.0, 2e5)
    moles = [4.4 / ig.gas("CO2").M, 2.8 / ig.gas("N2").M]
    by_moles = _mix([("CO2", moles[0]), ("N2", moles[1])], "moles", 500.0, 2e5)
    by_x = _mix([("CO2", 4.4 / 7.2), ("N2", 2.8 / 7.2)], "mass_fraction", 500.0, 2e5)
    ys = [n / sum(moles) for n in moles]
    by_y = _mix([("CO2", ys[0]), ("N2", ys[1])], "mole_fraction", 500.0, 2e5)
    for other in (by_moles, by_x, by_y):
        assert other.M == pytest.approx(by_mass.M)
        assert other.cp == pytest.approx(by_mass.cp)
        assert other.s == pytest.approx(by_mass.s)
        assert [c.y for c in other.components] == pytest.approx([c.y for c in by_mass.components])
    assert by_x.m_kg is None and by_x.V_m3 is None


def test_dalton_amagat_and_properties() -> None:
    r = _mix([("O2", 3.0), ("N2", 5.0), ("CH4", 12.0)], "mass", 350.0, 3e5)
    T, p = 350.0, 3e5
    assert sum(c.p_i for c in r.components) == pytest.approx(p)
    assert r.V_m3 is not None and sum(c.V_i for c in r.components) == pytest.approx(r.V_m3)
    for c in r.components:
        # p_i·V = N_i·R_u·T y p·V_i = N_i·R_u·T (vademecum §5.3 y §5.4)
        assert c.p_i * r.V_m3 == pytest.approx(c.n_mol * ig.R_U * T)
        assert p * c.V_i == pytest.approx(c.n_mol * ig.R_U * T)
    assert r.R == pytest.approx(sum(c.x * c.gas.R for c in r.components))
    assert r.cp == pytest.approx(sum(c.x * c.cp for c in r.components))
    assert r.cv == pytest.approx(r.cp - r.R)
    assert r.v_m3_per_kg == pytest.approx(r.V_m3 / r.m_kg)


def test_entropy_of_the_mixture_follows_the_vademecum() -> None:
    """s_M = Σ x_i·[s_i(T, p) − R_i·ln y_i] (vademecum §5.7)."""
    r = _mix([("CO2", 1.0), ("H2O", 1.0), ("N2", 6.0)], "moles", 800.0, ATM)
    T, p = 800.0, ATM
    expected = sum(
        c.x * (c.gas.s0(T) - c.gas.R * math.log(p / mx.P_REF_PA) - c.gas.R * math.log(c.y))
        for c in r.components
    )
    assert r.s == pytest.approx(expected)
    assert r.s_mixing == pytest.approx(-sum(c.x * c.gas.R * math.log(c.y) for c in r.components))
    assert r.s_mixing > 0.0


def test_water_that_would_condense() -> None:
    hot = mx.solve_mixture(
        mx.COMPOSITION_EXAMPLES["Humos del metano con 10 % de exceso de aire, a 1000 K"].inputs
    )
    assert not any("condensa" in n for n in hot.notes)
    flue = [("CO2", 0.087), ("H2O", 0.174), ("O2", 0.017), ("N2", 0.722)]
    cold = _mix(flue, "mole_fraction", 313.15)
    assert any("condensa" in n and "supera la de saturación" in n for n in cold.notes)


def test_mixture_messages() -> None:
    with pytest.raises(ValueError, match="al menos dos gases"):
        _mix([("N2", 1.0)], "mass")
    with pytest.raises(ValueError, match="una sola vez"):
        _mix([("N2", 1.0), ("N2", 2.0)], "mass")
    with pytest.raises(ValueError, match="no puede ser negativa"):
        _mix([("N2", 1.0), ("O2", -2.0)], "mass")
    with pytest.raises(ValueError, match="suman 0,9000: tienen que sumar 1"):
        _mix([("N2", 0.7), ("O2", 0.2)], "mole_fraction")
    with pytest.raises(ValueError, match="no tiene nada"):
        _mix([("N2", 0.0), ("O2", 0.0)], "mass")
    with pytest.raises(ValueError, match="presión de la mezcla"):
        _mix([("N2", 1.0), ("O2", 1.0)], "mass", 300.0, 0.0)
    with pytest.raises(ValueError, match="polinomios NASA del propano"):
        _mix([("CH4", 0.9), ("C3H8", 0.1)], "mole_fraction", 288.15)


# ---------------------------------------------------------------------
# Mezcla adiabática
# ---------------------------------------------------------------------


def _solve(name: str) -> mx.MixingResult:
    return mx.solve_mixing(mx.MIXING_EXAMPLES[name].inputs)


def test_tank_of_cengel() -> None:
    r = _solve(next(n for n in mx.MIXING_EXAMPLES if n.startswith("Tanque")))
    assert r.T_K - C == pytest.approx(32.2, abs=0.05)
    assert r.p_Pa / 1e3 == pytest.approx(114.5, abs=0.1)
    assert r.V_m3 == pytest.approx(sum(s.V_m3 for s in r.streams))
    assert r.S_gen > 0.0


def test_mixing_entropy_of_cengel() -> None:
    r = _solve("O₂ y CO₂ a 25 °C y 200 kPa: 3 kmol y 5 kmol (Çengel §13-3)")
    assert r.S_gen / 1e3 == pytest.approx(44.0, abs=0.05)
    assert r.X_dest / 1e3 == pytest.approx(13_100, rel=2e-3)
    assert r.S_gen_TP == pytest.approx(0.0, abs=1e-9)
    assert r.S_gen_mix == pytest.approx(-ig.R_U * (3000 * math.log(3 / 8) + 5000 * math.log(5 / 8)))


def test_same_gas_has_no_mixing_entropy() -> None:
    r = _solve("Cámara de mezcla: aire a 20 °C y a 80 °C")
    assert r.S_gen_mix == pytest.approx(0.0, abs=1e-12)
    assert r.S_gen > 0.0
    assert 20.0 + C < r.T_K < 80.0 + C
    assert "mismo gas" in r.notes[0]


@pytest.mark.parametrize("model", ["constant", "variable"])
@pytest.mark.parametrize("kind", ["tank", "flow"])
def test_energy_balances_close(model: str, kind: str) -> None:
    streams = (
        mx.MixingStream("CH4", 1.0, 300.0, 150e3),
        mx.MixingStream("air", 15.0, 700.0, 120e3),
        mx.MixingStream("CO2", 2.0, 450.0, 130e3),
    )
    r = mx.solve_mixing(mx.MixingInputs(kind, streams, model=model, p_out_Pa=110e3))  # type: ignore[arg-type]
    if kind == "tank":
        before = sum(s.amount * mx._u(ig.gas(s.gas_key), s.T_K, model) for s in streams)  # type: ignore[arg-type]
        after = sum(s.amount * mx._u(ig.gas(s.gas_key), r.T_K, model) for s in streams)  # type: ignore[arg-type]
    else:
        before = sum(s.amount * mx._h(ig.gas(s.gas_key), s.T_K, model) for s in streams)  # type: ignore[arg-type]
        after = sum(s.amount * mx._h(ig.gas(s.gas_key), r.T_K, model) for s in streams)  # type: ignore[arg-type]
    assert after == pytest.approx(before, rel=1e-9, abs=1e-6)
    assert r.S_gen > 0.0
    assert r.X_dest == pytest.approx(r.inputs.T0_K * r.S_gen)
    assert r.S_gen == pytest.approx(sum(s.amount * (s.ds_TP + s.ds_mix) for s in r.streams))


def test_mixing_messages() -> None:
    one = (mx.MixingStream("N2", 1.0, 300.0, 1e5),)
    with pytest.raises(ValueError, match="al menos dos corrientes"):
        mx.solve_mixing(mx.MixingInputs("tank", one))
    two = (mx.MixingStream("N2", 1.0, 300.0, 1e5), mx.MixingStream("O2", 1.0, 350.0, 2e5))
    with pytest.raises(ValueError, match="Falta la presión de salida"):
        mx.solve_mixing(mx.MixingInputs("flow", two))
    with pytest.raises(ValueError, match="no comprime"):
        mx.solve_mixing(mx.MixingInputs("flow", two, p_out_Pa=1.5e5))
    bad = (mx.MixingStream("N2", 0.0, 300.0, 1e5), mx.MixingStream("O2", 1.0, 350.0, 2e5))
    with pytest.raises(ValueError, match="corriente 1 \\(kg/s\\) tiene que ser positiva"):
        mx.solve_mixing(mx.MixingInputs("flow", bad, p_out_Pa=1e5))
    with pytest.raises(ValueError, match="T₀"):
        mx.solve_mixing(mx.MixingInputs("tank", two, T0_K=100.0))


@pytest.mark.parametrize("name", list(mx.MIXING_EXAMPLES))
def test_mixing_examples(name: str) -> None:
    r = _solve(name)
    data = mx.mixing_to_dict(r)
    json.dumps(data)
    assert data["S_gen"] == pytest.approx(r.S_gen)
    assert pickle.loads(pickle.dumps(r)) == r


@pytest.mark.parametrize("name", list(mx.COMPOSITION_EXAMPLES))
def test_composition_examples(name: str) -> None:
    r = mx.solve_mixture(mx.COMPOSITION_EXAMPLES[name].inputs)
    assert sum(c.x for c in r.components) == pytest.approx(1.0)
    assert sum(c.y for c in r.components) == pytest.approx(1.0)
    json.dumps(mx.mixture_to_dict(r))
    assert pickle.loads(pickle.dumps(r)) == r
