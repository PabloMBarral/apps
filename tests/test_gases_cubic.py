"""Tests de core.gases.cubic (Fase 9.2): Van der Waals y Peng–Robinson con sus raíces y Maxwell."""

from __future__ import annotations

import json
import math
import pickle

import CoolProp
import numpy as np
import pytest
from scipy.integrate import quad

from core.gases import cubic as cb
from core.gases import real as rg

C = 273.15


def _psat(key: str, T: float) -> float:
    state = CoolProp.AbstractState("HEOS", key)
    state.update(CoolProp.QT_INPUTS, 0.0, T)
    return float(state.p())


# ---------------------------------------------------------------------
# Maxwell
# ---------------------------------------------------------------------


def test_van_der_waals_saturation_is_universal() -> None:
    """En variables reducidas Van der Waals no depende del fluido: p_R,sat(0,9) = 0,647 y
    ω = −0,30."""
    for key in ("Nitrogen", "Water", "R134a"):
        fl = rg.real_fluid(key)
        sat = cb.maxwell(fl, "vdw", 0.9 * fl.T_cr)
        assert sat is not None
        assert sat.p_sat_Pa / fl.p_cr == pytest.approx(0.6470, abs=2e-4)
        assert cb.implied_omega(fl, "vdw") == pytest.approx(-0.302, abs=1e-3)
    ps, vf, vg = cb._vdw_reduced_maxwell(0.9)  # type: ignore[misc]
    assert ps == pytest.approx(0.6470, abs=2e-4)
    assert (vf, vg) == pytest.approx((0.6034, 2.3488), abs=2e-4)


@pytest.mark.parametrize("model", ["vdw", "pr"])
@pytest.mark.parametrize("Tr", [0.55, 0.7, 0.85, 0.95, 0.99])
def test_equal_areas(model: str, Tr: float) -> None:
    """∫ p·dv de v_f a v_g = p_sat·(v_g − v_f): las dos áreas del lazo son iguales."""
    fl = rg.real_fluid("n-Propane")
    T = Tr * fl.T_cr
    sat = cb.maxwell(fl, model, T)  # type: ignore[arg-type]
    assert sat is not None
    area, _ = quad(lambda v: rg.cubic_p(fl, model, T, v), sat.v_f, sat.v_g, limit=200)  # type: ignore[arg-type]
    assert area == pytest.approx(sat.p_sat_Pa * (sat.v_g - sat.v_f), rel=1e-8)
    assert sat.p_spinodal_liquid_Pa < sat.p_sat_Pa < sat.p_spinodal_vapor_Pa
    assert sat.Z_f < sat.Z_g
    # las dos raíces tienen la misma energía libre de Gibbs (φ_f = φ_g)
    A, B = rg.cubic_ab(fl, model, T, sat.p_sat_Pa)  # type: ignore[arg-type]
    assert cb.ln_phi(model, sat.Z_f, A, B) == pytest.approx(  # type: ignore[arg-type]
        cb.ln_phi(model, sat.Z_g, A, B),  # type: ignore[arg-type]
        abs=1e-9,
    )


def test_no_saturation_above_the_critical_temperature() -> None:
    fl = rg.real_fluid("Methane")
    assert cb.maxwell(fl, "vdw", fl.T_cr) is None
    assert cb.maxwell(fl, "pr", 1.2 * fl.T_cr) is None


def test_peng_robinson_follows_the_vapor_pressure() -> None:
    """Peng–Robinson ajusta α(T) con ω: la presión de vapor al 1,5 % y su ω implícito al 0,004."""
    for key in ("Argon", "Methane", "Nitrogen", "n-Propane", "R134a"):
        fl = rg.real_fluid(key)
        for Tr in (0.7, 0.8, 0.9, 0.95):
            sat = cb.maxwell(fl, "pr", Tr * fl.T_cr)
            assert sat is not None
            assert sat.p_sat_Pa == pytest.approx(_psat(key, Tr * fl.T_cr), rel=0.015)
        assert cb.implied_omega(fl, "pr") == pytest.approx(fl.omega, abs=0.004)


def test_van_der_waals_overshoots_the_vapor_pressure() -> None:
    """Con ω = −0,30, Van der Waals da presiones de vapor mucho más altas que las reales."""
    for key in ("Nitrogen", "Water", "R134a"):
        fl = rg.real_fluid(key)
        sat = cb.maxwell(fl, "vdw", 0.7 * fl.T_cr)
        assert sat is not None and sat.p_sat_Pa > 1.5 * _psat(key, 0.7 * fl.T_cr)


# ---------------------------------------------------------------------
# Las raíces
# ---------------------------------------------------------------------


def test_three_roots_with_their_names() -> None:
    fl = rg.real_fluid("R134a")
    sol = cb.cubic_solution(fl, "vdw", 1e6, 50.0 + C)
    assert [r.kind for r in sol.roots] == ["liquid", "unstable", "vapor"]
    assert sol.roots[1].ln_phi is None and not sol.roots[1].stable
    assert sol.phase == "vapor" and sol.stable.kind == "vapor"
    assert sol.saturation is not None and 1e6 < sol.saturation.p_sat_Pa
    zs = [r.Z for r in sol.roots]
    assert zs == sorted(zs)
    # la cúbica con sus coeficientes se anula en cada raíz
    c2, c1, c0 = sol.coefficients
    for z in zs:
        assert z**3 + c2 * z**2 + c1 * z + c0 == pytest.approx(0.0, abs=1e-12)
    assert sol.root("vapor").v_m3_per_kg == pytest.approx(0.02329, rel=1e-3)  # type: ignore[union-attr]
    assert sol.alpha == 1.0 and sol.kappa is None


def test_peng_robinson_constants() -> None:
    fl = rg.real_fluid("Water")
    T = 400.0
    sol = cb.cubic_solution(fl, "pr", 1e5, T)
    assert sol.kappa == pytest.approx(rg.pr_kappa(fl.omega))
    assert sol.alpha == pytest.approx((1 + sol.kappa * (1 - math.sqrt(T / fl.T_cr))) ** 2)  # type: ignore[operator]
    assert sol.a == pytest.approx(sol.a_cr * sol.alpha)
    assert sol.b == pytest.approx(0.07780 * fl.R * fl.T_cr / fl.p_cr)


def test_stable_root_follows_the_model_saturation() -> None:
    """Sobre la p_sat del modelo la estable es la del líquido; debajo, la del vapor."""
    fl = rg.real_fluid("n-Propane")
    T = 0.8 * fl.T_cr
    for model in ("vdw", "pr"):
        sat = cb.maxwell(fl, model, T)  # type: ignore[arg-type]
        assert sat is not None
        above = cb.cubic_solution(fl, model, 1.01 * sat.p_sat_Pa, T)  # type: ignore[arg-type]
        below = cb.cubic_solution(fl, model, 0.99 * sat.p_sat_Pa, T)  # type: ignore[arg-type]
        assert above.phase == "liquid" and below.phase == "vapor"


def test_single_root_kinds() -> None:
    fl = rg.real_fluid("Nitrogen")
    sup = cb.cubic_solution(fl, "vdw", 100e5, 175.0)
    assert [r.kind for r in sup.roots] == ["supercritical"] and sup.phase == "supercritical"
    # bajo T_cr con una sola raíz: vapor a p baja, líquido a p alta
    T = 0.8 * fl.T_cr
    assert cb.cubic_solution(fl, "pr", 1e3, T).roots[-1].kind == "vapor"
    liquid = cb.cubic_solution(fl, "pr", 3.0 * fl.p_cr, T)
    assert len(liquid.roots) == 1 and liquid.roots[0].kind == "liquid"


# ---------------------------------------------------------------------
# solve_cubic
# ---------------------------------------------------------------------


def test_refrigerant_134a_matches_the_compressibility_mode() -> None:
    r = cb.solve_cubic(cb.CubicInputs("R134a", 1e6, 50.0 + C))
    z = rg.compressibility(rg.CompressibilityInputs("R134a", "pT", 1e6, 50.0 + C))
    assert r.phase == "vapor"
    assert r.v_m3_per_kg == pytest.approx(z.v_m3_per_kg)
    for model in ("vdw", "pr"):
        assert r.model_v(model) == pytest.approx(z.model(model).v_m3_per_kg)  # type: ignore[arg-type]
    assert r.omega_vdw == pytest.approx(-0.302, abs=1e-3)
    assert r.omega_pr == pytest.approx(r.fluid.omega, abs=0.004)


def test_liquid_that_van_der_waals_calls_vapor() -> None:
    """R-134a a 10 bar y 20 °C es líquido; para Van der Waals (p_sat = 14 bar) es vapor."""
    r = cb.solve_cubic(cb.CubicInputs("R134a", 10e5, 20.0 + C))
    assert r.phase == "liquid" and r.p_sat_Pa == pytest.approx(5.717e5, rel=1e-3)
    assert r.vdw.phase == "vapor" and r.pr.phase == "liquid"
    assert r.model_v("vdw") == pytest.approx(r.vdw.root("liquid").v_m3_per_kg)  # type: ignore[union-attr]
    assert any("Para Van der Waals este estado es vapor" in n for n in r.notes)
    assert any("Presión de saturación a esta T" in n for n in r.notes)


def test_water_vapor_pressure_note() -> None:
    r = cb.solve_cubic(cb.CubicInputs("Water", 1e5, 20.0 + C))
    assert any("130 veces la real" in n for n in r.notes)
    assert any("el agua tiene 0,344" in n for n in r.notes)


def test_supercritical_and_model_without_the_phase() -> None:
    r = cb.solve_cubic(cb.CubicInputs("Nitrogen", 100e5, 175.0))
    assert r.phase == "supercritical" and r.p_sat_Pa is None
    assert r.model_v("vdw") is not None
    assert any("una sola raíz" in n for n in r.notes)
    # líquido a T_R = 0,95 apenas sobre p_sat: Van der Waals no tiene raíz de líquido
    fl = rg.real_fluid("R134a")
    T = 0.95 * fl.T_cr
    r = cb.solve_cubic(cb.CubicInputs("R134a", 1.02 * _psat("R134a", T), T))
    assert r.phase == "liquid" and r.model_v("vdw") is None
    assert r.model_v("pr") is not None


@pytest.mark.parametrize(
    "p,T,match",
    [
        (-1e5, 300.0, "positiva"),
        (1e5, -3.0, "positiva"),
        (1e5, 3000.0, "fuera del rango"),
        (5e9, 300.0, "supera la máxima"),
    ],
)
def test_errors(p: float, T: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cb.solve_cubic(cb.CubicInputs("Water", p, T))


def test_saturation_pressure_exactly() -> None:
    with pytest.raises(ValueError, match="justo la de saturación"):
        cb.solve_cubic(cb.CubicInputs("Water", _psat("Water", 373.15), 373.15))


# ---------------------------------------------------------------------
# Gráficos, export
# ---------------------------------------------------------------------


def test_reduced_isotherms() -> None:
    out = cb.vdw_reduced_isotherms()
    vr, pr = out["isotherms"][1.0]
    assert cb.vdw_reduced_p(1.0, 1.0) == pytest.approx(1.0)  # el punto crítico
    k = int(np.argmin(np.abs(np.array(vr) - 1.0)))
    assert pr[k] == pytest.approx(1.0, abs=2e-3)
    assert set(out["maxwell"]) == {0.8, 0.85, 0.9, 0.95}
    dv, dp = out["dome"]
    assert max(dp) == pytest.approx(1.0) and dv[dp.index(max(dp))] == pytest.approx(1.0)
    # la campana: el líquido a la izquierda de v_R = 1 y el vapor a la derecha
    assert dv[0] < 1.0 < dv[-1]


def test_fluid_isotherms_have_the_flat_part() -> None:
    T = 50.0 + C
    out = cb.fluid_isotherms("R134a", T)
    p_sat = out["p_sat"]
    state = CoolProp.AbstractState("HEOS", "R134a")
    state.update(CoolProp.QT_INPUTS, 0.0, T)
    v_f = 1.0 / state.rhomass()
    state.update(CoolProp.QT_INPUTS, 1.0, T)
    v_g = 1.0 / state.rhomass()
    inside = [p for v, p in zip(out["v"], out["p"]["coolprop"], strict=True) if v_f < v < v_g]
    assert inside and all(p == pytest.approx(p_sat, rel=1e-6) for p in inside)
    # el lazo de Peng–Robinson baja a presiones negativas: el líquido «estirado»
    assert min(p for p in out["p"]["pr"] if p is not None) < 0.0
    assert out["maxwell"]["vdw"].p_sat_Pa > out["maxwell"]["pr"].p_sat_Pa
    assert cb.fluid_isotherms("Nitrogen", 300.0)["maxwell"] == {"vdw": None, "pr": None}


def test_saturation_curves() -> None:
    """El agua es polar: Peng–Robinson se aparta 11 % a T_R = 0,5 y menos de 4 % desde 0,6."""
    out = cb.saturation_curves("Water")
    real = out["p_sat"]["coolprop"]
    assert real == sorted(real)
    T_cr = out["T_cr"]
    for T, vdw, pr, r in zip(out["T"], out["p_sat"]["vdw"], out["p_sat"]["pr"], real, strict=True):
        assert vdw is not None and vdw > r  # Van der Waals por arriba
        assert pr is not None and abs(pr / r - 1.0) < (0.04 if T >= 0.6 * T_cr else 0.12)


def test_examples_and_export() -> None:
    for name, ex in cb.CUBIC_EXAMPLES.items():
        r = cb.solve_cubic(ex.inputs)
        d = cb.cubic_to_dict(r)
        json.dumps(d)
        assert d["fase_real"] == r.phase, name
        assert len(d["van_der_waals"]["raices"]) == len(r.vdw.roots)
        assert pickle.loads(pickle.dumps(r)) == r
