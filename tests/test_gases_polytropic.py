"""Tests de core.gases.polytropic (Fase 9.1): transformaciones politrópicas."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import replace

import numpy as np
import pytest

from core.gases import ideal as ig
from core.gases import polytropic as pt

C = 273.15
PSI = 6894.757293168361

KINDS_AND_FINALS = [(kind, final) for kind in pt.PROCESS_KINDS for final in pt.allowed_finals(kind)]


def _example(prefix: str) -> pt.PolytropicInputs:
    return next(e.inputs for n, e in pt.PROCESS_EXAMPLES.items() if n.startswith(prefix))


def _value(kind: str, final: str) -> float:
    """Un dato final razonable para comprimir o calentar aire desde 100 kPa y 300 K."""
    return {"p2": 500e3, "ratio": 4.0 if kind != "isobaric" else 0.5, "T2": 520.0}[final]


# ---------------------------------------------------------------------
# Los ejemplos de Çengel
# ---------------------------------------------------------------------


def test_compressors_of_cengel() -> None:
    """Çengel §7-12: aire de 100 kPa y 300 K a 900 kPa."""
    inputs = _example("Compresor de aire")
    rows = {row.process: row for row in pt.process_comparison(inputs)}
    assert rows["polytropic"].result.w_f / 1e3 == pytest.approx(-246.4, abs=0.1)
    assert rows["adiabatic"].result.w_f / 1e3 == pytest.approx(-263.2, abs=0.1)
    assert rows["isothermal"].result.w_f / 1e3 == pytest.approx(-189.2, abs=0.1)
    assert rows["isobaric"].result is None and "p₂ = p₁" in rows["isobaric"].reason
    assert rows["polytropic"].is_data
    staged = pt.staged_compression(pt.StagedInputs("air", 100e3, 300.0, 900e3, 1.3, 2))
    assert staged.w_f / 1e3 == pytest.approx(-215.3, abs=0.1)
    assert staged.pressures_Pa[1] == pytest.approx(math.sqrt(100e3 * 900e3))


def test_car_engine_of_cengel() -> None:
    """Çengel §7-9: aire de 95 kPa y 22 °C, adiabática con v₁/v₂ = 8."""
    r = pt.solve_process(_example("Motor de auto"))
    assert r.state2.T_K == pytest.approx(662.7, abs=0.5)
    assert r.other_T2_K == pytest.approx(295.15 * 8**0.4, rel=2e-4)  # k = 1,400 a 25 °C
    assert r.ds == pytest.approx(0.0, abs=1e-9)
    assert r.q == 0.0


def test_helium_of_cengel() -> None:
    r = pt.solve_process(_example("Helio comprimido"))
    assert r.state2.p_Pa / PSI == pytest.approx(40.5, abs=0.05)
    assert r.n == pytest.approx(5.0 / 3.0)


# ---------------------------------------------------------------------
# Principios
# ---------------------------------------------------------------------


@pytest.mark.parametrize("model", ["constant", "variable"])
@pytest.mark.parametrize(("kind", "final"), KINDS_AND_FINALS)
def test_first_law_and_work_integrals(kind: str, final: str, model: str) -> None:
    inputs = pt.PolytropicInputs(
        "air",
        100e3,
        300.0,
        kind,
        final,
        _value(kind, final),
        n=1.25,
        model=model,  # type: ignore[arg-type]
    )
    r = pt.solve_process(inputs)
    # primer principio: q − w = Δu (cerrado) y q − w_f = Δh (abierto)
    assert r.q - r.w == pytest.approx(r.du, abs=1e-6 * max(1.0, abs(r.du)))
    assert r.q - r.w_f == pytest.approx(r.dh, abs=1e-6 * max(1.0, abs(r.dh)))
    # w = ∫p dv y w_f = −∫v dp a lo largo del camino
    curve = pt.process_curve(r, points=4001)
    v, p = np.array(curve["v"]), np.array(curve["p"])
    assert np.trapezoid(p, v) == pytest.approx(r.w, rel=1e-5, abs=1e-3)
    assert -np.trapezoid(v, p) == pytest.approx(r.w_f, rel=1e-5, abs=1e-3)
    # el camino arranca y termina en los estados
    assert curve["p"][0] == pytest.approx(r.state1.p_Pa)
    assert curve["T"][-1] == pytest.approx(r.state2.T_K, rel=1e-9)
    assert curve["s"][-1] - curve["s"][0] == pytest.approx(r.ds, abs=1e-9)


@pytest.mark.parametrize(("kind", "final"), KINDS_AND_FINALS)
def test_reversible_entropy_change_is_the_heat_over_T(kind: str, final: str) -> None:
    """Con c_p constante, Δs = ∫δq/T = c·ln(T₂/T₁) (q = c·dT), salvo en la isoterma (q/T)."""
    r = pt.solve_process(
        pt.PolytropicInputs("N2", 100e3, 300.0, kind, final, _value(kind, final), n=1.25)  # type: ignore[arg-type]
    )
    if kind == "isothermal":
        assert r.ds == pytest.approx(r.q / r.state1.T_K)
    else:
        assert r.c is not None
        assert r.ds == pytest.approx(r.c * math.log(r.state2.T_K / r.state1.T_K), abs=1e-9)


def test_specific_heat_of_the_polytropic() -> None:
    """c = c_v·(n − k)/(n − 1) (vademecum §6.5)."""
    r = pt.solve_process(_example("Nitrógeno en un cilindro"))
    g = r.gas
    cv, k = g.cv(ig.T_REF_K), g.k(ig.T_REF_K)
    assert r.c == pytest.approx(cv * (1.3 - k) / 0.3)
    assert r.q == pytest.approx(r.c * (r.state2.T_K - r.state1.T_K))
    assert r.w_f == pytest.approx(1.3 * r.w)
    assert r.W_total == pytest.approx(0.5 * r.w)


def test_polytropic_with_n_equal_to_k_is_the_adiabatic() -> None:
    k = ig.gas("air").k(ig.T_REF_K)
    adiabatic = pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "adiabatic", "p2", 8e5))
    poly = pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "polytropic", "p2", 8e5, n=k))
    assert poly.state2.T_K == pytest.approx(adiabatic.state2.T_K)
    assert poly.q == pytest.approx(0.0, abs=1e-6)
    assert poly.w_f == pytest.approx(adiabatic.w_f)


def test_isochoric_isobaric_isothermal() -> None:
    iso_v = pt.solve_process(_example("Aire calentado en un tanque"))
    assert iso_v.state2.v_m3_per_kg == pytest.approx(iso_v.state1.v_m3_per_kg)
    assert iso_v.w == 0.0 and math.isinf(iso_v.n)
    assert iso_v.q == pytest.approx(iso_v.du)
    iso_p = pt.solve_process(_example("CO₂ calentado"))
    assert iso_p.state2.p_Pa == pytest.approx(iso_p.state1.p_Pa)
    assert iso_p.state2.v_m3_per_kg == pytest.approx(2 * iso_p.state1.v_m3_per_kg)
    assert iso_p.w_f == 0.0 and iso_p.q == pytest.approx(iso_p.dh)
    iso_T = pt.solve_process(_example("Expansión isotérmica"))
    g = iso_T.gas
    assert iso_T.w == pytest.approx(g.R * 400.0 * math.log(10.0))
    assert iso_T.du == pytest.approx(0.0, abs=1e-9) and iso_T.q == pytest.approx(iso_T.w)


def test_variable_model_changes_the_heat_not_the_work() -> None:
    base = _example("Aire calentado en un tanque")
    constant = pt.solve_process(base)
    variable = pt.solve_process(replace(base, model="variable"))
    assert variable.state2.T_K == pytest.approx(constant.state2.T_K)
    assert variable.w_f == pytest.approx(constant.w_f)
    assert variable.q != pytest.approx(constant.q)
    assert any("cambia el calor" in n for n in constant.notes)


# ---------------------------------------------------------------------
# Etapas y n de dos estados
# ---------------------------------------------------------------------


def test_staged_compression() -> None:
    one = pt.staged_compression(pt.StagedInputs("air", 100e3, 300.0, 900e3, 1.3, 1))
    assert one.w_f == pytest.approx(one.w_f_single)
    assert one.saving == pytest.approx(0.0, abs=1e-12)
    assert one.q_intercoolers == 0.0
    three = pt.staged_compression(pt.StagedInputs("air", 100e3, 300.0, 900e3, 1.3, 3))
    assert three.stage_ratio == pytest.approx(9.0 ** (1 / 3))
    assert three.pressures_Pa[-1] == pytest.approx(900e3)
    assert three.w_f > pt.staged_compression(pt.StagedInputs("air", 1e5, 300.0, 9e5, 1.3, 2)).w_f
    assert three.q_intercoolers < 0.0  # el interenfriador saca calor
    isothermal = pt.staged_compression(pt.StagedInputs("air", 100e3, 300.0, 900e3, 1.0, 2))
    assert isothermal.saving == pytest.approx(0.0, abs=1e-12)  # con n = 1 no hay ahorro


def test_exponent_from_states_recovers_n() -> None:
    r = pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "polytropic", "p2", 6e5, n=1.27))
    s1, s2 = r.state1, r.state2
    by_v = pt.exponent_from_states("air", s1.p_Pa, s2.p_Pa, v1=s1.v_m3_per_kg, v2=s2.v_m3_per_kg)
    by_T = pt.exponent_from_states("air", s1.p_Pa, s2.p_Pa, T1_K=s1.T_K, T2_K=s2.T_K)
    assert by_v.n == pytest.approx(1.27) and by_T.n == pytest.approx(1.27)
    assert "cede calor" in by_T.text
    test = pt.exponent_from_states("air", **pt.EXPONENT_EXAMPLE)
    assert test.n == pytest.approx(1.3646, abs=1e-4)


# ---------------------------------------------------------------------
# Mensajes
# ---------------------------------------------------------------------


def test_process_messages() -> None:
    with pytest.raises(ValueError, match="no se puede dar la presión final p₂ \\(en una isóbara"):
        pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "isobaric", "p2", 2e5))
    with pytest.raises(ValueError, match="en una isoterma T₂ = T₁"):
        pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "isothermal", "T2", 400.0))
    with pytest.raises(ValueError, match="es una isoterma: elegí ese proceso"):
        pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "polytropic", "p2", 2e5, n=1.0))
    with pytest.raises(ValueError, match="es una isóbara"):
        pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "polytropic", "p2", 2e5, n=0.0))
    with pytest.raises(ValueError, match="tiene que ser positiva"):
        pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "adiabatic", "p2", -1.0))
    with pytest.raises(ValueError, match="polinomios NASA"):
        pt.solve_process(
            pt.PolytropicInputs("air", 1e5, 300.0, "adiabatic", "ratio", 50000.0, model="variable")
        )
    with pytest.raises(ValueError, match="fuera del rango"):
        pt.solve_process(pt.PolytropicInputs("air", 1e5, 300.0, "isochoric", "T2", 7000.0))
    with pytest.raises(ValueError, match="p₂ tiene que ser mayor que p₁"):
        pt.staged_compression(pt.StagedInputs("air", 5e5, 300.0, 1e5, 1.3, 2))
    with pytest.raises(ValueError, match="de 1 a 4"):
        pt.staged_compression(pt.StagedInputs("air", 1e5, 300.0, 9e5, 1.3, 6))
    with pytest.raises(ValueError, match="isóbara"):
        pt.exponent_from_states("air", 1e5, 1e5, T1_K=300.0, T2_K=400.0)
    with pytest.raises(ValueError, match="Dá los dos volúmenes"):
        pt.exponent_from_states("air", 1e5, 2e5)


@pytest.mark.parametrize("name", list(pt.PROCESS_EXAMPLES))
def test_examples(name: str) -> None:
    r = pt.solve_process(pt.PROCESS_EXAMPLES[name].inputs)
    data = pt.process_to_dict(r)
    json.dumps(data)
    assert data["w_f_J_per_kg"] == pytest.approx(r.w_f)
    assert pickle.loads(pickle.dumps(r)) == r
    for row in pt.process_comparison(r.inputs):
        assert row.result is not None or row.reason
