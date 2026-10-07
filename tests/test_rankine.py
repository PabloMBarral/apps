"""Tests de :mod:`core.cycles.rankine` — Fase 3.1a.

Valores de referencia de Çengel & Boles, *Termodinámica*, cap. 10:

- Ej. 10-1, Rankine ideal simple (3 MPa, 350 °C; 75 kPa): η = 26.0 %,
  x₄ = 0.886, w_T = 713.1 kJ/kg, w_B = 3.03 kJ/kg, q_H = 2728.6 kJ/kg;
  Carnot entre las mismas temperaturas 41.5 %.
- Ej. 10-3 (3 MPa, 350 °C; 10 kPa) a) 33.4 %, b) a 600 °C 37.3 %,
  c) a 15 MPa y 600 °C 43.0 % (x₄ = 0.804).
- Ej. 10-4, con recalentamiento (15 MPa, 600 °C; 4 MPa, 600 °C; 10 kPa):
  η = 45.0 % y 10.4 % de humedad a la salida (x₆ = 0.896).

Además se compara cada estado de TESPy con un cálculo a mano con CoolProp.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings

import CoolProp.CoolProp as CP
import pytest

from core.cycles.rankine import (
    RANKINE_EXAMPLES,
    RankineInputs,
    Reheat,
    default_sweep_values,
    rankine_isentropic_states,
    rankine_labeled_states,
    rankine_notes,
    rankine_steps,
    rankine_sweep,
    rankine_to_dict,
    solve_rankine,
)
from core.cycles.tespy_utils import new_network, solve
from core.export import dict_to_csv

C = 273.15


def _cengel_10_1() -> RankineInputs:
    return RankineInputs(3.0e6, 350 + C, 75.0e3)


def _hand_rankine(inputs: RankineInputs) -> list[float]:
    """Entalpías de cada estado con CoolProp, como se haría con tablas."""
    f = "Water"
    p_hi, p_lo, eta_t, eta_p = (
        inputs.p_boiler_Pa,
        inputs.p_condenser_Pa,
        inputs.eta_turbine,
        inputs.eta_pump,
    )
    h1 = CP.PropsSI("H", "P", p_lo, "Q", 0, f)
    s1 = CP.PropsSI("S", "P", p_lo, "Q", 0, f)
    h2 = h1 + (CP.PropsSI("H", "P", p_hi, "S", s1, f) - h1) / eta_p
    if inputs.T_turbine_in_K is None:
        h3, s3 = CP.PropsSI("H", "P", p_hi, "Q", 1, f), CP.PropsSI("S", "P", p_hi, "Q", 1, f)
    else:
        h3 = CP.PropsSI("H", "P", p_hi, "T", inputs.T_turbine_in_K, f)
        s3 = CP.PropsSI("S", "P", p_hi, "T", inputs.T_turbine_in_K, f)

    def expand(h_in: float, s_in: float, p_out: float) -> float:
        return h_in - eta_t * (h_in - CP.PropsSI("H", "P", p_out, "S", s_in, f))

    if inputs.reheat is None:
        return [h1, h2, h3, expand(h3, s3, p_lo)]
    rh = inputs.reheat
    h4 = expand(h3, s3, rh.p_Pa)
    h5 = CP.PropsSI("H", "P", rh.p_Pa, "T", rh.T_K, f)
    s5 = CP.PropsSI("S", "P", rh.p_Pa, "T", rh.T_K, f)
    return [h1, h2, h3, h4, h5, expand(h5, s5, p_lo)]


# ---------------------------------------------------------------------
# Valores del libro
# ---------------------------------------------------------------------


class TestCengel:
    def test_example_10_1(self) -> None:
        r = solve_rankine(_cengel_10_1())
        assert r.eta_th == pytest.approx(0.260, abs=0.002)
        assert r.x_turbine_out == pytest.approx(0.886, abs=0.002)
        assert r.w_turbine_J_per_kg == pytest.approx(713.1e3, abs=1.0e3)
        assert r.w_pump_J_per_kg == pytest.approx(3.03e3, abs=0.02e3)
        assert r.q_in_J_per_kg == pytest.approx(2728.6e3, abs=1.0e3)
        assert r.eta_carnot == pytest.approx(0.415, abs=0.002)

    @pytest.mark.parametrize(
        ("p_hi", "T_in", "eta", "x4"),
        [
            (3.0e6, 350 + C, 0.334, 0.813),  # a)
            (3.0e6, 600 + C, 0.373, 0.915),  # b) más sobrecalentamiento
            (15.0e6, 600 + C, 0.430, 0.804),  # c) más presión
        ],
    )
    def test_example_10_3(self, p_hi: float, T_in: float, eta: float, x4: float) -> None:
        r = solve_rankine(RankineInputs(p_hi, T_in, 10.0e3))
        assert r.eta_th == pytest.approx(eta, abs=0.002)
        assert r.x_turbine_out == pytest.approx(x4, abs=0.002)

    def test_example_10_4_reheat(self) -> None:
        r = solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3, reheat=Reheat(4.0e6, 600 + C)))
        assert r.eta_th == pytest.approx(0.450, abs=0.002)
        assert r.x_turbine_out == pytest.approx(0.896, abs=0.002)
        assert len(r.states) == 6


# ---------------------------------------------------------------------
# TESPy contra el cálculo a mano y balances
# ---------------------------------------------------------------------

_CASES = [
    _cengel_10_1(),
    RankineInputs(15.0e6, 600 + C, 10.0e3, eta_turbine=0.87, eta_pump=0.85),
    RankineInputs(15.0e6, 600 + C, 10.0e3, reheat=Reheat(4.0e6, 600 + C)),
    RankineInputs(15.0e6, 600 + C, 10.0e3, 0.85, 0.8, reheat=Reheat(3.0e6, 550 + C)),
    RankineInputs(8.0e6, None, 10.0e3),  # vapor saturado a la entrada
    RankineInputs(25.0e6, 600 + C, 5.0e3, 0.9, 0.85),  # supercrítico
]


@pytest.mark.parametrize("inputs", _CASES)
def test_states_match_hand_calculation(inputs: RankineInputs) -> None:
    r = solve_rankine(inputs)
    for state, h in zip(r.states, _hand_rankine(inputs), strict=True):
        assert state.h_J_per_kg == pytest.approx(h, rel=1e-6)


@pytest.mark.parametrize("inputs", _CASES)
def test_energy_balance_and_tespy_components(inputs: RankineInputs) -> None:
    r = solve_rankine(inputs)
    assert r.q_in_J_per_kg - r.q_out_J_per_kg == pytest.approx(r.w_net_J_per_kg, rel=1e-9)
    balances = dict(r.tespy_balances)  # TESPy, por 1 kg/s y positivo hacia el fluido
    turbines = sum(v for k, v in balances.items() if k.startswith("turbina"))
    assert -turbines == pytest.approx(r.w_turbine_J_per_kg, rel=1e-6)
    assert balances["bomba"] == pytest.approx(r.w_pump_J_per_kg, rel=1e-6)
    heat_in = balances["caldera"] + balances.get("recalentador", 0.0)
    assert heat_in == pytest.approx(r.q_in_J_per_kg, rel=1e-6)
    assert -balances["condensador"] == pytest.approx(r.q_out_J_per_kg, rel=1e-6)


@pytest.mark.parametrize("inputs", [c for c in _CASES if c.is_ideal])
def test_ideal_cycle_equals_carnot_at_mean_temperature(inputs: RankineInputs) -> None:
    # Salida húmeda: calor cedido a T_C constante y aportado de forma reversible,
    # así que η = 1 − T_C/T̄_H.
    r = solve_rankine(inputs)
    assert r.x_turbine_out is not None
    assert r.eta_mean_temperature == pytest.approx(r.eta_th, rel=1e-6)
    assert r.eta_th < r.eta_carnot


def test_superheated_exit_rejects_heat_above_the_condensing_temperature() -> None:
    # Ideal, pero la turbina descarga vapor sobrecalentado: parte del calor se cede por
    # encima de T_C (desrecalentamiento) y η queda por debajo de 1 − T_C/T̄_H.
    r = solve_rankine(RankineInputs(10.0e5, 500 + C, 2.0e5))
    assert r.x_turbine_out is None
    assert r.eta_th < r.eta_mean_temperature - 1e-3


def test_real_cycle_is_below_the_ideal_one() -> None:
    ideal = solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3))
    real = solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3, 0.87, 0.85))
    assert real.eta_th < ideal.eta_th
    assert real.x_turbine_out > ideal.x_turbine_out  # la irreversibilidad seca el vapor
    assert real.eta_th < real.eta_mean_temperature


def test_states_are_labeled_with_cengel_numbering() -> None:
    labels = [label for label, _ in rankine_labeled_states(solve_rankine(_cengel_10_1()))]
    assert labels[0].startswith("1 ") and labels[-1].startswith("4 ")
    assert solve_rankine(_cengel_10_1()).states[0].region == "saturated_liquid"
    reheat = solve_rankine(RANKINE_EXAMPLES[[k for k in RANKINE_EXAMPLES if "10-4" in k][0]])
    iso = [label for label, _ in rankine_isentropic_states(reheat)]
    assert iso == ["2s", "4s", "6s"]


def test_saturated_vapor_inlet() -> None:
    r = solve_rankine(RankineInputs(8.0e6, None, 10.0e3))
    assert r.states[2].region == "saturated_vapor"


def test_superheated_turbine_exit_has_no_quality() -> None:
    # Expansión corta desde vapor muy sobrecalentado: sale sobrecalentado.
    r = solve_rankine(RankineInputs(10.0e5, 500 + C, 2.0e5))
    assert r.x_turbine_out is None
    assert any("sobrecalentado" in note for note in rankine_notes(r))


def test_low_quality_note_and_supercritical_note() -> None:
    notes = rankine_notes(solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3)))
    assert any("erosiona" in note for note in notes)  # x₄ = 0.804
    notes = rankine_notes(solve_rankine(RankineInputs(25.0e6, 600 + C, 10.0e3)))
    assert any("supercrítico" in note for note in notes)


def test_net_power_sets_the_mass_flow() -> None:
    r = solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3, m_dot_kg_s=None, W_net_W=100.0e6))
    assert r.W_net_W == pytest.approx(100.0e6)
    assert r.m_dot_kg_s == pytest.approx(100.0e6 / r.w_net_J_per_kg)


def test_result_is_picklable_for_streamlit_cache() -> None:
    r = solve_rankine(_cengel_10_1())
    assert pickle.loads(pickle.dumps(r)).eta_th == pytest.approx(r.eta_th)


@pytest.mark.parametrize("name", list(RANKINE_EXAMPLES))
def test_every_example_solves_without_warnings(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_rankine(RANKINE_EXAMPLES[name])
    # Los ORC rinden menos (Fase 3.1c: de 8 % a 30 %).
    assert (0.05 if r.inputs.fluid != "Water" else 0.2) < r.eta_th < 0.5


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (RankineInputs(3.0e6, 200 + C, 10.0e3), "sería líquida"),
        (RankineInputs(3.0e6, 233.85 + C, 10.0e3), "justo la de saturación"),
        (RankineInputs(3.0e6, 233.85 + C, 10.0e3), "Vapor saturado seco"),
        (RankineInputs(10.0e3, 350 + C, 3.0e6), "menor que la de la caldera"),
        (RankineInputs(3.0e6, 350 + C, 300.0e5), "entre la del punto triple"),
        (RankineInputs(3.0e6, 350 + C, 100.0), "entre la del punto triple"),
        (RankineInputs(3.0e6, 350 + C, 10.0e3, eta_turbine=1.2), "entre 0 y 1"),
        (RankineInputs(3.0e6, 350 + C, 10.0e3, eta_pump=0.0), "entre 0 y 1"),
        (RankineInputs(25.0e6, None, 10.0e3), "no hay vapor saturado"),
        (RankineInputs(25.0e6, 300 + C, 10.0e3), "superar la crítica"),
        (RankineInputs(15.0e6, 600 + C, 10.0e3, reheat=Reheat(20.0e6, 600 + C)), "recalentamiento"),
        (RankineInputs(15.0e6, 600 + C, 10.0e3, reheat=Reheat(4.0e6, 200 + C)), "sería líquida"),
        (
            RankineInputs(15.0e6, 600 + C, 10.0e3, reheat=Reheat(4.0e6, 250.355 + C)),
            "subí su temperatura",
        ),
        (RankineInputs(3.0e6, 350 + C, 10.0e3, m_dot_kg_s=None), "caudal másico o la potencia"),
        (RankineInputs(3.0e6, 350 + C, 10.0e3, m_dot_kg_s=1.0, W_net_W=1.0e6), "uno de los dos"),
        (RankineInputs(3.0e6, 350 + C, 10.0e3, m_dot_kg_s=-1.0), "positivo"),
    ],
)
def test_invalid_inputs_are_explained(inputs: RankineInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_rankine(inputs)


def test_reheat_that_would_cool_is_explained() -> None:
    # TESPy lo resuelve sin quejarse (q_rec < 0): lo detecta el chequeo posterior.
    with pytest.raises(ValueError, match="tendría que enfriar"):
        solve_rankine(RankineInputs(15.0e6, 600 + C, 10.0e3, reheat=Reheat(10.0e6, 320 + C)))


def test_solve_explains_a_failed_network() -> None:
    from tespy.components import CycleCloser, Pump, SimpleHeatExchanger, Turbine
    from tespy.connections import Connection

    network = new_network()
    cc, sg, tu = CycleCloser("cc"), SimpleHeatExchanger("sg"), Turbine("tu")
    cd, fp = SimpleHeatExchanger("cd"), Pump("fp")
    c3 = Connection(cc, "out1", tu, "in1")
    c4 = Connection(tu, "out1", cd, "in1")
    c1 = Connection(cd, "out1", fp, "in1")
    c2 = Connection(fp, "out1", sg, "in1")
    c0 = Connection(sg, "out1", cc, "in1")
    network.add_conns(c3, c4, c1, c2, c0)
    sg.set_attr(pr=1)
    cd.set_attr(pr=1)
    tu.set_attr(eta_s=0.9)
    fp.set_attr(eta_s=0.8)
    c3.set_attr(T=623.15, p=30e5, m=1, fluid={"water": 1})
    c4.set_attr(p=40e5)  # la "turbina" tendría que comprimir
    c1.set_attr(x=0)
    with pytest.raises(ValueError, match="fuera del rango físico"):
        solve(network, what="el ciclo")


# ---------------------------------------------------------------------
# Barridos (Cengel §10-4: ¿cómo aumentar el rendimiento?)
# ---------------------------------------------------------------------


def _etas(inputs: RankineInputs, parameter: str) -> list[float]:
    values = default_sweep_values(inputs, parameter)  # type: ignore[arg-type]
    points = rankine_sweep(inputs, parameter, values)  # type: ignore[arg-type]
    assert len(points) == len(values)
    return [p.eta_th for p in points]


def test_sweeps_follow_cengel_10_6() -> None:
    base = RankineInputs(15.0e6, 600 + C, 10.0e3)
    eta_p = _etas(base, "p_boiler")
    eta_T = _etas(base, "T_turbine_in")
    eta_c = _etas(base, "p_condenser")
    assert eta_p == sorted(eta_p)  # más presión de caldera → más η
    assert eta_T == sorted(eta_T)  # más sobrecalentamiento → más η
    assert eta_c == sorted(eta_c, reverse=True)  # más presión de condensador → menos η


def test_sweep_skips_invalid_points() -> None:
    base = RankineInputs(3.0e6, 350 + C, 10.0e3)
    points = rankine_sweep(base, "T_turbine_in", [200 + C, 400 + C])
    assert [p.value_si for p in points] == [400 + C]


# ---------------------------------------------------------------------
# Procedimiento y export
# ---------------------------------------------------------------------


_UNITS = [("SI", r"\mathrm{J/kg}"), ("Técnico", r"\mathrm{kJ/kg}"), ("Inglés", r"\mathrm{Btu/lb}")]


@pytest.mark.parametrize(("system", "unit"), _UNITS)
@pytest.mark.parametrize("inputs", _CASES)
def test_steps_in_every_unit_system(inputs: RankineInputs, system: str, unit: str) -> None:
    steps = rankine_steps(solve_rankine(inputs), system)  # type: ignore[arg-type]
    tex = "\n".join(t for step in steps for t in step.latex)
    assert unit in tex
    for step in steps:
        for t in step.latex:
            assert t.count("{") == t.count("}"), (step.title, t)
    titles = [s.title for s in steps]
    assert ("4 → 5: recalentador" in titles) == (inputs.reheat is not None)
    assert (r"\eta_B" in tex) == (inputs.eta_pump != 1.0)
    assert "- -" not in tex


@pytest.mark.parametrize("name", list(RANKINE_EXAMPLES))
def test_procedure_cites_the_textbook_sections(name: str) -> None:
    """Cengel (7.ª a 9.ª ed.): §10-3 ciclo real, §10-5 recalentamiento, §10-6 regeneración."""
    result = solve_rankine(RANKINE_EXAMPLES[name])
    text = " ".join(f"{s.title} {s.text}" for s in rankine_steps(result, "Técnico"))
    assert ("§10-5" in text) == result.has_reheat
    if result.has_losses:
        assert "§10-3" in text


def test_wet_turbine_exit_uses_the_lever_rule() -> None:
    result = solve_rankine(_cengel_10_1())
    turbine = next(s for s in rankine_steps(result, "Técnico") if s.title.startswith("3 → 4"))
    assert any(r"x_{4s}" in t for t in turbine.latex)
    assert any(r"h_f + x_{4s}\,h_{fg}" in t for t in turbine.latex)  # como Cengel: h_f + x·h_fg
    assert any(t.startswith(r"x_{4} = x_{4s} =") for t in turbine.latex)  # turbina ideal
    assert "palanca" in turbine.text and "s₃" in turbine.text
    # En SI los números llevan ×10ⁿ: el término x·h_fg pasa a otro renglón (celular).
    turbine_si = next(s for s in rankine_steps(result, "SI") if s.title.startswith("3 → 4"))
    assert any(r"\\ &\quad + 0.886" in t for t in turbine_si.latex)


def test_export_is_serializable() -> None:
    data = rankine_to_dict(solve_rankine(RANKINE_EXAMPLES[next(iter(RANKINE_EXAMPLES))]), "Técnico")
    json.dumps(data, ensure_ascii=False)
    assert data["resultados"]["eta_termica"] == pytest.approx(0.260, abs=0.002)
    assert len(data["estados"]) == 4
    assert "resultados.eta_termica" in dict_to_csv(data)
    assert math.isfinite(data["resultados"]["potencia_neta"]["valor"])
