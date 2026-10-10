"""Tests del procedimiento LaTeX de los gases ideales (Fase 9.1).

Que cada paso esté bien formado en los tres sistemas de unidades y que los
números que se muestran sean los del resultado (el ancho y KaTeX se verifican
aparte, con un navegador).
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from core.gases import ideal as ig
from core.gases import mixture as mx
from core.gases import polytropic as pt
from core.gases.ideal_procedure import (
    exponent_steps,
    gas_symbol,
    mixing_steps,
    mixture_steps,
    process_steps,
    staged_steps,
    state_change_steps,
)
from core.latex import latex_number, latex_quantity
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, st in enumerate(steps, 1):
        assert st.title.startswith(f"{k}. ")
        assert st.text
        assert "None" not in st.text and "nan" not in st.text
        for tex in st.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert tex.count(r"\left") == tex.count(r"\right"), tex
            assert tex.count(r"\begin{aligned}") == tex.count(r"\end{aligned}"), tex
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex  # «- -3» sin paréntesis
            assert "nan" not in tex and "inf" not in tex.replace(r"\infty", ""), tex
            assert not re.search(r"_(\{[^{}]*\}|\w)_", tex), tex  # c_v_1: «Double subscript»
    return "\n".join(t for st in steps for t in st.latex)


def _q(value: float, kind: str, system: str) -> str:
    return latex_quantity(value, kind, system)  # type: ignore[arg-type]


def _titles(steps: list[ProcedureStep]) -> list[str]:
    return [s.title.split(". ", 1)[1] for s in steps]


# ---------------------------------------------------------------------
# Un gas entre dos estados
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(ig.STATE_EXAMPLES))
def test_state_change_steps(name: str, system: str) -> None:
    r = ig.solve_state_change(ig.STATE_EXAMPLES[name].inputs)
    steps = state_change_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _q(r.gas.R, "specific_entropy", system) in tex
    for st in (r.state1, r.state2):
        for value, kind in ((st.v_m3_per_kg, "specific_volume"), (st.p_Pa, "pressure")):
            assert latex_number(_convert(value, kind, system), 5) in tex
    for change in r.changes:
        assert _q(change.dh, "specific_enthalpy", system) in tex
        assert _q(change.du, "specific_enthalpy", system) in tex
        assert _q(change.ds, "specific_entropy", system) in tex
    assert _q(r.mean.cv, "specific_entropy", system) in tex  # c_v(T_m) a la vista
    assert latex_number(r.pr2, 5) in tex
    assert _q(r.vr2, "absolute_temperature", system) in tex
    assert "Con c_p variable" in _titles(steps)


def _convert(value: float, kind: str, system: str) -> float:
    from core.units_system import convert_from_si

    return convert_from_si(value, kind, system)  # type: ignore[arg-type]


def test_pv_needs_a_factor_outside_the_si() -> None:
    """p·v: 1 bar·m³/kg = 100 kJ/kg y 1 psia·ft³/lb = 0,18505 Btu/lb."""
    r = ig.solve_state_change(ig.STATE_EXAMPLES[next(iter(ig.STATE_EXAMPLES))].inputs)
    texts = {}
    for system in SYSTEMS:
        steps = state_change_steps(r, system)  # type: ignore[arg-type]
        texts[system] = steps[1]
    assert r"100 \cdot" in "\n".join(texts["Técnico"].latex)
    assert r"0.18505 \cdot" in "\n".join(texts["Inglés"].latex)
    assert (
        r"\cdot" not in "\n".join(texts["SI"].latex).split(r"\frac{R\,T_1}{p_1}")[1].split("}{")[1]
    )
    assert texts["Técnico"].text.endswith("1 bar·m³/kg = 100 kJ/kg.")
    # el texto del Inglés termina en punto (hasta esta versión el reemplazo de la coma lo comía)
    assert texts["Inglés"].text.endswith("1 psia·ft³/lb = 0,18505 Btu/lb.")
    assert "Ojo" not in texts["SI"].text


def test_helium_is_monatomic_in_the_procedure() -> None:
    inputs = ig.StateChangeInputs(
        "He", ig.StatePair("pT", 1e5, 300.0), ig.StatePair("pT", 3e5, 600.0)
    )
    steps = state_change_steps(ig.solve_state_change(inputs), "Técnico")
    tex = _check_steps(steps)
    assert r"\tfrac{5}{2}\,R" in tex
    assert "monoatómico" in steps[0].text


def test_every_pair_of_data_is_shown() -> None:
    g = ig.gas("N2")
    v = g.R * 400.0 / 2e5
    for pair, a, b in (("pT", 2e5, 400.0), ("pv", 2e5, v), ("Tv", 400.0, v)):
        inputs = ig.StateChangeInputs(
            "N2",
            ig.StatePair(pair, a, b),
            ig.StatePair("pT", 1e5, 300.0),  # type: ignore[arg-type]
        )
        tex = _check_steps(state_change_steps(ig.solve_state_change(inputs), "SI"))
        lhs = {"pT": "v_1 &=", "pv": "T_1 &=", "Tv": "p_1 &="}[pair]
        assert lhs in tex


# ---------------------------------------------------------------------
# Mezclas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("basis", ["mass", "moles", "mass_fraction", "mole_fraction"])
@pytest.mark.parametrize("name", list(mx.COMPOSITION_EXAMPLES))
def test_mixture_steps(name: str, basis: str, system: str) -> None:
    inputs = mx.COMPOSITION_EXAMPLES[name].inputs
    comps = inputs.components
    if basis in ("mass_fraction", "mole_fraction"):  # las cantidades pasan a sumar 1
        total = sum(c.amount for c in comps)
        comps = tuple(replace(c, amount=c.amount / total) for c in comps)
    r = mx.solve_mixture(replace(inputs, components=comps, basis=basis))  # type: ignore[arg-type]
    steps = mixture_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _q(r.M, "molar_mass", system) in tex
    assert _q(r.R, "specific_entropy", system) in tex
    assert _q(r.cp, "specific_entropy", system) in tex
    assert _q(r.s, "specific_entropy", system) in tex
    assert _q(r.s_mixing, "specific_entropy", system) in tex
    for c in r.components:
        assert latex_number(c.y, 4) in tex
        assert latex_number(c.x, 4) in tex
        assert _q(c.p_i, "pressure", system) in tex
    first = _titles(steps)[0]
    assert (
        first
        == {
            "mass": "Moles y masa de cada gas",
            "moles": "Moles y masa de cada gas",
            "mass_fraction": "De fracciones másicas a molares",
            "mole_fraction": "De fracciones molares a másicas",
        }[basis]
    )
    if r.V_m3 is not None:
        assert _q(r.V_m3, "volume", system) in tex


def test_mixing_entropy_with_the_molar_fractions() -> None:
    """−Σ x_i·R_i·ln y_i = −R_M·Σ y_i·ln y_i (x_i·R_i = y_i·R_M)."""
    r = mx.solve_mixture(mx.COMPOSITION_EXAMPLES["El aire seco (fracciones molares)"].inputs)
    y_ln_y = sum(c.y * math.log(c.y) for c in r.components)
    assert -r.R * y_ln_y == pytest.approx(r.s_mixing, rel=1e-12)
    tex = _check_steps(mixture_steps(r, "SI"))
    assert latex_number(y_ln_y, 5) in tex
    assert r"-R_M \sum y_i \ln y_i" in tex


# ---------------------------------------------------------------------
# Mezcla adiabática
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("model", ["constant", "variable"])
@pytest.mark.parametrize("name", list(mx.MIXING_EXAMPLES))
def test_mixing_steps(name: str, model: str, system: str) -> None:
    r = mx.solve_mixing(replace(mx.MIXING_EXAMPLES[name].inputs, model=model))  # type: ignore[arg-type]
    steps = mixing_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    tank = r.inputs.kind == "tank"
    assert _q(r.T_K, "absolute_temperature", system) in tex
    assert _q(r.S_gen, "entropy" if tank else "entropy_flow", system) in tex
    assert _q(r.X_dest, "energy" if tank else "power", system) in tex
    for st in r.streams:
        assert _q(st.ds_TP, "specific_entropy", system) in tex
        assert _q(st.ds_mix, "specific_entropy", system) in tex
        assert latex_number(st.y, 4) in tex
    titles = _titles(steps)
    assert ("La presión final" in titles) == tank
    assert ("El volumen del tanque" in titles) == tank
    if tank:
        assert _q(r.p_Pa, "pressure", system) in tex
        assert r.V_m3 is not None and _q(r.V_m3, "volume", system) in tex
    if model == "variable":
        # la T final se verifica: lo que entra y lo que sale dan lo mismo
        energy = "u" if tank else "h"
        m = "m" if tank else r"\dot{m}"
        before = next(t for t in tex.split("\n") if rf"\sum {m}_j\,{energy}_j(T_j)" in t)
        after = next(t for t in tex.split("\n") if rf"\sum {m}_j\,{energy}_j(T) &=" in t)
        assert before.rsplit("&= ", 1)[1] == after.rsplit("&= ", 1)[1]


def test_mixing_labels_have_no_double_subscripts() -> None:
    """(m·c_v)_j y no m_j c_v_j, que KaTeX rechaza («Double subscript»)."""
    name = next(n for n in mx.MIXING_EXAMPLES if n.startswith("Tanque"))
    r = mx.solve_mixing(replace(mx.MIXING_EXAMPLES[name].inputs, model="constant"))
    tex = _check_steps(mixing_steps(r, "Técnico"))
    assert r"(m\,c_v)_1" in tex
    assert r"\sum (m\,c_v)_j\,T_j" in tex
    flow = next(n for n in mx.MIXING_EXAMPLES if n.startswith("Cámara"))
    r = mx.solve_mixing(replace(mx.MIXING_EXAMPLES[flow].inputs, model="constant"))
    assert r"(\dot{m}\,c_p)_1" in _check_steps(mixing_steps(r, "Técnico"))


def test_same_temperature_and_pressure_shows_a_clean_zero() -> None:
    """Sin ruido de redondeo (6,8·10⁻¹⁸) cuando la corriente ya está a la T y la p final."""
    r = mx.solve_mixing(
        mx.MIXING_EXAMPLES[next(n for n in mx.MIXING_EXAMPLES if "3 kmol" in n)].inputs
    )
    assert all(st.ds_TP == 0.0 for st in r.streams)
    for system in SYSTEMS:
        tex = _check_steps(mixing_steps(r, system))  # type: ignore[arg-type]
        assert "10^{-1" not in tex


def test_moles_of_each_gas_in_the_mixing() -> None:
    name = next(n for n in mx.MIXING_EXAMPLES if n.startswith("Tanque"))
    r = mx.solve_mixing(mx.MIXING_EXAMPLES[name].inputs)
    tex = _check_steps(mixing_steps(r, "Técnico"))
    for st in r.streams:
        assert latex_number(st.amount / st.gas.M / 1000.0, 5) + r"\ \mathrm{kmol}" in tex
    flow = next(n for n in mx.MIXING_EXAMPLES if n.startswith("Premezcla"))
    r = mx.solve_mixing(mx.MIXING_EXAMPLES[flow].inputs)
    assert r"\mathrm{kmol/s}" in _check_steps(mixing_steps(r, "Técnico"))


# ---------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------


def _all_processes() -> list[pt.PolytropicInputs]:
    out = []
    for kind in pt.PROCESS_KINDS:
        # la isóbara con v₁/v₂ = 4 enfriaría el aire a 75 K: se expande al doble
        values = {"p2": 500e3, "ratio": 0.5 if kind == "isobaric" else 4.0, "T2": 450.0}
        for final in pt.allowed_finals(kind):
            for model in ("constant", "variable"):
                for system in ("closed", "open"):
                    out.append(
                        pt.PolytropicInputs(
                            "air",
                            100e3,
                            300.0,
                            kind,
                            final,
                            values[final],
                            n=1.25,
                            model=model,
                            system=system,
                            amount=2.0,  # type: ignore[arg-type]
                        )
                    )
    return out


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(pt.PROCESS_EXAMPLES))
def test_process_example_steps(name: str, system: str) -> None:
    r = pt.solve_process(pt.PROCESS_EXAMPLES[name].inputs)
    _check_process(r, system)


@pytest.mark.parametrize(
    "inputs",
    _all_processes(),
    ids=lambda i: f"{i.process}-{i.final}-{i.model}-{i.system}",
)
def test_every_process_and_datum(inputs: pt.PolytropicInputs) -> None:
    r = pt.solve_process(inputs)
    for system in SYSTEMS:
        _check_process(r, system)


def _check_process(r: pt.ProcessResult, system: str) -> None:
    steps = process_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    s2 = r.state2
    assert latex_number(_convert(s2.T_K, "absolute_temperature", system), 5) in tex
    assert latex_number(_convert(s2.p_Pa, "pressure", system), 5) in tex
    assert _q(s2.v_m3_per_kg, "specific_volume", system) in tex
    for value in (r.du, r.dh, r.w, r.w_f, r.q):
        assert _q(value, "specific_enthalpy", system) in tex
    assert _q(r.ds, "specific_entropy", system) in tex
    closed = r.inputs.system == "closed"
    assert _q(r.W_total, "energy" if closed else "power", system) in tex
    assert _q(r.Q_total, "energy" if closed else "power", system) in tex
    titles = _titles(steps)
    assert titles[-1] == ("Para toda la masa" if closed else "Para todo el caudal")
    if r.inputs.process == "adiabatic":
        assert r.ds == 0.0  # reversible: nada de 10⁻¹⁵
        assert _q(0.0, "specific_entropy", system) in tex
    if r.inputs.process == "polytropic" and r.inputs.model == "constant":
        assert r.c is not None and _q(r.c, "specific_entropy", system) in tex


def test_adiabatic_with_variable_cp_uses_the_relative_volume() -> None:
    """Çengel §7-9: v_r2 = v_r1/(v₁/v₂), y T₂ sale del polinomio."""
    name = next(n for n in pt.PROCESS_EXAMPLES if n.startswith("Motor de auto"))
    r = pt.solve_process(pt.PROCESS_EXAMPLES[name].inputs)
    tex = _check_steps(process_steps(r, "SI"))
    g = r.gas
    assert g.vr(r.state2.T_K) == pytest.approx(g.vr(r.state1.T_K) / 8.0, rel=1e-8)
    assert latex_quantity(g.vr(r.state2.T_K), "absolute_temperature", "SI") in tex
    assert r"\frac{v_{r1}}{v_1/v_2}" in tex


def test_adiabatic_with_variable_cp_and_final_temperature_shows_both_s0() -> None:
    inputs = pt.PolytropicInputs("air", 100e3, 300.0, "adiabatic", "T2", 600.0, model="variable")
    r = pt.solve_process(inputs)
    tex = _check_steps(process_steps(r, "Técnico"))
    for T in (300.0, 600.0):
        assert latex_quantity(r.gas.s0(T), "specific_entropy", "Técnico") in tex


# ---------------------------------------------------------------------
# Etapas y exponente
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("stages", [1, 2, 3, 4])
@pytest.mark.parametrize("n_exp", [1.0, 1.3, 1.4])
def test_staged_steps(stages: int, n_exp: float, system: str) -> None:
    r = pt.staged_compression(pt.StagedInputs("air", 100e3, 300.0, 900e3, n_exp, stages))
    steps = staged_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert _q(r.w_f, "specific_enthalpy", system) in tex
    assert latex_number(r.stage_ratio, 5) in tex
    for px in r.pressures_Pa[1:-1]:
        assert _q(px, "pressure", system) in tex
    assert ("Contra una sola etapa" in titles) == (stages > 1)
    assert ("Los interenfriadores" in titles) == (stages > 1 and n_exp != 1.0)
    if stages > 1:
        assert _q(r.w_f_single, "specific_enthalpy", system) in tex
        assert latex_number(r.saving * 100.0, 4) in tex
    if stages > 1 and n_exp != 1.0:
        assert _q(r.q_intercoolers, "specific_enthalpy", system) in tex
        assert ("El interenfriador lleva" in steps[2].text) == (stages == 2)


def test_two_stages_of_cengel() -> None:
    """Çengel §7-12: 215,3 kJ/kg en dos etapas con n = 1,3."""
    r = pt.staged_compression(pt.StagedInputs("air", 100e3, 300.0, 900e3, 1.3, 2))
    tex = _check_steps(staged_steps(r, "Técnico"))
    assert r.w_f / 1e3 == pytest.approx(-215.3, abs=0.1)
    assert latex_quantity(r.w_f, "specific_enthalpy", "Técnico") in tex
    assert r"p_{x,1} &= p_1\,r^{1} \\ &= 3\ \mathrm{bar}" in tex


@pytest.mark.parametrize("system", SYSTEMS)
def test_exponent_steps(system: str) -> None:
    E = pt.EXPONENT_EXAMPLE
    e = pt.exponent_from_states("air", E["p1_Pa"], E["p2_Pa"], T1_K=E["T1_K"], T2_K=E["T2_K"])
    steps = exponent_steps(
        e,
        system,
        E["p1_Pa"],
        E["p2_Pa"],
        T1_K=E["T1_K"],
        T2_K=E["T2_K"],  # type: ignore[arg-type]
    )
    tex = _check_steps(steps)
    assert latex_number(e.n, 5) in tex
    assert e.text in steps[0].text
    g = ig.gas("air")
    v1, v2 = g.R * E["T1_K"] / E["p1_Pa"], g.R * E["T2_K"] / E["p2_Pa"]
    e2 = pt.exponent_from_states("air", E["p1_Pa"], E["p2_Pa"], v1=v1, v2=v2)
    assert e2.n == pytest.approx(e.n, rel=1e-12)
    tex2 = _check_steps(exponent_steps(e2, system, E["p1_Pa"], E["p2_Pa"], v1=v1, v2=v2))  # type: ignore[arg-type]
    assert r"\frac{\ln(p_1/p_2)}{\ln(v_2/v_1)}" in tex2
    assert latex_number(e2.n, 5) in tex2


def test_every_gas_has_a_symbol() -> None:
    for key, g in ig.IDEAL_GASES.items():
        sym = gas_symbol(key)
        assert sym.count("{") == sym.count("}")
        assert " " not in sym.replace(r"\text{aire}", "")
        if key not in ("air",):
            assert g.formula.replace("₂", "_2").replace("₃", "_3").replace("₄", "_4")[:1] in sym
    with pytest.raises(ValueError, match="no está en la lista"):
        gas_symbol("Xe")
