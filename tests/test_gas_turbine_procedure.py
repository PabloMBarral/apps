"""Tests del procedimiento de la turbina de gas con etapas y regenerador — Fase 3.7.

Que los pasos sigan el orden de Cengel (§9-8 a §9-10), que los números que se
muestran sean los del resultado y que el LaTeX esté bien armado en los tres
sistemas de unidades (el ancho y la validez con KaTeX se miden aparte, en un
navegador).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from core.cycles.gas_turbine import GAS_TURBINE_EXAMPLES, GasTurbineInputs, solve_gas_turbine
from core.cycles.gas_turbine_procedure import gas_turbine_steps
from core.latex import latex_number

C = 273.15
NAMES = list(GAS_TURBINE_EXAMPLES)
EX_9_5, EX_9_6, EX_9_7, EX_9_8A, EX_9_8B, EX_IND, EX_MICRO, EX_AERO, EX_SEQ = NAMES
FULL = GasTurbineInputs(
    30.0,
    1400 + C,
    0.88,
    0.90,
    dp_combustor=0.03,
    compressor_stages=2,
    turbine_stages=2,
    T_intercool_K=30 + C,
    dp_intercooler=0.02,
    regenerator=0.85,
    dp_regenerator=0.03,
    W_net_W=100e6,
)
CASES = {**{n: GAS_TURBINE_EXAMPLES[n] for n in NAMES}, "2+2 metano con todo": FULL}
SYSTEMS = ["SI", "Técnico", "Inglés"]


def _titles(inputs: GasTurbineInputs, system: str = "Técnico") -> list[str]:
    return [s.title for s in gas_turbine_steps(solve_gas_turbine(inputs), system)]  # type: ignore[arg-type]


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(CASES))
def test_procedure_is_well_formed(name: str, system: str) -> None:
    r = solve_gas_turbine(CASES[name])
    steps = gas_turbine_steps(r, system)  # type: ignore[arg-type]
    titles = [s.title for s in steps]
    assert titles[0] == "Composición del aire"
    assert titles[-4:] == [
        "Trabajo neto y rendimiento",
        "Caudales y potencias",
        "Exergía destruida y perdida en cada componente",
        "Balance de exergía",
    ]
    st = r.states
    for stage in (*r.compressors, *r.turbines):
        title = f"{stage.name[0].upper()}{stage.name[1:]} ({st[stage.inlet].label} → "
        assert sum(t.startswith(title) for t in titles) == 1, (title, titles)
    for ic in r.intercoolers:
        assert any(t.startswith(ic.name[0].upper() + ic.name[1:] + " (") for t in titles)
    for cc in r.combustors:
        assert any(t.startswith(cc.name[0].upper() + cc.name[1:] + " (") for t in titles)
    assert any(t.startswith("Regenerador (") for t in titles) == (r.regenerator is not None)
    assert ("Presiones de cada etapa" in titles) == (
        len(r.compressors) > 1 or len(r.turbines) > 1 or r.inputs.dp_combustor > 0.0
    )
    for step in steps:
        assert step.text
        for tex in step.latex:
            assert tex.count("{") == tex.count("}"), (step.title, tex)
            assert "- -" not in tex and "+ -" not in tex, (step.title, tex)
            assert "nan" not in tex and "inf" not in tex, (step.title, tex)


def test_cengel_order_with_air_standard() -> None:
    """Aire estándar: compresores, turbinas, regenerador y recién ahí el calor (ejemplo 9-7)."""
    titles = _titles(GAS_TURBINE_EXAMPLES[EX_9_7])
    assert titles[:6] == [
        "Composición del aire",
        "Compresor (1 → 2)",
        "Turbina (3 → 4)",
        "Regenerador (2 → 5 y 4 → 6)",
        "Calentador (5 → 3), aire estándar",
        "Trabajo neto y rendimiento",
    ]


def test_order_with_combustion_and_stages() -> None:
    titles = _titles(FULL)
    assert titles[:11] == [
        "Composición del aire",
        "Presiones de cada etapa",
        "Compresor de baja (1 → 2)",
        "Interenfriador (2 → 3)",
        "Compresor de alta (3 → 4)",
        "Cámara de combustión (5 → 6)",
        "Composición de los gases de combustión",
        "Turbina de alta (6 → 7)",
        "Cámara de recalentamiento (7 → 8)",
        "Turbina de baja (8 → 9)",
        "Regenerador (4 → 5 y 9 → 10)",
    ]


def _latex(inputs: GasTurbineInputs, title_start: str, system: str = "Técnico") -> str:
    steps = gas_turbine_steps(solve_gas_turbine(inputs), system)  # type: ignore[arg-type]
    return " ".join(tex for s in steps if s.title.startswith(title_start) for tex in s.latex)


def test_numbers_shown_are_the_result() -> None:
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_7])
    net = _latex(r.inputs, "Trabajo neto")
    assert latex_number(r.eta_th * 100, 4) in net
    assert latex_number(r.back_work_ratio, 4) in net
    regen = _latex(r.inputs, "Regenerador")
    assert r.regenerator is not None
    assert latex_number(r.regenerator.q / 1e3, 5) in regen  # q_reg en kJ/kg
    assert latex_number(r.regenerator.q_max / 1e3, 5) in regen
    seq = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_SEQ])
    reheat = _latex(seq.inputs, "Cámara de recalentamiento")
    assert latex_number(seq.combustors[1].fuel, 5) in reheat
    assert latex_number(seq.fuel_air_ratio, 5) in reheat
    first = _latex(seq.inputs, "Cámara de combustión")
    assert latex_number(seq.combustors[0].fuel, 5) in first


def test_intermediate_pressure_is_the_geometric_mean() -> None:
    tex = _latex(GAS_TURBINE_EXAMPLES[EX_9_8A], "Presiones de cada etapa")
    assert r"\sqrt{p_{1}\,p_{4}}" in tex
    assert latex_number(8.0**0.5, 5) in tex  # r_C = √8
    given = _latex(GAS_TURBINE_EXAMPLES[EX_AERO], "Presiones de cada etapa")
    assert r"\sqrt" not in given and "p_{2} = 3.5" in given


def test_exergy_steps() -> None:
    tex = _latex(FULL, "Exergía destruida")
    for symbol in (
        r"x_{d,C1}",
        r"x_{d,C2}",
        r"x_{d,T1}",
        r"x_{d,T2}",
        r"x_{d,\text{cám}}",
        r"x_{d,\text{RH}}",
        r"x_{d,\mathrm{reg}}",
        r"x_{IC}",
        r"x_{\mathrm{esc}}",
    ):
        assert symbol in tex, symbol
    balance = _latex(FULL, "Balance de exergía")
    r = solve_gas_turbine(FULL)
    assert latex_number(r.eta_th * 100, 4) in balance  # η_II = η con x_comb ≈ PCI
    air = _latex(GAS_TURBINE_EXAMPLES[EX_9_8B], "Exergía destruida")
    assert r"x_{Q,1}" in air and r"x_{d,\text{cal}}" in air and r"x_{d,\text{RH}}" in air


def test_sizing_by_power_shows_the_air_flow() -> None:
    tex = _latex(FULL, "Caudales y potencias")
    assert r"\frac{\dot{W}_{TG}}{w_{\mathrm{neto}}}" in tex


def test_negative_s0_goes_to_a_new_line() -> None:
    """s° es negativa debajo de 25 °C: en el renglón con R·ln, el número va solo (celular)."""
    cold = replace(GAS_TURBINE_EXAMPLES[EX_IND], T_amb_K=-10 + C)
    tex = _latex(cold, "Compresor", "Inglés")
    assert r"\\ &\quad +" in tex
