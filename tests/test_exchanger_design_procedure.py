"""Tests del procedimiento LaTeX del diseño de intercambiadores (Fases 12.1 y 12.2)."""

from __future__ import annotations

import dataclasses
import math
import re

import pytest

from core.heat_transfer import double_pipe as dp
from core.heat_transfer import shell_tube as st
from core.heat_transfer import tube_bank as tb
from core.heat_transfer.design_common import FluidProperties
from core.heat_transfer.exchanger_design_procedure import (
    double_pipe_steps,
    shell_tube_steps,
    tube_bank_steps,
)
from core.latex import latex_number, latex_quantity, latex_value
from core.state_report import ProcedureStep

SYSTEMS = ("SI", "Técnico", "Inglés")
INCROPERA_BANK = "Calentador de aire en tresbolillo (Incropera, ejemplo 7.7)"
CENGEL_BANK = "Aire precalentado con agua geotérmica (Çengel y Ghajar, ejemplo 7-7)"
INCROPERA_DP = "Enfriador de aceite de una turbina de gas (Incropera, ejemplo 11.1)"
PROCESS_DP = "Calentador de agua de proceso (caños de acero, con ensuciamiento)"
PETERS = "Agua por el casco (Peters, Timmerhaus y West, ejemplo 14-6)"
KEROSENE = "Enfriador de kerosene con agua de río"
WATER = "Intercambiador agua–agua de 1 paso por el casco y 2 por los tubos"


def _check_steps(steps: list[ProcedureStep]) -> str:
    """Pasos numerados, LaTeX balanceado y sin signos dobles; devuelve todo el LaTeX junto."""
    assert steps
    for k, s in enumerate(steps, 1):
        assert s.title.startswith(f"{k}. ")
        assert s.text
        assert "None" not in s.text and not re.search(r"\de[+-]\d", s.text), s.text
        for tex in s.latex:
            assert tex.count("{") == tex.count("}"), tex
            assert tex.count(r"\left") == tex.count(r"\right"), tex
            assert tex.count(r"\begin{aligned}") == tex.count(r"\end{aligned}"), tex
            assert not re.search(r"[-+]\s*-\s*\d", tex), tex
            assert "nan" not in tex and "inf" not in tex, tex
            assert "Pack(" not in tex, tex
    return "\n".join(t for s in steps for t in s.latex)


def _titles(steps: list[ProcedureStep]) -> list[str]:
    return [s.title.split(". ", 1)[1] for s in steps]


def _step(steps: list[ProcedureStep], title: str) -> ProcedureStep:
    return next(s for s in steps if s.title.split(". ", 1)[1].startswith(title))


def _q(value: float, kind: str, system: str) -> str:
    return latex_quantity(value, kind, system)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Banco de tubos
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(tb.TUBE_BANK_EXAMPLES))
def test_tube_bank_steps(name: str, system: str) -> None:
    r = tb.solve_tube_bank(tb.TUBE_BANK_EXAMPLES[name].inputs)
    steps = tube_bank_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    assert _titles(steps) == [
        "Propiedades",
        "Velocidad máxima y Reynolds",
        "Nusselt (Zukauskas)",
        "Temperatura de salida",
        "Calor",
        "Caída de presión",
    ]
    for value, kind in (
        (r.V_max_m_s, "speed"),
        (r.h_W_per_m2K, "heat_transfer_coefficient"),
        (r.T_out_K, "temperature"),
        (r.Q_W, "heat_rate"),
        (r.dp_Pa, "pressure_drop"),
    ):
        assert _q(value, kind, system) in tex


def test_tube_bank_diagonal_and_transverse_paths() -> None:
    inputs = tb.TUBE_BANK_EXAMPLES[INCROPERA_BANK].inputs
    transverse = tube_bank_steps(tb.solve_tube_bank(inputs), "SI")
    assert "acá no" in _step(transverse, "Velocidad").text
    # S_D < (S_T + D)/2: el paso más angosto es la diagonal
    S_L = 0.97 * math.sqrt(((inputs.S_T_m + inputs.D_m) / 2) ** 2 - (inputs.S_T_m / 2) ** 2)
    tight = tb.solve_tube_bank(dataclasses.replace(inputs, S_L_m=S_L))
    assert tight.diagonal
    steps = tube_bank_steps(tight, "SI")
    assert "acá sí" in _step(steps, "Velocidad").text
    assert r"\frac{S_T}{2\,(S_D - D)}" in "\n".join(_step(steps, "Velocidad").latex)


def test_tube_bank_nusselt_texts() -> None:
    inputs = tb.TUBE_BANK_EXAMPLES[INCROPERA_BANK].inputs
    steps = tube_bank_steps(tb.solve_tube_bank(inputs), "SI")
    text = _step(steps, "Nusselt").text
    assert "Incropera (tabla 7.5)" in text and "C_2 de la tabla 7.6" in text
    assert "1000 a 200 000" in text
    # entre Re = 10² y 10³ Incropera usa el cilindro aislado
    slow = tb.solve_tube_bank(dataclasses.replace(inputs, V_m_s=0.25))
    assert 100 < slow.Re < 1000
    assert "cilindro aislado" in _step(tube_bank_steps(slow, "SI"), "Nusselt").text
    cengel = tb.solve_tube_bank(tb.TUBE_BANK_EXAMPLES[CENGEL_BANK].inputs)
    text = _step(tube_bank_steps(cengel, "SI"), "Nusselt").text
    assert "Çengel y Ghajar (tabla 7-2)" in text and "F de la tabla 7-3" in text


def test_tube_bank_friction_texts() -> None:
    # Çengel 7-7: P_L = 3,33 está fuera de las curvas de f (1,25 a 2,5)
    cengel = tb.solve_tube_bank(tb.TUBE_BANK_EXAMPLES[CENGEL_BANK].inputs)
    text = _step(tube_bank_steps(cengel, "SI"), "Caída de presión").text
    assert "fuera de las curvas: la de 2,5" in text
    assert "El alineado de referencia es el cuadrado" in text
    # justo en una curva
    inputs = tb.TUBE_BANK_EXAMPLES[INCROPERA_BANK].inputs
    on_curve = tb.solve_tube_bank(
        dataclasses.replace(inputs, S_T_m=2.0 * inputs.D_m, S_L_m=2.0 * inputs.D_m)
    )
    text = _step(tube_bank_steps(on_curve, "SI"), "Caída de presión").text
    assert "f en la curva de P_T = 2" in text
    assert "El tresbolillo de referencia es el equilátero" in text


@pytest.mark.parametrize("system", ["Técnico", "Inglés"])
def test_tube_bank_unit_factors(system: str) -> None:
    r = tb.solve_tube_bank(tb.TUBE_BANK_EXAMPLES[INCROPERA_BANK].inputs)
    steps = tube_bank_steps(r, system)  # type: ignore[arg-type]
    outlet = "\n".join(_step(steps, "Temperatura de salida").latex)
    assert {"Técnico": r"1000 \cdot", "Inglés": r"3600 \cdot"}[system] in outlet
    pressure = "\n".join(_step(steps, "Caída de presión").latex)
    assert (r"2 \cdot 32.174" in pressure) == (system == "Inglés")


# ---------------------------------------------------------------------
# Doble tubo
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(dp.DOUBLE_PIPE_EXAMPLES))
def test_double_pipe_steps(name: str, system: str) -> None:
    inputs = dp.DOUBLE_PIPE_EXAMPLES[name].inputs
    r = dp.solve_double_pipe(inputs)
    steps = double_pipe_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[0] == "Propiedades"
    assert titles[1].startswith("Lado del tubo (") and titles[2].startswith("Lado del anular (")
    per = "pie" if system == "Inglés" else "metro"
    assert titles[3] == f"U global (por {per} de tubo)"
    assert "Capacidades caloríficas" in titles
    assert titles[-1] == "Caídas de presión"
    assert ("Largo y horquillas" in titles) == (inputs.problem == "sizing")
    assert ("Largo y área" in titles) == (inputs.problem == "rating")
    for value, kind in (
        (r.tube.h_W_per_m2K, "heat_transfer_coefficient"),
        (r.annulus.h_W_per_m2K, "heat_transfer_coefficient"),
        (r.U_o_W_per_m2K, "heat_transfer_coefficient"),
        (r.dp_tube_Pa, "pressure_drop"),
        (r.dp_annulus_Pa, "pressure_drop"),
    ):
        assert _q(value, kind, system) in tex
    if system != "SI":
        assert _q(r.dp_tube_Pa, "pressure_loss", system) in tex


def test_double_pipe_annulus_kays_perkins() -> None:
    r = dp.solve_double_pipe(dp.DOUBLE_PIPE_EXAMPLES[INCROPERA_DP].inputs)
    assert r.annulus.correlation == "kays_perkins"
    steps = double_pipe_steps(r, "SI")
    annulus = _step(steps, "Lado del anular")
    tex = "\n".join(annulus.latex)
    # d_o/D = 0,5556 entre las filas 0,5 y 1 de la tabla 8.2: Nu_i = 5,642 (Incropera 11.1)
    assert r"\frac{0.5556 - 0.5}{1 - 0.5}" in tex
    assert r"5.74 + (4.86 - 5.74) \cdot 0.1111" in tex and "5.642" in tex
    assert "Kays y Perkins" in annulus.text
    # el f laminar del anular de Shah y London
    pressure = _step(steps, "Caídas de presión")
    assert "Shah y London" in pressure.text
    assert r"f_a &= \frac{95.46}{Re}" in "\n".join(pressure.latex)


def test_double_pipe_pressure_uses_the_installed_length() -> None:
    r = dp.solve_double_pipe(dp.DOUBLE_PIPE_EXAMPLES[INCROPERA_DP].inputs)
    assert r.L_installed_m > r.L_m
    steps = double_pipe_steps(r, "SI")
    pressure = _step(steps, "Caídas de presión")
    assert "largo instalado" in pressure.text
    assert r"\frac{72}{0.025}" in "\n".join(pressure.latex)
    # verificación: el largo es el del equipo y no se habla de "instalado"
    rating = dataclasses.replace(r.inputs, problem="rating", hairpins=3, leg_length_m=5.0)
    rated = dp.solve_double_pipe(rating)
    text = _step(double_pipe_steps(rated, "SI"), "Caídas de presión").text
    assert "instalado" not in text and "Las 3 vueltas" in text


def test_double_pipe_capacities_name_the_fluids() -> None:
    r = dp.solve_double_pipe(dp.DOUBLE_PIPE_EXAMPLES[INCROPERA_DP].inputs)
    text = _step(double_pipe_steps(r, "SI"), "Capacidades caloríficas").text
    assert "paso «Propiedades»" in text and "es un dato" not in text
    assert "caliente (aceite de motor)" in text


def test_double_pipe_u_step_lists_the_resistances() -> None:
    process = dp.solve_double_pipe(dp.DOUBLE_PIPE_EXAMPLES[PROCESS_DP].inputs)
    text = _step(double_pipe_steps(process, "SI"), "U global").text
    assert "ensuciamiento de adentro" in text and "ensuciamiento de afuera" in text
    assert "la pared" in text
    clean = dp.solve_double_pipe(dp.DOUBLE_PIPE_EXAMPLES[INCROPERA_DP].inputs)
    text = _step(double_pipe_steps(clean, "SI"), "U global").text
    assert "ensuciamiento" not in text and "pared es delgada" in text


# ---------------------------------------------------------------------
# Casco y tubos (Kern)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(st.SHELL_TUBE_EXAMPLES))
def test_shell_tube_steps(name: str, system: str) -> None:
    inputs = st.SHELL_TUBE_EXAMPLES[name].inputs
    r = st.solve_shell_tube(inputs)
    steps = shell_tube_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[:2] == ["Geometría", "Propiedades"]
    assert titles[2].startswith("Lado del casco: Kern (")
    assert titles[3].startswith("Lado de los tubos (")
    per = "pie" if system == "Inglés" else "metro"
    assert titles[4] == f"U de diseño, con el ensuciamiento (por {per} de tubo)"
    assert ("U limpio" in titles) == (r.R_d_required > 0.0)
    assert ("El criterio de Kern (R_d)" in titles) == (inputs.problem == "check")
    assert titles[-1] == "Caídas de presión"
    for value, kind in (
        (r.shell.h_W_per_m2K, "heat_transfer_coefficient"),
        (r.tube.h_W_per_m2K, "heat_transfer_coefficient"),
        (r.U_D_W_per_m2K, "heat_transfer_coefficient"),
        (r.area_m2, "area"),
        (r.shell.dp_Pa, "pressure_drop"),
        (r.dp_tube_Pa, "pressure_drop"),
    ):
        assert _q(value, kind, system) in tex
    if system != "SI":
        assert _q(r.shell.dp_Pa, "pressure_loss", system) in tex


def test_shell_side_kern_numbers_peters() -> None:
    r = st.solve_shell_tube(st.SHELL_TUBE_EXAMPLES[PETERS].inputs)
    s = r.shell
    shell = _step(shell_tube_steps(r, "SI"), "Lado del casco")
    tex = "\n".join(shell.latex)
    for value, kind in (
        (s.a_s_m2, "area"),
        (s.G_s, "mass_flux"),
        (s.D_e_m, "length"),
        (s.h_W_per_m2K, "heat_transfer_coefficient"),
    ):
        assert _q(value, kind, "SI") in tex
    # la celda cuadrada: P_T² − π·d_o²/4 sobre π·d_o
    assert r"A_{\text{libre}} &= P_T^2 - \frac{\pi\,d_o^2}{4}" in tex
    assert r"\mathcal{P}_{\text{moj}} &= \pi\,d_o" in tex
    assert "C' &= P_T - d_o" in tex
    # μ_p del enunciado (el libro da el agua a 30 °C y la pared a 40 °C)
    assert "μ_p del enunciado" in shell.text
    assert r"\left(\frac{0.000803}{0.000657}\right)^{0.14}" in tex


def test_shell_side_triangular_cell_and_coolprop_wall() -> None:
    water = st.SHELL_TUBE_EXAMPLES[WATER].inputs
    r = st.solve_shell_tube(dataclasses.replace(water, layout="triangular"))
    shell = _step(shell_tube_steps(r, "SI"), "Lado del casco")
    tex = "\n".join(shell.latex)
    assert r"\frac{\sqrt{3}}{4}\,P_T^2 - \frac{\pi\,d_o^2}{8}" in tex
    assert r"\frac{\pi\,d_o}{2}" in tex
    assert "medio triángulo equilátero" in shell.text
    assert "μ_p de CoolProp" in shell.text


def test_shell_tube_count_estimate_line() -> None:
    r = st.solve_shell_tube(st.SHELL_TUBE_EXAMPLES[KEROSENE].inputs)
    assert r.inputs.tube_count is None and r.N_t % r.inputs.tube_passes == 0
    geometry = _step(shell_tube_steps(r, "SI"), "Geometría")
    assert rf"\to {r.N_t}" in "\n".join(geometry.latex)
    assert "múltiplo de N_p" in geometry.text
    tubes = _step(shell_tube_steps(r, "SI"), "Lado de los tubos")
    assert "en promedio" not in tubes.text
    uneven = st.solve_shell_tube(dataclasses.replace(r.inputs, tube_count=178))
    tubes = _step(shell_tube_steps(uneven, "SI"), "Lado de los tubos")
    assert "44,5 tubos en paralelo en promedio" in tubes.text


def test_kern_check_step() -> None:
    r = st.solve_shell_tube(st.SHELL_TUBE_EXAMPLES[KEROSENE].inputs)
    steps = shell_tube_steps(r, "SI")
    check = _step(steps, "El criterio de Kern")
    tex = "\n".join(check.latex)
    assert _q(r.U_required_W_per_m2K or 0.0, "heat_transfer_coefficient", "SI") in tex
    assert _q(r.R_d_available or 0.0, "fouling_resistance", "SI") in tex
    assert _q(r.R_d_required, "fouling_resistance", "SI") in tex
    assert "el equipo sirve" in check.text
    assert "criterio de R_d" in _step(steps, "U limpio").text
    # acortado: limpio alcanza pero no tolera el ensuciamiento pedido
    short = st.solve_shell_tube(dataclasses.replace(r.inputs, L_m=5.0, baffles=32))
    assert "no tolera" in _step(shell_tube_steps(short, "SI"), "El criterio de Kern").text
    # en la verificación el U limpio no habla del criterio
    rating = st.solve_shell_tube(st.SHELL_TUBE_EXAMPLES[WATER].inputs)
    assert "recién limpio" in _step(shell_tube_steps(rating, "SI"), "U limpio").text


@pytest.mark.parametrize("system", SYSTEMS)
def test_shell_tube_pressure_factors(system: str) -> None:
    r = st.solve_shell_tube(st.SHELL_TUBE_EXAMPLES[WATER].inputs)
    pressure = _step(shell_tube_steps(r, system), "Caídas de presión")  # type: ignore[arg-type]
    tex = "\n".join(pressure.latex)
    assert ("32.174" in tex) == (system == "Inglés")
    assert rf"4 \cdot {r.inputs.tube_passes}" in tex
    assert f"({r.inputs.baffles} + 1)" in tex


def test_capacities_name_the_fluids_with_given_properties() -> None:
    viscous = FluidProperties(850.0, 2000.0, 0.004, 0.13, name="aceite liviano")
    inputs = st.SHELL_TUBE_EXAMPLES[KEROSENE].inputs
    r = st.solve_shell_tube(
        dataclasses.replace(inputs, hot=dataclasses.replace(inputs.hot, properties=viscous))
    )
    text = _step(shell_tube_steps(r, "Técnico"), "Capacidades caloríficas").text
    assert "es un dato" not in text and "1000" in text
    assert "(aceite liviano)" in text or "(agua)" in text


def test_kern_rotated_square_uses_the_square_cell() -> None:
    water = st.SHELL_TUBE_EXAMPLES[WATER].inputs
    r = st.solve_shell_tube(dataclasses.replace(water, layout="rotated_square"))
    shell = _step(shell_tube_steps(r, "SI"), "Lado del casco")
    tex = "\n".join(shell.latex)
    assert r"A_{\text{libre}} &= P_T^2 - \frac{\pi\,d_o^2}{4}" in tex
    assert "Kern no distingue el cuadrado rotado" in shell.text
    assert _q(r.shell.D_e_m, "length", "SI") in tex


# ---------------------------------------------------------------------
# Casco y tubos con Bell–Delaware (Fase 12.2)
# ---------------------------------------------------------------------

OIL_W = FluidProperties(852.0, 2131.0, 0.0325, 0.138, mu_wall_Pa_s=0.01, name="aceite")
BD_TITLES = [
    "Lado del casco: las fugas y el bypass",
    "Lado del casco: el banco ideal",
    "Lado del casco: los factores de corrección",
]


def _bd_inputs(name: str, **geometry: object) -> st.ShellTubeInputs:
    inputs = dataclasses.replace(st.SHELL_TUBE_EXAMPLES[name].inputs, shell_method="bell_delaware")
    if geometry:
        assert inputs.bell_delaware is not None
        inputs = dataclasses.replace(
            inputs, bell_delaware=dataclasses.replace(inputs.bell_delaware, **geometry)
        )
    return inputs


def _oil_shell(inputs: st.ShellTubeInputs, m_dot: float) -> st.ShellTubeInputs:
    """El casco con un aceite viscoso (flujo laminar en el casco)."""
    side = inputs.shell_side
    stream = getattr(inputs, side)
    oil = dataclasses.replace(stream, fluid=None, properties=OIL_W, m_dot_kg_s=m_dot)
    return dataclasses.replace(inputs, **{side: oil})


def _bd_tex(r: st.ShellTubeResult, title: str, system: str = "SI") -> tuple[str, str]:
    step = _step(shell_tube_steps(r, system), title)  # type: ignore[arg-type]
    return "\n".join(step.latex), step.text


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("name", list(st.SHELL_TUBE_EXAMPLES))
def test_bell_delaware_steps(name: str, system: str) -> None:
    r = st.solve_shell_tube(_bd_inputs(name))
    bd = r.bell_delaware
    assert bd is not None
    steps = shell_tube_steps(r, system)  # type: ignore[arg-type]
    tex = _check_steps(steps)
    titles = _titles(steps)
    assert titles[:2] == ["Geometría", "Propiedades"]
    assert titles[2].startswith("Lado del casco (")
    assert titles[2].endswith("la geometría de Bell–Delaware")
    assert titles[3:6] == BD_TITLES
    assert titles[6].startswith("Lado de los tubos (")
    assert titles[-2:] == [
        "Caída de presión en el casco (Bell–Delaware)",
        "Caída de presión en los tubos",
    ]
    assert not any(t.startswith("Lado del casco: Kern") for t in titles)
    for value, kind in (
        (bd.areas.S_m_m2, "area"),
        (bd.areas.S_w_m2, "area"),
        (bd.areas.S_sb_m2, "area"),
        (bd.areas.S_tb_m2, "area"),
        (bd.areas.S_b_m2, "area"),
        (bd.G_s, "mass_flux"),
        (bd.h_ideal_W_per_m2K, "heat_transfer_coefficient"),
        (bd.h_W_per_m2K, "heat_transfer_coefficient"),
        (r.U_D_W_per_m2K, "heat_transfer_coefficient"),
        (bd.dp_bi_Pa, "pressure_drop"),
        (bd.dp_wi_Pa, "pressure_drop"),
        (bd.dp_cross_Pa, "pressure_drop"),
        (bd.dp_window_Pa, "pressure_drop"),
        (bd.dp_ends_Pa, "pressure_drop"),
        (bd.dp_Pa, "pressure_drop"),
        (r.dp_tube_Pa, "pressure_drop"),
    ):
        assert _q(value, kind, system) in tex
    if system != "SI":
        assert _q(bd.dp_Pa, "pressure_loss", system) in tex
    fc = bd.factors
    for value in (fc.J_c, fc.J_l, fc.J_b, fc.R_l, fc.R_b, bd.ideal.j, bd.ideal.f):
        assert latex_number(value, 4) in tex
    # el U de diseño usa el h_o de Bell–Delaware
    u_tex = "\n".join(_step(steps, "U de diseño").latex)
    assert latex_value(bd.h_W_per_m2K, "heat_transfer_coefficient", system) in u_tex  # type: ignore[arg-type]
    assert latex_value(r.shell.h_W_per_m2K, "heat_transfer_coefficient", system) not in u_tex  # type: ignore[arg-type]


def test_bell_delaware_peters_geometry() -> None:
    """Peters et al. 14-6 con 405 tubos y L_bb = 33,3 mm: las áreas del libro."""
    r = st.solve_shell_tube(_bd_inputs(PETERS))
    tex, text = _bd_tex(r, "Lado del casco (agua)")
    assert r"S_m &= B\,\ell_m \\ &= 0.1524 \cdot 0.16727 \\ &= 0.025492\ \mathrm{m^{2}}" in tex
    assert r"S_w &= S_{wg} - S_{wt}" in tex and r"0.032979\ \mathrm{m^{2}}" in tex
    # cuadrado: los pasos son P_T (ni L_pp ni L_tp,ef)
    assert "L_{pp} &=" not in tex and r"\frac{D_s}{P_T}" in tex
    assert "cuadrado (90°)" in text
    tex, text = _bd_tex(r, "Lado del casco: las fugas")
    assert "(TEMA, RCB-4.3)" in text and "(TEMA, RCB-4.2)" in text
    assert "N_ss = 2 pares" in text
    tex, text = _bd_tex(r, "Lado del casco: el banco ideal")
    assert "de 10⁴ a 10⁵" in text and "a₂ = −0,395" in text
    assert "μ_p del enunciado" in text
    tex, _ = _bd_tex(r, "Lado del casco: los factores")
    assert r"J_c &= 0.55 + 0.72\,F_c" in tex
    assert r"J_s = 1 \quad" in tex and r"J_r = 1 \quad (Re_s \ge 100)" in tex


@pytest.mark.parametrize(
    ("layout", "pitch_line", "other"),
    [
        ("triangular", r"L_{pp} &= 0.866\,P_T", r"L_{tp,\text{ef}} &="),
        ("rotated_square", r"L_{pp} = L_{tp,\text{ef}} &= 0.707\,P_T", None),
        ("square", None, r"L_{pp}"),
    ],
)
def test_bell_delaware_layout_pitches(
    layout: str, pitch_line: str | None, other: str | None
) -> None:
    r = st.solve_shell_tube(dataclasses.replace(_bd_inputs(KEROSENE), layout=layout))
    tex, text = _bd_tex(r, "Lado del casco (")
    if pitch_line is not None:
        assert pitch_line in tex
    if other is not None:
        assert other not in tex
    angle = {"triangular": "30°", "rotated_square": "45°", "square": "90°"}[layout]
    assert f"({angle})" in text
    assert f"arreglo de {angle}" in _bd_tex(r, "Lado del casco: el banco ideal")[1]


def test_bell_delaware_bypass_and_strips() -> None:
    none = st.solve_shell_tube(_bd_inputs(KEROSENE, sealing_strips=0))
    tex, text = _bd_tex(none, "Lado del casco: los factores")
    fc, a = none.bell_delaware.factors, none.bell_delaware.areas  # type: ignore[union-attr]
    assert rf"e^{{-1.25 \cdot {latex_number(a.F_sbp, 4)}}}" in tex
    assert "sin tiras, el paréntesis vale 1" in text
    tex, _ = _bd_tex(none, "Caída de presión en el casco")
    assert rf"e^{{-3.7 \cdot {latex_number(a.F_sbp, 4)}}}" in tex
    assert fc.J_b < 1.0 and fc.R_b < 1.0
    # con r_ss ≥ ½ las tiras anulan el bypass
    many = st.solve_shell_tube(_bd_inputs(KEROSENE, sealing_strips=20))
    assert many.bell_delaware.areas.r_ss >= 0.5  # type: ignore[union-attr]
    assert r"J_b = 1 \quad (r_{ss} \ge 1/2)" in _bd_tex(many, "Lado del casco: los factores")[0]
    assert r"R_b = 1 \quad (r_{ss} \ge 1/2)" in _bd_tex(many, "Caída de presión en el casco")[0]
    # con tiras, el paréntesis (1 − (2·r_ss)^⅓)
    two = st.solve_shell_tube(_bd_inputs(KEROSENE, sealing_strips=2))
    assert r"\left(1 - (2 \cdot" in _bd_tex(two, "Lado del casco: los factores")[0]


def test_bell_delaware_lane_and_given_clearances() -> None:
    r = st.solve_shell_tube(_bd_inputs(KEROSENE, L_pl_m=0.016, L_sb_m=0.005, L_tb_m=0.0004))
    tex, text = _bd_tex(r, "Lado del casco: las fugas")
    assert r"S_b &= B\,(L_{bb} + L_{pl})" in tex
    assert "carril de los pasos" in text
    assert "TEMA" not in text
    plain = st.solve_shell_tube(_bd_inputs(KEROSENE))
    assert r"S_b &= B\,L_{bb}" in _bd_tex(plain, "Lado del casco: las fugas")[0]


def test_bell_delaware_unequal_end_spacings() -> None:
    base = _bd_inputs(KEROSENE)
    r = st.solve_shell_tube(_bd_inputs(KEROSENE, B_in_m=1.6 * base.B_m, B_out_m=1.2 * base.B_m))
    fc = r.bell_delaware.factors  # type: ignore[union-attr]
    tex, text = _bd_tex(r, "Lado del casco: los factores")
    assert r"L^*_{\text{ent}} &= \frac{B_{\text{ent}}}{B}" in tex
    assert r"&J_s = \frac{(N_b - 1)" in tex  # contra el margen
    assert latex_number(fc.J_s, 4) in tex and fc.J_s < 1.0
    assert "(n = 0,6)" in text
    tex, text = _bd_tex(r, "Caída de presión en el casco")
    assert r"R_{s,\text{ent}} &= \left(\frac{B}{B_{\text{ent}}}\right)^{2 - n'}" in tex
    assert latex_number(fc.R_s_in, 4) in tex and latex_number(fc.R_s_out, 4) in tex
    assert r"(R_{s,\text{ent}} + R_{s,\text{sal}})" in tex
    assert "n′ = 0,2" in text
    # iguales: J_s = R_s = 1 y los extremos con 2·Δp_bi
    equal = st.solve_shell_tube(base)
    assert (
        r"R_{s,\text{ent}} = R_{s,\text{sal}} = 1"
        in _bd_tex(equal, "Caída de presión en el casco")[0]
    )
    assert r"\Delta p_e &= 2\,\Delta p_{bi}" in _bd_tex(equal, "Caída de presión en el casco")[0]


@pytest.mark.parametrize("system", SYSTEMS)
def test_bell_delaware_laminar(system: str) -> None:
    r = st.solve_shell_tube(_oil_shell(_bd_inputs(KEROSENE), 2.0))
    bd = r.bell_delaware
    assert bd is not None and 20.0 < bd.Re < 100.0
    steps = shell_tube_steps(r, system)  # type: ignore[arg-type]
    _check_steps(steps)
    tex = "\n".join(_step(steps, "Lado del casco (aceite)").latex)
    assert r"\mathcal{P}_w &= \pi\,d_o\,N_t\,F_w" in tex
    assert _q(bd.areas.D_w_m, "length", system) in tex
    step = _step(steps, "Lado del casco: los factores")
    tex = "\n".join(step.latex)
    assert "laminar (Re_s < 100)" in step.text and "(C_bh = 1,35" in step.text
    assert r"J_r &= J_r^* + \frac{20 - Re_s}{80}" in tex
    assert latex_number(bd.factors.J_r, 4) in tex and latex_number(bd.factors.N_c, 4) in tex
    step = _step(steps, "Caída de presión en el casco")
    tex = "\n".join(step.latex)
    assert r"\beta_w &= \frac{N_{tcw}}{P_T - d_o} + \frac{B}{D_w^2}" in tex
    assert _q(bd.dp_wi_Pa, "pressure_drop", system) in tex
    assert ("g_c" in tex) == (system == "Inglés")
    assert "fricción viscosa" in step.text
    # Re ≤ 20: J_r = J_r*
    slow = st.solve_shell_tube(_oil_shell(_bd_inputs(KEROSENE), 0.5))
    assert slow.bell_delaware.Re <= 20.0  # type: ignore[union-attr]
    tex = "\n".join(_step(shell_tube_steps(slow, system), "Lado del casco: los factores").latex)  # type: ignore[arg-type]
    assert r"J_r = J_r^* = " in tex


def test_bell_delaware_jr_floor() -> None:
    """Con muchas hileras (N_c > 1600) J_r* baja de 0,4: el piso."""
    long = dataclasses.replace(_bd_inputs(KEROSENE), L_m=20.0, B_m=0.12, baffles=160)
    r = st.solve_shell_tube(_oil_shell(long, 0.4))
    fc = r.bell_delaware.factors  # type: ignore[union-attr]
    assert fc.J_r_star < 0.4 and fc.J_r == 0.4
    tex, _ = _bd_tex(r, "Lado del casco: los factores")
    assert r"J_r^{\,\prime} = J_r^* = " in tex
    assert r"J_r = \max(0.4,\ J_r^{\,\prime}) = 0.4" in tex


def test_bell_delaware_cut_without_window_tubes() -> None:
    """Un corte que no llega a los tubos: θ_ctl = 0, F_w = 0 y N_tcw = 0."""
    r = st.solve_shell_tube(_bd_inputs(PETERS, baffle_cut=0.10, L_bb_m=0.1))
    tex, _ = _bd_tex(r, "Lado del casco (agua)")
    assert r"\theta_{ctl} = 0 \quad \Rightarrow \quad F_w = 0" in tex
    assert r"N_{tcw} = 0 \quad (\ell_w \le 0)" in tex


@pytest.mark.parametrize("system", SYSTEMS)
def test_bell_delaware_unit_factors(system: str) -> None:
    r = st.solve_shell_tube(_bd_inputs(WATER))
    tex, text = _bd_tex(r, "Lado del casco: el banco ideal", system)
    factor = {"SI": None, "Técnico": "1000", "Inglés": "3600"}[system]
    h_line = next(
        line for line in tex.split("\n") if line.startswith(r"\begin{aligned}h_{\text{ideal}}")
    )
    if factor is None:
        assert r"&= 1000 \cdot" not in h_line and r"&= 3600 \cdot" not in h_line
    else:
        assert rf"&= {factor} \cdot" in h_line and factor in text
    tex, text = _bd_tex(r, "Caída de presión en el casco", system)
    assert ("32.174" in tex) == (system == "Inglés")
    assert ("g_c" in text) == (system == "Inglés")
