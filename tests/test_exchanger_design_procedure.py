"""Tests del procedimiento LaTeX del diseño de intercambiadores (Fase 12.1)."""

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
from core.latex import latex_quantity
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
