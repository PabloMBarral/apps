"""Tests de /Diseno_de_Intercambiadores (AppTest).

Fases 12.1 y 12.2. Los tres modos (banco de tubos, doble tubo y casco y tubos con Kern o con
Bell–Delaware), todos los ejemplos en los tres sistemas de unidades, los cambios de arreglo,
de constantes, de fluido, de lado, de problema y de calibre, los datos de Bell–Delaware
(corte, holguras, tiras, carril y extremos), la comparación con Kern, los errores, la
teoría, el procedimiento y la exportación.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.heat_transfer import double_pipe as dp
from core.heat_transfer import shell_tube as stt
from core.heat_transfer import tube_bank as tb
from core.heat_transfer.design_common import suggested_pressure
from core.units_system import convert_from_si

PAGE = str(Path(__file__).resolve().parents[1] / "app_pages" / "23_Diseno_de_Intercambiadores.py")
C = 273.15
M_BANK, M_DP, M_ST = "Banco de tubos", "Doble tubo", "Casco y tubos"
SYSTEMS = ("SI", "Técnico", "Inglés")


def _app() -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=300)
    at.run()
    _no_problems(at)
    return at


def _no_problems(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def _metric(at: AppTest, prefix: str) -> float:
    value = next(m.value for m in at.metric if m.label.startswith(prefix))
    value = value.split(" ")[0] if not re.match(r"^-?[\d ]+$", value) else value
    return float(value.replace(" ", "").replace(",", ".").replace("−", "-"))


def _mode(at: AppTest, mode: str) -> AppTest:
    at.radio(key="ix_mode").set_value(mode).run()
    _no_problems(at)
    return at


def _system(at: AppTest, system: str) -> AppTest:
    at.selectbox(key="units_system").set_value(system).run()
    return at


def _wkey(at: AppTest, key: str) -> str:
    system = at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"
    return f"{key}@{system}"


def _latex(at: AppTest) -> str:
    return "\n".join(str(x.value) for x in at.latex)


def _current_system(at: AppTest) -> str:
    return at.session_state["units_system"] if "units_system" in at.session_state else "Técnico"


# ---------------------------------------------------------------------
# Banco de tubos
# ---------------------------------------------------------------------


def test_default_is_the_incropera_bank() -> None:
    at = _app()
    assert [e.label for e in at.expander] == ["📖 Fórmulas teóricas", "🔬 Procedimiento"]
    name = list(tb.TUBE_BANK_EXAMPLES)[0]
    r = tb.solve_tube_bank(tb.TUBE_BANK_EXAMPLES[name].inputs)
    assert _metric(at, "h [") == pytest.approx(r.h_W_per_m2K, rel=1e-3)
    assert _metric(at, "Q̇ [") == pytest.approx(r.Q_W, rel=1e-3)
    assert _metric(at, "Δp [") == pytest.approx(r.dp_Pa, rel=1e-3)
    assert len(at.get("plotly_chart")) == 2
    assert len(at.get("download_button")) == 2
    tex = _latex(at)
    assert r"N_L\,\chi\,f" in tex  # la teoría
    assert r"\mathrm{Nu}_{D,N_L}" in tex  # el procedimiento


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("index", range(len(tb.TUBE_BANK_EXAMPLES)))
def test_every_bank_example(index: int, system: str) -> None:
    at = _app()
    _system(at, system)
    name = list(tb.TUBE_BANK_EXAMPLES)[index]
    at.selectbox(key="ib_example").set_value(name).run()
    _no_problems(at)
    r = tb.solve_tube_bank(tb.TUBE_BANK_EXAMPLES[name].inputs)
    Q = convert_from_si(r.Q_W, "heat_rate", system)  # type: ignore[arg-type]
    assert _metric(at, "Q̇ [") == pytest.approx(Q, rel=1e-3)
    assert len(at.get("plotly_chart")) == 2


def test_bank_arrangement_source_and_comparison() -> None:
    at = _app()
    i = tb.TUBE_BANK_EXAMPLES[list(tb.TUBE_BANK_EXAMPLES)[0]].inputs
    at.radio(key="ib_0_arr").set_value("aligned").run()
    _no_problems(at)
    aligned = tb.solve_tube_bank(replace(i, arrangement="aligned"))
    assert _metric(at, "h [") == pytest.approx(aligned.h_W_per_m2K, rel=1e-3)
    at.radio(key="ib_0_src").set_value("cengel").run()
    _no_problems(at)
    cengel = tb.solve_tube_bank(replace(i, arrangement="aligned", source="cengel"))
    assert _metric(at, "h [") == pytest.approx(cengel.h_W_per_m2K, rel=1e-3)
    # la tabla de los dos arreglos
    frames = [df.value for df in at.dataframe]
    assert any("Arreglo" in f.columns for f in frames)


def test_bank_other_fluid_starts_with_a_pressure_that_works() -> None:
    at = _app()
    at.selectbox(key="ib_0_fluid").set_value("Water").run()
    _no_problems(at)
    i = tb.TUBE_BANK_EXAMPLES[list(tb.TUBE_BANK_EXAMPLES)[0]].inputs
    p = suggested_pressure("Water", i.T_in_K, i.T_s_K, i.p_Pa)
    assert p > 1.0e5  # el agua tiene que seguir líquida a 70 °C
    key = _wkey(at, "ib_0_Water_p")
    assert at.number_input(key=key).value == pytest.approx(
        convert_from_si(p, "pressure", _current_system(at)), rel=1e-6
    )


def test_bank_errors() -> None:
    at = _app()
    at.number_input(key=_wkey(at, "ib_0_ST")).set_value(15.0).run()  # mm: menor que D
    assert at.error and "se tocan" in at.error[0].value


# ---------------------------------------------------------------------
# Doble tubo
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("index", range(len(dp.DOUBLE_PIPE_EXAMPLES)))
def test_every_double_pipe_example(index: int, system: str) -> None:
    at = _app()
    _system(at, system)
    _mode(at, M_DP)
    name = list(dp.DOUBLE_PIPE_EXAMPLES)[index]
    at.selectbox(key="id_example").set_value(name).run()
    _no_problems(at)
    r = dp.solve_double_pipe(dp.DOUBLE_PIPE_EXAMPLES[name].inputs)
    U = convert_from_si(r.U_o_W_per_m2K, "heat_transfer_coefficient", system)  # type: ignore[arg-type]
    assert _metric(at, "U_o [") == pytest.approx(U, rel=1e-3)
    assert _metric(at, "Horquillas") == r.hairpins
    assert len(at.get("plotly_chart")) == 2  # las resistencias y el perfil
    assert len(at.get("download_button")) == 2


def test_double_pipe_incropera_numbers() -> None:
    at = _app()
    _system(at, "SI")
    _mode(at, M_DP)
    assert _metric(at, "U_o [") == pytest.approx(38.27, abs=0.02)  # el libro: 38,1
    assert _metric(at, "Largo necesario") == pytest.approx(65.65, abs=0.05)  # el libro: 65,9
    assert _metric(at, "Horquillas") == 6
    assert "Kays y Perkins" in "\n".join(m.value for m in at.markdown)


def test_double_pipe_rating_and_targets() -> None:
    at = _app()
    _mode(at, M_DP)
    at.radio(key="id_0_problem").set_value("rating").run()
    _no_problems(at)
    # arranca con las horquillas que hacían falta: las salidas casi como las pedidas
    assert at.number_input(key="id_0_hp").value == 6
    assert any(m.label.startswith("Efectividad") for m in at.metric)
    at.radio(key="id_0_problem").set_value("sizing").run()
    at.radio(key="id_0_target").set_value("T_cold_out").run()
    _no_problems(at)
    i = dp.DOUBLE_PIPE_EXAMPLES[list(dp.DOUBLE_PIPE_EXAMPLES)[0]].inputs
    base = dp.solve_double_pipe(i)
    system = _current_system(at)
    value = at.number_input(key=_wkey(at, "id_0_T_cold_out")).value
    assert value == pytest.approx(convert_from_si(base.T_cold_out_K, "temperature", system), 1e-4)


def test_double_pipe_swap_and_fluid_change() -> None:
    at = _app()
    _mode(at, M_DP)
    at.radio(key="id_0_side").set_value("hot").run()
    _no_problems(at)
    assert any(m.label.startswith("h_i (tubo)") for m in at.metric)
    # el aceite (propiedades del libro) pasa a agua de CoolProp, con una presión que sirve
    at.selectbox(key="id_0_h_fluid").set_value("Water").run()
    _no_problems(at)
    assert at.number_input(key=_wkey(at, "id_0_h_Water_p")).value > 1.0


def test_double_pipe_errors() -> None:
    at = _app()
    _mode(at, M_DP)
    at.number_input(key=_wkey(at, "id_0_D")).set_value(20.0).run()  # mm: menor que d_o
    assert at.error


# ---------------------------------------------------------------------
# Casco y tubos (Kern)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("index", range(len(stt.SHELL_TUBE_EXAMPLES)))
def test_every_shell_tube_example(index: int, system: str) -> None:
    at = _app()
    _system(at, system)
    _mode(at, M_ST)
    name = list(stt.SHELL_TUBE_EXAMPLES)[index]
    at.selectbox(key="ik_example").set_value(name).run()
    _no_problems(at)
    r = stt.solve_shell_tube(stt.SHELL_TUBE_EXAMPLES[name].inputs)
    h = convert_from_si(r.shell.h_W_per_m2K, "heat_transfer_coefficient", system)  # type: ignore[arg-type]
    assert _metric(at, "h_o (casco") == pytest.approx(h, rel=1e-3)
    assert _metric(at, "Tubos N_t") == r.N_t
    # el barrido de los bafles y las resistencias; el perfil, solo con 1 paso (contracorriente)
    charts = 3 if r.inputs.tube_passes == 1 else 2
    assert len(at.get("plotly_chart")) == charts
    assert len(at.get("download_button")) == 2


def test_shell_tube_peters_kern() -> None:
    at = _app()
    _system(at, "SI")
    _mode(at, M_ST)
    assert _metric(at, "h_o (casco") == pytest.approx(3253, rel=2e-3)  # el libro: 3369 (−3 %)
    assert _metric(at, "Δp del casco") == pytest.approx(18_700, rel=0.01)  # Pa; el libro: 16,4 kPa


def test_shell_tube_check_verdicts() -> None:
    at = _app()
    _mode(at, M_ST)
    name = "Enfriador de kerosene con agua de río"
    at.selectbox(key="ik_example").set_value(name).run()
    _no_problems(at)
    assert at.success and "El equipo sirve" in at.success[0].value
    idx = list(stt.SHELL_TUBE_EXAMPLES).index(name)
    at.number_input(key=_wkey(at, f"ik_{idx}_L")).set_value(5.0).run()
    at.number_input(key=f"ik_{idx}_Nb").set_value(32).run()
    _no_problems(at)
    assert any("no tolera" in w.value for w in at.warning)
    at.number_input(key=_wkey(at, f"ik_{idx}_L")).set_value(2.0).run()
    at.number_input(key=f"ik_{idx}_Nb").set_value(12).run()
    _no_problems(at)
    assert any("Ni limpio alcanza" in w.value for w in at.warning)


def test_shell_tube_tube_count_and_bwg() -> None:
    at = _app()
    _mode(at, M_ST)
    at.radio(key="ik_0_ntmode").set_value("La cuento (dato)").run()
    _no_problems(at)
    at.number_input(key="ik_0_nt").set_value(300).run()
    _no_problems(at)
    assert _metric(at, "Tubos N_t") == 300
    at.selectbox(key="ik_0_bwg").set_value(0).run()  # d_i a elección
    _no_problems(at)
    assert at.number_input(key=_wkey(at, "ik_0_di")).value > 0
    at.selectbox(key="ik_0_bwg").set_value(12).run()
    _no_problems(at)


def test_shell_tube_errors() -> None:
    at = _app()
    _mode(at, M_ST)
    at.number_input(key=_wkey(at, "ik_0_PT")).set_value(19.0).run()  # mm: igual a d_o
    assert at.error and "se tocan" in at.error[0].value


# ---------------------------------------------------------------------
# Casco y tubos con Bell–Delaware (Fase 12.2)
# ---------------------------------------------------------------------

PETERS = "Agua por el casco (Peters, Timmerhaus y West, ejemplo 14-6)"
KEROSENE = "Enfriador de kerosene con agua de río"


def _bell_delaware(at: AppTest, name: str | None = None) -> tuple[AppTest, int]:
    _mode(at, M_ST)
    if name is not None:
        at.selectbox(key="ik_example").set_value(name).run()
        _no_problems(at)
    idx = list(stt.SHELL_TUBE_EXAMPLES).index(at.selectbox(key="ik_example").value)
    at.radio(key=f"ik_{idx}_method").set_value("bell_delaware").run()
    _no_problems(at)
    return at, idx


def _bd_result(name: str, **geometry: object) -> stt.ShellTubeResult:
    inputs = replace(stt.SHELL_TUBE_EXAMPLES[name].inputs, shell_method="bell_delaware")
    if geometry:
        assert inputs.bell_delaware is not None
        inputs = replace(inputs, bell_delaware=replace(inputs.bell_delaware, **geometry))
    return stt.solve_shell_tube(inputs)


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("index", range(len(stt.SHELL_TUBE_EXAMPLES)))
def test_every_shell_tube_example_with_bell_delaware(index: int, system: str) -> None:
    at = _app()
    _system(at, system)
    name = list(stt.SHELL_TUBE_EXAMPLES)[index]
    _bell_delaware(at, name)
    r = _bd_result(name)
    h = convert_from_si(r.h_shell_W_per_m2K, "heat_transfer_coefficient", system)  # type: ignore[arg-type]
    dp_s = convert_from_si(r.dp_shell_Pa, "pressure_loss", system)  # type: ignore[arg-type]
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(h, rel=1e-3)
    assert _metric(at, "Δp del casco") == pytest.approx(dp_s, rel=1e-3)
    # la cascada, las zonas, los dos barridos y las resistencias; el perfil, con 1 paso
    charts = 6 if r.inputs.tube_passes == 1 else 5
    assert len(at.get("plotly_chart")) == charts
    frames = [df.value for df in at.dataframe]
    assert any("Kern contra BD" in f.columns for f in frames)
    titles = [m.value for m in at.markdown]
    for title in ("#### El lado del casco (Bell–Delaware)", "#### Kern y Bell–Delaware"):
        assert title in titles
    assert "#### El lado del casco (Kern)" not in titles
    tex = _latex(at)
    assert r"h_o = h_{\text{ideal}}\,J_c\,J_l\,J_b\,J_s\,J_r" in tex  # la teoría
    assert r"\Delta p_c + \Delta p_w + \Delta p_e" in tex  # el procedimiento


def test_bell_delaware_peters_numbers() -> None:
    at = _app()
    _system(at, "SI")
    _bell_delaware(at)
    # el libro: 3080 W/(m²·K) (+4 %) y 9,48 kPa (el Δp no se reproduce: ver la nota del ejemplo)
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(3216, rel=2e-3)
    assert _metric(at, "Δp del casco") == pytest.approx(5912, rel=0.01)  # Pa
    frames = [df.value for df in at.dataframe]
    geometry = next(
        f for f in frames if "Símbolo" in f.columns and "S_m (flujo cruzado)" in set(f["Símbolo"])
    )
    row = geometry[geometry["Símbolo"] == "S_m (flujo cruzado)"].iloc[0]
    assert float(row["Valor"].replace(",", ".")) == pytest.approx(0.02549, rel=1e-3)
    # la comparación con Kern: Kern da más Δp
    table = next(f for f in frames if "Kern contra BD" in f.columns)
    dp_row = table[table["Magnitud"].str.startswith("Δp del casco")].iloc[0]
    assert dp_row["Kern contra BD"].startswith("+")


def test_bell_delaware_inputs_change_the_result() -> None:
    at = _app()
    _system(at, "Técnico")
    at, idx = _bell_delaware(at, KEROSENE)
    key = f"ik_{idx}"
    base = _bd_result(KEROSENE)
    h0 = _metric(at, "h_o (casco, Bell–Delaware)")
    assert h0 == pytest.approx(base.h_shell_W_per_m2K, rel=1e-3)
    # dos pares de tiras: el bypass baja menos el h_o
    at.number_input(key=f"{key}_Nss").set_value(2).run()
    _no_problems(at)
    strips = _bd_result(KEROSENE, sealing_strips=2)
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(
        strips.h_shell_W_per_m2K, rel=1e-3
    )
    assert strips.h_shell_W_per_m2K > base.h_shell_W_per_m2K
    # el corte
    at.number_input(key=f"{key}_cut").set_value(35.0).run()
    _no_problems(at)
    cut = _bd_result(KEROSENE, sealing_strips=2, baffle_cut=0.35)
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(
        cut.h_shell_W_per_m2K, rel=1e-3
    )
    # holguras a elección: arrancan con las de TEMA (mismo resultado)
    at.radio(key=f"{key}_clr").set_value("A elección").run()
    _no_problems(at)
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(
        cut.h_shell_W_per_m2K, rel=1e-3
    )
    at.number_input(key=_wkey(at, f"{key}_Lsb")).set_value(1.0).run()  # mm
    _no_problems(at)
    tight = _bd_result(KEROSENE, sealing_strips=2, baffle_cut=0.35, L_sb_m=0.001)
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(
        tight.h_shell_W_per_m2K, rel=1e-3
    )
    # separaciones de los extremos: arrancan en 1,5·B
    at.checkbox(key=f"{key}_ends").check().run()
    _no_problems(at)
    B = stt.SHELL_TUBE_EXAMPLES[KEROSENE].inputs.B_m
    ends = _bd_result(
        KEROSENE,
        sealing_strips=2,
        baffle_cut=0.35,
        L_sb_m=0.001,
        L_tb_m=tight.bell_delaware.areas.L_tb_m,  # type: ignore[union-attr]
        B_in_m=1.5 * B,
        B_out_m=1.5 * B,
    )
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(
        ends.h_shell_W_per_m2K, rel=1e-3
    )
    frames = [df.value for df in at.dataframe]
    coef = next(
        f for f in frames if "Símbolo" in f.columns and "J_s (extremos)" in set(f["Símbolo"])
    )
    J_s = coef[coef["Símbolo"] == "J_s (extremos)"].iloc[0]["Valor"]
    fc = ends.bell_delaware.factors  # type: ignore[union-attr]
    assert float(J_s.replace(",", ".")) == pytest.approx(fc.J_s, abs=1e-4) and fc.J_s < 1.0


def test_bell_delaware_rotated_square() -> None:
    at = _app()
    at, idx = _bell_delaware(at)
    at.radio(key=f"ik_{idx}_layout").set_value("rotated_square").run()
    _no_problems(at)
    r = _bd_result(PETERS)
    rotated = stt.solve_shell_tube(replace(r.inputs, layout="rotated_square"))
    system = _current_system(at)
    h = convert_from_si(rotated.h_shell_W_per_m2K, "heat_transfer_coefficient", system)  # type: ignore[arg-type]
    assert _metric(at, "h_o (casco, Bell–Delaware)") == pytest.approx(h, rel=1e-3)
    assert "arreglo de 45°" in "\n".join(m.value for m in at.markdown)
    assert len(at.get("download_button")) == 2


def test_bell_delaware_back_to_kern() -> None:
    at = _app()
    at, idx = _bell_delaware(at)
    at.radio(key=f"ik_{idx}_method").set_value("kern").run()
    _no_problems(at)
    titles = [m.value for m in at.markdown]
    assert "#### El lado del casco (Kern)" in titles
    assert "#### Kern y Bell–Delaware" not in titles
    assert any(m.label.startswith("h_o (casco, Kern)") for m in at.metric)


def test_bell_delaware_errors() -> None:
    at = _app()
    at, idx = _bell_delaware(at)
    at.checkbox(key=f"ik_{idx}_ends").check().run()
    at.number_input(key=_wkey(at, f"ik_{idx}_Bin")).set_value(0.0).run()
    assert at.error and "separaciones" in at.error[0].value
