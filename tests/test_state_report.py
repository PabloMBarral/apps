"""Tests de :mod:`core.state_report` — reporte didáctico del estado.

Cubre las tablas (propiedades y saturación) en los tres sistemas de
unidades, el resumen de región, las advertencias didácticas, el
procedimiento paso a paso (ramas por par y por región) y la
exportación JSON/CSV. El LaTeX generado se validó además con KaTeX
(el motor de ``st.latex``) sobre toda la grilla fluido × par × sistema;
acá se chequea en CI que las llaves estén balanceadas.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from core.fluids import (
    PAIR_KWARGS,
    SUPPORTED_FLUIDS,
    FluidState,
    fluid_state_from_pair,
    saturation_at_pressure,
    suggested_inputs,
)
from core.state_report import (
    INPUT_SPECS,
    PAIR_LABELS_ES,
    PAIR_ORDER,
    build_procedure,
    format_value,
    input_value_si,
    latex_number,
    latex_unit,
    property_rows,
    pv_energy_factor,
    region_summary,
    saturation_rows,
    state_notes,
    state_to_csv,
    state_to_dict,
)

WATER = "Water"


@pytest.fixture(scope="module")
def superheated() -> FluidState:
    """Cengel A-6: 1 MPa, 300 °C."""
    return fluid_state_from_pair(WATER, "TP", t=573.15, p=1.0e6)


@pytest.fixture(scope="module")
def wet() -> FluidState:
    """10 bar, h = 2000 kJ/kg → x ≈ 0.614."""
    return fluid_state_from_pair(WATER, "PH", p=1.0e6, h=2.0e6)


@pytest.fixture(scope="module")
def compressed() -> FluidState:
    """Cengel A-7: 5 MPa, 100 °C."""
    return fluid_state_from_pair(WATER, "TP", t=373.15, p=5.0e6)


def _by_symbol(rows: list) -> dict:
    return {r.symbol: r for r in rows}


# ---------------------------------------------------------------------
# Metadatos
# ---------------------------------------------------------------------


class TestMetadata:
    def test_pair_order_covers_every_pair_once(self) -> None:
        assert sorted(PAIR_ORDER) == sorted(PAIR_KWARGS)
        assert set(PAIR_LABELS_ES) == set(PAIR_KWARGS)

    def test_every_kwarg_has_input_spec(self) -> None:
        kwargs = {kw for pair in PAIR_KWARGS.values() for kw in pair}
        assert kwargs <= set(INPUT_SPECS)
        assert INPUT_SPECS["x"].kind is None

    def test_input_value_si(self, wet: FluidState, superheated: FluidState) -> None:
        assert input_value_si(wet, "p") == pytest.approx(1.0e6)
        assert input_value_si(wet, "x") == pytest.approx(wet.x)
        with pytest.raises(ValueError, match="título"):
            input_value_si(superheated, "x")
        with pytest.raises(ValueError, match="desconocida"):
            input_value_si(wet, "q")


# ---------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------


class TestPropertyRows:
    def test_superheated_values_tecnico(self, superheated: FluidState) -> None:
        rows = _by_symbol(property_rows(superheated, "Técnico"))
        assert rows["T"].value == pytest.approx(300.0)
        assert rows["T"].unit == "°C"
        assert rows["p"].value == pytest.approx(10.0)
        assert rows["v"].value == pytest.approx(0.25799, rel=2e-4)
        assert rows["h"].value == pytest.approx(3051.6, abs=0.15)
        assert rows["h"].unit == "kJ/kg"

    def test_superheat_row_and_no_subcooling(self, superheated: FluidState) -> None:
        rows = _by_symbol(property_rows(superheated, "Técnico"))
        assert rows["T − T_sat(p)"].value == pytest.approx(120.12, abs=0.02)
        assert "T_sat(p) − T" not in rows

    def test_superheat_in_fahrenheit_has_no_offset(self, superheated: FluidState) -> None:
        rows = _by_symbol(property_rows(superheated, "Inglés"))
        assert rows["T − T_sat(p)"].value == pytest.approx(120.12 * 1.8, abs=0.04)
        assert rows["T − T_sat(p)"].unit == "°F"

    def test_quality_undefined_outside_dome(self, superheated: FluidState) -> None:
        x = _by_symbol(property_rows(superheated, "Técnico"))["x"]
        assert x.value is None
        assert "campana" in x.note

    def test_subcooling_row(self, compressed: FluidState) -> None:
        rows = _by_symbol(property_rows(compressed, "Técnico"))
        assert rows["T_sat(p) − T"].value == pytest.approx(163.94, abs=0.02)

    def test_mixture_hides_single_phase_properties(self, wet: FluidState) -> None:
        rows = _by_symbol(property_rows(wet, "Técnico"))
        assert rows["x"].value == pytest.approx(wet.x)
        for sym in ("c_p", "c_v", "γ = c_p/c_v", "w", "μ", "k", "Pr = μ·c_p/k"):
            assert rows[sym].value is None
            assert "campana" in rows[sym].note

    def test_si_system(self, superheated: FluidState) -> None:
        rows = _by_symbol(property_rows(superheated, "SI"))
        assert rows["h"].value == pytest.approx(superheated.h_J_per_kg)
        assert rows["μ"].unit == "Pa·s"

    def test_transport_rows_present_for_liquid(self, compressed: FluidState) -> None:
        rows = _by_symbol(property_rows(compressed, "Técnico"))
        assert rows["Pr = μ·c_p/k"].value == pytest.approx(compressed.prandtl)
        assert rows["k"].unit == "W/(m·K)"


class TestSaturationRows:
    def test_pure_fluid_at_pressure(self) -> None:
        rows = saturation_rows(saturation_at_pressure(WATER, 1.0e6), "Técnico")
        by = _by_symbol(rows)
        assert len(rows) == 14
        assert [r.symbol for r in rows[:2]] == ["T_sat", "p_sat"]
        assert by["p_sat"].value == pytest.approx(10.0)
        assert by["T_sat"].value == pytest.approx(179.88, abs=0.01)
        assert by["h_f"].value == pytest.approx(762.51, abs=0.15)
        assert by["h_fg"].value == pytest.approx(2014.6, abs=0.15)
        assert by["h_g"].value == pytest.approx(2777.1, abs=0.15)
        assert by["s_g"].unit == "kJ/(kg·K)"

    def test_pure_fluid_at_temperature(self, compressed: FluidState) -> None:
        by = _by_symbol(saturation_rows(compressed.sat_at_T, "Técnico"))
        assert by["T_sat"].value == pytest.approx(100.0)
        assert by["p_sat"].value == pytest.approx(1.0142, abs=1e-4)

    def test_pseudo_pure_shows_bubble_and_dew(self) -> None:
        rows = saturation_rows(saturation_at_pressure("Air", 1.0e5), "SI")
        symbols = [r.symbol for r in rows]
        assert symbols[:3] == ["T_f", "T_g", "p_sat"]
        assert rows[1].value - rows[0].value > 2.0

    def test_pseudo_pure_at_temperature_shows_bubble_and_dew_pressures(self) -> None:
        from core.fluids import saturation_at_temperature

        rows = saturation_rows(saturation_at_temperature("Air", 90.0), "SI")
        assert [r.symbol for r in rows[:3]] == ["T_sat", "p_f", "p_g"]
        assert rows[1].value > rows[2].value  # p_burbuja > p_rocío a T fija


class TestFormatValue:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, "—"),
            (float("nan"), "—"),
            (0.0, "0"),
            (3051.632, "3051.63"),
            (0.001043153, "0.00104315"),
        ],
    )
    def test_format(self, value: float | None, expected: str) -> None:
        assert format_value(value) == expected


# ---------------------------------------------------------------------
# Resumen y notas
# ---------------------------------------------------------------------


class TestRegionSummary:
    def test_superheated(self, superheated: FluidState) -> None:
        text = region_summary(superheated, "Técnico")
        assert "Vapor sobrecalentado" in text
        assert "120.12 °C por encima" in text

    def test_wet(self, wet: FluidState) -> None:
        text = region_summary(wet, "Técnico")
        assert "Vapor húmedo" in text
        assert "61.4 %" in text

    def test_compressed(self, compressed: FluidState) -> None:
        assert "por debajo de la saturación" in region_summary(compressed, "Técnico")

    def test_supercritical(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=773.15, p=3.0e7)
        assert "supercrítico" in region_summary(state, "Técnico")

    def test_saturated_ends(self) -> None:
        liq = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=0.0)
        vap = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=1.0)
        assert "Líquido saturado" in region_summary(liq, "SI")
        assert "Vapor saturado seco" in region_summary(vap, "SI")


class TestStateNotes:
    def test_tp_on_saturation_line_warns(self) -> None:
        # El clásico "100 °C y 1 atm": queda a 0.03 K de la saturación.
        state = fluid_state_from_pair(WATER, "TP", t=373.15, p=101_325.0)
        notes = state_notes(state, "TP", "Técnico")
        assert any(n.kind == "warning" and "no son independientes" in n.text for n in notes)

    def test_same_state_with_other_pair_does_not_warn(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=373.15, p=101_325.0)
        notes = state_notes(state, "PH", "Técnico")
        assert not any("no son independientes" in n.text for n in notes)

    def test_near_critical_point(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=650.0, p=22.5e6)
        notes = state_notes(state, "TP", "Técnico")
        assert any(n.kind == "warning" and "punto crítico" in n.text for n in notes)

    def test_supercritical_info(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=773.15, p=3.0e7)
        assert any("no existe la campana" in n.text for n in state_notes(state, "TP", "SI"))

    def test_low_pressure_vapor_is_ideal_gas(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=573.15, p=1.0e3)
        assert any("gas ideal" in n.text for n in state_notes(state, "TP", "SI"))

    def test_dense_vapor_is_not_ideal_gas(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=673.15, p=15.0e6)
        assert any("error de" in n.text for n in state_notes(state, "TP", "SI"))

    def test_pseudo_pure_info(self) -> None:
        state = fluid_state_from_pair("Air", "TP", t=298.15, p=1.0e5)
        assert any("pseudo-puro" in n.text for n in state_notes(state, "TP", "SI"))

    def test_plain_wet_steam_has_no_notes(self, wet: FluidState) -> None:
        assert state_notes(wet, "PH", "Técnico") == []


# ---------------------------------------------------------------------
# Procedimiento
# ---------------------------------------------------------------------


def _titles(steps: list) -> list[str]:
    return [s.title for s in steps]


def _all_latex(steps: list) -> str:
    return "\n".join(t for s in steps for t in s.latex)


class TestProcedure:
    def test_wet_steam_from_p_h(self, wet: FluidState) -> None:
        steps = build_procedure(wet, "PH", "Técnico")
        titles = _titles(steps)
        assert titles[0] == "Datos"
        assert "Ubicación del estado" in titles
        assert "Título y demás propiedades (regla de la palanca)" in titles
        assert titles[-1] == "Verificación: h = u + p·v"
        tex = _all_latex(steps)
        assert r"x = \frac{h - h_f}{h_g - h_f}" in tex
        assert "0.61426" in tex
        assert r"\le h = 2000 \le" in tex

    def test_quality_given_uses_lever_rule(self) -> None:
        state = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=0.5)
        steps = build_procedure(state, "PX", "Técnico")
        assert "Regla de la palanca" in _titles(steps)
        tex = _all_latex(steps)
        for y in ("v", "u", "h", "s"):
            assert rf"{y} = {y}_f + x\,({y}_g - {y}_f)" in tex

    def test_compressed_liquid_includes_incompressible_approximation(
        self, compressed: FluidState
    ) -> None:
        steps = build_procedure(compressed, "TP", "Técnico")
        assert "Aproximación de líquido incompresible" in _titles(steps)
        tex = _all_latex(steps)
        assert r"p = 50\ \mathrm{bar} > p_{\mathrm{sat}}(T)" in tex
        assert r"h \approx h_f(T) + v_f(T)\,[p - p_{\mathrm{sat}}(T)]" in tex

    def test_superheated_includes_ideal_gas_check(self, superheated: FluidState) -> None:
        steps = build_procedure(superheated, "TP", "Técnico")
        assert "¿Gas ideal?" in _titles(steps)
        assert "0.97531" in _all_latex(steps)

    def test_mollier_wet_steam_quality_agrees(self) -> None:
        state = fluid_state_from_pair(WATER, "HS", h=2.0e6, s=5.0e3)
        steps = build_procedure(state, "HS", "Técnico")
        assert _titles(steps)[1] == "Diagrama de Mollier"
        x = latex_number(state.x, 5)
        tex = _all_latex(steps)
        assert tex.count(x) >= 2  # x despejado con h y con s

    def test_above_critical_pressure(self) -> None:
        state = fluid_state_from_pair(WATER, "PH", p=3.0e7, h=3.0e6)
        assert "Fuera del rango de saturación" in _titles(build_procedure(state, "PH", "SI"))

    def test_above_critical_temperature(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=700.0, p=1.0e7)
        assert "Fuera del rango de saturación" in _titles(build_procedure(state, "TP", "SI"))

    def test_ingles_units_and_pv_factor(self, superheated: FluidState) -> None:
        tex = _all_latex(build_procedure(superheated, "TP", "Inglés"))
        assert r"\mathrm{Btu/lb}" in tex
        assert "0.18505" in tex

    def test_unknown_pair(self, wet: FluidState) -> None:
        with pytest.raises(ValueError, match="no soportado"):
            build_procedure(wet, "XY", "SI")  # type: ignore[arg-type]

    @pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
    @pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
    def test_every_default_state_builds_balanced_latex(self, fluid: str, system: str) -> None:
        for pair in PAIR_ORDER:
            state = fluid_state_from_pair(fluid, pair, **suggested_inputs(fluid, pair))
            steps = build_procedure(state, pair, system)  # type: ignore[arg-type]
            assert steps[0].title == "Datos"
            assert steps[-1].title.startswith("Verificación")
            for step in steps:
                for tex in step.latex:
                    assert tex.count("{") == tex.count("}"), (fluid, pair, step.title, tex)


class TestLatexHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, "0"),
            (1.0142e5, r"1.0142\times 10^{5}"),
            (0.0010432, "0.0010432"),
            (2000.0, "2000"),
        ],
    )
    def test_latex_number(self, value: float, expected: str) -> None:
        assert latex_number(value) == expected

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("kJ/(kg·K)", r"\mathrm{kJ/(kg\cdot K)}"),
            ("°C", r"\mathrm{{}^{\circ}C}"),
            ("m³/kg", r"\mathrm{m^{3}/kg}"),
            ("m²/s", r"\mathrm{m^{2}/s}"),
        ],
    )
    def test_latex_unit(self, label: str, expected: str) -> None:
        assert latex_unit(label) == expected

    @pytest.mark.parametrize(
        "system,expected", [("SI", 1.0), ("Técnico", 100.0), ("Inglés", 0.185050)]
    )
    def test_pv_energy_factor(self, system: str, expected: float) -> None:
        # 1 psia·ft³/lb = 144 lbf·ft/lb = 144/778.169 Btu/lb.
        assert pv_energy_factor(system) == pytest.approx(expected, rel=1e-5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------


class TestExport:
    def test_dict_is_json_serializable(self, wet: FluidState) -> None:
        data = state_to_dict(wet, "PH", "Técnico")
        text = json.dumps(data, ensure_ascii=False)
        assert '"region": "saturated_mixture"' in text
        assert data["datos"]["p"] == {"valor": pytest.approx(10.0), "unidad": "bar"}
        assert data["datos"]["h"]["unidad"] == "kJ/kg"
        assert data["si"]["P_Pa"] == pytest.approx(1.0e6)
        assert data["saturacion_a_p"] is not None

    def test_dict_for_supercritical_has_no_saturation(self) -> None:
        state = fluid_state_from_pair(WATER, "TP", t=773.15, p=3.0e7)
        data = state_to_dict(state, "TP", "SI")
        assert data["saturacion_a_p"] is None and data["saturacion_a_T"] is None

    def test_quality_input_is_dimensionless(self) -> None:
        state = fluid_state_from_pair(WATER, "PX", p=1.0e5, x=0.25)
        assert state_to_dict(state, "PX", "SI")["datos"]["x"] == {"valor": 0.25, "unidad": "-"}

    def test_csv_round_trip(self, superheated: FluidState) -> None:
        text = state_to_csv(superheated, "TP", "Técnico")
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0][0] == "# fluido"
        assert rows[1] == ["grupo", "propiedad", "simbolo", "valor", "unidad", "nota"]
        temperature = next(r for r in rows[2:] if r[2] == "T")
        assert float(temperature[3]) == pytest.approx(300.0)
        # Incluye las dos tablas de saturación (a p y a T).
        assert any(r[0] == "Saturación a p" for r in rows)
        assert any(r[0] == "Saturación a T" for r in rows)


# ---------------------------------------------------------------------
# Tabla de varios estados
# ---------------------------------------------------------------------


class TestStatesTable:
    def test_rows_in_tecnico(self, superheated: FluidState, wet: FluidState) -> None:
        from core.state_report import states_table

        rows = states_table([("1", superheated), ("2", wet)], "Técnico")
        assert [r["Estado"] for r in rows] == ["1", "2"]
        assert rows[0]["T [°C]"] == pytest.approx(300.0)
        assert rows[0]["h [kJ/kg]"] == pytest.approx(3051.6, abs=0.15)
        assert rows[0]["x [-]"] is None
        assert rows[1]["x [-]"] == pytest.approx(wet.x)
        assert rows[1]["Región"].startswith("Vapor húmedo")
        assert rows[0]["Fluido"] == "Agua"

    def test_headers_follow_unit_system(self, superheated: FluidState) -> None:
        from core.state_report import states_table

        row = states_table([("A", superheated)], "Inglés")[0]
        assert "T [°F]" in row and "s [Btu/(lb·°R)]" in row

    def test_csv(self, superheated: FluidState, wet: FluidState) -> None:
        from core.state_report import states_table_csv

        text = states_table_csv([("1", superheated), ("2", wet)], "Técnico")
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 2
        assert rows[0]["x [-]"] == ""  # fuera de la campana
        assert float(rows[1]["p [bar]"]) == pytest.approx(10.0)

    def test_csv_of_empty_table(self) -> None:
        from core.state_report import states_table_csv

        assert states_table_csv([], "SI") == ""
