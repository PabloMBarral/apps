"""Tests de la exportación genérica (:mod:`core.export`) y de los dicts de
Interpolación e ISO 6976 que alimentan los botones de descarga."""

from __future__ import annotations

import csv
import io
import json

import pytest

from core.combustion.iso6976 import (
    GasComponent,
    ISO6976Inputs,
    ReferenceCondition,
    calculate,
    iso6976_to_dict,
)
from core.export import dict_to_csv, flatten
from core.interpolation import bilinear, interpolation_to_dict, linear_from_table


class TestFlatten:
    def test_nested_dicts_and_lists(self) -> None:
        data = {"a": {"b": 1}, "c": [{"d": 2}, {"d": 3}], "e": [1, 2], "f": None}
        assert flatten(data) == [
            ("a.b", 1),
            ("c[1].d", 2),
            ("c[2].d", 3),
            ("e", "1; 2"),
            ("f", None),
        ]

    def test_csv_has_header_and_empty_none(self) -> None:
        rows = list(csv.reader(io.StringIO(dict_to_csv({"x": 1.5, "y": None}))))
        assert rows == [["campo", "valor"], ["x", "1.5"], ["y", ""]]


class TestInterpolationExport:
    def test_linear(self) -> None:
        # Cengel A-4: h_f(100 °C) = 419.06, h_f(110 °C) = 461.42.
        result = linear_from_table(105.0, [100.0, 110.0], [419.06, 461.42])
        data = interpolation_to_dict(result, x_label="T [°C]", y_label="h_f [kJ/kg]")
        assert data["tipo"] == "lineal"
        assert data["variables"] == {"x": "T [°C]", "y": "h_f [kJ/kg]"}
        assert data["consulta"] == {"x": 105.0}
        assert data["nodos"] == {"x0": 100.0, "y0": 419.06, "x1": 110.0, "y1": 461.42}
        assert data["resultado"] == pytest.approx(440.24)
        json.dumps(data)  # serializable

    def test_bilinear_has_vertices_and_intermediates(self) -> None:
        result = bilinear(225.0, 2.0, 200.0, 250.0, 1.0, 3.0, 2875.5, 2865.6, 2974.5, 2967.7)
        data = interpolation_to_dict(result, z_label="h [kJ/kg]")
        assert data["tipo"].startswith("doble entrada")
        assert set(data["vertices"]) == {"x0", "x1", "y0", "y1", "z00", "z01", "z10", "z11"}
        assert set(data["intermedios"]) == {"z_en_y0", "z_en_y1"}
        assert "procedimiento.explicacion" in dict(flatten(data))

    def test_exact_node_omits_missing_fields(self) -> None:
        result = linear_from_table(100.0, [100.0, 110.0], [419.06, 461.42])
        data = interpolation_to_dict(result)
        assert data["nodo_exacto"] is True
        assert data["resultado"] == pytest.approx(419.06)


@pytest.fixture(scope="module")
def iso_data() -> dict:
    inputs = ISO6976Inputs(
        composition=[
            GasComponent("methane", 0.9, 0.0),
            GasComponent("ethane", 0.06, 0.0),
            GasComponent("nitrogen", 0.04, 0.0),
        ],
        combustion_reference=ReferenceCondition(15.0),
        metering_reference=ReferenceCondition(15.0),
        correlation_matrix="identity",
    )
    return iso6976_to_dict(calculate(inputs))


class TestISO6976Export:
    @pytest.fixture
    def data(self, iso_data: dict) -> dict:
        return iso_data

    def test_structure(self, data: dict) -> None:
        assert data["norma"] == "ISO 6976:2016"
        assert len(data["composicion"]) == 3
        assert [q["magnitud"].split()[0] for q in data["intermedios"]] == ["M", "s", "Z", "V_m"]
        assert len(data["resultados"]) == 10
        assert data["resultados"][0]["unidad"] == "kJ/mol"
        json.dumps(data, ensure_ascii=False)

    def test_csv(self, data: dict) -> None:
        text = dict_to_csv(data)
        assert "resultados[9].magnitud,W_G (Wobbe bruto)" in text
