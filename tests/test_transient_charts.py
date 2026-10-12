"""Tests de los gráficos de la conducción transitoria (Fase 11.1)."""

from __future__ import annotations

import pytest

from core.heat_transfer import transient as tr
from ui import transient_charts as tc

C = 273.15
TEC = {"t_scale": 1.0, "t_offset": -C, "t_unit": "°C"}
TEC_X = {**TEC, "x_scale": 1000.0, "x_unit": "mm"}
ENG = {"t_scale": 1.8, "t_offset": -459.67, "t_unit": "°F"}
ENG_X = {**ENG, "x_scale": 39.37007874015748, "x_unit": "in"}


def _names(fig) -> list[str]:  # type: ignore[no-untyped-def]
    return [t.name for t in fig.data]


@pytest.mark.parametrize(
    ("t_s", "unit"), [(30.0, "s"), (600.0, "min"), (5 * 3600.0, "h"), (90 * 86400.0, "días")]
)
def test_time_axis(t_s: float, unit: str) -> None:
    scale, label = tc.time_axis(t_s)
    assert label == unit
    assert 1.0 <= t_s / scale < 120.0
    assert tc.time_label(t_s).endswith(unit)
    assert "." not in tc.time_label(1.5 * 60.0)  # coma decimal


def test_lumped_figure() -> None:
    r = tr.solve_lumped(tr.LUMPED_EXAMPLES["La hora de la muerte (Çengel y Ghajar §4-1)"].inputs)
    fig = tc.lumped_figure(r, **TEC)
    names = _names(fig)
    assert names[:3] == ["exacta: centro", "exacta: media", "capacidad concentrada"]
    assert "el instante pedido" in names
    point = fig.data[names.index("el instante pedido")]
    scale, unit = tc.time_axis(tr.lumped_curve(r)[0][-1])
    assert point.x[0] == pytest.approx(r.time_s / scale)
    assert point.y[0] == pytest.approx(r.T_K - C)
    assert fig.layout.xaxis.title.text == f"t [{unit}]"
    # la constante de tiempo, dentro del eje
    assert any(t.text is not None and "1/b" in t.text for t in fig.data)


def test_lumped_figure_without_exact_solution() -> None:
    r = tr.solve_lumped(
        tr.LumpedInputs("custom", (1e-5, 0.006), tr.SOLIDS["iron"], 20.0, 400.0, 300.0, time_s=5.0)
    )
    names = _names(tc.lumped_figure(r, **ENG))
    assert "exacta: media" not in names and "capacidad concentrada" in names


def test_profiles_and_history() -> None:
    r = tr.solve_one_dimensional(
        tr.ONE_DIM_EXAMPLES["Placas de bronce en un horno (Çengel y Ghajar §4-2)"].inputs
    )
    fig = tc.profiles_figure(r, **TEC_X)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == 5
    # de claro a oscuro con el tiempo
    assert lines[0].line.color == tc._RAMP[0] and lines[-1].line.color == tc._RAMP[-1]
    assert lines[-1].x[-1] == pytest.approx(r.inputs.size * 1000.0)
    point = fig.data[-1]
    assert point.x[0] == pytest.approx(20.0) and point.y[0] == pytest.approx(r.T_K - C)
    assert fig.layout.xaxis.title.text == "x (desde el plano medio) [mm]"
    hist = tc.history_figure(r, **TEC)
    assert _names(hist)[:2] == ["un término", "serie completa"]
    assert hist.data[-1].y[0] == pytest.approx(r.T_center_K - C)
    assert hist.layout.yaxis.range[0] < 20.0 < 500.0 < hist.layout.yaxis.range[1]


def test_history_marks_tau_02() -> None:
    r = tr.solve_one_dimensional(
        tr.ONE_DIM_EXAMPLES["Eje de acero inoxidable (Çengel y Ghajar §4-2)"].inputs
    )
    fig = tc.history_figure(r, **ENG)
    texts = [a.text for a in fig.layout.annotations]
    assert "τ = 0,2" in texts
    assert fig.layout.yaxis.title.text == "T del centro [°F]"


def test_cylinder_axis_is_the_radius() -> None:
    r = tr.solve_one_dimensional(tr.ONE_DIM_EXAMPLES["Huevo hervido (Çengel y Ghajar §4-2)"].inputs)
    assert tc.profiles_figure(r, **ENG_X).layout.xaxis.title.text == "r (desde el centro) [in]"


def test_semi_infinite_figure() -> None:
    r = tr.solve_semi_infinite(
        tr.SEMI_INFINITE_EXAMPLES["Losa de hormigón con aire caliente (ejemplo propio)"].inputs
    )
    fig = tc.semi_infinite_figure(r, **TEC_X)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == 4
    assert "δ" in [a.text for a in fig.layout.annotations]
    point = fig.data[-1]
    assert point.x[0] == pytest.approx(20.0)
    assert point.y[0] == pytest.approx(r.T_K - C)


def test_contact_figure_has_two_panels() -> None:
    r = tr.solve_contact(tr.CONTACT_EXAMPLES["La mano en aluminio (ejemplo propio)"].inputs)
    fig = tc.contact_figure(r, **TEC_X)
    assert len(fig.data) == 6  # tres tiempos por panel
    assert {t.xaxis for t in fig.data} == {"x", "x2"}
    # cada tiempo, una sola entrada en la leyenda
    assert sum(bool(t.showlegend is not False) for t in fig.data) == 3
    texts = [a.text for a in fig.layout.annotations]
    assert "T_s" in texts and any("Aluminio" in t for t in texts)
    assert all(t.y[0] == pytest.approx(r.T_contact_K - C) for t in fig.data)


def test_product_map() -> None:
    r = tr.solve_product(
        tr.PRODUCT_EXAMPLES["Cilindro corto de bronce (Çengel y Ghajar §4-4)"].inputs
    )
    fig = tc.product_map_figure(r, **TEC_X)
    assert fig is not None
    contour = fig.data[0]
    assert contour.type == "contour"
    assert max(contour.x) == pytest.approx(50.0) and max(contour.y) == pytest.approx(60.0)
    assert fig.layout.xaxis.title.text == "r (desde el eje) [mm]"
    assert fig.layout.yaxis.title.text == "x (desde el plano medio) [mm]"
    # el punto pedido (el centro) y la temperatura del centro en el mapa
    assert (fig.data[1].x[0], fig.data[1].y[0]) == (0.0, 0.0)
    assert contour.z[0][0] == pytest.approx(r.T_K - C)


def test_product_map_with_a_semi_infinite_factor() -> None:
    r = tr.solve_product(
        tr.PRODUCT_EXAMPLES[
            "Cilindro semi-infinito de aluminio en agua (Çengel y Ghajar §4-4)"
        ].inputs
    )
    fig = tc.product_map_figure(r, **TEC_X)
    assert fig is not None
    assert fig.layout.yaxis.title.text == "profundidad (desde la cara) [mm]"
    assert fig.data[1].y[0] == pytest.approx(150.0)
