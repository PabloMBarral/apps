"""Tests de ui.radiation_charts y ui.exchanger_charts (Fase 8.2)."""

from __future__ import annotations

import math

import pytest

from core.heat_transfer import exchangers as hx
from core.heat_transfer import radiation as rd
from ui.exchanger_charts import (
    effectiveness_figure,
    f_factor_figure,
    profile_figure,
    short_name,
    type_comparison_figure,
    u_resistance_figure,
)
from ui.radiation_charts import (
    enclosure_figure,
    network_figure,
    planck_figure,
    spectral_match_figure,
    surface_curve_figure,
    view_factor_figure,
)

C = 273.15


def _trace(fig, name: str):  # noqa: ANN001, ANN202
    return next(t for t in fig.data if t.name == name)


def _bb(fragment: str) -> rd.BlackbodyResult:
    name = next(n for n in rd.BLACKBODY_EXAMPLES if fragment in n)
    return rd.solve_blackbody(rd.BLACKBODY_EXAMPLES[name].inputs)


# ---------------------------------------------------------------------
# Radiación
# ---------------------------------------------------------------------


def test_planck_curve_peaks_at_wien() -> None:
    r = _bb("Bola negra")
    fig = planck_figure(r, e_scale=1.0, e_unit="W/(m²·μm)")
    peak = _trace(fig, "λ_máx (Wien)")
    assert peak.x[0] == pytest.approx(r.lambda_max_m * 1e6)
    assert peak.y[0] == pytest.approx(r.E_blambda_max_W_per_m3 * 1e-6)
    curve = _trace(fig, "cuerpo negro a 800 K")
    assert max(curve.y) <= peak.y[0] * (1 + 1e-6)
    point = _trace(fig, "E_bλ en la λ pedida")
    assert point.y[0] == pytest.approx(3846.0, rel=2e-4)
    # en otro sistema, la escala de E_bλ
    fig_e = planck_figure(r, e_scale=0.317, e_unit="Btu/(h·ft²·μm)")
    assert _trace(fig_e, "λ_máx (Wien)").y[0] == pytest.approx(0.317 * peak.y[0])


def test_planck_with_bands_and_a_shaded_band() -> None:
    r = _bb("bandas")
    fig = planck_figure(r, e_scale=1.0, e_unit="W/(m²·μm)")
    black = _trace(fig, "cuerpo negro a 800 K")
    real = _trace(fig, "superficie real: ε(λ)·E_bλ")
    assert all(b >= e - 1e-9 for b, e in zip(black.y, real.y, strict=True))
    assert real.fill == "tozeroy"
    sun = _bb("El sol")
    fig = planck_figure(sun, e_scale=1.0, e_unit="W/(m²·μm)")
    assert len(fig.layout.shapes) == 1  # el visible, sombreado
    assert fig.layout.shapes[0].x0 == pytest.approx(0.40)


def test_spectral_match_normalizes_each_spectrum() -> None:
    r = _bb("selectivo")
    fig = spectral_match_figure(r)
    for t in fig.data[:2]:
        assert max(t.y) == pytest.approx(1.0, abs=2e-3)
    eps = _trace(fig, "ε(λ) = α(λ)")
    assert set(round(e, 6) for e in eps.y) == {0.95, 0.08}
    assert fig.layout.xaxis.type == "log"


def test_view_factor_figure_marks_the_data() -> None:
    inp = rd.VIEW_FACTOR_EXAMPLES[next(iter(rd.VIEW_FACTOR_EXAMPLES))].inputs
    r = rd.solve_view_factor(inp)
    label, xs, Fs = rd.view_factor_curve(inp)
    fig = view_factor_figure(label, xs, Fs, inp.c, r.F_ij, x_scale=1.0, x_unit="m", log_x=True)
    point = _trace(fig, "los datos")
    assert point.x[0] == pytest.approx(inp.c) and point.y[0] == pytest.approx(r.F_ij)
    assert fig.layout.xaxis.type == "log"
    # F baja al alejar las superficies
    assert Fs[0] > Fs[-1]


def test_network_shares_add_up() -> None:
    name = next(n for n in rd.TWO_SURFACE_EXAMPLES if "pantalla" in n)
    r = rd.solve_two_surface(rd.TWO_SURFACE_EXAMPLES[name].inputs)
    fig = network_figure(r)
    total = sum(sum(t.x) for t in fig.data)
    assert total == pytest.approx(100.0)
    assert {t.name for t in fig.data} == {"de superficie (1 − ε)/(A·ε)", "del espacio 1/(A·F)"}


def test_enclosure_bars_by_sign() -> None:
    name = next(n for n in rd.ENCLOSURE_EXAMPLES if "rerradiante" in n)
    r = rd.solve_enclosure(rd.ENCLOSURE_EXAMPLES[name].inputs)
    fig = enclosure_figure(r, q_scale=1e-3, q_unit="kW")
    assert all(x > 0 for x in _trace(fig, "entrega calor").x)
    assert all(x < 0 for x in _trace(fig, "recibe calor").x)
    assert list(_trace(fig, "rerradiante (Q̇ = 0)").y) == ["Pared aislada"]
    shown = [*_trace(fig, "entrega calor").x, *_trace(fig, "recibe calor").x]
    assert sum(shown) == pytest.approx(0.0, abs=1e-6)


def test_surface_curve_adds_both_mechanisms() -> None:
    r = rd.solve_surface_balance(rd.SURFACE_EXAMPLES[next(iter(rd.SURFACE_EXAMPLES))].inputs)
    curve = rd.surface_balance_curve(r)
    fig = surface_curve_figure(
        curve, r, t_scale=1.0, t_offset=-C, t_unit="°C", q_scale=1.0, q_unit="W/m²"
    )
    conv, rad, both = fig.data[0], fig.data[1], fig.data[2]
    for a, b, s in zip(conv.y, rad.y, both.y, strict=True):
        assert s == pytest.approx(a + b)
    point = _trace(fig, "los datos")
    assert point.x[0] == pytest.approx(200.0)
    assert point.y[0] == pytest.approx((r.Q_conv_W + r.Q_rad_W) / r.inputs.area_m2)


# ---------------------------------------------------------------------
# Intercambiadores
# ---------------------------------------------------------------------


def _rating(fragment: str) -> hx.ExchangerResult:
    name = next(n for n in hx.RATING_EXAMPLES if fragment in n)
    return hx.solve_rating(hx.RATING_EXAMPLES[name].inputs)


def test_effectiveness_family_and_point() -> None:
    r = _rating("11-9")
    fig = effectiveness_figure(r)
    family = [t for t in fig.data if t.name.startswith("C_r = ") and "datos" not in t.name]
    assert len(family) == 5
    point = _trace(fig, "los datos")
    assert point.x[0] == pytest.approx(r.NTU) and point.y[0] == pytest.approx(r.effectiveness)
    # C_r = 0: 1 − e^(−NTU) en cualquier tipo
    zero = family[0]
    for x, y in zip(zero.x, zero.y, strict=True):
        assert y == pytest.approx(1.0 - math.exp(-x), abs=1e-9)
    # la curva de los datos (C_r no está en la familia), con el punto encima
    data = _trace(fig, f"C_r = {r.C_r:.3g} (los datos)".replace(".", ","))
    assert data.line.color == "#eb6834"


def test_f_factor_only_where_it_applies() -> None:
    assert f_factor_figure(_rating("contracorriente")) is None
    assert f_factor_figure(_rating("Condensador")) is None  # C_r = 0
    r = _rating("11-9")
    fig = f_factor_figure(r)
    assert fig is not None
    point = _trace(fig, "los datos")
    assert r.P_R is not None
    assert point.x[0] == pytest.approx(r.P_R[0]) and point.y[0] == pytest.approx(r.F)
    for t in fig.data:
        assert all(0.5 - 1e-9 <= f <= 1.0 + 1e-9 for f in t.y)


def test_profile_of_the_double_pipe() -> None:
    assert profile_figure(_rating("11-9"), t_scale=1.0, t_offset=-C, t_unit="°C") is None
    r = _rating("contracorriente")
    fig = profile_figure(r, t_scale=1.0, t_offset=-C, t_unit="°C")
    assert fig is not None
    hot, cold = fig.data
    assert hot.y[0] == pytest.approx(r.hot.T_in_K - C) and hot.y[-1] == pytest.approx(
        r.hot.T_out_K - C
    )
    # contracorriente: el frío sale donde entra el caliente
    assert cold.y[0] == pytest.approx(r.cold.T_out_K - C)
    assert cold.y[-1] == pytest.approx(r.cold.T_in_K - C)


def test_type_comparison_highlights_the_current_type() -> None:
    r = _rating("11-9")
    rows = hx.type_comparison(r)
    fig = type_comparison_figure(rows, r.arrangement.name, area=False, scale=1e-3, unit="kW")
    t = fig.data[0]
    k = list(t.y).index(short_name(r.arrangement))
    assert t.y[k] == "Casco y tubos, 1 paso"
    assert t.marker.color[k] == "#2a78d6"
    assert t.x[k] == pytest.approx(r.Q_W * 1e-3)
    # el contracorriente transfiere más que todos
    counter = list(t.y).index("Contracorriente")
    assert t.x[counter] == pytest.approx(max(t.x))
    # los nombres cortos entran en un celular
    assert max(len(n) for n in t.y) <= 26


def test_u_resistance_shares() -> None:
    r = hx.overall_u(hx.U_EXAMPLES[next(iter(hx.U_EXAMPLES))].inputs)
    fig = u_resistance_figure(r)
    assert sum(sum(t.x) for t in fig.data) == pytest.approx(100.0)
    assert [t.name for t in fig.data] == ["convección", "ensuciamiento", "pared"]
