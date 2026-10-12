"""Gráficos de la conducción transitoria — Fase 11.1.

- :func:`lumped_figure`: T(t) con la capacidad concentrada, la constante de tiempo 1/b
  marcada y, si el cuerpo la tiene, la solución exacta (la media y el centro) en gris.
- :func:`profiles_figure`: T(x) en la pared, el cilindro o la esfera a varios tiempos.
- :func:`history_figure`: la temperatura del centro contra el tiempo con la serie y con un
  término (como un gráfico de Heisler), con τ = 0,2 marcado.
- :func:`semi_infinite_figure`: T(x) en el sólido semi-infinito a varios tiempos, con la
  penetración.
- :func:`contact_figure`: los dos sólidos que se tocan, en dos paneles con el mismo eje de
  temperatura (cada uno con su escala de largo: el calor llega mucho más lejos en un metal).
- :func:`product_map_figure`: el mapa de temperatura del producto en el plano de los dos
  primeros factores (el cilindro corto en r y x).

Colores: la paleta validada para daltonismo del proyecto. Los tiempos van en la rampa azul
ordinal (claro el primero, oscuro el último), el punto de los datos en naranja y el fluido o
la referencia en gris.

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.heat_transfer.transient`.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.heat_transfer.transient import (
    ContactResult,
    Factor,
    LumpedResult,
    OneDimResult,
    ProductResult,
    SemiInfiniteResult,
    center_history,
    contact_profiles,
    lumped_curve,
    lumped_exact_curve,
    product_map,
    semi_infinite_profiles,
    time_profiles,
)

__all__ = [
    "contact_figure",
    "history_figure",
    "lumped_figure",
    "product_map_figure",
    "profiles_figure",
    "semi_infinite_figure",
    "time_axis",
    "time_label",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_GRAY = "#6b7280"
_LIGHT_GRAY = "#9ca3af"
_INK = "#1f2937"
_GRID = "#e5e7eb"
#: Rampa azul ordinal (pasos 250 a 650 de la paleta validada).
_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
#: La misma rampa, desde casi blanco, para el mapa de temperatura.
_MAP_SCALE = [
    [0.0, "#eef4fc"],
    [0.25, "#86b6ef"],
    [0.5, "#5598e7"],
    [0.75, "#1c5cab"],
    [1.0, "#104281"],
]
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0}
_TIME_UNITS = (("s", 1.0), ("min", 60.0), ("h", 3600.0), ("días", 86400.0))


def _layout(
    fig: go.Figure,
    *,
    height: int,
    x_title: str,
    y_title: str,
    xaxis: dict[str, object] | None = None,
    yaxis: dict[str, object] | None = None,
) -> None:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": x_title, "gridcolor": _GRID, **(xaxis or {})},
        yaxis={"title": y_title, "gridcolor": _GRID, **(yaxis or {})},
        legend=_LEGEND_BELOW,
        separators=", ",
    )


def _comma(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _ramp(k: int, n: int) -> str:
    """El color k de n curvas de la rampa (de claro a oscuro)."""
    if n <= 1:
        return _RAMP[2]
    return _RAMP[round(k * (len(_RAMP) - 1) / (n - 1))]


def time_axis(t_max_s: float) -> tuple[float, str]:
    """(segundos por unidad, unidad) para un eje de tiempo que llega a ``t_max_s``."""
    if t_max_s < 120.0:
        return 1.0, "s"
    if t_max_s < 2.0 * 3600.0:
        return 60.0, "min"
    if t_max_s < 3.0 * 86400.0:
        return 3600.0, "h"
    return 86400.0, "días"


def time_label(t_s: float) -> str:
    """Un tiempo con la unidad que se lee mejor («45 min», «2,5 h»)."""
    scale, unit = time_axis(t_s)
    return f"{_comma(t_s / scale)} {unit}"


def _hline(fig: go.Figure, y: float, text: str, **kwargs: object) -> None:
    fig.add_hline(
        y=y,
        line={"color": _GRAY, "width": 1.2, "dash": "dash"},
        annotation_text=text,
        annotation_position="top left",
        annotation_font={"color": _GRAY},
        **kwargs,
    )


def _point(fig: go.Figure, x: float, y: float, name: str, **kwargs: object) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name=name,
            marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            **kwargs,
        )
    )


# ---------------------------------------------------------------------
# Capacidad concentrada
# ---------------------------------------------------------------------


def lumped_figure(r: LumpedResult, *, t_scale: float, t_offset: float, t_unit: str) -> go.Figure:
    """T(t) con la capacidad concentrada (y la exacta en gris, si el cuerpo la tiene).

    ``t_scale``/``t_offset`` pasan de K a la temperatura del sistema.
    """
    ts, Ts = lumped_curve(r)
    scale, unit = time_axis(ts[-1])

    def T(x: float) -> float:
        return x * t_scale + t_offset

    fig = go.Figure()
    exact = lumped_exact_curve(r)
    if exact is not None:
        te, means, centers = exact
        fig.add_trace(
            go.Scatter(
                x=[t / scale for t in te],
                y=[T(x) for x in centers],
                mode="lines",
                name="exacta: centro",
                line={"color": _LIGHT_GRAY, "width": 1.8, "dash": "dot"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[t / scale for t in te],
                y=[T(x) for x in means],
                mode="lines",
                name="exacta: media",
                line={"color": _GRAY, "width": 1.8, "dash": "dash"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[t / scale for t in ts],
            y=[T(x) for x in Ts],
            mode="lines",
            name="capacidad concentrada",
            line={"color": _BLUE, "width": 2.5},
        )
    )
    i = r.inputs
    _hline(fig, T(i.T_inf), "T∞")
    tc = r.time_constant_s
    if tc <= ts[-1]:
        T_tc = i.T_inf + (i.T_i - i.T_inf) * math.exp(-1.0)
        fig.add_trace(
            go.Scatter(
                x=[tc / scale, tc / scale],
                y=[T(i.T_i), T(T_tc)],
                mode="lines+text",
                name="constante de tiempo 1/b",
                line={"color": _INK, "width": 1, "dash": "dot"},
                text=["1/b", ""],
                textposition="middle right",
                showlegend=False,
                hoverinfo="skip",
            )
        )
    _point(fig, r.time_s / scale, T(r.T_K), "el instante pedido")
    _layout(fig, height=380, x_title=f"t [{unit}]", y_title=f"T [{t_unit}]")
    return fig


# ---------------------------------------------------------------------
# Pared, cilindro y esfera
# ---------------------------------------------------------------------


def _position_title(r: OneDimResult) -> str:
    return "x (desde el plano medio)" if r.inputs.geometry == "wall" else "r (desde el centro)"


def profiles_figure(
    r: OneDimResult,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    x_scale: float,
    x_unit: str,
) -> go.Figure:
    """T(x) a cinco tiempos hasta el pedido, con la posición pedida en naranja.

    ``x_scale`` pasa de m a la unidad de largo (mm, in).
    """
    i = r.inputs
    profiles = time_profiles(r)
    fig = go.Figure()

    def T(x: float) -> float:
        return x * t_scale + t_offset

    n = len(profiles)
    for k, (t, xi, Ts) in enumerate(profiles):
        fig.add_trace(
            go.Scatter(
                x=[x * i.size * x_scale for x in xi],
                y=[T(v) for v in Ts],
                mode="lines",
                name=f"t = {time_label(t)}",
                line={"color": _ramp(k, n), "width": 2.2 if k == n - 1 else 1.6},
            )
        )
    _hline(fig, T(i.T_inf), "T∞")
    _point(fig, i.position * i.size * x_scale, T(r.T_K), "la posición pedida")
    _layout(
        fig,
        height=380,
        x_title=f"{_position_title(r)} [{x_unit}]",
        y_title=f"T [{t_unit}]",
    )
    return fig


def history_figure(r: OneDimResult, *, t_scale: float, t_offset: float, t_unit: str) -> go.Figure:
    """La temperatura del centro contra el tiempo: la serie y la solución de un término."""
    i = r.inputs
    ts, series, one = center_history(r)
    scale, unit = time_axis(ts[-1])

    def T(x: float) -> float:
        return x * t_scale + t_offset

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[t / scale for t in ts],
            y=[T(x) for x in one],
            mode="lines",
            name="un término",
            line={"color": _GRAY, "width": 1.8, "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[t / scale for t in ts],
            y=[T(x) for x in series],
            mode="lines",
            name="serie completa",
            line={"color": _BLUE, "width": 2.5},
        )
    )
    t02 = 0.2 * i.size**2 / i.solid.alpha
    if t02 <= ts[-1]:
        fig.add_vline(
            x=t02 / scale,
            line={"color": _INK, "width": 1, "dash": "dot"},
            annotation_text="τ = 0,2",
            annotation_position="top right",
        )
    _hline(fig, T(i.T_inf), "T∞")
    _point(fig, r.time_s / scale, T(r.T_center_K), "el instante pedido")
    lo = min(i.T_i, i.T_inf)
    hi = max(i.T_i, i.T_inf)
    pad = 0.08 * (hi - lo)
    y_range = sorted((T(lo - pad), T(hi + pad)))
    _layout(
        fig,
        height=360,
        x_title=f"t [{unit}]",
        y_title=f"T del centro [{t_unit}]",
        yaxis={"range": y_range},
    )
    return fig


# ---------------------------------------------------------------------
# Semi-infinito y contacto
# ---------------------------------------------------------------------


def semi_infinite_figure(
    r: SemiInfiniteResult,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    x_scale: float,
    x_unit: str,
) -> go.Figure:
    """T(x) a cuatro tiempos hasta el pedido, con la penetración δ y el punto pedido."""
    i = r.inputs
    profiles = semi_infinite_profiles(r)
    fig = go.Figure()

    def T(x: float) -> float:
        return x * t_scale + t_offset

    n = len(profiles)
    for k, (t, xs, Ts) in enumerate(profiles):
        fig.add_trace(
            go.Scatter(
                x=[x * x_scale for x in xs],
                y=[T(v) for v in Ts],
                mode="lines",
                name=f"t = {time_label(t)}",
                line={"color": _ramp(k, n), "width": 2.2 if k == n - 1 else 1.6},
            )
        )
    _hline(fig, T(i.T_i), "T_i")
    fig.add_vline(
        x=r.penetration_m * x_scale,
        line={"color": _INK, "width": 1, "dash": "dot"},
        annotation_text="δ",
        annotation_position="top right",
    )
    _point(fig, r.depth_m * x_scale, T(r.T_K), "el punto pedido")
    _layout(
        fig,
        height=380,
        x_title=f"x (desde la superficie) [{x_unit}]",
        y_title=f"T [{t_unit}]",
    )
    return fig


def contact_figure(
    r: ContactResult,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    x_scale: float,
    x_unit: str,
) -> go.Figure:
    """Los dos sólidos que se tocan, en dos paneles con el mismo eje de temperatura."""
    i = r.inputs
    profiles = contact_profiles(r)

    def T(x: float) -> float:
        return x * t_scale + t_offset

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(f"A: {i.solid_a.name}", f"B: {i.solid_b.name}"),
    )
    fig.update_annotations(font_size=12)  # los títulos de los paneles
    n = len(profiles)
    for k, (t, xa, Ta, xb, Tb) in enumerate(profiles):
        color = _ramp(k, n)
        name = f"t = {time_label(t)}"
        fig.add_trace(
            go.Scatter(
                x=[x * x_scale for x in xa],
                y=[T(v) for v in Ta],
                mode="lines",
                name=name,
                legendgroup=name,
                line={"color": color, "width": 2.0},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[x * x_scale for x in xb],
                y=[T(v) for v in Tb],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=False,
                line={"color": color, "width": 2.0},
            ),
            row=1,
            col=2,
        )
    for col in (1, 2):
        fig.add_hline(
            y=T(r.T_contact_K),
            line={"color": _ORANGE, "width": 1.6, "dash": "dash"},
            row=1,
            col=col,
        )
    fig.add_annotation(
        x=0,
        y=T(r.T_contact_K),
        xref="x2",
        yref="y2",
        text="T_s",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"color": _ORANGE},
    )
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    fig.update_xaxes(title=f"x [{x_unit}]", gridcolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID)
    fig.update_yaxes(title=f"T [{t_unit}]", row=1, col=1)
    return fig


# ---------------------------------------------------------------------
# Producto
# ---------------------------------------------------------------------


def _axis_title(factors: tuple[Factor, ...], k: int) -> str:
    """El rótulo del eje del factor k: x, y o z en las paredes (en orden) y la cara 1, 2 o 3
    si hay más de un factor semi-infinito (la arista y la esquina)."""
    f = factors[k]
    if f.kind == "cylinder":
        return "r (desde el eje)"
    nth = sum(1 for g in factors[:k] if g.kind == f.kind)
    if f.kind == "wall":
        return f"{'xyz'[nth]} (desde el plano medio)"
    if sum(1 for g in factors if g.kind == "semi") > 1:
        return f"profundidad desde la cara {nth + 1}"
    return "profundidad (desde la cara)"


def _axis_point(f: Factor) -> float:
    """La coordenada del punto pedido en m."""
    return f.position if f.kind == "semi" else f.position * (f.size or 0.0)


def product_map_figure(
    r: ProductResult,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    x_scale: float,
    x_unit: str,
) -> go.Figure | None:
    """El mapa de temperatura en el plano de los dos primeros factores (con el resto en el
    punto pedido), con las isotermas rotuladas y el punto en naranja."""
    data = product_map(r)
    if data is None:
        return None
    xs, ys, grid = data
    f1, f2 = r.inputs.factors[0], r.inputs.factors[1]
    fig = go.Figure(
        go.Contour(
            x=[x * x_scale for x in xs],
            y=[y * x_scale for y in ys],
            z=[[v * t_scale + t_offset for v in row] for row in grid],
            colorscale=_MAP_SCALE,
            contours={"showlabels": True, "labelfont": {"size": 11, "color": _INK}},
            line={"width": 0.6, "color": "rgba(31, 41, 55, 0.35)"},
            colorbar={"title": {"text": t_unit}, "thickness": 12},
            hovertemplate="%{x:.3g}, %{y:.3g}<br>T = %{z:.4g}<extra></extra>",
            name="T",
        )
    )
    _point(
        fig,
        _axis_point(f1) * x_scale,
        _axis_point(f2) * x_scale,
        "el punto pedido",
        showlegend=False,
    )
    _layout(
        fig,
        height=400,
        x_title=f"{_axis_title(r.inputs.factors, 0)} [{x_unit}]",
        y_title=f"{_axis_title(r.inputs.factors, 1)} [{x_unit}]",
    )
    return fig
