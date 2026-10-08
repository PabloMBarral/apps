"""Gráficos de la exergía — Fase 7.

- :func:`grassmann_figure`: el **diagrama de Grassmann** de una planta. Una banda
  baja con la exergía que entra; cada componente desprende hacia la derecha la
  que destruye (gris) o la que se pierde (violeta), con el ancho proporcional, y
  lo que queda abajo es el producto (naranja). Vertical: cada componente tiene su
  renglón y su rótulo, así se lee en un celular (el Sankey de plotly con las
  corrientes reales, con los lazos del agua de alimentación, no se leía).
- :func:`mollier_exergy_figure`: el estado y el estado muerto en el diagrama
  h–s con la recta del ambiente, h = h₀ + T₀·(s − s₀); ψ es la distancia vertical
  del estado a esa recta.
- :func:`carnot_factor_figure` y :func:`finite_source_figure`: la exergía de un
  calor y de una fuente finita en función de la temperatura.
- :func:`fuel_ratio_figure`: φ = e/PCI de los combustibles de la tabla.
- :func:`component_efficiency_figure`: el rendimiento exergético de cada componente.

Colores: la paleta validada para daltonismo del proyecto (azul la exergía que
circula, gris la destruida, violeta la perdida, naranja el producto).

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en :mod:`core.exergy`.
"""

from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Sequence

import plotly.graph_objects as go

from core.exergy.chemical import SpeciesExergy
from core.exergy.physical import FiniteSourceExergy, PhysicalExergy
from core.exergy.plant import ComponentExergy, GrassmannRow
from core.fluids import fluid_limits, saturation_at_temperature

__all__ = [
    "carnot_factor_figure",
    "component_efficiency_figure",
    "finite_source_figure",
    "fuel_ratio_figure",
    "grassmann_figure",
    "mollier_exergy_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_VIOLET = "#4a3aa7"
_GRAY = "#6b7280"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_BLUE_FILL = "rgba(42, 120, 214, 0.28)"
_GRAY_FILL = "rgba(107, 114, 128, 0.42)"
_VIOLET_FILL = "rgba(74, 58, 167, 0.40)"
_ORANGE_FILL = "rgba(235, 104, 52, 0.45)"
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0}


def _fmt(x: float, sig: int = 4) -> str:
    """Número con separador de miles fino y coma decimal (como el resto de la página)."""
    if abs(x) >= 10**sig:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:.{sig}g}".replace(".", ",")


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


#: Caracteres por renglón de los rótulos del Grassmann (a 390 px entran ~32, a 11 px).
_LABEL_CHARS = 26
#: Largo de «rótulo · valores» que entra en un renglón arriba y abajo del Grassmann
#: (a 390 px, ~5,2 px por carácter: entran ~65; con margen para un celular de 360 px).
_HEADLINE_CHARS = 52


def _wrap_label(text: str, width: int = _LABEL_CHARS) -> str:
    """Parte un rótulo en renglones (``<br>``) sin separar un número de su unidad."""
    text = re.sub(r"(\d) (°C|K|°F|R|bar|kPa|%)", "\\1\u00a0\\2", text)
    return "<br>".join(textwrap.wrap(text, width)) or text


def _headline(label: str, values: str) -> tuple[str, int]:
    """«**rótulo** · valores» en un renglón, o en dos si no entra a 390 px."""
    if len(label) + len(values) + 3 <= _HEADLINE_CHARS:
        return f"<b>{label}</b> · {values}", 1
    return f"<b>{label}</b><br>{values}", 2


# ---------------------------------------------------------------------
# Diagrama de Grassmann
# ---------------------------------------------------------------------


def grassmann_figure(
    rows: Sequence[GrassmannRow],
    *,
    x_in: float,
    in_label: str,
    product_label: str,
    scale: float,
    unit: str,
) -> go.Figure:
    """El diagrama de Grassmann (vertical) de una planta.

    ``rows`` son las ramas en el orden del flujo (:func:`core.exergy.plant.grassmann_rows`),
    ``x_in`` la exergía que entra (W) y ``scale`` pasa de W a ``unit``. El producto
    es lo que queda: x_in − Σ(D + L).
    """
    band_w = 0.30  # ancho de la banda con el 100 %
    x0 = 0.03
    row_h = 1.0
    label_x = 0.55
    sc = band_w / x_in
    shapes: list[dict] = []
    ann: list[dict] = []
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_t: list[str] = []
    head, head_lines = _headline(in_label, f"{_fmt(x_in * scale)} {unit} · 100 %")
    ann.append(
        {
            "x": x0,
            "y": 0.12,
            "xanchor": "left",
            "yanchor": "bottom",
            "align": "left",
            "text": head,
            "showarrow": False,
            "font": {"size": 12, "color": _INK},
        }
    )
    y = 0.0
    remaining = x_in
    for row in rows:
        for amount, kind in ((row.destroyed_W, "d"), (row.loss_W, "l")):
            if amount <= 1e-9 * x_in:
                continue
            w_rem = remaining * sc
            t = amount * sc
            xr = x0 + w_rem
            y_top, y_bot = y, y - row_h
            # tramo de la banda hasta el renglón (sin la parte que se va)
            shapes.append(
                {
                    "type": "rect",
                    "x0": x0,
                    "x1": xr - t,
                    "y0": y_bot,
                    "y1": y_top,
                    "fillcolor": _BLUE_FILL,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
            # la rama: baja por el borde derecho de la banda y gira a la derecha
            yc = y_top - 0.32 * row_h
            th = max(min(t * 2.2, 0.55 * row_h), 0.012)
            xa, xb = xr - t, xr
            path = (
                f"M {xa},{y_top} L {xb},{y_top} C {xb},{yc} {xb},{yc} {xb + 0.03},{yc} "
                f"L {label_x - 0.02},{yc} L {label_x - 0.02},{yc - th} L {xb + 0.03},{yc - th} "
                f"C {xa},{yc - th} {xa},{yc - th} {xa},{y_top} Z"
            )
            fill, line = (_GRAY_FILL, _GRAY) if kind == "d" else (_VIOLET_FILL, _VIOLET)
            shapes.append(
                {
                    "type": "path",
                    "path": path,
                    "fillcolor": fill,
                    "line": {"width": 0.6, "color": line},
                    "layer": "below",
                }
            )
            what = "destruida" if kind == "d" else "perdida"
            name = _wrap_label(row.name if kind == "d" else f"{row.name} (pérdida)")
            ann.append(
                {
                    "x": label_x,
                    "y": yc - th / 2.0,
                    "xanchor": "left",
                    "yanchor": "middle",
                    "align": "left",
                    "showarrow": False,
                    "text": (
                        f"{name}<br><span style='color:{_GRAY}'>{_fmt(amount * scale)} {unit}"
                        f" · {_pct(amount / x_in)}</span>"
                    ),
                    "font": {"size": 11, "color": _INK},
                }
            )
            hover_x.append(label_x - 0.06)
            hover_y.append(yc - th / 2.0)
            hover_t.append(
                f"{row.name}: {_fmt(amount * scale)} {unit} {what} ({_pct(amount / x_in)} de "
                "lo que entra)"
            )
            remaining -= amount
            y = y_bot
    # el producto
    w_rem = max(remaining, 0.0) * sc
    shapes.append(
        {
            "type": "rect",
            "x0": x0,
            "x1": x0 + w_rem,
            "y0": y - 0.9 * row_h,
            "y1": y,
            "fillcolor": _ORANGE_FILL,
            "line": {"width": 0},
            "layer": "below",
        }
    )
    foot, foot_lines = _headline(
        product_label, f"{_fmt(remaining * scale)} {unit} · {_pct(remaining / x_in)}"
    )
    ann.append(
        {
            "x": x0,
            "y": y - 0.95 * row_h,
            "xanchor": "left",
            "yanchor": "top",
            "align": "left",
            "text": foot,
            "showarrow": False,
            "font": {"size": 12, "color": _INK},
        }
    )
    fig = go.Figure(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker={"size": 18, "color": "rgba(0,0,0,0)"},
            hovertext=hover_t,
            hoverinfo="text",
            showlegend=False,
        )
    )
    n_rows = max(1, len(hover_t))
    extra_top = 0.4 * (head_lines - 1)  # un renglón más arriba si el título se parte
    extra_bottom = 0.4 * (foot_lines - 1)
    fig.update_layout(
        shapes=shapes,
        annotations=ann,
        height=int(120 + 62 * n_rows + 25 * (head_lines + foot_lines - 2)),
        xaxis={"range": [0, 1], "visible": False, "fixedrange": True},
        yaxis={
            "range": [y - (1.55 + extra_bottom) * row_h, 0.55 + extra_top],
            "visible": False,
            "fixedrange": True,
        },
        margin={"l": 4, "r": 4, "t": 4, "b": 4},
        plot_bgcolor="white",
        paper_bgcolor="white",
        separators=", ",
    )
    return fig


# ---------------------------------------------------------------------
# Exergía física
# ---------------------------------------------------------------------


def mollier_exergy_figure(
    result: PhysicalExergy, *, h_scale: float, s_scale: float, h_unit: str, s_unit: str
) -> go.Figure:
    """El estado, el estado muerto y la recta del ambiente en el diagrama h–s.

    La recta h = h₀ + T₀·(s − s₀) pasa por el estado muerto con pendiente T₀: la
    distancia vertical del estado a la recta es ψ (Kotas, 1985; la construcción
    de Mollier). ``h_scale`` y ``s_scale`` pasan de SI a las unidades pedidas.
    """
    st, d = result.state, result.dead
    T0 = result.T0_K
    fig = go.Figure()
    # campana
    limits = fluid_limits(result.fluid)
    dome_l: list[tuple[float, float]] = []
    dome_g: list[tuple[float, float]] = []
    if limits.T_crit_K > limits.T_triple_K + 5.0:
        lo, hi = limits.T_triple_K + 0.5, limits.T_crit_K - 0.05
        for k in range(60):
            T = lo + (hi - lo) * (1.0 - (1.0 - k / 59.0) ** 2)
            try:
                sat = saturation_at_temperature(result.fluid, T)
            except ValueError:
                continue
            dome_l.append((sat.liquid.s_J_per_kg_K, sat.liquid.h_J_per_kg))
            dome_g.append((sat.vapor.s_J_per_kg_K, sat.vapor.h_J_per_kg))
    if dome_l:
        pts = dome_l + list(reversed(dome_g))
        fig.add_trace(
            go.Scatter(
                x=[p[0] * s_scale for p in pts],
                y=[p[1] * h_scale for p in pts],
                mode="lines",
                name="campana",
                line={"color": _GRAY, "width": 1.5},
                hoverinfo="skip",
            )
        )
    s_vals = [st.s_J_per_kg_K, d.s_J_per_kg_K, *(p[0] for p in dome_l + dome_g)]
    s_lo, s_hi = min(s_vals), max(s_vals)
    span = max(s_hi - s_lo, 1.0)
    s_line = [s_lo - 0.05 * span, s_hi + 0.05 * span]
    h_line = [d.h_J_per_kg + T0 * (s - d.s_J_per_kg_K) for s in s_line]
    fig.add_trace(
        go.Scatter(
            x=[s * s_scale for s in s_line],
            y=[h * h_scale for h in h_line],
            mode="lines",
            name="recta del ambiente: h₀ + T₀·(s − s₀)",
            line={"color": _INK, "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    h_on_line = d.h_J_per_kg + T0 * (st.s_J_per_kg_K - d.s_J_per_kg_K)
    fig.add_trace(
        go.Scatter(
            x=[st.s_J_per_kg_K * s_scale] * 2,
            y=[h_on_line * h_scale, st.h_J_per_kg * h_scale],
            mode="lines",
            name="ψ (distancia vertical)",
            line={"color": _BLUE, "width": 4},
            hovertemplate=f"ψ = {_fmt(result.psi_J_per_kg * h_scale)} {h_unit}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[d.s_J_per_kg_K * s_scale],
            y=[d.h_J_per_kg * h_scale],
            mode="markers+text",
            name="estado muerto (T₀, p₀)",
            marker={"size": 11, "color": _INK},
            text=["0"],
            textposition="bottom right",
            hovertemplate="estado muerto<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[st.s_J_per_kg_K * s_scale],
            y=[st.h_J_per_kg * h_scale],
            mode="markers+text",
            name="estado",
            marker={"size": 12, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            text=["estado"],
            textposition="top left",
            hovertemplate=f"h = %{{y:.5g}} {h_unit}<br>s = %{{x:.5g}} {s_unit}<extra></extra>",
        )
    )
    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": f"s [{s_unit}]", "gridcolor": _GRID},
        yaxis={"title": f"h [{h_unit}]", "gridcolor": _GRID},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


def carnot_factor_figure(
    T0_K: float, T_K: float, *, t_scale: float, t_offset: float, t_unit: str
) -> go.Figure:
    """El factor de Carnot 1 − T₀/T contra T, con la temperatura de la fuente marcada.

    ``t_scale`` y ``t_offset`` pasan de K a la temperatura del sistema.
    """
    lo = 0.5 * min(T0_K, T_K)
    hi = max(3.0 * T0_K, 1.25 * T_K)
    Ts = [lo + (hi - lo) * k / 120.0 for k in range(121)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[T * t_scale + t_offset for T in Ts],
            y=[1.0 - T0_K / T for T in Ts],
            mode="lines",
            name="1 − T₀/T",
            line={"color": _BLUE, "width": 2.5},
            hovertemplate=f"T = %{{x:.4g}} {t_unit}<br>1 − T₀/T = %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[T_K * t_scale + t_offset],
            y=[1.0 - T0_K / T_K],
            mode="markers",
            name="la fuente",
            marker={"size": 12, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=f"T = %{{x:.4g}} {t_unit}<br>1 − T₀/T = %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line={"color": _GRAY, "width": 1})
    fig.add_vline(x=T0_K * t_scale + t_offset, line={"color": _GRAY, "width": 1, "dash": "dot"})
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": f"T [{t_unit}]", "gridcolor": _GRID},
        yaxis={"title": "factor de Carnot", "gridcolor": _GRID, "range": [-1.5, 1.0]},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


def finite_source_figure(
    result: FiniteSourceExergy,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    e_scale: float,
    e_unit: str,
) -> go.Figure:
    """Φ contra la temperatura del cuerpo: cero en T₀ y positiva a los dos lados."""
    curve = result.curve()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[T * t_scale + t_offset for T, _ in curve],
            y=[phi * e_scale for _, phi in curve],
            mode="lines",
            name="Φ(T)",
            line={"color": _BLUE, "width": 2.5},
            hovertemplate=f"T = %{{x:.4g}} {t_unit}<br>Φ = %{{y:.4g}} {e_unit}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.T_K * t_scale + t_offset],
            y=[result.Phi_J * e_scale],
            mode="markers",
            name="el cuerpo",
            marker={"size": 12, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=f"T = %{{x:.4g}} {t_unit}<br>Φ = %{{y:.4g}} {e_unit}<extra></extra>",
        )
    )
    fig.add_vline(
        x=result.T0_K * t_scale + t_offset, line={"color": _GRAY, "width": 1, "dash": "dot"}
    )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": f"T [{t_unit}]", "gridcolor": _GRID},
        yaxis={"title": f"Φ [{e_unit}]", "gridcolor": _GRID, "rangemode": "tozero"},
        legend=_LEGEND_BELOW,
        separators=", ",
    )
    return fig


# ---------------------------------------------------------------------
# Exergía química
# ---------------------------------------------------------------------


def fuel_ratio_figure(rows: Sequence[SpeciesExergy], *, current: str | None = None) -> go.Figure:
    """φ = e/PCI de los combustibles de la tabla (un punto por sustancia).

    Los hidrocarburos quedan entre 1,03 y 1,08; el H₂ y el CO, por debajo de 1:
    su combustión baja la entropía (menos moles de gas), así que el trabajo
    máximo es menor que el calor (Kotas, 1985).
    """
    names = [r.species.name for r in rows]
    values = [r.ratio_lhv or 0.0 for r in rows]
    colors = [_ORANGE if r.species.key == current else _BLUE for r in rows]
    fig = go.Figure(
        go.Scatter(
            x=values,
            y=names,
            mode="markers+text",
            marker={"size": 11, "color": colors, "line": {"color": _INK, "width": 0.5}},
            text=[f"{v:.3f}".replace(".", ",") for v in values],
            textposition="middle right",
            textfont={"color": _INK, "size": 11},
            cliponaxis=False,
            hovertemplate="%{y}: φ = %{x:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=1.0, line={"color": _GRAY, "width": 1, "dash": "dash"})
    fig.update_layout(
        height=80 + 26 * len(rows),
        margin={"l": 10, "r": 30, "t": 20, "b": 10},
        xaxis={
            "title": "φ = e_química / PCI",
            "gridcolor": _GRID,
            "range": [min(values) - 0.02, max(values) + 0.05],
        },
        yaxis={"categoryorder": "array", "categoryarray": names, "automargin": True},
        separators=", ",
    )
    return fig


# ---------------------------------------------------------------------
# Planta
# ---------------------------------------------------------------------


def component_efficiency_figure(components: Sequence[ComponentExergy]) -> go.Figure:
    """El rendimiento exergético ε = P/F de cada componente que tiene producto."""
    rows = [c for c in components if c.efficiency is not None]
    # los nombres largos en dos renglones: a 390 px el eje se comía el gráfico
    names = [_wrap_label(c.name) for c in rows]
    values = [100.0 * (c.efficiency or 0.0) for c in rows]
    two_lines = any("<br>" in n for n in names)
    fig = go.Figure(
        go.Scatter(
            x=values,
            y=names,
            mode="markers+text",
            marker={"size": 11, "color": _BLUE, "line": {"color": _INK, "width": 0.5}},
            text=[f"{v:.1f} %".replace(".", ",") for v in values],
            textposition="middle left",
            textfont={"color": _INK, "size": 11},
            cliponaxis=False,
            hovertemplate="%{y}: ε = %{x:.1f} %<extra></extra>",
            showlegend=False,
        )
    )
    lo = min([*values, 100.0])
    fig.update_layout(
        height=80 + (36 if two_lines else 30) * max(1, len(rows)),
        margin={"l": 10, "r": 20, "t": 20, "b": 10},
        xaxis={
            "title": "ε = Ẋ_P / Ẋ_F [%]",
            "gridcolor": _GRID,
            "range": [max(0.0, math.floor(lo / 10.0) * 10.0 - 15.0), 101.0],
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(names)),
            "automargin": True,
            "tickfont": {"size": 11},
        },
        separators=", ",
    )
    return fig
