"""Gráficos de los intercambiadores de calor — Fase 8.2.

- :func:`effectiveness_figure`: ε(NTU) para varios C_r (el gráfico de los libros)
  con la curva de los datos y el punto.
- :func:`f_factor_figure`: F(P) para varios R (Bowman, Mueller y Nagle) con la
  curva de los datos y el punto.
- :func:`profile_figure`: las temperaturas a lo largo de un doble tubo.
- :func:`type_comparison_figure`: el mismo problema con cada tipo (gráfico de
  puntos: las diferencias pueden ser chicas y unas barras desde cero las
  esconderían).
- :func:`u_resistance_figure`: el peso de cada resistencia del U global.

Colores: la paleta validada para daltonismo del proyecto (la familia de curvas
en una rampa azul ordinal, los datos en naranja; el fluido caliente naranja y
el frío azul).

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.heat_transfer.exchangers`.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from core.heat_transfer.exchangers import (
    Arrangement,
    ExchangerResult,
    OverallUResult,
    TypeComparison,
    effectiveness,
    effectiveness_curves,
    f_curves,
    max_effectiveness,
    temperature_profile,
)

__all__ = [
    "effectiveness_figure",
    "f_factor_figure",
    "profile_figure",
    "short_name",
    "type_comparison_figure",
    "u_resistance_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_VIOLET = "#4a3aa7"
_GRAY = "#6b7280"
_INK = "#1f2937"
_GRID = "#e5e7eb"
#: Rampa azul ordinal (pasos 250 a 650 de la paleta validada).
_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0}


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


def _comma(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",")


def _label(x: float) -> str:
    """Un valor para rotular una barra o un punto: 36 980, 1832, 12,09 (sin «e+04»)."""
    if abs(x) >= 1e4:
        return f"{x:,.0f}".replace(",", "\u2009")
    return _comma(x)


def _legend_at_bottom(fig: go.Figure, height: int) -> None:
    """La leyenda al pie del gráfico, debajo del título del eje (en un gráfico bajo, la
    de ``_LEGEND_BELOW`` lo pisaba)."""
    fig.update_layout(
        height=height + 50,
        margin={"l": 10, "r": 10, "t": 30, "b": 80},
        legend={"orientation": "h", "yref": "container", "y": 0.0, "yanchor": "bottom", "x": 0},
    )


def _point(fig: go.Figure, x: float, y: float, hover: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            name="los datos",
            marker={"size": 12, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=hover,
        )
    )


def _ramp(k: int, n: int) -> str:
    """El color k de n curvas de la rampa (de claro a oscuro)."""
    if n <= 1:
        return _RAMP[2]
    return _RAMP[round(k * (len(_RAMP) - 1) / (n - 1))]


# ---------------------------------------------------------------------
# ε–NTU y F
# ---------------------------------------------------------------------


def effectiveness_figure(result: ExchangerResult) -> go.Figure:
    """ε(NTU) del tipo para C_r = 0; 0,25; 0,5; 0,75 y 1, y la curva de los datos."""
    arr, cr, ntu = result.arrangement, result.C_r, result.NTU
    ntu_max = max(5.0, 1.3 * ntu)
    family = effectiveness_curves(arr, ntu_max=ntu_max, mixed_is_cmin=result.mixed_is_cmin)
    fig = go.Figure()
    n = len(family)
    for k, (c, (xs, ys)) in enumerate(family.items()):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"C_r = {_comma(c, '.2g')}",
                line={"color": _ramp(k, n), "width": 1.8},
                hovertemplate=f"C_r = {_comma(c, '.2g')}<br>NTU = %{{x:.3g}}"
                "<br>ε = %{y:.3f}<extra></extra>",
            )
        )
    if all(abs(cr - c) > 1e-6 for c in family):
        xs = [ntu_max * j / 120 for j in range(121)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[effectiveness(arr, x, cr, result.mixed_is_cmin) for x in xs],
                mode="lines",
                name=f"C_r = {_comma(cr, '.3g')} (los datos)",
                line={"color": _ORANGE, "width": 2.5, "dash": "dash"},
                hovertemplate="NTU = %{x:.3g}<br>ε = %{y:.3f}<extra></extra>",
            )
        )
    _point(fig, ntu, result.effectiveness, "NTU = %{x:.4g}<br>ε = %{y:.4f}<extra></extra>")
    eps_max = max_effectiveness(arr, cr, result.mixed_is_cmin)
    if eps_max < 0.999:
        fig.add_hline(
            y=eps_max,
            line={"color": _GRAY, "width": 1.2, "dash": "dot"},
            annotation_text=f"ε máx. con C_r = {_comma(cr, '.3g')}",
            annotation_position="bottom right",
        )
    _layout(
        fig,
        height=380,
        x_title="NTU = UA/C_mín",
        y_title="efectividad ε",
        xaxis={"range": [0.0, ntu_max]},
        yaxis={"range": [0.0, 1.02]},
    )
    return fig


def f_factor_figure(result: ExchangerResult) -> go.Figure | None:
    """F(P) para varios R y la curva de los datos; None si F = 1 (doble tubo o un fluido que
    cambia de fase)."""
    PR = result.P_R
    if PR is None or result.arrangement.kind in ("counter", "parallel") or result.C_r <= 0.0:
        return None
    P0, R0 = PR
    arr = result.arrangement
    R_values = (0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0)
    family = f_curves(arr, R_values)
    fig = go.Figure()
    for k, (R, (Ps, Fs)) in enumerate(family.items()):
        if not Ps:
            continue
        fig.add_trace(
            go.Scatter(
                x=Ps,
                y=Fs,
                mode="lines",
                name=f"R = {_comma(R, '.2g')}",
                line={"color": _ramp(k, len(family)), "width": 1.8},
                hovertemplate=f"R = {_comma(R, '.2g')}<br>P = %{{x:.3f}}"
                "<br>F = %{y:.3f}<extra></extra>",
            )
        )
    if all(abs(R0 - R) > 1e-6 for R in R_values):
        Ps, Fs = f_curves(arr, (R0,))[R0]
        if Ps:
            fig.add_trace(
                go.Scatter(
                    x=Ps,
                    y=Fs,
                    mode="lines",
                    name=f"R = {_comma(R0, '.3g')} (los datos)",
                    line={"color": _ORANGE, "width": 2.5, "dash": "dash"},
                    hovertemplate="P = %{x:.3f}<br>F = %{y:.3f}<extra></extra>",
                )
            )
    _point(fig, P0, result.F, "P = %{x:.4f}<br>F = %{y:.4f}<extra></extra>")
    _layout(
        fig,
        height=380,
        x_title="P = (t₂ − t₁)/(T₁ − t₁)",
        y_title="factor de corrección F",
        xaxis={"range": [0.0, 1.0]},
        yaxis={"range": [0.5, 1.02]},
    )
    return fig


# ---------------------------------------------------------------------
# Temperaturas a lo largo del doble tubo
# ---------------------------------------------------------------------


def profile_figure(
    result: ExchangerResult, *, t_scale: float, t_offset: float, t_unit: str
) -> go.Figure | None:
    """T_h y T_c contra la fracción del área desde la entrada del caliente (None si no es
    un doble tubo)."""
    pts = temperature_profile(result)
    if pts is None:
        return None
    xs = [p.x for p in pts]
    fig = go.Figure()
    for attr, name, color in (
        ("T_hot_K", f"caliente ({result.hot.name})", _ORANGE),
        ("T_cold_K", f"frío ({result.cold.name})", _BLUE),
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[getattr(p, attr) * t_scale + t_offset for p in pts],
                mode="lines",
                name=name,
                line={"color": color, "width": 2.8},
                hovertemplate=f"{name}<br>A_x/A = %{{x:.2f}}<br>T = %{{y:.4g}} {t_unit}"
                "<extra></extra>",
            )
        )
    _layout(
        fig,
        height=340,
        x_title="fracción del área desde la entrada del caliente",
        y_title=f"T [{t_unit}]",
        xaxis={"range": [0.0, 1.0]},
    )
    return fig


# ---------------------------------------------------------------------
# Comparación de tipos
# ---------------------------------------------------------------------


def short_name(arr: Arrangement) -> str:
    """El nombre corto de un tipo, para el eje del gráfico (los completos no entraban a
    390 px)."""
    if arr.kind == "parallel":
        return "Paralelo"
    if arr.kind == "counter":
        return "Contracorriente"
    if arr.kind == "shell":
        n = arr.shell_passes
        return f"Casco y tubos, {n} paso{'s' if n > 1 else ''}"
    if arr.kind == "cross_unmixed":
        return "Cruzado sin mezclar"
    return f"Cruzado, {'caliente' if arr.mixed == 'hot' else 'frío'} mezclado"


def type_comparison_figure(
    rows: Sequence[TypeComparison],
    current: str,
    *,
    area: bool,
    scale: float,
    unit: str,
) -> go.Figure:
    """Q̇ (verificación) o A (dimensionamiento) de cada tipo: el elegido en azul, los demás
    en gris; los que no pueden con los datos no tienen punto."""
    names = [short_name(r.arrangement) for r in rows]
    ok = [r for r in rows if (r.A_m2 if area else r.Q_W) is not None]
    values = [float((r.A_m2 if area else r.Q_W) or 0.0) * scale for r in ok]
    colors = [_BLUE if r.arrangement.name == current else _GRAY for r in ok]
    what = "A" if area else "Q̇"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=values,
            y=[short_name(r.arrangement) for r in ok],
            mode="markers+text",
            name=what,
            marker={"size": 12, "color": colors, "line": {"color": colors, "width": 2}},
            text=[_label(v) for v in values],
            textposition="top center",
            textfont={"color": _INK, "size": 11},
            cliponaxis=False,
            showlegend=False,
            hovertemplate=f"%{{y}}<br>{what} = %{{x:.4g}} {unit}<extra></extra>",
        )
    )
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    pad = 0.15 * (hi - lo or hi or 1.0)
    _layout(
        fig,
        height=110 + 48 * len(names),
        x_title=f"{what} [{unit}]",
        y_title="",
        xaxis={"range": [max(0.0, lo - pad), hi + pad]},
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(names)),
            "automargin": True,
            "tickfont": {"size": 11},
        },
    )
    return fig


# ---------------------------------------------------------------------
# U global
# ---------------------------------------------------------------------


def u_resistance_figure(result: OverallUResult) -> go.Figure:
    """El peso de cada resistencia (en % del total): el lado que manda."""
    names = [r.label for r in result.resistances]
    fig = go.Figure()
    for kind, color in (("convección", _BLUE), ("ensuciamiento", _ORANGE), ("pared", _VIOLET)):
        rows = [r for r in result.resistances if r.kind == kind]
        if not rows:
            continue
        shares = [100.0 * result.share(r) for r in rows]
        fig.add_trace(
            go.Bar(
                x=shares,
                y=[r.label for r in rows],
                orientation="h",
                name=kind,
                marker={"color": color},
                text=[_comma(s, ".3g") + " %" for s in shares],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.3g} % del total<extra></extra>",
            )
        )
    height = 110 + 40 * len(names)
    _layout(
        fig,
        height=height,
        x_title="del total [%]",
        y_title="",
        xaxis={"range": [0.0, 118.0]},
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(names)),
            "automargin": True,
        },
    )
    fig.update_layout(bargap=0.35)
    _legend_at_bottom(fig, height)
    return fig
