"""Gráficos de la radiación — Fase 8.2.

- :func:`planck_figure`: E_bλ(λ) del cuerpo negro con λ_máx, la banda pedida
  sombreada y, con ε(λ) en bandas, la emisión real ε·E_bλ.
- :func:`spectral_match_figure`: ε(λ) junto a los espectros normalizados de la
  superficie y de la fuente: por qué α ≠ ε (un absorbedor selectivo).
- :func:`view_factor_figure`: F_ij contra la separación, con el punto de los datos.
- :func:`network_figure`: el peso de cada resistencia de la red de radiación.
- :func:`enclosure_figure`: el calor neto de cada superficie del recinto.
- :func:`surface_curve_figure`: q_conv y q_rad contra T_s.

Colores: la paleta validada para daltonismo del proyecto (azul el cuerpo negro y
lo frío, naranja la superficie real, lo caliente y el punto de los datos,
violeta las resistencias del espacio).

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.heat_transfer.radiation`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import plotly.graph_objects as go

from core.heat_transfer.radiation import (
    C3,
    BlackbodyResult,
    EnclosureResult,
    SurfaceBalanceResult,
    TwoSurfaceResult,
    planck,
)

__all__ = [
    "enclosure_figure",
    "network_figure",
    "planck_figure",
    "spectral_match_figure",
    "surface_curve_figure",
    "view_factor_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_VIOLET = "#4a3aa7"
_GRAY = "#6b7280"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_BAND_FILL = "rgba(107, 114, 128, 0.12)"
_ORANGE_FILL = "rgba(235, 104, 52, 0.18)"
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0}
_UM = 1e-6


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


def _log_ticks(lo: float, hi: float) -> dict[str, object]:
    """Marcas 1-2-5 con el número completo en un eje logarítmico (plotly rotula 0,2 como
    «2» en las marcas intermedias)."""
    vals: list[float] = []
    for exp in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1.0, 2.0, 5.0):
            v = m * 10.0**exp
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                vals.append(v)
    return {"tickvals": vals, "ticktext": [_comma(v, ".3g") for v in vals]}


def _epsilon_at(result: BlackbodyResult, lam_m: float) -> float:
    """ε(λ) de la superficie escalonada (1 si no hay bandas)."""
    if not result.rows:
        return 1.0
    for row in result.rows:
        if row.upper_m is None or lam_m < row.upper_m:
            return row.emissivity
    return result.rows[-1].emissivity


# ---------------------------------------------------------------------
# Cuerpo negro
# ---------------------------------------------------------------------


def planck_figure(result: BlackbodyResult, *, e_scale: float, e_unit: str) -> go.Figure:
    """E_bλ(λ) a la temperatura de la superficie (λ en μm, E_bλ por μm).

    ``e_scale`` pasa de W/(m²·μm) a la unidad del sistema. Con bandas se dibuja
    también ε(λ)·E_bλ (rellena: su área es E); la banda pedida va sombreada.
    """
    i = result.inputs
    lam_max = result.lambda_max_m
    hi = 6.0 * lam_max
    for edge in (i.band_upper_m, i.band_lower_m, result.lambda_target_m, i.lambda_point_m):
        if edge is not None:
            hi = max(hi, 1.15 * edge)
    hi = min(hi, 25.0 * lam_max)
    n = 240
    lams = [hi * k / n for k in range(1, n + 1)]
    for row in result.rows:  # los cortes de las bandas, para que el escalón se vea vertical
        if row.upper_m is not None and row.upper_m < hi:
            lams += [row.upper_m * (1 - 1e-9), row.upper_m * (1 + 1e-9)]
    lams.sort()
    xs = [x / _UM for x in lams]

    def E(lam: float) -> float:  # W/m³ → unidad del sistema por μm
        return planck(lam, i.T_K) * _UM * e_scale

    fig = go.Figure()
    if i.band_lower_m is not None or i.band_upper_m is not None:
        lo_b = (i.band_lower_m or 0.0) / _UM
        hi_b = (i.band_upper_m or hi) / _UM
        fig.add_vrect(
            x0=lo_b,
            x1=min(hi_b, hi / _UM),
            fillcolor=_BAND_FILL,
            line_width=0,
            layer="below",
            annotation_text=f"banda: {_comma(100.0 * (result.band_fraction or 0.0), '.3g')} %",
            annotation_position="top left",
            annotation_font={"size": 11, "color": _INK},
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[E(x) for x in lams],
            mode="lines",
            name=f"cuerpo negro a {_comma(i.T_K, '.5g')} K",
            line={"color": _BLUE, "width": 2.5},
            hovertemplate=f"λ = %{{x:.3g}} μm<br>E_bλ = %{{y:.4g}} {e_unit}<extra></extra>",
        )
    )
    if result.rows:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[_epsilon_at(result, x) * E(x) for x in lams],
                mode="lines",
                name="superficie real: ε(λ)·E_bλ",
                line={"color": _ORANGE, "width": 2.5},
                fill="tozeroy",
                fillcolor=_ORANGE_FILL,
                hovertemplate=f"λ = %{{x:.3g}} μm<br>ε·E_bλ = %{{y:.4g}} {e_unit}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[lam_max / _UM],
            y=[E(lam_max)],
            mode="markers+text",
            name="λ_máx (Wien)",
            marker={"size": 9, "color": _BLUE, "line": {"color": _INK, "width": 1}},
            text=[f"λ_máx = {_comma(lam_max / _UM, '.3g')} μm"],
            textposition="top right",
            textfont={"size": 11, "color": _INK},
            cliponaxis=False,
            showlegend=False,
            hovertemplate=f"λ_máx = %{{x:.4g}} μm<br>%{{y:.4g}} {e_unit}<extra></extra>",
        )
    )
    if i.lambda_point_m is not None:
        lam = i.lambda_point_m
        fig.add_trace(
            go.Scatter(
                x=[lam / _UM],
                y=[E(lam)],
                mode="markers",
                name="E_bλ en la λ pedida",
                marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
                hovertemplate=f"λ = %{{x:.4g}} μm<br>E_bλ = %{{y:.4g}} {e_unit}<extra></extra>",
            )
        )
    if result.lambda_target_m is not None and i.fraction_target is not None:
        fig.add_vline(
            x=result.lambda_target_m / _UM,
            line={"color": _GRAY, "width": 1.5, "dash": "dot"},
            annotation_text=f"{_comma(100.0 * i.fraction_target, '.3g')} % antes",
            annotation_position="top right",
        )
    _layout(
        fig,
        height=360,
        x_title="λ [μm]",
        y_title=f"E_bλ [{e_unit}]",
        xaxis={"range": [0.0, hi / _UM]},
        yaxis={"rangemode": "tozero"},
    )
    return fig


def spectral_match_figure(result: BlackbodyResult) -> go.Figure:
    """ε(λ) y los espectros de la superficie y de la fuente, cada uno sobre su máximo.

    Todo es adimensional entre 0 y 1 (un solo eje): donde la fuente emite, la
    superficie absorbe con su ε(λ) (α); donde emite la superficie, pesa su ε.
    """
    i = result.inputs
    assert i.T_source_K is not None
    lo = 0.2 * C3 / max(i.T_K, i.T_source_K)
    hi = 12.0 * C3 / min(i.T_K, i.T_source_K)
    n = 300
    lams = [lo * (hi / lo) ** (k / (n - 1)) for k in range(n)]
    for row in result.rows:
        if row.upper_m is not None and lo < row.upper_m < hi:
            lams += [row.upper_m * (1 - 1e-9), row.upper_m * (1 + 1e-9)]
    lams.sort()
    xs = [x / _UM for x in lams]
    fig = go.Figure()
    for T, name, color in (
        (i.T_source_K, f"fuente a {_comma(i.T_source_K, '.5g')} K", _ORANGE),
        (i.T_K, f"superficie a {_comma(i.T_K, '.5g')} K", _BLUE),
    ):
        peak = planck(C3 / T, T)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[planck(x, T) / peak for x in lams],
                mode="lines",
                name=f"{name} (E_bλ/E_bλ,máx)",
                line={"color": color, "width": 2.5},
                hovertemplate=f"{name}<br>λ = %{{x:.3g}} μm<br>%{{y:.3f}}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[_epsilon_at(result, x) for x in lams],
            mode="lines",
            name="ε(λ) = α(λ)",
            line={"color": _INK, "width": 2, "dash": "dash"},
            hovertemplate="λ = %{x:.3g} μm<br>ε = %{y:.3f}<extra></extra>",
        )
    )
    _layout(
        fig,
        height=360,
        x_title="λ [μm]",
        y_title="ε, y cada espectro sobre su máximo",
        xaxis={
            "type": "log",
            "range": [math.log10(lo / _UM), math.log10(hi / _UM)],
            **_log_ticks(lo / _UM, hi / _UM),
        },
        yaxis={"range": [0.0, 1.08]},
    )
    return fig


# ---------------------------------------------------------------------
# Factor de forma
# ---------------------------------------------------------------------


def view_factor_figure(
    label: str,
    xs: Sequence[float],
    Fs: Sequence[float],
    x0: float,
    F0: float,
    *,
    x_scale: float,
    x_unit: str,
    log_x: bool,
) -> go.Figure:
    """F_ij contra la medida que cambia (la separación, casi siempre)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[x * x_scale for x in xs],
            y=list(Fs),
            mode="lines",
            name="F_ij",
            line={"color": _BLUE, "width": 2.5},
            hovertemplate=f"{label} = %{{x:.4g}} {x_unit}<br>F = %{{y:.4f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0 * x_scale],
            y=[F0],
            mode="markers",
            name="los datos",
            marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=f"{label} = %{{x:.4g}} {x_unit}<br>F = %{{y:.4f}}<extra></extra>",
        )
    )
    _layout(
        fig,
        height=330,
        x_title=f"{label} [{x_unit}]",
        y_title="F_ij",
        xaxis=(
            {"type": "log", **_log_ticks(min(xs) * x_scale, max(xs) * x_scale)} if log_x else None
        ),
        yaxis={"range": [0.0, 1.02]},
    )
    return fig


# ---------------------------------------------------------------------
# Dos superficies y recintos
# ---------------------------------------------------------------------


def network_figure(result: TwoSurfaceResult) -> go.Figure:
    """El peso de cada resistencia de la red (en % del total): dónde está el freno."""
    names = [r.label for r in result.resistances]
    fig = go.Figure()
    for kind, color, label in (
        ("superficie", _BLUE, "de superficie (1 − ε)/(A·ε)"),
        ("espacio", _VIOLET, "del espacio 1/(A·F)"),
    ):
        rows = [r for r in result.resistances if r.kind == kind]
        if not rows:
            continue
        shares = [100.0 * result.share(r) for r in rows]
        fig.add_trace(
            go.Bar(
                x=shares,
                y=[r.label for r in rows],
                orientation="h",
                name=label,
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
        x_title="del total de la red [%]",
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


def enclosure_figure(result: EnclosureResult, *, q_scale: float, q_unit: str) -> go.Figure:
    """El calor neto de cada superficie: naranja la que entrega, azul la que recibe."""
    names = [s.name for s in result.inputs.surfaces]
    qs = [q * q_scale for q in result.Q_W]
    big = max(abs(q) for q in qs) or 1.0
    fig = go.Figure()
    sides = ((True, _ORANGE, "entrega calor"), (False, _BLUE, "recibe calor"))
    for positive, color, label in sides:
        rows = [
            (n, q)
            for n, q in zip(names, qs, strict=True)
            if (q > 1e-9 * big if positive else q < -1e-9 * big)
        ]
        if not rows:
            continue
        fig.add_trace(
            go.Bar(
                x=[q for _, q in rows],
                y=[n for n, _ in rows],
                orientation="h",
                name=label,
                marker={"color": color},
                text=[_label(q) for _, q in rows],
                # adentro: afuera, el de una barra negativa se pisaba con el nombre
                textposition="inside",
                insidetextanchor="middle",
                textfont={"color": "white"},
                hovertemplate=f"%{{y}}<br>Q̇ = %{{x:.4g}} {q_unit}<extra></extra>",
            )
        )
    zero = [n for n, q in zip(names, qs, strict=True) if abs(q) <= 1e-9 * big]
    if zero:  # las rerradiantes: un punto sobre el cero, para que no falten
        fig.add_trace(
            go.Scatter(
                x=[0.0] * len(zero),
                y=zero,
                mode="markers+text",
                name="rerradiante (Q̇ = 0)",
                marker={"size": 9, "color": _GRAY, "symbol": "diamond"},
                text=["0"] * len(zero),
                textposition="middle right",
                hovertemplate="%{y}<br>Q̇ = 0<extra></extra>",
            )
        )
    pad = 0.35 * big
    height = 110 + 48 * len(names)
    _layout(
        fig,
        height=height,
        x_title=f"Q̇ neto [{q_unit}]",
        y_title="",
        xaxis={
            "range": [min(0.0, min(qs)) - pad, max(0.0, max(qs)) + pad],
            "zeroline": True,
            "zerolinecolor": _INK,
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(names)),
            "automargin": True,
        },
    )
    fig.update_layout(bargap=0.35, barmode="relative", uniformtext={"minsize": 10, "mode": "hide"})
    _legend_at_bottom(fig, height)
    return fig


# ---------------------------------------------------------------------
# Convección y radiación
# ---------------------------------------------------------------------


def surface_curve_figure(
    curve: tuple[Sequence[float], Sequence[float], Sequence[float]],
    result: SurfaceBalanceResult,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    q_scale: float,
    q_unit: str,
) -> go.Figure:
    """q_conv, q_rad y la suma contra T_s (por unidad de área), con el punto de los datos."""
    Ts, q_conv, q_rad = curve
    x = [T * t_scale + t_offset for T in Ts]
    fig = go.Figure()
    for ys, name, color, dash in (
        (q_conv, "convección h·(T_s − T∞)", _BLUE, "solid"),
        (q_rad, "radiación ε·σ·(T_s⁴ − T_alr⁴)", _ORANGE, "solid"),
        ([a + b for a, b in zip(q_conv, q_rad, strict=True)], "las dos", _INK, "dash"),
    ):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[y * q_scale for y in ys],
                mode="lines",
                name=name,
                line={"color": color, "width": 2.5 if dash == "solid" else 2, "dash": dash},
                hovertemplate=f"T_s = %{{x:.4g}} {t_unit}<br>q = %{{y:.4g}} {q_unit}"
                "<extra></extra>",
            )
        )
    i = result.inputs
    q_now = (result.Q_conv_W + result.Q_rad_W) / i.area_m2
    fig.add_trace(
        go.Scatter(
            x=[result.T_s_K * t_scale + t_offset],
            y=[q_now * q_scale],
            mode="markers",
            name="los datos",
            marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=f"T_s = %{{x:.4g}} {t_unit}<br>q = %{{y:.4g}} {q_unit}<extra></extra>",
        )
    )
    _layout(
        fig,
        height=360,
        x_title=f"T_s [{t_unit}]",
        y_title=f"q″ que pierde [{q_unit}]",
        yaxis={"zeroline": True, "zerolinecolor": _GRAY},
    )
    return fig
