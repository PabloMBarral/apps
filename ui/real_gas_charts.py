"""Gráficos de los gases reales — Fase 9.2.

- :func:`generalized_chart_figure`: la carta generalizada (Z⁰ de Lee y Kesler contra
  p_R, con las isotermas T_R y la campana), con el estado real y el de la carta.
- :func:`z_isotherm_figure`: Z(p) a la temperatura del dato con los seis modelos.
- :func:`fluid_isotherm_figure`: p(v) a T del fluido real, de Van der Waals y de
  Peng–Robinson, con sus rectas de Maxwell y el estado.
- :func:`vdw_isotherms_figure`: las isotermas reducidas de Van der Waals con su lazo,
  su campana y las dos áreas iguales de una isoterma sombreadas.
- :func:`saturation_figure`: p_sat(T) real, de Van der Waals y de Peng–Robinson.
- :func:`convergence_figure`: cómo baja la diferencia entre los dos lados de cada
  relación de Maxwell al achicar los pasos (log–log, pendiente 2).
- :func:`inversion_figure`: la curva de inversión de Joule–Thomson con las líneas de h
  constante y el estado.
- :func:`clausius_figure`: p_sat(T) real contra la de Clausius–Clapeyron con h_fg
  constante.

Colores: la paleta de referencia validada para daltonismo. El fluido real (CoolProp)
va en tinta, gruesa; el gas ideal, gris rayado; Lee–Kesler con ω azul, Van der Waals
naranja, la carta (Z⁰) aguamarina y Peng–Robinson magenta, siempre los mismos. Las
familias de isotermas van en la rampa azul ordinal (validada en los intercambiadores)
con rótulos directos.

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en :mod:`core.gases`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from core.gases.cubic import vdw_reduced_p
from core.gases.relations import RelationsResult
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "MODEL_COLORS",
    "clausius_figure",
    "convergence_figure",
    "fluid_isotherm_figure",
    "generalized_chart_figure",
    "inversion_figure",
    "saturation_figure",
    "vdw_isotherms_figure",
    "z_isotherm_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_MAGENTA = "#e87ba4"
_GRAY = "#6b7280"
_LIGHT = "#9ca3af"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")

#: Cada modelo con su color y su trazo, siempre los mismos.
MODEL_COLORS: dict[str, str] = {
    "coolprop": _INK,
    "ideal": _GRAY,
    "lk0": _AQUA,
    "lk": _BLUE,
    "vdw": _ORANGE,
    "pr": _MAGENTA,
}
_MODEL_NAMES = {
    "coolprop": "real (CoolProp)",
    "ideal": "gas ideal",
    "lk0": "carta (Z⁰)",
    "lk": "Lee–Kesler con ω",
    "vdw": "Van der Waals",
    "pr": "Peng–Robinson",
}
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0}
_STATE_MARKER = {
    "size": 11,
    "color": _INK,
    "symbol": "diamond",
    "line": {"color": "white", "width": 2},
}


def _layout(
    fig: go.Figure,
    *,
    height: int,
    x_title: str,
    y_title: str,
    legend: dict[str, object] | None = None,
) -> None:
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": x_title, "gridcolor": _GRID, "zerolinecolor": _GRID},
        yaxis={"title": y_title, "gridcolor": _GRID, "zerolinecolor": _GRID},
        legend=legend or _LEGEND_BELOW,
        separators=", ",
        hoverlabel={"namelength": -1},
    )


def _comma(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",")


def _ramp(k: int, n: int) -> str:
    """El color k de n curvas de la rampa (de claro a oscuro)."""
    if n <= 1:
        return _RAMP[2]
    return _RAMP[round(k * (len(_RAMP) - 1) / (n - 1))]


def _conv(
    values: Sequence[float | None], kind: QuantityKind, system: UnitSystem
) -> list[float | None]:
    return [None if v is None else convert_from_si(float(v), kind, system) for v in values]


def _axis(name: str, kind: QuantityKind, system: UnitSystem) -> str:
    return f"{name} [{unit_label(kind, system)}]"


# ---------------------------------------------------------------------
# La carta generalizada
# ---------------------------------------------------------------------


def generalized_chart_figure(
    chart: dict[str, Any],
    state: tuple[float, float] | None = None,
    model: tuple[float, float] | None = None,
) -> go.Figure:
    """Z⁰ contra p_R con las isotermas T_R y la campana (Çengel, figura 3-51, calculada).

    ``state`` es (p_R, Z) del fluido real y ``model`` el (p_R, Z⁰) de la carta.
    """
    fig = go.Figure()
    isotherms = chart["isotherms"]
    n = len(isotherms)
    for k, (Tr, (pr, z)) in enumerate(sorted(isotherms.items())):
        color = _ramp(k, n)
        fig.add_trace(
            go.Scatter(
                x=pr,
                y=z,
                mode="lines",
                name="isotermas de T_R constante",
                legendgroup="iso",
                showlegend=k == 0,
                line={"color": color, "width": 1.8},
                hovertemplate=f"T_R = {_comma(Tr)}<br>p_R = %{{x:.3g}}<br>Z⁰ = %{{y:.3g}}"
                "<extra></extra>",
            )
        )
        x_lab, y_lab = _label_point(Tr, pr, z)
        fig.add_annotation(
            x=math.log10(x_lab),  # en un eje logarítmico, la posición va en log₁₀
            y=y_lab,
            text=_comma(Tr),
            showarrow=False,
            font={"size": 11, "color": _INK},
            bgcolor="rgba(255,255,255,0.75)",
            xshift=-12 if Tr < 1.0 else 0,
            yshift=0 if Tr < 1.0 else 8,
        )
    prs, zf, zg = chart["dome"]
    fig.add_trace(
        go.Scatter(
            x=[*prs, *reversed(prs)],
            y=[*zf, *reversed(zg)],
            mode="lines",
            name="campana (saturación)",
            line={"color": _GRAY, "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    if model is not None:
        fig.add_trace(
            go.Scatter(
                x=[model[0]],
                y=[model[1]],
                mode="markers",
                name="la carta (Z⁰)",
                marker={"size": 14, "color": _AQUA, "line": {"color": "white", "width": 2}},
                hovertemplate="carta: p_R = %{x:.3g}, Z⁰ = %{y:.4g}<extra></extra>",
            )
        )
    if state is not None:
        fig.add_trace(
            go.Scatter(
                x=[state[0]],
                y=[state[1]],
                mode="markers",
                name="el estado (CoolProp)",
                marker=_STATE_MARKER,
                hovertemplate="real: p_R = %{x:.3g}, Z = %{y:.4g}<extra></extra>",
            )
        )
    _layout(fig, height=420, x_title="presión reducida p_R", y_title="Z")
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    fig.update_xaxes(
        type="log",
        range=[math.log10(0.009), math.log10(11.0)],
        tickvals=ticks,
        ticktext=[_comma(t, "g") for t in ticks],
    )
    fig.update_yaxes(range=[0, 1.65])
    return fig


def _label_point(Tr: float, pr: Sequence[float], z: Sequence[float]) -> tuple[float, float]:
    """Dónde va el rótulo de una isoterma: el vapor saturado bajo T_R = 1, el mínimo hasta
    T_R = 2 y p_R = 5 arriba (donde las curvas se separan)."""
    xs, zs = np.asarray(pr), np.asarray(z)
    if Tr < 1.0:  # a media altura del salto de la saturación (no tapa el vapor saturado)
        jumps = np.nonzero(np.diff(zs) < -0.2)[0]
        k = int(jumps[0]) if len(jumps) else 0
        return float(xs[k]), 0.5 * float(zs[k] + zs[min(k + 1, len(zs) - 1)])
    if Tr <= 2.0:
        k = int(np.argmin(zs))
        return float(xs[k]), float(zs[k])
    return 5.0, float(np.interp(5.0, xs, zs))


# ---------------------------------------------------------------------
# Z(p) a la T del dato
# ---------------------------------------------------------------------


def z_isotherm_figure(
    data: dict[str, Any], system: UnitSystem, state: tuple[float, float] | None = None
) -> go.Figure:
    """Z contra p a la temperatura del dato con los seis modelos; ``state`` = (p, Z)."""
    p_kind: QuantityKind = "pressure"
    x = _conv(data["p"], p_kind, system)
    fig = go.Figure()
    for kind in ("coolprop", "ideal", "lk0", "lk", "vdw", "pr"):
        zs = data["Z"][kind]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=zs,
                mode="lines",
                name=_MODEL_NAMES[kind],
                connectgaps=False,
                line={
                    "color": MODEL_COLORS[kind],
                    "width": 4 if kind == "coolprop" else 2,
                    "dash": "dash" if kind == "ideal" else "solid",
                },
                hovertemplate=f"{_MODEL_NAMES[kind]}: Z = %{{y:.4g}}<br>p = %{{x:.4g}} "
                f"{unit_label(p_kind, system)}<extra></extra>",
            )
        )
    if data.get("p_sat") is not None:
        x_sat = convert_from_si(data["p_sat"], p_kind, system)
        fig.add_vline(
            x=x_sat,
            line={"color": _LIGHT, "width": 1, "dash": "dot"},
            annotation_text="p_sat",
            annotation_position="top",
            annotation_font={"color": _GRAY, "size": 11},
        )
    if state is not None:
        fig.add_trace(
            go.Scatter(
                x=[convert_from_si(state[0], p_kind, system)],
                y=[state[1]],
                mode="markers",
                name="el estado",
                marker=_STATE_MARKER,
                hovertemplate="el estado: Z = %{y:.4g}<extra></extra>",
            )
        )
    _layout(fig, height=400, x_title=_axis("presión", p_kind, system), y_title="Z = p·v/(R·T)")
    return fig


# ---------------------------------------------------------------------
# Las isotermas p(v)
# ---------------------------------------------------------------------


def fluid_isotherm_figure(
    data: dict[str, Any],
    system: UnitSystem,
    state: tuple[float, float] | None = None,
    p_cr: float | None = None,
) -> go.Figure:
    """p(v) a T: el fluido real, Van der Waals y Peng–Robinson con sus rectas de Maxwell.

    ``state`` es (v, p) del dato. El eje de v es logarítmico; el de p se recorta para
    que se vea el lazo (Van der Waals llega a presiones negativas).
    """
    p_kind: QuantityKind = "pressure"
    v_kind: QuantityKind = "specific_volume"
    xs = _conv(data["v"], v_kind, system)
    fig = go.Figure()
    for kind in ("coolprop", "vdw", "pr"):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=_conv(data["p"][kind], p_kind, system),
                mode="lines",
                name=_MODEL_NAMES[kind],
                legendgroup=kind,
                connectgaps=False,
                line={"color": MODEL_COLORS[kind], "width": 4 if kind == "coolprop" else 2},
                hovertemplate=f"{_MODEL_NAMES[kind]}<br>v = %{{x:.4g}}<br>p = %{{y:.4g}}"
                "<extra></extra>",
            )
        )
    sats: list[float] = [s for s in (data.get("p_sat"),) if s is not None]
    for kind, sat in data["maxwell"].items():
        if sat is None:
            continue
        sats.append(sat.p_sat_Pa)
        p_line = convert_from_si(sat.p_sat_Pa, p_kind, system)
        fig.add_trace(
            go.Scatter(
                x=_conv([sat.v_f, sat.v_g], v_kind, system),
                y=[p_line, p_line],
                mode="lines+markers",
                name=f"Maxwell, {_MODEL_NAMES[kind]}",
                legendgroup=kind,
                showlegend=False,
                line={"color": MODEL_COLORS[kind], "width": 2, "dash": "dash"},
                marker={"size": 7, "color": MODEL_COLORS[kind]},
                hovertemplate=f"p_sat de {_MODEL_NAMES[kind]} = %{{y:.4g}}<extra></extra>",
            )
        )
    ref = max([*sats, state[1] if state else 0.0, 0.5 * (p_cr or 0.0)])
    top = convert_from_si(1.8 * ref, p_kind, system)
    fig.update_yaxes(range=[-0.6 * top, top])
    if state is not None:
        fig.add_trace(
            go.Scatter(
                x=[convert_from_si(state[0], v_kind, system)],
                y=[convert_from_si(state[1], p_kind, system)],
                mode="markers",
                name="el estado",
                marker=_STATE_MARKER,
                hovertemplate="el estado<extra></extra>",
            )
        )
    _layout(
        fig,
        height=420,
        x_title=_axis("volumen específico", v_kind, system),
        y_title=_axis("presión", p_kind, system),
    )
    fig.update_xaxes(type="log")
    return fig


def _middle_root(Tr: float, ps: float) -> float:
    """El v_R del medio de Van der Waals reducido a p_R: 3p·v³ − (p + 8T)·v² + 9v − 3 = 0."""
    roots = sorted(
        float(r.real) for r in np.roots([3 * ps, -(ps + 8 * Tr), 9, -3]) if abs(r.imag) < 1e-9
    )
    return roots[1] if len(roots) == 3 else roots[0]


def vdw_isotherms_figure(iso: dict[str, Any], highlight: float | None = None) -> go.Figure:
    """Las isotermas reducidas de Van der Waals (vademecum §8.4) con la campana y, en la
    T_R ``highlight``, las dos áreas iguales de la construcción de Maxwell sombreadas."""
    fig = go.Figure()
    family = sorted(iso["isotherms"].items())
    n = len(family)
    for k, (Tr, (vr, pr)) in enumerate(family):
        fig.add_trace(
            go.Scatter(
                x=vr,
                y=pr,
                mode="lines",
                name=f"T_R = {_comma(Tr)}",
                line={"color": _ramp(k, n), "width": 2.4 if Tr == highlight else 1.6},
                hovertemplate=f"T_R = {_comma(Tr)}<br>v_R = %{{x:.3g}}<br>p_R = %{{y:.3g}}"
                "<extra></extra>",
            )
        )
    dv, dp = iso["dome"]
    fig.add_trace(
        go.Scatter(
            x=dv,
            y=dp,
            mode="lines",
            name="campana de Van der Waals",
            line={"color": _GRAY, "width": 1.5, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    if highlight is not None and highlight in iso["maxwell"]:
        ps, vf, vg = iso["maxwell"][highlight]
        vm = _middle_root(highlight, ps)
        for k, ((a, b), color) in enumerate(
            (((vf, vm), "rgba(235, 104, 52, 0.35)"), ((vm, vg), "rgba(42, 120, 214, 0.45)"))
        ):
            vs = np.geomspace(a, b, 60)
            ys = vdw_reduced_p(highlight, vs)
            fig.add_trace(
                go.Scatter(
                    x=[*vs, b, a],
                    y=[*ys, ps, ps],
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    line={"width": 0},
                    name="las dos áreas iguales",
                    legendgroup="areas",
                    showlegend=k == 0,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[vf, vg],
                y=[ps, ps],
                mode="lines+markers",
                name=f"Maxwell a T_R = {_comma(highlight)}",
                line={"color": _INK, "width": 2},
                marker={"size": 7, "color": _INK},
                hovertemplate="p_R,sat = %{y:.4g}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[1.0],
            mode="markers",
            name="punto crítico",
            marker={"size": 9, "color": _INK, "symbol": "circle-open", "line": {"width": 2}},
            hovertemplate="punto crítico (v_R = p_R = T_R = 1)<extra></extra>",
        )
    )
    _layout(fig, height=460, x_title="volumen reducido v_R = v/(3b)", y_title="p_R")
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0})
    fig.update_xaxes(type="log", range=[math.log10(0.4), math.log10(20.0)])
    fig.update_yaxes(range=[-1.0, 2.2])
    return fig


def saturation_figure(curves: dict[str, Any], system: UnitSystem) -> go.Figure:
    """p_sat(T) del fluido real, de Van der Waals y de Peng–Robinson (eje p logarítmico)."""
    T_kind: QuantityKind = "temperature"
    p_kind: QuantityKind = "pressure"
    xs = _conv(curves["T"], T_kind, system)
    fig = go.Figure()
    for kind in ("coolprop", "vdw", "pr"):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=_conv(curves["p_sat"][kind], p_kind, system),
                mode="lines",
                name=_MODEL_NAMES[kind],
                line={"color": MODEL_COLORS[kind], "width": 4 if kind == "coolprop" else 2},
                hovertemplate=f"{_MODEL_NAMES[kind]}<br>T = %{{x:.4g}}<br>p_sat = %{{y:.4g}}"
                "<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[convert_from_si(curves["T_cr"], T_kind, system)],
            y=[convert_from_si(curves["p_cr"], p_kind, system)],
            mode="markers",
            name="punto crítico",
            marker={"size": 9, "color": _INK, "symbol": "circle-open", "line": {"width": 2}},
            hovertemplate="punto crítico<extra></extra>",
        )
    )
    _layout(
        fig,
        height=380,
        x_title=_axis("temperatura", T_kind, system),
        y_title=_axis("presión de saturación", p_kind, system),
    )
    fig.update_yaxes(type="log")
    return fig


# ---------------------------------------------------------------------
# Relaciones de Maxwell, Joule–Thomson y Clapeyron
# ---------------------------------------------------------------------

_RELATION_NAMES = {
    "u": "(∂T/∂v)_s = −(∂p/∂s)_v",
    "h": "(∂T/∂p)_s = (∂v/∂s)_p",
    "f": "(∂s/∂v)_T = (∂p/∂T)_v",
    "g": "(∂s/∂p)_T = −(∂v/∂T)_p",
}
_RELATION_COLORS = {"u": _BLUE, "h": _ORANGE, "f": _AQUA, "g": _MAGENTA}


def convergence_figure(r: RelationsResult) -> go.Figure:
    """La diferencia relativa entre los dos lados de cada relación contra el paso (log–log):
    con diferencias centradas baja como Δ² (la recta gris)."""
    fig = go.Figure()
    fracs = [f for f, _ in r.convergence]
    low = math.inf
    for k, m in enumerate(r.maxwell):
        gaps = [100.0 * abs(g[k]) for _, g in r.convergence]
        low = min(low, gaps[0])
        fig.add_trace(
            go.Scatter(
                x=fracs,
                y=gaps,
                mode="lines+markers",
                name=_RELATION_NAMES[m.potential],
                line={"color": _RELATION_COLORS[m.potential], "width": 2},
                marker={"size": 8},
                hovertemplate="paso ×%{x}<br>diferencia = %{y:.3g} %<extra></extra>",
            )
        )
    if 0 < low < math.inf:
        fig.add_trace(
            go.Scatter(
                x=fracs,
                y=[0.5 * low * f**2 for f in fracs],
                mode="lines",
                name="pendiente 2 (error ∝ Δ²)",
                line={"color": _GRAY, "width": 1.5, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    _layout(
        fig,
        height=380,
        x_title="fracción de los pasos ΔT y Δp",
        y_title="diferencia entre los lados [%]",
    )
    fig.update_xaxes(type="log", tickvals=fracs, ticktext=["1", "1/2", "1/4"])
    fig.update_yaxes(type="log")
    return fig


def inversion_figure(
    curve: dict[str, list[float]],
    lines: dict[float, tuple[list[float], list[float]]],
    system: UnitSystem,
    state: tuple[float, float] | None = None,
) -> go.Figure:
    """La curva de inversión de Joule–Thomson en el plano T–p: adentro (sombreado) el gas
    se enfría al estrangularlo; las líneas de h constante tienen su máximo sobre ella.
    ``state`` = (p, T)."""
    T_kind: QuantityKind = "temperature"
    p_kind: QuantityKind = "pressure"
    fig = go.Figure()
    for k, (_T0, (ps, Ts)) in enumerate(sorted(lines.items())):
        fig.add_trace(
            go.Scatter(
                x=_conv(ps, p_kind, system),
                y=_conv(Ts, T_kind, system),
                mode="lines",
                name="h constante (una válvula)",
                legendgroup="h",
                showlegend=k == 0,
                line={"color": _LIGHT, "width": 1.3},
                hoverinfo="skip",
            )
        )
    xs = _conv(curve["p"], p_kind, system)
    ys = _conv(curve["T"], T_kind, system)
    # la zona donde se enfría: la curva y, si la corta la ecuación de estado, su borde
    ex = _conv(curve.get("p_edge", []), p_kind, system)
    ey = _conv(curve.get("T_edge", []), T_kind, system)
    zone_x, zone_y = [*xs, *ex], [*ys, *ey]
    fig.add_trace(
        go.Scatter(
            x=[*zone_x, zone_x[0]] if zone_x else [],
            y=[*zone_y, zone_y[0]] if zone_y else [],
            mode="lines",
            fill="toself",
            fillcolor="rgba(42, 120, 214, 0.12)",
            line={"width": 0},
            legendgroup="inv",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="curva de inversión (adentro se enfría)",
            legendgroup="inv",
            line={"color": _BLUE, "width": 2.5},
            hovertemplate="μ_JT = 0<br>p = %{x:.4g}<br>T = %{y:.4g}<extra></extra>",
        )
    )
    if ex:
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.15 * max(x for x in zone_x if x is not None)],
                y=[ey[0], ey[0]],
                mode="lines",
                name="T máxima de la ecuación de estado",
                line={"color": _GRAY, "width": 1.5, "dash": "dot"},
                hovertemplate="T máx. = %{y:.4g}<extra></extra>",
            )
        )
    if state is not None:
        fig.add_trace(
            go.Scatter(
                x=[convert_from_si(state[0], p_kind, system)],
                y=[convert_from_si(state[1], T_kind, system)],
                mode="markers",
                name="el estado",
                marker=_STATE_MARKER,
                hovertemplate="el estado<extra></extra>",
            )
        )
    _layout(
        fig,
        height=420,
        x_title=_axis("presión", p_kind, system),
        y_title=_axis("temperatura", T_kind, system),
    )
    return fig


def clausius_figure(
    curve: dict[str, list[float]],
    system: UnitSystem,
    point: tuple[float, float],
    second: tuple[float, float, float] | None = None,
) -> go.Figure:
    """p_sat(T) real contra Clausius–Clapeyron con h_fg constante (eje p logarítmico).

    ``point`` = (T₁, p₁); ``second`` = (T₂, p₂ real, p₂ de Clausius–Clapeyron).
    """
    T_kind: QuantityKind = "temperature"
    p_kind: QuantityKind = "pressure"
    xs = _conv(curve["T"], T_kind, system)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=_conv(curve["cc"], p_kind, system),
            mode="lines",
            name="Clausius–Clapeyron (h_fg constante)",
            line={"color": _ORANGE, "width": 2, "dash": "dash"},
            hovertemplate="Clausius–Clapeyron: %{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=_conv(curve["real"], p_kind, system),
            mode="lines",
            name="real (CoolProp)",
            line={"color": _INK, "width": 3},
            hovertemplate="real: %{y:.4g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[convert_from_si(point[0], T_kind, system)],
            y=[convert_from_si(point[1], p_kind, system)],
            mode="markers",
            name="T₁ (el dato)",
            marker=_STATE_MARKER,
            hovertemplate="T₁<extra></extra>",
        )
    )
    if second is not None:
        T2, p2, p2cc = second
        fig.add_trace(
            go.Scatter(
                x=[convert_from_si(T2, T_kind, system)] * 2,
                y=_conv([p2, p2cc], p_kind, system),
                mode="markers",
                name="T₂: real y Clausius–Clapeyron",
                marker={
                    "size": 9,
                    "color": [_INK, _ORANGE],
                    "line": {"color": "white", "width": 2},
                },
                hovertemplate="T₂: %{y:.4g}<extra></extra>",
            )
        )
    _layout(
        fig,
        height=380,
        x_title=_axis("temperatura", T_kind, system),
        y_title=_axis("presión de saturación", p_kind, system),
    )
    fig.update_yaxes(type="log")
    return fig
