"""Gráficos de transferencia de calor — Fase 8.1.

- :func:`conduction_profile_figure`: la temperatura a través de la pared, el
  caño o la esfera, con cada capa sombreada y los fluidos de los dos lados (la
  caída en la película de convección se dibuja punteada).
- :func:`insulation_figure`: Q̇ (o la temperatura del lado 1) contra el radio
  exterior, con el radio crítico.
- :func:`fin_profile_figure`: T(x) a lo largo de la aleta, con las otras
  condiciones de la punta en gris.
- :func:`fin_efficiency_figure`: η contra m·L_c (la curva de los libros).
- :func:`nu_curve_figure`: la correlación como Nu(Re) o Nu(Ra), en escala log.
- :func:`correlation_figure`: el h de cada correlación (gráfico de puntos: las
  diferencias son de pocos %, unas barras desde cero las esconderían).
- :func:`tube_figure`: la temperatura media del fluido y la de la pared a lo
  largo del tubo.

Colores: la paleta validada para daltonismo del proyecto (azul el sólido o la
correlación usada, naranja los fluidos y el punto de los datos, gris lo que se
compara).

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.heat_transfer`.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from core.heat_transfer.conduction import (
    ConductionResult,
    SweepPoint,
    temperature_profile,
)
from core.heat_transfer.convection import Alternative, InternalFlowResult, tube_profile
from core.heat_transfer.fins import FinResult, TipCondition, efficiency_curve, fin_profile

__all__ = [
    "conduction_profile_figure",
    "correlation_figure",
    "fin_efficiency_figure",
    "fin_profile_figure",
    "insulation_figure",
    "nu_curve_figure",
    "tube_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_VIOLET = "#4a3aa7"
_GRAY = "#6b7280"
_LIGHT_GRAY = "#9ca3af"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_LAYER_FILLS = ("rgba(42, 120, 214, 0.10)", "rgba(42, 120, 214, 0.20)")
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0}

_TIP_NAMES: dict[TipCondition, str] = {
    "convective": "punta convectiva",
    "adiabatic": "punta adiabática",
    "corrected": "longitud corregida",
    "temperature": "punta a T dada",
    "infinite": "aleta infinita",
}


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


# ---------------------------------------------------------------------
# Conducción
# ---------------------------------------------------------------------


def conduction_profile_figure(
    result: ConductionResult,
    *,
    t_scale: float,
    t_offset: float,
    t_unit: str,
    x_scale: float,
    x_unit: str,
) -> go.Figure:
    """T contra x (pared) o r (cilindro y esfera) a través de las capas.

    ``t_scale``/``t_offset`` pasan de K a la temperatura del sistema y
    ``x_scale`` de m a la unidad de largo.
    """
    i = result.inputs
    faces = result.faces_m
    width = faces[-1] - faces[0]
    fig = go.Figure()
    T_all: list[float] = []

    def T(x: float) -> float:
        return x * t_scale + t_offset

    for k, (x1, x2) in enumerate(zip(faces, faces[1:], strict=False)):
        fig.add_vrect(
            x0=x1 * x_scale,
            x1=x2 * x_scale,
            fillcolor=_LAYER_FILLS[k % 2],
            line_width=0,
            layer="below",
            annotation_text=str(k + 1),
            annotation_position="top",
            annotation_font={"size": 11, "color": _INK},
        )
    for n_seg, seg in enumerate(temperature_profile(result)):
        fig.add_trace(
            go.Scatter(
                x=[x * x_scale for x in seg.x_m],
                y=[T(t) for t in seg.T_K],
                mode="lines",
                name="sólido",
                legendgroup="solid",
                showlegend=n_seg == 0,
                line={"color": _BLUE, "width": 3},
                hovertemplate=f"capa {seg.layer + 1}<br>%{{x:.4g}} {x_unit}"
                f"<br>T = %{{y:.4g}} {t_unit}<extra></extra>",
            )
        )
        T_all.extend(seg.T_K)
    # el contacto: un salto de temperatura en la interfaz
    for r in result.resistances:
        if r.kind == "contacto":
            fig.add_trace(
                go.Scatter(
                    x=[r.x_from_m * x_scale] * 2,
                    y=[T(r.T_from_K), T(r.T_to_K)],
                    mode="lines",
                    name="salto en el contacto",
                    line={"color": _VIOLET, "width": 3},
                    hovertemplate=f"ΔT del contacto<br>%{{y:.4g}} {t_unit}<extra></extra>",
                )
            )
    # los fluidos: una recta a T∞ y la caída en la película (punteada)
    pad = 0.18 * width
    shown = False
    for r in result.resistances:
        if r.kind != "convección":
            continue
        first = r is result.resistances[0]
        x_face = r.x_from_m
        T_fluid, T_wall = (r.T_from_K, r.T_to_K) if first else (r.T_to_K, r.T_from_K)
        x_far = x_face - pad if first else x_face + pad
        x_near = x_face - 0.35 * pad if first else x_face + 0.35 * pad
        fig.add_trace(
            go.Scatter(
                x=[x_far * x_scale, x_near * x_scale, x_face * x_scale],
                y=[T(T_fluid), T(T_fluid), T(T_wall)],
                mode="lines",
                name="fluido (T∞ y la película)",
                legendgroup="fluid",
                showlegend=not shown,
                line={"color": _ORANGE, "width": 2.5},
                hovertemplate=f"%{{x:.4g}} {x_unit}<br>T = %{{y:.4g}} {t_unit}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=x_far * x_scale,
            y=T(T_fluid),
            text="T∞,1" if first else "T∞,2",
            showarrow=False,
            yshift=10,
            xanchor="left" if first else "right",
            font={"size": 11, "color": _INK},
        )
        T_all.append(T_fluid)
        shown = True
    if i.inner.kind == "heat":
        fig.add_annotation(
            x=faces[0] * x_scale,
            y=T(result.T_side1_K),
            text="Q̇ entra",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-25,
            font={"size": 11, "color": _INK},
        )
    lo, hi = min(T_all), max(T_all)
    margin = 0.12 * (hi - lo or 1.0)
    x_name = "x" if i.geometry == "plane" else "r"
    _layout(
        fig,
        height=380,
        x_title=f"{x_name} [{x_unit}]",
        y_title=f"T [{t_unit}]",
        yaxis={"range": sorted([T(lo - margin), T(hi + margin)])},
    )
    return fig


def insulation_figure(
    points: Sequence[SweepPoint],
    result: ConductionResult,
    *,
    heat_given: bool,
    x_scale: float,
    x_unit: str,
    y_scale: float,
    y_offset: float,
    y_unit: str,
) -> go.Figure:
    """Q̇ (o T del lado 1 con el calor dado) contra el radio exterior; r_cr punteado."""
    ys = [(p.T_side1_K if heat_given else p.Q_W) * y_scale + y_offset for p in points]
    label = "T del lado 1" if heat_given else "Q̇"
    hover = f"r = %{{x:.4g}} {x_unit}<br>{label} = %{{y:.4g}} {y_unit}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[p.r_outer_m * x_scale for p in points],
            y=ys,
            mode="lines",
            name=label,
            line={"color": _BLUE, "width": 2.5},
            hovertemplate=hover,
        )
    )
    now = (result.T_side1_K if heat_given else result.Q_W) * y_scale + y_offset
    fig.add_trace(
        go.Scatter(
            x=[result.faces_m[-1] * x_scale],
            y=[now],
            mode="markers",
            name="los datos",
            marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=hover,
        )
    )
    r_cr = result.critical_radius_m
    if r_cr is not None:
        fig.add_vline(
            x=r_cr * x_scale,
            line={"color": _GRAY, "width": 1.5, "dash": "dot"},
            annotation_text="r_cr",
            annotation_position="top",
        )
    _layout(fig, height=340, x_title=f"radio exterior [{x_unit}]", y_title=f"{label} [{y_unit}]")
    return fig


# ---------------------------------------------------------------------
# Aletas
# ---------------------------------------------------------------------


def fin_profile_figure(
    result: FinResult,
    others: dict[TipCondition, FinResult],
    *,
    x_scale: float,
    x_unit: str,
    t_scale: float,
    t_offset: float,
    t_unit: str,
) -> go.Figure:
    """T(x) de la aleta, y en gris la misma aleta con las otras condiciones de la punta."""
    fig = go.Figure()
    T_inf = result.inputs.T_inf_K * t_scale + t_offset
    for tip, other in others.items():
        if tip == result.inputs.tip:
            continue
        xs, Ts = fin_profile(other)
        fig.add_trace(
            go.Scatter(
                x=[x * x_scale for x in xs],
                y=[t * t_scale + t_offset for t in Ts],
                mode="lines",
                name=_TIP_NAMES[tip],
                line={"color": _LIGHT_GRAY, "width": 1.5, "dash": "dot"},
                hovertemplate=f"{_TIP_NAMES[tip]}<br>x = %{{x:.4g}} {x_unit}"
                f"<br>T = %{{y:.4g}} {t_unit}<extra></extra>",
            )
        )
    xs, Ts = fin_profile(result)
    fig.add_trace(
        go.Scatter(
            x=[x * x_scale for x in xs],
            y=[t * t_scale + t_offset for t in Ts],
            mode="lines",
            name=_TIP_NAMES[result.inputs.tip],
            line={"color": _BLUE, "width": 3},
            hovertemplate=f"x = %{{x:.4g}} {x_unit}<br>T = %{{y:.4g}} {t_unit}<extra></extra>",
        )
    )
    fig.add_hline(
        y=T_inf,
        line={"color": _ORANGE, "width": 1.5, "dash": "dash"},
        annotation_text="T∞",
        annotation_position="bottom right",
    )
    x_name = "r − r₁" if result.inputs.shape == "annular" else "x"
    _layout(fig, height=360, x_title=f"{x_name} [{x_unit}]", y_title=f"T [{t_unit}]")
    return fig


def fin_efficiency_figure(result: FinResult) -> go.Figure:
    """η contra m·L_c para la forma de la aleta, con el punto de los datos."""
    xs, etas = efficiency_curve(result)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=etas,
            mode="lines",
            name="η(m·L_c)",
            line={"color": _BLUE, "width": 2.5},
            hovertemplate="m·L_c = %{x:.3g}<br>η = %{y:.3f}<extra></extra>",
        )
    )
    if result.efficiency is not None:
        fig.add_trace(
            go.Scatter(
                x=[result.mL],
                y=[result.efficiency],
                mode="markers",
                name="la aleta",
                marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
                hovertemplate="m·L_c = %{x:.3g}<br>η = %{y:.3f}<extra></extra>",
            )
        )
    _layout(fig, height=330, x_title="m·L_c", y_title="eficiencia η", yaxis={"range": [0, 1.05]})
    return fig


# ---------------------------------------------------------------------
# Convección
# ---------------------------------------------------------------------


def nu_curve_figure(
    var: str, xs: Sequence[float], nus: Sequence[float], x0: float, Nu0: float, name: str
) -> go.Figure:
    """La correlación usada como curva Nu(Re) o Nu(Ra), en escala logarítmica."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(xs),
            y=list(nus),
            mode="lines",
            name=name,
            line={"color": _BLUE, "width": 2.5},
            hovertemplate=f"{var} = %{{x:.4g}}<br>Nu = %{{y:.4g}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x0],
            y=[Nu0],
            mode="markers",
            name="los datos",
            marker={"size": 11, "color": _ORANGE, "line": {"color": _INK, "width": 1}},
            hovertemplate=f"{var} = %{{x:.4g}}<br>Nu = %{{y:.4g}}<extra></extra>",
        )
    )
    _layout(
        fig,
        height=340,
        x_title=var,
        y_title="Nu",
        xaxis={"type": "log", "exponentformat": "power"},
        yaxis={"type": "log"},
    )
    return fig


def correlation_figure(
    alternatives: Sequence[Alternative], main: str, *, h_scale: float, h_unit: str
) -> go.Figure:
    """El h de cada correlación: la usada en azul, las demás en gris; hueco si está fuera de
    su rango."""
    names = [a.name for a in alternatives]
    fig = go.Figure()
    for in_range, label in ((True, "en su rango"), (False, "fuera de su rango")):
        rows = [a for a in alternatives if a.in_range == in_range]
        if not rows:
            continue
        fig.add_trace(
            go.Scatter(
                x=[a.h_W_per_m2K * h_scale for a in rows],
                y=[a.name for a in rows],
                mode="markers+text",
                name=label,
                marker={
                    "size": 12,
                    "color": [_BLUE if a.key == main else _GRAY for a in rows]
                    if in_range
                    else "white",
                    "line": {
                        "color": [_BLUE if a.key == main else _GRAY for a in rows],
                        "width": 2,
                    },
                    "symbol": "circle" if in_range else "circle-open",
                },
                text=[f"{a.h_W_per_m2K * h_scale:.4g}".replace(".", ",") for a in rows],
                textposition="top center",
                textfont={"color": _INK, "size": 11},
                cliponaxis=False,
                hovertemplate=f"%{{y}}<br>h = %{{x:.4g}} {h_unit}<extra></extra>",
            )
        )
    hs = [a.h_W_per_m2K * h_scale for a in alternatives]
    lo, hi = min(hs), max(hs)
    pad = 0.15 * (hi - lo or hi)
    _layout(
        fig,
        height=110 + 48 * len(names),
        x_title=f"h [{h_unit}]",
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


def tube_figure(
    result: InternalFlowResult,
    *,
    x_scale: float,
    x_unit: str,
    t_scale: float,
    t_offset: float,
    t_unit: str,
) -> go.Figure:
    """T_m(x) del fluido y T_s(x) de la pared a lo largo del tubo (Incropera, §8.3)."""
    xs, Tm, Ts = tube_profile(result)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[x * x_scale for x in xs],
            y=[t * t_scale + t_offset for t in Ts],
            mode="lines",
            name="pared T_s",
            line={"color": _ORANGE, "width": 2.5, "dash": "dash"},
            hovertemplate=f"x = %{{x:.4g}} {x_unit}<br>T_s = %{{y:.4g}} {t_unit}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x * x_scale for x in xs],
            y=[t * t_scale + t_offset for t in Tm],
            mode="lines",
            name="fluido T_m",
            line={"color": _BLUE, "width": 3},
            hovertemplate=f"x = %{{x:.4g}} {x_unit}<br>T_m = %{{y:.4g}} {t_unit}<extra></extra>",
        )
    )
    all_T = [*Tm, *Ts]
    lo, hi = min(all_T), max(all_T)
    margin = 0.08 * (hi - lo or 1.0)
    bounds = sorted([(lo - margin) * t_scale + t_offset, (hi + margin) * t_scale + t_offset])
    _layout(
        fig,
        height=340,
        x_title=f"x [{x_unit}]",
        y_title=f"T [{t_unit}]",
        yaxis={"range": bounds},
    )
    return fig
