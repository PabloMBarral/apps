"""Gráficos de los gases ideales — Fase 9.1.

- :func:`cp_figure`: c_p(T) del polinomio NASA, con la recta del c_p constante
  (25 °C), el c_p a la temperatura media y el tramo del proceso sombreado.
- :func:`composition_figure`: las fracciones másicas y molares de una mezcla, en
  barras apiladas al 100 %.
- :func:`mixing_entropy_figure`: la entropía generada en una mezcla adiabática,
  por corriente: la parte de igualar T y p y la de mezclar gases distintos.
- :func:`pv_figure` y :func:`ts_figure`: los cinco caminos desde el mismo estado
  inicial hasta el mismo dato final, con el del dato resaltado.
- :func:`work_figure`: el trabajo y el calor de cada camino (gráfico de puntos).
- :func:`staged_figure`: la compresión en etapas en el p–v, contra una etapa y la
  isoterma.

Colores: la paleta de referencia validada para daltonismo, en su orden fijo
(azul, naranja, aguamarina, amarillo, magenta, verde): cada proceso tiene el
suyo y los componentes de una mezcla toman los colores en el orden en que se
cargan. Los tres claros no llegan a 3:1 contra el fondo: las barras llevan sus
porcentajes y la página trae las tablas con los mismos números.

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.gases`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from core.gases.ideal import IdealGas
from core.gases.mixture import MixingResult, MixtureResult
from core.gases.polytropic import (
    PROCESS_KINDS,
    ComparisonRow,
    ProcessKind,
    StagedResult,
    process_curve,
    staged_curves,
)
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "PROCESS_COLORS",
    "composition_figure",
    "cp_figure",
    "far_paths",
    "mixing_entropy_figure",
    "process_label",
    "pv_figure",
    "staged_figure",
    "ts_figure",
    "work_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_YELLOW = "#eda100"
_MAGENTA = "#e87ba4"
_GREEN = "#008300"
_GRAY = "#6b7280"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_BAND = "rgba(42, 120, 214, 0.10)"

#: El orden fijo de la paleta (nunca se recicla: las mezclas van hasta 6 gases).
_SLOTS = (_BLUE, _ORANGE, _AQUA, _YELLOW, _MAGENTA, _GREEN)
#: Texto dentro de una barra: blanco sobre los oscuros, tinta sobre los claros.
_TEXT_ON = {
    _BLUE: "white",
    _ORANGE: _INK,
    _AQUA: _INK,
    _YELLOW: _INK,
    _MAGENTA: _INK,
    _GREEN: "white",
}
#: Cada proceso con su color, siempre el mismo (también cuando falta alguno).
PROCESS_COLORS: dict[ProcessKind, str] = {
    "isochoric": _BLUE,
    "isobaric": _ORANGE,
    "isothermal": _AQUA,
    "adiabatic": _YELLOW,
    "polytropic": _MAGENTA,
}
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0}


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


def _comma(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",")


def _conv(values: Sequence[float], kind: QuantityKind, system: UnitSystem) -> list[float]:
    return [convert_from_si(float(v), kind, system) for v in values]


def _axis(name: str, kind: QuantityKind, system: UnitSystem) -> str:
    return f"{name} [{unit_label(kind, system)}]"


def far_paths(rows: Sequence[ComparisonRow]) -> set[ProcessKind]:
    """Los caminos que se van de escala en el T–s y en el trabajo.

    Uno cuya T₂ se aleja de T₁ más de 5 veces lo que se aleja la del dato (o
    50 K): la isócora hasta la p₂ de un compresor llega a 9·T₁ y aplastaría a
    los demás. En el T–s queda oculto (se ve tocándolo en la leyenda) y en el
    gráfico del trabajo no va; sus números están en la tabla.
    """
    data = next((r.result for r in rows if r.is_data and r.result is not None), None)
    if data is None:
        return set()
    T1 = data.state1.T_K
    ref = 5.0 * max(abs(data.state2.T_K - T1), 50.0)
    return {
        row.process
        for row in rows
        if row.result is not None and not row.is_data and abs(row.result.state2.T_K - T1) > ref
    }


def process_label(row: ComparisonRow) -> str:
    """El nombre corto del proceso («Politrópica n = 1,3»)."""
    name = PROCESS_KINDS[row.process].split(" (")[0]
    if name.startswith("Adiabática"):
        name = "Adiabática"
    if row.process == "polytropic" and row.result is not None:
        name += f" n = {_comma(row.result.n, '.3g')}"
    return name


# ---------------------------------------------------------------------
# c_p(T)
# ---------------------------------------------------------------------


def cp_figure(g: IdealGas, T1_K: float, T2_K: float, system: UnitSystem) -> go.Figure:
    """c_p(T) del gas de 200 K (o su mínimo) a 2000 K (o más, si el proceso sigue)."""
    lo = max(g.T_min_K, 200.0)
    hi = min(g.T_max_K, max(2000.0, 1.15 * max(T1_K, T2_K)))
    Ts = np.linspace(lo, hi, 160)
    T_kind: QuantityKind = "temperature"
    c_kind: QuantityKind = "specific_heat"
    x = _conv(Ts, T_kind, system)
    T_unit = unit_label(T_kind, system)
    c_unit = unit_label(c_kind, system)
    fig = go.Figure()
    a, b = sorted((convert_from_si(T1_K, T_kind, system), convert_from_si(T2_K, T_kind, system)))
    if b - a > 1e-9:
        fig.add_vrect(
            x0=a,
            x1=b,
            fillcolor=_BAND,
            line_width=0,
            layer="below",
            annotation_text="el proceso",
            annotation_position="top left",
            annotation_font={"color": _GRAY, "size": 12},
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=_conv([g.cp(float(T)) for T in Ts], c_kind, system),
            mode="lines",
            name="c_p(T) (polinomio NASA)",
            line={"color": _BLUE, "width": 2},
            hovertemplate=f"T = %{{x:.4g}} {T_unit}<br>c_p = %{{y:.4g}} {c_unit}<extra></extra>",
        )
    )
    cp_ref = convert_from_si(g.cp_ref, c_kind, system)
    fig.add_trace(
        go.Scatter(
            x=[x[0], x[-1]],
            y=[cp_ref, cp_ref],
            mode="lines",
            name="c_p constante (25 °C)",
            line={"color": _GRAY, "width": 2, "dash": "dash"},
            hovertemplate=f"c_p(25 °C) = {_comma(cp_ref)} {c_unit}<extra></extra>",
        )
    )
    T_m = 0.5 * (T1_K + T2_K)
    fig.add_trace(
        go.Scatter(
            x=[convert_from_si(T_m, T_kind, system)],
            y=[convert_from_si(g.cp(T_m), c_kind, system)],
            mode="markers",
            name="c_p a la T media",
            marker={"size": 10, "color": _ORANGE, "line": {"color": "white", "width": 2}},
            hovertemplate=f"T media = %{{x:.4g}} {T_unit}<br>c_p = %{{y:.4g}} {c_unit}"
            "<extra></extra>",
        )
    )
    _layout(
        fig,
        height=340,
        x_title=_axis("temperatura", T_kind, system),
        y_title=_axis("c_p", c_kind, system),
    )
    return fig


# ---------------------------------------------------------------------
# Mezclas
# ---------------------------------------------------------------------


def composition_figure(result: MixtureResult) -> go.Figure:
    """Fracciones másicas y molares de la mezcla, en barras apiladas al 100 %."""
    rows = ("másica", "molar")
    fig = go.Figure()
    for k, c in enumerate(result.components):
        color = _SLOTS[k % len(_SLOTS)]
        xs = [100.0 * c.x, 100.0 * c.y]
        name = c.gas.formula if c.gas.formula != "—" else c.gas.name
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=list(rows),
                x=xs,
                name=name,
                marker={"color": color, "line": {"color": "white", "width": 2}},
                text=[f"{v:.1f} %".replace(".", ",") if v >= 6.0 else "" for v in xs],
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                insidetextfont={"color": _TEXT_ON[color]},
                hovertemplate=f"{name}: %{{x:.4g}} %<extra>fracción %{{y}}</extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=220,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"range": [0, 100], "title": "fracción [%]", "gridcolor": _GRID},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "traceorder": "normal"},
        uniformtext={"minsize": 10, "mode": "hide"},
        separators=", ",
    )
    return fig


def mixing_entropy_figure(result: MixingResult, system: UnitSystem) -> go.Figure:
    """m·Δs de cada corriente: igualar T y p (puede ser negativa) y mezclarse; y el total."""
    tank = result.inputs.kind == "tank"
    kind: QuantityKind = "entropy" if tank else "entropy_flow"
    unit = unit_label(kind, system)
    labels = []
    for j, st in enumerate(result.streams, 1):
        name = st.gas.formula if st.gas.formula != "—" else st.gas.name
        labels.append(f"{j}: {name}")
    labels.append("Total")
    tp = [st.amount * st.ds_TP for st in result.streams] + [result.S_gen_TP]
    mix = [st.amount * st.ds_mix for st in result.streams] + [result.S_gen_mix]
    fig = go.Figure()
    for name, values, color in (
        ("igualar T y p", tp, _BLUE),
        ("mezclar gases distintos", mix, _ORANGE),
    ):
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=labels,
                x=_conv(values, kind, system),
                name=name,
                marker={"color": color, "line": {"color": "white", "width": 2}},
                hovertemplate=f"{name}: %{{x:.4g}} {unit}<extra>%{{y}}</extra>",
            )
        )
    fig.add_vline(x=0.0, line={"color": _GRAY, "width": 1})
    fig.update_layout(
        barmode="relative",
        height=120 + 46 * len(labels),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={
            "title": f"{'S' if tank else 'Ṡ'}_gen de cada corriente [{unit}]",
            "gridcolor": _GRID,
            "zerolinecolor": _GRID,
        },
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "traceorder": "normal"},
        separators=", ",
    )
    return fig


# ---------------------------------------------------------------------
# Los cinco caminos
# ---------------------------------------------------------------------


def _paths_figure(
    rows: Sequence[ComparisonRow],
    system: UnitSystem,
    x_key: str,
    y_key: str,
    x_kind: QuantityKind,
    y_kind: QuantityKind,
    titles: tuple[str, str],
    hidden: set[ProcessKind] | None = None,
) -> go.Figure:
    fig = go.Figure()
    x_unit, y_unit = unit_label(x_kind, system), unit_label(y_kind, system)
    data_points: tuple[list[float], list[float]] | None = None
    for row in rows:
        if row.result is None:
            continue
        curve = process_curve(row.result)
        xs = _conv(curve[x_key], x_kind, system)
        ys = _conv(curve[y_key], y_kind, system)
        name = process_label(row) + (" (el dato)" if row.is_data else "")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=name,
                line={"color": PROCESS_COLORS[row.process], "width": 4 if row.is_data else 2},
                hovertemplate=f"{name}<br>{titles[0]} = %{{x:.4g}} {x_unit}<br>"
                f"{titles[1]} = %{{y:.4g}} {y_unit}<extra></extra>",
                visible="legendonly" if hidden and row.process in hidden else True,
            )
        )
        if row.is_data:
            data_points = ([xs[0], xs[-1]], [ys[0], ys[-1]])
    if data_points is not None:
        fig.add_trace(
            go.Scatter(
                x=data_points[0],
                y=data_points[1],
                mode="markers+text",
                name="estados 1 y 2",
                text=["1", "2"],
                textposition="top right",
                textfont={"color": _INK, "size": 13},
                marker={"size": 9, "color": _INK, "line": {"color": "white", "width": 2}},
                hovertemplate=f"{titles[0]} = %{{x:.4g}} {x_unit}<br>"
                f"{titles[1]} = %{{y:.4g}} {y_unit}<extra>estado %{{text}}</extra>",
                showlegend=False,
                cliponaxis=False,  # el rótulo de un estado en el borde no se corta
            )
        )
    _layout(
        fig,
        height=400,
        x_title=_axis(titles[0], x_kind, system),
        y_title=_axis(titles[1], y_kind, system),
    )
    return fig


def pv_figure(rows: Sequence[ComparisonRow], system: UnitSystem) -> go.Figure:
    """Los caminos en el p–v (el área a la izquierda es el trabajo de circulación)."""
    return _paths_figure(rows, system, "v", "p", "specific_volume", "pressure", ("v", "p"))


def ts_figure(rows: Sequence[ComparisonRow], system: UnitSystem) -> go.Figure:
    """Los caminos en el T–s (el área debajo es el calor de un proceso reversible); los que se
    van de escala (:func:`far_paths`) quedan ocultos hasta tocarlos en la leyenda."""
    return _paths_figure(
        rows, system, "s", "T", "specific_entropy", "temperature", ("s", "T"), far_paths(rows)
    )


def work_figure(rows: Sequence[ComparisonRow], system: UnitSystem, *, closed: bool) -> go.Figure:
    """El trabajo (w en un sistema cerrado, w_f en uno abierto) y el calor de cada camino."""
    kind: QuantityKind = "specific_enthalpy"
    unit = unit_label(kind, system)
    far = far_paths(rows)
    ok = [row for row in rows if row.result is not None and row.process not in far]
    labels = [("▸ " if row.is_data else "") + process_label(row) for row in ok]
    work = [row.result.w if closed else row.result.w_f for row in ok]  # type: ignore[union-attr]
    heat = [row.result.q for row in ok]  # type: ignore[union-attr]
    w_name = "trabajo w" if closed else "trabajo de circulación w_f"
    fig = go.Figure()
    for name, values, color, symbol in (
        (w_name, work, _BLUE, "circle"),
        ("calor q", heat, _ORANGE, "diamond"),
    ):
        fig.add_trace(
            go.Scatter(
                x=_conv(values, kind, system),
                y=labels,
                mode="markers",
                name=name,
                marker={
                    "size": 12,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "white", "width": 2},
                },
                hovertemplate=f"{name}: %{{x:.4g}} {unit}<extra>%{{y}}</extra>",
            )
        )
    fig.add_vline(x=0.0, line={"color": _GRAY, "width": 1})
    _layout(
        fig,
        height=150 + 42 * len(ok),
        x_title=_axis("por kg", kind, system),
        y_title="",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_yaxes(autorange="reversed", gridcolor=_GRID)
    return fig


# ---------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------


def staged_figure(result: StagedResult, system: UnitSystem) -> go.Figure:
    """La compresión en etapas con interenfriamiento contra una etapa y la isoterma."""
    curves = staged_curves(result)
    v_kind: QuantityKind = "specific_volume"
    p_kind: QuantityKind = "pressure"
    v_unit, p_unit = unit_label(v_kind, system), unit_label(p_kind, system)
    N = result.inputs.stages
    fig = go.Figure()
    for key, name, color, dash in (
        ("single", "una etapa", _ORANGE, "solid"),
        ("isothermal", "isoterma a T₁ (infinitas etapas)", _GRAY, "dot"),
        ("stages", f"{N} etapas con interenfriamiento", _BLUE, "solid"),
    ):
        if key == "stages" and N == 1:
            continue
        c = curves[key]
        fig.add_trace(
            go.Scatter(
                x=_conv(c["v"], v_kind, system),
                y=_conv(c["p"], p_kind, system),
                mode="lines",
                name=name,
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=f"{name}<br>v = %{{x:.4g}} {v_unit}<br>p = %{{y:.4g}} {p_unit}"
                "<extra></extra>",
            )
        )
    _layout(
        fig,
        height=380,
        x_title=_axis("v", v_kind, system),
        y_title=_axis("p", p_kind, system),
    )
    return fig
