"""Gráficos del segundo principio — Fase 10.2.

- :func:`limit_figure`: el rendimiento (o el COP) de la máquina reversible en función de la
  temperatura de una fuente, con la zona imposible arriba de la curva y la máquina como un
  punto: si cae arriba, viola el teorema de Carnot.
- :func:`heat_exergy_figure`: la exergía de un calor a cada temperatura por la que pasa
  (vademecum §11.5): baja escalón por escalón y lo que pierde es T₀·S_gen.

Los balances en cascada y los diagramas p–v y T–s del ciclo de Carnot salen de
:mod:`ui.balance_charts`. Colores: la paleta de referencia validada para daltonismo (la
máquina en azul, la curva de Carnot en tinta, la zona imposible en gris claro).
"""

from __future__ import annotations

import plotly.graph_objects as go

from core.balances.second_law import (
    PERFORMANCE_NAMES,
    HeatPathResult,
    MachineResult,
    carnot_performance,
)
from core.units_system import UnitSystem, convert_from_si, unit_label
from ui.balance_charts import value_axis

__all__ = ["heat_exergy_figure", "limit_figure"]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_ZONE = "rgba(156, 163, 175, 0.22)"
_INK = "#1f2937"
_GRID = "#e5e7eb"
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0}
_POINTS = 120


def _comma(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _temps(r: MachineResult) -> tuple[list[float], float, str]:
    """Las temperaturas (K) del eje x, la de la máquina y qué fuente se mueve."""
    inp = r.inputs
    T_H, T_C = inp.T_H_K, inp.T_C_K
    if r.kind == "engine":
        hi = max(2.0 * T_H - T_C, 1.5 * T_H)
        lo = min(T_C * 1.01, T_C + 0.5 * (T_H - T_C))  # T_H apenas sobre T_C: adentro igual
        return [lo + (hi - lo) * k / (_POINTS - 1) for k in range(_POINTS)], T_H, "T_H"
    # refrigerador y bomba de calor: se mueve la fuente fría, hasta un COP alto
    target = 2.5 * max(r.performance, r.performance_rev)
    if r.kind == "refrigerator":
        hi = T_H * target / (1.0 + target)  # COP_R,rev = T_C/(T_H − T_C) = target
    else:
        hi = T_H * (1.0 - 1.0 / target)  # COP_B,rev = T_H/(T_H − T_C) = target
    hi = max(hi, T_C + 0.6 * (T_H - T_C))
    lo = max(T_C - 2.0 * (T_H - T_C), 0.3 * T_H)
    lo = min(lo, max(T_C - 0.1 * (hi - T_C), 0.5 * T_C))  # T_C bajo 0,3·T_H: adentro igual
    return [lo + (hi - lo) * k / (_POINTS - 1) for k in range(_POINTS)], T_C, "T_C"


def limit_figure(r: MachineResult, system: UnitSystem) -> go.Figure:
    """η o COP de la máquina reversible contra la temperatura de una fuente, con la zona
    imposible arriba y la máquina como un punto (vademecum §9.2 y §9.3)."""
    inp = r.inputs
    temps, T_point, moving = _temps(r)
    if r.kind == "engine":
        ys = [carnot_performance("engine", T, inp.T_C_K) for T in temps]
    else:
        ys = [carnot_performance(r.kind, inp.T_H_K, T) for T in temps]
    xs = [convert_from_si(T, "temperature", system) for T in temps]
    y_top = max(max(ys), r.performance, r.performance_rev) * 1.15
    if r.kind == "engine":
        y_top = min(max(y_top, 0.2), 1.05)
    ys = [min(y, y_top) for y in ys]
    perf = PERFORMANCE_NAMES[r.kind]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": _INK, "width": 2},
            name="Máquina reversible (Carnot)",
            hovertemplate=f"{moving} = %{{x:.4g}}<br>{perf},rev = %{{y:.4g}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[y_top] * len(xs),
            mode="lines",
            line={"width": 0, "color": _ZONE},
            fill="tonexty",
            fillcolor=_ZONE,
            name="Imposible",
            hoverinfo="skip",
        )
    )
    x_point = convert_from_si(T_point, "temperature", system)
    impossible = r.verdict == "imposible"
    # el rótulo, lejos de la curva (abajo si es posible, arriba si no) y hacia el lado con
    # lugar: a 390 px «esta máquina» se cortaba en el borde derecho
    vertical = "top" if impossible else "bottom"
    if not impossible and r.performance < 0.15 * y_top:
        vertical = "top"
    elif impossible and r.performance > 0.8 * y_top:
        vertical = "bottom"
    right_half = x_point - xs[0] > 0.5 * (xs[-1] - xs[0])
    position = f"{vertical} {'left' if right_half else 'right'}"
    fig.add_trace(
        go.Scatter(
            x=[x_point],
            y=[r.performance],
            mode="markers+text",
            marker={
                "size": 12,
                "color": _ORANGE if impossible else _BLUE,
                "line": {"color": "white", "width": 2},
            },
            text=["esta máquina"],
            textposition=position,
            textfont={"color": _INK, "size": 13},
            name="Esta máquina",
            cliponaxis=False,
            hovertemplate=f"{perf} = %{{y:.4g}}<extra></extra>",
        )
    )
    unit = unit_label("temperature", system)
    what = {
        "engine": "temperatura de la fuente caliente T_H",
        "refrigerator": "temperatura del espacio frío T_C",
        "heat_pump": "temperatura de la fuente fría T_C",
    }[r.kind]
    title = {
        "engine": "El rendimiento máximo depende de T_H",
        "refrigerator": "El COP máximo depende de T_C",
        "heat_pump": "El COP máximo depende de T_C",
    }[r.kind]
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": title, "x": 0},
        xaxis={"title": f"{what} [{unit}]", "gridcolor": _GRID},
        yaxis={"title": perf, "gridcolor": _GRID, "range": [0, y_top], "zerolinecolor": _GRID},
        legend=_LEGEND_BELOW,
        separators=", ",
        hoverlabel={"namelength": -1},
    )
    return fig


def _wrap(text: str, width: int = 11) -> str:
    """Parte un rótulo en renglones de ~width caracteres (sin cortar palabras)."""
    lines: list[str] = []
    for word in text.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return "<br>".join(lines)


def heat_exergy_figure(r: HeatPathResult, system: UnitSystem) -> go.Figure:
    """La exergía del calor en cada punto del camino: baja a medida que baja T."""
    inp = r.inputs
    kind = "heat_rate" if inp.rates else "energy"
    values = [convert_from_si(x, kind, system) for x in r.exergies]
    t_unit = unit_label("temperature", system)
    labels = [
        f"{_wrap(name)}<br>{_comma(convert_from_si(T, 'temperature', system), '.4g')} {t_unit}"
        for name, T in zip(inp.points, inp.temperatures, strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": _BLUE},
            text=[_comma(v) for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y:.4g}<extra></extra>",
            name="Exergía del calor",
        )
    )
    fig.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": "La exergía del calor a cada temperatura", "x": 0},
        yaxis={
            "title": unit_label(kind, system),
            "gridcolor": _GRID,
            "zerolinecolor": _INK,
            **value_axis(values),
        },
        xaxis={"type": "category", "tickangle": 0},
        showlegend=False,
        separators=", ",
        bargap=0.45,
    )
    return fig
