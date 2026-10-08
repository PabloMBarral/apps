"""Carta psicrométrica interactiva (plotly) — Fase 4.

La carta de Cengel (figura A-31) y ASHRAE a cualquier presión: temperatura de
bulbo seco en el eje horizontal y humedad absoluta ω a la derecha; curvas de
humedad relativa cada 10 % (la de 100 % es la saturación), rectas de entalpía
y de volumen específico constantes, líneas de bulbo húmedo (ocultas al
principio: se prenden desde la leyenda) y la zona de confort de Cengel §14-6.

Encima van los estados numerados y los procesos con flechas, con la paleta de
referencia validada para daltonismo: calentamiento naranja, enfriamiento
azul, humidificación aguamarina y mezcla gris. Las líneas de la carta van en
gris, para que los procesos se destaquen.

Debajo de todo hay una grilla de puntos invisibles (la primera traza): al
pasar el mouse muestra el estado completo de ese punto, y al tocarlo la página
lo puede cargar como dato (``on_select`` de ``st.plotly_chart``).

Vive en ``ui/`` porque arma figuras para la página; los cálculos (líneas,
grilla y estados) están en :mod:`core.psychrometrics`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from core.hvac import HvacResult
from core.psychrometrics import (
    COMFORT_PHI,
    COMFORT_T_K,
    R_AIR,
    ChartLine,
    ChartWindow,
    MoistAirState,
    chart_grid,
    chart_lines,
    dew_point,
    humidity_ratio,
    moist_enthalpy,
    saturation_humidity_ratio,
    saturation_pressure,
    vapor_pressure,
    wet_bulb_temperature,
)
from core.state_report import format_value
from core.units_system import UnitSystem, convert_from_si, convert_to_si, unit_label

__all__ = [
    "GRID_CURVE",
    "ArrowKind",
    "ChartArrow",
    "ChartPoint",
    "grid_point_from_selection",
    "hvac_chart_items",
    "hvac_exergy_figure",
    "psychrometric_chart_figure",
]

#: La grilla para tocar es la primera traza de la figura.
GRID_CURVE = 0

ArrowKind = Literal["heating", "cooling", "humidification", "mixing"]
_ARROW_COLORS: dict[ArrowKind, str] = {
    "heating": "#eb6834",
    "cooling": "#2a78d6",
    "humidification": "#1baf7a",
    "mixing": "#6b7280",
}
_ARROW_NAMES: dict[ArrowKind, str] = {
    "heating": "calentamiento",
    "cooling": "enfriamiento",
    "humidification": "humidificación",
    "mixing": "mezcla",
}
_INK = "#1f2937"
_LINE = "rgba(90, 90, 90, 0.55)"
_LINE_SOFT = "rgba(90, 90, 90, 0.35)"
_LABEL = "rgba(70, 70, 70, 0.9)"
_COMFORT = "rgba(120, 120, 120, 0.13)"
#: Curvas de φ con rótulo (en %).
_PHI_LABELED = frozenset({10, 20, 30, 40, 60, 80})


@dataclass(frozen=True)
class ChartPoint:
    """Un estado numerado sobre la carta."""

    label: str
    state: MoistAirState
    name: str = ""


@dataclass(frozen=True)
class ChartArrow:
    """Un proceso: flecha de un estado a otro."""

    start: MoistAirState
    end: MoistAirState
    kind: ArrowKind


def _x(T_K: float, system: UnitSystem) -> float:
    return convert_from_si(T_K, "temperature", system)


def _fmt(value: float | None, sig: int = 4) -> str:
    return format_value(value, sig)


def _nice_values(lo: float, hi: float, step: float) -> list[float]:
    start = math.ceil(lo / step) * step
    return [start + k * step for k in range(int(math.floor((hi - start) / step)) + 1)]


def _family_values(p_Pa: float, window: ChartWindow, system: UnitSystem) -> dict[str, list[float]]:
    """Valores redondos (en las unidades del sistema) de h, v y T_bh, pasados a SI."""
    h_lo = moist_enthalpy(window.T_min_K, 0.0)
    h_hi = moist_enthalpy(window.T_max_K, window.omega_max)
    h_step = {"SI": 10e3, "Técnico": 10.0, "Inglés": 5.0}[system]
    h_values = _nice_values(
        convert_from_si(h_lo, "specific_enthalpy", system),
        convert_from_si(h_hi, "specific_enthalpy", system),
        h_step,
    )
    v_lo = R_AIR * window.T_min_K / p_Pa
    p_v = vapor_pressure(window.omega_max, p_Pa)
    v_hi = R_AIR * window.T_max_K / (p_Pa - p_v)
    v_lo_u = convert_from_si(v_lo, "specific_volume", system)
    v_hi_u = convert_from_si(v_hi, "specific_volume", system)
    v_step = 10 ** math.floor(math.log10((v_hi_u - v_lo_u) / 4))
    if (v_hi_u - v_lo_u) / v_step > 12:
        v_step *= 5 if (v_hi_u - v_lo_u) / v_step > 30 else 2
    v_values = _nice_values(v_lo_u, v_hi_u, v_step)
    T_step = {"SI": 5.0, "Técnico": 5.0, "Inglés": 10.0}[system]
    T_values = _nice_values(
        convert_from_si(window.T_min_K, "temperature", system),
        convert_from_si(window.T_max_K, "temperature", system),
        T_step,
    )
    return {
        "h": [convert_to_si(h, "specific_enthalpy", system) for h in h_values],
        "v": [convert_to_si(v, "specific_volume", system) for v in v_values],
        "T_wb": [convert_to_si(T, "temperature", system) for T in T_values],
    }


def _joined(lines: Sequence[ChartLine], system: UnitSystem) -> tuple[list[Any], list[Any]]:
    """Varias líneas en una sola traza, separadas con ``None`` (una entrada de leyenda)."""
    xs: list[Any] = []
    ys: list[Any] = []
    for line in lines:
        xs += [_x(T, system) for T in line.T_K] + [None]
        ys += list(line.omega) + [None]
    return xs, ys


def _inside(line: ChartLine, window: ChartWindow) -> list[tuple[float, float]]:
    return [
        (T, w)
        for T, w in zip(line.T_K, line.omega, strict=True)
        if window.T_min_K - 1e-9 <= T <= window.T_max_K + 1e-9 and 0.0 <= w <= window.omega_max
    ]


def _phi_label_spot(
    inside: list[tuple[float, float]], window: ChartWindow
) -> tuple[float, float] | None:
    """Dónde rotular una curva de φ: donde cruza una diagonal (como la carta de Cengel).

    La diagonal va de arriba al centro (u = 0,45; 0,95·ω_máx) a abajo a la derecha
    (u = 0,97; 0,2·ω_máx), con u la fracción del rango de T: así los rótulos quedan
    repartidos y no se amontonan arriba.
    """
    span = window.T_max_K - window.T_min_K
    for T, w in inside:
        u = (T - window.T_min_K) / span
        if not 0.45 <= u <= 0.97:
            continue
        v_diag = 0.95 + (0.2 - 0.95) * (u - 0.45) / (0.97 - 0.45)
        if w / window.omega_max >= v_diag:
            return T, w
    tail = [(T, w) for T, w in inside if (T - window.T_min_K) / span <= 0.97]
    return tail[-1] if tail else None


def _grid_trace(p_Pa: float, window: ChartWindow, system: UnitSystem) -> go.Scatter:
    """Puntos invisibles: hover con el estado completo y selección al tocar."""
    points = chart_grid(p_Pa, window)
    eh = unit_label("specific_enthalpy", system)
    ev = unit_label("specific_volume", system)
    et = unit_label("temperature", system)
    custom = []
    for T, w in points:
        p_v = vapor_pressure(w, p_Pa)
        T_dp = dew_point(w, p_Pa)
        custom.append(
            [
                _fmt(_x(T, system)),
                _fmt(100 * p_v / saturation_pressure(T), 3),
                _fmt(w, 4),
                _fmt(convert_from_si(moist_enthalpy(T, w), "specific_enthalpy", system)),
                _fmt(_x(wet_bulb_temperature(T, w, p_Pa), system)),
                "—" if T_dp is None else _fmt(_x(T_dp, system)),
                _fmt(convert_from_si(R_AIR * T / (p_Pa - p_v), "specific_volume", system)),
            ]
        )
    return go.Scatter(
        x=[_x(T, system) for T, _ in points],
        y=[w for _, w in points],
        mode="markers",
        marker={"size": 12, "opacity": 0.0, "color": _INK},
        customdata=custom,
        hovertemplate=(
            f"T = %{{customdata[0]}} {et}<br>φ = %{{customdata[1]}} %<br>"
            f"ω = %{{customdata[2]}}<br>h = %{{customdata[3]}} {eh}<br>"
            f"T_bh = %{{customdata[4]}} {et}<br>T_pr = %{{customdata[5]}} {et}<br>"
            f"v = %{{customdata[6]}} {ev}<extra>tocá para cargarlo</extra>"
        ),
        name="grilla",
        showlegend=False,
    )


def _comfort_trace(p_Pa: float, system: UnitSystem) -> go.Scatter:
    T_lo, T_hi = COMFORT_T_K
    phi_lo, phi_hi = COMFORT_PHI
    temps = np.linspace(T_lo, T_hi, 12)
    bottom = [humidity_ratio(phi_lo * saturation_pressure(float(T)), p_Pa) for T in temps]
    top = [humidity_ratio(phi_hi * saturation_pressure(float(T)), p_Pa) for T in temps[::-1]]
    xs = [_x(float(T), system) for T in temps] + [_x(float(T), system) for T in temps[::-1]]
    return go.Scatter(
        x=[*xs, xs[0]],
        y=[*bottom, *top, bottom[0]],
        mode="lines",
        fill="toself",
        fillcolor=_COMFORT,
        line={"width": 1, "color": "rgba(90, 90, 90, 0.5)", "dash": "dot"},
        name="confort",
        hoverinfo="skip",
    )


def _state_hover(state: MoistAirState, system: UnitSystem, title: str) -> str:
    et = unit_label("temperature", system)
    eh = unit_label("specific_enthalpy", system)
    return (
        f"<b>{title}</b><br>T = {_fmt(_x(state.T_K, system))} {et}<br>"
        f"φ = {_fmt(100 * state.phi, 3)} %<br>ω = {_fmt(state.omega, 4)}<br>"
        f"h = {_fmt(convert_from_si(state.h_J_per_kg, 'specific_enthalpy', system))} {eh}<br>"
        f"T_bh = {_fmt(_x(state.T_wb_K, system))} {et}"
    )


def psychrometric_chart_figure(
    p_Pa: float,
    system: UnitSystem,
    window: ChartWindow,
    *,
    points: Sequence[ChartPoint] = (),
    arrows: Sequence[ChartArrow] = (),
    grid: bool = True,
    comfort: bool = True,
    height: int = 560,
) -> go.Figure:
    """Carta psicrométrica a la presión ``p_Pa`` con estados y procesos.

    La primera traza es la grilla para tocar (:data:`GRID_CURVE`).
    """
    fig = go.Figure()
    if grid:
        fig.add_trace(_grid_trace(p_Pa, window, system))
    values = _family_values(p_Pa, window, system)
    lines = chart_lines(
        p_Pa,
        window,
        h_values=tuple(values["h"]),
        v_values=tuple(values["v"]),
        T_wb_values=tuple(values["T_wb"]),
    )
    phi_lines = [line for line in lines if line.family == "phi" and line.value < 1.0]
    saturation = next(line for line in lines if line.family == "phi" and line.value == 1.0)
    families: list[tuple[str, list[ChartLine], dict[str, Any], str, bool | str]] = [
        ("φ cada 10 %", phi_lines, {"color": _LINE, "width": 1}, "phi", True),
        (
            f"h [{unit_label('specific_enthalpy', system)}]",
            [line for line in lines if line.family == "h"],
            {"color": _LINE_SOFT, "width": 1, "dash": "dash"},
            "h",
            True,
        ),
        (
            f"v [{unit_label('specific_volume', system)}]",
            [line for line in lines if line.family == "v"],
            {"color": _LINE_SOFT, "width": 1, "dash": "dot"},
            "v",
            True,
        ),
        (
            "T_bh",
            [line for line in lines if line.family == "T_wb"],
            {"color": _LINE_SOFT, "width": 1, "dash": "dashdot"},
            "T_wb",
            "legendonly",
        ),
    ]
    if comfort:
        fig.add_trace(_comfort_trace(p_Pa, system))
    for name, group, style, family, visible in families:
        if not group:
            continue
        xs, ys = _joined(group, system)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=style,
                name=name,
                legendgroup=family,
                hoverinfo="skip",
                visible=visible,
            )
        )
        label_x, label_y, label_text, positions = [], [], [], []
        for line in group:
            inside = _inside(line, window)
            if len(inside) < 2:
                continue
            if family == "phi":
                if round(100 * line.value) not in _PHI_LABELED:
                    continue  # las curvas altas van muy juntas: en un celular se pisarían
                spot = _phi_label_spot(inside, window)
                if spot is None:
                    continue
                T, w = spot
                text, pos = f"{100 * line.value:.0f} %", "middle left"
            elif family == "h":
                sat = saturation_humidity_ratio(line.T_K[0], p_Pa)
                if sat is None or abs(line.omega[0] - sat) > 1e-6 * max(sat, 1e-3):
                    continue  # la recta no nace en la saturación dentro de la carta
                T, w = inside[0]
                if w > 0.85 * window.omega_max:
                    continue  # arriba se amontonaría con los rótulos de φ
                text = _fmt(convert_from_si(line.value, "specific_enthalpy", system), 3)
                pos = "top left"
            elif family == "v":
                T, w = inside[-1]
                text = _fmt(convert_from_si(line.value, "specific_volume", system), 3)
                pos = "top center" if w < 0.02 * window.omega_max else "middle left"
            else:
                T, w = inside[0]
                text, pos = _fmt(_x(line.value, system), 3), "top left"
            label_x.append(_x(T, system))
            label_y.append(w)
            label_text.append(text)
            positions.append(pos)
        if label_text:
            fig.add_trace(
                go.Scatter(
                    x=label_x,
                    y=label_y,
                    text=label_text,
                    mode="text",
                    textposition=positions,
                    textfont={"size": 10, "color": _LABEL},
                    legendgroup=family,
                    showlegend=False,
                    hoverinfo="skip",
                    visible=visible,
                )
            )
    xs, ys = _joined([saturation], system)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": _INK, "width": 2.5},
            name="saturación",
            hoverinfo="skip",
        )
    )
    for kind in dict.fromkeys(arrow.kind for arrow in arrows):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": _ARROW_COLORS[kind], "width": 3},
                name=_ARROW_NAMES[kind],
            )
        )
    for arrow in arrows:
        fig.add_annotation(
            x=_x(arrow.end.T_K, system),
            y=arrow.end.omega,
            ax=_x(arrow.start.T_K, system),
            ay=arrow.start.omega,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2.5,
            arrowcolor=_ARROW_COLORS[arrow.kind],
            standoff=7,
            startstandoff=7,
            text="",
        )
    if points:
        fig.add_trace(
            go.Scatter(
                x=[_x(pt.state.T_K, system) for pt in points],
                y=[pt.state.omega for pt in points],
                mode="markers+text",
                text=[pt.label for pt in points],
                textposition="top left",
                textfont={"size": 14, "color": _INK},
                marker={"size": 11, "color": _INK, "line": {"color": "white", "width": 2}},
                name="estados",
                showlegend=False,
                hovertext=[
                    _state_hover(pt.state, system, pt.name or f"Estado {pt.label}") for pt in points
                ],
                hoverinfo="text",
            )
        )
    t_unit = unit_label("temperature", system)
    w_unit = "lb/lb a.s." if system == "Inglés" else "kg/kg a.s."
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={
            "title": f"T de bulbo seco [{t_unit}]",
            "range": [_x(window.T_min_K, system), _x(window.T_max_K, system)],
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": f"ω [{w_unit}]",
            "side": "right",
            "range": [0, window.omega_max],
            "showgrid": False,
            "zeroline": False,
            "tickformat": ".3f" if window.omega_max < 0.2 else ".2f",
        },
        legend={"orientation": "h", "yanchor": "top", "y": -0.14, "x": 0, "font": {"size": 11}},
        hovermode="closest",
        clickmode="event+select",
        dragmode=False,
        separators=". ",
    )
    return fig


def grid_point_from_selection(
    selection: Mapping[str, Any] | None, system: UnitSystem
) -> tuple[float, float] | None:
    """(T en K, ω) del punto de la grilla que se tocó, o ``None``.

    ``selection`` es el estado que guarda ``st.plotly_chart`` con ``on_select``
    (``{"selection": {"points": [...]}}``); solo cuentan los puntos de la grilla.
    """
    if not selection:
        return None
    points = (selection.get("selection") or {}).get("points") or []
    for point in points:
        if point.get("curve_number") != GRID_CURVE:
            continue
        x, y = point.get("x"), point.get("y")
        if x is None or y is None:
            continue
        return convert_to_si(float(x), "temperature", system), float(y)
    return None


def hvac_chart_items(result: HvacResult) -> tuple[list[ChartPoint], list[ChartArrow]]:
    """Estados y flechas de un tren de procesos."""
    points = [ChartPoint(str(s.number), s.state, s.label) for s in result.streams]
    arrows: list[ChartArrow] = []
    for proc in result.processes:
        s1, s2 = proc.inlet.state, proc.outlet.state
        if proc.kind == "mixing":
            assert proc.other is not None
            arrows.append(ChartArrow(s1, s2, "mixing"))
            arrows.append(ChartArrow(proc.other.state, s2, "mixing"))
        elif proc.kind in ("heating", "heating_humidification") and s2.T_K >= s1.T_K:
            arrows.append(ChartArrow(s1, s2, "heating"))
        elif proc.kind in ("cooling", "cooling_dehumidification"):
            arrows.append(ChartArrow(s1, s2, "cooling"))
        else:
            arrows.append(ChartArrow(s1, s2, "humidification"))
    return points, arrows


def hvac_exergy_figure(result: HvacResult, system: UnitSystem) -> go.Figure:
    """Barras: la exergía destruida en cada proceso (con su valor)."""
    unit = unit_label("power", system)
    names = [f"{p.index}. {p.name.lower()}" for p in result.processes]
    values = [convert_from_si(p.X_destroyed_W, "power", system) for p in result.processes]
    fig = go.Figure(
        go.Bar(
            orientation="h",
            y=names,
            x=values,
            marker={"color": "#eb6834", "cornerradius": 4},
            text=[f"{_fmt(v, 3)} {unit}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}: %{{x:.4g}} {unit}<extra></extra>",
            name="destruida",
        )
    )
    top = max([*values, 1e-12])
    fig.update_layout(
        height=90 + 34 * len(names),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"title": f"Exergía destruida [{unit}]", "range": [0, 1.35 * top]},
        yaxis={"categoryorder": "array", "categoryarray": names, "autorange": "reversed"},
        showlegend=False,
        bargap=0.35,
        separators=". ",
    )
    return fig
