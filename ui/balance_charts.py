"""Gráficos de los balances de energía y de entropía — Fase 10.1.

- :func:`state_diagram`: el p–v (log–log) o el T–s del proceso, con la campana del
  fluido real, los caminos (rayados si no son cuasiestáticos) y los estados rotulados.
- :func:`waterfall_figure`: un balance como cascada (Q − W = ΔU, ΔS = Q/T_b + S_gen):
  cada término sube o baja hasta el total.
- :func:`evolution_figure`: el tanque que se llena o se vacía, p(m) y T(m) en dos paneles
  con el eje de la masa compartido (nunca un eje doble).
- :func:`exchanger_tq_figure`: la temperatura de cada corriente contra el calor que
  intercambia (a contracorriente).

Colores: la paleta de referencia validada para daltonismo (azul, naranja, aguamarina…);
la campana en gris, lo que entra en azul y lo que sale en naranja, los totales en gris.
Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en :mod:`core.balances`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.balances.common import upper_first
from core.balances.steady_flow import ExchangerResult
from core.balances.substance import Substance, ThermoState, saturation_dome, state
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "PATH_COLORS",
    "evolution_figure",
    "exchanger_tq_figure",
    "state_diagram",
    "waterfall_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_MAGENTA = "#e87ba4"
_GRAY = "#6b7280"
_LIGHT = "#9ca3af"
_INK = "#1f2937"
_GRID = "#e5e7eb"

#: Los caminos de un diagrama, en orden fijo: el real, el isoentrópico y uno más.
PATH_COLORS = (_BLUE, _AQUA, _MAGENTA)
_LEGEND_BELOW = {"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0}

Path = tuple[str, Sequence[ThermoState], bool]  # (nombre, estados, cuasiestático)


def _comma(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


def _power(mantissa: float, exponent: int) -> str:
    power = f"10<sup>{exponent}</sup>".replace("-", "−")
    return power if mantissa == 1.0 else f"{_comma(mantissa, 'g')}·{power}"


def _bar_label(x: float) -> str:
    """El valor de una barra con cuatro cifras y sin notación exponencial (156 300 J,
    −0,01365): con ``.4g`` lo que pasa de 10⁴ salía «1,563e+05». Espacio fino desde 10⁴,
    como en el procedimiento."""
    if x == 0.0 or not math.isfinite(x):
        return "0"
    if not 1e-4 <= abs(x) < 1e9:
        mantissa, exponent = f"{x:.3e}".split("e")
        return _power(float(mantissa), int(exponent))
    decimals = 3 - math.floor(math.log10(abs(x)))
    x = round(x, decimals)
    text = f"{x:,.{max(decimals, 0)}f}" if abs(x) >= 1e4 else f"{x:.{max(decimals, 0)}f}"
    return text.replace(",", " ").replace(".", ",").replace("-", "−")


def _log_ticks(values: Sequence[float]) -> dict[str, object]:
    """Marcas con el número completo en un eje logarítmico. Con más de una década plotly
    rotula las intermedias con un «2» o un «5» suelto (0,2 y 20 quedan iguales): hasta dos
    décadas y media van 1-2-5 y, con más, las potencias de 10. Con menos de una década,
    las de plotly (ya llevan el número completo)."""
    pos = [v for v in values if v > 0.0 and math.isfinite(v)]
    if not pos or math.log10(max(pos) / min(pos)) <= 1.0:
        return {}
    lo, hi = math.floor(math.log10(min(pos))), math.ceil(math.log10(max(pos)))
    factors = (1.0, 2.0, 5.0) if math.log10(max(pos) / min(pos)) <= 2.5 else (1.0,)
    vals, texts = [], []
    for exponent in range(lo - 1, hi + 2):
        for factor in factors:
            v = factor * 10.0**exponent
            vals.append(v)
            texts.append(_comma(v, "g") if 1e-3 <= v < 1e4 else _power(factor, exponent))
    return {"tickvals": vals, "ticktext": texts}


def _conv(values: Sequence[float], kind: QuantityKind, system: UnitSystem) -> list[float]:
    return [convert_from_si(float(v), kind, system) for v in values]


def _axis(name: str, kind: QuantityKind, system: UnitSystem) -> str:
    return f"{name} [{unit_label(kind, system)}]"


def _xy(
    states: Sequence[ThermoState], kind: Literal["pv", "Ts"], system: UnitSystem
) -> tuple[list[float], list[float]]:
    if kind == "pv":
        return (
            _conv([s.v for s in states], "specific_volume", system),
            _conv([s.p for s in states], "pressure", system),
        )
    return (
        _conv([s.s for s in states], "specific_entropy", system),
        _conv([s.T for s in states], "temperature", system),
    )


def state_diagram(
    kind: Literal["pv", "Ts"],
    sub: Substance,
    points: Sequence[tuple[str, ThermoState]],
    paths: Sequence[Path],
    system: UnitSystem,
) -> go.Figure:
    """El p–v (log–log) o el T–s con la campana (fluido real), los caminos y los estados.

    Un camino que no es cuasiestático (estado final dado, contra p_ext, una válvula) va
    rayado: es solo la recta que une los extremos. El rótulo de un estado isoentrópico (2s)
    va a la izquierda: queda cerca del real y los dos a la derecha se pisaban.
    """
    fig = go.Figure()
    xs: list[float] = []
    ys: list[float] = []
    dome = saturation_dome(sub)
    if dome:
        x, y = _xy(dome, kind, system)
        xs += x
        ys += y
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Campana",
                line={"color": _LIGHT, "width": 1.5},
                hoverinfo="skip",
            )
        )
    for k, (name, states, quasi) in enumerate(paths):
        if len(states) < 2:
            continue
        x, y = _xy(states, kind, system)
        xs += x
        ys += y
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line={
                    "color": PATH_COLORS[k % len(PATH_COLORS)],
                    "width": 2.5,
                    "dash": "solid" if quasi else "dash",
                },
            )
        )
    if points:
        x, y = _xy([s for _, s in points], kind, system)
        xs += x
        ys += y
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                text=[label for label, _ in points],
                textposition=[
                    "middle left" if label.endswith("s") else "top right" for label, _ in points
                ],
                textfont={"size": 14, "color": _INK},
                marker={"size": 9, "color": _INK},
                name="Estados",
                showlegend=False,
                cliponaxis=False,
            )
        )
    if kind == "pv":
        x_title = _axis("v", "specific_volume", system)
        y_title = _axis("p", "pressure", system)
    else:
        x_title = _axis("s", "specific_entropy", system)
        y_title = _axis("T", "temperature", system)
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": x_title, "gridcolor": _GRID, "zerolinecolor": _GRID},
        yaxis={"title": y_title, "gridcolor": _GRID, "zerolinecolor": _GRID},
        legend=_LEGEND_BELOW,
        separators=", ",
        hoverlabel={"namelength": -1},
        title={"text": "Diagrama p–v" if kind == "pv" else "Diagrama T–s", "x": 0},
    )
    if kind == "pv":
        fig.update_xaxes(type="log", **_log_ticks(xs))
        fig.update_yaxes(type="log", **_log_ticks(ys))
    return fig


def waterfall_figure(
    title: str,
    items: Sequence[tuple[str, float]],
    total_label: str,
    kind: QuantityKind,
    system: UnitSystem,
) -> go.Figure:
    """Un balance en cascada: cada término (+ entra, − sale) hasta el total.

    Lo que es ruido de redondeo frente a los términos pasa a 0: el Q de un llenado
    adiabático sumaba −9,9·10⁻¹⁰ J.
    """
    labels = [label for label, _ in items] + [total_label]
    values = _conv([v for _, v in items], kind, system)
    noise = 1e-9 * sum(abs(v) for v in values)
    values = [0.0 if abs(v) <= noise else v for v in values]
    total = sum(values)
    if abs(total) <= noise:
        total = 0.0
    unit = unit_label(kind, system)
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            x=labels,
            y=[*values, total],
            measure=["relative"] * len(values) + ["total"],
            text=[_bar_label(v) for v in [*values, total]],
            textposition="outside",
            increasing={"marker": {"color": _BLUE}},
            decreasing={"marker": {"color": _ORANGE}},
            totals={"marker": {"color": _GRAY}},
            connector={"line": {"color": _LIGHT, "width": 1}},
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title={"text": title, "x": 0},
        yaxis={"title": unit, "gridcolor": _GRID, "zerolinecolor": _INK},
        xaxis={"type": "category"},
        showlegend=False,
        separators=", ",
    )
    return fig


def evolution_figure(
    path: Sequence[tuple[float, ThermoState]],
    system: UnitSystem,
    title: str,
) -> go.Figure:
    """p(m) y T(m) del tanque en dos paneles con la masa compartida."""
    masses = _conv([m for m, _ in path], "mass", system)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(
            x=masses,
            y=_conv([s.p for _, s in path], "pressure", system),
            mode="lines",
            line={"color": _BLUE, "width": 2.5},
            name="p",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=masses,
            y=_conv([s.T for _, s in path], "temperature", system),
            mode="lines",
            line={"color": _ORANGE, "width": 2.5},
            name="T",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=_axis("p", "pressure", system), gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text=_axis("T", "temperature", system), gridcolor=_GRID, row=2, col=1)
    fig.update_xaxes(gridcolor=_GRID)
    fig.update_xaxes(title_text=_axis("masa en el tanque m", "mass", system), row=2, col=1)
    fig.update_layout(
        height=440,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        showlegend=False,
        separators=", ",
        title={"text": title, "x": 0},
    )
    return fig


def _stream_curve(
    sub: Substance, inlet: ThermoState, outlet: ThermoState, m_dot: float, points: int = 41
) -> tuple[list[float], list[float]]:
    """(Q acumulado desde el extremo frío de la corriente, T) a lo largo de h."""
    lo, hi = (inlet, outlet) if inlet.h <= outlet.h else (outlet, inlet)
    out_q, out_T = [], []
    for k in range(points):
        h = lo.h + (hi.h - lo.h) * k / (points - 1)
        p = lo.p + (hi.p - lo.p) * k / (points - 1)
        try:
            st = state(sub, "PH", p, h)
        except ValueError:
            continue
        out_q.append(m_dot * (h - lo.h))
        out_T.append(st.T)
    return out_q, out_T


def exchanger_tq_figure(r: ExchangerResult, system: UnitSystem) -> go.Figure:
    """T contra el calor intercambiado, a contracorriente: las dos corrientes empiezan en el
    extremo frío del intercambiador (la salida de la caliente y la entrada de la fría)."""
    fig = go.Figure()
    names = [
        r.inputs.stream_a.label or "Corriente A",
        r.inputs.stream_b.label or "Corriente B",
    ]
    subs = (r.inputs.stream_a.substance, r.inputs.stream_b.substance)
    hot = 0 if r.heat(0) < 0.0 else 1
    for i, color in ((hot, _ORANGE), (1 - hot, _BLUE)):
        qs, Ts = _stream_curve(subs[i], r.inlets[i], r.outlets[i], r.m_dots[i])
        fig.add_trace(
            go.Scatter(
                x=_conv(qs, "power", system),
                y=_conv(Ts, "temperature", system),
                mode="lines",
                name=f"{upper_first(names[i])} ({'se enfría' if i == hot else 'se calienta'})",
                line={"color": color, "width": 2.5},
            )
        )
    fig.update_layout(
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": _axis("Calor intercambiado Q̇", "power", system), "gridcolor": _GRID},
        yaxis={"title": _axis("T", "temperature", system), "gridcolor": _GRID},
        legend=_LEGEND_BELOW,
        separators=", ",
        title={"text": "Diagrama T–Q (a contracorriente)", "x": 0},
    )
    return fig
