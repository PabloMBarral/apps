"""Gráficos de la combustión — Fase 5.

- :func:`composition_figure`: los humos en base húmeda y seca (barras
  apiladas al 100 %).
- :func:`combustion_diagram_figure`: el diagrama de combustión: CO₂, O₂ y CO
  en los humos secos en función de λ, con el punto de los datos o el medido.
- :func:`energy_split_figure`: del PCI al calor útil, en cascada.
- :func:`sweep_figure`: las curvas de los barridos, con una línea en el valor
  de los datos.

Colores: la paleta de referencia validada para daltonismo (azul, naranja y
aguamarina; violeta para los otros gases de los humos, que en las barras solo
toca al naranja y al gris) y un gris claro para el N₂, el inerte que es casi
todo. Las barras llevan sus porcentajes escritos y la página trae la tabla
con los mismos números.

Vive en ``ui/`` porque arma figuras de plotly; el cálculo está en
:mod:`core.combustion`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from core.combustion.combustion import EnergySplit
from core.combustion.stoichiometry import Stoichiometry
from core.units_system import UnitSystem, convert_from_si, unit_label

__all__ = [
    "combustion_diagram_figure",
    "composition_figure",
    "energy_split_figure",
    "sweep_figure",
]

_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"
_VIOLET = "#4a3aa7"
_N2_GRAY = "#c4c8cf"
_MUTED = "#6b7280"
_INK = "#1f2937"

#: Color del texto dentro de cada barra: blanco sobre los oscuros, tinta sobre los claros.
_TEXT_ON = {_BLUE: "white", _AQUA: _INK, _ORANGE: _INK, _VIOLET: "white", _N2_GRAY: _INK}

#: Grupos de las barras de los humos: nombre, especies y color.
_GROUPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("CO₂", ("CO2",), _BLUE),
    ("H₂O", ("H2O",), _AQUA),
    ("O₂", ("O2",), _ORANGE),
    ("CO, H₂ y SO₂", ("CO", "H2", "SO2"), _VIOLET),
    ("N₂ y Ar", ("N2", "Ar"), _N2_GRAY),
)


def composition_figure(stoich: Stoichiometry) -> go.Figure:
    """Fracciones molares de los humos húmedos y secos, en barras apiladas al 100 %."""
    rows = ("húmedos", "secos")
    fractions = {"húmedos": stoich.wet_fractions, "secos": stoich.dry_fractions}
    fig = go.Figure()
    for name, keys, color in _GROUPS:
        xs = [100.0 * sum(fractions[r].get(k, 0.0) for k in keys) for r in rows]
        if max(xs) <= 0.0:
            continue
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=list(rows),
                x=xs,
                name=name,
                marker={"color": color, "line": {"color": "white", "width": 2}},
                text=[f"{x:.1f} %" if x >= 5.0 else "" for x in xs],
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                insidetextfont={"color": _TEXT_ON[color]},
                hovertemplate=f"{name}: %{{x:.4g}} %<extra>humos %{{y}}</extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=210,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"range": [0, 100], "title": "fracción molar [%]"},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "traceorder": "normal"},
        uniformtext={"minsize": 10, "mode": "hide"},
        separators=". ",
    )
    return fig


def combustion_diagram_figure(
    curves: Mapping[str, Sequence[float]],
    *,
    point_lambda: float | None = None,
    point: Mapping[str, float] | None = None,
    point_label: str = "tus datos",
) -> go.Figure:
    """Diagrama de combustión: CO₂, O₂ y CO en los humos secos contra λ.

    ``curves`` es la salida de :func:`core.combustion.stoichiometry.dry_gas_curves`
    (fracciones); ``point`` las fracciones secas en ``point_lambda`` (el punto
    de los datos o el de la medición).
    """
    lam = list(curves["lambda"])
    fig = go.Figure()
    series = (("CO2", "CO₂", _BLUE), ("O2", "O₂", _ORANGE), ("CO", "CO", _AQUA))
    marked = False
    for key, name, color in series:
        ys = [100.0 * v for v in curves[key]]
        if not ys or max(ys) <= 0.0:
            continue
        fig.add_trace(
            go.Scatter(
                x=lam,
                y=ys,
                mode="lines",
                name=name,
                line={"color": color, "width": 2},
                hovertemplate=f"λ = %{{x:.3f}}<br>{name} = %{{y:.3g}} %<extra></extra>",
            )
        )
        if point is not None and point_lambda is not None and point.get(key, 0.0) > 0.0:
            fig.add_trace(
                go.Scatter(
                    x=[point_lambda],
                    y=[100.0 * point[key]],
                    mode="markers",
                    name=point_label,
                    legendgroup="punto",
                    showlegend=False,
                    marker={
                        "color": color,
                        "size": 11,
                        "symbol": "diamond",
                        "line": {"color": "white", "width": 2},
                    },
                    hovertemplate=(
                        f"{point_label}<br>λ = %{{x:.4g}}<br>{name} = %{{y:.4g}} %<extra></extra>"
                    ),
                )
            )
            marked = True
    if marked:  # en la leyenda, el punto va en gris: su color es el de cada curva
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=point_label,
                legendgroup="punto",
                marker={"color": _MUTED, "size": 11, "symbol": "diamond"},
                hoverinfo="skip",
            )
        )
    if lam and min(lam) < 1.0 < max(lam):
        fig.add_vline(x=1.0, line={"color": _MUTED, "width": 1, "dash": "dot"})
    fig.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="λ (relación de aire)",
        yaxis_title="humos secos [%]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
    )
    return fig


def energy_split_figure(split: EnergySplit, per_kg: float, system: UnitSystem) -> go.Figure:
    """Del PCI al calor útil, por kg de combustible: lo que suma y lo que se pierde.

    Cada barra lleva su porcentaje del PCI; aumentos en aguamarina, pérdidas en
    naranja y los totales en azul.
    """
    items: list[tuple[str, float, str]] = [("PCI", split.lhv, "absolute")]
    if abs(split.sensible_reactants) > 1e-9 * split.lhv:
        items.append(("reactivos (sensible)", split.sensible_reactants, "relative"))
    if split.unburned > 1e-9 * split.lhv:
        items.append(("inquemados", -split.unburned, "relative"))
    items.append(("humos (sensible)", -split.stack_sensible, "relative"))
    if split.latent_recovered > 1e-9 * split.lhv:
        items.append(("latente recuperado", split.latent_recovered, "relative"))
    items.append(("calor útil", split.Q_out, "total"))
    unit = unit_label("specific_enthalpy", system)
    values = [convert_from_si(v * per_kg, "specific_enthalpy", system) for _, v, _ in items]
    text = [
        f"{100 * v / split.lhv:+.1f} %" if m == "relative" else f"{100 * v / split.lhv:.1f} %"
        for _, v, m in items
    ]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=[name for name, _, _ in items],
            x=values,
            measure=[m for _, _, m in items],
            text=text,
            textposition="outside",
            cliponaxis=False,
            increasing={"marker": {"color": _AQUA}},
            decreasing={"marker": {"color": _ORANGE}},
            totals={"marker": {"color": _BLUE}},
            connector={"line": {"color": _MUTED, "width": 1, "dash": "dot"}},
            hovertemplate=f"%{{y}}: %{{x:.5g}} {unit}<extra></extra>",
        )
    )
    top = max(values[0], values[-1], *(sum(values[: k + 1]) for k in range(len(values) - 1)))
    fig.update_layout(
        height=80 + 42 * len(items),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"title": f"por kg de combustible [{unit}]", "range": [0, 1.22 * top]},
        yaxis={"autorange": "reversed"},
        showlegend=False,
        separators=". ",
    )
    return fig


def sweep_figure(
    xs: Sequence[float],
    series: Mapping[str, Sequence[float | None]],
    *,
    title: str,
    x_title: str,
    y_title: str,
    current_x: float | None = None,
    colors: Sequence[str] = (_ORANGE, _BLUE, _AQUA),
) -> go.Figure:
    """Curvas de un barrido (los ``None`` se omiten) y el valor de los datos punteado.

    El título va arriba y la leyenda en su propio renglón, pegada a la izquierda:
    anclada más a la derecha no entra en un celular y plotly achica el gráfico
    para hacerle lugar.
    """
    fig = go.Figure()
    for (name, ys), color in zip(series.items(), colors, strict=False):
        pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
        if not pts:
            continue
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pts],
                y=[p[1] for p in pts],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"size": 5},
            )
        )
    if current_x is not None:
        fig.add_vline(x=current_x, line={"color": _MUTED, "width": 1, "dash": "dot"})
    legend = len(series) > 1
    fig.update_layout(
        height=330 if legend else 300,
        margin={"l": 10, "r": 10, "t": 70 if legend else 40, "b": 10},
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
    )
    return fig
