"""Gráficos de ciclos compartidos entre páginas — Fases 3.3 y 3.4.

- :func:`tq_figure`: el diagrama T–Q de la HRSG (/HRSG y /Ciclo_Combinado).
- :func:`render_rankine_diagram`: el ciclo de Rankine sobre el diagrama del
  agua u otro fluido (/Rankine y /Ciclo_Combinado).
- :func:`gas_turbine_ts_figure`: la turbina de gas en un T–s con entropías
  absolutas, con el enfriamiento de los gases en la HRSG.
- :func:`energy_sankey_figure`: a dónde va el calor del combustible en un
  ciclo combinado.
- :func:`multi_tq_figure`, :func:`exergy_split_figure` y
  :func:`exergy_sections_figure`: la HRSG de varias presiones (Fase 3.5).
- :func:`render_steam_cycle_diagram`, :func:`bottoming_exergy_figure` y
  :func:`configuration_figure`: el ciclo combinado de varias presiones, con
  recalentamiento (Fase 3.6).
- :func:`gas_turbine_cycle_figure`, :func:`improvement_figure` y
  :func:`gas_turbine_exergy_figure`: la turbina de gas con interenfriamiento,
  recalentamiento y regenerador (Fase 3.7).

Los colores de las exergías (útil, destruida, perdida) y de la comparación de
configuraciones son los tres primeros de la paleta de referencia validada para
daltonismo (azul, naranja y aguamarina): se distinguen también con protanopía y
deuteranopía; cada barra lleva su valor escrito.

Vive en ``ui/`` porque arma figuras de plotly y usa Streamlit; el cálculo
está en ``core/``.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.cycles.brayton import BraytonResult, brayton_ts_lines
from core.cycles.combined import CombinedResult
from core.cycles.combined_multi import BottomingExergy, ConfigurationRow, MultiCombinedResult
from core.cycles.gas_turbine import (
    GasTurbineExergy,
    GasTurbineResult,
    ImprovementRow,
    gas_turbine_lines,
)
from core.cycles.hrsg import HRSGResult, TQProfile
from core.cycles.hrsg_multi import HRSGExergy, MultiHRSGResult, MultiTQProfile
from core.cycles.rankine import RankineResult, rankine_labeled_states, rankine_segments
from core.diagrams import (
    DiagramSpec,
    DiagramType,
    ProcessOverlay,
    _join_with_gaps,
    isentropic_process,
    segments_overlays,
)
from core.fluids import StatePoint
from core.state_report import format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.diagrams import DiagramPoint, get_diagram, render_diagram_plotly

__all__ = [
    "LEVEL_COLORS",
    "bottoming_exergy_figure",
    "configuration_figure",
    "energy_sankey_figure",
    "exergy_sections_figure",
    "exergy_split_figure",
    "gas_turbine_cycle_figure",
    "gas_turbine_exergy_figure",
    "gas_turbine_ts_figure",
    "improvement_figure",
    "multi_tq_figure",
    "render_rankine_diagram",
    "render_steam_cycle_diagram",
    "tq_figure",
]

_SECTION_SHORT = {
    "sobrecalentador": "SH",
    "evaporador": "EV",
    "economizador": "ECO",
    "recalentador": "RH",
}
_REHEAT_COLOR = "#9467bd"
# Paleta de referencia validada para daltonismo (azul, naranja, aguamarina).
_USEFUL = "#2a78d6"
_DESTROYED = "#eb6834"
_LOST = "#1baf7a"
_GAS_COLOR = "#d62728"
_WATER_COLOR = "#1f77b4"
_AIR_COLOR = "#1f77b4"
_GREEN = "#2ca02c"


def _value(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Valor en ``system``, sin unidad."""
    return format_value(convert_from_si(value_si, kind, system), sig)


def tq_figure(result: HRSGResult, profile: TQProfile, system: UnitSystem) -> go.Figure:
    """Diagrama T–Q: gases y agua contra el calor transferido desde la chimenea."""

    def T(value_K: float) -> float:
        return convert_from_si(value_K, "temperature", system)

    def Q(value_W: float) -> float:
        return convert_from_si(value_W, "power", system)

    T_unit = unit_label("temperature", system)
    Q_unit = unit_label("power", system)
    dT_unit = unit_label("temperature_difference", system)
    hover = f"Q̇ = %{{x:,.0f}} {Q_unit}<br>T = %{{y:.1f}} {T_unit}"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[Q(q) for q in profile.Q_gas_W],
            y=[T(t) for t in profile.T_gas_K],
            mode="lines",
            name="gases",
            line={"color": _GAS_COLOR, "width": 3},
            hovertemplate=hover + "<extra>gases</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[Q(q) for q in profile.Q_water_W],
            y=[T(t) for t in profile.T_water_K],
            mode="lines",
            name="agua y vapor",
            line={"color": _WATER_COLOR, "width": 3},
            hovertemplate=hover + "<extra>agua y vapor</extra>",
        )
    )
    # Puntos de los gases: Q acumulado desde la chimenea.
    sections = result.sections
    gas_Q = [result.Q_W]
    for section in sections:
        gas_Q.append(gas_Q[-1] - section.Q_W)
    gas_pos = {"a": "top left", "b": "top left", "c": "top left", "d": "top center"}
    fig.add_trace(
        go.Scatter(
            x=[Q(q) for q in gas_Q],
            y=[T(t) for t in result.T_gas_K],
            mode="markers+text",
            text=list(result.gas_labels),
            textposition=[gas_pos[label] for label in result.gas_labels],
            textfont={"color": _GAS_COLOR, "size": 14},
            marker={"color": _GAS_COLOR, "size": 8},
            showlegend=False,
            hovertemplate=hover + "<extra>gases</extra>",
        )
    )
    # Puntos del agua. El 2 y el 3 (el escalón del approach) quedan a unos pocos
    # kelvin: los nombran las flechas del pinch y del approach.
    Q_eco = sections[-1].Q_W
    Q_ev = sections[-2].Q_W
    w = result.water
    water_Q = [0.0, Q_eco, Q_eco, Q_eco + Q_ev, result.Q_W][: len(w)]
    water_text = ["1", "", "", "4", "5"][: len(w)]
    fig.add_trace(
        go.Scatter(
            x=[Q(q) for q in water_Q],
            y=[T(s.T_K) for s in w],
            mode="markers+text",
            text=water_text,
            textposition="bottom right",
            textfont={"color": _WATER_COLOR, "size": 14},
            marker={"color": _WATER_COLOR, "size": 8},
            showlegend=False,
            hovertemplate=hover + "<extra>agua y vapor</extra>",
        )
    )
    # Límites entre secciones y sus nombres (arriba: abajo a la izquierda está el agua
    # de alimentación).
    bounds = [0.0, *profile.boundaries_W, result.Q_W]
    for q in profile.boundaries_W:
        fig.add_vline(x=Q(q), line_dash="dot", line_color="gray", line_width=1)
    names = [_SECTION_SHORT[s.name] for s in reversed(sections)]
    for name, lo, hi in zip(names, bounds, bounds[1:], strict=False):
        fig.add_annotation(
            x=Q(0.5 * (lo + hi)),
            y=0.99,
            yref="paper",
            text=name,
            showarrow=False,
            font={"color": "gray"},
            yanchor="top",
        )
    # Pinch (entre c y 3) y approach (entre 2 y 3): unos pocos kelvin, no se ven a
    # escala; las flechas vienen de zonas libres del evaporador (arriba de los gases
    # y abajo de la meseta), también en un celular.
    inputs = result.inputs
    pinch = _value(result.pinch_K, "temperature_difference", system, 3)
    fig.add_annotation(
        x=Q(Q_eco),
        y=T(result.T_sat_K + 0.5 * result.pinch_K),
        text=f"pinch {pinch} {dT_unit}<br>(c – 3)",
        showarrow=True,
        arrowhead=2,
        ax=55,
        ay=-85,
        font={"color": "#333333"},
    )
    if inputs.approach_K > 0.0:
        approach = _value(inputs.approach_K, "temperature_difference", system, 3)
        text = f"approach {approach} {dT_unit}<br>(2 → 3)"
    else:
        text = "approach 0<br>(2 = 3)"
    fig.add_annotation(
        x=Q(Q_eco),
        y=T(result.T_sat_K - 0.5 * inputs.approach_K),
        text=text,
        showarrow=True,
        arrowhead=2,
        ax=55,
        ay=65,
        font={"color": "#333333"},
    )
    fig.update_layout(
        height=440,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title=f"Q̇ desde la chimenea [{Q_unit}]",
        yaxis_title=f"T [{T_unit}]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
        hovermode="closest",
    )
    Q_max = Q(result.Q_W)  # margen para los rótulos de los extremos
    fig.update_xaxes(
        exponentformat="none", separatethousands=True, range=[-0.04 * Q_max, 1.07 * Q_max]
    )
    return fig


def render_rankine_diagram(
    result: RankineResult, system: UnitSystem, diagram_type: DiagramType, *, chart_key: str
) -> None:
    """El ciclo de Rankine sobre el diagrama del fluido (fluprodia + plotly).

    Estados numerados, procesos sobre sus isolíneas (:func:`core.diagrams.segments_overlays`)
    y, con turbina real, la expansión isoentrópica de referencia y los estados ks.
    Lo usan /Rankine y /Ciclo_Combinado.
    """
    labeled = rankine_labeled_states(result)
    states = [state.to_state_point() for _, state in labeled]
    points = [
        DiagramPoint(state=state, label=label.split()[0], color="#d62728")
        for (label, _), state in zip(labeled, states, strict=True)
    ]
    fluid = result.inputs.fluid
    diagram = get_diagram(fluid, system)
    spec = DiagramSpec(fluid=fluid, system=system)
    overlays: list[ProcessOverlay] = segments_overlays(
        diagram, spec, [(states[a], states[b]) for a, b in rankine_segments(result)]
    )
    if result.inputs.eta_turbine < 1.0:
        turbines = result.of_kind("turbine")
        casings: dict[str, tuple[int, int]] = {}
        for comp in turbines:
            start = casings.get(comp.casing, (comp.port("in").state, 0))[0]
            casings[comp.casing] = (start, comp.port("out").state)
        for start_i, end_i in casings.values():
            start = result.states[start_i]
            overlays.append(
                ProcessOverlay(
                    name=f"{start_i + 1} → {end_i + 1}s (isoentrópica de referencia)",
                    color="#2ca02c",
                    dash="dash",
                    coords_si=isentropic_process(
                        diagram,
                        spec,
                        s_J_per_kg_K=start.s_J_per_kg_K,
                        p_start_Pa=start.P_Pa,
                        p_end_Pa=result.states[end_i].P_Pa,
                    ),
                )
            )
        for comp, state_s in zip(turbines, result.turbine_out_s, strict=True):
            points.append(
                DiagramPoint(
                    state=state_s.to_state_point(),
                    label=f"{comp.port('out').state + 1}s",
                    color="#2ca02c",
                )
            )
    render_diagram_plotly(
        fluid=fluid,
        diagram_type=diagram_type,
        system=system,
        points=points,
        overlays=overlays,
        chart_key=chart_key,
        point_legend={"#d62728": "estados", "#2ca02c": "estados isoentrópicos (ks)"},
    )


def gas_turbine_ts_figure(
    result: BraytonResult, system: UnitSystem, T_stack_K: float | None = None
) -> go.Figure:
    """La turbina de gas en un diagrama T–s, con entropías absolutas.

    Con ``T_stack_K``, también el enfriamiento de los gases en la caldera de
    recuperación (4 → chimenea).
    """

    def T(value_K: float) -> float:
        return convert_from_si(value_K, "temperature", system)

    def s(value: float) -> float:
        return convert_from_si(value, "specific_entropy", system)

    T_unit = unit_label("temperature", system)
    s_unit = unit_label("specific_entropy", system)
    hover = f"s = %{{x:.4g}} {s_unit}<br>T = %{{y:.1f}} {T_unit}"
    styles = {
        "isobar": ("isobaras", {"color": "#bbbbbb", "width": 1}),
        "heat": ("calor a p constante", {"color": _GAS_COLOR, "width": 3}),
        "isentropic": ("isoentrópicas (2s, 4s)", {"color": _GREEN, "width": 2, "dash": "dash"}),
        "actual": (
            "compresión y expansión reales",
            {"color": "#555555", "width": 2, "dash": "dot"},
        ),
        "composition": ("cámara: cambia la composición", {"color": "#999999", "dash": "dot"}),
    }
    fig = go.Figure()
    shown: set[str] = set()
    for line in brayton_ts_lines(result, T_stack_K):
        name, style = styles[line.kind]
        fig.add_trace(
            go.Scatter(
                x=[s(v) for v in line.s_J_per_kg_K],
                y=[T(v) for v in line.T_K],
                mode="lines",
                name=name,
                legendgroup=line.kind,
                showlegend=line.kind not in shown and line.kind != "composition",
                line=style,
                hovertemplate=hover + f"<extra>{line.medium}, p = "
                f"{_value(line.p_Pa, 'pressure', system, 4)} {unit_label('pressure', system)}"
                "</extra>",
            )
        )
        shown.add(line.kind)
    states = list(result.states)
    labels = [st.label for st in states]
    xs = [s(st.s_J_per_kg_K) for st in states]
    ys = [T(st.T_K) for st in states]
    positions = {
        "1": "bottom right",
        "2s": "top left",
        "2": "middle right",
        "3": "top center",
        "4s": "bottom left",
        "4": "middle right",
    }
    if T_stack_K is not None:
        gas = result.gas
        labels.append("chim.")
        xs.append(s(gas.s(T_stack_K, result.state("4").P_Pa)))
        ys.append(T(T_stack_K))
        positions["chim."] = "bottom right"
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition=[positions[label] for label in labels],
            textfont={"size": 13},
            marker={"color": "#333333", "size": 8},
            showlegend=False,
            hovertemplate=hover + "<extra>%{text}</extra>",
        )
    )
    fig.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title=f"s [{s_unit}]",
        yaxis_title=f"T [{T_unit}]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
    )
    return fig


def energy_sankey_figure(result: CombinedResult, system: UnitSystem) -> go.Figure:
    """A dónde va el calor del combustible: trabajo de las dos turbinas, condensador y chimenea."""
    Q = result.Q_fuel_W
    unit = unit_label("power", system)
    source = "Calor" if result.gas_turbine.inputs.air_standard else "Combustible"
    # Rótulos en varios renglones (<br>): en un celular no se pisan entre columnas.
    nodes = [
        (source, Q, "#7f7f7f"),
        ("Ẇ turbina<br>de gas", result.W_gas_turbine_W, _GREEN),
        ("Escape", result.Q_exhaust_W, _GAS_COLOR),
        ("Vapor<br>(HRSG)", result.Q_hrsg_W, _WATER_COLOR),
        ("Ẇ ciclo<br>de vapor", result.W_steam_turbine_W, _GREEN),
        ("Condensador", result.Q_condenser_W, "#9467bd"),
        ("Chimenea", result.Q_stack_W, "#8c564b"),
    ]
    links = [(0, 1), (0, 2), (2, 3), (2, 6), (3, 4), (3, 5)]

    def label(name: str, value: float) -> str:
        return f"{name}<br>{value / Q * 100:.1f} %"

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=",.0f",
            valuesuffix=f" {unit}",
            node={
                "label": [label(n, v) for n, v, _ in nodes],
                "color": [c for _, _, c in nodes],
                "pad": 14,
                "thickness": 14,
            },
            link={
                "source": [a for a, _ in links],
                "target": [b for _, b in links],
                "value": [convert_from_si(nodes[b][1], "power", system) for _, b in links],
                "color": ["rgba(150,150,150,0.35)"] * len(links),
            },
        )
    )
    fig.update_layout(
        height=380,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        separators=". ",
        font={"size": 12},
    )
    return fig


# ---------------------------------------------------------------------
# HRSG de varias presiones (Fase 3.5)
# ---------------------------------------------------------------------

#: Un azul por nivel: más oscuro cuanto mayor la presión.
LEVEL_COLORS = {"alta": "#08306b", "media": "#2171b5", "baja": "#6baed6", "": _WATER_COLOR}
_LEVEL_SHORT = {"alta": "A", "media": "M", "baja": "B", "": ""}


def multi_tq_figure(
    result: MultiHRSGResult, profile: MultiTQProfile, system: UnitSystem
) -> go.Figure:
    """Diagrama T–Q de la HRSG en cascada: la curva del agua en «serrucho», un color por nivel.

    En cada evaporador, un segmento punteado marca el pinch (de T_sat a los gases).
    Los nombres de las secciones van arriba, salvo las muy angostas.
    """

    def T(value_K: float) -> float:
        return convert_from_si(value_K, "temperature", system)

    def Q(value_W: float) -> float:
        return convert_from_si(value_W, "power", system)

    T_unit = unit_label("temperature", system)
    Q_unit = unit_label("power", system)
    dT_unit = unit_label("temperature_difference", system)
    hover = f"Q̇ = %{{x:,.0f}} {Q_unit}<br>T = %{{y:.1f}} {T_unit}"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[Q(q) for q in profile.Q_gas_W],
            y=[T(t) for t in profile.T_gas_K],
            mode="lines",
            name="gases",
            line={"color": _GAS_COLOR, "width": 3},
            hovertemplate=hover + "<extra>gases</extra>",
        )
    )
    for i, lv in enumerate(result.levels):  # un trazo por nivel (sus secciones son contiguas)
        xs: list[float] = []
        ys: list[float] = []
        for seg in (s for s in profile.segments if s.level == i and s.kind != "recalentador"):
            xs += [Q(q) for q in seg.Q_W]
            ys += [T(t) for t in seg.T_K]
        name = f"agua y vapor de {lv.name}" if lv.name else "agua y vapor"
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=name,
                line={"color": LEVEL_COLORS[lv.name], "width": 3},
                hovertemplate=hover + f"<extra>{name}</extra>",
            )
        )
    # El recalentador, en paralelo con el sobrecalentador de alta (mismo tramo de Q).
    for seg in (s for s in profile.segments if s.kind == "recalentador"):
        fig.add_trace(
            go.Scatter(
                x=[Q(q) for q in seg.Q_W],
                y=[T(t) for t in seg.T_K],
                mode="lines",
                name="recalentador (en paralelo)",
                line={"color": _REHEAT_COLOR, "width": 3, "dash": "dash"},
                hovertemplate=hover + "<extra>recalentador</extra>",
            )
        )
    # El pinch de cada nivel: de T_sat a los gases, a la salida de gases del evaporador.
    for seg in (s for s in profile.segments if s.kind == "evaporador"):
        lv = result.levels[seg.level]
        q = Q(seg.Q_W[0])
        pinch = _value(lv.pinch_K, "temperature_difference", system, 3)
        label = f"pinch{' de ' + lv.name if lv.name else ''}: {pinch} {dT_unit}"
        fig.add_trace(
            go.Scatter(
                x=[q, q],
                y=[T(lv.T_sat_K), T(lv.T_pinch_gas_K)],
                mode="lines+markers",
                line={"color": "#333333", "dash": "dot", "width": 2},
                marker={"size": 5, "color": "#333333"},
                name="pinch de cada nivel",
                legendgroup="pinch",
                showlegend=seg.level == 0,
                hovertemplate=f"{label}<extra></extra>",
            )
        )
    # Puntos de los gases (a, b, c…): el Q acumulado desde la chimenea.
    points = result.gas_points
    m_g = result.inputs.m_gas_kg_s
    gas_Q = [m_g * (h - result.h_stack_J_per_kg) for _, _, h in points]
    # Un punto pegado al anterior (una sección muy chica) queda sin rótulo: se pisarían.
    texts: list[str] = []
    last_Q = None
    for (label, _, _), q in zip(points, gas_Q, strict=True):
        close = last_Q is not None and abs(last_Q - q) < 0.035 * result.Q_W
        texts.append("" if close else label)
        if not close:
            last_Q = q
    fig.add_trace(
        go.Scatter(
            x=[Q(q) for q in gas_Q],
            y=[T(t) for _, t, _ in points],
            mode="markers+text",
            text=texts,
            textposition="top left",
            textfont={"color": _GAS_COLOR, "size": 13},
            marker={"color": _GAS_COLOR, "size": 7},
            showlegend=False,
            hovertemplate=hover + "<extra>gases</extra>",
        )
    )
    # Límites y nombres de las secciones (los de las muy angostas no entrarían).
    for q in profile.boundaries_W:
        fig.add_vline(x=Q(q), line_dash="dot", line_color="lightgray", line_width=1)
    bounds = [0.0, *profile.boundaries_W, result.Q_W]
    # Un rótulo por tramo, desde la chimenea; los bancos en paralelo comparten el suyo.
    shorts: list[str] = []
    parallel = False
    for flows in reversed(result._flows()):
        kind, _, level = flows.section.name.partition(" de ")
        short = _SECTION_SHORT[kind] + (f" {_LEVEL_SHORT[level]}" if level else "")
        if flows.gas_share < 1.0 and parallel:
            shorts[-1] = f"{short} + {shorts[-1]}"
            continue
        parallel = flows.gas_share < 1.0
        shorts.append(short)
    row = 0
    for short, lo, hi in zip(shorts, bounds[:-1], bounds[1:], strict=True):
        if (hi - lo) < 0.05 * result.Q_W:
            continue
        fig.add_annotation(  # en dos alturas alternadas: en un celular no se pisan
            x=Q(0.5 * (lo + hi)),
            y=0.99 - 0.06 * (row % 2),
            yref="paper",
            text=short,
            showarrow=False,
            font={"color": "gray", "size": 11},
            yanchor="top",
        )
        row += 1
    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title=f"Q̇ desde la chimenea [{Q_unit}]",
        yaxis_title=f"T [{T_unit}]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
        hovermode="closest",
    )
    Q_max = Q(result.Q_W)
    fig.update_xaxes(
        exponentformat="none", separatethousands=True, range=[-0.04 * Q_max, 1.05 * Q_max]
    )
    return fig


_EXERGY_PARTS = (
    ("al agua y el vapor", _USEFUL),
    ("destruida", _DESTROYED),
    ("por la chimenea", _LOST),
)


def exergy_split_figure(rows: list[tuple[str, HRSGExergy]], system: UnitSystem) -> go.Figure:
    """Barras horizontales apiladas: a dónde va la exergía de los gases, en % de lo que traen."""
    fig = go.Figure()
    unit = unit_label("power", system)
    labels = [label for label, _ in rows]
    for k, (name, color) in enumerate(_EXERGY_PARTS):
        values = [
            (x.X_water_W, x.X_destroyed_W, x.X_stack_W)[k] / x.X_gas_in_W * 100.0 for _, x in rows
        ]
        absolute = [
            convert_from_si((x.X_water_W, x.X_destroyed_W, x.X_stack_W)[k], "power", system)
            for _, x in rows
        ]
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                name=name,
                marker_color=color,
                text=[f"{v:.1f} %" for v in values],
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                customdata=absolute,
                hovertemplate=f"{name}: %{{x:.1f}} %% (%{{customdata:,.0f}} {unit})<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=110 + 50 * len(rows),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"title": "% de la exergía de los gases", "range": [0, 100]},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.35, "x": 0, "traceorder": "normal"},
        uniformtext={"minsize": 10, "mode": "hide"},  # sin texto en los tramos angostos
        separators=". ",
    )
    return fig


def exergy_sections_figure(exergy: HRSGExergy, system: UnitSystem) -> go.Figure:
    """Exergía destruida en cada sección (barras horizontales, un color por nivel)."""
    unit = unit_label("power", system)
    names = [name for name, _ in exergy.destroyed_W]
    values = [convert_from_si(x, "power", system) for _, x in exergy.destroyed_W]
    colors = []
    for name in names:
        level = name.partition(" de ")[2]
        colors.append(LEVEL_COLORS.get(level, _WATER_COLOR))
    fig = go.Figure(
        go.Bar(
            y=names,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{x / exergy.X_destroyed_W * 100:.0f} %" for _, x in exergy.destroyed_W],
            textposition="outside",
            hovertemplate=f"%{{y}}: %{{x:,.0f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(
        height=80 + 28 * len(names),
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_range=[0, 1.25 * max(values)],  # lugar para el % de afuera
        xaxis_title=f"Ẋ destruida [{unit}]",
        yaxis={"autorange": "reversed"},
        separators=". ",
        showlegend=False,
    )
    fig.update_xaxes(exponentformat="none", separatethousands=True)
    return fig


# ---------------------------------------------------------------------
# Ciclo combinado de varias presiones (Fase 3.6)
# ---------------------------------------------------------------------


def render_steam_cycle_diagram(
    result: MultiCombinedResult, system: UnitSystem, diagram_type: DiagramType, *, chart_key: str
) -> None:
    """El ciclo de vapor con admisiones sobre el diagrama del agua (fluprodia + plotly).

    Estados numerados del ciclo; el agua de cada nivel en la HRSG (de la entrada
    a su economizador al vapor) y las bombas entre niveles, sin numerar; las
    turbinas, las mezclas, el recalentador, el desaireador y el condensador; y,
    con η_T < 1, la expansión isoentrópica de referencia de cada turbina.
    """
    cyc = result.steam
    hrsg = result.hrsg
    states = [s.to_state_point() for s in cyc.states]
    pairs: list[tuple[StatePoint, StatePoint]] = []
    if cyc.deaerator is None:
        pairs.append((states[0], states[cyc.feedwater]))
    else:
        cp, da, fw = cyc.condensate_pump_out, cyc.deaerator, cyc.feedwater
        pairs += [(states[0], states[cp]), (states[cp], states[da]), (states[da], states[fw])]
    levels = hrsg.levels
    pairs.append((states[cyc.feedwater], levels[-1].water[-1].to_state_point()))
    for i in range(len(levels) - 1):
        below, lv = levels[i + 1], levels[i]
        pairs.append((below.water[2].to_state_point(), lv.water[0].to_state_point()))
        pairs.append((lv.water[0].to_state_point(), lv.water[-1].to_state_point()))
    for t in cyc.turbines:
        if t.extraction is None:
            pairs.append((states[t.inlet], states[t.outlet]))
        else:
            pairs += [
                (states[t.inlet], states[t.extraction]),
                (states[t.extraction], states[t.outlet]),
            ]
    for a in cyc.admissions:
        pairs += [(states[a.turbine], states[a.outlet]), (states[a.steam], states[a.outlet])]
    if cyc.hot_reheat is not None and cyc.reheat_inlet is not None:
        pairs.append((states[cyc.reheat_inlet], states[cyc.hot_reheat]))
    if cyc.deaerator is not None and cyc.extraction is not None:
        pairs.append((states[cyc.extraction], states[cyc.deaerator]))
    pairs.append((states[cyc.exhaust], states[0]))
    diagram = get_diagram("Water", system)
    spec = DiagramSpec(fluid="Water", system=system)
    overlays: list[ProcessOverlay] = segments_overlays(diagram, spec, pairs)
    points = [
        DiagramPoint(state=s, label=str(k + 1), color=_GAS_COLOR) for k, s in enumerate(states)
    ]
    if result.inputs.steam.eta_turbine < 1.0:
        # Las isoentrópicas de todas las turbinas en una sola entrada de la leyenda: en un
        # celular, una por turbina ocupaba media figura.
        chunks = [
            isentropic_process(
                diagram,
                spec,
                s_J_per_kg_K=cyc.states[t.inlet].s_J_per_kg_K,
                p_start_Pa=cyc.states[t.inlet].P_Pa,
                p_end_Pa=cyc.states[t.outlet].P_Pa,
            )
            for t in cyc.turbines
        ]
        overlays.append(
            ProcessOverlay(
                name="expansiones isoentrópicas de referencia",
                color=_GREEN,
                dash="dash",
                coords_si=_join_with_gaps(chunks),
            )
        )
        for t in cyc.turbines:
            points.append(
                DiagramPoint(
                    state=t.outlet_s.to_state_point(), label=f"{t.outlet + 1}s", color=_GREEN
                )
            )
    render_diagram_plotly(
        fluid="Water",
        diagram_type=diagram_type,
        system=system,
        points=points,
        overlays=overlays,
        chart_key=chart_key,
        point_legend={_GAS_COLOR: "estados", _GREEN: "estados isoentrópicos (ks)"},
    )


_SHORT_WORDS = (
    ("sobrecalentador", "SH"),
    ("recalentador", "RH"),
    ("evaporador", "EV"),
    ("economizador", "ECO"),
    ("mezcla del vapor de ", "mezcla "),
    (" (recalentamiento frío)", " (RF)"),
    (" (turbina)", ""),
    ("turbina de ", "turbina "),
)


def _short(name: str) -> str:
    """Rótulo corto para el eje (en un celular no entran los nombres largos)."""
    for long, short in _SHORT_WORDS:
        name = name.replace(long, short)
    return name


def bottoming_exergy_figure(exergy: BottomingExergy, system: UnitSystem) -> go.Figure:
    """Barras horizontales: a dónde va la exergía de los gases de escape (% de lo que traen)."""
    X = exergy.X_gas_in_W
    unit = unit_label("power", system)
    rows: list[tuple[str, float, str]] = [("trabajo neto (vapor)", exergy.W_net_W, "trabajo")]
    rows += [(name, value, "destruida") for name, value in exergy.destroyed_W if value > 0.0]
    rows += [
        ("condensador", exergy.X_condenser_W, "perdida"),
        ("chimenea", exergy.X_stack_W, "perdida"),
    ]
    order = [_short(name) for name, _, _ in rows]
    fig = go.Figure()
    for group, color, legend in (
        ("trabajo", _USEFUL, "trabajo"),
        ("destruida", _DESTROYED, "destruida (T₀·S_gen)"),
        ("perdida", _LOST, "se va (condensador y chimenea)"),
    ):
        sel = [r for r in rows if r[2] == group]
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[_short(name) for name, _, _ in sel],
                x=[value / X * 100 for _, value, _ in sel],
                name=legend,
                marker={"color": color, "cornerradius": 4},
                text=[f"{value / X * 100:.1f} %" for _, value, _ in sel],
                textposition="outside",
                cliponaxis=False,
                customdata=[[name, _value(value, "power", system)] for name, value, _ in sel],
                hovertemplate=f"%{{customdata[0]}}: %{{customdata[1]}} {unit} (%{{x:.1f}} %)"
                "<extra></extra>",
            )
        )
    top = max(value for _, value, _ in rows) / X * 100
    fig.update_layout(
        height=110 + 26 * len(rows),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": "% de la exergía de los gases de escape", "range": [0, 1.25 * top]},
        yaxis={"categoryorder": "array", "categoryarray": order, "autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        bargap=0.3,
        separators=". ",
    )
    return fig


def configuration_figure(rows: list[ConfigurationRow]) -> go.Figure:
    """Rendimiento de cada configuración, con η_T constante y con la regla de Baumann.

    Un gráfico de puntos (dos marcas por fila, unidas): las diferencias son de
    décimas de punto, y unas barras desde cero las esconderían.
    """
    ok = [r for r in rows if r.result is not None and r.result_baumann is not None]
    labels = [r.label for r in ok]
    eta = [r.result.eta_th * 100 for r in ok]  # type: ignore[union-attr]
    eta_b = [r.result_baumann.eta_th * 100 for r in ok]  # type: ignore[union-attr]
    fig = go.Figure()
    for label, a, b in zip(labels, eta, eta_b, strict=True):
        fig.add_trace(
            go.Scatter(
                x=[a, b],
                y=[label, label],
                mode="lines",
                line={"color": "lightgray", "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for values, name, color, symbol in (
        (eta, "η_T constante", _USEFUL, "circle"),
        (eta_b, "con la regla de Baumann", _DESTROYED, "diamond"),
    ):
        fig.add_trace(
            go.Scatter(
                x=values,
                y=labels,
                mode="markers",
                name=name,
                marker={"color": color, "size": 11, "symbol": symbol},
                hovertemplate=f"%{{y}}: %{{x:.2f}} %<extra>{name}</extra>",
            )
        )
    fig.update_layout(
        height=90 + 38 * len(labels),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="η ciclo combinado [%]",
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
    )
    return fig


# ---------------------------------------------------------------------
# Turbina de gas con etapas y regenerador (Fase 3.7)
# ---------------------------------------------------------------------

#: Estilo de cada proceso del T–s: (leyenda, estilo de la línea). Los tres colores
#: son los primeros de la paleta validada para daltonismo; el resto, grises con trazo.
_GT_STYLES: dict[str, tuple[str, dict[str, object]]] = {
    "isobar": ("", {"color": "#c8c8c8", "width": 1}),
    "heat": ("cámaras", {"color": _DESTROYED, "width": 3}),
    "cooling": ("interenfriamiento", {"color": _USEFUL, "width": 3}),
    "regenerator": ("regenerador", {"color": _LOST, "width": 3}),
    "isentropic": ("isoentrópicas", {"color": "#555555", "width": 1.5, "dash": "dash"}),
    "actual": ("procesos reales", {"color": "#555555", "width": 2, "dash": "dot"}),
    "composition": ("", {"color": "#999999", "width": 1.5, "dash": "dot"}),
    "exhaust": ("escape → ambiente", {"color": "#999999", "width": 2, "dash": "dot"}),
}


def _gt_label_positions(result: GasTurbineResult) -> dict[int, str]:
    """Dónde va el rótulo de cada estado del T–s, según su papel en el ciclo."""
    pos: dict[int, str] = {}
    for c in result.compressors:
        pos[c.inlet] = "bottom center"
        pos[c.outlet_s] = "top left"
        pos[c.outlet] = "top right"
    for t in result.turbines:
        pos[t.inlet] = "top center"
        pos[t.outlet_s] = "bottom left"
        pos[t.outlet] = "bottom right"
    rg = result.regenerator
    if rg is not None:
        pos[rg.air_out] = "middle left"
        pos[rg.gas_out] = "middle right"
    return pos


def gas_turbine_cycle_figure(result: GasTurbineResult, system: UnitSystem) -> go.Figure:
    """La turbina de gas con etapas y regenerador en un T–s (entropías absolutas).

    Las cámaras en naranja, los interenfriadores en azul y el regenerador en
    aguamarina; las isoentrópicas rayadas y las compresiones y expansiones reales
    punteadas (solo de referencia). Las isobaras de referencia (p₁ y la descarga
    del compresor) van en gris, sin leyenda: la leyenda corta deja más lugar al
    diagrama en un celular. Los estados con la numeración de Cengel.
    """

    def T(value_K: float) -> float:
        return convert_from_si(value_K, "temperature", system)

    def s(value: float) -> float:
        return convert_from_si(value, "specific_entropy", system)

    T_unit = unit_label("temperature", system)
    s_unit = unit_label("specific_entropy", system)
    p_unit = unit_label("pressure", system)
    hover = f"s = %{{x:.4g}} {s_unit}<br>T = %{{y:.1f}} {T_unit}"
    fig = go.Figure()
    shown: set[str] = set()
    for line in gas_turbine_lines(result):
        name, style = _GT_STYLES[line.kind]
        p = line.p_Pa[0]
        fig.add_trace(
            go.Scatter(
                x=[s(v) for v in line.s_J_per_kg_K],
                y=[T(v) for v in line.T_K],
                mode="lines",
                name=name,
                legendgroup=line.kind,
                showlegend=bool(name) and line.kind not in shown,
                line=style,
                hovertemplate=hover
                + f"<extra>{line.medium}, p = {_value(p, 'pressure', system, 4)} {p_unit}</extra>",
            )
        )
        shown.add(line.kind)
    positions = _gt_label_positions(result)
    states = list(enumerate(result.states))
    fig.add_trace(
        go.Scatter(
            x=[s(st.s_J_per_kg_K) for _, st in states],
            y=[T(st.T_K) for _, st in states],
            mode="markers+text",
            text=[st.label for _, st in states],
            textposition=[positions.get(i, "top center") for i, _ in states],
            textfont={"size": 13},
            marker={"color": "#333333", "size": 8},
            customdata=[st.description for _, st in states],
            showlegend=False,
            hovertemplate=hover + "<extra>%{text}: %{customdata}</extra>",
        )
    )
    fig.update_layout(
        height=460,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title=f"s [{s_unit}]",
        yaxis_title=f"T [{T_unit}]",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        separators=". ",
    )
    return fig


_IMPROVEMENT_SHORT = {
    "Simple": "simple",
    "Con regenerador": "+ regenerador",
    "Interenfriamiento y recalentamiento": "+ IC + RH",
    "Interenfriamiento, recalentamiento y regenerador": "+ IC + RH + reg.",
}


def _improvement_label(label: str) -> str:
    if label in _IMPROVEMENT_SHORT:
        return _IMPROVEMENT_SHORT[label]
    if label.startswith("Con interenfriamiento"):
        return "+ IC"
    if label.startswith("Con recalentamiento"):
        return "+ RH"
    return label


def improvement_figure(rows: list[ImprovementRow], system: UnitSystem) -> go.Figure:
    """η y w_neto de cada configuración: dos paneles de puntos, la de los datos resaltada.

    Dos magnitudes con distintas unidades van en dos paneles (nunca en un eje
    doble). Las configuraciones que no tienen sentido físico no se dibujan.
    """
    ok = [r for r in rows if r.result is not None]
    labels = [_improvement_label(r.label) for r in ok]
    e_unit = unit_label("specific_enthalpy", system)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("η [%]", f"w_neto [{e_unit}]"),
    )
    etas = [r.result.eta_th * 100 for r in ok]  # type: ignore[union-attr]
    works = [
        convert_from_si(r.result.w_net_J_per_kg, "specific_enthalpy", system)  # type: ignore[union-attr]
        for r in ok
    ]
    current = [r.current for r in ok]
    for col, values, fmt in ((1, etas, ".1f"), (2, works, ".4g")):
        for highlight, name, color, symbol in (
            (False, "otras configuraciones", _USEFUL, "circle"),
            (True, "la de tus datos", _DESTROYED, "diamond"),
        ):
            sel = [k for k, c in enumerate(current) if c == highlight]
            fig.add_trace(
                go.Scatter(
                    x=[values[k] for k in sel],
                    y=[labels[k] for k in sel],
                    mode="markers+text",
                    text=[format(values[k], fmt) for k in sel],
                    textposition="middle right",
                    textfont={"size": 11},
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1 and bool(sel),
                    marker={"color": color, "size": 11, "symbol": symbol},
                    hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>",
                    cliponaxis=False,
                ),
                row=1,
                col=col,
            )
    for col, values in ((1, etas), (2, works)):
        lo, hi = min(values), max(values)
        pad = 0.08 * (hi - lo or 1.0)
        fig.update_xaxes(range=[lo - pad, hi + 3.5 * pad], row=1, col=col)
    fig.update_yaxes(categoryorder="array", categoryarray=labels, autorange="reversed")
    fig.update_layout(
        height=110 + 34 * len(labels),
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.12, "x": 0},
        separators=". ",
    )
    return fig


def gas_turbine_exergy_figure(exergy: GasTurbineExergy, system: UnitSystem) -> go.Figure:
    """Barras horizontales: a dónde va la exergía que entra (% de la del combustible o calor)."""
    x_in = exergy.x_in_J_per_kg
    unit = unit_label("specific_enthalpy", system)
    rows = [(i.name, i.x_J_per_kg, i.kind) for i in exergy.items if i.x_J_per_kg > 0.0]
    order = [name for name, _, _ in rows]
    fig = go.Figure()
    lost = (
        "se va (escape e interenfriadores)"
        if any(name.startswith("interenfriador") for name, _, _ in rows)
        else "se va con el escape"
    )
    for group, color, legend in (
        ("útil", _USEFUL, "trabajo neto"),
        ("destruida", _DESTROYED, "destruida (T₀·S_gen)"),
        ("perdida", _LOST, lost),
    ):
        sel = [r for r in rows if r[2] == group]
        if not sel:
            continue
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[name for name, _, _ in sel],
                x=[value / x_in * 100 for _, value, _ in sel],
                name=legend,
                marker={"color": color, "cornerradius": 4},
                text=[f"{value / x_in * 100:.1f} %" for _, value, _ in sel],
                textposition="outside",
                cliponaxis=False,
                customdata=[
                    [name, _value(value, "specific_enthalpy", system)] for name, value, _ in sel
                ],
                hovertemplate=f"%{{customdata[0]}}: %{{customdata[1]}} {unit} (%{{x:.1f}} %)"
                "<extra></extra>",
            )
        )
    top = max(value for _, value, _ in rows) / x_in * 100
    fig.update_layout(
        height=110 + 26 * len(rows),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": "% de la exergía que entra", "range": [0, 1.25 * top]},
        yaxis={"categoryorder": "array", "categoryarray": order, "autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        bargap=0.3,
        separators=". ",
    )
    return fig
