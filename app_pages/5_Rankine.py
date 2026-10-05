"""Página 5 — Ciclo de Rankine con TESPy (Fases 3.1a y 3.1b).

Ciclo de potencia de vapor de agua: simple, con recalentamiento y con
regeneración (hasta tres calentadores de agua de alimentación, abiertos o
cerrados), ideal o real (rendimientos isoentrópicos de turbina y bomba). El
cálculo lo hace :mod:`core.cycles.rankine` con una red de TESPy; la página
muestra el resultado, las extracciones, la tabla de estados, el ciclo sobre el
diagrama, el procedimiento "como con las tablas" (Cengel cap. 10), el agua de
enfriamiento del condensador, la exportación y barridos para ver cómo cambia
el rendimiento (Cengel §10-4 y §10-6).
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.rankine import (
    RANKINE_EXAMPLE_NOTES,
    RANKINE_EXAMPLES,
    X_TURBINE_MIN,
    CoolingWaterResult,
    FeedwaterHeater,
    RankineInputs,
    RankineResult,
    Reheat,
    SweepParameter,
    condenser_cooling_water,
    default_extraction_values,
    default_heater_pressures,
    default_sweep_values,
    extraction_pressure_sweep,
    rankine_labeled_states,
    rankine_notes,
    rankine_segments,
    rankine_steps,
    rankine_sweep,
    rankine_to_dict,
    solve_rankine,
)
from core.cycles.rankine_layout import PlantLayout, plant_layout
from core.diagrams import (
    DiagramSpec,
    DiagramType,
    ProcessOverlay,
    isentropic_process,
    segments_overlays,
)
from core.export import dict_to_csv
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, convert_to_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.diagrams import DiagramPoint, diagram_type_selector, get_diagram, render_diagram_plotly
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.12.0"
FLUID = "Water"

_EXAMPLES = list(RANKINE_EXAMPLES)
# Opciones fijas (sin número de estado): cambiarlas reiniciaría el widget.
_INLET_SUPERHEATED = "Vapor sobrecalentado (dar T)"
_INLET_SATURATED = "Vapor saturado seco (x = 1)"
_FLOW_MASS = "Caudal másico"
_FLOW_POWER = "Potencia neta"
_KINDS = {"Abierto": "open", "Cerrado": "closed"}
_POSITIONS = {2: ("de baja", "de alta"), 3: ("de baja", "intermedio", "de alta")}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind, str]] = {
    "Presión de la caldera": ("p_boiler", "pressure", "p caldera"),
    "Temperatura de entrada a la turbina": ("T_turbine_in", "temperature", "T entrada"),
    "Presión del condensador": ("p_condenser", "pressure", "p condensador"),
}


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses sin objetos de TESPy)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner="Resolviendo el ciclo con TESPy…")
def _solve_cached(inputs: RankineInputs) -> RankineResult:
    return solve_rankine(inputs)


@st.cache_data(show_spinner="Calculando el barrido…")
def _sweep_cached(
    inputs: RankineInputs, parameter: SweepParameter
) -> list[tuple[float, float, float, float | None]]:
    values = default_sweep_values(inputs, parameter)
    return [
        (p.value_si, p.eta_th, p.w_net_J_per_kg, p.x_turbine_out)
        for p in rankine_sweep(inputs, parameter, values)
    ]


@st.cache_data(show_spinner="Calculando el barrido…")
def _extraction_sweep_cached(
    inputs: RankineInputs, heater: int
) -> list[tuple[float, float, float | None, tuple[float, ...]]]:
    values = default_extraction_values(inputs, heater)
    return [
        (p.value_si, p.eta_th, p.x_turbine_out, p.fractions)
        for p in extraction_pressure_sweep(inputs, heater, values)
    ]


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _example_index() -> int:
    label = st.selectbox(
        "Ejemplo precargado",
        _EXAMPLES,
        key="rk_example",
        help=(
            "Ejemplos de Çengel & Boles, *Termodinámica*, cap. 10. Podés cambiar cualquier "
            "dato y volver a calcular."
        ),
    )
    return _EXAMPLES.index(label)


def _sub(index: int | None) -> str:
    """Subíndice con el número de estado (índice + 1), o nada si no se sabe."""
    return "" if index is None else str(index + 1).translate(_SUBSCRIPTS)


def _peek_pressure(key: str, default_si: float, system: UnitSystem) -> float:
    """Valor (SI) de un ``number_input_si`` que se dibuja más abajo en la página."""
    value = st.session_state.get(f"{key}@{system}")
    return default_si if value is None else convert_to_si(float(value), "pressure", system)


def _heater_defaults(
    base: RankineInputs, p_hi: float, p_lo: float, n: int
) -> tuple[list[FeedwaterHeater], bool]:
    """Calentadores del ejemplo, o ``n`` abiertos que reparten por igual T_sat."""
    if len(base.heaters) == n:
        return list(base.heaters), base.drain_forward
    return [FeedwaterHeater(p) for p in default_heater_pressures(p_hi, p_lo, n)], False


def _provisional_layout(ex: int, base: RankineInputs, system: UnitSystem) -> PlantLayout | None:
    """Numeración de estados con lo que está cargado, para los rótulos.

    Los widgets que la definen (recalentamiento y calentadores) están más
    abajo: se lee su valor de la corrida anterior en ``st.session_state``.
    """
    ss = st.session_state
    p_hi = _peek_pressure(f"rk_p_hi_{ex}", base.p_boiler_Pa, system)
    p_lo = _peek_pressure(f"rk_p_lo_{ex}", base.p_condenser_Pa, system)
    p_rh = None
    if ss.get(f"rk_reheat_{ex}", base.reheat is not None):
        default_rh = base.reheat.p_Pa if base.reheat else (p_hi * p_lo) ** 0.5
        p_rh = _peek_pressure(f"rk_p_rh_{ex}", default_rh, system)
    heaters: list[FeedwaterHeater] = []
    forward = False
    if ss.get(f"rk_regen_{ex}", bool(base.heaters)):
        n = int(ss.get(f"rk_nheat_{ex}", len(base.heaters) or 1))
        defaults, forward_default = _heater_defaults(base, p_hi, p_lo, n)
        for k, d in enumerate(defaults):
            default_kind = "Abierto" if d.kind == "open" else "Cerrado"
            kind = _KINDS[ss.get(f"rk_h{k}_kind_{ex}_{n}", default_kind)]
            heaters.append(
                FeedwaterHeater(_peek_pressure(f"rk_h{k}_p_{ex}_{n}", d.p_Pa, system), kind)
            )
        forward = bool(ss.get(f"rk_fwd_{ex}_{n}", forward_default)) and heaters[-1].kind == "closed"
    try:
        return plant_layout(heaters=heaters, reheat_p_Pa=p_rh, drain_forward=forward)
    except ValueError:  # p. ej. presiones desordenadas: la validación lo va a explicar
        try:
            return plant_layout(reheat_p_Pa=p_rh)
        except ValueError:
            return None


def _read_heaters(
    ex: int,
    base: RankineInputs,
    p_hi: float,
    p_lo: float,
    layout: PlantLayout | None,
) -> tuple[tuple[FeedwaterHeater, ...], bool]:
    """Bloque de regeneración: cantidad, tipo, presión y TTD de cada calentador."""
    if not st.checkbox(
        "Con regeneración (calentadores de agua de alimentación)",
        value=bool(base.heaters),
        key=f"rk_regen_{ex}",
        help=(
            "Cengel §10-6: se extrae vapor de la turbina para precalentar el agua que va a la "
            "caldera; así sube la temperatura media de aporte de calor."
        ),
    ):
        return (), False
    n = st.radio(
        "Cantidad de calentadores",
        (1, 2, 3),
        index=max(len(base.heaters), 1) - 1,
        horizontal=True,
        key=f"rk_nheat_{ex}",
    )
    st.caption(
        "De menor a mayor presión de extracción. **Abierto**: la extracción se mezcla con el "
        "agua y sale líquido saturado (después va una bomba). **Cerrado**: intercambiador; el "
        "agua sale a T_sat − TTD y el drenaje va en cascada al calentador de menor presión "
        "(o al condensador)."
    )
    defaults, forward_default = _heater_defaults(base, p_hi, p_lo, n)
    numbered = layout is not None and len(layout.extraction_states) == n
    heaters: list[FeedwaterHeater] = []
    for k, d in enumerate(defaults):
        with st.container(border=True):
            title = "Calentador" if n == 1 else f"Calentador {k + 1} ({_POSITIONS[n][k]})"
            st.markdown(f"**{title}**")
            kind = _KINDS[
                st.radio(
                    "Tipo",
                    tuple(_KINDS),
                    index=0 if d.kind == "open" else 1,
                    horizontal=True,
                    key=f"rk_h{k}_kind_{ex}_{n}",
                )
            ]
            state = layout.extraction_states[k] if numbered and layout else None
            p_ext = number_input_si(
                label=f"Presión de extracción p{_sub(state)}",
                kind="pressure",
                default_si=d.p_Pa,
                key=f"rk_h{k}_p_{ex}_{n}",
                format="%.5g",
            )
            ttd = 0.0
            if kind == "closed":
                ttd = number_input_si(
                    label="TTD (diferencia terminal)",
                    kind="temperature_difference",
                    default_si=d.ttd_K,
                    key=f"rk_h{k}_ttd_{ex}_{n}",
                    format="%.3g",
                    min_value_si=0.0,
                    help=(
                        "El agua de alimentación sale a T_sat(p_ext) − TTD. 0 = calentador "
                        "ideal (Cengel)."
                    ),
                )
            heaters.append(FeedwaterHeater(p_ext, kind, ttd))
    forward = False
    if heaters[-1].kind == "closed":
        forward = st.checkbox(
            "Bombear hacia adelante el drenaje del cerrado de mayor presión",
            value=forward_default,
            key=f"rk_fwd_{ex}_{n}",
            help=(
                "Como en Cengel 10-6: una bomba lleva el drenaje a la línea de agua de "
                "alimentación y se mezcla antes de la caldera. Si no, va en cascada hacia atrás."
            ),
        )
    return tuple(heaters), forward


def _read_cooling(ex: int) -> tuple[float, float] | None:
    """T de entrada y salida del agua de enfriamiento (SI), si se pide."""
    if not st.checkbox("Calcular el agua de enfriamiento del condensador", key=f"rk_cw_{ex}"):
        return None
    left, right = st.columns(2)
    with left:
        T_in = number_input_si(
            label="Agua: entrada",
            kind="temperature",
            default_si=293.15,
            key=f"rk_cw_in_{ex}",
            format="%.4g",
        )
    with right:
        T_out = number_input_si(
            label="Agua: salida",
            kind="temperature",
            default_si=303.15,
            key=f"rk_cw_out_{ex}",
            format="%.4g",
        )
    return T_in, T_out


def _read_inputs(ex: int, system: UnitSystem) -> tuple[RankineInputs, tuple[float, float] | None]:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    base = RANKINE_EXAMPLES[_EXAMPLES[ex]]
    layout = _provisional_layout(ex, base, system)
    t_in = layout.turbine_inlet if layout else None
    st.markdown("**Caldera y turbina**")
    p_hi = number_input_si(
        label=f"Presión de la caldera p{_sub(t_in)}",
        kind="pressure",
        default_si=base.p_boiler_Pa,
        key=f"rk_p_hi_{ex}",
        format="%.5g",
    )
    inlet = st.radio(
        "Entrada a la turbina",
        (_INLET_SUPERHEATED, _INLET_SATURATED),
        index=0 if base.T_turbine_in_K is not None else 1,
        horizontal=True,
        key=f"rk_inlet_{ex}",
    )
    T_in: float | None = None
    if inlet == _INLET_SUPERHEATED:
        T_in = number_input_si(
            label=f"Temperatura de entrada a la turbina T{_sub(t_in)}",
            kind="temperature",
            default_si=base.T_turbine_in_K or 873.15,
            key=f"rk_T_in_{ex}",
            format="%.5g",
        )
    p_lo = number_input_si(
        label=f"Presión del condensador p{_sub(layout.exhaust if layout else None)}",
        kind="pressure",
        default_si=base.p_condenser_Pa,
        key=f"rk_p_lo_{ex}",
        format="%.5g",
    )
    left, right = st.columns(2)
    eta_t = left.number_input(
        "η_T de la turbina",
        min_value=0.3,
        max_value=1.0,
        value=float(base.eta_turbine),
        step=0.01,
        key=f"rk_eta_t_{ex}",
        help=(
            "1 = turbina ideal (isoentrópica). Vademecum §10.4. Con extracciones, se mide "
            "desde la entrada de cada turbina."
        ),
    )
    eta_p = right.number_input(
        "η_B de la bomba",
        min_value=0.3,
        max_value=1.0,
        value=float(base.eta_pump),
        step=0.01,
        key=f"rk_eta_p_{ex}",
        help="1 = bomba ideal (isoentrópica). Vademecum §10.4.",
    )

    reheat: Reheat | None = None
    if st.checkbox("Con recalentamiento", value=base.reheat is not None, key=f"rk_reheat_{ex}"):
        default_rh = base.reheat or Reheat((p_hi * p_lo) ** 0.5, T_in or 873.15)
        rh_in = layout.reheat_in if layout else None
        rh_out = layout.reheat_out if layout else None
        p_rh = number_input_si(
            label=f"Presión de recalentamiento p{_sub(rh_in)} = p{_sub(rh_out)}",
            kind="pressure",
            default_si=default_rh.p_Pa,
            key=f"rk_p_rh_{ex}",
            format="%.5g",
        )
        T_rh = number_input_si(
            label=f"Temperatura de recalentamiento T{_sub(rh_out)}",
            kind="temperature",
            default_si=default_rh.T_K,
            key=f"rk_T_rh_{ex}",
            format="%.5g",
        )
        reheat = Reheat(p_rh, T_rh)

    st.markdown("**Regeneración**")
    heaters, forward = _read_heaters(ex, base, p_hi, p_lo, layout)

    st.markdown("**Tamaño de la planta**")
    flow = st.radio("Dato", (_FLOW_MASS, _FLOW_POWER), horizontal=True, key=f"rk_flow_{ex}")
    if flow == _FLOW_MASS:
        m_dot = number_input_si(
            label="Caudal másico ṁ (por la caldera)",
            kind="mass_flow",
            default_si=base.m_dot_kg_s or 1.0,
            key=f"rk_m_{ex}",
            format="%.5g",
        )
        flows: dict[str, float | None] = {"m_dot_kg_s": m_dot}
    else:
        power = number_input_si(
            label="Potencia neta Ẇ_neto",
            kind="power",
            default_si=100.0e6,
            key=f"rk_W_{ex}",
            format="%.5g",
        )
        flows = {"m_dot_kg_s": None, "W_net_W": power}
    cooling = _read_cooling(ex)
    inputs = RankineInputs(
        p_hi,
        T_in,
        p_lo,
        eta_t,
        eta_p,
        reheat,
        heaters=heaters,
        drain_forward=forward,
        **flows,  # type: ignore[arg-type]
    )
    return inputs, cooling


def _render_error(exc: ValueError) -> None:
    """Mensaje para el alumno arriba; el detalle técnico, aparte y chico."""
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico: {detail.strip()}")


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


def _number(value: float) -> str:
    """Número para mostrar: sin notación exponencial en los valores grandes."""
    if abs(value) >= 1.0e4:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, 5)


def _energy(value_si: float, system: UnitSystem) -> str:
    return _number(convert_from_si(value_si, "specific_enthalpy", system))


def _render_metrics(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None
) -> None:
    e_unit = unit_label("specific_enthalpy", system)
    x = result.x_turbine_out
    row1 = st.columns(3)
    row1[0].metric("η térmico [%]", f"{result.eta_th * 100:.2f}")
    row1[1].metric(f"w neto [{e_unit}]", _energy(result.w_net_J_per_kg, system))
    row1[2].metric("Título a la salida", "sobrecalentado" if x is None else f"{x:.3f}")
    row2 = st.columns(3)
    row2[0].metric("η Carnot [%]", f"{result.eta_carnot * 100:.2f}")
    row2[1].metric("BWR = w_B/w_T [%]", f"{result.back_work_ratio * 100:.2f}")
    power_unit = unit_label("power", system)
    row2[2].metric(
        f"Ẇ neto [{power_unit}]",
        _number(convert_from_si(result.W_net_W, "power", system)),
    )
    if cooling is not None:
        st.metric(
            f"ṁ agua de enfriamiento [{unit_label('mass_flow', system)}]",
            _number(convert_from_si(cooling.m_dot_kg_s, "mass_flow", system)),
        )
    st.caption(
        f"Caudal: {_number(convert_from_si(result.m_dot_kg_s, 'mass_flow', system))} "
        f"{unit_label('mass_flow', system)} · q_H = {_energy(result.q_in_J_per_kg, system)} "
        f"{e_unit} · q_C = {_energy(result.q_out_J_per_kg, system)} {e_unit}. "
        "BWR: relación de trabajo de retroceso."
        + (
            " Las energías son por kg de vapor que pasa por la caldera."
            if result.has_heaters
            else ""
        )
    )
    for note in rankine_notes(result):
        st.info(note)


def _render_extractions(result: RankineResult, system: UnitSystem) -> None:
    """Fracción y caudal de cada extracción (Cengel §10-6)."""
    if not result.has_heaters or result.layout is None:
        return
    p_unit = unit_label("pressure", system)
    m_unit = unit_label("mass_flow", system)
    rows = []
    for k, (name, heater, y) in enumerate(
        zip(
            result.layout.heater_names,
            result.inputs.heaters,
            result.extraction_fractions,
            strict=True,
        )
    ):
        rows.append(
            {
                "Calentador": name,
                "Estado n.º": result.layout.extraction_states[k] + 1,
                f"p [{p_unit}]": format_value(convert_from_si(heater.p_Pa, "pressure", system)),
                "y = ṁext/ṁ": f"{y:.4f}",
                f"ṁext [{m_unit}]": _number(
                    convert_from_si(y * result.m_dot_kg_s, "mass_flow", system)
                ),
            }
        )
    st.markdown("#### Extracciones")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_states_table(result: RankineResult, system: UnitSystem) -> None:
    frame = pd.DataFrame(states_table(rankine_labeled_states(result), system)).drop(
        columns="Fluido"
    )
    for column in frame.columns:
        if column not in ("Estado", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _render_tespy(result: RankineResult, system: UnitSystem) -> None:
    with st.expander("⚙️ Cómo lo resuelve TESPy", expanded=False):
        st.markdown(
            "TESPy arma la planta como una **red**: cada componente (bombas, caldera, turbinas, "
            "calentadores, condensador) aporta sus ecuaciones —balance de masa y energía, "
            "rendimiento isoentrópico, pérdida de carga— y cada conexión lleva caudal, presión "
            "y entalpía. Las extracciones son divisores de caudal (*Splitter*); los "
            "calentadores abiertos, mezcladores (*Merge*) con líquido saturado a la salida, y los "
            "cerrados, intercambiadores (*Condenser*). Con los datos resuelve el sistema con "
            "Newton. Estos son sus balances por kilogramo de vapor que pasa por la caldera "
            "(positivo: entra al fluido; negativo: sale); ya incluyen la fracción de caudal de "
            "cada componente y coinciden con los del procedimiento:"
        )
        unit = unit_label("specific_enthalpy", system)
        frame = pd.DataFrame(
            [
                {"Componente": name, f"Energía por kg [{unit}]": _energy(value, system)}
                for name, value in result.tespy_balances
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        if result.tespy_heaters:
            st.markdown(
                "Calor que cada calentador cerrado pasa de la extracción al agua de "
                "alimentación (es calor **interno** del ciclo: no entra ni sale):"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Calentador": name, f"Calor por kg [{unit}]": _energy(value, system)}
                        for name, value in result.tespy_heaters
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


def _cycle_diagram(result: RankineResult, system: UnitSystem) -> None:
    with st.expander("📈 Diagrama del ciclo", expanded=True):
        diagram_type: DiagramType = diagram_type_selector(key="rk_diagram_type", default="Ts")
        labeled = rankine_labeled_states(result)
        states = [state.to_state_point() for _, state in labeled]
        points = [
            DiagramPoint(state=state, label=label.split()[0], color="#d62728")
            for (label, _), state in zip(labeled, states, strict=True)
        ]
        try:
            diagram = get_diagram(FLUID, system)
            spec = DiagramSpec(fluid=FLUID, system=system)
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
                fluid=FLUID,
                diagram_type=diagram_type,
                system=system,
                points=points,
                overlays=overlays,
                chart_key="rk_diagram_chart",
                point_legend={"#d62728": "estados", "#2ca02c": "estados isoentrópicos (ks)"},
            )
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
        st.caption(
            "Caldera, condensador y calentadores se dibujan sobre su isobara real (la "
            "extracción condensa a la presión del calentador). En el ciclo ideal, bombas y "
            "turbina son isoentrópicas (verticales en T–s); en el real, la recta punteada que "
            "une la entrada y la salida es solo una referencia, y la verde es la expansión "
            "isoentrópica con la que se compara η_T. Las válvulas de los drenajes también son "
            "rectas de referencia (h constante)."
        )


def _render_procedure(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None
) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cómo se resuelve con las tablas de vapor, estado por estado (Cengel §10-2"
            + (" y §10-6" if result.has_heaters else "")
            + "). Los valores salen de la ecuación de estado (IAPWS-95), así que pueden "
            "diferir en el último decimal de los de una tabla impresa."
        )
        for i, step in enumerate(rankine_steps(result, system, cooling=cooling), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_export(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None
) -> None:
    st.markdown("#### 💾 Exportar")
    data = rankine_to_dict(result, system, cooling=cooling)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="rankine.csv",
        mime="text/csv",
        key="rk_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="rankine.json",
        mime="application/json",
        key="rk_download_json",
    )


def _line_chart(xs: list[float], ys: list[float], *, title: str, x_title: str, y_title: str):
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        height=270,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def _render_sweeps(inputs: RankineInputs, result: RankineResult, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo aumentar el rendimiento?", expanded=False):
        st.markdown(
            "Cengel §10-4: el rendimiento sube si se **baja la presión del condensador** (baja "
            "T_C), se **sobrecalienta más** o se **sube la presión de la caldera** (sube la "
            "temperatura media de aporte de calor T̄_H). Lo último baja el título a la salida de "
            "la turbina, y por eso se recalienta. Con **regeneración** (§10-6) cada extracción "
            "tiene una presión óptima. Elegí qué variar (el resto queda como en tu ciclo):"
        )
        names = result.layout.heater_names if result.layout else ()
        options = list(_SWEEPS) + [f"Presión de extracción del {name}" for name in names]
        choice = st.selectbox("Variable", options, key="rk_sweep_param")
        key: tuple = (
            ("extraction", options.index(choice) - len(_SWEEPS))
            if choice not in _SWEEPS
            else (_SWEEPS[choice][0],)
        )
        if st.button("Calcular barrido", key="rk_sweep_btn"):
            st.session_state["rk_sweep"] = (inputs, key)
        if st.session_state.get("rk_sweep") != (inputs, key):
            st.caption("Tocá «Calcular barrido» (resuelve el ciclo varias veces).")
            return
        if key[0] == "extraction":
            heater = key[1]
            extraction = _extraction_sweep_cached(inputs, heater)
            if not extraction:
                st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
                return
            unit = unit_label("pressure", system)
            xs = [convert_from_si(p[0], "pressure", system) for p in extraction]
            x_title = f"p de extracción [{unit}]"
            st.plotly_chart(
                _line_chart(
                    xs,
                    [p[1] * 100 for p in extraction],
                    title="Rendimiento térmico",
                    x_title=x_title,
                    y_title="η [%]",
                ),
                width="stretch",
                key="rk_sweep_eta",
            )
            st.plotly_chart(
                _line_chart(
                    xs,
                    [p[3][heater] for p in extraction],
                    title="Fracción de extracción",
                    x_title=x_title,
                    y_title="y [-]",
                ),
                width="stretch",
                key="rk_sweep_y",
            )
            st.caption(
                "Con poca presión la extracción calienta poco el agua; con mucha, se lleva vapor "
                "que todavía podía dar trabajo: entre las dos hay un óptimo."
            )
            return
        parameter, kind, axis = _SWEEPS[choice]
        points = _sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        unit = unit_label(kind, system)
        xs = [convert_from_si(p[0], kind, system) for p in points]
        x_title = f"{axis} [{unit}]"
        st.plotly_chart(
            _line_chart(
                xs,
                [p[1] * 100 for p in points],
                title="Rendimiento térmico",
                x_title=x_title,
                y_title="η [%]",
            ),
            width="stretch",
            key="rk_sweep_eta",
        )
        quality = [p[3] for p in points]
        if any(q is not None for q in quality):
            fig_x = _line_chart(
                xs,
                [q if q is not None else 1.0 for q in quality],
                title="Título a la salida de la turbina",
                x_title=x_title,
                y_title="x [-]",
            )
            fig_x.add_hline(
                y=X_TURBINE_MIN,
                line_dash="dot",
                annotation_text=f"x = {X_TURBINE_MIN} (límite práctico)",
            )
            st.plotly_chart(fig_x, width="stretch", key="rk_sweep_x")
            st.caption("Si el vapor sale sobrecalentado, el gráfico lo muestra como x = 1.")


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) no "
            "tiene una sección propia del ciclo de Rankine; las relaciones que se usan son las de "
            "**§3.3 *Sistema abierto, régimen permanente***, **§9.2 *Rendimientos térmicos***, "
            "§10.1 *Entropía*, **§10.4 *Rendimientos isoentrópicos***, §12 *Vapor húmedo* y §13 "
            "*Líquidos (incompresibles)*. En el libro: Çengel & Boles, cap. 10 (§10-2 a §10-7)."
        )
        st.markdown(
            "**Balance de cada componente** (§3.3, sin variaciones de energía cinética ni "
            "potencial; numeración de Cengel):"
        )
        st.latex(r"w_B = h_2 - h_1 \qquad q_H = h_3 - h_2")
        st.latex(r"w_T = h_3 - h_4 \qquad q_C = h_4 - h_1")
        st.markdown(
            "**Bomba** (líquido incompresible, §13) y **rendimientos isoentrópicos** (§10.4):"
        )
        st.latex(r"w_{B,s} \approx v_1\,(p_2 - p_1)")
        st.latex(r"\eta_B = \frac{h_{2s} - h_1}{h_2 - h_1}")
        st.latex(r"\eta_T = \frac{h_3 - h_4}{h_3 - h_{4s}}")
        st.markdown("**Rendimiento del ciclo** (§9.2), con temperaturas absolutas en Carnot:")
        st.latex(r"\eta_{\mathrm{MT}} = \frac{W}{Q_H} = 1 - \frac{Q_C}{Q_H}")
        st.latex(r"\eta_{\mathrm{Carnot}} = 1 - \frac{T_C}{T_H}")
        st.markdown(
            "**Temperatura media de aporte de calor** (de la definición de entropía, §10.1). En "
            "el ciclo ideal sin regeneración y con vapor húmedo a la salida de la turbina, el "
            "calor se cede a T_C constante y η = 1 − T_C/T̄_H:"
        )
        st.latex(r"\bar{T}_H = \frac{q_H}{s_3 - s_2}")
        st.markdown(
            "**Con recalentamiento** (estados 1 a 6): el calor entra en la caldera y en el "
            "recalentador, y el trabajo sale de las dos turbinas:"
        )
        st.latex(r"q_H = (h_3 - h_2) + (h_5 - h_4)")
        st.latex(r"w_T = (h_3 - h_4) + (h_5 - h_6)")
        st.markdown(
            "**Con regeneración** (Cengel §10-6) se extrae una fracción y del vapor para "
            "precalentar el agua de alimentación. En el calentador **abierto** se mezclan y sale "
            "líquido saturado; en el **cerrado** la extracción condensa (sale líquido saturado, el "
            "drenaje) y el agua sale a T_sat − TTD:"
        )
        st.latex(r"y\,h_{\mathrm{ext}} + (1 - y)\,h_{\mathrm{ent}} = h_{\mathrm{sal}}")
        st.latex(r"y\,(h_{\mathrm{ext}} - h_{\mathrm{dren}}) = h_{\mathrm{sal}} - h_{\mathrm{ent}}")
        st.markdown(
            "Todo se cuenta por kilogramo de vapor que pasa por la caldera: cada tramo pesa "
            "según su caudal (en el ejemplo 10-5, por la turbina después de la extracción pasa "
            "1 − y):"
        )
        st.latex(r"w_T = (h_5 - h_6) + (1 - y)\,(h_6 - h_7)")
        st.markdown(
            "La regeneración sube T̄_H (el agua llega más caliente a la caldera), pero η queda "
            "por debajo de 1 − T_C/T̄_H: en los calentadores se mezclan o intercambian calor "
            "corrientes a distinta temperatura, y eso genera entropía. Con turbina real, η_T se "
            "mide desde la entrada de cada turbina: cada extracción y la salida caen sobre la "
            "misma línea de expansión, h_k = h_e − η_T·(h_e − h_ks)."
        )
        st.markdown(
            "La **relación de trabajo de retroceso** BWR = w_B / w_T es chica en el Rankine "
            "(el líquido se bombea con poco trabajo), a diferencia del ciclo Brayton."
        )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Rankine", page_icon="♨️", layout="centered")

st.subheader(SUBJECT)
st.title("♨️ Ciclo de Rankine")
st.markdown(
    "Ciclo de potencia de vapor: bomba, caldera, turbina y condensador. Simple, con "
    "recalentamiento o con regeneración (calentadores de agua de alimentación abiertos o "
    "cerrados), ideal o real. Lo resuelve [TESPy](https://tespy.readthedocs.io) como una red de "
    "componentes, y el procedimiento muestra cómo se haría con las tablas."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Rankine")
render_units_selector()
system = get_current_system()

ex = _example_index()
inputs, cooling_T = _read_inputs(ex, system)

# Se calcula al elegir un ejemplo y cada vez que se toca el botón.
if st.session_state.get("rk_example_prev") != ex:
    st.session_state["rk_example_prev"] = ex
    st.session_state["rk_inputs"] = inputs
if st.button("Calcular ciclo", key="rk_btn", type="primary"):
    st.session_state["rk_inputs"] = inputs

computed: RankineInputs | None = st.session_state.get("rk_inputs")
if computed is not None:
    if computed != inputs:
        st.caption("Cambiaste datos: tocá «Calcular ciclo» para actualizar el resultado.")
    try:
        result = _solve_cached(computed)
    except ValueError as exc:
        _render_error(exc)
    else:
        cooling: CoolingWaterResult | None = None
        cooling_error = None
        if cooling_T is not None:
            try:
                cooling = condenser_cooling_water(result, *cooling_T)
            except ValueError as exc:
                cooling_error = str(exc)
        st.markdown("### Resultado")
        note = RANKINE_EXAMPLE_NOTES.get(_EXAMPLES[ex])
        if note:
            st.caption(f"📘 Sobre este ejemplo: {note}")
        _render_metrics(result, system, cooling)
        if cooling_error:
            st.warning(cooling_error)
        _render_extractions(result, system)
        st.markdown("#### Estados")
        _render_states_table(result, system)
        _cycle_diagram(result, system)
        _render_procedure(result, system, cooling)
        _render_tespy(result, system)
        _render_export(result, system, cooling)
        _render_sweeps(computed, result, system)
