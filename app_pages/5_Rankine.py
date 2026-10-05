"""Página 5 — Ciclo de Rankine con TESPy (Fase 3.1a).

Ciclo de potencia de vapor de agua, simple o con recalentamiento, ideal o
real (rendimientos isoentrópicos de turbina y bomba). El cálculo lo hace
:mod:`core.cycles.rankine` con una red de TESPy; la página muestra el
resultado, la tabla de estados, el ciclo sobre el diagrama, el
procedimiento "como con las tablas" (Cengel cap. 10), la exportación y
barridos para ver cómo cambia el rendimiento (Cengel §10-6).
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.rankine import (
    RANKINE_EXAMPLES,
    X_TURBINE_MIN,
    RankineInputs,
    RankineResult,
    Reheat,
    SweepParameter,
    default_sweep_values,
    rankine_isentropic_states,
    rankine_labeled_states,
    rankine_notes,
    rankine_steps,
    rankine_sweep,
    rankine_to_dict,
    solve_rankine,
)
from core.diagrams import (
    DiagramSpec,
    DiagramType,
    ProcessOverlay,
    cycle_overlays,
    isentropic_process,
)
from core.export import dict_to_csv
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.diagrams import DiagramPoint, diagram_type_selector, get_diagram, render_diagram_plotly
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.11.0"
FLUID = "Water"

_EXAMPLES = list(RANKINE_EXAMPLES)
_INLET_SUPERHEATED = "Vapor sobrecalentado (dar T₃)"
_INLET_SATURATED = "Vapor saturado seco (x₃ = 1)"
_FLOW_MASS = "Caudal másico"
_FLOW_POWER = "Potencia neta"

_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind, str]] = {
    "Presión de la caldera": ("p_boiler", "pressure", "p caldera"),
    "Temperatura de entrada a la turbina": ("T_turbine_in", "temperature", "T₃"),
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


def _read_inputs(ex: int) -> RankineInputs:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    base = RANKINE_EXAMPLES[_EXAMPLES[ex]]
    st.markdown("**Caldera y turbina**")
    p_hi = number_input_si(
        label="Presión de la caldera p₃",
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
            label="Temperatura de entrada a la turbina T₃",
            kind="temperature",
            default_si=base.T_turbine_in_K or 873.15,
            key=f"rk_T_in_{ex}",
            format="%.5g",
        )
    # Con recalentamiento la salida de la turbina de baja es el estado 6 (Cengel):
    # la casilla está más abajo, así que se lee su estado de la corrida anterior.
    reheat_on = bool(st.session_state.get(f"rk_reheat_{ex}", base.reheat is not None))
    p_lo = number_input_si(
        label=f"Presión del condensador {'p₆' if reheat_on else 'p₄'}",
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
        help="1 = turbina ideal (isoentrópica). Vademecum §10.4.",
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
        p_rh = number_input_si(
            label="Presión de recalentamiento p₄ = p₅",
            kind="pressure",
            default_si=default_rh.p_Pa,
            key=f"rk_p_rh_{ex}",
            format="%.5g",
        )
        T_rh = number_input_si(
            label="Temperatura de recalentamiento T₅",
            kind="temperature",
            default_si=default_rh.T_K,
            key=f"rk_T_rh_{ex}",
            format="%.5g",
        )
        reheat = Reheat(p_rh, T_rh)

    st.markdown("**Tamaño de la planta**")
    flow = st.radio("Dato", (_FLOW_MASS, _FLOW_POWER), horizontal=True, key=f"rk_flow_{ex}")
    if flow == _FLOW_MASS:
        m_dot = number_input_si(
            label="Caudal másico ṁ",
            kind="mass_flow",
            default_si=base.m_dot_kg_s or 1.0,
            key=f"rk_m_{ex}",
            format="%.5g",
        )
        return RankineInputs(p_hi, T_in, p_lo, eta_t, eta_p, reheat, m_dot_kg_s=m_dot)
    power = number_input_si(
        label="Potencia neta Ẇ_neto",
        kind="power",
        default_si=100.0e6,
        key=f"rk_W_{ex}",
        format="%.5g",
    )
    return RankineInputs(p_hi, T_in, p_lo, eta_t, eta_p, reheat, m_dot_kg_s=None, W_net_W=power)


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


def _render_metrics(result: RankineResult, system: UnitSystem) -> None:
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
    st.caption(
        f"Caudal: {_number(convert_from_si(result.m_dot_kg_s, 'mass_flow', system))} "
        f"{unit_label('mass_flow', system)} · q_H = {_energy(result.q_in_J_per_kg, system)} "
        f"{e_unit} · q_C = {_energy(result.q_out_J_per_kg, system)} {e_unit}. "
        "BWR: relación de trabajo de retroceso."
    )
    for note in rankine_notes(result):
        st.info(note)


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
            "TESPy arma la planta como una **red**: cada componente (bomba, caldera, turbina, "
            "condensador) aporta sus ecuaciones —balance de masa y energía, rendimiento "
            "isoentrópico, pérdida de carga— y cada conexión lleva caudal, presión y entalpía. "
            "Con los datos (p y T a la entrada de la turbina, p del condensador, x = 0 a la "
            "salida del condensador y los rendimientos) resuelve el sistema con Newton. Estos "
            "son sus balances por kilogramo (positivo: entra al fluido; negativo: sale), que "
            "coinciden con los del procedimiento:"
        )
        unit = unit_label("specific_enthalpy", system)
        frame = pd.DataFrame(
            [
                {"Componente": name, f"Energía por kg [{unit}]": _energy(value, system)}
                for name, value in result.tespy_balances
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")


def _cycle_diagram(result: RankineResult, system: UnitSystem) -> None:
    with st.expander("📈 Diagrama del ciclo", expanded=True):
        diagram_type: DiagramType = diagram_type_selector(key="rk_diagram_type", default="Ts")
        labeled = rankine_labeled_states(result)
        points = [
            DiagramPoint(state=state.to_state_point(), label=label.split()[0], color="#d62728")
            for label, state in labeled
        ]
        try:
            diagram = get_diagram(FLUID, system)
            spec = DiagramSpec(fluid=FLUID, system=system)
            overlays: list[ProcessOverlay] = cycle_overlays(
                diagram, spec, [state.to_state_point() for _, state in labeled], close=True
            )
            if result.inputs.eta_turbine < 1.0:
                for (i, _o), (label, state_s) in zip(
                    result.turbine_pairs, rankine_isentropic_states(result)[1:], strict=True
                ):
                    start = result.states[i]
                    overlays.append(
                        ProcessOverlay(
                            name=f"{i + 1} → {label} (isoentrópica de referencia)",
                            color="#2ca02c",
                            dash="dash",
                            coords_si=isentropic_process(
                                diagram,
                                spec,
                                s_J_per_kg_K=start.s_J_per_kg_K,
                                p_start_Pa=start.P_Pa,
                                p_end_Pa=state_s.P_Pa,
                            ),
                        )
                    )
                    points.append(
                        DiagramPoint(state=state_s.to_state_point(), label=label, color="#2ca02c")
                    )
            render_diagram_plotly(
                fluid=FLUID,
                diagram_type=diagram_type,
                system=system,
                points=points,
                overlays=overlays,
                chart_key="rk_diagram_chart",
            )
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
        st.caption(
            "Caldera y condensador se dibujan sobre su isobara real. En el ciclo ideal, bomba y "
            "turbina son isoentrópicas (verticales en T–s); en el real, la recta punteada que "
            "une la entrada y la salida es solo una referencia, y la verde es la expansión "
            "isoentrópica con la que se compara η_T."
        )


def _render_procedure(result: RankineResult, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cómo se resuelve con las tablas de vapor, estado por estado (Cengel §10-2). Los "
            "valores salen de la ecuación de estado (IAPWS-95), así que pueden diferir en el "
            "último decimal de los de una tabla impresa."
        )
        for i, step in enumerate(rankine_steps(result, system), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_export(result: RankineResult, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    data = rankine_to_dict(result, system)
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


def _render_sweeps(inputs: RankineInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo aumentar el rendimiento?", expanded=False):
        st.markdown(
            "Cengel §10-6: el rendimiento sube si se **baja la presión del condensador** (baja "
            "T_C), se **sobrecalienta más** o se **sube la presión de la caldera** (sube la "
            "temperatura media de aporte de calor T̄_H). Lo último baja el título a la salida de "
            "la turbina, y por eso se recalienta. Elegí qué variar (el resto queda como en tu "
            "ciclo):"
        )
        choice = st.selectbox("Variable", list(_SWEEPS), key="rk_sweep_param")
        parameter, kind, axis = _SWEEPS[choice]
        if st.button("Calcular barrido", key="rk_sweep_btn"):
            st.session_state["rk_sweep"] = (inputs, parameter)
        stored = st.session_state.get("rk_sweep")
        if stored != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve el ciclo unas siete veces).")
            return
        points = _sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        unit = unit_label(kind, system)
        xs = [convert_from_si(p[0], kind, system) for p in points]
        x_title = f"{axis} [{unit}]"
        eta = go.Figure(go.Scatter(x=xs, y=[p[1] * 100 for p in points], mode="lines+markers"))
        eta.update_layout(
            height=280,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            title="Rendimiento térmico",
            xaxis_title=x_title,
            yaxis_title="η [%]",
        )
        st.plotly_chart(eta, width="stretch", key="rk_sweep_eta")
        quality = [p[3] for p in points]
        if any(q is not None for q in quality):
            ys = [q if q is not None else 1.0 for q in quality]
            fig_x = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
            fig_x.add_hline(
                y=X_TURBINE_MIN,
                line_dash="dot",
                annotation_text=f"x = {X_TURBINE_MIN} (límite práctico)",
            )
            fig_x.update_layout(
                height=260,
                margin={"l": 10, "r": 10, "t": 30, "b": 10},
                title="Título a la salida de la turbina",
                xaxis_title=x_title,
                yaxis_title="x [-]",
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
            "el ciclo ideal con vapor húmedo a la salida de la turbina, el calor se cede a T_C "
            "constante y η = 1 − T_C/T̄_H:"
        )
        st.latex(r"\bar{T}_H = \frac{q_H}{s_3 - s_2}")
        st.markdown(
            "**Con recalentamiento** (estados 1 a 6): el calor entra en la caldera y en el "
            "recalentador, y el trabajo sale de las dos turbinas:"
        )
        st.latex(r"q_H = (h_3 - h_2) + (h_5 - h_4)")
        st.latex(r"w_T = (h_3 - h_4) + (h_5 - h_6)")
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
    "Ciclo de potencia de vapor: bomba, caldera, turbina y condensador. Simple o con "
    "recalentamiento, ideal o real (con los rendimientos isoentrópicos de turbina y bomba). "
    "Lo resuelve [TESPy](https://tespy.readthedocs.io) como una red de componentes, y el "
    "procedimiento muestra cómo se haría con las tablas."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Rankine")
render_units_selector()
system = get_current_system()

ex = _example_index()
inputs = _read_inputs(ex)

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
        st.markdown("### Resultado")
        _render_metrics(result, system)
        st.markdown("#### Estados")
        _render_states_table(result, system)
        _cycle_diagram(result, system)
        _render_procedure(result, system)
        _render_tespy(result, system)
        _render_export(result, system)
        _render_sweeps(computed, system)
