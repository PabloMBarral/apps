"""Página 3 — Rendimientos isoentrópicos.

Cuatro pestañas: turbina, compresor, bomba y compresor multietapa
(con intercooler opcional). En las tres primeras se puede operar en modo
**directo** (dado η_s y P_out → estado real) o **inverso** (dados los
estados de entrada y salida → η_s). El multietapa va solo en modo
directo y trae la comparación contra una sola etapa.

- Los valores iniciales dependen del fluido (siempre son calculables:
  :func:`core.isentropic.suggested_device_inputs`).
- Los estados se validan con :func:`core.fluids.fluid_state_from_pair`
  (rango de la ecuación de estado, mensajes en castellano).
- Tabla de estados, procedimiento LaTeX (Fase 1.5b: en el sistema de
  unidades activo), diagrama del proceso y exportación CSV/JSON.
- Cada resultado se guarda junto con su fluido: si se cambia el fluido,
  el resultado viejo deja de mostrarse.

Solo la bomba ofrece comparación contra el modelo de líquido
incompresible (``w_p ≈ v₁·Δp / η_s``).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.diagrams import (
    DiagramSpec,
    ProcessOverlay,
    isentropic_process,
    isobaric_process,
    linear_segment_overlay,
)
from core.fluids import (
    FLUID_NAMES_ES,
    SUPPORTED_FLUIDS,
    FluidState,
    StatePoint,
    fluid_state_from_pair,
)
from core.isentropic import (
    DeviceDefaults,
    IsentropicResult,
    IsentropicSteps,
    PolytropicResult,
    compressor_direct,
    compressor_inverse,
    compressor_multistage,
    isentropic_labeled_states,
    isentropic_steps,
    isentropic_to_dict,
    multistage_labeled_states,
    multistage_steps,
    multistage_to_dict,
    pump_direct,
    pump_incompressible_comparison,
    pump_inverse,
    suggested_device_inputs,
    summary_csv,
    turbine_direct,
    turbine_inverse,
)
from core.state_report import format_value, states_table
from core.units_system import UnitSystem
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.diagrams import (
    DiagramPoint,
    diagram_type_selector,
    get_diagram,
    render_diagram_plotly,
)
from ui.units_ui import (
    get_current_system,
    number_input_si,
    quantity_label,
    render_units_selector,
)

PAGE_VERSION = "0.10.0"

# Códigos de par independiente reconocidos por core.fluids, ordenados por uso didáctico.
_PAIR_LABELS_TO_CODE: dict[str, str] = {
    "T y P": "TP",
    "P y h": "PH",
    "P y x (saturado)": "PX",
    "T y x (saturado)": "TX",
    "P y s": "PS",
    "T y s": "TS",
    "h y s": "HS",
}
_PAIR_CODE_TO_LABEL: dict[str, str] = {v: k for k, v in _PAIR_LABELS_TO_CODE.items()}

# Símbolo → (kwarg de fluid_state_from_pair, magnitud, default en SI).
_SYMBOL_SPEC: dict[str, tuple[str, str, float]] = {
    "T": ("t", "temperature", 298.15),  # 25 °C
    "P": ("p", "pressure", 1.0e5),  # 1 bar
    "H": ("h", "specific_enthalpy", 2.5e6),  # 2500 kJ/kg
    "S": ("s", "specific_entropy", 5000.0),  # 5 kJ/(kg·K)
    "X": ("x", "dimensionless", 0.0),
}
_INPUT_FORMAT: dict[str, str] = {
    "temperature": "%.2f",
    "pressure": "%.5f",
    "specific_enthalpy": "%.3f",
    "specific_entropy": "%.5f",
}

# Cómo se presenta cada dispositivo de un solo paso.
_DEVICES: dict[str, dict[str, Any]] = {
    "turbine": {
        "prefix": "turb",
        "title": "Turbina",
        "work": "w_t",
        "in_header": "Estado 1 (entrada)",
        "diagram": "Ts",
        "functions": (turbine_direct, turbine_inverse),
    },
    "compressor": {
        "prefix": "comp",
        "title": "Compresor",
        "work": "w_c",
        "in_header": "Estado 1 (entrada, gas o vapor)",
        "diagram": "logph",
        "functions": (compressor_direct, compressor_inverse),
    },
    "pump": {
        "prefix": "pump",
        "title": "Bomba",
        "work": "w_p",
        "in_header": "Estado 1 (entrada, líquido)",
        "diagram": "Ts",
        "functions": (pump_direct, pump_inverse),
    },
}


# ---------------------------------------------------------------------
# Cálculo (cacheado) y valores iniciales
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _defaults_cached(fluid: str, device: str) -> DeviceDefaults:
    return suggested_device_inputs(fluid, device)  # type: ignore[arg-type]


@st.cache_data(show_spinner=False)
def _default_outlet_cached(fluid: str, device: str) -> tuple[float, float]:
    """(p, h) de la salida real con los valores iniciales: default del modo inverso."""
    d = _defaults_cached(fluid, device)
    state_in = fluid_state_from_pair(fluid, d.pair_in, **d.inlet).to_state_point()  # type: ignore[arg-type]
    direct = _DEVICES[device]["functions"][0]
    result = direct(fluid=fluid, state_in=state_in, p_out_Pa=d.p_out_Pa, eta_s=d.eta_s)
    return result.state_out_real.P_Pa, result.state_out_real.h_J_per_kg


@st.cache_data(show_spinner=False)
def _fluid_states_cached(
    fluid: str, labeled: tuple[tuple[str, float, float], ...]
) -> list[tuple[str, FluidState]]:
    """Completa (región, x, …) los estados de un resultado a partir de (p, h)."""
    return [(label, fluid_state_from_pair(fluid, "PH", p=p, h=h)) for label, p, h in labeled]


def _symbol_defaults(kwargs_si: dict[str, float]) -> dict[str, float]:
    """``{"t": …, "p": …}`` → ``{"T": …, "P": …}`` (claves de los inputs)."""
    return {kw.upper(): value for kw, value in kwargs_si.items()}


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _render_var_input(
    symbol: str,
    *,
    key: str,
    default_si: float,
) -> tuple[str, float]:
    """Renderiza un input para una propiedad y devuelve ``(kwarg_si_name, value_si)``.

    Para magnitudes con sistema (T/P/H/S) usa el sistema actual. Para X
    (calidad), el widget es plano sin conversión.
    """
    if symbol == "X":
        val_x = st.number_input(
            "Título x (calidad) [-]",
            value=float(default_si),
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
            key=key,
        )
        return "x", float(val_x)

    kw_name, kind, _ = _SYMBOL_SPEC[symbol]
    label_map = {
        "T": "Temperatura T",
        "P": "Presión P",
        "H": "Entalpía h",
        "S": "Entropía s",
    }
    value_si = number_input_si(
        label=label_map[symbol],
        kind=kind,  # type: ignore[arg-type]
        default_si=default_si,
        key=key,
        format=_INPUT_FORMAT[kind],
    )
    return kw_name, value_si


def _build_state_input(
    *,
    key_prefix: str,
    header: str,
    defaults_si: dict[str, float],
    default_pair_label: str,
) -> tuple[str, dict[str, float]]:
    """Renderiza el form para definir un estado y devuelve ``(pair_code, kwargs_si)``.

    ``defaults_si`` mapea símbolo → valor en SI. Para los símbolos faltantes
    se usa :data:`_SYMBOL_SPEC` como fallback.
    """
    st.markdown(f"**{header}**")
    pair_options = list(_PAIR_LABELS_TO_CODE.keys())
    pair_label = st.selectbox(
        "Par de variables",
        pair_options,
        index=pair_options.index(default_pair_label),
        key=f"{key_prefix}_pair_label",
    )
    pair_code = _PAIR_LABELS_TO_CODE[pair_label]
    char1, char2 = pair_code[0], pair_code[1]
    c1, c2 = st.columns(2)
    with c1:
        kw1, val1 = _render_var_input(
            char1,
            key=f"{key_prefix}_v1",
            default_si=defaults_si.get(char1, _SYMBOL_SPEC[char1][2]),
        )
    with c2:
        kw2, val2 = _render_var_input(
            char2,
            key=f"{key_prefix}_v2",
            default_si=defaults_si.get(char2, _SYMBOL_SPEC[char2][2]),
        )
    return pair_code, {kw1: val1, kw2: val2}


def _resolve_state(fluid: str, pair_code: str, kwargs_si: dict[str, float]) -> StatePoint:
    """Estado validado (rango de la ecuación de estado, mensajes en castellano)."""
    return fluid_state_from_pair(fluid, pair_code, **kwargs_si).to_state_point()  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Presentación de resultados
# ---------------------------------------------------------------------


def _render_error(exc: ValueError) -> None:
    """Mensaje para el alumno arriba; el detalle de CoolProp, aparte y chico."""
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico (CoolProp): {detail.strip()}")


def _render_states_table(
    fluid: str, labeled: list[tuple[str, StatePoint]], system: UnitSystem
) -> list[tuple[str, FluidState]]:
    """Tabla con T, p, v, u, h, s, x y región de cada estado (x = "—" fuera de la campana)."""
    states = _fluid_states_cached(
        fluid, tuple((label, s.P_Pa, s.h_J_per_kg) for label, s in labeled)
    )
    frame = pd.DataFrame(states_table(states, system)).drop(columns="Fluido")
    for column in frame.columns:
        if column not in ("Estado", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)
    return states


def _render_procedure(steps: IsentropicSteps) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.markdown("**Fórmula aplicada:**")
        st.latex(steps.formula_latex)
        st.markdown("**Con los valores ingresados:**")
        st.latex(steps.substituted_latex)
        st.markdown("**En palabras:**")
        st.write(steps.narrative_es)


def _render_export(data: dict[str, Any], *, base: str, key: str) -> None:
    st.markdown("#### 💾 Exportar")
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=summary_csv(data).encode("utf-8"),
        file_name=f"{base}.csv",
        mime="text/csv",
        key=f"{key}_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"{base}.json",
        mime="application/json",
        key=f"{key}_json",
    )


def _clear_state_for(prefix: str) -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(f"{prefix}_result") or k.startswith(f"{prefix}_compare_active"):
            del st.session_state[k]


def _format_specific_work(value_si: float) -> str:
    """Trabajo específico en unidades del sistema actual."""
    return quantity_label(value_si, "specific_enthalpy", precision=5)


def _stored_result(prefix: str, fluid: str) -> Any:
    """Resultado guardado de la pestaña, solo si es del fluido elegido."""
    stored = st.session_state.get(f"{prefix}_result")
    if stored is None or stored.get("fluid") != fluid:
        return None
    return stored["result"]


# ---------------------------------------------------------------------
# Overlays para el diagrama del proceso (Fase 1.5a)
# ---------------------------------------------------------------------


_DIAGRAM_CAPTION = (
    "ℹ️ Las **líneas rectas** que unen estados reales con sus contrapartes "
    "isoentrópicas (2s→2) son **referencias visuales**, no representan la "
    "trayectoria termodinámica real del fluido. El segmento 1→2s sí es la "
    "isoentrópica calculada por CoolProp."
)


def _isentropic_overlays(fluid: str, result: IsentropicResult) -> list[ProcessOverlay]:
    """Overlays para una expansión/compresión simple (turbina, compresor, bomba).

    Devuelve tres curvas:

    - 1 → 2s (isoentrópica real, calculada por fluprodia/CoolProp).
    - 2s → 2 (segmento recto, referencia visual).
    - 1 → 2 (segmento recto, ayuda a leer la diferencia con 1→2s).
    """
    system = get_current_system()
    try:
        diagram = get_diagram(fluid, system)
        proc = isentropic_process(
            diagram,
            DiagramSpec(fluid=fluid, system=system),
            s_J_per_kg_K=result.state_in.s_J_per_kg_K,
            p_start_Pa=result.state_in.P_Pa,
            p_end_Pa=result.state_out_isen.P_Pa,
        )
        curve_1_2s = ProcessOverlay(
            name="1 → 2s (isoentrópico)",
            color="#2ca02c",
            dash="solid",
            coords_si=proc,
        )
    except Exception:
        # Si CoolProp/fluprodia falla (p.ej. estado fuera de rango), caemos
        # a segmento recto para no romper la UI.
        curve_1_2s = linear_segment_overlay(
            name="1 → 2s (referencia)",
            color="#2ca02c",
            dash="dot",
            start=result.state_in,
            end=result.state_out_isen,
        )
    seg_2s_2 = linear_segment_overlay(
        name="2s → 2 (referencia)",
        color="#7f7f7f",
        dash="dot",
        start=result.state_out_isen,
        end=result.state_out_real,
    )
    seg_1_2 = linear_segment_overlay(
        name="1 → 2 (real, ref.)",
        color="#d62728",
        dash="dash",
        start=result.state_in,
        end=result.state_out_real,
    )
    return [curve_1_2s, seg_2s_2, seg_1_2]


def _polytropic_overlays(fluid: str, result: PolytropicResult) -> list[ProcessOverlay]:
    """Overlays para un compresor multietapa.

    Por cada etapa k: 1_k → 2s_k (isoentrópica), 2s_k → 2_k (segmento real).
    Si hay intercooler tras la etapa k, agrega 2_k → 1_{k+1} (isobárica).
    """
    system = get_current_system()
    overlays: list[ProcessOverlay] = []
    spec = DiagramSpec(fluid=fluid, system=system)
    try:
        diagram = get_diagram(fluid, system)
    except Exception:
        diagram = None

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
    ]

    for k, stage in enumerate(result.stages):
        color = palette[k % len(palette)]
        # 1_k → 2s_k (isoentrópica)
        if diagram is not None:
            try:
                proc = isentropic_process(
                    diagram,
                    spec,
                    s_J_per_kg_K=stage.state_in.s_J_per_kg_K,
                    p_start_Pa=stage.state_in.P_Pa,
                    p_end_Pa=stage.state_out_isen.P_Pa,
                )
                overlays.append(
                    ProcessOverlay(
                        name=f"Etapa {stage.index}: isoen.",
                        color=color,
                        dash="solid",
                        coords_si=proc,
                    )
                )
            except Exception:
                overlays.append(
                    linear_segment_overlay(
                        name=f"Etapa {stage.index}: isoen. (ref)",
                        color=color,
                        dash="dot",
                        start=stage.state_in,
                        end=stage.state_out_isen,
                    )
                )
        else:
            overlays.append(
                linear_segment_overlay(
                    name=f"Etapa {stage.index}: isoen. (ref)",
                    color=color,
                    dash="dot",
                    start=stage.state_in,
                    end=stage.state_out_isen,
                )
            )
        # 2s_k → 2_k (referencia visual)
        overlays.append(
            linear_segment_overlay(
                name=f"Etapa {stage.index}: real (ref)",
                color=color,
                dash="dash",
                start=stage.state_out_isen,
                end=stage.state_out_real,
            )
        )
        # Intercooler aguas abajo de la etapa k (isobárica)
        if stage.cooled_after and (k + 1) < len(result.stages):
            next_in = result.stages[k + 1].state_in
            if diagram is not None:
                try:
                    proc_ic = isobaric_process(
                        diagram,
                        spec,
                        p_Pa=stage.state_out_real.P_Pa,
                        t_start_K=stage.state_out_real.T_K,
                        t_end_K=next_in.T_K,
                    )
                    overlays.append(
                        ProcessOverlay(
                            name=f"Intercooler {stage.index}→{stage.index + 1}",
                            color="#1f77b4",
                            dash="solid",
                            coords_si=proc_ic,
                        )
                    )
                    continue
                except Exception:
                    pass
            overlays.append(
                linear_segment_overlay(
                    name=f"Intercooler {stage.index}→{stage.index + 1} (ref)",
                    color="#1f77b4",
                    dash="dot",
                    start=stage.state_out_real,
                    end=next_in,
                )
            )
    return overlays


def _render_isentropic_diagram(
    fluid: str,
    result: IsentropicResult,
    *,
    selector_key: str,
    chart_key: str,
    default_diagram: str = "Ts",
) -> None:
    """Expansor con el diagrama del proceso simple (turbina/compresor/bomba)."""
    with st.expander("📈 Diagrama del proceso", expanded=False):
        st.caption(_DIAGRAM_CAPTION)
        diagram_type = diagram_type_selector(
            key=selector_key,
            default=default_diagram,  # type: ignore[arg-type]
        )
        system = get_current_system()
        overlays = _isentropic_overlays(fluid, result)
        points = [
            DiagramPoint(state=result.state_in, label="1", color="#1f77b4"),
            DiagramPoint(state=result.state_out_isen, label="2s", color="#2ca02c"),
            DiagramPoint(state=result.state_out_real, label="2", color="#d62728"),
        ]
        render_diagram_plotly(
            fluid=fluid,
            diagram_type=diagram_type,
            system=system,
            points=points,
            overlays=overlays,
            chart_key=chart_key,
        )


def _render_polytropic_diagram(
    fluid: str,
    result: PolytropicResult,
    *,
    selector_key: str,
    chart_key: str,
) -> None:
    """Expansor con el diagrama del proceso multietapa."""
    with st.expander("📈 Diagrama del proceso multietapa", expanded=False):
        st.caption(_DIAGRAM_CAPTION)
        diagram_type = diagram_type_selector(key=selector_key, default="logph")
        system = get_current_system()
        overlays = _polytropic_overlays(fluid, result)
        points: list[DiagramPoint] = []
        points.append(DiagramPoint(state=result.stages[0].state_in, label="1", color="#1f77b4"))
        for stage in result.stages:
            points.append(
                DiagramPoint(
                    state=stage.state_out_isen,
                    label=f"{stage.index}s",
                    color="#2ca02c",
                )
            )
            points.append(
                DiagramPoint(
                    state=stage.state_out_real,
                    label=f"{stage.index}",
                    color="#d62728",
                )
            )
        render_diagram_plotly(
            fluid=fluid,
            diagram_type=diagram_type,
            system=system,
            points=points,
            overlays=overlays,
            chart_key=chart_key,
        )


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Las relaciones de esta página están en el [vademecum de la cátedra]"
            f"({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): **§10.4 *Rendimientos "
            "isoentrópicos*** (turbina, compresor, bomba), §3.3 *Sistema abierto en régimen "
            "permanente* y §6.3 *Compresión en dos etapas con interenfriamiento*."
        )
        st.markdown(
            "**Estado isoentrópico de salida** — misma entropía que la entrada, a la "
            "presión de salida:"
        )
        st.latex(r"h_{2s} = h(p_2,\ s_1)")
        st.markdown("**Turbina** (el real entrega menos trabajo que el ideal):")
        st.latex(
            r"\eta_{s,T} = \frac{w_{\mathrm{real}}}{w_{\mathrm{ideal}}}"
            r" = \frac{h_1 - h_2}{h_1 - h_{2s}}"
        )
        st.markdown("**Compresor y bomba** (el real consume más trabajo que el ideal):")
        st.latex(
            r"\eta_{s,C} = \frac{w_{\mathrm{ideal}}}{w_{\mathrm{real}}}"
            r" = \frac{h_{2s} - h_1}{h_2 - h_1}"
        )
        st.latex(
            r"\eta_{s,P} \simeq \frac{v\,(p_2 - p_1)}{h_2 - h_1}"
            r" \qquad \text{(líquido incompresible)}"
        )
        st.markdown(
            "**Primer principio** en régimen permanente, adiabático y sin variaciones de "
            "energía cinética ni potencial (§3.3):"
        )
        st.latex(r"w = h_1 - h_2")
        st.markdown(
            "**Compresor multietapa** — la misma relación de compresión en cada etapa; para "
            "dos etapas es la presión intermedia óptima (§6.3):"
        )
        st.latex(
            r"\Pi = \left(\frac{p_{\mathrm{out}}}{p_{\mathrm{in}}}\right)^{1/n}"
            r" \qquad p_x = \sqrt{p_1\,p_2}"
        )


# ---------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------


def _render_device_tab(device: str, fluid: str, system: UnitSystem) -> None:
    """Turbina, compresor o bomba: entradas, cálculo y resultado."""
    ui = _DEVICES[device]
    prefix = ui["prefix"]
    direct_fn, inverse_fn = ui["functions"]
    defaults = _defaults_cached(fluid, device)

    st.markdown(f"### {ui['title']}")
    mode = st.radio(
        "Modo",
        ["Directo (calcular salida dado η_s)", "Inverso (calcular η_s dados los estados)"],
        key=f"{prefix}_mode",
    )
    is_direct = mode.startswith("Directo")

    # Las keys llevan el fluido: al cambiarlo, los inputs vuelven a valores válidos.
    pair_in, kwargs_in = _build_state_input(
        key_prefix=f"{prefix}_in_{fluid}",
        header=ui["in_header"],
        defaults_si=_symbol_defaults(defaults.inlet),
        default_pair_label=_PAIR_CODE_TO_LABEL[defaults.pair_in],
    )
    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_Pa = number_input_si(
                label="Presión de salida P₂",
                kind="pressure",
                default_si=defaults.p_out_Pa,
                key=f"{prefix}_p_out_{fluid}",
                format=_INPUT_FORMAT["pressure"],
            )
        with c2:
            eta_s = st.number_input(
                "Rendimiento isoentrópico η_s",
                value=defaults.eta_s,
                min_value=0.01,
                max_value=1.00,
                step=0.01,
                format="%.4f",
                key=f"{prefix}_eta_{fluid}",
            )
    else:
        p_out_default, h_out_default = _default_outlet_cached(fluid, device)
        pair_out, kwargs_out = _build_state_input(
            key_prefix=f"{prefix}_out_{fluid}",
            header="Estado 2 (salida real)",
            defaults_si={"P": p_out_default, "H": h_out_default},
            default_pair_label="P y h",
        )

    if st.button("Calcular", key=f"{prefix}_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = direct_fn(
                    fluid=fluid, state_in=state_in, p_out_Pa=p_out_Pa, eta_s=float(eta_s)
                )
            else:
                state_out_real = _resolve_state(fluid, pair_out, kwargs_out)
                result = inverse_fn(fluid=fluid, state_in=state_in, state_out_real=state_out_real)
        except ValueError as exc:
            _render_error(exc)
            _clear_state_for(prefix)
        else:
            st.session_state[f"{prefix}_result"] = {"fluid": fluid, "result": result}

    result = _stored_result(prefix, fluid)
    if result is None:
        return
    assert isinstance(result, IsentropicResult)
    c1, c2 = st.columns(2)
    c1.metric("η_s", f"{result.eta_s:.4f}")
    c2.metric(
        f"Trabajo específico {ui['work']}", _format_specific_work(result.specific_work_J_per_kg)
    )
    if device == "compressor" and 0.0 < result.state_in.x < 1.0:
        st.warning(
            f"La entrada es vapor húmedo (x = {result.state_in.x:.3f}). El cálculo es "
            "válido, pero los compresores reales necesitan vapor saturado seco o "
            "sobrecalentado: el líquido arrastrado los daña (golpe de líquido).",
            icon="⚠️",
        )
    _render_states_table(fluid, isentropic_labeled_states(result), system)
    _render_procedure(isentropic_steps(result, system))
    _render_isentropic_diagram(
        fluid,
        result,
        selector_key=f"diag_type_{prefix}",
        chart_key=f"diag_chart_{prefix}",
        default_diagram=ui["diagram"],
    )
    if device == "pump":
        _render_pump_comparison(result, prefix)
    _render_export(
        isentropic_to_dict(result, system),
        base=f"{device}_{result.mode}_{fluid}".lower(),
        key=f"{prefix}_download",
    )


def _render_pump_comparison(result: IsentropicResult, prefix: str) -> None:
    if st.button("🆚 Comparar contra modelo incompresible", key=f"{prefix}_compare_btn"):
        st.session_state[f"{prefix}_compare_active"] = True
    if not st.session_state.get(f"{prefix}_compare_active", False):
        return
    st.markdown("#### 🆚 Modelo simplificado de líquido incompresible")
    st.caption(
        "Aproximación clásica para una bomba que mueve un líquido (vademecum §10.4): "
        r"$w_p \approx v_1 \cdot (P_2 - P_1) / \eta_s$. "
        "Compará contra la ecuación de estado para ver cuánto se aleja."
    )
    try:
        cmp = pump_incompressible_comparison(result)
    except ValueError as exc:
        st.warning(f"No pude armar el modelo simple: {exc}")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("w_p (ecuación de estado)", _format_specific_work(cmp.w_eos_J_per_kg))
    c2.metric("w_p (modelo simple)", _format_specific_work(cmp.w_incompressible_J_per_kg))
    c3.metric("Error relativo", f"{cmp.error_pct:.3f} %")
    st.caption(
        "Para agua subenfriada o líquido saturado y compresiones moderadas, el error típico "
        "es < 1 %. Crece cerca del punto crítico o con fluidos más compresibles."
    )


def _render_multistage_tab(fluid: str, system: UnitSystem) -> None:
    prefix = "poly"
    defaults = _defaults_cached(fluid, "multistage")
    st.markdown("### Compresor multietapa")
    st.caption(
        "Compresor de **n** etapas con relación de presión igual por etapa, "
        "η_s común a todas, e intercooler opcional entre etapas (la última no "
        "tiene intercooler aguas abajo). Solo modo directo."
    )
    pair_in, kwargs_in = _build_state_input(
        key_prefix=f"{prefix}_in_{fluid}",
        header="Estado 1 (entrada a la primera etapa)",
        defaults_si=_symbol_defaults(defaults.inlet),
        default_pair_label=_PAIR_CODE_TO_LABEL[defaults.pair_in],
    )
    st.markdown("**Parámetros del proceso**")
    c1, c2 = st.columns(2)
    with c1:
        p_out_Pa = number_input_si(
            label="Presión final P_out",
            kind="pressure",
            default_si=defaults.p_out_Pa,
            key=f"{prefix}_p_out_{fluid}",
            format=_INPUT_FORMAT["pressure"],
        )
        n_stages = st.number_input(
            "Número de etapas n",
            value=defaults.n_stages,
            min_value=1,
            max_value=10,
            step=1,
            key=f"{prefix}_n_{fluid}",
        )
    with c2:
        eta_s_stage = st.number_input(
            "η_s por etapa",
            value=defaults.eta_s,
            min_value=0.01,
            max_value=1.00,
            step=0.01,
            format="%.4f",
            key=f"{prefix}_eta_{fluid}",
        )
        intercool = st.checkbox(
            "Con intercooler entre etapas", value=True, key=f"{prefix}_intercool_{fluid}"
        )
    t_intercool_K = None
    if intercool:
        t_intercool_K = number_input_si(
            label="Temperatura del intercooler T_ic",
            kind="temperature",
            default_si=defaults.t_intercool_K or 298.15,
            key=f"{prefix}_tic_{fluid}",
            format=_INPUT_FORMAT["temperature"],
            help="Tiene que quedar por encima de la temperatura de saturación a la presión "
            "intermedia: si no, el fluido condensaría entre etapas.",
        )

    if st.button("Calcular", key=f"{prefix}_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            result = compressor_multistage(
                fluid=fluid,
                state_in=state_in,
                p_out_Pa=p_out_Pa,
                n_stages=int(n_stages),
                eta_s_per_stage=float(eta_s_stage),
                intercool=bool(intercool),
                t_intercool_K=t_intercool_K,
            )
        except ValueError as exc:
            _render_error(exc)
            _clear_state_for(prefix)
        else:
            st.session_state[f"{prefix}_result"] = {"fluid": fluid, "result": result}

    result = _stored_result(prefix, fluid)
    if result is None:
        return
    assert isinstance(result, PolytropicResult)
    c1, c2, c3 = st.columns(3)
    c1.metric("Δh total (multietapa)", _format_specific_work(result.total_delta_h_real_J_per_kg))
    c2.metric(
        "Δh 1 etapa equivalente",
        _format_specific_work(result.delta_h_single_stage_real_J_per_kg),
    )
    c3.metric("Ahorro vs 1 etapa", f"{result.saving_pct:+.2f} %")
    st.caption(
        f"Relación de compresión por etapa Π = {result.pressure_ratio_per_stage:.4f} "
        f"({'con' if result.intercool else 'sin'} intercooler)."
    )
    if result.saving_pct <= 0.0:
        st.info(
            "Con estos datos el escalonamiento no ahorra trabajo: cada etapa real genera "
            "entropía y calienta la entrada de la siguiente (efecto de recalentamiento), y "
            "el intercooler no alcanza a compensarlo. Pasa sin intercooler, o con fluidos "
            "como el R-1234yf, cuya salida de cada etapa queda apenas sobrecalentada.",
            icon="ℹ️",
        )
    _render_states_table(fluid, multistage_labeled_states(result), system)
    _render_procedure(multistage_steps(result, system))
    _render_polytropic_diagram(
        fluid,
        result,
        selector_key="diag_type_poly",
        chart_key="diag_chart_poly",
    )
    _render_export(
        multistage_to_dict(result, system),
        base=f"multietapa_{fluid}".lower(),
        key=f"{prefix}_download",
    )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Isoentrópicos", page_icon="⚙️", layout="centered")

sidebar_credits(version=PAGE_VERSION, page_name="Isoentrópicos")
render_units_selector()
system = get_current_system()

st.subheader(SUBJECT)
st.title("⚙️ Rendimientos isoentrópicos")
st.markdown(
    "Calculadora de turbina, compresor, bomba y compresor multietapa con "
    "intercooler. Modo **directo** (dado η_s calcular el estado real) o "
    "**inverso** (dados los dos estados, recuperar η_s)."
)
_render_theory()
st.markdown("---")

fluid = st.selectbox(
    "Fluido",
    SUPPORTED_FLUIDS,
    format_func=lambda f: FLUID_NAMES_ES.get(f, f),
    key="iso_fluid",
)

tab_turbine, tab_compressor, tab_pump, tab_polytropic = st.tabs(
    ["🌪️ Turbina", "🌀 Compresor", "💧 Bomba", "🔁 Multietapa"]
)
with tab_turbine:
    _render_device_tab("turbine", fluid, system)
with tab_compressor:
    _render_device_tab("compressor", fluid, system)
with tab_pump:
    _render_device_tab("pump", fluid, system)
with tab_polytropic:
    _render_multistage_tab(fluid, system)
