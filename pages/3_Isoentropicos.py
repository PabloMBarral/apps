"""Página 3 — Rendimientos isoentrópicos.

Cuatro pestañas: turbina, compresor, bomba y compresor multietapa
(politrópico, con intercooler opcional). En cada una se puede operar en
modo **directo** (dado η_s y P_out → calcular estado real) o **inverso**
(dados estados de entrada y salida → recuperar η_s). El multietapa va
solo en modo directo y trae built-in la comparación contra single-stage.

Solo la bomba ofrece comparación opt-in contra el modelo simplificado
de líquido incompresible (``w_p ≈ v_in · Δp / η_s``).

Unidades: la tabla de estados y los KPIs se muestran en el sistema
seleccionado en el sidebar (SI / Técnico / Inglés). Los pasos didácticos
del expansor "🔬 Procedimiento" se mantienen en sistema Técnico
(kJ/kg, °C, bar) — generados en ``core/isentropic.py`` para mantener la
consistencia con la bibliografía clásica (Cengel). La conversión de los
pasos al sistema actual queda pendiente para Fase 1.5.

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Diagrama T-s / h-s del proceso con fluprodia (Fase 1.5).
- Conversión de los pasos didácticos al sistema activo (Fase 1.5).
- Expansor `📖 Fórmulas teóricas` con link a vademecum-termo.
- Botón de exportar resultados (CSV / JSON).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from core.fluids import SUPPORTED_FLUIDS, StatePoint, state_from_pair
from core.isentropic import (
    IsentropicResult,
    PolytropicResult,
    compressor_direct,
    compressor_inverse,
    compressor_multistage,
    pump_direct,
    pump_inverse,
    turbine_direct,
    turbine_inverse,
)
from core.units_system import convert_from_si
from ui.branding import SUBJECT, sidebar_credits
from ui.units_ui import (
    current_unit_label,
    get_current_system,
    number_input_si,
    quantity_label,
    render_units_selector,
)

PAGE_VERSION = "0.7.0"

# Códigos de par independiente reconocidos por core.fluids.state_from_pair,
# ordenados por uso didáctico.
_PAIR_LABELS_TO_CODE: dict[str, str] = {
    "T y P": "TP",
    "P y h": "PH",
    "P y x (saturado)": "PX",
    "T y x (saturado)": "TX",
    "P y s": "PS",
    "T y s": "TS",
    "h y s": "HS",
}

# Símbolo → (kwarg en state_from_pair, kind para el sistema de unidades, default en SI).
_SYMBOL_SPEC: dict[str, tuple[str, str, float]] = {
    "T": ("t", "temperature", 298.15),  # 25 °C
    "P": ("p", "pressure", 1.0e5),  # 1 bar
    "H": ("h", "specific_enthalpy", 2.5e6),  # 2500 kJ/kg
    "S": ("s", "specific_entropy", 5000.0),  # 5 kJ/(kg·K)
    "X": ("x", "dimensionless", 0.0),
}


# ---------------------------------------------------------------------
# Helpers
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
        format="%.4f",
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
    return state_from_pair(fluid, pair_code, **kwargs_si)  # type: ignore[arg-type]


def _state_to_row(label: str, state: StatePoint) -> dict[str, Any]:
    """Una fila de la tabla de estados en el sistema actual."""
    sys = get_current_system()
    T_col = f"T [{current_unit_label('temperature')}]"
    P_col = f"P [{current_unit_label('pressure')}]"
    h_col = f"h [{current_unit_label('specific_enthalpy')}]"
    s_col = f"s [{current_unit_label('specific_entropy')}]"
    return {
        "estado": label,
        T_col: convert_from_si(state.T_K, "temperature", sys),
        P_col: convert_from_si(state.P_Pa, "pressure", sys),
        h_col: convert_from_si(state.h_J_per_kg, "specific_enthalpy", sys),
        s_col: convert_from_si(state.s_J_per_kg_K, "specific_entropy", sys),
        "x [-]": state.x,
    }


def _render_states_table(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format(precision=4), use_container_width=True)


def _render_procedure(steps: Any) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.caption(
            "ℹ️ Los pasos se muestran siempre en sistema **Técnico** "
            "(°C, bar, kJ/kg, kJ/(kg·K)) para consistencia con la "
            "bibliografía clásica (Cengel). La conversión al sistema "
            "activo de la tabla de estados queda pendiente para Fase 1.5."
        )
        st.markdown("**Fórmula aplicada:**")
        st.latex(steps.formula_latex)
        st.markdown("**Con los valores ingresados:**")
        st.latex(steps.substituted_latex)
        st.markdown("**En palabras:**")
        st.write(steps.narrative_es)


def _clear_state_for(prefix: str) -> None:
    for k in list(st.session_state.keys()):
        if k.startswith(f"{prefix}_result") or k.startswith(f"{prefix}_compare_active"):
            del st.session_state[k]


def _format_specific_work(value_si: float) -> str:
    """Trabajo específico en unidades del sistema actual."""
    return quantity_label(value_si, "specific_enthalpy", precision=4)


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Isoentrópicos", page_icon="⚙️", layout="centered")

st.subheader(SUBJECT)
st.title("⚙️ Rendimientos isoentrópicos")
st.markdown(
    "Calculadora de turbina, compresor, bomba y compresor multietapa con "
    "intercooler. Modo **directo** (dado η_s calcular el estado real) o "
    "**inverso** (dados los dos estados, recuperar η_s)."
)
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Isoentrópicos")
render_units_selector()

fluid = st.selectbox(
    "Fluido (CoolProp)",
    SUPPORTED_FLUIDS,
    index=0,
    key="iso_fluid",
)

tab_turbine, tab_compressor, tab_pump, tab_polytropic = st.tabs(
    ["🌪️ Turbina", "🌀 Compresor", "💧 Bomba", "🔁 Multietapa"]
)


# =====================================================================
# Tab — Turbina
# =====================================================================
with tab_turbine:
    st.markdown("### Turbina")
    mode = st.radio(
        "Modo",
        ["Directo (calcular salida dado η_s)", "Inverso (calcular η_s dados los estados)"],
        key="turb_mode",
        horizontal=False,
    )
    is_direct = mode.startswith("Directo")

    # Defaults: vapor sobrecalentado 30 bar, 400 °C.
    pair_in, kwargs_in = _build_state_input(
        key_prefix="turb_in",
        header="Estado 1 (entrada)",
        defaults_si={"T": 673.15, "P": 3.0e6},  # 400 °C, 30 bar
        default_pair_label="T y P",
    )

    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_Pa = number_input_si(
                label="Presión de salida P₂",
                kind="pressure",
                default_si=5.0e4,  # 0.5 bar
                key="turb_p_out",
                format="%.4f",
            )
        with c2:
            eta_s = st.number_input(
                "Rendimiento isoentrópico η_s",
                value=0.90,
                min_value=0.01,
                max_value=1.00,
                step=0.01,
                format="%.4f",
                key="turb_eta",
            )
    else:
        pair_out, kwargs_out = _build_state_input(
            key_prefix="turb_out",
            header="Estado 2 (salida real)",
            defaults_si={"T": 373.15, "P": 5.0e4},  # 100 °C, 0.5 bar
            default_pair_label="T y P",
        )

    if st.button("Calcular", key="turb_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = turbine_direct(
                    fluid=fluid,
                    state_in=state_in,
                    p_out_Pa=p_out_Pa,
                    eta_s=float(eta_s),
                )
            else:
                state_out_real = _resolve_state(fluid, pair_out, kwargs_out)
                result = turbine_inverse(
                    fluid=fluid,
                    state_in=state_in,
                    state_out_real=state_out_real,
                )
        except ValueError as exc:
            st.error(f"Error: {exc}")
            _clear_state_for("turb")
        else:
            st.session_state["turb_result"] = result

    if "turb_result" in st.session_state:
        result = st.session_state["turb_result"]
        assert isinstance(result, IsentropicResult)
        w_t = -result.delta_h_real_J_per_kg
        c1, c2 = st.columns(2)
        c1.metric("η_s", f"{result.eta_s:.4f}")
        c2.metric("Trabajo específico w_t", _format_specific_work(w_t))
        _render_states_table(
            [
                _state_to_row("1 (entrada)", result.state_in),
                _state_to_row("2s (salida isoentrópica)", result.state_out_isen),
                _state_to_row("2 (salida real)", result.state_out_real),
            ]
        )
        _render_procedure(result.steps)


# =====================================================================
# Tab — Compresor
# =====================================================================
with tab_compressor:
    st.markdown("### Compresor")
    mode = st.radio(
        "Modo",
        ["Directo (calcular salida dado η_s)", "Inverso (calcular η_s dados los estados)"],
        key="comp_mode",
        horizontal=False,
    )
    is_direct = mode.startswith("Directo")

    pair_in, kwargs_in = _build_state_input(
        key_prefix="comp_in",
        header="Estado 1 (entrada)",
        defaults_si={"T": 298.15, "P": 1.0e5},  # 25 °C, 1 bar
        default_pair_label="T y P",
    )

    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_Pa = number_input_si(
                label="Presión de salida P₂",
                kind="pressure",
                default_si=8.0e5,  # 8 bar
                key="comp_p_out",
                format="%.4f",
            )
        with c2:
            eta_s = st.number_input(
                "Rendimiento isoentrópico η_s",
                value=0.80,
                min_value=0.01,
                max_value=1.00,
                step=0.01,
                format="%.4f",
                key="comp_eta",
            )
    else:
        pair_out, kwargs_out = _build_state_input(
            key_prefix="comp_out",
            header="Estado 2 (salida real)",
            defaults_si={"T": 573.15, "P": 8.0e5},  # 300 °C, 8 bar
            default_pair_label="T y P",
        )

    if st.button("Calcular", key="comp_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = compressor_direct(
                    fluid=fluid,
                    state_in=state_in,
                    p_out_Pa=p_out_Pa,
                    eta_s=float(eta_s),
                )
            else:
                state_out_real = _resolve_state(fluid, pair_out, kwargs_out)
                result = compressor_inverse(
                    fluid=fluid,
                    state_in=state_in,
                    state_out_real=state_out_real,
                )
        except ValueError as exc:
            st.error(f"Error: {exc}")
            _clear_state_for("comp")
        else:
            st.session_state["comp_result"] = result

    if "comp_result" in st.session_state:
        result = st.session_state["comp_result"]
        assert isinstance(result, IsentropicResult)
        w_c = result.delta_h_real_J_per_kg
        c1, c2 = st.columns(2)
        c1.metric("η_s", f"{result.eta_s:.4f}")
        c2.metric("Trabajo específico w_c", _format_specific_work(w_c))
        _render_states_table(
            [
                _state_to_row("1 (entrada)", result.state_in),
                _state_to_row("2s (salida isoentrópica)", result.state_out_isen),
                _state_to_row("2 (salida real)", result.state_out_real),
            ]
        )
        _render_procedure(result.steps)


# =====================================================================
# Tab — Bomba
# =====================================================================
with tab_pump:
    st.markdown("### Bomba")
    mode = st.radio(
        "Modo",
        ["Directo (calcular salida dado η_s)", "Inverso (calcular η_s dados los estados)"],
        key="pump_mode",
        horizontal=False,
    )
    is_direct = mode.startswith("Directo")

    pair_in, kwargs_in = _build_state_input(
        key_prefix="pump_in",
        header="Estado 1 (entrada, líquido)",
        defaults_si={"P": 1.0e4, "X": 0.0, "T": 318.96},  # 0.1 bar, x=0
        default_pair_label="P y x (saturado)",
    )

    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_Pa = number_input_si(
                label="Presión de salida P₂",
                kind="pressure",
                default_si=1.5e7,  # 150 bar
                key="pump_p_out",
                format="%.4f",
            )
        with c2:
            eta_s = st.number_input(
                "Rendimiento isoentrópico η_s",
                value=0.85,
                min_value=0.01,
                max_value=1.00,
                step=0.01,
                format="%.4f",
                key="pump_eta",
            )
    else:
        pair_out, kwargs_out = _build_state_input(
            key_prefix="pump_out",
            header="Estado 2 (salida real)",
            defaults_si={"P": 1.5e7, "T": 319.15},  # 150 bar, 46 °C
            default_pair_label="T y P",
        )

    if st.button("Calcular", key="pump_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = pump_direct(
                    fluid=fluid,
                    state_in=state_in,
                    p_out_Pa=p_out_Pa,
                    eta_s=float(eta_s),
                )
            else:
                state_out_real = _resolve_state(fluid, pair_out, kwargs_out)
                result = pump_inverse(
                    fluid=fluid,
                    state_in=state_in,
                    state_out_real=state_out_real,
                )
        except ValueError as exc:
            st.error(f"Error: {exc}")
            _clear_state_for("pump")
        else:
            st.session_state["pump_result"] = result

    if "pump_result" in st.session_state:
        result = st.session_state["pump_result"]
        assert isinstance(result, IsentropicResult)
        w_p = result.delta_h_real_J_per_kg
        c1, c2 = st.columns(2)
        c1.metric("η_s", f"{result.eta_s:.4f}")
        c2.metric("Trabajo específico w_p", _format_specific_work(w_p))
        _render_states_table(
            [
                _state_to_row("1 (entrada)", result.state_in),
                _state_to_row("2s (salida isoentrópica)", result.state_out_isen),
                _state_to_row("2 (salida real)", result.state_out_real),
            ]
        )
        _render_procedure(result.steps)

        # Comparación opt-in vs modelo incompresible.
        if st.button("🆚 Comparar contra modelo incompresible", key="pump_compare_btn"):
            st.session_state["pump_compare_active"] = True

        if st.session_state.get("pump_compare_active", False):
            st.markdown("#### 🆚 Modelo simplificado de líquido incompresible")
            st.caption(
                "Aproximación clásica para una bomba que mueve un líquido: "
                r"$w_p \approx v_1 \cdot (P_2 - P_1) / \eta_s$. "
                "Compará contra la EOS de CoolProp para ver cuánto se aleja."
            )
            try:
                # v_1 = 1 / rho_1 a partir de CoolProp.
                import CoolProp.CoolProp as cp

                rho_in = cp.PropsSI("D", "P", result.state_in.P_Pa, "T", result.state_in.T_K, fluid)
                v_in = 1.0 / rho_in
                dp = result.state_out_real.P_Pa - result.state_in.P_Pa
                w_p_simple = v_in * dp / result.eta_s
                w_p_real = result.delta_h_real_J_per_kg
                err_pct = (
                    abs(w_p_real - w_p_simple) / abs(w_p_real) * 100.0
                    if w_p_real != 0.0
                    else float("nan")
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("w_p (EOS CoolProp)", _format_specific_work(w_p_real))
                c2.metric("w_p (modelo simple)", _format_specific_work(w_p_simple))
                c3.metric("Error relativo", f"{err_pct:.3f} %")
                st.caption(
                    "Para agua subenfriada / saturada líquida y compresiones "
                    "moderadas, el error típico es < 1 %. Crece cerca del "
                    "punto crítico o con fluidos muy compresibles."
                )
            except Exception as exc:
                st.warning(f"No pude armar el modelo simple: {exc}")


# =====================================================================
# Tab — Compresor multietapa (politrópico)
# =====================================================================
with tab_polytropic:
    st.markdown("### Compresor multietapa")
    st.caption(
        "Compresor de **n** etapas con relación de presión igual por etapa, "
        "η_s común a todas, e intercooler opcional entre etapas (la última no "
        "tiene intercooler aguas abajo). Solo modo directo."
    )

    pair_in, kwargs_in = _build_state_input(
        key_prefix="poly_in",
        header="Estado 1 (entrada al primer etapa)",
        defaults_si={"T": 298.15, "P": 1.0e5},  # 25 °C, 1 bar
        default_pair_label="T y P",
    )

    st.markdown("**Parámetros del proceso**")
    c1, c2 = st.columns(2)
    with c1:
        p_out_Pa = number_input_si(
            label="Presión final P_out",
            kind="pressure",
            default_si=2.7e6,  # 27 bar
            key="poly_p_out",
            format="%.4f",
        )
        n_stages = st.number_input(
            "Número de etapas n",
            value=3,
            min_value=1,
            max_value=10,
            step=1,
            key="poly_n",
        )
    with c2:
        eta_s_stage = st.number_input(
            "η_s por etapa",
            value=0.85,
            min_value=0.01,
            max_value=1.00,
            step=0.01,
            format="%.4f",
            key="poly_eta",
        )
        intercool = st.checkbox(
            "Con intercooler entre etapas",
            value=True,
            key="poly_intercool",
        )

    if intercool:
        t_intercool_K = number_input_si(
            label="Temperatura del intercooler T_ic",
            kind="temperature",
            default_si=298.15,  # 25 °C
            key="poly_tic",
            format="%.4f",
            help="Default = temperatura ambiente (25 °C). En la práctica se usa "
            "la temperatura de entrada al compresor para 'full cooling'.",
        )
    else:
        t_intercool_K = None

    if st.button("Calcular", key="poly_btn", type="primary"):
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
            st.error(f"Error: {exc}")
            _clear_state_for("poly")
        else:
            st.session_state["poly_result"] = result

    if "poly_result" in st.session_state:
        result = st.session_state["poly_result"]
        assert isinstance(result, PolytropicResult)
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Δh total (multietapa)",
            _format_specific_work(result.total_delta_h_real_J_per_kg),
        )
        c2.metric(
            "Δh 1 etapa equivalente",
            _format_specific_work(result.delta_h_single_stage_real_J_per_kg),
        )
        single = result.delta_h_single_stage_real_J_per_kg
        if single != 0.0:
            saving_pct = (single - result.total_delta_h_real_J_per_kg) / single * 100.0
            c3.metric("Ahorro vs 1 etapa", f"{saving_pct:+.2f} %")
        else:
            c3.metric("Ahorro vs 1 etapa", "—")
        st.caption(
            f"Relación de compresión por etapa Π = "
            f"{result.pressure_ratio_per_stage:.4f} "
            f"({'con' if result.intercool else 'sin'} intercooler)."
        )

        # Tabla de estados por etapa.
        rows: list[dict[str, Any]] = []
        rows.append(_state_to_row("1 (entrada)", result.stages[0].state_in))
        for stage in result.stages:
            rows.append(
                _state_to_row(
                    f"Etapa {stage.index} — salida isen.",
                    stage.state_out_isen,
                )
            )
            rows.append(
                _state_to_row(
                    f"Etapa {stage.index} — salida real",
                    stage.state_out_real,
                )
            )
            if stage.cooled_after:
                next_idx = stage.index + 1
                # El estado de entrada de la siguiente etapa es post-intercooler.
                if next_idx - 1 < len(result.stages):
                    rows.append(
                        _state_to_row(
                            f"→ Intercooler a etapa {next_idx}",
                            result.stages[next_idx - 1].state_in,
                        )
                    )
        _render_states_table(rows)
        _render_procedure(result.steps)
