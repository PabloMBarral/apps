"""Página 3 — Rendimientos isoentrópicos.

Cuatro pestañas: turbina, compresor, bomba y compresor multietapa
(politrópico, con intercooler opcional). En cada una se puede operar en
modo **directo** (dado η_s y P_out → calcular estado real) o **inverso**
(dados estados de entrada y salida → recuperar η_s). El multietapa va
solo en modo directo y trae built-in la comparación contra single-stage.

Solo la bomba ofrece comparación opt-in contra el modelo simplificado
de líquido incompresible (``w_p ≈ v_in · Δp / η_s``).

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Diagrama T-s / h-s del proceso con fluprodia (Fase 1.5).
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
from core.units import bar_to_pa, c_to_k, j_to_kj, k_to_c, kj_to_j, pa_to_bar
from ui.branding import SUBJECT, sidebar_credits

PAGE_VERSION = "0.6.0"

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _render_var_input(
    symbol: str,
    *,
    key: str,
    default: float,
) -> tuple[str, float]:
    """Renderiza un input numérico para una propiedad y devuelve
    ``(kwarg_si_name, value_si)``. ``default`` viene en unidades
    didácticas (°C / bar / kJ por kg / -)."""
    if symbol == "T":
        val_c = st.number_input(
            "Temperatura T [°C]",
            value=float(default),
            format="%.4f",
            key=key,
        )
        return "t", c_to_k(float(val_c))
    if symbol == "P":
        val_bar = st.number_input(
            "Presión P [bar(a)]",
            value=float(default),
            min_value=1.0e-6,
            format="%.4f",
            key=key,
        )
        return "p", bar_to_pa(float(val_bar))
    if symbol == "H":
        val_kj = st.number_input(
            "Entalpía h [kJ/kg]",
            value=float(default),
            format="%.4f",
            key=key,
        )
        return "h", kj_to_j(float(val_kj))
    if symbol == "S":
        val_kj = st.number_input(
            "Entropía s [kJ/(kg·K)]",
            value=float(default),
            format="%.4f",
            key=key,
        )
        return "s", kj_to_j(float(val_kj))
    if symbol == "X":
        val_x = st.number_input(
            "Título x (calidad) [-]",
            value=float(default),
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
            key=key,
        )
        return "x", float(val_x)
    raise ValueError(f"Símbolo no soportado: {symbol!r}")


def _build_state_input(
    *,
    key_prefix: str,
    header: str,
    defaults: dict[str, float],
    default_pair_label: str,
) -> tuple[str, dict[str, float]]:
    """Renderiza el form para definir un estado y devuelve ``(pair_code, kwargs_si)``."""
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
            default=defaults.get(char1, _default_for_symbol(char1)),
        )
    with c2:
        kw2, val2 = _render_var_input(
            char2,
            key=f"{key_prefix}_v2",
            default=defaults.get(char2, _default_for_symbol(char2)),
        )
    return pair_code, {kw1: val1, kw2: val2}


def _default_for_symbol(symbol: str) -> float:
    return {"T": 25.0, "P": 1.0, "H": 2500.0, "S": 5.0, "X": 0.0}[symbol]


def _resolve_state(fluid: str, pair_code: str, kwargs_si: dict[str, float]) -> StatePoint:
    return state_from_pair(fluid, pair_code, **kwargs_si)  # type: ignore[arg-type]


def _state_to_row(label: str, state: StatePoint) -> dict[str, Any]:
    return {
        "estado": label,
        "T [°C]": k_to_c(state.T_K),
        "P [bar]": pa_to_bar(state.P_Pa),
        "h [kJ/kg]": j_to_kj(state.h_J_per_kg),
        "s [kJ/(kg·K)]": j_to_kj(state.s_J_per_kg_K),
        "x [-]": state.x,
    }


def _render_states_table(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format(precision=4), use_container_width=True)


def _render_procedure(steps: Any) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
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


def _format_kj_per_kg(value_si: float) -> str:
    return f"{value_si / 1.0e3:.4g} kJ/kg"


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

    # Inputs de estado de entrada (defaults: vapor sobrecalentado 30 bar, 400 °C).
    pair_in, kwargs_in = _build_state_input(
        key_prefix="turb_in",
        header="Estado 1 (entrada)",
        defaults={"T": 400.0, "P": 30.0},
        default_pair_label="T y P",
    )

    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_bar = st.number_input(
                "Presión de salida P₂ [bar(a)]",
                value=0.5,
                min_value=1.0e-6,
                format="%.4f",
                key="turb_p_out",
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
            defaults={"T": 100.0, "P": 0.5},
            default_pair_label="T y P",
        )

    if st.button("Calcular", key="turb_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = turbine_direct(
                    fluid=fluid,
                    state_in=state_in,
                    p_out_Pa=bar_to_pa(float(p_out_bar)),
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
        c2.metric("Trabajo específico w_t", _format_kj_per_kg(w_t))
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
        defaults={"T": 25.0, "P": 1.0},
        default_pair_label="T y P",
    )

    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_bar = st.number_input(
                "Presión de salida P₂ [bar(a)]",
                value=8.0,
                min_value=1.0e-6,
                format="%.4f",
                key="comp_p_out",
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
            defaults={"T": 300.0, "P": 8.0},
            default_pair_label="T y P",
        )

    if st.button("Calcular", key="comp_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = compressor_direct(
                    fluid=fluid,
                    state_in=state_in,
                    p_out_Pa=bar_to_pa(float(p_out_bar)),
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
        c2.metric("Trabajo específico w_c", _format_kj_per_kg(w_c))
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
        defaults={"P": 0.10, "X": 0.0, "T": 45.81},
        default_pair_label="P y x (saturado)",
    )

    if is_direct:
        st.markdown("**Parámetros del proceso**")
        c1, c2 = st.columns(2)
        with c1:
            p_out_bar = st.number_input(
                "Presión de salida P₂ [bar(a)]",
                value=150.0,
                min_value=1.0e-6,
                format="%.4f",
                key="pump_p_out",
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
            defaults={"P": 150.0, "T": 46.0},
            default_pair_label="T y P",
        )

    if st.button("Calcular", key="pump_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            if is_direct:
                result = pump_direct(
                    fluid=fluid,
                    state_in=state_in,
                    p_out_Pa=bar_to_pa(float(p_out_bar)),
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
        c2.metric("Trabajo específico w_p", _format_kj_per_kg(w_p))
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
                # v_1 = 1 / rho_1 a partir de CoolProp (densidad del estado in).
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
                c1.metric("w_p (EOS CoolProp)", _format_kj_per_kg(w_p_real))
                c2.metric("w_p (modelo simple)", _format_kj_per_kg(w_p_simple))
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
        defaults={"T": 25.0, "P": 1.0},
        default_pair_label="T y P",
    )

    st.markdown("**Parámetros del proceso**")
    c1, c2 = st.columns(2)
    with c1:
        p_out_bar = st.number_input(
            "Presión final P_out [bar(a)]",
            value=27.0,
            min_value=1.0e-6,
            format="%.4f",
            key="poly_p_out",
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
        t_intercool_c = st.number_input(
            "Temperatura del intercooler T_ic [°C]",
            value=25.0,
            format="%.4f",
            key="poly_tic",
            help="Default = temperatura de entrada (full cooling).",
        )
    else:
        t_intercool_c = None

    if st.button("Calcular", key="poly_btn", type="primary"):
        try:
            state_in = _resolve_state(fluid, pair_in, kwargs_in)
            result = compressor_multistage(
                fluid=fluid,
                state_in=state_in,
                p_out_Pa=bar_to_pa(float(p_out_bar)),
                n_stages=int(n_stages),
                eta_s_per_stage=float(eta_s_stage),
                intercool=bool(intercool),
                t_intercool_K=(c_to_k(float(t_intercool_c)) if t_intercool_c is not None else None),
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
            _format_kj_per_kg(result.total_delta_h_real_J_per_kg),
        )
        c2.metric(
            "Δh 1 etapa equivalente",
            _format_kj_per_kg(result.delta_h_single_stage_real_J_per_kg),
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
