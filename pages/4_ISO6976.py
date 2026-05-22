"""Página 4 — ISO 6976:2016, poder calorífico de gases combustibles.

Calcula, a partir de una composición molar:

- masa molar M, factor de sumación s, compresibilidad Z y volumen molar V_m,
- calor de combustión bruto y neto (molar, másico, volumétrico),
- densidad, densidad relativa al aire e índices de Wobbe (G y N),

con propagación de incertidumbre estándar bajo matriz **identidad** (las
mediciones de composición se asumen independientes). La opción de matriz
de normalización (ISO 14912:2003) queda deferida a fase futura.

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Matriz de correlación de normalización vía ISO 14912:2003 Formula (69).
- Expansor `📖 Fórmulas teóricas` con link a vademecum-termo.
- Botón de exportar resultados (CSV / JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.combustion.iso6976 import (
    GasComponent,
    ISO6976Inputs,
    ISO6976Result,
    ISO6976Tables,
    Quantity,
    ReferenceCondition,
    calculate,
    load_tables,
)
from ui.branding import SUBJECT, sidebar_credits

PAGE_VERSION = "0.6.0"

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "iso6976_annex_d.json"

_T_COMBUSTION_OPTIONS = (0.0, 15.0, 15.55, 20.0, 25.0)
_T_METERING_OPTIONS = (0.0, 15.0, 15.55, 20.0)


# ---------------------------------------------------------------------
# Carga cacheada de tablas y de los ejemplos del Annex D
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_tables_cached() -> ISO6976Tables:
    return load_tables()


@st.cache_data(show_spinner=False)
def _load_annex_d() -> dict[str, Any]:
    if not _FIXTURE_PATH.exists():
        return {}
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _composition_from_example(example_key: str) -> pd.DataFrame:
    """Construye un DataFrame con la composición de un ejemplo del Annex D."""
    annex = _load_annex_d()
    if not annex or example_key not in annex:
        # Fallback: composición vacía con 1 fila placeholder.
        return pd.DataFrame({"componente": ["methane"], "x": [1.0], "u_x": [0.0]})
    spec = annex[example_key]["inputs"]["composition_mol_fraction"]
    return pd.DataFrame(
        {
            "componente": [c["name"] for c in spec],
            "x": [c["x"] for c in spec],
            "u_x": [c["u_x"] for c in spec],
        }
    )


def _references_from_example(example_key: str) -> tuple[float, float, float, float]:
    """Devuelve (T_comb, P_comb, T_meas, P_meas) del ejemplo."""
    annex = _load_annex_d()
    if not annex or example_key not in annex:
        return 15.0, 101.325, 15.0, 101.325
    ex = annex[example_key]
    if example_key == "example_3":
        # Ejemplo 3 tiene 4 sub-casos; usamos el caso A (15/15, identity) por default.
        case = ex["cases"]["case_a_iso_standard_identity"]
        c_ref = case["combustion_reference"]
        m_ref = case["metering_reference"]
    else:
        c_ref = ex["inputs"]["combustion_reference"]
        m_ref = ex["inputs"]["metering_reference"]
    return c_ref["T_celsius"], c_ref["P_kPa"], m_ref["T_celsius"], m_ref["P_kPa"]


# ---------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------


def _quantity_row(label: str, q: Quantity, unit: str) -> dict[str, Any]:
    return {
        "magnitud": label,
        "valor": q.value,
        "u (k=1)": q.u,
        "U (k=2)": q.U_k2,
        "unidad": unit,
    }


def _render_results_table(result: ISO6976Result) -> None:
    inter_rows = [
        _quantity_row("M (masa molar)", result.molar_mass_kg_per_kmol, "kg/kmol"),
        _quantity_row("s (factor de sumación)", result.summation_factor, "—"),
        _quantity_row("Z (compresibilidad)", result.compression_factor, "—"),
        _quantity_row("V_m (volumen molar)", result.molar_volume_m3_per_mol, "m³/mol"),
    ]
    st.markdown("**Intermedios:**")
    df_inter = pd.DataFrame(inter_rows)
    fmt_inter = {"valor": "{:.8g}", "u (k=1)": "{:.6g}", "U (k=2)": "{:.6g}"}
    st.dataframe(df_inter.style.format(fmt_inter), use_container_width=True)

    final_rows = [
        _quantity_row("H_c,G (bruto, molar)", result.Hc_G_molar_kJ_per_mol, "kJ/mol"),
        _quantity_row("H_c,N (neto, molar)", result.Hc_N_molar_kJ_per_mol, "kJ/mol"),
        _quantity_row("H_m,G (bruto, másico)", result.Hm_G_mass_MJ_per_kg, "MJ/kg"),
        _quantity_row("H_m,N (neto, másico)", result.Hm_N_mass_MJ_per_kg, "MJ/kg"),
        _quantity_row("H_v,G (bruto, volumétrico)", result.Hv_G_volume_MJ_per_m3, "MJ/m³"),
        _quantity_row("H_v,N (neto, volumétrico)", result.Hv_N_volume_MJ_per_m3, "MJ/m³"),
        _quantity_row("ρ (densidad)", result.density_kg_per_m3, "kg/m³"),
        _quantity_row("d (densidad relativa al aire)", result.relative_density, "—"),
        _quantity_row("W_G (Wobbe bruto)", result.Wobbe_gross_MJ_per_m3, "MJ/m³"),
        _quantity_row("W_N (Wobbe neto)", result.Wobbe_net_MJ_per_m3, "MJ/m³"),
    ]
    st.markdown("**Resultados:**")
    df_final = pd.DataFrame(final_rows)
    st.dataframe(
        df_final.style.format({"valor": "{:.6g}", "u (k=1)": "{:.6g}", "U (k=2)": "{:.6g}"}),
        use_container_width=True,
    )


def _render_procedure(result: ISO6976Result) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.markdown("**Fórmulas aplicadas:**")
        st.latex(result.steps.formula_latex)
        st.markdown("**Con los valores ingresados:**")
        st.latex(result.steps.substituted_latex)
        st.markdown("**En palabras:**")
        st.write(result.steps.narrative_es)


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="ISO 6976", page_icon="🔥", layout="wide")

st.subheader(SUBJECT)
st.title("🔥 ISO 6976:2016 — Poder calorífico de gases combustibles")
st.markdown(
    "Calcula poder calorífico (bruto y neto), densidad, densidad relativa "
    "al aire e índice de Wobbe a partir de la composición molar y las "
    "condiciones de referencia, con propagación de incertidumbre estándar."
)
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="ISO 6976")

# Cargo las tablas y el fixture de ejemplos.
tables = _load_tables_cached()
known_components = sorted(tables.components.keys())

# Selector de ejemplo precargado.
example_options = {
    "Personalizado (componer manualmente)": None,
    "Annex D.2 — 5 componentes (15/15 °C)": "example_1",
    "Annex D.3 — 5 componentes con vapor de agua (15.55/15.55 °C)": "example_2",
    "Annex D.4 — 11 componentes (15/15 °C, caso A)": "example_3",
}
example_label = st.selectbox(
    "Ejemplo precargado",
    list(example_options.keys()),
    index=1,
    key="iso_example_label",
    help=(
        "Carga la composición y condiciones de referencia de uno de los "
        "ejemplos del Annex D de la norma. 'Personalizado' empieza con "
        "una sola fila editable."
    ),
)
example_key = example_options[example_label]

# Si cambió el ejemplo, reseteo los inputs y el resultado.
sig = str(example_label)
if st.session_state.get("iso_example_sig") != sig:
    st.session_state["iso_example_sig"] = sig
    for k in (
        "iso_result",
        "iso_composition_editor",
    ):
        st.session_state.pop(k, None)
    if example_key is not None:
        T_c_default, P_c_default, T_m_default, P_m_default = _references_from_example(example_key)
    else:
        T_c_default, P_c_default, T_m_default, P_m_default = 15.0, 101.325, 15.0, 101.325
    st.session_state["iso_T_c"] = T_c_default
    st.session_state["iso_P_c"] = P_c_default
    st.session_state["iso_T_m"] = T_m_default
    st.session_state["iso_P_m"] = P_m_default

# Composition editor.
st.markdown("### Composición molar")
default_df = (
    _composition_from_example(example_key)
    if example_key is not None
    else pd.DataFrame({"componente": ["methane"], "x": [1.0], "u_x": [0.0]})
)
edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "componente": st.column_config.SelectboxColumn(
            "Componente",
            options=known_components,
            required=True,
            help="Uno de los 60 componentes tabulados en ISO 6976:2016 Tabla 1.",
        ),
        "x": st.column_config.NumberColumn(
            "Fracción molar x",
            min_value=0.0,
            max_value=1.0,
            format="%.6f",
            required=True,
        ),
        "u_x": st.column_config.NumberColumn(
            "u(x) absoluto",
            min_value=0.0,
            format="%.6f",
            default=0.0,
        ),
    },
    key=f"iso_composition_editor_{sig}",
)

# Limpio filas con valores no numéricos.
edited_clean = edited_df.dropna(subset=["componente", "x"]).copy()
edited_clean["x"] = pd.to_numeric(edited_clean["x"], errors="coerce")
edited_clean["u_x"] = pd.to_numeric(edited_clean["u_x"], errors="coerce").fillna(0.0)
edited_clean = edited_clean.dropna(subset=["x"])

total_x = float(edited_clean["x"].sum()) if not edited_clean.empty else 0.0
col_a, col_b = st.columns([3, 1])
col_a.caption(
    f"Σ xⱼ = {total_x:.6f}. La composición debe sumar exactamente 1 "
    f"(tolerancia 1e-6). La norma no permite auto-normalización."
)
col_b.metric("Σ xⱼ", f"{total_x:.6f}")

# Condiciones de referencia.
st.markdown("### Condiciones de referencia")
c_comb, c_meas = st.columns(2)
with c_comb:
    st.markdown("**Combustión**")
    T_c_idx = (
        _T_COMBUSTION_OPTIONS.index(st.session_state["iso_T_c"])
        if st.session_state["iso_T_c"] in _T_COMBUSTION_OPTIONS
        else 1
    )
    T_c = st.selectbox(
        "T_c [°C]",
        _T_COMBUSTION_OPTIONS,
        index=T_c_idx,
        key="iso_T_c_sel",
        help="Temperatura de referencia para la combustión (Tabla 3 de ISO 6976:2016).",
    )
    P_c = st.number_input(
        "P_c [kPa]",
        value=float(st.session_state["iso_P_c"]),
        min_value=1.0e-3,
        format="%.4f",
        key="iso_P_c_in",
    )
with c_meas:
    st.markdown("**Medición**")
    T_m_idx = (
        _T_METERING_OPTIONS.index(st.session_state["iso_T_m"])
        if st.session_state["iso_T_m"] in _T_METERING_OPTIONS
        else 1
    )
    T_m = st.selectbox(
        "T_m [°C]",
        _T_METERING_OPTIONS,
        index=T_m_idx,
        key="iso_T_m_sel",
        help="Temperatura de referencia para la medición (Tabla 2 + Annex A de ISO 6976:2016).",
    )
    P_m = st.number_input(
        "P_m [kPa]",
        value=float(st.session_state["iso_P_m"]),
        min_value=1.0e-3,
        format="%.4f",
        key="iso_P_m_in",
    )

# Matriz de correlación — solo identity por ahora.
st.markdown("### Matriz de correlación de composición")
st.radio(
    "Modelo",
    ["identity"],
    index=0,
    key="iso_corr",
    help="Mediciones de composición tratadas como independientes.",
    horizontal=True,
)
st.info(
    "ℹ️ La opción de **matriz de normalización** (ISO 14912:2003) está prevista "
    "para una fase futura. La matriz identidad da una sobre-estimación segura "
    "(_safe overestimate_) de la incertidumbre — la propia norma ISO 6976 "
    "§D.4.3.2 lo confirma."
)

if st.button("Calcular", key="iso_btn", type="primary"):
    try:
        composition = [
            GasComponent(name=str(row["componente"]), x=float(row["x"]), u_x=float(row["u_x"]))
            for _, row in edited_clean.iterrows()
        ]
        inputs = ISO6976Inputs(
            composition=composition,
            combustion_reference=ReferenceCondition(T_celsius=float(T_c), P_kPa=float(P_c)),
            metering_reference=ReferenceCondition(T_celsius=float(T_m), P_kPa=float(P_m)),
            correlation_matrix="identity",
        )
        result = calculate(inputs, tables=tables)
    except (ValueError, NotImplementedError) as exc:
        st.error(f"Error: {exc}")
        st.session_state.pop("iso_result", None)
    else:
        st.session_state["iso_result"] = result

if "iso_result" in st.session_state:
    result = st.session_state["iso_result"]
    assert isinstance(result, ISO6976Result)
    _render_results_table(result)
    _render_procedure(result)
