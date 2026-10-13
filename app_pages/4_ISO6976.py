"""Página 4 — ISO 6976:2016, poder calorífico de gases combustibles.

Calcula, a partir de una composición molar:

- masa molar M, factor de sumación s, compresibilidad Z y volumen molar V_m,
- calor de combustión bruto y neto (molar, másico, volumétrico),
- densidad, densidad relativa al aire e índices de Wobbe (G y N),

con propagación de incertidumbre estándar con la matriz **identidad** (las
u de la composición se tratan como independientes) o la de
**normalización** (la composición se normalizó a Σx = 1: las fracciones se
correlacionan; 0.30.0), con las u de la composición normalizada (las del
Anexo D, de las que se reconstruyen las del análisis crudo) o las del
análisis crudo.

Incluye el expansor `📖 Fórmulas teóricas` (vademecum §16.9 y las
fórmulas de la norma) y la descarga del resultado en CSV / JSON (Fase 1.7).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.combustion.iso6976 import (
    UNCERTAINTY_LATEX,
    GasComponent,
    ISO6976Inputs,
    ISO6976Result,
    ISO6976Tables,
    NormalizationFit,
    Quantity,
    ReferenceCondition,
    calculate,
    iso6976_to_dict,
    load_tables,
)
from core.export import dict_to_csv
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.units_ui import render_units_selector

PAGE_VERSION = "0.30.0"

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "iso6976_annex_d.json"

_T_COMBUSTION_OPTIONS = (0.0, 15.0, 15.55, 20.0, 25.0)
_T_METERING_OPTIONS = (0.0, 15.0, 15.55, 20.0)

_CORRELATIONS = {
    "identity": "Identidad (u independientes)",
    "normalization": "Normalización (ISO 14912:2003)",
}
_BASES = {
    "normalized": "De la composición normalizada (como el Anexo D)",
    "raw": "Del análisis crudo (antes de normalizar)",
}
#: Las salidas que compara la tabla de la identidad contra la normalización.
_COMPARED = (
    ("Hv_G_volume_MJ_per_m3", "H_v,G", "MJ/m³"),
    ("Hv_N_volume_MJ_per_m3", "H_v,N", "MJ/m³"),
    ("density_kg_per_m3", "ρ", "kg/m³"),
    ("relative_density", "d", "—"),
    ("Wobbe_gross_MJ_per_m3", "W_G", "MJ/m³"),
    ("Wobbe_net_MJ_per_m3", "W_N", "MJ/m³"),
)


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


def _references_from_example(
    example_key: str, case_key: str | None = None
) -> tuple[float, float, float, float]:
    """Devuelve (T_comb, P_comb, T_meas, P_meas) del ejemplo (y del caso, en el D.4)."""
    annex = _load_annex_d()
    if not annex or example_key not in annex:
        return 15.0, 101.325, 15.0, 101.325
    ex = annex[example_key]
    if example_key == "example_3":
        # Ejemplo 3 tiene 4 sub-casos; sin caso, el A (15/15, identity).
        case = ex["cases"][case_key or "case_a_iso_standard_identity"]
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
    st.dataframe(df_inter.style.format(fmt_inter), width="stretch")

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
        width="stretch",
    )


def _render_normalization(result: ISO6976Result, identity: ISO6976Result | None) -> None:
    """Las u del análisis crudo, la correlación con el componente principal y cuánto
    baja la incertidumbre respecto de la matriz identidad."""
    fit: NormalizationFit | None = result.normalization
    if fit is None:
        return
    st.markdown("#### Análisis crudo y correlaciones")
    main = max(range(len(fit.x)), key=lambda j: fit.x[j])
    rows = [
        {
            "componente": name,
            "x": x,
            "u(y) crudo": u_y,
            "u(x) normalizada": u_x,
            f"r con {fit.names[main]}": r,
        }
        for name, x, u_y, u_x, r in zip(
            fit.names, fit.x, fit.u_raw, fit.u_normalized, fit.correlation[main], strict=True
        )
    ]
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame.style.format(
            {
                "x": "{:.6f}",
                "u(y) crudo": "{:.3g}",
                "u(x) normalizada": "{:.3g}",
                f"r con {fit.names[main]}": "{:+.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=35 * (len(frame) + 1) + 3,
    )
    origin = (
        "Las u(y) del análisis crudo se reconstruyen de las u(x) que cargaste"
        if fit.basis == "normalized"
        else "Las u(x) de la composición normalizada salen de propagar la normalización "
        "desde las u(y) que cargaste"
    )
    own = (1.0 - fit.x[main]) ** 2
    share = (
        "casi toda sale de la de los demás componentes, porque la suya propia pesa"
        if own < 0.1
        else "una parte sale de la de los demás componentes; la suya propia pesa"
    )
    st.caption(
        f"{origin}: u²(xᵢ) = (1 − xᵢ)²·u²(yᵢ) + xᵢ²·Σₖ≠ᵢ u²(yₖ). La u de "
        f"{fit.names[main]} pasa de {fit.u_raw[main]:.3g} (cruda) a "
        f"{fit.u_normalized[main]:.3g}: {share} (1 − x)² = {own:.2g}."
    )
    st.caption(
        f"r es el coeficiente de correlación con {fit.names[main]}: negativo con los "
        "componentes que tienen incertidumbre propia (si suben, el principal baja) y "
        "levemente positivo con los que casi no tienen (se escalan con el mismo Σy)."
    )
    if identity is None:
        return
    st.markdown("#### ¿Cuánto cambia la incertidumbre?")
    comparison = []
    for attr, symbol, unit in _COMPARED:
        u_id = getattr(identity, attr).u
        u_norm = getattr(result, attr).u
        comparison.append(
            {
                "magnitud": symbol if unit == "—" else f"{symbol} [{unit}]",
                "cambio": f"{100.0 * (u_norm / u_id - 1.0):+.0f} %" if u_id > 0 else "—",
                "u identidad": u_id,
                "u normalización": u_norm,
            }
        )
    st.dataframe(
        pd.DataFrame(comparison).style.format(
            {"u identidad": "{:.6g}", "u normalización": "{:.6g}"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Las correlaciones negativas se compensan: si sube el etano, baja el metano, y el "
        "poder calorífico cambia menos que si los dos errores fueran independientes. La "
        "identidad las ignora y por eso sobre-estima (ISO 6976:2016 §D.4.3.2)."
    )


def _render_procedure(result: ISO6976Result) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.markdown("**Fórmulas aplicadas:**")
        st.latex(result.steps.formula_latex)
        st.markdown("**Con los valores ingresados:**")
        st.latex(result.steps.substituted_latex)
        if result.steps.uncertainty_latex:
            st.markdown("**Incertidumbre de la composición:**")
            for tex in result.steps.uncertainty_latex:
                st.latex(tex)
        st.markdown("**En palabras:**")
        st.write(result.steps.narrative_es)


def _render_export(result: ISO6976Result) -> None:
    """Descarga del resultado completo (composición, referencias, u y U)."""
    st.markdown("#### 💾 Exportar")
    data = iso6976_to_dict(result)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="iso6976.csv",
        mime="text/csv",
        key="iso_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="iso6976.json",
        mime="application/json",
        key="iso_download_json",
    )


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"La definición de poder calorífico superior e inferior está en el [vademecum "
            f"de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})), **§16.9 "
            "*Poder calorífico***. El bruto (G) de la norma es el superior (PCS: el agua de "
            "combustión condensa) y el neto (N) es el inferior (PCI: sale como vapor):"
        )
        st.latex(
            r"\overline{PCS} - \overline{PCI} = \nu_{\mathrm{H_2O}}\,\bar{h}_{fg}(T_{\mathrm{ref}})"
        )
        st.markdown("**Cálculo a partir de la composición** (ISO 6976:2016):")
        st.latex(
            r"\begin{aligned}M &= \sum_j x_j\,M_j \\ s &= \sum_j x_j\,s_j(T_m) \\"
            r"Z &= 1 - s^2 \\ V_m &= \frac{Z\,R\,T_m}{p_m}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}H_{c,G} &= \sum_j x_j\,H_{c,j}(T_c) \\"
            r"H_{c,N} &= H_{c,G} - \Big(\sum_j x_j\,\frac{b_j}{2}\Big)\,L_0(T_c)\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}H_v &= \frac{H_c}{V_m} \\"
            r"d &= \frac{M\,Z_{\mathrm{aire}}}{M_{\mathrm{aire}}\,Z} \\"
            r"W &= \frac{H_v}{\sqrt{d}}\end{aligned}"
        )
        st.markdown(
            "b_j es el número de átomos de hidrógeno del componente j (cada 2 H forman una "
            "molécula de agua) y L₀ el calor latente molar del agua."
        )
        st.markdown(
            "**Incertidumbre** (ISO 6976:2016 §8 y Anexo B): cada salida Y suma los aportes "
            "de la composición y de los datos de las tablas p (s_j, H_c,j, las masas "
            "atómicas, L₀…), con c = ∂Y/∂x y U = k·u con k = 2 (≈ 95 % de cobertura). Con "
            "la matriz **identidad** las u(x_j) son independientes:"
        )
        st.latex(UNCERTAINTY_LATEX["identity"][0])
        st.markdown(
            "Con la de **normalización** (ISO 14912:2003), la composición salió de "
            "normalizar las fracciones medidas, x_j = y_j/Σy_k, y la covarianza sale de las "
            "incertidumbres u(y) del análisis crudo:"
        )
        for tex in UNCERTAINTY_LATEX["normalization"]:
            st.latex(tex)
        st.markdown(
            "La última ecuación es la diagonal: con las u(x) ya normalizadas (como las "
            "publica el Anexo D) la app despeja las u(y) ≥ 0. No todas las u(x) pueden "
            "salir de una normalización: ninguna fracción puede tener menos incertidumbre "
            "que x_i·√(Σ u²(y_k)) de las demás, y con dos componentes las dos u(x) son "
            "iguales."
        )


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
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="ISO 6976")
render_units_selector()

st.info(
    "ℹ️ Esta página usa las unidades de la norma ISO 6976:2016 "
    "(kJ/mol, MJ/kg, MJ/m³, kg/m³) independientemente del sistema global "
    "de unidades seleccionado en el sidebar. La conversión a sistema "
    "inglés (Btu/lb·mol, etc.) según Annex C de la norma queda pendiente "
    "para fase futura."
)

# Cargo las tablas y el fixture de ejemplos.
tables = _load_tables_cached()
known_components = sorted(tables.components.keys())

# Selector de ejemplo precargado.
example_options: dict[str, tuple[str, str | None] | None] = {
    "Personalizado (componer manualmente)": None,
    "Annex D.2 — 5 componentes (15/15 °C)": ("example_1", None),
    "Annex D.3 — 5 componentes con vapor de agua (15.55/15.55 °C)": ("example_2", None),
    "Annex D.4 — 11 componentes (15/15 °C, caso A)": ("example_3", None),
    "Annex D.4 — 11 componentes (15/15 °C, caso B: normalización)": (
        "example_3",
        "case_b_iso_standard_normalization",
    ),
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
example_choice = example_options[example_label]
example_key = example_choice[0] if example_choice is not None else None
example_case = example_choice[1] if example_choice is not None else None

# Si cambió el ejemplo, reseteo los inputs y el resultado.
sig = str(example_label)
if st.session_state.get("iso_example_sig") != sig:
    st.session_state["iso_example_sig"] = sig
    for k in (
        "iso_result",
        "iso_result_identity",
        "iso_composition_editor",
    ):
        st.session_state.pop(k, None)
    if example_key is not None:
        T_c_default, P_c_default, T_m_default, P_m_default = _references_from_example(
            example_key, example_case
        )
    else:
        T_c_default, P_c_default, T_m_default, P_m_default = 15.0, 101.325, 15.0, 101.325
    st.session_state["iso_T_c"] = T_c_default
    st.session_state["iso_P_c"] = P_c_default
    st.session_state["iso_T_m"] = T_m_default
    st.session_state["iso_P_m"] = P_m_default
    # El caso B del D.4 es el de la matriz de normalización (con las u normalizadas).
    st.session_state["iso_corr"] = "normalization" if example_case else "identity"
    st.session_state["iso_basis"] = "normalized"

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
    width="stretch",
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

# Matriz de correlación de la composición.
st.markdown("### Matriz de correlación de composición")
correlation = st.radio(
    "Modelo",
    list(_CORRELATIONS),
    format_func=_CORRELATIONS.get,
    key="iso_corr",
    help=(
        "Identidad: las u(x) se tratan como independientes. Normalización: la "
        "composición se normalizó a Σx = 1, así que las fracciones se correlacionan."
    ),
    horizontal=True,
)
basis = "normalized"
if correlation == "normalization":
    basis = st.radio(
        "Las u de la composición son…",
        list(_BASES),
        format_func=_BASES.get,
        key="iso_basis",
        help=(
            "Las del Anexo D son las de la composición ya normalizada: la app reconstruye "
            "las del análisis crudo. Si tenés las del cromatógrafo antes de normalizar, "
            "elegí la segunda."
        ),
    )
    st.info(
        "ℹ️ Al normalizar (xⱼ = yⱼ/Σyₖ), si una fracción sube, las demás bajan: las "
        "fracciones quedan **correlacionadas**, sobre todo negativamente con el "
        "componente principal, y esas correlaciones se compensan en el poder "
        "calorífico y la densidad. La norma calcula la matriz con ISO 14912:2003 desde "
        "las incertidumbres del análisis crudo; si cargás las de la composición "
        "normalizada, la app reconstruye las crudas (con las del Anexo D, exactas)."
    )
else:
    st.info(
        "ℹ️ La matriz **identidad** trata las u(x) como independientes e ignora que la "
        "composición suma 1. Da una sobre-estimación segura (_safe overestimate_) de la "
        "incertidumbre — la propia norma ISO 6976 §D.4.3.2 lo confirma. Con la de "
        "**normalización** la incertidumbre baja (en el ejemplo D.4, 39 % la de H_v,G)."
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
            correlation_matrix=correlation,
            uncertainty_basis=basis,
        )
        result = calculate(inputs, tables=tables)
        identity = (
            calculate(dataclasses.replace(inputs, correlation_matrix="identity"), tables=tables)
            if correlation == "normalization"
            else None
        )
    except ValueError as exc:
        st.error(f"Error: {exc}")
        st.session_state.pop("iso_result", None)
        st.session_state.pop("iso_result_identity", None)
    else:
        st.session_state["iso_result"] = result
        st.session_state["iso_result_identity"] = identity

if "iso_result" in st.session_state:
    result = st.session_state["iso_result"]
    assert isinstance(result, ISO6976Result)
    _render_results_table(result)
    if result.normalization is not None:
        _render_normalization(result, st.session_state.get("iso_result_identity"))
    _render_procedure(result)
    _render_export(result)
