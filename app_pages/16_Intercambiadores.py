"""Página 16 — Intercambiadores de calor.

Fase 8.2. Cuatro modos (Cengel y Ghajar, cap. 11; Incropera, cap. 11):

- **Verificación** (ε-NTU): con el intercambiador conocido (U y A), el calor y
  las temperaturas de salida.
- **Dimensionamiento**: con lo que tiene que hacer (una salida o Q̇), el área y el
  largo de tubo, por ε-NTU y por la LMTD con F (dan lo mismo).
- **Ensayo con las cuatro temperaturas**: con U, el calor y los caudales; con un
  caudal, el U.
- **Coeficiente global U**: tubo o pared plana, con ensuciamiento.

Doble tubo (paralelo y contracorriente), casco y tubos (1 a 4 pasos de casco) y
flujo cruzado; c_p dado o de CoolProp; una corriente que cambia de fase. La
exergía destruida y el rendimiento exergético siguen al vademecum (§11.10 y
§13.2).

El cálculo vive en :mod:`core.heat_transfer.exchangers`; los gráficos, en
:mod:`ui.exchanger_charts`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.fluids import FLUID_NAMES_ES
from core.heat_transfer import exchangers as hx
from core.heat_transfer.convection import convection_fluids
from core.heat_transfer.exchangers_procedure import EPS_FORMULAS, exchanger_steps, overall_u_steps
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.exchanger_charts import (
    effectiveness_figure,
    f_factor_figure,
    profile_figure,
    type_comparison_figure,
    u_resistance_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.25.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_RATING = "Verificación: el intercambiador ya está (ε-NTU)"
_M_SIZING = "Dimensionamiento: el área que hace falta"
_M_TEST = "Ensayo: las cuatro temperaturas"
_M_U = "El coeficiente global U"
_MODES = (_M_RATING, _M_SIZING, _M_TEST, _M_U)

_TARGETS: dict[hx.SizingTarget, str] = {
    "T_hot_out": "La salida del caliente",
    "T_cold_out": "La salida del frío",
    "Q": "El calor Q̇",
}
_TEST_U = "U (salen Q̇ y los caudales)"
_TEST_FLOW = "Un caudal (sale U)"
_TEST_GIVEN = (_TEST_U, _TEST_FLOW)
_SIDES: dict[hx.Side, str] = {"hot": "el caliente", "cold": "el frío"}
_GEOMETRIES: dict[hx.WallGeometry, str] = {
    "tube": "Un tubo (doble tubo, casco y tubos)",
    "plane": "Una pared plana (placas)",
}
_CP_GIVEN = "cp"
_OTHER = "Otro valor"

_W: QuantityKind = "heat_rate"
_T: QuantityKind = "temperature"
_DT: QuantityKind = "temperature_difference"
_H: QuantityKind = "heat_transfer_coefficient"
_C: QuantityKind = "heat_capacity_rate"


# ---------------------------------------------------------------------
# Formato y utilidades
# ---------------------------------------------------------------------


def _num(value: float | None, sig: int = 5) -> str:
    if value is None:
        return "—"
    if abs(value) >= 10**sig:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, sig)


def _val(value_si: float | None, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    return "—" if value_si is None else _num(convert_from_si(value_si, kind, system), sig)


def _u(kind: QuantityKind, system: UnitSystem) -> str:
    return unit_label(kind, system)


def _pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100.0 * x:.{digits}f} %".replace(".", ",")


def _t_scale(system: UnitSystem) -> tuple[float, float]:
    """(escala, offset) para pasar de K a la temperatura del sistema."""
    offset = convert_from_si(0.0, _T, system)
    return convert_from_si(1.0, _T, system) - offset, offset


def _table(frame: pd.DataFrame) -> None:
    """Tabla con todas las filas a la vista (sin barra de desplazamiento)."""
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _metric_rows(items: list[tuple[str, str]], per_row: int = 2) -> None:
    for k in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, items[k : k + per_row], strict=False):
            col.metric(label, value)


def _render_steps(steps: list[ProcedureStep], caption: str = "") -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        if caption:
            st.caption(caption)
        for step in steps:
            st.markdown(f"**{step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_export(data: dict[str, Any], stem: str, key: str) -> None:
    st.markdown("#### 💾 Exportar")
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name=f"{stem}.csv",
        mime="text/csv",
        key=f"{key}_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"{stem}.json",
        mime="application/json",
        key=f"{key}_json",
    )


def _example(examples: dict[str, Any], key: str, help_text: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox("Ejemplo precargado", names, key=key, help=help_text)
    return name, names.index(name)


# ---------------------------------------------------------------------
# Datos del intercambiador y de las corrientes
# ---------------------------------------------------------------------


def _arrangement_inputs(default: hx.Arrangement, key: str) -> hx.Arrangement:
    kinds = list(hx.EXCHANGER_KINDS)
    kind: hx.ExchangerKind = st.selectbox(
        "Tipo de intercambiador",
        kinds,
        index=kinds.index(default.kind),
        format_func=lambda k: hx.EXCHANGER_KINDS[k],
        key=f"{key}_kind",
    )
    shell_passes, mixed, tubes = default.shell_passes, default.mixed, default.tubes
    sides = list(_SIDES)
    if kind == "shell":
        shell_passes = int(
            st.number_input(
                "Pasos de casco",
                min_value=1,
                max_value=4,
                value=default.shell_passes if default.kind == "shell" else 1,
                key=f"{key}_np",
                help="Con n pasos de casco (2n, 4n… pasos de tubo) el arreglo se acerca a un "
                "contracorriente.",
            )
        )
    if kind == "cross_mixed":
        mixed = st.radio(
            "El fluido mezclado (sin canales que lo guíen)",
            sides,
            index=sides.index(default.mixed),
            format_func=lambda s: _SIDES[s],
            key=f"{key}_mixed",
            horizontal=True,
        )
    if kind in ("shell", "cross_unmixed", "cross_mixed"):
        tubes = st.radio(
            "Por los tubos va",
            sides,
            index=sides.index(default.tubes),
            format_func=lambda s: _SIDES[s],
            key=f"{key}_tubes",
            horizontal=True,
            help="Solo para P y R como Cengel y Ghajar (t: el fluido de los tubos); no cambia "
            "el resultado.",
        )
    return hx.Arrangement(kind, shell_passes, mixed, tubes)


def _fluid_options() -> list[str]:
    return [_CP_GIVEN, *convection_fluids()]


def _fluid_name(f: str) -> str:
    return "c_p dado (como los libros)" if f == _CP_GIVEN else FLUID_NAMES_ES.get(f, f)


def _stream_inputs(
    title: str, default: hx.Stream, key: str, *, flow: bool, outlet: float | None = None
) -> tuple[hx.Stream, float | None]:
    """Una corriente; con ``outlet`` (el ensayo) pide también su temperatura de salida."""
    st.markdown(f"**{title}: {default.name}**")
    options = _fluid_options()
    current = (
        _CP_GIVEN if default.fluid is None or default.cp_J_per_kgK is not None else default.fluid
    )
    fluid = st.selectbox(
        "Fluido",
        options,
        index=options.index(current) if current in options else 0,
        format_func=_fluid_name,
        key=f"{key}_fluid",
    )
    phase = st.checkbox(
        "Cambia de fase (condensa o evapora a temperatura constante)",
        value=default.phase_change,
        key=f"{key}_pc",
    )
    left, right = st.columns(2)
    with left:
        T_in = number_input_si(
            label="T de saturación" if phase else "T de entrada",
            kind=_T,
            default_si=default.T_in_K,
            key=f"{key}_Tin",
            format="%.5g",
        )
    T_out = None
    if outlet is not None and not phase:
        with right:
            T_out = number_input_si(
                label="T de salida", kind=_T, default_si=outlet, key=f"{key}_Tout", format="%.5g"
            )
    m_dot = default.m_dot_kg_s
    if flow and not phase:
        with left if outlet is not None else right:
            m_dot = number_input_si(
                label="Caudal ṁ",
                kind="mass_flow",
                default_si=default.m_dot_kg_s or 1.0,
                key=f"{key}_m",
                format="%.5g",
                min_value_si=0.0,
            )
    cp = h_fg = None
    p = default.p_Pa
    if fluid == _CP_GIVEN:
        if phase:
            h_fg = number_input_si(
                label="Calor latente h_fg",
                kind="specific_enthalpy",
                default_si=default.h_fg_J_per_kg or 2_257_000.0,
                key=f"{key}_hfg",
                format="%.5g",
                min_value_si=0.0,
            )
        else:
            cp = number_input_si(
                label="Calor específico c_p",
                kind="specific_heat",
                default_si=default.cp_J_per_kgK or 4180.0,
                key=f"{key}_cp",
                format="%.5g",
                min_value_si=0.0,
            )
        name_fluid = None
    else:
        name_fluid = fluid
        if not phase:
            p = number_input_si(
                label="Presión",
                kind="pressure",
                default_si=default.p_Pa,
                key=f"{key}_p",
                format="%.5g",
                min_value_si=0.0,
                help="c_p sale de CoolProp a esta presión y a la temperatura media de la "
                "corriente; el fluido tiene que quedar en una sola fase.",
            )
        else:
            st.caption("h_fg y la presión de saturación salen de CoolProp a esa temperatura.")
    stream = hx.Stream(
        default.name, T_in, m_dot if not phase else 0.0, cp, name_fluid, p, phase, h_fg
    )
    return stream, T_out


def _T0_input(default: float, key: str) -> float:
    return number_input_si(
        label="T₀ del ambiente (para la exergía)",
        kind=_T,
        default_si=default,
        key=f"{key}_T0",
        format="%.5g",
        help="El estado muerto del vademecum (§11.1): la exergía destruida es T₀·Ṡ_gen.",
    )


def _U_input(default: float, key: str) -> float:
    return number_input_si(
        label="Coeficiente global U",
        kind=_H,
        default_si=default,
        key=f"{key}_U",
        format="%.5g",
        min_value_si=0.0,
        help="Sale del modo «El coeficiente global U» (Cengel y Ghajar, tabla 11-1, da valores "
        "típicos).",
    )


# ---------------------------------------------------------------------
# Los tres problemas
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _rating_cached(inputs: hx.RatingInputs) -> hx.ExchangerResult:
    return hx.solve_rating(inputs)


@st.cache_data(show_spinner=False)
def _sizing_cached(inputs: hx.SizingInputs) -> hx.ExchangerResult:
    return hx.solve_sizing(inputs)


@st.cache_data(show_spinner=False)
def _test_cached(inputs: hx.FourTemperatureInputs) -> hx.ExchangerResult:
    return hx.solve_four_temperatures(inputs)


@st.cache_data(show_spinner=False)
def _comparison_cached(result: hx.ExchangerResult) -> list[hx.TypeComparison]:
    return hx.type_comparison(result)


def _rating(system: UnitSystem) -> None:
    name, idx = _example(
        hx.RATING_EXAMPLES,
        "xv_example",
        "Ejemplos del cap. 11 de Cengel y Ghajar. Podés cambiar cualquier dato: el resultado se "
        "actualiza solo.",
    )
    ex = hx.RATING_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, hx.RatingInputs)
    key = f"xv_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    arr = _arrangement_inputs(inp.arrangement, key)
    hot, _ = _stream_inputs("Fluido caliente", inp.hot, f"{key}_h", flow=True)
    cold, _ = _stream_inputs("Fluido frío", inp.cold, f"{key}_c", flow=True)
    left, right = st.columns(2)
    with left:
        U = _U_input(inp.U_W_per_m2K, key)
    with right:
        A = number_input_si(
            label="Área A",
            kind="area",
            default_si=inp.A_m2,
            key=f"{key}_A",
            format="%.5g",
            min_value_si=0.0,
        )
    T0 = _T0_input(inp.T0_K, key)
    try:
        result = _rating_cached(hx.RatingInputs(arr, hot, cold, U, A, T0))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_exchanger(result, system, "xv")


def _sizing(system: UnitSystem) -> None:
    name, idx = _example(
        hx.SIZING_EXAMPLES,
        "xd_example",
        "Ejemplos de Cengel y Ghajar e Incropera. Podés cambiar cualquier dato.",
    )
    ex = hx.SIZING_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, hx.SizingInputs)
    key = f"xd_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    arr = _arrangement_inputs(inp.arrangement, key)
    hot, _ = _stream_inputs("Fluido caliente", inp.hot, f"{key}_h", flow=True)
    cold, _ = _stream_inputs("Fluido frío", inp.cold, f"{key}_c", flow=True)
    targets = list(_TARGETS)
    target: hx.SizingTarget = st.radio(
        "¿Qué tiene que hacer?",
        targets,
        index=targets.index(inp.target),
        format_func=lambda t: _TARGETS[t],
        key=f"{key}_target",
        horizontal=True,
    )
    left, right = st.columns(2)
    with left:
        if target == "Q":
            value = number_input_si(
                label="Calor Q̇",
                kind=_W,
                default_si=inp.target_value if inp.target == "Q" else 100_000.0,
                key=f"{key}_Q",
                format="%.5g",
                min_value_si=0.0,
            )
        else:
            stream = hot if target == "T_hot_out" else cold
            guess = stream.T_in_K + (-20.0 if target == "T_hot_out" else 20.0)
            value = number_input_si(
                label="T de salida pedida",
                kind=_T,
                default_si=inp.target_value if inp.target == target else guess,
                key=f"{key}_{target}",
                format="%.5g",
            )
    with right:
        U = _U_input(inp.U_W_per_m2K, key)
    D = None
    if st.checkbox(
        "El largo de tubo (con su diámetro)", value=inp.tube_D_m is not None, key=f"{key}_tube"
    ):
        D = number_input_si(
            label="Diámetro del tubo D",
            kind="small_length",
            default_si=inp.tube_D_m or 0.02,
            key=f"{key}_D",
            format="%.4g",
            min_value_si=0.0,
            help="El de la superficie a la que se refiere U: L = A/(π·D).",
        )
    T0 = _T0_input(inp.T0_K, key)
    try:
        result = _sizing_cached(hx.SizingInputs(arr, hot, cold, U, target, value, D, T0))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_exchanger(result, system, "xd")


def _test(system: UnitSystem) -> None:
    name, idx = _example(
        hx.TEST_EXAMPLES,
        "xe_example",
        "Ejemplos del cap. 11 de Cengel y Ghajar (11-3, 11-5 y 11-6). Podés cambiar cualquier "
        "dato.",
    )
    ex = hx.TEST_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, hx.FourTemperatureInputs)
    key = f"xe_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    arr = _arrangement_inputs(inp.arrangement, key)
    given = st.radio(
        "¿Qué es dato, además de las temperaturas?",
        _TEST_GIVEN,
        index=0 if inp.U_W_per_m2K is not None else 1,
        key=f"{key}_given",
        horizontal=True,
    )
    known: hx.Side = inp.known_flow
    if given == _TEST_FLOW:
        sides = list(_SIDES)
        known = st.radio(
            "El caudal conocido es el de",
            sides,
            index=sides.index(inp.known_flow),
            format_func=lambda s: _SIDES[s],
            key=f"{key}_known",
            horizontal=True,
        )
    hot, T_hot_out = _stream_inputs(
        "Fluido caliente",
        inp.hot,
        f"{key}_h",
        flow=given == _TEST_FLOW and known == "hot",
        outlet=inp.T_hot_out_K,
    )
    cold, T_cold_out = _stream_inputs(
        "Fluido frío",
        inp.cold,
        f"{key}_c",
        flow=given == _TEST_FLOW and known == "cold",
        outlet=inp.T_cold_out_K,
    )
    if given == _TEST_FLOW and known == "hot" and hot.phase_change:
        hot = replace(hot, m_dot_kg_s=_condensate_input(inp.hot, f"{key}_h"))
    if given == _TEST_FLOW and known == "cold" and cold.phase_change:
        cold = replace(cold, m_dot_kg_s=_condensate_input(inp.cold, f"{key}_c"))
    left, right = st.columns(2)
    with left:
        A = number_input_si(
            label="Área A",
            kind="area",
            default_si=inp.A_m2,
            key=f"{key}_A",
            format="%.5g",
            min_value_si=0.0,
        )
    U = None
    if given == _TEST_U:
        with right:
            U = _U_input(inp.U_W_per_m2K or 500.0, key)
    T0 = _T0_input(inp.T0_K, key)
    inputs = hx.FourTemperatureInputs(
        arr,
        hot,
        cold,
        T_hot_out if T_hot_out is not None else hot.T_in_K,
        T_cold_out if T_cold_out is not None else cold.T_in_K,
        A,
        U,
        known,
        T0,
    )
    try:
        result = _test_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_exchanger(result, system, "xe")


def _condensate_input(default: hx.Stream, key: str) -> float:
    return number_input_si(
        label="Caudal que cambia de fase",
        kind="mass_flow",
        default_si=default.m_dot_kg_s or 0.5,
        key=f"{key}_mpc",
        format="%.5g",
        min_value_si=0.0,
    )


def _stream_table(r: hx.ExchangerResult, system: UnitSystem) -> None:
    rows = []
    for sr, who in ((r.hot, "caliente"), (r.cold, "frío")):
        fluid = FLUID_NAMES_ES.get(sr.stream.fluid or "", "—")
        rows.append(
            {
                "Corriente": f"{who}: {sr.name}",
                "Fluido": fluid if sr.stream.cp_J_per_kgK is None else "c_p dado",
                f"T ent. [{_u(_T, system)}]": _val(sr.T_in_K, _T, system),
                f"T sal. [{_u(_T, system)}]": _val(sr.T_out_K, _T, system),
                f"ṁ [{_u('mass_flow', system)}]": _val(sr.m_dot_kg_s, "mass_flow", system),
                f"c_p [{_u('specific_heat', system)}]": _val(
                    sr.cp_J_per_kgK, "specific_heat", system
                ),
                f"C [{_u(_C, system)}]": "∞" if sr.phase_change else _val(sr.C_W_per_K, _C, system),
            }
        )
    _table(pd.DataFrame(rows))


def _render_exchanger(r: hx.ExchangerResult, system: UnitSystem, key: str) -> None:
    st.markdown("### Resultado")
    items = [(f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system))]
    if r.mode == "sizing":
        items.append((f"Área A [{_u('area', system)}]", _val(r.A_m2, "area", system)))
        if r.tube_length_m is not None:
            items.append(
                (f"Largo de tubo [{_u('length', system)}]", _val(r.tube_length_m, "length", system))
            )
    if r.mode == "test" and r.inputs.U_W_per_m2K is None:  # type: ignore[union-attr]
        items.append((f"U [{_u(_H, system)}]", _val(r.U_W_per_m2K, _H, system)))
    for sr, who in ((r.hot, "caliente"), (r.cold, "frío")):
        if sr.phase_change:
            items.append(
                (
                    f"ṁ que cambia de fase [{_u('mass_flow', system)}]",
                    _val(sr.m_dot_kg_s, "mass_flow", system),
                )
            )
        elif r.mode != "test":
            items.append((f"Salida del {who} [{_u(_T, system)}]", _val(sr.T_out_K, _T, system)))
        elif r.inputs.U_W_per_m2K is not None or r.inputs.known_flow != (
            "hot" if sr is r.hot else "cold"
        ):  # type: ignore[union-attr]
            items.append(
                (
                    f"ṁ del {who} [{_u('mass_flow', system)}]",
                    _val(sr.m_dot_kg_s, "mass_flow", system),
                )
            )
    items += [
        ("Efectividad ε", _num(r.effectiveness, 4)),
        ("NTU = UA/C_mín", _num(r.NTU, 4)),
        ("C_r = C_mín/C_máx", _num(r.C_r, 4)),
        ("Factor de corrección F", _num(r.F, 4)),
        (f"ΔT_ml [{_u(_DT, system)}]", _val(r.dT_lm_K, _DT, system)),
        (f"UA [{_u(_C, system)}]", _val(r.UA_W_per_K, _C, system)),
        (f"Exergía destruida [{_u(_W, system)}]", _val(r.exergy.X_dest_W, _W, system)),
        ("Rendimiento exergético", _pct(r.exergy.efficiency)),
    ]
    _metric_rows(items)
    warnings = set(r.warnings)
    for note in hx.exchanger_notes(r):
        (st.warning if note in warnings else st.info)(note)
    _stream_table(r, system)
    t_scale, t_offset = _t_scale(system)
    try:
        fig = profile_figure(r, t_scale=t_scale, t_offset=t_offset, t_unit=_u(_T, system))
        if fig is not None:
            st.markdown("#### Las temperaturas a lo largo del intercambiador")
            st.plotly_chart(fig, width="stretch", key=f"{key}_profile")
            st.caption(
                "En contracorriente la diferencia de temperaturas es más pareja (y el frío puede "
                "salir más caliente que el caliente); en paralelo es grande a la entrada y se "
                "achica hasta casi cerrarse."
            )
    except Exception as exc:  # noqa: BLE001 — un gráfico que falla no tumba la página
        st.warning(f"No se pudo dibujar el perfil: {exc}")
    st.markdown("#### ε–NTU")
    try:
        st.plotly_chart(effectiveness_figure(r), width="stretch", key=f"{key}_eps")
        st.caption(
            "El gráfico de los libros para este tipo (Cengel y Ghajar, fig. 11-26): ε crece con "
            "NTU y se aplana; con C_r más chico se llega más alto. El punto naranja son los "
            "datos."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar ε–NTU: {exc}")
    try:
        fig = f_factor_figure(r)
        if fig is not None:
            st.markdown("#### El factor de corrección F")
            st.plotly_chart(fig, width="stretch", key=f"{key}_F")
            st.caption(
                "F(P, R) como los gráficos de Bowman, Mueller y Nagle (1940): con P cerca del "
                "máximo F cae a pique. Por debajo de 0,75 el arreglo desperdicia área."
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar F: {exc}")
    _render_comparison(r, system, key)
    _render_steps(
        exchanger_steps(r, system),
        "Las capacidades, ε y NTU, el calor, la LMTD con F y la exergía, con los números en el "
        "sistema elegido.",
    )
    _render_export(hx.exchanger_to_dict(r, system), "intercambiador", f"{key}_dl")


def _render_comparison(r: hx.ExchangerResult, system: UnitSystem, key: str) -> None:
    area = r.mode != "rating"
    st.markdown(
        "#### Los otros tipos, con los mismos datos"
        + (" (el área para el mismo calor)" if area else " (el calor con la misma área)")
    )
    try:
        rows = _comparison_cached(r)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo comparar: {exc}")
        return
    kind: QuantityKind = "area" if area else _W
    _table(
        pd.DataFrame(
            [
                {
                    "Tipo": c.arrangement.name
                    + (" ←" if c.arrangement.name == r.arrangement.name else ""),
                    (f"A [{_u('area', system)}]" if area else f"Q̇ [{_u(_W, system)}]"): (
                        _val(c.A_m2 if area else c.Q_W, kind, system)
                        if c.error is None
                        else "no puede"
                    ),
                    "ε": _num(c.effectiveness, 4),
                    "F": _num(c.F, 4),
                }
                for c in rows
            ]
        )
    )
    try:
        st.plotly_chart(
            type_comparison_figure(
                rows,
                r.arrangement.name,
                area=area,
                scale=convert_from_si(1.0, kind, system),
                unit=_u(kind, system),
            ),
            width="stretch",
            key=f"{key}_cmp",
        )
        st.caption(
            "En azul el tipo elegido. El contracorriente es el que menos área necesita; los que "
            "dicen «no puede» no llegan con ningún área (su ε máximo es menor)."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar la comparación: {exc}")


# ---------------------------------------------------------------------
# U global
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _overall_u_cached(inputs: hx.OverallUInputs) -> hx.OverallUResult:
    return hx.overall_u(inputs)


def _fouling_input(label: str, default: float, key: str) -> float:
    options = [*hx.FOULING_FACTORS, _OTHER]
    current = next((k for k, v in hx.FOULING_FACTORS.items() if abs(v - default) < 1e-12), _OTHER)
    choice = st.selectbox(
        label,
        options,
        index=options.index(current),
        key=f"{key}_sel",
        help="Valores representativos de TEMA (Incropera, tabla 11.1; Cengel y Ghajar, tabla "
        "11-2).",
    )
    if choice != _OTHER:
        return hx.FOULING_FACTORS[choice]
    return number_input_si(
        label="R''_f",
        kind="fouling_resistance",
        default_si=default or 0.0002,
        key=f"{key}_v",
        format="%.4g",
        min_value_si=0.0,
    )


def _overall(system: UnitSystem) -> None:
    name, idx = _example(
        hx.U_EXAMPLES,
        "xu_example",
        "El U global con ensuciamiento (Cengel y Ghajar, §11-2). Podés cambiar cualquier dato.",
    )
    ex = hx.U_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, hx.OverallUInputs)
    key = f"xu_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    geos = list(_GEOMETRIES)
    geometry: hx.WallGeometry = st.radio(
        "La pared",
        geos,
        index=geos.index(inp.geometry),
        format_func=lambda g: _GEOMETRIES[g],
        key=f"{key}_geo",
        horizontal=True,
    )
    same = geometry == inp.geometry
    left, right = st.columns(2)
    with left:
        h_i = number_input_si(
            label="h adentro (lado i)" if geometry == "tube" else "h del lado 1",
            kind=_H,
            default_si=inp.h_i_W_per_m2K,
            key=f"{key}_hi",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        h_o = number_input_si(
            label="h afuera (lado o)" if geometry == "tube" else "h del lado 2",
            kind=_H,
            default_si=inp.h_o_W_per_m2K,
            key=f"{key}_ho",
            format="%.5g",
            min_value_si=0.0,
        )
    k = number_input_si(
        label="Conductividad de la pared k",
        kind="thermal_conductivity",
        default_si=inp.k_wall_W_per_mK,
        key=f"{key}_k",
        format="%.4g",
        min_value_si=0.0,
    )
    D_i, D_o, L, t, A = inp.D_i_m, inp.D_o_m, inp.L_m, inp.thickness_m, inp.A_m2
    g_key = f"{key}_{geometry}"
    if geometry == "tube":
        left, right = st.columns(2)
        with left:
            D_i = number_input_si(
                label="Diámetro interior D_i",
                kind="small_length",
                default_si=inp.D_i_m if same else 0.02,
                key=f"{g_key}_Di",
                format="%.4g",
                min_value_si=0.0,
            )
            L = number_input_si(
                label="Largo L",
                kind="length",
                default_si=inp.L_m if same else 1.0,
                key=f"{g_key}_L",
                format="%.4g",
                min_value_si=0.0,
            )
        with right:
            D_o = number_input_si(
                label="Diámetro exterior D_o",
                kind="small_length",
                default_si=inp.D_o_m if same else 0.025,
                key=f"{g_key}_Do",
                format="%.4g",
                min_value_si=0.0,
            )
    else:
        left, right = st.columns(2)
        with left:
            t = number_input_si(
                label="Espesor t",
                kind="small_length",
                default_si=inp.thickness_m if same else 0.001,
                key=f"{g_key}_t",
                format="%.4g",
                min_value_si=0.0,
            )
        with right:
            A = number_input_si(
                label="Área A",
                kind="area",
                default_si=inp.A_m2 if same else 1.0,
                key=f"{g_key}_A",
                format="%.4g",
                min_value_si=0.0,
            )
    R_i = _fouling_input(
        "Ensuciamiento adentro" if geometry == "tube" else "Ensuciamiento del lado 1",
        inp.R_f_i_m2K_per_W,
        f"{key}_fi",
    )
    R_o = _fouling_input(
        "Ensuciamiento afuera" if geometry == "tube" else "Ensuciamiento del lado 2",
        inp.R_f_o_m2K_per_W,
        f"{key}_fo",
    )
    inputs = hx.OverallUInputs(geometry, h_i, h_o, k, D_i, D_o, L, t, A, R_i, R_o)
    try:
        result = _overall_u_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_overall(result, system)


def _render_overall(r: hx.OverallUResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    R: QuantityKind = "thermal_resistance"
    items = []
    if r.inputs.geometry == "tube":
        items += [
            (f"U_i (área de adentro) [{_u(_H, system)}]", _val(r.U_i_W_per_m2K, _H, system)),
            (f"U_o (área de afuera) [{_u(_H, system)}]", _val(r.U_o_W_per_m2K, _H, system)),
        ]
    else:
        items.append((f"U [{_u(_H, system)}]", _val(r.U_i_W_per_m2K, _H, system)))
    items += [
        (f"UA [{_u(_C, system)}]", _val(r.UA_W_per_K, _C, system)),
        (f"R total [{_u(R, system)}]", _val(r.R_total_K_per_W, R, system)),
        (f"U_i limpio [{_u(_H, system)}]", _val(r.U_i_clean_W_per_m2K, _H, system)),
    ]
    _metric_rows(items)
    for note in hx.overall_u_notes(r):
        st.info(note)
    _table(
        pd.DataFrame(
            [
                {
                    "Resistencia": res.label,
                    f"R [{_u(R, system)}]": _val(res.R_K_per_W, R, system),
                    "Del total": _pct(r.share(res)),
                }
                for res in r.resistances
            ]
        )
    )
    try:
        st.plotly_chart(u_resistance_figure(r), width="stretch", key="xu_bars")
        st.caption(
            "Las resistencias van en serie: la mayor es la que fija U (el lado del gas, si hay "
            "uno)."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar las resistencias: {exc}")
    _render_steps(overall_u_steps(r, system))
    _render_export(hx.overall_u_to_dict(r, system), "coeficiente_U", "xu_dl")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§11.10** (el rendimiento exergético de un intercambiador adiabático) y **§13.2** "
            "(Δs de un líquido). El resto sigue a **Çengel y Ghajar (2015)**, cap. 11, y a "
            "**Incropera et al. (2007)**, cap. 11."
        )
        st.markdown("**Balance y coeficiente global** (§11-2 y §11-3):")
        st.latex(
            r"\begin{aligned}\dot{Q} &= C_h\,(T_{h,\text{ent}} - T_{h,\text{sal}}) \\ "
            r"&= C_c\,(T_{c,\text{sal}} - T_{c,\text{ent}})\end{aligned}"
        )
        st.latex(r"C = \dot{m}\,c_p \qquad \dot{Q} = U\,A\,F\,\Delta T_{ml}")
        st.latex(
            r"\begin{aligned}\frac{1}{UA} &= \frac{1}{h_i A_i} + \frac{R''_{f,i}}{A_i} + "
            r"R_{\text{pared}} \\ &\quad + \frac{R''_{f,o}}{A_o} + \frac{1}{h_o A_o}\end{aligned}"
        )
        st.markdown(
            "**LMTD** (§11-4), con las diferencias de los extremos de un contracorriente; F "
            "corrige para los otros tipos con P y R (t: el fluido de los tubos):"
        )
        st.latex(r"\Delta T_{ml} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1/\Delta T_2)}")
        st.latex(r"P = \frac{t_2 - t_1}{T_1 - t_1} \qquad R = \frac{T_1 - T_2}{t_2 - t_1}")
        st.markdown("**ε-NTU** (§11-5):")
        st.latex(r"\varepsilon = \frac{\dot{Q}}{\dot{Q}_{\text{máx}}}")
        st.latex(r"\dot{Q}_{\text{máx}} = C_{\text{mín}}\,(T_{h,\text{ent}} - T_{c,\text{ent}})")
        st.latex(
            r"NTU = \frac{UA}{C_{\text{mín}}} \qquad C_r = \frac{C_{\text{mín}}}{C_{\text{máx}}}"
        )
        st.markdown(
            "Las relaciones ε(NTU, C_r) de cada tipo (Incropera, tabla 11.3; Cengel y Ghajar, "
            "tabla 11-4). Con un fluido que cambia de fase, C_r = 0 y todas dan "
            "ε = 1 − e^(−NTU). El flujo cruzado sin mezclar usa la serie exacta (Mason, 1955):"
        )
        for title, keys in (
            ("Doble tubo, flujo paralelo", ("parallel",)),
            ("Doble tubo, contracorriente", ("counter",)),
            ("Casco y tubos: un paso de casco y n pasos", ("shell", "shell_n")),
            ("Flujo cruzado, los dos sin mezclar", ("cross_unmixed",)),
            ("Flujo cruzado, C_máx mezclado", ("cross_cmax_mixed",)),
            ("Flujo cruzado, C_mín mezclado", ("cross_cmin_mixed",)),
        ):
            st.markdown(f"- {title}:")
            for k in keys:
                for tex in EPS_FORMULAS[k]:
                    st.latex(tex)
        st.markdown(
            "**F para cualquier tipo**: el cociente de los NTU de un contracorriente y del tipo "
            "para el mismo ε y C_r (coincide con Bowman, Mueller y Nagle, 1940):"
        )
        st.latex(r"F = \frac{NTU_{\text{cc}}(\varepsilon, C_r)}{NTU}")
        st.markdown(
            "**Exergía** (vademecum §13.2 y §11.10): con c_p constante, la entropía de cada "
            "corriente y la exergía destruida; el rendimiento exergético, si el frío gana "
            "exergía:"
        )
        st.latex(r"\Delta \dot{S} = C \ln\frac{T_{\text{sal}}}{T_{\text{ent}}}")
        st.latex(r"\dot{X}_{\text{dest}} = T_0\,(\Delta\dot{S}_h + \Delta\dot{S}_c)")
        st.latex(
            r"\eta_{\text{ex}} = \frac{\dot{m}_c\,(\psi_{c,\text{sal}} - \psi_{c,\text{ent}})}"
            r"{\dot{m}_h\,(\psi_{h,\text{ent}} - \psi_{h,\text{sal}})}"
        )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Intercambiadores de calor", page_icon="🔄", layout="centered")

st.subheader(SUBJECT)
st.title("🔄 Intercambiadores de calor")
st.markdown(
    "La **verificación** de un intercambiador (ε-NTU), su **dimensionamiento** (por ε-NTU y por "
    "la LMTD con el factor F), el **ensayo** con las cuatro temperaturas y el **coeficiente "
    "global U** con ensuciamiento. Doble tubo, casco y tubos y flujo cruzado; con c_p dado o de "
    "CoolProp, y con una corriente que cambia de fase. Con la exergía destruida."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Intercambiadores de calor")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="hx_mode")
if mode == _M_RATING:
    _rating(system)
elif mode == _M_SIZING:
    _sizing(system)
elif mode == _M_TEST:
    _test(system)
else:
    _overall(system)
