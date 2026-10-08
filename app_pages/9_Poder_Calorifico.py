"""Página 9 — Poder calorífico por correlaciones.

Fase 6. El PCS de un combustible sólido o líquido estimado desde su análisis
elemental (Dulong, Boie, Channiwala y Parikh) o inmediato (Parikh et al.,
Cordero et al.), comparando las correlaciones entre sí y contra el PCS medido
o el exacto (las h_f de una sustancia pura). Los datos van en cualquier base
(tal cual, seca o seca y sin cenizas); sale el PCS en las tres y el PCI
(vademecum §16.9).

«¿Qué tan buenas son?» muestra cada correlación contra las 536 biomasas de
Ghugare et al. (2014) y los carbones de Argonne (Vorres, 1990).

El cálculo vive en :mod:`core.combustion.heating_value`; los gráficos, en
:mod:`ui.heating_value_charts`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from core.combustion.heating_value import (
    ANALYSIS_KINDS,
    BASES,
    CORRELATIONS,
    FUEL_TYPES,
    HEATING_VALUE_EXAMPLE_NOTES,
    HEATING_VALUE_EXAMPLES,
    Basis,
    FuelType,
    HeatingValueInputs,
    HeatingValueResult,
    applicable_correlations,
    argonne_deviations,
    dataset_fit,
    default_moisture_values,
    element_heating_values,
    heating_value_to_dict,
    hydrogen_water_ratio,
    moisture_sweep,
    solve_heating_value,
    to_basis,
    water_hfg_J_per_kg,
)
from core.combustion.heating_value_procedure import heating_value_steps
from core.export import dict_to_csv
from core.state_report import format_value
from core.units_system import UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.combustion_charts import sweep_figure
from ui.heating_value_charts import (
    ERROR_RANGE,
    ComparisonRow,
    comparison_figure,
    error_vs_oxygen_figure,
    parity_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.22.0"

# Opciones fijas de los radios y selectores: cambiarlas reiniciaría el widget.
_AN_ULT = "Elemental (último)"
_AN_PROX = "Inmediato (próximo)"
_AN_BOTH = "Los dos"
_ANALYSES = (_AN_ULT, _AN_PROX, _AN_BOTH)
_BASIS_LIST: list[Basis] = ["ar", "d", "daf"]
_BASIS_LABELS = ["Tal cual (como se recibe)", "Base seca", "Seca y sin cenizas"]
_TYPE_LIST: list[FuelType] = list(FUEL_TYPES)
_TYPE_LABELS = [FUEL_TYPES[t] for t in _TYPE_LIST]
_NAMES = list(HEATING_VALUE_EXAMPLES)
_ULT_FIELDS = (
    ("C", "Carbono C"),
    ("H", "Hidrógeno H"),
    ("O", "Oxígeno O"),
    ("N", "Nitrógeno N"),
    ("S", "Azufre S"),
)
_PROX_FIELDS = (("VM", "Materia volátil MV"), ("FC", "Carbono fijo CF"))
#: Correlaciones elementales (las que se pueden validar con los datos de Ghugare).
_ULT_KEYS = [k for k, c in CORRELATIONS.items() if c.analysis == "ultimate"]
#: Nombre de cada fila de las tablas del análisis.
_ROW_NAMES = {
    "C": "C",
    "H": "H",
    "O": "O",
    "N": "N",
    "S": "S",
    "VM": "Materia volátil",
    "FC": "Carbono fijo",
    "A": "Cenizas",
    "W": "Humedad",
}
_EH = "specific_enthalpy"
#: Nombres cortos de las correlaciones para las columnas de las tablas (celular).
_SHORT_NAMES = {
    "channiwala_parikh": "CyP",
    "boie": "Boie",
    "dulong": "Dulong",
    "parikh": "Parikh",
    "cordero": "Cordero",
}


# ---------------------------------------------------------------------
# Formato y utilidades
# ---------------------------------------------------------------------


def _num(value: float | None, sig: int = 5) -> str:
    if value is None:
        return "—"
    if abs(value) >= 10**sig:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, sig)


def _hv(value_si: float | None, system: UnitSystem, sig: int = 5) -> str:
    """Un poder calorífico en el sistema (sin unidad)."""
    return "—" if value_si is None else _num(convert_from_si(value_si, _EH, system), sig)


def _signed_pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:+.1f} %"


def _table(frame: pd.DataFrame) -> None:
    """Tabla con todas las filas a la vista (sin barra de desplazamiento)."""
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _metric_rows(items: list[tuple[str, str]], per_row: int = 3) -> None:
    for k in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, items[k : k + per_row], strict=False):
            col.metric(label, value)


@st.cache_data(show_spinner=False)
def _solve_cached(inputs: HeatingValueInputs) -> HeatingValueResult:
    return solve_heating_value(inputs)


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------


def _default_analysis(example: HeatingValueInputs) -> str:
    if example.ultimate is not None and example.proximate is not None:
        return _AN_BOTH
    return _AN_ULT if example.ultimate is not None else _AN_PROX


def _number(label: str, value: float, key: str, help: str | None = None) -> float:  # noqa: A002
    return float(
        st.number_input(
            f"{label} [%]",
            min_value=0.0,
            max_value=100.0,
            value=round(value, 4),
            step=0.1,
            format="%.3f",
            key=key,
            help=help,
        )
    )


def _read_fields(
    fields: tuple[tuple[str, str], ...],
    defaults: dict[str, float],
    key: str,
    skip: str | None,
) -> dict[str, float]:
    """Los porcentajes de un análisis, de a dos por fila (en el celular conserva el orden)."""
    shown = [(attr, label) for attr, label in fields if attr != skip]
    out: dict[str, float] = {}
    for i, (attr, label) in enumerate(shown):
        if i % 2 == 0:
            cols = st.columns(2)
        with cols[i % 2]:
            out[attr] = _number(label, defaults.get(attr, 0.0), f"{key}_{attr}")
    return out


def _read_inputs(system: UnitSystem) -> tuple[HeatingValueInputs | None, str, str]:
    """Los datos de la página: devuelve (datos o ``None`` si hay un error, ejemplo, key)."""
    choice = st.selectbox(
        "Ejemplo precargado",
        _NAMES,
        key="hv_example",
        help="Biomasas de Ghugare et al. (2014) y carbones de Argonne (Vorres, 1990) con el PCS "
        "medido, y sustancias puras con el PCS exacto. Podés cambiar cualquier dato: el "
        "resultado se actualiza solo.",
    )
    ex = _NAMES.index(choice)
    example = HEATING_VALUE_EXAMPLES[choice]
    key = f"hv_{ex}"
    last: HeatingValueInputs = st.session_state.get(f"{key}_last", example)

    st.markdown("#### Combustible")
    type_label = st.selectbox(
        "Tipo de combustible",
        _TYPE_LABELS,
        index=_TYPE_LIST.index(example.fuel_type),
        key=f"{key}_type",
        help="Cada correlación se ajustó con ciertos combustibles: la tabla marca las que usás "
        "fuera de su dominio.",
    )
    fuel_type = _TYPE_LIST[_TYPE_LABELS.index(type_label)]
    analysis = st.radio(
        "Análisis que tenés",
        _ANALYSES,
        index=_ANALYSES.index(_default_analysis(example)),
        horizontal=True,
        key=f"{key}_an",
        help="Elemental (último): C, H, O, N y S. Inmediato (próximo): humedad, materia volátil, "
        "carbono fijo y cenizas, lo que mide un laboratorio de carbón o de biomasa.",
    )
    basis_label = st.radio(
        "Base de los datos",
        _BASIS_LABELS,
        index=_BASIS_LIST.index(example.basis),
        horizontal=True,
        key=f"{key}_basis",
        help="Tal cual: como llega el combustible, con su humedad. Seca: sin la humedad. Seca y "
        "sin cenizas: solo la materia combustible (las cenizas se dan en base seca).",
    )
    basis = _BASIS_LIST[_BASIS_LABELS.index(basis_label)]
    use_ult = analysis in (_AN_ULT, _AN_BOTH)
    use_prox = analysis in (_AN_PROX, _AN_BOTH)

    # Valores de arranque: el último resultado de este ejemplo (o el ejemplo) en la base elegida.
    W, A_d = last.moisture, last.ash_d
    ult_src = last.ultimate or example.ultimate
    prox_src = last.proximate or example.proximate
    ult_def = (
        {k: 100.0 * v for k, v in to_basis(ult_src.as_dict(), basis, W, A_d).items()}
        if ult_src is not None
        else {"C": 50.0, "H": 6.0, "O": 40.0, "N": 0.5, "S": 0.1}
    )
    prox_def = (
        {k: 100.0 * v for k, v in to_basis(prox_src.as_dict(), basis, W, A_d).items()}
        if prox_src is not None
        else {"VM": 75.0, "FC": 20.0}
    )
    ash_default = 100.0 * (A_d * (1.0 - W) if basis == "ar" else A_d)

    left, right = st.columns(2)
    with left:
        moisture = _number(
            "Humedad tal cual W",
            100.0 * W,
            f"{key}_W",
            help="La del combustible como llega (el bagazo sale del trapiche con ~50 %). En base "
            "seca o sin cenizas no entra en la suma, pero hace falta para pasar a tal cual.",
        )
    with right:
        ash = _number(
            "Cenizas" + (" (base seca)" if basis == "daf" else ""),
            ash_default,
            f"{key}_A{'ar' if basis == 'ar' else 'd'}",
            help="Lo que queda sin quemar (inertes minerales). En la base seca y sin cenizas se "
            "da en base seca, para poder volver.",
        )

    ultimate_pct = proximate_pct = None
    o_diff = fc_diff = False
    if use_ult:
        st.markdown(f"**Análisis elemental** ({BASES[basis]}, % en masa)")
        o_diff = st.checkbox(
            "Oxígeno por diferencia",
            value=False,
            key=f"{key}_odiff",
            help="Los laboratorios suelen dar el O como 100 − (los demás): así lo hace la norma "
            "ASTM D3176-24.",
        )
        ultimate_pct = _read_fields(_ULT_FIELDS, ult_def, f"{key}_{basis}", "O" if o_diff else None)
    if use_prox:
        st.markdown(f"**Análisis inmediato** ({BASES[basis]}, % en masa)")
        fc_diff = st.checkbox(
            "Carbono fijo por diferencia",
            value=True,
            key=f"{key}_fcdiff",
            help="El carbono fijo no se mide: es 100 − (humedad + materia volátil + cenizas), "
            "como en la norma ASTM D3172-13(2021)e1.",
        )
        proximate_pct = _read_fields(
            _PROX_FIELDS, prox_def, f"{key}_{basis}", "FC" if fc_diff else None
        )
    H_d_pct = None
    if not use_ult:
        if st.checkbox(
            "Conozco el H (para el PCI)",
            value=last.H_d is not None,
            key=f"{key}_hknown",
            help="Sin el H no se sabe cuánta agua forma el combustible al quemarse, así que no hay "
            "PCI. En la biomasa ronda el 6 % en base seca.",
        ):
            H_d_pct = _number("H en base seca", 100.0 * (last.H_d or 0.05), f"{key}_Hd")

    st.markdown("#### Referencia")
    reference = None
    ref_basis: Basis = "d"
    ref_kind = "measured"
    ex_ref = example.reference
    if ex_ref is not None and ex_ref.kind == "exact":
        use_ref = st.checkbox(
            "Comparar con el PCS exacto (de las entalpías de formación)",
            value=True,
            key=f"{key}_ref",
        )
        if use_ref:
            reference, ref_kind = ex_ref.hhv_d_J_per_kg, "exact"
            st.caption(
                f"PCS exacto en base seca: {_hv(reference, system)} "
                f"{unit_label(_EH, system)} (vademecum §16.9, con las h_f de NASA)."
            )
    else:
        use_ref = st.checkbox(
            "Comparar con el PCS medido",
            value=ex_ref is not None,
            key=f"{key}_ref",
            help="El de una bomba calorimétrica: con él, cada correlación muestra su desvío.",
        )
        if use_ref:
            left, right = st.columns(2)
            with left:
                rb_label = st.selectbox(
                    "Base del PCS medido", _BASIS_LABELS, index=1, key=f"{key}_refb"
                )
            ref_basis = _BASIS_LIST[_BASIS_LABELS.index(rb_label)]
            ref_d = ex_ref.hhv_d_J_per_kg if ex_ref is not None else 20e6
            default = {
                "ar": ref_d * (1.0 - W),
                "d": ref_d,
                "daf": ref_d / (1.0 - A_d),
            }[ref_basis]
            with right:
                reference = number_input_si(
                    label="PCS medido",
                    kind=_EH,
                    default_si=default,
                    key=f"{key}_refv_{ref_basis}",
                    format="%.0f" if system == "SI" else "%.1f",
                    min_value_si=1e6,
                    max_value_si=150e6,
                )

    try:
        inputs = HeatingValueInputs.from_basis(
            basis,
            name=example.name,
            fuel_type=fuel_type,
            ultimate_pct=ultimate_pct,
            proximate_pct=proximate_pct,
            ash_pct=ash,
            moisture_pct=moisture,
            o_by_difference=o_diff,
            fc_by_difference=fc_diff,
            H_d_pct=H_d_pct,
            reference_J_per_kg=reference,
            reference_basis=ref_basis,
            reference_kind=ref_kind,  # type: ignore[arg-type]
            reference_source=ex_ref.source if ex_ref is not None else "",
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return None, choice, key
    if ref_kind == "exact" and example.ultimate is not None and inputs.ultimate is not None:
        changed = any(
            abs(getattr(inputs.ultimate, k) - getattr(example.ultimate, k)) > 1e-4
            for k in ("C", "H", "O", "N", "S", "A")
        )
        if changed:
            st.caption(
                "Cambiaste la composición: el PCS exacto ya no corresponde a este combustible, "
                "así que no se compara."
            )
            inputs = replace(inputs, reference=None)

    keys = applicable_correlations(inputs)
    names = [CORRELATIONS[k].name for k in keys]
    main_name = st.selectbox(
        "Correlación principal (para las métricas, el PCI y el barrido)",
        names,
        key=f"{key}_main_{'u' if use_ult else ''}{'p' if use_prox else ''}",
        help="Con el análisis elemental, Channiwala y Parikh (la de menor error en general); con "
        "el inmediato, Parikh et al.",
    )
    inputs = replace(inputs, main=keys[names.index(main_name)])
    return inputs, choice, key


# ---------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------


def _render_metrics(result: HeatingValueResult, system: UnitSystem) -> None:
    E = unit_label(_EH, system)
    main = result.main
    items = [
        (f"PCS seco [{E}]", _hv(main.hhv_d, system)),
        (f"PCS tal cual [{E}]", _hv(main.hhv_ar, system)),
        (f"PCI tal cual [{E}]", _hv(main.lhv_ar, system)),
    ]
    ref = result.reference
    if ref is not None:
        items.append((f"{ref.name} seco [{E}]", _hv(ref.hhv_d, system)))
        items.append(("Desvío de la principal", _signed_pct(main.deviation)))
    w_star = result.zero_lhv_moisture
    if w_star is not None:
        items.append(("Humedad con PCI nulo W*", f"{format_value(100.0 * w_star, 3)} %"))
    _metric_rows(items)
    st.caption(f"Correlación principal: **{main.name}**.")


def _render_comparison(result: HeatingValueResult, system: UnitSystem) -> None:
    st.markdown("#### Comparación de las correlaciones")
    E = unit_label(_EH, system)
    scale = convert_from_si(1.0, _EH, system)
    rows = [
        ComparisonRow(e.name, e.hhv_d * scale, e.analysis, e.flagged, e.deviation)
        for e in result.estimates
    ]
    ref = result.reference
    fig = comparison_figure(
        rows,
        unit=E,
        reference=None if ref is None else ref.hhv_d * scale,
        reference_label=ref.name if ref is not None else "",
    )
    st.plotly_chart(fig, width="stretch", key="hv_comparison")
    st.caption(
        "Cada punto es el PCS en base seca de una correlación"
        + (", con su desvío respecto de la referencia (la línea de trazos)" if ref else "")
        + ". Azul: con el análisis elemental; naranja: con el inmediato; hueco: fuera del rango "
        "o del tipo de combustible con que se ajustó."
    )
    data = []
    for e in [*result.estimates, *([ref] if ref is not None else [])]:
        corr = CORRELATIONS.get(e.key)
        data.append(
            {
                "Correlación": e.name,
                f"PCS seco [{E}]": _hv(e.hhv_d, system),
                "Desvío": _signed_pct(e.deviation),
                f"PCS tal cual [{E}]": _hv(e.hhv_ar, system),
                f"PCI tal cual [{E}]": _hv(e.lhv_ar, system),
                f"PCS seco sin cenizas [{E}]": _hv(e.hhv_daf, system),
                "Análisis": ANALYSIS_KINDS[e.analysis].split(" ")[0] if corr else "—",
                "Avisos": "; ".join(
                    [*e.warnings, *([] if e.in_type else ["fuera de su tipo de combustible"])]
                )
                or "—",
            }
        )
    _table(pd.DataFrame(data))


def _render_bases(result: HeatingValueResult) -> None:
    inp = result.inputs
    st.markdown("#### El análisis en las tres bases")
    st.caption("En % en masa. El H y el O son los de la materia seca: la humedad va aparte.")
    dry: dict[str, float] = {}
    if inp.ultimate is not None:
        dry.update({k: getattr(inp.ultimate, k) for k in ("C", "H", "O", "N", "S")})
    if inp.proximate is not None:
        dry.update({"VM": inp.proximate.VM, "FC": inp.proximate.FC})
    dry["A"] = inp.ash_d
    W = inp.moisture
    conv = {b: to_basis(dry, b, W, inp.ash_d) for b in _BASIS_LIST}
    rows = []
    for k in [*dry, "W"]:
        rows.append(
            {
                "Componente": _ROW_NAMES[k],
                "Tal cual [%]": _num(100.0 * conv["ar"][k], 4),
                "Seca [%]": "—" if k == "W" else _num(100.0 * conv["d"][k], 4),
                "Seca y sin cenizas [%]": "—"
                if k in ("A", "W")
                else _num(100.0 * conv["daf"][k], 4),
            }
        )
    _table(pd.DataFrame(rows))
    if result.water_formed_d is not None:
        st.caption(
            f"El H forma {format_value(result.water_formed_d, 4)} kg de agua por kg seco "
            f"({format_value(hydrogen_water_ratio(), 4)}·H): con la humedad, es el agua que en "
            "el PCI sale como vapor."
        )


def _render_dataset(result: HeatingValueResult, system: UnitSystem) -> None:
    with st.expander("🎯 ¿Qué tan buenas son las correlaciones?", expanded=False):
        st.markdown(
            "**536 biomasas con el PCS medido** (Ghugare et al., 2014): el análisis elemental en "
            "base seca de maderas, residuos agrícolas, cáscaras, carbonizados y lodos, cada una "
            "con su bomba calorimétrica."
        )
        label = st.selectbox(
            "Correlación",
            [CORRELATIONS[k].name for k in _ULT_KEYS],
            key="hv_ds_corr",
        )
        key = _ULT_KEYS[[CORRELATIONS[k].name for k in _ULT_KEYS].index(label)]
        fit = dataset_fit(key)
        inp = result.inputs
        current = current_err = None
        if inp.ultimate is not None and result.reference is not None:
            est = result.estimate(key)
            current = (result.reference.hhv_d, est.hhv_d)
            current_err = (inp.ultimate.O, est.deviation or 0.0)
        E = unit_label(_EH, system)
        scale = convert_from_si(1.0, _EH, system)
        st.plotly_chart(
            parity_figure(fit, name=label, scale=scale, unit=E, current=current),
            width="stretch",
            key="hv_parity",
        )
        st.plotly_chart(
            error_vs_oxygen_figure(fit, name="El error según el O", current=current_err),
            width="stretch",
            key="hv_error_o",
        )
        lo, hi = ERROR_RANGE
        outside = sum(not lo <= 100.0 * e <= hi for e in fit.errors)
        st.caption(
            "Dulong supone que el O ya está unido al H como agua: con poco O (carbonizados) "
            "acierta, y con el 40–50 % de O de la biomasa subestima cada vez más. Las de ajuste "
            "(Boie, Channiwala y Parikh) no tienen esa tendencia. Fuera de la escala quedan "
            f"{outside} muestras con datos dudosos (por ejemplo, con casi nada de H)."
        )
        rows = []
        for k in _ULT_KEYS:
            f = dataset_fit(k)
            rows.append(
                {
                    "Correlación": CORRELATIONS[k].name,
                    "AAE": f"{100.0 * f.aae:.2f} %",
                    "ABE": f"{100.0 * f.abe:+.2f} %",
                    "Dentro de ±10 %": f"{100.0 * f.within_10:.0f} %",
                    "Error que informan los autores": CORRELATIONS[k].reported or "—",
                }
            )
        _table(pd.DataFrame(rows))
        st.caption(
            "AAE: error absoluto medio, Σ|e|/n; ABE: error medio con signo, Σe/n (el sesgo), con "
            "e = (estimado − medido)/medido, como en Channiwala y Parikh (2002)."
        )
        st.markdown(
            "**Carbones de Argonne** (Vorres, 1990): cinco carbones de EE. UU. de distinto rango, "
            "con los dos análisis y el PCS medido."
        )
        coal_rows = []
        for row in argonne_deviations():
            coal = row["coal"]
            item = {"Carbón": coal.name}
            for k in CORRELATIONS:
                item[_SHORT_NAMES[k]] = _signed_pct(row["deviations"][k])
            item["Rango"] = coal.rank
            coal_rows.append(item)
        _table(pd.DataFrame(coal_rows))
        st.caption("CyP: Channiwala y Parikh (2002); Parikh: Parikh et al. (2005).")
        st.caption(
            "Con el elemental, las tres aciertan a ±3 %. Con el inmediato subestiman 7–23 %, "
            "aunque los carbones estén dentro de sus rangos de ajuste: la materia volátil de un "
            "carbón bituminoso tiene mucho más H (vale más) que la de la madera, y una "
            "correlación con MV y CF no las distingue."
        )


def _render_sweep(result: HeatingValueResult, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo cambia con la humedad?", expanded=False):
        values = default_moisture_values()
        sw = moisture_sweep(result, values)
        scale = convert_from_si(1.0, _EH, system)
        series = {
            "PCS tal cual": [None if v is None else v * scale for v in sw["PCS"]],
            "PCI tal cual": [None if v is None else v * scale for v in sw["PCI"]],
        }
        fig = sweep_figure(
            [100.0 * w for w in values],
            series,
            title=f"Con {result.main.name}",
            x_title="humedad tal cual W [%]",
            y_title=f"poder calorífico [{unit_label(_EH, system)}]",
            current_x=100.0 * result.inputs.moisture,
        )
        st.plotly_chart(fig, width="stretch", key="hv_sweep")
        w_star = result.zero_lhv_moisture
        text = (
            "El análisis seco no cambia: cada kg tal cual tiene (1 − W) kg de combustible seco, "
            "así que el PCS baja en línea recta. El PCI baja más rápido porque además hay que "
            "evaporar esa agua."
        )
        if w_star is not None:
            text += (
                f" Con W* = {format_value(100.0 * w_star, 3)} % el PCI se anula: el combustible "
                "no alcanza a evaporar su propia agua."
            )
        st.caption(text + " La línea punteada marca tus datos.")


def _render_export(data: dict[str, Any]) -> None:
    st.markdown("#### 💾 Exportar")
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="poder_calorifico.csv",
        mime="text/csv",
        key="hv_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="poder_calorifico.json",
        mime="application/json",
        key="hv_download_json",
    )


def _render_results(result: HeatingValueResult, name: str, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    note = HEATING_VALUE_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    _render_metrics(result, system)
    for item in result.notes:
        st.info(item)
    _render_comparison(result, system)
    _render_bases(result)
    _render_dataset(result, system)
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cada correlación en su forma publicada (los % en masa en base seca y el PCS en "
            "MJ/kg) con los números reemplazados; el resto, en el sistema de unidades elegido."
        )
        for step in heating_value_steps(result, system):
            st.markdown(f"**{step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)
    _render_export(heating_value_to_dict(result, system))
    _render_sweep(result, system)


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§16.9 *Poder calorífico*** (la definición, el PCS y el PCI). Las correlaciones no "
            "están en el vademecum: acá van con su trabajo original. Çengel & Boles, §15-3 "
            "(poder calorífico)."
        )
        st.markdown(
            "**Poder calorífico** (§16.9): el calor que libera la combustión completa con los "
            "reactivos y los productos a 25 °C. El PCS deja el agua de los humos líquida; el PCI, "
            "como vapor."
        )
        st.latex(
            r"\overline{PC} = \sum_\mathrm{reac} \nu_j\,\bar h_{f,j}"
            r" - \sum_\mathrm{prod} \nu_i\,\bar h_{f,i}"
        )
        st.markdown(
            "En una sustancia pura sale exacto de las h_f (la página Combustión). Un carbón o una "
            "biomasa no tiene una h_f conocida: se mide en una bomba calorimétrica o se estima "
            "con una **correlación** sobre su análisis."
        )
        st.markdown(
            "**Análisis.** El *elemental* (último) da C, H, O, N y S (ASTM D3176-24; el O suele "
            "salir por diferencia). El *inmediato* (próximo) da la humedad, la materia volátil "
            "(lo que se desprende al calentar sin aire a ~900–950 °C), las cenizas y el carbono "
            "fijo por diferencia (ASTM D3172-13(2021)e1)."
        )
        st.markdown(
            "**Bases** (ASTM D3180-25): *tal cual* (con la humedad W), *seca* y *seca y sin "
            "cenizas* (solo la materia combustible). La humedad y las cenizas no aportan calor:"
        )
        st.latex(r"x_\mathrm{s} = \frac{x_\mathrm{tc}}{1 - W}")
        st.latex(r"x_\mathrm{sscz} = \frac{x_\mathrm{s}}{1 - \mathit{Cz}_\mathrm{s}}")
        st.latex(r"PCS_\mathrm{tc} = PCS_\mathrm{s}\,(1 - W)")
        st.markdown("**Correlaciones** (PCS en MJ/kg, con los % en masa **en base seca**):")
        for corr in CORRELATIONS.values():
            ranges = (
                "; rango de ajuste: "
                + ", ".join(
                    f"{_RANGE_NAMES.get(v, v)} {format_value(lo, 4)}–{format_value(hi, 4)} %"
                    for v, lo, hi in corr.ranges
                )
                if corr.ranges
                else ""
            )
            reported = f" Error informado: {corr.reported}." if corr.reported else ""
            doi = f" [doi:{corr.doi}](https://doi.org/{corr.doi})" if corr.doi else ""
            st.markdown(
                f"*{corr.name}* — {ANALYSIS_KINDS[corr.analysis]}; {corr.fuels}{ranges}."
                f"{reported} {corr.reference}{doi}"
            )
            st.latex(rf"\begin{{aligned}}PCS_\mathrm{{s}} &= {corr.latex}\end{{aligned}}")
        hv = element_heating_values()
        st.markdown(
            "**La base física de Dulong.** Cada elemento aporta su propio PCS, que sale de las "
            f"h_f de NASA: C (grafito) → CO₂, {format_value(hv['C'] / 1e6, 4)} MJ/kg; H₂ → "
            f"H₂O(ℓ), {format_value(hv['H'] / 1e6, 4)} MJ/kg; S → SO₂, "
            f"{format_value(hv['S'] / 1e6, 3)} MJ/kg (Dulong usa 33,83, 144,3 y 9,42). El O del "
            "combustible se supone ya unido al H como agua: solo se quema el *hidrógeno "
            "disponible*, H − O/8. Ignora la h_f del combustible, así que se pasa en los "
            "hidrocarburos (+11 % en el metano) y se queda corta con mucho O (biomasa)."
        )
        st.markdown(
            "**PCI** (§16.9): el agua que forma el H, M_H₂O/(2·M_H) ≈ 8,94 kg por kg de H (el "
            "«9·H» de los libros), y la humedad salen como vapor y se llevan h_fg a 25 °C "
            f"({format_value(water_hfg_J_per_kg() / 1000, 5)} kJ/kg; el vademecum redondea a "
            "2442):"
        )
        st.latex(r"PCI_\mathrm{s} = PCS_\mathrm{s} - h_{fg}\,8.94\,H_\mathrm{s}")
        st.latex(r"PCI_\mathrm{tc} = PCI_\mathrm{s}\,(1 - W) - h_{fg}\,W")
        st.latex(r"W^* = \frac{PCI_\mathrm{s}}{PCI_\mathrm{s} + h_{fg}}")
        st.markdown(
            "La bomba calorimétrica mide a volumen constante; las normas (ISO 18125:2017 para "
            "biocombustibles sólidos) corrigen el PCI a presión constante, una diferencia de "
            "décimas de %."
        )
        st.markdown(
            "**Cómo se mide el error** (Channiwala y Parikh, 2002), con e = (estimado − "
            "medido)/medido en cada muestra:"
        )
        st.latex(r"AAE = \frac{1}{n}\sum |e| \qquad ABE = \frac{1}{n}\sum e")


_RANGE_NAMES = {"A": "cenizas", "VM": "MV", "FC": "CF"}


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Poder calorífico", page_icon="🪵", layout="centered")

st.subheader(SUBJECT)
st.title("🪵 Poder calorífico por correlaciones")
st.markdown(
    "El PCS de un carbón o una biomasa se mide en una bomba calorimétrica o se **estima con una "
    "correlación** sobre su análisis elemental o inmediato. Acá comparás las correlaciones "
    "clásicas (Dulong, Boie, Channiwala y Parikh, Parikh, Cordero) entre sí y contra el PCS "
    "medido o el exacto, pasás de una base a otra (tal cual, seca, sin cenizas), sacás el PCI y "
    "ves cuánto se equivoca cada una con datos reales."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Poder calorífico")
render_units_selector()
system = get_current_system()

inputs, example_name, example_key = _read_inputs(system)
if inputs is not None:
    try:
        result = _solve_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
    else:
        st.session_state[f"{example_key}_last"] = inputs
        _render_results(result, example_name, system)
