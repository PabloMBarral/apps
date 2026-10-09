"""Página 14 — Transferencia de calor.

Fase 8.1. Tres modos:

- **Conducción** (Cengel y Ghajar, §3-1 a §3-5): la red de resistencias de una
  pared plana, un caño o una esfera de varias capas, con contacto, partes en
  paralelo, convección en los bordes y el radio crítico de aislación.
- **Aletas** (Incropera, §3.6): recta, de aguja o anular, con cada condición de la
  punta, la eficiencia, la efectividad y un arreglo de aletas.
- **Convección**: el coeficiente h de las correlaciones de Nusselt, forzada
  externa (placa, cilindro, esfera), interna (un tubo) y natural.

El cálculo vive en :mod:`core.heat_transfer`; los gráficos, en
:mod:`ui.heat_transfer_charts`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.fluids import FLUID_NAMES_ES
from core.heat_transfer import conduction as cd
from core.heat_transfer import convection as cv
from core.heat_transfer import fins as fn
from core.heat_transfer.conduction_procedure import conduction_steps
from core.heat_transfer.convection_procedure import convection_steps, internal_steps
from core.heat_transfer.fins_procedure import fin_steps
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.heat_transfer_charts import (
    conduction_profile_figure,
    correlation_figure,
    fin_efficiency_figure,
    fin_profile_figure,
    insulation_figure,
    nu_curve_figure,
    tube_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.24.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_COND = "Conducción: paredes, caños y esferas"
_M_FIN = "Aletas"
_M_CONV = "Convección: el coeficiente h"
_MODES = (_M_COND, _M_FIN, _M_CONV)

_V_EXT = "Forzada externa"
_V_INT = "Forzada interna (un tubo)"
_V_NAT = "Natural"
_CONV_WHAT = (_V_EXT, _V_INT, _V_NAT)

_BOUNDARY_LABELS: dict[cd.BoundaryKind, str] = {
    "fluid": "Un fluido (T∞ y h)",
    "surface": "Una superficie a T dada",
    "heat": "Un calor dado (Q̇ que entra)",
}
_AUTO = "auto"

_W: QuantityKind = "heat_rate"
_H: QuantityKind = "heat_transfer_coefficient"
_K: QuantityKind = "thermal_conductivity"
_T: QuantityKind = "temperature"


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


def _notes(notes: list[str]) -> None:
    for note in notes:
        st.info(note)


def _example(examples: dict[str, Any], key: str, help_text: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox("Ejemplo precargado", names, key=key, help=help_text)
    return name, names.index(name)


# ---------------------------------------------------------------------
# Conducción
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _conduction_cached(inputs: cd.ConductionInputs) -> cd.ConductionResult:
    return cd.solve_conduction(inputs)


@st.cache_data(show_spinner=False)
def _sweep_cached(inputs: cd.ConductionInputs) -> list[cd.SweepPoint]:
    return cd.insulation_sweep(inputs, cd.default_insulation_radii(inputs))


def _boundary_inputs(title: str, default: cd.Boundary, key: str) -> cd.Boundary:
    kinds = list(_BOUNDARY_LABELS)
    kind: cd.BoundaryKind = st.radio(
        title,
        kinds,
        index=kinds.index(default.kind),
        format_func=lambda k: _BOUNDARY_LABELS[k],
        key=f"{key}_kind",
        horizontal=True,
    )
    if kind == "heat":
        Q = number_input_si(
            label="Calor que entra Q̇",
            kind=_W,
            default_si=default.Q_W or 100.0,
            key=f"{key}_Q",
            format="%.5g",
            help="El que se genera adentro (un alambre por efecto Joule) o el que entra por ese "
            "lado; la otra temperatura tiene que ser dato.",
        )
        return cd.Boundary("heat", Q_W=Q)
    left, right = st.columns(2)
    with left:
        T = number_input_si(
            label="T∞ del fluido" if kind == "fluid" else "T de la superficie",
            kind=_T,
            default_si=default.T_K or 293.15,
            key=f"{key}_T",
            format="%.5g",
        )
    if kind == "surface":
        return cd.Boundary("surface", T_K=T)
    with right:
        h = number_input_si(
            label="h",
            kind=_H,
            default_si=default.h_W_per_m2K or 10.0,
            key=f"{key}_h",
            format="%.5g",
            min_value_si=0.0,
            help="Convección (y radiación, si se la combina en un h equivalente).",
        )
    return cd.Boundary("fluid", T_K=T, h_W_per_m2K=h)


def _layer_inputs(
    j: int, default: cd.Layer, geometry: cd.Geometry, last: bool, key: str
) -> cd.Layer:
    st.markdown(f"**Capa {j + 1}: {default.name}**")
    left, right = st.columns(2)
    with left:
        t = number_input_si(
            label="Espesor",
            kind="small_length",
            default_si=default.thickness_m,
            key=f"{key}_t",
            format="%.5g",
            min_value_si=0.0,
        )
    parts: tuple[cd.ParallelPart, ...] = ()
    k = default.k_W_per_mK
    if default.parts and geometry == "plane":
        right.caption("Partes en paralelo (k y fracción del área):")
        rows = []
        for p_i, part in enumerate(default.parts):
            a, b = st.columns(2)
            with a:
                k_p = number_input_si(
                    label=f"k {part.name}",
                    kind=_K,
                    default_si=part.k_W_per_mK,
                    key=f"{key}_p{p_i}_k",
                    format="%.4g",
                )
            with b:
                f = st.number_input(
                    f"Fracción del área {part.name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(part.area_fraction),
                    step=0.01,
                    format="%.4f",
                    key=f"{key}_p{p_i}_f",
                )
            rows.append(cd.ParallelPart(part.name, k_p, float(f)))
        parts = tuple(rows)
    else:
        with right:
            k = number_input_si(
                label="Conductividad k",
                kind=_K,
                default_si=default.k_W_per_mK or default.k_effective or 1.0,
                key=f"{key}_k",
                format="%.4g",
            )
    contact = 0.0
    if not last and st.checkbox(
        "Resistencia de contacto con la capa siguiente",
        value=default.contact_m2K_per_W > 0.0,
        key=f"{key}_c",
    ):
        h_c = number_input_si(
            label="Conductancia de contacto h_c = 1/R''_c",
            kind=_H,
            default_si=1.0 / default.contact_m2K_per_W if default.contact_m2K_per_W else 10_000.0,
            key=f"{key}_hc",
            format="%.5g",
            min_value_si=0.0,
            help="Cengel y Ghajar, tabla 3-2: metales pulidos y apretados, de 3000 a 70 000 "
            "W/(m²·K).",
        )
        contact = 1.0 / h_c if h_c > 0.0 else 0.0
    return cd.Layer(default.name, t, k if not parts else 0.0, parts, contact)


def _conduction(system: UnitSystem) -> None:
    name, idx = _example(
        cd.CONDUCTION_EXAMPLES,
        "qc_example",
        "Ejemplos del cap. 3 de Cengel y Ghajar. Podés cambiar cualquier dato: el resultado se "
        "actualiza solo.",
    )
    ex = cd.CONDUCTION_EXAMPLES[name]
    key = f"qc_{idx}"
    note = cd.CONDUCTION_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 {note}")
    geos = list(cd.GEOMETRY_NAMES)
    geometry: cd.Geometry = st.radio(
        "Geometría",
        geos,
        index=geos.index(ex.geometry),
        format_func=lambda g: cd.GEOMETRY_NAMES[g],
        key=f"{key}_geo",
        horizontal=True,
    )
    area, length, r_in = ex.area_m2, ex.length_m, ex.r_inner_m
    left, right = st.columns(2)
    if geometry == "plane":
        with left:
            area = number_input_si(
                label="Área A", kind="area", default_si=ex.area_m2, key=f"{key}_A", format="%.5g"
            )
    else:
        with left:
            r_in = number_input_si(
                label="Radio interior r₁",
                kind="small_length",
                default_si=ex.r_inner_m,
                key=f"{key}_r1",
                format="%.5g",
                help="Para un alambre macizo, el radio del alambre (el de la cara interior de "
                "la primera capa).",
            )
        if geometry == "cylinder":
            with right:
                length = number_input_si(
                    label="Largo L",
                    kind="length",
                    default_si=ex.length_m,
                    key=f"{key}_L",
                    format="%.5g",
                )
    n_layers = int(
        st.number_input(
            "Cantidad de capas", min_value=1, max_value=5, value=len(ex.layers), key=f"{key}_n"
        )
    )
    layers = []
    for j in range(n_layers):
        default = ex.layers[j] if j < len(ex.layers) else cd.Layer(f"capa {j + 1}", 0.01, 1.0)
        layers.append(
            _layer_inputs(j, default, geometry, j == n_layers - 1, f"{key}_{geometry}_{j}")
        )
    inner = _boundary_inputs("Lado 1 (la izquierda o el interior)", ex.inner, f"{key}_s1")
    outer = _boundary_inputs("Lado 2 (la derecha o el exterior)", ex.outer, f"{key}_s2")
    inputs = cd.ConductionInputs(geometry, tuple(layers), inner, outer, area, length, r_in)
    try:
        result = _conduction_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_conduction(result, system)


def _render_conduction(r: cd.ConductionResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items = [
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system)),
        (
            f"R total [{_u('thermal_resistance', system)}]",
            _val(r.R_total_K_per_W, "thermal_resistance", system),
        ),
    ]
    if i.geometry == "plane":
        items.append((f"U [{_u(_H, system)}]", _val(r.U_inner_W_per_m2K, _H, system)))
        items.append(
            (f"q″ [{_u('heat_flux', system)}]", _val(r.q_inner_W_per_m2, "heat_flux", system))
        )
    else:
        items.append(
            (f"U₁ (cara interior) [{_u(_H, system)}]", _val(r.U_inner_W_per_m2K, _H, system))
        )
        items.append(
            (f"U₂ (cara exterior) [{_u(_H, system)}]", _val(r.U_outer_W_per_m2K, _H, system))
        )
    if r.Q_per_length_W_per_m is not None:
        items.append(
            (
                f"Q̇/L [{_u('linear_heat_rate', system)}]",
                _val(r.Q_per_length_W_per_m, "linear_heat_rate", system),
            )
        )
    if i.inner.kind == "heat":
        items.append((f"T del lado 1 [{_u(_T, system)}]", _val(r.T_side1_K, _T, system)))
    if r.critical_radius_m is not None:
        items.append(
            (
                f"Radio crítico r_cr [{_u('small_length', system)}]",
                _val(r.critical_radius_m, "small_length", system),
            )
        )
    _metric_rows(items)
    _notes(cd.conduction_notes(r))
    st.markdown("#### La red de resistencias")
    ut, ur = _u(_T, system), _u("thermal_resistance", system)
    _table(
        pd.DataFrame(
            [
                {
                    "Elemento": res.label,
                    f"R [{ur}]": _val(res.R_K_per_W, "thermal_resistance", system),
                    "Del total": _pct(r.share(res)),
                    f"T desde [{ut}]": _val(res.T_from_K, _T, system),
                    f"T hasta [{ut}]": _val(res.T_to_K, _T, system),
                }
                for res in r.resistances
            ]
        )
    )
    parts = [(res, p) for res in r.resistances for p in res.parts]
    if parts:
        st.markdown("#### El calor por cada parte de la capa compuesta")
        _table(
            pd.DataFrame(
                [
                    {
                        "Parte": p.name,
                        f"R [{ur}]": _val(p.R_K_per_W, "thermal_resistance", system),
                        f"Q̇ [{_u(_W, system)}]": _val(p.Q_W, _W, system),
                        "Del Q̇ de la capa": _pct(p.Q_W / r.Q_W if r.Q_W else None),
                    }
                    for _res, p in parts
                ]
            )
        )
    st.markdown("#### El perfil de temperatura")
    t_scale, t_offset = _t_scale(system)
    x_scale = convert_from_si(1.0, "small_length", system)
    try:
        fig = conduction_profile_figure(
            r,
            t_scale=t_scale,
            t_offset=t_offset,
            t_unit=ut,
            x_scale=x_scale,
            x_unit=_u("small_length", system),
        )
        st.plotly_chart(fig, width="stretch", key="qc_profile")
        st.caption(
            "Cada capa sombreada con su número. "
            + {
                "plane": "En una pared plana el perfil es recto en cada capa: la pendiente es "
                "Q̇/(k·A), mayor donde k es menor.",
                "cylinder": "En un cilindro el perfil es logarítmico: el área crece con r y la "
                "pendiente baja hacia afuera.",
                "sphere": "En una esfera el perfil va como 1/r.",
            }[i.geometry]
            + " En naranja los fluidos: la caída punteada es la de la película de convección."
        )
    except Exception as exc:  # noqa: BLE001 — un gráfico que falla no tumba la página
        st.warning(f"No se pudo dibujar el perfil: {exc}")
    if i.geometry != "plane" and i.outer.kind == "fluid":
        st.markdown("#### Radio crítico de aislación")
        heat_given = i.inner.kind == "heat"
        try:
            points = _sweep_cached(i)
            if heat_given:
                y_scale, y_offset, y_unit = t_scale, t_offset, ut
            else:
                y_scale, y_offset, y_unit = convert_from_si(1.0, _W, system), 0.0, _u(_W, system)
            st.plotly_chart(
                insulation_figure(
                    points,
                    r,
                    heat_given=heat_given,
                    x_scale=x_scale,
                    x_unit=_u("small_length", system),
                    y_scale=y_scale,
                    y_offset=y_offset,
                    y_unit=y_unit,
                ),
                width="stretch",
                key="qc_sweep",
            )
            st.caption(
                "La capa exterior cambia de espesor y lo demás queda igual. "
                + (
                    "Con el calor dado, la temperatura del lado 1 tiene un mínimo en r_cr: hasta "
                    "ahí, aislar enfría el alambre."
                    if heat_given
                    else "Con las temperaturas dadas, Q̇ tiene un máximo en r_cr: hasta ahí, "
                    "aislar aumenta la pérdida de calor."
                )
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"No se pudo dibujar el barrido: {exc}")
    _render_steps(
        conduction_steps(r, system),
        "Las resistencias, el calor y las temperaturas con los números en el sistema elegido.",
    )
    _render_export(cd.conduction_to_dict(r, system), "conduccion", "qc_dl")


# ---------------------------------------------------------------------
# Aletas
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _fin_cached(fin: fn.FinInputs) -> tuple[fn.FinResult, dict[str, fn.FinResult]]:
    return fn.solve_fin(fin), dict(fn.tip_comparison(fin))


@st.cache_data(show_spinner=False)
def _array_cached(inputs: fn.FinArrayInputs) -> fn.FinArrayResult:
    return fn.solve_fin_array(inputs)


def _fins(system: UnitSystem) -> None:
    name, idx = _example(
        fn.FIN_EXAMPLES,
        "qa_example",
        "Ejemplos de Incropera (§3.6) y de Cengel y Ghajar (§3-6). Podés cambiar cualquier "
        "dato: el resultado se actualiza solo.",
    )
    ex = fn.FIN_EXAMPLES[name]
    key = f"qa_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    f = ex.fin
    shapes = list(fn.FIN_SHAPES)
    shape: fn.FinShape = st.radio(
        "Forma",
        shapes,
        index=shapes.index(f.shape),
        format_func=lambda s: fn.FIN_SHAPES[s],
        key=f"{key}_shape",
        horizontal=True,
    )
    tips = list(fn.TIP_CONDITIONS)
    tip: fn.TipCondition = st.selectbox(
        "Condición de la punta",
        tips,
        index=tips.index(f.tip),
        format_func=lambda t: fn.TIP_CONDITIONS[t],
        key=f"{key}_tip",
        help="La convectiva es la exacta; la longitud corregida L_c la aproxima con una punta "
        "adiabática; la infinita vale si mL > 2,65.",
    )
    left, right = st.columns(2)
    with left:
        k = number_input_si(
            label="Conductividad k", kind=_K, default_si=f.k_W_per_mK, key=f"{key}_k", format="%.4g"
        )
        T_b = number_input_si(
            label="T de la base T_b", kind=_T, default_si=f.T_base_K, key=f"{key}_Tb", format="%.5g"
        )
    with right:
        h = number_input_si(
            label="h del fluido", kind=_H, default_si=f.h_W_per_m2K, key=f"{key}_h", format="%.4g"
        )
        T_inf = number_input_si(
            label="T∞ del fluido", kind=_T, default_si=f.T_inf_K, key=f"{key}_Tinf", format="%.5g"
        )
    left, right = st.columns(2)
    with left:
        L = number_input_si(
            label="Largo L" if shape != "annular" else "Largo radial L = r₂ − r₁",
            kind="small_length",
            default_si=f.length_m,
            key=f"{key}_L",
            format="%.5g",
        )
    thickness, width, diameter, r_base = f.thickness_m, f.width_m, f.diameter_m, f.r_base_m
    with right:
        if shape == "pin":
            diameter = number_input_si(
                label="Diámetro D",
                kind="small_length",
                default_si=f.diameter_m or 0.005,
                key=f"{key}_D",
                format="%.5g",
            )
        else:
            thickness = number_input_si(
                label="Espesor t",
                kind="small_length",
                default_si=f.thickness_m or 0.0015,
                key=f"{key}_t",
                format="%.5g",
            )
    if shape == "straight":
        width = number_input_si(
            label="Ancho w",
            kind="small_length",
            default_si=f.width_m if f.shape == "straight" else 0.05,
            key=f"{key}_w",
            format="%.5g",
        )
    elif shape == "annular":
        r_base = number_input_si(
            label="Radio de la base r₁ (el exterior del caño)",
            kind="small_length",
            default_si=f.r_base_m or 0.025,
            key=f"{key}_r1",
            format="%.5g",
        )
    T_tip = None
    if tip == "temperature":
        T_tip = number_input_si(
            label="T de la punta T_L",
            kind=_T,
            default_si=f.T_tip_K or 0.5 * (f.T_base_K + f.T_inf_K),
            key=f"{key}_TL",
            format="%.5g",
        )
    fin = fn.FinInputs(shape, k, h, T_b, T_inf, L, thickness, width, diameter, r_base, tip, T_tip)
    array_inputs = None
    if st.checkbox("Un arreglo de aletas", value=ex.n_fins is not None, key=f"{key}_arr"):
        a, b = st.columns(2)
        with a:
            n = int(
                st.number_input(
                    "Cantidad de aletas N", min_value=1, value=ex.n_fins or 10, key=f"{key}_N"
                )
            )
        with b:
            base = number_input_si(
                label="Área de la base (sin aletas)",
                kind="area",
                default_si=ex.base_area_m2 or 0.01,
                key=f"{key}_Ab",
                format="%.5g",
            )
        array_inputs = fn.FinArrayInputs(fin, n, base)
    try:
        result, others = _fin_cached(fin)
        array = _array_cached(array_inputs) if array_inputs is not None else None
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_fin(result, others, array, system)  # type: ignore[arg-type]


def _render_fin(
    r: fn.FinResult,
    others: dict[fn.TipCondition, fn.FinResult],
    array: fn.FinArrayResult | None,
    system: UnitSystem,
) -> None:
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"Q̇ de la aleta [{_u(_W, system)}]", _val(r.Q_W, _W, system)),
            ("Eficiencia η", _num(r.efficiency, 4)),
            ("Efectividad ε", _num(r.effectiveness, 4)),
            (f"T de la punta [{_u(_T, system)}]", _val(r.T_tip_K, _T, system)),
            ("m·L", _num(r.mL, 4)),
            ("Biot h·δ/k", _num(r.biot, 3)),
        ]
    )
    _notes(fn.fin_notes(r))
    if array is not None:
        st.markdown("#### El arreglo")
        _metric_rows(
            [
                (f"Q̇ con aletas [{_u(_W, system)}]", _val(array.Q_total_W, _W, system)),
                (f"Q̇ sin aletas [{_u(_W, system)}]", _val(array.Q_no_fins_W, _W, system)),
                ("Eficiencia global η_o", _num(array.overall_efficiency, 4)),
                ("Efectividad del arreglo", _num(array.effectiveness, 4)),
            ]
        )
        _notes(fn.fin_array_notes(array))
    st.markdown("#### La misma aleta con cada condición de la punta")
    _table(
        pd.DataFrame(
            [
                {
                    "Punta": fn.TIP_CONDITIONS[tip] + (" ←" if tip == r.inputs.tip else ""),
                    f"Q̇ [{_u(_W, system)}]": _val(o.Q_W, _W, system),
                    "η": _num(o.efficiency, 4),
                    f"T punta [{_u(_T, system)}]": _val(o.T_tip_K, _T, system),
                }
                for tip, o in others.items()
            ]
        )
    )
    t_scale, t_offset = _t_scale(system)
    try:
        st.plotly_chart(
            fin_profile_figure(
                r,
                others,
                x_scale=convert_from_si(1.0, "small_length", system),
                x_unit=_u("small_length", system),
                t_scale=t_scale,
                t_offset=t_offset,
                t_unit=_u(_T, system),
            ),
            width="stretch",
            key="qa_profile",
        )
        st.caption(
            "La temperatura cae a lo largo de la aleta hacia la del fluido (la raya naranja). "
            "En gris, la misma aleta con las otras condiciones de la punta: si mL es chico casi "
            "no se distinguen."
        )
        st.plotly_chart(fin_efficiency_figure(r), width="stretch", key="qa_eta")
        st.caption(
            "La eficiencia baja con m·L_c: una aleta larga o poco conductora tiene la punta casi "
            "a la temperatura del fluido y esa parte transfiere poco."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar el gráfico: {exc}")
    _render_steps(fin_steps(r, system, array))
    _render_export(fn.fin_to_dict(r, system, array), "aleta", "qa_dl")


# ---------------------------------------------------------------------
# Convección
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _external_cached(inputs: cv.ExternalFlowInputs) -> cv.ConvectionResult:
    return cv.solve_external(inputs)


@st.cache_data(show_spinner=False)
def _natural_cached(inputs: cv.NaturalConvectionInputs) -> cv.ConvectionResult:
    return cv.solve_natural(inputs)


@st.cache_data(show_spinner=False)
def _internal_cached(inputs: cv.InternalFlowInputs) -> cv.InternalFlowResult:
    return cv.solve_internal(inputs)


def _fluid_input(default: str, key: str) -> str:
    fluids = list(cv.convection_fluids())
    return st.selectbox(
        "Fluido",
        fluids,
        index=fluids.index(default) if default in fluids else 0,
        format_func=lambda f: FLUID_NAMES_ES.get(f, f),
        key=key,
    )


def _temperatures(
    default_s: float, default_inf: float, key: str, inf_label: str
) -> tuple[float, float]:
    left, right = st.columns(2)
    with left:
        T_s = number_input_si(
            label="T de la superficie T_s",
            kind=_T,
            default_si=default_s,
            key=f"{key}_Ts",
            format="%.5g",
        )
    with right:
        T_inf = number_input_si(
            label=inf_label, kind=_T, default_si=default_inf, key=f"{key}_Tinf", format="%.5g"
        )
    return T_s, T_inf


def _pressure(default: float, key: str) -> float:
    return number_input_si(
        label="Presión del fluido",
        kind="pressure",
        default_si=default,
        key=f"{key}_p",
        format="%.5g",
        help="Las propiedades de CoolProp dependen de la presión (y la saturación también: "
        "con agua hay que mirar que no hierva).",
    )


def _external(system: UnitSystem) -> None:
    name, idx = _example(
        cv.EXTERNAL_EXAMPLES,
        "qv_ext_example",
        "Ejemplos del cap. 7 de Cengel y Ghajar. Podés cambiar cualquier dato.",
    )
    ex = cv.EXTERNAL_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, cv.ExternalFlowInputs)
    key = f"qv_e{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    left, right = st.columns(2)
    with left:
        fluid = _fluid_input(inp.fluid, f"{key}_fluid")
    with right:
        geos = list(cv.EXTERNAL_GEOMETRIES)
        geometry: cv.ExternalGeometry = st.selectbox(
            "Geometría",
            geos,
            index=geos.index(inp.geometry),
            format_func=lambda g: cv.EXTERNAL_GEOMETRIES[g],
            key=f"{key}_geo",
        )
    V = number_input_si(
        label="Velocidad del fluido V",
        kind="speed",
        default_si=inp.V_m_per_s,
        key=f"{key}_V",
        format="%.4g",
    )
    T_s, T_inf = _temperatures(inp.T_s_K, inp.T_inf_K, key, "T∞ del fluido")
    left, right = st.columns(2)
    width = inp.width_m
    with left:
        if geometry == "plate":
            L = number_input_si(
                label="Largo en el sentido del flujo L",
                kind="length",
                default_si=inp.length_m if inp.geometry == "plate" else 1.0,
                key=f"{key}_plate_L",
                format="%.4g",
            )
        else:
            L = number_input_si(
                label="Diámetro D",
                kind="small_length",
                default_si=inp.length_m if inp.geometry != "plate" else 0.05,
                key=f"{key}_{geometry}_D",
                format="%.4g",
            )
    with right:
        if geometry != "sphere":
            width = number_input_si(
                label="Ancho w" if geometry == "plate" else "Largo del cilindro",
                kind="length",
                default_si=inp.width_m,
                key=f"{key}_{geometry}_w",
                format="%.4g",
            )
    p = _pressure(inp.p_Pa, key)
    tripped = False
    if geometry == "plate":
        tripped = st.checkbox(
            "Turbulenta desde el borde de ataque (un alambre o una rugosidad la dispara)",
            value=inp.tripped,
            key=f"{key}_trip",
        )
    try:
        result = _external_cached(
            cv.ExternalFlowInputs(fluid, geometry, V, T_s, T_inf, L, width, p, tripped)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_convection(result, system, "qv_ext")


def _natural(system: UnitSystem) -> None:
    name, idx = _example(
        cv.NATURAL_EXAMPLES,
        "qv_nat_example",
        "Ejemplos del cap. 9 de Cengel y Ghajar. Podés cambiar cualquier dato.",
    )
    ex = cv.NATURAL_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, cv.NaturalConvectionInputs)
    key = f"qv_n{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    left, right = st.columns(2)
    with left:
        fluid = _fluid_input(inp.fluid, f"{key}_fluid")
    with right:
        geos = list(cv.NATURAL_GEOMETRIES)
        geometry: cv.NaturalGeometry = st.selectbox(
            "Geometría",
            geos,
            index=geos.index(inp.geometry),
            format_func=lambda g: cv.NATURAL_GEOMETRIES[g],
            key=f"{key}_geo",
        )
    T_s, T_inf = _temperatures(inp.T_s_K, inp.T_inf_K, key, "T∞ del fluido quieto")
    plate = geometry in ("vertical_plate", "horizontal_plate_up", "horizontal_plate_down")
    left, right = st.columns(2)
    width = inp.width_m
    with left:
        if plate:
            L = number_input_si(
                label="Alto L" if geometry == "vertical_plate" else "Largo",
                kind="length",
                default_si=inp.length_m,
                key=f"{key}_{geometry}_L",
                format="%.4g",
            )
        else:
            L = number_input_si(
                label="Diámetro D",
                kind="small_length",
                default_si=inp.length_m,
                key=f"{key}_{geometry}_D",
                format="%.4g",
            )
    with right:
        if geometry != "sphere":
            width = number_input_si(
                label="Ancho" if plate else "Largo del cilindro",
                kind="length",
                default_si=inp.width_m,
                key=f"{key}_{geometry}_w",
                format="%.4g",
            )
    p = _pressure(inp.p_Pa, key)
    try:
        result = _natural_cached(
            cv.NaturalConvectionInputs(fluid, geometry, T_s, T_inf, L, width, p)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_convection(result, system, "qv_nat")


def _props_table(props: cv.Properties, system: UnitSystem, natural: bool) -> None:
    rows = [
        ("ρ", props.rho_kg_per_m3, "density"),
        ("μ", props.mu_Pa_s, "dynamic_viscosity"),
        ("ν = μ/ρ", props.nu_m2_per_s, "diffusivity"),
        ("k", props.k_W_per_mK, _K),
        ("c_p", props.cp_J_per_kgK, "specific_heat"),
    ]
    data = [
        {"Propiedad": sym, "Valor": _val(v, kind, system), "Unidad": _u(kind, system)}  # type: ignore[arg-type]
        for sym, v, kind in rows
    ]
    data.append({"Propiedad": "Pr", "Valor": _num(props.Pr, 4), "Unidad": "—"})
    if natural:
        data.append(
            {
                "Propiedad": "β",
                "Valor": _val(props.beta_per_K, "expansion_coefficient", system),
                "Unidad": _u("expansion_coefficient", system),
            }
        )
    _table(pd.DataFrame(data))


def _comparison(
    alternatives: tuple[cv.Alternative, ...], main: str, system: UnitSystem, key: str
) -> None:
    st.markdown("#### Las correlaciones, comparadas")
    _table(
        pd.DataFrame(
            [
                {
                    "Correlación": a.name + (" ←" if a.key == main else ""),
                    "Nu": _num(a.Nu, 4),
                    f"h [{_u(_H, system)}]": _val(a.h_W_per_m2K, _H, system),
                    "¿En su rango?": "sí" if a.in_range else "no",
                    "Rango": cv.CORRELATIONS[a.key].validity,
                }
                for a in alternatives
            ]
        )
    )
    if len(alternatives) > 1:
        st.plotly_chart(
            correlation_figure(
                alternatives, main, h_scale=convert_from_si(1.0, _H, system), h_unit=_u(_H, system)
            ),
            width="stretch",
            key=f"{key}_corr",
        )


def _nu_curve(result: cv.ConvectionResult | cv.InternalFlowResult, key: str) -> None:
    var, xs, nus = cv.nu_curve(result)
    x0 = (result.Ra if var == "Ra" else result.Re) or 1.0
    name = cv.CORRELATIONS[result.correlation].name
    st.plotly_chart(
        nu_curve_figure(var, xs, nus, x0, result.Nu, name), width="stretch", key=f"{key}_nu"
    )
    st.caption(
        f"La correlación usada como curva Nu({var}), con el Pr de los datos: el punto naranja "
        "es este caso."
    )


def _render_convection(r: cv.ConvectionResult, system: UnitSystem, key: str) -> None:
    st.markdown("### Resultado")
    items = []
    if r.is_natural:
        items += [("Grashof Gr", _num(r.Gr, 4)), ("Rayleigh Ra", _num(r.Ra, 4))]
    else:
        items.append(("Reynolds Re", _num(r.Re, 4)))
    items += [
        ("Nusselt Nu", _num(r.Nu, 4)),
        (f"h [{_u(_H, system)}]", _val(r.h_W_per_m2K, _H, system)),
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system)),
        (f"q″ [{_u('heat_flux', system)}]", _val(r.q_W_per_m2, "heat_flux", system)),
    ]
    _metric_rows(items)
    corr = cv.CORRELATIONS[r.correlation]
    st.caption(f"Régimen: **{r.regime}**. Correlación: **{corr.name}** ({corr.validity}).")
    _notes(cv.convection_notes(r))
    where = (
        "a T∞ (Whitaker)"
        if (not r.is_natural and r.inputs.geometry == "sphere")
        else "a la temperatura de película T_f"
    )
    st.markdown(
        f"#### Propiedades del fluido {where}: {_val(r.props.T_K, _T, system)} {_u(_T, system)}"
    )
    _props_table(r.props, system, r.is_natural)
    _comparison(r.alternatives, r.correlation, system, key)
    try:
        _nu_curve(r, key)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar la curva: {exc}")
    _render_steps(convection_steps(r, system))
    _render_export(cv.convection_to_dict(r, system), "conveccion", f"{key}_dl")


def _internal(system: UnitSystem) -> None:
    name, idx = _example(
        cv.INTERNAL_EXAMPLES,
        "qv_int_example",
        "Ejemplos del cap. 8 de Cengel y Ghajar. Podés cambiar cualquier dato.",
    )
    ex = cv.INTERNAL_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, cv.InternalFlowInputs)
    key = f"qv_i{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid = _fluid_input(inp.fluid, f"{key}_fluid")
    left, right = st.columns(2)
    with left:
        D = number_input_si(
            label="Diámetro interior D",
            kind="small_length",
            default_si=inp.D_m,
            key=f"{key}_D",
            format="%.4g",
        )
        m_dot = number_input_si(
            label="Caudal ṁ",
            kind="mass_flow",
            default_si=inp.m_dot_kg_s,
            key=f"{key}_m",
            format="%.4g",
        )
    with right:
        L = number_input_si(
            label="Largo L", kind="length", default_si=inp.L_m, key=f"{key}_L", format="%.4g"
        )
        T_in = number_input_si(
            label="T de entrada", kind=_T, default_si=inp.T_in_K, key=f"{key}_Tin", format="%.5g"
        )
    conditions = list(cv.INTERNAL_CONDITIONS)
    condition: cv.InternalCondition = st.radio(
        "La pared",
        conditions,
        index=conditions.index(inp.condition),
        format_func=lambda c: cv.INTERNAL_CONDITIONS[c],
        key=f"{key}_cond",
        horizontal=True,
    )
    T_s = q = None
    if condition == "constant_T":
        T_s = number_input_si(
            label="T de la pared T_s",
            kind=_T,
            default_si=inp.T_s_K or inp.T_in_K + 40.0,
            key=f"{key}_Ts",
            format="%.5g",
        )
    else:
        q = number_input_si(
            label="Flujo de calor por la pared q″ (+ entra al fluido)",
            kind="heat_flux",
            default_si=inp.q_W_per_m2 or 1000.0,
            key=f"{key}_q",
            format="%.5g",
        )
    p = _pressure(inp.p_Pa, key)
    options = [_AUTO, *cv.TUBE_CORRELATIONS]
    correlation = st.selectbox(
        "Correlación",
        options,
        index=options.index(inp.correlation),
        format_func=lambda c: (
            "Automática (Hausen o 3,66/4,36 si es laminar, Gnielinski si no)"
            if c == _AUTO
            else cv.CORRELATIONS[c].name
        ),
        key=f"{key}_corr",
    )
    try:
        result = _internal_cached(
            cv.InternalFlowInputs(fluid, D, L, m_dot, T_in, condition, T_s, q, p, correlation)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_internal(result, system)


def _render_internal(r: cv.InternalFlowResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    items = [
        ("Reynolds Re", _num(r.Re, 4)),
        ("Nusselt Nu", _num(r.Nu, 4)),
        (f"h [{_u(_H, system)}]", _val(r.h_W_per_m2K, _H, system)),
        (f"T de salida [{_u(_T, system)}]", _val(r.T_out_K, _T, system)),
        (f"Q̇ al fluido [{_u(_W, system)}]", _val(r.Q_W, _W, system)),
        (f"Δp [{_u('pressure_drop', system)}]", _val(r.dp_Pa, "pressure_drop", system)),
    ]
    if r.T_s_out_K is not None:
        items.append(
            (f"T de la pared a la salida [{_u(_T, system)}]", _val(r.T_s_out_K, _T, system))
        )
    if r.dT_lm_K is not None:
        items.append(
            (
                f"ΔT_ml [{_u('temperature_difference', system)}]",
                _val(r.dT_lm_K, "temperature_difference", system),
            )
        )
    _metric_rows(items)
    corr = cv.CORRELATIONS[r.correlation]
    st.caption(f"Régimen: **{r.regime}**. Correlación: **{corr.name}** ({corr.validity}).")
    _notes(cv.internal_notes(r))
    t_scale, t_offset = _t_scale(system)
    try:
        st.plotly_chart(
            tube_figure(
                r,
                x_scale=convert_from_si(1.0, "length", system),
                x_unit=_u("length", system),
                t_scale=t_scale,
                t_offset=t_offset,
                t_unit=_u(_T, system),
            ),
            width="stretch",
            key="qv_int_tube",
        )
        st.caption(
            "Con la pared a T constante, el fluido se acerca a ella en forma exponencial; con "
            "flujo constante, sube lineal y la pared lo acompaña q″/h más arriba (con el h "
            "promedio; cerca de la entrada h es mayor)."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar el perfil: {exc}")
    T_m = f"{_val(r.T_mean_K, _T, system)} {_u(_T, system)}"
    st.markdown(f"#### Propiedades a la temperatura media T_m: {T_m}")
    _props_table(r.props, system, natural=False)
    _comparison(r.alternatives, r.correlation, system, "qv_int")
    try:
        _nu_curve(r, "qv_int")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar la curva: {exc}")
    _render_steps(internal_steps(r, system))
    _render_export(cv.internal_to_dict(r, system), "tubo", "qv_int_dl")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) "
            "todavía no tiene un capítulo de transferencia de calor: las fórmulas siguen a "
            "**Çengel y Ghajar (2015)**, caps. 3, 7, 8 y 9, y a **Incropera et al. (2007)**, "
            "caps. 3, 7, 8 y 9."
        )
        st.markdown(
            "**Conducción en régimen permanente** (ley de Fourier, sin generación). Cada "
            "elemento entre dos temperaturas es una resistencia R = ΔT/Q̇:"
        )
        st.latex(r"R_{\text{pared}} = \frac{L}{k\,A}")
        st.latex(r"R_{\text{cil}} = \frac{\ln(r_2/r_1)}{2\pi\,k\,L}")
        st.latex(r"R_{\text{esf}} = \frac{r_2 - r_1}{4\pi\,k\,r_1\,r_2}")
        st.latex(r"R_{\text{conv}} = \frac{1}{h\,A} \qquad R_c = \frac{R''_c}{A}")
        st.markdown(
            "En serie se suman; en paralelo (caras isotérmicas) se suman las inversas. El "
            "calor y el coeficiente global:"
        )
        st.latex(r"\dot{Q} = \frac{T_{\infty,1} - T_{\infty,2}}{\sum R} = U\,A\,\Delta T")
        st.markdown(
            "**Radio crítico de aislación**: agregar aislación a un cilindro o una esfera "
            "aumenta el calor mientras r < r_cr:"
        )
        st.latex(r"r_{\text{cr,cil}} = \frac{k}{h} \qquad r_{\text{cr,esf}} = \frac{2k}{h}")
        st.markdown(
            "**Aletas de sección constante** (temperatura uniforme en cada sección, Bi < 0,1): "
            "θ = T − T∞ cumple θ'' = m²·θ (Incropera, tabla 3.4):"
        )
        st.latex(r"m = \sqrt{\frac{h\,P}{k\,A_c}} \qquad M = \sqrt{h\,P\,k\,A_c}\;\theta_b")
        st.latex(r"\dot{Q}_{\text{adiab}} = M\,\tanh mL \qquad \dot{Q}_{\infty} = M")
        st.latex(
            r"\dot{Q}_{\text{conv}} = M\,\frac{\sinh mL + \frac{h}{mk}\cosh mL}"
            r"{\cosh mL + \frac{h}{mk}\sinh mL}"
        )
        st.latex(r"L_c = L + \frac{t}{2}\ \text{(recta)},\quad L + \frac{D}{4}\ \text{(aguja)}")
        st.latex(
            r"\eta_f = \frac{\dot{Q}}{h\,A_f\,\theta_b} \qquad "
            r"\varepsilon_f = \frac{\dot{Q}}{h\,A_c\,\theta_b}"
        )
        st.latex(r"\eta_o = 1 - \frac{N\,A_f}{A_t}\,(1 - \eta_f)")
        st.markdown(
            "La **aleta anular** sigue la ecuación de Bessel modificada de orden cero: "
            "θ = C₁·I₀(mr) + C₂·K₀(mr), con m = √(2h/(k·t))."
        )
        st.markdown(
            "**Convección**: h sale del número de Nusselt, que las correlaciones dan en función "
            "de Re (forzada) o Ra (natural) y Pr:"
        )
        st.latex(r"Nu = \frac{h\,L_c}{k} \qquad Re = \frac{V\,L_c}{\nu}")
        st.latex(r"Pr = \frac{\nu}{\alpha} = \frac{\mu\,c_p}{k}")
        st.latex(r"Gr = \frac{g\,\beta\,(T_s - T_\infty)\,L_c^3}{\nu^2}")
        st.latex(r"Ra = Gr\,Pr")
        st.latex(r"\dot{Q} = h\,A\,(T_s - T_\infty)")
        st.markdown(
            "Las propiedades van a la temperatura de película T_f = (T_s + T∞)/2 (salvo en "
            "Whitaker, a T∞) o, en un tubo, a la media del fluido. Las correlaciones de la "
            "página:"
        )
        for c in cv.CORRELATIONS.values():
            st.markdown(f"- **{c.name}** ({c.validity}):")
            st.latex(c.latex)
        st.markdown(
            "**Tubo**: con la pared a T_s constante el fluido sale a "
            "T_sal = T_s − (T_s − T_ent)·e^(−hA/(ṁc_p)) y Q̇ = h·A·ΔT_ml; con flujo constante, "
            "T_sal = T_ent + q″·A/(ṁ·c_p). La caída de presión, de Darcy–Weisbach:"
        )
        st.latex(r"\Delta p = f\,\frac{L}{D}\,\frac{\rho\,V^2}{2}")


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Transferencia de calor", page_icon="🧱", layout="centered")

st.subheader(SUBJECT)
st.title("🧱 Transferencia de calor")
st.markdown(
    "La **conducción** a través de paredes, caños y esferas de varias capas (la red de "
    "resistencias térmicas y el radio crítico de aislación), las **aletas** (con su eficiencia "
    "y su efectividad) y el coeficiente de **convección** h de las correlaciones de Nusselt, "
    "forzada y natural. La radiación y los intercambiadores llegan en la próxima entrega."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Transferencia de calor")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="ht_mode")
if mode == _M_COND:
    _conduction(system)
elif mode == _M_FIN:
    _fins(system)
else:
    what = st.radio("¿Qué convección?", _CONV_WHAT, key="qv_what", horizontal=True)
    if what == _V_EXT:
        _external(system)
    elif what == _V_INT:
        _internal(system)
    else:
        _natural(system)
