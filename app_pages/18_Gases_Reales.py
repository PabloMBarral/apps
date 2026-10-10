"""Página 18 — Gases reales.

Fase 9.2. Tres modos (vademecum §7, §8 y §15):

- **El factor de compresibilidad**: con dos de p, T y v, Z y la incógnita con seis
  modelos (gas ideal, la carta generalizada, Lee–Kesler con ω, Van der Waals,
  Peng–Robinson y CoolProp como referencia), la carta generalizada calculada y Z(p) a
  la temperatura del dato.
- **Van der Waals y Peng–Robinson**: las raíces de cada cúbica en (p, T), cuál es la
  estable, la construcción de Maxwell (áreas iguales) y la presión de vapor de cada
  modelo contra la real.
- **Funciones características y Maxwell**: las cuatro relaciones de Maxwell exactas y
  por diferencias finitas (como con las tablas), α, κ_T y la relación de Mayer
  generalizada; la ecuación de Clapeyron y la de Clausius–Clapeyron; el coeficiente de
  Joule–Thomson con su curva de inversión.

El cálculo vive en :mod:`core.gases`; los gráficos, en :mod:`ui.real_gas_charts`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.gases import cubic as cb
from core.gases import real as rg
from core.gases import relations as rl
from core.gases.real_procedure import (
    clapeyron_steps,
    compressibility_steps,
    cubic_steps,
    joule_thomson_steps,
    relations_steps,
)
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.real_gas_charts import (
    clausius_figure,
    convergence_figure,
    fluid_isotherm_figure,
    generalized_chart_figure,
    inversion_figure,
    saturation_figure,
    vdw_isotherms_figure,
    z_isotherm_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.27.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_Z = "El factor de compresibilidad"
_M_CUBIC = "Van der Waals y Peng–Robinson"
_M_REL = "Funciones características y Maxwell"
_MODES = (_M_Z, _M_CUBIC, _M_REL)

_REL_MAXWELL = "Relaciones de Maxwell en un estado"
_REL_CLAP = "Clapeyron (el cambio de fase)"
_REL_JT = "Joule–Thomson"
_REL_KINDS = (_REL_MAXWELL, _REL_CLAP, _REL_JT)

_PAIRS: dict[rg.StatePairKind, str] = {"pT": "p y T", "Tv": "T y v", "pv": "p y v"}
_PHASES = {"liquid": "líquido", "vapor": "vapor", "supercritical": "supercrítico"}

_P: QuantityKind = "pressure"
_T: QuantityKind = "temperature"
_TA: QuantityKind = "absolute_temperature"
_DT: QuantityKind = "temperature_difference"
_V: QuantityKind = "specific_volume"
_E: QuantityKind = "specific_enthalpy"
_S: QuantityKind = "specific_entropy"
_C: QuantityKind = "specific_heat"
_UNKNOWN_KIND: dict[str, QuantityKind] = {"v": _V, "p": _P, "T": _T}
_UNKNOWN_NAME = {"v": "v", "p": "p", "T": "T"}


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


def _signed_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.{digits}f} %".replace(".", ",")


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


def _example(examples: dict[str, Any], key: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox(
        "Ejemplo precargado",
        names,
        key=key,
        help="Podés cambiar el fluido y cualquier dato: el resultado se actualiza solo.",
    )
    return name, names.index(name)


def _fluid_select(default: str, key: str) -> str:
    options = list(rg.REAL_FLUIDS)
    return st.selectbox(
        "Fluido",
        options,
        index=options.index(default) if default in options else 0,
        format_func=lambda k: rg.real_fluid(k).label,
        key=key,
        help="Los nueve de la tabla de propiedades críticas del vademecum (§7.5) y los "
        "refrigerantes e hidrocarburos de la app. El punto crítico y ω, de CoolProp.",
    )


def _diagram(build: Any, key: str, caption: str = "") -> None:
    """Un gráfico que falla no tumba la página."""
    try:
        st.plotly_chart(build(), width="stretch", key=key)
        if caption:
            st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar el gráfico: {exc}")


def _p_input(label: str, default: float, key: str) -> float:
    return number_input_si(
        label=label, kind=_P, default_si=default, key=key, format="%.5g", min_value_si=0.0
    )


def _T_input(label: str, default: float, key: str) -> float:
    return number_input_si(label=label, kind=_T, default_si=default, key=key, format="%.5g")


def _critical_rows(fl: rg.RealFluid, system: UnitSystem) -> list[dict[str, str]]:
    return [
        {"Propiedad": "T_cr", "Valor": _val(fl.T_cr, _T, system), "Unidad": _u(_T, system)},
        {"Propiedad": "p_cr", "Valor": _val(fl.p_cr, _P, system), "Unidad": _u(_P, system)},
        {"Propiedad": "v_cr", "Valor": _val(fl.v_cr, _V, system), "Unidad": _u(_V, system)},
        {"Propiedad": "Z_cr", "Valor": _num(fl.Z_cr, 4), "Unidad": "—"},
        {"Propiedad": "ω (factor acéntrico)", "Valor": _num(fl.omega, 4), "Unidad": "—"},
    ]


# ---------------------------------------------------------------------
# El factor de compresibilidad
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _z_cached(inputs: rg.CompressibilityInputs) -> rg.CompressibilityResult:
    return rg.compressibility(inputs)


@st.cache_data(show_spinner=False)
def _chart_cached() -> dict[str, Any]:
    return rg.generalized_chart()


@st.cache_data(show_spinner=False)
def _z_isotherm_cached(fluid_key: str, T_K: float, p_max: float) -> dict[str, Any]:
    return rg.z_isotherm(fluid_key, T_K, p_max)


def _z_mode(system: UnitSystem) -> None:
    name, idx = _example(rg.REAL_EXAMPLES, "rz_example")
    ex = rg.REAL_EXAMPLES[name]
    key = f"rz_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid = _fluid_select(ex.inputs.fluid_key, f"{key}_fluid")
    pairs = list(_PAIRS)
    pair: rg.StatePairKind = st.radio(
        "Con",
        pairs,
        index=pairs.index(ex.inputs.pair),
        format_func=lambda k: _PAIRS[k],
        key=f"{key}_pair",
        horizontal=True,
    )
    # los valores por defecto de cualquier par salen del estado real del ejemplo
    base = _z_cached(ex.inputs)
    values = {"p": base.p_Pa, "T": base.T_K, "v": base.v_m3_per_kg}
    left, right = st.columns(2)
    data: dict[str, float] = {}
    for col, var in zip((left, right), (pair[0], pair[1]), strict=True):
        with col:
            if var == "p":
                data["p"] = _p_input("Presión p (absoluta)", values["p"], f"{key}_{pair}_p")
            elif var == "T":
                data["T"] = _T_input("Temperatura T", values["T"], f"{key}_{pair}_T")
            else:
                data["v"] = number_input_si(
                    label="Volumen específico v",
                    kind=_V,
                    default_si=values["v"],
                    key=f"{key}_{pair}_v",
                    format="%.5g",
                    min_value_si=0.0,
                )
    a, b = {"pT": (data.get("p"), data.get("T")), "Tv": (data.get("T"), data.get("v"))}.get(
        pair, (data.get("p"), data.get("v"))
    )
    try:
        r = _z_cached(rg.CompressibilityInputs(fluid, pair, float(a or 0), float(b or 0)))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_compressibility(r, system)


def _render_compressibility(r: rg.CompressibilityResult, system: UnitSystem) -> None:
    fl = r.fluid
    kind = _UNKNOWN_KIND[r.unknown]
    unknown = {"v": r.v_m3_per_kg, "p": r.p_Pa, "T": r.T_K}[r.unknown]
    best = r.best
    st.markdown("### Resultado")
    _metric_rows(
        [
            ("Z real (CoolProp)", _num(r.Z, 4)),
            (f"{_UNKNOWN_NAME[r.unknown]} real [{_u(kind, system)}]", _val(unknown, kind, system)),
            ("Error del gas ideal", _signed_pct(r.model("ideal").error, 1)),
            (
                "El que menos se equivoca",
                "—" if best is None else rg.MODELS[best.model].split(" (")[0],
            ),
        ]
    )
    for note in r.notes:
        st.info(note)

    st.markdown("#### Los seis modelos")
    rows = []
    for m in r.models:
        value = {"v": m.v_m3_per_kg, "p": m.p_Pa, "T": m.T_K}[r.unknown]
        rows.append(
            {
                "Modelo": rg.MODELS[m.model],
                "Z": _num(m.Z, 4),
                f"{_UNKNOWN_NAME[r.unknown]} [{_u(kind, system)}]": _val(value, kind, system),
                "Error": "—" if m.model == "coolprop" else _signed_pct(m.error),
            }
        )
    _table(pd.DataFrame(rows))
    missing = [m for m in r.models if m.Z is None]
    for m in missing:
        st.warning(f"**{rg.MODELS[m.model]}**: {m.note}")
    st.caption(
        "El error es el de la incógnita contra CoolProp, la ecuación de estado de referencia. "
        "La carta generalizada es el fluido simple de Lee y Kesler (dos parámetros, como la de "
        "Çengel); con ω suma el fluido de referencia (tres parámetros)."
    )

    st.markdown("#### El fluido y las variables reducidas")
    _table(
        pd.DataFrame(
            [
                *_critical_rows(fl, system),
                {"Propiedad": "T_R = T/T_cr", "Valor": _num(r.T_R, 4), "Unidad": "—"},
                {"Propiedad": "p_R = p/p_cr", "Valor": _num(r.p_R, 4), "Unidad": "—"},
                {"Propiedad": "v_R = v/v_cr", "Valor": _num(r.v_R, 4), "Unidad": "—"},
                {
                    "Propiedad": "v′_R = v·p_cr/(R·T_cr)",
                    "Valor": _num(r.v_R_pseudo, 4),
                    "Unidad": "—",
                },
                {"Propiedad": "fase", "Valor": _PHASES.get(r.phase, r.phase), "Unidad": "—"},
            ]
        )
    )

    p_max = min(max(2.0 * fl.p_cr, 1.3 * r.p_Pa), fl.p_max)
    _diagram(
        lambda: z_isotherm_figure(_z_isotherm_cached(fl.key, r.T_K, p_max), system, (r.p_Pa, r.Z)),
        "rz_iso",
        "Z contra p a la temperatura del dato. A p → 0 todos dan Z = 1 (gas ideal); bajo la "
        "temperatura crítica, en p_sat el vapor se condensa y Z salta al del líquido. Cada "
        "modelo toma la raíz de la misma fase que el fluido real.",
    )
    lk0 = r.model("lk0")
    on_chart = 0.01 <= r.p_R <= 10.0
    model_point = (
        (lk0.p_Pa / fl.p_cr, lk0.Z)
        if lk0.Z is not None and lk0.p_Pa is not None and 0.01 <= lk0.p_Pa / fl.p_cr <= 10.0
        else None
    )
    _diagram(
        lambda: generalized_chart_figure(
            _chart_cached(), (r.p_R, r.Z) if on_chart else None, model_point
        ),
        "rz_chart",
        "La carta generalizada (Çengel, figura 3-51), calculada con el fluido simple de Lee y "
        "Kesler: Z⁰ contra p_R para cada T_R, con la campana. "
        + (
            "El rombo es el estado real; el círculo, lo que da la carta."
            if on_chart
            else f"El estado (p_R = {_num(r.p_R, 3)}) queda fuera de la carta (0,01 a 10)."
        ),
    )
    _render_steps(compressibility_steps(r, system))
    _render_export(rg.compressibility_to_dict(r), "factor_de_compresibilidad", "rz_dl")


# ---------------------------------------------------------------------
# Van der Waals y Peng–Robinson
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cubic_cached(inputs: cb.CubicInputs) -> cb.CubicResult:
    return cb.solve_cubic(inputs)


@st.cache_data(show_spinner=False)
def _fluid_isotherms_cached(fluid_key: str, T_K: float) -> dict[str, Any]:
    return cb.fluid_isotherms(fluid_key, T_K)


@st.cache_data(show_spinner=False)
def _vdw_isotherms_cached(Tr_values: tuple[float, ...]) -> dict[str, Any]:
    return cb.vdw_reduced_isotherms(Tr_values)


@st.cache_data(show_spinner=False)
def _saturation_cached(fluid_key: str) -> dict[str, Any]:
    return cb.saturation_curves(fluid_key)


def _cubic_mode(system: UnitSystem) -> None:
    name, idx = _example(cb.CUBIC_EXAMPLES, "rw_example")
    ex = cb.CUBIC_EXAMPLES[name]
    key = f"rw_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid = _fluid_select(ex.inputs.fluid_key, f"{key}_fluid")
    left, right = st.columns(2)
    with left:
        p = _p_input("Presión p (absoluta)", ex.inputs.p_Pa, f"{key}_p")
    with right:
        T = _T_input("Temperatura T", ex.inputs.T_K, f"{key}_T")
    try:
        r = _cubic_cached(cb.CubicInputs(fluid, p, T))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_cubic(r, system)


def _root_rows(r: cb.CubicResult, system: UnitSystem) -> list[dict[str, str]]:
    rows = []
    for model in ("vdw", "pr"):
        sol = r.solution(model)  # type: ignore[arg-type]
        for root in sol.roots:
            rows.append(
                {
                    "Modelo": cb.CUBIC_MODELS[model],  # type: ignore[index]
                    "Z": _num(root.Z, 5),
                    f"v [{_u(_V, system)}]": _val(root.v_m3_per_kg, _V, system),
                    "Raíz": cb.ROOT_NAMES[root.kind],
                    "ln φ": _num(root.ln_phi, 4),
                    "Estable": "✓" if root.stable else "",
                }
            )
    return rows


def _render_cubic(r: cb.CubicResult, system: UnitSystem) -> None:
    fl = r.fluid
    st.markdown("### Resultado")
    items = [
        (f"Real: {_PHASES[r.phase]}, v [{_u(_V, system)}]", _val(r.v_m3_per_kg, _V, system)),
    ]
    for model in ("vdw", "pr"):
        v = r.model_v(model)  # type: ignore[arg-type]
        name = cb.CUBIC_MODELS[model]  # type: ignore[index]
        err = "—" if v is None else _signed_pct(v / r.v_m3_per_kg - 1.0, 1)
        items.append((f"{name}: error en v", err))
    items.append(("Z_cr real (Van der Waals: 0,375)", _num(fl.Z_cr, 4)))
    _metric_rows(items)
    for note in r.notes:
        st.info(note)

    st.markdown("#### Las raíces de cada cúbica")
    _table(pd.DataFrame(_root_rows(r, system)))
    st.caption(
        "Con tres raíces, la menor es el líquido, la mayor el vapor y la del medio no tiene "
        "sentido físico. La estable es la de menor coeficiente de fugacidad φ (la de menor "
        "energía libre de Gibbs); la otra es metaestable."
    )

    st.markdown("#### La saturación a esta temperatura")
    sat_rows = []
    for label, p_sat, v_f, v_g, omega in (
        ("Real (CoolProp)", r.p_sat_Pa, r.v_f, r.v_g, fl.omega),
        *(
            (
                cb.CUBIC_MODELS[m],  # type: ignore[index]
                s.p_sat_Pa if s else None,
                s.v_f if s else None,
                s.v_g if s else None,
                r.omega_vdw if m == "vdw" else r.omega_pr,
            )
            for m, s in (("vdw", r.vdw.saturation), ("pr", r.pr.saturation))
        ),
    ):
        sat_rows.append(
            {
                "": label,
                f"p_sat [{_u(_P, system)}]": _val(p_sat, _P, system),
                f"v_f [{_u(_V, system)}]": _val(v_f, _V, system),
                f"v_g [{_u(_V, system)}]": _val(v_g, _V, system),
                "ω": _num(omega, 3),
            }
        )
    _table(pd.DataFrame(sat_rows))
    st.caption(
        "La presión de saturación de cada cúbica sale de la construcción de Maxwell (áreas "
        "iguales). El ω de cada modelo es el que implica su presión de vapor a T_R = 0,7."
    )

    _diagram(
        lambda: fluid_isotherm_figure(
            _fluid_isotherms_cached(fl.key, r.inputs.T_K),
            system,
            (r.v_m3_per_kg, r.inputs.p_Pa),
            fl.p_cr,
        ),
        "rw_iso",
        "La isoterma del dato: la real tiene un tramo horizontal (la campana); las cúbicas, un "
        "lazo que llega a presiones negativas. Las rayadas son las rectas de Maxwell de cada "
        "cúbica: su presión de saturación.",
    )
    highlight = round(r.T_R, 2) if 0.5 <= r.T_R < 0.99 else 0.9
    family = tuple(sorted({0.8, 0.9, 1.0, 1.1, 1.2, highlight}))
    _diagram(
        lambda: vdw_isotherms_figure(_vdw_isotherms_cached(family), highlight),
        "rw_vdw",
        f"Van der Waals en forma reducida (vademecum §8.4), igual para todos los fluidos. A "
        f"T_R = {_num(highlight, 3)} la recta de Maxwell corta dos áreas iguales en el lazo.",
    )
    _diagram(
        lambda: saturation_figure(_saturation_cached(fl.key), system),
        "rw_sat",
        "La presión de vapor: Van der Waals la sobreestima mucho (su ω es −0,30); Peng–Robinson "
        "ajusta α(T) con el ω del fluido y casi coincide con la real.",
    )
    _render_steps(cubic_steps(r, system))
    _render_export(cb.cubic_to_dict(r), "van_der_waals_y_peng_robinson", "rw_dl")


# ---------------------------------------------------------------------
# Funciones características y Maxwell
# ---------------------------------------------------------------------

_DERIV_NAMES = {
    "dT_dv_s": "(∂T/∂v)_s",
    "dp_ds_v": "(∂p/∂s)_v",
    "dT_dp_s": "(∂T/∂p)_s",
    "dv_ds_p": "(∂v/∂s)_p",
    "ds_dv_T": "(∂s/∂v)_T",
    "dp_dT_v": "(∂p/∂T)_v",
    "ds_dp_T": "(∂s/∂p)_T",
    "dv_dT_p": "(∂v/∂T)_p",
}
_TARGET: dict[str, QuantityKind] = {
    "u": "temperature_per_volume",
    "h": "temperature_per_pressure",
    "f": "pressure_per_temperature",
    "g": "volume_per_temperature",
}
_NATURAL_NAMES = {
    "du_ds_v": "(∂u/∂s)_v",
    "du_dv_s": "(∂u/∂v)_s",
    "dh_ds_p": "(∂h/∂s)_p",
    "dh_dp_s": "(∂h/∂p)_s",
    "df_dT_v": "(∂f/∂T)_v",
    "df_dv_T": "(∂f/∂v)_T",
    "dg_dT_p": "(∂g/∂T)_p",
    "dg_dp_T": "(∂g/∂p)_T",
}
_NATURAL_KIND: dict[str, QuantityKind] = {"T": _TA, "-p": _P, "v": _V, "-s": _S}


@st.cache_data(show_spinner=False)
def _relations_cached(inputs: rl.RelationsInputs) -> rl.RelationsResult:
    return rl.relations(inputs)


@st.cache_data(show_spinner=False)
def _clapeyron_cached(inputs: rl.ClapeyronInputs) -> rl.ClapeyronResult:
    return rl.clapeyron(inputs)


@st.cache_data(show_spinner=False)
def _jt_cached(inputs: rl.JouleThomsonInputs) -> rl.JouleThomsonResult:
    return rl.joule_thomson(inputs)


@st.cache_data(show_spinner=False)
def _inversion_cached(fluid_key: str) -> dict[str, list[float]]:
    return rl.inversion_curve(fluid_key, 80)


@st.cache_data(show_spinner=False)
def _isenthalps_cached(
    fluid_key: str, T_values: tuple[float, ...], p_ref: float, p_max: float
) -> dict[float, tuple[list[float], list[float]]]:
    return rl.isenthalps(fluid_key, T_values, p_ref, p_max)


def _maxwell_mode(system: UnitSystem) -> None:
    name, idx = _example(rl.RELATIONS_EXAMPLES, "rm_example")
    ex = rl.RELATIONS_EXAMPLES[name]
    key = f"rm_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid = _fluid_select(ex.inputs.fluid_key, f"{key}_fluid")
    left, right = st.columns(2)
    with left:
        p = _p_input("Presión p (absoluta)", ex.inputs.p_Pa, f"{key}_p")
        dT = number_input_si(
            label="Paso ΔT",
            kind=_DT,
            default_si=ex.inputs.dT_K,
            key=f"{key}_dT",
            format="%.4g",
            min_value_si=0.0,
            help="Las diferencias usan T ± ΔT (en la isobara y la isócora), como dos entradas "
            "de una tabla.",
        )
    with right:
        T = _T_input("Temperatura T", ex.inputs.T_K, f"{key}_T")
        dp = number_input_si(
            label="Paso Δp",
            kind=_P,
            default_si=ex.inputs.dp_Pa or 0.1 * ex.inputs.p_Pa,
            key=f"{key}_dp",
            format="%.4g",
            min_value_si=0.0,
            help="Las diferencias usan p ± Δp (en la isoterma y la isoentrópica).",
        )
    try:
        r = _relations_cached(rl.RelationsInputs(fluid, p, T, dT, dp))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_relations(r, system)


def _render_relations(r: rl.RelationsResult, system: UnitSystem) -> None:
    s = r.state
    st.markdown("### Resultado")
    _metric_rows(
        [
            (
                f"α [{_u('expansion_coefficient', system)}]",
                _val(r.alpha, "expansion_coefficient", system, 4),
            ),
            (
                f"κ_T [{_u('isothermal_compressibility', system)}]",
                _val(r.kappa_T, "isothermal_compressibility", system, 4),
            ),
            (f"c_p − c_v [{_u(_C, system)}]", _val(r.cp - r.cv, _C, system, 4)),
            (
                f"μ_JT [{_u('temperature_per_pressure', system)}]",
                _val(r.mu_JT, "temperature_per_pressure", system, 4),
            ),
        ]
    )
    for note in r.notes:
        st.info(note)

    st.markdown("#### Los potenciales")
    _table(
        pd.DataFrame(
            [
                {"Función": name, "Valor": _val(value, kind, system), "Unidad": _u(kind, system)}
                for name, value, kind in (
                    ("v", s.v, _V),
                    ("u", s.u, _E),
                    ("h = u + p·v", s.h, _E),
                    ("s", s.s, _S),
                    ("f = u − T·s", s.f, _E),
                    ("g = h − T·s", s.g, _E),
                    ("c_p", r.cp, _C),
                    ("c_v", r.cv, _C),
                )
            ]
        )
    )
    st.caption("Los valores dependen del estado de referencia del fluido (el de CoolProp).")

    st.markdown("#### Las cuatro relaciones de Maxwell")
    rows = []
    for m in r.maxwell:
        target = _TARGET[m.potential]
        rows.append(
            {
                "Relación": f"{_DERIV_NAMES[m.left.key]} = {'−' if m.sign < 0 else ''}"
                f"{_DERIV_NAMES[m.right.key]}",
                "Unidad": _u(target, system),
                "Exacta": _val(m.left.exact, target, system),
                "Izquierda (dif.)": _val(m.left.finite, target, system),
                "Derecha (dif.)": _val(m.sign * m.right.finite, target, system),
                "Diferencia": "—" if r.degenerate else _signed_pct(m.gap_finite),
            }
        )
    _table(pd.DataFrame(rows))
    st.caption(
        f"Con ΔT = {_val(r.dT, _DT, system, 4)} {_u(_DT, system)} y Δp = "
        f"{_val(r.dp, _P, system, 4)} {_u(_P, system)}: las diferencias centradas, como con las "
        "tablas (Çengel §12-2); la exacta es la derivada de CoolProp, igual de los dos lados."
    )
    if not r.degenerate:
        _diagram(
            lambda: convergence_figure(r),
            "rm_conv",
            "Con los pasos a la mitad la diferencia entre los dos lados baja unas 4 veces: el "
            "error de una diferencia centrada va con Δ².",
        )

    st.markdown("#### Las derivadas de los potenciales (vademecum §15.2)")
    _table(
        pd.DataFrame(
            [
                {
                    "Derivada": _NATURAL_NAMES[nd.key],
                    "Debe dar": nd.expected_symbol.replace("-", "−"),
                    "Por diferencias": _val(nd.value, _NATURAL_KIND[nd.expected_symbol], system),
                    "Valor": _val(nd.expected, _NATURAL_KIND[nd.expected_symbol], system),
                    "Unidad": _u(_NATURAL_KIND[nd.expected_symbol], system),
                }
                for nd in r.natural
            ]
        )
    )

    st.markdown("#### Contra el gas ideal")
    _table(
        pd.DataFrame(
            [
                {"Magnitud": "α·T", "Real": _num(r.alpha * s.T, 4), "Gas ideal": "1"},
                {"Magnitud": "κ_T·p", "Real": _num(r.kappa_T * s.p, 4), "Gas ideal": "1"},
                {
                    "Magnitud": "(c_p − c_v)/R",
                    "Real": _num((r.cp - r.cv) / r.fluid.R, 4),
                    "Gas ideal": "1",
                },
                {
                    "Magnitud": "T·v·α²/κ_T ÷ (c_p − c_v)",
                    "Real": _num(r.mayer / (r.cp - r.cv), 6) if r.cp != r.cv else "—",
                    "Gas ideal": "1",
                },
            ]
        )
    )
    _render_steps(relations_steps(r, system))
    _render_export(rl.relations_to_dict(r), "relaciones_de_maxwell", "rm_dl")


def _clapeyron_mode(system: UnitSystem) -> None:
    name, idx = _example(rl.CLAPEYRON_EXAMPLES, "rc_example")
    ex = rl.CLAPEYRON_EXAMPLES[name]
    key = f"rc_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid = _fluid_select(ex.inputs.fluid_key, f"{key}_fluid")
    left, right = st.columns(2)
    with left:
        T = _T_input("Temperatura de saturación T₁", ex.inputs.T_K, f"{key}_T")
        T2 = _T_input("Otra temperatura T₂", ex.inputs.T2_K or ex.inputs.T_K - 10.0, f"{key}_T2")
    with right:
        dT = number_input_si(
            label="Paso ΔT",
            kind=_DT,
            default_si=ex.inputs.dT_K,
            key=f"{key}_dT",
            format="%.4g",
            min_value_si=0.0,
            help="(dp/dT)_sat con p_sat a T ± ΔT, como con dos entradas de la tabla.",
        )
    try:
        r = _clapeyron_cached(rl.ClapeyronInputs(fluid, T, dT, T2))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"h_fg real [{_u(_E, system)}]", _val(r.h_fg, _E, system)),
            (f"h_fg con Clapeyron [{_u(_E, system)}]", _val(r.h_fg_clapeyron, _E, system)),
            (f"h_fg con Clausius–Clapeyron [{_u(_E, system)}]", _val(r.h_fg_cc, _E, system)),
            ("Error de Clausius–Clapeyron", _signed_pct(r.h_fg_cc / r.h_fg - 1.0, 1)),
        ]
    )
    for note in r.notes:
        st.info(note)
    _table(
        pd.DataFrame(
            [
                {"": "p_sat(T₁)", "Valor": _val(r.p_sat, _P, system), "Unidad": _u(_P, system)},
                {"": "v_f", "Valor": _val(r.v_f, _V, system), "Unidad": _u(_V, system)},
                {"": "v_g", "Valor": _val(r.v_g, _V, system), "Unidad": _u(_V, system)},
                {
                    "": "(dp/dT)_sat exacta",
                    "Valor": _val(r.dpdT_exact, "pressure_per_temperature", system),
                    "Unidad": _u("pressure_per_temperature", system),
                },
                {
                    "": "(dp/dT)_sat por diferencias",
                    "Valor": _val(r.dpdT_finite, "pressure_per_temperature", system),
                    "Unidad": _u("pressure_per_temperature", system),
                },
                {
                    "": "p_sat(T₂) real",
                    "Valor": _val(r.p_sat2, _P, system),
                    "Unidad": _u(_P, system),
                },
                {
                    "": "p_sat(T₂) con Clausius–Clapeyron",
                    "Valor": _val(r.p_sat2_cc, _P, system),
                    "Unidad": _u(_P, system),
                },
            ]
        )
    )
    second = (
        (r.T2, r.p_sat2, r.p_sat2_cc)
        if r.T2 is not None and r.p_sat2 is not None and r.p_sat2_cc is not None
        else None
    )
    _diagram(
        lambda: clausius_figure(rl.clausius_curve(r), system, (r.T, r.p_sat), second),
        "rc_curve",
        "La presión de saturación real y la de Clausius–Clapeyron con el h_fg de T₁ constante: "
        "coinciden cerca de T₁ y se separan al acercarse al punto crítico.",
    )
    _render_steps(clapeyron_steps(r, system))
    _render_export(rl.clapeyron_to_dict(r), "clapeyron", "rc_dl")


def _jt_mode(system: UnitSystem) -> None:
    name, idx = _example(rl.JT_EXAMPLES, "rj_example")
    ex = rl.JT_EXAMPLES[name]
    key = f"rj_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid = _fluid_select(ex.inputs.fluid_key, f"{key}_fluid")
    left, right = st.columns(2)
    with left:
        p = _p_input("Presión p (absoluta)", ex.inputs.p_Pa, f"{key}_p")
        dp = number_input_si(
            label="Paso Δp",
            kind=_P,
            default_si=ex.inputs.dp_Pa or 0.1 * ex.inputs.p_Pa,
            key=f"{key}_dp",
            format="%.4g",
            min_value_si=0.0,
        )
    with right:
        T = _T_input("Temperatura T", ex.inputs.T_K, f"{key}_T")
    try:
        r = _jt_cached(rl.JouleThomsonInputs(fluid, p, T, dp))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    jt: QuantityKind = "temperature_per_pressure"
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"μ_JT [{_u(jt, system)}]", _val(r.mu, jt, system, 4)),
            (f"μ_JT por diferencias [{_u(jt, system)}]", _val(r.mu_finite, jt, system, 4)),
            *(
                (f"T de inversión {k + 1} [{_u(_T, system)}]", _val(T_inv, _T, system, 4))
                for k, T_inv in enumerate(r.T_inversion)
            ),
        ]
    )
    for note in r.notes:
        st.info(note)
    fl = r.fluid
    curve = _inversion_cached(fl.key)
    p_max = 1.15 * max(curve["p"]) if curve["p"] else 3.0 * p
    T_hi = max(curve["T"]) if curve["T"] else 2.0 * T
    T_lo = max(min(curve["T"]) if curve["T"] else fl.T_min, fl.T_min * 1.05)
    T_values = tuple(
        round(T_lo + k * (min(T_hi, fl.T_max * 0.95) - T_lo) / 6.0, 2) for k in range(1, 6)
    )
    _diagram(
        lambda: inversion_figure(
            curve,
            _isenthalps_cached(fl.key, T_values, min(1e5, 0.5 * p), p_max),
            system,
            (p, T),
        ),
        "rj_inv",
        "Adentro de la curva de inversión (μ_JT > 0) el gas se enfría al estrangularlo; afuera "
        "se calienta. Cada línea gris es una válvula (h constante): su máximo cae sobre la curva.",
    )
    _render_steps(joule_thomson_steps(r, system))
    _render_export(rl.joule_thomson_to_dict(r), "joule_thomson", "rj_dl")


def _relations_mode(system: UnitSystem) -> None:
    kind = st.radio("¿Qué calculás?", _REL_KINDS, key="rm_kind", horizontal=True)
    if kind == _REL_MAXWELL:
        _maxwell_mode(system)
    elif kind == _REL_CLAP:
        _clapeyron_mode(system)
    else:
        _jt_mode(system)


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§7** (estados correspondientes), **§8** (Van der Waals) y **§15** (funciones "
            "características). Lee–Kesler, Peng–Robinson, Clapeyron y Joule–Thomson no están en "
            "el vademecum: se citan sus fuentes (Lee y Kesler, 1975; Peng y Robinson, 1976; "
            "Çengel §12-3 y §12-5)."
        )
        st.markdown("**Estados correspondientes** (§7.1 a §7.4):")
        st.latex(
            r"p_R = \frac{p}{p_{cr}} \qquad T_R = \frac{T}{T_{cr}} \qquad v_R = \frac{v}{v_{cr}}"
        )
        st.latex(r"Z = \frac{p\,v}{R\,T} = Z(p_R,\,T_R)")
        st.latex(r"v'_R = \frac{v\,p_{cr}}{R\,T_{cr}} = Z_{cr}\,v_R")
        st.markdown(
            "**Lee y Kesler (1975)**: el fluido simple da la carta generalizada, Z⁰(p_R, T_R); "
            "con el factor acéntrico ω y el fluido de referencia (n-octano, ω_r = 0,3978):"
        )
        st.latex(r"Z = Z^0 + \frac{\omega}{\omega_r}\left(Z^r - Z^0\right)")
        st.latex(r"\omega = -\log_{10}\left.\frac{p_{sat}}{p_{cr}}\right|_{T_R = 0.7} - 1")
        st.markdown("**Van der Waals** (§8.1 a §8.4):")
        st.latex(r"\left(p + \frac{a}{v^2}\right)(v - b) = R\,T")
        st.latex(r"a = \frac{27\,R^2\,T_{cr}^2}{64\,p_{cr}} \qquad b = \frac{R\,T_{cr}}{8\,p_{cr}}")
        st.latex(r"v_{cr} = 3\,b \qquad Z_{cr} = \frac{3}{8}")
        st.latex(r"\left(p_R + \frac{3}{v_R^2}\right)\left(3\,v_R - 1\right) = 8\,T_R")
        st.latex(r"Z^3 - (1 + B)\,Z^2 + A\,Z - A\,B = 0")
        st.latex(r"A = \frac{a\,p}{(R\,T)^2} \qquad B = \frac{b\,p}{R\,T}")
        st.markdown("**Peng y Robinson (1976)**, con α(T) ajustada con ω:")
        st.latex(r"p = \frac{R\,T}{v - b} - \frac{a\,\alpha}{v^2 + 2\,b\,v - b^2}")
        st.latex(r"\alpha = \left[1 + \kappa\left(1 - \sqrt{T_R}\right)\right]^2")
        st.latex(r"\kappa = 0.37464 + 1.54226\,\omega - 0.26992\,\omega^2")
        st.markdown(
            "**Construcción de Maxwell**: la presión de saturación de una cúbica corta dos "
            "áreas iguales en el lazo (lo mismo que igualar la energía libre de Gibbs del "
            "líquido y del vapor):"
        )
        st.latex(r"\int_{v_f}^{v_g} p\,dv = p_{sat}\,(v_g - v_f)")
        st.markdown("**Funciones características** (§15.1 y §15.2):")
        st.latex(r"\begin{aligned}h &= u + p\,v \\ f &= u - T\,s \\ g &= h - T\,s\end{aligned}")
        st.latex(
            r"\begin{aligned}du &= T\,ds - p\,dv \\ dh &= T\,ds + v\,dp \\ "
            r"df &= -s\,dT - p\,dv \\ dg &= -s\,dT + v\,dp\end{aligned}"
        )
        st.markdown("**Relaciones de Maxwell** (§15.3):")
        for left, right in (
            (
                r"\left(\frac{\partial T}{\partial v}\right)_s",
                r"-\left(\frac{\partial p}{\partial s}\right)_v",
            ),
            (
                r"\left(\frac{\partial T}{\partial p}\right)_s",
                r"\left(\frac{\partial v}{\partial s}\right)_p",
            ),
            (
                r"\left(\frac{\partial s}{\partial v}\right)_T",
                r"\left(\frac{\partial p}{\partial T}\right)_v",
            ),
            (
                r"\left(\frac{\partial s}{\partial p}\right)_T",
                r"-\left(\frac{\partial v}{\partial T}\right)_p",
            ),
        ):
            st.latex(f"{left} = {right}")
        st.markdown("**Coeficientes térmicos y Mayer generalizada** (§15.4 y §15.5):")
        st.latex(
            r"\alpha = \frac{1}{v}\left(\frac{\partial v}{\partial T}\right)_p \qquad "
            r"\kappa_T = -\frac{1}{v}\left(\frac{\partial v}{\partial p}\right)_T"
        )
        st.latex(r"c_p - c_v = \frac{T\,v\,\alpha^2}{\kappa_T}")
        st.markdown("**Clapeyron** y **Clausius–Clapeyron** (Çengel §12-3):")
        st.latex(r"\left(\frac{dp}{dT}\right)_{sat} = \frac{h_{fg}}{T\,v_{fg}}")
        st.latex(
            r"\ln\frac{p_2}{p_1} \approx \frac{h_{fg}}{R}\left(\frac{1}{T_1} - \frac{1}{T_2}\right)"
        )
        st.markdown("**Joule–Thomson** (Çengel §12-5):")
        st.latex(
            r"\mu_{JT} = \left(\frac{\partial T}{\partial p}\right)_h = "
            r"\frac{T\left(\frac{\partial v}{\partial T}\right)_p - v}{c_p}"
        )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Gases reales", page_icon="🫧", layout="centered")

st.subheader(SUBJECT)
st.title("🫧 Gases reales")
st.markdown(
    "El **factor de compresibilidad** Z con seis modelos (gas ideal, la carta generalizada, "
    "Lee–Kesler, Van der Waals, Peng–Robinson y la ecuación de estado de referencia), las "
    "**cúbicas** de Van der Waals y Peng–Robinson con sus raíces y la construcción de Maxwell, "
    "y las **funciones características**: las relaciones de Maxwell con las tablas, la "
    "relación de Mayer generalizada, Clapeyron y Joule–Thomson."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Gases reales")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="rg_mode")
if mode == _M_Z:
    _z_mode(system)
elif mode == _M_CUBIC:
    _cubic_mode(system)
else:
    _relations_mode(system)
