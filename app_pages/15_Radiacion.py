"""Página 15 — Radiación.

Fase 8.2. Cinco modos:

- **Cuerpo negro y superficies reales** (Cengel y Ghajar, §12-3 y §12-5): Planck,
  Wien y Stefan–Boltzmann, la fracción emitida en una banda, ε(λ) en bandas y la
  absortividad para la radiación de otra fuente (el sol).
- **Factores de forma** (§13-1 y §13-2): ocho geometrías, con la reciprocidad y la
  regla de la suma.
- **Dos superficies y pantallas** (§13-4 y §13-5): la red de resistencias de
  placas, cilindros y esferas concéntricos, con hasta tres pantallas.
- **Recinto de varias superficies** (§13-4): el método de las radiosidades, con
  superficies a T dada, con el calor dado o rerradiantes.
- **Radiación y convección juntas** (Incropera, §1.2; Cengel y Ghajar, §13-5): el
  balance de una superficie (con el sol) y el error de una termocupla.

El cálculo vive en :mod:`core.heat_transfer.radiation`; los gráficos, en
:mod:`ui.radiation_charts`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.heat_transfer import radiation as rd
from core.heat_transfer.radiation_procedure import (
    blackbody_steps,
    enclosure_steps,
    surface_balance_steps,
    thermocouple_steps,
    two_surface_steps,
    view_factor_steps,
)
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.radiation_charts import (
    enclosure_figure,
    network_figure,
    planck_figure,
    spectral_match_figure,
    surface_curve_figure,
    view_factor_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.25.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_BB = "Cuerpo negro y superficies reales"
_M_VF = "Factores de forma"
_M_TS = "Dos superficies y pantallas"
_M_EN = "Recinto de varias superficies"
_M_CR = "Radiación y convección juntas"
_MODES = (_M_BB, _M_VF, _M_TS, _M_EN, _M_CR)

_W_SURF = "Una superficie que pierde calor"
_W_TC = "Una termocupla"
_COMBINED_WHAT = (_W_SURF, _W_TC)

_GIVEN_T = "La temperatura de la superficie"
_GIVEN_Q = "El calor que recibe (sale la T de equilibrio)"
_SURFACE_GIVEN = (_GIVEN_T, _GIVEN_Q)

_C_T = "T dada"
_C_Q = "Q̇ dado"
_C_R = "Rerradiante (Q̇ = 0)"
_CONDITIONS = (_C_T, _C_Q, _C_R)

_W: QuantityKind = "heat_rate"
_E: QuantityKind = "heat_flux"
_TA: QuantityKind = "absolute_temperature"
_T: QuantityKind = "temperature"
_LAM: QuantityKind = "wavelength"
_EL: QuantityKind = "spectral_emissive_power"
_H: QuantityKind = "heat_transfer_coefficient"

#: Medidas por defecto de cada factor de forma (cuando se cambia de geometría).
_VF_DEFAULTS: dict[rd.ViewGeometry, tuple[float, float, float]] = {
    "parallel_rectangles": (1.0, 1.0, 1.0),
    "coaxial_disks": (0.5, 0.5, 1.0),
    "perpendicular_rectangles": (1.0, 1.0, 1.0),
    "parallel_plates_2d": (1.0, 1.0, 1.0),
    "perpendicular_plates_2d": (1.0, 1.0, 0.0),
    "three_sided_2d": (1.0, 1.0, 1.0),
    "parallel_cylinders_2d": (0.05, 0.1, 0.0),
    "tube_row_2d": (0.05, 0.1, 0.0),
}

#: Las medidas de cada configuración de recinto (cuando se cambia de configuración).
_LAYOUT_DEFAULTS: dict[rd.EnclosureLayout, tuple[float, ...]] = {
    "triangular_duct": (1.0, 1.0, 1.0),
    "cylindrical_furnace": (1.0, 1.0),
    "open_cavity": (0.05, 0.2),
    "plates_surroundings": (1.0, 1.0, 0.5),
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


def _emissivity(label: str, default: float, key: str) -> float:
    return float(
        st.number_input(
            label,
            min_value=0.001,
            max_value=1.0,
            value=float(default),
            step=0.05,
            format="%.3f",
            key=key,
        )
    )


def _abs_temperature(label: str, default: float, key: str, help_text: str | None = None) -> float:
    return number_input_si(
        label=label,
        kind=_TA,
        default_si=default,
        key=key,
        format="%.5g",
        min_value_si=1.0,
        help=help_text,
    )


_ABS_HELP = "Las fórmulas de la radiación van con la temperatura absoluta (K, o °R en el Inglés)."


# ---------------------------------------------------------------------
# Cuerpo negro y superficies reales
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _blackbody_cached(inputs: rd.BlackbodyInputs) -> rd.BlackbodyResult:
    return rd.solve_blackbody(inputs)


def _band_inputs(inp: rd.BlackbodyInputs, key: str) -> tuple[rd.SpectralBand, ...]:
    n = int(
        st.number_input(
            "Cantidad de bandas",
            min_value=1,
            max_value=4,
            value=len(inp.bands) or 2,
            key=f"{key}_nb",
        )
    )
    rows: list[rd.SpectralBand] = []
    previous = 0.0
    for j in range(n):
        default = inp.bands[j] if j < len(inp.bands) else rd.SpectralBand(0.5, None)
        left, right = st.columns(2)
        with left:
            eps = st.number_input(
                f"ε de la banda {j + 1}",
                min_value=0.0,
                max_value=1.0,
                value=float(default.emissivity),
                step=0.05,
                format="%.3f",
                key=f"{key}_b{j}_e",
            )
        upper = None
        if j < n - 1:
            with right:
                upper = number_input_si(
                    label=f"Hasta λ (banda {j + 1})",
                    kind=_LAM,
                    default_si=default.upper_m or max(previous, 1e-6) + 3.0e-6,
                    key=f"{key}_b{j}_u",
                    format="%.4g",
                    min_value_si=0.0,
                )
            previous = upper
        else:
            right.caption("La última banda llega hasta λ → ∞.")
        rows.append(rd.SpectralBand(float(eps), upper))
    return tuple(rows)


def _blackbody(system: UnitSystem) -> None:
    name, idx = _example(
        rd.BLACKBODY_EXAMPLES,
        "rb_example",
        "Ejemplos de los caps. 12 de Cengel y Ghajar e Incropera. Podés cambiar cualquier dato: el "
        "resultado se actualiza solo.",
    )
    ex = rd.BLACKBODY_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, rd.BlackbodyInputs)
    key = f"rb_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    T = _abs_temperature("Temperatura de la superficie T", inp.T_K, f"{key}_T", _ABS_HELP)
    lam_point = None
    if st.checkbox(
        "E_bλ en una longitud de onda", value=inp.lambda_point_m is not None, key=f"{key}_pt"
    ):
        lam_point = number_input_si(
            label="Longitud de onda λ",
            kind=_LAM,
            default_si=inp.lambda_point_m or 3.0e-6,
            key=f"{key}_lam",
            format="%.4g",
            min_value_si=0.0,
        )
    band_lo = band_hi = None
    if st.checkbox(
        "La fracción que se emite en una banda (el visible: 0,40 a 0,76 μm)",
        value=inp.band_upper_m is not None or inp.band_lower_m is not None,
        key=f"{key}_band",
    ):
        left, right = st.columns(2)
        with left:
            band_lo = number_input_si(
                label="Desde λ₁",
                kind=_LAM,
                default_si=inp.band_lower_m or 0.40e-6,
                key=f"{key}_l1",
                format="%.4g",
                min_value_si=0.0,
            )
        with right:
            band_hi = number_input_si(
                label="Hasta λ₂",
                kind=_LAM,
                default_si=inp.band_upper_m or 0.76e-6,
                key=f"{key}_l2",
                format="%.4g",
                min_value_si=0.0,
            )
    fraction = None
    if st.checkbox(
        "¿Por debajo de qué λ se emite una fracción dada?",
        value=inp.fraction_target is not None,
        key=f"{key}_frac",
    ):
        fraction = (
            st.number_input(
                "Fracción [%]",
                min_value=0.1,
                max_value=99.9,
                value=100.0 * (inp.fraction_target or 0.5),
                step=5.0,
                format="%.4g",
                key=f"{key}_f",
            )
            / 100.0
        )
    bands: tuple[rd.SpectralBand, ...] = ()
    T_source = None
    if st.checkbox(
        "Una superficie real: ε(λ) escalonada en bandas", value=bool(inp.bands), key=f"{key}_real"
    ):
        bands = _band_inputs(inp, key)
        if st.checkbox(
            "La absortividad para la radiación de una fuente (el sol: 5800 K)",
            value=inp.T_source_K is not None,
            key=f"{key}_src",
        ):
            T_source = _abs_temperature(
                "Temperatura de la fuente", inp.T_source_K or 5800.0, f"{key}_Tsrc"
            )
    inputs = rd.BlackbodyInputs(T, band_lo, band_hi, fraction, bands, T_source, lam_point)
    try:
        result = _blackbody_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_blackbody(result, system)


def _band_table(rows: tuple[rd.BandRow, ...], symbol: str) -> None:
    _table(
        pd.DataFrame(
            [
                {
                    "Banda": k,
                    "λ [μm]": f"{_num(r.lower_m * 1e6, 4)} a "
                    + ("∞" if r.upper_m is None else _num(r.upper_m * 1e6, 4)),
                    symbol: _num(r.emissivity, 3),
                    "Δf": _num(r.fraction, 4),
                    f"{symbol}·Δf": _num(r.contribution, 4),
                }
                for k, r in enumerate(rows, 1)
            ]
        )
    )


def _render_blackbody(r: rd.BlackbodyResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items = [
        (f"E_b = σT⁴ [{_u(_E, system)}]", _val(r.E_b_W_per_m2, _E, system)),
        (f"λ_máx [{_u(_LAM, system)}]", _val(r.lambda_max_m, _LAM, system, 4)),
        (f"E_bλ,máx [{_u(_EL, system)}]", _val(r.E_blambda_max_W_per_m3, _EL, system)),
    ]
    if r.E_blambda_point_W_per_m3 is not None:
        items.append(
            (f"E_bλ en λ [{_u(_EL, system)}]", _val(r.E_blambda_point_W_per_m3, _EL, system))
        )
    if r.band_fraction is not None:
        items.append(("Fracción en la banda", _pct(r.band_fraction, 2)))
        items.append(
            (f"Emisión en la banda [{_u(_E, system)}]", _val(r.band_power_W_per_m2, _E, system))
        )
    if r.lambda_target_m is not None:
        items.append(
            (f"λ de esa fracción [{_u(_LAM, system)}]", _val(r.lambda_target_m, _LAM, system, 4))
        )
    if r.emissivity is not None:
        items.append(("Emisividad total ε", _num(r.emissivity, 4)))
        items.append((f"E = ε·E_b [{_u(_E, system)}]", _val(r.E_W_per_m2, _E, system)))
    if r.absorptivity is not None:
        items.append(("Absortividad α (de la fuente)", _num(r.absorptivity, 4)))
    _metric_rows(items)
    _notes(rd.blackbody_notes(r))
    st.markdown("#### La curva de Planck")
    try:
        e_scale = convert_from_si(1e6, _EL, system)  # 1 W/(m²·μm) en la unidad del sistema
        st.plotly_chart(
            planck_figure(r, e_scale=e_scale, e_unit=_u(_EL, system)),
            width="stretch",
            key="rb_planck",
        )
        caption = (
            "El área bajo la curva es E_b = σT⁴; el máximo cae en λ_máx = C₃/T (Wien) y se corre "
            "a λ más cortas al calentar."
        )
        if r.rows:
            caption += " En naranja la emisión real ε(λ)·E_bλ: su área es E = ε·E_b."
        st.caption(caption)
    except Exception as exc:  # noqa: BLE001 — un gráfico que falla no tumba la página
        st.warning(f"No se pudo dibujar la curva: {exc}")
    if r.rows:
        st.markdown(f"#### ε(λ) a {_num(i.T_K, 5)} K: el peso de cada banda")
        _band_table(r.rows, "ε")
    if r.source_rows:
        st.markdown(f"#### α para la radiación de la fuente a {_num(i.T_source_K, 5)} K")
        _band_table(r.source_rows, "α")
        try:
            st.plotly_chart(spectral_match_figure(r), width="stretch", key="rb_match")
            st.caption(
                "Cada espectro dividido por su máximo, con ε(λ) en el mismo eje (todo entre 0 y "
                "1). La superficie absorbe la radiación de la fuente con su ε(λ) donde la fuente "
                "emite, y emite con su ε(λ) donde emite ella: por eso α ≠ ε."
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"No se pudo dibujar la comparación: {exc}")
    _render_steps(blackbody_steps(r, system))
    _render_export(rd.blackbody_to_dict(r, system), "cuerpo_negro", "rb_dl")


# ---------------------------------------------------------------------
# Factores de forma
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _view_factor_cached(inputs: rd.ViewFactorInputs) -> rd.ViewFactorResult:
    return rd.solve_view_factor(inputs)


@st.cache_data(show_spinner=False)
def _view_curve_cached(inputs: rd.ViewFactorInputs) -> tuple[str, list[float], list[float]]:
    return rd.view_factor_curve(inputs)


def _view_factor(system: UnitSystem) -> None:
    name, idx = _example(
        rd.VIEW_FACTOR_EXAMPLES,
        "rv_example",
        "Geometrías de las tablas 13.1 y 13.2 de Incropera (13-1 y 13-2 de Cengel y Ghajar). "
        "Podés cambiar cualquier medida.",
    )
    ex = rd.VIEW_FACTOR_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, rd.ViewFactorInputs)
    key = f"rv_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    geos = list(rd.VIEW_FACTORS)
    geometry: rd.ViewGeometry = st.selectbox(
        "Geometría",
        geos,
        index=geos.index(inp.geometry),
        format_func=lambda g: rd.VIEW_FACTORS[g].name,
        key=f"{key}_geo",
    )
    vf = rd.VIEW_FACTORS[geometry]
    st.caption(
        f"De la superficie i a la j. Fuente: {vf.source}."
        + (" Superficies muy largas: se calcula por metro de largo." if vf.two_d else "")
    )
    defaults = (inp.a, inp.b, inp.c) if geometry == inp.geometry else _VF_DEFAULTS[geometry]
    dims: list[float] = []
    for k in range(0, len(vf.dims), 2):
        cols = st.columns(2)
        for col, j in zip(cols, range(k, min(k + 2, len(vf.dims))), strict=False):
            with col:
                dims.append(
                    number_input_si(
                        label=vf.labels[j],
                        kind="length",
                        default_si=defaults[j],
                        key=f"{key}_{geometry}_{j}",
                        format="%.4g",
                        min_value_si=0.0,
                    )
                )
    inputs = rd.ViewFactorInputs(geometry, *dims)
    try:
        result = _view_factor_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_view_factor(result, system)


def _render_view_factor(r: rd.ViewFactorResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    vf = rd.VIEW_FACTORS[r.inputs.geometry]
    a_kind: QuantityKind = "length" if vf.two_d else "area"
    per = " (por metro de largo)" if vf.two_d else ""
    _metric_rows(
        [
            ("F_ij (de i a j)", _num(r.F_ij, 4)),
            ("F_ji (reciprocidad)", _num(r.F_ji, 4)),
            ("De i al resto: 1 − F_ij", _num(r.F_i_rest, 4)),
            (f"A_i{per} [{_u(a_kind, system)}]", _val(r.A_i, a_kind, system)),
            (f"A_j{per} [{_u(a_kind, system)}]", _val(r.A_j, a_kind, system)),
        ]
    )
    _notes(rd.view_factor_notes(r))
    st.latex(vf.latex)
    if r.ratios:
        _table(pd.DataFrame([{"Parámetro": k, "Valor": _num(v, 4)} for k, v in r.ratios.items()]))
    st.markdown("#### F_ij al cambiar la separación")
    try:
        label, xs, Fs = _view_curve_cached(r.inputs)
        values = (r.inputs.a, r.inputs.b, r.inputs.c)
        x0 = values[vf.dims.index(label)]
        log_x = r.inputs.geometry not in ("three_sided_2d", "parallel_cylinders_2d", "tube_row_2d")
        st.plotly_chart(
            view_factor_figure(
                label,
                xs,
                Fs,
                x0,
                r.F_ij,
                x_scale=convert_from_si(1.0, "length", system),
                x_unit=_u("length", system),
                log_x=log_x,
            ),
            width="stretch",
            key="rv_curve",
        )
        st.caption(
            f"Con las demás medidas fijas: al alejarlas (más {label}) cada superficie ve menos "
            "a la otra. El punto naranja son los datos."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar la curva: {exc}")
    _render_steps(view_factor_steps(r, system))
    _render_export(rd.view_factor_to_dict(r, system), "factor_de_forma", "rv_dl")


# ---------------------------------------------------------------------
# Dos superficies y pantallas
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _two_surface_cached(inputs: rd.TwoSurfaceInputs) -> rd.TwoSurfaceResult:
    return rd.solve_two_surface(inputs)


def _two_surface(system: UnitSystem) -> None:
    name, idx = _example(
        rd.TWO_SURFACE_EXAMPLES,
        "rt_example",
        "Ejemplos del cap. 13 de Cengel y Ghajar e Incropera. Podés cambiar cualquier dato.",
    )
    ex = rd.TWO_SURFACE_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, rd.TwoSurfaceInputs)
    key = f"rt_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    geos = list(rd.TWO_SURFACE_GEOMETRIES)
    geometry: rd.TwoSurfaceGeometry = st.radio(
        "Geometría",
        geos,
        index=geos.index(inp.geometry),
        format_func=lambda g: rd.TWO_SURFACE_GEOMETRIES[g],
        key=f"{key}_geo",
    )
    same = geometry == inp.geometry
    inner = geometry in ("concentric_cylinders", "concentric_spheres")
    left, right = st.columns(2)
    with left:
        T1 = _abs_temperature(
            "T₁ (la de adentro)" if inner else "T₁", inp.T1_K, f"{key}_T1", _ABS_HELP
        )
        eps1 = _emissivity("ε₁", inp.eps1, f"{key}_e1")
    with right:
        T2 = _abs_temperature("T₂ (la de afuera)" if inner else "T₂", inp.T2_K, f"{key}_T2")
        eps2 = _emissivity("ε₂", inp.eps2, f"{key}_e2")
    A1, A2, F12 = inp.A1_m2, inp.A2_m2, inp.F12
    r1, r2, L = inp.r1_m, inp.r2_m, inp.L_m
    g_key = f"{key}_{geometry}"
    if geometry == "parallel_plates":
        A1 = number_input_si(
            label="Área A",
            kind="area",
            default_si=inp.A1_m2 if same else 1.0,
            key=f"{g_key}_A",
            format="%.5g",
            min_value_si=0.0,
            help="Con 1 m² el calor es el flujo q″ (lo que dan los libros).",
        )
    elif inner:
        left, right = st.columns(2)
        with left:
            r1 = number_input_si(
                label="Radio r₁",
                kind="small_length",
                default_si=inp.r1_m if same else 0.05,
                key=f"{g_key}_r1",
                format="%.5g",
                min_value_si=0.0,
            )
        with right:
            r2 = number_input_si(
                label="Radio r₂",
                kind="small_length",
                default_si=inp.r2_m if same else 0.1,
                key=f"{g_key}_r2",
                format="%.5g",
                min_value_si=0.0,
            )
        if geometry == "concentric_cylinders":
            L = number_input_si(
                label="Largo L",
                kind="length",
                default_si=inp.L_m if same else 1.0,
                key=f"{g_key}_L",
                format="%.5g",
                min_value_si=0.0,
            )
    else:
        A1 = number_input_si(
            label="Área A₁" + (" del objeto" if geometry == "small_object" else ""),
            kind="area",
            default_si=inp.A1_m2 if same else 1.0,
            key=f"{g_key}_A1",
            format="%.5g",
            min_value_si=0.0,
        )
        if geometry == "general":
            left, right = st.columns(2)
            with left:
                A2 = number_input_si(
                    label="Área A₂",
                    kind="area",
                    default_si=inp.A2_m2 if same else 2.0,
                    key=f"{g_key}_A2",
                    format="%.5g",
                    min_value_si=0.0,
                )
            with right:
                F12 = float(
                    st.number_input(
                        "F₁₂",
                        min_value=0.001,
                        max_value=1.0,
                        value=float(inp.F12) if same else 1.0,
                        step=0.05,
                        format="%.4f",
                        key=f"{g_key}_F12",
                    )
                )
    shields: tuple[rd.Shield, ...] = ()
    if geometry in ("parallel_plates", "concentric_cylinders", "concentric_spheres"):
        n = int(
            st.number_input(
                "Pantallas de radiación",
                min_value=0,
                max_value=3,
                value=len(inp.shields) if same else 0,
                key=f"{g_key}_ns",
                help="Láminas delgadas y brillantes entre las dos superficies: cada una suma "
                "resistencias a la red.",
            )
        )
        rows = []
        for j in range(n):
            default = inp.shields[j] if same and j < len(inp.shields) else rd.Shield(0.1, 0.1, 0.0)
            st.markdown(f"**Pantalla {j + 1}**")
            cols = st.columns(3 if inner else 2)
            with cols[0]:
                e_a = _emissivity("ε de la cara 1", default.eps_1, f"{g_key}_s{j}_e1")
            with cols[1]:
                e_b = _emissivity("ε de la cara 2", default.eps_2, f"{g_key}_s{j}_e2")
            r_s = 0.0
            if inner:
                with cols[2]:
                    r_s = number_input_si(
                        label="Radio",
                        kind="small_length",
                        default_si=default.r_m or r1 + (r2 - r1) * (j + 1) / (n + 1),
                        key=f"{g_key}_s{j}_r",
                        format="%.5g",
                        min_value_si=0.0,
                    )
            rows.append(rd.Shield(e_a, e_b, r_s))
        shields = tuple(rows)
    inputs = rd.TwoSurfaceInputs(geometry, T1, T2, eps1, eps2, A1, A2, F12, r1, r2, L, shields)
    try:
        result = _two_surface_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_two_surface(result, system)


def _inv_area_label(system: UnitSystem) -> str:
    return "1/ft²" if system == "Inglés" else "1/m²"


def _render_two_surface(r: rd.TwoSurfaceResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items = [(f"Q̇ de 1 a 2 [{_u(_W, system)}]", _val(r.Q_W, _W, system))]
    if i.geometry == "parallel_plates":
        items.append((f"q″ = Q̇/A [{_u(_E, system)}]", _val(r.Q_W / i.A1_m2, _E, system)))
    if i.shields:
        items.append((f"Q̇ sin pantallas [{_u(_W, system)}]", _val(r.Q_no_shields_W, _W, system)))
        items.append(("Con pantallas pasa", _pct(r.reduction)))
        for k, T in enumerate(r.shield_T_K, 1):
            items.append((f"T de la pantalla {k} [{_u(_TA, system)}]", _val(T, _TA, system)))
    items.append((f"J₁ [{_u(_E, system)}]", _val(r.radiosities[0], _E, system)))
    items.append((f"J₂ [{_u(_E, system)}]", _val(r.radiosities[1], _E, system)))
    _metric_rows(items)
    _notes(rd.two_surface_notes(r))
    st.markdown("#### La red de resistencias")
    inv = 1.0 / convert_from_si(1.0, "area", system)  # 1/m² → 1/ft²
    _table(
        pd.DataFrame(
            [
                {
                    "Resistencia": res.label,
                    "Tipo": res.kind,
                    f"R [{_inv_area_label(system)}]": _num(res.R_per_m2 * inv, 4),
                    "Del total": _pct(r.share(res)),
                }
                for res in r.resistances
            ]
        )
    )
    try:
        st.plotly_chart(network_figure(r), width="stretch", key="rt_network")
        st.caption(
            "Q̇ = (E_b1 − E_b2)/ΣR, como una corriente en una red en serie: manda la resistencia "
            "más grande. Con superficies brillantes (ε chica) pesan las de superficie."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar la red: {exc}")
    _render_steps(two_surface_steps(r, system))
    _render_export(rd.two_surface_to_dict(r, system), "dos_superficies", "rt_dl")


# ---------------------------------------------------------------------
# Recinto
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _enclosure_cached(inputs: rd.EnclosureInputs) -> rd.EnclosureResult:
    return rd.solve_enclosure(inputs)


def _surface_inputs(
    k: int, default: rd.EnclosureSurface, opening: bool, key: str
) -> rd.EnclosureSurface:
    st.markdown(f"**{k + 1}. {default.name}**")
    if opening:
        T = _abs_temperature(
            "T de los alrededores",
            default.T_K or 300.0,
            f"{key}_T",
            "Una abertura se trata "
            "como una superficie negra a la temperatura de lo que se ve a través de ella.",
        )
        return rd.EnclosureSurface(default.name, 0.0, 1.0, T)
    current = _C_R if default.reradiating else (_C_Q if default.T_K is None else _C_T)
    condition = st.radio(
        "Condición",
        _CONDITIONS,
        index=_CONDITIONS.index(current),
        key=f"{key}_cond",
        horizontal=True,
        label_visibility="collapsed",
    )
    if condition == _C_R:
        return rd.EnclosureSurface(default.name, 0.0, default.emissivity, None, 0.0)
    left, right = st.columns(2)
    with left:
        eps = _emissivity("ε", default.emissivity, f"{key}_e")
    with right:
        if condition == _C_T:
            T = _abs_temperature("T", default.T_K or 500.0, f"{key}_T")
            return rd.EnclosureSurface(default.name, 0.0, eps, T)
        Q = number_input_si(
            label="Q̇ que se le entrega",
            kind=_W,
            default_si=default.Q_W if default.Q_W else 1000.0,
            key=f"{key}_Q",
            format="%.5g",
            help="Positivo si la superficie entrega calor al recinto (un calefactor); negativo "
            "si lo recibe.",
        )
    return rd.EnclosureSurface(default.name, 0.0, eps, None, Q)


def _enclosure(system: UnitSystem) -> None:
    name, idx = _example(
        rd.ENCLOSURE_EXAMPLES,
        "re_example",
        "Recintos de tres superficies grises y difusas (Cengel y Ghajar, §13-4; Incropera, "
        "§13.3). Podés cambiar la geometría y la condición de cada superficie.",
    )
    ex = rd.ENCLOSURE_EXAMPLES[name]
    case = ex.case
    assert case is not None
    key = f"re_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    layouts = list(rd.ENCLOSURE_LAYOUTS)
    layout: rd.EnclosureLayout = st.selectbox(
        "Configuración",
        layouts,
        index=layouts.index(case.layout),
        format_func=lambda g: rd.ENCLOSURE_LAYOUTS[g].name,
        key=f"{key}_layout",
    )
    info = rd.ENCLOSURE_LAYOUTS[layout]
    same = layout == case.layout
    if info.per_length:
        st.caption(
            "Un ducto largo: se calcula un tramo de 1 m (las áreas son el ancho por 1 m y los "
            "calores van por metro de largo)."
        )
    l_key = f"{key}_{layout}"
    defaults = case.dims if same else _LAYOUT_DEFAULTS[layout]
    dims: list[float] = []
    for k in range(0, len(info.dims), 2):
        cols = st.columns(2)
        for col, j in zip(cols, range(k, min(k + 2, len(info.dims))), strict=False):
            with col:
                dims.append(
                    number_input_si(
                        label=info.dims[j],
                        kind="length",
                        default_si=defaults[j],
                        key=f"{l_key}_d{j}",
                        format="%.4g",
                        min_value_si=0.0,
                    )
                )
    surfaces = []
    for k in range(3):
        if same:
            default = case.surfaces[k]
        else:
            default = rd.EnclosureSurface(info.surfaces[k], 0.0, 0.8, 500.0 - 100.0 * k)
        opening = info.surroundings and k == 2
        surfaces.append(_surface_inputs(k, default, opening, f"{l_key}_s{k}"))
    try:
        inputs = rd.enclosure_from_layout(layout, tuple(dims), tuple(surfaces))
        result = _enclosure_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_enclosure(result, system)


def _render_enclosure(r: rd.EnclosureResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    S = r.inputs.surfaces
    _metric_rows(
        [
            (f"Q̇ de «{s.name}» [{_u(_W, system)}]", _val(q, _W, system))
            for s, q in zip(S, r.Q_W, strict=True)
        ]
    )
    _notes(rd.enclosure_notes(r))
    _table(
        pd.DataFrame(
            [
                {
                    "Superficie": f"{k}. {s.name}",
                    f"A [{_u('area', system)}]": _val(s.area_m2, "area", system),
                    "ε": "—" if s.reradiating else _num(s.emissivity, 3),
                    f"T [{_u(_TA, system)}]": _val(T, _TA, system),
                    f"J [{_u(_E, system)}]": _val(J, _E, system),
                    f"Q̇ [{_u(_W, system)}]": _val(q, _W, system),
                }
                for k, (s, T, J, q) in enumerate(zip(S, r.T_K, r.J_W_per_m2, r.Q_W, strict=True), 1)
            ]
        )
    )
    try:
        st.plotly_chart(
            enclosure_figure(r, q_scale=convert_from_si(1.0, _W, system), q_unit=_u(_W, system)),
            width="stretch",
            key="re_bars",
        )
        st.caption(
            "Lo que entregan unas superficies (naranja) lo reciben las otras (azul): el recinto "
            "está cerrado y ΣQ̇ = 0."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar el balance: {exc}")
    st.markdown("#### Los factores de forma y el intercambio neto entre pares")
    n = len(S)
    _table(
        pd.DataFrame(
            [
                {
                    "de \\ a": f"{k + 1}. {S[k].name}",
                    **{str(j + 1): _num(r.inputs.F[k][j], 4) for j in range(n)},
                }
                for k in range(n)
            ]
        )
    )
    _table(
        pd.DataFrame(
            [
                {
                    "De": f"{a + 1}. {S[a].name}",
                    "A": f"{b + 1}. {S[b].name}",
                    f"Q̇ neto [{_u(_W, system)}]": _val(r.Q_between(a, b), _W, system),
                }
                for a in range(n)
                for b in range(a + 1, n)
            ]
        )
    )
    st.caption("Q̇_ij = A_i·F_ij·(J_i − J_j): positivo si va de la primera a la segunda.")
    _render_steps(enclosure_steps(r, system))
    _render_export(rd.enclosure_to_dict(r, system), "recinto", "re_dl")


# ---------------------------------------------------------------------
# Radiación y convección juntas
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _surface_cached(inputs: rd.SurfaceBalanceInputs) -> rd.SurfaceBalanceResult:
    return rd.solve_surface_balance(inputs)


@st.cache_data(show_spinner=False)
def _thermocouple_cached(inputs: rd.ThermocoupleInputs) -> rd.ThermocoupleResult:
    return rd.solve_thermocouple(inputs)


def _surface(system: UnitSystem) -> None:
    name, idx = _example(
        rd.SURFACE_EXAMPLES,
        "rc_example",
        "Ejemplos de Incropera (cap. 1) y Cengel y Ghajar (radiación solar). Podés cambiar "
        "cualquier dato.",
    )
    ex = rd.SURFACE_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, rd.SurfaceBalanceInputs)
    key = f"rc_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    left, right = st.columns(2)
    with left:
        eps = _emissivity("Emisividad ε", inp.emissivity, f"{key}_e")
        T_inf = number_input_si(
            label="T∞ del aire", kind=_T, default_si=inp.T_inf_K, key=f"{key}_Tinf", format="%.5g"
        )
    with right:
        h = number_input_si(
            label="h de convección",
            kind=_H,
            default_si=inp.h_W_per_m2K,
            key=f"{key}_h",
            format="%.5g",
            min_value_si=0.0,
        )
        T_surr = number_input_si(
            label="T de los alrededores T_alr",
            kind=_T,
            default_si=inp.T_surr_K,
            key=f"{key}_Tsur",
            format="%.5g",
            help="La de las paredes que ve la superficie (o la del cielo, afuera).",
        )
    area = number_input_si(
        label="Área A",
        kind="area",
        default_si=inp.area_m2,
        key=f"{key}_A",
        format="%.5g",
        min_value_si=0.0,
    )
    given = st.radio(
        "¿Qué es dato?",
        _SURFACE_GIVEN,
        index=0 if inp.T_s_K is not None else 1,
        key=f"{key}_given",
        horizontal=True,
    )
    T_s = Q_in = None
    if given == _GIVEN_T:
        T_s = number_input_si(
            label="T de la superficie T_s",
            kind=_T,
            default_si=inp.T_s_K or inp.T_inf_K + 50.0,
            key=f"{key}_Ts",
            format="%.5g",
        )
    else:
        Q_in = number_input_si(
            label="Calor que recibe la superficie",
            kind=_W,
            default_si=inp.Q_in_W if inp.Q_in_W is not None else 100.0,
            key=f"{key}_Q",
            format="%.5g",
            help="Una resistencia eléctrica, o lo que llega por conducción desde adentro: sale "
            "por convección y radiación.",
        )
    alpha, G = 0.0, 0.0
    if st.checkbox("Al sol", value=inp.alpha_solar > 0.0, key=f"{key}_sun"):
        left, right = st.columns(2)
        with left:
            alpha = float(
                st.number_input(
                    "Absortividad solar α_s",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(inp.alpha_solar) if inp.alpha_solar > 0.0 else 0.9,
                    step=0.05,
                    format="%.3f",
                    key=f"{key}_as",
                )
            )
        with right:
            G = number_input_si(
                label="Radiación solar G",
                kind=_E,
                default_si=inp.G_solar_W_per_m2 or 800.0,
                key=f"{key}_G",
                format="%.5g",
                min_value_si=0.0,
            )
    inputs = rd.SurfaceBalanceInputs(eps, h, T_inf, T_surr, area, T_s, Q_in, alpha, G)
    try:
        result = _surface_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_surface(result, system)


def _render_surface(r: rd.SurfaceBalanceResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items = []
    if i.Q_in_W is not None:
        items.append((f"T_s de equilibrio [{_u(_T, system)}]", _val(r.T_s_K, _T, system)))
    items += [
        (f"Q̇ por convección [{_u(_W, system)}]", _val(r.Q_conv_W, _W, system)),
        (f"Q̇ por radiación [{_u(_W, system)}]", _val(r.Q_rad_W, _W, system)),
    ]
    if r.Q_solar_W > 0.0:
        items.append((f"Q̇ que absorbe del sol [{_u(_W, system)}]", _val(r.Q_solar_W, _W, system)))
    items += [
        (f"Q̇ neto que pierde [{_u(_W, system)}]", _val(r.Q_lost_W, _W, system)),
        (f"h_rad [{_u(_H, system)}]", _val(r.h_rad_W_per_m2K, _H, system)),
        ("La radiación es", _pct(r.radiation_share) + " del calor"),
    ]
    _metric_rows(items)
    _notes(rd.surface_balance_notes(r))
    t_scale, t_offset = _t_scale(system)
    try:
        st.plotly_chart(
            surface_curve_figure(
                rd.surface_balance_curve(r),
                r,
                t_scale=t_scale,
                t_offset=t_offset,
                t_unit=_u(_T, system),
                q_scale=convert_from_si(1.0, _E, system),
                q_unit=_u(_E, system),
            ),
            width="stretch",
            key="rc_curve",
        )
        st.caption(
            "Por unidad de área y sin el sol: la convección crece lineal con T_s y la radiación "
            "con T⁴, así que a temperaturas altas manda la radiación."
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar la curva: {exc}")
    _render_steps(surface_balance_steps(r, system))
    _render_export(rd.surface_to_dict(r, system), "conveccion_y_radiacion", "rc_dl")


def _thermocouple(system: UnitSystem) -> None:
    name, idx = _example(
        rd.THERMOCOUPLE_EXAMPLES,
        "rk_example",
        "El error de radiación de una termocupla (Cengel y Ghajar, §13-5).",
    )
    ex = rd.THERMOCOUPLE_EXAMPLES[name]
    inp = ex.inputs
    assert isinstance(inp, rd.ThermocoupleInputs)
    key = f"rk_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    left, right = st.columns(2)
    with left:
        T_th = number_input_si(
            label="Lo que marca la termocupla",
            kind=_T,
            default_si=inp.T_reading_K,
            key=f"{key}_Tth",
            format="%.5g",
        )
        eps = _emissivity("ε de la junta", inp.emissivity, f"{key}_e")
    with right:
        T_w = number_input_si(
            label="T de las paredes del ducto",
            kind=_T,
            default_si=inp.T_wall_K,
            key=f"{key}_Tw",
            format="%.5g",
        )
        h = number_input_si(
            label="h sobre la junta",
            kind=_H,
            default_si=inp.h_W_per_m2K,
            key=f"{key}_h",
            format="%.5g",
            min_value_si=0.0,
        )
    try:
        result = _thermocouple_cached(rd.ThermocoupleInputs(T_th, T_w, eps, h))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"T real del gas [{_u(_T, system)}]", _val(result.T_gas_K, _T, system)),
            (
                f"Error de la lectura [{_u('temperature_difference', system)}]",
                _val(result.error_K, "temperature_difference", system),
            ),
            (f"q_rad de la junta [{_u(_E, system)}]", _val(result.q_rad_W_per_m2, _E, system)),
        ]
    )
    _notes(rd.thermocouple_notes(result))
    _render_steps(thermocouple_steps(result, system))
    _render_export(rd.thermocouple_to_dict(result, system), "termocupla", "rk_dl")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) "
            "todavía no tiene un capítulo de radiación: las fórmulas siguen a **Çengel y Ghajar "
            "(2015)**, caps. 12 y 13, y a **Incropera et al. (2007)**, caps. 12 y 13. Las "
            "constantes son las de CODATA 2018."
        )
        st.markdown("**Cuerpo negro**: Planck, Wien y Stefan–Boltzmann (§12-3):")
        st.latex(
            r"E_{b\lambda} = \frac{C_1}{\lambda^5\left[\exp\left(\frac{C_2}{\lambda T}\right) - "
            r"1\right]}"
        )
        st.latex(r"\lambda_{\text{máx}}\,T = C_3 = 2897.8\ \mu\mathrm{m\cdot K}")
        st.latex(r"E_b = \sigma\,T^4")
        st.markdown(
            "La fracción emitida entre 0 y λ depende solo de λT (la tabla 12-2): en una banda, "
            "Δf = f(λ₂T) − f(λ₁T)."
        )
        st.latex(r"f_{0-\lambda} = \frac{1}{\sigma T^4}\int_0^{\lambda} E_{b\lambda}\,d\lambda")
        st.markdown(
            "**Superficies reales** (§12-5): con ε(λ) escalonada, la emisividad total pesa cada "
            "banda con la emisión del cuerpo negro a la temperatura de la superficie, y la "
            "absortividad, con la de la fuente (Kirchhoff espectral: α_λ = ε_λ):"
        )
        st.latex(r"\varepsilon(T) = \sum_i \varepsilon_i\,\Delta f_i(T)")
        st.latex(r"\alpha = \sum_i \varepsilon_i\,\Delta f_i(T_{\text{fuente}})")
        st.markdown(
            "**Factor de forma** (§13-1 y §13-2): la fracción de lo que deja i que llega a j, "
            "con reciprocidad y la regla de la suma; en superficies largas, las cuerdas cruzadas "
            "de Hottel:"
        )
        st.latex(r"A_i\,F_{ij} = A_j\,F_{ji} \qquad \sum_j F_{ij} = 1")
        st.latex(r"F_{ij} = \frac{(\text{cruzadas}) - (\text{no cruzadas})}{2\,(\text{la de } i)}")
        st.markdown(
            "**Superficies grises y difusas** (§13-4): la radiosidad J es lo que deja la "
            "superficie (lo que emite más lo que refleja). Cada superficie es una resistencia y "
            "el espacio entre dos, otra:"
        )
        st.latex(r"R_i = \frac{1 - \varepsilon_i}{A_i\,\varepsilon_i}")
        st.latex(r"R_{ij} = \frac{1}{A_i\,F_{ij}}")
        st.latex(
            r"\dot{Q}_{12} = \frac{E_{b1} - E_{b2}}{\frac{1 - \varepsilon_1}{A_1 \varepsilon_1} + "
            r"\frac{1}{A_1 F_{12}} + \frac{1 - \varepsilon_2}{A_2 \varepsilon_2}}"
        )
        st.markdown(
            "Placas paralelas, y cada pantalla suma (1/ε de cada cara − 1) al denominador (§13-5):"
        )
        st.latex(
            r"q = \frac{\sigma\,(T_1^4 - T_2^4)}{\frac{1}{\varepsilon_1} + \frac{1}{\varepsilon_2} "
            r"- 1}"
        )
        st.markdown(
            "**Recinto de N superficies** (método de las radiosidades): una ecuación por "
            "superficie, con su temperatura o con su calor dado (rerradiante: Q̇ = 0):"
        )
        st.latex(
            r"\frac{E_{bi} - J_i}{(1 - \varepsilon_i)/\varepsilon_i} = "
            r"\sum_j F_{ij}\,(J_i - J_j)"
        )
        st.latex(r"\dot{Q}_i = A_i \sum_j F_{ij}\,(J_i - J_j)")
        st.markdown(
            "**Radiación y convección juntas** (Incropera, §1.2.3): con un recinto grande "
            "alrededor, la radiación se escribe como una convección con h_rad:"
        )
        st.latex(
            r"\dot{Q} = h\,A\,(T_s - T_\infty) + \varepsilon\,\sigma\,A\,(T_s^4 - T_{\text{alr}}^4)"
        )
        st.latex(
            r"h_{\text{rad}} = \varepsilon\,\sigma\,(T_s + T_{\text{alr}})"
            r"(T_s^2 + T_{\text{alr}}^2)"
        )
        st.markdown(
            "**Termocupla** (§13-5): la junta gana por convección lo que pierde por radiación con "
            "las paredes, así que marca menos que el gas:"
        )
        st.latex(r"T_f = T_{\text{th}} + \frac{\varepsilon\,\sigma\,(T_{\text{th}}^4 - T_w^4)}{h}")


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Radiación", page_icon="☀️", layout="centered")

st.subheader(SUBJECT)
st.title("☀️ Radiación")
st.markdown(
    "El **cuerpo negro** (Planck, Wien y Stefan–Boltzmann) y las **superficies reales** (ε por "
    "bandas, α para el sol), los **factores de forma**, el intercambio entre **dos superficies** "
    "con pantallas, los **recintos** de varias superficies grises (radiosidades) y la radiación "
    "**junto con la convección** (y el error de una termocupla)."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Radiación")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="rd_mode")
if mode == _M_BB:
    _blackbody(system)
elif mode == _M_VF:
    _view_factor(system)
elif mode == _M_TS:
    _two_surface(system)
elif mode == _M_EN:
    _enclosure(system)
else:
    what = st.radio("¿Qué caso?", _COMBINED_WHAT, key="rc_what", horizontal=True)
    if what == _W_SURF:
        _surface(system)
    else:
        _thermocouple(system)
