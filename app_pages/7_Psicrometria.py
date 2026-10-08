"""Página 7 — Psicrometría: aire húmedo, procesos de acondicionamiento y torre de enfriamiento.

Fase 4. El modelo del vademecum §14 (aire seco y vapor como gases ideales; Çengel
& Boles cap. 14) en tres modos:

- **Estado del aire húmedo** con trece pares de datos (T, φ, T_bh, T_pr, ω, h),
  la presión dada o por altura, todas las propiedades, la exergía, la
  comparación con el modelo de gas real de CoolProp y una carta psicrométrica
  que se puede tocar para cargar un estado.
- **Procesos de acondicionamiento** (§14.12): un tren de hasta cuatro procesos
  (calentamiento o enfriamiento sensible, calentamiento con humidificación,
  serpentín, humidificación adiabática y mezcla), con el recorrido en la carta,
  el calor, el agua y la exergía destruida en cada uno.
- **Torre de enfriamiento** (§14.12.7): caudal de aire, agua de reposición,
  rango, aproximación y exergía.

El cálculo vive en :mod:`core.psychrometrics` y :mod:`core.hvac`; la carta, en
:mod:`ui.psychro_chart`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.export import dict_to_csv
from core.hvac import (
    HVAC_EXAMPLE_NOTES,
    HVAC_EXAMPLES,
    MAX_PROCESSES,
    PROCESS_SECTIONS,
    TOWER_EXAMPLE_NOTES,
    TOWER_EXAMPLES,
    AdiabaticHumidification,
    AdiabaticMixing,
    AirFlow,
    AirInlet,
    CoolingDehumidification,
    CoolingTowerInputs,
    CoolingTowerResult,
    HeatingHumidification,
    HvacInputs,
    HvacResult,
    ProcessSpec,
    SensibleProcess,
    cooling_tower_notes,
    cooling_tower_sweep,
    cooling_tower_to_dict,
    default_hvac_sweep_values,
    default_tower_sweep_values,
    hvac_notes,
    hvac_sweep,
    hvac_to_dict,
    solve_cooling_tower,
    solve_hvac,
)
from core.psychrometrics import (
    P_MAX_PA,
    P_MIN_PA,
    P_SEA_LEVEL_PA,
    PAIR_LABELS,
    PAIRS,
    STATE_EXAMPLE_NOTES,
    STATE_EXAMPLES,
    T_MAX_K,
    T_MIN_K,
    Z_MAX_M,
    Z_MIN_M,
    CoolPropComparison,
    DeadState,
    MoistAirState,
    altitude_sweep,
    chart_window,
    coolprop_comparison,
    default_altitude_values,
    moist_air_notes,
    moist_air_state,
    moist_air_to_dict,
    pressure_from_altitude,
    room_masses,
    state_from_T_omega,
)
from core.psychrometrics_procedure import cooling_tower_steps, hvac_steps, moist_air_steps
from core.state_report import format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.psychro_chart import (
    ChartArrow,
    ChartPoint,
    grid_point_from_selection,
    hvac_chart_items,
    hvac_exergy_figure,
    psychrometric_chart_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.20.0"

# Opciones fijas de los radios y selectores: cambiarlas reiniciaría el widget.
_MODE_STATE = "Estado del aire húmedo"
_MODE_HVAC = "Procesos de acondicionamiento"
_MODE_TOWER = "Torre de enfriamiento"
_MODES = (_MODE_STATE, _MODE_HVAC, _MODE_TOWER)
_P_SEA = "Nivel del mar"
_P_ALT = "Por altura"
_P_GIVEN = "Presión dada"
_FLOW_VOLUME = "Caudal volumétrico"
_FLOW_DRY = "Caudal de aire seco"
_PROC_SENSIBLE = "Calentamiento o enfriamiento sensible"
_PROC_HEAT_HUM = "Calentamiento con humidificación"
_PROC_COIL = "Enfriamiento y deshumidificación"
_PROC_ADIABATIC = "Humidificación adiabática"
_PROC_MIX = "Mezcla con otra corriente"
_PROC_TYPES = (_PROC_SENSIBLE, _PROC_HEAT_HUM, _PROC_COIL, _PROC_ADIABATIC, _PROC_MIX)
_WATER_VAPOR = "Vapor saturado"
_WATER_LIQUID = "Agua líquida"
_N_PROCESSES = tuple(str(n) for n in range(1, MAX_PROCESSES + 1))

_STATE_NAMES = list(STATE_EXAMPLES)
_HVAC_NAMES = list(HVAC_EXAMPLES)
_TOWER_NAMES = list(TOWER_EXAMPLES)
_PAIR_OPTIONS = [PAIR_LABELS[pair] for pair in PAIRS]
_PAIR_BY_LABEL = {PAIR_LABELS[pair]: pair for pair in PAIRS}

_VAR_LABELS = {
    "T": "Temperatura de bulbo seco T",
    "T_wb": "Temperatura de bulbo húmedo T_bh",
    "T_dp": "Temperatura de punto de rocío T_pr",
}
_T_MOISTURE_MIN_K = 173.15


# ---------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------


def _value(value_si: float | None, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    if value_si is None:
        return "—"
    return format_value(convert_from_si(value_si, kind, system), sig)


def _fmt(value_si: float | None, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    if value_si is None:
        return "—"
    return f"{_value(value_si, kind, system, sig)} {unit_label(kind, system)}"


def _pct(fraction: float | None) -> str:
    return "—" if fraction is None else f"{format_value(100 * fraction, 4)} %"


def _T_format(system: UnitSystem) -> str:
    return "%.2f"


def _p_format(system: UnitSystem) -> str:
    return {"SI": "%.0f", "Técnico": "%.5f", "Inglés": "%.4f"}[system]


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _state_cached(p_Pa: float, pair: tuple[str, str], a: float, b: float) -> MoistAirState:
    return moist_air_state(p_Pa, pair, a, b)


@st.cache_data(show_spinner=False)
def _comparison_cached(
    state: MoistAirState, pair: tuple[str, str], a: float, b: float
) -> CoolPropComparison:
    return coolprop_comparison(state, pair, a, b)


@st.cache_data(show_spinner=False)
def _chart_cached(
    p_Pa: float,
    system: UnitSystem,
    points: tuple[ChartPoint, ...],
    arrows: tuple[ChartArrow, ...],
    extra_T: tuple[float, ...] = (),
) -> go.Figure:
    window = chart_window(p_Pa, tuple(pt.state for pt in points), extra_T_K=extra_T)
    return psychrometric_chart_figure(p_Pa, system, window, points=points, arrows=arrows)


@st.cache_data(show_spinner=False)
def _hvac_cached(inputs: HvacInputs) -> HvacResult:
    return solve_hvac(inputs)


@st.cache_data(show_spinner=False)
def _tower_cached(inputs: CoolingTowerInputs) -> CoolingTowerResult:
    return solve_cooling_tower(inputs)


@st.cache_data(show_spinner="Calculando el barrido…")
def _altitude_sweep_cached(
    T_K: float, phi: float
) -> list[tuple[float, float, float, float, float]]:
    return [
        (pt.altitude_m, pt.state.omega, pt.state.T_wb_K, pt.state.rho_kg_per_m3, pt.state.p_Pa)
        for pt in altitude_sweep(T_K, phi, default_altitude_values())
    ]


@st.cache_data(show_spinner="Calculando el barrido…")
def _hvac_sweep_cached(inputs: HvacInputs, parameter: str) -> list[tuple[float, ...]]:
    values = default_hvac_sweep_values(inputs, parameter)  # type: ignore[arg-type]
    return [
        (p.value, p.heating_W, p.cooling_W, p.water_added_kg_s, p.water_removed_kg_s)
        for p in hvac_sweep(inputs, parameter, values)  # type: ignore[arg-type]
    ]


@st.cache_data(show_spinner="Calculando el barrido…")
def _tower_sweep_cached(inputs: CoolingTowerInputs, parameter: str) -> list[tuple[float, ...]]:
    values = default_tower_sweep_values(inputs, parameter)  # type: ignore[arg-type]
    return [
        (p.value, p.m_dry_air_kg_s, p.V_air_in_m3_s, p.makeup_kg_s, p.approach_K)
        for p in cooling_tower_sweep(inputs, parameter, values)  # type: ignore[arg-type]
    ]


# ---------------------------------------------------------------------
# Entradas comunes
# ---------------------------------------------------------------------


def _example(label: str, names: list[str], key: str) -> int:
    choice = st.selectbox(
        label, names, key=key, help="Podés cambiar cualquier dato: el resultado se actualiza solo."
    )
    return names.index(choice)


def _read_pressure(
    key: str, p_default: float, z_default: float | None, system: UnitSystem
) -> tuple[float, float | None]:
    """Presión total: nivel del mar, por altura (atmósfera estándar) o dada."""
    if z_default is not None:
        index = 1
    elif math.isclose(p_default, P_SEA_LEVEL_PA, rel_tol=1e-9):
        index = 0
    else:
        index = 2
    mode = st.radio(
        "Presión total",
        (_P_SEA, _P_ALT, _P_GIVEN),
        index=index,
        horizontal=True,
        key=f"{key}_pmode",
        help=(
            "La presión atmosférica baja con la altura y cambia la humedad absoluta: con la "
            "misma T y φ, en la altura el aire lleva más vapor por kg de aire seco."
        ),
    )
    if mode == _P_SEA:
        return P_SEA_LEVEL_PA, None
    if mode == _P_ALT:
        z = st.number_input(
            "Altura sobre el nivel del mar [m]",
            min_value=Z_MIN_M,
            max_value=Z_MAX_M,
            value=float(z_default if z_default is not None else 1000.0),
            step=100.0,
            format="%.0f",
            key=f"{key}_z",
            help="Atmósfera estándar de ASHRAE (HoF 2017, cap. 1, ec. 3). Mendoza: 750 m; "
            "Bariloche: 900 m; La Quiaca: 3440 m.",
        )
        p = pressure_from_altitude(float(z))
        st.caption(f"Presión de la atmósfera estándar a esa altura: {_fmt(p, 'pressure', system)}")
        return p, float(z)
    p = number_input_si(
        label="Presión total",
        kind="pressure",
        default_si=p_default,
        key=f"{key}_p",
        format=_p_format(system),
        min_value_si=P_MIN_PA,
        max_value_si=P_MAX_PA,
        help="De 0,4 a 10 bar. A presiones altas el modelo ideal se aparta (mirá la comparación "
        "con CoolProp).",
    )
    return p, None


def _read_T(
    key: str, label: str, default_K: float, *, low: float = T_MIN_K, help: str | None = None
) -> float:  # noqa: A002
    return number_input_si(
        label=label,
        kind="temperature",
        default_si=default_K,
        key=key,
        format="%.2f",
        min_value_si=low,
        max_value_si=T_MAX_K,
        help=help,
    )


def _read_phi(
    key: str, label: str, default: float, *, low: float = 0.0, help: str | None = None
) -> float:  # noqa: A002
    value = st.number_input(
        f"{label} [%]",
        min_value=low,
        max_value=100.0,
        value=float(round(100.0 * default, 2)),
        step=5.0,
        format="%.2f",
        key=key,
        help=help,
    )
    return float(value) / 100.0


def _read_dead(
    key: str,
    default: DeadState | None,
    inlet: tuple[float, float],
    p: float,
    *,
    allow_inlet: bool = True,
) -> DeadState | None:
    """El ambiente para la exergía (``None``: el aire que entra)."""
    with st.expander("🌍 Ambiente para la exergía (estado muerto)", expanded=False):
        if allow_inlet:
            use_inlet = st.checkbox(
                "El ambiente es el aire que entra",
                value=default is None,
                key=f"{key}_dead_inlet",
                help="La exergía se mide contra el aire del ambiente: con ese aire, ψ = 0.",
            )
            if use_inlet:
                return None
        T0_default, phi0_default = (default.T0_K, default.phi0) if default else inlet
        left, right = st.columns(2)
        with left:
            T0 = _read_T(f"{key}_T0", "Temperatura del ambiente T₀", T0_default)
        with right:
            phi0 = _read_phi(
                f"{key}_phi0", "Humedad relativa del ambiente φ₀", phi0_default, low=1.0
            )
        return DeadState(T0, phi0, p)


# ---------------------------------------------------------------------
# Modo 1: estado del aire húmedo
# ---------------------------------------------------------------------


def _state_values(state: MoistAirState) -> dict[str, float | None]:
    return {
        "T": state.T_K,
        "phi": state.phi,
        "T_wb": state.T_wb_K,
        "T_dp": state.T_dp_K,
        "omega": state.omega,
        "h": state.h_J_per_kg,
    }


def _read_variable(key: str, name: str, default: float, system: UnitSystem) -> float:
    if name == "T":
        return _read_T(f"{key}_T", _VAR_LABELS["T"], default)
    if name in ("T_wb", "T_dp"):
        return _read_T(f"{key}_{name}", _VAR_LABELS[name], default, low=_T_MOISTURE_MIN_K)
    if name == "phi":
        return _read_phi(f"{key}_phi", "Humedad relativa φ", default)
    if name == "omega":
        return float(
            st.number_input(
                "Humedad absoluta ω [kg/kg a.s.]"
                if system != "Inglés"
                else "Humedad absoluta ω [lb/lb a.s.]",
                min_value=0.0,
                max_value=5.0,
                value=float(round(default, 6)),
                step=0.001,
                format="%.5f",
                key=f"{key}_omega",
                help="kg de vapor por kg de aire seco (el mismo número en lb/lb).",
            )
        )
    return number_input_si(
        label="Entalpía h (por kg de aire seco)",
        kind="specific_enthalpy",
        default_si=default,
        key=f"{key}_h",
        format="%.3f" if system != "SI" else "%.0f",
        help="Con la referencia de ASHRAE: h = 0 para el aire seco a 0 °C.",
    )


def _on_state_chart_select(key: str, chart_key: str, p_Pa: float) -> None:
    """Al tocar la carta: carga T y φ del punto en los datos (par T y φ)."""
    system = get_current_system()
    point = grid_point_from_selection(st.session_state.get(chart_key), system)
    if point is None:
        return
    T_K, omega = point
    T_user = round(convert_from_si(T_K, "temperature", system), 1)
    T_K = T_K + (T_user - convert_from_si(T_K, "temperature", system)) / (
        1.8 if system == "Inglés" else 1.0
    )
    try:
        state = state_from_T_omega(p_Pa, T_K, omega)
    except ValueError:
        return
    st.session_state[f"{key}_pair"] = PAIR_LABELS[("T", "phi")]
    st.session_state[f"{key}_T@{system}"] = T_user
    st.session_state[f"{key}_phi"] = round(100.0 * min(state.phi, 1.0), 1)
    st.session_state[f"{key}_chart_n"] = st.session_state.get(f"{key}_chart_n", 0) + 1


def _render_state_mode(system: UnitSystem) -> None:
    ex = _example("Ejemplo precargado", _STATE_NAMES, "ps_example")
    name = _STATE_NAMES[ex]
    example = STATE_EXAMPLES[name]
    key = f"ps_{ex}"
    p, z = _read_pressure(key, example.p_Pa, example.altitude_m, system)
    base = moist_air_state(example.pressure_Pa, example.pair, example.first, example.second)
    last = st.session_state.get(f"{key}_last", base)
    defaults = _state_values(last)
    pair_label = st.selectbox(
        "Datos",
        _PAIR_OPTIONS,
        index=_PAIR_OPTIONS.index(PAIR_LABELS[example.pair]),
        key=f"{key}_pair",
        help="Con la presión, dos propiedades independientes fijan el estado. ω y T_pr dicen lo "
        "mismo (la presión del vapor), así que no van juntas.",
    )
    pair = _PAIR_BY_LABEL[pair_label]
    left, right = st.columns(2)
    values = []
    for col, var in zip((left, right), pair, strict=True):
        with col:
            default = defaults.get(var)
            if default is None:
                default = _state_values(base)[var] or 0.0
            values.append(_read_variable(key, var, float(default), system))
    a, b = values
    room = st.checkbox(
        "Calcular las masas de aire y de vapor en un recinto",
        value=example.room_volume_m3 is not None,
        key=f"{key}_room",
    )
    volume = None
    if room:
        V_label = "Volumen del recinto [ft³]" if system == "Inglés" else "Volumen del recinto [m³]"
        V_factor = 0.3048**3 if system == "Inglés" else 1.0
        V_user = st.number_input(
            V_label,
            min_value=0.001,
            max_value=1.0e7,
            value=float((example.room_volume_m3 or 75.0) / V_factor),
            key=f"{key}_V@{'ft' if system == 'Inglés' else 'm'}",
        )
        volume = float(V_user) * V_factor
    dead = _read_dead(
        key,
        DeadState(example.T0_K, example.phi0, p),
        (example.T0_K, example.phi0),
        p,
        allow_inlet=False,
    )
    try:
        state = _state_cached(p, pair, a, b)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.session_state[f"{key}_last"] = state
    assert dead is not None
    st.markdown("### Resultado")
    note = STATE_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    _render_state_metrics(state, system)
    for item in moist_air_notes(state):
        st.info(item)
    _render_state_chart(state, dead, key, p, system)
    if volume is not None:
        m_a, m_v = room_masses(state, volume)
        row = st.columns(2)
        row[0].metric(
            f"Masa de aire seco [{unit_label('mass_flow', system).split('/')[0]}]",
            format_value(m_a if system != "Inglés" else m_a / 0.45359237, 4),
        )
        row[1].metric(
            f"Masa de vapor [{unit_label('mass_flow', system).split('/')[0]}]",
            format_value(m_v if system != "Inglés" else m_v / 0.45359237, 4),
        )
    _render_state_table(state, dead, system)
    _render_comparison(state, pair, a, b, system)
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Las fórmulas del vademecum §14 con los números reemplazados, en el sistema de "
            "unidades elegido. Las presiones de saturación salen de las tablas (A-4 sobre agua "
            "líquida, A-8 sobre hielo)."
        )
        steps = moist_air_steps(
            state, system, pair=pair, given=(a, b), dead=dead, altitude_m=z, room_volume_m3=volume
        )
        for i, step in enumerate(steps, start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)
    _render_export(
        moist_air_to_dict(
            state, system, pair=pair, given=(a, b), dead=dead, altitude_m=z, room_volume_m3=volume
        ),
        "aire_humedo",
        "ps",
    )
    _render_altitude_sweep(state, system)


def _render_state_metrics(state: MoistAirState, system: UnitSystem) -> None:
    w_unit = "lb/lb" if system == "Inglés" else "kg/kg"
    row1 = st.columns(3)
    row1[0].metric(f"ω [{w_unit} a.s.]", format_value(state.omega, 4))
    row1[1].metric("φ [%]", format_value(100 * state.phi, 4))
    row1[2].metric(
        f"h [{unit_label('specific_enthalpy', system)}]",
        _value(state.h_J_per_kg, "specific_enthalpy", system, 4),
    )
    row2 = st.columns(3)
    t_unit = unit_label("temperature", system)
    row2[0].metric(f"T_bh [{t_unit}]", _value(state.T_wb_K, "temperature", system, 4))
    row2[1].metric(f"T_pr [{t_unit}]", _value(state.T_dp_K, "temperature", system, 4))
    row2[2].metric(
        f"v [{unit_label('specific_volume', system)}]",
        _value(state.v_m3_per_kg, "specific_volume", system, 4),
    )


def _render_state_chart(
    state: MoistAirState, dead: DeadState, key: str, p: float, system: UnitSystem
) -> None:
    st.markdown("#### Carta psicrométrica")
    points = [ChartPoint("", state, "Estado")]
    if abs(dead.T0_K - state.T_K) > 1e-6 or abs(dead.phi0 - state.phi) > 1e-6:
        try:
            ambient = state_from_T_omega(p, dead.T0_K, dead.omega0)
            points.append(ChartPoint("0", ambient, "Ambiente (estado muerto)"))
        except ValueError:
            pass
    chart_key = f"{key}_chart_{st.session_state.get(f'{key}_chart_n', 0)}"
    try:
        fig = _chart_cached(p, system, tuple(points), ())
        st.plotly_chart(
            fig,
            width="stretch",
            key=chart_key,
            on_select=lambda: _on_state_chart_select(key, chart_key, p),
            selection_mode="points",
        )
    except Exception as exc:  # la carta no debe tumbar la página
        st.warning(f"No se pudo dibujar la carta: {exc}")
        return
    st.caption(
        f"Carta a {_fmt(p, 'pressure', system, 4)}. Pasá el mouse (o tocá) para ver el estado de "
        "cualquier punto; al tocarlo se cargan su T y su φ como datos. En la leyenda se prenden "
        "y apagan las familias de líneas (las de bulbo húmedo empiezan apagadas: son casi "
        "paralelas a las de h). El 0 es el ambiente de la exergía."
    )


def _render_state_table(state: MoistAirState, dead: DeadState, system: UnitSystem) -> None:
    st.markdown("#### Propiedades")
    thermo, mixing = state.psi_parts(dead)
    eh: QuantityKind = "specific_enthalpy"
    w_unit = "lb/lb a.s." if system == "Inglés" else "kg/kg a.s."
    rows = [
        (
            "p",
            "presión total",
            _value(state.p_Pa, "pressure", system),
            unit_label("pressure", system),
        ),
        (
            "p_v",
            "presión del vapor",
            _value(state.p_v_Pa, "pressure", system),
            unit_label("pressure", system),
        ),
        (
            "p_a",
            "presión del aire seco",
            _value(state.p_a_Pa, "pressure", system),
            unit_label("pressure", system),
        ),
        (
            "p_vs",
            "presión de saturación a T" + (" (hielo)" if state.T_K < 273.16 else ""),
            _value(state.p_vs_Pa, "pressure", system),
            unit_label("pressure", system),
        ),
        ("ω", "humedad absoluta", format_value(state.omega, 5), w_unit),
        ("ω_s", "humedad de saturación a T", format_value(state.omega_s, 5), w_unit),
        ("φ", "humedad relativa", format_value(100 * state.phi, 5), "%"),
        (
            "μ",
            "grado de saturación",
            format_value(None if state.mu is None else 100 * state.mu, 5),
            "%",
        ),
        (
            "h",
            "entalpía (por kg de a.s.)",
            _value(state.h_J_per_kg, eh, system),
            unit_label(eh, system),
        ),
        (
            "c_p,ah",
            "calor específico (por kg de a.s.)",
            _value(state.cp_J_per_kg_K, "specific_heat", system),
            unit_label("specific_heat", system),
        ),
        (
            "v",
            "volumen (por kg de a.s.)",
            _value(state.v_m3_per_kg, "specific_volume", system),
            unit_label("specific_volume", system),
        ),
        (
            "v_ah",
            "volumen (por kg de aire húmedo)",
            _value(state.v_moist_m3_per_kg, "specific_volume", system),
            unit_label("specific_volume", system),
        ),
        (
            "ρ",
            "densidad del aire húmedo",
            _value(state.rho_kg_per_m3, "density", system),
            unit_label("density", system),
        ),
        (
            "R_ah",
            "constante del aire húmedo",
            _value(state.R_J_per_kg_K, "specific_heat", system),
            unit_label("specific_heat", system),
        ),
        (
            "T_bh",
            "bulbo húmedo (saturación adiabática)",
            _value(state.T_wb_K, "temperature", system),
            unit_label("temperature", system),
        ),
        (
            "T_pr",
            "punto de rocío" + (" (escarcha)" if state.dew_over_ice else ""),
            _value(state.T_dp_K, "temperature", system),
            unit_label("temperature", system),
        ),
        (
            "s",
            "entropía (por kg de a.s.)",
            _value(state.s_J_per_kg_K, "specific_entropy", system),
            unit_label("specific_entropy", system),
        ),
        ("ψ_tm", "exergía térmica y mecánica", _value(thermo, eh, system), unit_label(eh, system)),
        ("ψ_qu", "exergía química (de mezcla)", _value(mixing, eh, system), unit_label(eh, system)),
        ("ψ", "exergía de flujo", _value(thermo + mixing, eh, system), unit_label(eh, system)),
    ]
    frame = pd.DataFrame(rows, columns=["Símbolo", "Propiedad", "Valor", "Unidad"])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        f"Exergía contra el ambiente a {_fmt(dead.T0_K, 'temperature', system, 4)} y "
        f"φ₀ = {_pct(dead.phi0)} (se cambia en «Ambiente para la exergía»)."
    )


def _render_comparison(
    state: MoistAirState, pair: tuple[str, str], a: float, b: float, system: UnitSystem
) -> None:
    st.markdown("#### ¿Cuánto se aparta del aire húmedo real?")
    comparison = _comparison_cached(state, pair, a, b)
    if comparison.error:
        st.caption(comparison.error)
        return
    kinds: dict[str, QuantityKind | None] = {
        "T": "temperature",
        "T_wb": "temperature",
        "T_dp": "temperature",
        "h": "specific_enthalpy",
        "v": "specific_volume",
        "rho": "density",
        "omega": None,
        "phi": None,
    }
    rows = []
    for row in comparison.rows:
        kind = kinds[row.key]
        if row.key == "phi":
            model, real, unit = _pct(row.model), _pct(row.coolprop), ""
        elif kind is None:
            model, real, unit = format_value(row.model, 5), format_value(row.coolprop, 5), ""
        else:
            model, real = _value(row.model, kind, system), _value(row.coolprop, kind, system)
            unit = unit_label(kind, system)
        diff = row.difference
        if diff is None:
            diff_text = "—"
        elif row.is_temperature:
            dT = convert_from_si(diff, "temperature_difference", system)
            diff_text = f"{format_value(dT, 3)} {unit_label('temperature_difference', system)}"
        else:
            diff_text = f"{format_value(100 * diff, 2)} %"
        rows.append((row.label, model, real, unit, diff_text))
    st.dataframe(
        pd.DataFrame(
            rows,
            columns=[
                "Propiedad",
                "Gas ideal (vademecum)",
                "CoolProp (RP-1485)",
                "Unidad",
                "Diferencia",
            ],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Con los mismos datos, el modelo de gas real de CoolProp (ASHRAE RP-1485, Herrmann, "
        "Kretzschmar y Gatley, 2009) tiene en cuenta el «factor de mejora» (el aire ayuda a que "
        "entre un poco más de vapor: ω_s sube ~0,4 % a 1 atm) y que la mezcla no es un gas ideal. "
        "A presión atmosférica la diferencia es menor que la de leer la carta."
    )


def _render_altitude_sweep(state: MoistAirState, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo cambia con la altura?", expanded=False):
        st.markdown(
            "El mismo aire (misma T y φ) a distintas alturas, de 0 a 5000 m: cambia la presión "
            "total y con ella la humedad absoluta, el bulbo húmedo y la densidad."
        )
        if st.button("Calcular barrido", key="ps_sweep_btn"):
            st.session_state["ps_sweep"] = (state.T_K, state.phi)
        if st.session_state.get("ps_sweep") != (state.T_K, state.phi):
            st.caption("Tocá «Calcular barrido».")
            return
        points = _altitude_sweep_cached(state.T_K, state.phi)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        zs = [pt[0] for pt in points]
        w_unit = "lb/lb" if system == "Inglés" else "kg/kg"
        st.plotly_chart(
            _line_chart(
                zs,
                [pt[1] for pt in points],
                title="Humedad absoluta",
                x_title="Altura [m]",
                y_title=f"ω [{w_unit} a.s.]",
            ),
            width="stretch",
            key="ps_sweep_w",
        )
        st.plotly_chart(
            _line_chart(
                zs,
                [convert_from_si(pt[2], "temperature", system) for pt in points],
                title="Temperatura de bulbo húmedo",
                x_title="Altura [m]",
                y_title=f"T_bh [{unit_label('temperature', system)}]",
            ),
            width="stretch",
            key="ps_sweep_wb",
        )
        st.plotly_chart(
            _line_chart(
                zs,
                [convert_from_si(pt[3], "density", system) for pt in points],
                title="Densidad del aire húmedo",
                x_title="Altura [m]",
                y_title=f"ρ [{unit_label('density', system)}]",
            ),
            width="stretch",
            key="ps_sweep_rho",
        )
        st.caption(
            "A menor presión, ω_s = 0,622·p_vs/(p − p_vs) crece: con la misma φ hay más vapor por "
            "kg de aire seco. El bulbo húmedo baja (el agua se evapora más fácil) y el aire es "
            "menos denso: por eso en la altura los ventiladores mueven menos masa con el mismo "
            "caudal."
        )


# ---------------------------------------------------------------------
# Modo 2: procesos de acondicionamiento
# ---------------------------------------------------------------------


def _spec_label(spec: ProcessSpec | None) -> str:
    if isinstance(spec, HeatingHumidification):
        return _PROC_HEAT_HUM
    if isinstance(spec, CoolingDehumidification):
        return _PROC_COIL
    if isinstance(spec, AdiabaticHumidification):
        return _PROC_ADIABATIC
    if isinstance(spec, AdiabaticMixing):
        return _PROC_MIX
    return _PROC_SENSIBLE


def _read_source(key: str, default: float | None, help_text: str) -> float | None:
    chosen = st.checkbox(
        "Indicar la temperatura de la fuente de calor (para la exergía)",
        value=default is not None,
        key=f"{key}_src_on",
        help=help_text,
    )
    if not chosen:
        return None
    return _read_T(
        f"{key}_src", "Temperatura de la fuente T_b", default if default is not None else 333.15
    )


def _read_process(
    key: str, k: int, default: ProcessSpec | None, flow_kind: str, system: UnitSystem
) -> ProcessSpec:
    pkey = f"{key}_p{k}"
    with st.container(border=True):
        kind = st.selectbox(
            f"Proceso {k + 1}",
            _PROC_TYPES,
            index=_PROC_TYPES.index(_spec_label(default)),
            key=f"{pkey}_type",
        )
        left, right = st.columns(2)
        if kind == _PROC_SENSIBLE:
            d = default if isinstance(default, SensibleProcess) else SensibleProcess(298.15)
            with left:
                T_out = _read_T(f"{pkey}_Tout", "T de salida", d.T_out_K)
            with right:
                T_src = _read_source(
                    pkey,
                    d.T_source_K,
                    "Si no la indicás: para calentar, agua caliente a 60 °C (o 10 K más que la "
                    "salida); para enfriar, un serpentín 5 K más frío que la salida, sin bajar del "
                    "punto de rocío.",
                )
            return SensibleProcess(T_out, T_src)
        if kind == _PROC_HEAT_HUM:
            d = (
                default
                if isinstance(default, HeatingHumidification)
                else HeatingHumidification(298.15, 0.5)
            )
            with left:
                T_out = _read_T(f"{pkey}_Tout", "T de salida", d.T_out_K)
                phi_out = _read_phi(f"{pkey}_phi", "φ de salida", d.phi_out, low=0.1)
            with right:
                water = st.radio(
                    "Agua que se agrega",
                    (_WATER_VAPOR, _WATER_LIQUID),
                    index=0 if d.water == "vapor" else 1,
                    horizontal=True,
                    key=f"{pkey}_water",
                )
                T_w = _read_T(f"{pkey}_Tw", "Temperatura del agua T_w", d.T_water_K)
                T_src = _read_source(
                    pkey,
                    d.T_source_K,
                    "Si no la indicás: agua caliente a 60 °C (o 10 K más que la salida).",
                )
            return HeatingHumidification(
                T_out, phi_out, "vapor" if water == _WATER_VAPOR else "liquid", T_w, T_src
            )
        if kind == _PROC_COIL:
            d = (
                default
                if isinstance(default, CoolingDehumidification)
                else CoolingDehumidification(287.15, 0.95)
            )
            with left:
                T_out = _read_T(f"{pkey}_Tout", "T de salida", d.T_out_K)
                phi_out = _read_phi(f"{pkey}_phi", "φ de salida", d.phi_out, low=1.0)
            with right:
                same = st.checkbox(
                    "El condensado sale a la T del aire (como Cengel)",
                    value=d.T_water_K is None or math.isclose(d.T_water_K, d.T_out_K),
                    key=f"{pkey}_Tw_same",
                    help="El condensado sale a la temperatura de la superficie fría del serpentín "
                    "(vademecum §14.12.3), que no puede ser mayor que la del aire que sale.",
                )
                T_w = (
                    None
                    if same
                    else _read_T(
                        f"{pkey}_Tw",
                        "T del condensado (superficie)",
                        d.T_water_K or d.T_out_K - 2.0,
                    )
                )
            return CoolingDehumidification(T_out, phi_out, T_w)
        if kind == _PROC_ADIABATIC:
            d = (
                default
                if isinstance(default, AdiabaticHumidification)
                else AdiabaticHumidification(0.8)
            )
            with left:
                phi_out = _read_phi(f"{pkey}_phi", "φ de salida", d.phi_out, low=1.0)
                water = st.radio(
                    "Agua que se agrega",
                    (_WATER_LIQUID, _WATER_VAPOR),
                    index=0 if d.water == "liquid" else 1,
                    horizontal=True,
                    key=f"{pkey}_water",
                )
            with right:
                if water == _WATER_LIQUID:
                    recirc = st.checkbox(
                        "El agua recircula (llega al bulbo húmedo del aire)",
                        value=d.T_water_K is None,
                        key=f"{pkey}_recirc",
                    )
                    T_w = (
                        None
                        if recirc
                        else _read_T(
                            f"{pkey}_Tw", "Temperatura del agua T_w", d.T_water_K or 288.15
                        )
                    )
                else:
                    T_w = _read_T(
                        f"{pkey}_Tws",
                        "Temperatura del vapor T_w",
                        d.T_water_K if d.T_water_K and d.T_water_K > 373 else 373.15,
                    )
            return AdiabaticHumidification(
                phi_out, "liquid" if water == _WATER_LIQUID else "vapor", T_w
            )
        d = (
            default
            if isinstance(default, AdiabaticMixing)
            else AdiabaticMixing(305.15, 0.5, AirFlow(0.5, flow_kind))
        )  # type: ignore[arg-type]
        with left:
            T = _read_T(f"{pkey}_Tmix", "T de la otra corriente", d.T_K)
            phi = _read_phi(f"{pkey}_phimix", "φ de la otra corriente", d.phi, low=0.1)
        with right:
            flow = _read_flow_value(
                f"{pkey}_flow", flow_kind, d.flow.value, system, "de la otra corriente"
            )
        return AdiabaticMixing(T, phi, AirFlow(flow, flow_kind))  # type: ignore[arg-type]


def _read_flow_value(
    key: str, flow_kind: str, default: float, system: UnitSystem, what: str = ""
) -> float:
    kind: QuantityKind = "volume_flow" if flow_kind == "volume" else "mass_flow"
    label = ("Caudal volumétrico " if flow_kind == "volume" else "Caudal de aire seco ") + what
    return number_input_si(
        label=label.strip(),
        kind=kind,
        default_si=default,
        key=key,
        format="%.4f",
        min_value_si=1e-6,
        max_value_si=1e5,
        help="El volumétrico se mide en el estado de esa corriente: ṁ_a = V̇/v.",
    )


def _render_hvac_mode(system: UnitSystem) -> None:
    ex = _example("Ejemplo precargado", _HVAC_NAMES, "hv_example")
    name = _HVAC_NAMES[ex]
    example = HVAC_EXAMPLES[name]
    key = f"hv_{ex}"
    p, z = _read_pressure(key, example.p_Pa, example.altitude_m, system)
    st.markdown("#### Aire que entra (1)")
    left, right = st.columns(2)
    with left:
        T_in = _read_T(f"{key}_Tin", "Temperatura T₁", example.inlet.T_K)
        phi_in = _read_phi(f"{key}_phiin", "Humedad relativa φ₁", example.inlet.phi, low=0.1)
    with right:
        flow_label = st.radio(
            "Caudal",
            (_FLOW_VOLUME, _FLOW_DRY),
            index=0 if example.inlet.flow.kind == "volume" else 1,
            horizontal=True,
            key=f"{key}_flowkind",
        )
        flow_kind = "volume" if flow_label == _FLOW_VOLUME else "dry_air"
        flow = _read_flow_value(
            f"{key}_flow_{flow_kind}", flow_kind, example.inlet.flow.value, system
        )
    st.markdown("#### Procesos")
    n = int(
        st.radio(
            "Cantidad de procesos",
            _N_PROCESSES,
            index=len(example.processes) - 1,
            horizontal=True,
            key=f"{key}_n",
        )
    )
    processes = []
    for k in range(n):
        default = example.processes[k] if k < len(example.processes) else None
        processes.append(_read_process(key, k, default, flow_kind, system))
    dead = _read_dead(key, example.dead, (T_in, phi_in), p)
    try:
        inputs = HvacInputs(
            p,
            AirInlet(T_in, phi_in, AirFlow(flow, flow_kind)),
            tuple(processes),
            dead=dead,
            altitude_m=z,
        )  # type: ignore[arg-type]
        result = _hvac_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    note = HVAC_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    _render_hvac_metrics(result, system)
    for item in hvac_notes(result):
        st.info(item)
    st.markdown("#### El proceso en la carta")
    points, arrows = hvac_chart_items(result)
    try:
        st.plotly_chart(
            _chart_cached(p, system, tuple(points), tuple(arrows)), width="stretch", key="hv_chart"
        )
    except Exception as exc:  # la carta no debe tumbar la página
        st.warning(f"No se pudo dibujar la carta: {exc}")
    st.caption(
        "Calentamiento en naranja, enfriamiento en azul, humidificación en aguamarina y mezcla "
        "en gris. El calentamiento sensible es horizontal (ω constante); el enfriador "
        "evaporativo sigue casi una recta de h constante; la mezcla cae sobre la recta que une "
        "las dos corrientes."
    )
    _render_hvac_tables(result, system)
    _render_hvac_exergy(result, system)
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Los balances de masa de aire seco, de agua y de energía de cada proceso (vademecum "
            "§14.12), con los números reemplazados en el sistema de unidades elegido."
        )
        for i, step in enumerate(hvac_steps(result, system), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)
    _render_export(hvac_to_dict(result, system), "procesos_aire_humedo", "hv")
    _render_hvac_sweep(inputs, system)


def _render_hvac_metrics(result: HvacResult, system: UnitSystem) -> None:
    P = unit_label("power", system)
    M = unit_label("mass_flow", system)
    row1 = st.columns(2)
    row1[0].metric(f"Calor entregado [{P}]", _value(result.heating_W, "power", system, 4))
    row1[1].metric(f"Calor quitado [{P}]", _value(result.cooling_W, "power", system, 4))
    row2 = st.columns(2)
    row2[0].metric(f"Agua agregada [{M}]", _value(result.water_added_kg_s, "mass_flow", system, 4))
    row2[1].metric(
        f"Agua condensada [{M}]", _value(result.water_removed_kg_s, "mass_flow", system, 4)
    )
    out = result.outlet.state
    row3 = st.columns(3)
    row3[0].metric(
        f"T de salida [{unit_label('temperature', system)}]",
        _value(out.T_K, "temperature", system, 4),
    )
    row3[1].metric("φ de salida [%]", format_value(100 * out.phi, 4))
    row3[2].metric(f"Ẋ destruida [{P}]", _value(result.X_destroyed_W, "power", system, 4))


def _render_hvac_tables(result: HvacResult, system: UnitSystem) -> None:
    st.markdown("#### Estados")
    dead = result.dead
    t_unit = unit_label("temperature", system)
    eh = unit_label("specific_enthalpy", system)
    rows = []
    for s in result.streams:
        st_ = s.state
        rows.append(
            {
                "Estado": s.number,
                "Descripción": s.label,
                f"T [{t_unit}]": _value(st_.T_K, "temperature", system, 4),
                "φ [%]": format_value(100 * st_.phi, 4),
                "ω": format_value(st_.omega, 4),
                f"h [{eh}]": _value(st_.h_J_per_kg, "specific_enthalpy", system, 5),
                f"T_bh [{t_unit}]": _value(st_.T_wb_K, "temperature", system, 4),
                f"ṁ_a [{unit_label('mass_flow', system)}]": _value(
                    s.m_dry_air_kg_s, "mass_flow", system, 4
                ),
                f"V̇ [{unit_label('volume_flow', system)}]": _value(
                    s.V_m3_s, "volume_flow", system, 4
                ),
                f"ψ [{eh}]": _value(st_.psi(dead), "specific_enthalpy", system, 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("#### Procesos")
    P = unit_label("power", system)
    rows = []
    for proc in result.processes:
        rows.append(
            {
                "Proceso": proc.label,
                "Vademecum": PROCESS_SECTIONS[proc.kind],
                f"Q̇ [{P}]": _value(proc.Q_W, "power", system, 4),
                f"ṁ_w [{unit_label('mass_flow', system)}]": _value(
                    proc.m_water_kg_s, "mass_flow", system, 4
                ),
                f"T_b [{t_unit}]": _value(proc.T_source_K, "temperature", system, 4),
                f"Ẋ destruida [{P}]": _value(proc.X_destroyed_W, "power", system, 4),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Q̇ > 0: calor que recibe el aire; Q̇ < 0: calor que se le quita. ṁ_w > 0: agua que se "
        "agrega; ṁ_w < 0: agua que condensa. T_b es la temperatura de la fuente de calor (o de "
        "la superficie del serpentín), con la que se cuenta la exergía del calor."
    )


def _render_hvac_exergy(result: HvacResult, system: UnitSystem) -> None:
    st.markdown("#### Exergía: ¿dónde se pierde?")
    try:
        st.plotly_chart(hvac_exergy_figure(result, system), width="stretch", key="hv_exergy")
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    dead = result.dead
    st.caption(
        f"Contra el ambiente a {_fmt(dead.T0_K, 'temperature', system, 4)} y φ₀ = "
        f"{_pct(dead.phi0)}. Calentar con agua caliente o vapor muy por encima de la "
        "temperatura del aire, mezclar corrientes distintas o evaporar agua en aire seco son "
        "procesos irreversibles: destruyen exergía (vademecum §11). Con una fuente más cercana "
        "a la temperatura del aire, se destruye menos."
    )


_HVAC_SWEEPS = {
    "Temperatura del aire que entra": ("T_in", "temperature", "T₁"),
    "Humedad relativa del aire que entra": ("phi_in", None, "φ₁ [%]"),
}


def _render_hvac_sweep(inputs: HvacInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo cambian las cargas con el aire exterior?", expanded=False):
        st.markdown(
            "El mismo tren con otro aire de entrada (el resto queda igual): cuánto calor y agua "
            "hacen falta."
        )
        choice = st.selectbox("Variable", list(_HVAC_SWEEPS), key="hv_sweep_param")
        parameter, kind, axis = _HVAC_SWEEPS[choice]
        if st.button("Calcular barrido", key="hv_sweep_btn"):
            st.session_state["hv_sweep"] = (inputs, parameter)
        if st.session_state.get("hv_sweep") != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve el tren varias veces).")
            return
        points = _hvac_sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        if kind is None:
            xs = [100 * pt[0] for pt in points]
            x_title = axis
        else:
            xs = [convert_from_si(pt[0], kind, system) for pt in points]
            x_title = f"{axis} [{unit_label(kind, system)}]"
        P = unit_label("power", system)
        st.plotly_chart(
            _lines_chart(
                xs,
                {
                    "entregado": [convert_from_si(pt[1], "power", system) for pt in points],
                    "quitado": [convert_from_si(pt[2], "power", system) for pt in points],
                },
                title="Calor",
                x_title=x_title,
                y_title=f"Q̇ [{P}]",
            ),
            width="stretch",
            key="hv_sweep_q",
        )
        M = unit_label("mass_flow", system)
        st.plotly_chart(
            _lines_chart(
                xs,
                {
                    "agregada": [convert_from_si(pt[3], "mass_flow", system) for pt in points],
                    "condensada": [convert_from_si(pt[4], "mass_flow", system) for pt in points],
                },
                title="Agua",
                x_title=x_title,
                y_title=f"ṁ_w [{M}]",
            ),
            width="stretch",
            key="hv_sweep_w",
        )
        skipped = len(default_hvac_sweep_values(inputs, parameter)) - len(points)  # type: ignore[arg-type]
        if skipped:
            st.caption(
                f"{skipped} valor(es) se omiten: con ese aire algún proceso no tiene sentido (un "
                "serpentín que no condensa, un humidificador que no humidifica, niebla…)."
            )


# ---------------------------------------------------------------------
# Modo 3: torre de enfriamiento
# ---------------------------------------------------------------------


def _render_tower_mode(system: UnitSystem) -> None:
    ex = _example("Ejemplo precargado", _TOWER_NAMES, "ct_example")
    name = _TOWER_NAMES[ex]
    example = TOWER_EXAMPLES[name]
    key = f"ct_{ex}"
    p, z = _read_pressure(key, example.p_Pa, example.altitude_m, system)
    st.markdown("#### Agua caliente (3 → 4)")
    cols = st.columns(3)
    with cols[0]:
        m_w = number_input_si(
            label="Caudal de agua ṁ₃",
            kind="mass_flow",
            default_si=example.m_water_in_kg_s,
            key=f"{key}_mw",
            format="%.3f",
            min_value_si=1e-4,
            max_value_si=1e6,
        )
    with cols[1]:
        T_w_in = _read_T(f"{key}_Twin", "T del agua que entra T₃", example.T_water_in_K)
    with cols[2]:
        T_w_out = _read_T(f"{key}_Twout", "T del agua que sale T₄", example.T_water_out_K)
    st.markdown("#### Aire (1 → 2)")
    left, right = st.columns(2)
    with left:
        T_a1 = _read_T(f"{key}_Ta1", "T del aire que entra T₁", example.T_air_in_K)
        phi_a1 = _read_phi(f"{key}_phia1", "φ del aire que entra", example.phi_air_in, low=1.0)
    with right:
        T_a2 = _read_T(f"{key}_Ta2", "T del aire que sale T₂", example.T_air_out_K)
        phi_a2 = _read_phi(f"{key}_phia2", "φ del aire que sale", example.phi_air_out, low=1.0)
    dead = _read_dead(key, example.dead, (T_a1, phi_a1), p)
    inputs = CoolingTowerInputs(
        p, m_w, T_w_in, T_w_out, T_a1, phi_a1, T_a2, phi_a2, dead=dead, altitude_m=z
    )
    try:
        result = _tower_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    note = TOWER_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    M = unit_label("mass_flow", system)
    row1 = st.columns(3)
    row1[0].metric(f"ṁ aire seco [{M}]", _value(result.m_dry_air_kg_s, "mass_flow", system, 4))
    row1[1].metric(
        f"V̇ aire [{unit_label('volume_flow', system)}]",
        _value(result.V_air_in_m3_s, "volume_flow", system, 4),
    )
    row1[2].metric(f"Reposición [{M}]", _value(result.makeup_kg_s, "mass_flow", system, 4))
    dT = unit_label("temperature_difference", system)
    row2 = st.columns(3)
    row2[0].metric(f"Rango [{dT}]", _value(result.range_K, "temperature_difference", system, 4))
    row2[1].metric(
        f"Aproximación [{dT}]", _value(result.approach_K, "temperature_difference", system, 4)
    )
    row2[2].metric("Efectividad", format_value(result.effectiveness, 3))
    row3 = st.columns(3)
    row3[0].metric("L/G", format_value(result.water_to_air, 3))
    row3[1].metric(f"Calor [{unit_label('power', system)}]", _value(result.Q_W, "power", system, 4))
    row3[2].metric(
        f"Ẋ destruida [{unit_label('power', system)}]",
        _value(result.X_destroyed_W, "power", system, 4),
    )
    for item in cooling_tower_notes(result):
        st.info(item)
    st.markdown("#### El aire en la carta")
    points = (
        ChartPoint("1", result.air_in, "Aire que entra"),
        ChartPoint("2", result.air_out, "Aire que sale"),
    )
    arrows = (ChartArrow(result.air_in, result.air_out, "humidification"),)
    try:
        st.plotly_chart(
            _chart_cached(p, system, points, arrows, (T_w_in, T_w_out)),
            width="stretch",
            key="ct_chart",
        )
    except Exception as exc:  # la carta no debe tumbar la página
        st.warning(f"No se pudo dibujar la carta: {exc}")
    st.caption(
        "El aire se calienta y se humedece: sale con más entalpía. El agua no puede enfriarse "
        "por debajo del bulbo húmedo del aire que entra "
        f"({_fmt(result.air_in.T_wb_K, 'temperature', system, 4)}): ese es el límite de la torre."
    )
    _render_tower_table(result, system)
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Los balances de masa de aire seco, de agua y de energía de la torre (vademecum "
            "§14.12.7; Cengel ejemplo 14-9), en el sistema de unidades elegido."
        )
        for i, step in enumerate(cooling_tower_steps(result, system), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)
    _render_export(cooling_tower_to_dict(result, system), "torre_de_enfriamiento", "ct")
    _render_tower_sweep(inputs, system)


def _render_tower_table(result: CoolingTowerResult, system: UnitSystem) -> None:
    st.markdown("#### Estados")
    ti = result.inputs
    t_unit = unit_label("temperature", system)
    eh = unit_label("specific_enthalpy", system)
    M = unit_label("mass_flow", system)
    rows = [
        {
            "Estado": "1",
            "Descripción": "aire que entra",
            f"T [{t_unit}]": _value(result.air_in.T_K, "temperature", system, 4),
            "φ [%]": format_value(100 * result.air_in.phi, 4),
            "ω": format_value(result.air_in.omega, 4),
            f"h [{eh}]": _value(result.air_in.h_J_per_kg, "specific_enthalpy", system, 5),
            f"ṁ [{M}]": _value(result.m_dry_air_kg_s, "mass_flow", system, 4) + " (a.s.)",
        },
        {
            "Estado": "2",
            "Descripción": "aire que sale",
            f"T [{t_unit}]": _value(result.air_out.T_K, "temperature", system, 4),
            "φ [%]": format_value(100 * result.air_out.phi, 4),
            "ω": format_value(result.air_out.omega, 4),
            f"h [{eh}]": _value(result.air_out.h_J_per_kg, "specific_enthalpy", system, 5),
            f"ṁ [{M}]": _value(result.m_dry_air_kg_s, "mass_flow", system, 4) + " (a.s.)",
        },
        {
            "Estado": "3",
            "Descripción": "agua caliente",
            f"T [{t_unit}]": _value(ti.T_water_in_K, "temperature", system, 4),
            "φ [%]": "—",
            "ω": "—",
            f"h [{eh}]": _value(result.h_water_in, "specific_enthalpy", system, 5),
            f"ṁ [{M}]": _value(ti.m_water_in_kg_s, "mass_flow", system, 4),
        },
        {
            "Estado": "4",
            "Descripción": "agua enfriada",
            f"T [{t_unit}]": _value(ti.T_water_out_K, "temperature", system, 4),
            "φ [%]": "—",
            "ω": "—",
            f"h [{eh}]": _value(result.h_water_out, "specific_enthalpy", system, 5),
            f"ṁ [{M}]": _value(result.m_water_out_kg_s, "mass_flow", system, 4),
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


_TOWER_SWEEPS = {
    "Temperatura del agua que sale (aproximación)": ("T_water_out", "temperature", "T₄"),
    "Temperatura del aire que entra": ("T_air_in", "temperature", "T₁"),
    "Humedad relativa del aire que entra": ("phi_air_in", None, "φ₁ [%]"),
}


def _render_tower_sweep(inputs: CoolingTowerInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo cambia la torre?", expanded=False):
        st.markdown("Elegí qué variar (el resto queda como en tu torre):")
        choice = st.selectbox("Variable", list(_TOWER_SWEEPS), key="ct_sweep_param")
        parameter, kind, axis = _TOWER_SWEEPS[choice]
        if st.button("Calcular barrido", key="ct_sweep_btn"):
            st.session_state["ct_sweep"] = (inputs, parameter)
        if st.session_state.get("ct_sweep") != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve la torre varias veces).")
            return
        points = _tower_sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        if kind is None:
            xs = [100 * pt[0] for pt in points]
            x_title = axis
        else:
            xs = [convert_from_si(pt[0], kind, system) for pt in points]
            x_title = f"{axis} [{unit_label(kind, system)}]"
        M = unit_label("mass_flow", system)
        st.plotly_chart(
            _line_chart(
                xs,
                [convert_from_si(pt[1], "mass_flow", system) for pt in points],
                title="Caudal de aire seco",
                x_title=x_title,
                y_title=f"ṁ_a [{M}]",
            ),
            width="stretch",
            key="ct_sweep_air",
        )
        st.plotly_chart(
            _line_chart(
                xs,
                [convert_from_si(pt[3], "mass_flow", system) for pt in points],
                title="Agua de reposición (evaporada)",
                x_title=x_title,
                y_title=f"ṁ_rep [{M}]",
            ),
            width="stretch",
            key="ct_sweep_makeup",
        )
        st.caption(
            "Con menos aproximación (agua más fría) hace falta mucho más aire: cerca del bulbo "
            "húmedo, cada grado cuesta más. Con aire más seco o más frío la torre enfría mejor."
        )


# ---------------------------------------------------------------------
# Gráficos simples y export
# ---------------------------------------------------------------------


def _line_chart(
    xs: list[float], ys: list[float], *, title: str, x_title: str, y_title: str
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=xs, y=ys, mode="lines+markers", line={"color": "#2a78d6", "width": 2})
    )
    fig.update_layout(
        height=270,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        separators=". ",
    )
    return fig


def _lines_chart(
    xs: list[float], series: dict[str, list[float]], *, title: str, x_title: str, y_title: str
) -> go.Figure:
    colors = ("#eb6834", "#2a78d6")
    fig = go.Figure()
    for (name, ys), color in zip(series.items(), colors, strict=False):
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=name, line={"color": color, "width": 2}
            )
        )
    fig.update_layout(
        height=290,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.4},
        separators=". ",
    )
    return fig


def _render_export(data: dict[str, Any], file_name: str, key: str) -> None:
    st.markdown("#### 💾 Exportar")
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name=f"{file_name}.csv",
        mime="text/csv",
        key=f"{key}_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name=f"{file_name}.json",
        mime="application/json",
        key=f"{key}_download_json",
    )


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§14 *Aire húmedo (psicrometría)*** (§14.1 a §14.11, el estado; §14.12, los "
            "procesos) y §11 *Exergía*. Aire seco y vapor de agua como **gases ideales** con "
            "calores específicos constantes; todo **por kg de aire seco**, que se conserva. "
            "Çengel & Boles, cap. 14 (§14-1 a §14-7 y la carta de la figura A-31)."
        )
        st.markdown(
            "**Presiones y humedades** (§14.1 a §14.4): ley de Dalton y la definición de ω."
        )
        st.latex(r"p = p_a + p_v \qquad \omega = 0.622\,\frac{p_v}{p - p_v}")
        st.latex(r"\varphi = \frac{p_v}{p_{vs}(T)} \qquad \mu = \frac{\omega}{\omega_s}")
        st.markdown(
            "p_vs(T) es la presión de saturación del agua a T (tabla A-4) o, bajo 0 °C, del "
            "hielo (tabla A-8): por debajo de 0 °C el punto de rocío es de escarcha."
        )
        st.markdown(
            "**Entalpía, volumen y densidad** (§14.5 a §14.8), con t = T − 273,15 K, "
            "c_p,a = 1,005 y c_p,v = 1,864 kJ/(kg·K) y r₀ = 2501 kJ/kg:"
        )
        st.latex(r"h = c_{p,a}\,t + \omega\,(r_0 + c_{p,v}\,t)")
        st.latex(r"v = \frac{R_a\,T}{p_a} \qquad \rho = \frac{1 + \omega}{v}")
        st.markdown(
            "**Bulbo húmedo = saturación adiabática** (§14.12.5): el aire queda saturado al "
            "recibir agua a T_bh sin intercambiar calor. **Punto de rocío**: donde el vapor se "
            "satura a su presión parcial."
        )
        st.latex(r"h + (\omega_s^{*} - \omega)\,h_w^{*} = h^{*}")
        st.latex(r"T_{pr} = T_{\mathrm{sat}}(p_v)")
        st.markdown("**Exergía de flujo** (§14.11), contra el ambiente (T₀, p₀, ω₀):")
        st.latex(
            r"\begin{aligned}\psi_{tm} &= c_{p,ah}\Bigl[(T - T_0) - T_0\ln\frac{T}{T_0}\Bigr]"
            r" \\ &\quad + \nu\,R_a\,T_0\ln\frac{p}{p_0}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}\psi_{qu} &= R_a\,T_0\Bigl[\nu\ln\frac{\nu_0}{\nu}"
            r" \\ &\quad + (\nu - 1)\ln\frac{\omega}{\omega_0}\Bigr]\end{aligned}"
        )
        st.markdown("con ν = 1 + 1,608·ω (moles de aire húmedo por mol de aire seco).")
        st.markdown(
            "**Procesos** (§14.12), en régimen permanente; ṁ_a se conserva y el agua entra o "
            "sale con h_w de tablas. Calentamiento o enfriamiento sensible (ω constante) y "
            "calentamiento con humidificación:"
        )
        st.latex(r"\dot Q = \dot m_a\,(h_2 - h_1)")
        st.latex(r"\dot Q + \dot m_w\,h_w = \dot m_a\,(h_2 - h_1)")
        st.markdown("Serpentín de enfriamiento y deshumidificación (el condensado sale a T_w):")
        st.latex(r"\dot Q = \dot m_a\,\bigl[(h_2 - h_1) + (\omega_1 - \omega_2)\,h_w\bigr]")
        st.markdown("Humidificación adiabática (enfriador evaporativo) y mezcla adiabática:")
        st.latex(r"h_2 = h_1 + (\omega_2 - \omega_1)\,h_w")
        st.latex(
            r"\frac{\dot m_{a,1}}{\dot m_{a,2}} = \frac{\omega_2 - \omega_3}{\omega_3 - \omega_1}"
            r" = \frac{h_2 - h_3}{h_3 - h_1}"
        )
        st.markdown("**Torre de enfriamiento** (§14.12.7): agua 3 → 4, aire 1 → 2.")
        st.latex(
            r"\dot m_a = \frac{\dot m_3\,(h_3 - h_4)}{(h_2 - h_1) - (\omega_2 - \omega_1)\,h_4}"
        )
        st.markdown(
            "**Presión con la altura Z** (ASHRAE, *Handbook—Fundamentals* 2017, cap. 1), en metros:"
        )
        st.latex(r"p = p_0\,(1 - 2.25577\times 10^{-5}\,Z)^{5.2559}")
        st.latex(r"p_0 = 101.325\ \mathrm{kPa}")


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Psicrometría", page_icon="🌫️", layout="centered")

st.subheader(SUBJECT)
st.title("🌫️ Psicrometría: aire húmedo")
st.markdown(
    "El aire húmedo es una mezcla de aire seco y vapor de agua. Con la presión y dos datos "
    "(por ejemplo, la temperatura y la humedad relativa) salen todas sus propiedades y su "
    "lugar en la **carta psicrométrica**. Con eso se resuelven los **procesos de "
    "acondicionamiento** (calentar, enfriar y secar, humidificar, mezclar) y las **torres de "
    "enfriamiento**."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Psicrometría")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, horizontal=True, key="ps_mode")
if mode == _MODE_STATE:
    _render_state_mode(system)
elif mode == _MODE_HVAC:
    _render_hvac_mode(system)
else:
    _render_tower_mode(system)
