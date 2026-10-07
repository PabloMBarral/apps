"""Página 10 — Caldera de recuperación (HRSG) de una, dos o tres presiones (Fases 3.3 y 3.5).

Arma el diagrama T–Q de una HRSG a partir de los gases que entran
(temperatura, composición y caudal), la temperatura del agua de alimentación
y, para cada nivel de presión, la presión de evaporación, el pinch, el
approach y la temperatura del vapor sobrecalentado (o vapor saturado). Con
una presión el cálculo lo hace :mod:`core.cycles.hrsg`; con dos o tres,
:mod:`core.cycles.hrsg_multi` (niveles en cascada). La página muestra los
caudales de vapor, el diagrama, las secciones, los estados, la exergía
(cuánto vale el calor recuperado y cuánto se destruye), la comparación entre
vapor saturado y sobrecalentado o entre 1, 2 y 3 presiones, el
procedimiento, la exportación y barridos.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.hrsg import (
    GAS_NAMES,
    GAS_SPECIES,
    HRSG_EXAMPLE_NOTES,
    HRSG_EXAMPLES,
    HRSG_EXAMPLES_EXCESS_AIR,
    FlueGas,
    HRSGInputs,
    HRSGResult,
    SweepParameter,
    TQProfile,
    default_steam_temperature,
    default_sweep_values,
    exhaust_composition,
    hrsg_notes,
    hrsg_sweep,
    hrsg_to_dict,
    hrsg_tq_profile,
    other_steam_option,
    solve_hrsg,
)
from core.cycles.hrsg_multi import (
    MULTI_HRSG_EXAMPLE_NOTES,
    MULTI_HRSG_EXAMPLES,
    HRSGExergy,
    MultiHRSGInputs,
    MultiHRSGResult,
    MultiSweepParameter,
    MultiTQProfile,
    PressureLevel,
    default_multi_sweep_values,
    from_single,
    hrsg_exergy,
    level_comparison,
    level_names,
    multi_hrsg_notes,
    multi_hrsg_sweep,
    multi_hrsg_to_dict,
    multi_tq_profile,
    solve_multi_hrsg,
)
from core.cycles.hrsg_multi_procedure import exergy_step, multi_hrsg_steps
from core.cycles.hrsg_procedure import hrsg_steps
from core.export import dict_to_csv
from core.fluids import fluid_state_from_pair
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.cycle_charts import (
    LEVEL_COLORS,
    exergy_sections_figure,
    exergy_split_figure,
    multi_tq_figure,
    tq_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.18.0"

_EXAMPLES = list(HRSG_EXAMPLES)
# Opciones fijas: cambiarlas reiniciaría el widget.
_STEAM_SH = "Sobrecalentado"
_STEAM_SAT = "Saturado (sin sobrecalentador)"
_COMP_PRESET = "Gases de metano con exceso de aire λ"
_COMP_MANUAL = "Cargar la composición"
_BASES = {"Molar (% en volumen)": "molar", "Másica (%)": "mass"}
_WATER_LABELS = (
    "1 alimentación",
    "2 salida del economizador",
    "3 líquido saturado (domo)",
    "4 vapor saturado",
    "5 vapor sobrecalentado",
)
_GAS_PLACES = {
    "a": "entrada",
    "b": "salida del sobrecalentador",
    "c": "salida del evaporador (pinch)",
    "d": "chimenea",
}

_LEVEL_OPTIONS = ("1 presión", "2 presiones", "3 presiones")
_MULTI_EXAMPLES = {
    n: [name for name, i in MULTI_HRSG_EXAMPLES.items() if len(i.levels) == n] for n in (2, 3)
}
_WATER_STATES = (
    "entrada al economizador",
    "salida del economizador",
    "líquido saturado (domo)",
    "vapor saturado",
    "vapor sobrecalentado",
)
_LETTER = {"alta": "A", "media": "M", "baja": "B"}

# (parámetro, magnitud del eje, rótulo del eje)
_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind, str]] = {
    "Pinch": ("pinch", "temperature_difference", "ΔT pinch"),
    "Approach": ("approach", "temperature_difference", "ΔT approach"),
    "Presión de evaporación": ("p_steam", "pressure", "p evaporación"),
    "Temperatura del vapor": ("T_steam", "temperature", "T vapor"),
    "Temperatura del agua de alimentación": ("T_feedwater", "temperature", "T alimentación"),
    "Temperatura de los gases": ("T_gas_in", "temperature", "T gases"),
}
_SWEEP_NOTES: dict[str, str] = {
    "pinch": (
        "Con menos pinch los gases salen del evaporador más fríos: se recupera más calor "
        "arriba del pinch, sale más vapor y la chimenea queda más fría. Pero la diferencia de "
        "temperatura media baja y el evaporador y el economizador necesitan más área (más "
        "caros): por eso se diseña con unos 5 a 15 K."
    ),
    "approach": (
        "Con más approach el agua llega más fría al domo: cada kilogramo de vapor necesita un "
        "poco más de calor arriba del pinch (sale menos vapor) y la chimenea queda más "
        "caliente. Se deja unos pocos kelvin para que el economizador no evapore cuando la "
        "caldera trabaja a carga parcial."
    ),
    "p_steam": (
        "Con más presión el vapor vale más (más exergía: la turbina saca más trabajo de cada "
        "kilogramo), pero T_sat sube, los gases salen del evaporador más calientes y se "
        "recupera menos calor: menos vapor y la chimenea más caliente. Ese compromiso lleva a "
        "las calderas de dos y tres presiones. Cerca de la crítica h_fg se achica y el caudal "
        "vuelve a subir."
    ),
    "T_steam": (
        "Más sobrecalentamiento: cada kilogramo de vapor necesita más calor arriba del pinch, "
        "así que sale menos vapor y la chimenea queda más caliente; a cambio, el vapor vale "
        "más y llega más seco al final de la turbina (Cengel §10-4)."
    ),
    "T_feedwater": (
        "El caudal de vapor no cambia: lo fija el tramo entre la entrada de los gases y el "
        "pinch, donde el agua entra a T_sat − approach. El agua de alimentación más caliente "
        "solo deja menos calor para el economizador y la chimenea queda más caliente; pero "
        "tiene que quedar por encima del punto de rocío de los gases."
    ),
    "T_gas_in": (
        "Gases más calientes: hay más calor arriba del pinch, sale más vapor y, como por el "
        "economizador pasa más agua, la chimenea queda más fría."
    ),
}


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _solve_cached(inputs: HRSGInputs) -> tuple[HRSGResult, TQProfile]:
    result = solve_hrsg(inputs)
    return result, hrsg_tq_profile(result)


@st.cache_data(show_spinner=False)
def _other_cached(inputs: HRSGInputs) -> tuple[HRSGInputs | None, HRSGResult | None, str]:
    """La misma caldera con la otra opción de vapor (o por qué no se puede)."""
    other = other_steam_option(inputs)
    if other is None:
        return None, None, ""
    try:
        return other, solve_hrsg(other), ""
    except ValueError as exc:
        return other, None, str(exc)


@st.cache_data(show_spinner="Calculando el barrido…")
def _sweep_cached(
    inputs: HRSGInputs, parameter: SweepParameter
) -> list[tuple[float, float, float, float]]:
    values = default_sweep_values(inputs, parameter)
    return [
        (p.value_si, p.m_steam_kg_s, p.T_stack_K, p.Q_W)
        for p in hrsg_sweep(inputs, parameter, values)
    ]


@st.cache_data(show_spinner=False)
def _exergy_single_cached(inputs: HRSGInputs) -> HRSGExergy:
    """La exergía de la caldera de una presión, con el modelo de varias presiones (1 nivel)."""
    return hrsg_exergy(solve_multi_hrsg(from_single(inputs)))


@st.cache_data(show_spinner=False)
def _solve_multi_cached(inputs: MultiHRSGInputs) -> tuple[MultiHRSGResult, MultiTQProfile]:
    result = solve_multi_hrsg(inputs)
    return result, multi_tq_profile(result)


@st.cache_data(show_spinner=False)
def _comparison_cached(
    inputs: MultiHRSGInputs,
) -> list[tuple[str, MultiHRSGResult | None, HRSGExergy | None, str]]:
    return [
        (label, r, None if r is None else hrsg_exergy(r), error)
        for label, r, error in level_comparison(inputs)
    ]


@st.cache_data(show_spinner="Calculando el barrido…")
def _multi_sweep_cached(
    inputs: MultiHRSGInputs, parameter: MultiSweepParameter
) -> list[tuple[float, tuple[float, ...], float, float]]:
    values = default_multi_sweep_values(inputs, parameter)
    return [
        (p.value_si, p.m_steam_kg_s, p.T_stack_K, p.exergy_efficiency)
        for p in multi_hrsg_sweep(inputs, parameter, values)
    ]


# ---------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------


def _number(value: float) -> str:
    """Número para mostrar: sin notación exponencial en los valores grandes."""
    if abs(value) >= 1.0e4:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, 5)


def _value(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Valor en ``system``, sin unidad."""
    if kind == "power":
        return _number(convert_from_si(value_si, kind, system))
    return format_value(convert_from_si(value_si, kind, system), sig)


def _fmt(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Valor con su unidad."""
    return f"{_value(value_si, kind, system, sig)} {unit_label(kind, system)}"


def _T_sat(p_Pa: float) -> float | None:
    try:
        return fluid_state_from_pair("Water", "PX", p=p_Pa, x=0.0).T_K
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _example_index() -> int:
    label = st.selectbox(
        "Ejemplo precargado",
        _EXAMPLES,
        key="hr_example",
        help="Podés cambiar cualquier dato: el resultado se actualiza solo.",
    )
    return _EXAMPLES.index(label)


def _carried_fractions(key: str, basis: str) -> dict[str, float] | None:
    """Al cambiar de fracción molar a másica (o al revés), la composición que estaba cargada.

    Streamlit conserva los valores de los widgets de la corrida anterior
    hasta el final de la actual (como el cambio de unidades en
    :mod:`ui.units_ui`). ``None`` si no hay nada que convertir.
    """
    if f"{key}_{basis}_{GAS_SPECIES[0]}" in st.session_state:
        return None
    other = "mass" if basis == "molar" else "molar"
    values = [st.session_state.get(f"{key}_{other}_{s}") for s in GAS_SPECIES]
    if any(v is None for v in values):
        return None
    try:
        gas = FlueGas.from_fractions(dict(zip(GAS_SPECIES, values, strict=True)), basis=other)  # type: ignore[arg-type]
    except ValueError:
        return None
    return gas.mole_fractions if basis == "molar" else gas.mass_fractions


def _read_gas(key: str, name: str, base: HRSGInputs | MultiHRSGInputs) -> FlueGas | None:
    """Composición (preset con λ o cargada) y presión de los gases."""
    excess = HRSG_EXAMPLES_EXCESS_AIR.get(name)
    mode = st.radio(
        "Composición",
        (_COMP_PRESET, _COMP_MANUAL),
        index=0 if excess is not None else 1,
        horizontal=True,
        key=f"{key}_comp",
        help=(
            "El preset quema metano con el aire técnico del vademecum (§16.1: 21 % O₂ y 79 % "
            "N₂). Si no, cargá N₂, O₂, CO₂, H₂O y Ar."
        ),
    )
    basis = "molar"
    if mode == _COMP_PRESET:
        lam = st.number_input(
            "Exceso de aire λ",
            min_value=1.0,
            max_value=10.0,
            value=float(excess if excess is not None else 3.0),
            step=0.1,
            format="%.3g",
            key=f"{key}_lambda",
            help=(
                "λ = aire real / aire estequiométrico (vademecum §16.2). Una turbina de gas "
                "trabaja con λ ≈ 2,5 a 3,5: el aire de más baja la temperatura a la entrada de "
                "la turbina."
            ),
        )
        fractions = exhaust_composition(float(lam))
    else:
        basis = _BASES[st.radio("Fracciones", tuple(_BASES), horizontal=True, key=f"{key}_basis")]
        defaults = _carried_fractions(key, basis) or (
            base.gas.mole_fractions if basis == "molar" else base.gas.mass_fractions
        )
        fractions = {}
        for col, species in zip(st.columns(len(GAS_SPECIES)), GAS_SPECIES, strict=True):
            with col:
                fractions[species] = (
                    st.number_input(
                        f"{GAS_NAMES[species]} [%]",
                        min_value=0.0,
                        max_value=100.0,
                        value=round(defaults[species] * 100.0, 2),
                        step=0.1,
                        format="%.2f",
                        key=f"{key}_{basis}_{species}",
                    )
                    / 100.0
                )
        total = sum(fractions.values()) * 100.0
        if total > 0.0 and abs(total - 100.0) > 0.05:
            st.caption(f"Suman {format_value(total, 5)} %: se normalizan a 100 %.")
    p_gas = number_input_si(
        label="Presión de los gases",
        kind="pressure",
        default_si=base.gas.p_Pa,
        key=f"{key}_pg",
        format="%.5g",
        min_value_si=0.0,
        help=(
            "Solo cuenta para el punto de rocío: la presión parcial del vapor de agua es "
            "y_H₂O·p (ley de Dalton). A la salida de una turbina de gas, casi la atmosférica."
        ),
    )
    try:
        gas = FlueGas.from_fractions(fractions, basis=basis, p_Pa=p_gas)  # type: ignore[arg-type]
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return None
    y = gas.mole_fractions
    parts = [f"{GAS_NAMES[s]} {format_value(y[s] * 100.0, 4)} %" for s in GAS_SPECIES if y[s] > 0]
    st.caption(
        "En fracción molar: "
        + " · ".join(parts)
        + f". M = {format_value(gas.M_kg_per_mol * 1e3, 4)} kg/kmol."
    )
    return gas


def _read_inputs(ex: int) -> HRSGInputs | None:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    name = _EXAMPLES[ex]
    base = HRSG_EXAMPLES[name]
    key = f"hr_{ex}"
    st.markdown("#### Gases")
    left, right = st.columns(2)
    with left:
        T_gas = number_input_si(
            label="Entrada de los gases Tₐ",
            kind="temperature",
            default_si=base.T_gas_in_K,
            key=f"{key}_Tg",
            format="%.4g",
            help="El escape de la turbina de gas (o los gases calientes que se aprovechan).",
        )
    with right:
        m_gas = number_input_si(
            label="Caudal de gases ṁ_g",
            kind="mass_flow",
            default_si=base.m_gas_kg_s,
            key=f"{key}_mg",
            format="%.5g",
            min_value_si=0.0,
        )
    gas = _read_gas(key, name, base)

    st.markdown("#### Agua y vapor")
    left, right = st.columns(2)
    with left:
        p_steam = number_input_si(
            label="Presión de evaporación p",
            kind="pressure",
            default_si=base.p_steam_Pa,
            key=f"{key}_p",
            format="%.5g",
            min_value_si=0.0,
            help=(
                "La del domo. El agua y el vapor la mantienen en toda la caldera (sin pérdidas "
                "de carga)."
            ),
        )
        T_sat = _T_sat(p_steam)
        if T_sat is None:
            st.caption("Sin saturación a esa presión (revisá el dato).")
        else:
            st.caption(f"T_sat = {_fmt(T_sat, 'temperature', get_current_system())}")
    with right:
        T_feed = number_input_si(
            label="Agua de alimentación T₁",
            kind="temperature",
            default_si=base.T_feedwater_K,
            key=f"{key}_Tfw",
            format="%.4g",
            help="Como llega a la caldera: después del desaireador o de los precalentadores.",
        )
    steam = st.radio(
        "Vapor",
        (_STEAM_SH, _STEAM_SAT),
        index=0 if base.superheated else 1,
        horizontal=True,
        key=f"{key}_steam",
        help=(
            "Sobrecalentado para una turbina; saturado, sin sobrecalentador, para vapor de proceso."
        ),
    )
    T_steam = None
    if steam == _STEAM_SH:
        default = base.T_steam_K or default_steam_temperature(base)
        if default is None:
            default = (T_sat or base.T_feedwater_K) + 50.0
        T_steam = number_input_si(
            label="Vapor sobrecalentado T₅",
            kind="temperature",
            default_si=default,
            key=f"{key}_Ts",
            format="%.4g",
            help="Suele quedar unos 20 a 30 K por debajo de los gases que entran.",
        )

    st.markdown("#### Diseño")
    left, right = st.columns(2)
    with left:
        pinch = number_input_si(
            label="Pinch ΔT_pinch",
            kind="temperature_difference",
            default_si=base.pinch_K,
            key=f"{key}_pinch",
            format="%.3g",
            min_value_si=0.0,
            help=(
                "Los gases a la salida del evaporador menos T_sat: la menor diferencia de "
                "temperatura de la caldera. Se diseña con unos 5 a 15 K."
            ),
        )
    with right:
        approach = number_input_si(
            label="Approach ΔT_approach",
            kind="temperature_difference",
            default_si=base.approach_K,
            key=f"{key}_approach",
            format="%.3g",
            min_value_si=0.0,
            help=(
                "T_sat menos el agua que sale del economizador: lo que le falta para hervir. "
                "Unos 3 a 10 K."
            ),
        )
    if gas is None:
        return None
    return HRSGInputs(T_gas, m_gas, gas, p_steam, T_feed, T_steam, pinch, approach)


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


def _render_metrics(result: HRSGResult, system: UnitSystem) -> None:
    T_unit = unit_label("temperature", system)
    row1 = st.columns(3)
    row1[0].metric(
        f"ṁ vapor [{unit_label('mass_flow', system)}]",
        _value(result.m_steam_kg_s, "mass_flow", system),
    )
    row1[1].metric(f"Q̇ [{unit_label('power', system)}]", _value(result.Q_W, "power", system))
    row1[2].metric(f"T chimenea [{T_unit}]", _value(result.T_stack_K, "temperature", system, 4))
    row2 = st.columns(3)
    row2[0].metric(f"T_sat [{T_unit}]", _value(result.T_sat_K, "temperature", system, 4))
    if result.T_after_superheater_K is not None:
        row2[1].metric(
            f"Gases tras el SH [{T_unit}]",
            _value(result.T_after_superheater_K, "temperature", system, 4),
        )
    else:
        row2[1].metric(
            f"Gases tras el EV [{T_unit}]",
            _value(result.T_pinch_gas_K, "temperature", system, 4),
        )
    row2[2].metric("Aprovechamiento [%]", f"{result.recovery * 100:.1f}")
    gas = result.inputs.gas
    dew = result.dew_point_K
    parts = [
        f"Gases: M = {format_value(gas.M_kg_per_mol * 1e3, 4)} kg/kmol",
        f"R = {_fmt(gas.R_J_per_kg_K, 'specific_heat', system, 4)}",
        f"c_p medio = {_fmt(result.cp_gas_mean_J_per_kg_K, 'specific_heat', system, 4)} entre la "
        "entrada y la chimenea",
    ]
    if dew is not None:
        parts.append(f"punto de rocío {_fmt(dew, 'temperature', system, 4)}")
    st.caption(
        " · ".join(parts) + ". El aprovechamiento compara Q̇ con el calor que cederían los gases "
        f"enfriándose hasta {_fmt(result.inputs.T_ref_K, 'temperature', system, 4)}."
    )
    for note in hrsg_notes(result):
        st.info(note)


def _render_tq(result: HRSGResult, profile: TQProfile, system: UnitSystem) -> None:
    st.markdown("#### Diagrama T–Q")
    try:
        st.plotly_chart(tq_figure(result, profile, system), width="stretch", key="hr_tq_chart")
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    sh = " y el sobrecalentador (SH)" if result.inputs.superheated else ""
    st.caption(
        "Cada curva es la temperatura de una corriente contra el calor que intercambió desde la "
        f"chimenea: a la izquierda el economizador (ECO), después el evaporador (EV){sh}. Los "
        "gases (a → d) van de derecha a izquierda; el agua (1 → "
        f"{len(result.water)}), de izquierda a derecha. La curva de los gases no es una recta "
        "porque su c_p crece con T. El escalón vertical del agua (2 → 3) es el approach: el agua "
        "del economizador se termina de calentar al mezclarse en el domo, y los tubos del "
        "evaporador la ven a T_sat. Las curvas se acercan más en el pinch (c y 3): ese punto "
        "fija el caudal de vapor."
    )


def _render_sections(result: HRSGResult, system: UnitSystem) -> None:
    st.markdown("#### Secciones")
    T_unit = unit_label("temperature", system)
    dT_unit = unit_label("temperature_difference", system)
    rows = []
    for s in result.sections:
        rows.append(
            {
                "Sección": s.name,
                f"Q̇ [{unit_label('power', system)}]": _value(s.Q_W, "power", system),
                "% de Q̇": f"{s.Q_W / result.Q_W * 100:.1f}",
                f"Gases [{T_unit}]": (
                    f"{_value(s.T_gas_in_K, 'temperature', system, 4)} → "
                    f"{_value(s.T_gas_out_K, 'temperature', system, 4)}"
                ),
                f"Agua [{T_unit}]": (
                    f"{_value(s.T_water_in_K, 'temperature', system, 4)} → "
                    f"{_value(s.T_water_out_K, 'temperature', system, 4)}"
                ),
                f"ΔT caliente [{dT_unit}]": _value(s.dT_hot_K, "temperature_difference", system, 3),
                f"ΔT frío [{dT_unit}]": _value(s.dT_cold_K, "temperature_difference", system, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "En el sentido de los gases. ΔT caliente: entre los gases que entran y el agua (o vapor) "
        "que sale; ΔT frío: entre los gases que salen y el agua que entra. El ΔT frío del "
        "evaporador es el pinch; el caliente del economizador, pinch + approach."
    )


def _render_comparison(inputs: HRSGInputs, result: HRSGResult, system: UnitSystem) -> None:
    st.markdown("#### ¿Vapor saturado o sobrecalentado?")
    other, other_result, error = _other_cached(inputs)
    if other is None:
        st.caption(
            "Los gases no llegan a sobrecalentar el vapor (harían falta unos 35 K por encima de "
            "T_sat): no hay con qué comparar."
        )
        return

    def describe(i: HRSGInputs) -> str:
        if i.T_steam_K is None:
            return "Saturado"
        return f"Sobrecalentado a {_fmt(i.T_steam_K, 'temperature', system, 4)}"

    st.caption(
        f"Esta caldera produce vapor {describe(inputs).lower()}; la otra columna es la misma "
        "caldera (mismos gases, pinch y approach) con la otra opción."
    )

    that = describe(other)
    if other_result is None:
        st.caption(f"{that}: {error}")
        return
    T_unit = unit_label("temperature", system)
    e_unit = unit_label("specific_enthalpy", system)

    def column(r: HRSGResult) -> list[str]:
        return [
            _value(r.m_steam_kg_s, "mass_flow", system),
            _value(r.h_steam_J_per_kg - r.water[0].h_J_per_kg, "specific_enthalpy", system),
            _value(r.Q_W, "power", system),
            _value(r.T_stack_K, "temperature", system, 4),
            f"{r.recovery * 100:.1f}",
        ]

    frame = pd.DataFrame(
        {
            "Magnitud": [
                f"ṁ vapor [{unit_label('mass_flow', system)}]",
                f"Calor por kg de vapor [{e_unit}]",
                f"Q̇ [{unit_label('power', system)}]",
                f"T chimenea [{T_unit}]",
                "Aprovechamiento [%]",
            ],
            "Esta caldera": column(result),
            that: column(other_result),
        }
    )
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        "Con el mismo pinch, sin sobrecalentador cada kilogramo de vapor necesita menos calor "
        "arriba del pinch (h₄ − h₂ en vez de h₅ − h₂): sale más vapor, por el economizador pasa "
        "más agua y la chimenea queda más fría. El vapor sobrecalentado vale más para una "
        "turbina (más exergía y menos humedad al final de la expansión, Cengel §10-4); el "
        "saturado alcanza para calentar un proceso, que aprovecha su calor de condensación."
    )


def _render_states(result: HRSGResult, system: UnitSystem) -> None:
    st.markdown("#### Estados")
    st.markdown("**Agua y vapor**")
    labeled = list(zip(_WATER_LABELS, result.water, strict=False))
    frame = pd.DataFrame(states_table(labeled, system))
    frame = frame.drop(
        columns=[c for c in frame.columns if c == "Fluido" or c.startswith(("v [", "u ["))]
    )
    for column in frame.columns:
        if column not in ("Estado", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.markdown("**Gases**")
    T_unit = unit_label("temperature", system)
    e_unit = unit_label("specific_enthalpy", system)
    rows = [
        {
            "Punto": label,
            "Dónde": _GAS_PLACES[label],
            f"T [{T_unit}]": _value(T, "temperature", system),
            f"h_g [{e_unit}]": _value(h, "specific_enthalpy", system),
        }
        for label, T, h in zip(
            result.gas_labels, result.T_gas_K, result.h_gas_J_per_kg, strict=True
        )
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("h_g medida desde 25 °C (h_g = 0 ahí): solo importan sus diferencias.")


def _render_procedure(result: HRSGResult, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cómo se resuelve a mano, sección por sección. El agua y el vapor salen de IAPWS-95 "
            "(lo mismo que las tablas de Cengel); los gases, del gas ideal de CoolProp."
        )
        steps = hrsg_steps(result, system)
        # La exergía, con el modelo de varias presiones (un nivel da la misma caldera).
        steps.insert(
            -1 if steps[-1].title.startswith("Punto de rocío") else len(steps),
            exergy_step(_solve_multi_cached(from_single(result.inputs))[0], system),
        )
        for i, step in enumerate(steps, start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_export(result: HRSGResult, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    data = hrsg_to_dict(result, system)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="hrsg.csv",
        mime="text/csv",
        key="hr_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="hrsg.json",
        mime="application/json",
        key="hr_download_json",
    )


def _chart(
    xs: list[float], ys: list[float], *, title: str, x_title: str, y_title: str, log_x: bool
) -> go.Figure:
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        height=270,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        separators=". ",
    )
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def _render_sweeps(inputs: HRSGInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo cambia la caldera?", expanded=False):
        st.markdown(
            "Elegí qué variar (el resto queda como en tu caldera) para ver cómo cambian el caudal "
            "de vapor y la temperatura de chimenea:"
        )
        sweeps = dict(_SWEEPS)
        if not inputs.superheated:
            del sweeps["Temperatura del vapor"]
        choice = st.selectbox("Variable", list(sweeps), key="hr_sweep_param")
        parameter, kind, axis = sweeps[choice]
        if st.button("Calcular barrido", key="hr_sweep_btn"):
            st.session_state["hr_sweep"] = (inputs, parameter)
        if st.session_state.get("hr_sweep") != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve la caldera varias veces).")
            return
        points = _sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        xs = [convert_from_si(p[0], kind, system) for p in points]
        x_title = f"{axis} [{unit_label(kind, system)}]"
        log_x = parameter == "p_steam"
        st.plotly_chart(
            _chart(
                xs,
                [convert_from_si(p[1], "mass_flow", system) for p in points],
                title="Caudal de vapor",
                x_title=x_title,
                y_title=f"ṁ vapor [{unit_label('mass_flow', system)}]",
                log_x=log_x,
            ),
            width="stretch",
            key="hr_sweep_steam",
        )
        st.plotly_chart(
            _chart(
                xs,
                [convert_from_si(p[2], "temperature", system) for p in points],
                title="Temperatura de chimenea",
                x_title=x_title,
                y_title=f"T [{unit_label('temperature', system)}]",
                log_x=log_x,
            ),
            width="stretch",
            key="hr_sweep_stack",
        )
        skipped = len(default_sweep_values(inputs, parameter)) - len(points)
        if skipped:
            st.caption(
                f"{skipped} valor(es) del barrido no tienen sentido físico (cruce de "
                "temperaturas, condensación en la chimenea…) y se omiten."
            )
        st.caption(_SWEEP_NOTES[parameter])


def _render_single_exergy(inputs: HRSGInputs, system: UnitSystem) -> None:
    st.markdown("#### Exergía: cuánto vale el calor recuperado")
    exergy = _exergy_single_cached(inputs)
    _exergy_metrics(exergy, system)
    try:
        st.plotly_chart(
            exergy_sections_figure(exergy, system), width="stretch", key="hr_exergy_sections"
        )
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    st.caption(
        f"Con el ambiente a {_fmt(exergy.T0_K, 'temperature', system, 4)}: de la exergía que "
        "traen los gases (lo máximo que se podría convertir en trabajo), una parte la gana el "
        "agua, otra se destruye porque el calor pasa con diferencia de temperatura (más en las "
        "secciones donde las curvas del T–Q están más separadas) y otra se va por la chimenea. "
        "Una caldera de dos o tres presiones acerca las curvas: probalo con el selector de "
        "arriba."
    )


def _exergy_metrics(exergy: HRSGExergy, system: UnitSystem) -> None:
    P = unit_label("power", system)
    row = st.columns(4)
    row[0].metric(f"Ẋ gases [{P}]", _value(exergy.X_gas_in_W, "power", system))
    row[1].metric(f"Ẋ al agua [{P}]", _value(exergy.X_water_W, "power", system))
    row[2].metric(f"Ẋ destruida [{P}]", _value(exergy.X_destroyed_W, "power", system))
    row[3].metric(f"Ẋ chimenea [{P}]", _value(exergy.X_stack_W, "power", system))
    st.caption(
        f"Rendimiento exergético η_II = Ẋ al agua / Ẋ gases = {exergy.efficiency * 100:.1f} %."
    )


# ---------------------------------------------------------------------
# Dos y tres presiones (Fase 3.5)
# ---------------------------------------------------------------------


def _multi_example(n: int) -> str:
    return st.selectbox(
        "Ejemplo precargado",
        _MULTI_EXAMPLES[n],
        key=f"hm{n}_example",
        help="Podés cambiar cualquier dato: el resultado se actualiza solo.",
    )


def _read_level(key: str, name: str, base: PressureLevel, system: UnitSystem) -> PressureLevel:
    st.markdown(f"**Nivel de {name}**")
    left, right = st.columns(2)
    with left:
        p = number_input_si(
            label=f"Presión de {name}",
            kind="pressure",
            default_si=base.p_Pa,
            key=f"{key}_p",
            format="%.5g",
            min_value_si=0.0,
        )
        T_sat = _T_sat(p)
        st.caption(
            "Sin saturación a esa presión."
            if T_sat is None
            else f"T_sat = {_fmt(T_sat, 'temperature', system)}"
        )
    with right:
        steam = st.radio(
            f"Vapor de {name}",
            (_STEAM_SH, _STEAM_SAT),
            index=0 if base.superheated else 1,
            horizontal=True,
            key=f"{key}_steam",
        )
        T_steam = None
        if steam == _STEAM_SH:
            default = base.T_steam_K or ((T_sat or 373.15) + 30.0)
            T_steam = number_input_si(
                label=f"Vapor sobrecalentado de {name}",
                kind="temperature",
                default_si=default,
                key=f"{key}_Ts",
                format="%.4g",
            )
    left, right = st.columns(2)
    with left:
        pinch = number_input_si(
            label=f"Pinch de {name}",
            kind="temperature_difference",
            default_si=base.pinch_K,
            key=f"{key}_pinch",
            format="%.3g",
            min_value_si=0.0,
        )
    with right:
        approach = number_input_si(
            label=f"Approach de {name}",
            kind="temperature_difference",
            default_si=base.approach_K,
            key=f"{key}_approach",
            format="%.3g",
            min_value_si=0.0,
        )
    return PressureLevel(p, T_steam, pinch, approach)


def _read_multi_inputs(n: int, name: str) -> MultiHRSGInputs | None:
    """Widgets con los valores del ejemplo (su posición y la cantidad de niveles van en la key)."""
    base = MULTI_HRSG_EXAMPLES[name]
    key = f"hm{n}_{_MULTI_EXAMPLES[n].index(name)}"
    system = get_current_system()
    st.markdown("#### Gases")
    left, right = st.columns(2)
    with left:
        T_gas = number_input_si(
            label="Entrada de los gases Tₐ",
            kind="temperature",
            default_si=base.T_gas_in_K,
            key=f"{key}_Tg",
            format="%.4g",
            help="El escape de la turbina de gas (o los gases calientes que se aprovechan).",
        )
    with right:
        m_gas = number_input_si(
            label="Caudal de gases ṁ_g",
            kind="mass_flow",
            default_si=base.m_gas_kg_s,
            key=f"{key}_mg",
            format="%.5g",
            min_value_si=0.0,
        )
    gas = _read_gas(key, name, base)
    st.markdown("#### Agua y niveles de presión")
    T_feed = number_input_si(
        label="Agua de alimentación",
        kind="temperature",
        default_si=base.T_feedwater_K,
        key=f"{key}_Tfw",
        format="%.4g",
        help="Entra al economizador de baja; de ahí sube, por las bombas, a los otros niveles.",
    )
    levels = tuple(
        _read_level(f"{key}_{_LETTER[nm]}", nm, lv, system)
        for nm, lv in zip(level_names(n), base.levels, strict=True)
    )
    if gas is None:
        return None
    return MultiHRSGInputs(T_gas, m_gas, gas, levels, T_feed)


def _render_multi_metrics(result: MultiHRSGResult, exergy: HRSGExergy, system: UnitSystem) -> None:
    T_unit = unit_label("temperature", system)
    m_unit = unit_label("mass_flow", system)
    row1 = st.columns(3)
    row1[0].metric(f"ṁ vapor total [{m_unit}]", _value(result.m_steam_kg_s, "mass_flow", system))
    row1[1].metric(f"Q̇ [{unit_label('power', system)}]", _value(result.Q_W, "power", system))
    row1[2].metric(f"T chimenea [{T_unit}]", _value(result.T_stack_K, "temperature", system, 4))
    row2 = st.columns(len(result.levels))
    for col, lv in zip(row2, result.levels, strict=True):
        col.metric(f"ṁ vapor de {lv.name} [{m_unit}]", _value(lv.m_steam_kg_s, "mass_flow", system))
    row3 = st.columns(3)
    row3[0].metric("Aprovechamiento [%]", f"{result.recovery * 100:.1f}")
    row3[1].metric("η exergético [%]", f"{exergy.efficiency * 100:.1f}")
    row3[2].metric(
        f"Ẇ bombas [{unit_label('power', system)}]", _value(result.W_pumps_W, "power", system)
    )
    dew = result.dew_point_K
    st.caption(
        "El aprovechamiento compara Q̇ con el calor que cederían los gases enfriándose hasta "
        f"{_fmt(result.inputs.T_ref_K, 'temperature', system, 4)}; el rendimiento exergético, la "
        "exergía que gana el agua con la que traen los gases. Las bombas llevan el agua de un "
        "domo al nivel siguiente (fuera de la caldera)"
        + (
            ""
            if dew is None
            else f". Punto de rocío de los gases: {_fmt(dew, 'temperature', system, 4)}"
        )
        + "."
    )
    for note in multi_hrsg_notes(result):
        st.info(note)


def _render_multi_tq(result: MultiHRSGResult, profile: MultiTQProfile, system: UnitSystem) -> None:
    st.markdown("#### Diagrama T–Q")
    try:
        st.plotly_chart(
            multi_tq_figure(result, profile, system), width="stretch", key="hm_tq_chart"
        )
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    st.caption(
        "Las secciones van, desde la chimenea, del economizador de baja al sobrecalentador de "
        "alta (los nombres arriba: ECO, EV y SH, con A alta, M media y B baja; las muy angostas "
        "no llevan nombre). La curva del agua es un serrucho: el economizador de cada nivel "
        "arranca a la temperatura del domo de abajo, porque su bomba trae el líquido saturado "
        "de ese domo. Los segmentos punteados son el pinch de cada nivel (los gases a la salida "
        "de su evaporador contra T_sat). Comparada con una sola presión, la curva del agua "
        "sigue más de cerca a la de los gases: la chimenea queda más fría."
    )


def _render_multi_sections(result: MultiHRSGResult, system: UnitSystem) -> None:
    st.markdown("#### Secciones")
    T_unit = unit_label("temperature", system)
    dT_unit = unit_label("temperature_difference", system)
    rows = [
        {
            "Sección": s.name,
            f"Q̇ [{unit_label('power', system)}]": _value(s.Q_W, "power", system),
            "% de Q̇": f"{s.Q_W / result.Q_W * 100:.1f}",
            f"Gases [{T_unit}]": (
                f"{_value(s.T_gas_in_K, 'temperature', system, 4)} → "
                f"{_value(s.T_gas_out_K, 'temperature', system, 4)}"
            ),
            f"Agua [{T_unit}]": (
                f"{_value(s.T_water_in_K, 'temperature', system, 4)} → "
                f"{_value(s.T_water_out_K, 'temperature', system, 4)}"
            ),
            f"ΔT caliente [{dT_unit}]": _value(s.dT_hot_K, "temperature_difference", system, 3),
            f"ΔT frío [{dT_unit}]": _value(s.dT_cold_K, "temperature_difference", system, 3),
        }
        for s in result.sections
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "En el sentido de los gases. El evaporador de cada nivel incluye el domo: ahí se "
        "termina de calentar (de T_sat − approach a T_sat) también el agua que sube a los "
        "niveles de mayor presión."
    )


def _render_levels_comparison(inputs: MultiHRSGInputs, system: UnitSystem) -> None:
    st.markdown("#### ¿Cuánto ganás con más presiones?")
    rows = _comparison_cached(inputs)
    T_unit = unit_label("temperature", system)
    P = unit_label("power", system)
    table = []
    bars: list[tuple[str, HRSGExergy]] = []
    for label, r, x, error in rows:
        if r is None or x is None:
            st.caption(f"{label}: {error}")
            continue
        bars.append((label, x))
        table.append(
            {
                "Caldera": label,
                f"T chimenea [{T_unit}]": _value(r.T_stack_K, "temperature", system, 4),
                f"Q̇ [{P}]": _value(r.Q_W, "power", system),
                f"ṁ vapor [{unit_label('mass_flow', system)}]": " + ".join(
                    _value(lv.m_steam_kg_s, "mass_flow", system, 4) for lv in r.levels
                ),
                "Aprovechamiento [%]": f"{r.recovery * 100:.1f}",
                "η exergético [%]": f"{x.efficiency * 100:.1f}",
                f"Ẋ destruida [{P}]": _value(x.X_destroyed_W, "power", system),
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    if bars:
        try:
            st.plotly_chart(exergy_split_figure(bars, system), width="stretch", key="hm_split")
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
    st.caption(
        "Los mismos gases, la misma agua de alimentación y los mismos niveles que cargaste: "
        "solo el de alta, alta y baja, y los tres. Con un solo nivel de alta presión los gases "
        "tienen que salir por encima de su T_sat + pinch y del agua que calienta su "
        "economizador: la chimenea queda caliente. Un nivel de baja usa ese calor (más Q̇ y "
        "chimenea más fría) y, como el agua recibe el calor más cerca de la temperatura de los "
        "gases, se destruye menos exergía. El nivel de media reparte mejor el vapor: más "
        "exergía por kilogramo que el de baja."
    )


def _render_multi_exergy(exergy: HRSGExergy, system: UnitSystem) -> None:
    st.markdown("#### Exergía: ¿dónde se destruye?")
    _exergy_metrics(exergy, system)
    try:
        st.plotly_chart(
            exergy_sections_figure(exergy, system), width="stretch", key="hm_exergy_sections"
        )
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    st.caption(
        "X_destruida = T₀·S_gen de cada sección (Cengel cap. 8): crece con la diferencia de "
        "temperatura entre los gases y el agua. Las secciones con las curvas del T–Q más "
        "separadas (el evaporador y el economizador de alta, donde los gases todavía están "
        "muy calientes) son las que más destruyen."
    )


def _render_multi_states(result: MultiHRSGResult, system: UnitSystem) -> None:
    st.markdown("#### Estados")
    st.markdown("**Agua y vapor**")
    labeled = []
    for lv in result.levels:
        letter = _LETTER[lv.name]
        for k, (desc, state) in enumerate(zip(_WATER_STATES, lv.water, strict=False), start=1):
            labeled.append((f"{k}{letter} {desc} ({lv.name})", state))
    frame = pd.DataFrame(states_table(labeled, system))
    frame = frame.drop(
        columns=[c for c in frame.columns if c == "Fluido" or c.startswith(("v [", "u ["))]
    )
    for column in frame.columns:
        if column not in ("Estado", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        "Cada nivel numera sus estados como la caldera de una presión: 1 entra al economizador "
        "(el agua de alimentación en el de baja; en los otros, la que trae la bomba del domo de "
        "abajo), 2 sale del economizador, 3 y 4 saturados, 5 sobrecalentado."
    )
    st.markdown("**Gases**")
    T_unit = unit_label("temperature", system)
    e_unit = unit_label("specific_enthalpy", system)
    places = ["entrada", *(f"salida del {s.name}" for s in result.sections)]
    places[-1] = "chimenea"
    rows = [
        {
            "Punto": label,
            "Dónde": place,
            f"T [{T_unit}]": _value(T, "temperature", system),
            f"h_g [{e_unit}]": _value(h, "specific_enthalpy", system),
        }
        for (label, T, h), place in zip(result.gas_points, places, strict=True)
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_multi_procedure(result: MultiHRSGResult, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cómo se resuelve a mano, nivel por nivel, de mayor a menor presión (Kehlhofer et "
            "al., cap. 5). El agua y el vapor salen de IAPWS-95 (lo mismo que las tablas de "
            "Cengel); los gases, del gas ideal de CoolProp."
        )
        for i, step in enumerate(multi_hrsg_steps(result, system), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_multi_export(result: MultiHRSGResult, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    data = multi_hrsg_to_dict(result, system)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="hrsg_varias_presiones.csv",
        mime="text/csv",
        key="hm_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="hrsg_varias_presiones.json",
        mime="application/json",
        key="hm_download_json",
    )


_MULTI_SWEEPS: dict[str, tuple[MultiSweepParameter, QuantityKind, str]] = {
    "Presión de baja": ("p_low", "pressure", "p baja"),
    "Presión de alta": ("p_high", "pressure", "p alta"),
    "Pinch (todos los niveles)": ("pinch", "temperature_difference", "ΔT pinch"),
    "Temperatura del agua de alimentación": ("T_feedwater", "temperature", "T alimentación"),
}
_MULTI_SWEEP_NOTES: dict[str, str] = {
    "p_low": (
        "Con la presión de baja más baja el domo de baja hierve más frío: los gases se enfrían "
        "más y la chimenea baja. Pero el vapor de baja vale menos (menos exergía por "
        "kilogramo), y si baja demasiado el economizador de alta arranca más frío y le quita "
        "calor al nivel de baja."
    ),
    "p_high": (
        "Más presión de alta: el vapor de alta vale más y sube el rendimiento exergético, "
        "aunque salga un poco menos de vapor de alta. La chimenea casi no cambia: la fija el "
        "nivel de baja."
    ),
    "pinch": (
        "Menos pinch en todos los niveles: más vapor, chimenea más fría y menos exergía "
        "destruida, a cambio de más área (calderas más grandes y caras)."
    ),
    "T_feedwater": (
        "El agua de alimentación más caliente deja menos calor para el economizador de baja: "
        "la chimenea queda más caliente. Tiene que quedar por encima del punto de rocío de los "
        "gases."
    ),
}


def _render_multi_sweeps(inputs: MultiHRSGInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo cambia la caldera?", expanded=False):
        st.markdown(
            "Elegí qué variar (el resto queda como en tu caldera) para ver cómo cambian la "
            "chimenea, el rendimiento exergético y el vapor de cada nivel:"
        )
        choice = st.selectbox("Variable", list(_MULTI_SWEEPS), key="hm_sweep_param")
        parameter, kind, axis = _MULTI_SWEEPS[choice]
        if st.button("Calcular barrido", key="hm_sweep_btn"):
            st.session_state["hm_sweep"] = (inputs, parameter)
        if st.session_state.get("hm_sweep") != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve la caldera varias veces).")
            return
        points = _multi_sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        xs = [convert_from_si(p[0], kind, system) for p in points]
        x_title = f"{axis} [{unit_label(kind, system)}]"
        log_x = parameter in ("p_low", "p_high")
        st.plotly_chart(
            _chart(
                xs,
                [convert_from_si(p[2], "temperature", system) for p in points],
                title="Temperatura de chimenea",
                x_title=x_title,
                y_title=f"T [{unit_label('temperature', system)}]",
                log_x=log_x,
            ),
            width="stretch",
            key="hm_sweep_stack",
        )
        st.plotly_chart(
            _chart(
                xs,
                [p[3] * 100.0 for p in points],
                title="Rendimiento exergético",
                x_title=x_title,
                y_title="η_II [%]",
                log_x=log_x,
            ),
            width="stretch",
            key="hm_sweep_exergy",
        )
        fig = go.Figure()
        for i, name in enumerate(level_names(len(inputs.levels))):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[convert_from_si(p[1][i], "mass_flow", system) for p in points],
                    mode="lines+markers",
                    name=f"de {name}",
                    line={"color": LEVEL_COLORS[name]},
                )
            )
        fig.update_layout(
            height=280,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            title="Vapor de cada nivel",
            xaxis_title=x_title,
            yaxis_title=f"ṁ vapor [{unit_label('mass_flow', system)}]",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            separators=". ",
        )
        if log_x:
            fig.update_xaxes(type="log")
        st.plotly_chart(fig, width="stretch", key="hm_sweep_steam")
        skipped = len(default_multi_sweep_values(inputs, parameter)) - len(points)
        if skipped:
            st.caption(
                f"{skipped} valor(es) del barrido no tienen sentido físico (vapor más caliente "
                "que los gases que le llegan, cruces, presiones demasiado cerca…) y se omiten."
            )
        st.caption(_MULTI_SWEEP_NOTES[parameter])


def _render_multi(n: int, system: UnitSystem) -> None:
    name = _multi_example(n)
    inputs = _read_multi_inputs(n, name)
    if inputs is None:
        return
    try:
        result, profile = _solve_multi_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    exergy = hrsg_exergy(result)
    st.markdown("### Resultado")
    note = MULTI_HRSG_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    _render_multi_metrics(result, exergy, system)
    _render_multi_tq(result, profile, system)
    _render_multi_sections(result, system)
    _render_levels_comparison(inputs, system)
    _render_multi_exergy(exergy, system)
    _render_multi_states(result, system)
    _render_multi_procedure(result, system)
    _render_multi_export(result, system)
    _render_multi_sweeps(inputs, system)


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§3.3 *Sistema abierto, régimen permanente*** (el balance de cada sección), "
            "**§5 *Mezclas de gases ideales*** (§5.1 y §5.2 fracciones, masa molar y R; §5.3 "
            "ley de Dalton; §5.6 propiedades de la mezcla), §4.8 *Polinomios NASA*, §12 *Vapor "
            "húmedo*, §13 *Líquidos* y §16 *Combustión* (§16.1 aire técnico, §16.2 exceso de "
            "aire) y §11 *Exergía*. El diseño con pinch y approach y las calderas de varias "
            "presiones, de Kehlhofer, Hannemann, Stirnimann y Rukes, *Combined-Cycle Gas & "
            "Steam Turbine Power Plants* (3.ª ed., PennWell, 2009, cap. 5); el ciclo combinado, "
            "en Çengel & Boles §10-9."
        )
        st.markdown(
            "**Pinch y approach.** Los gases salen del evaporador (c) a T_sat + ΔT_pinch y el "
            "agua sale del economizador (2) a T_sat − ΔT_approach:"
        )
        st.latex(r"T_c = T_{\mathrm{sat}} + \Delta T_{\mathrm{pinch}}")
        st.latex(r"T_2 = T_{\mathrm{sat}} - \Delta T_{\mathrm{approach}}")
        st.markdown(
            "**Caudal de vapor.** Entre la entrada de los gases (a) y el pinch (c) están el "
            "sobrecalentador y el evaporador; el balance de ese tramo (adiabático y sin trabajo, "
            "§3.3) da el caudal de vapor (5 es el vapor sobrecalentado; con vapor saturado, 4):"
        )
        st.latex(r"\dot{m}_v\,(h_5 - h_2) = \dot{m}_g\,\bigl[h_g(T_a) - h_g(T_c)\bigr]")
        st.markdown(
            "El economizador usa lo que les queda a los gases y fija la chimenea (d). Por eso "
            "la temperatura del agua de alimentación no cambia el caudal de vapor:"
        )
        st.latex(r"\dot{m}_v\,(h_2 - h_1) = \dot{m}_g\,\bigl[h_g(T_c) - h_g(T_d)\bigr]")
        st.markdown("**Sobrecalentador** y **evaporador** (con el domo), entre a, b y c:")
        st.latex(r"\dot{m}_v\,(h_5 - h_4) = \dot{m}_g\,\bigl[h_g(T_a) - h_g(T_b)\bigr]")
        st.latex(r"\dot{m}_v\,(h_4 - h_2) = \dot{m}_g\,\bigl[h_g(T_b) - h_g(T_c)\bigr]")
        st.markdown(
            "**Gases** (§5): mezcla de gases ideales; con las fracciones molares yᵢ salen la "
            "masa molar, las fracciones másicas y la entalpía de la mezcla, que depende solo "
            "de T (la de cada componente, del gas ideal de CoolProp: lo mismo que los "
            "polinomios NASA, §4.8):"
        )
        st.latex(r"M = \sum_i y_i\,M_i \qquad w_i = \frac{y_i\,M_i}{M}")
        st.latex(r"h_g(T) = \sum_i w_i\,\bigl[h_i(T) - h_i(25\ {}^{\circ}\mathrm{C})\bigr]")
        st.markdown(
            "**Gases de metano con exceso de aire λ** (§16.2), con el aire técnico (§16.1):"
        )
        st.latex(
            r"\begin{aligned}&\mathrm{CH_4} + 2\lambda\left(\mathrm{O_2}"
            r" + \tfrac{0.79}{0.21}\,\mathrm{N_2}\right) \\ &\quad\to \mathrm{CO_2}"
            r" + 2\,\mathrm{H_2O} \\ &\qquad + 2(\lambda - 1)\,\mathrm{O_2}"
            r" + 2\lambda\,\tfrac{0.79}{0.21}\,\mathrm{N_2}\end{aligned}"
        )
        st.markdown(
            "**Aprovechamiento**: el calor recuperado sobre el que cederían los gases "
            "enfriándose hasta 15 °C (la referencia de las normas ISO)."
        )
        st.latex(
            r"\eta_{\mathrm{rec}} = \frac{\dot{Q}}"
            r"{\dot{m}_g\,\bigl[h_g(T_a) - h_g(T_{\mathrm{ref}})\bigr]}"
        )
        st.markdown(
            "**Punto de rocío** (§5.3, ley de Dalton): el vapor de agua de los gases condensa "
            "por debajo de su temperatura de saturación a la presión parcial. La chimenea y "
            "los tubos del economizador tienen que quedar por encima (con azufre, el "
            "condensado es ácido)."
        )
        st.latex(
            r"p_{\mathrm{H_2O}} = y_{\mathrm{H_2O}}\,p \qquad"
            r" T_{\text{rocío}} = T_{\mathrm{sat}}(p_{\mathrm{H_2O}})"
        )
        st.markdown(
            "**Dos y tres presiones, en cascada** (Kehlhofer et al., cap. 5). Los gases pasan "
            "por los niveles de mayor a menor presión. El economizador de baja calienta toda el "
            "agua; cada domo evapora su vapor y manda el resto, como líquido saturado, a la "
            "bomba del nivel siguiente. Para cada nivel, el tramo de los gases desde que llegan "
            "(T_llega) hasta su pinch evapora su vapor y termina de calentar, en el domo, el "
            "agua que sube a los niveles de mayor presión:"
        )
        st.latex(
            r"\begin{aligned}&\dot{m}_i\,(h_{5,i} - h_{2,i}) \\ &\quad"
            r" + \dot{m}_{\mathrm{sube}}\,(h_{3,i} - h_{2,i}) \\ &= \dot{m}_g\,"
            r"\bigl[h_g(T_{\mathrm{llega}}) - h_g(T_{\mathrm{pinch},i})\bigr]\end{aligned}"
        )
        st.latex(r"T_{\mathrm{pinch},i} = T_{\mathrm{sat},i} + \Delta T_{\mathrm{pinch},i}")
        st.markdown(
            "Su economizador calienta su vapor y el de los niveles de mayor presión "
            "(ṁ_ECO = ṁ_i + ṁ_sube) hasta T_sat − approach."
        )
        st.markdown(
            "**Exergía** (§11 *Exergía*: §11.8 trabajo perdido, §11.10 rendimiento exergético; "
            "Çengel & Boles cap. 8). Con el ambiente a T₀ = 15 °C, la exergía de los gases (a "
            "presión constante) es lo máximo que se podría convertir en trabajo al enfriarlos "
            "hasta T₀. Se reparte en la que gana el agua, la destruida (T₀·S_gen de cada "
            "sección) y la que se va por la chimenea:"
        )
        st.latex(
            r"\begin{aligned}x_g(T) &= \bigl[h_g(T) - h_g(T_0)\bigr] \\ &\quad"
            r" - T_0\,\bigl[s^{\circ}(T) - s^{\circ}(T_0)\bigr]\end{aligned}"
        )
        st.latex(
            r"\dot{X}_{\mathrm{gases}} = \dot{X}_{\mathrm{agua}} + \dot{X}_{\mathrm{dest}}"
            r" + \dot{X}_{\mathrm{chim}}"
        )
        st.latex(
            r"\dot{X}_{\mathrm{dest}} = T_0\,\dot{S}_{\mathrm{gen}} \qquad \eta_{\mathrm{II}}"
            r" = \frac{\dot{X}_{\mathrm{agua}}}{\dot{X}_{\mathrm{gases}}}"
        )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="HRSG", page_icon="🏭", layout="centered")

st.subheader(SUBJECT)
st.title("🏭 Caldera de recuperación (HRSG)")
st.markdown(
    "Una caldera de recuperación aprovecha los gases calientes (por ejemplo, el escape de una "
    "turbina de gas): pasan por el **sobrecalentador**, el **evaporador** y el **economizador**, "
    "y el agua hace el camino inverso. Con el **pinch** y el **approach**, el balance de energía "
    "de cada sección da el caudal de vapor, la temperatura de chimenea y el **diagrama T–Q**, "
    "con vapor sobrecalentado o saturado. Con **dos o tres niveles de presión**, la caldera "
    "aprovecha más los gases y destruye menos exergía."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="HRSG")
render_units_selector()
system = get_current_system()

n_levels = 1 + _LEVEL_OPTIONS.index(
    st.radio(
        "Niveles de presión",
        _LEVEL_OPTIONS,
        horizontal=True,
        key="hr_levels",
        help=(
            "Con dos o tres niveles, cada uno con su sobrecalentador, evaporador y economizador, "
            "los gases se enfrían más (Kehlhofer et al., cap. 5)."
        ),
    )
)

if n_levels > 1:
    _render_multi(n_levels, system)
else:
    ex = _example_index()
    inputs = _read_inputs(ex)
    if inputs is not None:
        try:
            result, profile = _solve_cached(inputs)
        except ValueError as exc:
            st.error(str(exc), icon="🚫")
        else:
            st.markdown("### Resultado")
            note = HRSG_EXAMPLE_NOTES.get(_EXAMPLES[ex])
            if note:
                st.caption(f"📘 Sobre este ejemplo: {note}")
            _render_metrics(result, system)
            _render_tq(result, profile, system)
            _render_sections(result, system)
            _render_comparison(inputs, result, system)
            _render_single_exergy(inputs, system)
            _render_states(result, system)
            _render_procedure(result, system)
            _render_export(result, system)
            _render_sweeps(inputs, system)
