"""Página 10 — Caldera de recuperación (HRSG) de una presión (Fase 3.3).

Arma el diagrama T–Q de una HRSG de una presión a partir de los gases que
entran (temperatura, composición y caudal), la presión de evaporación, la
temperatura del agua de alimentación, el pinch, el approach y la temperatura
del vapor sobrecalentado, o con vapor saturado (sin sobrecalentador). El
cálculo lo hace :mod:`core.cycles.hrsg` (balances de energía sección por
sección, gases ideales y agua IAPWS-95); la página muestra el caudal de
vapor, el diagrama, las secciones, los estados, la comparación entre vapor
saturado y sobrecalentado, el procedimiento, la exportación y barridos.
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
from core.cycles.hrsg_procedure import hrsg_steps
from core.export import dict_to_csv
from core.fluids import fluid_state_from_pair
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.cycle_charts import tq_figure
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.15.0"

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


def _read_gas(key: str, name: str, base: HRSGInputs) -> FlueGas | None:
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
        for i, step in enumerate(hrsg_steps(result, system), start=1):
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
            "aire). El diseño con pinch y approach, de Kehlhofer, Hannemann, Stirnimann y Rukes, "
            "*Combined-Cycle Gas & Steam Turbine Power Plants* (3.ª ed., PennWell, 2009); el "
            "ciclo combinado, en Çengel & Boles §10-9."
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


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="HRSG", page_icon="🏭", layout="centered")

st.subheader(SUBJECT)
st.title("🏭 Caldera de recuperación (HRSG)")
st.markdown(
    "Una caldera de recuperación de una presión aprovecha los gases calientes (por ejemplo, el "
    "escape de una turbina de gas): pasan por el **sobrecalentador**, el **evaporador** y el "
    "**economizador**, y el agua hace el camino inverso. Con el **pinch** y el **approach**, el "
    "balance de energía de cada sección da el caudal de vapor, la temperatura de chimenea y el "
    "**diagrama T–Q**, con vapor sobrecalentado o saturado."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="HRSG")
render_units_selector()
system = get_current_system()

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
        _render_states(result, system)
        _render_procedure(result, system)
        _render_export(result, system)
        _render_sweeps(inputs, system)
