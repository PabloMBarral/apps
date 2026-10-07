"""Página 12 — Ciclo combinado gas–vapor de una presión (Fase 3.4).

Una turbina de gas (compresor, cámara de combustión y turbina, con metano o
con el modelo de aire estándar), una caldera de recuperación (HRSG) de una
presión y un ciclo de Rankine con desaireador opcional. El cálculo lo hace
:mod:`core.cycles.combined` (turbina de gas y HRSG con gases ideales; el ciclo
de vapor con TESPy). La página muestra los rendimientos y las potencias, a
dónde va la energía (Sankey), los diagramas de cada parte, las tablas de
estados, el procedimiento, el control de la turbina de gas con TESPy, la
exportación y barridos.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.brayton import METHANE, BraytonInputs, BraytonTespy, brayton_tespy
from core.cycles.combined import (
    COMBINED_EXAMPLE_NOTES,
    COMBINED_EXAMPLES,
    CombinedInputs,
    CombinedResult,
    SteamCycle,
    SweepParameter,
    combined_notes,
    combined_sweep,
    combined_to_dict,
    default_sweep_values,
    solve_combined,
)
from core.cycles.combined_procedure import combined_sections
from core.cycles.hrsg import hrsg_tq_profile
from core.cycles.rankine import rankine_labeled_states
from core.diagrams import DiagramType
from core.export import dict_to_csv
from core.fluids import fluid_state_from_pair
from core.ideal_gas import AIR_DRY, AIR_TECHNICAL, GAS_NAMES, GAS_SPECIES
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.cycle_charts import (
    energy_sankey_figure,
    gas_turbine_ts_figure,
    render_rankine_diagram,
    tq_figure,
)
from ui.diagrams import diagram_type_selector
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.16.0"

_EXAMPLES = list(COMBINED_EXAMPLES)
# Opciones fijas: cambiarlas reiniciaría el widget.
_MODEL_FUEL = "Combustión de metano"
_MODEL_AIR = "Aire estándar (sin combustible)"
_AIRS = {"Seco (N₂, O₂, Ar, CO₂)": AIR_DRY, "Técnico (21 % O₂, 79 % N₂)": AIR_TECHNICAL}
_SIZE_AIR = "Caudal de aire"
_SIZE_POWER = "Potencia neta total"
_STEAM_SH = "Sobrecalentado"
_STEAM_SAT = "Saturado"
_DESIGN_PINCH = "Por pinch"
_DESIGN_STACK = "Por temperatura de chimenea"
_GAS_PLACES = {
    "1": "entrada al compresor",
    "2s": "salida isoentrópica del compresor",
    "2": "salida del compresor",
    "3": "entrada a la turbina",
    "4s": "salida isoentrópica de la turbina",
    "4": "escape (a la HRSG)",
}

# (parámetro, magnitud del eje, rótulo del eje)
_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind | None, str]] = {
    "Relación de presiones": ("pressure_ratio", None, "r_p"),
    "Temperatura de entrada a la turbina": ("T_turbine_in", "temperature", "TIT"),
    "Presión del vapor": ("p_steam", "pressure", "p vapor"),
}
_SWEEP_PINCH = "Pinch"
_SWEEP_STACK = "Temperatura de chimenea"
_SWEEP_NOTES: dict[str, str] = {
    "pressure_ratio": (
        "Con más relación de presiones la turbina de gas sola rinde más, pero sus gases salen "
        "más fríos y el ciclo de vapor aprovecha menos: el ciclo combinado tiene su óptimo en una "
        "relación de presiones más baja que la de la turbina de gas sola (Kehlhofer et al.). "
        "Cuando el escape se enfría, el vapor se sobrecalienta como mucho hasta 25 K por debajo "
        "de él."
    ),
    "T_turbine_in": (
        "Una TIT más alta mejora la turbina de gas y, sobre todo, calienta el escape: el ciclo "
        "de vapor recupera más. Por eso las turbinas modernas trabajan con 1400–1600 °C (con "
        "álabes refrigerados). El vapor se sobrecalienta como mucho hasta 25 K por debajo del "
        "escape."
    ),
    "p_steam": (
        "Con más presión el vapor vale más (mejor η del ciclo de vapor) pero la HRSG de una "
        "presión recupera menos calor: la chimenea queda más caliente. Las plantas grandes usan "
        "dos o tres presiones para tener las dos cosas."
    ),
    "pinch": (
        "Menos pinch: más vapor y más rendimiento, a cambio de una caldera más grande (más "
        "área). Se diseña con unos 5 a 15 K."
    ),
    "T_stack": (
        "Con la chimenea más fría se recupera más calor del escape y sube el rendimiento, hasta "
        "que el pinch se achica demasiado (o los gases cruzan la temperatura del agua)."
    ),
}


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner="Resolviendo el ciclo combinado…")
def _solve_cached(inputs: CombinedInputs) -> CombinedResult:
    return solve_combined(inputs)


@st.cache_data(show_spinner="Resolviendo la turbina de gas con TESPy…")
def _tespy_cached(inputs: CombinedInputs) -> BraytonTespy | str:
    try:
        return brayton_tespy(solve_combined(inputs).gas_turbine)
    except ValueError as exc:
        return str(exc)


@st.cache_data(show_spinner="Calculando el barrido…")
def _sweep_cached(
    inputs: CombinedInputs, parameter: SweepParameter
) -> list[tuple[float, float, float, float]]:
    values = default_sweep_values(inputs, parameter)
    return [
        (p.value_si, p.eta, p.eta_gas_turbine, p.eta_steam)
        for p in combined_sweep(inputs, parameter, values)
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
    if kind == "power":
        return _number(convert_from_si(value_si, kind, system))
    return format_value(convert_from_si(value_si, kind, system), sig)


def _fmt(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    return f"{_value(value_si, kind, system, sig)} {unit_label(kind, system)}"


def _default_stack_K(base: CombinedInputs) -> float:
    """Chimenea por defecto del diseño por temperatura de chimenea.

    La del ejemplo si la trae; si no, la que sale del diseño por pinch del
    ejemplo, redondeada hacia arriba a 5 °C (el pinch queda un poco mayor y
    no hay cruce de temperaturas).
    """
    if base.T_stack_K is not None:
        return base.T_stack_K
    T_stack = _solve_cached(base).hrsg.T_stack_K
    return 273.15 + 5.0 * math.ceil((T_stack - 273.15) / 5.0)


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
        key="cc_example",
        help="Podés cambiar cualquier dato y volver a calcular.",
    )
    return _EXAMPLES.index(label)


def _read_gas_turbine(key: str, base: BraytonInputs) -> BraytonInputs:
    st.markdown("#### Turbina de gas")
    model = st.radio(
        "Modelo",
        (_MODEL_FUEL, _MODEL_AIR),
        index=1 if base.air_standard else 0,
        horizontal=True,
        key=f"{key}_model",
        help=(
            "Con combustión, el metano se quema con el aire y por la turbina pasan los gases. "
            "Aire estándar (Cengel §9-3): la cámara es un intercambiador que calienta el aire."
        ),
    )
    air_names = list(_AIRS)
    default_air = next(n for n, a in _AIRS.items() if a.y == base.air.y)
    air_name = st.radio(
        "Aire",
        air_names,
        index=air_names.index(default_air),
        horizontal=True,
        key=f"{key}_air",
        help=(
            "El aire seco (con argón) es el de las tablas de Cengel; el técnico, el del "
            "vademecum §16.1 para los cálculos de combustión."
        ),
    )
    left, right = st.columns(2)
    with left:
        T_amb = number_input_si(
            label="Aire ambiente T₁",
            kind="temperature",
            default_si=base.T_amb_K,
            key=f"{key}_T1",
            format="%.4g",
            help="Condiciones ISO: 15 °C y 1,013 bar.",
        )
        rp = st.number_input(
            "Relación de presiones r_p",
            min_value=1.0,
            max_value=60.0,
            value=float(base.pressure_ratio),
            step=0.5,
            format="%.4g",
            key=f"{key}_rp",
        )
        TIT = number_input_si(
            label="Entrada a la turbina T₃ (TIT)",
            kind="temperature",
            default_si=base.T_turbine_in_K,
            key=f"{key}_TIT",
            format="%.5g",
            help="Las turbinas modernas trabajan con 1400 a 1600 °C (álabes refrigerados).",
        )
    with right:
        p_amb = number_input_si(
            label="Presión ambiente p₁",
            kind="pressure",
            default_si=base.p_amb_Pa,
            key=f"{key}_p1",
            format="%.5g",
            min_value_si=0.0,
        )
        eta_c = st.number_input(
            "η_C del compresor",
            min_value=0.3,
            max_value=1.0,
            value=float(base.eta_compressor),
            step=0.01,
            format="%.3f",
            key=f"{key}_etac",
            help="1 = isoentrópico (vademecum §10.4).",
        )
        eta_t = st.number_input(
            "η_T de la turbina",
            min_value=0.3,
            max_value=1.0,
            value=float(base.eta_turbine),
            step=0.01,
            format="%.3f",
            key=f"{key}_etat",
        )
    dp = st.number_input(
        "Caída de presión en la cámara [%]",
        min_value=0.0,
        max_value=19.9,
        value=float(base.dp_combustor * 100.0),
        step=0.5,
        format="%.3g",
        key=f"{key}_dp",
        help="p₃ = p₂·(1 − Δp). Típico: 3 a 5 %.",
    )
    return replace(
        base,
        pressure_ratio=float(rp),
        T_turbine_in_K=TIT,
        eta_compressor=float(eta_c),
        eta_turbine=float(eta_t),
        T_amb_K=T_amb,
        p_amb_Pa=p_amb,
        dp_combustor=float(dp) / 100.0,
        air=_AIRS[air_name],
        fuel=None if model == _MODEL_AIR else METHANE,
    )


def _read_hrsg(key: str, base: CombinedInputs) -> dict[str, float | None]:
    st.markdown("#### Caldera de recuperación (HRSG)")
    system = get_current_system()
    left, right = st.columns(2)
    with left:
        p_steam = number_input_si(
            label="Presión del vapor",
            kind="pressure",
            default_si=base.p_steam_Pa,
            key=f"{key}_ps",
            format="%.5g",
            min_value_si=0.0,
            help="La del domo de la HRSG y la de entrada a la turbina de vapor.",
        )
        T_sat = _T_sat(p_steam)
        st.caption(
            "Sin saturación a esa presión."
            if T_sat is None
            else f"T_sat = {_fmt(T_sat, 'temperature', system)}"
        )
    with right:
        steam = st.radio(
            "Vapor",
            (_STEAM_SH, _STEAM_SAT),
            index=0 if base.T_steam_K is not None else 1,
            horizontal=True,
            key=f"{key}_steam",
        )
        T_steam = None
        if steam == _STEAM_SH:
            T_steam = number_input_si(
                label="Vapor sobrecalentado",
                kind="temperature",
                default_si=base.T_steam_K or 813.15,
                key=f"{key}_Ts",
                format="%.4g",
                help="Unos 25 K por debajo del escape de la turbina de gas, hasta ~565 °C.",
            )
    design = st.radio(
        "Diseño de la HRSG",
        (_DESIGN_PINCH, _DESIGN_STACK),
        index=1 if base.T_stack_K is not None else 0,
        horizontal=True,
        key=f"{key}_design",
        help=(
            "Por pinch (Kehlhofer): la menor diferencia de temperatura fija el caudal de vapor. "
            "Por temperatura de chimenea (Cengel §10-9): el pinch es un resultado."
        ),
    )
    left, right = st.columns(2)
    pinch, T_stack = base.pinch_K, None
    with left:
        if design == _DESIGN_PINCH:
            pinch = number_input_si(
                label="Pinch",
                kind="temperature_difference",
                default_si=base.pinch_K,
                key=f"{key}_pinch",
                format="%.3g",
                min_value_si=0.0,
            )
        else:
            T_stack = number_input_si(
                label="Chimenea",
                kind="temperature",
                default_si=_default_stack_K(base),
                key=f"{key}_Tstack",
                format="%.4g",
            )
    with right:
        approach = number_input_si(
            label="Approach",
            kind="temperature_difference",
            default_si=base.approach_K,
            key=f"{key}_approach",
            format="%.3g",
            min_value_si=0.0,
        )
    return {
        "p_steam_Pa": p_steam,
        "T_steam_K": T_steam,
        "pinch_K": pinch,
        "approach_K": approach,
        "T_stack_K": T_stack,
    }


def _read_steam(key: str, base: SteamCycle) -> SteamCycle:
    st.markdown("#### Ciclo de vapor")
    system = get_current_system()
    left, right = st.columns(2)
    with left:
        p_cond = number_input_si(
            label="Presión del condensador",
            kind="pressure",
            default_si=base.p_condenser_Pa,
            key=f"{key}_pc",
            format="%.4g",
            min_value_si=0.0,
        )
        T_cond = _T_sat(p_cond)
        if T_cond is not None:
            st.caption(f"T_sat = {_fmt(T_cond, 'temperature', system)}")
        eta_st = st.number_input(
            "η_T de la turbina de vapor",
            min_value=0.3,
            max_value=1.0,
            value=float(base.eta_turbine),
            step=0.01,
            format="%.3f",
            key=f"{key}_etast",
        )
    with right:
        eta_p = st.number_input(
            "η_B de la bomba",
            min_value=0.3,
            max_value=1.0,
            value=float(base.eta_pump),
            step=0.01,
            format="%.3f",
            key=f"{key}_etap",
        )
    deaerator = st.checkbox(
        "Desaireador (calentador abierto con una extracción)",
        value=base.deaerator_p_Pa is not None,
        key=f"{key}_da",
        help=(
            "Calienta el agua de alimentación (y le saca los gases disueltos) con vapor de la "
            "turbina: el agua llega a la HRSG a 105–120 °C, por encima del punto de rocío de los "
            "gases (Cengel §10-6)."
        ),
    )
    p_da = None
    if deaerator:
        p_da = number_input_si(
            label="Presión del desaireador",
            kind="pressure",
            default_si=base.deaerator_p_Pa or 1.5e5,
            key=f"{key}_pda",
            format="%.4g",
            min_value_si=0.0,
        )
    return SteamCycle(p_cond, float(eta_st), float(eta_p), p_da)


def _read_inputs(ex: int) -> CombinedInputs:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    base = COMBINED_EXAMPLES[_EXAMPLES[ex]]
    key = f"cc_{ex}"
    gt = _read_gas_turbine(key, base.gas_turbine)
    hrsg = _read_hrsg(key, base)
    steam = _read_steam(key, base.steam)
    st.markdown("#### Tamaño")
    size = st.radio(
        "Dato",
        (_SIZE_AIR, _SIZE_POWER),
        index=0,
        horizontal=True,
        key=f"{key}_size",
        help="Todo escala con el caudal: los rendimientos no cambian.",
    )
    W_net = None
    if size == _SIZE_AIR:
        m_air = number_input_si(
            label="Caudal de aire",
            kind="mass_flow",
            default_si=base.gas_turbine.m_air_kg_s,
            key=f"{key}_mair",
            format="%.5g",
            min_value_si=0.0,
        )
        gt = replace(gt, m_air_kg_s=m_air)
    else:
        W_net = number_input_si(
            label="Potencia neta del ciclo combinado",
            kind="power",
            default_si=300e6,
            key=f"{key}_W",
            format="%.6g",
            min_value_si=0.0,
        )
    return CombinedInputs(
        gas_turbine=gt,
        steam=steam,
        W_net_W=W_net,
        **hrsg,  # type: ignore[arg-type]
    )


def _render_error(exc: ValueError) -> None:
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico: {detail.strip()}")


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


def _render_metrics(result: CombinedResult, system: UnitSystem) -> None:
    P = unit_label("power", system)
    T_unit = unit_label("temperature", system)
    gt = result.gas_turbine
    row1 = st.columns(3)
    row1[0].metric("η ciclo combinado [%]", f"{result.eta_th * 100:.2f}")
    row1[1].metric(f"Ẇ neto [{P}]", _value(result.W_net_W, "power", system))
    heat = "Q̇ entrada" if gt.inputs.air_standard else "Q̇ combustible"
    row1[2].metric(f"{heat} [{P}]", _value(result.Q_fuel_W, "power", system))
    row2 = st.columns(3)
    row2[0].metric("η turbina de gas [%]", f"{result.eta_gas_turbine * 100:.2f}")
    row2[1].metric(f"Ẇ turbina de gas [{P}]", _value(result.W_gas_turbine_W, "power", system))
    row2[2].metric(f"T escape [{T_unit}]", _value(gt.T_exhaust_K, "temperature", system, 4))
    row3 = st.columns(3)
    row3[0].metric("η ciclo de vapor [%]", f"{result.eta_steam * 100:.2f}")
    row3[1].metric(f"Ẇ ciclo de vapor [{P}]", _value(result.W_steam_turbine_W, "power", system))
    row3[2].metric(
        f"ṁ vapor [{unit_label('mass_flow', system)}]",
        _value(result.hrsg.m_steam_kg_s, "mass_flow", system),
    )
    m_unit = unit_label("mass_flow", system)
    parts = [f"ṁ aire = {_value(gt.m_air_kg_s, 'mass_flow', system)} {m_unit}"]
    if not gt.inputs.air_standard:
        parts += [
            f"ṁ combustible = {_value(gt.m_fuel_kg_s, 'mass_flow', system)} {m_unit}",
            f"λ = {format_value(gt.excess_air, 4)}",
        ]
    parts += [
        f"ṁ_v/ṁ_g = {format_value(result.steam_gas_ratio, 4)}",
        f"η_HRSG = {result.eta_hrsg * 100:.1f} %",
        f"chimenea {_fmt(result.hrsg.T_stack_K, 'temperature', system, 4)}",
        f"heat rate {_number(result.heat_rate_kJ_per_kWh)} kJ/kWh",
    ]
    st.caption(" · ".join(parts) + ".")
    for note in combined_notes(result, system):
        st.info(note)


def _render_energy(result: CombinedResult, system: UnitSystem) -> None:
    st.markdown("#### ¿A dónde va la energía?")
    try:
        st.plotly_chart(energy_sankey_figure(result, system), width="stretch", key="cc_sankey")
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    st.caption(
        "De cada 100 unidades de calor del combustible, la turbina de gas convierte en trabajo "
        "una parte; el resto sale con el escape. La HRSG le pasa la mayor parte al vapor, que "
        "en el ciclo de vapor da más trabajo y le cede el resto al condensador; lo que no "
        "recupera la HRSG se va por la chimenea."
    )


def _gas_table(result: CombinedResult, system: UnitSystem) -> pd.DataFrame:
    T_unit = unit_label("temperature", system)
    p_unit = unit_label("pressure", system)
    e_unit = unit_label("specific_enthalpy", system)
    s_unit = unit_label("specific_entropy", system)
    rows = [
        {
            "Estado": st_.label,
            "Dónde": _GAS_PLACES[st_.label],
            f"T [{T_unit}]": _value(st_.T_K, "temperature", system),
            f"p [{p_unit}]": _value(st_.P_Pa, "pressure", system),
            f"h [{e_unit}]": _value(st_.h_J_per_kg, "specific_enthalpy", system),
            f"s [{s_unit}]": _value(st_.s_J_per_kg_K, "specific_entropy", system),
            "Medio": st_.medium,
        }
        for st_ in result.gas_turbine.states
    ]
    return pd.DataFrame(rows)


def _render_gas_turbine(result: CombinedResult, system: UnitSystem) -> None:
    st.markdown("#### Turbina de gas")
    try:
        st.plotly_chart(
            gas_turbine_ts_figure(result.gas_turbine, system, result.hrsg.T_stack_K),
            width="stretch",
            key="cc_gt_ts",
        )
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    mixing = (
        ""
        if result.gas_turbine.inputs.air_standard
        else " El tramo gris horizontal en 2 es la cámara: a la misma T y p, los gases (con el "
        "CO₂ y el H₂O de la combustión) tienen otra entropía que el aire."
    )
    st.caption(
        "Entropías absolutas (con las de NIST-JANAF a 25 °C y 1 bar), así el aire (1 → 2) y los "
        "gases (3 → 4) quedan en la misma escala. La compresión y la expansión reales se unen "
        "con una recta punteada solo como referencia; las verdes son las isoentrópicas con las "
        "que se comparan los rendimientos." + mixing + " Los gases se enfrían en la HRSG a la "
        "presión del escape, de 4 hasta la chimenea."
    )
    st.dataframe(_gas_table(result, system), hide_index=True, width="stretch")
    gas = result.gas_turbine.gas
    y = gas.mole_fractions
    st.caption(
        "h medida desde 25 °C (la referencia del poder calorífico). Gases de escape: "
        + " · ".join(f"{GAS_NAMES[s]} {y[s] * 100:.3g} %" for s in GAS_SPECIES if y[s] > 0)
        + " (fracción molar)."
    )


def _render_hrsg(result: CombinedResult, system: UnitSystem) -> None:
    st.markdown("#### Caldera de recuperación")
    hrsg = result.hrsg
    try:
        st.plotly_chart(
            tq_figure(hrsg, hrsg_tq_profile(hrsg), system), width="stretch", key="cc_tq"
        )
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    T_unit = unit_label("temperature", system)
    rows = [
        {
            "Sección": s.name,
            f"Q̇ [{unit_label('power', system)}]": _value(s.Q_W, "power", system),
            f"Gases [{T_unit}]": (
                f"{_value(s.T_gas_in_K, 'temperature', system, 4)} → "
                f"{_value(s.T_gas_out_K, 'temperature', system, 4)}"
            ),
            f"Agua [{T_unit}]": (
                f"{_value(s.T_water_in_K, 'temperature', system, 4)} → "
                f"{_value(s.T_water_out_K, 'temperature', system, 4)}"
            ),
        }
        for s in hrsg.sections
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    how = (
        "la chimenea es dato y el pinch sale"
        if hrsg.inputs.by_stack
        else "el pinch fija el caudal de vapor y la chimenea sale"
    )
    st.caption(
        f"Pinch {_fmt(hrsg.pinch_K, 'temperature_difference', system, 3)} y chimenea "
        f"{_fmt(hrsg.T_stack_K, 'temperature', system, 4)}: {how} del balance. El agua de "
        "alimentación es la que entrega la bomba del ciclo de vapor."
    )


def _render_steam(result: CombinedResult, system: UnitSystem) -> None:
    st.markdown("#### Ciclo de vapor")
    with st.expander("📈 Diagrama del ciclo de vapor", expanded=False):
        diagram_type: DiagramType = diagram_type_selector(key="cc_diagram_type", default="Ts")
        try:
            render_rankine_diagram(result.steam, system, diagram_type, chart_key="cc_steam_chart")
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
    frame = pd.DataFrame(states_table(rankine_labeled_states(result.steam), system))
    frame = frame.drop(columns=[c for c in frame.columns if c == "Fluido" or c.startswith("u [")])
    for column in frame.columns:
        if column not in ("Estado", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(
        "La caldera del ciclo de vapor es la HRSG: el agua entra con el estado que entrega la "
        "bomba y sale como el vapor de la HRSG."
    )


def _render_tespy(inputs: CombinedInputs, result: CombinedResult, system: UnitSystem) -> None:
    with st.expander("⚙️ Control con TESPy (turbina de gas)", expanded=False):
        st.markdown(
            "La misma turbina de gas resuelta con [TESPy](https://tespy.readthedocs.io): "
            "*Compressor*, *DiabaticCombustionChamber* y *Turbine*, como su tutorial de turbina de "
            "gas (con aire estándar, la cámara es un *SimpleHeatExchanger*). TESPy evalúa cada "
            "componente de la mezcla a su presión parcial con su ecuación de estado real: por "
            "eso difiere unas décimas de % del cálculo con gases ideales (más con relaciones de "
            "presión muy altas)."
        )
        control = _tespy_cached(inputs)
        if isinstance(control, str):
            st.warning(f"TESPy no pudo resolver esta turbina de gas: {control}")
            return
        gt = result.gas_turbine
        T_unit = unit_label("temperature", system)
        e_unit = unit_label("specific_enthalpy", system)
        rows = [
            (
                f"T₂ [{T_unit}]",
                _value(gt.state("2").T_K, "temperature", system),
                _value(control.T2_K, "temperature", system),
            ),
            (
                f"T₄ [{T_unit}]",
                _value(gt.T_exhaust_K, "temperature", system),
                _value(control.T4_K, "temperature", system),
            ),
            (
                f"w_neto [{e_unit}]",
                _value(gt.w_net_J_per_kg, "specific_enthalpy", system),
                _value(control.w_net_J_per_kg, "specific_enthalpy", system),
            ),
            ("η turbina de gas [%]", f"{gt.eta_th * 100:.3f}", f"{control.eta_th * 100:.3f}"),
        ]
        if control.excess_air is not None and gt.excess_air is not None:
            rows.append(("λ", format_value(gt.excess_air, 5), format_value(control.excess_air, 5)))
            rows.append(
                (
                    "f (kg combustible/kg aire)",
                    format_value(gt.fuel_air_ratio, 5),
                    format_value(control.fuel_air_ratio, 5),
                )
            )
        st.dataframe(
            pd.DataFrame(rows, columns=["Magnitud", "Gases ideales", "TESPy"]),
            hide_index=True,
            width="stretch",
        )


def _render_procedure(result: CombinedResult, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cómo se resuelve a mano, parte por parte: la turbina de gas con las tablas de gas "
            "ideal (Cengel §9-8), la HRSG con sus balances (Kehlhofer et al.) y el ciclo de vapor "
            "con las tablas de vapor (Cengel §10-2 a §10-6); al final, el acople y el rendimiento "
            "del ciclo combinado (Cengel §10-9)."
        )
        n = 0
        for title, steps in combined_sections(result, system):
            st.markdown(f"##### {title}")
            for step in steps:
                n += 1
                st.markdown(f"**{n}. {step.title}**")
                if step.text:
                    st.markdown(step.text)
                for tex in step.latex:
                    st.latex(tex)


def _render_export(result: CombinedResult, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    data = combined_to_dict(result, system)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="ciclo_combinado.csv",
        mime="text/csv",
        key="cc_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="ciclo_combinado.json",
        mime="application/json",
        key="cc_download_json",
    )


def _render_sweeps(inputs: CombinedInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo mejorar el ciclo combinado?", expanded=False):
        st.markdown(
            "Elegí qué variar (el resto queda como en tu ciclo) para ver cómo cambian los "
            "rendimientos del ciclo combinado, de la turbina de gas y del ciclo de vapor:"
        )
        sweeps = dict(_SWEEPS)
        if inputs.T_stack_K is None:
            sweeps[_SWEEP_PINCH] = ("pinch", "temperature_difference", "ΔT pinch")
        else:
            sweeps[_SWEEP_STACK] = ("T_stack", "temperature", "T chimenea")
        choice = st.selectbox("Variable", list(sweeps), key="cc_sweep_param")
        parameter, kind, axis = sweeps[choice]
        if st.button("Calcular barrido", key="cc_sweep_btn"):
            st.session_state["cc_sweep"] = (inputs, parameter)
        if st.session_state.get("cc_sweep") != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve el ciclo combinado varias veces).")
            return
        points = _sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        if kind is None:
            xs = [p[0] for p in points]
            x_title = f"{axis} [-]"
        else:
            xs = [convert_from_si(p[0], kind, system) for p in points]
            x_title = f"{axis} [{unit_label(kind, system)}]"
        fig = go.Figure()
        for k, name in ((1, "ciclo combinado"), (2, "turbina de gas"), (3, "ciclo de vapor")):
            fig.add_trace(
                go.Scatter(x=xs, y=[p[k] * 100 for p in points], mode="lines+markers", name=name)
            )
        fig.update_layout(
            height=320,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            title="Rendimientos",
            xaxis_title=x_title,
            yaxis_title="η [%]",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            separators=". ",
        )
        if parameter in ("pressure_ratio", "p_steam"):
            fig.update_xaxes(type="log")
        st.plotly_chart(fig, width="stretch", key="cc_sweep_eta")
        skipped = len(default_sweep_values(inputs, parameter)) - len(points)
        if skipped:
            st.caption(
                f"{skipped} valor(es) del barrido no tienen sentido físico (cruce de "
                "temperaturas en la HRSG, vapor más caliente que los gases…) y se omiten."
            )
        st.caption(_SWEEP_NOTES[parameter])


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "§3.3 *Sistema abierto, régimen permanente*, **§5 *Mezclas de gases ideales***, "
            "§4.8 *Polinomios NASA*, **§10.4 *Rendimientos isoentrópicos*** y **§16 "
            "*Combustión*** (§16.1 aire, §16.2 exceso de aire). En el libro: Çengel & Boles §9-8 "
            "(ciclo Brayton), §9-3 (aire estándar) y §10-9 (ciclos combinados); Kehlhofer, "
            "Hannemann, Stirnimann y Rukes, *Combined-Cycle Gas & Steam Turbine Power Plants* "
            "(3.ª ed., PennWell, 2009). El poder calorífico, de ISO 6976:2016 a 25 °C."
        )
        st.markdown(
            "**Compresor y turbina** con calores específicos variables: la isoentrópica sale de "
            "la función s° = s°(T) (la columna s° de la tabla A-17 de Cengel para el aire; "
            "s°₁ = s°(T₁)) y el rendimiento isoentrópico da el estado real:"
        )
        st.latex(r"s^{\circ}_{2s} = s^{\circ}_1 + R\,\ln\frac{p_2}{p_1}")
        st.latex(r"\eta_C = \frac{h_{2s} - h_1}{h_2 - h_1}")
        st.latex(r"\eta_T = \frac{h_3 - h_4}{h_3 - h_{4s}}")
        st.markdown(
            "**Cámara de combustión** adiabática, por kilogramo de aire, con las entalpías "
            "medidas desde 25 °C (la referencia del PCI); f es la relación combustible/aire y "
            "λ = f_t/f el exceso de aire (§16.2):"
        )
        st.latex(r"h_a(T_2) + f\,\mathrm{PCI} = (1 + f)\,h_g(T_3)")
        st.latex(r"\mathrm{PCI}_m = \mathrm{PCS}_m - \tfrac{b}{2}\,L_0")
        st.markdown("**Turbina de gas**, por kilogramo de aire:")
        st.latex(r"w_{\mathrm{neto}} = (1 + f)\,(h_3 - h_4) - (h_2 - h_1)")
        st.latex(r"\eta_{TG} = \frac{w_{\mathrm{neto}}}{f\,\mathrm{PCI}}")
        st.markdown(
            "**HRSG** (ver la página HRSG): con el pinch, el tramo entre la entrada de los gases "
            "y la salida del evaporador da el caudal de vapor; con la temperatura de chimenea, "
            "el balance de toda la caldera:"
        )
        st.latex(
            r"\begin{aligned}\dot{m}_v\,(h_5 - h_1) &= \dot{m}_g\,\bigl[h_g(T_4) \\ &\quad"
            r" - h_g(T_{\mathrm{chim}})\bigr]\end{aligned}"
        )
        st.markdown(
            "**Ciclo combinado**: el calor entra una sola vez, en la turbina de gas. Con "
            "η_HRSG = Q̇_HRSG / (Q̇_comb − Ẇ_TG), la parte del calor del escape que recupera la "
            "caldera (Kehlhofer et al.):"
        )
        st.latex(r"\eta_{CC} = \frac{\dot{W}_{TG} + \dot{W}_{TV}}{\dot{Q}_{\mathrm{comb}}}")
        st.latex(r"\eta_{CC} = \eta_{TG} + \eta_{\mathrm{HRSG}}\,\eta_{TV}\,(1 - \eta_{TG})")
        st.latex(
            r"\begin{aligned}\dot{Q}_{\mathrm{comb}} &= \dot{W}_{TG} + \dot{W}_{TV} \\ &\quad"
            r" + \dot{Q}_{\mathrm{cond}} + \dot{Q}_{\mathrm{chim}}\end{aligned}"
        )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Ciclo combinado", page_icon="⚡", layout="centered")

st.subheader(SUBJECT)
st.title("⚡ Ciclo combinado gas–vapor")
st.markdown(
    "Una **turbina de gas** quema el combustible; sus gases de escape, todavía a 500–650 °C, "
    "producen vapor en una **caldera de recuperación** (HRSG) de una presión, y el vapor mueve "
    "un **ciclo de Rankine**. Con el mismo combustible se obtiene bastante más trabajo que con "
    "la turbina de gas sola."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Ciclo combinado")
render_units_selector()
system = get_current_system()

ex = _example_index()
inputs = _read_inputs(ex)

# Se calcula al elegir un ejemplo y cada vez que se toca el botón.
if st.session_state.get("cc_example_prev") != ex:
    st.session_state["cc_example_prev"] = ex
    st.session_state["cc_inputs"] = inputs
if st.button("Calcular ciclo combinado", key="cc_btn", type="primary"):
    st.session_state["cc_inputs"] = inputs

computed: CombinedInputs | None = st.session_state.get("cc_inputs")
if computed is not None:
    if computed != inputs:
        st.caption("Cambiaste datos: tocá «Calcular ciclo combinado» para actualizar el resultado.")
    try:
        result = _solve_cached(computed)
    except ValueError as exc:
        _render_error(exc)
    else:
        st.markdown("### Resultado")
        note = COMBINED_EXAMPLE_NOTES.get(_EXAMPLES[ex])
        if note:
            st.caption(f"📘 Sobre este ejemplo: {note}")
        _render_metrics(result, system)
        _render_energy(result, system)
        _render_gas_turbine(result, system)
        _render_hrsg(result, system)
        _render_steam(result, system)
        _render_procedure(result, system)
        _render_tespy(computed, result, system)
        _render_export(result, system)
        _render_sweeps(computed, system)
