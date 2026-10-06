"""Página 6 — Refrigeración por compresión de vapor con TESPy (Fase 3.2).

Ciclo simple (ideal o real), de dos etapas con cámara de evaporación
instantánea («economizador») y en cascada (con uno o dos refrigerantes),
usado como refrigerador o como bomba de calor. El cálculo lo hace
:mod:`core.cycles.refrigeration` con una red de TESPy; la página muestra el
COP, la tabla de estados, el ciclo sobre el diagrama log p–h (u otro), el
procedimiento "como con las tablas" (Cengel cap. 11), la exergía destruida en
cada componente si se dan las temperaturas de las fuentes, la exportación y
barridos para ver cómo cambia el COP.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.refrigeration import (
    REFRIGERATION_EXAMPLE_NOTES,
    REFRIGERATION_EXAMPLES,
    REFRIGERATION_EXAMPLES_BY_TEMPERATURE,
    REFRIGERATION_FLUIDS,
    Cascade,
    RefrigerationInputs,
    RefrigerationLayout,
    RefrigerationResult,
    Reservoirs,
    SweepParameter,
    condensation_pressure,
    condensation_temperature,
    default_flash_pressure,
    default_sweep_values,
    evaporation_pressure,
    evaporation_temperature,
    refrigeration_diagram_groups,
    refrigeration_labeled_states,
    refrigeration_layout,
    refrigeration_notes,
    refrigeration_steps,
    refrigeration_sweep,
    refrigeration_to_dict,
    solve_refrigeration,
    suggested_refrigeration_inputs,
)
from core.diagrams import (
    DiagramSpec,
    DiagramType,
    ProcessOverlay,
    isentropic_process,
    segments_overlays,
)
from core.export import dict_to_csv
from core.fluids import FLUID_NAMES_ES, fluid_limits
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.diagrams import DiagramPoint, diagram_type_selector, get_diagram, render_diagram_plotly
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.14.0"

_EXAMPLES = list(REFRIGERATION_EXAMPLES)
# Opciones fijas (sin números de estado): cambiarlas reiniciaría el widget.
_USES = {"Refrigerador": "refrigerator", "Bomba de calor": "heat_pump"}
_CYCLES = {
    "Simple": "simple",
    "Dos etapas con cámara": "flash",
    "Cascada": "cascade",
}
_LEVEL_P = "Presión"
_LEVEL_T = "Temperatura de saturación"
_FLOW_MASS = "Caudal másico"
_FLOW_CAPACITY = "Capacidad"
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# (parámetro, magnitud del eje, rótulo del eje)
_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind | None, str]] = {
    "Temperatura de evaporación": ("T_evap", "temperature", "T evaporación"),
    "Temperatura de condensación": ("T_cond", "temperature", "T condensación"),
    "Rendimiento del compresor": ("eta_compressor", None, "η_C"),
    "Subenfriamiento": ("subcooling", "temperature_difference", "ΔT subenfriamiento"),
    "Sobrecalentamiento": ("superheat", "temperature_difference", "ΔT sobrecalentamiento"),
}
_FLASH_SWEEP = "Presión de la cámara"
_CASCADE_SWEEP = "Temperatura intermedia de la cascada"


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses sin objetos de TESPy)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner="Resolviendo el ciclo con TESPy…")
def _solve_cached(inputs: RefrigerationInputs) -> RefrigerationResult:
    return solve_refrigeration(inputs)


@st.cache_data(show_spinner="Calculando el barrido…")
def _sweep_cached(
    inputs: RefrigerationInputs, parameter: SweepParameter
) -> list[tuple[float, float, float, float]]:
    values = default_sweep_values(inputs, parameter)
    return [
        (p.value_si, p.COP, p.COP_carnot, p.T_discharge_K)
        for p in refrigeration_sweep(inputs, parameter, values)
    ]


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _example_index() -> int:
    label = st.selectbox(
        "Ejemplo precargado",
        _EXAMPLES,
        key="rf_example",
        help=(
            "Ejemplos de Çengel & Boles, *Termodinámica* (8.ª ed.), cap. 11, y otros con "
            "refrigerantes naturales y bomba de calor. Podés cambiar cualquier dato y volver a "
            "calcular."
        ),
    )
    return _EXAMPLES.index(label)


def _sub(index: int) -> str:
    """Número de estado (índice + 1) como subíndice: 0 → ₁."""
    return str(index + 1).translate(_SUBSCRIPTS)


def _base_inputs(ex: int, fluid: str) -> RefrigerationInputs:
    """Datos del ejemplo o, si se eligió otro refrigerante, los sugeridos para él."""
    example = REFRIGERATION_EXAMPLES[_EXAMPLES[ex]]
    return example if fluid == example.fluid else suggested_refrigeration_inputs(fluid)


def _by_temperature(ex: int, fluid: str) -> bool:
    """Los ejemplos propios (y los datos sugeridos) se plantean con temperaturas."""
    name = _EXAMPLES[ex]
    return name in REFRIGERATION_EXAMPLES_BY_TEMPERATURE or (
        fluid != REFRIGERATION_EXAMPLES[name].fluid
    )


def _level(
    *,
    mode: str,
    label: str,
    fluid: str,
    phase: str,
    default_p: float,
    key: str,
    help: str | None = None,  # noqa: A002 — como st.number_input
) -> float | None:
    """Un nivel de presión, dado por presión o por temperatura de saturación (SI).

    ``phase``: ``"g"`` (evapora: punto de rocío) o ``"f"`` (condensa: burbuja).
    Devuelve ``None`` (y muestra el error) si la temperatura no tiene saturación.
    """
    to_T = evaporation_temperature if phase == "g" else condensation_temperature
    to_p = evaporation_pressure if phase == "g" else condensation_pressure
    if mode == _LEVEL_P:
        p = number_input_si(
            label=f"Presión {label}",
            kind="pressure",
            default_si=default_p,
            key=f"{key}_p",
            format="%.5g",
            min_value_si=0.0,
            help=help,
        )
        try:
            T = to_T(fluid, p)
        except ValueError:
            st.caption("Sin saturación a esa presión (revisá el dato).")
        else:
            st.caption(f"T de saturación: {_fmt(T, 'temperature')}")
        return p
    T = number_input_si(
        label=f"Temperatura {label}",
        kind="temperature",
        default_si=round(to_T(fluid, default_p) - 273.15, 1) + 273.15,
        key=f"{key}_T",
        format="%.4g",
        help=help,
    )
    try:
        p = to_p(fluid, T)
    except ValueError as exc:
        st.error(f"No hay saturación a esa temperatura: {exc}", icon="🚫")
        return None
    st.caption(f"p de saturación: {_fmt(p, 'pressure')}")
    return p


def _fmt(value_si: float, kind: QuantityKind) -> str:
    system = get_current_system()
    return f"{format_value(convert_from_si(value_si, kind, system), 5)} {unit_label(kind, system)}"


def _default_cascade(base: RefrigerationInputs, fluid: str, fluid_low: str) -> Cascade:
    """La del ejemplo o, si no, el intercambiador a la temperatura media (ΔT = 0)."""
    if base.cascade is not None and base.fluid_low == fluid_low and base.fluid == fluid:
        return base.cascade
    T_low = evaporation_temperature(base.fluid_low, base.p_evap_Pa)
    T_high = condensation_temperature(fluid, base.p_cond_Pa)
    T_mid = min(0.5 * (T_low + T_high), fluid_limits(fluid_low).T_crit_K - 5.0)
    return Cascade(
        condensation_pressure(fluid_low, T_mid),
        evaporation_pressure(fluid, T_mid),
        fluid_low=None if fluid_low == fluid else fluid_low,
    )


def _read_losses(
    key: str, base: RefrigerationInputs, p_evap: float, p_cond: float
) -> tuple[float, float, float, float]:
    """Bloque del ciclo real (Cengel §11-4): ΔT_sob, ΔT_sub, Δp_evap y Δp_cond (SI)."""
    if not st.checkbox(
        "Ciclo real: sobrecalentamiento, subenfriamiento y caídas de presión",
        value=base.has_losses,
        key=f"{key}_real",
        help=(
            "Cengel §11-4: el vapor sale del evaporador algo sobrecalentado (así no llegan gotas "
            "al compresor), el líquido sale del condensador algo subenfriado y la fricción baja "
            "la presión en los dos intercambiadores."
        ),
    ):
        return 0.0, 0.0, 0.0, 0.0
    real = base.has_losses
    left, right = st.columns(2)
    with left:
        superheat = number_input_si(
            label="Sobrecalentamiento",
            kind="temperature_difference",
            default_si=base.superheat_K if real else 5.0,
            key=f"{key}_sh",
            format="%.3g",
            min_value_si=0.0,
            help="Sobre la temperatura de saturación (rocío) a la salida del evaporador.",
        )
        dp_evap = number_input_si(
            label="Δp en el evaporador",
            kind="pressure",
            default_si=base.dp_evap_Pa if real else 0.05 * p_evap,
            key=f"{key}_dpe",
            format="%.4g",
            min_value_si=0.0,
        )
    with right:
        subcooling = number_input_si(
            label="Subenfriamiento",
            kind="temperature_difference",
            default_si=base.subcooling_K if real else 3.0,
            key=f"{key}_sc",
            format="%.3g",
            min_value_si=0.0,
            help="Bajo la temperatura de saturación (burbuja) a la salida del condensador.",
        )
        dp_cond = number_input_si(
            label="Δp en el condensador",
            kind="pressure",
            default_si=base.dp_cond_Pa if real else 0.05 * p_cond,
            key=f"{key}_dpc",
            format="%.4g",
            min_value_si=0.0,
        )
    return superheat, subcooling, dp_evap, dp_cond


def _read_reservoirs(
    key: str, base: RefrigerationInputs, heat_pump: bool, T1: float, T3: float
) -> Reservoirs | None:
    """Temperaturas de las fuentes para el segundo principio (Cengel §11-5)."""
    if not st.checkbox(
        "Segundo principio: exergía destruida en cada componente",
        value=base.reservoirs is not None,
        key=f"{key}_ex",
        help=(
            "Con las temperaturas de las fuentes se calcula cuánta exergía destruye cada "
            "componente, el trabajo mínimo y el rendimiento exergético (Cengel §11-5)."
        ),
    ):
        return None
    if base.reservoirs is not None:
        cold, hot = base.reservoirs.T_cold_K, base.reservoirs.T_hot_K
    else:
        cold, hot = round(T1 - 273.15 + 5.0) + 273.15, round(T3 - 273.15 - 5.0) + 273.15
    cold_label = (
        "Fuente fría: ambiente exterior" if heat_pump else "Fuente fría: espacio refrigerado"
    )
    hot_label = (
        "Fuente caliente: espacio calefaccionado" if heat_pump else "Fuente caliente: ambiente"
    )
    left, right = st.columns(2)
    with left:
        T_cold = number_input_si(
            label=f"{cold_label} T_C",
            kind="temperature",
            default_si=cold,
            key=f"{key}_TC",
            format="%.4g",
        )
    with right:
        T_hot = number_input_si(
            label=f"{hot_label} T_H",
            kind="temperature",
            default_si=hot,
            key=f"{key}_TH",
            format="%.4g",
        )
    st.caption(
        "El ambiente es el estado muerto T₀: "
        + ("la fuente fría (el aire exterior)." if heat_pump else "la fuente caliente.")
    )
    return Reservoirs(T_cold, T_hot)


def _read_inputs(ex: int) -> RefrigerationInputs | None:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    example = REFRIGERATION_EXAMPLES[_EXAMPLES[ex]]
    fluid = st.selectbox(
        "Refrigerante",
        REFRIGERATION_FLUIDS,
        index=REFRIGERATION_FLUIDS.index(example.fluid),
        format_func=lambda f: FLUID_NAMES_ES.get(f, f),
        key=f"rf_fluid_{ex}",
        help=(
            "En la cascada, el del ciclo de alta (el que condensa contra el ambiente). Cengel "
            "§11-6: conviene que evapore por encima de 1 atm y condense a una presión moderada."
        ),
    )
    base = _base_inputs(ex, fluid)
    key = f"rf_{ex}_{fluid}"
    use = _USES[
        st.radio(
            "Uso",
            tuple(_USES),
            index=1 if base.heat_pump else 0,
            horizontal=True,
            key=f"{key}_use",
        )
    ]
    cycle = _CYCLES[
        st.radio(
            "Ciclo",
            tuple(_CYCLES),
            index=list(_CYCLES.values()).index(base.cycle),
            horizontal=True,
            key=f"{key}_cycle",
            help=(
                "Cámara: el líquido se estrangula hasta una presión intermedia y el vapor que se "
                "forma va directo al compresor de alta. Cascada: dos ciclos unidos por un "
                "intercambiador (Cengel §11-8)."
            ),
        )
    ]
    fluid_low = fluid
    if cycle == "cascade":
        default_low = base.fluid_low if base.cascade is not None else fluid
        fluid_low = st.selectbox(
            "Refrigerante del ciclo de baja",
            REFRIGERATION_FLUIDS,
            index=REFRIGERATION_FLUIDS.index(default_low),
            format_func=lambda f: FLUID_NAMES_ES.get(f, f),
            key=f"{key}_low",
            help="Por ejemplo, CO₂ abajo (evapora a muy baja temperatura) y amoníaco arriba.",
        )
    layout: RefrigerationLayout = refrigeration_layout(cycle, fluid, fluid_low)  # type: ignore[arg-type]
    mode = st.radio(
        "Niveles de presión dados por",
        (_LEVEL_P, _LEVEL_T),
        index=1 if _by_temperature(ex, fluid) else 0,
        horizontal=True,
        key=f"{key}_mode",
        help=(
            "Con temperaturas, la presión de evaporación es la de rocío y la de condensación la "
            "de burbuja (iguales en un fluido puro; el R-410A tiene un pequeño deslizamiento)."
        ),
    )
    lkey = f"{key}_{mode[0]}"
    evap_out = layout.one("evaporator").outlet
    cond_in = layout.one("condenser").inlet
    base_p_evap = (
        base.p_evap_Pa
        if fluid_low == base.fluid_low
        else evaporation_pressure(
            fluid_low, evaporation_temperature(base.fluid_low, base.p_evap_Pa)
        )
    )
    left, right = st.columns(2)
    with left:
        p_evap = _level(
            mode=mode,
            label=f"del evaporador p{_sub(evap_out)}" if mode == _LEVEL_P else "de evaporación",
            fluid=fluid_low,
            phase="g",
            default_p=base_p_evap,
            key=f"{lkey}_ev_{fluid_low}",
            help="A la salida del evaporador: la presión de aspiración del compresor.",
        )
    with right:
        p_cond = _level(
            mode=mode,
            label=f"del condensador p{_sub(cond_in)}" if mode == _LEVEL_P else "de condensación",
            fluid=fluid,
            phase="f",
            default_p=base.p_cond_Pa,
            key=f"{lkey}_co",
            help="A la entrada del condensador: la presión de descarga del compresor.",
        )
    p_flash = None
    cascade = None
    if cycle == "flash":
        tank = layout.one("flash_tank").inlet
        default = base.p_flash_Pa or default_flash_pressure(
            p_evap or base.p_evap_Pa, p_cond or base.p_cond_Pa
        )
        p_flash = _level(
            mode=mode,
            label=f"de la cámara p{_sub(tank)}" if mode == _LEVEL_P else "de la cámara",
            fluid=fluid,
            phase="f",
            default_p=default,
            key=f"{lkey}_fl",
            help=(
                "Presión intermedia. Por defecto, la media geométrica de las otras dos; el "
                "barrido muestra dónde está el óptimo."
            ),
        )
        if p_flash is None:
            return None
    elif cycle == "cascade":
        hx = layout.one("cascade_hx")
        default = _default_cascade(base, fluid, fluid_low)
        st.markdown("**Intercambiador de la cascada**")
        cols = st.columns(2)
        with cols[0]:
            p_low = _level(
                mode=mode,
                label=f"condensación baja p{_sub(hx.inlets[0])}"
                if mode == _LEVEL_P
                else "de condensación (baja)",
                fluid=fluid_low,
                phase="f",
                default_p=default.p_cond_low_Pa,
                key=f"{lkey}_cl_{fluid_low}",
            )
        with cols[1]:
            p_high = _level(
                mode=mode,
                label=f"evaporación alta p{_sub(hx.outlets[1])}"
                if mode == _LEVEL_P
                else "de evaporación (alta)",
                fluid=fluid,
                phase="g",
                default_p=default.p_evap_high_Pa,
                key=f"{lkey}_eh_{fluid_low}",
            )
        st.caption(
            "El ciclo de baja tiene que condensar a una temperatura mayor o igual que la de "
            "evaporación del de alta (igual en el ejemplo 11-4 de Cengel; en la práctica, unos "
            "5 K más)."
        )
        if p_low is None or p_high is None:
            return None
        cascade = Cascade(p_low, p_high, fluid_low=None if fluid_low == fluid else fluid_low)
    if p_evap is None or p_cond is None:
        return None

    eta = st.number_input(
        "η_C del compresor" + ("" if cycle == "simple" else " (los dos)"),
        min_value=0.3,
        max_value=1.0,
        value=float(base.eta_compressor),
        step=0.01,
        format="%.3f",
        key=f"{key}_eta",
        help="1 = compresor ideal (isoentrópico), como en Cengel §11-3. Vademecum §10.4.",
    )
    st.markdown("**Ciclo real**")
    superheat, subcooling, dp_evap, dp_cond = _read_losses(key, base, p_evap, p_cond)

    st.markdown("**Tamaño**")
    flow = st.radio(
        "Dato",
        (_FLOW_MASS, _FLOW_CAPACITY),
        index=0 if base.m_dot_kg_s is not None else 1,
        horizontal=True,
        key=f"{key}_flow",
    )
    flows: dict[str, float | None]
    if flow == _FLOW_MASS:
        m_dot = number_input_si(
            label="Caudal por el condensador ṁ"
            + (" (ciclo de alta)" if cycle == "cascade" else ""),
            kind="mass_flow",
            default_si=base.m_dot_kg_s or 0.05,
            key=f"{key}_m",
            format="%.5g",
            min_value_si=0.0,
            help="La base de Cengel: el caudal que pasa por el condensador.",
        )
        flows = {"m_dot_kg_s": m_dot, "capacity_W": None}
    else:
        heat_pump = use == "heat_pump"
        capacity = number_input_si(
            label="Capacidad de calefacción Q̇_H" if heat_pump else "Capacidad de refrigeración Q̇_C",
            kind="power",
            default_si=base.capacity_W or 10.0e3,
            key=f"{key}_cap_{use}",
            format="%.5g",
            min_value_si=0.0,
        )
        flows = {"m_dot_kg_s": None, "capacity_W": capacity}

    st.markdown("**Segundo principio**")
    try:
        T1 = evaporation_temperature(fluid_low, p_evap) + superheat
        T3 = condensation_temperature(fluid, p_cond - dp_cond) - subcooling
    except ValueError:
        T1, T3 = 263.15, 303.15
    reservoirs = _read_reservoirs(key, base, use == "heat_pump", T1, T3)
    return RefrigerationInputs(
        p_evap,
        p_cond,
        float(eta),
        fluid,
        superheat,
        subcooling,
        dp_evap,
        dp_cond,
        application=use,  # type: ignore[arg-type]
        p_flash_Pa=p_flash,
        cascade=cascade,
        reservoirs=reservoirs,
        **flows,  # type: ignore[arg-type]
    )


def _render_error(exc: ValueError) -> None:
    """Mensaje para el alumno arriba; el detalle técnico, aparte y chico."""
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico: {detail.strip()}")


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


def _number(value: float) -> str:
    """Número para mostrar: sin notación exponencial en los valores grandes."""
    if abs(value) >= 1.0e4:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, 5)


def _power(value_si: float, system: UnitSystem) -> str:
    return _number(convert_from_si(value_si, "power", system))


def _energy(value_si: float, system: UnitSystem) -> str:
    return _number(convert_from_si(value_si, "specific_enthalpy", system))


def _render_metrics(result: RefrigerationResult, system: UnitSystem) -> None:
    p_unit = unit_label("power", system)
    heat_pump = result.inputs.heat_pump
    row1 = st.columns(3)
    row1[0].metric("COP_B" if heat_pump else "COP_R", f"{result.COP:.3f}")
    row1[1].metric(f"Q̇_C [{p_unit}]", _power(result.Q_cold_W, system))
    row1[2].metric(f"Ẇ [{p_unit}]", _power(result.W_W, system))
    row2 = st.columns(3)
    row2[0].metric("COP de Carnot", f"{result.COP_carnot:.3f}")
    row2[1].metric(f"Q̇_H [{p_unit}]", _power(result.Q_hot_W, system))
    T_unit = unit_label("temperature", system)
    row2[2].metric(
        f"T de descarga [{T_unit}]",
        format_value(convert_from_si(result.T_discharge_K, "temperature", system), 4),
    )
    e_unit = unit_label("specific_enthalpy", system)
    m_unit = unit_label("mass_flow", system)
    x = result.x_evaporator_in
    m_cond = _number(convert_from_si(result.m_cond_kg_s, "mass_flow", system))
    parts = [f"ṁ por el condensador: {m_cond} {m_unit}"]
    if result.cycle != "simple":
        m_evap = _number(convert_from_si(result.m_evap_kg_s, "mass_flow", system))
        parts.append(f"por el evaporador: {m_evap} {m_unit}")
    q_c = _energy(result.q_cold_J_per_kg, system)
    w = _energy(result.w_J_per_kg, system)
    q_h = _energy(result.q_hot_J_per_kg, system)
    parts.append(f"q_C = {q_c}, w = {w} y q_H = {q_h} {e_unit} por kg que pasa por el condensador")
    if not heat_pump:
        parts.append(f"{result.tons_of_refrigeration:.3g} toneladas de refrigeración")
    volume = format_value(
        convert_from_si(result.suction_volume_flow_m3_s, "volume_flow", system), 4
    )
    parts.append(f"V̇ aspirado: {volume} {unit_label('volume_flow', system)}")
    if x is not None:
        parts.append(f"título a la entrada del evaporador: {x:.3f}")
    ratios = ", ".join(f"{r:.3g}" for r in result.pressure_ratios)
    parts.append(f"relación de presiones: {ratios}")
    st.caption(
        " · ".join(parts) + ". COP de Carnot entre las temperaturas de evaporación y condensación "
        f"({format_value(convert_from_si(result.T_evap_K, 'temperature', system), 4)} y "
        f"{format_value(convert_from_si(result.T_cond_K, 'temperature', system), 4)} {T_unit})."
    )
    for note in refrigeration_notes(result, system):
        st.info(note)


def _render_exergy(result: RefrigerationResult, system: UnitSystem) -> None:
    """Exergía destruida por componente, con barras e interpretación (Cengel §11-5)."""
    exergy = result.exergy
    if exergy is None:
        return
    p_unit = unit_label("power", system)
    st.markdown("#### Segundo principio")
    cols = st.columns(3)
    cols[0].metric("Rendimiento exergético [%]", f"{exergy.eta_ex * 100:.1f}")
    cols[1].metric(f"Ẇ mínimo [{p_unit}]", _power(exergy.W_min_W, system))
    cols[2].metric("COP reversible", f"{result.COP_rev:.3f}")
    rows = [
        {
            "Componente": name,
            f"Ẋ destruida [{p_unit}]": _power(x, system),
            "% de Ẇ": f"{x / exergy.W_W * 100:.1f}",
        }
        for name, x in exergy.destroyed
    ]
    rows.append(
        {
            "Componente": "trabajo mínimo (exergía del efecto útil)",
            f"Ẋ destruida [{p_unit}]": _power(exergy.W_min_W, system),
            "% de Ẇ": f"{exergy.W_min_W / exergy.W_W * 100:.1f}",
        }
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    names = [name for name, _ in exergy.destroyed] + ["Ẇ mínimo"]
    values = [convert_from_si(x, "power", system) for _, x in exergy.destroyed]
    values.append(convert_from_si(exergy.W_min_W, "power", system))
    colors = ["#d62728"] * len(exergy.destroyed) + ["#2ca02c"]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3g}" for v in values],
        )
    )
    fig.update_layout(
        height=60 + 32 * len(names),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=f"En qué se va la potencia del compresor [{p_unit}]",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch", key="rf_exergy_bars")
    worst = max(exergy.destroyed, key=lambda item: item[1])
    st.caption(
        "Ẇ = Ẇ_mín + ΣẊ_dest: lo que el compresor consume por encima del trabajo de un ciclo "
        f"reversible entre las fuentes se destruye en los componentes. Acá, la mayor parte se "
        f"pierde en: **{worst[0]}**. Detalle y explicación de cada uno en el procedimiento."
    )


def _render_states_table(result: RefrigerationResult, system: UnitSystem) -> None:
    frame = pd.DataFrame(states_table(refrigeration_labeled_states(result), system))
    if len(set(result.layout.fluids)) == 1:
        frame = frame.drop(columns="Fluido")
    for column in frame.columns:
        if column not in ("Estado", "Región", "Fluido"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _render_tespy(result: RefrigerationResult, system: UnitSystem) -> None:
    with st.expander("⚙️ Cómo lo resuelve TESPy", expanded=False):
        st.markdown(
            "TESPy arma el ciclo como una **red**, igual que su tutorial de bomba de calor: "
            "cada componente (compresores, condensador, válvulas, evaporador) aporta sus "
            "ecuaciones —balance de masa y energía, rendimiento isoentrópico, pérdida de carga— "
            "y cada conexión lleva caudal, presión y entalpía. La cámara de evaporación "
            "instantánea es un separador de gotas (*DropletSeparator*: sale líquido y vapor "
            "saturados) seguido de un mezclador (*Merge*); el intercambiador de la cascada, un "
            "*HeatExchanger*. Con los datos resuelve el sistema con Newton. Estos son sus "
            "balances (positivo: entra al refrigerante; negativo: sale), que coinciden con los "
            "del procedimiento:"
        )
        unit = unit_label("power", system)
        frame = pd.DataFrame(
            [
                {"Componente": name, f"Potencia o calor [{unit}]": _power(value, system)}
                for name, value in result.tespy_balances
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        for name, value in result.tespy_internal:
            st.caption(
                f"Calor que el ciclo de baja le pasa al de alta en el {name}: "
                f"{_power(value, system)} {unit} (calor interno: no entra ni sale del sistema)."
            )


def _draw_group(
    result: RefrigerationResult,
    system: UnitSystem,
    diagram_type: DiagramType,
    fluid: str,
    indices: tuple[int, ...],
    segments: tuple[tuple[int, int], ...],
    chart_key: str,
) -> None:
    states = [s.to_state_point() for s in result.states]
    points = [DiagramPoint(state=states[i], label=str(i + 1), color="#1f77b4") for i in indices]
    diagram = get_diagram(fluid, system)
    spec = DiagramSpec(fluid=fluid, system=system)
    overlays: list[ProcessOverlay] = segments_overlays(
        diagram, spec, [(states[a], states[b]) for a, b in segments]
    )
    if result.inputs.eta_compressor < 1.0:
        for comp, state_s in zip(result.compressors, result.compressors_out_s, strict=True):
            if comp.inlet not in indices:
                continue
            start = result.states[comp.inlet]
            overlays.append(
                ProcessOverlay(
                    name=f"{comp.inlet + 1} → {comp.outlet + 1}s (isoentrópica de referencia)",
                    color="#2ca02c",
                    dash="dash",
                    coords_si=isentropic_process(
                        diagram,
                        spec,
                        s_J_per_kg_K=start.s_J_per_kg_K,
                        p_start_Pa=start.P_Pa,
                        p_end_Pa=state_s.P_Pa,
                    ),
                )
            )
            points.append(
                DiagramPoint(
                    state=state_s.to_state_point(), label=f"{comp.outlet + 1}s", color="#2ca02c"
                )
            )
    render_diagram_plotly(
        fluid=fluid,
        diagram_type=diagram_type,
        system=system,
        points=points,
        overlays=overlays,
        chart_key=chart_key,
        point_legend={"#1f77b4": "estados", "#2ca02c": "salida isoentrópica (ks)"},
    )


def _cycle_diagram(result: RefrigerationResult, system: UnitSystem) -> None:
    with st.expander("📈 Diagrama del ciclo", expanded=True):
        diagram_type: DiagramType = diagram_type_selector(key="rf_diagram_type", default="logph")
        groups = refrigeration_diagram_groups(result)
        try:
            if len(groups) == 1:
                _draw_group(result, system, diagram_type, *groups[0], chart_key="rf_diagram_chart")
            else:
                tabs = st.tabs(
                    [
                        f"{title} ({FLUID_NAMES_ES.get(fluid, fluid)})"
                        for title, (fluid, _, _) in zip(
                            ("Ciclo de baja", "Ciclo de alta"), groups, strict=True
                        )
                    ]
                )
                for tab, group, chart_key in zip(
                    tabs, groups, ("rf_diagram_low", "rf_diagram_high"), strict=True
                ):
                    with tab:
                        _draw_group(result, system, diagram_type, *group, chart_key=chart_key)
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
        devices = ["el evaporador", "el condensador"]
        if result.cycle == "flash":
            devices += ["la cámara", "la mezcla"]
        if result.cycle == "cascade":
            devices.append("el intercambiador de la cascada")
        friction = (
            " (con caída de presión, sobre una curva en la que la presión baja de a poco)"
            if result.inputs.dp_evap_Pa or result.inputs.dp_cond_Pa
            else ""
        )
        st.caption(
            f"{', '.join(devices[:-1]).capitalize()} y {devices[-1]} se dibujan sobre su "
            f"isobara{friction}. Las válvulas, rayadas sobre su línea de h constante: en log p–h "
            "son verticales; el estrangulamiento es irreversible y sus estados intermedios no "
            "son de equilibrio. El compresor ideal es isoentrópico; el real se une con una "
            "recta punteada de referencia y la verde es la compresión isoentrópica con la que "
            "se compara η_C."
        )


def _render_procedure(result: RefrigerationResult, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        tables = "R134a" in result.layout.fluids
        st.caption(
            "Cómo se resuelve estado por estado (Cengel §11-3"
            + (", §11-4" if not result.inputs.is_ideal else "")
            + (", §11-5" if result.exergy is not None else "")
            + (", §11-8" if result.cycle != "simple" else "")
            + "). Los valores salen de la ecuación de estado (de Helmholtz, en CoolProp)"
            + (
                "; las h y s del R-134a difieren de las tablas del libro en una constante (ver "
                "la nota del resultado)."
                if tables
                else "."
            )
        )
        for i, step in enumerate(refrigeration_steps(result, system), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_export(result: RefrigerationResult, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    data = refrigeration_to_dict(result, system)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="refrigeracion.csv",
        mime="text/csv",
        key="rf_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="refrigeracion.json",
        mime="application/json",
        key="rf_download_json",
    )


def _chart(
    xs: list[float], series: dict[str, list[float]], *, title: str, x_title: str, y_title: str
) -> go.Figure:
    fig = go.Figure()
    for name, ys in series.items():
        dash = "dot" if "Carnot" in name else "solid"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name, line={"dash": dash}))
    fig.update_layout(
        height=290,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def _render_sweeps(inputs: RefrigerationInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo mejorar el COP?", expanded=False):
        st.markdown(
            "El COP sube si el refrigerante **evapora más caliente** o **condensa más frío** "
            "(menos diferencia de temperaturas: por eso no conviene enfriar más de lo necesario), "
            "si el compresor es mejor o si se **subenfría** el líquido (Cengel §11-4). Con "
            "cámara o cascada, la presión intermedia tiene un óptimo. Elegí qué variar (el resto "
            "queda como en tu ciclo):"
        )
        sweeps = dict(_SWEEPS)
        if inputs.p_flash_Pa is not None:
            sweeps[_FLASH_SWEEP] = ("p_flash", "pressure", "p cámara")
        if inputs.cascade is not None:
            sweeps[_CASCADE_SWEEP] = ("T_cascade", "temperature", "T condensación (baja)")
        choice = st.selectbox("Variable", list(sweeps), key="rf_sweep_param")
        parameter, kind, axis = sweeps[choice]
        if st.button("Calcular barrido", key="rf_sweep_btn"):
            st.session_state["rf_sweep"] = (inputs, parameter)
        if st.session_state.get("rf_sweep") != (inputs, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve el ciclo varias veces).")
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
        cop = "COP_B" if inputs.heat_pump else "COP_R"
        st.plotly_chart(
            _chart(
                xs,
                {cop: [p[1] for p in points], "Carnot": [p[2] for p in points]},
                title="Coeficiente de operación",
                x_title=x_title,
                y_title="COP [-]",
            ),
            width="stretch",
            key="rf_sweep_cop",
        )
        T_unit = unit_label("temperature", system)
        st.plotly_chart(
            _chart(
                xs,
                {"descarga": [convert_from_si(p[3], "temperature", system) for p in points]},
                title="Temperatura de descarga del compresor",
                x_title=x_title,
                y_title=f"T [{T_unit}]",
            ),
            width="stretch",
            key="rf_sweep_T",
        )
        if parameter == "p_flash":
            st.caption(
                "Con poca presión en la cámara casi todo se evapora en ella y el compresor de "
                "alta trabaja de más; con mucha, la cámara casi no separa vapor: entre las dos "
                "hay un óptimo, cerca de la media geométrica de las presiones."
            )
        elif parameter == "T_cascade":
            st.caption(
                "Al subir la temperatura intermedia trabaja más el ciclo de baja y menos el de "
                "alta: el óptimo reparte la diferencia de temperaturas entre los dos."
            )


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§9.3 *Coeficientes de operación***, **§10.4 *Rendimientos isoentrópicos*** "
            "(compresor), §3.3 *Sistema abierto, régimen permanente*, §12 *Vapor húmedo* y, "
            "para el segundo principio, **§11 *Exergía*** (§11.5 exergía de un calor, §11.8 "
            "trabajo perdido, §11.10 rendimiento exergético). En el libro: Çengel & Boles, "
            "cap. 11 (8.ª ed.: §11-3 ideal, §11-4 real, §11-5 segundo principio, §11-7 bombas "
            "de calor y §11-8 cascada y cámara; en la 7.ª no está el §11-5 y los siguientes "
            "corren un número)."
        )
        st.markdown(
            "**Coeficientes de operación** (§9.3). Q_C es el calor que se saca de la fuente fría "
            "(el Q_L de Cengel) y Q_H el que se entrega a la caliente:"
        )
        st.latex(r"\mathrm{COP}_R = \frac{Q_C}{W} \qquad \mathrm{COP}_B = \frac{Q_H}{W}")
        st.latex(r"\mathrm{COP}_B = \mathrm{COP}_R + 1")
        st.latex(r"\mathrm{COP}_{R,\mathrm{rev}} = \frac{T_C}{T_H - T_C}")
        st.latex(r"\mathrm{COP}_{B,\mathrm{rev}} = \frac{T_H}{T_H - T_C}")
        st.markdown(
            "**Ciclo simple** (numeración de Cengel: 1 entrada al compresor, 2 salida, 3 salida "
            "del condensador, 4 salida de la válvula), por kilogramo de refrigerante:"
        )
        st.latex(r"q_C = h_1 - h_4 \qquad w = h_2 - h_1")
        st.latex(r"q_H = h_2 - h_3 = q_C + w")
        st.markdown(
            "**Compresor** (§10.4) y **válvula de expansión** (estrangulamiento: sin calor ni "
            "trabajo, h constante; §3.3). Después de la válvula hay vapor húmedo y el título sale "
            "de la regla de la palanca (§12):"
        )
        st.latex(r"\eta_C = \frac{h_{2s} - h_1}{h_2 - h_1}")
        st.latex(r"h_4 = h_3 \qquad x_4 = \frac{h_4 - h_f}{h_{fg}}")
        st.markdown(
            "**Ciclo real** (Cengel §11-4): el vapor sale del evaporador sobrecalentado "
            "(ΔT_sob sobre la saturación, así no llegan gotas al compresor) y el líquido sale del "
            "condensador subenfriado (ΔT_sub, más efecto refrigerante); la fricción baja la "
            "presión en los intercambiadores."
        )
        st.markdown(
            "**Dos etapas con cámara de evaporación instantánea** (Cengel §11-8, ejemplo 11-5): "
            "el líquido se estrangula hasta la cámara (6), donde una fracción x₆ se evapora y va "
            "directo a la mezcla con la descarga del compresor de baja; por el evaporador pasa "
            "1 − x₆. Por kilogramo que pasa por el condensador:"
        )
        st.latex(r"h_9 = x_6\,h_3 + (1 - x_6)\,h_2")
        st.latex(r"q_C = (1 - x_6)\,(h_1 - h_8)")
        st.latex(r"w = (1 - x_6)\,(h_2 - h_1) + (h_4 - h_9)")
        st.markdown(
            "**Cascada** (Cengel §11-8, ejemplo 11-4): el condensador del ciclo de baja (B) es el "
            "evaporador del de alta (A). El balance del intercambiador fija los caudales:"
        )
        st.latex(r"\dot{m}_A\,(h_5 - h_8) = \dot{m}_B\,(h_2 - h_3)")
        st.markdown(
            "**Segundo principio** (Cengel §11-5): la exergía destruida en cada componente es el "
            "trabajo perdido (§11.8) y el trabajo mínimo es la exergía del efecto útil (§11.5). "
            "En un refrigerador el ambiente es la fuente caliente, T₀ = T_H:"
        )
        st.latex(r"\dot{X}_{\mathrm{dest}} = T_0\,\dot{S}_{\mathrm{gen}}")
        st.latex(r"\dot{W}_{\text{mín}} = \dot{Q}_C\left(\frac{T_0}{T_C} - 1\right)")
        st.latex(r"\dot{W} = \dot{W}_{\text{mín}} + \sum \dot{X}_{\mathrm{dest}}")
        st.latex(
            r"\eta_{\mathrm{ex},R} = \frac{\dot{W}_{\text{mín}}}{\dot{W}}"
            r" = \frac{\mathrm{COP}_R}{\mathrm{COP}_{R,\mathrm{rev}}}"
        )
        st.markdown(
            "**Tonelada de refrigeración**: 1 TR = 211 kJ/min ≈ 3,517 kW, el calor para "
            "congelar una tonelada (corta) de agua a 0 °C en un día (Cengel §11-1)."
        )
        st.markdown(
            "**Tablas del R-134a**: Cengel (A-11 a A-13) toma h = s = 0 para el líquido "
            "saturado a −40 °C; CoolProp, la referencia del IIR (h = 200 kJ/kg y s = 1 kJ/(kg·K) "
            "a 0 °C). Las h y s difieren en una constante y las diferencias son iguales."
        )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Refrigeración", page_icon="❄️", layout="centered")

st.subheader(SUBJECT)
st.title("❄️ Refrigeración por compresión de vapor")
st.markdown(
    "Compresor, condensador, válvula de expansión y evaporador: simple, de dos etapas con "
    "cámara de evaporación instantánea o en cascada, ideal o real, como refrigerador o como "
    "bomba de calor. Lo resuelve [TESPy](https://tespy.readthedocs.io) como una red de "
    "componentes; el procedimiento muestra cómo se haría con las tablas y, con las "
    "temperaturas de las fuentes, dónde se destruye la exergía."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Refrigeración")
render_units_selector()
system = get_current_system()

ex = _example_index()
inputs = _read_inputs(ex)

# Se calcula al elegir un ejemplo y cada vez que se toca el botón.
if inputs is not None and st.session_state.get("rf_example_prev") != ex:
    st.session_state["rf_example_prev"] = ex
    st.session_state["rf_inputs"] = inputs
if st.button("Calcular ciclo", key="rf_btn", type="primary") and inputs is not None:
    st.session_state["rf_inputs"] = inputs

computed: RefrigerationInputs | None = st.session_state.get("rf_inputs")
if computed is not None:
    if computed != inputs:
        st.caption("Cambiaste datos: tocá «Calcular ciclo» para actualizar el resultado.")
    try:
        result = _solve_cached(computed)
    except ValueError as exc:
        _render_error(exc)
    else:
        st.markdown("### Resultado")
        note = REFRIGERATION_EXAMPLE_NOTES.get(_EXAMPLES[ex])
        if note:
            st.caption(f"📘 Sobre este ejemplo: {note}")
        _render_metrics(result, system)
        _render_exergy(result, system)
        st.markdown("#### Estados")
        _render_states_table(result, system)
        _cycle_diagram(result, system)
        _render_procedure(result, system)
        _render_tespy(result, system)
        _render_export(result, system)
        _render_sweeps(computed, system)
