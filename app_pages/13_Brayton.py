"""Página 13 — Turbina de gas: ciclo Brayton con interenfriamiento, recalentamiento y regenerador.

Fase 3.7. La turbina de gas de la Fase 3.4 (compresor, cámara de combustión con
metano o aire estándar y turbina) con las tres mejoras del ciclo Brayton de
Çengel & Boles (§9-9 y §9-10): compresión en etapas con interenfriamiento,
expansión en etapas con recalentamiento (con combustible, una segunda cámara:
combustión secuencial) y regenerador. El cálculo vive en
:mod:`core.cycles.gas_turbine` (directo, con TESPy de control). La página
muestra el rendimiento, el trabajo y la relación de trabajo de retroceso, el
diagrama T–s, las tablas de estados y de componentes, cuánto suma cada mejora,
la exergía destruida en cada componente, el procedimiento, el control con
TESPy, la exportación y barridos.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.brayton import METHANE
from core.cycles.gas_turbine import (
    GAS_TURBINE_EXAMPLE_NOTES,
    GAS_TURBINE_EXAMPLES,
    GasTurbineInputs,
    GasTurbineResult,
    GasTurbineSweepPoint,
    GasTurbineTespy,
    ImprovementRow,
    SweepParameter,
    default_gas_turbine_sweep_values,
    gas_turbine_exergy,
    gas_turbine_notes,
    gas_turbine_sweep,
    gas_turbine_tespy,
    gas_turbine_to_dict,
    improvement_comparison,
    solve_gas_turbine,
)
from core.cycles.gas_turbine_procedure import gas_turbine_steps
from core.export import dict_to_csv
from core.ideal_gas import AIR_DRY, AIR_TECHNICAL, GAS_NAMES, GAS_SPECIES
from core.state_report import format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.cycle_charts import (
    gas_turbine_cycle_figure,
    gas_turbine_exergy_figure,
    improvement_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.19.0"

_EXAMPLES = list(GAS_TURBINE_EXAMPLES)
# Opciones fijas: cambiarlas reiniciaría el widget.
_MODEL_FUEL = "Combustión de metano"
_MODEL_AIR = "Aire estándar (sin combustible)"
_AIRS = {"Seco (N₂, O₂, Ar, CO₂)": AIR_DRY, "Técnico (21 % O₂, 79 % N₂)": AIR_TECHNICAL}
_STAGES = ("1 etapa", "2 etapas", "3 etapas")
_PRESSURES_OPTIMAL = "Óptimas (relaciones iguales)"
_PRESSURES_GIVEN = "A elección"
_SIZE_AIR = "Caudal de aire"
_SIZE_POWER = "Potencia neta"

# (parámetro, magnitud del eje, rótulo del eje)
_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind | None, str]] = {
    "Relación de presiones": ("pressure_ratio", None, "r_p"),
    "Temperatura de entrada a la turbina": ("T_turbine_in", "temperature", "TIT"),
    "Temperatura ambiente": ("T_amb", "temperature", "T₁"),
    "Cantidad de etapas (compresión y expansión)": ("stages", None, "etapas"),
}
_SWEEP_REGEN = "Efectividad del regenerador"
_SWEEP_P_IC = "Presión del interenfriamiento"
_SWEEP_P_RH = "Presión del recalentamiento"
_SWEEP_NOTES: dict[str, str] = {
    "pressure_ratio": (
        "Sin regenerador, el rendimiento sube con r_p y el trabajo neto tiene un máximo: a r_p "
        "muy alta el compresor se lleva casi todo el trabajo de la turbina (Cengel §9-8). Con "
        "regenerador pasa al revés: el rendimiento es mayor a r_p bajas y la curva se corta "
        "cuando los gases salen de la turbina más fríos que el aire del compresor (T₄ ≤ T₂): "
        "ahí el regenerador ya no sirve (§9-9)."
    ),
    "T_turbine_in": (
        "Una TIT más alta sube el rendimiento y, sobre todo, el trabajo por kg de aire: por "
        "eso las turbinas modernas trabajan con 1400–1600 °C, con álabes refrigerados."
    ),
    "T_amb": (
        "Con el aire más caliente el compresor trabaja más (y entra menos masa por el mismo "
        "volumen): en verano la turbina de gas entrega menos potencia y rinde menos."
    ),
    "stages": (
        "Con más etapas de compresión (con interenfriamiento) y de expansión (con "
        "recalentamiento), la compresión se acerca a una isoterma a T₁ y la expansión a una "
        "isoterma a la TIT. Con un regenerador ideal, el ciclo tiende al de Ericsson, cuyo "
        "rendimiento es el de Carnot (línea punteada, Cengel §9-10). Sin regenerador, más "
        "etapas dan más trabajo pero no más rendimiento."
    ),
    "regenerator": (
        "Más efectividad: el aire entra más caliente a la cámara y hace falta menos "
        "combustible. Pero ε → 1 pide un área infinita, y un regenerador más grande también "
        "pierde más presión: los reales tienen ε de 0,6 a 0,9."
    ),
    "p_intercool": (
        "El trabajo de compresión es mínimo cuando las dos etapas tienen la misma relación de "
        "presiones: p_x = √(p₁·p₂) (línea punteada, vademecum §6.3)."
    ),
    "p_reheat": (
        "El trabajo de la turbina es máximo con la misma relación de presiones en las dos etapas "
        "(línea punteada). Recalentar a presión más alta deja el escape más frío: es lo que "
        "hacen las turbinas de combustión secuencial."
    ),
}


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner="Resolviendo la turbina de gas…")
def _solve_cached(inputs: GasTurbineInputs) -> GasTurbineResult:
    return solve_gas_turbine(inputs)


@st.cache_data(show_spinner="Comparando las mejoras (resuelve la turbina varias veces)…")
def _comparison_cached(inputs: GasTurbineInputs) -> list[ImprovementRow]:
    return improvement_comparison(inputs)


@st.cache_data(show_spinner="Resolviendo la turbina de gas con TESPy…")
def _tespy_cached(inputs: GasTurbineInputs) -> GasTurbineTespy | str:
    try:
        return gas_turbine_tespy(solve_gas_turbine(inputs))
    except ValueError as exc:
        return str(exc)


@st.cache_data(show_spinner="Calculando el barrido…")
def _sweep_cached(
    inputs: GasTurbineInputs, parameter: SweepParameter
) -> tuple[list[GasTurbineSweepPoint], list[GasTurbineSweepPoint], int]:
    """Los puntos del barrido, los del ciclo simple (referencia de r_p) y cuántos valores hubo."""
    values = default_gas_turbine_sweep_values(inputs, parameter)
    points = gas_turbine_sweep(inputs, parameter, values)
    reference: list[GasTurbineSweepPoint] = []
    if parameter == "pressure_ratio" and inputs.configuration != "simple":
        simple = replace(
            inputs,
            compressor_stages=1,
            turbine_stages=1,
            regenerator=None,
            p_intercool_Pa=(),
            p_reheat_Pa=(),
        )
        reference = gas_turbine_sweep(simple, parameter, values)
    return points, reference, len(values)


# ---------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------


def _number(value: float) -> str:
    """Número para mostrar: sin notación exponencial en los valores grandes."""
    if abs(value) >= 1.0e4:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, 5)


def _value(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    if kind in ("power", "mass_flow"):
        return _number(convert_from_si(value_si, kind, system))
    return format_value(convert_from_si(value_si, kind, system), sig)


def _fmt(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    return f"{_value(value_si, kind, system, sig)} {unit_label(kind, system)}"


def _render_error(exc: ValueError) -> None:
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico: {detail.strip()}")


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _example_index() -> int:
    label = st.selectbox(
        "Ejemplo precargado",
        _EXAMPLES,
        key="bt_example",
        help="Podés cambiar cualquier dato y volver a calcular.",
    )
    return _EXAMPLES.index(label)


def _read_basic(key: str, base: GasTurbineInputs) -> GasTurbineInputs:
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
            help="La del compresor completo: p de descarga / p₁.",
        )
        TIT = number_input_si(
            label="Entrada a la turbina (TIT)",
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
            "η_C de cada compresor",
            min_value=0.3,
            max_value=1.0,
            value=float(base.eta_compressor),
            step=0.01,
            format="%.3f",
            key=f"{key}_etac",
            help="1 = isoentrópico (vademecum §10.4).",
        )
        eta_t = st.number_input(
            "η_T de cada turbina",
            min_value=0.3,
            max_value=1.0,
            value=float(base.eta_turbine),
            step=0.01,
            format="%.3f",
            key=f"{key}_etat",
        )
    dp = st.number_input(
        "Caída de presión en cada cámara [%]",
        min_value=0.0,
        max_value=19.9,
        value=float(base.dp_combustor * 100.0),
        step=0.5,
        format="%.3g",
        key=f"{key}_dp",
        help="La salida de la cámara queda a p·(1 − Δp). Típico: 3 a 5 %.",
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


def _equal_pressures(p_lo: float, p_hi: float, n: int) -> list[float]:
    """Presiones intermedias con relaciones iguales entre p_lo y p_hi (sin pérdidas)."""
    r = (p_hi / p_lo) ** (1.0 / n)
    return [p_lo * r**k for k in range(1, n)]


def _read_pressures(
    key: str,
    what: str,
    n: int,
    given: tuple[float, ...],
    p_lo: float,
    p_hi: float,
    descending: bool,
) -> tuple[float, ...]:
    """Las presiones intermedias: óptimas (vacío) o a elección del alumno."""
    mode = st.radio(
        f"Presiones de {what}",
        (_PRESSURES_OPTIMAL, _PRESSURES_GIVEN),
        index=1 if given else 0,
        horizontal=True,
        key=f"{key}_mode",
        help=(
            "Con relaciones de presión iguales en cada etapa, el trabajo es el óptimo "
            "(vademecum §6.3: con dos etapas, p_x = √(p₁·p₂))."
        ),
    )
    system = get_current_system()
    defaults = _equal_pressures(p_lo, p_hi, n)
    if descending:
        defaults = defaults[::-1]
    if mode == _PRESSURES_OPTIMAL:
        shown = ", ".join(_fmt(p, "pressure", system, 4) for p in defaults)
        st.caption(f"Sin pérdidas de carga: {shown}.")
        return ()
    values = []
    for k in range(n - 1):
        default = given[k] if len(given) == n - 1 else defaults[k]
        values.append(
            number_input_si(
                label=f"Presión de {what} {k + 1}" if n > 2 else f"Presión de {what}",
                kind="pressure",
                default_si=default,
                key=f"{key}{n}_{k}",
                format="%.5g",
                min_value_si=0.0,
            )
        )
    return tuple(values)


def _read_stages(key: str, base: GasTurbineInputs, current: GasTurbineInputs) -> GasTurbineInputs:
    p1 = current.p_amb_Pa
    p2 = p1 * current.pressure_ratio
    st.markdown("#### Interenfriamiento")
    nc = (
        _STAGES.index(
            st.radio(
                "Etapas de compresión",
                _STAGES,
                index=base.compressor_stages - 1,
                horizontal=True,
                key=f"{key}_nc",
                help="Entre etapa y etapa el aire se enfría a presión constante (Cengel §9-10).",
            )
        )
        + 1
    )
    T_ic, dp_ic, p_ic = None, 0.0, ()
    if nc > 1:
        left, right = st.columns(2)
        with left:
            T_ic = number_input_si(
                label="T después de interenfriar",
                kind="temperature",
                default_si=base.T_intercool,
                key=f"{key}_Tic",
                format="%.4g",
                help="Lo habitual: la del ambiente (interenfriamiento perfecto) o unos grados más.",
            )
        with right:
            dp_ic = (
                st.number_input(
                    "Δp de cada interenfriador [%]",
                    min_value=0.0,
                    max_value=19.9,
                    value=float(base.dp_intercooler * 100.0),
                    step=0.5,
                    format="%.3g",
                    key=f"{key}_dpic",
                )
                / 100.0
            )
        p_ic = _read_pressures(
            f"{key}_pic",
            "interenfriamiento",
            nc,
            base.p_intercool_Pa if base.compressor_stages == nc else (),
            p1,
            p2,
            descending=False,
        )
    st.markdown("#### Recalentamiento")
    nt = (
        _STAGES.index(
            st.radio(
                "Etapas de expansión",
                _STAGES,
                index=base.turbine_stages - 1,
                horizontal=True,
                key=f"{key}_nt",
                help=(
                    "Entre etapa y etapa los gases se vuelven a calentar (Cengel §9-10). Con "
                    "combustible, en una segunda cámara que quema en los gases (combustión "
                    "secuencial)."
                ),
            )
        )
        + 1
    )
    T_rh, p_rh = None, ()
    if nt > 1:
        T_rh = number_input_si(
            label="T de recalentamiento",
            kind="temperature",
            default_si=base.T_reheat,
            key=f"{key}_Trh",
            format="%.5g",
            help="Lo habitual: la misma TIT.",
        )
        p_rh = _read_pressures(
            f"{key}_prh",
            "recalentamiento",
            nt,
            base.p_reheat_Pa if base.turbine_stages == nt else (),
            p1,
            p2,
            descending=True,
        )
    st.markdown("#### Regenerador")
    regen = st.checkbox(
        "Regenerador (el escape precalienta el aire del compresor)",
        value=base.regenerated,
        key=f"{key}_regen",
        help="Sirve si los gases salen de la turbina más calientes que el aire del compresor.",
    )
    eps, dp_rg = None, 0.0
    if regen:
        left, right = st.columns(2)
        with left:
            eps = st.number_input(
                "Efectividad ε",
                min_value=0.01,
                max_value=1.0,
                value=float(base.regenerator if base.regenerator is not None else 0.8),
                step=0.01,
                format="%.3f",
                key=f"{key}_eps",
                help="1 = regenerador ideal (área infinita). Los reales: 0,6 a 0,9 (Cengel §9-9).",
            )
        with right:
            dp_rg = (
                st.number_input(
                    "Δp de cada lado [%]",
                    min_value=0.0,
                    max_value=19.9,
                    value=float(base.dp_regenerator * 100.0),
                    step=0.5,
                    format="%.3g",
                    key=f"{key}_dprg",
                )
                / 100.0
            )
    return replace(
        current,
        compressor_stages=nc,
        turbine_stages=nt,
        T_intercool_K=T_ic,
        T_reheat_K=T_rh,
        dp_intercooler=dp_ic,
        p_intercool_Pa=p_ic,
        p_reheat_Pa=p_rh,
        regenerator=float(eps) if eps is not None else None,
        dp_regenerator=dp_rg,
    )


def _read_size(key: str, current: GasTurbineInputs) -> GasTurbineInputs:
    """Caudal de aire o potencia neta (todo escala con el caudal)."""
    st.markdown("#### Tamaño")
    size = st.radio(
        "Dato",
        (_SIZE_AIR, _SIZE_POWER),
        index=0,
        horizontal=True,
        key=f"{key}_size",
        help="Todo escala con el caudal: los rendimientos no cambian.",
    )
    if size == _SIZE_AIR:
        m_air = number_input_si(
            label="Caudal de aire",
            kind="mass_flow",
            default_si=current.m_air_kg_s,
            key=f"{key}_mair",
            format="%.5g",
            min_value_si=0.0,
        )
        return replace(current, m_air_kg_s=m_air, W_net_W=None)
    W_net = number_input_si(
        label="Potencia neta",
        kind="power",
        default_si=100e6,
        key=f"{key}_W",
        format="%.6g",
        min_value_si=0.0,
    )
    return replace(current, W_net_W=W_net)


def _read_inputs(ex: int) -> GasTurbineInputs:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    base = GAS_TURBINE_EXAMPLES[_EXAMPLES[ex]]
    key = f"bt_{ex}"
    current = _read_basic(key, base)
    current = _read_stages(key, base, current)
    return _read_size(key, current)


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


def _render_metrics(result: GasTurbineResult, system: UnitSystem) -> None:
    P = unit_label("power", system)
    T_unit = unit_label("temperature", system)
    e_unit = unit_label("specific_enthalpy", system)
    row1 = st.columns(3)
    row1[0].metric("η turbina de gas [%]", f"{result.eta_th * 100:.2f}")
    row1[1].metric(f"w neto [{e_unit}]", _value(result.w_net_J_per_kg, "specific_enthalpy", system))
    row1[2].metric("r_bw = w_C/w_T", f"{result.back_work_ratio:.3f}")
    row2 = st.columns(3)
    row2[0].metric(f"Ẇ neto [{P}]", _value(result.W_net_W, "power", system))
    row2[1].metric(f"T escape [{T_unit}]", _value(result.T_exhaust_K, "temperature", system, 4))
    if result.excess_air is not None:
        row2[2].metric("Exceso de aire λ", f"{result.excess_air:.3g}")
    else:
        row2[2].metric(f"Q̇ entrada [{P}]", _value(result.Q_in_W, "power", system))
    q_in = result.q_in_J_per_kg
    parts = [f"{result.w_net_J_per_kg / q_in * 100:.1f} % sale como trabajo neto"]
    if result.intercoolers:
        parts.append(
            f"{result.q_intercoolers_J_per_kg / q_in * 100:.1f} % al agua de los interenfriadores"
        )
    parts.append(f"{result.q_exhaust_J_per_kg / q_in * 100:.1f} % con el escape")
    heat = "del calor que entra" if result.inputs.air_standard else "de la energía del combustible"
    st.caption(
        f"Configuración: {result.inputs.configuration}. De cada 100 unidades {heat}, "
        + ", ".join(parts[:-1])
        + f" y {parts[-1]}."
        + (
            f" El regenerador recircula {result.q_regenerator_J_per_kg / q_in * 100:.1f} % más "
            "dentro del ciclo."
            if result.regenerator is not None
            else ""
        )
    )
    for note in gas_turbine_notes(result):
        st.info(note)


def _render_diagram(result: GasTurbineResult, system: UnitSystem) -> None:
    st.markdown("#### Diagrama T–s")
    try:
        st.plotly_chart(gas_turbine_cycle_figure(result, system), width="stretch", key="bt_ts")
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    mixing = (
        ""
        if result.inputs.air_standard
        else " Los tramos grises horizontales a la entrada de cada cámara son el cambio de "
        "composición: a la misma T y p, los gases (con el CO₂ y el H₂O de la combustión) tienen "
        "otra entropía que el aire."
    )
    st.caption(
        "Entropías absolutas (con las de NIST-JANAF a 25 °C y 1 bar), así el aire y los gases "
        "quedan en la misma escala. Las líneas grises finas son las isobaras de p₁ y de la "
        "descarga del compresor. Las compresiones y expansiones reales se unen con rectas "
        "punteadas solo como referencia; las rayadas son las isoentrópicas con las que se "
        "comparan los rendimientos." + mixing + " En un ciclo abierto, el escape se enfría en "
        "el ambiente hasta T₁: eso cierra el ciclo."
    )


def _states_table(result: GasTurbineResult, system: UnitSystem) -> pd.DataFrame:
    T_unit = unit_label("temperature", system)
    p_unit = unit_label("pressure", system)
    e_unit = unit_label("specific_enthalpy", system)
    s_unit = unit_label("specific_entropy", system)
    return pd.DataFrame(
        [
            {
                "Estado": s.label,
                f"T [{T_unit}]": _value(s.T_K, "temperature", system),
                f"p [{p_unit}]": _value(s.P_Pa, "pressure", system),
                f"h [{e_unit}]": _value(s.h_J_per_kg, "specific_enthalpy", system),
                f"s [{s_unit}]": _value(s.s_J_per_kg_K, "specific_entropy", system),
                "Dónde": s.description,
                "Medio": s.medium,
            }
            for s in result.states
        ]
    )


def _components_table(result: GasTurbineResult, system: UnitSystem) -> pd.DataFrame:
    e_unit = unit_label("specific_enthalpy", system)
    P = unit_label("power", system)
    st_ = result.states
    m = result.m_air_kg_s
    rows = []

    def add(name: str, a: int, b: int, what: str, value: float) -> None:
        rows.append(
            {
                "Componente": name,
                "Estados": f"{st_[a].label} → {st_[b].label}",
                "Qué": what,
                f"Por kg de aire [{e_unit}]": _value(value, "specific_enthalpy", system),
                f"Potencia [{P}]": _value(m * value, "power", system),
            }
        )

    for k, c in enumerate(result.compressors):
        add(c.name, c.inlet, c.outlet, "w (consume)", result.stage_work(c))
        if k < len(result.intercoolers):
            ic = result.intercoolers[k]
            add(ic.name, ic.inlet, ic.outlet, "q (saca)", result.intercooler_heat(ic))
    rg = result.regenerator
    if rg is not None:
        add("regenerador (aire)", rg.air_in, rg.air_out, "q (interno)", rg.q)
    for cc, t in zip(result.combustors, result.turbines, strict=True):
        add(cc.name, cc.inlet, cc.outlet, "q (entra)", cc.q)
        add(t.name, t.inlet, t.outlet, "w (entrega)", result.stage_work(t))
    if rg is not None:
        add("regenerador (gases)", rg.gas_in, rg.gas_out, "q (interno)", rg.q)
    return pd.DataFrame(rows)


def _render_tables(result: GasTurbineResult, system: UnitSystem) -> None:
    st.markdown("#### Estados")
    st.dataframe(_states_table(result, system), hide_index=True, width="stretch")
    gas = result.gas
    y = gas.mole_fractions
    fuel = ""
    if result.inputs.fuel is not None:
        fuel = (
            f" Combustible: {_fmt(result.m_fuel_kg_s, 'mass_flow', system)} "
            f"(f = {result.fuel_air_ratio:.4g} kg por kg de aire). Gases de escape: "
            + " · ".join(f"{GAS_NAMES[s]} {y[s] * 100:.3g} %" for s in GAS_SPECIES if y[s] > 0)
            + " (fracción molar)."
        )
    st.caption(
        "Numeración de Cengel: 1–4 en el ciclo simple, 5 y 6 para el regenerador (figura 9-38) "
        "y en el sentido del flujo cuando hay etapas (figura 9-43). Los estados con «s» son los "
        "isoentrópicos. h medida desde 25 °C (la referencia del poder calorífico)." + fuel
    )
    st.markdown("#### Componentes")
    st.dataframe(_components_table(result, system), hide_index=True, width="stretch")


def _render_comparison(computed: GasTurbineInputs, system: UnitSystem) -> None:
    st.markdown("#### ¿Cuánto ganás con cada mejora?")
    rows = _comparison_cached(replace(computed, W_net_W=None))
    T_unit = unit_label("temperature", system)
    e_unit = unit_label("specific_enthalpy", system)
    table = []
    for row in rows:
        mark = " ◆" if row.current else ""
        if row.result is None:
            table.append(
                {
                    "Configuración": row.label + mark,
                    "η [%]": "—",
                    f"w neto [{e_unit}]": "—",
                    "r_bw": "—",
                    f"T escape [{T_unit}]": "—",
                }
            )
            continue
        r = row.result
        table.append(
            {
                "Configuración": row.label + mark,
                "η [%]": f"{r.eta_th * 100:.2f}",
                f"w neto [{e_unit}]": _value(r.w_net_J_per_kg, "specific_enthalpy", system, 4),
                "r_bw": f"{r.back_work_ratio:.3f}",
                f"T escape [{T_unit}]": _value(r.T_exhaust_K, "temperature", system, 4),
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    try:
        st.plotly_chart(improvement_figure(rows, system), width="stretch", key="bt_improvements")
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    for row in rows:
        if row.result is None:
            st.caption(f"{row.label}: no se puede con estos datos. {row.error}")
    eps = computed.regenerator if computed.regenerated else 0.80
    st.caption(
        "La misma turbina (r_p, TIT y rendimientos de tus datos), con regenerador de "
        f"ε = {eps:.3g} y con las etapas de tus datos (o dos). ◆ es tu configuración. Sin "
        "regenerador, interenfriar y recalentar suben el trabajo neto pero bajan el "
        "rendimiento: el calor entra a menor temperatura media y sale a mayor. Con "
        "regenerador, las tres mejoras "
        "juntas dan el mejor rendimiento (Cengel §9-10)."
    )


def _render_exergy(result: GasTurbineResult, system: UnitSystem) -> None:
    st.markdown("#### Exergía: ¿dónde se pierde?")
    x = gas_turbine_exergy(result)
    try:
        st.plotly_chart(gas_turbine_exergy_figure(x, system), width="stretch", key="bt_exergy")
    except Exception as exc:  # el diagrama no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    what = (
        "del combustible (≈ PCI, vademecum §16.13)"
        if result.inputs.fuel is not None
        else "del calor que entra, Q·(1 − T₀/T)"
    )
    destroyed = max(
        (i for i in x.items if i.kind == "destruida"), key=lambda i: i.x_J_per_kg, default=None
    )
    biggest = (
        f" La mayor destrucción es la de {destroyed.name}: "
        + (
            "quemar el combustible con un salto de temperatura enorme es muy irreversible."
            if destroyed.name.startswith("cámara")
            else "ahí está la mayor irreversibilidad del ciclo."
        )
        if destroyed is not None and destroyed.x_J_per_kg > 0.0
        else ""
    )
    st.caption(
        f"De la exergía {what}, con T₀ = {_fmt(x.T0_K, 'temperature', system, 4)}, la turbina de "
        f"gas convierte en trabajo el {x.efficiency * 100:.1f} %. Compresores, turbinas y "
        "regenerador destruyen exergía por fricción y por pasar calor con diferencia de "
        "temperatura (T₀·S_gen); el escape y el agua de los interenfriadores se llevan el resto "
        "(Cengel §9-12)." + biggest
    )


def _render_procedure(result: GasTurbineResult, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Cómo se resuelve a mano, con las tablas de gas ideal (Cengel §9-8 a §9-10): cada "
            "etapa del compresor, los interenfriadores, las cámaras y las turbinas, el regenerador "
            "(después de la turbina, como el ejemplo 9-7), el rendimiento y la exergía."
        )
        for n, step in enumerate(gas_turbine_steps(result, system), start=1):
            st.markdown(f"**{n}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_tespy(inputs: GasTurbineInputs, result: GasTurbineResult, system: UnitSystem) -> None:
    with st.expander("⚙️ Control con TESPy", expanded=False):
        st.markdown(
            "La misma turbina de gas resuelta con [TESPy](https://tespy.readthedocs.io): un "
            "*Compressor* por etapa, *SimpleHeatExchanger* en los interenfriadores, un "
            "*HeatExchanger* con la efectividad del lado frío en el regenerador, "
            "*DiabaticCombustionChamber* en cada cámara (con aire estándar, *SimpleHeatExchanger*) "
            "y una *Turbine* por etapa, como su tutorial de turbina de gas. TESPy evalúa cada "
            "componente de la mezcla a su presión parcial con su ecuación de estado real: por eso "
            "difiere unas décimas de % del cálculo con gases ideales."
        )
        control = _tespy_cached(replace(inputs, W_net_W=None, m_air_kg_s=1.0))
        if isinstance(control, str):
            st.warning(f"TESPy no pudo resolver esta turbina de gas: {control}")
            return
        T_unit = unit_label("temperature", system)
        e_unit = unit_label("specific_enthalpy", system)
        rows = [
            (
                f"w_C [{e_unit}]",
                _value(result.w_compressor_J_per_kg, "specific_enthalpy", system),
                _value(control.w_compressor_J_per_kg, "specific_enthalpy", system),
            ),
            (
                f"w_T [{e_unit}]",
                _value(result.w_turbine_J_per_kg, "specific_enthalpy", system),
                _value(control.w_turbine_J_per_kg, "specific_enthalpy", system),
            ),
            (
                f"w_neto [{e_unit}]",
                _value(result.w_net_J_per_kg, "specific_enthalpy", system),
                _value(control.w_net_J_per_kg, "specific_enthalpy", system),
            ),
            (
                f"T escape [{T_unit}]",
                _value(result.T_exhaust_K, "temperature", system),
                _value(control.T_exhaust_K, "temperature", system),
            ),
            ("η [%]", f"{result.eta_th * 100:.3f}", f"{control.eta_th * 100:.3f}"),
        ]
        rg = result.regenerator
        if rg is not None and control.T_regenerator_air_K is not None:
            rows.insert(
                3,
                (
                    f"T aire del regenerador [{T_unit}]",
                    _value(result.states[rg.air_out].T_K, "temperature", system),
                    _value(control.T_regenerator_air_K, "temperature", system),
                ),
            )
        if result.inputs.fuel is not None:
            rows.append(
                (
                    "f (kg combustible/kg aire)",
                    format_value(result.fuel_air_ratio, 5),
                    format_value(control.fuel_air_ratio, 5),
                )
            )
        st.dataframe(
            pd.DataFrame(rows, columns=["Magnitud", "Gases ideales", "TESPy"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"TESPy resolvió la red en {control.seconds:.2f} s (por kg de aire).")


def _render_export(result: GasTurbineResult, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    data = gas_turbine_to_dict(result, system)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="turbina_de_gas.csv",
        mime="text/csv",
        key="bt_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="turbina_de_gas.json",
        mime="application/json",
        key="bt_download_json",
    )


def _sweep_axis(
    points: list[GasTurbineSweepPoint],
    kind: QuantityKind | None,
    parameter: SweepParameter,
    system: UnitSystem,
) -> list[float]:
    if parameter in ("p_intercool", "p_reheat"):
        return [convert_from_si(p.value, "pressure", system) for p in points]
    if kind is None:
        return [p.value for p in points]
    return [convert_from_si(p.value, kind, system) for p in points]


def _render_sweeps(inputs: GasTurbineInputs, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo mejorar la turbina de gas?", expanded=False):
        st.markdown(
            "Elegí qué variar (el resto queda como en tu turbina) para ver cómo cambian el "
            "rendimiento y el trabajo neto:"
        )
        sweeps: dict[str, tuple[SweepParameter, QuantityKind | None, str]] = dict(_SWEEPS)
        if inputs.regenerated:
            sweeps[_SWEEP_REGEN] = ("regenerator", None, "ε")
        if inputs.compressor_stages == 2:
            sweeps[_SWEEP_P_IC] = ("p_intercool", "pressure", "p interenfriamiento")
        if inputs.turbine_stages == 2:
            sweeps[_SWEEP_P_RH] = ("p_reheat", "pressure", "p recalentamiento")
        choice = st.selectbox("Variable", list(sweeps), key="bt_sweep_param")
        parameter, kind, axis = sweeps[choice]
        if parameter in ("p_intercool", "p_reheat"):
            kind = "pressure"
        case = replace(inputs, W_net_W=None)
        if st.button("Calcular barrido", key="bt_sweep_btn"):
            st.session_state["bt_sweep"] = (case, parameter)
        if st.session_state.get("bt_sweep") != (case, parameter):
            st.caption("Tocá «Calcular barrido» (resuelve la turbina varias veces).")
            return
        points, reference, n_values = _sweep_cached(case, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        xs = _sweep_axis(points, kind, parameter, system)
        x_title = f"{axis} [-]" if kind is None else f"{axis} [{unit_label(kind, system)}]"
        e_unit = unit_label("specific_enthalpy", system)
        log_x = parameter in ("pressure_ratio", "p_intercool", "p_reheat")
        for metric, title, y_title, key in (
            ("eta", "Rendimiento", "η [%]", "bt_sweep_eta"),
            ("w", "Trabajo neto por kg de aire", f"w neto [{e_unit}]", "bt_sweep_w"),
        ):

            def ys(pts: list[GasTurbineSweepPoint], metric: str = metric) -> list[float]:
                if metric == "eta":
                    return [p.eta * 100 for p in pts]
                return [convert_from_si(p.w_net_J_per_kg, "specific_enthalpy", system) for p in pts]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=xs, y=ys(points), mode="lines+markers", name="tu configuración")
            )
            if reference:
                fig.add_trace(
                    go.Scatter(
                        x=_sweep_axis(reference, kind, parameter, system),
                        y=ys(reference),
                        mode="lines",
                        name="ciclo simple",
                        line={"dash": "dash", "color": "#888888"},
                    )
                )
            if metric == "eta" and parameter == "stages":
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=[p.eta_carnot * 100 for p in points],
                        mode="lines",
                        name="Carnot (Ericsson)",
                        line={"dash": "dot", "color": "#888888"},
                    )
                )
            if parameter in ("p_intercool", "p_reheat"):
                p1 = inputs.p_amb_Pa
                optimum = convert_from_si(
                    (p1 * p1 * inputs.pressure_ratio) ** 0.5, "pressure", system
                )
                fig.add_vline(x=optimum, line_dash="dot", line_color="#888888")
            fig.update_layout(
                height=300,
                margin={"l": 10, "r": 10, "t": 30, "b": 10},
                title=title,
                xaxis_title=x_title,
                yaxis_title=y_title,
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
                separators=". ",
                showlegend=bool(reference) or parameter == "stages",
            )
            if log_x:
                fig.update_xaxes(type="log")
            st.plotly_chart(fig, width="stretch", key=key)
        skipped = n_values - len(points)
        if skipped:
            st.caption(
                f"{skipped} valor(es) del barrido no tienen sentido físico con estos datos (por "
                "ejemplo, un regenerador que enfriaría el aire o una cámara sin oxígeno) y se "
                "omiten."
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
            "**§6.3 *Trabajo de circulación*** (compresión en dos etapas con interenfriamiento), "
            "§9.2 *Rendimientos térmicos*, **§10.4 *Rendimientos isoentrópicos***, **§11 "
            "*Exergía*** y **§16 *Combustión*** (§16.13: la exergía del combustible ≈ PCI). En el "
            "libro: Çengel & Boles §9-8 (ciclo Brayton), §9-9 (regeneración), §9-10 "
            "(interenfriamiento, recalentamiento y regeneración) y §9-12 (segundo principio). El "
            "poder calorífico, de ISO 6976:2016 a 25 °C."
        )
        st.markdown(
            "**Compresor y turbina** con calores específicos variables (por etapa): la "
            "isoentrópica sale de la función s° = s°(T) (la columna s° de la tabla A-17 de "
            "Cengel para el aire) y el rendimiento isoentrópico da el estado real:"
        )
        st.latex(r"s^{\circ}_{2s} = s^{\circ}_1 + R\,\ln\frac{p_2}{p_1}")
        st.latex(r"\eta_C = \frac{h_{2s} - h_1}{h_2 - h_1}")
        st.latex(r"\eta_T = \frac{h_3 - h_4}{h_3 - h_{4s}}")
        st.markdown(
            "**Interenfriamiento**: entre etapas el aire se enfría a presión constante. Con el "
            "aire enfriado hasta la misma temperatura y el mismo rendimiento en las dos etapas, el "
            "trabajo es mínimo cuando las relaciones de presión son iguales (vademecum §6.3):"
        )
        st.latex(r"p_x = \sqrt{p_1\,p_2} \qquad \frac{p_x}{p_1} = \frac{p_2}{p_x}")
        st.markdown(
            "**Recalentamiento**: entre etapas de la turbina los gases se vuelven a calentar; con "
            "la misma temperatura, el trabajo es máximo con relaciones iguales. Con combustible, "
            "una segunda cámara quema más combustible en los gases (que todavía tienen O₂):"
        )
        st.latex(r"(1 + F_1)\,h_{\mathrm{ent}} + f_2\,\mathrm{PCI} = (1 + F_2)\,h_{\mathrm{sal}}")
        st.markdown(
            "**Regenerador**: los gases de escape calientan el aire que sale del compresor. "
            "Sirve si T₄ > T₂; su efectividad es la parte del calor máximo que se logra (Cengel "
            "§9-9), con las entalpías del aire:"
        )
        st.latex(r"\varepsilon = \frac{h_5 - h_2}{h_4 - h_2} \approx \frac{T_5 - T_2}{T_4 - T_2}")
        st.markdown(
            "**Rendimiento**, por kilogramo de aire (con combustión, por la turbina pasan 1 + F "
            "kg de gases):"
        )
        st.latex(
            r"\eta = \frac{w_{\mathrm{neto}}}{q_{\mathrm{ent}}} = \frac{w_T - w_C}{F\,\mathrm{PCI}}"
        )
        st.latex(r"r_{bw} = \frac{w_C}{w_T}")
        st.markdown(
            "Con aire estándar frío (c_p constante, k = 1,4), el Brayton ideal y el ideal con "
            "regenerador cumplen (Cengel §9-8 y §9-9):"
        )
        st.latex(r"\eta_{\mathrm{Brayton}} = 1 - \frac{1}{r_p^{(k-1)/k}}")
        st.latex(r"\eta_{\mathrm{regen}} = 1 - \frac{T_1}{T_3}\,r_p^{(k-1)/k}")
        st.markdown(
            "Con muchas etapas y regenerador ideal, el ciclo tiende al de **Ericsson** (compresión "
            "y expansión isotérmicas), que tiene el rendimiento de Carnot:"
        )
        st.latex(r"\eta_{\mathrm{Ericsson}} = 1 - \frac{T_{\min}}{T_{\max}}")
        st.markdown(
            "**Exergía** (vademecum §11): la que entra sale como trabajo, se destruye en cada "
            "componente (T₀·S_gen) o se pierde con el escape y el agua de los interenfriadores. "
            "η_II = w_neto / x_ent:"
        )
        st.latex(
            r"\begin{aligned}x_{\mathrm{ent}} &= w_{\mathrm{neto}} + \textstyle\sum x_d"
            r" \\ &\quad + x_{\mathrm{esc}} + \textstyle\sum x_{IC}\end{aligned}"
        )
        st.latex(r"\psi = (h - h_0) - T_0\,(s - s_0)")


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Brayton", page_icon="🌀", layout="centered")

st.subheader(SUBJECT)
st.title("🌀 Turbina de gas (ciclo Brayton)")
st.markdown(
    "El aire se comprime, se calienta quemando combustible y se expande en la turbina. "
    "Acá podés sumarle las tres mejoras del ciclo Brayton: **comprimir en etapas con "
    "interenfriamiento**, **expandir en etapas con recalentamiento** (una segunda cámara de "
    "combustión) y un **regenerador** que precalienta el aire con el escape. Mirá cuánto suma "
    "cada una al rendimiento y al trabajo, y dónde se pierde la exergía."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Brayton")
render_units_selector()
system = get_current_system()

ex = _example_index()
note = GAS_TURBINE_EXAMPLE_NOTES.get(_EXAMPLES[ex])
if note:
    st.caption(f"📘 Sobre este ejemplo: {note}")
inputs = _read_inputs(ex)

# Se calcula al elegir un ejemplo y cada vez que se toca el botón.
if st.session_state.get("bt_example_prev") != ex:
    st.session_state["bt_example_prev"] = ex
    st.session_state["bt_inputs"] = inputs
if st.button("Calcular turbina de gas", key="bt_btn", type="primary"):
    st.session_state["bt_inputs"] = inputs

computed: GasTurbineInputs | None = st.session_state.get("bt_inputs")
if computed is not None:
    if computed != inputs:
        st.caption("Cambiaste datos: tocá «Calcular turbina de gas» para actualizar el resultado.")
    try:
        result = _solve_cached(computed)
    except ValueError as exc:
        _render_error(exc)
    else:
        st.markdown("### Resultado")
        _render_metrics(result, system)
        _render_diagram(result, system)
        _render_tables(result, system)
        _render_comparison(computed, system)
        _render_exergy(result, system)
        _render_procedure(result, system)
        _render_tespy(computed, result, system)
        _render_export(result, system)
        _render_sweeps(computed, system)
