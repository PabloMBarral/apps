"""Página 8 — Combustión: estequiometría, humos, llama adiabática, calor y análisis de humos.

Fase 5. El modelo del vademecum §16 (Çengel & Boles, cap. 15 y §16-1 a §16-3),
con los polinomios NASA de McBride, Zehe & Gordon (2002), en dos modos:

- **Combustión**: un combustible (mezcla gaseosa, líquido puro o análisis
  elemental) con aire técnico, aire seco u oxígeno (con la humedad del
  ambiente y, si hace falta, precalentado); la cantidad de aire de seis formas
  (λ, e, % de aire teórico, φ, AC o el O₂ de los humos secos). Resultados: el
  aire, los humos en base húmeda y seca, los puntos de rocío (del agua y
  ácido), el PCS y el PCI, la llama adiabática con combustión completa y con
  disociación (equilibrio químico, verificado con las K_p), el calor con los
  humos a una temperatura dada (con condensación y rendimiento) y el segundo
  principio.
- **Análisis de humos**: λ, el aire, el agua y la pérdida por CO desde el O₂
  (y el CO) medido o desde un Orsat, con el diagrama de combustión.

El cálculo vive en :mod:`core.combustion`; los gráficos, en
:mod:`ui.combustion_charts`.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd
import streamlit as st

from core.combustion.combustion import (
    COMBUSTION_EXAMPLE_NOTES,
    COMBUSTION_EXAMPLES,
    FLUE_GAS_EXAMPLE_NOTES,
    FLUE_GAS_EXAMPLES,
    CombustionInputs,
    CombustionResult,
    FlueGasInputs,
    FlueGasResult,
    air_temperature_sweep,
    combustion_notes,
    combustion_to_dict,
    default_air_temperature_values,
    default_lambda_values,
    default_products_temperature_values,
    flue_gas_notes,
    flue_gas_to_dict,
    lambda_sweep,
    products_temperature_sweep,
    solve_combustion,
    solve_flue_gas,
)
from core.combustion.combustion_procedure import combustion_steps, flue_gas_steps
from core.combustion.fuels import (
    FUEL_KINDS,
    FUELS,
    GAS_COMPONENTS,
    LIQUID_COMPONENTS,
    Fuel,
    FuelKind,
    UltimateAnalysis,
)
from core.combustion.heating_value import CORRELATIONS, estimate_hhv_as_fired
from core.combustion.stoichiometry import (
    AIR_SPEC_KINDS,
    OXIDIZER_KINDS,
    AirSpec,
    AirSpecKind,
    Oxidizer,
    OxidizerKind,
    Stoichiometry,
    dry_gas_curves,
)
from core.export import dict_to_csv
from core.state_report import format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.combustion_charts import (
    combustion_diagram_figure,
    composition_figure,
    energy_split_figure,
    sweep_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.22.0"

# Opciones fijas de los radios y selectores: cambiarlas reiniciaría el widget.
_MODE_COMB = "Combustión: llama y calor"
_MODE_FLUE = "Análisis de humos"
_MODES = (_MODE_COMB, _MODE_FLUE)
_PROC_P = "Presión constante (hogar, cámara)"
_PROC_V = "Volumen constante (recipiente rígido)"
_METHOD_O2 = "Solo O₂ (analizador)"
_METHOD_ORSAT = "Orsat (CO₂, CO y O₂)"
_SW_LAMBDA = "Exceso de aire λ"
_SW_AIR = "Temperatura del comburente (precalentamiento)"
_SW_STACK = "Temperatura de los humos a la salida"

_KIND_LIST: list[FuelKind] = list(FUEL_KINDS)
_KIND_LABELS = [FUEL_KINDS[k] for k in _KIND_LIST]
_OX_LIST: list[OxidizerKind] = list(OXIDIZER_KINDS)
_OX_LABELS = [OXIDIZER_KINDS[k] for k in _OX_LIST]
_SPEC_LIST: list[AirSpecKind] = list(AIR_SPEC_KINDS)
_SPEC_LABELS = [AIR_SPEC_KINDS[k] for k in _SPEC_LIST]
_COMB_NAMES = list(COMBUSTION_EXAMPLES)
_FLUE_NAMES = list(FLUE_GAS_EXAMPLES)

#: De dónde sale el PCS de un análisis elemental: dato o una correlación (Fase 6).
_HHV_KEYS: tuple[str | None, ...] = (None, "channiwala_parikh", "boie", "dulong")
_HHV_SOURCES = (
    "Dato (de laboratorio)",
    *(f"Estimado con {CORRELATIONS[k].name}" for k in _HHV_KEYS if k is not None),
)

#: Datos del análisis elemental: atributo y nombre.
_ANALYSIS_FIELDS = (
    ("C", "Carbono C"),
    ("H", "Hidrógeno H"),
    ("O", "Oxígeno O"),
    ("N", "Nitrógeno N"),
    ("S", "Azufre S"),
    ("W", "Humedad W"),
    ("A", "Cenizas A"),
)

#: Valores de arranque de cada forma de dar el aire (si todavía no hay resultado).
_SPEC_FALLBACK: dict[AirSpecKind, float] = {
    "lambda": 1.2,
    "excess": 0.2,
    "theoretical": 1.2,
    "equivalence": 1.0 / 1.2,
    "AC": 17.0,
    "o2_dry": 0.035,
}

# Productos en el orden de las tablas.
_ORDER = ("CO2", "CO", "H2O", "H2", "SO2", "O2", "N2", "Ar", "OH", "H", "O", "NO", "N")
_KP_LABELS = {
    "CO2": "CO₂ ⇌ CO + ½ O₂",
    "H2O": "H₂O ⇌ H₂ + ½ O₂",
    "NO": "½ N₂ + ½ O₂ ⇌ NO",
}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Paleta de referencia validada para daltonismo (la de ui.combustion_charts).
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"


# ---------------------------------------------------------------------
# Combustibles de los selectores
# ---------------------------------------------------------------------


def _gas_presets() -> dict[str, Fuel]:
    """Mezclas gaseosas: las de la biblioteca y las de los ejemplos."""
    out = {name: fuel for name, fuel in FUELS.items() if fuel.kind == "gas"}
    for inp in (*COMBUSTION_EXAMPLES.values(), *FLUE_GAS_EXAMPLES.values()):
        if inp.fuel.kind == "gas":
            out.setdefault(inp.fuel.name, inp.fuel)
    return out


def _liquid_fuels() -> dict[str, Fuel]:
    """Un combustible por cada líquido puro (con el nombre de la biblioteca si está)."""
    out: dict[str, Fuel] = {}
    for key, label in LIQUID_COMPONENTS.items():
        known = next(
            (f for f in FUELS.values() if f.kind == "liquid" and f.components == ((key, 1.0),)),
            None,
        )
        out[key] = known or Fuel.mixture(label[0].upper() + label[1:], {key: 1.0}, kind="liquid")
    return out


_GAS_PRESETS = _gas_presets()
_GAS_NAMES = list(_GAS_PRESETS)
_LIQUID_FUELS = _liquid_fuels()
_LIQUID_KEYS = list(_LIQUID_FUELS)
_ANALYSIS_PRESETS = {name: fuel for name, fuel in FUELS.items() if fuel.kind == "analysis"}
_ANALYSIS_NAMES = list(_ANALYSIS_PRESETS)


# ---------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------


def _num(value: float | None, sig: int = 5) -> str:
    """Número con ``sig`` cifras; los grandes, enteros con espacio de miles."""
    if value is None:
        return "—"
    if abs(value) >= 10**sig:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, sig)


def _value(value_si: float | None, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    if value_si is None:
        return "—"
    return _num(convert_from_si(value_si, kind, system), sig)


def _fmt(value_si: float | None, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    if value_si is None:
        return "—"
    return f"{_value(value_si, kind, system, sig)} {unit_label(kind, system)}"


def _pct(fraction: float | None, sig: int = 4) -> str:
    return "—" if fraction is None else format_value(100.0 * fraction, sig)


def _formula(key: str) -> str:
    """``CO2`` → ``CO₂``; ``C8H18(l)`` → ``C₈H₁₈(l)``."""
    body, liquid = (key[:-3], "(l)") if key.endswith("(l)") else (key, "")
    return re.sub(r"\d+", lambda m: m.group(0).translate(_SUBSCRIPTS), body) + liquid


def _ratio_unit(system: UnitSystem) -> str:
    return "lb/lb" if system == "Inglés" else "kg/kg"


def _molar_mass_unit(system: UnitSystem) -> str:
    return "lb/lbmol" if system == "Inglés" else "kg/kmol"


def _moles_unit(fuel: Fuel, system: UnitSystem) -> str:
    """Moles por unidad de combustible: kmol/kmol (especies) o kmol/kg (análisis)."""
    if fuel.per_mol:
        return {"SI": "mol/mol", "Técnico": "kmol/kmol", "Inglés": "lbmol/lbmol"}[system]
    return {"SI": "mol/kg", "Técnico": "kmol/kg", "Inglés": "lbmol/lb"}[system]


def _moles(value: float, fuel: Fuel, system: UnitSystem, sig: int = 5) -> str:
    """Moles por unidad en el sistema: por kg van en kmol/kg (o lbmol/lb), salvo en SI."""
    scale = 1.0 if fuel.per_mol or system == "SI" else 1e-3
    return _num(value * scale, sig)


def _basis_name(fuel: Fuel, system: UnitSystem) -> str:
    if fuel.per_mol:
        return {"SI": "mol", "Técnico": "kmol", "Inglés": "lbmol"}[system]
    return "lb" if system == "Inglés" else "kg"


def _p_format(system: UnitSystem) -> str:
    return {"SI": "%.0f", "Técnico": "%.5f", "Inglés": "%.4f"}[system]


def _table(frame: pd.DataFrame) -> None:
    """Tabla con todas las filas a la vista (sin barra de desplazamiento)."""
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _metric_rows(items: list[tuple[str, str]], per_row: int = 3) -> None:
    for k in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, items[k : k + per_row], strict=False):
            col.metric(label, value)


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _solve_cached(inputs: CombustionInputs) -> CombustionResult:
    return solve_combustion(inputs)


@st.cache_data(show_spinner=False)
def _flue_cached(inputs: FlueGasInputs) -> FlueGasResult:
    return solve_flue_gas(inputs)


@st.cache_data(show_spinner=False)
def _curves_cached(
    fuel: Fuel, oxidizer: Oxidizer, p_Pa: float, lam_high: float
) -> dict[str, list[float]]:
    lambdas = [0.5 + (lam_high - 0.5) * k / 160 for k in range(161)]
    return dry_gas_curves(fuel, oxidizer, lambdas, p_Pa=p_Pa)


@st.cache_data(show_spinner="Calculando el barrido…")
def _lambda_sweep_cached(inputs: CombustionInputs) -> dict[str, list[float | None]]:
    return lambda_sweep(inputs, default_lambda_values(inputs), equilibrium=inputs.equilibrium)


@st.cache_data(show_spinner="Calculando el barrido…")
def _air_sweep_cached(inputs: CombustionInputs) -> dict[str, list[float | None]]:
    return air_temperature_sweep(
        inputs, default_air_temperature_values(), equilibrium=inputs.equilibrium
    )


@st.cache_data(show_spinner="Calculando el barrido…")
def _stack_sweep_cached(inputs: CombustionInputs) -> dict[str, list[float | None]]:
    return products_temperature_sweep(inputs, default_products_temperature_values(inputs))


# ---------------------------------------------------------------------
# Entradas comunes
# ---------------------------------------------------------------------


def _example(label: str, names: list[str], key: str) -> int:
    choice = st.selectbox(
        label, names, key=key, help="Podés cambiar cualquier dato: el resultado se actualiza solo."
    )
    return names.index(choice)


def _read_fuel(key: str, default: Fuel, system: UnitSystem) -> Fuel | None:
    """El combustible: mezcla gaseosa, líquido puro o análisis elemental.

    Devuelve ``None`` (con el error a la vista) si los datos no forman un
    combustible válido.
    """
    label = st.radio(
        "Clase de combustible",
        _KIND_LABELS,
        index=_KIND_LIST.index(default.kind),
        horizontal=True,
        key=f"{key}_kind",
        help="Los gases y los líquidos puros se calculan por kmol de combustible (como "
        "Cengel); un análisis elemental (carbón, fueloil, biomasa), por kg.",
    )
    kind = _KIND_LIST[_KIND_LABELS.index(label)]
    try:
        if kind == "gas":
            return _read_gas(key, default)
        if kind == "liquid":
            return _read_liquid(key, default)
        return _read_analysis(key, default, system)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return None


def _read_gas(key: str, default: Fuel) -> Fuel:
    base_name = default.name if default.name in _GAS_PRESETS else _GAS_NAMES[0]
    name = st.selectbox(
        "Gas",
        _GAS_NAMES,
        index=_GAS_NAMES.index(base_name),
        key=f"{key}_gas",
        help="Carga una composición típica; después podés cambiar los componentes y sus "
        "fracciones.",
    )
    base = _GAS_PRESETS[name]
    gkey = f"{key}_gas{_GAS_NAMES.index(name)}"
    comps = dict(base.components)
    selected = st.multiselect(
        "Componentes",
        list(GAS_COMPONENTS),
        default=list(comps),
        format_func=lambda k: GAS_COMPONENTS[k],
        key=f"{gkey}_comp",
    )
    if not selected:
        raise ValueError("Elegí al menos un componente del gas.")
    fractions: dict[str, float] = {}
    for i, comp in enumerate(selected):
        if i % 2 == 0:
            cols = st.columns(2)  # una fila por par: en el celular conserva el orden
        with cols[i % 2]:
            fractions[comp] = (
                st.number_input(
                    f"{GAS_COMPONENTS[comp]} [% molar]",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(100.0 * comps.get(comp, 0.0), 4),
                    step=1.0,
                    format="%.3f",
                    key=f"{gkey}_{comp}",
                )
                / 100.0
            )
    total = sum(fractions.values())
    if total > 0.0 and abs(total - 1.0) > 1e-6:
        st.caption(f"Las fracciones suman {format_value(100 * total, 5)} %: se normalizan a 100 %.")
    same = set(fractions) == set(comps) and all(
        abs(fractions[k] - comps[k]) < 1e-7 for k in fractions
    )
    return base if same else Fuel.mixture("Mezcla gaseosa", fractions)


def _read_liquid(key: str, default: Fuel) -> Fuel:
    default_key = default.components[0][0] if default.kind == "liquid" else "C8H18(l)"
    comp = st.selectbox(
        "Líquido",
        _LIQUID_KEYS,
        index=_LIQUID_KEYS.index(default_key),
        format_func=lambda k: LIQUID_COMPONENTS[k],
        key=f"{key}_liq",
        help="El propano y el butano líquidos (GLP a presión) salen del gas de NASA menos el "
        "calor latente de CoolProp a la temperatura del combustible.",
    )
    return _LIQUID_FUELS[comp]


def _read_analysis(key: str, default: Fuel, system: UnitSystem) -> Fuel:
    base_name = default.name if default.name in _ANALYSIS_PRESETS else _ANALYSIS_NAMES[0]
    name = st.selectbox(
        "Combustible",
        _ANALYSIS_NAMES,
        index=_ANALYSIS_NAMES.index(base_name),
        key=f"{key}_an",
        help="Carga un análisis típico; después podés cambiar cada fracción y el PCS.",
    )
    base = _ANALYSIS_PRESETS[name]
    assert base.analysis is not None and base.hhv_J_per_kg is not None
    akey = f"{key}_an{_ANALYSIS_NAMES.index(name)}"
    st.caption(
        "Fracciones en masa tal cual se quema (con la humedad y las cenizas): tienen que sumar "
        "100 %."
    )
    values: dict[str, float] = {}
    for i, (attr, label) in enumerate(_ANALYSIS_FIELDS):
        if i % 2 == 0:
            cols = st.columns(2)  # una fila por par: en el celular conserva el orden
        with cols[i % 2]:
            values[attr] = (
                st.number_input(
                    f"{label} [% en masa]",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(100.0 * getattr(base.analysis, attr), 4),
                    step=0.1,
                    format="%.3f",
                    key=f"{akey}_{attr}",
                )
                / 100.0
            )
    st.caption(f"Suma: {format_value(100 * sum(values.values()), 6)} %")
    source = st.selectbox(
        "Poder calorífico superior (PCS)",
        _HHV_SOURCES,
        key=f"{akey}_hhvsrc",
        help="Un dato del laboratorio (bomba calorimétrica) o estimado con una correlación "
        "sobre el análisis elemental. La página Poder calorífico las compara y muestra cuánto "
        "se equivoca cada una.",
    )
    correlation = _HHV_KEYS[_HHV_SOURCES.index(source)]
    analysis = UltimateAnalysis(**values)
    # Un error del análisis salta después de dibujar los dos campos: si no, el c_p no se
    # dibuja en esa corrida y Streamlit lo reinicia.
    estimate_error: ValueError | None = None
    left, right = st.columns(2)
    with left:
        if correlation is None:
            hhv = number_input_si(
                label="PCS tal cual se quema",
                kind="specific_enthalpy",
                default_si=base.hhv_J_per_kg,
                key=f"{akey}_hhv",
                format="%.0f" if system == "SI" else "%.1f",
                min_value_si=1e6,
                max_value_si=150e6,
                help="Poder calorífico superior (con el agua de los humos líquida), un dato del "
                "laboratorio (bomba calorimétrica).",
            )
        else:
            try:
                analysis.validate()
                hhv = estimate_hhv_as_fired(analysis, correlation)
            except ValueError as exc:
                estimate_error, hhv = exc, math.nan
            else:
                st.metric(
                    f"PCS tal cual estimado [{unit_label('specific_enthalpy', system)}]",
                    _value(hhv, "specific_enthalpy", system),
                    help="La correlación va con los % en base seca (cada fracción dividida "
                    "por 1 − W) y el resultado se pasa a tal cual: PCS·(1 − W).",
                )
    with right:
        cp = number_input_si(
            label="c_p del combustible",
            kind="specific_heat",
            default_si=base.cp_J_per_kg_K or 1500.0,
            key=f"{akey}_cp",
            format="%.0f" if system == "SI" else "%.3f",
            min_value_si=100.0,
            max_value_si=10_000.0,
            help="Solo cuenta si el combustible entra a otra temperatura que 25 °C (el fueloil "
            "se precalienta para pulverizarlo).",
        )
    if estimate_error is not None:
        raise estimate_error
    same = all(abs(v - getattr(base.analysis, a)) < 1e-9 for a, v in values.items())
    same = same and abs(cp - (base.cp_J_per_kg_K or 1500.0)) < 1e-6 * cp
    if correlation is not None:
        label = name if same else f"{name} (modificado)"
        return Fuel.from_analysis(label, analysis, hhv, cp, hhv_correlation=correlation)
    same = same and abs(hhv - base.hhv_J_per_kg) < 1e-6 * hhv
    return base if same else Fuel.from_analysis(f"{name} (modificado)", analysis, hhv, cp)


def _read_oxidizer(
    key: str, default: Oxidizer, T0_default: float, system: UnitSystem, *, preheat: bool
) -> tuple[Oxidizer, float]:
    """Comburente, temperatura ambiente T₀ y humedad (y el precalentamiento)."""
    label = st.radio(
        "Comburente",
        _OX_LABELS,
        index=_OX_LIST.index(default.kind),
        horizontal=True,
        key=f"{key}_ox",
        help="Aire técnico del vademecum (§16.1: 21 % de O₂ y 79 % de N₂), aire seco de las "
        "tablas de Cengel (con Ar y CO₂) u oxígeno puro (oxicombustión).",
    )
    kind = _OX_LIST[_OX_LABELS.index(label)]
    left, right = st.columns(2)
    with left:
        T0 = number_input_si(
            label="Temperatura ambiente T₀",
            kind="temperature",
            default_si=T0_default,
            key=f"{key}_T0",
            format="%.2f",
            min_value_si=223.15,
            max_value_si=333.15,
            help="La del aire que se toma del ambiente; también es el estado muerto de la exergía.",
        )
    phi = 0.0
    if kind != "oxygen":
        with right:
            phi = (
                st.number_input(
                    "Humedad relativa del aire φ [%]",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(100.0 * default.phi, 2),
                    step=10.0,
                    format="%.1f",
                    key=f"{key}_phi",
                    help="La del aire ambiente, a T₀ (Cengel 15-3). Ese vapor pasa a los humos; "
                    "precalentar el aire no lo cambia.",
                )
                / 100.0
            )
    T_air = T0
    if preheat:
        hot_default = default.T_K > T0_default + 0.01
        hot = st.checkbox(
            "El comburente entra más caliente que el ambiente (precalentado o comprimido)",
            value=hot_default,
            key=f"{key}_hot",
        )
        if hot:
            T_air = number_input_si(
                label="Temperatura del comburente",
                kind="temperature",
                default_si=default.T_K if hot_default else T0_default + 175.0,
                key=f"{key}_Ta",
                format="%.2f",
                min_value_si=223.15,
                max_value_si=1573.15,
                help="Un precalentador de aire (con los mismos humos) sube la llama y ahorra "
                "combustible; en un motor, la mezcla llega comprimida y caliente.",
            )
    humidity_T = T0 if abs(T_air - T0) > 1e-9 else None
    return Oxidizer(kind, T_air, phi, T_humidity_K=humidity_T), T0


def _air_values(stoich: Stoichiometry) -> dict[AirSpecKind, float]:
    """La cantidad de aire del último resultado, en cada forma de darla."""
    lam = stoich.lam
    return {
        "lambda": lam,
        "excess": lam - 1.0,
        "theoretical": lam,
        "equivalence": 1.0 / lam,
        "AC": stoich.AC,
        "o2_dry": stoich.dry_fractions.get("O2", 0.0),
    }


def _read_air(key: str, example: AirSpec, system: UnitSystem) -> AirSpec:
    """La cantidad de aire: λ, e, % teórico, φ, AC u O₂ en humos secos."""
    label = st.selectbox(
        "Cómo se da",
        _SPEC_LABELS,
        index=_SPEC_LIST.index(example.kind),
        key=f"{key}_spec",
        help="Todas dicen lo mismo: λ = a/a_s = 1 + e = 1/φ y AC = λ·AC_s (vademecum §16.2 y "
        "§16.4). En una caldera se mide el O₂ de los humos secos.",
    )
    kind = _SPEC_LIST[_SPEC_LABELS.index(label)]
    last: dict[AirSpecKind, float] | None = st.session_state.get(f"{key}_last")
    if last is not None:
        value = last[kind]
    elif kind == example.kind:
        value = example.value
    else:
        value = _SPEC_FALLBACK[kind]
    widget = f"{key}_air_{kind}"
    if kind == "lambda":
        return AirSpec(
            kind,
            st.number_input(
                "λ = a/a_s",
                min_value=0.3,
                max_value=20.0,
                value=round(value, 5),
                step=0.05,
                format="%.4f",
                key=widget,
            ),
        )
    if kind == "excess":
        e = st.number_input(
            "Exceso de aire e [%]",
            min_value=-70.0,
            max_value=1900.0,
            value=round(100.0 * value, 4),
            step=5.0,
            format="%.2f",
            key=widget,
            help="Negativo: defecto de aire (λ < 1).",
        )
        return AirSpec(kind, e / 100.0)
    if kind == "theoretical":
        t = st.number_input(
            "Aire teórico [%]",
            min_value=30.0,
            max_value=2000.0,
            value=round(100.0 * value, 4),
            step=5.0,
            format="%.2f",
            key=widget,
            help="100 % es el estequiométrico; 150 %, 50 % de exceso (Cengel).",
        )
        return AirSpec(kind, t / 100.0)
    if kind == "equivalence":
        return AirSpec(
            kind,
            st.number_input(
                "Relación de equivalencia φ = 1/λ",
                min_value=0.05,
                max_value=3.3,
                value=round(value, 5),
                step=0.05,
                format="%.4f",
                key=widget,
                help="φ < 1: mezcla pobre (sobra aire); φ > 1: rica.",
            ),
        )
    if kind == "AC":
        return AirSpec(
            kind,
            st.number_input(
                f"AC (aire seco) [{_ratio_unit(system)}]",
                min_value=0.01,
                max_value=2000.0,
                value=round(value, 5),
                step=0.5,
                format="%.4f",
                key=widget,
                help="Masa de aire seco por unidad de masa de combustible (el mismo número en "
                "kg/kg y en lb/lb).",
            ),
        )
    o2 = st.number_input(
        "O₂ en los humos secos [%]",
        min_value=0.0,
        max_value=20.9,
        value=round(100.0 * value, 4),
        step=0.5,
        format="%.3f",
        key=widget,
        help="Lo que mide el analizador de la chimenea (en base seca: el agua condensa antes "
        "de la celda).",
    )
    return AirSpec(kind, o2 / 100.0)


# ---------------------------------------------------------------------
# Modo 1: combustión
# ---------------------------------------------------------------------


def _render_combustion_mode(system: UnitSystem) -> None:
    ex = _example("Ejemplo precargado", _COMB_NAMES, "cb_example")
    name = _COMB_NAMES[ex]
    example = COMBUSTION_EXAMPLES[name]
    key = f"cb_{ex}"

    st.markdown("#### Combustible")
    fuel = _read_fuel(key, example.fuel, system)
    st.markdown("#### Comburente")
    oxidizer, T0 = _read_oxidizer(key, example.oxidizer, example.T0_K, system, preheat=True)
    st.markdown("#### Cantidad de aire")
    air = _read_air(key, example.air, system)
    has_carbon = fuel is None or fuel.elements()["C"] > 0.0
    co_fraction = 0.0
    if has_carbon:
        co_fraction = (
            st.number_input(
                "Carbono que sale como CO [%]",
                min_value=0.0,
                max_value=99.0,
                value=round(100.0 * example.co_fraction, 4),
                step=1.0,
                format="%.2f",
                key=f"{key}_co",
                help="Combustión incompleta con oxígeno de sobra (Cengel 15-6). Con defecto de "
                "aire (λ < 1) no se usa: el CO sale del balance.",
            )
            / 100.0
        )

    st.markdown("#### Condiciones")
    process_label = st.radio(
        "Proceso",
        (_PROC_P, _PROC_V),
        index=0 if example.process == "p" else 1,
        horizontal=True,
        key=f"{key}_proc",
        help="A presión constante: un hogar, una cámara de combustión (flujo). A volumen "
        "constante: una bomba calorimétrica o el ciclo Otto ideal (Cengel 15-7).",
    )
    process = "p" if process_label == _PROC_P else "v"
    left, right = st.columns(2)
    with left:
        p = number_input_si(
            label="Presión de los reactivos",
            kind="pressure",
            default_si=example.p_Pa,
            key=f"{key}_p",
            format=_p_format(system),
            min_value_si=0.1e5,
            max_value_si=100e5,
            help="A presión constante es la de toda la combustión; a volumen constante, la "
            "inicial (la final sale del cálculo).",
        )
    with right:
        T_fuel = number_input_si(
            label="Temperatura del combustible",
            kind="temperature",
            default_si=example.T_fuel_K,
            key=f"{key}_Tf",
            format="%.2f",
            min_value_si=223.15,
            max_value_si=1273.15,
        )

    st.markdown("#### Calor, caudal y equilibrio")
    T_out = T_sink = None
    heat = st.checkbox(
        "Calcular el calor que entregan los humos al enfriarse hasta T_s",
        value=example.T_products_K is not None,
        key=f"{key}_heat",
    )
    if heat:
        left, right = st.columns(2)
        with left:
            T_out = number_input_si(
                label="Temperatura de los humos a la salida T_s",
                kind="temperature",
                default_si=example.T_products_K or 423.15,
                key=f"{key}_Ts",
                format="%.2f",
                min_value_si=273.16,
                max_value_si=3000.0,
                help="La de la chimenea de una caldera (o la de los productos en Cengel 15-6 y "
                "15-7). Bajo el punto de rocío condensa parte del agua (caldera de "
                "condensación).",
            )
        with right:
            T_sink = number_input_si(
                label="Temperatura a la que se entrega el calor T_b",
                kind="temperature",
                default_si=example.T_sink_K or example.T0_K,
                key=f"{key}_Tb",
                format="%.2f",
                min_value_si=200.0,
                max_value_si=3000.0,
                help="La del agua o el vapor que recibe el calor (para la exergía). Si va al "
                "ambiente, T_b = T₀ (Cengel 15-11).",
            )
    flow = st.checkbox(
        "Dar el caudal de combustible (para los totales)",
        value=example.m_fuel_kg_s is not None,
        key=f"{key}_flow",
    )
    m_fuel = None
    if flow:
        m_fuel = number_input_si(
            label="Caudal de combustible",
            kind="mass_flow",
            default_si=example.m_fuel_kg_s or 0.01,
            key=f"{key}_m",
            format="%.6f",
            min_value_si=1e-7,
            max_value_si=1e4,
        )
    has_sulfur = fuel is not None and fuel.elements()["S"] > 0.0
    so3 = example.so3_conversion
    if has_sulfur:
        so3 = (
            st.number_input(
                "SO₂ que pasa a SO₃ [%]",
                min_value=0.0,
                max_value=20.0,
                value=round(100.0 * example.so3_conversion, 4),
                step=0.5,
                format="%.2f",
                key=f"{key}_so3",
                help="Del 1 al 5 % en una caldera (más con vanadio en las cenizas): el SO₃ con "
                "el agua forma ácido sulfúrico y fija el rocío ácido.",
            )
            / 100.0
        )
    equilibrium = st.checkbox(
        "Calcular también la llama con disociación (equilibrio químico)",
        value=example.equilibrium,
        key=f"{key}_eq",
        help="A más de ~1800 K parte del CO₂ y del H₂O se separa (en CO, H₂, OH, O, H…) y la "
        "llama queda más fría (Cengel §16-2).",
    )
    if fuel is None:
        return
    inputs = CombustionInputs(
        fuel=fuel,
        oxidizer=oxidizer,
        air=air,
        co_fraction=co_fraction,
        p_Pa=p,
        T_fuel_K=T_fuel,
        process=process,  # type: ignore[arg-type]
        T_products_K=T_out,
        T_sink_K=T_sink,
        T0_K=T0,
        m_fuel_kg_s=m_fuel,
        so3_conversion=so3,
        equilibrium=equilibrium,
    )
    try:
        result = _solve_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.session_state[f"{key}_last"] = _air_values(result.stoich)

    st.markdown("### Resultado")
    note = COMBUSTION_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    _render_main_metrics(result, system)
    for item in combustion_notes(result):
        st.info(item)
    _render_air_and_products(result, system)
    _render_flame(result, system)
    if result.heat is not None:
        _render_heat(result, system)
    if result.second_law is not None:
        _render_second_law(result, system)
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Las fórmulas del vademecum §16 con los números reemplazados, en el sistema de "
            "unidades elegido. Las entalpías y entropías de cada especie salen de los "
            "polinomios NASA (McBride et al., 2002): son las de las tablas A-18 a A-26 de "
            "Cengel con más decimales."
        )
        for step in combustion_steps(result, system):
            st.markdown(f"**{step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)
    _render_export(combustion_to_dict(result, system), "combustion", "cb")
    _render_sweeps(result, system)


def _render_main_metrics(result: CombustionResult, system: UnitSystem) -> None:
    s = result.stoich
    T = unit_label("temperature", system)
    E = unit_label("specific_enthalpy", system)
    eq = result.flame_eq
    items = [
        ("λ", format_value(s.lam, 4)),
        (f"AC [{_ratio_unit(system)}]", format_value(s.AC, 4)),
        (f"PCI [{E}]", _value(result.lhv * result.per_kg, "specific_enthalpy", system)),
        (f"T_ad completa [{T}]", _value(result.flame.T_K, "temperature", system, 4)),
        (
            f"T_ad con disociación [{T}]",
            _value(eq.T_K if eq else None, "temperature", system, 4),
        ),
        (f"T_pr del agua [{T}]", _value(s.dew_point_K, "temperature", system, 4)),
    ]
    if result.inputs.process == "v":
        P = unit_label("pressure", system)
        p_end = _value(result.flame.p_Pa, "pressure", system)
        items.append((f"p al final de la llama [{P}]", p_end))
    heat = result.heat
    if heat is not None:
        if heat.split is not None:
            items.append(("η sobre el PCI [%]", _pct(heat.eta_lhv)))
        total = result.total(heat.Q_out)
        if total is not None:
            items.append((f"Q̇ [{unit_label('power', system)}]", _value(total, "power", system)))
    _metric_rows(items)


def _render_air_and_products(result: CombustionResult, system: UnitSystem) -> None:
    s = result.stoich
    fuel = result.fuel
    st.markdown("#### Aire y humos")
    st.caption(
        f"Por {_basis_name(fuel, system)} de combustible. Base húmeda: con el agua de los "
        "humos; seca: sin ella, como la mide un analizador (vademecum §16.3)."
    )
    _render_stoich_table(result, system)
    _render_products_table(s, system)
    try:
        st.plotly_chart(composition_figure(s), width="stretch", key="cb_composition")
    except Exception as exc:  # un gráfico no debe tumbar la página
        st.warning(f"No se pudo dibujar el gráfico: {exc}")
    if fuel.elements()["C"] > 0.0:
        _render_diagram(s, s.lam, s.dry_fractions, "tus datos", "cb_diagram")


def _render_stoich_table(result: CombustionResult, system: UnitSystem) -> None:
    s = result.stoich
    fuel = result.fuel
    mol = _moles_unit(fuel, system)
    ratio = _ratio_unit(system)
    T = unit_label("temperature", system)
    rows: list[tuple[str, str, str, str]] = [
        ("O₂,t", "oxígeno teórico", _moles(s.o2_theoretical, fuel, system), mol),
        ("a_s", "comburente teórico", _moles(s.air_theoretical, fuel, system), mol),
        ("a", "comburente", _moles(s.air_moles, fuel, system), mol),
        ("λ", "relación de aire a/a_s", format_value(s.lam, 5), "—"),
        ("e", "exceso de aire", _pct(s.excess, 5), "%"),
        ("φ", "relación de equivalencia 1/λ", format_value(s.equivalence_ratio, 5), "—"),
        ("AC_s", "aire seco estequiométrico", format_value(s.AC_s, 5), ratio),
        ("AC", "aire seco", format_value(s.AC, 5), ratio),
        ("GC", "humos húmedos por masa de combustible", format_value(s.GC, 5), ratio),
        ("n_g", "humos húmedos", _moles(s.n_gas, fuel, system), mol),
        ("n_g,s", "humos secos", _moles(s.n_dry, fuel, system), mol),
        (
            "M_g",
            "masa molar de los humos",
            format_value(1e3 * s.M_gas, 5),
            _molar_mass_unit(system),
        ),
        (
            "V_N",
            "humos húmedos a 0 °C y 1 atm (por kg)",
            _value(s.normal_volume, "specific_volume", system),
            unit_label("specific_volume", system),
        ),
        (
            "V_N,s",
            "humos secos a 0 °C y 1 atm (por kg)",
            _value(s.normal_volume_dry, "specific_volume", system),
            unit_label("specific_volume", system),
        ),
    ]
    if fuel.elements()["C"] > 0.0 and s.oxidizer.kind != "oxygen":
        rows.append(("CO₂,máx", "CO₂ seco con λ = 1", _pct(s.co2_max), "%"))
    dew = _value(s.dew_point_K, "temperature", system)
    rows.append(("T_pr", "punto de rocío del agua", dew, T))
    acid = s.acid_dew_point_K(result.inputs.so3_conversion)
    if acid is not None:
        rows.append(("T_pr,a", "punto de rocío ácido", _value(acid, "temperature", system), T))
    frame = pd.DataFrame(rows, columns=["Símbolo", "Magnitud", "Valor", "Unidad"])
    frame = frame[["Símbolo", "Valor", "Unidad", "Magnitud"]]
    _table(frame)


def _render_products_table(s: Stoichiometry, system: UnitSystem) -> None:
    fuel = s.fuel
    wet, dry, mass = s.wet_fractions, s.dry_fractions, s.mass_fractions
    mol = _moles_unit(fuel, system)
    rows = [
        {
            "Especie": _formula(k),
            f"n [{mol}]": _moles(n, fuel, system),
            "húmedos [%]": _pct(wet.get(k)),
            "secos [%]": "—" if k == "H2O" else _pct(dry.get(k)),
            "en masa [%]": _pct(mass.get(k)),
        }
        for k, n in ((k, s.products[k]) for k in _ORDER if s.products.get(k, 0.0) > 0.0)
    ]
    _table(pd.DataFrame(rows))


def _render_diagram(
    s: Stoichiometry, lam: float, point: dict[str, float], label: str, chart_key: str
) -> None:
    st.markdown("##### Diagrama de combustión")
    try:
        curves = _curves_cached(s.fuel, s.oxidizer, s.p_Pa, max(2.5, 1.25 * lam))
        fig = combustion_diagram_figure(curves, point_lambda=lam, point=point, point_label=label)
        st.plotly_chart(fig, width="stretch", key=chart_key)
    except Exception as exc:  # un gráfico no debe tumbar la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
        return
    st.caption(
        "CO₂, O₂ y CO en los humos secos en función de λ (combustión completa con λ ≥ 1; con "
        "λ < 1, la regla de Cengel 15-8 c). El CO₂ es máximo con λ = 1 y el O₂ crece con el "
        "exceso de aire: con uno de los dos medido se sabe λ."
    )


def _render_flame(result: CombustionResult, system: UnitSystem) -> None:
    st.markdown("#### Llama adiabática")
    flame, eq = result.flame, result.flame_eq
    text = (
        f"Con combustión completa los humos llegan a {_fmt(flame.T_K, 'temperature', system, 5)} "
        "(vademecum §16.8)"
    )
    if eq is not None:
        text += (
            f"; con disociación, a {_fmt(eq.T_K, 'temperature', system, 5)}: en equilibrio "
            "químico (mínima energía libre de Gibbs, como NASA CEA) parte del CO₂ y del H₂O se "
            "separa y eso absorbe energía."
        )
    else:
        text += "."
    if result.inputs.process == "v":
        p0 = _fmt(result.inputs.p_Pa, "pressure", system)
        text += (
            f" A volumen constante la presión sube de {p0} a {_fmt(flame.p_Pa, 'pressure', system)}"
        )
        text += f" ({_fmt(eq.p_Pa, 'pressure', system)} con disociación)." if eq else "."
    st.markdown(text)
    y_c = flame.fractions
    y_e = eq.fractions if eq else {}
    keys = [k for k in _ORDER if y_c.get(k, 0.0) > 0.0 or y_e.get(k, 0.0) >= 1e-6]
    rows = []
    for k in keys:
        row = {"Especie": _formula(k), "completa [%]": _pct(y_c.get(k)) if k in y_c else "—"}
        if eq is not None:
            row["con disociación [%]"] = _pct(y_e.get(k)) if k in y_e else "—"
        rows.append(row)
    _table(pd.DataFrame(rows))
    if eq is not None:
        st.caption(
            "Fracciones molares de los humos en la llama (base húmeda). En equilibrio aparecen "
            "CO, H₂, OH, O, H y NO aun con aire de sobra; al enfriarse vuelven a CO₂, H₂O, O₂ "
            "y N₂ (salvo el NO, que queda congelado por la cinética)."
        )
    if result.kp_checks:
        st.markdown("##### Verificación con las constantes de equilibrio")
        rows_kp = [
            {
                "Reacción": _KP_LABELS.get(k.key, k.key),
                "ln K_p (tablas)": format_value(k.ln_Kp_tables, 5),
                "ln K_p (composición)": format_value(k.ln_Kp_composition, 5),
            }
            for k in result.kp_checks
        ]
        _table(pd.DataFrame(rows_kp))
        st.caption(
            "ln K_p = −ΔG°(T)/(R_u·T) con los g° de los polinomios, contra "
            "K_p = Π (y_i·p/p°)^ν_i con la composición calculada (Cengel §16-2, p° = 1 bar): "
            "si coinciden, la mezcla está en equilibrio."
        )


def _render_heat(result: CombustionResult, system: UnitSystem) -> None:
    heat = result.heat
    assert heat is not None
    st.markdown("#### Calor con los humos a T_s")
    E = unit_label("specific_enthalpy", system)
    fuel = result.fuel
    items = []
    if fuel.per_mol:
        items.append(
            (
                f"Calor por {_basis_name(fuel, system)} [{unit_label('molar_enthalpy', system)}]",
                _value(heat.Q_out, "molar_enthalpy", system),
            )
        )
    q_kg = _value(heat.Q_out * result.per_kg, "specific_enthalpy", system)
    items.append((f"Calor por kg [{E}]", q_kg))
    if heat.split is not None:
        items.append(("η sobre el PCS [%]", _pct(heat.eta_hhv)))
    water = result.stoich.products.get("H2O", 0.0)
    items.append(("Agua que condensa [%]", _pct(heat.condensed / water if water > 0 else 0.0)))
    _metric_rows(items)
    if heat.split is None:
        st.caption(
            "A volumen constante el calor sale del balance de energía interna, "
            "Q = U_r − U_p(T_s) (Cengel 15-7); la presión final es "
            f"{_fmt(heat.p_Pa, 'pressure', system)}."
        )
        return
    try:
        st.plotly_chart(
            energy_split_figure(heat.split, result.per_kg, system),
            width="stretch",
            key="cb_energy",
        )
    except Exception as exc:  # un gráfico no debe tumbar la página
        st.warning(f"No se pudo dibujar el gráfico: {exc}")
    st.caption(
        "Del PCI al calor útil, por kg de combustible: suma el calor sensible con el que entran "
        "los reactivos (si están a más de 25 °C), se pierde lo que no se quemó (CO, H₂) y el "
        "calor sensible de los humos a T_s (con el agua como vapor), y se recupera el latente "
        "del agua que condensa. Los porcentajes son sobre el PCI."
    )


def _render_second_law(result: CombustionResult, system: UnitSystem) -> None:
    sl = result.second_law
    assert sl is not None
    st.markdown("#### Segundo principio")
    S = unit_label("molar_entropy", system)
    H = unit_label("molar_enthalpy", system)
    items = [
        (f"S_gen adiabática [{S}]", _value(sl.S_gen_adiabatic, "molar_entropy", system)),
        (f"X_dest adiabática [{H}]", _value(sl.X_dest_adiabatic, "molar_enthalpy", system)),
        ("X_dest / PCI [%]", _pct(sl.X_dest_adiabatic / sl.lhv)),
    ]
    if sl.S_gen_heat is not None:
        items += [
            (f"S_gen con el calor [{S}]", _value(sl.S_gen_heat, "molar_entropy", system)),
            (f"X_dest con el calor [{H}]", _value(sl.X_dest_heat, "molar_enthalpy", system)),
            (f"X del calor a T_b [{H}]", _value(sl.X_heat, "molar_enthalpy", system)),
        ]
    _metric_rows(items)
    text = (
        f"Por {_basis_name(result.fuel, system)} de combustible, con T₀ = "
        f"{_fmt(sl.T0_K, 'temperature', system, 4)}. La exergía del combustible es casi su PCI "
        "(vademecum §16.13): la combustión adiabática destruye una buena parte porque convierte "
        "energía química en calor a una temperatura finita (Cengel 15-10)."
    )
    if sl.T_sink_K is not None and sl.X_heat is not None:
        text += (
            f" Con el calor entregado a T_b = {_fmt(sl.T_sink_K, 'temperature', system, 4)} queda "
            f"útil el {_pct(sl.X_heat / sl.lhv, 3)} % de la exergía del combustible "
            "(Cengel 15-11): cuanto más baja T_b, menos vale ese calor."
        )
    st.caption(text)


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------


def _render_sweeps(result: CombustionResult, system: UnitSystem) -> None:
    inputs = result.inputs
    with st.expander("📊 ¿Cómo cambia? (barridos)", expanded=False):
        choice = st.selectbox(
            "Variable",
            (_SW_LAMBDA, _SW_AIR, _SW_STACK),
            key="cb_sweep_kind",
            help="El resto de los datos queda igual.",
        )
        if choice == _SW_STACK and inputs.process != "p":
            st.caption(
                "El rendimiento con la temperatura de los humos es para una combustión a "
                "presión constante (un hogar o una caldera)."
            )
            return
        if st.button("Calcular barrido", key="cb_sweep_btn"):
            st.session_state["cb_sweep"] = (inputs, choice)
        if st.session_state.get("cb_sweep") != (inputs, choice):
            st.caption("Tocá «Calcular barrido» (resuelve la combustión unas 40 veces).")
            return
        if choice == _SW_LAMBDA:
            _render_lambda_sweep(result, system)
        elif choice == _SW_AIR:
            _render_air_sweep(inputs, system)
        else:
            _render_stack_sweep(inputs, system)


def _temps(values: list[float | None], system: UnitSystem) -> list[float | None]:
    return [None if v is None else convert_from_si(v, "temperature", system) for v in values]


def _percent(values: list[float | None]) -> list[float | None]:
    return [None if v is None else 100.0 * v for v in values]


def _render_lambda_sweep(result: CombustionResult, system: UnitSystem) -> None:
    inputs = result.inputs
    sw = _lambda_sweep_cached(inputs)
    lam = [float(v) for v in sw["lambda"] if v is not None]
    T = unit_label("temperature", system)
    flame = {"completa": _temps(sw["T_ad"], system)}
    if inputs.equilibrium:
        flame["con disociación"] = _temps(sw["T_ad_eq"], system)
    charts: list[tuple[str, str, dict[str, list[float | None]], tuple[str, ...]]] = [
        ("Temperatura adiabática de llama", f"T_ad [{T}]", flame, (_ORANGE, _BLUE)),
        (
            "Humos secos",
            "fracción molar [%]",
            {
                "CO₂": _percent(sw["CO2_dry"]),
                "O₂": _percent(sw["O2_dry"]),
                "CO": _percent(sw["CO_dry"]),
            },
            (_BLUE, _ORANGE, _AQUA),
        ),
        ("Relación aire–combustible", f"AC [{_ratio_unit(system)}]", {"AC": sw["AC"]}, (_BLUE,)),
        ("Punto de rocío del agua", f"T_pr [{T}]", {"T_pr": _temps(sw["T_dew"], system)}, (_BLUE,)),
    ]
    if any(v is not None for v in sw["eta_lhv"]):
        charts.append(
            ("Rendimiento sobre el PCI", "η [%]", {"η": _percent(sw["eta_lhv"])}, (_BLUE,))
        )
    for k, (title, y_title, series, colors) in enumerate(charts):
        fig = sweep_figure(
            lam,
            series,
            title=title,
            x_title="λ",
            y_title=y_title,
            current_x=result.stoich.lam,
            colors=colors,
        )
        st.plotly_chart(fig, width="stretch", key=f"cb_sweep_lam_{k}")
    st.caption(
        "La llama es máxima cerca de λ = 1: con menos aire falta oxígeno (CO y H₂ sin quemar) y "
        "con más, el aire de sobra se calienta. Con disociación el pico se achata. El "
        "rendimiento cae con el exceso: son más humos que se van calientes por la chimenea. "
        "La línea punteada marca tus datos."
    )
    _skipped(len(default_lambda_values(inputs)), len(lam))


def _render_air_sweep(inputs: CombustionInputs, system: UnitSystem) -> None:
    sw = _air_sweep_cached(inputs)
    T = unit_label("temperature", system)
    xs = [float(x) for x in _temps(sw["T_air"], system) if x is not None]
    series = {"completa": _temps(sw["T_ad"], system)}
    if inputs.equilibrium:
        series["con disociación"] = _temps(sw["T_ad_eq"], system)
    st.plotly_chart(
        sweep_figure(
            xs,
            series,
            title="Temperatura adiabática de llama",
            x_title=f"T del comburente [{T}]",
            y_title=f"T_ad [{T}]",
            current_x=convert_from_si(inputs.oxidizer.T_K, "temperature", system),
            colors=(_ORANGE, _BLUE),
        ),
        width="stretch",
        key="cb_sweep_air",
    )
    st.caption(
        "Precalentar el aire sube la llama casi lo mismo que se lo calienta (un poco menos: "
        "los humos tienen más masa y más c_p que el aire). Con disociación la ganancia es "
        "menor: cuanto más caliente, más se disocia. La humedad del aire queda la del ambiente. "
        "La línea punteada marca tus datos."
    )
    _skipped(len(default_air_temperature_values()), len(xs))


def _render_stack_sweep(inputs: CombustionInputs, system: UnitSystem) -> None:
    sw = _stack_sweep_cached(inputs)
    T = unit_label("temperature", system)
    xs = [float(x) for x in _temps(sw["T_out"], system) if x is not None]
    current = (
        convert_from_si(inputs.T_products_K, "temperature", system)
        if inputs.T_products_K is not None
        else None
    )
    st.plotly_chart(
        sweep_figure(
            xs,
            {"sobre el PCI": _percent(sw["eta_lhv"]), "sobre el PCS": _percent(sw["eta_hhv"])},
            title="Rendimiento",
            x_title=f"T_s [{T}]",
            y_title="η [%]",
            current_x=current,
            colors=(_BLUE, _ORANGE),
        ),
        width="stretch",
        key="cb_sweep_stack_eta",
    )
    st.plotly_chart(
        sweep_figure(
            xs,
            {"condensa": _percent(sw["condensed_fraction"])},
            title="Agua de los humos que condensa",
            x_title=f"T_s [{T}]",
            y_title="del agua formada [%]",
            current_x=current,
            colors=(_AQUA,),
        ),
        width="stretch",
        key="cb_sweep_stack_water",
    )
    st.caption(
        "Enfriar más los humos saca más calor. Bajo el punto de rocío el agua empieza a "
        "condensar y devuelve su latente: el rendimiento sube más rápido (caldera de "
        "condensación) y sobre el PCI puede pasar el 100 %. La línea punteada marca tus datos."
    )
    _skipped(len(default_products_temperature_values(inputs)), len(xs))


def _skipped(total: int, done: int) -> None:
    if done < total:
        st.caption(
            f"{total - done} valor(es) se omiten: con esos datos la combustión no tiene sentido "
            "físico (hollín, humos más calientes que la llama, aire que condensa…)."
        )


# ---------------------------------------------------------------------
# Modo 2: análisis de humos
# ---------------------------------------------------------------------


def _render_flue_mode(system: UnitSystem) -> None:
    ex = _example("Ejemplo precargado", _FLUE_NAMES, "fg_example")
    name = _FLUE_NAMES[ex]
    example = FLUE_GAS_EXAMPLES[name]
    key = f"fg_{ex}"

    st.markdown("#### Combustible")
    fuel = _read_fuel(key, example.fuel, system)
    st.markdown("#### Comburente")
    oxidizer, _ = _read_oxidizer(key, example.oxidizer, example.oxidizer.T_K, system, preheat=False)
    st.markdown("#### Humos secos medidos")
    method_label = st.radio(
        "Medición",
        (_METHOD_O2, _METHOD_ORSAT),
        index=0 if example.method == "o2" else 1,
        horizontal=True,
        key=f"{key}_method",
        help="Un analizador de caldera mide el O₂ (y el CO, en ppm); el Orsat absorbe por "
        "turno el CO₂ (con el SO₂), el O₂ y el CO, y el N₂ queda por diferencia.",
    )
    co2 = 0.0
    if method_label == _METHOD_O2:
        method = "o2"
        left, right = st.columns(2)
        with left:
            o2 = (
                st.number_input(
                    "O₂ [%]",
                    min_value=0.0,
                    max_value=20.9,
                    value=round(100.0 * example.o2_dry, 4),
                    step=0.1,
                    format="%.2f",
                    key=f"{key}_o2",
                )
                / 100.0
            )
        with right:
            co = (
                st.number_input(
                    "CO [ppm]",
                    min_value=0.0,
                    max_value=99_999.0,
                    value=round(1e6 * example.co_dry, 2),
                    step=10.0,
                    format="%.1f",
                    key=f"{key}_coppm",
                )
                / 1e6
            )
    else:
        method = "orsat"
        cols = st.columns(3)
        with cols[0]:
            co2 = (
                st.number_input(
                    "CO₂ [%]",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(100.0 * (example.co2_dry or 0.12), 4),
                    step=0.1,
                    format="%.2f",
                    key=f"{key}_co2",
                )
                / 100.0
            )
        with cols[1]:
            co = (
                st.number_input(
                    "CO [%]",
                    min_value=0.0,
                    max_value=10.0,
                    value=round(100.0 * example.co_dry, 4),
                    step=0.01,
                    format="%.3f",
                    key=f"{key}_co",
                )
                / 100.0
            )
        with cols[2]:
            o2 = (
                st.number_input(
                    "O₂ [%]",
                    min_value=0.0,
                    max_value=21.0,
                    value=round(100.0 * example.o2_dry, 4),
                    step=0.1,
                    format="%.2f",
                    key=f"{key}_o2orsat",
                )
                / 100.0
            )
        st.caption(
            f"N₂ por diferencia: {format_value(100 * (1 - co2 - co - o2), 5)} %. El CO₂ del "
            "Orsat incluye el SO₂ (se absorben en la misma pipeta)."
        )
    left, right = st.columns(2)
    with left:
        p = number_input_si(
            label="Presión de los humos",
            kind="pressure",
            default_si=example.p_Pa,
            key=f"{key}_p",
            format=_p_format(system),
            min_value_si=0.1e5,
            max_value_si=100e5,
        )
    with right:
        T_cool = number_input_si(
            label="Temperatura de enfriamiento",
            kind="temperature",
            default_si=example.T_cool_K,
            key=f"{key}_Tc",
            format="%.2f",
            min_value_si=273.16,
            max_value_si=600.0,
            help="Para ver cuánta agua condensa si los humos se enfrían hasta esa temperatura "
            "(Cengel 15-4 c).",
        )
    if fuel is None:
        return
    inputs = FlueGasInputs(
        fuel=fuel,
        oxidizer=oxidizer,
        method=method,  # type: ignore[arg-type]
        o2_dry=o2,
        co_dry=co,
        co2_dry=co2,
        p_Pa=p,
        T_cool_K=T_cool,
    )
    try:
        result = _flue_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return

    st.markdown("### Resultado")
    note = FLUE_GAS_EXAMPLE_NOTES.get(name)
    if note:
        st.caption(f"📘 Sobre este ejemplo: {note}")
    s = result.stoich
    water = s.products.get("H2O", 0.0)
    _metric_rows(
        [
            ("λ", format_value(s.lam, 4)),
            ("Aire teórico [%]", _pct(s.lam)),
            (f"AC [{_ratio_unit(system)}]", format_value(s.AC, 4)),
            ("Exceso de aire e [%]", _pct(s.excess)),
            ("Pérdida por CO [% del PCI]", _pct(result.unburned_fraction, 3)),
            ("Agua que condensa [%]", _pct(result.condensed / water if water > 0 else 0.0)),
        ]
    )
    for item in flue_gas_notes(result):
        st.info(item)
    if result.orsat is not None:
        _render_orsat(result, system)
    st.markdown("#### Humos")
    st.caption(f"Los que corresponden a esa λ, por {_basis_name(fuel, system)} de combustible.")
    _render_products_table(s, system)
    if fuel.elements()["C"] > 0.0:
        if result.orsat is not None:
            point = {"CO2": inputs.co2_dry, "O2": inputs.o2_dry, "CO": inputs.co_dry}
        else:
            point = s.dry_fractions
        _render_diagram(s, s.lam, point, "medido", "fg_diagram")
    with st.expander("🔬 Procedimiento", expanded=False):
        st.caption(
            "Los balances de cada elemento con los números reemplazados (Cengel 15-4), en el "
            "sistema de unidades elegido."
        )
        for step in flue_gas_steps(result, system):
            st.markdown(f"**{step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)
    _render_export(flue_gas_to_dict(result, system), "analisis_de_humos", "fg")


def _render_orsat(result: FlueGasResult, system: UnitSystem) -> None:
    o = result.orsat
    assert o is not None
    fuel = result.inputs.fuel
    st.markdown("#### Balances del Orsat")
    unit = _basis_name(fuel, system)
    mol = {"SI": "mol", "Técnico": "kmol", "Inglés": "lbmol"}[system]
    # Por kg: kg de combustible cada 100 kmol de humos (o lb cada 100 lbmol), salvo en SI.
    x = o.fuel_per_100 * (1.0 if fuel.per_mol or system == "SI" else 1e3)
    rows = [
        ("x", _num(x, 5), f"combustible (balance de C) [{unit}]"),
        ("a", _num(o.air_per_100, 5), f"comburente (balance de N₂) [{mol}]"),
        ("n_H₂O", _num(o.water_per_100, 5), f"agua (balance de H) [{mol}]"),
        ("ΔO", _num(o.oxygen_residual, 4), f"residuo del balance de O [{mol}]"),
        ("ΔO/O", _pct(o.oxygen_residual_relative, 3), "residuo relativo [%]"),
    ]
    _table(pd.DataFrame(rows, columns=["Símbolo", "Valor", "Magnitud"]))
    st.caption(
        f"Cada 100 {mol} de humos secos. Si el residuo del balance de oxígeno es chico, la "
        "medición es coherente con el combustible."
    )


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


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

#: Tabla de §16.12 del vademecum (Klein & Nellis, 2012, sobre McBride et al., 2002).
_VADEMECUM_TABLE = (
    ("C (grafito)", "sól.", "0", "5.734", "12.00"),
    ("H₂", "gas", "0", "130.673", "2.016"),
    ("O₂", "gas", "0", "205.137", "32.00"),
    ("N₂", "gas", "0", "191.598", "28.01"),
    ("CO", "gas", "−110 528", "197.648", "28.01"),
    ("CO₂", "gas", "−393 486", "213.774", "44.01"),
    ("H₂O", "gas", "−241 811", "188.818", "18.02"),
    ("H₂O", "líq.", "−285 813", "69.938", "18.02"),
    ("SO₂", "gas", "−296 792", "248.207", "64.06"),
    ("CH₄ (metano)", "gas", "−74 595", "186.360", "16.04"),
    ("C₂H₆ (etano)", "gas", "−83 846", "229.207", "30.07"),
    ("C₃H₈ (propano)", "gas", "−104 674", "270.298", "44.10"),
    ("C₈H₁₈ (n-octano)", "líq.", "−250 302", "360.833", "114.20"),
    ("CH₃OH (metanol)", "líq.", "−239 004", "126.976", "32.04"),
    ("C₂H₅OH (etanol)", "líq.", "−277 402", "159.206", "46.07"),
)


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§16 *Combustión*** (§16.1 aire técnico; §16.2 estequiometría; §16.3 fracciones "
            "de los humos; §16.4 y §16.5 AC y GC; §16.6 a §16.10 entalpía, primer principio, "
            "llama adiabática y poder calorífico; §16.11 entropía; §16.12 tabla; §16.13 "
            "exergía) y §4.8 *Polinomios NASA*. Çengel & Boles, cap. 15 (§15-1 a §15-7) y "
            "cap. 16 (§16-1 a §16-3, equilibrio químico)."
        )
        st.markdown("**Aire técnico** (§16.1): 3,762 moles de N₂ por mol de O₂.")
        st.latex(r"y_{a,\mathrm{O_2}} = 0.21 \qquad y_{a,\mathrm{N_2}} = 0.79")
        st.latex(r"M_a = 28.85\ \mathrm{kg/kmol}")
        st.markdown(
            "**Oxígeno y aire teóricos** (§16.2), por kmol de combustible "
            "C_nH_mO_pN_qS_r (en una mezcla, cada átomo se suma con las fracciones molares):"
        )
        st.latex(r"\mathrm{O_{2,t}} = n + \frac{m}{4} + r - \frac{p}{2}")
        st.latex(r"a_s = \frac{\mathrm{O_{2,t}}}{y_{a,\mathrm{O_2}}}")
        st.latex(r"\lambda = \frac{a}{a_s} = 1 + e = \frac{1}{\phi}")
        st.markdown(
            "**Reacción real** con λ ≥ 1 (combustión completa): todo el C a CO₂, el H a H₂O, "
            "el S a SO₂ y el N a N₂; sobra O₂."
        )
        st.latex(
            r"\begin{aligned}b &= n \qquad c = \frac{m}{2} \qquad e_{\mathrm{SO_2}} = r \\"
            r" d &= a\,y_{a,\mathrm{N_2}} + \frac{q}{2} \\ f &= (\lambda - 1)\,"
            r"\mathrm{O_{2,t}}\end{aligned}"
        )
        st.markdown(
            "Con λ < 1 (Cengel 15-8 c) el H se quema a H₂O y el S a SO₂, y el C reparte el O₂ "
            "que queda entre CO₂ y CO; si no alcanza ni para CO, se forma hollín."
        )
        st.markdown("**Humos** (§16.3 a §16.5), en base húmeda y seca:")
        st.latex(r"n_g = b + c + d + e + f")
        st.latex(r"y_{g,i} = \frac{n_i}{n_g} \qquad y_{g,s,i} = \frac{n_i}{n_g - c}")
        st.latex(r"AC = \lambda\,\frac{a_s\,M_a}{M_\mathrm{comb}} \qquad GC = 1 + AC")
        st.markdown(
            "**Puntos de rocío**: el del agua es la temperatura de saturación a su presión "
            "parcial; el ácido (H₂SO₄), la correlación de Verhoff & Banchero (1974), con las "
            "presiones parciales del agua y del SO₃ en mmHg y T en K:"
        )
        st.latex(r"T_{pr} = T_\mathrm{sat}(y_{\mathrm{H_2O}}\,p)")
        st.latex(
            r"\begin{aligned}\frac{1000}{T_{pr,a}} &= 2.276 - 0.0294\ln p_w \\ &\quad"
            r" - 0.0858\ln p_{s} \\ &\quad + 0.0062\ln p_w\,\ln p_{s}\end{aligned}"
        )
        st.markdown(
            "**Entalpía estándar** (§16.6): la de formación a 25 °C más la sensible, con los "
            "polinomios NASA (§4.8; acá en la forma de 9 coeficientes de McBride et al., 2002, "
            "de 200 a 6000 K):"
        )
        st.latex(r"\bar h(T) = \bar h_f + \bigl[\bar h(T) - \bar h(T_\mathrm{ref})\bigr]")
        st.latex(
            r"\begin{aligned}\frac{\bar c_p}{R_u} &= a_1 T^{-2} + a_2 T^{-1} + a_3 \\ &\quad"
            r" + a_4 T + a_5 T^2 \\ &\quad + a_6 T^3 + a_7 T^4\end{aligned}"
        )
        st.markdown(
            "**Primer principio** (§16.7) y **llama adiabática** (§16.8), por kmol de "
            "combustible; a volumen constante, con ū = h̄ − R_u·T para los gases (§16.10, "
            "Cengel 15-7):"
        )
        st.latex(
            r"\bar q - \bar w = \sum_\mathrm{prod} \nu_i\,\bar h_i"
            r" - \sum_\mathrm{reac} \nu_j\,\bar h_j"
        )
        st.latex(
            r"\sum_\mathrm{prod} \nu_i\,\bar h_i(T_{ad})"
            r" = \sum_\mathrm{reac} \nu_j\,\bar h_j(T_r)"
        )
        st.markdown(
            "**Poder calorífico** (§16.9): con los reactivos y los productos a 25 °C. El PCS "
            "deja el agua líquida; el PCI, como vapor."
        )
        st.latex(
            r"\overline{PC} = \sum_\mathrm{reac} \nu_j\,\bar h_{f,j}"
            r" - \sum_\mathrm{prod} \nu_i\,\bar h_{f,i}"
        )
        st.latex(r"\overline{PCS} - \overline{PCI} = c\,\bar h_{fg}(25\,^\circ\mathrm{C})")
        st.markdown(
            "Con los datos de NASA, h̄_fg = 44 004 kJ/kmol ≈ 2442 kJ/kg. Para un análisis "
            "elemental el PCS es un dato y PCS − PCI = h_fg·(agua formada + humedad)."
        )
        st.markdown(
            "**Disociación** (Cengel §16-1 a §16-3): a la temperatura de la llama los productos "
            "se reparten entre CO₂, CO, H₂O, H₂, O₂, N₂, OH, H, O, NO, N, Ar y SO₂ con la mínima "
            "energía libre de Gibbs (como NASA CEA; Gordon & McBride, 1994). Cada reacción "
            "cumple su constante de equilibrio (p° = 1 bar):"
        )
        st.latex(r"\ln K_p = -\frac{\Delta G^\circ(T)}{R_u\,T}")
        st.latex(r"K_p = \prod_i \left(\frac{y_i\,p}{p^\circ}\right)^{\nu_i}")
        st.markdown(
            "**Calor con los humos a T_s** (Cengel 15-6 y 15-11): si T_s está bajo el punto de "
            "rocío, el vapor queda a su presión de saturación y el resto condensa."
        )
        st.latex(r"Q_\mathrm{sal} = H_r - H_p(T_s) \qquad \eta = \frac{Q_\mathrm{sal}}{PCI}")
        st.markdown(
            "**Entropía y exergía** (§16.11 y §16.13; Cengel 15-10 y 15-11), cada gas a su "
            "presión parcial y el calor entregado a T_b:"
        )
        st.latex(r"\bar s_i = \bar s_i^\circ(T) - R_u \ln\frac{y_i\,p}{p^\circ}")
        st.latex(r"S_\mathrm{gen} = S_p - S_r + \frac{Q_\mathrm{sal}}{T_b}")
        st.latex(r"X_\mathrm{dest} = T_0\,S_\mathrm{gen}")
        st.latex(r"\overline{X}_\mathrm{comb} \approx \overline{PCI}")
        st.markdown(
            "**Tabla de §16.12** (h̄_f y s̄° a 25 °C; Klein & Nellis, 2012, sobre McBride et "
            "al., 2002). La app usa los polinomios NASA de esos mismos datos: las h̄_f "
            "difieren menos de 0,05 % (el CO₂ de NASA es −393 510 kJ/kmol) y las s̄° de los "
            "líquidos, hasta 0,9 kJ/(kmol·K)."
        )
        _table(
            pd.DataFrame(
                _VADEMECUM_TABLE,
                columns=["Sustancia", "Estado", "h̄_f [kJ/kmol]", "s̄° [kJ/(kmol·K)]", "M [kg/kmol]"],
            )
        )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Combustión", page_icon="🕯️", layout="centered")

st.subheader(SUBJECT)
st.title("🕯️ Combustión")
st.markdown(
    "Un combustible se quema con aire: con su composición y la cantidad de aire salen los "
    "**humos** (en base húmeda y seca), los **puntos de rocío**, el **poder calorífico**, la "
    "**temperatura adiabática de llama** (con combustión completa y con disociación) y el "
    "**calor** que entregan los humos al enfriarse, con el rendimiento y la exergía destruida. "
    "Al revés, con lo que mide un analizador en la chimenea (el O₂ o un Orsat) sale cuánto aire "
    "se usó."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Combustión")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, horizontal=True, key="cb_mode")
if mode == _MODE_COMB:
    _render_combustion_mode(system)
else:
    _render_flue_mode(system)
