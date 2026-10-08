"""Página 11 — Exergía.

Fase 7. Tres modos:

- **Exergía física** (vademecum §11.1 a §11.6): de un fluido (masa o flujo, con la
  parte térmica y la mecánica, la cinética y la potencial), de un calor a
  temperatura T y de un cuerpo que se enfría o se calienta.
- **Exergía química** (Szargut et al., 1988): de una sustancia (la tabla y el
  método de Szargut), de una mezcla de gases y de un combustible por su análisis
  elemental (Szargut y Styrylska, 1964).
- **Una planta, componente por componente**: la exergía que gasta, produce,
  destruye y pierde cada componente de un Rankine, una refrigeración, una turbina
  de gas o un ciclo combinado, con el **diagrama de Grassmann** (Bejan,
  Tsatsaronis y Moran, 1996).

El cálculo vive en :mod:`core.exergy`; los gráficos, en :mod:`ui.exergy_charts`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from core.combustion.heating_value import (
    HEATING_VALUE_EXAMPLES,
    DryUltimate,
    hhv_channiwala_parikh,
)
from core.cycles.combined import COMBINED_EXAMPLES, CombinedInputs
from core.cycles.combined_multi import (
    COMBINED_MULTI_EXAMPLES,
    MultiCombinedInputs,
    from_combined_with_pinch,
    solve_combined_multi,
)
from core.cycles.gas_turbine import GAS_TURBINE_EXAMPLES, GasTurbineInputs, solve_gas_turbine
from core.cycles.rankine import RANKINE_EXAMPLES, RankineInputs, solve_rankine
from core.cycles.refrigeration import (
    REFRIGERATION_EXAMPLES,
    RefrigerationInputs,
    Reservoirs,
    default_reservoirs,
    solve_refrigeration,
)
from core.exergy import chemical as ch
from core.exergy import physical as ph
from core.exergy import plant as pl
from core.exergy.exergy_procedure import (
    finite_source_steps,
    fuel_steps,
    heat_steps,
    mixture_steps,
    physical_steps,
    plant_steps,
    species_steps,
)
from core.export import dict_to_csv
from core.fluids import FLUID_NAMES_ES, PAIR_KWARGS, SUPPORTED_FLUIDS, PairCode, suggested_inputs
from core.state_report import INPUT_SPECS, PAIR_LABELS_ES, PAIR_ORDER, ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.exergy_charts import (
    carnot_factor_figure,
    component_efficiency_figure,
    finite_source_figure,
    fuel_ratio_figure,
    grassmann_figure,
    mollier_exergy_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.23.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_PHYS = "Exergía física: un fluido, un calor o un cuerpo"
_M_CHEM = "Exergía química (Szargut)"
_M_PLANT = "Una planta, componente por componente (Grassmann)"
_MODES = (_M_PHYS, _M_CHEM, _M_PLANT)

_F_FLUID = "Un fluido (masa o flujo)"
_F_HEAT = "Un calor a temperatura T"
_F_BODY = "Un cuerpo que se enfría o se calienta"
_PHYS_WHAT = (_F_FLUID, _F_HEAT, _F_BODY)

_AMOUNTS: tuple[ph.Amount, ...] = ("por kg", "masa", "caudal")
_AMOUNT_LABELS = ("Por kg", "Una masa (sistema cerrado)", "Un caudal (flujo)")

_Q_SPECIES = "Una sustancia"
_Q_MIX = "Una mezcla de gases"
_Q_FUEL = "Un combustible por su análisis elemental"
_CHEM_WHAT = (_Q_SPECIES, _Q_MIX, _Q_FUEL)

_C_RANKINE = "Ciclo de Rankine"
_C_REFRIG = "Refrigeración o bomba de calor"
_C_GT = "Turbina de gas (Brayton)"
_C_CC = "Ciclo combinado"
_CYCLES = (_C_RANKINE, _C_REFRIG, _C_GT, _C_CC)
#: Lo que guarda cada página de ciclo en la sesión (lo último que se calculó ahí).
_SESSION_KEYS = {
    _C_RANKINE: "rk_inputs",
    _C_REFRIG: "rf_inputs",
    _C_GT: "bt_inputs",
    _C_CC: "cc_inputs",
}
_PAGE_NAMES = {
    _C_RANKINE: "Rankine",
    _C_REFRIG: "Refrigeración",
    _C_GT: "Brayton",
    _C_CC: "Ciclo combinado",
}
_SRC_EXAMPLE = "Un ejemplo"
_SRC_LAST = "El último que calculaste en la página del ciclo"

_FUEL_MODELS: tuple[pl.FuelModel, ...] = ("szargut", "pci")
_FUEL_MODEL_LABELS = (
    "Su exergía química (Szargut et al., 1988)",
    "≈ PCI (vademecum §16.13)",
)

_EH: QuantityKind = "specific_enthalpy"
_EM: QuantityKind = "molar_enthalpy"
_W: QuantityKind = "power"

_FUEL_EXAMPLES = {
    name: inp
    for name, inp in HEATING_VALUE_EXAMPLES.items()
    if inp.ultimate is not None and inp.fuel_type != "gas"
}
#: Los de una presión (Fase 3.4) se pasan al modelo de varias presiones al resolver.
_COMBINED_ALL: dict[str, CombinedInputs | MultiCombinedInputs] = {
    **{f"Una presión — {n}": i for n, i in COMBINED_EXAMPLES.items()},
    **COMBINED_MULTI_EXAMPLES,
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


def _ambient_inputs(key: str, default: ph.Ambient) -> tuple[float, float]:
    left, right = st.columns(2)
    with left:
        T0 = number_input_si(
            label="Ambiente T₀",
            kind="temperature",
            default_si=default.T0_K,
            key=f"{key}_T0",
            format="%.4g",
            help="El estado muerto (vademecum §11.1): el aire o el agua que rodean al sistema.",
        )
    with right:
        p0 = number_input_si(
            label="Ambiente p₀",
            kind="pressure",
            default_si=default.p0_Pa,
            key=f"{key}_p0",
            format="%.5g",
        )
    return T0, p0


# ---------------------------------------------------------------------
# Exergía física
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _physical_cached(
    fluid: str,
    pair: str,
    values: tuple[tuple[str, float], ...],
    ambient: ph.Ambient,
    speed: float,
    height: float,
    amount: str,
    mass: float | None,
    m_dot: float | None,
) -> ph.PhysicalExergy:
    return ph.physical_exergy(
        fluid,
        pair,  # type: ignore[arg-type]
        ambient=ambient,
        speed_m_per_s=speed,
        height_m=height,
        amount=amount,  # type: ignore[arg-type]
        mass_kg=mass,
        m_dot_kg_s=m_dot,
        **dict(values),
    )


def _input_format(kind: QuantityKind | None) -> str:
    return "%.6g" if kind in ("specific_volume", "specific_entropy") else "%.5g"


def _state_inputs(
    fluid: str, pair: PairCode, defaults: dict[str, float], key: str
) -> dict[str, float]:
    """Los dos datos del par, como en Propiedades (devuelve SI)."""
    values: dict[str, float] = {}
    for column, kw in zip(st.columns(2), PAIR_KWARGS[pair], strict=True):
        spec = INPUT_SPECS[kw]
        with column:
            if spec.kind is None:
                values[kw] = float(
                    st.number_input(
                        f"{spec.name} {spec.symbol} [-]",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(defaults[kw]),
                        step=0.05,
                        format="%.4f",
                        key=f"{key}_{kw}",
                    )
                )
            else:
                values[kw] = number_input_si(
                    label=f"{spec.name} {spec.symbol}",
                    kind=spec.kind,
                    default_si=float(defaults[kw]),
                    key=f"{key}_{kw}",
                    format=_input_format(spec.kind),
                )
    return values


def _fluid_and_pair(
    key: str, fluid_default: str, pair_default: PairCode, label: str = ""
) -> tuple[str, PairCode]:
    left, right = st.columns(2)
    with left:
        fluid = st.selectbox(
            f"Fluido{label}",
            SUPPORTED_FLUIDS,
            index=SUPPORTED_FLUIDS.index(fluid_default),
            format_func=lambda f: FLUID_NAMES_ES.get(f, f),
            key=f"{key}_fluid",
        )
    with right:
        pair: PairCode = st.selectbox(
            f"Propiedades conocidas{label}",
            PAIR_ORDER,
            index=PAIR_ORDER.index(pair_default),
            format_func=lambda p: PAIR_LABELS_ES[p],
            key=f"{key}_pair",
        )
    return fluid, pair


def _physical_fluid(system: UnitSystem) -> None:
    names = list(ph.PHYSICAL_EXAMPLES)
    name = st.selectbox(
        "Ejemplo precargado",
        names,
        key="xf_example",
        help="Ejemplos del cap. 8 de Cengel y del 10-8. Podés cambiar cualquier dato: el "
        "resultado se actualiza solo.",
    )
    ex = ph.PHYSICAL_EXAMPLES[name]
    key = f"xf_{names.index(name)}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    fluid, pair = _fluid_and_pair(key, ex.fluid, ex.pair)
    defaults = ex.values if (fluid, pair) == (ex.fluid, ex.pair) else suggested_inputs(fluid, pair)
    values = _state_inputs(fluid, pair, defaults, f"{key}_{fluid}_{pair}")
    T0, p0 = _ambient_inputs(key, ex.ambient)
    left, right = st.columns(2)
    with left:
        speed = number_input_si(
            label="Velocidad V",
            kind="speed",
            default_si=ex.speed_m_per_s,
            key=f"{key}_V",
            format="%.4g",
            min_value_si=0.0,
        )
    with right:
        height = number_input_si(
            label="Altura z",
            kind="length",
            default_si=ex.height_m,
            key=f"{key}_z",
            format="%.4g",
            help="Sobre el nivel de referencia del ambiente (la energía potencial g·z).",
        )
    amount_label = st.radio(
        "¿Para cuánto fluido?",
        _AMOUNT_LABELS,
        index=_AMOUNTS.index(ex.amount),
        key=f"{key}_amount",
        horizontal=True,
    )
    amount = _AMOUNTS[_AMOUNT_LABELS.index(amount_label)]
    mass = m_dot = None
    if amount == "masa":
        mass = number_input_si(
            label="Masa m",
            kind="mass",
            default_si=ex.mass_kg or 1.0,
            key=f"{key}_m",
            format="%.5g",
        )
    elif amount == "caudal":
        m_dot = number_input_si(
            label="Caudal ṁ",
            kind="mass_flow",
            default_si=ex.m_dot_kg_s or 1.0,
            key=f"{key}_mdot",
            format="%.5g",
        )
    second_values: tuple[str, PairCode, dict[str, float]] | None = None
    if st.checkbox(
        "Comparar con un segundo estado (proceso 1 → 2: el trabajo reversible)",
        value=ex.second is not None,
        key=f"{key}_second",
    ):
        pair2_default, values2_default = ex.second or ("TP", suggested_inputs(fluid, "TP"))
        pair2: PairCode = st.selectbox(
            "Propiedades conocidas del estado 2",
            PAIR_ORDER,
            index=PAIR_ORDER.index(pair2_default),
            format_func=lambda p: PAIR_LABELS_ES[p],
            key=f"{key}_pair2",
        )
        d2 = (
            values2_default
            if pair2 == pair2_default and fluid == ex.fluid
            else suggested_inputs(fluid, pair2)
        )
        second_values = (fluid, pair2, _state_inputs(fluid, pair2, d2, f"{key}_{fluid}_2_{pair2}"))
    try:
        ambient = ph.Ambient(T0, p0)
        result = _physical_cached(
            fluid,
            pair,
            tuple(sorted(values.items())),
            ambient,
            speed,
            height,
            amount,
            mass,
            m_dot,
        )
        second = None
        if second_values is not None:
            f2, p2, v2 = second_values
            second = _physical_cached(
                f2, p2, tuple(sorted(v2.items())), ambient, 0.0, 0.0, "por kg", None, None
            )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_physical(result, second, system)


def _render_physical(
    r: ph.PhysicalExergy, second: ph.PhysicalExergy | None, system: UnitSystem
) -> None:
    st.markdown("### Resultado")
    ue = _u(_EH, system)
    items = [
        (f"ψ, exergía de flujo [{ue}]", _val(r.psi_J_per_kg, _EH, system)),
        (f"φ, exergía de la masa [{ue}]", _val(r.phi_J_per_kg, _EH, system)),
        (f"ψ térmica [{ue}]", _val(r.psi_thermal_J_per_kg, _EH, system)),
        (f"ψ mecánica [{ue}]", _val(r.psi_mechanical_J_per_kg, _EH, system)),
    ]
    if r.X_J is not None:
        items.append((f"X de la masa [{_u('energy', system)}]", _val(r.X_J, "energy", system)))
    if r.X_W is not None:
        items.append((f"Ẋ del flujo [{_u(_W, system)}]", _val(r.X_W, _W, system)))
    if second is not None:
        proc = ph.ProcessExergy(r, second)
        items.append((f"Δψ = ψ₂ − ψ₁ [{ue}]", _val(proc.dpsi_J_per_kg, _EH, system)))
        items.append((f"h₂ − h₁ [{ue}]", _val(proc.dh_J_per_kg, _EH, system)))
    _metric_rows(items)
    for note in ph.physical_notes(r):
        st.info(note)
    rows = []
    specs = (
        ("T", "temperature", lambda s: s.T_K),
        ("p", "pressure", lambda s: s.P_Pa),
        ("h", _EH, lambda s: s.h_J_per_kg),
        ("s", "specific_entropy", lambda s: s.s_J_per_kg_K),
        ("u", _EH, lambda s: s.u_J_per_kg),
        ("v", "specific_volume", lambda s: s.v_m3_per_kg),
    )
    for sym, kind, get in specs:
        row = {
            "Propiedad": sym,
            "Estado": _val(get(r.state), kind, system),  # type: ignore[arg-type]
            "Estado muerto": _val(get(r.dead), kind, system),  # type: ignore[arg-type]
            "A T₀ y p": _val(get(r.intermediate), kind, system),  # type: ignore[arg-type]
        }
        if second is not None:
            row["Estado 2"] = _val(get(second.state), kind, system)  # type: ignore[arg-type]
        row["Unidad"] = _u(kind, system)  # type: ignore[arg-type]
        rows.append(row)
    _table(pd.DataFrame(rows))
    st.markdown("#### El estado en el diagrama h–s")
    try:
        h_scale = convert_from_si(1.0, _EH, system)
        s_scale = convert_from_si(1.0, "specific_entropy", system)
        fig = mollier_exergy_figure(
            r,
            h_scale=h_scale,
            s_scale=s_scale,
            h_unit=_u(_EH, system),
            s_unit=_u("specific_entropy", system),
        )
        st.plotly_chart(fig, width="stretch", key="xf_mollier")
        st.caption(
            "La recta punteada pasa por el estado muerto con pendiente T₀: "
            "h = h₀ + T₀·(s − s₀). La exergía de flujo ψ es la distancia vertical del estado a "
            "esa recta (Kotas, 1985): lo que el estado tiene de entalpía por encima de lo que "
            "hay que entregarle al ambiente."
        )
    except Exception as exc:  # noqa: BLE001 — un gráfico que falla no tumba la página
        st.warning(f"No se pudo dibujar el diagrama: {exc}")
    _render_steps(
        physical_steps(r, system, second),
        "El estado muerto y las diferencias con los números del estado, en el sistema elegido.",
    )
    _render_export(ph.physical_exergy_to_dict(r, system, second), "exergia_fisica", "xf_dl")


def _physical_heat(system: UnitSystem) -> None:
    names = list(ph.HEAT_EXAMPLES)
    name = st.selectbox("Ejemplo precargado", names, key="xh_example")
    ex = ph.HEAT_EXAMPLES[name]
    key = f"xh_{names.index(name)}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    left, right = st.columns(2)
    with left:
        Q = number_input_si(
            label="Calor Q̇", kind=_W, default_si=ex.Q_W, key=f"{key}_Q", format="%.5g"
        )
    with right:
        T = number_input_si(
            label="Temperatura de la fuente T",
            kind="temperature",
            default_si=ex.T_K,
            key=f"{key}_T",
            format="%.5g",
        )
    T0 = number_input_si(
        label="Ambiente T₀",
        kind="temperature",
        default_si=ex.T0_K,
        key=f"{key}_T0",
        format="%.5g",
    )
    try:
        r = ph.heat_exergy(Q, T, T0)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"Ẋ_Q, exergía del calor [{_u(_W, system)}]", _val(r.X_W, _W, system)),
            (f"Anergía Q̇·T₀/T [{_u(_W, system)}]", _val(r.anergy_W, _W, system)),
            ("Factor de Carnot 1 − T₀/T", _num(r.carnot, 4)),
        ]
    )
    for note in ph.heat_notes(r):
        st.info(note)
    t_scale = convert_from_si(1.0, "temperature", system) - convert_from_si(
        0.0, "temperature", system
    )
    t_offset = convert_from_si(0.0, "temperature", system)
    st.plotly_chart(
        carnot_factor_figure(
            r.T0_K, r.T_K, t_scale=t_scale, t_offset=t_offset, t_unit=_u("temperature", system)
        ),
        width="stretch",
        key="xh_carnot",
    )
    st.caption(
        "El factor de Carnot dice qué parte del calor es exergía: crece con la temperatura de "
        "la fuente y se vuelve negativo debajo de T₀ (la línea punteada)."
    )
    _render_steps(heat_steps(r, system))
    data = {
        "Q": _val(r.Q_W, _W, system),
        "T": _val(r.T_K, "temperature", system),
        "T0": _val(r.T0_K, "temperature", system),
        "factor_de_Carnot": r.carnot,
        "X_Q": _val(r.X_W, _W, system),
        "anergia": _val(r.anergy_W, _W, system),
        "unidad_potencia": _u(_W, system),
    }
    _render_export(data, "exergia_calor", "xh_dl")


def _physical_body(system: UnitSystem) -> None:
    names = list(ph.FINITE_SOURCE_EXAMPLES)
    name = st.selectbox("Ejemplo precargado", names, key="xs_example")
    ex = ph.FINITE_SOURCE_EXAMPLES[name]
    key = f"xs_{names.index(name)}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    left, right = st.columns(2)
    with left:
        m = number_input_si(
            label="Masa m", kind="mass", default_si=ex.m_kg, key=f"{key}_m", format="%.5g"
        )
    with right:
        c = number_input_si(
            label="Calor específico c",
            kind="specific_heat",
            default_si=ex.c_J_per_kg_K,
            key=f"{key}_c",
            format="%.4g",
        )
    left, right = st.columns(2)
    with left:
        T = number_input_si(
            label="Temperatura del cuerpo T",
            kind="temperature",
            default_si=ex.T_K,
            key=f"{key}_T",
            format="%.5g",
        )
    with right:
        T0 = number_input_si(
            label="Ambiente T₀",
            kind="temperature",
            default_si=ex.T0_K,
            key=f"{key}_T0",
            format="%.5g",
        )
    try:
        r = ph.finite_source_exergy(m, c, T, T0)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    ue = _u("energy", system)
    _metric_rows(
        [
            (f"Φ, exergía del cuerpo [{ue}]", _val(r.Phi_J, "energy", system)),
            (f"Calor hasta T₀, m·c·(T − T₀) [{ue}]", _val(r.Q_J, "energy", system)),
            ("Φ / |Q|", _pct(r.fraction_of_heat)),
        ]
    )
    for note in ph.finite_source_notes(r):
        st.info(note)
    t_scale = convert_from_si(1.0, "temperature", system) - convert_from_si(
        0.0, "temperature", system
    )
    t_offset = convert_from_si(0.0, "temperature", system)
    st.plotly_chart(
        finite_source_figure(
            r,
            t_scale=t_scale,
            t_offset=t_offset,
            t_unit=_u("temperature", system),
            e_scale=convert_from_si(1.0, "energy", system),
            e_unit=ue,
        ),
        width="stretch",
        key="xs_curve",
    )
    _render_steps(finite_source_steps(r, system))
    data = {
        "m": _val(r.m_kg, "mass", system),
        "c": _val(r.c_J_per_kg_K, "specific_heat", system),
        "T": _val(r.T_K, "temperature", system),
        "T0": _val(r.T0_K, "temperature", system),
        "Phi": _val(r.Phi_J, "energy", system),
        "Q": _val(r.Q_J, "energy", system),
        "Phi_sobre_Q": r.fraction_of_heat,
    }
    _render_export(data, "exergia_cuerpo", "xs_dl")


# ---------------------------------------------------------------------
# Exergía química
# ---------------------------------------------------------------------


def _formula_text(key: str) -> str:
    sp = ch.chemical_species()[key]
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    body = "".join(el + ("" if n == 1 else str(int(n)).translate(sub)) for el, n in sp.formula)
    return f"{body}, {sp.phase_label}"


def _chemical_species(system: UnitSystem, model: ch.ReferenceModel) -> None:
    table = ch.chemical_species()
    keys = list(table)
    key = st.selectbox(
        "Sustancia",
        keys,
        index=keys.index("CH4"),
        format_func=lambda k: f"{table[k].name} ({_formula_text(k)})",
        key="xq_species",
    )
    try:
        r = ch.species_exergy(key, model)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    items = [
        (f"ē, exergía química [{_u(_EM, system)}]", _val(r.e_J_per_mol, _EM, system, 6)),
        (f"e, por masa [{_u(_EH, system)}]", _val(r.e_J_per_kg, _EH, system)),
    ]
    if r.ratio_lhv is not None:
        items.append(("φ = e/PCI", _num(r.ratio_lhv, 4)))
        items.append(("e/PCS", _num(r.ratio_hhv, 4)))
    if r.x_reference is not None:
        items.append(("x⁰⁰ en el aire de referencia", _num(r.x_reference, 4)))
    _metric_rows(items)
    st.caption(f"Fuente: {r.source} ({ch.MODELS[model]}).")
    if r.method_deviation is not None and r.method is not None:
        diff = r.method.e_J_per_mol - r.e_J_per_mol
        sign = "+" if diff >= 0 else "−"
        um = _u(_EM, system)
        st.info(
            "El método de Szargut (Δg_f de los polinomios NASA + los elementos) da "
            f"{_val(r.method.e_J_per_mol, _EM, system, 6)} {um}, "
            f"{sign}{_val(abs(diff), _EM, system, 3)} {um} "
            f"({sign}{100 * abs(r.method_deviation):.2f} %)".replace(".", ",")
            + " respecto de la tabla."
        )
    rows = []
    for k, sp in table.items():
        rows.append(
            {
                "Sustancia": sp.name,
                "Fórmula": _formula_text(k),
                f"Szargut 1988 [{_u(_EM, system)}]": _val(
                    sp.tabulated_J_per_mol("szargut1988"), _EM, system, 6
                ),
                f"Ahrendts 1980 [{_u(_EM, system)}]": _val(
                    sp.tabulated_J_per_mol("ahrendts1980"), _EM, system, 6
                ),
            }
        )
    with st.expander("📋 La tabla completa (los dos modelos)", expanded=False):
        st.caption(
            "Exergía química molar estándar a 25 °C. Vacío: el modelo no la tabula (la página "
            "la calcula con el método de Szargut cuando hay datos de NASA)."
        )
        _table(pd.DataFrame(rows))
    st.markdown("#### φ = e/PCI de los combustibles")
    st.plotly_chart(
        fuel_ratio_figure(ch.fuel_ratio_table(model), current=key),
        width="stretch",
        key="xq_ratio",
    )
    st.caption(
        "El vademecum aproxima la exergía de un combustible con su PCI (§16.13). Los "
        "hidrocarburos tienen un poco más (φ ≈ 1,03–1,08): su combustión aumenta la cantidad de "
        "moles de gas y la entropía. El H₂ y el CO quedan por debajo de 1: su combustión la baja."
    )
    _render_steps(species_steps(r, system))
    _render_export(ch.species_exergy_to_dict(r, system), "exergia_quimica", "xq_dl_species")


def _chemical_mixture(system: UnitSystem, model: ch.ReferenceModel) -> None:
    names = list(ch.MIXTURE_EXAMPLES)
    name = st.selectbox("Ejemplo precargado", names, key="xq_mix_example")
    ex = ch.MIXTURE_EXAMPLES[name]
    key = f"xq_mix_{names.index(name)}"
    table = ch.chemical_species()
    gases = [k for k, sp in table.items() if sp.phase == "g"]
    chosen = st.multiselect(
        "Componentes",
        gases,
        default=list(ex),
        format_func=lambda k: f"{table[k].name} ({_formula_text(k).split(',')[0]})",
        key=f"{key}_components",
    )
    fractions: dict[str, float] = {}
    for i, k in enumerate(chosen):
        if i % 2 == 0:
            cols = st.columns(2)
        with cols[i % 2]:
            fractions[k] = float(
                st.number_input(
                    f"{table[k].name} [fracción molar]",
                    min_value=0.0,
                    max_value=1.0,
                    value=round(float(ex.get(k, 0.0)), 6),
                    step=0.01,
                    format="%.4f",
                    key=f"{key}_{k}",
                )
            )
    total = sum(fractions.values())
    if fractions and abs(total - 1.0) > 1e-3:
        st.caption(f"Las fracciones suman {total:.4f}: se normalizan.".replace(".", ","))
    try:
        r = ch.mixture_chemical_exergy(fractions, model)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    items = [
        (f"ē de la mezcla [{_u(_EM, system)}]", _val(r.e_J_per_mol, _EM, system)),
        (f"e por masa [{_u(_EH, system)}]", _val(r.e_J_per_kg, _EH, system)),
    ]
    if r.condenses:
        items.append(("Agua que condensa [mol/mol]", _num(r.n_liquid, 4)))
    _metric_rows(items)
    if r.condenses:
        st.info(
            "A 25 °C y 1 atm parte del vapor condensa: el gas queda saturado y el agua líquida "
            "entra con su exergía química (0,9 kJ/mol). Sin esa corrección se contaría vapor "
            "por encima de su presión de saturación."
        )
    rows = [
        {
            "Componente": table[row.key].name,
            "x": _num(row.x_in, 4),
            "x en el gas": _num(row.x_gas, 4),
            f"ē [{_u(_EM, system)}]": _val(row.e_J_per_mol, _EM, system),
            "x·ē": _val(row.x_e_J_per_mol, _EM, system),
            "R̄T₀·x·ln x": _val(row.mixing_J_per_mol, _EM, system),
        }
        for row in r.rows
    ]
    _table(pd.DataFrame(rows))
    st.caption(
        "La exergía de la mezcla es la de sus componentes puros menos la que se pierde al "
        "mezclarlos (el término R̄T₀·Σx·ln x, negativo)."
    )
    _render_steps(mixture_steps(r, system))
    _render_export(ch.mixture_exergy_to_dict(r, system), "exergia_mezcla", "xq_dl_mix")


_ULT_FIELDS = (
    ("C", "Carbono C"),
    ("H", "Hidrógeno H"),
    ("O", "Oxígeno O"),
    ("N", "Nitrógeno N"),
    ("S", "Azufre S"),
    ("A", "Cenizas"),
)


def _chemical_fuel(system: UnitSystem, model: ch.ReferenceModel) -> None:
    names = list(_FUEL_EXAMPLES)
    name = st.selectbox(
        "Ejemplo precargado",
        names,
        key="xq_fuel_example",
        help="Los combustibles de la página Poder calorífico con análisis elemental y PCS "
        "medido o exacto.",
    )
    ex = _FUEL_EXAMPLES[name]
    key = f"xq_fuel_{names.index(name)}"
    assert ex.ultimate is not None
    kind_label = st.radio(
        "Tipo de combustible",
        ("Sólido", "Líquido"),
        index=1 if ex.fuel_type == "liquid" else 0,
        key=f"{key}_kind",
        horizontal=True,
    )
    kind: ch.FuelKind = "líquido" if kind_label == "Líquido" else "sólido"
    st.markdown("Análisis elemental **en base seca** [% en masa]:")
    pct = ex.ultimate.pct()
    values: dict[str, float] = {}
    for i, (attr, label) in enumerate(_ULT_FIELDS):
        if i % 2 == 0:
            cols = st.columns(2)
        with cols[i % 2]:
            values[attr] = float(
                st.number_input(
                    f"{label} [%]",
                    min_value=0.0,
                    max_value=100.0,
                    value=round(pct[attr], 4),
                    step=0.1,
                    format="%.3f",
                    key=f"{key}_{attr}",
                )
            )
    left, right = st.columns(2)
    with left:
        W = float(
            st.number_input(
                "Humedad tal cual W [%]",
                min_value=0.0,
                max_value=99.0,
                value=round(100.0 * ex.moisture, 4),
                step=0.5,
                format="%.3f",
                key=f"{key}_W",
            )
        )
    estimate = st.checkbox(
        "Estimar el PCS con Channiwala y Parikh (2002)",
        value=ex.reference is None,
        key=f"{key}_estimate",
    )
    ultimate = DryUltimate(**{k: v / 100.0 for k, v in values.items()})
    hhv_d: float | None = None
    with right:
        if not estimate:
            hhv_default = (
                ex.reference.hhv_d_J_per_kg
                if ex.reference is not None
                else hhv_channiwala_parikh(ultimate)
            )
            hhv_d = number_input_si(
                label="PCS seco",
                kind=_EH,
                default_si=hhv_default,
                key=f"{key}_hhv",
                format="%.5g",
            )
    try:
        if hhv_d is None:
            ultimate.validate()
            hhv_d = hhv_channiwala_parikh(ultimate)
            st.caption(f"PCS seco estimado: {_val(hhv_d, _EH, system)} {_u(_EH, system)}.")
        r = ch.fuel_chemical_exergy(ultimate, hhv_d, moisture=W / 100.0, kind=kind, model=model)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    ue = _u(_EH, system)
    _metric_rows(
        [
            ("β = e/PCI de la materia seca", _num(r.beta, 4)),
            (f"e química tal cual [{ue}]", _val(r.e_J_per_kg, _EH, system)),
            ("e/PCI tal cual", _num(r.ratio_lhv, 4)),
            ("e/PCS tal cual", _num(r.ratio_hhv, 4)),
        ]
    )
    branch = {
        "carbón": "la forma de los carbones (o/c ≤ 0,667)",
        "madera": "la forma de la madera y la biomasa (0,667 < o/c ≤ 2,67)",
        "líquido": "la forma de los combustibles líquidos",
    }[r.branch]
    st.info(
        f"Con o/c = {_num(r.o_c, 3)} se usa {branch} de Szargut y Styrylska (1964). En base "
        "seca la exergía química de un combustible sólido o líquido supera al PCI en un 3–15 % "
        "(β ≈ 1,03–1,15). Con humedad, e/PCI tal cual sube más: el PCI descuenta el calor para "
        "evaporar el agua (W·h_fg) y la exergía no (β multiplica PCI + W·h_fg)."
    )
    for w in r.warnings:
        st.warning(w)
    rows = [
        {"Término": "Materia orgánica β·PCI*", f"[{ue}]": _val(r.e_organic_J_per_kg, _EH, system)},
        {"Término": "Azufre (e_S − PCI_S)·S", f"[{ue}]": _val(r.e_sulfur_J_per_kg, _EH, system)},
        {"Término": "Humedad e_w·W", f"[{ue}]": _val(r.e_water_J_per_kg, _EH, system)},
        {"Término": "Total", f"[{ue}]": _val(r.e_J_per_kg, _EH, system)},
        {"Término": "PCI tal cual", f"[{ue}]": _val(r.lhv_ar_J_per_kg, _EH, system)},
        {"Término": "PCS tal cual", f"[{ue}]": _val(r.hhv_ar_J_per_kg, _EH, system)},
    ]
    _table(pd.DataFrame(rows))
    _render_steps(fuel_steps(r, system))
    _render_export(ch.fuel_exergy_to_dict(r, system), "exergia_combustible", "xq_dl_fuel")


# ---------------------------------------------------------------------
# Planta
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _rankine_plant(
    inputs: RankineInputs, ambient: ph.Ambient, T_H: float | None, T_L: float | None
) -> pl.PlantExergy:
    # el ciclo sale de su propia caché: cambiar T₀, T_H o T_L no vuelve a correr TESPy
    result = _rankine_cached(inputs)
    return pl.rankine_exergy(result, ambient=ambient, T_source_K=T_H, T_sink_K=T_L)


@st.cache_data(show_spinner=False)
def _rankine_cached(inputs: RankineInputs):  # noqa: ANN202 — el resultado de TESPy
    return solve_rankine(inputs)


@st.cache_data(show_spinner=False)
def _refrigeration_cached(inputs: RefrigerationInputs):  # noqa: ANN202
    return solve_refrigeration(inputs)


@st.cache_data(show_spinner=False)
def _refrigeration_plant(inputs: RefrigerationInputs, reservoirs: Reservoirs) -> pl.PlantExergy:
    return pl.refrigeration_exergy(_refrigeration_cached(inputs), reservoirs)


@st.cache_data(show_spinner=False)
def _gas_turbine_plant(inputs: GasTurbineInputs, model: str) -> tuple[pl.PlantExergy, float]:
    result = solve_gas_turbine(inputs)
    return pl.gas_turbine_plant_exergy(result, model), result.eta_th  # type: ignore[arg-type]


@st.cache_data(show_spinner=False)
def _combined_plant(
    inputs: CombinedInputs | MultiCombinedInputs, model: str
) -> tuple[pl.PlantExergy, float]:
    if isinstance(inputs, CombinedInputs):  # una presión (Fase 3.4), también por chimenea
        inputs = from_combined_with_pinch(inputs)
    result = solve_combined_multi(inputs)
    return pl.combined_plant_exergy(result, model), result.eta_th  # type: ignore[arg-type]


def _examples_for(cycle: str) -> dict[str, Any]:
    return {
        _C_RANKINE: RANKINE_EXAMPLES,
        _C_REFRIG: REFRIGERATION_EXAMPLES,
        _C_GT: GAS_TURBINE_EXAMPLES,
        _C_CC: _COMBINED_ALL,
    }[cycle]


def _cycle_inputs(cycle: str) -> tuple[Any, str] | None:
    """Los datos del ciclo (un ejemplo o lo último de su página) y una key para los widgets."""
    last = st.session_state.get(_SESSION_KEYS[cycle])
    cid = _CYCLES.index(cycle)
    source = _SRC_EXAMPLE
    if last is not None:
        source = st.radio(
            "Datos del ciclo",
            (_SRC_EXAMPLE, _SRC_LAST),
            index=1,
            key=f"xp_{cid}_source",
            horizontal=True,
            help=f"La página {_PAGE_NAMES[cycle]} guarda lo último que calculaste.",
        )
    if source == _SRC_LAST:
        st.caption(f"Con los datos que calculaste en la página {_PAGE_NAMES[cycle]}.")
        # la huella de los datos va en la key: otro ciclo arranca con sus valores por defecto
        digest = hashlib.sha1(repr(last).encode(), usedforsecurity=False).hexdigest()[:8]
        return last, f"xp_{cid}_last_{digest}"
    examples = _examples_for(cycle)
    names = list(examples)
    name = st.selectbox("Ejemplo", names, key=f"xp_{cid}_example")
    return examples[name], f"xp_{cid}_{names.index(name)}"


def _plant_mode(system: UnitSystem) -> None:
    cycle = st.radio("Planta", _CYCLES, key="xp_cycle", horizontal=True)
    got = _cycle_inputs(cycle)
    if got is None:
        return
    inputs, key = got
    try:
        with st.spinner("Resolviendo el ciclo…"):
            plant, context = _plant_with_settings(cycle, inputs, key, system)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_plant(plant, context, key, system)


def _plant_with_settings(
    cycle: str, inputs: Any, key: str, system: UnitSystem
) -> tuple[pl.PlantExergy, list[tuple[str, str]]]:
    """Lee los datos del análisis de cada planta y la resuelve (con caché)."""
    if cycle == _C_RANKINE:
        result = _rankine_cached(inputs)
        T0, p0 = _ambient_inputs(key, ph.Ambient(290.0, 100_000.0))
        left, right = st.columns(2)
        with left:
            T_H = number_input_si(
                label="Fuente de calor T_H",
                kind="temperature",
                default_si=pl.default_rankine_source_K(result),
                key=f"{key}_TH",
                format="%.5g",
                help="El hogar de Cengel 10-8 está a 1600 K. Con un ORC, la fuente es más fría "
                "(por defecto, 50 K arriba del vapor).",
            )
        with right:
            T_L = number_input_si(
                label="Sumidero del condensador T_L",
                kind="temperature",
                default_si=T0,
                key=f"{key}_TL",
                format="%.5g",
                help="El agua de enfriamiento o el ambiente. Igual a T₀: el calor no lleva "
                "exergía. Más caliente: su exergía es una pérdida.",
            )
        plant = _rankine_plant(inputs, ph.Ambient(T0, p0), T_H, T_L)
        context = [("η térmico", _pct(result.eta_th))]
        return plant, context
    if cycle == _C_REFRIG:
        result = _refrigeration_cached(inputs)
        res = default_reservoirs(result)
        left, right = st.columns(2)
        with left:
            T_C = number_input_si(
                label="Fuente fría T_C",
                kind="temperature",
                default_si=res.T_cold_K,
                key=f"{key}_TC",
                format="%.5g",
                help="El espacio refrigerado (o el exterior en una bomba de calor).",
            )
        with right:
            T_Hr = number_input_si(
                label="Fuente caliente T_H",
                kind="temperature",
                default_si=res.T_hot_K,
                key=f"{key}_THr",
                format="%.5g",
                help="El ambiente (o el espacio calefaccionado en una bomba de calor).",
            )
        plant = _refrigeration_plant(inputs, Reservoirs(T_C, T_Hr))
        cop = result.COP_B if result.inputs.heat_pump else result.COP_R
        return plant, [("COP", _num(cop, 4))]
    gas_turbine = inputs if cycle == _C_GT else inputs.gas_turbine
    model: pl.FuelModel = "szargut"
    if gas_turbine.fuel is None:
        st.caption(
            "Aire estándar: no hay combustible y el calor de cada cámara entra con su exergía, "
            "Q̇·(1 − T₀/T), con T la temperatura a la que sale el aire (§11.5)."
        )
    else:
        label = st.radio(
            "Exergía del combustible",
            _FUEL_MODEL_LABELS,
            key=f"{key}_model",
            horizontal=True,
        )
        model = _FUEL_MODELS[_FUEL_MODEL_LABELS.index(label)]
    if cycle == _C_GT:
        plant, eta = _gas_turbine_plant(inputs, model)
    else:
        plant, eta = _combined_plant(inputs, model)
    return plant, [("η térmico", _pct(eta))]


def _render_plant(
    plant: pl.PlantExergy, context: list[tuple[str, str]], key: str, system: UnitSystem
) -> None:
    st.markdown("### Resultado")
    uw = _u(_W, system)
    items = [
        (f"Exergía que entra [{uw}]", _val(plant.fuel_W, _W, system)),
        (f"Producto [{uw}]", _val(plant.product_W, _W, system)),
        ("Rendimiento exergético η_II", _pct(plant.efficiency)),
        (f"Exergía destruida [{uw}]", _val(plant.destroyed_W, _W, system)),
        (f"Exergía perdida [{uw}]", _val(plant.loss_W, _W, system)),
        *context,
    ]
    _metric_rows(items)
    st.markdown("#### Diagrama de Grassmann")
    rows = pl.grassmann_rows(plant)
    in_label = "Exergía que entra" if len(plant.inputs) > 1 else plant.inputs[0][0]
    product = "Trabajo neto" if plant.kind != "refrigeration" else plant.products[0][0]
    try:
        fig = grassmann_figure(
            rows,
            x_in=plant.fuel_W,
            in_label=in_label,
            product_label=product,
            scale=convert_from_si(1.0, _W, system),
            unit=uw,
        )
        st.plotly_chart(fig, width="stretch", key=f"{key}_grassmann")
        st.caption(
            "La banda azul es la exergía que entra y recorre la planta en el sentido del flujo; "
            "en cada componente se desprende hacia la derecha la que se destruye (gris) o se "
            "pierde (violeta), con el ancho proporcional. Lo que llega abajo es el producto. Los "
            "componentes que no destruyen nada (ideales) no tienen rama."
        )
    except Exception as exc:  # noqa: BLE001 — un gráfico que falla no tumba la página
        st.warning(f"No se pudo dibujar el diagrama de Grassmann: {exc}")
    for note in pl.plant_notes(plant):
        st.info(note)
    st.markdown("#### Cada componente")
    table_rows = []
    for c in plant.components:
        table_rows.append(
            {
                "Componente": c.name,
                f"Ẋ_D [{uw}]": "—" if c.is_loss else _val(c.destroyed_W, _W, system),
                "y_D": _pct(plant.y_D(c)) if not c.is_loss else "—",
                "ε": _pct(c.efficiency),
                f"Ẋ_F [{uw}]": _val(c.fuel_W, _W, system),
                f"Ẋ_P [{uw}]": _val(c.product_W, _W, system),
                f"Ẋ_L [{uw}]": _val(c.loss_W, _W, system) if c.loss_W else "—",
                "y*_D": _pct(plant.y_star_D(c)) if not c.is_loss else "—",
            }
        )
    _table(pd.DataFrame(table_rows))
    st.caption(
        "Ẋ_F: la exergía que gasta el componente; Ẋ_P: la que produce; Ẋ_D = T₀·Ṡ_gen, la que "
        "destruye; Ẋ_L, la que sale de la planta. ε = Ẋ_P/Ẋ_F; y_D = Ẋ_D/Ẋ_entra; "
        "y*_D = Ẋ_D/ΣẊ_D. Los disipativos (válvulas, condensador) no tienen producto."
    )
    if any(c.efficiency is not None for c in plant.components):
        st.plotly_chart(
            component_efficiency_figure(plant.components),
            width="stretch",
            key=f"{key}_eps",
        )
        st.caption(
            "ε de cada componente: un componente puede destruir poco en total y aun así ser "
            "malo (ε bajo), o al revés."
        )
    if plant.streams:
        with st.expander("🌊 Exergía de cada corriente", expanded=False):
            stream_rows = [
                {
                    "Corriente": s.label,
                    f"ṁ [{_u('mass_flow', system)}]": _val(s.m_kg_s, "mass_flow", system),
                    f"ψ [{_u(_EH, system)}]": _val(s.psi_J_per_kg, _EH, system),
                    **(
                        {f"e química [{_u(_EH, system)}]": _val(s.e_ch_J_per_kg, _EH, system)}
                        if any(x.e_ch_J_per_kg for x in plant.streams)
                        else {}
                    ),
                    f"Ẋ [{uw}]": _val(s.X_W, _W, system),
                }
                for s in plant.streams
            ]
            _table(pd.DataFrame(stream_rows))
    _render_steps(plant_steps(plant, system))
    _render_export(pl.plant_exergy_to_dict(plant, system), "exergia_planta", f"{key}_dl")


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§11 *Exergía*** (§11.1 estado muerto, §11.2 masa, §11.3 flujo, §11.4 espacio "
            "vacío, §11.5 calor, §11.6 fuente finita, §11.8 trabajo perdido, §11.9 balance, "
            "§11.10 rendimiento exergético) y **§16.13** (la exergía de un combustible). En el "
            "libro: Çengel & Boles, cap. 8."
        )
        st.markdown(
            "**Exergía física.** El trabajo máximo que se obtiene llevando el sistema al **estado "
            "muerto**, en equilibrio térmico y mecánico con el ambiente (T₀, p₀):"
        )
        st.latex(
            r"\begin{aligned}\phi &= (u - u_0) + p_0\,(v - v_0) \\ &\quad"
            r" - T_0\,(s - s_0) \ge 0\end{aligned}"
        )
        st.latex(r"\psi = (h - h_0) - T_0\,(s - s_0)")
        st.latex(r"x = \psi + \frac{V^2}{2} + g\,z")
        st.markdown(
            "ψ (flujo) puede ser negativa, φ (masa) nunca. Separada en una parte **térmica** "
            "(enfriar a p constante hasta T₀) y una **mecánica** (de p a p₀ a T₀; Kotas, 1985):"
        )
        st.latex(r"\psi_T = (h - h_i) - T_0\,(s - s_i)")
        st.latex(r"\psi_M = (h_i - h_0) - T_0\,(s_i - s_0)")
        st.markdown("**Calor y fuente finita** (§11.5 y §11.6):")
        st.latex(r"\dot X_Q = \dot Q\left(1 - \frac{T_0}{T}\right)")
        st.latex(r"\Phi = m\,c\left[(T - T_0) - T_0\ln\frac{T}{T_0}\right]")
        st.markdown(
            "**Exergía química.** El trabajo de llevar la sustancia, ya a T₀ y p₀, al "
            "equilibrio *químico* con el ambiente, reaccionando y mezclándose con sus componentes "
            "de referencia. Depende del **ambiente de referencia**: el *modelo II* de Szargut, "
            "Morris y Steward (1988) imita al real (aire con 2,2 kPa de vapor y 33,5 Pa de CO₂, "
            "el agua de mar, la corteza) y el *modelo I* de Ahrendts (1980) es un ambiente en "
            "equilibrio (las dos columnas de la tabla A-26 de Moran y Shapiro). Los gases del "
            "aire tienen exergía porque en el ambiente están diluidos:"
        )
        st.latex(r"\bar e^{ch} = -\bar R\,T_0\,\ln x^{00}")
        st.markdown(
            "Un compuesto, con el **método de Szargut**: se forma desde sus elementos y cada uno "
            "lleva la exergía de su sustancia de referencia."
        )
        st.latex(r"\bar e^{ch} = \Delta\bar g_f + \sum \nu\,\bar e^{ch}_{ref}")
        st.markdown("Una mezcla de gases ideales (con el agua que condensa a 25 °C):")
        st.latex(r"\bar e^{ch}_{mez} = \sum x_i\,\bar e^{ch}_i + \bar R\,T_0\sum x_i\ln x_i")
        st.markdown(
            "**Combustibles sólidos y líquidos** (Szargut y Styrylska, 1964; Kotas, 1985): sin "
            "la entropía de un carbón o una biomasa, se usa β = e/PCI de la materia seca en "
            "función de h/c, o/c, n/c y s/c (en masa):"
        )
        st.latex(
            r"\begin{aligned}\beta_\text{carbón} &= 1.0437 + 0.1882\,\tfrac{h}{c} \\ &\quad"
            r" + 0.0610\,\tfrac{o}{c} + 0.0404\,\tfrac{n}{c}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}e^{ch} &= \beta\,(PCI + W\,h_{fg}) \\ &\quad"
            r" + (e_S - PCI_S)\,S + e_w\,W\end{aligned}"
        )
        st.markdown(
            "El vademecum aproxima e ≈ PCI (§16.13): para los hidrocarburos φ = e/PCI ≈ 1,04 "
            "y para un carbón o una biomasa, 1,05–1,15."
        )
        st.markdown(
            "**Planta, por componente** (Bejan, Tsatsaronis y Moran, 1996, cap. 3): cada "
            "componente gasta una exergía Ẋ_F («combustible») para producir Ẋ_P; lo que falta se "
            "destruye (T₀·Ṡ_gen, §11.8) o se pierde (sale de la planta):"
        )
        st.latex(r"\dot X_F = \dot X_P + \dot X_D + \dot X_L")
        st.latex(r"\varepsilon = \frac{\dot X_P}{\dot X_F}")
        st.latex(r"\dot X_{entra} = \dot X_{producto} + \sum \dot X_D + \sum \dot X_L")
        st.latex(r"\eta_{II} = \frac{\dot X_{producto}}{\dot X_{entra}}")
        st.markdown(
            "El **diagrama de Grassmann** dibuja ese balance: una banda con la exergía que entra "
            "que se angosta en cada componente con lo que se destruye o se pierde. Convención: "
            "el calor que va al ambiente a T₀ (condensador, interenfriadores) no lleva exergía, "
            "así que lo que pierde el fluido ahí se **destruye** en ese componente (Cengel 10-8); "
            "lo que sale con una corriente (escape, chimenea) es **pérdida**. Las páginas "
            "Brayton y Ciclo combinado muestran el calor al agua de enfriamiento aparte."
        )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Exergía", page_icon="🔋", layout="centered")

st.subheader(SUBJECT)
st.title("🔋 Exergía")
st.markdown(
    "La **exergía** es el trabajo máximo que se puede obtener de algo llevándolo al "
    "equilibrio con el ambiente: dice cuánto *vale* la energía. Acá calculás la exergía "
    "**física** de un fluido, un calor o un cuerpo; la **química** de una sustancia, una mezcla "
    "o un combustible (tablas de Szargut); y la de **una planta completa, componente por "
    "componente**, con el diagrama de Grassmann para ver dónde se destruye."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Exergía")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="ex_mode")
if mode == _M_PHYS:
    what = st.radio("¿De qué?", _PHYS_WHAT, key="xf_what", horizontal=True)
    if what == _F_FLUID:
        _physical_fluid(system)
    elif what == _F_HEAT:
        _physical_heat(system)
    else:
        _physical_body(system)
elif mode == _M_CHEM:
    model: ch.ReferenceModel = st.selectbox(
        "Ambiente de referencia",
        list(ch.MODELS),
        format_func=lambda m: ch.MODELS[m],
        key="xq_model",
    )
    what = st.radio("¿De qué?", _CHEM_WHAT, key="xq_what", horizontal=True)
    if what == _Q_SPECIES:
        _chemical_species(system, model)
    elif what == _Q_MIX:
        _chemical_mixture(system, model)
    else:
        _chemical_fuel(system, model)
else:
    _plant_mode(system)
