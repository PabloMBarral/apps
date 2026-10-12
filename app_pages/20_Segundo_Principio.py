"""Página 20 — Segundo principio.

Fase 10.2. El segundo principio en cuatro modos (vademecum §9, §10.2 y §11):

- **Máquinas**: una máquina térmica, un refrigerador o una bomba de calor entre dos
  fuentes, con dos datos de Q_H, Q_C, W y el rendimiento (o el COP), o uno con la máquina
  reversible: el primer principio da lo que falta y el segundo, el veredicto contra
  Carnot, la entropía generada, la exergía destruida y el rendimiento exergético.
- **Clausius**: un ciclo que intercambia calor con varias fuentes (ΣQ/T ≤ 0).
- **Ciclo de Carnot**: con un fluido real o un gas ideal, proceso por proceso, en un
  sistema cerrado o en flujo estacionario.
- **Calor y entropía**: la entropía que genera un calor al pasar de una temperatura a otra
  (entre dos fuentes o a través de una pared) y la exergía que pierde.

Una máquina o un ciclo imposible no es un error: la página lo dice y muestra igual el
primer principio. El cálculo vive en :mod:`core.balances.second_law`; los gráficos, en
:mod:`ui.second_law_charts` y :mod:`ui.balance_charts`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core import fluids as fl
from core.balances import second_law as sl
from core.balances import substance as sb
from core.balances.second_law_procedure import (
    carnot_steps,
    clausius_steps,
    heat_path_steps,
    machine_steps,
)
from core.export import dict_to_csv
from core.gases import ideal as gi
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.balance_charts import state_diagram, waterfall_figure
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.second_law_charts import heat_exergy_figure, limit_figure
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.29.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_MACHINE = "Máquinas"
_M_CLAUSIUS = "Clausius"
_M_CARNOT = "Ciclo de Carnot"
_M_HEAT = "Calor y entropía"
_MODES = (_M_MACHINE, _M_CLAUSIUS, _M_CARNOT, _M_HEAT)

_T = "temperature"
_P = "pressure"
_V = "specific_volume"
_E = "specific_enthalpy"
_S = "specific_entropy"
_EN = "energy"
_M = "mass"

_DATA_OPTIONS = ("QH_QC", "QH_W", "QC_W", "QH_p", "QC_p", "W_p", "QH_rev", "QC_rev", "W_rev")
_KIND_LABELS = {"fluid": "Fluido real", "ideal_gas": "Gas ideal"}
_MODEL_LABELS = {"constant": "c_p constante (25 °C)", "variable": "c_p variable (NASA)"}
_PROP_LABELS = {"p": "presión p", "x": "título x", "v": "volumen específico v"}


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


def _table(frame: pd.DataFrame) -> None:
    """Tabla con todas las filas a la vista (sin barra de desplazamiento)."""
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _metric_rows(items: list[tuple[str, str]], per_row: int = 2) -> None:
    for k in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, items[k : k + per_row], strict=False):
            col.metric(label, value)


def _render_steps(steps: list[ProcedureStep]) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
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
        help="Podés cambiar cualquier dato: el resultado se actualiza solo.",
    )
    return name, names.index(name)


def _diagram(build: Any, key: str, caption: str = "") -> None:
    """Un gráfico que falla no tumba la página."""
    try:
        st.plotly_chart(build(), width="stretch", key=key)
        if caption:
            st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar el gráfico: {exc}")


def _verdict_box(verdict: str, violation: str | None) -> None:
    """El veredicto del segundo principio (vademecum §9.1 y §10.2)."""
    if violation:
        st.error(f"**Imposible.** {violation}", icon="🚫")
    elif verdict == "reversible":
        st.success("**Reversible**: S_gen = 0 (el límite de Carnot).", icon="✅")
    else:
        st.info("**Irreversible**: S_gen > 0 (posible, con exergía destruida).", icon="ℹ️")


def _notes(notes: tuple[str, ...]) -> None:
    for note in notes:
        st.caption(f"💡 {note}")


def _rates_input(default: bool, key: str) -> bool:
    choice = st.radio(
        "Calores y trabajo",
        ("cycle", "rate"),
        index=1 if default else 0,
        format_func=lambda c: "Por ciclo (energía)" if c == "cycle" else "Por unidad de tiempo",
        key=key,
        horizontal=True,
    )
    return choice == "rate"


def _ambient_input(default_K: float, key: str, label: str = "Ambiente T₀") -> float:
    return number_input_si(
        label=label,
        kind=_T,
        default_si=default_K,
        key=key,
        format="%.2f",
        help="La temperatura del ambiente para la exergía destruida, X_dest = T₀·S_gen "
        "(vademecum §11.8).",
    )


# ---------------------------------------------------------------------
# Máquinas
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _machine_cached(inputs: sl.MachineInputs) -> sl.MachineResult:
    return sl.solve_machine(inputs)


def _data_label(code: str, kind: sl.MachineKind) -> str:
    perf = "η" if kind == "engine" else sl.PERFORMANCE_NAMES[kind]
    names = {"QH": "Q_H", "QC": "Q_C", "W": "W", "p": perf}
    a, b = code.split("_")
    if b == "rev":
        return f"{names[a]} y es reversible (Carnot)"
    return f"{names[a]} y {names[b]}"


def _data_code(inp: sl.MachineInputs) -> str:
    names = {"Q_H": "QH", "Q_C": "QC", "W": "W", "performance": "p", "reversible": "rev"}
    code = "_".join(names[d] for d in inp.data)
    return code if code in _DATA_OPTIONS else "QH_QC"


def _possible_performance(kind: sl.MachineKind, T_H: float, T_C: float) -> float:
    """A mitad de camino entre la peor máquina y la reversible: siempre posible (y con
    COP_B ≥ 1 en la bomba de calor)."""
    rev = sl.carnot_performance(kind, max(T_H, T_C + 1.0), max(T_C, 1.0))
    if kind == "heat_pump":
        return 1.0 + 0.5 * (rev - 1.0)
    return 0.5 * rev


def _machine_mode(system: UnitSystem) -> None:
    name, idx = _example(sl.MACHINE_EXAMPLES, "sm_example")
    ex = sl.MACHINE_EXAMPLES[name]
    st.caption(f"📘 {ex.description}")
    key = f"sm_{idx}"
    kinds = list(sl.MACHINES)
    kind: sl.MachineKind = st.radio(
        "Máquina",
        kinds,
        index=kinds.index(ex.inputs.kind),
        format_func=lambda k: sl.MACHINES[k],
        key=f"{key}_kind",
        horizontal=True,
    )
    left, right = st.columns(2)
    with left:
        T_H = number_input_si(
            label="Fuente caliente T_H",
            kind=_T,
            default_si=ex.inputs.T_H_K,
            key=f"{key}_TH",
            format="%.2f",
            help="La temperatura de la fuente caliente (el horno, la cocina, la casa).",
        )
    with right:
        T_C = number_input_si(
            label="Fuente fría T_C",
            kind=_T,
            default_si=ex.inputs.T_C_K,
            key=f"{key}_TC",
            format="%.2f",
            help="La temperatura de la fuente fría (el río, el espacio refrigerado, afuera).",
        )
    rates = _rates_input(ex.inputs.rates, f"{key}_rates")
    E: QuantityKind = "power" if rates else _EN
    code = st.selectbox(
        "Datos",
        _DATA_OPTIONS,
        index=_DATA_OPTIONS.index(_data_code(ex.inputs)),
        format_func=lambda c: _data_label(c, kind),
        key=f"{key}_{kind}_data",
        help="Con dos datos el primer principio da el tercero (W = Q_H − Q_C).",
    )
    # valores por defecto: los del ejemplo (resuelto) si es la misma máquina; si no, una
    # máquina posible con el mismo trabajo (con los calores del ejemplo, una bomba de calor
    # entre las fuentes de la central sería imposible)
    try:
        base: sl.MachineResult | None = _machine_cached(ex.inputs)
    except ValueError:
        base = None
    if base is not None and kind == ex.inputs.kind:
        Q_H0, Q_C0, W0, p0 = base.Q_H, base.Q_C, base.W, base.performance
    else:
        W0 = base.W if base is not None and base.W > 0 else 400e3
        p0 = _possible_performance(kind, T_H, T_C)
        if kind == "engine":
            Q_H0 = W0 / p0
            Q_C0 = Q_H0 - W0
        elif kind == "refrigerator":
            Q_C0 = p0 * W0
            Q_H0 = Q_C0 + W0
        else:
            Q_H0 = p0 * W0
            Q_C0 = Q_H0 - W0
    a, b = code.split("_")
    values: dict[str, float] = {}
    cols = st.columns(2)
    defaults = {"QH": Q_H0, "QC": Q_C0, "W": W0}
    labels = {"QH": "Calor de la fuente caliente Q_H", "QC": "Calor de la fuente fría Q_C"}
    labels["W"] = "Trabajo W"
    for col, item in zip(cols, (a, b), strict=True):
        if item == "rev":
            continue
        with col:
            if item == "p":
                perf = "Rendimiento η" if kind == "engine" else sl.PERFORMANCE_NAMES[kind]
                values["performance"] = st.number_input(
                    perf,
                    value=float(p0),
                    min_value=0.0,
                    format="%.4g",
                    key=f"{key}_{kind}_perf",
                    help="η = W/Q_H (máquina), COP_R = Q_C/W, COP_B = Q_H/W (vademecum §9.2 "
                    "y §9.3).",
                )
            else:
                values[{"QH": "Q_H", "QC": "Q_C", "W": "W"}[item]] = number_input_si(
                    label=labels[item],
                    kind=E,
                    default_si=defaults[item],
                    key=f"{key}_{kind}_{'r' if rates else 'c'}_{item}",
                    format="%.6g",
                    min_value_si=0.0,
                )
    other_T0 = st.checkbox(
        "El ambiente está a otra temperatura",
        value=ex.inputs.T0_K is not None,
        key=f"{key}_hasT0",
        help="Por defecto el ambiente es la fuente de afuera: la fría de la máquina térmica y "
        "de la bomba de calor, la caliente del refrigerador.",
    )
    T0 = None
    if other_T0:
        T0 = _ambient_input(ex.inputs.T0_K or 298.15, f"{key}_Tamb")
    inputs = sl.MachineInputs(
        kind,
        T_H,
        T_C,
        rates=rates,
        reversible=b == "rev",
        T0_K=T0,
        **values,
    )
    try:
        r = _machine_cached(inputs)
    except ValueError as exc:
        st.error(str(exc))
        return
    _render_machine(r, system)


def _render_machine(r: sl.MachineResult, system: UnitSystem) -> None:
    inp = r.inputs
    E: QuantityKind = "power" if inp.rates else _EN
    S: QuantityKind = "entropy_flow" if inp.rates else "entropy"
    dot = "̇" if inp.rates else ""
    perf = sl.PERFORMANCE_NAMES[r.kind]
    rev = sl.REVERSIBLE_NAMES[r.kind]
    st.markdown("### Resultado")
    _verdict_box(r.verdict, r.violation)
    _metric_rows(
        [
            (f"Trabajo W{dot} [{_u(E, system)}]", _val(r.W, E, system)),
            (f"{perf} (de {rev})", f"{_num(r.performance, 4)} ({_num(r.performance_rev, 4)})"),
            (f"Calor Q{dot}_H [{_u(E, system)}]", _val(r.Q_H, E, system)),
            (f"Calor Q{dot}_C [{_u(E, system)}]", _val(r.Q_C, E, system)),
            (f"S{dot}_gen [{_u(S, system)}]", _val(r.S_gen, S, system)),
            (f"X{dot}_dest [{_u(E, system)}]", _val(r.X_dest, E, system)),
        ]
    )
    same = {
        "engine": "con el mismo Q_H",
        "refrigerator": "con el mismo Q_C",
        "heat_pump": "con el mismo Q_H",
    }[r.kind]
    reversible = sl.solve_machine(
        sl.MachineInputs(
            r.kind,
            inp.T_H_K,
            inp.T_C_K,
            Q_H=r.Q_H if r.kind != "refrigerator" else None,
            Q_C=r.Q_C if r.kind == "refrigerator" else None,
            reversible=True,
            rates=inp.rates,
        )
    )
    rows = []
    for symbol, a, b_, kind in (
        (f"Q{dot}_H", r.Q_H, reversible.Q_H, E),
        (f"Q{dot}_C", r.Q_C, reversible.Q_C, E),
        (f"W{dot}", r.W, reversible.W, E),
    ):
        rows.append(
            {
                "Magnitud": symbol,
                "Unidad": _u(kind, system),
                "Esta máquina": _val(a, kind, system),
                f"Reversible ({same})": _val(b_, kind, system),
            }
        )
    rows.append(
        {
            "Magnitud": perf,
            "Unidad": "—",
            "Esta máquina": _num(r.performance, 4),
            f"Reversible ({same})": _num(r.performance_rev, 4),
        }
    )
    rows.append(
        {
            "Magnitud": "η_ex",
            "Unidad": "—",
            "Esta máquina": _num(r.eta_ex, 4),
            f"Reversible ({same})": "1",
        }
    )
    _table(pd.DataFrame(rows))
    _notes(r.notes)
    _diagram(lambda: limit_figure(r, system), "sm_limit")
    if r.kind == "engine":
        items = [(f"Q{dot}_H", r.Q_H), (f"−Q{dot}_C", -r.Q_C)]
        title, total = "Primer principio: W = Q_H − Q_C", f"W{dot}"
    else:
        items = [(f"Q{dot}_C", r.Q_C), (f"W{dot}", r.W)]
        title, total = "Primer principio: Q_H = Q_C + W", f"Q{dot}_H"
    _diagram(lambda: waterfall_figure(title, items, total, E, system), "sm_energy")
    _render_steps(machine_steps(r, system))
    _render_export(sl.machine_to_dict(r), "segundo_principio_maquina", "sm_export")


# ---------------------------------------------------------------------
# Clausius
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _clausius_cached(inputs: sl.ClausiusInputs) -> sl.ClausiusResult:
    return sl.solve_clausius(inputs)


def _clausius_mode(system: UnitSystem) -> None:
    name, idx = _example(sl.CLAUSIUS_EXAMPLES, "sc_example")
    ex = sl.CLAUSIUS_EXAMPLES[name]
    st.caption(f"📘 {ex.description}")
    key = f"sc_{idx}"
    rates = _rates_input(ex.inputs.rates, f"{key}_rates")
    E: QuantityKind = "power" if rates else _EN
    n_res = int(
        st.number_input(
            "Cantidad de fuentes",
            min_value=2,
            max_value=6,
            value=len(ex.inputs.reservoirs),
            step=1,
            key=f"{key}_n",
        )
    )
    st.caption(
        "Q > 0 es el calor que el ciclo **recibe** de la fuente; Q < 0, el que le **entrega** "
        "(vademecum §9.1)."
    )
    reservoirs = []
    for k in range(n_res):
        base = (
            ex.inputs.reservoirs[k]
            if k < len(ex.inputs.reservoirs)
            else sl.Reservoir(300.0, -100e3, "")
        )
        label = base.label or f"la fuente {k + 1}"
        left, right = st.columns(2)
        with left:
            T = number_input_si(
                label=f"T de {label}",
                kind=_T,
                default_si=base.T_K,
                key=f"{key}_T{k}",
                format="%.2f",
            )
        with right:
            Q = number_input_si(
                label=f"Q de {label}",
                kind=E,
                default_si=base.Q,
                key=f"{key}_{'r' if rates else 'c'}_Q{k}",
                format="%.6g",
            )
        reservoirs.append(sl.Reservoir(T, Q, base.label if k < len(ex.inputs.reservoirs) else ""))
    T0 = _ambient_input(ex.inputs.T0_K, f"{key}_Tamb")
    try:
        r = _clausius_cached(sl.ClausiusInputs(tuple(reservoirs), rates=rates, T0_K=T0))
    except ValueError as exc:
        st.error(str(exc))
        return
    _render_clausius(r, system)


def _render_clausius(r: sl.ClausiusResult, system: UnitSystem) -> None:
    inp = r.inputs
    E: QuantityKind = "power" if inp.rates else _EN
    S: QuantityKind = "entropy_flow" if inp.rates else "entropy"
    dot = "̇" if inp.rates else ""
    st.markdown("### Resultado")
    _verdict_box(r.verdict, r.violation)
    items = [
        (f"ΣQ{dot}/T [{_u(S, system)}]", _val(r.clausius_sum, S, system)),
        (f"S{dot}_gen [{_u(S, system)}]", _val(r.S_gen, S, system)),
        (f"Trabajo neto W{dot} [{_u(E, system)}]", _val(r.W, E, system)),
        (f"X{dot}_dest [{_u(E, system)}]", _val(r.X_dest, E, system)),
    ]
    if r.efficiency is not None:
        items.append(("Rendimiento η", _num(r.efficiency, 4)))
        items.append(("1 − T_mín/T_máx", _num(r.carnot_limit, 4)))
    _metric_rows(items)
    rows = []
    for k, (res, term) in enumerate(zip(inp.reservoirs, r.terms, strict=True), 1):
        rows.append(
            {
                "Fuente": res.label or f"fuente {k}",
                f"T [{_u(_T, system)}]": _val(res.T_K, _T, system),
                f"Q [{_u(E, system)}]": _val(res.Q, E, system),
                f"Q/T [{_u(S, system)}]": _val(term, S, system),
            }
        )
    _table(pd.DataFrame(rows))
    _notes(r.notes)
    terms = [
        (res.label or f"fuente {k}", term)
        for k, (res, term) in enumerate(zip(inp.reservoirs, r.terms, strict=True), 1)
    ]
    _diagram(
        lambda: waterfall_figure(
            "Desigualdad de Clausius: ΣQ/T ≤ 0", terms, f"ΣQ{dot}/T", S, system
        ),
        "sc_terms",
    )
    _render_steps(clausius_steps(r, system))
    _render_export(sl.clausius_to_dict(r), "segundo_principio_clausius", "sc_export")


# ---------------------------------------------------------------------
# Ciclo de Carnot
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _carnot_cached(inputs: sl.CarnotInputs) -> sl.CarnotResult:
    return sl.solve_carnot(inputs)


def _tag(sub: sb.Substance) -> str:
    extra = sub.model if sub.kind == "ideal_gas" else ""
    return f"{sub.kind}-{sub.key}-{extra}".replace("(", "").replace(")", "")


def _substance_input(default: sb.Substance, key: str) -> sb.Substance:
    """Fluido real o gas ideal (un incompresible no hace un ciclo de Carnot)."""
    options = list(_KIND_LABELS)
    kind = st.radio(
        "Sustancia",
        options,
        index=options.index(default.kind) if default.kind in options else 0,
        format_func=lambda k: _KIND_LABELS[k],
        key=f"{key}_kind",
        horizontal=True,
    )
    if kind == "fluid":
        fluids = list(fl.SUPPORTED_FLUIDS)
        d = default.key if default.kind == "fluid" else "Water"
        name = st.selectbox(
            "Fluido",
            fluids,
            index=fluids.index(d),
            format_func=lambda k: fl.FLUID_NAMES_ES.get(k, k),
            key=f"{key}_fluid",
            help="Las propiedades salen de la ecuación de estado de CoolProp (las «tablas»).",
        )
        return sb.fluid_substance(name)
    gases = list(gi.IDEAL_GASES)
    d = default.key if default.kind == "ideal_gas" else "air"
    left, right = st.columns(2)
    with left:
        gas = st.selectbox(
            "Gas",
            gases,
            index=gases.index(d),
            format_func=lambda k: gi.gas(k).name,
            key=f"{key}_gas",
        )
    models = ["constant", "variable"]
    dm = default.model if default.kind == "ideal_gas" else "variable"
    with right:
        model = st.radio(
            "Modelo",
            models,
            index=models.index(dm),
            format_func=lambda m: _MODEL_LABELS[m],
            key=f"{key}_model",
        )
    return sb.ideal_gas_substance(gas, model)  # type: ignore[arg-type]


def _carnot_defaults(ex: sl.CarnotInputs, sub: sb.Substance) -> sl.CarnotInputs:
    """Los datos del ejemplo si es la misma sustancia; si no, unos que funcionan."""
    if sub == ex.substance:
        return ex
    if sub.kind == "ideal_gas":
        return sl.CarnotInputs(sub, 1000.0, 300.0, "p", 20e5, "p", 10e5, ex.mass_kg, ex.flow)
    limits = fl.fluid_limits(sub.key)
    T_lo = max(limits.T_triple_K, limits.T_min_K)
    T_H = T_lo + 0.85 * (limits.T_crit_K - T_lo)
    T_C = T_lo + 0.45 * (limits.T_crit_K - T_lo)
    return sl.CarnotInputs(sub, T_H, T_C, "x", 0.0, "x", 1.0, ex.mass_kg, ex.flow)


def _prop_input(
    sub: sb.Substance, prop0: sl.CarnotProp, value0: float, key: str, label: str
) -> tuple[sl.CarnotProp, float]:
    props: list[sl.CarnotProp] = ["p", "x", "v"] if sub.kind == "fluid" else ["p", "v"]
    left, right = st.columns(2)
    with left:
        prop = st.selectbox(
            f"Estado {label}: con",
            props,
            index=props.index(prop0) if prop0 in props else 0,
            format_func=lambda p: _PROP_LABELS[p],
            key=f"{key}_prop{label}",
        )
    with right:
        if prop == "x":
            value = st.number_input(
                f"x{'₁' if label == '1' else '₂'}",
                min_value=0.0,
                max_value=1.0,
                value=float(value0) if prop0 == "x" else (0.0 if label == "1" else 1.0),
                format="%.4f",
                key=f"{key}_x{label}",
            )
        else:
            kind: QuantityKind = _P if prop == "p" else _V
            default = value0 if prop0 == prop else (20e5 if label == "1" else 10e5)
            if prop == "v" and prop0 != "v":
                default = 0.05 if label == "1" else 0.2
            value = number_input_si(
                label=f"{'p' if prop == 'p' else 'v'}{'₁' if label == '1' else '₂'}",
                kind=kind,
                default_si=default,
                key=f"{key}_{prop}{label}",
                format="%.6g",
                min_value_si=0.0,
            )
    return prop, float(value)


def _carnot_mode(system: UnitSystem) -> None:
    name, idx = _example(sl.CARNOT_EXAMPLES, "sk_example")
    ex = sl.CARNOT_EXAMPLES[name]
    st.caption(f"📘 {ex.description}")
    key = f"sk_{idx}"
    sub = _substance_input(ex.inputs.substance, f"{key}_sub")
    d = _carnot_defaults(ex.inputs, sub)
    tag = f"{key}_{_tag(sub)}"
    left, right = st.columns(2)
    with left:
        T_H = number_input_si(
            label="Isoterma caliente T_H", kind=_T, default_si=d.T_H_K, key=f"{tag}_TH",
            format="%.2f",
        )  # fmt: skip
    with right:
        T_C = number_input_si(
            label="Isoterma fría T_C", kind=_T, default_si=d.T_C_K, key=f"{tag}_TC",
            format="%.2f",
        )  # fmt: skip
    st.caption("Los estados 1 (empieza a recibir calor) y 2 (termina) están a T_H.")
    prop1, value1 = _prop_input(sub, d.prop1, d.value1, tag, "1")
    prop2, value2 = _prop_input(sub, d.prop2, d.value2, tag, "2")
    left, right = st.columns(2)
    with left:
        mass = number_input_si(
            label="Masa m", kind=_M, default_si=d.mass_kg, key=f"{key}_m", format="%.4g",
            min_value_si=0.0,
        )  # fmt: skip
    with right:
        flows = list(sl.FLOWS)
        flow = st.radio(
            "Sistema",
            flows,
            index=flows.index(d.flow),
            format_func=lambda f: "Cerrado (pistón)" if f == "closed" else "Flujo estacionario",
            key=f"{key}_flow",
        )
    inputs = sl.CarnotInputs(sub, T_H, T_C, prop1, value1, prop2, value2, mass, flow)
    try:
        r = _carnot_cached(inputs)
    except ValueError as exc:
        st.error(str(exc))
        return
    _render_carnot(r, system)


def _states_table(r: sl.CarnotResult, system: UnitSystem) -> None:
    rows = []
    for symbol, kind, attr in (
        ("p", _P, "p"),
        ("T", _T, "T"),
        ("v", _V, "v"),
        ("u", _E, "u"),
        ("h", _E, "h"),
        ("s", _S, "s"),
    ):
        row = {"Propiedad": symbol, "Unidad": _u(kind, system)}
        for k, st_ in enumerate(r.states, 1):
            row[str(k)] = _val(getattr(st_, attr), kind, system)
        rows.append(row)
    row = {"Propiedad": "x", "Unidad": "—"}
    for k, st_ in enumerate(r.states, 1):
        row[str(k)] = _num(st_.x, 4) if st_.x is not None else "—"
    rows.append(row)
    row = {"Propiedad": "Región", "Unidad": "—"}
    for k, st_ in enumerate(r.states, 1):
        row[str(k)] = st_.region_es
    rows.append(row)
    _table(pd.DataFrame(rows))


def _render_carnot(r: sl.CarnotResult, system: UnitSystem) -> None:
    inp = r.inputs
    st.markdown("### Resultado")
    st.success(
        f"**Reversible**: η = 1 − T_C/T_H = {_num(r.eta_carnot, 4)}, con cualquier fluido.",
        icon="✅",
    )
    _metric_rows(
        [
            (f"Calor recibido Q_H [{_u(_EN, system)}]", _val(r.Q_H, _EN, system)),
            (f"Calor cedido Q_C [{_u(_EN, system)}]", _val(r.Q_C, _EN, system)),
            (f"Trabajo neto W [{_u(_EN, system)}]", _val(r.W_net, _EN, system)),
            ("Rendimiento η", _num(r.eta, 4)),
        ]
    )
    _states_table(r, system)
    e_name = "ΔU" if inp.flow == "closed" else "ΔH"
    rows = [
        {
            "Proceso": p.name,
            f"Q [{_u(_EN, system)}]": _val(p.Q, _EN, system),
            f"W [{_u(_EN, system)}]": _val(p.W, _EN, system),
            f"{e_name} [{_u(_EN, system)}]": _val(p.dE, _EN, system),
        }
        for p in r.processes
    ]
    _table(pd.DataFrame(rows))
    _notes(r.notes)
    points = [(str(k), s) for k, s in enumerate(r.states, 1)]
    paths = [("Ciclo de Carnot", r.path, True)]
    _diagram(lambda: state_diagram("Ts", inp.substance, points, paths, system), "sk_ts")
    _diagram(lambda: state_diagram("pv", inp.substance, points, paths, system), "sk_pv")
    items = [(f"W {p.start}→{p.end}", p.W) for p in r.processes]
    _diagram(
        lambda: waterfall_figure("El trabajo de cada proceso", items, "W neto", _EN, system),
        "sk_work",
    )
    _render_steps(carnot_steps(r, system))
    _render_export(sl.carnot_to_dict(r), "segundo_principio_carnot", "sk_export")


# ---------------------------------------------------------------------
# Calor y entropía
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _heat_cached(inputs: sl.HeatPathInputs) -> sl.HeatPathResult:
    return sl.solve_heat_path(inputs)


def _heat_mode(system: UnitSystem) -> None:
    name, idx = _example(sl.HEAT_PATH_EXAMPLES, "sq_example")
    ex = sl.HEAT_PATH_EXAMPLES[name]
    st.caption(f"📘 {ex.description}")
    key = f"sq_{idx}"
    rates = _rates_input(ex.inputs.rates, f"{key}_rates")
    E: QuantityKind = "heat_rate" if rates else _EN
    Q = number_input_si(
        label="Calor que pasa Q",
        kind=E,
        default_si=ex.inputs.Q,
        key=f"{key}_{'r' if rates else 'c'}_Q",
        format="%.6g",
        min_value_si=0.0,
    )
    n_pts = int(
        st.number_input(
            "Cantidad de temperaturas",
            min_value=2,
            max_value=6,
            value=len(ex.inputs.temperatures),
            step=1,
            key=f"{key}_n",
            help="El calor pasa de la primera a la última; cada tramo entre dos genera entropía.",
        )
    )
    temps, points = [], []
    n_ex = len(ex.inputs.temperatures)
    cols = st.columns(2)
    for k in range(n_pts):
        if k < n_ex:
            T0, label = ex.inputs.temperatures[k], ex.inputs.points[k]
        else:
            T0 = ex.inputs.temperatures[-1] - 10.0 * (k - n_ex + 1)
            label = f"el punto {k + 1}"
        with cols[k % 2]:
            temps.append(
                number_input_si(
                    label=f"T de {label}",
                    kind=_T,
                    default_si=T0,
                    key=f"{key}_T{k}",
                    format="%.2f",
                )  # fmt: skip
            )
        points.append(label)
    zones = [
        ex.inputs.zones[k] if k < len(ex.inputs.zones) and n_pts == n_ex else f"el tramo {k + 1}"
        for k in range(n_pts - 1)
    ]
    T0 = _ambient_input(ex.inputs.T0_K, f"{key}_Tamb")
    inputs = sl.HeatPathInputs(Q, tuple(temps), tuple(points), tuple(zones), rates, T0)
    try:
        r = _heat_cached(inputs)
    except ValueError as exc:
        st.error(str(exc))
        return
    _render_heat(r, system)


def _render_heat(r: sl.HeatPathResult, system: UnitSystem) -> None:
    inp = r.inputs
    E: QuantityKind = "heat_rate" if inp.rates else _EN
    S: QuantityKind = "entropy_rate" if inp.rates else "entropy"
    dot = "̇" if inp.rates else ""
    st.markdown("### Resultado")
    _verdict_box(r.verdict, r.violation)
    X = r.exergies
    _metric_rows(
        [
            (f"S{dot}_gen total [{_u(S, system)}]", _val(r.S_gen, S, system)),
            (f"X{dot}_dest [{_u(E, system)}]", _val(r.X_dest, E, system)),
            (f"Exergía al entrar [{_u(E, system)}]", _val(X[0], E, system)),
            (f"Exergía al salir [{_u(E, system)}]", _val(X[-1], E, system)),
        ]
    )
    rows = [
        {
            "Tramo": z.name,
            f"T entra [{_u(_T, system)}]": _val(z.T_in, _T, system),
            f"T sale [{_u(_T, system)}]": _val(z.T_out, _T, system),
            f"S{dot}_gen [{_u(S, system)}]": _val(z.S_gen, S, system),
        }
        for z in r.zones
    ]
    _table(pd.DataFrame(rows))
    _notes(r.notes)
    items = [(z.name, z.S_gen) for z in r.zones]
    _diagram(
        lambda: waterfall_figure("La entropía que genera cada tramo", items, "S_gen", S, system),
        "sq_sgen",
    )
    _diagram(lambda: heat_exergy_figure(r, system), "sq_exergy")
    _render_steps(heat_path_steps(r, system))
    _render_export(sl.heat_path_to_dict(r), "segundo_principio_calor", "sq_export")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§9** (segundo principio), **§10** (entropía) y **§11** (exergía). Las máquinas, el "
            "ciclo de Carnot y la entropía del calor siguen a Çengel y Boles (cap. 6 y 7)."
        )
        st.markdown(
            "**Desigualdad de Clausius** (§9.1), con Q > 0 el calor que el ciclo recibe de la "
            "fuente a la temperatura absoluta T:"
        )
        st.latex(r"\oint \frac{\delta Q}{T} \le 0 \qquad \sum \frac{Q}{T} \le 0")
        st.markdown("**Máquina térmica** (§9.2) y teorema de Carnot (§9.2.1):")
        st.latex(r"\eta = \frac{W}{Q_H} = 1 - \frac{Q_C}{Q_H}")
        st.latex(r"\eta_{\mathrm{Carnot}} = 1 - \frac{T_C}{T_H}")
        st.latex(r"0 \le \eta \le \eta_{\mathrm{Carnot}}")
        st.markdown("**Refrigerador y bomba de calor** (§9.3):")
        st.latex(
            r"\begin{aligned}\mathrm{COP}_R &= \frac{Q_C}{W} \\ "
            r"\mathrm{COP}_{R,\mathrm{rev}} &= \frac{T_C}{T_H - T_C}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}\mathrm{COP}_B &= \frac{Q_H}{W} \\ "
            r"\mathrm{COP}_{B,\mathrm{rev}} &= \frac{T_H}{T_H - T_C}\end{aligned}"
        )
        st.latex(r"\mathrm{COP}_B = \mathrm{COP}_R + 1")
        st.markdown(
            "**Entropía** (§10.1 y §10.2): una fuente que recibe Q a T cambia su entropía en "
            "Q/T; el ciclo vuelve a su estado, así que la entropía generada es la de las "
            "fuentes:"
        )
        st.latex(r"dS = \left(\frac{\delta Q}{T}\right)_{\mathrm{rev}}")
        st.latex(r"S_{\mathrm{gen}} = \Delta S_{\mathrm{sist}} + \Delta S_{\mathrm{med}} \ge 0")
        st.latex(r"S_{\mathrm{gen}} = -\sum \frac{Q}{T}")
        st.markdown(
            "**Exergía** (§11.5, §11.8 y §11.10): la de un calor, la destruida y el rendimiento "
            "exergético de cada máquina:"
        )
        st.latex(r"X_Q = Q\left(1 - \frac{T_0}{T}\right)")
        st.latex(r"X_{\mathrm{dest}} = T_0\,S_{\mathrm{gen}}")
        st.latex(r"\eta_{\mathrm{ex,MT}} = \frac{W}{Q_H\left(1 - \frac{T_0}{T_H}\right)}")
        st.latex(r"\eta_{\mathrm{ex,R}} = \frac{Q_C\left(\frac{T_0}{T_C} - 1\right)}{W}")
        st.latex(r"\eta_{\mathrm{ex,B}} = \frac{Q_H\left(1 - \frac{T_0}{T_H}\right)}{W}")
        st.markdown(
            "**Ciclo de Carnot** (Çengel §6-7): dos isotermas y dos isoentrópicas; con "
            "cualquier fluido,"
        )
        st.latex(
            r"\begin{aligned}Q_H &= m\,T_H\,(s_2 - s_1) \\ "
            r"Q_C &= m\,T_C\,(s_2 - s_1)\end{aligned}"
        )
        st.markdown(
            "**Un calor entre dos temperaturas** (Çengel §7-2 y §7-13): genera entropía "
            "aunque nada cambie de estado."
        )
        st.latex(r"S_{\mathrm{gen}} = \frac{Q}{T_{\mathrm{sale}}} - \frac{Q}{T_{\mathrm{entra}}}")


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Segundo principio", page_icon="🔁", layout="centered")

st.subheader(SUBJECT)
st.title("🔁 Segundo principio")
st.markdown(
    "Las **máquinas térmicas, los refrigeradores y las bombas de calor** contra el límite de "
    "**Carnot**, la **desigualdad de Clausius** de un ciclo con varias fuentes, el **ciclo de "
    "Carnot** de un fluido real o un gas ideal, y la **entropía que genera un calor** al "
    "pasar de una temperatura a otra: si el proceso es posible, cuánta entropía genera y "
    "cuánta exergía destruye."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Segundo principio")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="sp_mode", horizontal=True)
if mode == _M_MACHINE:
    _machine_mode(system)
elif mode == _M_CLAUSIUS:
    _clausius_mode(system)
elif mode == _M_CARNOT:
    _carnot_mode(system)
else:
    _heat_mode(system)
