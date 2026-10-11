"""Página 19 — Primer principio.

Fase 10.1. Los balances de energía y de entropía en tres modos (vademecum §3, §10 y §13),
con un fluido real (CoolProp), un gas ideal (c_p constante o variable) o un incompresible:

- **Sistema cerrado**: un proceso (volumen, presión o temperatura constante, politrópico,
  adiabático reversible, contra una presión exterior o con el estado final dado), con el
  trabajo de frontera, el calor, ΔU, ΔS, la entropía generada y la exergía destruida; y el
  equilibrio térmico entre dos cuerpos o con un reservorio.
- **Flujo estacionario**: un dispositivo (tobera, difusor, válvula, turbina, compresor,
  bomba, calentador), la cámara de mezcla y el intercambiador sin mezcla.
- **Régimen transitorio**: el llenado y el vaciado de un tanque rígido.

Con S_gen < 0 el proceso es imposible: la página lo dice y muestra igual el balance de
energía (el primer principio no lo prohíbe). El cálculo vive en :mod:`core.balances`; los
gráficos, en :mod:`ui.balance_charts`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from core import fluids as fl
from core.balances import closed as cl
from core.balances import steady_flow as sf
from core.balances import substance as sb
from core.balances import transient as tr
from core.balances.balances_procedure import (
    charging_steps,
    closed_steps,
    device_steps,
    discharging_steps,
    equilibrium_steps,
    exchanger_steps,
    mixing_steps,
)
from core.export import dict_to_csv
from core.gases import ideal as gi
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.balance_charts import (
    evolution_figure,
    exchanger_tq_figure,
    state_diagram,
    waterfall_figure,
)
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.28.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_CLOSED = "Sistema cerrado"
_M_FLOW = "Flujo estacionario"
_M_TRANS = "Régimen transitorio"
_MODES = (_M_CLOSED, _M_FLOW, _M_TRANS)
_C_PROC = "Un proceso"
_C_EQ = "Equilibrio térmico"
_F_DEV = "Un dispositivo"
_F_MIX = "Cámara de mezcla"
_F_HX = "Intercambiador"
_T_CHG = "Llenado de un tanque"
_T_DIS = "Vaciado de un tanque"

_P: QuantityKind = "pressure"
_T: QuantityKind = "temperature"
_V: QuantityKind = "specific_volume"
_E: QuantityKind = "specific_enthalpy"
_S: QuantityKind = "specific_entropy"
_EN: QuantityKind = "energy"
_EN_S: QuantityKind = "entropy"
_M: QuantityKind = "mass"
_VOL: QuantityKind = "volume"
_PW: QuantityKind = "power"
_SF: QuantityKind = "entropy_flow"
_MF: QuantityKind = "mass_flow"

_KIND_LABELS = {"fluid": "Fluido real", "ideal_gas": "Gas ideal", "incompressible": "Incompresible"}
_MODEL_LABELS = {"constant": "c_p constante (25 °C)", "variable": "c_p variable (NASA)"}
_LETTERS: dict[str, tuple[str, QuantityKind | None]] = {
    "T": ("Temperatura T", _T),
    "P": ("Presión p (absoluta)", _P),
    "V": ("Volumen específico v", _V),
    "H": ("Entalpía h", _E),
    "S": ("Entropía s", _S),
    "U": ("Energía interna u", _E),
    "X": ("Título x", None),
}
_END_LABELS = {
    "T": "T₂",
    "p": "p₂",
    "v": "v₂",
    "V": "V₂",
    "x": "x₂",
    "Q": "El calor Q",
    "W": "El trabajo W",
}
_OUT_LABELS = {
    "T": "T₂",
    "x": "x₂",
    "eta": "η_s",
    "V": "ω₂",
    "w": "El trabajo w",
    "q": "El calor q",
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
        help="Podés cambiar la sustancia y cualquier dato: el resultado se actualiza solo.",
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


def _verdict_box(S_gen: float, violation: str | None, what: str) -> None:
    """El veredicto del segundo principio (vademecum §10.2)."""
    if violation:
        st.error(f"**Imposible.** {violation}", icon="🚫")
    elif S_gen == 0.0:
        st.success(f"**Reversible**: {what} = 0.", icon="✅")
    else:
        st.info(f"**Irreversible**: {what} > 0 (posible, con exergía destruida).", icon="ℹ️")


def _notes(notes: tuple[str, ...]) -> None:
    for note in notes:
        st.caption(f"💡 {note}")


def _tag(sub: sb.Substance) -> str:
    """Una etiqueta de la sustancia para las keys de los widgets que dependen de ella."""
    extra = sub.model if sub.kind == "ideal_gas" else ""
    return f"{sub.kind}-{sub.key}-{extra}".replace("(", "").replace(")", "")


def _states_table(columns: list[tuple[str, sb.ThermoState]], system: UnitSystem) -> None:
    """Las propiedades por fila y un estado por columna (se lee mejor en el celular)."""
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
        for label, s in columns:
            row[label] = _val(getattr(s, attr), kind, system)
        rows.append(row)
    row = {"Propiedad": "x", "Unidad": "—"}
    for label, s in columns:
        row[label] = _num(s.x, 4) if s.x is not None else "—"
    rows.append(row)
    row = {"Propiedad": "Región", "Unidad": "—"}
    for label, s in columns:
        row[label] = s.region_es
    rows.append(row)
    _table(pd.DataFrame(rows))


# ---------------------------------------------------------------------
# Sustancia y estados
# ---------------------------------------------------------------------


def _material_input(default: sb.Substance, key: str, label: str = "Material") -> sb.Substance:
    options = [*sb.INCOMPRESSIBLES, "custom"]
    d = default.key if default.kind == "incompressible" else "water"
    mat = st.selectbox(
        label,
        options,
        index=options.index(d),
        format_func=lambda k: "A elección (c y ρ)" if k == "custom" else sb.INCOMPRESSIBLES[k].name,
        key=f"{key}_mat",
    )
    base = (
        default.material
        if default.kind == "incompressible" and default.key == mat
        else (sb.INCOMPRESSIBLES.get(mat) or sb.INCOMPRESSIBLES["water"])
    )
    left, right = st.columns(2)
    with left:
        c = number_input_si(
            label="Calor específico c",
            kind="specific_heat",
            default_si=base.c,
            key=f"{key}_{mat}_c",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        rho = number_input_si(
            label="Densidad ρ",
            kind="density",
            default_si=base.rho,
            key=f"{key}_{mat}_rho",
            format="%.5g",
            min_value_si=0.0,
        )
    return sb.incompressible_substance(mat, c=c, rho=rho)


def _substance_input(
    default: sb.Substance, key: str, kinds: tuple[str, ...] = tuple(_KIND_LABELS)
) -> sb.Substance:
    """Fluido real, gas ideal (con su modelo) o incompresible (con c y ρ)."""
    options = list(kinds)
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
    if kind == "ideal_gas":
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
    return _material_input(default, key)


@st.cache_data(show_spinner=False)
def _state_cached(sub: sb.Substance, pair: sb.PairCode, a: float, b: float) -> sb.ThermoState:
    return sb.state(sub, pair, a, b)


def _fallback_state(sub: sb.Substance) -> sb.ThermoState:
    if sub.kind == "fluid":
        guess = fl.suggested_inputs(sub.key, "TP")
        return _state_cached(sub, "TP", guess["t"], guess["p"])
    if sub.kind == "ideal_gas":
        return _state_cached(sub, "TP", 300.0, 1e5)
    return _state_cached(sub, "TP", 298.15, 101325.0)


def _default_state(
    sub: sb.Substance, ex_sub: sb.Substance, ex_state: sb.ThermoState
) -> sb.ThermoState:
    """El estado del ejemplo; con otra sustancia, su T y su p (o uno de la sustancia)."""
    if sub == ex_sub:
        return ex_state
    try:
        return _state_cached(sub, "TP", ex_state.T, ex_state.p)
    except ValueError:
        return _fallback_state(sub)


def _state_input(
    sub: sb.Substance,
    default_pair: sb.PairCode,
    default: sb.ThermoState,
    key: str,
    title: str = "",
) -> tuple[sb.PairCode, float, float]:
    """Un estado con dos propiedades (el par se elige)."""
    if title:
        st.markdown(f"**{title}**")
    tag = _tag(sub)
    pairs = list(sb.allowed_pairs(sub))
    dp = default_pair if default_pair in pairs else pairs[0]
    pair: sb.PairCode = st.selectbox(
        "Datos",
        pairs,
        index=pairs.index(dp),
        format_func=lambda p: sb.PAIR_NAMES[p],
        key=f"{key}_{tag}_pair",
    )
    values = {
        "T": default.T,
        "P": default.p,
        "V": default.v,
        "H": default.h,
        "S": default.s,
        "U": default.u,
        "X": default.x if default.x is not None else 0.5,
    }
    out: list[float] = []
    cols = st.columns(2)
    for col, letter in zip(cols, pair, strict=True):
        label, kind = _LETTERS[letter]
        with col:
            if kind is None:
                out.append(
                    float(
                        st.number_input(
                            label,
                            min_value=0.0,
                            max_value=1.0,
                            value=float(values["X"]),
                            format="%.4f",
                            key=f"{key}_{tag}_{pair}_X",
                        )
                    )
                )
            else:
                out.append(
                    number_input_si(
                        label=label,
                        kind=kind,
                        default_si=values[letter],
                        key=f"{key}_{tag}_{pair}_{letter}",
                        format="%.6g",
                        min_value_si=0.0 if letter in "PV" else None,
                    )
                )
    return pair, out[0], out[1]


def _ambient_inputs(
    T_b: float | None, T0: float, key: str, heat_label: str = "el calor"
) -> tuple[float | None, float]:
    """La temperatura de la fuente o el medio del calor (T_b) y la del ambiente (T₀)."""
    other = st.checkbox(
        f"{heat_label.capitalize()} pasa desde (o hacia) un medio a otra temperatura que el "
        "ambiente",
        value=T_b is not None,
        key=f"{key}_has_Tb",
        help="T_b es la temperatura del medio en la frontera por donde pasa el calor (el T_k "
        "del vademecum §10.3). Si no la das, es la del ambiente.",
    )
    left, right = st.columns(2)
    Tb_out: float | None = None
    if other:
        with left:
            Tb_out = number_input_si(
                label="Temperatura del medio T_b",
                kind=_T,
                default_si=T_b if T_b is not None else T0 + 100.0,
                key=f"{key}_Tb",
                format="%.5g",
            )
    with right if other else left:
        T0_out = number_input_si(
            label="Temperatura del ambiente T₀",
            kind=_T,
            default_si=T0,
            key=f"{key}_T0",
            format="%.5g",
            help="El estado muerto para la exergía destruida X_dest = T₀·S_gen.",
        )
    return Tb_out, T0_out


def _pv_ts(
    sub: sb.Substance,
    points: list[tuple[str, sb.ThermoState]],
    paths: list[tuple[str, Any, bool]],
    system: UnitSystem,
    key: str,
    caption: str = "",
) -> None:
    _diagram(lambda: state_diagram("pv", sub, points, paths, system), f"{key}_pv")
    _diagram(lambda: state_diagram("Ts", sub, points, paths, system), f"{key}_ts", caption)


def _isentrope(sub: sb.Substance, s1: sb.ThermoState, p2: float) -> list[sb.ThermoState]:
    out = []
    for p in np.geomspace(s1.p, p2, 31):
        try:
            out.append(sb.state(sub, "PS", float(p), s1.s))
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------
# Sistema cerrado
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _closed_cached(inputs: cl.ClosedInputs) -> cl.ClosedResult:
    return cl.solve_closed(inputs)


@st.cache_data(show_spinner=False)
def _equilibrium_cached(inputs: cl.EquilibriumInputs) -> cl.EquilibriumResult:
    return cl.solve_equilibrium(inputs)


def _end_default(
    end: str,
    proc: str,
    ex: cl.ClosedInputs,
    ref: cl.ClosedResult | None,
    s1: sb.ThermoState,
    m: float,
) -> float:
    """El dato final por defecto: el del ejemplo, el del resultado del ejemplo o uno razonable."""
    if ex.process == proc and ex.end == end and math.isfinite(ex.end_value):
        return ex.end_value
    if ref is not None:
        s2 = ref.state2
        guess = {"T": s2.T, "p": s2.p, "v": s2.v, "V": ref.V2, "Q": ref.Q, "W": ref.W_b}
        if s2.x is not None:
            guess["x"] = s2.x
        if end in guess:
            return guess[end]
    return {
        "T": s1.T + 50.0,
        "p": 2.0 * s1.p,
        "v": 0.5 * s1.v,
        "V": 0.5 * m * s1.v,
        "x": 1.0,
        "Q": 50e3 * m,
        "W": 0.0,
    }[end]


def _end_input(end: str, default: float, key: str) -> float:
    if end == "x":
        return float(
            st.number_input(
                "x₂", 0.0, 1.0, value=float(min(max(default, 0.0), 1.0)), format="%.4f", key=key
            )
        )
    kinds: dict[str, tuple[str, QuantityKind]] = {
        "T": ("Temperatura final T₂", _T),
        "p": ("Presión final p₂", _P),
        "v": ("Volumen específico final v₂", _V),
        "V": ("Volumen final V₂", _VOL),
        "Q": ("Calor que recibe Q (< 0 si cede)", _EN),
        "W": ("Trabajo que hace W (< 0 si recibe)", _EN),
    }
    label, kind = kinds[end]
    return number_input_si(
        label=label,
        kind=kind,
        default_si=default,
        key=key,
        format="%.6g",
        min_value_si=0.0 if end in "pvV" else None,
    )


def _closed_process_mode(system: UnitSystem) -> None:
    name, idx = _example(cl.CLOSED_EXAMPLES, "pc_example")
    ex = cl.CLOSED_EXAMPLES[name]
    inp = ex.inputs
    key = f"pc_{idx}"
    st.caption(f"📘 {ex.description}")
    try:
        ref: cl.ClosedResult | None = _closed_cached(inp)
    except ValueError:
        ref = None
    sub = _substance_input(inp.substance, f"{key}_sub")
    tag = _tag(sub)
    ex_s1 = _state_cached(inp.substance, inp.pair1, inp.a1, inp.b1)
    if sub != inp.substance:
        ref = None
    pair1, a1, b1 = _state_input(
        sub, inp.pair1, _default_state(sub, inp.substance, ex_s1), f"{key}_1", "Estado 1"
    )
    try:
        s1 = _state_cached(sub, pair1, a1, b1)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    ex_mass = inp.mass_kg if inp.mass_kg is not None else (inp.volume_m3 or 1.0) / ex_s1.v
    amounts = ["mass", "volume"]
    amount = st.radio(
        "¿Qué cantidad das?",
        amounts,
        index=0 if inp.mass_kg is not None else 1,
        format_func=lambda a: "La masa m" if a == "mass" else "El volumen inicial V₁",
        key=f"{key}_amount",
        horizontal=True,
    )
    if amount == "mass":
        mass: float | None = number_input_si(
            label="Masa m",
            kind=_M,
            default_si=ex_mass,
            key=f"{key}_m",
            format="%.5g",
            min_value_si=0.0,
        )
        volume: float | None = None
        m = mass or 1.0
    else:
        volume = number_input_si(
            label="Volumen inicial V₁",
            kind=_VOL,
            default_si=inp.volume_m3 if inp.volume_m3 is not None else ex_mass * ex_s1.v,
            key=f"{key}_V",
            format="%.5g",
            min_value_si=0.0,
        )
        mass = None
        m = (volume or 1.0) / s1.v
    procs = list(cl.allowed_processes(sub))
    proc = st.selectbox(
        "Proceso",
        procs,
        index=procs.index(inp.process) if inp.process in procs else 0,
        format_func=lambda p: cl.PROCESSES[p],
        key=f"{key}_{tag}_proc",
    )
    ends = list(cl.allowed_ends(proc, sub))
    end = st.radio(
        "¿Qué sabés del final?",
        ends,
        index=ends.index(inp.end) if inp.end in ends else 0,
        format_func=lambda e: _END_LABELS[e],
        key=f"{key}_{tag}_{proc}_end",
        horizontal=True,
    )
    kw: dict[str, Any] = {}
    if proc == "general":
        st.markdown("**Estado 2**")
        ex_s2 = (
            _state_cached(inp.substance, inp.end_pair, inp.end_a, inp.end_b)
            if inp.end_pair is not None
            else s1
        )
        end_pair, end_a, end_b = _state_input(
            sub, inp.end_pair or "TP", _default_state(sub, inp.substance, ex_s2), f"{key}_2"
        )
        kw.update(end_pair=end_pair, end_a=end_a, end_b=end_b)
    value = _end_input(
        end, _end_default(end, proc, inp, ref, s1, m), f"{key}_{tag}_{proc}_{end}_value"
    )
    if proc == "polytropic":
        kw["n"] = float(
            st.number_input(
                "Exponente n (p·vⁿ = cte)",
                0.05,
                5.0,
                value=float(inp.n),
                step=0.05,
                format="%.3f",
                key=f"{key}_n",
            )
        )
    if proc == "p_ext":
        kw["p_ext_Pa"] = number_input_si(
            label="Presión exterior p_ext",
            kind=_P,
            default_si=inp.p_ext_Pa if inp.p_ext_Pa is not None else 0.5 * s1.p,
            key=f"{key}_pext",
            format="%.5g",
            min_value_si=0.0,
        )
    if proc in ("v_const", "p_const", "p_ext") or sub.kind == "incompressible":
        kw["W_in_J"] = number_input_si(
            label="Otro trabajo que entra W_ent (resistencia, paleta)",
            kind=_EN,
            default_si=inp.W_in_J,
            key=f"{key}_Win",
            format="%.5g",
            help="Trabajo que entra por otra vía que la frontera: siempre se disipa y genera "
            "entropía.",
        )
    T_b, T0 = _ambient_inputs(inp.T_b_K, inp.T0_K, key)
    try:
        r = _closed_cached(
            cl.ClosedInputs(
                sub,
                pair1,
                a1,
                b1,
                mass_kg=mass,
                volume_m3=volume,
                process=proc,
                end=end,
                end_value=value,
                T_b_K=T_b,
                T0_K=T0,
                **kw,
            )
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_closed(r, system)


def _render_closed(r: cl.ClosedResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    _verdict_box(r.S_gen, r.violation, "S_gen")
    _metric_rows(
        [
            (f"Trabajo de frontera W_b [{_u(_EN, system)}]", _val(r.W_b, _EN, system)),
            (f"Calor Q [{_u(_EN, system)}]", _val(r.Q, _EN, system)),
            (f"ΔU [{_u(_EN, system)}]", _val(r.dU, _EN, system)),
            (f"ΔS [{_u(_EN_S, system)}]", _val(r.dS, _EN_S, system)),
            (f"S_gen [{_u(_EN_S, system)}]", _val(r.S_gen, _EN_S, system)),
            (f"X_dest = T₀·S_gen [{_u(_EN, system)}]", _val(r.X_dest, _EN, system)),
        ]
    )
    _notes(r.notes)
    st.markdown("#### Estados")
    cols = [("1", r.state1), ("2", r.state2)]
    if r.reversible_state is not None:
        cols.append(("2s (reversible)", r.reversible_state))
    _states_table(cols, system)
    st.caption(
        f"Masa m = {_val(r.mass, _M, system)} {_u(_M, system)}; V₁ = {_val(r.V1, _VOL, system)} "
        f"y V₂ = {_val(r.V2, _VOL, system)} {_u(_VOL, system)}."
    )
    items = [("Q", r.Q), ("−W_b", -r.W_b)]
    if r.W_in:
        items.append(("W_ent", r.W_in))
    _diagram(
        lambda: waterfall_figure("Primer principio: Q − W = ΔU", items, "ΔU", _EN, system),
        "pc_energy",
    )
    _diagram(
        lambda: waterfall_figure(
            "Segundo principio: ΔS = Q/T_b + S_gen",
            [("Q/T_b", r.S_heat), ("S_gen", r.S_gen)],
            "ΔS",
            _EN_S,
            system,
        ),
        "pc_entropy",
        "Lo que entra con el calor (Q/T_b) más lo que se genera (S_gen) es lo que cambia la "
        "entropía del sistema.",
    )
    sub = r.inputs.substance
    if sub.kind != "incompressible":
        paths: list[tuple[str, Any, bool]] = [("Proceso", r.path, r.quasi_static)]
        points = [("1", r.state1), ("2", r.state2)]
        if r.reversible_state is not None:
            paths.append(
                ("Expansión reversible", _isentrope(sub, r.state1, r.reversible_state.p), True)
            )
            points.append(("2s", r.reversible_state))
        _pv_ts(
            sub,
            points,
            paths,
            system,
            "pc",
            "Rayado: no hay un camino de estados de equilibrio (solo se une el principio con el "
            "final)."
            if not r.quasi_static
            else "",
        )
    _render_steps(closed_steps(r, system))
    _render_export(cl.closed_to_dict(r), "sistema_cerrado", "pc_dl")


def _body_input(default: cl.Body, key: str, title: str) -> cl.Body:
    st.markdown(f"**{title}**")
    sub = _material_input(default.substance, key)
    left, right = st.columns(2)
    with left:
        m = number_input_si(
            label="Masa",
            kind=_M,
            default_si=default.mass_kg,
            key=f"{key}_m",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        T = number_input_si(
            label="Temperatura inicial",
            kind=_T,
            default_si=default.T_K,
            key=f"{key}_T",
            format="%.5g",
        )
    return cl.Body(sub, m, T, default.label)


def _equilibrium_mode(system: UnitSystem) -> None:
    name, idx = _example(cl.EQUILIBRIUM_EXAMPLES, "pe_example")
    ex = cl.EQUILIBRIUM_EXAMPLES[name]
    inp = ex.inputs
    key = f"pe_{idx}"
    st.caption(f"📘 {ex.description}")
    a = _body_input(inp.body_a, f"{key}_a", f"Cuerpo a ({inp.body_a.label or 'el caliente'})")
    others = ["body", "reservoir"]
    other = st.radio(
        "¿Con qué se equilibra?",
        others,
        index=0 if inp.body_b is not None else 1,
        format_func=lambda o: (
            "Con otro cuerpo (recipiente aislado)" if o == "body" else "Con un reservorio"
        ),
        key=f"{key}_other",
        horizontal=True,
    )
    b: cl.Body | None = None
    T_res: float | None = None
    if other == "body":
        default_b = inp.body_b or cl.Body(
            sb.incompressible_substance("water"), 10.0, 293.15, "agua"
        )
        b = _body_input(default_b, f"{key}_b", f"Cuerpo b ({default_b.label or 'el frío'})")
    else:
        T_res = number_input_si(
            label="Temperatura del reservorio",
            kind=_T,
            default_si=inp.T_reservoir_K if inp.T_reservoir_K is not None else inp.T0_K,
            key=f"{key}_Tres",
            format="%.5g",
        )
    T0 = number_input_si(
        label="Temperatura del ambiente T₀",
        kind=_T,
        default_si=inp.T0_K,
        key=f"{key}_T0",
        format="%.5g",
    )
    try:
        r = _equilibrium_cached(cl.EquilibriumInputs(a, b, T_res, T0))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _verdict_box(r.S_gen, None, "S_gen")
    _metric_rows(
        [
            (f"Temperatura final [{_u(_T, system)}]", _val(r.T_final, _T, system)),
            (f"S_gen [{_u(_EN_S, system)}]", _val(r.S_gen, _EN_S, system)),
            (f"Calor del cuerpo a [{_u(_EN, system)}]", _val(r.changes[0].Q, _EN, system)),
            (f"X_dest [{_u(_EN, system)}]", _val(r.X_dest, _EN, system)),
        ]
    )
    _notes(r.notes)
    rows = [
        {
            "Parte": c.body.label or c.body.substance.label,
            f"T₁ [{_u(_T, system)}]": _val(c.body.T_K, _T, system),
            f"Q [{_u(_EN, system)}]": _val(c.Q, _EN, system),
            f"ΔS [{_u(_EN_S, system)}]": _val(c.dS, _EN_S, system),
        }
        for c in r.changes
    ]
    if r.reservoir_dS is not None:
        rows.append(
            {
                "Parte": "reservorio",
                f"T₁ [{_u(_T, system)}]": _val(r.T_final, _T, system),
                f"Q [{_u(_EN, system)}]": _val(-r.changes[0].Q, _EN, system),
                f"ΔS [{_u(_EN_S, system)}]": _val(r.reservoir_dS, _EN_S, system),
            }
        )
    _table(pd.DataFrame(rows))
    dS_parts = [c.dS for c in r.changes]
    if r.reservoir_dS is not None:
        dS_parts.append(r.reservoir_dS)
    items = [(f"ΔS ({row['Parte']})", d) for row, d in zip(rows, dS_parts, strict=True)]
    _diagram(
        lambda: waterfall_figure("La entropía de cada parte", items, "S_gen", _EN_S, system),
        "pe_entropy",
        "El que se enfría pierde entropía, pero el que recibe el calor gana más (lo recibe a "
        "menor temperatura): la suma es la entropía generada.",
    )
    _render_steps(equilibrium_steps(r, system))
    _render_export(cl.equilibrium_to_dict(r), "equilibrio_termico", "pe_dl")


def _closed_mode(system: UnitSystem) -> None:
    kind = st.radio("¿Qué calculás?", (_C_PROC, _C_EQ), key="pc_kind", horizontal=True)
    if kind == _C_PROC:
        _closed_process_mode(system)
    else:
        _equilibrium_mode(system)


# ---------------------------------------------------------------------
# Flujo estacionario
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _device_cached(inputs: sf.DeviceInputs) -> sf.DeviceResult:
    return sf.solve_device(inputs)


@st.cache_data(show_spinner=False)
def _mixing_cached(inputs: sf.MixingInputs) -> sf.MixingResult:
    return sf.solve_mixing(inputs)


@st.cache_data(show_spinner=False)
def _exchanger_cached(inputs: sf.ExchangerInputs) -> sf.ExchangerResult:
    return sf.solve_exchanger(inputs)


def _out_default(
    out: str, ex: sf.DeviceInputs, ref: sf.DeviceResult | None, s1: sb.ThermoState
) -> float:
    if ex.out == out and math.isfinite(ex.out_value):
        return ex.out_value
    if ref is not None:
        guess = {"T": ref.state2.T, "eta": 0.85, "V": ref.V2, "w": ref.w, "q": ref.q}
        if ref.state2.x is not None:
            guess["x"] = ref.state2.x
        if out in guess:
            return guess[out]
    return {"T": s1.T, "x": 0.9, "eta": 0.85, "V": 100.0, "w": 0.0, "q": 0.0}[out]


def _basis_default(value: float, ex_rates: bool, rates: bool, m_dot: float) -> float:
    """Un calor o un trabajo del ejemplo, pasado a la base pedida (por kg o potencia)."""
    if ex_rates == rates:
        return value
    return value * m_dot if rates else value / m_dot


def _device_mode(system: UnitSystem) -> None:
    name, idx = _example(sf.DEVICE_EXAMPLES, "pd_example")
    ex = sf.DEVICE_EXAMPLES[name]
    inp = ex.inputs
    key = f"pd_{idx}"
    st.caption(f"📘 {ex.description}")
    try:
        ref: sf.DeviceResult | None = _device_cached(inp)
    except ValueError:
        ref = None
    sub = _substance_input(inp.substance, f"{key}_sub")
    tag = _tag(sub)
    if sub != inp.substance:
        ref = None
    devices = list(sf.allowed_devices(sub))
    dev = st.selectbox(
        "Dispositivo",
        devices,
        index=devices.index(inp.device) if inp.device in devices else 0,
        format_func=lambda d: sf.DEVICES[d],
        key=f"{key}_{tag}_dev",
    )
    ex_s1 = _state_cached(inp.substance, inp.pair1, inp.a1, inp.b1)
    pair1, a1, b1 = _state_input(
        sub, inp.pair1, _default_state(sub, inp.substance, ex_s1), f"{key}_1", "Entrada"
    )
    try:
        s1 = _state_cached(sub, pair1, a1, b1)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    left, right = st.columns(2)
    with left:
        V1 = number_input_si(
            label="Velocidad ω₁",
            kind="speed",
            default_si=inp.V1,
            key=f"{key}_V1",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        z1 = number_input_si(
            label="Altura z₁", kind="length", default_si=inp.z1, key=f"{key}_z1", format="%.5g"
        )
    st.markdown("**Salida**")
    p2: float | None
    if dev == "heater":
        same_p = st.checkbox(
            "Sin pérdida de carga (p₂ = p₁)", value=inp.p2_Pa is None, key=f"{key}_samep"
        )
        p2 = (
            None
            if same_p
            else number_input_si(
                label="Presión de salida p₂",
                kind=_P,
                default_si=inp.p2_Pa if inp.p2_Pa is not None else 0.95 * s1.p,
                key=f"{key}_p2",
                format="%.5g",
                min_value_si=0.0,
            )
        )
    else:
        default_p2 = (
            inp.p2_Pa
            if inp.p2_Pa is not None and inp.device == dev
            else (2.0 * s1.p if dev in ("compressor", "pump", "diffuser") else 0.5 * s1.p)
        )
        p2 = number_input_si(
            label="Presión de salida p₂",
            kind=_P,
            default_si=default_p2,
            key=f"{key}_{dev}_p2",
            format="%.5g",
            min_value_si=0.0,
        )
    outs = list(sf.allowed_out(dev, sub))
    out = "T"
    value = math.nan
    if outs:
        out = st.radio(
            "¿Qué más sabés de la salida?",
            outs,
            index=outs.index(inp.out) if inp.out in outs and inp.device == dev else 0,
            format_func=lambda o: _OUT_LABELS[o],
            key=f"{key}_{tag}_{dev}_out",
            horizontal=True,
        )
        rates = st.checkbox(
            "Dar el calor y el trabajo como potencias (Q̇, Ẇ) y no por kg",
            value=inp.rates,
            key=f"{key}_rates",
        )
        dflt = _out_default(out, inp, ref, s1)
        vkey = f"{key}_{tag}_{dev}_{out}{'_r' if rates else ''}"
        if out == "T":
            value = number_input_si(
                label="Temperatura de salida T₂", kind=_T, default_si=dflt, key=vkey, format="%.5g"
            )
        elif out in ("x", "eta"):
            value = float(
                st.number_input(
                    "x₂" if out == "x" else "Rendimiento isoentrópico η_s",
                    0.0,
                    1.0,
                    value=float(min(max(dflt, 0.0), 1.0)),
                    format="%.4f",
                    key=vkey,
                )
            )
        elif out == "V":
            value = number_input_si(
                label="Velocidad de salida ω₂",
                kind="speed",
                default_si=dflt,
                key=vkey,
                format="%.5g",
                min_value_si=0.0,
            )
        else:
            label = "Trabajo que hace" if out == "w" else "Calor que recibe"
            if rates:
                d = _basis_default(
                    dflt, inp.rates and inp.out == out, True, ref.m_dot if ref is not None else 1.0
                )
                value = number_input_si(
                    label=f"{label} ({'Ẇ' if out == 'w' else 'Q̇'})",
                    kind=_PW,
                    default_si=d,
                    key=vkey,
                    format="%.5g",
                )
            else:
                value = number_input_si(
                    label=f"{label} ({out}, por kg)",
                    kind=_E,
                    default_si=dflt,
                    key=vkey,
                    format="%.5g",
                )
    else:
        rates = False
    V2: float | None = None
    if dev not in ("nozzle", "diffuser", "valve"):
        if st.checkbox(
            "La velocidad de salida es distinta de la de entrada",
            value=inp.V2 is not None,
            key=f"{key}_hasV2",
        ):
            V2 = number_input_si(
                label="Velocidad ω₂",
                kind="speed",
                default_si=inp.V2 if inp.V2 is not None else V1,
                key=f"{key}_V2",
                format="%.5g",
                min_value_si=0.0,
            )
    z2 = number_input_si(
        label="Altura z₂", kind="length", default_si=inp.z2, key=f"{key}_z2", format="%.5g"
    )
    q_known, w_known = 0.0, 0.0
    m_ref = ref.m_dot if ref is not None else 1.0
    known_kind = _PW if rates else _E
    suffix = " (potencia)" if rates else " (por kg)"
    rkey = "_r" if rates else ""
    if dev != "valve" and out != "eta" and dev != "heater":
        q_known = number_input_si(
            label=f"Calor que recibe{suffix} (< 0 si pierde)",
            kind=known_kind,
            default_si=_basis_default(inp.q, inp.rates, rates, m_ref),
            key=f"{key}_q{rkey}",
            format="%.5g",
        )
    if dev == "heater":
        w_known = number_input_si(
            label=f"Trabajo que hace{suffix} (< 0 si entra: una resistencia)",
            kind=known_kind,
            default_si=_basis_default(inp.w, inp.rates, rates, m_ref),
            key=f"{key}_w{rkey}",
            format="%.5g",
        )
    flows = ["m", "Vdot", "inlet", "power"]
    flow = st.radio(
        "Caudal",
        flows,
        index=flows.index(inp.flow) if inp.device == dev else 0,
        format_func=lambda f: sf.FLOW_NAMES[f],
        key=f"{key}_flow",
    )
    fkinds: dict[str, tuple[str, QuantityKind]] = {
        "m": ("Caudal másico ṁ", _MF),
        "Vdot": ("Caudal volumétrico V̇₁", "volume_flow"),
        "inlet": ("Área de entrada A₁", "area"),
        "power": ("Potencia |Ẇ| o |Q̇|", _PW),
    }
    flabel, fkind = fkinds[flow]
    flow_value = number_input_si(
        label=flabel,
        kind=fkind,
        default_si=inp.flow_value
        if inp.flow == flow
        else {"m": 1.0, "Vdot": 0.1, "inlet": 0.01, "power": 1e6}[flow],
        key=f"{key}_flow_{flow}",
        format="%.5g",
        min_value_si=0.0,
    )
    A1: float | None = None
    if flow in ("m", "Vdot") and st.checkbox(
        "Dar el área de entrada A₁ (ω₁ = ṁ·v₁/A₁)", value=inp.A1_m2 is not None, key=f"{key}_hasA1"
    ):
        A1 = number_input_si(
            label="Área de entrada A₁",
            kind="area",
            default_si=inp.A1_m2 if inp.A1_m2 is not None else 0.01,
            key=f"{key}_A1",
            format="%.5g",
            min_value_si=0.0,
        )
    T_b, T0 = _ambient_inputs(inp.T_b_K, inp.T0_K, key)
    try:
        r = _device_cached(
            sf.DeviceInputs(
                dev,
                sub,
                pair1,
                a1,
                b1,
                p2_Pa=p2,
                out=out,
                out_value=value,
                flow=flow,
                flow_value=flow_value,
                V1=V1,
                A1_m2=A1,
                V2=V2,
                z1=z1,
                z2=z2,
                q=q_known,
                w=w_known,
                rates=rates,
                T_b_K=T_b,
                T0_K=T0,
            )
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_device(r, system)


def _render_device(r: sf.DeviceResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    _verdict_box(r.s_gen, r.violation, "s_gen")
    items = [
        (f"Caudal ṁ [{_u(_MF, system)}]", _val(r.m_dot, _MF, system)),
        (f"T₂ [{_u(_T, system)}]", _val(r.state2.T, _T, system)),
    ]
    if r.w or r.inputs.device in ("turbine", "compressor", "pump"):
        items.append((f"Potencia Ẇ [{_u(_PW, system)}]", _val(r.W_dot, _PW, system)))
    if r.q or r.inputs.device == "heater":
        items.append((f"Calor Q̇ [{_u(_PW, system)}]", _val(r.Q_dot, _PW, system)))
    if r.inputs.device in ("nozzle", "diffuser") or r.V1 or r.V2:
        items.append((f"ω₂ [{_u('speed', system)}]", _val(r.V2, "speed", system)))
    if r.eta_s is not None:
        items.append(("η_s", f"{100 * r.eta_s:.1f} %".replace(".", ",")))
    items += [
        (f"s_gen [{_u(_S, system)}]", _val(r.s_gen, _S, system)),
        (f"Ẋ_dest = T₀·Ṡ_gen [{_u(_PW, system)}]", _val(r.X_dest_dot, _PW, system)),
    ]
    _metric_rows(items)
    _notes(r.notes)
    st.markdown("#### Estados")
    cols = [("1", r.state1), ("2", r.state2)]
    if r.state2s is not None:
        cols.append(("2s", r.state2s))
    _states_table(cols, system)
    extra = []
    if r.A1 is not None:
        extra.append(f"A₁ = {_val(r.A1, 'area', system)}")
    if r.A2 is not None:
        extra.append(f"A₂ = {_val(r.A2, 'area', system)}")
    if extra:
        st.caption(f"Áreas: {', '.join(extra)} {_u('area', system)}.")
    terms = [("q", r.q), ("−w", -r.w)]
    _diagram(
        lambda: waterfall_figure(
            "Primer principio por kg: q − w = Δh + Δe_c + Δe_p",
            terms,
            "Δh + Δe_c + Δe_p",
            _E,
            system,
        ),
        "pd_energy",
    )
    _diagram(
        lambda: waterfall_figure(
            "Segundo principio por kg: s₂ − s₁ = q/T_b + s_gen",
            [("q/T_b", r.q / r.T_b), ("s_gen", r.s_gen)],
            "s₂ − s₁",
            _S,
            system,
        ),
        "pd_entropy",
    )
    sub = r.inputs.substance
    if sub.kind != "incompressible":
        dev = r.inputs.device
        quasi = len(r.path) > 2 and dev != "valve"
        name = {"valve": "Estrangulamiento (h constante)", "heater": "Calentamiento a p constante"}
        paths: list[tuple[str, Any, bool]] = [(name.get(dev, "Proceso real"), r.path, quasi)]
        points = [("1", r.state1), ("2", r.state2)]
        if r.path_s:
            paths.append(("Isoentrópico", r.path_s, True))
            assert r.state2s is not None
            points.append(("2s", r.state2s))
        _pv_ts(
            sub,
            points,
            paths,
            system,
            "pd",
            "Rayado: el proceso real no es una sucesión de estados de equilibrio; solo se "
            "unen la entrada y la salida.",
        )
    _render_steps(device_steps(r, system))
    _render_export(sf.device_to_dict(r), "dispositivo", "pd_dl")


def _mixing_mode(system: UnitSystem) -> None:
    name, idx = _example(sf.MIXING_EXAMPLES, "pm_example")
    ex = sf.MIXING_EXAMPLES[name]
    inp = ex.inputs
    key = f"pm_{idx}"
    st.caption(f"📘 {ex.description}")
    sub = _substance_input(inp.substance, f"{key}_sub", ("fluid", "ideal_gas", "incompressible"))
    unknowns = ["m1", "m2", "out"]
    ex_unknown = "m1" if inp.inlet_1.m_dot is None else "m2" if inp.inlet_2.m_dot is None else "out"
    unknown = st.radio(
        "¿Qué falta?",
        unknowns,
        index=unknowns.index(ex_unknown),
        format_func=lambda u: {
            "m1": "El caudal de la entrada 1",
            "m2": "El caudal de la entrada 2",
            "out": "La salida (con los dos caudales)",
        }[u],
        key=f"{key}_unknown",
    )
    try:
        ref_mix: sf.MixingResult | None = _mixing_cached(inp)
    except ValueError:
        ref_mix = None
    streams = []
    ex_states = [_state_cached(inp.substance, s.pair, s.a, s.b) for s in (inp.inlet_1, inp.inlet_2)]
    for k, (stream, other) in enumerate(
        ((inp.inlet_1, inp.inlet_2), (inp.inlet_2, inp.inlet_1)), 1
    ):
        ex_st = ex_states[k - 1]
        pair, a, b = _state_input(
            sub,
            stream.pair,
            _default_state(sub, inp.substance, ex_st),
            f"{key}_{k}",
            f"Entrada {k}" + (f" ({stream.label})" if stream.label else ""),
        )
        m_dot: float | None = None
        if unknown != f"m{k}":
            d = (
                stream.m_dot
                if stream.m_dot is not None
                else (ref_mix.m_dots[k - 1] if ref_mix is not None else (other.m_dot or 1.0))
            )
            m_dot = number_input_si(
                label=f"Caudal ṁ{k}",
                kind=_MF,
                default_si=d,
                key=f"{key}_m{k}",
                format="%.5g",
                min_value_si=0.0,
            )
        streams.append(sf.MixingStream(pair, a, b, m_dot, stream.label))
    st.markdown("**Salida**")
    out = "h"
    value = math.nan
    if unknown != "out":
        outs = ["T", "x"] if sub.kind == "fluid" else ["T"]
        out = st.radio(
            "Dato de la salida",
            outs,
            index=0,
            format_func=lambda o: _OUT_LABELS[o],
            key=f"{key}_out",
            horizontal=True,
        )
        if out == "T":
            value = number_input_si(
                label="Temperatura de salida T₃",
                kind=_T,
                default_si=inp.out_value
                if inp.out == "T"
                else (ref_mix.outlet.T if ref_mix is not None else ex_states[0].T),
                key=f"{key}_T3",
                format="%.5g",
            )
        else:
            value = float(
                st.number_input("x₃", 0.0, 1.0, value=0.5, format="%.4f", key=f"{key}_x3")
            )
    same_p = st.checkbox(
        "La salida a la menor de las presiones de entrada",
        value=inp.p_out_Pa is None,
        key=f"{key}_samep",
    )
    p3 = (
        None
        if same_p
        else number_input_si(
            label="Presión de salida p₃",
            kind=_P,
            default_si=inp.p_out_Pa or 1e5,
            key=f"{key}_p3",
            format="%.5g",
            min_value_si=0.0,
        )
    )
    Q = number_input_si(
        label="Calor que recibe la cámara Q̇ (< 0 si pierde)",
        kind=_PW,
        default_si=inp.Q_dot_W,
        key=f"{key}_Q",
        format="%.5g",
    )
    T_b, T0 = _ambient_inputs(inp.T_b_K, inp.T0_K, key)
    try:
        r = _mixing_cached(
            sf.MixingInputs(sub, streams[0], streams[1], out, value, p3, Q, T_b, T0)  # type: ignore[arg-type]
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _verdict_box(r.S_gen_dot, r.violation, "Ṡ_gen")
    _metric_rows(
        [
            (f"ṁ₁ [{_u(_MF, system)}]", _val(r.m_dots[0], _MF, system)),
            (f"ṁ₂ [{_u(_MF, system)}]", _val(r.m_dots[1], _MF, system)),
            (f"T₃ [{_u(_T, system)}]", _val(r.outlet.T, _T, system)),
            (f"ṁ₃ [{_u(_MF, system)}]", _val(r.m_out, _MF, system)),
            (f"Ṡ_gen [{_u(_SF, system)}]", _val(r.S_gen_dot, _SF, system)),
            (f"Ẋ_dest [{_u(_PW, system)}]", _val(r.X_dest_dot, _PW, system)),
        ]
    )
    _notes(r.notes)
    _states_table([("1", r.inlets[0]), ("2", r.inlets[1]), ("3", r.outlet)], system)
    if sub.kind != "incompressible":
        _diagram(
            lambda: state_diagram(
                "Ts",
                sub,
                [("1", r.inlets[0]), ("2", r.inlets[1]), ("3", r.outlet)],
                [
                    ("1 → 3", (r.inlets[0], r.outlet), False),
                    ("2 → 3", (r.inlets[1], r.outlet), False),
                ],
                system,
            ),
            "pm_ts",
            "Las dos corrientes terminan en el mismo estado 3: la que estaba más fría gana "
            "entropía y la más caliente pierde, pero la suma (pesada con los caudales) sube.",
        )
    _render_steps(mixing_steps(r, system))
    _render_export(sf.mixing_to_dict(r), "camara_de_mezcla", "pm_dl")


def _stream_input(
    default: sf.ExchangerStream,
    key: str,
    title: str,
    unknown: str,
    tag: str,
    ref: tuple[float, sb.ThermoState] | None,
) -> sf.ExchangerStream:
    """Una corriente; ``ref`` es (caudal, salida) del resultado del ejemplo, para los valores
    por defecto de lo que en el ejemplo era la incógnita."""
    st.markdown(f"**{title}**")
    sub = _substance_input(default.substance, f"{key}_sub")
    ex_st = _state_cached(default.substance, default.pair, default.a, default.b)
    pair, a, b = _state_input(
        sub, default.pair, _default_state(sub, default.substance, ex_st), f"{key}_in"
    )
    m_dot: float | None = None
    if unknown != f"m{tag}":
        m_dot = number_input_si(
            label="Caudal ṁ",
            kind=_MF,
            default_si=default.m_dot
            if default.m_dot is not None
            else (ref[0] if ref is not None else 1.0),
            key=f"{key}_m",
            format="%.5g",
            min_value_si=0.0,
        )
    out: str = "h"
    value = math.nan
    if unknown != f"out{tag}":
        outs = ["T", "x"] if sub.kind == "fluid" else ["T"]
        out = st.radio(
            "Dato de la salida",
            outs,
            index=0,
            format_func=lambda o: _OUT_LABELS[o],
            key=f"{key}_out",
            horizontal=True,
        )
        if out == "T":
            value = number_input_si(
                label="Temperatura de salida",
                kind=_T,
                default_si=default.out_value
                if default.out == "T"
                else (ref[1].T if ref is not None else ex_st.T),
                key=f"{key}_Tout",
                format="%.5g",
            )
        else:
            value = float(
                st.number_input(
                    "x de salida", 0.0, 1.0, value=1.0, format="%.4f", key=f"{key}_xout"
                )
            )
    return sf.ExchangerStream(sub, pair, a, b, m_dot, out, value, None, default.label)  # type: ignore[arg-type]


def _exchanger_mode(system: UnitSystem) -> None:
    name, idx = _example(sf.EXCHANGER_EXAMPLES, "px_example")
    ex = sf.EXCHANGER_EXAMPLES[name]
    inp = ex.inputs
    key = f"px_{idx}"
    st.caption(f"📘 {ex.description}")
    sa, sb_ = inp.stream_a, inp.stream_b
    unknowns = ["mA", "mB", "outA", "outB"]
    ex_unknown = (
        "mA"
        if sa.m_dot is None
        else "mB"
        if sb_.m_dot is None
        else "outA"
        if sa.out == "h"
        else "outB"
    )
    names = {
        "mA": f"El caudal de A ({sa.label or 'A'})",
        "mB": f"El caudal de B ({sb_.label or 'B'})",
        "outA": f"La salida de A ({sa.label or 'A'})",
        "outB": f"La salida de B ({sb_.label or 'B'})",
    }
    unknown = st.radio(
        "¿Qué falta?",
        unknowns,
        index=unknowns.index(ex_unknown),
        format_func=lambda u: names[u],
        key=f"{key}_unknown",
    )
    try:
        ref = _exchanger_cached(inp)
        refs = [(ref.m_dots[i], ref.outlets[i]) for i in range(2)]
    except ValueError:
        refs = [None, None]  # type: ignore[list-item]
    a = _stream_input(sa, f"{key}_A", f"Corriente A ({sa.label or 'A'})", unknown, "A", refs[0])
    b = _stream_input(sb_, f"{key}_B", f"Corriente B ({sb_.label or 'B'})", unknown, "B", refs[1])
    Q = number_input_si(
        label="Calor que recibe desde afuera Q̇ (< 0 si pierde)",
        kind=_PW,
        default_si=inp.Q_dot_W,
        key=f"{key}_Q",
        format="%.5g",
    )
    T_b, T0 = _ambient_inputs(inp.T_b_K, inp.T0_K, key, "el calor perdido")
    try:
        r = _exchanger_cached(sf.ExchangerInputs(a, b, Q, T_b, T0))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _verdict_box(r.S_gen_dot, r.violation, "Ṡ_gen")
    _metric_rows(
        [
            (f"ṁ_A [{_u(_MF, system)}]", _val(r.m_dots[0], _MF, system)),
            (f"ṁ_B [{_u(_MF, system)}]", _val(r.m_dots[1], _MF, system)),
            (f"Calor transferido [{_u(_PW, system)}]", _val(r.Q_transfer, _PW, system)),
            (f"Salida de A [{_u(_T, system)}]", _val(r.outlets[0].T, _T, system)),
            (f"Salida de B [{_u(_T, system)}]", _val(r.outlets[1].T, _T, system)),
            (f"Ṡ_gen [{_u(_SF, system)}]", _val(r.S_gen_dot, _SF, system)),
        ]
    )
    st.caption(f"Ẋ_dest = T₀·Ṡ_gen = {_val(r.X_dest_dot, _PW, system)} {_u(_PW, system)}.")
    _notes(r.notes)
    _states_table(
        [("A1", r.inlets[0]), ("A2", r.outlets[0]), ("B1", r.inlets[1]), ("B2", r.outlets[1])],
        system,
    )
    _diagram(
        lambda: exchanger_tq_figure(r, system),
        "px_tq",
        "Cada curva es la temperatura de una corriente contra el calor que intercambió, desde "
        "el extremo frío. A contracorriente, las curvas no se pueden cruzar.",
    )
    _diagram(
        lambda: waterfall_figure(
            "La entropía de cada corriente",
            [
                (f"Δ{'Ṡ'}_A", r.entropy_change(0)),
                (f"Δ{'Ṡ'}_B", r.entropy_change(1)),
                ("−Q̇/T_b", -inp.Q_dot_W / r.T_b if inp.Q_dot_W else 0.0),
            ],
            "Ṡ_gen",
            _SF,
            system,
        ),
        "px_entropy",
    )
    _render_steps(exchanger_steps(r, system))
    _render_export(sf.exchanger_to_dict(r), "intercambiador", "px_dl")


def _flow_mode(system: UnitSystem) -> None:
    kind = st.radio("¿Qué calculás?", (_F_DEV, _F_MIX, _F_HX), key="pf_kind", horizontal=True)
    if kind == _F_DEV:
        _device_mode(system)
    elif kind == _F_MIX:
        _mixing_mode(system)
    else:
        _exchanger_mode(system)


# ---------------------------------------------------------------------
# Régimen transitorio
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _charging_cached(inputs: tr.ChargingInputs) -> tr.ChargingResult:
    return tr.solve_charging(inputs)


@st.cache_data(show_spinner=False)
def _discharging_cached(inputs: tr.DischargingInputs) -> tr.DischargingResult:
    return tr.solve_discharging(inputs)


def _charging_mode(system: UnitSystem) -> None:
    name, idx = _example(tr.CHARGING_EXAMPLES, "pl_example")
    ex = tr.CHARGING_EXAMPLES[name]
    inp = ex.inputs
    key = f"pl_{idx}"
    st.caption(f"📘 {ex.description}")
    sub = _substance_input(inp.substance, f"{key}_sub", ("fluid", "ideal_gas"))
    V = number_input_si(
        label="Volumen del tanque V",
        kind=_VOL,
        default_si=inp.volume_m3,
        key=f"{key}_V",
        format="%.5g",
        min_value_si=0.0,
    )
    ex_line = _state_cached(inp.substance, inp.line_pair, inp.line_a, inp.line_b)
    lpair, la, lb = _state_input(
        sub,
        inp.line_pair,
        _default_state(sub, inp.substance, ex_line),
        f"{key}_line",
        "Línea (lo que entra)",
    )
    try:
        line = _state_cached(sub, lpair, la, lb)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    empty = st.radio(
        "¿Cómo está el tanque al principio?",
        ["empty", "full"],
        index=0 if inp.initial_pair is None else 1,
        format_func=lambda e: "Vacío" if e == "empty" else "Con fluido",
        key=f"{key}_empty",
        horizontal=True,
    )
    ipair: sb.PairCode | None = None
    ia = ib = math.nan
    if empty == "full":
        ex_init = (
            _state_cached(inp.substance, inp.initial_pair, inp.initial_a, inp.initial_b)
            if inp.initial_pair is not None
            else line
        )
        ipair, ia, ib = _state_input(
            sub,
            inp.initial_pair or "TP",
            _default_state(sub, inp.substance, ex_init),
            f"{key}_ini",
            "Estado inicial del tanque",
        )
    p2 = number_input_si(
        label="Presión final p₂",
        kind=_P,
        default_si=min(inp.p2_Pa, line.p),
        key=f"{key}_p2",
        format="%.5g",
        min_value_si=0.0,
    )
    end = st.radio(
        "¿Qué sabés del final?",
        ["Q", "T"],
        index=0 if inp.end == "Q" else 1,
        format_func=lambda e: "El calor (0: aislado)" if e == "Q" else "La temperatura final T₂",
        key=f"{key}_end",
        horizontal=True,
    )
    Q = 0.0
    T2 = math.nan
    if end == "Q":
        Q = number_input_si(
            label="Calor que recibe el tanque Q (< 0 si cede)",
            kind=_EN,
            default_si=inp.Q_J,
            key=f"{key}_Q",
            format="%.5g",
        )
    else:
        T2 = number_input_si(
            label="Temperatura final T₂",
            kind=_T,
            default_si=inp.T2_K if math.isfinite(inp.T2_K) else line.T,
            key=f"{key}_T2",
            format="%.5g",
        )
    T_b, T0 = _ambient_inputs(inp.T_b_K, inp.T0_K, key)
    try:
        r = _charging_cached(
            tr.ChargingInputs(sub, V, lpair, la, lb, p2, ipair, ia, ib, end, Q, T2, T_b, T0)  # type: ignore[arg-type]
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _verdict_box(r.S_gen, r.violation, "S_gen")
    _metric_rows(
        [
            (f"T₂ [{_u(_T, system)}]", _val(r.state2.T, _T, system)),
            (f"Masa que entra m_e [{_u(_M, system)}]", _val(r.m_in, _M, system)),
            (f"Calor Q [{_u(_EN, system)}]", _val(r.Q, _EN, system)),
            (f"Masa final m₂ [{_u(_M, system)}]", _val(r.m2, _M, system)),
            (f"S_gen [{_u(_EN_S, system)}]", _val(r.S_gen, _EN_S, system)),
            (f"X_dest [{_u(_EN, system)}]", _val(r.X_dest, _EN, system)),
        ]
    )
    _notes(r.notes)
    cols = (
        [("Línea", r.line)]
        + ([("1", r.state1)] if r.state1 is not None else [])
        + [("2", r.state2)]
    )
    _states_table(cols, system)
    _diagram(lambda: evolution_figure(r.path, system, "El tanque mientras se llena"), "pl_evol")
    s1 = r.state1
    items = ([("m₁(u₂ − u₁)", r.m1 * (r.state2.u - s1.u))] if s1 is not None else []) + [
        ("m_e(u₂ − h_ℓ)", r.m_in * (r.state2.u - r.line.h))
    ]
    _diagram(
        lambda: waterfall_figure(
            "Primer principio: Q = m₁(u₂ − u₁) + m_e(u₂ − h_ℓ)", items, "Q", _EN, system
        ),
        "pl_energy",
        "Lo que ya estaba (m₁) y lo que entra (m_e): sin calor, lo que entra trae su entalpía y "
        "termina con u₂ > u_línea (el trabajo de flujo p·v se vuelve energía interna).",
    )
    _render_steps(charging_steps(r, system))
    _render_export(tr.charging_to_dict(r), "llenado", "pl_dl")


def _discharging_mode(system: UnitSystem) -> None:
    name, idx = _example(tr.DISCHARGING_EXAMPLES, "pv_example")
    ex = tr.DISCHARGING_EXAMPLES[name]
    inp = ex.inputs
    key = f"pv_{idx}"
    st.caption(f"📘 {ex.description}")
    sub = _substance_input(inp.substance, f"{key}_sub", ("fluid", "ideal_gas"))
    V = number_input_si(
        label="Volumen del tanque V",
        kind=_VOL,
        default_si=inp.volume_m3,
        key=f"{key}_V",
        format="%.5g",
        min_value_si=0.0,
    )
    ex_s1 = _state_cached(inp.substance, inp.pair1, inp.a1, inp.b1)
    pair1, a1, b1 = _state_input(
        sub, inp.pair1, _default_state(sub, inp.substance, ex_s1), f"{key}_1", "Estado inicial"
    )
    p2 = number_input_si(
        label="Presión final p₂",
        kind=_P,
        default_si=inp.p2_Pa,
        key=f"{key}_p2",
        format="%.5g",
        min_value_si=0.0,
    )
    modes = list(tr.DISCHARGE_MODES)
    mode = st.radio(
        "¿Cómo se vacía?",
        modes,
        index=modes.index(inp.mode),
        format_func=lambda m: tr.DISCHARGE_MODES[m],
        key=f"{key}_mode",
    )
    p_out: float | None = None
    if st.checkbox(
        "Incluir la válvula (lo que sale termina a la presión de afuera)",
        value=inp.p_out_Pa is not None,
        key=f"{key}_valve",
    ):
        p_out = number_input_si(
            label="Presión de afuera p_sal",
            kind=_P,
            default_si=inp.p_out_Pa if inp.p_out_Pa is not None else 101325.0,
            key=f"{key}_pout",
            format="%.5g",
            min_value_si=0.0,
        )
    T_b: float | None = None
    if mode == "isothermal":
        T_b, T0 = _ambient_inputs(inp.T_b_K, inp.T0_K, key)
    else:
        T0 = number_input_si(
            label="Temperatura del ambiente T₀",
            kind=_T,
            default_si=inp.T0_K,
            key=f"{key}_T0",
            format="%.5g",
        )
    try:
        r = _discharging_cached(
            tr.DischargingInputs(sub, V, pair1, a1, b1, p2, mode, p_out, T_b, T0)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _verdict_box(r.S_gen, r.violation, "S_gen")
    _metric_rows(
        [
            (f"T₂ [{_u(_T, system)}]", _val(r.state2.T, _T, system)),
            (f"Masa que sale [{_u(_M, system)}]", _val(r.m_out, _M, system)),
            (f"∫h·dm [{_u(_EN, system)}]", _val(r.H_out, _EN, system)),
            (f"Calor Q [{_u(_EN, system)}]", _val(r.Q, _EN, system)),
            (f"Q con flujo uniforme [{_u(_EN, system)}]", _val(r.Q_uniform, _EN, system)),
            (f"S_gen [{_u(_EN_S, system)}]", _val(r.S_gen, _EN_S, system)),
        ]
    )
    _notes(r.notes)
    _states_table([("1", r.state1), ("2", r.state2)], system)
    _diagram(lambda: evolution_figure(r.path, system, "El tanque mientras se vacía"), "pv_evol")
    _pv_ts(
        sub,
        [("1", r.state1), ("2", r.state2)],
        [("Lo que queda en el tanque", [s for _, s in r.path], True)],
        system,
        "pv",
    )
    _render_steps(discharging_steps(r, system))
    _render_export(tr.discharging_to_dict(r), "vaciado", "pv_dl")


def _transient_mode(system: UnitSystem) -> None:
    kind = st.radio("¿Qué calculás?", (_T_CHG, _T_DIS), key="pt_kind", horizontal=True)
    if kind == _T_CHG:
        _charging_mode(system)
    else:
        _discharging_mode(system)


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§3** (primer principio), **§10** (entropía: generación y balances, rendimientos "
            "isoentrópicos) y **§13** (líquidos y sólidos incompresibles). Los dispositivos y los "
            "tanques siguen a Çengel y Boles (cap. 4, 5 y 7)."
        )
        st.markdown("**Sistema cerrado** (§3.1), con W el trabajo que hace el sistema:")
        st.latex(r"Q - W = \Delta U + \Delta E_c + \Delta E_p")
        st.markdown("**Trabajo de expansión** cuasiestático (§3.5) y contra p_ext (§3.6):")
        st.latex(r"W = m \int_1^2 p\,dv \qquad W = m\,p_{\mathrm{ext}}\,(v_2 - v_1)")
        st.markdown("**Flujo estacionario** (§3.3), por unidad de masa de una corriente:")
        st.latex(r"q - w = (h_2 - h_1) + \frac{\omega_2^2 - \omega_1^2}{2} + g\,(z_2 - z_1)")
        st.markdown("**Régimen transitorio** (§3.4):")
        st.latex(
            r"\begin{aligned}Q - W &= \Delta U_{VC} + \sum_{\mathrm{sal}} m\,h \\ "
            r"&\quad - \sum_{\mathrm{ent}} m\,h\end{aligned}"
        )
        st.markdown(
            "**Entropía** (§10.2 y §10.3), con T_k la temperatura del medio en la frontera por "
            "donde pasa cada calor (acá T_b):"
        )
        st.latex(r"S_{\mathrm{gen}} = \Delta S_{\mathrm{sist}} + \Delta S_{\mathrm{med}} \ge 0")
        st.latex(r"S_2 - S_1 = \sum_k \frac{Q_k}{T_k} + S_{\mathrm{gen}}")
        st.latex(
            r"\begin{aligned}0 &= \sum_k \frac{\dot{Q}_k}{T_k} + \sum_{\mathrm{ent}} \dot{m}\,s \\ "
            r"&\quad - \sum_{\mathrm{sal}} \dot{m}\,s + \dot{S}_{\mathrm{gen}}\end{aligned}"
        )
        st.latex(r"X_{\mathrm{dest}} = T_0\,S_{\mathrm{gen}}")
        st.markdown(
            "S_gen > 0: irreversible; S_gen = 0: reversible; S_gen < 0: imposible (el balance de "
            "energía puede cerrar igual). Si el calor entra desde una fuente siempre más fría que "
            "el sistema, es imposible aunque la S_gen total salga positiva (Clausius, §9.1)."
        )
        st.markdown("**Rendimientos isoentrópicos** (§10.4), con h₂ₛ = h(p₂, s₁):")
        st.latex(r"\eta_T = \frac{h_1 - h_2}{h_1 - h_{2s}}")
        st.latex(r"\eta_C = \frac{h_{2s} - h_1}{h_2 - h_1}")
        st.latex(r"\eta_N = \frac{\omega_2^2}{\omega_{2s}^2}")
        st.markdown("**Incompresibles** (§13): v y c constantes.")
        st.latex(r"\Delta u = c\,\Delta T \qquad \Delta h = c\,\Delta T + v\,\Delta p")
        st.latex(r"\Delta s = c \ln\frac{T_2}{T_1}")


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Primer principio", page_icon="⚖️", layout="centered")

st.subheader(SUBJECT)
st.title("⚖️ Primer principio")
st.markdown(
    "Los **balances de energía y de entropía** de un **sistema cerrado** (siete procesos y el "
    "equilibrio térmico), de los **dispositivos de flujo estacionario** (tobera, difusor, "
    "válvula, turbina, compresor, bomba, calentador, la cámara de mezcla y el intercambiador) y "
    "del **llenado y el vaciado de un tanque**, con un fluido real, un gas ideal o un "
    "incompresible: el trabajo, el calor, la entropía generada, la exergía destruida y si el "
    "proceso es posible."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Primer principio")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="pp_mode")
if mode == _M_CLOSED:
    _closed_mode(system)
elif mode == _M_FLOW:
    _flow_mode(system)
else:
    _transient_mode(system)
