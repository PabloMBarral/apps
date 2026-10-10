"""Página 17 — Gases ideales.

Fase 9.1. Tres modos (vademecum §4, §5, §6 y §10.5):

- **Un gas entre dos estados**: con dos de p, T y v en cada estado, Δu, Δh y Δs
  con c_p constante a 25 °C, con el c_p a la temperatura media y con c_p
  variable (polinomios NASA), y cuánto te equivocás con los dos primeros.
- **Mezclas de gases**: la composición (por masas, moles o fracciones) con sus
  propiedades, Dalton, Amagat y la entropía de mezcla; y la mezcla adiabática de
  2 a 4 corrientes en un tanque o en una cámara de flujo permanente, con la
  entropía generada y la exergía destruida.
- **Transformaciones**: una politrópica (isócora, isóbara, isoterma, adiabática
  o con n) con los cinco caminos hasta el mismo dato, la compresión en etapas con
  interenfriamiento y el exponente n de dos estados medidos.

El cálculo vive en :mod:`core.gases`; los gráficos, en :mod:`ui.gas_charts`.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.gases import ideal as ig
from core.gases import mixture as mx
from core.gases import polytropic as pt
from core.gases.ideal_procedure import (
    exponent_steps,
    mixing_steps,
    mixture_steps,
    process_steps,
    staged_steps,
    state_change_steps,
)
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.gas_charts import (
    composition_figure,
    cp_figure,
    far_paths,
    mixing_entropy_figure,
    process_label,
    pv_figure,
    staged_figure,
    ts_figure,
    work_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.26.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_GAS = "Un gas entre dos estados"
_M_MIX = "Mezclas de gases"
_M_PROC = "Transformaciones"
_MODES = (_M_GAS, _M_MIX, _M_PROC)

_MIX_COMP = "Composición y propiedades"
_MIX_ADIA = "Mezcla adiabática (tanque o cámara)"
_MIX_KINDS = (_MIX_COMP, _MIX_ADIA)

_PROC_ONE = "Un proceso y los cinco caminos"
_PROC_STAGED = "Compresión en etapas"
_PROC_N = "El exponente n de dos estados"
_PROC_KINDS = (_PROC_ONE, _PROC_STAGED, _PROC_N)

_PAIRS: dict[ig.PairKind, str] = {"pT": "p y T", "pv": "p y v", "Tv": "T y v"}
_BASES: dict[mx.CompositionBasis, str] = {
    "mass": "Masas",
    "moles": "Moles",
    "mass_fraction": "Fracciones másicas x",
    "mole_fraction": "Fracciones molares y",
}
_MODELS: dict[ig.GasModel, str] = {
    "constant": "c_p constante (25 °C)",
    "variable": "c_p variable (NASA)",
}
_SYSTEMS: dict[pt.SystemKind, str] = {
    "closed": "Cerrado (una masa)",
    "open": "Abierto (un caudal)",
}
_MIXING: dict[mx.MixingKind, str] = {
    "tank": "Tanque rígido con tabiques (U = cte)",
    "flow": "Cámara de mezcla de flujo permanente (H = cte)",
}
_EXP_DATA = {"T": "p y T", "v": "p y v"}

_P: QuantityKind = "pressure"
_T: QuantityKind = "temperature"
_V: QuantityKind = "specific_volume"
_E: QuantityKind = "specific_enthalpy"
_S: QuantityKind = "specific_entropy"
_C: QuantityKind = "specific_heat"


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


def _signed_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{'+' if x >= 0 else '−'}{abs(x) * 100:.2f} %".replace(".", ",")


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


def _example(examples: dict[str, Any], key: str, help_text: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox("Ejemplo precargado", names, key=key, help=help_text)
    return name, names.index(name)


def _gas_name(key: str) -> str:
    g = ig.gas(key)
    return g.name if g.formula == "—" else f"{g.name} ({g.formula})"


def _gas_select(default: str, key: str, label: str = "Gas") -> str:
    options = list(ig.IDEAL_GASES)
    return st.selectbox(
        label,
        options,
        index=options.index(default) if default in options else 0,
        format_func=_gas_name,
        key=key,
        help="Los nueve de la tabla del vademecum (§4.7) y algunos más de los polinomios NASA "
        "(McBride et al., 2002).",
    )


def _diagram(build: Any, key: str, caption: str = "") -> None:
    """Un gráfico que falla no tumba la página."""
    try:
        st.plotly_chart(build(), width="stretch", key=key)
        if caption:
            st.caption(caption)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"No se pudo dibujar el gráfico: {exc}")


# ---------------------------------------------------------------------
# Un gas entre dos estados
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _state_cached(inputs: ig.StateChangeInputs) -> ig.StateChange:
    return ig.solve_state_change(inputs)


def _pair_inputs(title: str, default: ig.StatePair, gas_key: str, key: str) -> ig.StatePair:
    """Un estado con dos de p, T y v (el par se elige)."""
    st.markdown(f"**{title}**")
    pairs = list(_PAIRS)
    pair: ig.PairKind = st.radio(
        "Con",
        pairs,
        index=pairs.index(default.pair),
        format_func=lambda k: _PAIRS[k],
        key=f"{key}_pair",
        horizontal=True,
    )
    # los valores por defecto de cualquier par salen del estado del ejemplo
    base = ig.state_from_pair(ig.gas(gas_key), default)
    values = {"p": base.p_Pa, "T": base.T_K, "v": base.v_m3_per_kg}
    widgets = {
        "p": ("Presión p (absoluta)", _P),
        "T": ("Temperatura T", _T),
        "v": ("Volumen específico v", _V),
    }
    out: list[float] = []
    cols = st.columns(2)
    for col, name in zip(cols, (pair[0], pair[1]), strict=True):
        label, kind = widgets[name]
        with col:
            out.append(
                number_input_si(
                    label=label,
                    kind=kind,
                    default_si=values[name],
                    key=f"{key}_{pair}_{name}",
                    format="%.5g",
                    min_value_si=0.0 if name != "T" else None,
                )
            )
    return ig.StatePair(pair, out[0], out[1])


def _gas_mode(system: UnitSystem) -> None:
    name, idx = _example(
        ig.STATE_EXAMPLES,
        "gi_example",
        "Podés cambiar el gas y cualquier dato: el resultado se actualiza solo.",
    )
    ex = ig.STATE_EXAMPLES[name]
    inp = ex.inputs
    key = f"gi_{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    gas_key = _gas_select(inp.gas_key, f"{key}_gas")
    s1 = _pair_inputs("Estado 1", inp.state1, inp.gas_key, f"{key}_1")
    s2 = _pair_inputs("Estado 2", inp.state2, inp.gas_key, f"{key}_2")
    try:
        result = _state_cached(ig.StateChangeInputs(gas_key, s1, s2))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_state_change(result, system)


def _render_state_change(r: ig.StateChange, system: UnitSystem) -> None:
    g = r.gas
    var = r.variable
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"Δh, c_p variable [{_u(_E, system)}]", _val(var.dh, _E, system)),
            (f"Δu, c_p variable [{_u(_E, system)}]", _val(var.du, _E, system)),
            (f"Δs, c_p variable [{_u(_S, system)}]", _val(var.ds, _S, system)),
            ("Error de Δh con c_p a 25 °C", _signed_pct(r.error("constant", "dh"))),
        ]
    )
    for note in r.notes:
        st.info(note)

    st.markdown("#### Los dos estados")
    rows = []
    for symbol, kind, a, b in (
        ("p", _P, r.state1.p_Pa, r.state2.p_Pa),
        ("T", _T, r.state1.T_K, r.state2.T_K),
        ("v", _V, r.state1.v_m3_per_kg, r.state2.v_m3_per_kg),
        ("c_p", _C, r.cp1, r.cp2),
        ("c_v", _C, r.cp1 - g.R, r.cp2 - g.R),
        ("h (desde 25 °C)", _E, g.h(r.state1.T_K), g.h(r.state2.T_K)),
        ("u (desde 25 °C)", _E, g.u(r.state1.T_K), g.u(r.state2.T_K)),
        ("s° (desde 25 °C)", _S, r.s01, r.s02),
        (
            "s (25 °C y 1 bar)",
            _S,
            ig.model_s(g, r.state1.T_K, r.state1.p_Pa, "variable"),
            ig.model_s(g, r.state2.T_K, r.state2.p_Pa, "variable"),
        ),
    ):
        rows.append(
            {
                "Propiedad": symbol,
                "Unidad": _u(kind, system),
                "Estado 1": _val(a, kind, system),
                "Estado 2": _val(b, kind, system),
            }
        )
    for symbol, a, b in (
        ("k = c_p/c_v", r.cp1 / (r.cp1 - g.R), r.cp2 / (r.cp2 - g.R)),
        ("p_r", r.pr1, r.pr2),
        ("Z (CoolProp)", r.Z1, r.Z2),
    ):
        rows.append({"Propiedad": symbol, "Unidad": "—", "Estado 1": _num(a), "Estado 2": _num(b)})
    rows.append(
        {
            "Propiedad": "v_r",
            "Unidad": _u("absolute_temperature", system),
            "Estado 1": _val(r.vr1, "absolute_temperature", system),
            "Estado 2": _val(r.vr2, "absolute_temperature", system),
        }
    )
    _table(pd.DataFrame(rows))
    st.caption(
        "h, u y s° con c_p variable, desde 25 °C (las tablas de los libros usan otras "
        "referencias: lo que coincide son las diferencias). Z = p·v/(R·T) de CoolProp, para ver "
        "si el gas ideal vale: 1 es gas ideal."
    )

    st.markdown("#### ¿Cuánto te equivocás con c_p constante?")
    labels = {
        "constant": "c_p constante a 25 °C",
        "mean": "c_p a la temperatura media",
        "variable": "c_p variable (NASA)",
    }
    _table(
        pd.DataFrame(
            [
                {
                    "Modelo": labels[c.model],
                    f"c_p [{_u(_C, system)}]": _val(c.cp, _C, system),
                    f"Δh [{_u(_E, system)}]": _val(c.dh, _E, system),
                    f"Δu [{_u(_E, system)}]": _val(c.du, _E, system),
                    f"Δs [{_u(_S, system)}]": _val(c.ds, _S, system),
                    "Error en Δh": "—"
                    if c.model == "variable"
                    else _signed_pct(r.error(c.model, "dh")),
                    "Error en Δs": "—"
                    if c.model == "variable"
                    else _signed_pct(r.error(c.model, "ds")),
                }
                for c in r.changes
            ]
        )
    )
    st.caption(
        "El c_p de c_p variable es el medio del proceso, Δh/ΔT. El error es contra c_p variable, "
        "que integra el polinomio."
    )
    _diagram(
        lambda: cp_figure(g, r.state1.T_K, r.state2.T_K, system),
        "gi_cp",
        "c_p crece con la temperatura (se excitan las vibraciones de la molécula); en el helio, "
        "monoatómico, es constante. El c_p a la temperatura media representa bien todo el tramo "
        "si la curva es casi recta.",
    )
    _render_steps(state_change_steps(r, system))
    _render_export(ig.state_change_to_dict(r), "gas_ideal", "gi_dl")


# ---------------------------------------------------------------------
# Mezclas
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _mixture_cached(inputs: mx.MixtureInputs) -> mx.MixtureResult:
    return mx.solve_mixture(inputs)


@st.cache_data(show_spinner=False)
def _mixing_cached(inputs: mx.MixingInputs) -> mx.MixingResult:
    return mx.solve_mixing(inputs)


def _next_gas(used: list[str]) -> str:
    return next(k for k in ig.IDEAL_GASES if k not in used)


def _amount_default(
    basis: mx.CompositionBasis,
    j: int,
    example: mx.MixtureInputs,
    last: mx.MixtureResult | None,
) -> float:
    """Cantidad por defecto: la del ejemplo, la del último resultado pasado a la base, o una."""
    if example.basis == basis and j < len(example.components):
        return example.components[j].amount
    if last is not None and j < len(last.components):
        c = last.components[j]
        if basis == "mass":
            return c.m_kg if c.m_kg is not None else c.x
        if basis == "moles":
            return c.n_mol if c.n_mol is not None else 1000.0 * c.y
        return c.x if basis == "mass_fraction" else c.y
    return {"mass": 1.0, "moles": 1000.0}.get(basis, 0.0)


def _amount_input(basis: mx.CompositionBasis, default: float, key: str, label: str) -> float:
    if basis == "mass":
        return number_input_si(
            label=label, kind="mass", default_si=default, key=key, format="%.5g", min_value_si=0.0
        )
    if basis == "moles":
        return number_input_si(
            label=label, kind="amount", default_si=default, key=key, format="%.5g", min_value_si=0.0
        )
    return float(
        st.number_input(
            label,
            min_value=0.0,
            max_value=1.0,
            value=float(default),
            format="%.4f",
            key=key,
        )
    )


def _composition_mode(system: UnitSystem) -> None:
    name, idx = _example(
        mx.COMPOSITION_EXAMPLES,
        "gm_c_example",
        "Podés cambiar los gases, las cantidades, la forma de dar la composición, T y p.",
    )
    ex = mx.COMPOSITION_EXAMPLES[name]
    inp = ex.inputs
    key = f"gm_c{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    bases = list(_BASES)
    basis: mx.CompositionBasis = st.radio(
        "¿Cómo das la composición?",
        bases,
        index=bases.index(inp.basis),
        format_func=lambda b: _BASES[b],
        key=f"{key}_basis",
        horizontal=True,
    )
    count = int(
        st.number_input(
            "Cantidad de gases",
            min_value=2,
            max_value=6,
            value=len(inp.components),
            key=f"{key}_n",
        )
    )
    last: mx.MixtureResult | None = st.session_state.get(f"{key}_last")
    labels = {
        "mass": "Masa",
        "moles": "Moles",
        "mass_fraction": "Fracción másica x",
        "mole_fraction": "Fracción molar y",
    }
    comps: list[mx.Component] = []
    used: list[str] = []
    for j in range(count):
        default_gas = inp.components[j].gas_key if j < len(inp.components) else _next_gas(used)
        left, right = st.columns(2)
        with left:
            gas_key = _gas_select(default_gas, f"{key}_{j}_gas", label=f"Gas {j + 1}")
        with right:
            amount = _amount_input(
                basis,
                _amount_default(basis, j, inp, last),
                f"{key}_{basis}_{j}",
                labels[basis],
            )
        used.append(gas_key)
        comps.append(mx.Component(gas_key, amount))
    left, right = st.columns(2)
    with left:
        T = number_input_si(
            label="Temperatura de la mezcla",
            kind=_T,
            default_si=inp.T_K,
            key=f"{key}_T",
            format="%.5g",
        )
    with right:
        p = number_input_si(
            label="Presión de la mezcla",
            kind=_P,
            default_si=inp.p_Pa,
            key=f"{key}_p",
            format="%.5g",
            min_value_si=0.0,
        )
    try:
        result = _mixture_cached(mx.MixtureInputs(tuple(comps), basis, T, p))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.session_state[f"{key}_last"] = result
    _render_mixture(result, system)


def _render_mixture(r: mx.MixtureResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    M: QuantityKind = "molar_mass"
    _metric_rows(
        [
            (f"M_M [{_u(M, system)}]", _val(r.M, M, system)),
            (f"R_M [{_u(_S, system)}]", _val(r.R, _S, system)),
            (f"c_p,M [{_u(_C, system)}]", _val(r.cp, _C, system)),
            ("k_M", _num(r.k, 4)),
        ]
    )
    for note in r.notes:
        st.info(note)
    rows = []
    for c in r.components:
        row = {
            "Gas": _gas_name(c.gas.key),
            "x": _num(c.x, 4),
            "y": _num(c.y, 4),
            f"M [{_u(M, system)}]": _val(c.gas.M, M, system),
            f"p_i [{_u(_P, system)}]": _val(c.p_i, _P, system),
        }
        if r.V_m3 is not None:
            row[f"V_i [{_u('volume', system)}]"] = _val(c.V_i, "volume", system)
        row[f"c_p [{_u(_C, system)}]"] = _val(c.cp, _C, system)
        row[f"s_i [{_u(_S, system)}]"] = _val(c.s_i if c.y > 0.0 else None, _S, system)
        rows.append(row)
    _table(pd.DataFrame(rows))
    extra = [
        f"c_v,M = {_val(r.cv, _C, system)} {_u(_C, system)}",
        f"v = {_val(r.v_m3_per_kg, _V, system)} {_u(_V, system)}",
        f"s_M = {_val(r.s, _S, system)} {_u(_S, system)}",
        f"de mezclar: {_val(r.s_mixing, _S, system)} {_u(_S, system)}",
    ]
    if r.m_kg is not None and r.V_m3 is not None:
        extra.append(f"V = {_val(r.V_m3, 'volume', system)} {_u('volume', system)}")
    st.caption(
        " · ".join(extra) + ". s con s = 0 a 25 °C y 1 bar; cada gas a su presión parcial p_i."
    )
    _diagram(
        lambda: composition_figure(r),
        "gm_comp",
        "La fracción másica pesa más a los gases de masa molar grande (CO₂, Ar) y la molar a los "
        "livianos (H₂, CH₄).",
    )
    _render_steps(mixture_steps(r, system))
    _render_export(mx.mixture_to_dict(r), "mezcla", "gm_c_dl")


def _mixing_mode(system: UnitSystem) -> None:
    name, idx = _example(
        mx.MIXING_EXAMPLES,
        "gm_a_example",
        "Podés cambiar dónde se mezclan, los gases, sus cantidades, temperaturas y presiones.",
    )
    ex = mx.MIXING_EXAMPLES[name]
    inp = ex.inputs
    key = f"gm_a{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    kinds = list(_MIXING)
    kind: mx.MixingKind = st.radio(
        "Dónde se mezclan",
        kinds,
        index=kinds.index(inp.kind),
        format_func=lambda k: _MIXING[k],
        key=f"{key}_kind",
    )
    models = list(_MODELS)
    model: ig.GasModel = st.radio(
        "Modelo",
        models,
        index=models.index(inp.model),
        format_func=lambda m: _MODELS[m],
        key=f"{key}_model",
        horizontal=True,
    )
    tank = kind == "tank"
    count = int(
        st.number_input(
            "Cantidad de compartimientos" if tank else "Cantidad de corrientes",
            min_value=2,
            max_value=4,
            value=len(inp.streams),
            key=f"{key}_n",
        )
    )
    streams: list[mx.MixingStream] = []
    used: list[str] = []
    for j in range(count):
        st.markdown(f"**{'Compartimiento' if tank else 'Corriente'} {j + 1}**")
        base = (
            inp.streams[j]
            if j < len(inp.streams)
            else mx.MixingStream(_next_gas(used), 1.0, 300.0, 100e3)
        )
        left, right = st.columns(2)
        with left:
            gas_key = _gas_select(base.gas_key, f"{key}_{j}_gas")
            T = number_input_si(
                label="Temperatura", kind=_T, default_si=base.T_K, key=f"{key}_{j}_T", format="%.5g"
            )
        with right:
            amount = number_input_si(
                label="Masa" if tank else "Caudal",
                kind="mass" if tank else "mass_flow",
                default_si=base.amount,
                key=f"{key}_{j}_m_{kind}",
                format="%.5g",
                min_value_si=0.0,
            )
            p = number_input_si(
                label="Presión (absoluta)",
                kind=_P,
                default_si=base.p_Pa,
                key=f"{key}_{j}_p",
                format="%.5g",
                min_value_si=0.0,
            )
        used.append(gas_key)
        streams.append(mx.MixingStream(gas_key, amount, T, p))
    p_out = None
    left, right = st.columns(2)
    if not tank:
        with left:
            p_out = number_input_si(
                label="Presión de salida",
                kind=_P,
                default_si=inp.p_out_Pa or min(s.p_Pa for s in streams),
                key=f"{key}_pout",
                format="%.5g",
                min_value_si=0.0,
                help="No puede ser mayor que la de la entrada más baja: una cámara de mezcla no "
                "comprime.",
            )
    with right if not tank else left:
        T0 = number_input_si(
            label="T₀ del ambiente (para la exergía)",
            kind=_T,
            default_si=inp.T0_K,
            key=f"{key}_T0",
            format="%.5g",
            help="El estado muerto del vademecum (§11.1): la exergía destruida es T₀·S_gen.",
        )
    try:
        result = _mixing_cached(mx.MixingInputs(kind, tuple(streams), model, T0, p_out))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_mixing(result, system)


def _render_mixing(r: mx.MixingResult, system: UnitSystem) -> None:
    tank = r.inputs.kind == "tank"
    s_kind: QuantityKind = "entropy" if tank else "entropy_flow"
    x_kind: QuantityKind = "energy" if tank else "power"
    st.markdown("### Resultado")
    _metric_rows(
        [
            (f"T final [{_u(_T, system)}]", _val(r.T_K, _T, system)),
            (f"p final [{_u(_P, system)}]", _val(r.p_Pa, _P, system)),
            (f"{'S' if tank else 'Ṡ'}_gen [{_u(s_kind, system)}]", _val(r.S_gen, s_kind, system)),
            (f"{'X' if tank else 'Ẋ'}_dest [{_u(x_kind, system)}]", _val(r.X_dest, x_kind, system)),
        ]
    )
    for note in r.notes:
        st.info(note)
    amount: QuantityKind = "mass" if tank else "mass_flow"
    _table(
        pd.DataFrame(
            [
                {
                    "#": j,
                    "Gas": _gas_name(st_.gas.key),
                    f"{'m' if tank else 'ṁ'} [{_u(amount, system)}]": _val(
                        st_.amount, amount, system
                    ),
                    f"T [{_u(_T, system)}]": _val(st_.T_K, _T, system),
                    f"p [{_u(_P, system)}]": _val(st_.p_Pa, _P, system),
                    "y final": _num(st_.y, 4),
                    f"Δs_TP [{_u(_S, system)}]": _val(st_.ds_TP, _S, system),
                    f"Δs_mez [{_u(_S, system)}]": _val(st_.ds_mix, _S, system),
                }
                for j, st_ in enumerate(r.streams, 1)
            ]
        )
    )
    if r.V_m3 is not None:
        st.caption(f"Volumen del tanque: {_val(r.V_m3, 'volume', system)} {_u('volume', system)}.")
    _diagram(
        lambda: mixing_entropy_figure(r, system),
        "gm_sgen",
        "Igualar temperaturas y presiones: la corriente caliente pierde entropía y la fría gana "
        "más (la suma es positiva). Mezclar gases distintos suma siempre (−R·ln y > 0); dos "
        "corrientes del mismo gas no suman nada.",
    )
    _render_steps(mixing_steps(r, system))
    _render_export(mx.mixing_to_dict(r), "mezcla_adiabatica", "gm_a_dl")


def _mixtures_mode(system: UnitSystem) -> None:
    kind = st.radio("¿Qué mezcla?", _MIX_KINDS, key="gm_kind", horizontal=True)
    if kind == _MIX_COMP:
        _composition_mode(system)
    else:
        _mixing_mode(system)


# ---------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _process_cached(inputs: pt.PolytropicInputs) -> pt.ProcessResult:
    return pt.solve_process(inputs)


@st.cache_data(show_spinner=False)
def _comparison_cached(inputs: pt.PolytropicInputs) -> tuple[pt.ComparisonRow, ...]:
    return pt.process_comparison(inputs)


@st.cache_data(show_spinner=False)
def _staged_cached(inputs: pt.StagedInputs) -> pt.StagedResult:
    return pt.staged_compression(inputs)


def _final_defaults(ex: pt.PolytropicInputs) -> dict[pt.FinalSpec, float]:
    """El dato final del ejemplo y, para los otros, valores que tengan sentido."""
    out: dict[pt.FinalSpec, float] = {"p2": 5.0 * ex.p1_Pa, "ratio": 5.0, "T2": 1.5 * ex.T1_K}
    try:
        r = _process_cached(ex)
    except ValueError:
        return out
    if abs(r.state2.p_Pa / r.state1.p_Pa - 1.0) > 1e-6:
        out["p2"] = r.state2.p_Pa
    ratio = r.state1.v_m3_per_kg / r.state2.v_m3_per_kg
    if abs(ratio - 1.0) > 1e-6:
        out["ratio"] = ratio
    if abs(r.state2.T_K - r.state1.T_K) > 1e-6:
        out["T2"] = r.state2.T_K
    out[ex.final] = ex.final_value
    return out


def _process_mode(system: UnitSystem) -> None:
    name, idx = _example(
        pt.PROCESS_EXAMPLES,
        "gp_p_example",
        "Podés cambiar el gas, el proceso, el dato final y el modelo: el resultado se actualiza "
        "solo.",
    )
    ex = pt.PROCESS_EXAMPLES[name]
    inp = ex.inputs
    key = f"gp_p{idx}"
    if ex.note:
        st.caption(f"📘 {ex.note}")
    gas_key = _gas_select(inp.gas_key, f"{key}_gas")
    left, right = st.columns(2)
    models = list(_MODELS)
    with left:
        model: ig.GasModel = st.radio(
            "Modelo",
            models,
            index=models.index(inp.model),
            format_func=lambda m: _MODELS[m],
            key=f"{key}_model",
        )
    systems = list(_SYSTEMS)
    with right:
        sys_kind: pt.SystemKind = st.radio(
            "Sistema",
            systems,
            index=systems.index(inp.system),
            format_func=lambda s: _SYSTEMS[s],
            key=f"{key}_sys",
        )
    left, right = st.columns(2)
    with left:
        p1 = number_input_si(
            label="p₁ (absoluta)",
            kind=_P,
            default_si=inp.p1_Pa,
            key=f"{key}_p1",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        T1 = number_input_si(
            label="T₁", kind=_T, default_si=inp.T1_K, key=f"{key}_T1", format="%.5g"
        )
    kinds = list(pt.PROCESS_KINDS)
    process: pt.ProcessKind = st.radio(
        "Proceso",
        kinds,
        index=kinds.index(inp.process),
        format_func=lambda k: pt.PROCESS_KINDS[k],
        key=f"{key}_proc",
    )
    finals = list(pt.allowed_finals(process))
    final: pt.FinalSpec = st.radio(
        "El estado final, con",
        finals,
        index=finals.index(inp.final) if inp.final in finals else 0,
        format_func=lambda f: pt.FINAL_SPECS[f],
        key=f"{key}_final_{process}",
        horizontal=True,
    )
    defaults = _final_defaults(inp)
    left, right = st.columns(2)
    with left:
        if final == "p2":
            value = number_input_si(
                label="p₂ (absoluta)",
                kind=_P,
                default_si=defaults["p2"],
                key=f"{key}_p2",
                format="%.5g",
                min_value_si=0.0,
            )
        elif final == "ratio":
            value = float(
                st.number_input(
                    "v₁/v₂ (mayor que 1 comprime)",
                    min_value=0.0,
                    value=float(defaults["ratio"]),
                    format="%.4g",
                    key=f"{key}_ratio",
                )
            )
        else:
            value = number_input_si(
                label="T₂", kind=_T, default_si=defaults["T2"], key=f"{key}_T2", format="%.5g"
            )
    n_exp = inp.n
    with right:
        if process == "polytropic":
            n_exp = float(
                st.number_input(
                    "Exponente n",
                    min_value=-20.0,
                    max_value=20.0,
                    value=float(inp.n),
                    step=0.05,
                    format="%.4g",
                    key=f"{key}_n",
                    help="Un compresor real va de n = 1 (isoterma) a algo más que k (adiabática "
                    "con fricción); con refrigeración, 1 < n < k.",
                )
            )
    amount = number_input_si(
        label="Masa" if sys_kind == "closed" else "Caudal",
        kind="mass" if sys_kind == "closed" else "mass_flow",
        default_si=inp.amount,
        key=f"{key}_amount_{sys_kind}",
        format="%.5g",
        min_value_si=0.0,
    )
    inputs = pt.PolytropicInputs(
        gas_key, p1, T1, process, final, value, n=n_exp, model=model, system=sys_kind, amount=amount
    )
    try:
        result = _process_cached(inputs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_process(result, system)
    _render_paths(inputs, system)
    _render_steps(process_steps(result, system))
    _render_export(pt.process_to_dict(result), "transformacion", "gp_p_dl")


def _render_process(r: pt.ProcessResult, system: UnitSystem) -> None:
    closed = r.inputs.system == "closed"
    total: QuantityKind = "energy" if closed else "power"
    st.markdown("### Resultado")
    work = r.w if closed else r.w_f
    _metric_rows(
        [
            (f"T₂ [{_u(_T, system)}]", _val(r.state2.T_K, _T, system)),
            (f"p₂ [{_u(_P, system)}]", _val(r.state2.p_Pa, _P, system)),
            (f"{'w' if closed else 'w_f'} [{_u(_E, system)}]", _val(work, _E, system)),
            (f"q [{_u(_E, system)}]", _val(r.q, _E, system)),
            (f"{'W' if closed else 'Ẇ_f'} [{_u(total, system)}]", _val(r.W_total, total, system)),
            (f"{'Q' if closed else 'Q̇'} [{_u(total, system)}]", _val(r.Q_total, total, system)),
        ]
    )
    for note in r.notes:
        st.info(note)
    st.markdown("#### Los estados")
    _table(
        pd.DataFrame(
            [
                {
                    "Estado": label,
                    f"p [{_u(_P, system)}]": _val(s.p_Pa, _P, system),
                    f"T [{_u(_T, system)}]": _val(s.T_K, _T, system),
                    f"v [{_u(_V, system)}]": _val(s.v_m3_per_kg, _V, system),
                }
                for label, s in (("1", r.state1), ("2", r.state2))
            ]
        )
    )
    st.markdown("#### Energía y entropía, por kg")
    n_txt = "∞" if math.isinf(r.n) else _num(r.n, 4)
    rows = [
        ("Δu", r.du, _E),
        ("Δh", r.dh, _E),
        ("Δs", r.ds, _S),
        ("w = ∫p dv", r.w, _E),
        ("w_f = −∫v dp", r.w_f, _E),
        ("q", r.q, _E),
    ]
    _table(
        pd.DataFrame(
            [
                {"Magnitud": name, "Unidad": _u(kind, system), "Valor": _val(value, kind, system)}
                for name, value, kind in rows
            ]
        )
    )
    st.caption(
        f"n = {n_txt} y c = {_val(r.c, _C, system)} {_u(_C, system)} (el calor por kelvin). "
        f"Primer principio en el sistema cerrado: q − w = {_val(r.q - r.w, _E, system)} y "
        f"Δu = {_val(r.du, _E, system)} {_u(_E, system)}. Trabajo positivo si lo hace el gas."
    )


def _render_paths(inputs: pt.PolytropicInputs, system: UnitSystem) -> None:
    st.markdown("#### Los cinco caminos")
    st.caption(
        "Los cinco procesos desde el mismo estado inicial hasta el mismo dato final (la tabla "
        "resumen del vademecum §6.4.5, con números). La politrópica usa tu n o, si el proceso es "
        "otro, n = 1,3."
    )
    try:
        rows = _comparison_cached(inputs)
    except ValueError as exc:
        st.warning(str(exc))
        return
    closed = inputs.system == "closed"
    table = []
    for row in rows:
        name = ("▸ " if row.is_data else "") + process_label(row)
        r = row.result
        if r is None:
            table.append({"Proceso": name, f"T₂ [{_u(_T, system)}]": row.reason})
            continue
        table.append(
            {
                "Proceso": name,
                f"T₂ [{_u(_T, system)}]": _val(r.state2.T_K, _T, system),
                f"p₂ [{_u(_P, system)}]": _val(r.state2.p_Pa, _P, system),
                f"v₂ [{_u(_V, system)}]": _val(r.state2.v_m3_per_kg, _V, system),
                f"w [{_u(_E, system)}]": _val(r.w, _E, system),
                f"w_f [{_u(_E, system)}]": _val(r.w_f, _E, system),
                f"q [{_u(_E, system)}]": _val(r.q, _E, system),
                f"Δs [{_u(_S, system)}]": _val(r.ds, _S, system),
            }
        )
    _table(pd.DataFrame(table).fillna(""))
    _diagram(
        lambda: pv_figure(rows, system),
        "gp_pv",
        "En el p–v, el área debajo de cada camino es w = ∫p dv y el área a su izquierda, "
        "w_f = −∫v dp. Con el dato resaltado y sus estados 1 y 2.",
    )
    far = [row for row in rows if row.process in far_paths(rows) and row.result is not None]
    if far:
        parts = [
            f"{process_label(row).lower()} (T₂ = "
            f"{_val(row.result.state2.T_K, _T, system)} {_u(_T, system)})"  # type: ignore[union-attr]
            for row in far
        ]
        st.caption(
            f"{'La ' + parts[0] if len(far) == 1 else 'La ' + ' y la '.join(parts)} se "
            f"{'va' if len(far) == 1 else 'van'} de escala: en el T–s "
            f"{'queda oculta' if len(far) == 1 else 'quedan ocultas'} (tocá su nombre en la "
            "leyenda para verla) y no va en el gráfico del trabajo. Sus números están en la tabla."
        )
    _diagram(
        lambda: ts_figure(rows, system),
        "gp_ts",
        "En el T–s el área debajo de un camino reversible es el calor q = ∫T ds; la adiabática "
        "reversible es vertical (Δs = 0).",
    )
    _diagram(
        lambda: work_figure(rows, system, closed=closed),
        "gp_work",
        "El trabajo y el calor de cada camino, por kg, positivos si los hace o los recibe el gas "
        "(la convención del vademecum). Comprimiendo hasta la misma presión, la isoterma es la que "
        "menos trabajo pide y la adiabática la que más.",
    )


def _staged_mode(system: UnitSystem) -> None:
    base = pt.PROCESS_EXAMPLES[next(iter(pt.PROCESS_EXAMPLES))].inputs
    st.caption(
        "📘 El compresor de Çengel §7-12: aire de 100 kPa y 300 K a 900 kPa con n = 1,3 "
        "(215,3 kJ/kg en dos etapas)."
    )
    key = "gp_s"
    gas_key = _gas_select(base.gas_key, f"{key}_gas")
    left, right = st.columns(2)
    with left:
        p1 = number_input_si(
            label="p₁ (absoluta)",
            kind=_P,
            default_si=base.p1_Pa,
            key=f"{key}_p1",
            format="%.5g",
            min_value_si=0.0,
        )
        p2 = number_input_si(
            label="p₂ (absoluta)",
            kind=_P,
            default_si=base.final_value,
            key=f"{key}_p2",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        T1 = number_input_si(
            label="T₁", kind=_T, default_si=base.T1_K, key=f"{key}_T1", format="%.5g"
        )
        n_exp = float(
            st.number_input(
                "Exponente n",
                min_value=1.0,
                max_value=3.0,
                value=1.3,
                step=0.05,
                format="%.4g",
                key=f"{key}_n",
            )
        )
    stages = int(
        st.number_input("Cantidad de etapas N", min_value=1, max_value=4, value=2, key=f"{key}_N")
    )
    st.caption(
        "Interenfriamiento perfecto: el gas entra a cada etapa a T₁, sin pérdida de carga en los "
        "interenfriadores y con el mismo n en todas las etapas (vademecum §6.3.1)."
    )
    try:
        r = _staged_cached(pt.StagedInputs(gas_key, p1, T1, p2, n_exp, stages))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _metric_rows(
        [
            (
                f"w_f en {stages} etapa{'s' if stages > 1 else ''} [{_u(_E, system)}]",
                _val(r.w_f, _E, system),
            ),
            (f"w_f en una etapa [{_u(_E, system)}]", _val(r.w_f_single, _E, system)),
            ("Ahorro de trabajo", _pct(r.saving)),
            (f"T a la salida de cada etapa [{_u(_T, system)}]", _val(r.T_out_K, _T, system)),
        ]
    )
    rows = [
        {
            "Etapa": j,
            f"p entrada [{_u(_P, system)}]": _val(r.pressures_Pa[j - 1], _P, system),
            f"p salida [{_u(_P, system)}]": _val(r.pressures_Pa[j], _P, system),
            f"T salida [{_u(_T, system)}]": _val(r.T_out_K, _T, system),
        }
        for j in range(1, stages + 1)
    ]
    _table(pd.DataFrame(rows))
    if stages > 1:
        st.caption(
            f"Calor de los interenfriadores: {_val(r.q_intercoolers, _E, system)} "
            f"{_u(_E, system)} (sale del gas). En una etapa el gas saldría a "
            f"{_val(r.T_out_single_K, _T, system)} {_u(_T, system)}."
        )
    _diagram(
        lambda: staged_figure(r, system),
        "gp_staged",
        "El trabajo es el área a la izquierda de cada camino: las etapas se acercan a la isoterma "
        "(el mínimo, con infinitas etapas).",
    )
    _render_steps(staged_steps(r, system))
    _render_export(pt.staged_to_dict(r), "compresion_en_etapas", "gp_s_dl")


def _exponent_mode(system: UnitSystem) -> None:
    E = pt.EXPONENT_EXAMPLE
    st.caption(
        "📘 El ensayo de un compresor de aire: de 100 kPa y 20 °C a 600 kPa y 200 °C. ¿Qué n tuvo "
        "la compresión?"
    )
    key = "gp_n"
    gas_key = _gas_select("air", f"{key}_gas")
    data = st.radio(
        "Medidas",
        list(_EXP_DATA),
        format_func=lambda d: _EXP_DATA[d],
        key=f"{key}_data",
        horizontal=True,
    )
    g = ig.gas(gas_key)
    left, right = st.columns(2)
    with left:
        p1 = number_input_si(
            label="p₁ (absoluta)",
            kind=_P,
            default_si=E["p1_Pa"],
            key=f"{key}_p1",
            format="%.5g",
            min_value_si=0.0,
        )
    with right:
        p2 = number_input_si(
            label="p₂ (absoluta)",
            kind=_P,
            default_si=E["p2_Pa"],
            key=f"{key}_p2",
            format="%.5g",
            min_value_si=0.0,
        )
    left, right = st.columns(2)
    kwargs: dict[str, float] = {}
    if data == "T":
        with left:
            kwargs["T1_K"] = number_input_si(
                label="T₁", kind=_T, default_si=E["T1_K"], key=f"{key}_T1", format="%.5g"
            )
        with right:
            kwargs["T2_K"] = number_input_si(
                label="T₂", kind=_T, default_si=E["T2_K"], key=f"{key}_T2", format="%.5g"
            )
    else:
        with left:
            kwargs["v1"] = number_input_si(
                label="v₁",
                kind=_V,
                default_si=g.R * E["T1_K"] / E["p1_Pa"],
                key=f"{key}_v1",
                format="%.5g",
                min_value_si=0.0,
            )
        with right:
            kwargs["v2"] = number_input_si(
                label="v₂",
                kind=_V,
                default_si=g.R * E["T2_K"] / E["p2_Pa"],
                key=f"{key}_v2",
                format="%.5g",
                min_value_si=0.0,
            )
    try:
        e = pt.exponent_from_states(gas_key, p1, p2, **kwargs)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    _metric_rows([("Exponente politrópico n", _num(e.n, 4)), ("k del gas (25 °C)", _num(e.k, 4))])
    st.info(e.text)
    _render_steps(exponent_steps(e, system, p1, p2, **kwargs))
    _render_export(pt.exponent_to_dict(e, gas_key, p1, p2, **kwargs), "exponente", "gp_n_dl")


def _processes_mode(system: UnitSystem) -> None:
    kind = st.radio("¿Qué calculás?", _PROC_KINDS, key="gp_kind", horizontal=True)
    if kind == _PROC_ONE:
        _process_mode(system)
    elif kind == _PROC_STAGED:
        _staged_mode(system)
    else:
        _exponent_mode(system)


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Del [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): "
            "**§4** (gas ideal), **§5** (mezclas), **§6** (transformaciones politrópicas), "
            "**§3.5** (trabajo) y **§10.5** (entropía de los gases ideales). Las propiedades con "
            "c_p variable salen de los polinomios NASA de McBride, Zehe y Gordon (2002)."
        )
        st.markdown("**Gas ideal** (§4.2 a §4.6):")
        st.latex(r"p\,v = R\,T \qquad R = \frac{R_u}{M}")
        st.latex(r"h(T) = u(T) + R\,T")
        st.latex(r"c_p - c_v = R \qquad k = \frac{c_p}{c_v}")
        st.latex(r"\Delta u = \int_1^2 c_v\,dT \qquad \Delta h = \int_1^2 c_p\,dT")
        st.markdown("**Entropía** (§10.5), con s° = ∫c_p dT/T desde 25 °C:")
        st.latex(r"\Delta s = c_p \ln\frac{T_2}{T_1} - R \ln\frac{p_2}{p_1}")
        st.latex(r"\Delta s = s^{\circ}_2 - s^{\circ}_1 - R \ln\frac{p_2}{p_1}")
        st.latex(r"p_r = e^{s^{\circ}/R} \qquad v_r = \frac{T}{p_r}")
        st.latex(r"\left(\frac{p_2}{p_1}\right)_s = \frac{p_{r2}}{p_{r1}}")
        st.latex(r"\left(\frac{v_2}{v_1}\right)_s = \frac{v_{r2}}{v_{r1}}")
        st.markdown("**Mezclas** (§5): x es la fracción másica e y la molar.")
        st.latex(r"y_i = \frac{N_i}{N_M} \qquad x_i = \frac{m_i}{m_M}")
        st.latex(r"x_i = \frac{y_i M_i}{\sum_j y_j M_j}")
        st.latex(r"M_M = \sum_i y_i M_i")
        st.latex(r"R_M = \frac{R_u}{M_M} = \sum_i x_i R_i")
        st.latex(r"p_i = y_i\,p \qquad V_i = y_i\,V")
        st.latex(r"c_{p,M} = \sum_i x_i\,c_{p,i}")
        st.latex(
            r"\begin{aligned}s_M &= \sum_i x_i\,s_i(T, p_i) \\ "
            r"&= \sum_i x_i \left[s_i(T, p) - R_i \ln y_i\right]\end{aligned}"
        )
        st.markdown(
            "**Mezcla adiabática** (Çengel §13-3): U constante en un tanque rígido, H constante en "
            "una cámara de flujo permanente; la exergía destruida es T₀·S_gen (vademecum §11.8)."
        )
        st.latex(
            r"\begin{aligned}S_{\text{gen}} &= \sum_j m_j \big[s_j(T, y_j\,p) \\ "
            r"&\qquad - s_j(T_j, p_j)\big]\end{aligned}"
        )
        st.markdown("**Politrópicas** (§6):")
        st.latex(
            r"p\,v^n = \text{cte} \qquad \frac{T_2}{T_1} = \left(\frac{p_2}{p_1}\right)^{(n-1)/n}"
        )
        st.latex(r"w = \int_1^2 p\,dv = \frac{R\,(T_2 - T_1)}{1 - n}")
        st.latex(r"w_f = -\int_1^2 v\,dp = n\,w")
        st.latex(r"w_{\text{isoterma}} = R\,T \ln\frac{v_2}{v_1} = R\,T \ln\frac{p_1}{p_2}")
        st.latex(r"c = c_v\,\frac{n - k}{n - 1} \qquad q = c\,(T_2 - T_1)")
        st.markdown(
            "Isócora n = ±∞, isóbara n = 0, isoterma n = 1, adiabática reversible n = k (con "
            "c_p constante; §6.4.5). Con c_p variable, la adiabática reversible sigue Δs = 0 con "
            "s° o con v_r (§10.5.3). El exponente de dos estados (§6.1):"
        )
        st.latex(r"n = \frac{\ln(p_1/p_2)}{\ln(v_2/v_1)}")
        st.latex(r"\frac{n - 1}{n} = \frac{\ln(T_2/T_1)}{\ln(p_2/p_1)}")
        st.markdown(
            "**Compresión en N etapas con interenfriamiento perfecto** (§6.3.1): la misma "
            "relación de presiones en cada etapa da el mínimo trabajo."
        )
        st.latex(r"r = \left(\frac{p_2}{p_1}\right)^{1/N}")
        st.latex(r"w_f = -\frac{N\,n\,R\,T_1}{n - 1}\left[r^{(n-1)/n} - 1\right]")


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Gases ideales", page_icon="🎈", layout="centered")

st.subheader(SUBJECT)
st.title("🎈 Gases ideales")
st.markdown(
    "Un **gas ideal entre dos estados** con c_p constante o variable (y cuánto te equivocás con "
    "c_p constante), las **mezclas** de gases ideales (composición, Dalton, Amagat, entropía y la "
    "mezcla adiabática con la exergía destruida) y las **transformaciones politrópicas**: los "
    "cinco caminos hasta el mismo dato, la compresión en etapas y el exponente n de dos estados."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Gases ideales")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="ig_mode")
if mode == _M_GAS:
    _gas_mode(system)
elif mode == _M_MIX:
    _mixtures_mode(system)
else:
    _processes_mode(system)
