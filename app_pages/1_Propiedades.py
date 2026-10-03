"""Página 1 — Estado termodinámico de un fluido puro (pensada para el agua).

El alumno elige el fluido y un par de propiedades independientes (T-p,
p-x, T-x, p-h, p-s, T-s, h-s, T-v, p-v, p-u) y obtiene:

- la región (líquido comprimido, vapor húmedo, vapor sobrecalentado,
  supercrítico) con su sobrecalentamiento, subenfriamiento o título;
- la tabla completa de propiedades (incluye cp, cv, transporte y Z) y la
  fila de las tablas de saturación a la p y a la T del estado;
- advertencias didácticas (p. ej. T-p sobre la curva de saturación);
- el procedimiento para resolverlo con las tablas, en LaTeX con valores;
- el estado sobre el diagrama (log p–h, T–s, h–s, p–log v);
- exportación CSV / JSON.

Toda la lógica vive en :mod:`core.fluids` y :mod:`core.state_report`;
esta página solo presenta. El estado calculado se guarda en
``st.session_state`` (como datos de entrada en SI) para que no
desaparezca al interactuar con el diagrama ni al cambiar de sistema de
unidades.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from core.diagrams import DiagramSpec, cycle_overlays
from core.fluids import (
    FLUID_NAMES_ES,
    PAIR_KWARGS,
    SUPPORTED_FLUIDS,
    FluidState,
    PairCode,
    fluid_limits,
    fluid_state_from_pair,
    suggested_inputs,
)
from core.state_report import (
    INPUT_SPECS,
    PAIR_LABELS_ES,
    PAIR_ORDER,
    PropertyRow,
    build_procedure,
    format_value,
    property_rows,
    region_summary,
    saturation_rows,
    state_notes,
    state_to_csv,
    state_to_dict,
    states_table,
    states_table_csv,
)
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.diagrams import (
    DiagramPoint,
    auto_point_colors,
    diagram_type_selector,
    get_diagram,
    render_diagram_plotly,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.9.0"

_RESULT_KEY = "prop_result"
_LAST_INPUTS_KEY = "prop_last_inputs"
_STATES_KEY = "prop_states"  # tabla de estados: [{label, fluid, pair, inputs_si}]
_STATE_LABEL_KEY = "prop_state_label"
_TOAST_KEY = "prop_toast"

_REGION_ICONS: dict[str, str] = {
    "compressed_liquid": "💧",
    "saturated_liquid": "💧",
    "saturated_mixture": "♨️",
    "saturated_vapor": "☁️",
    "superheated_vapor": "🔥",
    "supercritical": "🌀",
}


# ---------------------------------------------------------------------
# Cálculo (cacheado)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _state_cached(fluid: str, pair: str, inputs: tuple[tuple[str, float], ...]) -> FluidState:
    """Wrapper cacheado sobre :func:`core.fluids.fluid_state_from_pair` (SI)."""
    return fluid_state_from_pair(fluid, pair, **dict(inputs))  # type: ignore[arg-type]


@st.cache_data(show_spinner=False)
def _suggested_cached(fluid: str, pair: str) -> dict[str, float]:
    return suggested_inputs(fluid, pair)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _input_format(kind: QuantityKind, system: UnitSystem) -> str:
    """Decimales razonables para cada magnitud y sistema."""
    if kind == "temperature":
        return "%.2f"
    if kind == "pressure":
        return {"SI": "%.1f", "Técnico": "%.5f", "Inglés": "%.4f"}[system]
    if kind == "specific_enthalpy":
        return "%.1f" if system == "SI" else "%.3f"
    if kind == "specific_entropy":
        return "%.3f" if system == "SI" else "%.5f"
    return "%.6f"


def _render_inputs(fluid: str, pair: PairCode, system: UnitSystem) -> dict[str, float]:
    """Dos inputs del par elegido; devuelve ``{kwarg: valor SI}``.

    El valor inicial es el último calculado para (fluido, par) o, si no
    hay, uno sugerido que siempre es un estado válido del fluido.
    """
    last = st.session_state.get(_LAST_INPUTS_KEY, {}).get(f"{fluid}|{pair}")
    defaults = last if last is not None else _suggested_cached(fluid, pair)
    values: dict[str, float] = {}
    for column, kw in zip(st.columns(2), PAIR_KWARGS[pair], strict=True):
        spec = INPUT_SPECS[kw]
        key = f"prop_in_{fluid}_{pair}_{kw}"
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
                        key=key,
                        help="0 = líquido saturado, 1 = vapor saturado seco.",
                    )
                )
            else:
                values[kw] = number_input_si(
                    label=f"{spec.name} {spec.symbol}",
                    kind=spec.kind,
                    default_si=float(defaults[kw]),
                    key=key,
                    format=_input_format(spec.kind, system),
                )
    return values


# ---------------------------------------------------------------------
# Tabla de estados (para armar un ciclo estado por estado)
# ---------------------------------------------------------------------


def _table_entries() -> list[dict[str, Any]]:
    return st.session_state.setdefault(_STATES_KEY, [])


def _next_label(entries: list[dict[str, Any]]) -> str:
    """Primer número natural que todavía no se usó como nombre de estado."""
    used = {e["label"] for e in entries}
    n = 1
    while str(n) in used:
        n += 1
    return str(n)


def _entry_state(entry: dict[str, Any]) -> FluidState | None:
    try:
        return _state_cached(
            entry["fluid"], entry["pair"], tuple(sorted(entry["inputs_si"].items()))
        )
    except ValueError:
        return None


def _labeled_states(fluid: str | None = None) -> list[tuple[str, FluidState]]:
    """Estados de la tabla (opcionalmente solo los de ``fluid``), ya calculados."""
    out: list[tuple[str, FluidState]] = []
    for entry in _table_entries():
        if fluid is not None and entry["fluid"] != fluid:
            continue
        state = _entry_state(entry)
        if state is not None:
            out.append((entry["label"], state))
    return out


def _add_result_to_table() -> None:
    """Callback del botón: agrega (o reemplaza, si el nombre existe) el estado actual."""
    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        return
    entries = _table_entries()
    label = str(st.session_state.get(_STATE_LABEL_KEY, "")).strip() or _next_label(entries)
    new_entry = {
        "label": label,
        "fluid": result["fluid"],
        "pair": result["pair"],
        "inputs_si": dict(result["inputs_si"]),
    }
    for i, entry in enumerate(entries):
        if entry["label"] == label:
            entries[i] = new_entry
            st.session_state[_TOAST_KEY] = f"Estado «{label}» actualizado en la tabla."
            break
    else:
        entries.append(new_entry)
        st.session_state[_TOAST_KEY] = f"Estado «{label}» agregado a la tabla."
    st.session_state[_STATE_LABEL_KEY] = _next_label(entries)


def _remove_last_state() -> None:
    entries = _table_entries()
    if entries:
        entries.pop()
    st.session_state[_STATE_LABEL_KEY] = _next_label(entries)


def _clear_states() -> None:
    st.session_state[_STATES_KEY] = []
    st.session_state[_STATE_LABEL_KEY] = "1"


def _render_add_to_table() -> None:
    st.session_state.setdefault(_STATE_LABEL_KEY, _next_label(_table_entries()))
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    left.text_input(
        "Nombre del estado",
        key=_STATE_LABEL_KEY,
        help="Por ejemplo 1, 2s o «salida de turbina». Si ya existe, se reemplaza.",
    )
    right.button(
        "➕ Agregar a la tabla de estados",
        key="prop_add_state",
        on_click=_add_result_to_table,
        help="Para armar un ciclo: calculá cada estado y agregalo con su nombre.",
    )


def _render_states_table(system: UnitSystem) -> None:
    labeled = _labeled_states()
    if not labeled:
        return
    st.markdown("---")
    st.markdown("### 🗂️ Tabla de estados")
    st.caption(
        "Los estados que fuiste agregando (por ejemplo, los de un ciclo), recalculados en el "
        "sistema de unidades activo. Activá «Mostrar la tabla de estados» en el diagrama para "
        "verlos juntos."
    )
    rows = states_table(labeled, system)
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column not in ("Estado", "Fluido", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    if frame["Fluido"].nunique() == 1:
        frame = frame.drop(columns="Fluido")
    _show_table(frame)
    left, middle, right = st.columns(3)
    left.download_button(
        "Descargar tabla (CSV)",
        data=states_table_csv(labeled, system).encode("utf-8"),
        file_name="tabla_de_estados.csv",
        mime="text/csv",
        key="prop_download_states",
    )
    middle.button("↩️ Quitar el último", key="prop_remove_last", on_click=_remove_last_state)
    right.button("🗑️ Vaciar la tabla", key="prop_clear_states", on_click=_clear_states)


# ---------------------------------------------------------------------
# Presentación del resultado
# ---------------------------------------------------------------------


def _metric_value(value_si: float | None, kind: QuantityKind | None, system: UnitSystem) -> str:
    """Número con unidad (para textos y captions)."""
    if value_si is None:
        return "—"
    if kind is None:
        return format_value(value_si, 4)
    return f"{format_value(convert_from_si(value_si, kind, system), 5)} {unit_label(kind, system)}"


def _metric(
    column: DeltaGenerator,
    label: str,
    value_si: float | None,
    kind: QuantityKind | None,
    system: UnitSystem,
    help: str | None = None,  # noqa: A002 — replica la API de st.metric
) -> None:
    """``st.metric`` con la unidad en el rótulo: el número solo entra en 4 columnas."""
    unit = "-" if kind is None else unit_label(kind, system)
    value = (
        "—"
        if value_si is None
        else format_value(value_si if kind is None else convert_from_si(value_si, kind, system), 5)
    )
    column.metric(f"{label} [{unit}]", value, help=help)


def _render_metrics(state: FluidState, system: UnitSystem) -> None:
    first = st.columns(4)
    _metric(first[0], "Temperatura T", state.T_K, "temperature", system)
    _metric(first[1], "Presión p", state.P_Pa, "pressure", system)
    _metric(first[2], "Volumen v", state.v_m3_per_kg, "specific_volume", system)
    _metric(
        first[3],
        "Título x",
        state.x,
        None,
        system,
        help="Solo se define dentro de la campana (mezcla líquido–vapor).",
    )
    second = st.columns(4)
    _metric(second[0], "Energía int. u", state.u_J_per_kg, "specific_enthalpy", system)
    _metric(second[1], "Entalpía h", state.h_J_per_kg, "specific_enthalpy", system)
    _metric(second[2], "Entropía s", state.s_J_per_kg_K, "specific_entropy", system)
    if state.superheat_K is not None:
        _metric(
            second[3],
            "Sobrecalent.",
            state.superheat_K,
            "temperature_difference",
            system,
            help="Grado de sobrecalentamiento: T − T_sat(p).",
        )
    elif state.subcooling_K is not None:
        _metric(
            second[3],
            "Subenfriam.",
            state.subcooling_K,
            "temperature_difference",
            system,
            help="Grado de subenfriamiento: T_sat(p) − T.",
        )
    else:
        _metric(second[3], "Densidad ρ", state.rho_kg_per_m3, "density", system)


def _rows_frame(rows: list[PropertyRow]) -> pd.DataFrame:
    """Tabla para mostrar, pensada para pantallas angostas: primero el
    símbolo, el valor y la unidad; el nombre largo al final."""
    return pd.DataFrame(
        {
            "Símbolo": [r.symbol for r in rows],
            "Valor": [format_value(r.value) for r in rows],
            "Unidad": [r.unit for r in rows],
            "Propiedad": [r.name for r in rows],
        }
    )


def _show_table(frame: pd.DataFrame) -> None:
    # Alto fijo para que entren todas las filas sin scroll interno.
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _render_properties(state: FluidState, system: UnitSystem) -> None:
    """Una tabla por grupo; las propiedades no definidas se explican al pie."""
    rows = property_rows(state, system)
    groups = list(dict.fromkeys(r.group for r in rows))
    for group in groups:
        group_rows = [r for r in rows if r.group == group]
        st.markdown(f"**{group}**")
        _show_table(_rows_frame(group_rows))
        notes = sorted({r.note for r in group_rows if r.note})
        if notes:
            st.caption("— : " + "; ".join(notes) + ".")


def _render_saturation(state: FluidState, system: UnitSystem) -> None:
    """Fila de las tablas de saturación a la p y a la T del estado, lado a lado."""
    columns: list[tuple[str, list[PropertyRow]]] = []
    if state.sat_at_P is not None:
        header = f"a p = {_metric_value(state.P_Pa, 'pressure', system)}"
        columns.append((header, saturation_rows(state.sat_at_P, system)))
    # Dentro de la campana la saturación a T es la misma que a p: no se repite.
    if state.sat_at_T is not None and not state.is_two_phase:
        header = f"a T = {_metric_value(state.T_K, 'temperature', system)}"
        columns.append((header, saturation_rows(state.sat_at_T, system)))
    if not columns:
        st.info(
            "No hay saturación a la p ni a la T de este estado: está por encima del punto "
            "crítico (o por debajo del triple).",
            icon="ℹ️",
        )
        return

    # Unión de filas por símbolo (en el orden en que aparecen).
    order: dict[str, PropertyRow] = {}
    for _, rows in columns:
        for r in rows:
            order.setdefault(r.symbol, r)
    data: dict[str, list[str]] = {"Símbolo": list(order)}
    for header, rows in columns:
        by_symbol = {r.symbol: r for r in rows}
        data[header] = [
            format_value(by_symbol[sym].value) if sym in by_symbol else "" for sym in order
        ]
    data["Unidad"] = [r.unit for r in order.values()]
    data["Propiedad"] = [r.name for r in order.values()]
    _show_table(pd.DataFrame(data))
    tables = " y ".join(
        "por presión (Cengel A-5)" if h.startswith("a p") else "por temperatura (Cengel A-4)"
        for h, _ in columns
    )
    st.caption(
        f"Lo que se lee en las tablas de saturación {tables}. Subíndice f: líquido "
        "saturado · g: vapor saturado · fg = g − f (h_fg es el calor latente de "
        "vaporización)."
    )


def _render_procedure(state: FluidState, pair: PairCode, system: UnitSystem) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.caption(
            "Cómo se resuelve con las tablas de vapor, paso a paso. Los valores salen de la "
            "ecuación de estado (IAPWS-95 para el agua), así que pueden diferir en el último "
            "decimal de los de una tabla impresa. En el celular, las ecuaciones largas se "
            "deslizan hacia el costado."
        )
        for i, step in enumerate(build_procedure(state, pair, system), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_diagram(state: FluidState, system: UnitSystem) -> None:
    with st.expander("📈 Diagrama del fluido", expanded=False):
        diagram_type = diagram_type_selector(key="prop_diagram_type", default="Ts")
        table = _labeled_states(state.fluid)
        show_table = connect = False
        if table:
            left, right = st.columns(2)
            show_table = left.toggle(
                "Mostrar la tabla de estados", value=True, key="prop_diagram_show_table"
            )
            connect = right.toggle(
                "Unir los estados en orden (ciclo)",
                value=False,
                key="prop_diagram_connect",
                disabled=not show_table or len(table) < 2,
            )

        points: list[DiagramPoint] = []
        if show_table:
            colors = auto_point_colors(len(table))
            points += [
                DiagramPoint(state=s.to_state_point(), label=label, color=color)
                for (label, s), color in zip(table, colors, strict=True)
            ]
        # El estado actual, salvo que ya esté en la tabla (mismo punto).
        if not any(s == state for _, s in table) or not show_table:
            points.append(
                DiagramPoint(state=state.to_state_point(), label="estado", color="#d62728")
            )
        try:
            overlays = []
            if show_table and connect and len(table) >= 2:
                overlays = cycle_overlays(
                    get_diagram(state.fluid, system),
                    DiagramSpec(fluid=state.fluid, system=system),
                    [s.to_state_point() for _, s in table],
                    close=True,
                )
            render_diagram_plotly(
                fluid=state.fluid,
                diagram_type=diagram_type,
                system=system,
                points=points,
                overlays=overlays,
                chart_key="prop_diagram_chart",
            )
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
        caption = (
            "Isolíneas: T y p constantes y título constante dentro de la campana "
            "(fluprodia + CoolProp). El punto rojo es el estado calculado."
        )
        if connect:
            caption += (
                " Dos estados consecutivos con la misma p se unen por la isobárica, y con la "
                "misma s por la isoentrópica (línea llena). Si no comparten ninguna, la recta "
                "punteada es solo una referencia: no es la trayectoria real del proceso."
            )
        st.caption(caption)


def _render_export(state: FluidState, pair: PairCode, system: UnitSystem) -> None:
    st.markdown("#### 💾 Exportar")
    base = f"estado_{state.fluid}_{pair}".lower()
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=state_to_csv(state, pair, system).encode("utf-8"),
        file_name=f"{base}.csv",
        mime="text/csv",
        key="prop_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(state_to_dict(state, pair, system), ensure_ascii=False, indent=2),
        file_name=f"{base}.json",
        mime="application/json",
        key="prop_download_json",
    )


def _render_state(state: FluidState, pair: PairCode, system: UnitSystem) -> None:
    st.markdown(f"### Resultado — {PAIR_LABELS_ES[pair]}")
    st.info(region_summary(state, system), icon=_REGION_ICONS[state.region])
    for note in state_notes(state, pair, system):
        if note.kind == "warning":
            st.warning(note.text, icon="⚠️")
        else:
            st.info(note.text, icon="ℹ️")

    _render_metrics(state, system)
    _render_add_to_table()

    st.markdown("#### Propiedades del estado")
    _render_properties(state, system)

    st.markdown("#### Propiedades de saturación")
    _render_saturation(state, system)

    _render_procedure(state, pair, system)
    _render_diagram(state, system)
    _render_export(state, pair, system)


def _render_error(exc: ValueError) -> None:
    """Mensaje para el alumno arriba; el detalle de CoolProp, aparte y chico."""
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico (CoolProp): {detail.strip()}")


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"Las relaciones que usa esta página están en el [vademecum de la cátedra]"
            f"({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})): §3.2 *Entalpía*, "
            "§7.3 *Factor de compresibilidad*, **§12 *Vapor húmedo*** y **§13 *Líquidos y "
            "sólidos (incompresibles)***."
        )
        st.markdown(
            "**Postulado de estado.** Para una sustancia pura simple compresible, dos "
            "propiedades intensivas *independientes* fijan el estado. Dentro de la campana "
            "T y p **no** son independientes (T = T_sat(p)): hace falta otra propiedad, "
            "como el título."
        )
        st.markdown(
            "**Vapor húmedo** (§12) — título, regla de la palanca (para y = v, u, h o s) y "
            "calor latente:"
        )
        st.latex(r"x \equiv \frac{m_g}{m_f + m_g}")
        st.latex(r"y = y_f + x\,(y_g - y_f)")
        st.latex(r"x = \frac{y - y_f}{y_g - y_f} \qquad h_{fg} \equiv h_g - h_f")
        st.markdown("**Entalpía** (§3.2):")
        st.latex(r"h \equiv u + p\,v")
        st.markdown(
            "**Líquido comprimido** — si no hay tabla, aproximación de líquido incompresible "
            "a la misma temperatura (§13; Cengel §3-5):"
        )
        st.latex(r"v \approx v_f(T) \quad u \approx u_f(T) \quad s \approx s_f(T)")
        st.latex(r"h \approx h_f(T) + v_f(T)\,[p - p_{\mathrm{sat}}(T)]")
        st.markdown(
            "**Vapor sobrecalentado** — grado de sobrecalentamiento y desvío respecto del "
            "gas ideal (§7.3):"
        )
        st.latex(r"\Delta T_{\mathrm{sob}} = T - T_{\mathrm{sat}}(p) \qquad Z = \frac{p\,v}{R\,T}")


def _fluid_caption(fluid: str, system: UnitSystem) -> str:
    lim = fluid_limits(fluid)
    model = "IAPWS-95" if fluid == "Water" else "ecuación de estado de Helmholtz"
    return (
        f"Punto crítico: T_c = {_metric_value(lim.T_crit_K, 'temperature', system)}, "
        f"p_c = {_metric_value(lim.P_crit_Pa, 'pressure', system)} · Punto triple: "
        f"T_t = {_metric_value(lim.T_triple_K, 'temperature', system)}, "
        f"p_t = {_metric_value(lim.P_triple_Pa, 'pressure', system)} · "
        f"R = {_metric_value(lim.R_J_per_kg_K, 'specific_entropy', system)} · "
        f"Modelo: {model} (CoolProp)."
    )


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

st.set_page_config(page_title="Propiedades", page_icon="💧", layout="centered")

sidebar_credits(version=PAGE_VERSION, page_name="Propiedades")
render_units_selector()
system = get_current_system()

st.subheader(SUBJECT)
st.title("💧 Estado termodinámico")
st.markdown(
    "Ingresá **dos propiedades independientes** y obtené el estado completo: en qué región "
    "está (líquido comprimido, vapor húmedo, vapor sobrecalentado o supercrítico), todas "
    "sus propiedades, la fila de las tablas de saturación, el procedimiento para "
    "resolverlo con las tablas y el punto sobre el diagrama. Las unidades se eligen en el "
    "menú lateral."
)
_render_theory()
st.markdown("---")

fluid = st.selectbox(
    "Fluido",
    SUPPORTED_FLUIDS,
    format_func=lambda f: FLUID_NAMES_ES.get(f, f),
    key="prop_fluid",
)
st.caption(_fluid_caption(fluid, system))
pair: PairCode = st.selectbox(
    "Propiedades conocidas",
    PAIR_ORDER,
    format_func=lambda p: PAIR_LABELS_ES[p],
    key="prop_pair",
)

with st.form(key="prop_form"):
    inputs_si = _render_inputs(fluid, pair, system)
    submitted = st.form_submit_button("Calcular estado", type="primary")

if submitted:
    st.session_state[_RESULT_KEY] = {"fluid": fluid, "pair": pair, "inputs_si": inputs_si}
    st.session_state.setdefault(_LAST_INPUTS_KEY, {})[f"{fluid}|{pair}"] = inputs_si

result = st.session_state.get(_RESULT_KEY)
if result is not None and result["fluid"] == fluid and result["pair"] == pair:
    try:
        state = _state_cached(fluid, pair, tuple(sorted(result["inputs_si"].items())))
    except ValueError as exc:
        _render_error(exc)
    else:
        _render_state(state, pair, system)
else:
    st.info(
        "Elegí el fluido y el par de propiedades, cargá los valores y tocá **Calcular "
        "estado**. Los valores iniciales son un estado de ejemplo válido.",
        icon="👆",
    )

_render_states_table(system)

toast = st.session_state.pop(_TOAST_KEY, None)
if toast:
    st.toast(toast, icon="🗂️")
