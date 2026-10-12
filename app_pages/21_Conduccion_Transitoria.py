"""Página 21 — Conducción transitoria.

Fase 11.1. Cuatro modos:

- **Capacidad concentrada** (Çengel y Ghajar, §4-1; Incropera, §5.1 a §5.3): un cuerpo de
  temperatura uniforme (Bi < 0,1), con la constante de tiempo y la comparación con la
  solución exacta.
- **Pared, cilindro largo o esfera** (§4-2): la solución de un término (λ₁ y A₁) y la serie
  completa, la temperatura del centro y de cualquier posición, y el calor.
- **Sólido semi-infinito** (§4-3): la superficie a T_s, con un flujo constante o con
  convección; y el contacto de dos sólidos.
- **Varias dimensiones** (§4-4): el producto de soluciones de una dimensión y la regla de
  Langston para el calor.

El cálculo vive en :mod:`core.heat_transfer.transient`; los gráficos, en
:mod:`ui.transient_charts`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.heat_transfer import transient as tr
from core.heat_transfer.transient_procedure import (
    contact_steps,
    lumped_steps,
    one_dim_steps,
    product_steps,
    semi_infinite_steps,
)
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.transient_charts import (
    contact_figure,
    history_figure,
    lumped_figure,
    product_map_figure,
    profiles_figure,
    semi_infinite_figure,
    time_label,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.30.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_LUMP = "Capacidad concentrada"
_M_1D = "Pared, cilindro largo o esfera"
_M_SEMI = "Sólido semi-infinito"
_M_PROD = "Varias dimensiones (producto)"
_MODES = (_M_LUMP, _M_1D, _M_SEMI, _M_PROD)

_W_SURF = "Una superficie con T, flujo o convección"
_W_CONTACT = "Dos sólidos que se tocan"
_SEMI_WHAT = (_W_SURF, _W_CONTACT)

_G_TIME = "El tiempo (sale la temperatura)"
_G_TEMP = "Una temperatura (sale el tiempo)"
_GIVEN = (_G_TIME, _G_TEMP)

_U_T = "La temperatura en x y t"
_U_X = "La profundidad a la que llega una temperatura"
_U_t = "El tiempo que tarda una temperatura en llegar a x"
_SEMI_UNKNOWN = (_U_T, _U_X, _U_t)

_OTHER = "Otro (k, ρ y c a elección)"

_T: QuantityKind = "temperature"
_DT: QuantityKind = "temperature_difference"
_X: QuantityKind = "small_length"
#: El semi-infinito va en m o ft: los caños enterrados están a casi un metro y δ llega a
#: varios metros en meses (en mm se leía «4000»).
_XL: QuantityKind = "length"
_H: QuantityKind = "heat_transfer_coefficient"
_FLUX: QuantityKind = "heat_flux"

_TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0, "días": 86400.0}

#: Medidas por defecto de cada cuerpo (cuando se cambia de cuerpo).
_BODY_DEFAULTS: dict[tr.Body, tuple[float, ...]] = {
    "sphere": (0.02,),
    "cylinder": (0.02, 0.1),
    "long_cylinder": (0.02,),
    "plate": (0.01,),
    "cube": (0.02,),
    "box": (0.02, 0.03, 0.04),
    "custom": (1.0e-5, 0.006),
}
_DIM_LABELS = {
    "D": "Diámetro D",
    "L": "Largo L",
    "e": "Espesor e",
    "a": "Arista a",
    "b": "Arista b",
    "c": "Arista c",
    "V": "Volumen V",
    "A": "Área de intercambio A",
}

#: Medidas y punto por defecto de cada forma del producto (cuando se cambia de forma).
_SHAPE_DEFAULTS: dict[tr.ProductShape, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "short_cylinder": ((0.1, 0.12), (0.0, 0.0)),
    "semi_cylinder": ((0.2,), (0.0, 0.05)),
    "bar": ((0.1, 0.05), (0.0, 0.0)),
    "box": ((0.2, 0.1, 0.06), (0.0, 0.0, 0.0)),
    "semi_plate": ((0.1,), (0.0, 0.02)),
    "quarter": ((), (0.01, 0.01)),
    "corner": ((), (0.01, 0.01, 0.01)),
}
_SHAPE_DIMS: dict[tr.ProductShape, tuple[str, ...]] = {
    "short_cylinder": ("Diámetro D", "Alto H"),
    "semi_cylinder": ("Diámetro D",),
    "bar": ("Lado a", "Lado b"),
    "box": ("Lado a", "Lado b", "Lado c"),
    "semi_plate": ("Espesor e",),
    "quarter": (),
    "corner": (),
}
#: El punto de cada forma: (rótulo, True si es una profundidad en vez de una ξ).
_SHAPE_POINT: dict[tr.ProductShape, tuple[tuple[str, bool], ...]] = {
    "short_cylinder": (("r/r₀", False), ("x/L (desde el plano medio, L = H/2)", False)),
    "semi_cylinder": (("r/r₀", False), ("Profundidad desde la cara expuesta", True)),
    "bar": (("x/L₁ (L₁ = a/2)", False), ("y/L₂ (L₂ = b/2)", False)),
    "box": (("x/L₁ (L₁ = a/2)", False), ("y/L₂ (L₂ = b/2)", False), ("z/L₃ (L₃ = c/2)", False)),
    "semi_plate": (("x/L (L = e/2)", False), ("Profundidad desde el borde", True)),
    "quarter": (("Profundidad desde la cara 1", True), ("Profundidad desde la cara 2", True)),
    "corner": (
        ("Profundidad desde la cara 1", True),
        ("Profundidad desde la cara 2", True),
        ("Profundidad desde la cara 3", True),
    ),
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


def _t_scale(system: UnitSystem) -> tuple[float, float]:
    """(escala, offset) para pasar de K a la temperatura del sistema."""
    offset = convert_from_si(0.0, _T, system)
    return convert_from_si(1.0, _T, system) - offset, offset


def _chart_units(system: UnitSystem, x_kind: QuantityKind = _X) -> dict[str, Any]:
    t_scale, t_offset = _t_scale(system)
    return {
        "t_scale": t_scale,
        "t_offset": t_offset,
        "t_unit": _u(_T, system),
        "x_scale": convert_from_si(1.0, x_kind, system),
        "x_unit": _u(x_kind, system),
    }


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


def _notes(notes: tuple[str, ...]) -> None:
    for note in notes:
        st.info(note)


def _chart(fig: Any, key: str, caption: str = "") -> None:
    """Un gráfico que falla no tumba la página."""
    try:
        st.plotly_chart(fig, width="stretch", key=key)
        if caption:
            st.caption(caption)
    except Exception as exc:  # noqa: BLE001 — se avisa y la página sigue
        st.warning(f"No se pudo dibujar el gráfico: {exc}")


def _example(examples: dict[str, tr.TransientExample], key: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox(
        "Ejemplo precargado",
        names,
        key=key,
        help=(
            "Ejemplos de Çengel y Ghajar (cap. 4) e Incropera (cap. 5), y algunos propios. "
            "Podés cambiar cualquier dato: el resultado se actualiza solo."
        ),
    )
    st.caption(f"📘 {examples[name].description}")
    return name, names.index(name)


def _auto_unit(t_s: float) -> str:
    if t_s < 120.0:
        return "s"
    if t_s < 2.0 * 3600.0:
        return "min"
    if t_s < 3.0 * 86400.0:
        return "h"
    return "días"


def _time_input(label: str, default_s: float, key: str) -> float:
    """Un tiempo en s, min, h o días (la unidad se elige al lado); devuelve segundos.

    Como :func:`ui.units_ui.number_input_si`, el widget lleva la unidad en la key: al
    cambiarla se crea uno nuevo con el mismo tiempo en la unidad nueva.
    """
    units = list(_TIME_UNITS)
    left, right = st.columns([3, 1])
    with right:
        unit = st.selectbox(
            "Unidad", units, index=units.index(_auto_unit(default_s)), key=f"{key}_unit"
        )
    widget_key = f"{key}@{unit}"
    value = default_s / _TIME_UNITS[unit]
    if widget_key not in st.session_state:
        for other in units:
            previous = st.session_state.get(f"{key}@{other}")
            if other != unit and previous is not None:
                value = float(previous) * _TIME_UNITS[other] / _TIME_UNITS[unit]
                break
    with left:
        shown = st.number_input(
            f"{label} [{unit}]", min_value=0.0, value=float(value), format="%.5g", key=widget_key
        )
    return float(shown) * _TIME_UNITS[unit]


def _temperature(label: str, default_K: float, key: str, help_text: str | None = None) -> float:
    return number_input_si(
        label=label, kind=_T, default_si=default_K, key=key, format="%.5g", help=help_text
    )


def _length(label: str, default_m: float, key: str, kind: QuantityKind = _X) -> float:
    return number_input_si(
        label=label, kind=kind, default_si=default_m, key=key, format="%.5g", min_value_si=0.0
    )


def _h_input(default: float, key: str) -> float:
    return number_input_si(
        label="Coeficiente de convección h",
        kind=_H,
        default_si=default,
        key=key,
        format="%.5g",
        min_value_si=0.0,
        help="Con la radiación incluida si corresponde (el h combinado de los ejemplos).",
    )


def _solid_input(default: tr.Solid, key: str, label: str = "Material") -> tr.Solid:
    """El sólido de la tabla (Incropera, a 300 K) o uno a elección."""
    table = list(tr.SOLIDS.values())
    names = [s.name for s in table] + [_OTHER]
    index = names.index(default.name) if default in table else len(names) - 1
    choice = st.selectbox(
        label,
        names,
        index=index,
        key=f"{key}_mat",
        help="k, ρ y c a 300 K de Incropera et al. (2007), tablas A.1 (metales) y A.3.",
    )
    system = get_current_system()
    if choice != _OTHER:
        solid = table[names.index(choice)]
    else:
        base = default if default not in table else tr.SOLIDS["carbon_steel"]
        cols = st.columns(3)
        with cols[0]:
            k = number_input_si(
                label="k",
                kind="thermal_conductivity",
                default_si=base.k,
                key=f"{key}_k",
                format="%.4g",
                min_value_si=0.0,
            )
        with cols[1]:
            rho = number_input_si(
                label="ρ",
                kind="density",
                default_si=base.rho,
                key=f"{key}_rho",
                format="%.5g",
                min_value_si=0.0,
            )
        with cols[2]:
            c = number_input_si(
                label="c",
                kind="specific_heat",
                default_si=base.c,
                key=f"{key}_c",
                format="%.5g",
                min_value_si=0.0,
            )
        name = base.name if default not in table else "Material a elección"
        solid = tr.Solid(name, k, rho, c)
    if solid.k > 0.0 and solid.rho > 0.0 and solid.c > 0.0:
        st.caption(
            f"k = {_val(solid.k, 'thermal_conductivity', system, 4)} "
            f"{_u('thermal_conductivity', system)} · ρ = {_val(solid.rho, 'density', system, 4)} "
            f"{_u('density', system)} · c = {_val(solid.c, 'specific_heat', system, 4)} "
            f"{_u('specific_heat', system)} · α = {_val(solid.alpha, 'diffusivity', system, 4)} "
            f"{_u('diffusivity', system)}"
        )
    return solid


# ---------------------------------------------------------------------
# Capacidad concentrada
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _lumped_cached(inputs: tr.LumpedInputs) -> tr.LumpedResult:
    return tr.solve_lumped(inputs)


def _lumped(system: UnitSystem) -> None:
    _name, idx = _example(tr.LUMPED_EXAMPLES, "tl_example")
    inp: tr.LumpedInputs = tr.LUMPED_EXAMPLES[_name].inputs
    key = f"tl_{idx}"
    bodies = list(tr.BODIES)
    body: tr.Body = st.selectbox(
        "Cuerpo",
        bodies,
        index=bodies.index(inp.body),
        format_func=lambda b: tr.BODIES[b],
        key=f"{key}_body",
    )
    defaults = inp.dims if body == inp.body else _BODY_DEFAULTS[body]
    dims: list[float] = []
    names = tr.BODY_DIMENSIONS[body]
    cols = st.columns(len(names))
    for col, dim, default in zip(cols, names, defaults, strict=True):
        with col:
            if dim == "V":
                value = number_input_si(
                    label=_DIM_LABELS[dim],
                    kind="volume",
                    default_si=default,
                    key=f"{key}_{body}_{dim}",
                    format="%.5g",
                    min_value_si=0.0,
                )
            elif dim == "A":
                value = number_input_si(
                    label=_DIM_LABELS[dim],
                    kind="area",
                    default_si=default,
                    key=f"{key}_{body}_{dim}",
                    format="%.5g",
                    min_value_si=0.0,
                )
            else:
                value = _length(_DIM_LABELS[dim], default, f"{key}_{body}_{dim}")
        dims.append(value)
    solid = _solid_input(inp.solid, key)
    h = _h_input(inp.h, f"{key}_h")
    left, right = st.columns(2)
    with left:
        T_i = _temperature("Temperatura inicial T_i", inp.T_i, f"{key}_Ti")
    with right:
        T_inf = _temperature("Temperatura del fluido T∞", inp.T_inf, f"{key}_Tinf")
    given = st.radio(
        "¿Qué dato tenés?",
        _GIVEN,
        index=1 if inp.T_target_K is not None else 0,
        key=f"{key}_given",
        horizontal=True,
    )
    time_s = T_target = None
    if given == _G_TIME:
        time_s = _time_input("Tiempo t", inp.time_s or 60.0, f"{key}_t")
    else:
        default_T = inp.T_target_K or (T_inf + 0.5 * (T_i - T_inf))
        T_target = _temperature("Temperatura que tiene que alcanzar", default_T, f"{key}_Tt")
    try:
        r = _lumped_cached(
            tr.LumpedInputs(body, tuple(dims), solid, h, T_i, T_inf, time_s, T_target)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_lumped(r, system)


def _render_lumped(r: tr.LumpedResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    q_kind: QuantityKind = {"plate": "energy_per_area", "long_cylinder": "energy_per_length"}.get(
        i.body, "energy"
    )  # type: ignore[assignment]
    verb = "entregado" if i.T_i > i.T_inf else "recibido"
    items = [
        ("Biot", _num(r.Bi, 4)),
        (f"L_c [{_u(_X, system)}]", _val(r.Lc, _X, system, 4)),
        ("Constante de tiempo 1/b", time_label(r.time_constant_s)),
    ]
    if i.T_target_K is None:
        items.append(
            (f"T a t = {time_label(r.time_s)} [{_u(_T, system)}]", _val(r.T_K, _T, system))
        )
    else:
        items.append(("Tiempo t", time_label(r.time_s)))
    items.append((f"Calor {verb} Q [{_u(q_kind, system)}]", _val(abs(r.Q_J), q_kind, system)))
    items.append(("Q/Q_máx", _pct(1.0 - r.theta)))
    _metric_rows(items)
    _notes(r.notes)
    st.markdown("#### La temperatura en el tiempo")
    _chart(
        lumped_figure(r, **{k: v for k, v in _chart_units(system).items() if k.startswith("t_")}),
        "tl_chart",
        "La curva azul es la capacidad concentrada; en gris, la solución exacta (la temperatura "
        "media y la del centro), que coinciden con ella cuando Bi < 0,1. La línea punteada marca "
        "la constante de tiempo 1/b.",
    )
    _render_steps(lumped_steps(r, system))
    _render_export(tr.lumped_to_dict(r, system), "capacidad_concentrada", "tl_dl")


# ---------------------------------------------------------------------
# Pared, cilindro y esfera
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _one_dim_cached(inputs: tr.OneDimInputs) -> tr.OneDimResult:
    return tr.solve_one_dimensional(inputs)


def _one_dim(system: UnitSystem) -> None:
    _name, idx = _example(tr.ONE_DIM_EXAMPLES, "tw_example")
    inp: tr.OneDimInputs = tr.ONE_DIM_EXAMPLES[_name].inputs
    key = f"tw_{idx}"
    geometries = list(tr.GEOMETRIES)
    geometry: tr.Geometry1D = st.radio(
        "Geometría",
        geometries,
        index=geometries.index(inp.geometry),
        format_func=lambda g: tr.GEOMETRIES[g],
        key=f"{key}_geo",
        horizontal=True,
    )
    size_label = (
        "Semiespesor L (la mitad del espesor)" if geometry == "wall" else "Radio exterior r₀"
    )
    size = _length(size_label, inp.size, f"{key}_size")
    solid = _solid_input(inp.solid, key)
    h = _h_input(inp.h, f"{key}_h")
    left, right = st.columns(2)
    with left:
        T_i = _temperature("Temperatura inicial T_i", inp.T_i, f"{key}_Ti")
    with right:
        T_inf = _temperature("Temperatura del fluido T∞", inp.T_inf, f"{key}_Tinf")
    position = float(
        st.number_input(
            "Posición x/L" if geometry == "wall" else "Posición r/r₀",
            min_value=0.0,
            max_value=1.0,
            value=float(inp.position),
            step=0.1,
            format="%.3g",
            key=f"{key}_pos",
            help="0 es el centro (el plano medio de la pared) y 1 la superficie.",
        )
    )
    given = st.radio(
        "¿Qué dato tenés?",
        (_G_TIME, "La temperatura del centro (sale el tiempo)"),
        index=1 if inp.T_center_target_K is not None else 0,
        key=f"{key}_given",
        horizontal=True,
    )
    time_s = T_target = None
    if given == _G_TIME:
        time_s = _time_input("Tiempo t", inp.time_s or 600.0, f"{key}_t")
    else:
        default_T = inp.T_center_target_K or (T_inf + 0.5 * (T_i - T_inf))
        T_target = _temperature("Temperatura del centro buscada", default_T, f"{key}_Tt")
    try:
        r = _one_dim_cached(
            tr.OneDimInputs(geometry, size, solid, h, T_i, T_inf, position, time_s, T_target)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_one_dim(r, system)


def _render_one_dim(r: tr.OneDimResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    q_kind: QuantityKind = {"wall": "energy_per_area", "cylinder": "energy_per_length"}.get(
        i.geometry, "energy"
    )  # type: ignore[assignment]
    verb = "entregado" if i.T_i > i.T_inf else "recibido"
    items = [
        ("Biot", _num(r.Bi, 4)),
        ("Fourier τ", _num(r.tau, 4)),
        ("λ₁", _num(r.lambda1, 5)),
        ("A₁", _num(r.A1, 5)),
    ]
    if i.T_center_target_K is not None:
        items.append(("Tiempo t", time_label(r.time_s)))
    items += [
        (f"T del centro [{_u(_T, system)}]", _val(r.T_center_K, _T, system)),
        (f"T de la superficie [{_u(_T, system)}]", _val(r.T_surface_K, _T, system)),
    ]
    if 0.0 < i.position < 1.0:
        items.append((f"T en la posición [{_u(_T, system)}]", _val(r.T_K, _T, system)))
    items += [
        ("Q/Q_máx", _pct(r.heat_ratio)),
        (f"Calor {verb} Q [{_u(q_kind, system)}]", _val(abs(r.Q), q_kind, system)),
    ]
    _metric_rows(items)
    st.markdown("#### Un término contra la serie completa")
    rows = []
    if i.T_center_target_K is not None:
        rows.append(
            {
                "": "Tiempo t",
                "Un término": time_label(r.time_one_term_s or 0.0),
                "Serie completa": time_label(r.time_s),
            }
        )
    else:
        rows.append(
            {
                "": "θ₀ del centro",
                "Un término": _num(r.theta0_one, 4),
                "Serie completa": _num(r.theta0, 4),
            }
        )
        rows.append(
            {
                "": f"T del centro [{_u(_T, system)}]",
                "Un término": _val(r.T_center_one_K, _T, system),
                "Serie completa": _val(r.T_center_K, _T, system),
            }
        )
    rows.append(
        {
            "": "Q/Q_máx",
            "Un término": _num(r.heat_ratio_one, 4),
            "Serie completa": _num(r.heat_ratio, 4),
        }
    )
    _table(pd.DataFrame(rows))
    _notes(r.notes)
    units = _chart_units(system)
    st.markdown("#### Perfiles de temperatura")
    _chart(
        profiles_figure(r, **units),
        "tw_profiles",
        "Del más claro al más oscuro, cinco tiempos hasta el pedido. La superficie se acerca "
        "primero a T∞ y el centro después.",
    )
    st.markdown("#### La temperatura del centro en el tiempo")
    _chart(
        history_figure(r, **{k: v for k, v in units.items() if k.startswith("t_")}),
        "tw_history",
        "Lo que leen los gráficos de Heisler. La solución de un término (gris) se separa de la "
        "serie a tiempos cortos: desde τ = 0,2 casi coinciden.",
    )
    _render_steps(one_dim_steps(r, system))
    _render_export(tr.one_dim_to_dict(r, system), "conduccion_transitoria", "tw_dl")


# ---------------------------------------------------------------------
# Sólido semi-infinito y contacto
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _semi_cached(inputs: tr.SemiInfiniteInputs) -> tr.SemiInfiniteResult:
    return tr.solve_semi_infinite(inputs)


@st.cache_data(show_spinner=False)
def _contact_cached(inputs: tr.ContactInputs) -> tr.ContactResult:
    return tr.solve_contact(inputs)


def _semi(system: UnitSystem) -> None:
    _name, idx = _example(tr.SEMI_INFINITE_EXAMPLES, "ti_example")
    inp: tr.SemiInfiniteInputs = tr.SEMI_INFINITE_EXAMPLES[_name].inputs
    key = f"ti_{idx}"
    boundaries = list(tr.SEMI_BOUNDARIES)
    boundary: tr.SemiBoundary = st.radio(
        "Condición en la superficie",
        boundaries,
        index=boundaries.index(inp.boundary),
        format_func=lambda b: tr.SEMI_BOUNDARIES[b],
        key=f"{key}_bnd",
    )
    solid = _solid_input(inp.solid, key)
    T_i = _temperature("Temperatura inicial T_i", inp.T_i, f"{key}_Ti")
    T_s = q_s = h = T_inf = None
    if boundary == "temperature":
        T_s = _temperature(
            "Temperatura de la superficie T_s", inp.T_s or inp.T_i + 50.0, f"{key}_Ts"
        )
    elif boundary == "flux":
        q_s = number_input_si(
            label="Flujo que entra q̇_s",
            kind=_FLUX,
            default_si=inp.q_s or 1000.0,
            key=f"{key}_qs",
            format="%.5g",
            help="Positivo si el calor entra al sólido; negativo si sale.",
        )
    else:
        left, right = st.columns(2)
        with left:
            h = _h_input(inp.h or 25.0, f"{key}_h")
        with right:
            T_inf = _temperature(
                "Temperatura del fluido T∞", inp.T_inf or inp.T_i + 50.0, f"{key}_Tinf"
            )
    if inp.T_target_K is None:
        default_unknown = 0
    else:
        default_unknown = 1 if inp.depth_m is None else 2
    unknown = st.radio("¿Qué buscás?", _SEMI_UNKNOWN, index=default_unknown, key=f"{key}_unk")
    depth = time_s = T_target = None
    if unknown != _U_X:
        depth = _length(
            "Profundidad x (desde la superficie)",
            0.02 if inp.depth_m is None else inp.depth_m,
            f"{key}_depth",
            _XL,
        )
    if unknown != _U_t:
        time_s = _time_input("Tiempo t", inp.time_s or 3600.0, f"{key}_t")
    if unknown != _U_T:
        far = T_s if T_s is not None else T_inf
        if far is None:
            far = T_i + (30.0 if (q_s or 0.0) > 0.0 else -30.0)
        default_T = inp.T_target_K or (T_i + 0.5 * (far - T_i))
        T_target = _temperature("Temperatura buscada", default_T, f"{key}_Tt")
    try:
        r = _semi_cached(
            tr.SemiInfiniteInputs(boundary, solid, T_i, T_s, q_s, h, T_inf, depth, time_s, T_target)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_semi(r, system)


def _render_semi(r: tr.SemiInfiniteResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items: list[tuple[str, str]] = []
    if i.T_target_K is not None and i.depth_m is None:
        items.append((f"Profundidad x [{_u(_XL, system)}]", _val(r.depth_m, _XL, system)))
    elif i.T_target_K is not None:
        items.append(("Tiempo t", time_label(r.time_s)))
    else:
        items.append((f"T en x y t [{_u(_T, system)}]", _val(r.T_K, _T, system)))
    items += [
        (f"T de la superficie [{_u(_T, system)}]", _val(r.T_surface_K, _T, system)),
        ("η = x/(2√(αt))", _num(r.eta, 4)),
        (f"q̇ en la superficie [{_u(_FLUX, system)}]", _val(r.surface_flux, _FLUX, system)),
        (
            f"Energía que entró Q/A [{_u('energy_per_area', system)}]",
            _val(r.energy_per_area, "energy_per_area", system),
        ),
        (f"Penetración δ [{_u(_XL, system)}]", _val(r.penetration_m, _XL, system)),
    ]
    _metric_rows(items)
    _notes(r.notes)
    st.markdown("#### Perfiles de temperatura")
    _chart(
        semi_infinite_figure(r, **_chart_units(system, _XL)),
        "ti_profiles",
        "Del más claro al más oscuro, cuatro tiempos hasta el pedido; la línea punteada es la "
        "penetración δ al último.",
    )
    _render_steps(semi_infinite_steps(r, system))
    _render_export(tr.semi_infinite_to_dict(r, system), "semi_infinito", "ti_dl")


def _contact(system: UnitSystem) -> None:
    _name, idx = _example(tr.CONTACT_EXAMPLES, "tc_example")
    inp: tr.ContactInputs = tr.CONTACT_EXAMPLES[_name].inputs
    key = f"tc_{idx}"
    st.markdown("**Sólido A**")
    solid_a = _solid_input(inp.solid_a, f"{key}_a", "Material de A")
    T_a = _temperature("Temperatura de A", inp.T_a, f"{key}_Ta")
    st.markdown("**Sólido B**")
    solid_b = _solid_input(inp.solid_b, f"{key}_b", "Material de B")
    T_b = _temperature("Temperatura de B", inp.T_b, f"{key}_Tb")
    time_s = None
    if st.checkbox(
        "El flujo por el contacto a un tiempo dado", value=inp.time_s is not None, key=f"{key}_q"
    ):
        time_s = _time_input("Tiempo t", inp.time_s or 1.0, f"{key}_t")
    try:
        r = _contact_cached(tr.ContactInputs(solid_a, T_a, solid_b, T_b, time_s))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    st.markdown("### Resultado")
    items = [
        (f"Temperatura de contacto [{_u(_T, system)}]", _val(r.T_contact_K, _T, system)),
        ("Peso de A en la temperatura", _pct(r.share_a)),
    ]
    if r.flux_W_m2 is not None:
        items.append((f"q̇ de A a B [{_u(_FLUX, system)}]", _val(r.flux_W_m2, _FLUX, system)))
    _metric_rows(items)
    _notes(r.notes)
    st.markdown("#### Los dos sólidos")
    _chart(
        contact_figure(r, **_chart_units(system)),
        "tc_profiles",
        "Cada sólido con su escala de largo: el calor llega mucho más lejos en el de mayor α. La "
        "superficie de contacto queda a T_s todo el tiempo.",
    )
    _render_steps(contact_steps(r, system))
    _render_export(tr.contact_to_dict(r, system), "contacto", "tc_dl")


# ---------------------------------------------------------------------
# Producto
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _product_cached(inputs: tr.ProductInputs) -> tr.ProductResult:
    return tr.solve_product(inputs)


def _dims_and_point(inp: tr.ProductInputs) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Las medidas (diámetros, espesores) y el punto de un ejemplo, desde sus factores."""
    dims = tuple(2.0 * (f.size or 0.0) for f in inp.factors if f.kind != "semi")
    return dims, tuple(f.position for f in inp.factors)


def _product(system: UnitSystem) -> None:
    _name, idx = _example(tr.PRODUCT_EXAMPLES, "tp_example")
    inp: tr.ProductInputs = tr.PRODUCT_EXAMPLES[_name].inputs
    key = f"tp_{idx}"
    shapes = list(tr.PRODUCT_SHAPES)
    shape: tr.ProductShape = st.selectbox(
        "Forma",
        shapes,
        index=shapes.index(inp.shape),
        format_func=lambda s: tr.PRODUCT_SHAPES[s],
        key=f"{key}_shape",
    )
    dims0, point0 = _dims_and_point(inp) if shape == inp.shape else _SHAPE_DEFAULTS[shape]
    dims: list[float] = []
    labels = _SHAPE_DIMS[shape]
    if labels:
        cols = st.columns(len(labels))
        for j, (col, label) in enumerate(zip(cols, labels, strict=True)):
            with col:
                dims.append(_length(label, dims0[j], f"{key}_{shape}_d{j}"))
    st.markdown("**El punto**")
    point: list[float] = []
    for j, (label, is_depth) in enumerate(_SHAPE_POINT[shape]):
        if is_depth:
            point.append(_length(label, point0[j], f"{key}_{shape}_p{j}"))
        else:
            point.append(
                float(
                    st.number_input(
                        label,
                        min_value=0.0,
                        max_value=1.0,
                        value=float(point0[j]),
                        step=0.1,
                        format="%.3g",
                        key=f"{key}_{shape}_p{j}",
                        help="0 es el centro (o el eje) y 1 la superficie.",
                    )
                )
            )
    solid = _solid_input(inp.solid, key)
    h = _h_input(inp.h, f"{key}_h")
    left, right = st.columns(2)
    with left:
        T_i = _temperature("Temperatura inicial T_i", inp.T_i, f"{key}_Ti")
    with right:
        T_inf = _temperature("Temperatura del fluido T∞", inp.T_inf, f"{key}_Tinf")
    time_s = _time_input("Tiempo t", inp.time_s, f"{key}_t")
    try:
        factors = tr.product_factors(shape, tuple(dims), tuple(point))
        r = _product_cached(tr.ProductInputs(shape, factors, solid, h, T_i, T_inf, time_s))
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_product(r, system)


def _render_product(r: tr.ProductResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    verb = "entregado" if i.T_i > i.T_inf else "recibido"
    items = [
        (f"T en el punto [{_u(_T, system)}]", _val(r.T_K, _T, system)),
        ("θ = (T − T∞)/(T_i − T∞)", _num(r.theta, 4)),
    ]
    if r.heat_ratio is not None:
        items.append(("Q/Q_máx (Langston)", _pct(r.heat_ratio)))
    if r.Q is not None:
        items.append((f"Calor {verb} Q [{_u('energy', system)}]", _val(abs(r.Q), "energy", system)))
    _metric_rows(items)
    _table(
        pd.DataFrame(
            [
                {
                    "Factor": f.factor.label or f.factor.kind,
                    "Bi": _num(f.Bi, 4),
                    "τ": _num(f.tau, 4),
                    "θ": _num(f.theta, 4),
                    "Q/Q_máx": _num(f.heat_ratio, 4),
                }
                for f in r.factors
            ]
        )
    )
    _notes(r.notes)
    fig = product_map_figure(r, **_chart_units(system))
    if fig is not None:
        st.markdown("#### El mapa de temperatura")
        _chart(
            fig,
            "tp_map",
            "En el plano de los dos primeros factores, con el resto en el punto pedido; el punto "
            "naranja es el pedido.",
        )
    _render_steps(product_steps(r, system))
    _render_export(tr.product_to_dict(r, system), "producto_de_soluciones", "tp_dl")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) "
            "todavía no tiene un capítulo de transferencia de calor: las fórmulas siguen a "
            "**Çengel y Ghajar (2015)**, cap. 4, y a **Incropera et al. (2007)**, cap. 5. Las "
            "soluciones en serie son las de Heisler (1947) y las del semi-infinito, las de "
            "Carslaw y Jaeger (1959)."
        )
        st.markdown(
            "**Capacidad concentrada** (§4-1): con Bi < 0,1 la temperatura del cuerpo es "
            "uniforme y el balance de energía da una exponencial:"
        )
        st.latex(r"\mathrm{Bi} = \frac{h\,L_c}{k} \qquad L_c = \frac{V}{A}")
        st.latex(
            r"\begin{aligned}\frac{T - T_\infty}{T_i - T_\infty} &= e^{-b\,t} \\ "
            r"b &= \frac{h\,A}{\rho\,V\,c}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}Q &= m\,c\,(T_i - T) \\ "
            r"Q_{\text{máx}} &= m\,c\,(T_i - T_\infty)\end{aligned}"
        )
        st.markdown(
            "**Pared, cilindro largo y esfera** (§4-2): con θ = (T − T∞)/(T_i − T∞), "
            "ξ = x/L (o r/r₀), Bi = h·L/k y el número de Fourier τ = α·t/L², la solución es una "
            "serie; con τ > 0,2 alcanza con el primer término:"
        )
        st.latex(r"\theta = \sum_{n=1}^{\infty} A_n\,e^{-\lambda_n^2\,\tau}\,X(\lambda_n\,\xi)")
        st.latex(
            r"\theta_0 = A_1\,e^{-\lambda_1^2\,\tau} \qquad \theta = \theta_0\,X(\lambda_1\,\xi)"
        )
        st.markdown("Las raíces λ_n y los coeficientes A_n (la tabla 4-1 del libro):")
        st.latex(
            r"\begin{aligned}\text{pared: } &\lambda \tan\lambda = \mathrm{Bi} \\ "
            r"&A = \frac{4\sin\lambda}{2\lambda + \sin 2\lambda} \\ "
            r"&X = \cos(\lambda\,\xi)\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}\text{cilindro: } &\lambda\,\frac{J_1(\lambda)}{J_0(\lambda)} = "
            r"\mathrm{Bi} \\ "
            r"&A = \frac{2}{\lambda}\,\frac{J_1(\lambda)}{J_0^2(\lambda) + J_1^2(\lambda)} \\ "
            r"&X = J_0(\lambda\,\xi)\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}\text{esfera: } &1 - \lambda\cot\lambda = \mathrm{Bi} \\ "
            r"&A = \frac{4\,(\sin\lambda - \lambda\cos\lambda)}{2\lambda - \sin 2\lambda} \\ "
            r"&X = \frac{\sin(\lambda\,\xi)}{\lambda\,\xi}\end{aligned}"
        )
        st.markdown("El calor, como fracción del máximo (con θ₀ de un término):")
        st.latex(
            r"\begin{aligned}&\text{pared:} \\ &\frac{Q}{Q_{\text{máx}}} = "
            r"1 - \theta_0\,\frac{\sin\lambda_1}{\lambda_1} \\ "
            r"&\text{cilindro:} \\ &\frac{Q}{Q_{\text{máx}}} = "
            r"1 - 2\,\theta_0\,\frac{J_1(\lambda_1)}{\lambda_1} \\ "
            r"&\text{esfera:} \\ &\frac{Q}{Q_{\text{máx}}} = "
            r"1 - 3\,\theta_0\,\frac{\sin\lambda_1 - \lambda_1\cos\lambda_1}{\lambda_1^3}"
            r"\end{aligned}"
        )
        st.markdown(
            "**Sólido semi-infinito** (§4-3), con η = x/(2·√(α·t)): la superficie a T_s, un "
            "flujo constante y la convección:"
        )
        st.latex(r"\frac{T - T_i}{T_s - T_i} = \operatorname{erfc}(\eta)")
        st.latex(
            r"\begin{aligned}T - T_i &= \frac{2\,\dot{q}_s}{k}\sqrt{\frac{\alpha\,t}{\pi}}\,"
            r"e^{-\eta^2} \\ &\quad - \frac{\dot{q}_s\,x}{k}\,\operatorname{erfc}(\eta)"
            r"\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}&\frac{T - T_i}{T_\infty - T_i} = \operatorname{erfc}(\eta) \\ "
            r"&\quad - e^{\frac{h\,x}{k} + \frac{h^2 \alpha t}{k^2}}\,"
            r"\operatorname{erfc}\left(\eta + \frac{h\sqrt{\alpha t}}{k}\right)\end{aligned}"
        )
        st.markdown(
            "Dos sólidos que se tocan: la superficie de contacto queda a una temperatura "
            "constante, pesada con √(k·ρ·c):"
        )
        st.latex(
            r"T_s = \frac{\sqrt{(k\rho c)_A}\,T_A + \sqrt{(k\rho c)_B}\,T_B}"
            r"{\sqrt{(k\rho c)_A} + \sqrt{(k\rho c)_B}}"
        )
        st.markdown(
            "**Varias dimensiones** (§4-4): un sólido que es la intersección de otros tiene el "
            "producto de sus θ, y el calor se combina con la regla de Langston (1982):"
        )
        st.latex(r"\theta_{\text{cil. corto}} = \theta_{\text{pared}}\,\theta_{\text{cil}}")
        st.latex(
            r"\begin{aligned}\frac{Q}{Q_{\text{máx}}} &= q_1 + q_2\,(1 - q_1) \\ "
            r"&\quad + q_3\,(1 - q_1)(1 - q_2)\end{aligned}"
        )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Conducción transitoria", page_icon="⏱️", layout="centered")

st.subheader(SUBJECT)
st.title("⏱️ Conducción transitoria")
st.markdown(
    "Cómo cambia la temperatura de un sólido en el tiempo: la **capacidad concentrada** "
    "(Bi < 0,1), la **pared, el cilindro y la esfera** con la solución de un término y la serie "
    "completa, el **sólido semi-infinito** (y el contacto de dos sólidos) y los cuerpos de "
    "**varias dimensiones** como producto de soluciones."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Conducción transitoria")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="cd_mode")
if mode == _M_LUMP:
    _lumped(system)
elif mode == _M_1D:
    _one_dim(system)
elif mode == _M_SEMI:
    what = st.radio("¿Qué caso?", _SEMI_WHAT, key="ti_what", horizontal=True)
    if what == _W_SURF:
        _semi(system)
    else:
        _contact(system)
else:
    _product(system)
