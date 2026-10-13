"""Página 22 — Ebullición y condensación.

Fase 11.2. Dos modos:

- **Ebullición en pileta** (Çengel y Ghajar, §10-1 y §10-2; Incropera, §10.3 y §10.4): la
  curva de ebullición con la convección natural, la nucleada (Rohsenow), el flujo crítico
  (Zuber con Lienhard y Dhir), la transición, el flujo mínimo y la película (Bromley y
  Berenson) con la radiación; con la temperatura de la superficie, el flujo o el flujo
  crítico como dato.
- **Condensación en película** (§10-4 a §10-7): la pared vertical con los tres regímenes de
  la película, la placa inclinada, el tubo horizontal, el banco de tubos, la esfera y el
  interior de un tubo (Chato), con el vapor sobrecalentado y las gotas como nota.

El cálculo vive en :mod:`core.heat_transfer.boiling` y
:mod:`core.heat_transfer.condensation`; los gráficos, en :mod:`ui.phase_change_charts`.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.heat_transfer import boiling as bo
from core.heat_transfer import condensation as co
from core.heat_transfer.phase_change import (
    PHASE_CHANGE_FLUIDS,
    pressure_limits,
    saturation_pressure,
    saturation_state,
)
from core.heat_transfer.phase_change_procedure import boiling_steps, condensation_steps
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.phase_change_charts import (
    boiling_curve_figure,
    comparison_figure,
    condensation_curve_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.31.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_BOIL = "Ebullición en pileta"
_M_COND = "Condensación en película"
_MODES = (_M_BOIL, _M_COND)

_BY_P = "La presión"
_BY_T = "La temperatura de saturación"
_LEVEL = (_BY_P, _BY_T)

_OTHER_SURFACE = "Otra (C_sf y n a elección)"

_T: QuantityKind = "temperature"
_DT: QuantityKind = "temperature_difference"
_P: QuantityKind = "pressure"
_D: QuantityKind = "small_length"
_L: QuantityKind = "length"
_QF: QuantityKind = "heat_flux"
_H: QuantityKind = "heat_transfer_coefficient"
_W: QuantityKind = "heat_rate"
_M: QuantityKind = "mass_flow"
_HFG: QuantityKind = "specific_enthalpy"

_ATM = 101_325.0


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


def _pct(x: float, digits: int = 0) -> str:
    return f"{100.0 * x:.{digits}f} %".replace(".", ",")


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


def _example(examples: dict[str, Any], key: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox(
        "Ejemplo precargado",
        names,
        key=key,
        help=(
            "Ejemplos de Çengel y Ghajar (cap. 10) y algunos propios. Podés cambiar cualquier "
            "dato: el resultado se actualiza solo."
        ),
    )
    st.caption(f"📘 {examples[name].description}")
    return name, names.index(name)


def _fluid_input(default: str, key: str) -> str:
    fluids = list(PHASE_CHANGE_FLUIDS)
    return st.selectbox(
        "Fluido",
        fluids,
        index=fluids.index(default),
        format_func=lambda f: PHASE_CHANGE_FLUIDS[f],
        key=key,
        help="Fluidos puros con tensión superficial y propiedades de transporte en CoolProp.",
    )


def _level_input(fluid: str, default_p: float, key: str) -> float | None:
    """La presión de saturación, dada directamente o con T_sat. ``None`` si no se puede."""
    by = st.radio("¿Qué dato de la saturación tenés?", _LEVEL, key=f"{key}_by", horizontal=True)
    p_tr, p_cr = pressure_limits(fluid)
    try:
        if by == _BY_P:
            p = number_input_si(
                label="Presión p",
                kind=_P,
                default_si=default_p,
                key=f"{key}_{fluid}_p",
                format="%.5g",
                min_value_si=0.0,
                help=(
                    f"Entre el punto triple ({_num(p_tr / 1e5, 4)} bar) y el crítico "
                    f"({_num(p_cr / 1e5, 4)} bar)."
                ),
            )
        else:
            T_sat = number_input_si(
                label="Temperatura de saturación T_sat",
                kind=_T,
                default_si=saturation_state(fluid, default_p).T_sat_K,
                key=f"{key}_{fluid}_Tsat",
                format="%.5g",
            )
            p = saturation_pressure(fluid, T_sat)
        saturation_state(fluid, p)  # valida la presión antes de seguir
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return None
    return p


def _default_pressure(fluid: str, example_fluid: str, example_p: float) -> float:
    return example_p if fluid == example_fluid else _ATM


# ---------------------------------------------------------------------
# Ebullición en pileta
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _boiling_cached(inputs: bo.PoolBoilingInputs) -> bo.PoolBoilingResult:
    return bo.solve_pool_boiling(inputs)


@st.cache_data(show_spinner=False)
def _curve_cached(r: bo.PoolBoilingResult) -> bo.BoilingCurve:
    return bo.boiling_curve(r)


def _surface_input(
    fluid: str, inp: bo.PoolBoilingInputs, key: str
) -> tuple[float, float, str | None]:
    """C_sf y n de la tabla (Vachon, Nix y Tanger) o a elección: (C_sf, n, clave)."""
    table = bo.surfaces_for(fluid)
    names = [s.name for s in table.values()] + [_OTHER_SURFACE]
    if fluid == inp.fluid and inp.surface in table:
        index = list(table).index(inp.surface)
    elif fluid == inp.fluid or not table:
        index = len(names) - 1
    else:
        index = 0
    choice = st.selectbox(
        "Superficie (C_sf y n de Rohsenow)",
        names,
        index=index,
        key=f"{key}_{fluid}_surf",
        help=(
            "Incropera et al. (2007), tabla 10.1, de Vachon, Nix y Tanger (1968). Para otros "
            "pares hay que medirlos: C_sf depende de la rugosidad y de cómo se terminó la "
            "superficie."
        ),
    )
    if choice != _OTHER_SURFACE:
        surface_key = list(table)[names.index(choice)]
        s = table[surface_key]
        st.caption(f"C_sf = {_num(s.C_sf, 3)} · n = {_num(s.n, 2)}")
        return s.C_sf, s.n, surface_key
    same = fluid == inp.fluid
    C_default = inp.C_sf if same else 0.013
    n_default = inp.n if same else (1.0 if fluid == "Water" else 1.7)
    left, right = st.columns(2)
    with left:
        C_sf = float(
            st.number_input(
                "C_sf",
                min_value=0.0,
                value=float(C_default),
                step=0.0005,
                format="%.4f",
                key=f"{key}_{fluid}_Csf",
            )
        )
    with right:
        n = float(
            st.number_input(
                "n",
                min_value=0.0,
                value=float(n_default),
                step=0.1,
                format="%.2f",
                key=f"{key}_{fluid}_n",
                help="1,0 para el agua y 1,7 para los demás líquidos (Rohsenow).",
            )
        )
    return C_sf, n, None


def _boiling(system: UnitSystem) -> None:
    _name, idx = _example(bo.BOILING_EXAMPLES, "eb_example")
    inp: bo.PoolBoilingInputs = bo.BOILING_EXAMPLES[_name].inputs
    key = f"eb_{idx}"
    fluid = _fluid_input(inp.fluid, f"{key}_fluid")
    p = _level_input(fluid, _default_pressure(fluid, inp.fluid, inp.p_Pa), key)
    if p is None:
        return
    heaters = list(bo.HEATERS)
    heater: bo.Heater = st.radio(
        "Calentador",
        heaters,
        index=heaters.index(inp.heater),
        format_func=lambda h: bo.HEATERS[h],
        key=f"{key}_heater",
    )
    left, right = st.columns(2)
    with left:
        D = number_input_si(
            label="Diámetro D",
            kind=_D,
            default_si=inp.D_m,
            key=f"{key}_D",
            format="%.5g",
            min_value_si=0.0,
        )
    L = 1.0
    if heater == "cylinder":
        with right:
            L = number_input_si(
                label="Largo L",
                kind=_L,
                default_si=inp.L_m,
                key=f"{key}_L",
                format="%.5g",
                min_value_si=0.0,
            )
    C_sf, n, surface = _surface_input(fluid, inp, key)
    emissivity = float(
        st.number_input(
            "Emisividad ε de la superficie (para la película)",
            min_value=0.0,
            max_value=1.0,
            value=float(inp.emissivity),
            step=0.05,
            format="%.2f",
            key=f"{key}_eps",
            help="La radiación cuenta solo en la ebullición en película.",
        )
    )
    modes = list(bo.BOILING_MODES)
    mode: bo.BoilingMode = st.radio(
        "¿Qué dato tenés?",
        modes,
        index=modes.index(inp.mode),
        format_func=lambda m: bo.BOILING_MODES[m],
        key=f"{key}_mode",
    )
    T_s = q = None
    if mode == "temperature":
        T_sat = saturation_state(fluid, p).T_sat_K
        default_T = inp.T_s_K if (fluid == inp.fluid and inp.T_s_K) else T_sat + 10.0
        T_s = number_input_si(
            label="Temperatura de la superficie T_s",
            kind=_T,
            default_si=default_T,
            key=f"{key}_{fluid}_Ts",
            format="%.5g",
        )
    elif mode == "flux":
        default_q = inp.q_W_per_m2 if (fluid == inp.fluid and inp.q_W_per_m2) else 5.0e4
        q = number_input_si(
            label="Flujo de calor q̇",
            kind=_QF,
            default_si=default_q,
            key=f"{key}_q",
            format="%.5g",
            min_value_si=0.0,
        )
    try:
        r = _boiling_cached(
            bo.PoolBoilingInputs(fluid, p, heater, D, C_sf, n, mode, T_s, q, L, emissivity, surface)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_boiling(r, system)


def _render_boiling(r: bo.PoolBoilingResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    st.markdown(f"**Régimen: {bo.REGIMES[r.regime]}**" + (" (burnout)" if r.burnout else ""))
    items = [
        (f"T_s [{_u(_T, system)}]", _val(r.T_s_K, _T, system)),
        (f"ΔT_e = T_s − T_sat [{_u(_DT, system)}]", _val(r.dT_K, _DT, system, 4)),
        (f"q̇ [{_u(_QF, system)}]", _val(r.q, _QF, system, 4)),
        (f"h = q̇/ΔT_e [{_u(_H, system)}]", _val(r.h, _H, system, 4)),
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system, 4)),
        (f"ṁ evaporada [{_u(_M, system)}]", _val(r.m_dot, _M, system, 4)),
        (f"q̇_máx [{_u(_QF, system)}]", _val(r.q_max, _QF, system, 4)),
        ("q̇/q̇_máx", _pct(r.fraction_of_critical)),
    ]
    _metric_rows(items)
    if i.heater == "cylinder":
        st.caption(
            f"Q̇ y ṁ con el largo del cilindro, L = {_val(i.L_m, _L, system)} {_u(_L, system)}."
        )
    _notes(r.notes)
    st.markdown("#### La curva de ebullición")
    curve = _curve_cached(r)
    _chart(
        boiling_curve_figure(
            curve,
            r,
            q_scale=convert_from_si(1.0, _QF, system),
            q_unit=_u(_QF, system),
            dT_scale=convert_from_si(1.0, _DT, system),
            dT_unit=_u(_DT, system),
        ),
        "eb_chart",
        "En gris, la convección natural; en azul, la nucleada (Rohsenow) hasta el flujo "
        "crítico; en gris claro rayado, la transición (una interpolación: no hay correlación "
        "confiable) hasta el flujo mínimo (Leidenfrost); en naranja, la película (Bromley). "
        "Con un flujo dado mayor que el crítico, la superficie salta a la película: el "
        "«burnout».",
    )
    rows = [
        (
            "Inicio de la nucleada",
            r.dT_onset,
            bo.q_rohsenow(
                r.sat.mu_l,
                r.sat.h_fg,
                r.sat.rho_l,
                r.sat.rho_v,
                r.sat.sigma,
                r.sat.cp_l,
                r.sat.Pr_l,
                i.C_sf,
                i.n,
                r.dT_onset,
            ),
        ),
        ("Flujo crítico", r.dT_max, r.q_max),
    ]
    if r.minimum_case == "crossing":
        rows.append(("Flujo mínimo (Leidenfrost)", r.dT_min, r.q_min))
    rows.append(("Tus datos", r.dT_K, r.q))
    _table(
        pd.DataFrame(
            {
                "Punto": [p for p, _, _ in rows],
                f"ΔT_e [{_u(_DT, system)}]": [_val(dT, _DT, system, 4) for _, dT, _ in rows],
                f"q̇ [{_u(_QF, system)}]": [_val(q, _QF, system, 4) for _, _, q in rows],
            }
        )
    )
    s = r.sat
    st.markdown("#### Propiedades a la saturación")
    props = [
        ("T_sat", _val(s.T_sat_K, _T, system), _u(_T, system)),
        ("ρ_ℓ", _val(s.rho_l, "density", system), _u("density", system)),
        ("ρ_v", _val(s.rho_v, "density", system), _u("density", system)),
        ("h_fg", _val(s.h_fg, _HFG, system), _u(_HFG, system)),
        ("σ", _val(s.sigma, "surface_tension", system), _u("surface_tension", system)),
        ("μ_ℓ", _val(s.mu_l, "dynamic_viscosity", system), _u("dynamic_viscosity", system)),
        ("c_pℓ", _val(s.cp_l, "specific_heat", system), _u("specific_heat", system)),
        ("k_ℓ", _val(s.k_l, "thermal_conductivity", system), _u("thermal_conductivity", system)),
        ("Pr_ℓ", _num(s.Pr_l, 4), "—"),
        ("L*", _num(r.L_star, 4), "—"),
        ("C_cr", _num(r.C_cr, 3), "—"),
    ]
    _table(pd.DataFrame(props, columns=["Símbolo", "Valor", "Unidad"]))
    _render_steps(boiling_steps(r, system))
    _render_export(bo.pool_boiling_to_dict(r, system), "ebullicion_en_pileta", "eb_dl")


# ---------------------------------------------------------------------
# Condensación en película
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _condensation_cached(inputs: co.CondensationInputs) -> co.CondensationResult:
    return co.solve_condensation(inputs)


@st.cache_data(show_spinner=False)
def _cond_extras_cached(
    r: co.CondensationResult,
) -> tuple[co.CondensationCurve, tuple[co.ComparisonRow, ...]]:
    return co.condensation_curve(r), co.geometry_comparison(r)


_L_LABELS: dict[co.Geometry, str] = {
    "vertical_plate": "Alto L",
    "vertical_tube": "Alto del tubo L",
    "inclined_plate": "Largo L (a lo largo de la pendiente)",
    "horizontal_tube": "Largo del tubo L",
    "tube_bank": "Largo de los tubos L",
    "inside_tube": "Largo del tubo L",
}


def _condensation(system: UnitSystem) -> None:
    _name, idx = _example(co.CONDENSATION_EXAMPLES, "ec_example")
    inp: co.CondensationInputs = co.CONDENSATION_EXAMPLES[_name].inputs
    key = f"ec_{idx}"
    fluid = _fluid_input(inp.fluid, f"{key}_fluid")
    p = _level_input(fluid, _default_pressure(fluid, inp.fluid, inp.p_Pa), key)
    if p is None:
        return
    geometries = list(co.GEOMETRIES)
    geometry: co.Geometry = st.selectbox(
        "Geometría",
        geometries,
        index=geometries.index(inp.geometry),
        format_func=lambda g: co.GEOMETRIES[g],
        key=f"{key}_geo",
    )
    plate = geometry in ("vertical_plate", "inclined_plate")
    cols = st.columns(2)
    L = W = 1.0
    D = inp.D_m
    if geometry != "sphere":
        with cols[0]:
            L = number_input_si(
                label=_L_LABELS[geometry],
                kind=_L,
                default_si=inp.L_m,
                key=f"{key}_L",
                format="%.5g",
                min_value_si=0.0,
            )
    with cols[1]:
        if plate:
            W = number_input_si(
                label="Ancho W",
                kind=_L,
                default_si=inp.W_m,
                key=f"{key}_W",
                format="%.5g",
                min_value_si=0.0,
            )
        else:
            D = number_input_si(
                label="Diámetro D",
                kind=_D,
                default_si=inp.D_m,
                key=f"{key}_D",
                format="%.5g",
                min_value_si=0.0,
            )
    angle, rows, columns, V_v = inp.angle_deg, inp.rows, inp.columns, inp.V_v_m_s
    if geometry == "inclined_plate":
        angle = float(
            st.number_input(
                "Inclinación θ respecto de la vertical [°]",
                min_value=0.0,
                max_value=89.0,
                value=float(inp.angle_deg or 30.0),
                step=5.0,
                format="%.4g",
                key=f"{key}_angle",
            )
        )
    elif geometry == "tube_bank":
        left, right = st.columns(2)
        with left:
            rows = int(
                st.number_input(
                    "Tubos por columna N (de alto)",
                    min_value=1,
                    max_value=50,
                    value=max(int(inp.rows), 1) if inp.geometry == "tube_bank" else 3,
                    step=1,
                    key=f"{key}_rows",
                )
            )
        with right:
            columns = int(
                st.number_input(
                    "Columnas (de ancho)",
                    min_value=1,
                    max_value=200,
                    value=max(int(inp.columns), 1) if inp.geometry == "tube_bank" else 4,
                    step=1,
                    key=f"{key}_cols",
                )
            )
    elif geometry == "inside_tube":
        if st.checkbox(
            "Conozco la velocidad del vapor a la entrada (para el rango de Chato)",
            value=inp.V_v_m_s is not None,
            key=f"{key}_hasV",
        ):
            V_v = number_input_si(
                label="Velocidad del vapor V_v",
                kind="speed",
                default_si=inp.V_v_m_s or 1.0,
                key=f"{key}_V",
                format="%.4g",
                min_value_si=0.0,
            )
        else:
            V_v = None
    T_sat = saturation_state(fluid, p).T_sat_K
    default_Ts = inp.T_s_K if fluid == inp.fluid else T_sat - 10.0
    T_s = number_input_si(
        label="Temperatura de la pared T_s",
        kind=_T,
        default_si=default_Ts,
        key=f"{key}_{fluid}_Ts",
        format="%.5g",
    )
    T_v = None
    if st.checkbox("Vapor sobrecalentado", value=inp.T_v_K is not None, key=f"{key}_{fluid}_sh"):
        T_v = number_input_si(
            label="Temperatura del vapor T_v",
            kind=_T,
            default_si=inp.T_v_K if (fluid == inp.fluid and inp.T_v_K) else T_sat + 20.0,
            key=f"{key}_{fluid}_Tv",
            format="%.5g",
        )
    try:
        r = _condensation_cached(
            co.CondensationInputs(fluid, p, geometry, T_s, L, W, D, angle, rows, columns, T_v, V_v)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_condensation(r, system)


def _render_condensation(r: co.CondensationResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    tall = r.Re is not None
    st.markdown(f"**Película {co.FILM_REGIMES[r.regime]}**")
    items = [
        (f"h [{_u(_H, system)}]", _val(r.h, _H, system, 4)),
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system, 4)),
        (f"ṁ condensada [{_u(_M, system)}]", _val(r.m_dot, _M, system, 4)),
        (f"q̇ = h·ΔT [{_u(_QF, system)}]", _val(r.q, _QF, system, 4)),
        (f"h*_fg [{_u(_HFG, system)}]", _val(r.h_fg_star, _HFG, system)),
        (f"ΔT = T_sat − T_s [{_u(_DT, system)}]", _val(r.dT_K, _DT, system, 4)),
    ]
    if tall:
        items.append(("Re de la película", _num(r.Re, 4)))
        items.append((f"Espesor δ al pie [{_u(_D, system)}]", _val(r.delta_m, _D, system, 3)))
    if r.h_single is not None:
        items.append((f"h de un tubo h₁ [{_u(_H, system)}]", _val(r.h_single, _H, system, 4)))
    if r.Re_vapor is not None:
        items.append(("Re del vapor", _num(r.Re_vapor, 4)))
    if r.h_dropwise is not None:
        items.append(
            (f"h en gotas (Griffith) [{_u(_H, system)}]", _val(r.h_dropwise, _H, system, 4))
        )
    _metric_rows(items)
    if i.geometry in ("horizontal_tube", "inside_tube", "vertical_tube", "tube_bank"):
        st.caption(
            f"Q̇ y ṁ con {'cada tubo' if i.geometry != 'tube_bank' else f'los {r.tubes} tubos'} "
            f"de L = {_val(i.L_m, _L, system)} {_u(_L, system)}."
        )
    _notes(r.notes)
    curve, comparison = _cond_extras_cached(r)
    st.markdown("#### El coeficiente según el tamaño")
    _chart(
        condensation_curve_figure(
            curve,
            r,
            h_scale=convert_from_si(1.0, _H, system),
            h_unit=_u(_H, system),
            x_scale=convert_from_si(1.0, _L if tall else _D, system),
            x_unit=_u(_L if tall else _D, system),
        ),
        "ec_curve",
        "En una pared alta la película engorda hacia abajo y h baja, hasta que las ondas y "
        "después la turbulencia lo vuelven a subir; abajo, el Re de la película con los "
        "límites de 30 y 1800."
        if tall
        else "En un tubo o una esfera, h baja como D^−¼: los tubos finos condensan mejor.",
    )
    st.markdown("#### La misma pared en otras geometrías")
    _chart(
        comparison_figure(
            comparison, h_scale=convert_from_si(1.0, _H, system), h_unit=_u(_H, system)
        ),
        "ec_comparison",
        "Las mismas temperaturas y medidas en las otras geometrías de la familia. Con película "
        "laminar, h_vertical/h_horizontal = 1,29·(D/L)^¼: un tubo acostado condensa mejor si "
        "L > 2,77·D.",
    )
    _table(
        pd.DataFrame(
            {
                "Geometría": [row.name for row in comparison],
                f"h [{_u(_H, system)}]": [_val(row.h, _H, system, 4) for row in comparison],
                f"q̇ [{_u(_QF, system)}]": [_val(row.q, _QF, system, 4) for row in comparison],
                "Película": [co.FILM_REGIMES[row.regime] for row in comparison],
            }
        )
    )
    s, liq = r.sat, r.liquid
    st.markdown("#### Propiedades")
    props = [
        ("T_sat", _val(s.T_sat_K, _T, system), _u(_T, system)),
        ("h_fg", _val(s.h_fg, _HFG, system), _u(_HFG, system)),
        ("ρ_v", _val(s.rho_v, "density", system), _u("density", system)),
        ("T_f (película)", _val(liq.T_K, _T, system), _u(_T, system)),
        ("ρ_ℓ", _val(liq.rho, "density", system), _u("density", system)),
        ("μ_ℓ", _val(liq.mu, "dynamic_viscosity", system), _u("dynamic_viscosity", system)),
        ("c_pℓ", _val(liq.cp, "specific_heat", system), _u("specific_heat", system)),
        ("k_ℓ", _val(liq.k, "thermal_conductivity", system), _u("thermal_conductivity", system)),
        ("Pr_ℓ", _num(liq.Pr, 4), "—"),
    ]
    _table(pd.DataFrame(props, columns=["Símbolo", "Valor", "Unidad"]))
    _render_steps(condensation_steps(r, system))
    _render_export(co.condensation_to_dict(r, system), "condensacion_en_pelicula", "ec_dl")


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) "
            "todavía no tiene un capítulo de transferencia de calor: las fórmulas siguen a "
            "**Çengel y Ghajar (2015)**, cap. 10, y a **Incropera et al. (2007)**, cap. 10, con "
            "las propiedades de CoolProp."
        )
        st.markdown(
            "**Ebullición en pileta** (§10-2). La fuerza impulsora es el exceso de temperatura "
            "ΔT_e = T_s − T_sat. En la **nucleada**, Rohsenow (1952), con C_sf y n del par "
            "fluido–superficie (Vachon, Nix y Tanger, 1968):"
        )
        st.latex(
            r"\begin{aligned}\dot q &= \mu_\ell\,h_{fg}"
            r"\left[\frac{g\,(\rho_\ell - \rho_v)}{\sigma}\right]^{1/2}\psi^3 \\ "
            r"\psi &= \frac{c_{p\ell}\,\Delta T_e}{C_{sf}\,h_{fg}\,\mathrm{Pr}_\ell^{\,n}}"
            r"\end{aligned}"
        )
        st.markdown(
            "El **flujo crítico** (Zuber, 1958), con C_cr de Lienhard y Dhir (1973) según la "
            "geometría y su tamaño adimensional L* (0,149 en una placa grande, 0,12 en un "
            "cilindro grande, 0,11 en una esfera grande):"
        )
        st.latex(
            r"\begin{aligned}\dot q_{\text{máx}} &= C_{cr}\,h_{fg}"
            r"\left[\sigma\,g\,\rho_v^2\,(\rho_\ell - \rho_v)\right]^{1/4} \\ "
            r"L^* &= L\left[\frac{g\,(\rho_\ell - \rho_v)}{\sigma}\right]^{1/2}\end{aligned}"
        )
        st.markdown(
            "El **flujo mínimo** (el punto de Leidenfrost; Zuber, con el 0,09 de Berenson):"
        )
        st.latex(
            r"\dot q_{\text{mín}} = 0.09\,\rho_v\,h_{fg}"
            r"\left[\frac{\sigma\,g\,(\rho_\ell - \rho_v)}{(\rho_\ell + \rho_v)^2}\right]^{1/4}"
        )
        st.markdown(
            "La **película** (Bromley, 1950; C = 0,62 en un cilindro y 0,67 en una esfera; en "
            "una placa, Berenson con 0,425 y la longitud capilar), con el vapor a "
            "T_f = (T_s + T_sat)/2 y la radiación:"
        )
        st.latex(
            r"\begin{aligned}\dot q_{\text{pel}} &= C\left[\frac{g\,k_v^3\,\rho_v\,"
            r"(\rho_\ell - \rho_v)\,h'_{fg}}{\mu_v\,D\,\Delta T_e}\right]^{1/4}\Delta T_e \\ "
            r"h'_{fg} &= h_{fg} + 0.4\,c_{pv}\,\Delta T_e\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}\dot q &= \dot q_{\text{pel}} + \tfrac{3}{4}\,\dot q_{\text{rad}} \\ "
            r"\dot q_{\text{rad}} &= \varepsilon\,\sigma_{\text{SB}}\,"
            r"(T_s^4 - T_{\mathrm{sat}}^4)\end{aligned}"
        )
        st.markdown(
            "La **transición** no tiene una correlación confiable: la app interpola en log–log "
            "entre el flujo crítico y el mínimo."
        )
        st.markdown(
            "**Condensación en película** (§10-5). El calor latente corregido suma el "
            "subenfriamiento del condensado (y el sobrecalentamiento del vapor), y el número de "
            "Reynolds de la película dice el régimen:"
        )
        st.latex(
            r"\begin{aligned}h^*_{fg} &= h_{fg} + 0.68\,c_{p\ell}\,(T_{\mathrm{sat}} - T_s) \\ "
            r"\mathrm{Re} &= \frac{4\,\dot m}{p\,\mu_\ell} = "
            r"\frac{4\,L\,h\,(T_{\mathrm{sat}} - T_s)}{\mu_\ell\,h^*_{fg}}\end{aligned}"
        )
        st.markdown("Placa vertical: laminar (Re < 30, Nusselt, 1916):")
        st.latex(
            r"h = 0.943\left[\frac{g\,\rho_\ell\,(\rho_\ell - \rho_v)\,h^*_{fg}\,k_\ell^3}"
            r"{\mu_\ell\,(T_{\mathrm{sat}} - T_s)\,L}\right]^{1/4}"
        )
        st.markdown("Con ondas (30 < Re < 1800, Kutateladze, 1963) y turbulenta (Labuntsov, 1957):")
        st.latex(
            r"h = \frac{\mathrm{Re}\,k_\ell}{1.08\,\mathrm{Re}^{1.22} - 5.2}"
            r"\left(\frac{g}{\nu_\ell^2}\right)^{1/3}"
        )
        st.latex(
            r"h = \frac{\mathrm{Re}\,k_\ell\,(g/\nu_\ell^2)^{1/3}}"
            r"{8750 + 58\,\mathrm{Pr}^{-0.5}\,(\mathrm{Re}^{0.75} - 253)}"
        )
        st.markdown(
            "Placa inclinada θ de la vertical: h_θ = h·(cos θ)^¼. Tubo horizontal y esfera "
            "(siempre laminares) y un banco de N tubos por columna:"
        )
        st.latex(
            r"\begin{aligned}h &= 0.729\left[\frac{g\,\rho_\ell\,(\rho_\ell - \rho_v)\,"
            r"h^*_{fg}\,k_\ell^3}{\mu_\ell\,(T_{\mathrm{sat}} - T_s)\,D}\right]^{1/4} \\ "
            r"h_N &= h_1\,N^{-1/4}\end{aligned}"
        )
        st.markdown(
            "Adentro de un tubo horizontal con poco vapor (Chato, 1962; Re_v < 35 000), con su "
            "propio calor latente, y las gotas del agua (Griffith, 1983, §10-7):"
        )
        st.latex(
            r"h = 0.555\left[\frac{g\,\rho_\ell\,(\rho_\ell - \rho_v)\,k_\ell^3\,h'_{fg}}"
            r"{\mu_\ell\,(T_{\mathrm{sat}} - T_s)\,D}\right]^{1/4}"
        )
        st.latex(
            r"\begin{aligned}h'_{fg} &= h_{fg} + \tfrac{3}{8}\,c_{p\ell}\,(T_{\mathrm{sat}} - T_s)"
            r" \\ h_{\text{gotas}} &= 51\,104 + 2044\,T_{\mathrm{sat}}\end{aligned}"
        )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Ebullición y condensación", page_icon="🫖", layout="centered")

st.subheader(SUBJECT)
st.title("🫖 Ebullición y condensación")
st.markdown(
    "El calor con cambio de fase: la **ebullición en pileta** sobre un calentador sumergido "
    "(la curva de ebullición, de la convección natural a la película, con el flujo crítico) y "
    "la **condensación en película** sobre una pared, un tubo, un banco o adentro de un tubo."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Ebullición y condensación")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés calcular?", _MODES, key="bc_mode", horizontal=True)
if mode == _M_BOIL:
    _boiling(system)
else:
    _condensation(system)
