"""Página 23 — Diseño de intercambiadores.

Fases 12.1 y 12.2. Tres modos:

- **Banco de tubos** (Çengel y Ghajar, §7-4; Incropera, §7.6): Zukauskas con la corrección por
  filas, V_máx por el plano transversal o por la diagonal, la salida con las propiedades a la
  temperatura media y Δp con f y χ de los gráficos de Zukauskas (la digitalización de ht).
- **Doble tubo** (Incropera, §8.7 y §11.3; Çengel y Ghajar, §11-2; Kern, cap. 6): el h de cada
  lado (las correlaciones de la 8.1 y la tabla del anular de Kays y Perkins), el U armado con
  el ensuciamiento, el largo y las horquillas (o las salidas de un equipo dado) y Δp.
- **Casco y tubos** (Kern, 1950, cap. 7; Kakaç y Liu, 2002, cap. 8): el lado del casco con
  Kern o con Bell–Delaware (Fase 12.2; Taborek, 1983; Thome, 2004; Serth, 2014: el corte, las
  holguras, el bypass y las tiras, los cinco factores y el Δp por zonas, con Kern al lado),
  el de los tubos con la 8.1, U_C y U_D, el ε-NTU de un paso por el casco, el criterio de R_d
  y Δp de los dos lados.

El cálculo vive en :mod:`core.heat_transfer.tube_bank`, :mod:`core.heat_transfer.double_pipe`
y :mod:`core.heat_transfer.shell_tube`; los gráficos, en :mod:`ui.exchanger_design_charts`
(el perfil de temperaturas y las resistencias, en :mod:`ui.exchanger_charts`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from core.export import dict_to_csv
from core.fluids import FLUID_NAMES_ES
from core.heat_transfer import double_pipe as dp
from core.heat_transfer import shell_tube as stt
from core.heat_transfer import tube_bank as tb
from core.heat_transfer.bell_delaware import (
    BAFFLE_CUT_RANGE,
    BUNDLE_CLEARANCE_HINTS,
    BellDelawareGeometry,
    tema_shell_baffle_clearance,
    tema_tube_hole_clearance,
)
from core.heat_transfer.design_common import (
    TUBE_SIDE_CORRELATIONS,
    DesignStream,
    FluidProperties,
    StreamProps,
    TubeFlow,
    stream_name,
    suggested_pressure,
)
from core.heat_transfer.exchanger_design_procedure import (
    double_pipe_steps,
    shell_tube_steps,
    tube_bank_steps,
)
from core.heat_transfer.exchangers import FOULING_FACTORS, ExchangerResult, OverallUResult
from core.state_report import ProcedureStep, format_value
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.exchanger_charts import profile_figure, u_resistance_figure
from ui.exchanger_design_charts import (
    baffle_sweep_figure,
    dp_zones_figure,
    friction_chart_figure,
    h_cascade_figure,
    methods_sweep_figure,
    velocity_sweep_figure,
)
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.33.0"

# Opciones fijas de los radios: cambiarlas reiniciaría el widget.
_M_BANK = "Banco de tubos"
_M_DP = "Doble tubo"
_M_ST = "Casco y tubos"
_MODES = (_M_BANK, _M_DP, _M_ST)

_GIVEN = "given"
_OTHER = "Otro valor"
_BWG_OTHER = 0
_NT_ESTIMATE = "Estimada con el diámetro del casco"
_NT_GIVEN = "La cuento (dato)"
_NT_MODES = (_NT_ESTIMATE, _NT_GIVEN)
_CLR_TEMA = "Las de TEMA"
_CLR_GIVEN = "A elección"
_CLR_MODES = (_CLR_TEMA, _CLR_GIVEN)

_TARGETS: dict[str, str] = {
    "T_hot_out": "La T de salida del caliente",
    "T_cold_out": "La T de salida del frío",
    "Q": "El calor Q̇",
}
_SIDES: dict[str, str] = {"hot": "La caliente", "cold": "La fría"}

_T: QuantityKind = "temperature"
_DT: QuantityKind = "temperature_difference"
_P: QuantityKind = "pressure"
_D: QuantityKind = "small_length"
_L: QuantityKind = "length"
_A: QuantityKind = "area"
_V: QuantityKind = "speed"
_H: QuantityKind = "heat_transfer_coefficient"
_W: QuantityKind = "heat_rate"
_WL: QuantityKind = "linear_heat_rate"
_M: QuantityKind = "mass_flow"
_G: QuantityKind = "mass_flux"
_PD: QuantityKind = "pressure_drop"
_PL: QuantityKind = "pressure_loss"
_RF: QuantityKind = "fouling_resistance"
_K: QuantityKind = "thermal_conductivity"
_MU: QuantityKind = "dynamic_viscosity"
_CP: QuantityKind = "specific_heat"
_RHO: QuantityKind = "density"
_R: QuantityKind = "thermal_resistance"


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


def _signed(x: float) -> str:
    """Una diferencia relativa con su signo: «+12 %», «−3,4 %»."""
    v = 100.0 * abs(x)
    text = f"{v:.0f}" if v >= 10.0 else f"{v:.1f}".replace(".", ",")
    return f"{'+' if x >= 0 else '−'}{text} %"


def _t_scale(system: UnitSystem) -> tuple[float, float]:
    """(escala, offset) para pasar de K a la temperatura del sistema."""
    offset = convert_from_si(0.0, _T, system)
    return convert_from_si(1.0, _T, system) - offset, offset


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


def _messages(warnings: tuple[str, ...], notes: list[str]) -> None:
    """Los avisos (lo que conviene revisar) en amarillo y las notas en azul."""
    for w in warnings:
        st.warning(w)
    for note in notes:
        if note not in warnings:
            st.info(note)


def _chart(fig: Any, key: str, caption: str = "") -> None:
    """Un gráfico que falla no tumba la página."""
    try:
        if fig is None:
            return
        st.plotly_chart(fig, width="stretch", key=key)
        if caption:
            st.caption(caption)
    except Exception as exc:  # noqa: BLE001 — se avisa y la página sigue
        st.warning(f"No se pudo dibujar el gráfico: {exc}")


def _example(examples: dict[str, Any], key: str, books: str) -> tuple[str, int]:
    names = list(examples)
    name = st.selectbox(
        "Ejemplo precargado",
        names,
        key=key,
        help=(
            f"Ejemplos de {books} y algunos propios. Podés cambiar cualquier dato: el resultado "
            "se actualiza solo."
        ),
    )
    st.caption(f"📘 {examples[name].description}")
    return name, names.index(name)


def _fluid_name(f: str) -> str:
    return FLUID_NAMES_ES.get(f, f)


def _int_input(label: str, default: int, key: str, help_text: str | None = None) -> int:
    return int(
        st.number_input(label, min_value=1, value=int(default), step=1, key=key, help=help_text)
    )


def _fouling_input(label: str, default: float, key: str) -> float:
    options = [*FOULING_FACTORS, _OTHER]
    current = next((k for k, v in FOULING_FACTORS.items() if abs(v - default) < 1e-12), _OTHER)
    choice = st.selectbox(
        label,
        options,
        index=options.index(current),
        key=f"{key}_sel",
        help="Valores representativos de TEMA (Incropera, tabla 11.1; Çengel y Ghajar, tabla "
        "11-2).",
    )
    if choice != _OTHER:
        return FOULING_FACTORS[choice]
    return number_input_si(
        label="R''_f",
        kind=_RF,
        default_si=default or 0.0002,
        key=f"{key}_v",
        format="%.4g",
        min_value_si=0.0,
    )


def _props_table(
    rows: list[tuple[str, float | None, QuantityKind | None]], system: UnitSystem
) -> None:
    _table(
        pd.DataFrame(
            [
                (
                    sym,
                    _num(value, 4) if kind is None else _val(value, kind, system),
                    "—" if kind is None else _u(kind, system),
                )
                for sym, value, kind in rows
            ],
            columns=["Símbolo", "Valor", "Unidad"],
        )
    )


def _stream_props_rows(
    name: str, props: StreamProps, system: UnitSystem
) -> list[tuple[str, float | None, QuantityKind | None]]:
    return [
        (f"T_m ({name})", props.T_K, _T),
        (f"ρ ({name})", props.rho_kg_per_m3, _RHO),
        (f"c_p ({name})", props.cp_J_per_kgK, _CP),
        (f"μ ({name})", props.mu_Pa_s, _MU),
        (f"k ({name})", props.k_W_per_mK, _K),
        (f"Pr ({name})", props.Pr, None),
    ]


def _resistances(u: OverallUResult, system: UnitSystem, key: str) -> None:
    """El peso de cada resistencia del U por metro: el lado que manda."""
    st.markdown("#### ¿Qué lado manda?")
    _table(
        pd.DataFrame(
            [
                {
                    "Resistencia (por metro)": res.label,
                    f"R [{_u(_R, system)}]": _val(res.R_K_per_W, _R, system, 4),
                    "Del total": _pct(u.share(res)),
                }
                for res in u.resistances
            ]
        )
    )
    _chart(
        u_resistance_figure(u),
        key,
        "Las resistencias van en serie: la mayor fija el U. Mejorar el otro lado casi no cambia "
        "nada.",
    )


def _profile(ex: ExchangerResult, system: UnitSystem, key: str) -> None:
    t_scale, t_offset = _t_scale(system)
    try:
        fig = profile_figure(ex, t_scale=t_scale, t_offset=t_offset, t_unit=_u(_T, system))
    except Exception as exc:  # noqa: BLE001 — un gráfico que falla no tumba la página
        st.warning(f"No se pudo dibujar el perfil: {exc}")
        return
    if fig is None:
        return
    st.markdown("#### Las temperaturas a lo largo del equipo")
    _chart(
        fig,
        key,
        "En contracorriente la diferencia de temperaturas es más pareja (y el frío puede salir más "
        "caliente que el caliente); en paralelo es grande a la entrada y se achica.",
    )


# ---------------------------------------------------------------------
# Corrientes (doble tubo y casco y tubos)
# ---------------------------------------------------------------------


def _stream_inputs(
    title: str, default: DesignStream, key: str, T_range: tuple[float, float]
) -> DesignStream:
    """Una corriente: un fluido de CoolProp o propiedades a elección, T de entrada, caudal,
    presión y ensuciamiento."""
    st.markdown(f"**{title}**")
    options = [_GIVEN, *tb.tube_bank_fluids()]
    current = _GIVEN if default.properties is not None else (default.fluid or "Water")
    fluid = st.selectbox(
        "Fluido",
        options,
        index=options.index(current) if current in options else 1,
        format_func=lambda f: (
            "Propiedades a elección (como los libros)" if f == _GIVEN else _fluid_name(f)
        ),
        key=f"{key}_fluid",
        help="Un fluido de CoolProp (las propiedades a la temperatura media, que se itera) o "
        "propiedades constantes, como las de un enunciado (los aceites de los libros).",
    )
    left, right = st.columns(2)
    with left:
        T_in = number_input_si(
            label="T de entrada",
            kind=_T,
            default_si=default.T_in_K,
            key=f"{key}_Tin",
            format="%.5g",
        )
    with right:
        m_dot = number_input_si(
            label="Caudal ṁ",
            kind=_M,
            default_si=default.m_dot_kg_s,
            key=f"{key}_m",
            format="%.5g",
            min_value_si=0.0,
        )
    if fluid == _GIVEN:
        g = default.properties or FluidProperties(850.0, 2000.0, 0.01, 0.13, name="aceite")
        name = st.text_input("Nombre del fluido", value=g.name, key=f"{key}_name")
        c1, c2 = st.columns(2)
        with c1:
            rho = number_input_si(
                label="Densidad ρ",
                kind=_RHO,
                default_si=g.rho_kg_per_m3,
                key=f"{key}_rho",
                format="%.5g",
                min_value_si=0.0,
            )
            mu = number_input_si(
                label="Viscosidad μ",
                kind=_MU,
                default_si=g.mu_Pa_s,
                key=f"{key}_mu",
                format="%.5g",
                min_value_si=0.0,
            )
        with c2:
            cp = number_input_si(
                label="Calor específico c_p",
                kind=_CP,
                default_si=g.cp_J_per_kgK,
                key=f"{key}_cp",
                format="%.5g",
                min_value_si=0.0,
            )
            k = number_input_si(
                label="Conductividad k",
                kind=_K,
                default_si=g.k_W_per_mK,
                key=f"{key}_k",
                format="%.5g",
                min_value_si=0.0,
            )
        mu_wall = None
        if st.checkbox(
            "Conozco μ a la temperatura de la pared (para (μ/μ_p)^0,14)",
            value=g.mu_wall_Pa_s is not None,
            key=f"{key}_hasmuw",
        ):
            mu_wall = number_input_si(
                label="Viscosidad junto a la pared μ_p",
                kind=_MU,
                default_si=g.mu_wall_Pa_s or g.mu_Pa_s,
                key=f"{key}_muw",
                format="%.5g",
                min_value_si=0.0,
            )
        props: FluidProperties | None = FluidProperties(
            rho, cp, mu, k, mu_wall, name=name.strip() or "fluido a elección"
        )
        stream_fluid, p = None, default.p_Pa
    else:
        props, stream_fluid = None, fluid
        p_default = (
            default.p_Pa
            if fluid == default.fluid
            else suggested_pressure(fluid, T_range[0], T_range[1], default.p_Pa)
        )
        p = number_input_si(
            label="Presión p",
            kind=_P,
            default_si=p_default,
            key=f"{key}_{fluid}_p",
            format="%.5g",
            min_value_si=0.0,
            help="Las propiedades salen de CoolProp a esta presión y a la temperatura media de la "
            "corriente; el fluido tiene que quedar en una sola fase.",
        )
    R_f = _fouling_input("Ensuciamiento R''_f", default.R_f_m2K_per_W, f"{key}_Rf")
    return DesignStream(T_in, m_dot, stream_fluid, p, props, R_f)


def _target_inputs(
    target_default: str,
    value_default: float,
    solved: dict[str, float],
    key: str,
) -> tuple[str, float]:
    """Lo que se le pide al equipo: una T de salida o el calor."""
    names = list(_TARGETS)
    target = st.radio(
        "¿Qué le pedís al equipo?",
        names,
        index=names.index(target_default),
        format_func=lambda t: _TARGETS[t],
        key=f"{key}_target",
    )
    default = value_default if target == target_default else solved[target]
    if target == "Q":
        value = number_input_si(
            label="Calor Q̇",
            kind=_W,
            default_si=default,
            key=f"{key}_Q",
            format="%.5g",
            min_value_si=0.0,
        )
    else:
        value = number_input_si(
            label=_TARGETS[target],
            kind=_T,
            default_si=default,
            key=f"{key}_{target}",
            format="%.5g",
        )
    return target, value


def _solved_targets(T_hot_out: float, T_cold_out: float, Q: float) -> dict[str, float]:
    return {"T_hot_out": T_hot_out, "T_cold_out": T_cold_out, "Q": Q}


def _side_table(
    rows: list[tuple[str, str, TubeFlow, float, StreamProps]], system: UnitSystem
) -> None:
    """Los dos lados: velocidad, Re, régimen, correlación, Nu, h y Δp."""
    _table(
        pd.DataFrame(
            [
                {
                    "Lado": where,
                    "Fluido": name,
                    f"u [{_u(_V, system)}]": _val(flow.velocity_m_s, _V, system, 4),
                    "Re": _num(flow.Re, 4),
                    "Régimen": flow.regime,
                    "Nu": _num(flow.Nu, 4),
                    f"h [{_u(_H, system)}]": _val(flow.h_W_per_m2K, _H, system, 4),
                    f"Δp [{_u(_PL, system)}]": _val(dp_Pa, _PL, system, 4),
                }
                for where, name, flow, dp_Pa, _props in rows
            ]
        )
    )


# ---------------------------------------------------------------------
# Banco de tubos
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _bank_cached(inputs: tb.TubeBankInputs) -> tb.TubeBankResult:
    return tb.solve_tube_bank(inputs)


@st.cache_data(show_spinner=False)
def _bank_extras_cached(
    inputs: tb.TubeBankInputs,
) -> tuple[list[tb.SweepPoint], dict[str, tb.TubeBankResult]]:
    return tb.velocity_sweep(inputs), dict(tb.compare_arrangements(inputs))


def _bank(system: UnitSystem) -> None:
    _name, idx = _example(
        tb.TUBE_BANK_EXAMPLES, "ib_example", "Incropera (cap. 7), Çengel y Ghajar (cap. 7)"
    )
    inp = tb.TUBE_BANK_EXAMPLES[_name].inputs
    key = f"ib_{idx}"
    fluids = list(tb.tube_bank_fluids())
    fluid = st.selectbox(
        "Fluido que cruza el banco",
        fluids,
        index=fluids.index(inp.fluid),
        format_func=_fluid_name,
        key=f"{key}_fluid",
        help="Fluidos de CoolProp con propiedades de transporte (los de la convección).",
    )
    left, right = st.columns(2)
    with left:
        T_in = number_input_si(
            label="T de entrada", kind=_T, default_si=inp.T_in_K, key=f"{key}_Tin", format="%.5g"
        )
        V = number_input_si(
            label="Velocidad de llegada V",
            kind=_V,
            default_si=inp.V_m_s,
            key=f"{key}_V",
            format="%.5g",
            min_value_si=0.0,
            help="La del fluido antes de entrar al banco (en la sección libre del frente).",
        )
    with right:
        T_s = number_input_si(
            label="T de los tubos T_s",
            kind=_T,
            default_si=inp.T_s_K,
            key=f"{key}_Ts",
            format="%.5g",
            help="Los tubos a temperatura constante, como en los libros (por adentro condensa "
            "vapor o circula mucho caudal).",
        )
        p_default = (
            inp.p_Pa
            if fluid == inp.fluid
            else suggested_pressure(
                fluid, min(inp.T_in_K, inp.T_s_K), max(inp.T_in_K, inp.T_s_K), inp.p_Pa
            )
        )
        p = number_input_si(
            label="Presión p",
            kind=_P,
            default_si=p_default,
            key=f"{key}_{fluid}_p",
            format="%.5g",
            min_value_si=0.0,
        )
    arrangements = list(tb.ARRANGEMENTS)
    arrangement = st.radio(
        "Arreglo",
        arrangements,
        index=arrangements.index(inp.arrangement),
        format_func=lambda a: tb.ARRANGEMENTS[a],
        key=f"{key}_arr",
        horizontal=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        D = number_input_si(
            label="Diámetro D",
            kind=_D,
            default_si=inp.D_m,
            key=f"{key}_D",
            format="%.5g",
            min_value_si=0.0,
        )
    with c2:
        S_T = number_input_si(
            label="Paso S_T",
            kind=_D,
            default_si=inp.S_T_m,
            key=f"{key}_ST",
            format="%.5g",
            min_value_si=0.0,
            help="Transversal: entre los centros de dos tubos de una fila.",
        )
    with c3:
        S_L = number_input_si(
            label="Paso S_L",
            kind=_D,
            default_si=inp.S_L_m,
            key=f"{key}_SL",
            format="%.5g",
            min_value_si=0.0,
            help="Longitudinal: entre dos filas, en el sentido del flujo.",
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        N_L = _int_input("Filas N_L", inp.N_L, f"{key}_NL", "En el sentido del flujo.")
    with c2:
        N_T = _int_input("Tubos por fila N_T", inp.N_T, f"{key}_NT")
    with c3:
        L = number_input_si(
            label="Largo L",
            kind=_L,
            default_si=inp.L_m,
            key=f"{key}_L",
            format="%.5g",
            min_value_si=0.0,
        )
    sources = list(tb.CONSTANT_SOURCES)
    source = st.radio(
        "Constantes de Zukauskas",
        sources,
        index=sources.index(inp.source),
        format_func=lambda s: tb.CONSTANT_SOURCES[s],
        key=f"{key}_src",
        horizontal=True,
        help="Los dos libros tabulan los mismos datos de Zukauskas (1972) con tramos de Re "
        "y redondeos distintos.",
    )
    try:
        r = _bank_cached(
            tb.TubeBankInputs(fluid, p, V, T_in, T_s, D, S_T, S_L, N_L, N_T, L, arrangement, source)
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_bank(r, system)


def _render_bank(r: tb.TubeBankResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items = [
        (f"h [{_u(_H, system)}]", _val(r.h_W_per_m2K, _H, system, 4)),
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system, 4)),
        (f"T de salida [{_u(_T, system)}]", _val(r.T_out_K, _T, system)),
        (f"Δp [{_u(_PD, system)}]", _val(r.dp_Pa, _PD, system, 4)),
        (f"V_máx [{_u(_V, system)}]", _val(r.V_max_m_s, _V, system, 4)),
        ("Re = ρ·V_máx·D/μ", _num(r.Re, 4)),
        ("Nu (con la corrección por filas)", _num(r.Nu, 4)),
        (f"ṁ [{_u(_M, system)}]", _val(r.m_dot_kg_s, _M, system, 4)),
        (f"Q̇ por largo de tubo [{_u(_WL, system)}]", _val(r.Q_per_length_W_m, _WL, system, 4)),
        ("f · χ", f"{_num(r.f, 3)} · {_num(r.chi, 3)}"),
    ]
    _metric_rows(items)
    st.caption(
        f"{i.N_L} filas de {i.N_T} tubos ({r.N} en total) de L = {_val(i.L_m, _L, system)} "
        f"{_u(_L, system)}; A_s = {_val(r.area_m2, _A, system, 4)} {_u(_A, system)}."
    )
    _messages(r.warnings, tb.tube_bank_notes(r))
    sweep, comparison = _bank_extras_cached(i)
    reference = "alineado cuadrado" if i.arrangement == "aligned" else "tresbolillo equilátero"
    st.markdown(f"#### El f de Zukauskas ({reference})")
    _chart(
        friction_chart_figure(
            tb.friction_curves(i.arrangement),
            i.arrangement,
            Re=r.Re,
            f=r.f,
            pitch=r.friction.pitch,
        ),
        "ib_friction",
        "Como en los libros (Incropera, figs. 7.13 y 7.14): f contra Re, una curva por paso, "
        "para el arreglo de referencia (alineado cuadrado o tresbolillo equilátero). Con "
        "otros pasos, χ corrige: Δp = N_L·χ·f·ρ·V_máx²/2.",
    )
    st.markdown("#### h y Δp contra la velocidad")
    _chart(
        velocity_sweep_figure(
            sweep,
            V_now=i.V_m_s,
            h_now=r.h_W_per_m2K,
            dp_now=r.dp_Pa,
            V_scale=convert_from_si(1.0, _V, system),
            V_unit=_u(_V, system),
            h_scale=convert_from_si(1.0, _H, system),
            h_unit=_u(_H, system),
            dp_scale=convert_from_si(1.0, _PD, system),
            dp_unit=_u(_PD, system),
        ),
        "ib_sweep",
        "En log–log las dos son rectas: h crece como V^m (m ≈ 0,6) y Δp casi como V² (f baja "
        "un poco con Re). Al duplicar la velocidad, h sube ~50 % y Δp casi cuatro veces: el "
        "compromiso del diseño.",
    )
    if len(comparison) == 2:
        st.markdown("#### Alineado o tresbolillo")
        _table(
            pd.DataFrame(
                [
                    {
                        "Arreglo": tb.ARRANGEMENTS[a],  # type: ignore[index]
                        f"h [{_u(_H, system)}]": _val(c.h_W_per_m2K, _H, system, 4),
                        f"Q̇ [{_u(_W, system)}]": _val(c.Q_W, _W, system, 4),
                        f"Δp [{_u(_PD, system)}]": _val(c.dp_Pa, _PD, system, 4),
                    }
                    for a, c in comparison.items()
                ]
            )
        )
        st.caption(
            "Los mismos pasos, D, velocidad y filas. El tresbolillo mezcla más el fluido: más h "
            "y, en general, más Δp."
        )
    st.markdown("#### Propiedades")
    _props_table(
        [
            ("T_m (media, iterada)", r.T_mean_K, _T),
            ("ρ", r.rho_kg_per_m3, _RHO),
            ("μ", r.mu_Pa_s, _MU),
            ("k", r.k_W_per_mK, _K),
            ("c_p", r.cp_J_per_kgK, _CP),
            ("Pr", r.Pr, None),
            ("Pr_s (a T_s)", r.Pr_s, None),
            ("ρ_ent (para ṁ)", r.rho_in_kg_per_m3, _RHO),
        ],
        system,
    )
    _render_steps(tube_bank_steps(r, system))
    _render_export(tb.tube_bank_to_dict(r, system), "banco_de_tubos", "ib_dl")


# ---------------------------------------------------------------------
# Doble tubo
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _dp_cached(inputs: dp.DoublePipeInputs) -> dp.DoublePipeResult:
    return dp.solve_double_pipe(inputs)


@st.cache_data(show_spinner=False)
def _dp_swap_cached(inputs: dp.DoublePipeInputs) -> dp.DoublePipeResult | None:
    return dp.swap_sides(inputs)


def _double_pipe(system: UnitSystem) -> None:
    _name, idx = _example(
        dp.DOUBLE_PIPE_EXAMPLES, "id_example", "Incropera (cap. 11), Çengel y Ghajar (cap. 11)"
    )
    inp = dp.DOUBLE_PIPE_EXAMPLES[_name].inputs
    key = f"id_{idx}"
    T_range = (inp.cold.T_in_K, inp.hot.T_in_K)
    hot = _stream_inputs(f"Fluido caliente ({stream_name(inp.hot)})", inp.hot, f"{key}_h", T_range)
    cold = _stream_inputs(f"Fluido frío ({stream_name(inp.cold)})", inp.cold, f"{key}_c", T_range)
    st.markdown("**Geometría**")
    sides = list(_SIDES)
    tube_side = st.radio(
        "¿Qué corriente va por el tubo interior?",
        sides,
        index=sides.index(inp.tube_side),
        format_func=lambda s: _SIDES[s],
        key=f"{key}_side",
        horizontal=True,
        help="La otra va por el anular, entre el tubo y el caño exterior.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        d_i = number_input_si(
            label="Tubo, d_i",
            kind=_D,
            default_si=inp.d_i_m,
            key=f"{key}_di",
            format="%.5g",
            min_value_si=0.0,
        )
    with c2:
        d_o = number_input_si(
            label="Tubo, d_o",
            kind=_D,
            default_si=inp.d_o_m,
            key=f"{key}_do",
            format="%.5g",
            min_value_si=0.0,
            help="Igual a d_i: pared delgada (no cuenta su resistencia).",
        )
    with c3:
        D = number_input_si(
            label="Caño exterior, D",
            kind=_D,
            default_si=inp.D_m,
            key=f"{key}_D",
            format="%.5g",
            min_value_si=0.0,
            help="El diámetro interior del caño exterior: el anular va de d_o a D.",
        )
    c1, c2 = st.columns(2)
    with c1:
        k_wall = number_input_si(
            label="k de la pared",
            kind=_K,
            default_si=inp.k_wall_W_per_mK,
            key=f"{key}_kw",
            format="%.5g",
            min_value_si=0.0,
            help="45 W/(m·K) el acero al carbono, 15 el inoxidable, 400 el cobre.",
        )
    with c2:
        leg = number_input_si(
            label="Largo de una rama",
            kind=_L,
            default_si=inp.leg_length_m,
            key=f"{key}_leg",
            format="%.5g",
            min_value_si=0.0,
            help="Una horquilla tiene dos ramas rectas unidas por una curva.",
        )
    flows = list(dp.FLOWS)
    flow = st.radio(
        "Sentido",
        flows,
        index=flows.index(inp.flow),
        format_func=lambda f: dp.FLOWS[f],
        key=f"{key}_flow",
        horizontal=True,
    )
    problems = list(dp.PROBLEMS)
    problem = st.radio(
        "Problema",
        problems,
        index=problems.index(inp.problem),
        format_func=lambda p: dp.PROBLEMS[p],
        key=f"{key}_problem",
    )
    target, target_value, hairpins = inp.target, inp.target_value, inp.hairpins
    try:
        base: dp.DoublePipeResult | None = _dp_cached(inp)
    except ValueError:
        base = None
    if problem == "sizing":
        solved = (
            _solved_targets(base.T_hot_out_K, base.T_cold_out_K, base.Q_W)
            if base is not None
            else _solved_targets(inp.hot.T_in_K, inp.cold.T_in_K, 0.0)
        )
        target, target_value = _target_inputs(inp.target, inp.target_value, solved, key)
    else:
        default_n = inp.hairpins if inp.problem == "rating" or base is None else base.hairpins
        hairpins = _int_input("Horquillas", default_n, f"{key}_hp", "Cada una, dos ramas.")
    with st.expander("Correlaciones de cada lado", expanded=False):
        tube_options = list(TUBE_SIDE_CORRELATIONS)
        tube_corr = st.selectbox(
            "Tubo interior",
            tube_options,
            index=tube_options.index(inp.tube_correlation),
            format_func=lambda c: TUBE_SIDE_CORRELATIONS[c],
            key=f"{key}_tcorr",
        )
        ann_options = list(dp.ANNULUS_CORRELATIONS)
        ann_corr = st.selectbox(
            "Anular",
            ann_options,
            index=ann_options.index(inp.annulus_correlation),
            format_func=lambda c: dp.ANNULUS_CORRELATIONS[c],
            key=f"{key}_acorr",
        )
    try:
        r = _dp_cached(
            dp.DoublePipeInputs(
                hot=hot,
                cold=cold,
                d_i_m=d_i,
                d_o_m=d_o,
                D_m=D,
                tube_side=tube_side,
                k_wall_W_per_mK=k_wall,
                flow=flow,
                problem=problem,
                target=target,
                target_value=target_value,
                hairpins=hairpins,
                leg_length_m=leg,
                tube_correlation=tube_corr,
                annulus_correlation=ann_corr,
            )
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_double_pipe(r, system)


def _render_double_pipe(r: dp.DoublePipeResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    items = [
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system, 4)),
        (f"U_o [{_u(_H, system)}]", _val(r.U_o_W_per_m2K, _H, system, 4)),
        (f"T de salida del caliente [{_u(_T, system)}]", _val(r.T_hot_out_K, _T, system)),
        (f"T de salida del frío [{_u(_T, system)}]", _val(r.T_cold_out_K, _T, system)),
    ]
    if i.problem == "sizing":
        items += [
            (f"Largo necesario [{_u(_L, system)}]", _val(r.L_m, _L, system, 4)),
            ("Horquillas", str(r.hairpins)),
            (f"Largo instalado [{_u(_L, system)}]", _val(r.L_installed_m, _L, system, 4)),
            ("Sobrediseño", _pct(r.overdesign)),
        ]
    else:
        items += [
            (f"Largo L [{_u(_L, system)}]", _val(r.L_m, _L, system, 4)),
            ("Efectividad ε", _num(r.exchanger.effectiveness, 4)),
        ]
    items += [
        (f"h_i (tubo) [{_u(_H, system)}]", _val(r.tube.h_W_per_m2K, _H, system, 4)),
        (f"h_o (anular) [{_u(_H, system)}]", _val(r.annulus.h_W_per_m2K, _H, system, 4)),
        (f"Δp del tubo [{_u(_PL, system)}]", _val(r.dp_tube_Pa, _PL, system, 4)),
        (f"Δp del anular [{_u(_PL, system)}]", _val(r.dp_annulus_Pa, _PL, system, 4)),
        (f"Área A [{_u(_A, system)}]", _val(r.area_m2, _A, system, 4)),
        (f"U_o limpio [{_u(_H, system)}]", _val(r.U_o_clean_W_per_m2K, _H, system, 4)),
    ]
    _metric_rows(items)
    _messages(r.warnings, dp.double_pipe_notes(r))
    tube_s, ann_s = (i.cold, i.hot) if r.tube_stream == "cold" else (i.hot, i.cold)
    _side_table(
        [
            ("Tubo", stream_name(tube_s), r.tube, r.dp_tube_Pa, r.tube_props),
            ("Anular", stream_name(ann_s), r.annulus, r.dp_annulus_Pa, r.annulus_props),
        ],
        system,
    )
    returns = _val(r.dp_returns_Pa, _PL, system, 3)
    st.caption(
        f"Las vueltas de las horquillas suman en el anular ~{returns} {_u(_PL, system)} (una "
        "carga de velocidad por horquilla, Kern), aparte del Δp por fricción."
    )
    _resistances(r.u_result, system, "id_resist")
    _profile(r.exchanger, system, "id_profile")
    swapped = _dp_swap_cached(i)
    st.markdown("#### ¿Y si las corrientes van al revés?")
    if swapped is None:
        st.info(
            "Con las corrientes al revés (la del tubo por el anular) el problema no tiene "
            "solución: probá con otros diámetros."
        )
    else:
        what = "Largo necesario" if i.problem == "sizing" else "Q̇"
        unit = _u(_L, system) if i.problem == "sizing" else _u(_W, system)
        rows = []
        for label, res in (("Como está", r), ("Al revés", swapped)):
            rows.append(
                {
                    "Caso": label,
                    "Por el tubo": stream_name(
                        res.inputs.cold if res.tube_stream == "cold" else res.inputs.hot
                    ),
                    f"U_o [{_u(_H, system)}]": _val(res.U_o_W_per_m2K, _H, system, 4),
                    f"{what} [{unit}]": _val(
                        res.L_m if i.problem == "sizing" else res.Q_W,
                        _L if i.problem == "sizing" else _W,
                        system,
                        4,
                    ),
                    f"Δp tubo [{_u(_PL, system)}]": _val(res.dp_tube_Pa, _PL, system, 3),
                    f"Δp anular [{_u(_PL, system)}]": _val(res.dp_annulus_Pa, _PL, system, 3),
                }
            )
        _table(pd.DataFrame(rows))
        st.caption(
            "El fluido viscoso (o el de menor caudal) conviene donde más velocidad tenga; la "
            "decisión mira el U y también la caída de presión de cada lado."
        )
    st.markdown("#### Propiedades")
    hot_props = r.annulus_props if r.annulus_stream == "hot" else r.tube_props
    cold_props = r.tube_props if r.annulus_stream == "hot" else r.annulus_props
    _props_table(
        _stream_props_rows("caliente", hot_props, system)
        + _stream_props_rows("frío", cold_props, system),
        system,
    )
    _render_steps(double_pipe_steps(r, system))
    _render_export(dp.double_pipe_to_dict(r, system), "doble_tubo", "id_dl")


# ---------------------------------------------------------------------
# Casco y tubos (Kern o Bell–Delaware)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _st_cached(inputs: stt.ShellTubeInputs) -> stt.ShellTubeResult:
    return stt.solve_shell_tube(inputs)


@st.cache_data(show_spinner=False)
def _baffle_sweep_cached(inputs: stt.ShellTubeInputs) -> list[stt.BaffleSweepPoint]:
    return stt.baffle_sweep(inputs)


@st.cache_data(show_spinner=False)
def _cut_sweep_cached(inputs: stt.ShellTubeInputs) -> list[stt.BaffleCutSweepPoint]:
    return stt.baffle_cut_sweep(inputs)


def _kern_of(inputs: stt.ShellTubeInputs) -> stt.ShellTubeInputs:
    """El mismo equipo con Kern en el casco (para comparar)."""
    return replace(inputs, shell_method="kern", bell_delaware=None)


def _bd_geometry_inputs(
    inp: stt.ShellTubeInputs, key: str, system: UnitSystem, D_s: float, d_o: float, B: float
) -> BellDelawareGeometry:
    """Lo que Bell–Delaware pide además de lo de Kern."""
    geo = inp.bell_delaware or BellDelawareGeometry(L_bb_m=0.012)
    st.markdown("**Bell–Delaware**")
    lo, hi = BAFFLE_CUT_RANGE
    c1, c2 = st.columns(2)
    with c1:
        cut = st.number_input(
            "Corte de los bafles B_c [%]",
            min_value=1.0,
            max_value=49.0,
            value=round(100.0 * geo.baffle_cut, 4),
            step=1.0,
            format="%.4g",
            key=f"{key}_cut",
            help=f"La altura de la ventana, en % de D_s. Las correlaciones van del "
            f"{100 * lo:.0f} al {100 * hi:.0f} %; lo habitual, del 20 al 35 %.",
        )
    with c2:
        L_bb = number_input_si(
            label="Holgura haz–casco L_bb",
            kind=_D,
            default_si=geo.L_bb_m,
            key=f"{key}_Lbb",
            format="%.4g",
            min_value_si=0.0,
            help="Diametral: D_s menos el diámetro del haz. Es un dato del equipo (depende del "
            "cabezal); por ella pasa el bypass.",
        )
    hints = "; ".join(
        f"{name[0].lower()}{name[1:]}, {_val(a, _D, system, 2)} a {_val(b, _D, system, 2)} "
        f"{_u(_D, system)}"
        for name, (a, b) in BUNDLE_CLEARANCE_HINTS.items()
    )
    st.caption(f"L_bb según el cabezal: {hints}.")
    clr = st.radio(
        "Holguras casco–bafle y tubo–bafle",
        _CLR_MODES,
        index=0 if geo.L_sb_m is None and geo.L_tb_m is None else 1,
        key=f"{key}_clr",
        horizontal=True,
    )
    tema_sb = tema_shell_baffle_clearance(D_s)
    tema_tb = tema_tube_hole_clearance(d_o, 2.0 * B)
    L_sb = L_tb = None
    if clr == _CLR_GIVEN:
        c1, c2 = st.columns(2)
        with c1:
            L_sb = number_input_si(
                label="Casco–bafle L_sb",
                kind=_D,
                default_si=geo.L_sb_m if geo.L_sb_m is not None else tema_sb,
                key=f"{key}_Lsb",
                format="%.4g",
                min_value_si=0.0,
                help="Diametral: el diámetro interior del casco menos el del bafle.",
            )
        with c2:
            L_tb = number_input_si(
                label="Tubo–bafle L_tb",
                kind=_D,
                default_si=geo.L_tb_m if geo.L_tb_m is not None else tema_tb,
                key=f"{key}_Ltb",
                format="%.4g",
                min_value_si=0.0,
                help="Diametral: el agujero del bafle menos d_o.",
            )
    else:
        st.caption(
            f"TEMA: L_sb = {_val(tema_sb, _D, system, 3)} {_u(_D, system)} (RCB-4.3, según "
            f"D_s) y L_tb = {_val(tema_tb, _D, system, 3)} {_u(_D, system)} (RCB-4.2: el "
            "tramo sin apoyo de un tubo es 2·B)."
        )
    c1, c2 = st.columns(2)
    with c1:
        strips = int(
            st.number_input(
                "Pares de tiras de sellado N_ss",
                min_value=0,
                value=int(geo.sealing_strips),
                step=1,
                key=f"{key}_Nss",
                help="Cortan el bypass entre el haz y el casco; con la mitad de las hileras "
                "que cruza el fluido (N_ss ≥ N_tcc/2) lo anulan.",
            )
        )
    with c2:
        L_pl = number_input_si(
            label="Carril de los pasos L_pl",
            kind=_D,
            default_si=geo.L_pl_m,
            key=f"{key}_Lpl",
            format="%.4g",
            min_value_si=0.0,
            help="El ancho del carril que dejan las divisiones de los pasos, si va en la "
            "dirección del flujo (0 si no hay o si va perpendicular).",
        )
    ends = st.checkbox(
        "Separaciones de la entrada y de la salida distintas de B",
        value=geo.B_in_m is not None or geo.B_out_m is not None,
        key=f"{key}_ends",
        help="Las toberas obligan a separar más los primeros y los últimos bafles: ahí el "
        "fluido va más lento (J_s y R_s).",
    )
    B_in = B_out = None
    if ends:
        c1, c2 = st.columns(2)
        with c1:
            B_in = number_input_si(
                label="B de la entrada",
                kind=_L,
                default_si=geo.B_in_m or 1.5 * B,
                key=f"{key}_Bin",
                format="%.5g",
                min_value_si=0.0,
            )
        with c2:
            B_out = number_input_si(
                label="B de la salida",
                kind=_L,
                default_si=geo.B_out_m or 1.5 * B,
                key=f"{key}_Bout",
                format="%.5g",
                min_value_si=0.0,
            )
    return BellDelawareGeometry(
        L_bb_m=L_bb,
        baffle_cut=cut / 100.0,
        L_sb_m=L_sb,
        L_tb_m=L_tb,
        sealing_strips=strips,
        L_pl_m=L_pl,
        B_in_m=B_in,
        B_out_m=B_out,
    )


def _bwg_of(d_o: float, d_i: float) -> int:
    """El calibre BWG de esos diámetros, o 0 si no es ninguno."""
    for bwg in stt.BWG_WALL_IN:
        if abs(stt.tube_inner_diameter(d_o, bwg) - d_i) < 1e-6:
            return bwg
    return _BWG_OTHER


def _shell_tube(system: UnitSystem) -> None:
    _name, idx = _example(
        stt.SHELL_TUBE_EXAMPLES, "ik_example", "Kern (1950), Peters, Timmerhaus y West (2003)"
    )
    inp = stt.SHELL_TUBE_EXAMPLES[_name].inputs
    key = f"ik_{idx}"
    methods = list(stt.SHELL_METHODS)
    method = st.radio(
        "Método del lado del casco",
        methods,
        index=methods.index(inp.shell_method),
        format_func=lambda m: stt.SHELL_METHODS[m],
        key=f"{key}_method",
        horizontal=True,
        help="Kern (1950) supone que todo el fluido cruza el haz. Bell–Delaware (Taborek, "
        "1983) corrige el banco ideal por el corte de los bafles, las fugas por las holguras, "
        "el bypass entre el haz y el casco, las separaciones de los extremos y el flujo "
        "laminar.",
    )
    T_range = (inp.cold.T_in_K, inp.hot.T_in_K)
    hot = _stream_inputs(f"Fluido caliente ({stream_name(inp.hot)})", inp.hot, f"{key}_h", T_range)
    cold = _stream_inputs(f"Fluido frío ({stream_name(inp.cold)})", inp.cold, f"{key}_c", T_range)
    sides = list(_SIDES)
    shell_side = st.radio(
        "¿Qué corriente va por el casco?",
        sides,
        index=sides.index(inp.shell_side),
        format_func=lambda s: _SIDES[s],
        key=f"{key}_side",
        horizontal=True,
        help="La otra va por los tubos. Por los tubos conviene el fluido sucio, corrosivo o a "
        "alta presión (se limpian y se hacen más fácil).",
    )
    st.markdown("**El casco**")
    c1, c2, c3 = st.columns(3)
    with c1:
        D_s = number_input_si(
            label="Diámetro D_s",
            kind=_L,
            default_si=inp.D_s_m,
            key=f"{key}_Ds",
            format="%.5g",
            min_value_si=0.0,
        )
    with c2:
        B = number_input_si(
            label="Bafles cada B",
            kind=_L,
            default_si=inp.B_m,
            key=f"{key}_B",
            format="%.5g",
            min_value_si=0.0,
            help="Bafles segmentados (Kern los supone con el 25 % de corte). Lo habitual: de "
            "D_s/5 a D_s.",
        )
    with c3:
        baffles = _int_input(
            "Bafles N_b",
            inp.baffles,
            f"{key}_Nb",
            "N_b + 1 = L/B si están parejos; con las tapas más separadas, menos.",
        )
    st.markdown("**Los tubos**")
    c1, c2 = st.columns(2)
    with c1:
        d_o = number_input_si(
            label="Diámetro exterior d_o",
            kind=_D,
            default_si=inp.d_o_m,
            key=f"{key}_do",
            format="%.5g",
            min_value_si=0.0,
        )
    with c2:
        bwg_options = [*stt.BWG_WALL_IN, _BWG_OTHER]
        bwg = st.selectbox(
            "Calibre (espesor de pared)",
            bwg_options,
            index=bwg_options.index(_bwg_of(inp.d_o_m, inp.d_i_m)),
            format_func=lambda b: "d_i a elección" if b == _BWG_OTHER else f"BWG {b}",
            key=f"{key}_bwg",
            help="BWG: Birmingham Wire Gauge (TEMA; Kern, tabla 10). Más alto, pared más fina.",
        )
    if bwg == _BWG_OTHER:
        d_i = number_input_si(
            label="Diámetro interior d_i",
            kind=_D,
            default_si=inp.d_i_m,
            key=f"{key}_di",
            format="%.5g",
            min_value_si=0.0,
        )
    else:
        d_i = stt.tube_inner_diameter(d_o, bwg)
        st.caption(f"d_i = {_val(d_i, _D, system, 4)} {_u(_D, system)}")
    c1, c2 = st.columns(2)
    with c1:
        L = number_input_si(
            label="Largo L",
            kind=_L,
            default_si=inp.L_m,
            key=f"{key}_L",
            format="%.5g",
            min_value_si=0.0,
        )
    with c2:
        P_T = number_input_si(
            label="Paso P_T",
            kind=_D,
            default_si=inp.P_T_m,
            key=f"{key}_PT",
            format="%.5g",
            min_value_si=0.0,
            help="Entre los centros de dos tubos vecinos; lo habitual, 1,25·d_o.",
        )
    layouts = list(stt.LAYOUTS)
    layout = st.radio(
        "Arreglo",
        layouts,
        index=layouts.index(inp.layout),
        format_func=lambda a: stt.LAYOUTS[a],
        key=f"{key}_layout",
        horizontal=True,
        help="El cuadrado se limpia por afuera; el triangular entra más tubos.",
    )
    c1, c2 = st.columns(2)
    with c1:
        passes = st.selectbox(
            "Pasos por los tubos N_p",
            stt.TUBE_PASSES,
            index=stt.TUBE_PASSES.index(inp.tube_passes),
            key=f"{key}_np",
            help="Con 1 paso es un contracorriente; con más, un paso por el casco y N_p por los "
            "tubos (F < 1).",
        )
    with c2:
        k_wall = number_input_si(
            label="k de la pared",
            kind=_K,
            default_si=inp.k_wall_W_per_mK,
            key=f"{key}_kw",
            format="%.5g",
            min_value_si=0.0,
        )
    nt_mode = st.radio(
        "Cantidad de tubos N_t",
        _NT_MODES,
        index=0 if inp.tube_count is None else 1,
        key=f"{key}_ntmode",
        horizontal=True,
    )
    tube_count = None
    if nt_mode == _NT_GIVEN:
        estimate = stt.estimate_tube_count(D_s, P_T, layout, passes) if P_T > 0 else 1
        tube_count = _int_input(
            "N_t", inp.tube_count or max(estimate, 1), f"{key}_nt", "Los tubos del haz, en total."
        )
    geometry = (
        _bd_geometry_inputs(inp, key, system, D_s, d_o, B) if method == "bell_delaware" else None
    )
    problems = list(stt.SHELL_PROBLEMS)
    problem = st.radio(
        "Problema",
        problems,
        index=problems.index(inp.problem),
        format_func=lambda p: stt.SHELL_PROBLEMS[p],
        key=f"{key}_problem",
    )
    target, target_value = inp.target, inp.target_value
    if problem == "check":
        try:
            base = _st_cached(inp)
            solved = _solved_targets(base.T_hot_out_K, base.T_cold_out_K, base.Q_W)
        except ValueError:
            solved = _solved_targets(inp.hot.T_in_K, inp.cold.T_in_K, 0.0)
        default_target = inp.target if inp.problem == "check" else "T_hot_out"
        default_value = inp.target_value if inp.problem == "check" else solved["T_hot_out"]
        target, target_value = _target_inputs(default_target, default_value, solved, key)
    with st.expander("Correlación del lado de los tubos", expanded=False):
        tube_options = list(TUBE_SIDE_CORRELATIONS)
        tube_corr = st.selectbox(
            "Tubos",
            tube_options,
            index=tube_options.index(inp.tube_correlation),
            format_func=lambda c: TUBE_SIDE_CORRELATIONS[c],
            key=f"{key}_tcorr",
        )
    try:
        r = _st_cached(
            stt.ShellTubeInputs(
                hot=hot,
                cold=cold,
                D_s_m=D_s,
                d_o_m=d_o,
                d_i_m=d_i,
                L_m=L,
                P_T_m=P_T,
                B_m=B,
                baffles=baffles,
                layout=layout,
                tube_passes=passes,
                tube_count=tube_count,
                shell_side=shell_side,
                k_wall_W_per_mK=k_wall,
                problem=problem,
                target=target,
                target_value=target_value,
                tube_correlation=tube_corr,
                shell_method=method,
                bell_delaware=geometry,
            )
        )
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return
    _render_shell_tube(r, system)


def _render_shell_tube(r: stt.ShellTubeResult, system: UnitSystem) -> None:
    st.markdown("### Resultado")
    i = r.inputs
    if i.problem == "check":
        Rd, Rreq = r.R_d_available or 0.0, r.R_d_required
        if Rd >= Rreq:
            st.success(
                "**El equipo sirve**: tolera más ensuciamiento que el pedido (Kern).", icon="✅"
            )
        elif Rd >= 0.0:
            st.warning(
                "**Limpio alcanza, pero no tolera el ensuciamiento pedido**: cuando se ensucie "
                "no va a cumplir.",
                icon="⚠️",
            )
        else:
            st.warning("**Ni limpio alcanza**: hace falta más área.", icon="🚫")
    items = [
        (f"Q̇ [{_u(_W, system)}]", _val(r.Q_W, _W, system, 4)),
        (f"U_D (con ensuciamiento) [{_u(_H, system)}]", _val(r.U_D_W_per_m2K, _H, system, 4)),
        (f"T de salida del caliente [{_u(_T, system)}]", _val(r.T_hot_out_K, _T, system)),
        (f"T de salida del frío [{_u(_T, system)}]", _val(r.T_cold_out_K, _T, system)),
    ]
    if i.problem == "check":
        items += [
            (f"U requerido [{_u(_H, system)}]", _val(r.U_required_W_per_m2K, _H, system, 4)),
            (f"U_C (limpio) [{_u(_H, system)}]", _val(r.U_C_W_per_m2K, _H, system, 4)),
            (f"R_d que tolera [{_u(_RF, system)}]", _val(r.R_d_available, _RF, system, 3)),
            (f"R_d pedido [{_u(_RF, system)}]", _val(r.R_d_required, _RF, system, 3)),
            (f"Área que hay [{_u(_A, system)}]", _val(r.area_m2, _A, system, 4)),
            ("Sobrediseño (con U_D)", _pct(r.overdesign)),
        ]
    else:
        items += [
            (f"U_C (limpio) [{_u(_H, system)}]", _val(r.U_C_W_per_m2K, _H, system, 4)),
            (f"Área A_o [{_u(_A, system)}]", _val(r.area_m2, _A, system, 4)),
            ("Efectividad ε", _num(r.exchanger.effectiveness, 4)),
            ("Factor F", _num(r.exchanger.F, 4)),
        ]
    items += [
        (
            f"h_o (casco, {r.method_label}) [{_u(_H, system)}]",
            _val(r.h_shell_W_per_m2K, _H, system, 4),
        ),
        (f"h_i (tubos) [{_u(_H, system)}]", _val(r.tube.h_W_per_m2K, _H, system, 4)),
        (f"Δp del casco [{_u(_PL, system)}]", _val(r.dp_shell_Pa, _PL, system, 4)),
        (f"Δp de los tubos [{_u(_PL, system)}]", _val(r.dp_tube_Pa, _PL, system, 4)),
        ("Tubos N_t", str(r.N_t)),
        (f"Velocidad en los tubos [{_u(_V, system)}]", _val(r.tube.velocity_m_s, _V, system, 3)),
    ]
    _metric_rows(items)
    _messages(r.warnings, stt.shell_tube_notes(r))
    if r.bell_delaware is not None:
        _render_bell_delaware(r, system)
    else:
        _render_kern_side(r, system)
    _resistances(r.u_dirty, system, "ik_resist")
    _profile(r.exchanger, system, "ik_profile")
    st.markdown("#### Propiedades")
    hot_props = r.shell_props if i.shell_side == "hot" else r.tube_props
    cold_props = r.tube_props if i.shell_side == "hot" else r.shell_props
    _props_table(
        _stream_props_rows("caliente", hot_props, system)
        + _stream_props_rows("frío", cold_props, system),
        system,
    )
    _render_steps(shell_tube_steps(r, system))
    _render_export(stt.shell_tube_to_dict(r, system), f"casco_y_tubos_{i.shell_method}", "ik_dl")


def _render_kern_side(r: stt.ShellTubeResult, system: UnitSystem) -> None:
    i, s = r.inputs, r.shell
    st.markdown("#### El lado del casco (Kern)")
    _props_table(
        [
            ("a_s (flujo cruzado)", s.a_s_m2, _A),
            ("G_s = ṁ/a_s", s.G_s, _G),
            ("D_e", s.D_e_m, _D),
            ("Re_s", s.Re, None),
            ("φ_s = (μ/μ_p)^0,14", s.phi, None),
            ("Nu_s", s.Nu, None),
            ("f (Kakaç y Liu)", s.f, None),
            ("Cruces del haz N_b + 1", float(s.crossings), None),
        ],
        system,
    )
    st.markdown("#### h_o y Δp contra los bafles")
    _chart(
        baffle_sweep_figure(
            _baffle_sweep_cached(i),
            B_now=i.B_m,
            h_now=s.h_W_per_m2K,
            dp_now=s.dp_Pa,
            B_scale=convert_from_si(1.0, _L, system),
            B_unit=_u(_L, system),
            h_scale=convert_from_si(1.0, _H, system),
            h_unit=_u(_H, system),
            dp_scale=convert_from_si(1.0, _PL, system),
            dp_unit=_u(_PL, system),
        ),
        "ik_baffles",
        "De D_s/5 a D_s, con N_b = L/B − 1. Bafles más juntos: más velocidad y más h_o, pero "
        "la caída de presión crece mucho más rápido (más cruces y cada uno más rápido).",
    )


def _render_bell_delaware(r: stt.ShellTubeResult, system: UnitSystem) -> None:
    i, bd = r.inputs, r.bell_delaware
    assert bd is not None
    a, fc, ib = bd.areas, bd.factors, bd.ideal
    st.markdown("#### El lado del casco (Bell–Delaware)")
    geometry_rows: list[tuple[str, float | None, QuantityKind | None]] = [
        ("S_m (flujo cruzado)", a.S_m_m2, _A),
        ("S_w (una ventana)", a.S_w_m2, _A),
        ("S_sb (casco–bafle)", a.S_sb_m2, _A),
        ("S_tb (tubos–bafle)", a.S_tb_m2, _A),
        ("S_b (bypass)", a.S_b_m2, _A),
        ("F_c (tubos entre puntas)", a.F_c, None),
        ("N_tcc (hileras en el cruce)", a.N_tcc, None),
        ("N_tcw (hileras en la ventana)", a.N_tcw, None),
        ("r_s = S_sb/(S_sb + S_tb)", a.r_s, None),
        ("r_lm = (S_sb + S_tb)/S_m", a.r_lm, None),
        ("F_sbp = S_b/S_m", a.F_sbp, None),
        ("r_ss = N_ss/N_tcc", a.r_ss, None),
        ("L_sb (casco–bafle)", a.L_sb_m, _D),
        ("L_tb (tubo–bafle)", a.L_tb_m, _D),
    ]
    if fc.laminar:
        geometry_rows.append(("D_w (de la ventana)", a.D_w_m, _D))
    _props_table(geometry_rows, system)
    st.markdown("#### Del banco ideal al h_o")
    factors = [
        ("J_c (corte)", fc.J_c),
        ("J_l (fugas)", fc.J_l),
        ("J_b (bypass)", fc.J_b),
        ("J_s (extremos)", fc.J_s),
        ("J_r (laminar)", fc.J_r),
    ]
    _chart(
        h_cascade_figure(
            bd.h_ideal_W_per_m2K,
            factors,
            h_scale=convert_from_si(1.0, _H, system),
            h_unit=_u(_H, system),
        ),
        "ik_cascade",
        "Cada escalón es lo que cambia el h con ese factor (×J, sobre el h que llega). Los "
        "que valen 1 no aparecen. J_c pasa de 1 con ventanas chicas (el fluido se acelera).",
    )
    _props_table(
        [
            ("Re_s = d_o·ṁ/(μ·S_m)", bd.Re, None),
            ("j_i (Taborek)", ib.j, None),
            ("φ_s = (μ/μ_p)^0,14", bd.phi, None),
            ("h_ideal", bd.h_ideal_W_per_m2K, _H),
            *[(name, J, None) for name, J in factors],
            ("J_c·J_l·J_b·J_s·J_r", fc.J_total, None),
            ("h_o", bd.h_W_per_m2K, _H),
        ],
        system,
    )
    st.markdown("#### La caída de presión por zonas")
    _chart(
        dp_zones_figure(
            [
                ("Flujo cruzado", bd.dp_cross_ideal_Pa, bd.dp_cross_Pa),
                ("Ventanas", bd.dp_window_ideal_Pa, bd.dp_window_Pa),
                ("Extremos", bd.dp_ends_ideal_Pa, bd.dp_ends_Pa),
            ],
            dp_scale=convert_from_si(1.0, _PL, system),
            dp_unit=_u(_PL, system),
        ),
        "ik_zones",
        f"Los N_b − 1 = {i.baffles - 1} compartimientos centrales, las N_b = {i.baffles} "
        "ventanas y los dos extremos. Las fugas (R_l) y el bypass (R_b) bajan el Δp: el fluido "
        "que no cruza el haz casi no pierde presión.",
    )
    _props_table(
        [
            ("f_i (Taborek)", ib.f, None),
            ("Δp_bi (un compartimiento ideal)", bd.dp_bi_Pa, _PL),
            ("Δp_wi (una ventana ideal)", bd.dp_wi_Pa, _PL),
            ("R_l (fugas)", fc.R_l, None),
            ("R_b (bypass)", fc.R_b, None),
            ("R_s (entrada)", fc.R_s_in, None),
            ("R_s (salida)", fc.R_s_out, None),
            ("Δp del flujo cruzado", bd.dp_cross_Pa, _PL),
            ("Δp de las ventanas", bd.dp_window_Pa, _PL),
            ("Δp de los extremos", bd.dp_ends_Pa, _PL),
            ("Δp del casco", bd.dp_Pa, _PL),
        ],
        system,
    )
    st.markdown("#### Kern y Bell–Delaware")
    try:
        kern = _st_cached(_kern_of(i))
    except ValueError as exc:
        st.warning(f"Con Kern no se puede resolver este equipo: {exc}")
        kern = None
    if kern is not None:
        rows: list[tuple[str, float | None, float | None, QuantityKind]] = [
            ("h_o", r.h_shell_W_per_m2K, kern.h_shell_W_per_m2K, _H),
            ("Δp del casco", r.dp_shell_Pa, kern.dp_shell_Pa, _PL),
            ("U_D", r.U_D_W_per_m2K, kern.U_D_W_per_m2K, _H),
        ]
        if i.problem == "check":
            rows.append(("R_d que tolera", r.R_d_available, kern.R_d_available, _RF))
        else:
            rows += [
                ("Q̇", r.Q_W, kern.Q_W, _W),
                ("T de salida del caliente", r.T_hot_out_K, kern.T_hot_out_K, _T),
                ("T de salida del frío", r.T_cold_out_K, kern.T_cold_out_K, _T),
            ]
        _table(
            pd.DataFrame(
                [
                    {
                        "Magnitud": f"{name} [{_u(kind, system)}]",
                        "Bell–Delaware": _val(b, kind, system, 4),
                        "Kern": _val(k, kind, system, 4),
                        "Kern contra BD": "—"
                        if kind in (_T, _RF) or not b or k is None
                        else _signed(k / b - 1.0),
                    }
                    for name, b, k, kind in rows
                ]
            )
        )
        st.caption(
            "Cada método con sus propias temperaturas de salida y de pared (la nota de arriba "
            "compara con las mismas propiedades). Kern supone que todo el fluido cruza el haz."
        )
    st.markdown("#### h_o y Δp contra los bafles")
    h_scale, h_unit = convert_from_si(1.0, _H, system), _u(_H, system)
    dp_scale, dp_unit = convert_from_si(1.0, _PL, system), _u(_PL, system)
    L_scale, L_unit = convert_from_si(1.0, _L, system), _u(_L, system)
    series: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for name, inputs in (("Bell–Delaware", i), ("Kern", _kern_of(i))):
        pts = _baffle_sweep_cached(inputs)
        series[name] = (
            [p.B_m * L_scale for p in pts],
            [p.h_o_W_per_m2K * h_scale for p in pts],
            [p.dp_shell_Pa * dp_scale for p in pts],
        )
    _chart(
        methods_sweep_figure(
            series,
            x_now=i.B_m * L_scale,
            h_now=r.h_shell_W_per_m2K * h_scale,
            dp_now=r.dp_shell_Pa * dp_scale,
            x_title=f"Separación de los bafles B [{L_unit}]",
            h_title=f"h_o [{h_unit}]",
            dp_title=f"Δp del casco [{dp_unit}]",
            hover_x="B",
        ),
        "ik_baffles",
        "De D_s/5 a D_s, con N_b = L/B − 1 y los extremos como en los datos. Los dos métodos "
        "ven el mismo compromiso (bafles más juntos: más h_o y mucho más Δp), pero Kern no "
        "descuenta las fugas ni el bypass. Los quiebres de Bell–Delaware vienen de las tablas: "
        "la de Taborek salta en Re = 10⁴ a 90° y en 10³ a 45°, y TEMA achica L_tb cuando el "
        "tramo sin apoyo de un tubo (2·B) pasa de 36″ (914 mm).",
    )
    st.markdown("#### h_o y Δp contra el corte de los bafles")
    pts = _cut_sweep_cached(i)
    cut_series: dict[str, tuple[list[float], list[float], list[float]]] = {
        "Bell–Delaware": (
            [100.0 * p.baffle_cut for p in pts],
            [p.h_o_W_per_m2K * h_scale for p in pts],
            [p.dp_shell_Pa * dp_scale for p in pts],
        )
    }
    if kern is not None:
        lo, hi = BAFFLE_CUT_RANGE
        cut_series["Kern"] = (
            [100.0 * lo, 100.0 * hi],
            [kern.h_shell_W_per_m2K * h_scale] * 2,
            [kern.dp_shell_Pa * dp_scale] * 2,
        )
    _chart(
        methods_sweep_figure(
            cut_series,
            x_now=100.0 * bd.geometry.baffle_cut,
            h_now=r.h_shell_W_per_m2K * h_scale,
            dp_now=r.dp_shell_Pa * dp_scale,
            x_title="Corte de los bafles B_c [%]",
            h_title=f"h_o [{h_unit}]",
            dp_title=f"Δp del casco [{dp_unit}]",
            hover_x="B_c [%]",
            log_x=False,
        ),
        "ik_cut",
        "Del 15 al 45 % de D_s. Un corte chico acelera el fluido en las ventanas (más h_o y "
        "mucho más Δp); uno grande deja más tubos en las ventanas, donde el flujo es paralelo. "
        "Kern no ve el corte (supone el 25 %): es una recta.",
    )


# ---------------------------------------------------------------------
# Teoría
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) "
            "todavía no tiene un capítulo de transferencia de calor: las fórmulas siguen a "
            "**Çengel y Ghajar (2015)** (§7-4 y cap. 11), **Incropera et al. (2007)** (§7.6, "
            "§8.7 y cap. 11), **Kern (1950)** y **Kakaç y Liu (2002)** y, para Bell–Delaware, "
            "**Taborek (1983)**, **Thome (2004)** y **Serth (2014)**, con las propiedades de "
            "CoolProp."
        )
        st.markdown(
            "**Banco de tubos** (Zukauskas, 1972). La velocidad máxima es la del paso más "
            "angosto: el plano transversal o, en un tresbolillo con S_D < (S_T + D)/2, la "
            "diagonal S_D = √(S_L² + (S_T/2)²):"
        )
        st.latex(
            r"\begin{aligned}V_{\text{máx}} &= \frac{S_T}{S_T - D}\,V \\ "
            r"V_{\text{máx}} &= \frac{S_T}{2\,(S_D - D)}\,V\end{aligned}"
        )
        st.markdown(
            "Con las propiedades a la temperatura media y Pr_s a la de los tubos; C, m y n de "
            "la tabla de cada libro, y la corrección por filas (F o C₂) si hay menos de 16 o 20:"
        )
        st.latex(
            r"\mathrm{Nu}_D = C\,\mathrm{Re}^{m}\,\mathrm{Pr}^{n}"
            r"\left(\frac{\mathrm{Pr}}{\mathrm{Pr}_s}\right)^{1/4}"
        )
        st.latex(
            r"\begin{aligned}T_{\text{sal}} &= T_s - (T_s - T_{\text{ent}})\,"
            r"e^{-h A_s/(\dot m c_p)} \\ \dot Q &= h\,A_s\,\Delta T_{ml}\end{aligned}"
        )
        st.markdown(
            "La caída de presión, con f y χ de los gráficos (no tienen fórmula; acá, la "
            "digitalización de la biblioteca ht):"
        )
        st.latex(r"\Delta p = N_L\,\chi\,f\,\frac{\rho\,V_{\text{máx}}^2}{2}")
        st.markdown(
            "**Doble tubo**. El anular usa el diámetro hidráulico; en laminar, el Nu de la "
            "pared interior de Kays y Perkins (Incropera, tabla 8.2) y el f·Re de Shah y London "
            "(1978), con κ = d_o/D:"
        )
        st.latex(
            r"\begin{aligned}D_h &= D - d_o \\ f\,\mathrm{Re} &= "
            r"\frac{64\,(1 - \kappa)^2}{1 + \kappa^2 + (1 - \kappa^2)/\ln\kappa}\end{aligned}"
        )
        st.markdown("El U por metro de tubo, referido al área exterior (Çengel y Ghajar, §11-2):")
        st.latex(
            r"\begin{aligned}\frac{1}{U_o\,\pi\,d_o} &= \frac{1}{h_i\,\pi\,d_i} + "
            r"\frac{R''_{f,i}}{\pi\,d_i} \\ &\quad + \frac{\ln(d_o/d_i)}{2\pi\,k} \\ "
            r"&\quad + \frac{R''_{f,o}}{\pi\,d_o} + \frac{1}{h_o\,\pi\,d_o}\end{aligned}"
        )
        st.markdown(
            "Con ε-NTU (o la LMTD) sale el área; el largo y las horquillas de dos ramas, "
            "redondeadas para arriba. La caída de presión, con el largo instalado:"
        )
        st.latex(
            r"\begin{aligned}L &= \frac{A}{\pi\,d_o} \qquad N = \left\lceil"
            r"\frac{L}{2\,L_{\text{rama}}}\right\rceil \\ \Delta p &= f\,\frac{L}{D_h}\,"
            r"\frac{\rho\,u^2}{2}\end{aligned}"
        )
        st.markdown(
            "**Casco y tubos con Kern** (1950, cap. 7). El área de flujo cruzado en el ecuador "
            "del casco, con la luz C' = P_T − d_o entre dos tubos, y el diámetro equivalente: "
            "cuatro veces el área libre de la celda sobre el perímetro mojado (cuadrado y "
            "triangular):"
        )
        st.latex(
            r"\begin{aligned}a_s &= \frac{D_s\,C'\,B}{P_T} \qquad G_s = \frac{\dot m}{a_s} \\ "
            r"\mathrm{Re}_s &= \frac{G_s\,D_e}{\mu}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}D_e &= \frac{4\,(P_T^2 - \pi\,d_o^2/4)}{\pi\,d_o} \\ "
            r"D_e &= \frac{4\,(\frac{\sqrt{3}}{4}\,P_T^2 - \pi\,d_o^2/8)}{\pi\,d_o/2}"
            r"\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}h_o &= 0.36\,\frac{k}{D_e}\,\mathrm{Re}_s^{0.55}\,\mathrm{Pr}^{1/3}"
            r"\,\phi_s \\ \phi_s &= \left(\frac{\mu}{\mu_p}\right)^{0.14}\end{aligned}"
        )
        st.markdown(
            "La caída de presión del casco, una por cada cruce del haz, con el ajuste de Kakaç y "
            "Liu del f de Kern; la de los tubos, con cuatro cargas de velocidad por paso en las "
            "vueltas:"
        )
        st.latex(
            r"\begin{aligned}\Delta p_s &= \frac{f\,G_s^2\,D_s\,(N_b + 1)}{2\,\rho\,D_e\,\phi_s}"
            r" \\ f &= e^{0.576 - 0.19\,\ln\mathrm{Re}_s}\end{aligned}"
        )
        st.latex(r"\Delta p_t = \left(f\,\frac{L\,N_p}{d_i} + 4\,N_p\right)\frac{\rho\,u^2}{2}")
        st.markdown(
            "Sin el dato, la cantidad de tubos sale del casco (Kakaç y Liu): CTP = 0,93, 0,90 "
            "o 0,85 con 1, 2 o más pasos y CL = 0,87 triangular o 1 cuadrado:"
        )
        st.latex(r"N_t = 0.785\,\frac{CTP}{CL}\,\frac{D_s^2}{P_T^2}")
        st.markdown(
            "**El criterio de Kern**: con el área que hay, el U que hace falta para el servicio "
            "y el ensuciamiento que el equipo tolera, contra el pedido:"
        )
        st.latex(
            r"\begin{aligned}U_{\text{req}} &= \frac{\dot Q}{A_o\,F\,\Delta T_{ml}} \\ "
            r"R_d &= \frac{U_C - U_{\text{req}}}{U_C\,U_{\text{req}}}\end{aligned}"
        )
        st.latex(r"R_d \ge R''_{f,o} + R''_{f,i}\,\frac{d_o}{d_i}")
        st.markdown(
            "**Casco y tubos con Bell–Delaware** (Bell, 1963 y 1988; Taborek, 1983; Thome, "
            "2004, cap. 3; Serth, 2014, cap. 6). Tinker (1951) partió el flujo del casco en "
            "corrientes: la que cruza el haz (B), la que se fuga entre los tubos y los agujeros "
            "de los bafles (A), la que pasa entre el haz y el casco (C), la que se fuga entre "
            "los bafles y el casco (E) y la del carril de los pasos (F). Kern supone que todo "
            "es B; Bell–Delaware parte del h de un banco de tubos ideal y lo corrige con un "
            "factor por cada efecto:"
        )
        st.latex(r"h_o = h_{\text{ideal}}\,J_c\,J_l\,J_b\,J_s\,J_r")
        st.markdown(
            "El banco ideal va con la velocidad másica del área de flujo cruzado en la línea de "
            "centros (S_m: la luz entre los tubos y el canal entre el haz y el casco) y j_i de "
            "la tabla de Taborek para el arreglo de 30°, 45° o 90°:"
        )
        st.latex(
            r"\begin{aligned}S_m &= B\left[L_{bb} + \frac{D_{ctl}}{L_{tp,\text{ef}}}\,"
            r"(P_T - d_o)\right] \\ h_{\text{ideal}} &= j_i\,c_p\,\frac{\dot m}{S_m}\,"
            r"\mathrm{Pr}^{-2/3}\,\phi_s\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}j_i &= a_1\left(\frac{1.33}{P_T/d_o}\right)^{a}\mathrm{Re}_s^{a_2}"
            r" \\ a &= \frac{a_3}{1 + 0.14\,\mathrm{Re}_s^{a_4}}\end{aligned}"
        )
        st.markdown(
            "Los factores: J_c con F_c, la fracción de los tubos entre las puntas de los "
            "bafles; J_l con las áreas de las fugas (r_s = S_sb/(S_sb + S_tb) y r_lm = "
            "(S_sb + S_tb)/S_m); J_b con la del bypass (F_sbp = S_b/S_m) y las tiras de "
            "sellado por hilera (r_ss = N_ss/N_tcc), y J_s con las separaciones de los "
            "extremos sobre la central (L* = B_ext/B):"
        )
        st.latex(
            r"\begin{aligned}J_c &= 0.55 + 0.72\,F_c \\ J_l &= 0.44\,(1 - r_s) \\ "
            r"&\quad + \left[1 - 0.44\,(1 - r_s)\right]e^{-2.2\,r_{lm}}\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}J_b &= e^{-C_{bh}\,F_{sbp}\left(1 - \sqrt[3]{2\,r_{ss}}"
            r"\right)} \\ J_s &= \frac{N_b - 1 + L^{*\,1-n}_{\text{ent}} + "
            r"L^{*\,1-n}_{\text{sal}}}{N_b - 1 + L^*_{\text{ent}} + L^*_{\text{sal}}}"
            r"\end{aligned}"
        )
        st.markdown(
            "C_bh = 1,25 y n = 0,6 en turbulento (1,35 y ⅓ con Re_s < 100); con r_ss ≥ ½ las "
            "tiras anulan el bypass (J_b = 1). J_r vale 1 con Re_s ≥ 100; con Re_s ≤ 20, "
            "(10/N_c)^0,18, con N_c las hileras que cruza todo el fluido, interpolado en el "
            "medio y nunca menos de 0,4."
        )
        st.markdown(
            "La caída de presión va por zonas: los N_b − 1 compartimientos centrales, las N_b "
            "ventanas y los dos extremos, con un compartimiento y una ventana ideales "
            "(G_w = ṁ/√(S_m·S_w)) y los factores de las fugas (R_l), del bypass (R_b) y de los "
            "extremos (R_s):"
        )
        st.latex(
            r"\begin{aligned}\Delta p_{bi} &= \frac{2\,f_i\,N_{tcc}\,G_s^2}{\rho\,\phi_s}"
            r" \\ \Delta p_{wi} &= (2 + 0.6\,N_{tcw})\,\frac{G_w^2}{2\,\rho}"
            r"\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}\Delta p_s &= (N_b - 1)\,\Delta p_{bi}\,R_b\,R_l \\ "
            r"&\quad + N_b\,\Delta p_{wi}\,R_l \\ &\quad + \Delta p_{bi}\left(1 + "
            r"\frac{N_{tcw}}{N_{tcc}}\right)R_b \\ &\qquad \cdot (R_{s,\text{ent}} + "
            r"R_{s,\text{sal}})\end{aligned}"
        )
        st.latex(
            r"\begin{aligned}R_l &= e^{-1.33\,(1 + r_s)\,r_{lm}^{\,p}} \\ "
            r"R_b &= e^{-C_{bp}\,F_{sbp}\left(1 - \sqrt[3]{2\,r_{ss}}\right)}"
            r"\end{aligned}"
        )
        st.markdown(
            "Con p = 0,8 − 0,15·(1 + r_s), C_bp = 3,7 (4,5 en laminar) y R_s = (B/B_ext)^(2 − n′), "
            "n′ = 0,2 (1 en laminar). En laminar, la ventana suma además la fricción viscosa. "
            "Las holguras casco–bafle y tubo–bafle por defecto son las de **TEMA (2007)**: "
            "RCB-4.3 y RCB-4.2. La del haz con el casco depende del cabezal: es un dato del "
            "equipo."
        )


# ---------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------

st.set_page_config(page_title="Diseño de intercambiadores", page_icon="🛠️", layout="centered")

st.subheader(SUBJECT)
st.title("🛠️ Diseño de intercambiadores")
st.markdown(
    "Del h de cada lado al equipo: un **banco de tubos** que cruza un fluido (Zukauskas), un "
    "**doble tubo** dimensionado en horquillas y un **casco y tubos** con el método de Kern o "
    "el de Bell–Delaware, con el U armado, el ensuciamiento y las caídas de presión."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Diseño de intercambiadores")
render_units_selector()
system = get_current_system()

mode = st.radio("¿Qué querés diseñar?", _MODES, key="ix_mode", horizontal=True)
if mode == _M_BANK:
    _bank(system)
elif mode == _M_DP:
    _double_pipe(system)
else:
    _shell_tube(system)
