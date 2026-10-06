"""Página 5 — Ciclo de Rankine con TESPy (Fases 3.1a, 3.1b y 3.1c).

Ciclo de potencia de vapor: simple, con recalentamiento y con regeneración
(hasta tres calentadores de agua de alimentación, abiertos o cerrados, con
subenfriador de drenaje, desrecalentador y drenajes bombeados), ideal o real
(rendimientos isoentrópicos y pérdidas de carga y de calor, Cengel §10-3), con
agua o con un fluido orgánico (ORC, con recuperador). El cálculo lo hace
:mod:`core.cycles.rankine` con una red de TESPy; la página muestra el
resultado, las extracciones, la tabla de estados, el ciclo sobre el diagrama,
el procedimiento "como con las tablas" (Cengel cap. 10), el agua de
enfriamiento del condensador, la exportación y barridos para ver cómo cambia
el rendimiento (Cengel §10-4 y §10-6).
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.cycles.rankine import (
    RANKINE_EXAMPLE_NOTES,
    RANKINE_EXAMPLES,
    RANKINE_FLUIDS,
    X_TURBINE_MIN,
    CoolingWaterResult,
    FeedwaterHeater,
    Losses,
    PipeLoss,
    RankineInputs,
    RankineResult,
    Recuperator,
    Reheat,
    SweepParameter,
    condenser_cooling_water,
    default_extraction_values,
    default_heater_pressures,
    default_sweep_values,
    extraction_pressure_sweep,
    rankine_labeled_states,
    rankine_notes,
    rankine_steps,
    rankine_sweep,
    rankine_to_dict,
    solve_rankine,
    suggested_rankine_inputs,
)
from core.cycles.rankine_layout import PlantLayout, plant_layout
from core.diagrams import DiagramType
from core.export import dict_to_csv
from core.fluids import FLUID_NAMES_ES, fluid_limits, saturation_at_pressure
from core.state_report import format_value, states_table
from core.units_system import QuantityKind, UnitSystem, convert_from_si, convert_to_si, unit_label
from ui.branding import SUBJECT, VADEMECUM_DOI_URL, VADEMECUM_PDF_URL, sidebar_credits
from ui.cycle_charts import render_rankine_diagram
from ui.diagrams import diagram_type_selector
from ui.units_ui import get_current_system, number_input_si, render_units_selector

PAGE_VERSION = "0.14.0"
FLUID = "Water"

_EXAMPLES = list(RANKINE_EXAMPLES)
# Opciones fijas (sin número de estado): cambiarlas reiniciaría el widget.
_INLET_SUPERHEATED = "Vapor sobrecalentado (dar T)"
_INLET_SATURATED = "Vapor saturado seco (x = 1)"
_FLOW_MASS = "Caudal másico"
_FLOW_POWER = "Potencia neta"
_KINDS = {"Abierto": "open", "Cerrado": "closed"}
_POSITIONS = {2: ("de baja", "de alta"), 3: ("de baja", "intermedio", "de alta")}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind | None, str]] = {
    "Presión de la caldera": ("p_boiler", "pressure", "p caldera"),
    "Temperatura de entrada a la turbina": ("T_turbine_in", "temperature", "T entrada"),
    "Presión del condensador": ("p_condenser", "pressure", "p condensador"),
}
_ORC_SWEEPS: dict[str, tuple[SweepParameter, QuantityKind | None, str]] = {
    "Presión del evaporador": ("p_boiler", "pressure", "p evaporador"),
    "Temperatura de entrada a la turbina": ("T_turbine_in", "temperature", "T entrada"),
    "Presión del condensador": ("p_condenser", "pressure", "p condensador"),
}
_REC_SWEEP = "Efectividad del recuperador"


# ---------------------------------------------------------------------
# Cálculo cacheado (los resultados son dataclasses sin objetos de TESPy)
# ---------------------------------------------------------------------


@st.cache_data(show_spinner="Resolviendo el ciclo con TESPy…")
def _solve_cached(inputs: RankineInputs) -> RankineResult:
    return solve_rankine(inputs)


@st.cache_data(show_spinner="Calculando el barrido…")
def _sweep_cached(
    inputs: RankineInputs, parameter: SweepParameter
) -> list[tuple[float, float, float, float | None]]:
    values = default_sweep_values(inputs, parameter)
    return [
        (p.value_si, p.eta_th, p.w_net_J_per_kg, p.x_turbine_out)
        for p in rankine_sweep(inputs, parameter, values)
    ]


@st.cache_data(show_spinner="Calculando el barrido…")
def _extraction_sweep_cached(
    inputs: RankineInputs, heater: int
) -> list[tuple[float, float, float | None, tuple[float, ...]]]:
    values = default_extraction_values(inputs, heater)
    return [
        (p.value_si, p.eta_th, p.x_turbine_out, p.fractions)
        for p in extraction_pressure_sweep(inputs, heater, values)
    ]


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------


def _example_index() -> int:
    label = st.selectbox(
        "Ejemplo precargado",
        _EXAMPLES,
        key="rk_example",
        help=(
            "Ejemplos de Çengel & Boles, *Termodinámica*, cap. 10, y de ciclos orgánicos (ORC). "
            "Podés cambiar cualquier dato y volver a calcular."
        ),
    )
    return _EXAMPLES.index(label)


def _sub(index: int | None) -> str:
    """Subíndice con el número de estado (índice + 1), o nada si no se sabe."""
    return "" if index is None else str(index + 1).translate(_SUBSCRIPTS)


def _peek_pressure(key: str, default_si: float, system: UnitSystem) -> float:
    """Valor (SI) de un ``number_input_si`` que se dibuja más abajo en la página."""
    value = st.session_state.get(f"{key}@{system}")
    return default_si if value is None else convert_to_si(float(value), "pressure", system)


def _suffix(fluid: str) -> str:
    """Las keys de los widgets llevan el fluido (salvo el agua, que conserva las de la 0.12.0):
    al cambiar de fluido, cada dato arranca con los valores por defecto de ese fluido."""
    return "" if fluid == FLUID else f"_{fluid}"


def _base_inputs(ex: int, fluid: str) -> RankineInputs:
    """Datos del ejemplo o, si se eligió otro fluido, los sugeridos para ese fluido."""
    example = RANKINE_EXAMPLES[_EXAMPLES[ex]]
    return example if fluid == example.fluid else suggested_rankine_inputs(fluid)


def _boiler_name(fluid: str) -> str:
    return "caldera" if fluid == FLUID else "evaporador"


def _heater_defaults(
    base: RankineInputs, p_hi: float, p_lo: float, n: int
) -> list[FeedwaterHeater]:
    """Calentadores del ejemplo, o ``n`` abiertos que reparten por igual T_sat.

    El drenaje bombeado del ejemplo (``drain_forward`` de todo el ciclo, como
    Cengel 10-6) pasa a la marca del calentador de mayor presión.
    """
    if len(base.heaters) == n:
        return [
            replace(h, drain_forward=h.drain_forward or (base.drain_forward and k == n - 1))
            for k, h in enumerate(base.heaters)
        ]
    pressures = default_heater_pressures(p_hi, p_lo, n, fluid=base.fluid)
    return [FeedwaterHeater(p) for p in pressures]


def _default_losses(base: RankineInputs, p_hi: float, p_lo: float) -> Losses:
    """Pérdidas del ejemplo o, si no tiene, unas como las de Cengel 10-2 (proporcionales)."""
    if base.losses.any:
        return base.losses
    p_rh = base.reheat.p_Pa if base.reheat is not None else 0.0
    return Losses(
        dp_boiler_Pa=0.05 * p_hi,
        dp_reheater_Pa=0.05 * p_rh,
        dp_condenser_Pa=0.1 * p_lo,
        subcooling_K=5.0,
        feed_pipe=PipeLoss(0.005 * p_hi, 4.0),
        steam_pipe=PipeLoss(0.015 * p_hi, 25.0),
    )


def _provisional_layout(
    ex: int, fluid: str, base: RankineInputs, system: UnitSystem
) -> PlantLayout | None:
    """Numeración de estados con lo que está cargado, para los rótulos.

    Los widgets que la definen (recalentamiento, calentadores, cañerías y
    recuperador) están más abajo: se lee su valor de la corrida anterior en
    ``st.session_state``.
    """
    ss = st.session_state
    sfx = _suffix(fluid)
    p_hi = _peek_pressure(f"rk_p_hi_{ex}{sfx}", base.p_boiler_Pa, system)
    p_lo = _peek_pressure(f"rk_p_lo_{ex}{sfx}", base.p_condenser_Pa, system)
    p_rh = None
    if ss.get(f"rk_reheat_{ex}{sfx}", base.reheat is not None):
        default_rh = base.reheat.p_Pa if base.reheat else (p_hi * p_lo) ** 0.5
        p_rh = _peek_pressure(f"rk_p_rh_{ex}{sfx}", default_rh, system)
    heaters: list[FeedwaterHeater] = []
    regen = ss.get(f"rk_regen_{ex}{sfx}", bool(base.heaters))
    if regen:
        n = int(ss.get(f"rk_nheat_{ex}{sfx}", len(base.heaters) or 1))
        for k, d in enumerate(_heater_defaults(base, p_hi, p_lo, n)):
            default_kind = "Abierto" if d.kind == "open" else "Cerrado"
            kind = _KINDS[ss.get(f"rk_h{k}_kind_{ex}_{n}{sfx}", default_kind)]
            forward = kind == "closed" and bool(
                ss.get(f"rk_h{k}_fwd_{ex}_{n}{sfx}", d.drain_forward)
            )
            heaters.append(
                FeedwaterHeater(
                    _peek_pressure(f"rk_h{k}_p_{ex}_{n}{sfx}", d.p_Pa, system),
                    kind,
                    drain_forward=forward,
                )
            )
    losses = ss.get(f"rk_loss_{ex}{sfx}", base.losses.any)
    defaults = _default_losses(base, p_hi, p_lo)
    feed = losses and ss.get(f"rk_fpipe_{ex}{sfx}", defaults.feed_pipe is not None)
    steam = losses and ss.get(f"rk_spipe_{ex}{sfx}", defaults.steam_pipe is not None)
    recuperator = (
        fluid != FLUID and not regen and ss.get(f"rk_rec_{ex}{sfx}", base.recuperator is not None)
    )
    kwargs = {
        "feed_pipe": bool(feed),
        "steam_pipe": bool(steam),
        "recuperator": bool(recuperator),
        "boiler": _boiler_name(fluid),
    }
    try:
        return plant_layout(heaters=heaters, reheat_p_Pa=p_rh, **kwargs)  # type: ignore[arg-type]
    except ValueError:  # p. ej. presiones desordenadas: la validación lo va a explicar
        try:
            return plant_layout(reheat_p_Pa=p_rh, **kwargs)  # type: ignore[arg-type]
        except ValueError:
            return None


def _read_heaters(
    ex: int,
    sfx: str,
    base: RankineInputs,
    p_hi: float,
    p_lo: float,
    layout: PlantLayout | None,
    with_losses: bool,
) -> tuple[FeedwaterHeater, ...]:
    """Bloque de regeneración: cantidad, tipo, presión y datos de cada calentador."""
    if not st.checkbox(
        "Con regeneración (calentadores de agua de alimentación)",
        value=bool(base.heaters),
        key=f"rk_regen_{ex}{sfx}",
        help=(
            "Cengel §10-6: se extrae vapor de la turbina para precalentar el agua que va a la "
            "caldera; así sube la temperatura media de aporte de calor."
        ),
    ):
        return ()
    n = st.radio(
        "Cantidad de calentadores",
        (1, 2, 3),
        index=max(len(base.heaters), 1) - 1,
        horizontal=True,
        key=f"rk_nheat_{ex}{sfx}",
    )
    st.caption(
        "De menor a mayor presión de extracción. **Abierto**: la extracción se mezcla con el "
        "agua y sale líquido saturado (después va una bomba). **Cerrado**: intercambiador; el "
        "agua sale a T_sat − TTD y el drenaje va en cascada al calentador de menor presión "
        "(o al condensador), o se bombea hacia adelante a la línea de agua."
    )
    defaults = _heater_defaults(base, p_hi, p_lo, n)
    numbered = layout is not None and len(layout.extraction_states) == n
    heaters: list[FeedwaterHeater] = []
    for k, d in enumerate(defaults):
        with st.container(border=True):
            title = "Calentador" if n == 1 else f"Calentador {k + 1} ({_POSITIONS[n][k]})"
            st.markdown(f"**{title}**")
            kind = _KINDS[
                st.radio(
                    "Tipo",
                    tuple(_KINDS),
                    index=0 if d.kind == "open" else 1,
                    horizontal=True,
                    key=f"rk_h{k}_kind_{ex}_{n}{sfx}",
                )
            ]
            state = layout.extraction_states[k] if numbered and layout else None
            p_ext = number_input_si(
                label=f"Presión de extracción p{_sub(state)}",
                kind="pressure",
                default_si=d.p_Pa,
                key=f"rk_h{k}_p_{ex}_{n}{sfx}",
                format="%.5g",
            )
            if kind == "open":
                heaters.append(FeedwaterHeater(p_ext))
                continue
            desuperheater = st.checkbox(
                "Con desrecalentador (el agua puede salir por encima de T_sat: TTD < 0)",
                value=d.desuperheater,
                key=f"rk_h{k}_ds_{ex}_{n}{sfx}",
                help=(
                    "La extracción sobrecalentada se enfría primero hasta vapor saturado "
                    "calentando el agua que ya pasó por la zona de condensación. Necesita que la "
                    "extracción salga sobrecalentada."
                ),
            )
            ttd = number_input_si(
                label="TTD (diferencia terminal)",
                kind="temperature_difference",
                default_si=d.ttd_K,
                key=f"rk_h{k}_ttd_{ex}_{n}{sfx}",
                format="%.3g",
                help=(
                    "El agua de alimentación sale a T_sat(p_ext) − TTD. 0 = calentador ideal "
                    "(Cengel); negativo, solo con desrecalentador."
                ),
            )
            dca = None
            if st.checkbox(
                "Con subenfriador de drenaje",
                value=d.dca_K is not None,
                key=f"rk_h{k}_sc_{ex}_{n}{sfx}",
                help=(
                    "El drenaje se sigue enfriando contra el agua que entra: sale a la "
                    "temperatura de esa agua más el DCA (drain cooler approach)."
                ),
            ):
                dca = number_input_si(
                    label="DCA (aproximación del subenfriador)",
                    kind="temperature_difference",
                    default_si=d.dca_K if d.dca_K is not None else 5.6,
                    key=f"rk_h{k}_dca_{ex}_{n}{sfx}",
                    format="%.3g",
                    min_value_si=0.0,
                )
            forward = st.checkbox(
                "Bombear el drenaje hacia adelante (a la línea de agua)",
                value=d.drain_forward,
                key=f"rk_h{k}_fwd_{ex}_{n}{sfx}",
                help=(
                    "Como en Cengel 10-6: una bomba lleva el drenaje a una cámara de mezcla en la "
                    "línea de agua, justo después del calentador. Si no, va en cascada hacia "
                    "atrás por una válvula."
                ),
            )
            dp = 0.0
            if with_losses:
                dp = number_input_si(
                    label="Δp del agua en los tubos",
                    kind="pressure",
                    default_si=d.dp_Pa if d.dp_Pa else 0.01 * p_hi,
                    key=f"rk_h{k}_dp_{ex}_{n}{sfx}",
                    format="%.4g",
                    min_value_si=0.0,
                )
            heaters.append(
                FeedwaterHeater(
                    p_ext,
                    kind,
                    ttd,
                    dca_K=dca,
                    desuperheater=desuperheater,
                    drain_forward=forward,
                    dp_Pa=dp,
                )
            )
    return tuple(heaters)


def _read_losses(
    ex: int,
    sfx: str,
    base: RankineInputs,
    p_hi: float,
    p_lo: float,
    reheat: Reheat | None,
    layout: PlantLayout | None,
) -> Losses:
    """Bloque del ciclo real (Cengel §10-3): caídas de presión y de temperatura."""
    if not st.checkbox(
        "Con pérdidas (ciclo real: caídas de presión y de calor)",
        value=base.losses.any,
        key=f"rk_loss_{ex}{sfx}",
        help=(
            "Cengel §10-3: la fricción baja la presión en la caldera, el condensador y las "
            "cañerías (la bomba tiene que compensarlo), las cañerías pierden calor y el "
            "condensado se subenfría para que la bomba no cavite."
        ),
    ):
        return Losses()
    d = _default_losses(base, p_hi, p_lo)
    left, right = st.columns(2)
    with left:
        dp_boiler = number_input_si(
            label=f"Δp en la {_boiler_name(base.fluid)}",
            kind="pressure",
            default_si=d.dp_boiler_Pa,
            key=f"rk_dpb_{ex}{sfx}",
            format="%.4g",
            min_value_si=0.0,
        )
        subcooling = number_input_si(
            label="Subenfriamiento del condensado",
            kind="temperature_difference",
            default_si=d.subcooling_K,
            key=f"rk_sub_{ex}{sfx}",
            format="%.3g",
            min_value_si=0.0,
        )
    with right:
        dp_cond = number_input_si(
            label="Δp en el condensador",
            kind="pressure",
            default_si=d.dp_condenser_Pa,
            key=f"rk_dpc_{ex}{sfx}",
            format="%.4g",
            min_value_si=0.0,
        )
        dp_rh = 0.0
        if reheat is not None:
            dp_rh = number_input_si(
                label="Δp en el recalentador",
                kind="pressure",
                default_si=d.dp_reheater_Pa or 0.05 * reheat.p_Pa,
                key=f"rk_dpr_{ex}{sfx}",
                format="%.4g",
                min_value_si=0.0,
            )

    def pipe(what: str, key: str, default: PipeLoss | None, numbers: str) -> PipeLoss | None:
        if not st.checkbox(
            f"Cañería {what} {numbers}".rstrip(),
            value=default is not None,
            key=f"rk_{key}_{ex}{sfx}",
        ):
            return None
        fallback = default or PipeLoss(0.01 * p_hi, 5.0)
        cols = st.columns(2)
        with cols[0]:
            dp = number_input_si(
                label="Δp",
                kind="pressure",
                default_si=fallback.dp_Pa,
                key=f"rk_{key}_dp_{ex}{sfx}",
                format="%.4g",
                min_value_si=0.0,
            )
        with cols[1]:
            dT = number_input_si(
                label="ΔT (pierde calor)",
                kind="temperature_difference",
                default_si=fallback.dT_K,
                key=f"rk_{key}_dT_{ex}{sfx}",
                format="%.3g",
                min_value_si=0.0,
            )
        return PipeLoss(dp, dT)

    numbers_steam = numbers_feed = ""
    if layout is not None and layout.boiler_out != layout.turbine_inlet:
        numbers_steam = f"({layout.boiler_out + 1} → {layout.turbine_inlet + 1})"
    if layout is not None:
        pipes = [c for c in layout.of_kind("pipe") if c.port("out").state == layout.boiler_in]
        if pipes:
            numbers_feed = f"({pipes[0].port('in').state + 1} → {layout.boiler_in + 1})"
    boiler = _boiler_name(base.fluid)
    steam = pipe(
        f"{'de la' if boiler == 'caldera' else 'del'} {boiler} a la turbina",
        "spipe",
        d.steam_pipe,
        numbers_steam,
    )
    feed = pipe(
        f"de alimentación (hasta {'la' if boiler == 'caldera' else 'el'} {boiler})",
        "fpipe",
        d.feed_pipe,
        numbers_feed,
    )
    return Losses(
        dp_boiler_Pa=dp_boiler,
        dp_reheater_Pa=dp_rh,
        dp_condenser_Pa=dp_cond,
        subcooling_K=subcooling,
        feed_pipe=feed,
        steam_pipe=steam,
    )


def _read_recuperator(ex: int, sfx: str, base: RankineInputs) -> Recuperator | None:
    """Recuperador del ORC (sin calentadores): efectividad ε = q/q_máx."""
    if not st.checkbox(
        "Con recuperador (el escape de la turbina precalienta el líquido)",
        value=base.recuperator is not None,
        key=f"rk_rec_{ex}{sfx}",
        help=(
            "Con fluidos secos el vapor sale sobrecalentado de la turbina: el recuperador "
            "aprovecha ese calor. ε = q/q_máx (como el regenerador del ciclo Brayton, Cengel "
            "§9-9)."
        ),
    ):
        return None
    eps = st.number_input(
        "Efectividad ε del recuperador",
        min_value=0.05,
        max_value=0.99,
        value=float(base.recuperator.effectiveness if base.recuperator else 0.8),
        step=0.05,
        key=f"rk_rec_eps_{ex}{sfx}",
    )
    return Recuperator(float(eps))


def _read_cooling(ex: int) -> tuple[float, float] | None:
    """T de entrada y salida del agua de enfriamiento (SI), si se pide."""
    if not st.checkbox("Calcular el agua de enfriamiento del condensador", key=f"rk_cw_{ex}"):
        return None
    left, right = st.columns(2)
    with left:
        T_in = number_input_si(
            label="Agua: entrada",
            kind="temperature",
            default_si=293.15,
            key=f"rk_cw_in_{ex}",
            format="%.4g",
        )
    with right:
        T_out = number_input_si(
            label="Agua: salida",
            kind="temperature",
            default_si=303.15,
            key=f"rk_cw_out_{ex}",
            format="%.4g",
        )
    return T_in, T_out


def _read_inputs(ex: int, system: UnitSystem) -> tuple[RankineInputs, tuple[float, float] | None]:
    """Widgets con los valores del ejemplo ``ex`` (su número va en cada key)."""
    example = RANKINE_EXAMPLES[_EXAMPLES[ex]]
    fluid = st.selectbox(
        "Fluido de trabajo",
        RANKINE_FLUIDS,
        index=RANKINE_FLUIDS.index(example.fluid),
        format_func=lambda f: FLUID_NAMES_ES.get(f, f),
        key=f"rk_fluid_{ex}",
        help=(
            "Agua: el ciclo de vapor de las centrales. Los demás son fluidos orgánicos (ORC), "
            "para fuentes de calor de baja temperatura (geotermia, calor residual, solar)."
        ),
    )
    sfx = _suffix(fluid)
    base = _base_inputs(ex, fluid)
    boiler = _boiler_name(fluid)
    layout = _provisional_layout(ex, fluid, base, system)
    t_in = layout.turbine_inlet if layout else None
    real = bool(st.session_state.get(f"rk_loss_{ex}{sfx}", base.losses.any))
    st.markdown(f"**{boiler.capitalize()} y turbina**")
    p_label = (
        "Presión de entrada a la turbina"
        if real
        else ("Presión de la caldera" if boiler == "caldera" else "Presión del evaporador")
    )
    p_hi = number_input_si(
        label=f"{p_label} p{_sub(t_in)}",
        kind="pressure",
        default_si=base.p_boiler_Pa,
        key=f"rk_p_hi_{ex}{sfx}",
        format="%.5g",
    )
    inlet = st.radio(
        "Entrada a la turbina",
        (_INLET_SUPERHEATED, _INLET_SATURATED),
        index=0 if base.T_turbine_in_K is not None else 1,
        horizontal=True,
        key=f"rk_inlet_{ex}{sfx}",
    )
    T_in: float | None = None
    if inlet == _INLET_SUPERHEATED:
        T_in = number_input_si(
            label=f"Temperatura de entrada a la turbina T{_sub(t_in)}",
            kind="temperature",
            default_si=base.T_turbine_in_K or _superheated_default(fluid, p_hi),
            key=f"rk_T_in_{ex}{sfx}",
            format="%.5g",
        )
    p_lo = number_input_si(
        label=f"Presión del condensador p{_sub(layout.exhaust if layout else None)}",
        kind="pressure",
        default_si=base.p_condenser_Pa,
        key=f"rk_p_lo_{ex}{sfx}",
        format="%.5g",
    )
    left, right = st.columns(2)
    eta_t = left.number_input(
        "η_T de la turbina",
        min_value=0.3,
        max_value=1.0,
        value=float(base.eta_turbine),
        step=0.01,
        key=f"rk_eta_t_{ex}{sfx}",
        help=(
            "1 = turbina ideal (isoentrópica). Vademecum §10.4. Con extracciones, se mide "
            "desde la entrada de cada turbina."
        ),
    )
    eta_p = right.number_input(
        "η_B de la bomba",
        min_value=0.3,
        max_value=1.0,
        value=float(base.eta_pump),
        step=0.01,
        key=f"rk_eta_p_{ex}{sfx}",
        help="1 = bomba ideal (isoentrópica). Vademecum §10.4.",
    )

    reheat: Reheat | None = None
    if st.checkbox(
        "Con recalentamiento", value=base.reheat is not None, key=f"rk_reheat_{ex}{sfx}"
    ):
        default_rh = base.reheat or Reheat((p_hi * p_lo) ** 0.5, T_in or 873.15)
        rh_in = layout.reheat_in if layout else None
        rh_out = layout.reheat_out if layout else None
        p_rh = number_input_si(
            label=f"Presión de recalentamiento p{_sub(rh_in)}"
            + ("" if real else f" = p{_sub(rh_out)}"),
            kind="pressure",
            default_si=default_rh.p_Pa,
            key=f"rk_p_rh_{ex}{sfx}",
            format="%.5g",
        )
        T_rh = number_input_si(
            label=f"Temperatura de recalentamiento T{_sub(rh_out)}",
            kind="temperature",
            default_si=default_rh.T_K,
            key=f"rk_T_rh_{ex}{sfx}",
            format="%.5g",
        )
        reheat = Reheat(p_rh, T_rh)

    st.markdown("**Pérdidas**")
    losses = _read_losses(ex, sfx, base, p_hi, p_lo, reheat, layout)

    st.markdown("**Regeneración**")
    heaters = _read_heaters(ex, sfx, base, p_hi, p_lo, layout, losses.any)

    recuperator: Recuperator | None = None
    if fluid != FLUID:
        st.markdown("**Recuperador**")
        if heaters:
            st.caption(
                "El recuperador se usa en el ciclo sin calentadores de agua de alimentación."
            )
        else:
            recuperator = _read_recuperator(ex, sfx, base)

    st.markdown("**Tamaño de la planta**")
    flow = st.radio("Dato", (_FLOW_MASS, _FLOW_POWER), horizontal=True, key=f"rk_flow_{ex}")
    if flow == _FLOW_MASS:
        m_dot = number_input_si(
            label=f"Caudal másico ṁ (por {'la' if boiler == 'caldera' else 'el'} {boiler})",
            kind="mass_flow",
            default_si=base.m_dot_kg_s or 1.0,
            key=f"rk_m_{ex}{sfx}",
            format="%.5g",
        )
        flows: dict[str, float | None] = {"m_dot_kg_s": m_dot}
    else:
        power = number_input_si(
            label="Potencia neta Ẇ_neto",
            kind="power",
            default_si=100.0e6 if fluid == FLUID else 1.0e6,
            key=f"rk_W_{ex}{sfx}",
            format="%.5g",
        )
        flows = {"m_dot_kg_s": None, "W_net_W": power}
    cooling = _read_cooling(ex)
    inputs = RankineInputs(
        p_hi,
        T_in,
        p_lo,
        eta_t,
        eta_p,
        reheat,
        heaters=heaters,
        losses=losses,
        fluid=fluid,
        recuperator=recuperator,
        **flows,  # type: ignore[arg-type]
    )
    return inputs, cooling


def _superheated_default(fluid: str, p_Pa: float) -> float:
    """T por defecto si se pide vapor sobrecalentado: 600 °C con agua; con un fluido
    de ORC, 10 K sobre la saturación (sin pasar la T máx. de su ecuación de estado)."""
    if fluid == FLUID:
        return 873.15
    try:
        T_sat = saturation_at_pressure(fluid, p_Pa).T_sat_K
    except ValueError:
        return suggested_rankine_inputs(fluid).T_turbine_in_K or 400.0
    return float(min(round(T_sat + 10.0), fluid_limits(fluid).T_max_K - 1.0))


def _render_error(exc: ValueError) -> None:
    """Mensaje para el alumno arriba; el detalle técnico, aparte y chico."""
    message, _, detail = str(exc).partition("Detalle técnico:")
    st.error(message.strip(), icon="🚫")
    if detail:
        st.caption(f"Detalle técnico: {detail.strip()}")


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


def _number(value: float) -> str:
    """Número para mostrar: sin notación exponencial en los valores grandes."""
    if abs(value) >= 1.0e4:
        return f"{value:,.0f}".replace(",", " ")
    return format_value(value, 5)


def _energy(value_si: float, system: UnitSystem) -> str:
    return _number(convert_from_si(value_si, "specific_enthalpy", system))


def _render_metrics(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None
) -> None:
    e_unit = unit_label("specific_enthalpy", system)
    x = result.x_turbine_out
    row1 = st.columns(3)
    row1[0].metric("η térmico [%]", f"{result.eta_th * 100:.2f}")
    row1[1].metric(f"w neto [{e_unit}]", _energy(result.w_net_J_per_kg, system))
    row1[2].metric("Título a la salida", "sobrecalentado" if x is None else f"{x:.3f}")
    row2 = st.columns(3)
    row2[0].metric("η Carnot [%]", f"{result.eta_carnot * 100:.2f}")
    row2[1].metric("BWR = w_B/w_T [%]", f"{result.back_work_ratio * 100:.2f}")
    power_unit = unit_label("power", system)
    row2[2].metric(
        f"Ẇ neto [{power_unit}]",
        _number(convert_from_si(result.W_net_W, "power", system)),
    )
    if cooling is not None:
        st.metric(
            f"ṁ agua de enfriamiento [{unit_label('mass_flow', system)}]",
            _number(convert_from_si(cooling.m_dot_kg_s, "mass_flow", system)),
        )
    extras = ""
    if result.q_loss_J_per_kg > 0.0:
        extras += f" · q_pérd = {_energy(result.q_loss_J_per_kg, system)} {e_unit}"
    if result.has_recuperator:
        extras += f" · q_rec = {_energy(result.q_recuperator_J_per_kg, system)} {e_unit}"
    boiler = result.inputs.boiler_name
    st.caption(
        f"Caudal: {_number(convert_from_si(result.m_dot_kg_s, 'mass_flow', system))} "
        f"{unit_label('mass_flow', system)} · q_H = {_energy(result.q_in_J_per_kg, system)} "
        f"{e_unit} · q_C = {_energy(result.q_out_J_per_kg, system)} {e_unit}{extras}. "
        "BWR: relación de trabajo de retroceso."
        + (
            f" Las energías son por kg de vapor que pasa por "
            f"{'la' if boiler == 'caldera' else 'el'} {boiler}."
            if result.has_heaters
            else ""
        )
        + (" q_pérd: calor que pierden las cañerías." if extras.startswith(" · q_pérd") else "")
        + (" q_rec: calor interno del recuperador." if result.has_recuperator else "")
    )
    for note in rankine_notes(result, system):
        st.info(note)


def _render_extractions(result: RankineResult, system: UnitSystem) -> None:
    """Fracción y caudal de cada extracción (Cengel §10-6)."""
    if not result.has_heaters or result.layout is None:
        return
    p_unit = unit_label("pressure", system)
    m_unit = unit_label("mass_flow", system)
    rows = []
    for k, (name, heater, y) in enumerate(
        zip(
            result.layout.heater_names,
            result.inputs.heaters,
            result.extraction_fractions,
            strict=True,
        )
    ):
        rows.append(
            {
                "Calentador": name,
                "Estado n.º": result.layout.extraction_states[k] + 1,
                f"p [{p_unit}]": format_value(convert_from_si(heater.p_Pa, "pressure", system)),
                "y = ṁext/ṁ": f"{y:.4f}",
                f"ṁext [{m_unit}]": _number(
                    convert_from_si(y * result.m_dot_kg_s, "mass_flow", system)
                ),
            }
        )
    st.markdown("#### Extracciones")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_states_table(result: RankineResult, system: UnitSystem) -> None:
    frame = pd.DataFrame(states_table(rankine_labeled_states(result), system)).drop(
        columns="Fluido"
    )
    for column in frame.columns:
        if column not in ("Estado", "Región"):
            frame[column] = [format_value(v) for v in frame[column]]
    st.dataframe(frame, hide_index=True, width="stretch", height=35 * (len(frame) + 1) + 3)


def _render_tespy(result: RankineResult, system: UnitSystem) -> None:
    with st.expander("⚙️ Cómo lo resuelve TESPy", expanded=False):
        st.markdown(
            "TESPy arma la planta como una **red**: cada componente (bombas, caldera, turbinas, "
            "calentadores, condensador) aporta sus ecuaciones —balance de masa y energía, "
            "rendimiento isoentrópico, pérdida de carga— y cada conexión lleva caudal, presión "
            "y entalpía. Las extracciones son divisores de caudal (*Splitter*); los "
            "calentadores abiertos, mezcladores (*Merge*) con líquido saturado a la salida, y los "
            "cerrados, intercambiadores (*Condenser*, con subenfriamiento si tienen subenfriador "
            "de drenaje, y un *Desuperheater* antes si tienen desrecalentador). Las cañerías son "
            "*Pipe* y el recuperador del ORC, un *HeatExchanger*. Con los datos resuelve el "
            "sistema con Newton. Estos son sus balances por kilogramo de vapor que pasa por la "
            "caldera (positivo: entra al fluido; negativo: sale); ya incluyen la fracción de "
            "caudal de cada componente y coinciden con los del procedimiento:"
        )
        unit = unit_label("specific_enthalpy", system)
        frame = pd.DataFrame(
            [
                {"Componente": name, f"Energía por kg [{unit}]": _energy(value, system)}
                for name, value in result.tespy_balances
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")
        if result.tespy_heaters:
            st.markdown(
                "Calor que cada calentador cerrado pasa de la extracción al agua de "
                "alimentación, o el recuperador del escape al líquido (es calor **interno** del "
                "ciclo: no entra ni sale):"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Equipo": name, f"Calor por kg [{unit}]": _energy(value, system)}
                        for name, value in result.tespy_heaters
                    ]
                ),
                hide_index=True,
                width="stretch",
            )


def _cycle_diagram(result: RankineResult, system: UnitSystem) -> None:
    with st.expander("📈 Diagrama del ciclo", expanded=True):
        diagram_type: DiagramType = diagram_type_selector(key="rk_diagram_type", default="Ts")
        try:
            render_rankine_diagram(result, system, diagram_type, chart_key="rk_diagram_chart")
        except Exception as exc:  # el diagrama no debe tumbar la página
            st.warning(f"No se pudo dibujar el diagrama: {exc}")
        st.caption(
            "Caldera, condensador y calentadores se dibujan sobre su isobara real (la "
            "extracción condensa a la presión del calentador)"
            + (
                "; con pérdidas de carga, sobre una curva en la que la presión baja de a poco"
                if result.has_losses
                else ""
            )
            + ". En el ciclo ideal, bombas y "
            "turbina son isoentrópicas (verticales en T–s); en el real, la recta punteada que "
            "une la entrada y la salida es solo una referencia, y la verde es la expansión "
            "isoentrópica con la que se compara η_T. Las válvulas de los drenajes (y una "
            "cañería que pierde presión sin perder calor) se dibujan rayadas sobre su línea de h "
            "constante: el estrangulamiento es irreversible y sus estados intermedios no son de "
            "equilibrio."
            + (
                " El recuperador aparece dos veces: el escape que se enfría y el líquido que se "
                "calienta, cada uno sobre su isobara."
                if result.has_recuperator
                else ""
            )
        )


def _join_es(items: list[str]) -> str:
    """«a», «a y b», «a, b y c»."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} y {items[-1]}"


def _render_procedure(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None
) -> None:
    with st.expander("🔬 Procedimiento", expanded=False):
        sections = ["§10-2"]
        if result.has_losses:
            sections.append("§10-3")
        if result.has_reheat:
            sections.append("§10-5")
        if result.has_heaters:
            sections.append("§10-6")
        water = result.inputs.fluid == FLUID
        st.caption(
            "Cómo se resuelve "
            + ("con las tablas de vapor" if water else "con las propiedades del fluido")
            + ", estado por estado (Cengel "
            + _join_es(sections)
            + "). Los valores salen de la ecuación de estado ("
            + ("IAPWS-95" if water else "de Helmholtz, en CoolProp")
            + "), así que pueden diferir en el último decimal de los de una tabla impresa."
        )
        for i, step in enumerate(rankine_steps(result, system, cooling=cooling), start=1):
            st.markdown(f"**{i}. {step.title}**")
            if step.text:
                st.markdown(step.text)
            for tex in step.latex:
                st.latex(tex)


def _render_export(
    result: RankineResult, system: UnitSystem, cooling: CoolingWaterResult | None
) -> None:
    st.markdown("#### 💾 Exportar")
    data = rankine_to_dict(result, system, cooling=cooling)
    left, right = st.columns(2)
    left.download_button(
        "Descargar CSV",
        data=dict_to_csv(data).encode("utf-8"),
        file_name="rankine.csv",
        mime="text/csv",
        key="rk_download_csv",
    )
    right.download_button(
        "Descargar JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2),
        file_name="rankine.json",
        mime="application/json",
        key="rk_download_json",
    )


def _line_chart(xs: list[float], ys: list[float], *, title: str, x_title: str, y_title: str):
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(
        height=270,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def _render_sweeps(inputs: RankineInputs, result: RankineResult, system: UnitSystem) -> None:
    with st.expander("📊 ¿Cómo aumentar el rendimiento?", expanded=False):
        st.markdown(
            "Cengel §10-4: el rendimiento sube si se **baja la presión del condensador** (baja "
            "T_C), se **sobrecalienta más** o se **sube la presión de la caldera** (sube la "
            "temperatura media de aporte de calor T̄_H). Lo último baja el título a la salida de "
            "la turbina, y por eso se recalienta. Con **regeneración** (§10-6) cada extracción "
            "tiene una presión óptima, y en un ORC con fluido seco ayuda el **recuperador**. "
            "Elegí qué variar (el resto queda como en tu ciclo):"
        )
        names = result.layout.heater_names if result.layout else ()
        sweeps = dict(_SWEEPS if inputs.fluid == FLUID else _ORC_SWEEPS)
        if inputs.recuperator is not None:
            sweeps[_REC_SWEEP] = ("recuperator_eff", None, "ε")
        extractions = [f"Presión de extracción del {name}" for name in names]
        options = list(sweeps) + extractions
        choice = st.selectbox("Variable", options, key="rk_sweep_param")
        key: tuple = (
            ("extraction", extractions.index(choice))
            if choice in extractions
            else (sweeps[choice][0],)
        )
        if st.button("Calcular barrido", key="rk_sweep_btn"):
            st.session_state["rk_sweep"] = (inputs, key)
        if st.session_state.get("rk_sweep") != (inputs, key):
            st.caption("Tocá «Calcular barrido» (resuelve el ciclo varias veces).")
            return
        if key[0] == "extraction":
            heater = key[1]
            extraction = _extraction_sweep_cached(inputs, heater)
            if not extraction:
                st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
                return
            unit = unit_label("pressure", system)
            xs = [convert_from_si(p[0], "pressure", system) for p in extraction]
            x_title = f"p de extracción [{unit}]"
            st.plotly_chart(
                _line_chart(
                    xs,
                    [p[1] * 100 for p in extraction],
                    title="Rendimiento térmico",
                    x_title=x_title,
                    y_title="η [%]",
                ),
                width="stretch",
                key="rk_sweep_eta",
            )
            st.plotly_chart(
                _line_chart(
                    xs,
                    [p[3][heater] for p in extraction],
                    title="Fracción de extracción",
                    x_title=x_title,
                    y_title="y [-]",
                ),
                width="stretch",
                key="rk_sweep_y",
            )
            st.caption(
                "Con poca presión la extracción calienta poco el agua; con mucha, se lleva vapor "
                "que todavía podía dar trabajo: entre las dos hay un óptimo."
            )
            return
        parameter, kind, axis = sweeps[choice]
        points = _sweep_cached(inputs, parameter)
        if not points:
            st.warning("Ningún punto del barrido tiene sentido físico con estos datos.")
            return
        if kind is None:  # efectividad del recuperador (adimensional)
            xs = [p[0] for p in points]
            x_title = f"{axis} [-]"
        else:
            xs = [convert_from_si(p[0], kind, system) for p in points]
            x_title = f"{axis} [{unit_label(kind, system)}]"
        st.plotly_chart(
            _line_chart(
                xs,
                [p[1] * 100 for p in points],
                title="Rendimiento térmico",
                x_title=x_title,
                y_title="η [%]",
            ),
            width="stretch",
            key="rk_sweep_eta",
        )
        quality = [p[3] for p in points]
        if any(q is not None for q in quality):
            fig_x = _line_chart(
                xs,
                [q if q is not None else 1.0 for q in quality],
                title="Título a la salida de la turbina",
                x_title=x_title,
                y_title="x [-]",
            )
            fig_x.add_hline(
                y=X_TURBINE_MIN,
                line_dash="dot",
                annotation_text=f"x = {X_TURBINE_MIN} (límite práctico)",
            )
            st.plotly_chart(fig_x, width="stretch", key="rk_sweep_x")
            st.caption("Si el vapor sale sobrecalentado, el gráfico lo muestra como x = 1.")
        if parameter == "recuperator_eff":
            st.caption(
                "Más efectividad, más calor recuperado y menos calor de la fuente para el mismo "
                "trabajo; el costo es un recuperador más grande (ε → 1 pide área infinita)."
            )


# ---------------------------------------------------------------------
# Fórmulas teóricas
# ---------------------------------------------------------------------


def _render_theory() -> None:
    with st.expander("📖 Fórmulas teóricas", expanded=False):
        st.markdown(
            f"El [vademecum de la cátedra]({VADEMECUM_PDF_URL}) ([DOI]({VADEMECUM_DOI_URL})) no "
            "tiene una sección propia del ciclo de Rankine; las relaciones que se usan son las de "
            "**§3.3 *Sistema abierto, régimen permanente***, **§9.2 *Rendimientos térmicos***, "
            "§10.1 *Entropía*, **§10.4 *Rendimientos isoentrópicos***, §12 *Vapor húmedo* y §13 "
            "*Líquidos (incompresibles)*. En el libro: Çengel & Boles, cap. 10 (§10-2 a §10-7)."
        )
        st.markdown(
            "**Balance de cada componente** (§3.3, sin variaciones de energía cinética ni "
            "potencial; numeración de Cengel):"
        )
        st.latex(r"w_B = h_2 - h_1 \qquad q_H = h_3 - h_2")
        st.latex(r"w_T = h_3 - h_4 \qquad q_C = h_4 - h_1")
        st.markdown(
            "**Bomba** (líquido incompresible, §13) y **rendimientos isoentrópicos** (§10.4):"
        )
        st.latex(r"w_{B,s} \approx v_1\,(p_2 - p_1)")
        st.latex(r"\eta_B = \frac{h_{2s} - h_1}{h_2 - h_1}")
        st.latex(r"\eta_T = \frac{h_3 - h_4}{h_3 - h_{4s}}")
        st.markdown("**Rendimiento del ciclo** (§9.2), con temperaturas absolutas en Carnot:")
        st.latex(r"\eta_{\mathrm{MT}} = \frac{W}{Q_H} = 1 - \frac{Q_C}{Q_H}")
        st.latex(r"\eta_{\mathrm{Carnot}} = 1 - \frac{T_C}{T_H}")
        st.markdown(
            "**Temperatura media de aporte de calor** (de la definición de entropía, §10.1). En "
            "el ciclo ideal sin regeneración y con vapor húmedo a la salida de la turbina, el "
            "calor se cede a T_C constante y η = 1 − T_C/T̄_H:"
        )
        st.latex(r"\bar{T}_H = \frac{q_H}{s_3 - s_2}")
        st.markdown(
            "**Con recalentamiento** (Cengel §10-5; estados 1 a 6): el calor entra en la "
            "caldera y en el recalentador, y el trabajo sale de las dos turbinas:"
        )
        st.latex(r"q_H = (h_3 - h_2) + (h_5 - h_4)")
        st.latex(r"w_T = (h_3 - h_4) + (h_5 - h_6)")
        st.markdown(
            "**Con regeneración** (Cengel §10-6) se extrae una fracción y del vapor para "
            "precalentar el agua de alimentación. En el calentador **abierto** se mezclan y sale "
            "líquido saturado; en el **cerrado** la extracción condensa (sale líquido saturado, el "
            "drenaje) y el agua sale a T_sat − TTD:"
        )
        st.latex(r"y\,h_{\mathrm{ext}} + (1 - y)\,h_{\mathrm{ent}} = h_{\mathrm{sal}}")
        st.latex(r"y\,(h_{\mathrm{ext}} - h_{\mathrm{dren}}) = h_{\mathrm{sal}} - h_{\mathrm{ent}}")
        st.markdown(
            "Todo se cuenta por kilogramo de vapor que pasa por la caldera: cada tramo pesa "
            "según su caudal (en el ejemplo 10-5, por la turbina después de la extracción pasa "
            "1 − y):"
        )
        st.latex(r"w_T = (h_5 - h_6) + (1 - y)\,(h_6 - h_7)")
        st.markdown(
            "La regeneración sube T̄_H (el agua llega más caliente a la caldera), pero η queda "
            "por debajo de 1 − T_C/T̄_H: en los calentadores se mezclan o intercambian calor "
            "corrientes a distinta temperatura, y eso genera entropía. Con turbina real, η_T se "
            "mide desde la entrada de cada turbina: cada extracción y la salida caen sobre la "
            "misma línea de expansión, h_k = h_e − η_T·(h_e − h_ks)."
        )
        st.markdown(
            "**Ciclo real** (Cengel §10-3): por la fricción el fluido pierde presión en la "
            "caldera, el condensador y las cañerías, así que la bomba tiene que entregar más "
            "presión que la que recibe la turbina; las cañerías pierden calor hacia el ambiente "
            "y el condensado se subenfría para que la bomba no cavite. El primer principio suma "
            "el calor perdido:"
        )
        st.latex(r"w_{\mathrm{neto}} = q_H - q_C - q_{\text{pérd}}")
        st.markdown(
            "**Calentadores cerrados reales**: el agua sale a T_sat(p_ext) − TTD (con "
            "**desrecalentador** el TTD puede ser negativo: la extracción sobrecalentada calienta "
            "el agua por encima de la saturación) y, con **subenfriador de drenaje**, el drenaje "
            "sale a la temperatura del agua que entra más el DCA:"
        )
        st.latex(r"\mathrm{TTD} = T_{\mathrm{sat}}(p_{\mathrm{ext}}) - T_{\mathrm{sal}}")
        st.latex(r"\mathrm{DCA} = T_{\mathrm{dren}} - T_{\mathrm{ent}}")
        st.markdown(
            "Si el drenaje de un cerrado intermedio se **bombea hacia adelante**, la mezcla "
            "cambia el agua que entra al calentador siguiente y las fracciones de extracción "
            "quedan acopladas: los balances se resuelven juntos, como un sistema."
        )
        st.markdown(
            "**Ciclo de Rankine orgánico (ORC)**: con un fluido orgánico se aprovechan fuentes de "
            "calor de baja temperatura (geotermia, calor residual, solar; Quoilin et al., 2013, "
            "*Renewable and Sustainable Energy Reviews* 22, 168–186). Según la pendiente de su "
            "curva de vapor saturado en T–s, el fluido es **húmedo** (ξ < 0, como el agua: la "
            "expansión entra en la campana), **seco** (ξ > 0: termina sobrecalentada) o **casi "
            "isoentrópico** (Chen, Goswami y Stefanakos, 2010, *Renewable and Sustainable Energy "
            "Reviews* 14, 3059–3067):"
        )
        st.latex(r"\xi = \frac{ds_g}{dT}")
        st.markdown(
            "Con un fluido seco, el **recuperador** pasa calor del escape de la turbina al "
            "líquido que sale de la bomba; su efectividad compara lo que pasa con lo máximo que "
            "podría pasar (la misma idea que el regenerador del ciclo Brayton, Cengel §9-9):"
        )
        st.latex(r"\varepsilon = \frac{q}{q_{\text{máx}}}")
        st.markdown(
            "La **relación de trabajo de retroceso** BWR = w_B / w_T es chica en el Rankine "
            "(el líquido se bombea con poco trabajo), a diferencia del ciclo Brayton."
        )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Rankine", page_icon="♨️", layout="centered")

st.subheader(SUBJECT)
st.title("♨️ Ciclo de Rankine")
st.markdown(
    "Ciclo de potencia de vapor: bomba, caldera, turbina y condensador. Simple, con "
    "recalentamiento o con regeneración (calentadores de agua de alimentación abiertos o "
    "cerrados), ideal o real (con pérdidas de carga y de calor), con agua o con un fluido "
    "orgánico (ORC, con recuperador). Lo resuelve [TESPy](https://tespy.readthedocs.io) como "
    "una red de componentes, y el procedimiento muestra cómo se haría con las tablas."
)
_render_theory()
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Rankine")
render_units_selector()
system = get_current_system()

ex = _example_index()
inputs, cooling_T = _read_inputs(ex, system)

# Se calcula al elegir un ejemplo y cada vez que se toca el botón.
if st.session_state.get("rk_example_prev") != ex:
    st.session_state["rk_example_prev"] = ex
    st.session_state["rk_inputs"] = inputs
if st.button("Calcular ciclo", key="rk_btn", type="primary"):
    st.session_state["rk_inputs"] = inputs

computed: RankineInputs | None = st.session_state.get("rk_inputs")
if computed is not None:
    if computed != inputs:
        st.caption("Cambiaste datos: tocá «Calcular ciclo» para actualizar el resultado.")
    try:
        result = _solve_cached(computed)
    except ValueError as exc:
        _render_error(exc)
    else:
        cooling: CoolingWaterResult | None = None
        cooling_error = None
        if cooling_T is not None:
            try:
                cooling = condenser_cooling_water(result, *cooling_T)
            except ValueError as exc:
                cooling_error = str(exc)
        st.markdown("### Resultado")
        note = RANKINE_EXAMPLE_NOTES.get(_EXAMPLES[ex])
        if note:
            st.caption(f"📘 Sobre este ejemplo: {note}")
        _render_metrics(result, system, cooling)
        if cooling_error:
            st.warning(cooling_error)
        _render_extractions(result, system)
        st.markdown("#### Estados")
        _render_states_table(result, system)
        _cycle_diagram(result, system)
        _render_procedure(result, system, cooling)
        _render_tespy(result, system)
        _render_export(result, system, cooling)
        _render_sweeps(computed, result, system)
