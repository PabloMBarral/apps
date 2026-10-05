"""Diagramas de propiedades termodinámicas — Fase 1.5a.

Wrapper tipado sobre :mod:`fluprodia` (v4.2+) para construir diagramas
log p–h, T–s, h–s y p–log v de los fluidos puros/pseudo-puros de
:data:`core.fluids.SUPPORTED_FLUIDS`. El módulo **no importa Streamlit**:
el cache `@st.cache_resource` vive en :mod:`ui.diagrams`.

La API pública recibe siempre valores en **SI** (Pa, K, J/kg, J/(kg·K))
y devuelve overlays de procesos con arrays también en SI. La capa UI
se encarga de convertir al sistema activo del usuario en el momento
del render.

Por qué `FLUPRODIA_UNITS` vive acá y no en :mod:`core.units_system`
------------------------------------------------------------------
Es metadata específica de fluprodia (strings pint-compatibles, con
quirks como ``Btu/(lb*degR)`` en sistema inglés) que el resto del
proyecto no necesita conocer.

Cita
----
Witte, F. *fluprodia: Fluid Property Diagrams*.
<https://github.com/fwitte/fluprodia> — Apache 2.0.

Wraps CoolProp para la EOS:
Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014).
"Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the
Open-Source Thermophysical Property Library CoolProp".
*Industrial & Engineering Chemistry Research*, 53(6), 2498-2508.
DOI: 10.1021/ie4033999
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import CoolProp.CoolProp as cp
import numpy as np
from fluprodia import FluidPropertyDiagram

from core.fluids import SUPPORTED_FLUIDS, StatePoint
from core.units_system import UnitSystem, convert_from_si

# ---------------------------------------------------------------------
# Tipos públicos
# ---------------------------------------------------------------------

DiagramType = Literal["logph", "Ts", "hs", "plogv"]
SUPPORTED_DIAGRAM_TYPES: tuple[DiagramType, ...] = ("logph", "Ts", "hs", "plogv")


@dataclass(frozen=True)
class DiagramSpec:
    """Especificación inmutable de un diagrama: fluido + sistema de unidades."""

    fluid: str
    system: UnitSystem


@dataclass(frozen=True)
class FluidRange:
    """Rango operativo de un fluido para la ventana inicial del diagrama.

    Valores en SI (K, Pa). La conversión al sistema visual vive en el
    consumidor.
    """

    T_K_min: float
    T_K_max: float
    p_Pa_min: float
    p_Pa_max: float


# ---------------------------------------------------------------------
# Mapeo a strings pint-compatibles que fluprodia acepta
# ---------------------------------------------------------------------
#
# Nota crítica del research:
#   - En sistema "Inglés", la entropía debe pasarse como ``Btu/(lb*degR)``
#     (o ``Btu/lb/degR``); ``Btu/lbR`` rompe pint.
#   - "K" y "kgK" son sufijos compactos válidos de pint y fluprodia
#     internamente los expande a "K" y "kg·K".

FLUPRODIA_UNITS: dict[UnitSystem, dict[str, str]] = {
    "SI": {
        "T": "K",
        "p": "Pa",
        "h": "J/kg",
        "s": "J/kgK",
        "vol": "m^3/kg",
    },
    "Técnico": {
        "T": "°C",
        "p": "bar",
        "h": "kJ/kg",
        "s": "kJ/kgK",
        "vol": "m^3/kg",
    },
    "Inglés": {
        "T": "°F",
        "p": "psi",
        "h": "Btu/lb",
        "s": "Btu/(lb*degR)",
        "vol": "ft^3/lb",
    },
}


# ---------------------------------------------------------------------
# Rangos operativos por fluido (en SI)
# ---------------------------------------------------------------------
#
# Una sola fuente de verdad para la ventana inicial. Pensados como los
# rangos didácticos típicos de Cengel: agua hasta supercrítico medio,
# refrigerantes en la ventana de uso de heladeras / aires, CO2 con foco
# en la zona crítica/transcrítica.

DEFAULT_RANGES: dict[str, FluidRange] = {
    "Water": FluidRange(T_K_min=273.15, T_K_max=873.15, p_Pa_min=1.0e3, p_Pa_max=3.0e7),
    "R134a": FluidRange(T_K_min=223.15, T_K_max=423.15, p_Pa_min=1.0e4, p_Pa_max=4.0e6),
    "R410A": FluidRange(T_K_min=223.15, T_K_max=423.15, p_Pa_min=1.0e4, p_Pa_max=5.0e6),
    "R1234yf": FluidRange(T_K_min=223.15, T_K_max=423.15, p_Pa_min=1.0e4, p_Pa_max=4.0e6),
    "Ammonia": FluidRange(T_K_min=223.15, T_K_max=473.15, p_Pa_min=1.0e4, p_Pa_max=1.0e7),
    "CarbonDioxide": FluidRange(T_K_min=218.15, T_K_max=423.15, p_Pa_min=5.0e5, p_Pa_max=2.0e7),
    "Air": FluidRange(T_K_min=173.15, T_K_max=1273.15, p_Pa_min=1.0e4, p_Pa_max=1.0e7),
}


# Ventana genérica de respaldo (la que se usaba antes de calcularla por fluido).
_FALLBACK_WINDOW_SI: dict[str, tuple[float, float]] = {
    "h": (0.0, 4.0e6),
    "s": (0.0, 1.0e4),
    "vol": (1.0e-4, 1.0e1),
}


def axis_window_si(fluid: str) -> dict[str, tuple[float, float]]:
    """Ventana inicial (en SI) de cada propiedad de eje para ``fluid``.

    ``T`` y ``p`` salen de :data:`DEFAULT_RANGES`. ``h``, ``s`` y ``vol``
    se calculan con CoolProp en las cuatro esquinas del rectángulo (T, p)
    del fluido: el mínimo cae en (T_min, p_max) — líquido frío comprimido —
    y el máximo en (T_max, p_min) — vapor caliente a baja presión. Así la
    campana de cada fluido ocupa una fracción razonable del gráfico (con
    los rangos fijos pensados para agua, la campana del R134a quedaba en
    ~10 % del ancho del log p–h, y en p–log v el vapor de agua a menos de
    ~0,15 bar quedaba fuera de la ventana).

    Devuelve un dict ``{prop: (min, max)}`` con las keys
    ``'T', 'p', 'h', 's', 'vol'``.
    """
    _validate_fluid(fluid)
    rng = DEFAULT_RANGES[fluid]
    # CoolProp no resuelve por debajo de su T mínima (para el agua, el
    # punto triple 0,01 °C, apenas arriba del T_K_min = 0 °C del rango).
    t_low = max(rng.T_K_min, float(cp.PropsSI("Tmin", fluid)) + 0.01)
    values: dict[str, list[float]] = {"h": [], "s": [], "vol": []}
    for t_K in (t_low, rng.T_K_max):
        for p_Pa in (rng.p_Pa_min, rng.p_Pa_max):
            try:
                h = float(cp.PropsSI("H", "T", t_K, "P", p_Pa, fluid))
                s = float(cp.PropsSI("S", "T", t_K, "P", p_Pa, fluid))
                rho = float(cp.PropsSI("D", "T", t_K, "P", p_Pa, fluid))
            except ValueError:
                continue
            values["h"].append(h)
            values["s"].append(s)
            values["vol"].append(1.0 / rho)

    window: dict[str, tuple[float, float]] = {
        "T": (rng.T_K_min, rng.T_K_max),
        "p": (rng.p_Pa_min, rng.p_Pa_max),
    }
    for prop, vals in values.items():
        if vals:
            window[prop] = (min(vals), max(vals))
        else:  # pragma: no cover — ninguna esquina resolvió; ventana genérica
            window[prop] = _FALLBACK_WINDOW_SI[prop]
    return window


def expand_window(lo: float, hi: float, values: Iterable[float]) -> tuple[float, float]:
    """Agranda ``(lo, hi)`` para que contenga todos los ``values`` finitos.

    Pensado para que los estados y procesos superpuestos al diagrama
    nunca queden fuera de la ventana visible.
    """
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return lo, hi
    return min(lo, *finite), max(hi, *finite)


def pad_window(lo: float, hi: float, *, log: bool, frac: float = 0.05) -> tuple[float, float]:
    """Agrega un margen ``frac`` a cada lado de la ventana.

    En ejes logarítmicos el margen se aplica sobre ``log10`` (requiere
    ``lo > 0``). Si la ventana es degenerada (``lo == hi``), abre un
    margen relativo al valor.
    """
    if log:
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(f"Un eje logarítmico requiere límites positivos (recibí {lo}, {hi}).")
        log_lo, log_hi = math.log10(lo), math.log10(hi)
        span = (log_hi - log_lo) or 1.0
        return 10.0 ** (log_lo - frac * span), 10.0 ** (log_hi + frac * span)
    span = (hi - lo) or (abs(lo) or 1.0)
    return lo - frac * span, hi + frac * span


# ---------------------------------------------------------------------
# Mapa diagrama → (prop_x, prop_y, x_logscale, y_logscale)
# ---------------------------------------------------------------------
#
# Las keys 'h','T','s','vol','p','Q' coinciden con las que devuelve
# ``calc_individual_isoline`` y con las que acepta ``draw_isolines_plotly``.
# Las escalas tienen que coincidir con las que dibuja fluprodia
# (``FluidPropertyDiagram.supported_diagrams``): en "p–log v" solo v es
# logarítmico, p es lineal.

AXIS_MAP: dict[DiagramType, tuple[str, str, bool, bool]] = {
    "logph": ("h", "p", False, True),
    "Ts": ("s", "T", False, False),
    "hs": ("s", "h", False, False),
    "plogv": ("vol", "p", True, False),
}

# Para cada axis-property, qué QuantityKind de units_system aplica.
# Q (calidad) no convierte; vol no se muestra en la mayoría de overlays.
_PROP_TO_KIND: dict[str, str] = {
    "h": "specific_enthalpy",
    "T": "temperature",
    "s": "specific_entropy",
    "p": "pressure",
    "vol": "specific_volume",
}


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------


def _validate_fluid(fluid: str) -> None:
    if fluid not in SUPPORTED_FLUIDS:
        raise ValueError(
            f"Fluido '{fluid}' no soportado para diagramas. Usá uno de: {SUPPORTED_FLUIDS}."
        )


def _validate_system(system: str) -> None:
    if system not in FLUPRODIA_UNITS:
        raise ValueError(
            f"Sistema de unidades '{system}' no soportado. Usá uno de: {list(FLUPRODIA_UNITS)}."
        )


def _validate_diagram_type(diagram_type: str) -> None:
    if diagram_type not in AXIS_MAP:
        raise ValueError(
            f"Tipo de diagrama '{diagram_type}' no soportado. Usá uno de: {list(AXIS_MAP)}."
        )


# ---------------------------------------------------------------------
# Construcción del diagrama
# ---------------------------------------------------------------------


def _isoline_grid(fluid: str, system: UnitSystem = "Técnico") -> dict[str, np.ndarray]:
    """Isolíneas "redondas" (T, p, Q) del fluido, en las unidades de ``system``.

    ``FluidPropertyDiagram.set_isolines`` interpreta los valores en las
    unidades activas (las de ``set_unit_system``), así que se generan
    directamente en las del diagrama: °C y bar (Técnico), K y Pa (SI),
    °F y psia (Inglés). Hasta la versión 0.9.0 se generaban siempre en
    °C / bar: en SI quedaban isotermas de 0 a 600 K e isobaras de 0,01 a
    1000 Pa (y con aire, isotermas en kelvin negativos que hacían fallar
    a CoolProp); en Inglés, isotermas de 0 a 600 °F.
    """
    rng = DEFAULT_RANGES[fluid]
    # Temperatura: paso de 25 o 50 grados (°C o K) según el rango; en °F,
    # el doble (el grado Fahrenheit es 1/1,8 del Celsius).
    step_T = 25.0 if rng.T_K_max - rng.T_K_min <= 400.0 else 50.0
    if system == "Inglés":
        step_T *= 2.0
    T_min = convert_from_si(rng.T_K_min, "temperature", system)
    T_max = convert_from_si(rng.T_K_max, "temperature", system)
    T_isolines = np.arange(
        np.ceil(T_min / step_T) * step_T,
        np.floor(T_max / step_T) * step_T + step_T / 2,
        step_T,
    )
    # Presión: una isobara por década que cubra el rango.
    p_min = convert_from_si(rng.p_Pa_min, "pressure", system)
    p_max = convert_from_si(rng.p_Pa_max, "pressure", system)
    log_min = int(np.floor(np.log10(p_min)))
    log_max = int(np.ceil(np.log10(p_max)))
    p_isolines = np.array([10.0**k for k in range(log_min, log_max + 1)])
    # Calidad: 11 puntos en [0, 1].
    Q_isolines = np.linspace(0.0, 1.0, 11)
    return {"T": T_isolines, "p": p_isolines, "Q": Q_isolines}


def build_diagram(spec: DiagramSpec) -> FluidPropertyDiagram:
    """Construye un :class:`FluidPropertyDiagram` listo para dibujar.

    Hace ``set_unit_system``, define isolíneas limpias y llama a
    ``calc_isolines()``. La llamada es costosa (~0.5–1.1 s según fluido);
    cacheala desde la UI con ``@st.cache_resource``.

    Raises
    ------
    ValueError
        Si ``spec.fluid`` no está en :data:`SUPPORTED_FLUIDS`, o el
        sistema no está en :data:`FLUPRODIA_UNITS`.
    """
    _validate_fluid(spec.fluid)
    _validate_system(spec.system)

    diagram = FluidPropertyDiagram(spec.fluid, backend=None)
    diagram.set_unit_system(**FLUPRODIA_UNITS[spec.system])

    # Después de set_unit_system: los valores se leen en esas unidades.
    isolines = _isoline_grid(spec.fluid, spec.system)
    diagram.set_isolines(**isolines)
    diagram.calc_isolines()
    return diagram


# ---------------------------------------------------------------------
# Procesos (wrappers sobre calc_individual_isoline)
# ---------------------------------------------------------------------


def _process_to_si(raw: dict[str, np.ndarray], spec: DiagramSpec) -> dict[str, np.ndarray]:
    """Convierte el dict de salida de fluprodia (en unidades del diagrama)
    a SI por cada magnitud. ``Q`` (calidad) queda igual."""
    out: dict[str, np.ndarray] = {}
    system = spec.system
    for prop, arr in raw.items():
        if prop == "Q":
            out["Q"] = np.asarray(arr, dtype=float)
            continue
        kind = _PROP_TO_KIND.get(prop)
        if kind is None:
            out[prop] = np.asarray(arr, dtype=float)
            continue
        # Convertimos punto a punto user→SI usando la tabla de units_system.
        # Para arrays vectorizamos llamando convert_to_si en bucle (n=100).
        from core.units_system import convert_to_si

        out[prop] = np.array(
            [convert_to_si(float(v), kind, system) for v in arr],  # type: ignore[arg-type]
            dtype=float,
        )
    return out


def isentropic_process(
    diagram: FluidPropertyDiagram,
    spec: DiagramSpec,
    *,
    s_J_per_kg_K: float,
    p_start_Pa: float,
    p_end_Pa: float,
) -> dict[str, np.ndarray]:
    """Proceso isoentrópico s = cte entre dos presiones.

    Parámetros
    ----------
    diagram :
        Diagrama ya construido (con `set_unit_system` aplicado).
    spec :
        Para conocer el sistema de unidades del diagrama y convertir
        de SI a esas unidades antes de invocar fluprodia.
    s_J_per_kg_K, p_start_Pa, p_end_Pa :
        Inputs en SI.

    Devuelve
    --------
    dict con keys ``'p', 'T', 'h', 's', 'vol', 'Q'`` y arrays numpy
    **en SI**.
    """
    s_user = convert_from_si(s_J_per_kg_K, "specific_entropy", spec.system)
    p_start_user = convert_from_si(p_start_Pa, "pressure", spec.system)
    p_end_user = convert_from_si(p_end_Pa, "pressure", spec.system)
    raw = diagram.calc_individual_isoline(
        isoline_property="s",
        isoline_value=float(s_user),
        starting_point_property="p",
        starting_point_value=float(p_start_user),
        ending_point_property="p",
        ending_point_value=float(p_end_user),
    )
    return _process_to_si(raw, spec)


def isobaric_process(
    diagram: FluidPropertyDiagram,
    spec: DiagramSpec,
    *,
    p_Pa: float,
    t_start_K: float,
    t_end_K: float,
) -> dict[str, np.ndarray]:
    """Proceso isobárico p = cte entre dos temperaturas (en SI)."""
    p_user = convert_from_si(p_Pa, "pressure", spec.system)
    t_start_user = convert_from_si(t_start_K, "temperature", spec.system)
    t_end_user = convert_from_si(t_end_K, "temperature", spec.system)
    raw = diagram.calc_individual_isoline(
        isoline_property="p",
        isoline_value=float(p_user),
        starting_point_property="T",
        starting_point_value=float(t_start_user),
        ending_point_property="T",
        ending_point_value=float(t_end_user),
    )
    return _process_to_si(raw, spec)


def isobaric_process_by_enthalpy(
    diagram: FluidPropertyDiagram,
    spec: DiagramSpec,
    *,
    p_Pa: float,
    h_start_J_per_kg: float,
    h_end_J_per_kg: float,
) -> dict[str, np.ndarray]:
    """Proceso isobárico p = cte entre dos entalpías (en SI).

    A diferencia de :func:`isobaric_process` (parametrizada por T), esta
    versión recorre bien el tramo dentro de la campana, donde la
    temperatura es constante y la entalpía no: es la que hace falta para
    dibujar una caldera o un condensador (p. ej. 2→3 en un Rankine).
    """
    p_user = convert_from_si(p_Pa, "pressure", spec.system)
    h_start_user = convert_from_si(h_start_J_per_kg, "specific_enthalpy", spec.system)
    h_end_user = convert_from_si(h_end_J_per_kg, "specific_enthalpy", spec.system)
    raw = diagram.calc_individual_isoline(
        isoline_property="p",
        isoline_value=float(p_user),
        starting_point_property="h",
        starting_point_value=float(h_start_user),
        ending_point_property="h",
        ending_point_value=float(h_end_user),
    )
    return _process_to_si(raw, spec)


def isothermal_process(
    diagram: FluidPropertyDiagram,
    spec: DiagramSpec,
    *,
    t_K: float,
    p_start_Pa: float,
    p_end_Pa: float,
) -> dict[str, np.ndarray]:
    """Proceso isotérmico T = cte entre dos presiones (en SI)."""
    t_user = convert_from_si(t_K, "temperature", spec.system)
    p_start_user = convert_from_si(p_start_Pa, "pressure", spec.system)
    p_end_user = convert_from_si(p_end_Pa, "pressure", spec.system)
    raw = diagram.calc_individual_isoline(
        isoline_property="T",
        isoline_value=float(t_user),
        starting_point_property="p",
        starting_point_value=float(p_start_user),
        ending_point_property="p",
        ending_point_value=float(p_end_user),
    )
    return _process_to_si(raw, spec)


# ---------------------------------------------------------------------
# Conversión SI → coordenadas del diagrama (para overlays en la UI)
# ---------------------------------------------------------------------


def state_to_diagram_coords(
    state: StatePoint,
    diagram_type: DiagramType,
    system: UnitSystem,
) -> tuple[float, float]:
    """Convierte un StatePoint en coordenadas ``(x, y)`` del diagrama.

    Devuelve los valores **en las unidades del sistema** (no en SI),
    listos para pasar a plotly.
    """
    _validate_diagram_type(diagram_type)
    _validate_system(system)
    prop_x, prop_y, _x_log, _y_log = AXIS_MAP[diagram_type]
    x = _state_property(state, prop_x, system)
    y = _state_property(state, prop_y, system)
    return x, y


def process_to_diagram_coords(
    process_si: dict[str, np.ndarray],
    diagram_type: DiagramType,
    system: UnitSystem,
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte un dict de proceso (SI) a arrays ``(x, y)`` del diagrama."""
    _validate_diagram_type(diagram_type)
    _validate_system(system)
    prop_x, prop_y, _x_log, _y_log = AXIS_MAP[diagram_type]
    x = _process_array_in_system(process_si, prop_x, system)
    y = _process_array_in_system(process_si, prop_y, system)
    return x, y


def state_property_si(state: StatePoint, prop: str) -> float:
    """Valor en SI de la propiedad de eje ``prop`` (``'T','p','h','s','vol'``).

    Raises
    ------
    ValueError
        Si ``prop`` no es una propiedad de eje, o si se pide ``'vol'`` de
        un ``StatePoint`` construido a mano sin volumen específico.
    """
    if prop == "vol":
        if state.v_m3_per_kg is None:
            raise ValueError(
                "El estado no trae volumen específico (v_m3_per_kg=None), así "
                "que no se puede ubicar en el diagrama p–log v. Construilo con "
                "core.fluids.state_from_pair, que siempre calcula v = 1/ρ."
            )
        return state.v_m3_per_kg
    si_value = {
        "T": state.T_K,
        "p": state.P_Pa,
        "h": state.h_J_per_kg,
        "s": state.s_J_per_kg_K,
    }.get(prop)
    if si_value is None:
        raise ValueError(
            f"Propiedad '{prop}' no es un eje de diagrama. Usá una de: {sorted(_PROP_TO_KIND)}."
        )
    return si_value


def _state_property(state: StatePoint, prop: str, system: UnitSystem) -> float:
    """Devuelve el valor de ``prop`` para ``state`` en unidades del sistema."""
    kind = _PROP_TO_KIND[prop]
    return convert_from_si(state_property_si(state, prop), kind, system)  # type: ignore[arg-type]


def _process_array_in_system(
    process_si: dict[str, np.ndarray], prop: str, system: UnitSystem
) -> np.ndarray:
    """Convierte el array SI de ``prop`` a unidades del sistema."""
    if prop not in process_si:
        raise ValueError(
            f"El proceso no incluye la propiedad '{prop}'. Disponibles: {sorted(process_si)}."
        )
    arr_si = np.asarray(process_si[prop], dtype=float)
    if prop == "Q":
        return arr_si  # calidad es adimensional, no convierte
    kind = _PROP_TO_KIND[prop]
    return np.array(
        [convert_from_si(float(v), kind, system) for v in arr_si],  # type: ignore[arg-type]
        dtype=float,
    )


# ---------------------------------------------------------------------
# Helper para overlays plotly desde la UI
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessOverlay:
    """Curva conceptual (isolínea o segmento) para superponer al diagrama.

    Atributos
    ---------
    name : str
        Etiqueta de la leyenda.
    color : str
        Color CSS (ej. ``'red'``, ``'#1f77b4'``).
    dash : Literal['solid','dot','dash','dashdot']
        Estilo de línea de plotly.
    coords_si : dict[str, np.ndarray]
        Arrays en SI con claves coincidentes con las de los procesos
        (``'p', 'T', 'h', 's', 'vol', 'Q'``). Las arrays que no apliquen
        a algún diagrama pueden venir con NaN.
    """

    name: str
    color: str
    dash: str
    coords_si: dict[str, np.ndarray]


def linear_segment_overlay(
    *,
    name: str,
    color: str,
    dash: str,
    start: StatePoint,
    end: StatePoint,
) -> ProcessOverlay:
    """Construye un :class:`ProcessOverlay` de 2 puntos (segmento recto).

    El volumen específico se toma de los estados cuando lo traen; si
    alguno no lo tiene, ese extremo queda en NaN (plotly no lo dibuja).
    """

    def _vol(state: StatePoint) -> float:
        return np.nan if state.v_m3_per_kg is None else state.v_m3_per_kg

    return ProcessOverlay(
        name=name,
        color=color,
        dash=dash,
        coords_si={
            "p": np.array([start.P_Pa, end.P_Pa], dtype=float),
            "T": np.array([start.T_K, end.T_K], dtype=float),
            "h": np.array([start.h_J_per_kg, end.h_J_per_kg], dtype=float),
            "s": np.array([start.s_J_per_kg_K, end.s_J_per_kg_K], dtype=float),
            "vol": np.array([_vol(start), _vol(end)], dtype=float),
            "Q": np.array([start.x, end.x], dtype=float),
        },
    )


SegmentKind = Literal["isobaric", "isentropic", "straight"]

# Tolerancia relativa para considerar que dos estados comparten p o s.
_SAME_PROPERTY_RTOL = 1e-4


def _shares(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= _SAME_PROPERTY_RTOL * max(abs(a), abs(b), scale)


def _straight_coords(start: StatePoint, end: StatePoint) -> dict[str, np.ndarray]:
    return linear_segment_overlay(name="", color="", dash="", start=start, end=end).coords_si


def segment_between(
    diagram: FluidPropertyDiagram,
    spec: DiagramSpec,
    start: StatePoint,
    end: StatePoint,
) -> tuple[SegmentKind, dict[str, np.ndarray]]:
    """Trayectoria para unir dos estados consecutivos de un ciclo (en SI).

    - Misma presión → isobárica real (caldera, condensador, evaporador).
    - Misma entropía → isoentrópica real (bomba, turbina o compresor
      ideales).
    - Si no comparten ninguna, o si fluprodia no puede trazarla → segmento
      recto, que es **solo una referencia visual** (p. ej. una turbina
      real o una válvula de expansión no son procesos cuasiestáticos con
      una trayectoria definida en el diagrama).
    """
    if _shares(start.P_Pa, end.P_Pa, scale=1.0) and not _shares(
        start.h_J_per_kg, end.h_J_per_kg, scale=1.0e3
    ):
        try:
            coords = isobaric_process_by_enthalpy(
                diagram,
                spec,
                p_Pa=start.P_Pa,
                h_start_J_per_kg=start.h_J_per_kg,
                h_end_J_per_kg=end.h_J_per_kg,
            )
            return "isobaric", coords
        except Exception:  # fluprodia/CoolProp no pudo: se cae a la recta
            pass
    if _shares(start.s_J_per_kg_K, end.s_J_per_kg_K, scale=1.0e3) and not _shares(
        start.P_Pa, end.P_Pa, scale=1.0
    ):
        try:
            coords = isentropic_process(
                diagram,
                spec,
                s_J_per_kg_K=start.s_J_per_kg_K,
                p_start_Pa=start.P_Pa,
                p_end_Pa=end.P_Pa,
            )
            return "isentropic", coords
        except Exception:
            pass
    return "straight", _straight_coords(start, end)


def _join_with_gaps(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatena tramos separándolos con NaN (plotly corta la línea ahí)."""
    keys = ("p", "T", "h", "s", "vol", "Q")
    out: dict[str, list[float]] = {k: [] for k in keys}
    for i, chunk in enumerate(chunks):
        if i:
            for k in keys:
                out[k].append(np.nan)
        n = len(chunk["p"])
        for k in keys:
            values = chunk.get(k)
            out[k].extend(np.full(n, np.nan) if values is None else np.asarray(values, dtype=float))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def cycle_overlays(
    diagram: FluidPropertyDiagram,
    spec: DiagramSpec,
    states: Sequence[StatePoint],
    *,
    close: bool = True,
) -> list[ProcessOverlay]:
    """Overlays para unir ``states`` en orden (y cerrar el ciclo si ``close``).

    Devuelve hasta dos curvas: los tramos isobáricos/isoentrópicos reales
    (línea llena) y las uniones rectas de referencia (punteada). Ver
    :func:`segment_between`.
    """
    seq = list(states)
    if len(seq) < 2:
        return []
    pairs = list(zip(seq[:-1], seq[1:], strict=True))
    if close and len(seq) > 2:
        pairs.append((seq[-1], seq[0]))
    real: list[dict[str, np.ndarray]] = []
    straight: list[dict[str, np.ndarray]] = []
    for start, end in pairs:
        kind, coords = segment_between(diagram, spec, start, end)
        (straight if kind == "straight" else real).append(coords)
    overlays: list[ProcessOverlay] = []
    if real:
        overlays.append(
            ProcessOverlay(
                name="procesos a p o s constante",
                color="#444444",
                dash="solid",
                coords_si=_join_with_gaps(real),
            )
        )
    if straight:
        overlays.append(
            ProcessOverlay(
                name="uniones rectas (referencia)",
                color="#444444",
                dash="dot",
                coords_si=_join_with_gaps(straight),
            )
        )
    return overlays


# ---------------------------------------------------------------------
# Serialización (delegada a fluprodia)
# ---------------------------------------------------------------------


def to_json(diagram: FluidPropertyDiagram, path: str) -> None:
    """Persiste el diagrama (incluyendo isolíneas calculadas) en disco."""
    diagram.to_json(path)


def from_json(path: str) -> FluidPropertyDiagram:
    """Carga un diagrama previamente persistido con :func:`to_json`."""
    return FluidPropertyDiagram.from_json(path)


# Suprime warning de import no usado (Any se reserva para el futuro).
_ = Any
