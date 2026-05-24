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

from dataclasses import dataclass
from typing import Any, Literal

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


# ---------------------------------------------------------------------
# Mapa diagrama → (prop_x, prop_y, x_logscale, y_logscale)
# ---------------------------------------------------------------------
#
# Las keys 'h','T','s','vol','p','Q' coinciden con las que devuelve
# ``calc_individual_isoline`` y con las que acepta ``draw_isolines_plotly``.

AXIS_MAP: dict[DiagramType, tuple[str, str, bool, bool]] = {
    "logph": ("h", "p", False, True),
    "Ts": ("s", "T", False, False),
    "hs": ("s", "h", False, False),
    "plogv": ("vol", "p", True, True),
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


def _isoline_grid(fluid: str) -> dict[str, np.ndarray]:
    """Retorna las isolíneas "limpias" (T, p, Q) para el fluido.

    Valores expresados en unidades del sistema **Técnico** porque
    fluprodia tiene `set_unit_system` configurado y este helper solo
    se llama después de eso, con system fijo a `Técnico`. Para los
    otros sistemas, la conversión la hace fluprodia internamente
    al cambiar la unidad de la magnitud.
    """
    rng = DEFAULT_RANGES[fluid]
    # Temperatura en °C: paso 25 o 50 °C según el rango.
    T_min_C = rng.T_K_min - 273.15
    T_max_C = rng.T_K_max - 273.15
    span = T_max_C - T_min_C
    step_T = 25.0 if span <= 400.0 else 50.0
    T_isolines = np.arange(
        np.ceil(T_min_C / step_T) * step_T,
        np.floor(T_max_C / step_T) * step_T + step_T / 2,
        step_T,
    )
    # Presión en bar: una década, decades cubiertas según rango.
    p_min_bar = rng.p_Pa_min * 1.0e-5
    p_max_bar = rng.p_Pa_max * 1.0e-5
    log_min = int(np.floor(np.log10(p_min_bar)))
    log_max = int(np.ceil(np.log10(p_max_bar)))
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

    isolines = _isoline_grid(spec.fluid)
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


def _state_property(state: StatePoint, prop: str, system: UnitSystem) -> float:
    """Devuelve el valor de ``prop`` para ``state`` en unidades del sistema."""
    si_value = {
        "T": state.T_K,
        "p": state.P_Pa,
        "h": state.h_J_per_kg,
        "s": state.s_J_per_kg_K,
    }.get(prop)
    if si_value is None:
        raise ValueError(
            f"Propiedad '{prop}' no extraíble de StatePoint (faltaría 'vol'). "
            f"Para diagramas p–v use process_to_diagram_coords con datos de "
            f"un proceso, o computá v=1/ρ desde CoolProp aparte."
        )
    kind = _PROP_TO_KIND[prop]
    return convert_from_si(si_value, kind, system)  # type: ignore[arg-type]


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
    """Construye un :class:`ProcessOverlay` de 2 puntos (segmento recto)."""
    return ProcessOverlay(
        name=name,
        color=color,
        dash=dash,
        coords_si={
            "p": np.array([start.P_Pa, end.P_Pa], dtype=float),
            "T": np.array([start.T_K, end.T_K], dtype=float),
            "h": np.array([start.h_J_per_kg, end.h_J_per_kg], dtype=float),
            "s": np.array([start.s_J_per_kg_K, end.s_J_per_kg_K], dtype=float),
            "vol": np.array([np.nan, np.nan], dtype=float),
            "Q": np.array([start.x, end.x], dtype=float),
        },
    )


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
