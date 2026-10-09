"""Propiedades termofísicas — wrappers tipados sobre CoolProp.

Toda función opera en unidades SI: Pa, K, J/kg, J/(kg·K). La conversión
desde/hacia unidades didácticas (bar, °C, kJ/kg) es responsabilidad de
la capa de UI. Este módulo no importa Streamlit.

Dos niveles de API:

- :func:`state_from_pair` → :class:`StatePoint` (T, p, h, s, x, v). Es
  la que usan los módulos de cálculo (isoentrópicos, diagramas).
- :func:`fluid_state_from_pair` → :class:`FluidState`: el estado
  completo que necesita el alumno — región (líquido comprimido, vapor
  húmedo, vapor sobrecalentado, supercrítico), u, ρ, cp, cv, velocidad
  del sonido, propiedades de transporte, factor de compresibilidad y las
  propiedades de saturación a la presión y a la temperatura del estado
  (lo mismo que se lee en las tablas A-4/A-5 de Cengel). Valida rangos
  y traduce los errores de CoolProp a mensajes en castellano.

Cita
----
Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014).
"Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the
Open-Source Thermophysical Property Library CoolProp".
*Industrial & Engineering Chemistry Research*, 53(6), 2498-2508.
DOI: 10.1021/ie4033999

Para el agua, CoolProp implementa la formulación IAPWS-95:
Wagner, W., & Pruß, A. (2002). "The IAPWS Formulation 1995 for the
Thermodynamic Properties of Ordinary Water Substance for General and
Scientific Use". *J. Phys. Chem. Ref. Data*, 31(2), 387-535.
DOI: 10.1063/1.1461829

Convención de regiones y nomenclatura f/g: Cengel, Y. A., & Boles, M. A.
(2015). *Thermodynamics: An Engineering Approach* (8th ed., cap. 3).
McGraw-Hill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import CoolProp
import CoolProp.CoolProp as cp

PairCode = Literal["TP", "PH", "HS", "PX", "TX", "PS", "TS", "TV", "PV", "PU"]

# Fluidos puros/pseudo-puros aceptados por CoolProp que se ofrecen como
# default en las UIs. Mantenerla acá centraliza el set para todas las
# páginas (propiedades, interpolación, isoentrópicos, ciclos, ...).
SUPPORTED_FLUIDS: list[str] = [
    "Water",
    "R134a",
    "R410A",
    "R1234yf",
    "Ammonia",
    "CarbonDioxide",
    "Air",
    # Fluidos orgánicos para el ciclo de Rankine orgánico (ORC, Fase 3.1c).
    "R245fa",
    "R1233zd(E)",
    "Isopentane",
    "Toluene",
    # Refrigerantes para la refrigeración por compresión de vapor (Fase 3.2):
    # el HFC de los aires acondicionados actuales y dos naturales (Cengel §11-6).
    "R32",
    "Propane",
    "IsoButane",
]

#: Nombre en castellano de cada fluido soportado (para títulos y selectores).
FLUID_NAMES_ES: dict[str, str] = {
    "Water": "Agua",
    "R134a": "R-134a",
    "R410A": "R-410A",
    "R1234yf": "R-1234yf",
    "Ammonia": "Amoníaco (R-717)",
    "CarbonDioxide": "Dióxido de carbono (R-744)",
    "Air": "Aire",
    "R245fa": "R-245fa",
    "R1233zd(E)": "R-1233zd(E)",
    "Isopentane": "Isopentano (R-601a)",
    "Toluene": "Tolueno",
    "R32": "R-32",
    "Propane": "Propano (R-290)",
    "IsoButane": "Isobutano (R-600a)",
}

# (kwarg en la API, símbolo CoolProp). El volumen específico viaja como
# densidad ("D" = 1/v) porque es la variable que acepta CoolProp.
_PAIR_KEYS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "TP": (("t", "T"), ("p", "P")),
    "PH": (("p", "P"), ("h", "H")),
    "HS": (("h", "H"), ("s", "S")),
    "PX": (("p", "P"), ("x", "Q")),
    "TX": (("t", "T"), ("x", "Q")),
    "PS": (("p", "P"), ("s", "S")),
    "TS": (("t", "T"), ("s", "S")),
    "TV": (("t", "T"), ("v", "D")),
    "PV": (("p", "P"), ("v", "D")),
    "PU": (("p", "P"), ("u", "U")),
}

#: kwargs que espera cada par, en el orden del código (``"PH"`` → ``("p", "h")``).
PAIR_KWARGS: dict[str, tuple[str, str]] = {
    pair: (k1, k2) for pair, ((k1, _), (k2, _)) in _PAIR_KEYS.items()
}

# Índices de entrada de la API de bajo nivel (AbstractState) por símbolo.
_COOLPROP_INPUT_INDEX: dict[str, int] = {
    "T": CoolProp.iT,
    "P": CoolProp.iP,
    "H": CoolProp.iHmass,
    "S": CoolProp.iSmass,
    "Q": CoolProp.iQ,
    "D": CoolProp.iDmass,
    "U": CoolProp.iUmass,
}

# Backend de ecuación de estado de Helmholtz (IAPWS-95 para el agua).
_BACKEND = "HEOS"


@dataclass(frozen=True)
class StatePoint:
    """Estado termodinámico puntual de un fluido puro, en SI.

    Atributos
    ---------
    T_K : float
        Temperatura en kelvin.
    P_Pa : float
        Presión absoluta en pascal.
    h_J_per_kg : float
        Entalpía específica en J/kg.
    s_J_per_kg_K : float
        Entropía específica en J/(kg·K).
    x : float
        Título de vapor (calidad), adimensional. Sigue la convención
        de CoolProp: ``-1`` indica que el estado está fuera de la
        región bifásica (líquido subenfriado o vapor sobrecalentado).
    v_m3_per_kg : float | None
        Volumen específico en m³/kg (= 1/ρ). Es opcional para no romper
        el código que construye ``StatePoint`` a mano;
        :func:`state_from_pair` siempre lo completa. Hace falta para
        ubicar el estado en el diagrama p–log v.
    """

    T_K: float
    P_Pa: float
    h_J_per_kg: float
    s_J_per_kg_K: float
    x: float
    v_m3_per_kg: float | None = None


def state_from_pair(fluid: str, pair: PairCode, **kwargs: float) -> StatePoint:
    """Resuelve el estado completo a partir de un par independiente.

    Parámetros
    ----------
    fluid : str
        Nombre del fluido aceptado por CoolProp (p.ej. ``"Water"``).
    pair : PairCode
        Código del par independiente. Uno de
        ``"TP"``, ``"PH"``, ``"HS"``, ``"PX"``, ``"TX"``, ``"PS"``, ``"TS"``,
        ``"TV"``, ``"PV"``, ``"PU"``.
    **kwargs : float
        Valores numéricos en SI. Claves esperadas según ``pair``:

        =====  ========================
        pair   kwargs requeridos
        =====  ========================
        TP     ``t`` [K], ``p`` [Pa]
        PH     ``p`` [Pa], ``h`` [J/kg]
        HS     ``h`` [J/kg], ``s`` [J/(kg·K)]
        PX     ``p`` [Pa], ``x`` [-]
        TX     ``t`` [K], ``x`` [-]
        PS     ``p`` [Pa], ``s`` [J/(kg·K)]
        TS     ``t`` [K], ``s`` [J/(kg·K)]
        TV     ``t`` [K], ``v`` [m³/kg]
        PV     ``p`` [Pa], ``v`` [m³/kg]
        PU     ``p`` [Pa], ``u`` [J/kg]
        =====  ========================

    Devuelve
    --------
    StatePoint
        Estado con todas las propiedades en SI.

    Excepciones
    -----------
    ValueError
        Si el par no está soportado, si faltan kwargs, si los valores
        están fuera de rango físico, o si CoolProp no logra resolver
        el estado para el fluido y los inputs dados.
    """
    (kw1, cp1, val1), (kw2, cp2, val2) = _parse_pair(pair, kwargs)
    in1 = _to_coolprop_value(kw1, val1)
    in2 = _to_coolprop_value(kw2, val2)

    try:
        T_K = _solve(cp1, in1, cp2, in2, "T", fluid)
        P_Pa = _solve(cp1, in1, cp2, in2, "P", fluid)
        h_J_per_kg = _solve(cp1, in1, cp2, in2, "H", fluid)
        s_J_per_kg_K = _solve(cp1, in1, cp2, in2, "S", fluid)
        x = _normalize_quality(_solve(cp1, in1, cp2, in2, "Q", fluid))
        rho_kg_per_m3 = _solve(cp1, in1, cp2, in2, "D", fluid)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"CoolProp no pudo resolver el estado de {fluid} para el par "
            f"{pair} con {kw1}={val1}, {kw2}={val2}. Revisá que los valores "
            f"sean físicamente consistentes y estén dentro del rango del "
            f"fluido. Detalle interno: {exc}"
        ) from exc

    return StatePoint(
        T_K=float(T_K),
        P_Pa=float(P_Pa),
        h_J_per_kg=_zero_if_noise(float(h_J_per_kg), _ENERGY_ZERO_J_PER_KG),
        s_J_per_kg_K=_zero_if_noise(float(s_J_per_kg_K), _ENTROPY_ZERO_J_PER_KG_K),
        x=float(x),
        v_m3_per_kg=1.0 / float(rho_kg_per_m3),
    )


# Tolerancia para aceptar como saturado un título que CoolProp devuelve
# apenas fuera de [0, 1] por redondeo numérico.
_QUALITY_TOL: float = 1e-9

# Ruido numérico alrededor del estado de referencia de cada fluido (donde
# u, h o s valen 0 por convención): el agua en el punto triple (IAPWS:
# u_f = s_f = 0) da u_f = −6.6×10⁻⁸ J/kg y el aire líquido saturado a
# 1 atm (NBP: h_f = s_f = 0) da h_f = −0.012 J/kg, que la página mostraba
# como "−6.6048×10⁻¹¹ kJ/kg". Por debajo de estos valores (SI) la
# propiedad es cero dentro de la precisión de la ecuación de estado; un
# estado real nunca está tan cerca de la referencia sin estar en ella
# (h_f del agua en el punto triple, p·v = 0.61 J/kg, queda intacta).
_ENERGY_ZERO_J_PER_KG: float = 0.1
_ENTROPY_ZERO_J_PER_KG_K: float = 1e-3


def _zero_if_noise(value: float, tol: float) -> float:
    """``0.0`` si ``value`` es ruido alrededor del estado de referencia."""
    return 0.0 if abs(value) < tol else value


def _normalize_quality(q: float) -> float:
    """Lleva el título de CoolProp a la convención de :class:`StatePoint`.

    Para estados monofásicos CoolProp devuelve un centinela fuera de
    [0, 1]: ``-1`` en la mayoría de los pares, pero ``10000`` cuando el
    estado se resuelve con h y s. Cualquier valor fuera de [0, 1] se
    normaliza a ``-1``; los desbordes de redondeo (±1e-9) se recortan
    al borde de la campana.
    """
    q = float(q)
    if -_QUALITY_TOL <= q <= 1.0 + _QUALITY_TOL:
        return min(max(q, 0.0), 1.0)
    return -1.0


def _parse_pair(
    pair: str, kwargs: dict[str, float]
) -> tuple[tuple[str, str, float], tuple[str, str, float]]:
    """Valida el par y sus kwargs; devuelve ``((kw, símbolo, valor), ...)``."""
    if pair not in _PAIR_KEYS:
        raise ValueError(f"Par '{pair}' no soportado. Usá uno de: {sorted(_PAIR_KEYS)}.")

    (kw1, cp1), (kw2, cp2) = _PAIR_KEYS[pair]
    if kw1 not in kwargs or kw2 not in kwargs:
        raise ValueError(
            f"El par '{pair}' requiere argumentos '{kw1}' y '{kw2}', pero recibí {sorted(kwargs)}."
        )

    val1 = float(kwargs[kw1])
    val2 = float(kwargs[kw2])
    _validate_si_inputs(kw1, val1)
    _validate_si_inputs(kw2, val2)
    return (kw1, cp1, val1), (kw2, cp2, val2)


def _to_coolprop_value(kw: str, value: float) -> float:
    """Valor que recibe CoolProp: el volumen específico viaja como ρ = 1/v."""
    return 1.0 / value if kw == "v" else value


def _solve(in1: str, val1: float, in2: str, val2: float, out: str, fluid: str) -> float:
    """Resuelve ``out`` con CoolProp, o devuelve el input si coincide."""
    if in1 == out:
        return val1
    if in2 == out:
        return val2
    return cp.PropsSI(out, in1, val1, in2, val2, fluid)


def _validate_si_inputs(kw: str, val: float) -> None:
    """Valida rangos físicos de un input antes de invocar a CoolProp.

    Mensajes orientados al alumno: explican qué entrada está fuera de
    rango y por qué, no solo "ValueError".
    """
    if not math.isfinite(val):
        raise ValueError(f"El valor de '{kw}' tiene que ser un número finito. Recibí {val}.")
    if kw == "p" and val <= 0.0:
        raise ValueError(
            f"La presión absoluta debe ser estrictamente positiva. "
            f"Recibí p = {val} Pa, que no es físico."
        )
    if kw == "t" and val <= 0.0:
        raise ValueError(
            f"La temperatura absoluta debe ser estrictamente positiva. "
            f"Recibí T = {val} K, que no es físico."
        )
    if kw == "x" and not (0.0 <= val <= 1.0):
        raise ValueError(
            f"El título de vapor x debe estar entre 0 y 1 (ambos inclusive). Recibí x = {val}."
        )
    if kw == "v" and val <= 0.0:
        raise ValueError(
            f"El volumen específico debe ser estrictamente positivo. "
            f"Recibí v = {val} m³/kg, que no es físico."
        )


# =====================================================================
# Estado completo para el alumno
# =====================================================================


@dataclass(frozen=True)
class FluidLimits:
    """Puntos característicos y rango de validez de la ecuación de estado.

    Todo en SI. ``R_J_per_kg_K`` es la constante particular R = R_u / M,
    útil para comparar con el modelo de gas ideal. ``is_pure`` es False
    para los pseudo-puros (mezclas tratadas como un fluido, p. ej. aire o
    R410A): CoolProp no admite para ellos un título 0 < x < 1 como dato.
    """

    fluid: str
    is_pure: bool
    T_triple_K: float
    P_triple_Pa: float
    T_crit_K: float
    P_crit_Pa: float
    v_crit_m3_per_kg: float
    T_min_K: float
    T_max_K: float
    P_max_Pa: float
    R_J_per_kg_K: float
    M_kg_per_mol: float


def fluid_limits(fluid: str) -> FluidLimits:
    """Punto triple, punto crítico y rango de validez de ``fluid`` (SI).

    Raises
    ------
    ValueError
        Si CoolProp no reconoce el fluido.
    """
    state = _abstract_state(fluid)
    return FluidLimits(
        fluid=fluid,
        is_pure=cp.get_fluid_param_string(fluid, "pure") == "true",
        T_triple_K=float(state.Ttriple()),
        P_triple_Pa=float(state.trivial_keyed_output(CoolProp.iP_triple)),
        T_crit_K=float(state.T_critical()),
        P_crit_Pa=float(state.p_critical()),
        v_crit_m3_per_kg=1.0 / float(state.rhomass_critical()),
        T_min_K=float(state.Tmin()),
        T_max_K=float(state.Tmax()),
        P_max_Pa=float(state.pmax()),
        R_J_per_kg_K=float(state.gas_constant()) / float(state.molar_mass()),
        M_kg_per_mol=float(state.molar_mass()),
    )


@dataclass(frozen=True)
class SaturatedPhase:
    """Una fase en saturación: líquido saturado (f) o vapor saturado (g)."""

    T_K: float
    P_Pa: float
    v_m3_per_kg: float
    u_J_per_kg: float
    h_J_per_kg: float
    s_J_per_kg_K: float


@dataclass(frozen=True)
class SaturationProperties:
    """Propiedades de saturación a una presión o temperatura dada.

    Es la fila de las tablas de saturación (Cengel A-4 por temperatura,
    A-5 por presión): ``liquid`` es el líquido saturado (subíndice f) y
    ``vapor`` el vapor saturado (subíndice g).

    Para fluidos puros ``liquid.T_K == vapor.T_K`` (y lo mismo con la
    presión). Los pseudo-puros (aire, R410A) tienen deslizamiento
    (*glide*): a presión fija, el vapor saturado (punto de rocío) está
    más caliente que el líquido saturado (punto de burbuja).
    """

    fluid: str
    basis: Literal["P", "T"]
    liquid: SaturatedPhase
    vapor: SaturatedPhase

    @property
    def T_sat_K(self) -> float:
        """Temperatura de saturación (de burbuja, si hay glide)."""
        return self.liquid.T_K

    @property
    def P_sat_Pa(self) -> float:
        """Presión de saturación (de burbuja, si hay glide)."""
        return self.liquid.P_Pa

    @property
    def glide_K(self) -> float:
        """T_rocío − T_burbuja. Cero para fluidos puros."""
        return self.vapor.T_K - self.liquid.T_K

    @property
    def v_fg_m3_per_kg(self) -> float:
        return self.vapor.v_m3_per_kg - self.liquid.v_m3_per_kg

    @property
    def u_fg_J_per_kg(self) -> float:
        return self.vapor.u_J_per_kg - self.liquid.u_J_per_kg

    @property
    def h_fg_J_per_kg(self) -> float:
        """Calor latente de vaporización h_fg = h_g − h_f (vademecum §12.3)."""
        return self.vapor.h_J_per_kg - self.liquid.h_J_per_kg

    @property
    def s_fg_J_per_kg_K(self) -> float:
        return self.vapor.s_J_per_kg_K - self.liquid.s_J_per_kg_K


def saturation_at_pressure(fluid: str, P_Pa: float) -> SaturationProperties:
    """Propiedades de saturación a la presión ``P_Pa`` (tabla "por presión").

    Raises
    ------
    ValueError
        Si la presión no está entre la del punto triple y la crítica
        (fuera de ese rango no existe equilibrio líquido–vapor).
    """
    limits = fluid_limits(fluid)
    if not (limits.P_triple_Pa <= P_Pa < limits.P_crit_Pa):
        raise ValueError(
            f"No hay saturación líquido–vapor a p = {_fmt_p(P_Pa)}: para "
            f"{_el(fluid)} la presión de saturación va del punto triple "
            f"({_fmt_p(limits.P_triple_Pa)}) al punto crítico ({_fmt_p(limits.P_crit_Pa)})."
        )
    return _saturation(fluid, "P", P_Pa)


def saturation_at_temperature(fluid: str, T_K: float) -> SaturationProperties:
    """Propiedades de saturación a la temperatura ``T_K`` (tabla "por temperatura").

    Raises
    ------
    ValueError
        Si la temperatura no está entre la del punto triple y la crítica.
    """
    limits = fluid_limits(fluid)
    if not (limits.T_triple_K <= T_K < limits.T_crit_K):
        raise ValueError(
            f"No hay saturación líquido–vapor a T = {_fmt_T(T_K)}: para "
            f"{_el(fluid)} la temperatura de saturación va del punto triple "
            f"({_fmt_T(limits.T_triple_K)}) al punto crítico ({_fmt_T(limits.T_crit_K)})."
        )
    return _saturation(fluid, "T", T_K)


def _saturation(fluid: str, basis: Literal["P", "T"], value: float) -> SaturationProperties:
    state = _abstract_state(fluid)
    pair = CoolProp.PQ_INPUTS if basis == "P" else CoolProp.QT_INPUTS
    phases: list[SaturatedPhase] = []
    for q in (0.0, 1.0):
        try:
            if basis == "P":
                state.update(pair, value, q)
            else:
                state.update(pair, q, value)
        except Exception as exc:
            raise ValueError(
                f"CoolProp no pudo calcular la saturación {_del(fluid)} "
                f"({'p' if basis == 'P' else 'T'} = "
                f"{_fmt_p(value) if basis == 'P' else _fmt_T(value)}). Detalle técnico: {exc}"
            ) from exc
        phases.append(
            SaturatedPhase(
                T_K=float(state.T()),
                P_Pa=float(state.p()),
                v_m3_per_kg=1.0 / float(state.rhomass()),
                u_J_per_kg=_zero_if_noise(float(state.umass()), _ENERGY_ZERO_J_PER_KG),
                h_J_per_kg=_zero_if_noise(float(state.hmass()), _ENERGY_ZERO_J_PER_KG),
                s_J_per_kg_K=_zero_if_noise(float(state.smass()), _ENTROPY_ZERO_J_PER_KG_K),
            )
        )
    return SaturationProperties(fluid=fluid, basis=basis, liquid=phases[0], vapor=phases[1])


Region = Literal[
    "compressed_liquid",
    "saturated_liquid",
    "saturated_mixture",
    "saturated_vapor",
    "superheated_vapor",
    "supercritical",
]

#: Nombre de cada región en castellano, con la nomenclatura de la cátedra.
REGION_LABELS_ES: dict[str, str] = {
    "compressed_liquid": "Líquido comprimido (subenfriado)",
    "saturated_liquid": "Líquido saturado",
    "saturated_mixture": "Vapor húmedo (mezcla líquido–vapor saturada)",
    "saturated_vapor": "Vapor saturado seco",
    "superheated_vapor": "Vapor sobrecalentado",
    "supercritical": "Fluido supercrítico",
}

_TWO_PHASE_REGIONS: frozenset[str] = frozenset(
    {"saturated_liquid", "saturated_mixture", "saturated_vapor"}
)


def classify_region(phase_index: int, quality: float) -> tuple[Region, float | None]:
    """Región del estado (convención de Cengel) y título si corresponde.

    Parámetros
    ----------
    phase_index : int
        Índice de fase de CoolProp (``AbstractState.phase()``).
    quality : float
        Título que devuelve CoolProp (con sus centinelas -1 / 10000).

    Notas
    -----
    CoolProp distingue ``supercritical_liquid`` (p > p_c, T < T_c) y
    ``supercritical_gas`` (T > T_c, p < p_c). Siguiendo a Cengel, el
    primero se reporta como líquido comprimido y el segundo como vapor
    sobrecalentado; solo T > T_c **y** p > p_c es "fluido supercrítico".
    """
    q = _normalize_quality(quality)
    if phase_index == CoolProp.iphase_twophase or q >= 0.0:
        x = min(max(q, 0.0), 1.0)
        if x <= _QUALITY_TOL:
            return "saturated_liquid", 0.0
        if x >= 1.0 - _QUALITY_TOL:
            return "saturated_vapor", 1.0
        return "saturated_mixture", x
    if phase_index in (CoolProp.iphase_liquid, CoolProp.iphase_supercritical_liquid):
        return "compressed_liquid", None
    if phase_index in (CoolProp.iphase_gas, CoolProp.iphase_supercritical_gas):
        return "superheated_vapor", None
    return "supercritical", None


@dataclass(frozen=True)
class FluidState:
    """Estado termodinámico completo de un fluido puro, en SI.

    Las propiedades que no están definidas en el estado valen ``None``:
    ``x`` fuera de la campana; cp, cv, velocidad del sonido y propiedades
    de transporte dentro de la mezcla bifásica (CoolProp devuelve números
    sin significado físico ahí; cerca del punto crítico, cp < 0);
    ``sat_at_P`` / ``sat_at_T`` cuando p o T están fuera del rango de
    saturación (encima del crítico o debajo del triple).
    """

    fluid: str
    T_K: float
    P_Pa: float
    v_m3_per_kg: float
    u_J_per_kg: float
    h_J_per_kg: float
    s_J_per_kg_K: float
    x: float | None
    region: Region
    cp_J_per_kg_K: float | None
    cv_J_per_kg_K: float | None
    speed_of_sound_m_per_s: float | None
    viscosity_Pa_s: float | None
    conductivity_W_per_m_K: float | None
    Z: float
    sat_at_P: SaturationProperties | None
    sat_at_T: SaturationProperties | None
    limits: FluidLimits

    @property
    def rho_kg_per_m3(self) -> float:
        return 1.0 / self.v_m3_per_kg

    @property
    def is_two_phase(self) -> bool:
        """True para líquido saturado, vapor húmedo y vapor saturado."""
        return self.region in _TWO_PHASE_REGIONS

    @property
    def gamma(self) -> float | None:
        """Relación de calores específicos k = cp/cv."""
        if self.cp_J_per_kg_K is None or self.cv_J_per_kg_K is None:
            return None
        return self.cp_J_per_kg_K / self.cv_J_per_kg_K

    @property
    def prandtl(self) -> float | None:
        """Número de Prandtl Pr = μ·cp/k."""
        if None in (self.viscosity_Pa_s, self.cp_J_per_kg_K, self.conductivity_W_per_m_K):
            return None
        return self.viscosity_Pa_s * self.cp_J_per_kg_K / self.conductivity_W_per_m_K  # type: ignore[operator]

    @property
    def kinematic_viscosity_m2_per_s(self) -> float | None:
        """Viscosidad cinemática ν = μ/ρ = μ·v."""
        if self.viscosity_Pa_s is None:
            return None
        return self.viscosity_Pa_s * self.v_m3_per_kg

    @property
    def thermal_diffusivity_m2_per_s(self) -> float | None:
        """Difusividad térmica α = k/(ρ·cp)."""
        if self.conductivity_W_per_m_K is None or self.cp_J_per_kg_K is None:
            return None
        return self.conductivity_W_per_m_K * self.v_m3_per_kg / self.cp_J_per_kg_K

    @property
    def superheat_K(self) -> float | None:
        """Grado de sobrecalentamiento T − T_sat(p) (vapor sobrecalentado)."""
        if self.region != "superheated_vapor" or self.sat_at_P is None:
            return None
        return self.T_K - self.sat_at_P.vapor.T_K

    @property
    def subcooling_K(self) -> float | None:
        """Grado de subenfriamiento T_sat(p) − T (líquido comprimido)."""
        if self.region != "compressed_liquid" or self.sat_at_P is None:
            return None
        return self.sat_at_P.liquid.T_K - self.T_K

    def to_state_point(self) -> StatePoint:
        """Versión reducida para los módulos que trabajan con :class:`StatePoint`."""
        return StatePoint(
            T_K=self.T_K,
            P_Pa=self.P_Pa,
            h_J_per_kg=self.h_J_per_kg,
            s_J_per_kg_K=self.s_J_per_kg_K,
            x=-1.0 if self.x is None else self.x,
            v_m3_per_kg=self.v_m3_per_kg,
        )


def fluid_state_from_pair(fluid: str, pair: PairCode, **kwargs: float) -> FluidState:
    """Estado completo de ``fluid`` a partir de dos propiedades independientes.

    Mismos pares y kwargs (en SI) que :func:`state_from_pair`. Antes de
    llamar a CoolProp valida que los datos estén dentro del rango del
    fluido (punto triple, punto crítico para el título, máximos de la
    ecuación de estado) y, después, que el estado resultante también lo
    esté: con pares como T-s o h-s, CoolProp puede "resolver" estados
    absurdos (p. ej. R134a a 44 000 bar) extrapolando la ecuación.

    Raises
    ------
    ValueError
        Con un mensaje en castellano que explica qué dato está fuera de
        rango y por qué.
    """
    (kw1, cp1, val1), (kw2, cp2, val2) = _parse_pair(pair, kwargs)
    limits = fluid_limits(fluid)
    for kw, val in ((kw1, val1), (kw2, val2)):
        _validate_against_limits(kw, val, pair, limits)

    state = _abstract_state(fluid)
    inputs_desc = f"{_describe_input(kw1, val1)} y {_describe_input(kw2, val2)}"
    lever_x = (
        _pseudo_pure_lever_quality(fluid, val1, kw2, val2, limits)
        if not limits.is_pure and kw1 == "p"
        else None
    )
    try:
        if lever_x is not None:
            state.update(CoolProp.PQ_INPUTS, val1, lever_x)
        else:
            update_pair, in1, in2 = cp.generate_update_pair(
                _COOLPROP_INPUT_INDEX[cp1],
                _to_coolprop_value(kw1, val1),
                _COOLPROP_INPUT_INDEX[cp2],
                _to_coolprop_value(kw2, val2),
            )
            state.update(update_pair, in1, in2)
        T_K = float(state.T())
        P_Pa = float(state.p())
    except Exception as exc:
        raise _student_error(exc, fluid, inputs_desc) from exc

    _validate_result_in_range(T_K, P_Pa, limits, inputs_desc)

    region, x = classify_region(int(state.phase()), float(state.Q()))
    single_phase_or_saturated_end = region != "saturated_mixture"

    def _optional(getter: str) -> float | None:
        """Propiedad que puede no estar definida (o ser basura numérica)."""
        if not single_phase_or_saturated_end:
            return None
        try:
            value = float(getattr(state, getter)())
        except Exception:
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    return FluidState(
        fluid=fluid,
        T_K=T_K,
        P_Pa=P_Pa,
        v_m3_per_kg=1.0 / float(state.rhomass()),
        u_J_per_kg=_zero_if_noise(float(state.umass()), _ENERGY_ZERO_J_PER_KG),
        h_J_per_kg=_zero_if_noise(float(state.hmass()), _ENERGY_ZERO_J_PER_KG),
        s_J_per_kg_K=_zero_if_noise(float(state.smass()), _ENTROPY_ZERO_J_PER_KG_K),
        x=x,
        region=region,
        cp_J_per_kg_K=_optional("cpmass"),
        cv_J_per_kg_K=_optional("cvmass"),
        speed_of_sound_m_per_s=_optional("speed_sound"),
        viscosity_Pa_s=_optional("viscosity"),
        conductivity_W_per_m_K=_optional("conductivity"),
        Z=P_Pa / (float(state.rhomass()) * limits.R_J_per_kg_K * T_K),
        sat_at_P=_saturation_or_none(fluid, "P", P_Pa, limits),
        sat_at_T=_saturation_or_none(fluid, "T", T_K, limits),
        limits=limits,
    )


# Propiedad específica de una fase saturada, por kwarg del par.
_LEVER_ATTRS: dict[str, str] = {
    "h": "h_J_per_kg",
    "s": "s_J_per_kg_K",
    "v": "v_m3_per_kg",
    "u": "u_J_per_kg",
}


def _pseudo_pure_lever_quality(
    fluid: str, p_Pa: float, kw: str, value: float, limits: FluidLimits
) -> float | None:
    """Título de un pseudo-puro con p y (h, s, v o u) dentro de la campana.

    Con deslizamiento de temperatura (aire a 1 atm: de 78,9 K en el punto
    de burbuja a 81,7 K en el de rocío), el cálculo de CoolProp con p y h
    no reconoce la campana cerca de la línea de burbuja: en el borde
    devuelve "líquido" y apenas adentro falla. En su modelo cada propiedad
    específica es lineal en el título a p constante (regla de la palanca,
    vademecum §12.2; Cengel §3-5), así que el título se despeja de ahí y
    el estado se calcula con (p, x). Devuelve ``None`` si el dato cae
    fuera de la campana (o p está fuera del rango de saturación).
    """
    attr = _LEVER_ATTRS.get(kw)
    if attr is None or not limits.P_triple_Pa <= p_Pa < limits.P_crit_Pa:
        return None
    try:
        sat = _saturation(fluid, "P", p_Pa)
    except ValueError:
        return None
    y_f = float(getattr(sat.liquid, attr))
    y_g = float(getattr(sat.vapor, attr))
    x = (value - y_f) / (y_g - y_f)
    if -_QUALITY_TOL <= x <= 1.0 + _QUALITY_TOL:
        return min(max(x, 0.0), 1.0)
    return None


def _saturation_or_none(
    fluid: str, basis: Literal["P", "T"], value: float, limits: FluidLimits
) -> SaturationProperties | None:
    """Saturación a p (o T) del estado, o None si no existe en ese punto."""
    if basis == "P":
        inside = limits.P_triple_Pa <= value < limits.P_crit_Pa
    else:
        inside = limits.T_triple_K <= value < limits.T_crit_K
    if not inside:
        return None
    try:
        return _saturation(fluid, basis, value)
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Valores sugeridos para los formularios
# ---------------------------------------------------------------------

# Agua: estados de libro (Cengel, cap. 3 y 10), uno por par, que cubren
# las distintas regiones. Comparables directamente con las tablas.
_WATER_SUGGESTED: dict[str, dict[str, float]] = {
    "TP": {"t": 573.15, "p": 1.0e6},  # 300 °C, 10 bar → vapor sobrecalentado (A-6)
    "PH": {"p": 1.0e6, "h": 2.0e6},  # 10 bar, 2000 kJ/kg → vapor húmedo
    "HS": {"h": 3.0e6, "s": 7.0e3},  # diagrama de Mollier → vapor sobrecalentado
    "PX": {"p": 1.0e5, "x": 0.5},  # 1 bar, x = 0.5
    "TX": {"t": 373.15, "x": 0.5},  # 100 °C, x = 0.5
    "PS": {"p": 1.0e4, "s": 7.0e3},  # salida de turbina a 0,1 bar → vapor húmedo
    "TS": {"t": 573.15, "s": 7.0e3},  # 300 °C → vapor sobrecalentado
    "TV": {"t": 393.15, "v": 0.5},  # tanque rígido a 120 °C → vapor húmedo
    "PV": {"p": 5.0e5, "v": 0.5},  # 5 bar → vapor sobrecalentado
    "PU": {"p": 2.0e5, "u": 2.0e6},  # sistema cerrado a 2 bar → vapor húmedo
}


def suggested_inputs(fluid: str, pair: PairCode) -> dict[str, float]:
    """Valores iniciales (en SI) para los widgets de ``pair``.

    Siempre corresponden a un estado válido del fluido (lo verifica la
    suite de tests para cada fluido × par). Para el agua son estados de
    libro; para el resto se arman a partir de una temperatura de
    saturación de referencia (0 °C cuando cae dentro de la campana): una
    mezcla con x = 0.5 (x = 1 en los pseudo-puros) y un vapor
    sobrecalentado 20 K por encima. Para los gases permanentes
    (T_c < 0 °C, como el aire) los pares que no involucran el título usan
    aire ambiente (25 °C, 1 atm).
    """
    if pair not in _PAIR_KEYS:
        raise ValueError(f"Par '{pair}' no soportado. Usá uno de: {sorted(_PAIR_KEYS)}.")
    if fluid == "Water":
        return dict(_WATER_SUGGESTED[pair])

    limits = fluid_limits(fluid)
    T_ref = min(max(273.15, limits.T_triple_K + 10.0), limits.T_crit_K - 15.0)
    P_ref = _saturation(fluid, "T", T_ref).P_sat_Pa
    # La mezcla se arma con p-h (regla de la palanca sobre h). Los
    # pseudo-puros no aceptan 0 < x < 1 junto con T: su título sugerido
    # en T-x es 1 (vapor saturado, la salida de un evaporador).
    sat_P = _saturation(fluid, "P", P_ref)
    mix = fluid_state_from_pair(
        fluid, "PH", p=P_ref, h=sat_P.liquid.h_J_per_kg + 0.5 * sat_P.h_fg_J_per_kg
    )
    x_ref_T = 0.5 if limits.is_pure else 1.0
    if limits.T_crit_K < 273.15:
        gas = fluid_state_from_pair(fluid, "TP", t=298.15, p=101_325.0)
        mix_like = gas
    else:
        gas = fluid_state_from_pair(fluid, "TP", t=T_ref + 20.0, p=P_ref)
        mix_like = mix

    candidates: dict[str, dict[str, float]] = {
        "TP": {"t": gas.T_K, "p": gas.P_Pa},
        "PH": {"p": mix_like.P_Pa, "h": mix_like.h_J_per_kg},
        "HS": {"h": gas.h_J_per_kg, "s": gas.s_J_per_kg_K},
        "PX": {"p": P_ref, "x": 0.5},
        "TX": {"t": T_ref, "x": x_ref_T},
        "PS": {"p": mix_like.P_Pa, "s": mix_like.s_J_per_kg_K},
        "TS": {"t": gas.T_K, "s": gas.s_J_per_kg_K},
        "TV": {"t": mix_like.T_K, "v": mix_like.v_m3_per_kg},
        "PV": {"p": gas.P_Pa, "v": gas.v_m3_per_kg},
        "PU": {"p": mix_like.P_Pa, "u": mix_like.u_J_per_kg},
    }
    return candidates[pair]


# Estado de referencia de las tablas de Cengel para los refrigerantes que tabula:
# h = s = 0 para el líquido saturado a −40 °C (ASHRAE). CoolProp usa la del IIR
# (h = 200 kJ/kg y s = 1 kJ/(kg·K) para el líquido saturado a 0 °C). El agua no
# necesita corrección: las dos usan u = s = 0 para el líquido en el punto triple.
_TEXTBOOK_REFERENCE_T_K: dict[str, float] = {"R134a": 233.15}


def textbook_reference_offset(fluid: str) -> tuple[float, float] | None:
    """``(Δh, Δs)`` = valor de CoolProp − valor de las tablas de Cengel (SI), o ``None``.

    Cengel (tablas A-11 a A-13) tabula el R-134a con h = s = 0 para el líquido
    saturado a −40 °C; CoolProp usa la referencia del IIR. Restando Δh y Δs a
    los valores de CoolProp se obtienen los del libro; las diferencias de h y
    s (calores, trabajos, COP) no cambian. ``None`` si no hace falta corregir.
    """
    T_ref = _TEXTBOOK_REFERENCE_T_K.get(fluid)
    if T_ref is None:
        return None
    liquid = _saturation(fluid, "T", T_ref).liquid
    return liquid.h_J_per_kg, liquid.s_J_per_kg_K


# ---------------------------------------------------------------------
# Helpers internos: CoolProp, validación y mensajes
# ---------------------------------------------------------------------


def _abstract_state(fluid: str) -> CoolProp.AbstractState:
    try:
        return CoolProp.AbstractState(_BACKEND, fluid)
    except Exception as exc:
        raise ValueError(
            f"CoolProp no reconoce el fluido '{fluid}'. Fluidos del proyecto: {SUPPORTED_FLUIDS}."
        ) from exc


# Para redactar los mensajes: "el agua" (a tónica), "el R-134a", "el aire"...
_FLUID_WITH_ARTICLE: dict[str, str] = {
    "Water": "el agua",
    "R134a": "el R-134a",
    "R410A": "el R-410A",
    "R1234yf": "el R-1234yf",
    "Ammonia": "el amoníaco",
    "CarbonDioxide": "el dióxido de carbono",
    "Air": "el aire",
    "R245fa": "el R-245fa",
    "R1233zd(E)": "el R-1233zd(E)",
    "Isopentane": "el isopentano",
    "Toluene": "el tolueno",
    "R32": "el R-32",
    "Propane": "el propano",
    "IsoButane": "el isobutano",
}


def _el(fluid: str) -> str:
    """``"el agua"``, ``"el R-134a"``, ... para usar dentro de una oración."""
    return _FLUID_WITH_ARTICLE.get(fluid, f"el fluido {fluid}")


def fluid_with_article(fluid: str) -> str:
    """``"el agua"``, ``"el R-134a"``, ... (público: lo usan los mensajes de otros módulos)."""
    return _el(fluid)


def _del(fluid: str) -> str:
    """``"del agua"``, ``"del R-134a"``, ... (contracción de + el)."""
    return "del " + _el(fluid)[3:]


def _fmt_T(T_K: float) -> str:
    return f"{T_K:.6g} K ({T_K - 273.15:.5g} °C)"


def _fmt_p(P_Pa: float) -> str:
    return f"{P_Pa:.6g} Pa ({P_Pa / 1e5:.6g} bar)"


def _describe_input(kw: str, value: float) -> str:
    """Descripción legible de un dato (SI y, entre paréntesis, Técnico)."""
    if kw == "t":
        return f"T = {_fmt_T(value)}"
    if kw == "p":
        return f"p = {_fmt_p(value)}"
    if kw in ("h", "u"):
        return f"{kw} = {value:.6g} J/kg ({value / 1e3:.6g} kJ/kg)"
    if kw == "s":
        return f"s = {value:.6g} J/(kg·K) ({value / 1e3:.6g} kJ/(kg·K))"
    if kw == "v":
        return f"v = {value:.6g} m³/kg"
    return f"{kw} = {value:.6g}"


def _validate_against_limits(kw: str, value: float, pair: str, limits: FluidLimits) -> None:
    """Rangos que dependen del fluido (los universales están en _validate_si_inputs)."""
    fluid = limits.fluid
    if kw == "t":
        if value < limits.T_min_K:
            solid = ", es decir hielo" if limits.fluid == "Water" else ""
            raise ValueError(
                f"La temperatura T = {_fmt_T(value)} está por debajo del límite inferior "
                f"del modelo {_del(fluid)} ({_fmt_T(limits.T_min_K)}, el punto triple). Por "
                f"debajo de ese valor el fluido puede estar sólido{solid}: CoolProp solo "
                f"modela líquido, vapor y fluido supercrítico."
            )
        if value > limits.T_max_K:
            raise ValueError(
                f"La temperatura T = {_fmt_T(value)} supera el máximo de validez de la "
                f"ecuación de estado {_del(fluid)} ({_fmt_T(limits.T_max_K)})."
            )
    if kw == "p" and value > limits.P_max_Pa:
        raise ValueError(
            f"La presión p = {_fmt_p(value)} supera el máximo de validez de la ecuación "
            f"de estado {_del(fluid)} ({_fmt_p(limits.P_max_Pa)})."
        )
    if pair == "TX" and kw == "x" and 0.0 < value < 1.0 and not limits.is_pure:
        raise ValueError(
            f"{_el(fluid).capitalize()} es un fluido pseudo-puro (una mezcla tratada como "
            f"un solo fluido): dentro de la campana su temperatura no es constante, va de "
            f"la de burbuja (x = 0) a la de rocío (x = 1) a la misma presión. Por eso con T "
            f"solo están definidos el líquido saturado (x = 0) y el vapor saturado "
            f"(x = 1). Para un estado dentro de la campana usá el par p-x (o p-h)."
        )
    # El título solo existe dentro de la campana: la variable "ancla" (p o T)
    # tiene que estar entre el punto triple y el crítico.
    if pair == "PX" and kw == "p":
        if value >= limits.P_crit_Pa:
            raise ValueError(
                f"Con p = {_fmt_p(value)} se está en o por encima de la presión crítica "
                f"{_del(fluid)} (p_c = {_fmt_p(limits.P_crit_Pa)}): ahí no hay cambio de "
                f"fase, no existe la campana y el título x no está definido. Usá otro par de "
                f"variables (por ejemplo T y p)."
            )
        if value < limits.P_triple_Pa:
            raise ValueError(
                f"Con p = {_fmt_p(value)} se está por debajo de la presión del punto triple "
                f"{_del(fluid)} (p_t = {_fmt_p(limits.P_triple_Pa)}): no existe equilibrio "
                f"líquido–vapor, así que el título x no está definido."
            )
    if pair == "TX" and kw == "t":
        if value >= limits.T_crit_K:
            raise ValueError(
                f"Con T = {_fmt_T(value)} se está en o por encima de la temperatura crítica "
                f"{_del(fluid)} (T_c = {_fmt_T(limits.T_crit_K)}): ahí no hay cambio de "
                f"fase, no existe la campana y el título x no está definido. Usá otro par de "
                f"variables (por ejemplo T y p)."
            )
        if value < limits.T_triple_K:
            raise ValueError(
                f"Con T = {_fmt_T(value)} se está por debajo del punto triple {_del(fluid)} "
                f"(T_t = {_fmt_T(limits.T_triple_K)}): no existe equilibrio líquido–vapor."
            )


def _validate_result_in_range(T_K: float, P_Pa: float, limits: FluidLimits, inputs: str) -> None:
    """El estado resuelto tiene que caer dentro del rango de la ecuación de estado."""
    problems: list[str] = []
    if P_Pa > limits.P_max_Pa * (1.0 + 1e-9):
        problems.append(f"p = {_fmt_p(P_Pa)} (máximo {_fmt_p(limits.P_max_Pa)})")
    if T_K > limits.T_max_K * (1.0 + 1e-9):
        problems.append(f"T = {_fmt_T(T_K)} (máximo {_fmt_T(limits.T_max_K)})")
    if T_K < limits.T_min_K * (1.0 - 1e-9):
        problems.append(f"T = {_fmt_T(T_K)} (mínimo {_fmt_T(limits.T_min_K)})")
    if problems:
        raise ValueError(
            f"Con {inputs}, el estado {_del(limits.fluid)} que resulta cae fuera del rango "
            f"de validez de su ecuación de estado: {'; '.join(problems)}. Seguramente la "
            f"combinación de valores no corresponde a un estado físico del fluido; revisá "
            f"que las dos propiedades sean coherentes entre sí."
        )


def _student_error(exc: Exception, fluid: str, inputs: str) -> ValueError:
    """Traduce un error de CoolProp a un mensaje que explique la causa."""
    detail = str(exc)
    lowered = detail.lower()
    if "pseudo-pure" in lowered:
        reason = (
            "es un fluido pseudo-puro y CoolProp no define sus estados de dos fases a "
            "partir del título; usá el par p-h con h = h_f + x·(h_g − h_f)"
        )
    elif "tmelt" in lowered or "melting" in lowered:
        solid = " (hielo)" if fluid == "Water" else ""
        reason = (
            f"a esa presión la temperatura queda por debajo de la de fusión, así que el "
            f"fluido estaría sólido{solid}. CoolProp solo modela líquido, vapor y fluido "
            f"supercrítico"
        )
    elif any(
        key in lowered
        for key in ("bracket", "could not find", "unable to solve", "out of range", "no solution")
    ):
        reason = (
            "no existe un estado del fluido con esa combinación de valores dentro del rango "
            "de su ecuación de estado. Revisá que sean coherentes: por ejemplo, que la "
            "entalpía o la entropía correspondan a ese fluido (su estado de referencia "
            "puede no ser el de tus tablas) y a la presión o temperatura ingresada"
        )
    else:
        reason = "CoolProp no pudo resolver el estado con esos valores"
    return ValueError(
        f"No se pudo calcular el estado {_del(fluid)} con {inputs}: {reason}. "
        f"Detalle técnico: {detail}"
    )
