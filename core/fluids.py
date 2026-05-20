"""Propiedades termofísicas — wrappers tipados sobre CoolProp.

Toda función opera en unidades SI: Pa, K, J/kg, J/(kg·K). La conversión
desde/hacia unidades didácticas (bar, °C, kJ/kg) es responsabilidad de
la capa de UI. Este módulo no importa Streamlit.

Cita
----
Bell, I. H., Wronski, J., Quoilin, S., & Lemort, V. (2014).
"Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the
Open-Source Thermophysical Property Library CoolProp".
*Industrial & Engineering Chemistry Research*, 53(6), 2498-2508.
DOI: 10.1021/ie4033999
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import CoolProp.CoolProp as cp

PairCode = Literal["TP", "PH", "HS", "PX", "TX", "PS", "TS"]

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
]

# (kwarg en la API, símbolo CoolProp)
_PAIR_KEYS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "TP": (("t", "T"), ("p", "P")),
    "PH": (("p", "P"), ("h", "H")),
    "HS": (("h", "H"), ("s", "S")),
    "PX": (("p", "P"), ("x", "Q")),
    "TX": (("t", "T"), ("x", "Q")),
    "PS": (("p", "P"), ("s", "S")),
    "TS": (("t", "T"), ("s", "S")),
}


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
    """

    T_K: float
    P_Pa: float
    h_J_per_kg: float
    s_J_per_kg_K: float
    x: float


def state_from_pair(fluid: str, pair: PairCode, **kwargs: float) -> StatePoint:
    """Resuelve el estado completo a partir de un par independiente.

    Parámetros
    ----------
    fluid : str
        Nombre del fluido aceptado por CoolProp (p.ej. ``"Water"``).
    pair : PairCode
        Código del par independiente. Uno de
        ``"TP"``, ``"PH"``, ``"HS"``, ``"PX"``, ``"TX"``, ``"PS"``, ``"TS"``.
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

    try:
        T_K = _solve(cp1, val1, cp2, val2, "T", fluid)
        P_Pa = _solve(cp1, val1, cp2, val2, "P", fluid)
        h_J_per_kg = _solve(cp1, val1, cp2, val2, "H", fluid)
        s_J_per_kg_K = _solve(cp1, val1, cp2, val2, "S", fluid)
        x = _solve(cp1, val1, cp2, val2, "Q", fluid)
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
        h_J_per_kg=float(h_J_per_kg),
        s_J_per_kg_K=float(s_J_per_kg_K),
        x=float(x),
    )


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
