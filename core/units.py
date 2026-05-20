"""Conversiones de unidades entre la UI didáctica y el SI interno.

La UI trabaja con unidades cómodas para el alumno (bar(a), °C, kJ/kg,
kJ/(kg·K)) y `core/` trabaja siempre en SI (Pa, K, J/kg, J/(kg·K)).
Estas funciones son puras y sin estado.
"""

from __future__ import annotations

ZERO_CELSIUS_IN_KELVIN: float = 273.15
BAR_IN_PASCAL: float = 1.0e5
KILO: float = 1.0e3


def c_to_k(t_celsius: float) -> float:
    """Convierte temperatura de °C a K."""
    return t_celsius + ZERO_CELSIUS_IN_KELVIN


def k_to_c(t_kelvin: float) -> float:
    """Convierte temperatura de K a °C."""
    return t_kelvin - ZERO_CELSIUS_IN_KELVIN


def bar_to_pa(p_bar: float) -> float:
    """Convierte presión absoluta de bar a Pa."""
    return p_bar * BAR_IN_PASCAL


def pa_to_bar(p_pa: float) -> float:
    """Convierte presión absoluta de Pa a bar."""
    return p_pa / BAR_IN_PASCAL


def kj_to_j(value_kj: float) -> float:
    """Convierte una magnitud específica de kJ/(kg[·K]) a J/(kg[·K]).

    Sirve igual para entalpía (kJ/kg ↔ J/kg) y para entropía
    (kJ/(kg·K) ↔ J/(kg·K)) porque la conversión de prefijo es la misma.
    """
    return value_kj * KILO


def j_to_kj(value_j: float) -> float:
    """Convierte una magnitud específica de J/(kg[·K]) a kJ/(kg[·K])."""
    return value_j / KILO
