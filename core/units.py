"""Conversiones de unidades y normalizadores de headers entre la UI y el SI.

La UI didáctica trabaja con bar(a), °C, kJ/kg y kJ/(kg·K) por defecto.
``core/`` trabaja siempre en SI (Pa, K, J/kg, J/(kg·K)).

Este módulo ofrece:

1. **Conversiones simples** (``c_to_k``, ``bar_to_pa``, ``kj_to_j``, …)
   para uso directo cuando la unidad ya se conoce.
2. **Parser de headers** (``parse_header``) que extrae (símbolo, unidad)
   de strings tipo ``"T [°C]"`` o ``"h_f [kJ/kg]"``.
3. **Normalizadores tolerantes a notación** (``normalize_*_unit``,
   ``temperature_to_kelvin``, ``pressure_to_pascal``,
   ``specific_from_si``) que reconocen muchas variantes del mismo
   símbolo y devuelven ``None`` cuando no entienden la unidad. Pensado
   para tablas pegadas por alumnos donde el header puede venir como
   ``"kJ/(kg·K)"``, ``"kJ/kg-K"``, ``"kJ kg^-1 K^-1"``, etc.
"""

from __future__ import annotations

import re
from typing import Literal

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------

ZERO_CELSIUS_IN_KELVIN: float = 273.15
BAR_IN_PASCAL: float = 1.0e5
KILO: float = 1.0e3

SpecificPrefix = Literal["si", "kilo"]
TemperatureCanonical = Literal["C", "K"]
PressureCanonical = Literal["bar", "Pa", "kPa", "MPa"]


# ---------------------------------------------------------------------
# Conversiones simples
# ---------------------------------------------------------------------


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
    """Convierte una magnitud específica de kJ/(kg[·K]) a J/(kg[·K])."""
    return value_kj * KILO


def j_to_kj(value_j: float) -> float:
    """Convierte una magnitud específica de J/(kg[·K]) a kJ/(kg[·K])."""
    return value_j / KILO


# ---------------------------------------------------------------------
# Parser de headers (símbolo + unidad)
# ---------------------------------------------------------------------

# Captura "símbolo [unidad]" o "símbolo (unidad)". Si no hay unidad,
# captura solo el símbolo.
_HEADER_RE = re.compile(r"^(.+?)(?:\s*[\[\(](.+?)[\]\)])?\s*$")


def parse_header(name: str) -> tuple[str, str]:
    """Separa ``"símbolo [unidad]"`` en ``(símbolo, unidad)``.

    El símbolo se devuelve en minúsculas, sin guiones bajos. La unidad
    se devuelve preservando el case (la normalización es responsabilidad
    de ``normalize_*_unit``).

    Ejemplos
    --------
    >>> parse_header("T [°C]")
    ('t', '°C')
    >>> parse_header("h_f [kJ/kg]")
    ('hf', 'kJ/kg')
    >>> parse_header("s [kJ/(kg·K)]")
    ('s', 'kJ/(kg·K)')
    >>> parse_header("Temperature")
    ('temperature', '')
    """
    s = str(name).strip()
    m = _HEADER_RE.match(s)
    if not m:
        return s.lower().replace("_", ""), ""
    base = m.group(1).strip().lower().replace("_", "")
    unit = (m.group(2) or "").strip()
    return base, unit


# ---------------------------------------------------------------------
# Normalizadores tolerantes a notación
# ---------------------------------------------------------------------


def normalize_temperature_unit(unit: str) -> TemperatureCanonical | None:
    """Reconoce variantes de °C y K; devuelve la forma canónica."""
    u = unit.strip().lower()
    if u in ("", "°c", "ºc", "c", "celsius", "deg c", "degc"):
        return "C"
    if u in ("k", "kelvin"):
        return "K"
    return None


def temperature_to_kelvin(value: float, unit: str) -> float | None:
    """Convierte ``value`` (en ``unit``) a kelvin. Devuelve ``None`` si la
    unidad no se reconoce."""
    canonical = normalize_temperature_unit(unit)
    if canonical == "C":
        return value + ZERO_CELSIUS_IN_KELVIN
    if canonical == "K":
        return value
    return None


def normalize_pressure_unit(unit: str) -> PressureCanonical | None:
    """Reconoce variantes de bar, Pa, kPa, MPa; devuelve la forma canónica."""
    u = unit.strip().lower()
    if u in ("", "bar", "bar(a)", "bara"):
        return "bar"
    if u == "pa":
        return "Pa"
    if u == "kpa":
        return "kPa"
    if u == "mpa":
        return "MPa"
    return None


def pressure_to_pascal(value: float, unit: str) -> float | None:
    """Convierte ``value`` (en ``unit``) a pascal. Devuelve ``None`` si la
    unidad no se reconoce."""
    canonical = normalize_pressure_unit(unit)
    if canonical == "bar":
        return value * BAR_IN_PASCAL
    if canonical == "Pa":
        return value
    if canonical == "kPa":
        return value * 1.0e3
    if canonical == "MPa":
        return value * 1.0e6
    return None


_SPECIFIC_TARGETS = ("/kg", "/kgk")


def normalize_specific_unit(unit: str) -> SpecificPrefix | None:
    """Reconoce variantes de J/kg, kJ/kg, J/(kg·K) y kJ/(kg·K).

    Devuelve ``"kilo"`` si la unidad lleva prefijo kJ, ``"si"`` si lleva
    prefijo J (sin k), y ``None`` si no la reconoce. Si la unidad está
    vacía, asume ``"kilo"`` (default didáctico del proyecto).

    Variantes aceptadas (case-insensitive, con/sin espacios):

    - h, cp: ``J/kg``, ``kJ/kg``, ``J kg^-1``, ``kJ kg^-1``
    - s, cp: ``J/(kg·K)``, ``kJ/(kg·K)``, ``J/(kg K)``, ``J/(kg.K)``,
      ``J/kg-K``, ``J/kg/K``, ``J kg^-1 K^-1`` y todas las variantes
      kJ correspondientes.

    No reconoce ``BTU/lb``, ``cal/g``, ni unidades fuera del SI.
    """
    s = unit.strip().lower()
    if not s:
        # Sin unidad: default didáctico del proyecto = kJ/kg.
        return "kilo"

    if s.startswith("kj"):
        is_kilo = True
        rest = s[2:]
    elif s.startswith("j"):
        is_kilo = False
        rest = s[1:]
    else:
        return None

    # Canonicalización: barrer todos los separadores y dejar solo
    # "/kg" o "/kgk" (h vs s/cp).
    rest = re.sub(r"\^\s*-?\s*1", "", rest)  # ^-1 / ^1
    rest = re.sub(r"\s+", "", rest)  # cualquier whitespace
    for sep in ("·", "*", ".", "-", "(", ")"):
        rest = rest.replace(sep, "")
    if not rest.startswith("/"):
        rest = "/" + rest
    # Drop slashes internos (mantener solo el primero).
    rest = "/" + rest[1:].replace("/", "")

    if rest in _SPECIFIC_TARGETS:
        return "kilo" if is_kilo else "si"
    return None


def specific_from_si(value_si: float, unit: str) -> float | None:
    """Convierte una magnitud específica (h o s) de SI a la unidad dada.

    Devuelve ``None`` si la unidad no es reconocida.
    """
    prefix = normalize_specific_unit(unit)
    if prefix == "kilo":
        return value_si / KILO
    if prefix == "si":
        return value_si
    return None
