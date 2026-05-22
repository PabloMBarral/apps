"""Sistema global de unidades — Fase 1.4.

Define tres sistemas de unidades (``SI``, ``Técnico``, ``Inglés``) y
expone cuatro funciones públicas que la UI usa para convertir y
mostrar magnitudes termodinámicas. El core de cálculo (``core.fluids``,
``core.isentropic``, ``core.interpolation``, ``core.combustion``) sigue
operando en SI internamente; este módulo solo vive en el borde UI.

Magnitudes soportadas (``QuantityKind``)
----------------------------------------

============ =========== ============== ============================
kind         SI          Técnico        Inglés
============ =========== ============== ============================
temperature  K           °C             °F
pressure     Pa          bar            psia
specific_enthalpy  J/kg  kJ/kg          Btu/lb
specific_entropy   J/(kg·K)  kJ/(kg·K)  Btu/(lb·°R)
specific_volume    m³/kg     m³/kg      ft³/lb
specific_heat      J/(kg·K)  kJ/(kg·K)  Btu/(lb·°R)
============ =========== ============== ============================

Constantes NIST (exactas)
-------------------------
- 1 lb = 0.45359237 kg.
- 1 ft = 0.3048 m.
- 1 in = 0.0254 m, 1 in² = 6.4516e-4 m².
- 1 lbf = 4.4482216152605 N.
- 1 psi = 6894.757293168361 Pa  (= lbf/in²).
- 1 Btu_IT/lb = 2326 J/kg  (definición exacta de la tabla internacional).
- 1 Btu_IT/(lb·°R) = 4186.8 J/(kg·K)  (consistente con ΔT_R = ΔT_K · 5/9).
- T conversions:
    K → °C: subtraer 273.15;
    K → °F: multiplicar por 9/5 y subtraer 459.67.

Las conversiones son siempre por factores fijos NIST (no se usa CoolProp
para unidades; CoolProp solo aparece para propiedades termofísicas).
"""

from __future__ import annotations

from typing import Literal

UnitSystem = Literal["SI", "Técnico", "Inglés"]
QuantityKind = Literal[
    "temperature",
    "pressure",
    "specific_enthalpy",
    "specific_entropy",
    "specific_volume",
    "specific_heat",
]

DEFAULT_SYSTEM: UnitSystem = "Técnico"
SUPPORTED_SYSTEMS: tuple[UnitSystem, ...] = ("SI", "Técnico", "Inglés")

# ---------------------------------------------------------------------
# Constantes de conversión (todas NIST CODATA, exactas)
# ---------------------------------------------------------------------

_LB_PER_KG: float = 0.45359237  # exacto
_FT_PER_M: float = 0.3048  # exacto
_PSI_PER_PA: float = 6894.757293168361  # NIST, 13 cifras significativas
_BTU_PER_LB_J_PER_KG: float = 2326.0  # 1 Btu_IT/lb = 2326 J/kg, exacto
_BTU_PER_LB_R_J_PER_KG_K: float = 4186.8  # 1 Btu_IT/(lb·°R) = 4186.8 J/(kg·K)
_FT3_PER_LB_TO_M3_PER_KG: float = (_FT_PER_M**3) / _LB_PER_KG
# ≈ 0.062427960576145 m³/kg por ft³/lb (inverso: 16.0184633739537 ft³/lb por m³/kg)

# ---------------------------------------------------------------------
# Tabla central (kind, system) → (factor, offset, label)
# ---------------------------------------------------------------------
#
# Convención:
#   value_user = factor · value_si + offset
#   value_si   = (value_user − offset) / factor
#
# Para magnitudes sin offset (todas excepto temperatura), offset = 0.
# Para temperaturas:
#   °C = K − 273.15           → factor = 1, offset = −273.15
#   °F = K · 9/5 − 459.67     → factor = 9/5, offset = −459.67

_UNIT_TABLE: dict[QuantityKind, dict[UnitSystem, tuple[float, float, str]]] = {
    "temperature": {
        "SI": (1.0, 0.0, "K"),
        "Técnico": (1.0, -273.15, "°C"),
        "Inglés": (9.0 / 5.0, -459.67, "°F"),
    },
    "pressure": {
        "SI": (1.0, 0.0, "Pa"),
        "Técnico": (1.0e-5, 0.0, "bar"),
        "Inglés": (1.0 / _PSI_PER_PA, 0.0, "psia"),
    },
    "specific_enthalpy": {
        "SI": (1.0, 0.0, "J/kg"),
        "Técnico": (1.0e-3, 0.0, "kJ/kg"),
        "Inglés": (1.0 / _BTU_PER_LB_J_PER_KG, 0.0, "Btu/lb"),
    },
    "specific_entropy": {
        "SI": (1.0, 0.0, "J/(kg·K)"),
        "Técnico": (1.0e-3, 0.0, "kJ/(kg·K)"),
        "Inglés": (1.0 / _BTU_PER_LB_R_J_PER_KG_K, 0.0, "Btu/(lb·°R)"),
    },
    "specific_volume": {
        "SI": (1.0, 0.0, "m³/kg"),
        "Técnico": (1.0, 0.0, "m³/kg"),
        "Inglés": (1.0 / _FT3_PER_LB_TO_M3_PER_KG, 0.0, "ft³/lb"),
    },
    "specific_heat": {
        "SI": (1.0, 0.0, "J/(kg·K)"),
        "Técnico": (1.0e-3, 0.0, "kJ/(kg·K)"),
        "Inglés": (1.0 / _BTU_PER_LB_R_J_PER_KG_K, 0.0, "Btu/(lb·°R)"),
    },
}


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------


def _entry(kind: QuantityKind, system: UnitSystem) -> tuple[float, float, str]:
    if kind not in _UNIT_TABLE:
        raise ValueError(f"QuantityKind no soportado: {kind!r}. Opciones: {list(_UNIT_TABLE)}.")
    by_system = _UNIT_TABLE[kind]
    if system not in by_system:
        raise ValueError(f"UnitSystem no soportado: {system!r}. Opciones: {list(by_system)}.")
    return by_system[system]


# ---------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------


def unit_label(kind: QuantityKind, system: UnitSystem) -> str:
    """Etiqueta textual de la unidad. Ej.: ``'K'``, ``'°C'``, ``'kJ/kg'``."""
    return _entry(kind, system)[2]


def convert_from_si(value_si: float, kind: QuantityKind, system: UnitSystem) -> float:
    """Valor en SI → valor en el sistema dado (sin formato).

    Aplica la transformación afín ``value_user = factor · value_si + offset``.
    """
    factor, offset, _ = _entry(kind, system)
    return factor * float(value_si) + offset


def convert_to_si(value_user: float, kind: QuantityKind, system: UnitSystem) -> float:
    """Valor en el sistema dado → valor en SI (inversa de :func:`convert_from_si`).

    Aplica ``value_si = (value_user − offset) / factor``.
    """
    factor, offset, _ = _entry(kind, system)
    return (float(value_user) - offset) / factor


def parse_user_input(value: float, kind: QuantityKind, system: UnitSystem) -> float:
    """Alias didáctico de :func:`convert_to_si` para uso desde la UI."""
    return convert_to_si(value, kind, system)


def format_quantity(
    value_si: float,
    kind: QuantityKind,
    system: UnitSystem,
    *,
    precision: int = 4,
) -> str:
    """Convierte un valor SI al sistema dado y lo formatea con su unidad.

    Usa la notación ``g`` (cifras significativas). Para una elección de
    decimales fijos, llamá :func:`convert_from_si` y :func:`unit_label`
    por separado.

    Ejemplos
    --------
    >>> format_quantity(298.15, "temperature", "Técnico")
    '25 °C'
    >>> format_quantity(298.15, "temperature", "Inglés")
    '77 °F'
    """
    converted = convert_from_si(value_si, kind, system)
    label = unit_label(kind, system)
    return f"{converted:.{precision}g} {label}"
