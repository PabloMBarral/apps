"""Sistema global de unidades — Fase 1.4.

Define tres sistemas de unidades (``SI``, ``Técnico``, ``Inglés``) y
expone cuatro funciones públicas que la UI usa para convertir y
mostrar magnitudes termodinámicas. El core de cálculo (``core.fluids``,
``core.isentropic``, ``core.interpolation``, ``core.combustion``) sigue
operando en SI internamente; este módulo solo vive en el borde UI.

Magnitudes soportadas (``QuantityKind``)
----------------------------------------

====================== =========== ============== ============================
kind                   SI          Técnico        Inglés
====================== =========== ============== ============================
temperature            K           °C             °F
temperature_difference K           °C             °F  (sin offset: ΔT)
pressure               Pa          bar            psia
specific_enthalpy      J/kg        kJ/kg          Btu/lb
specific_entropy       J/(kg·K)    kJ/(kg·K)      Btu/(lb·°R)
specific_volume        m³/kg       m³/kg          ft³/lb
specific_heat          J/(kg·K)    kJ/(kg·K)      Btu/(lb·°R)
density                kg/m³       kg/m³          lb/ft³
speed                  m/s         m/s            ft/s
dynamic_viscosity      Pa·s        Pa·s           lb/(ft·s)
thermal_conductivity   W/(m·K)     W/(m·K)        Btu/(h·ft·°F)
diffusivity            m²/s        m²/s           ft²/s
====================== =========== ============== ============================

``specific_enthalpy`` también se usa para energía interna, calor latente
y trabajo específicos (misma unidad). ``diffusivity`` cubre la
viscosidad cinemática ν y la difusividad térmica α.

Constantes NIST (exactas)
-------------------------
- 1 lb = 0.45359237 kg.
- 1 ft = 0.3048 m.
- 1 in = 0.0254 m, 1 in² = 6.4516e-4 m².
- 1 lbf = 4.4482216152605 N.
- 1 psi = 6894.757293168361 Pa  (= lbf/in²).
- 1 Btu_IT/lb = 2326 J/kg  (definición exacta de la tabla internacional).
- 1 Btu_IT/(lb·°R) = 4186.8 J/(kg·K)  (consistente con ΔT_R = ΔT_K · 5/9).
- 1 Btu_IT = 1055.05585262 J, 1 h = 3600 s →
  1 Btu/(h·ft·°F) = 1055.05585262 / 609.6 W/(m·K) ≈ 1.730735 W/(m·K).
- T conversions:
    K → °C: subtraer 273.15;
    K → °F: multiplicar por 9/5 y subtraer 459.67.
- Diferencias de temperatura (vademecum-termo §1.3): Δt[°C] = ΔT[K] y
  Δt[°F] = ΔT[°R] = 9/5 · ΔT[K] — sin offset.

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
    "temperature_difference",
    "density",
    "speed",
    "dynamic_viscosity",
    "thermal_conductivity",
    "diffusivity",
    "mass_flow",
    "power",
    "volume_flow",
    "molar_enthalpy",
    "molar_entropy",
    "mass",
    "energy",
    "length",
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
# Por la misma cuenta, 1 kg/m³ = 0.062427960576145 lb/ft³.
_LB_PER_FT_S_TO_PA_S: float = _LB_PER_KG / _FT_PER_M  # 1 lb/(ft·s) ≈ 1.488164 Pa·s
_BTU_IT_J: float = 1055.05585262  # exacto (Btu de la tabla internacional)
_BTU_PER_H_FT_F_TO_W_PER_M_K: float = _BTU_IT_J / (3600.0 * _FT_PER_M * 5.0 / 9.0)
# ≈ 1.730734666 W/(m·K) por Btu/(h·ft·°F)

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
    "temperature_difference": {
        "SI": (1.0, 0.0, "K"),
        "Técnico": (1.0, 0.0, "°C"),
        "Inglés": (9.0 / 5.0, 0.0, "°F"),
    },
    "density": {
        "SI": (1.0, 0.0, "kg/m³"),
        "Técnico": (1.0, 0.0, "kg/m³"),
        "Inglés": (_FT3_PER_LB_TO_M3_PER_KG, 0.0, "lb/ft³"),
    },
    "speed": {
        "SI": (1.0, 0.0, "m/s"),
        "Técnico": (1.0, 0.0, "m/s"),
        "Inglés": (1.0 / _FT_PER_M, 0.0, "ft/s"),
    },
    "dynamic_viscosity": {
        "SI": (1.0, 0.0, "Pa·s"),
        "Técnico": (1.0, 0.0, "Pa·s"),
        "Inglés": (1.0 / _LB_PER_FT_S_TO_PA_S, 0.0, "lb/(ft·s)"),
    },
    "thermal_conductivity": {
        "SI": (1.0, 0.0, "W/(m·K)"),
        "Técnico": (1.0, 0.0, "W/(m·K)"),
        "Inglés": (1.0 / _BTU_PER_H_FT_F_TO_W_PER_M_K, 0.0, "Btu/(h·ft·°F)"),
    },
    "diffusivity": {
        "SI": (1.0, 0.0, "m²/s"),
        "Técnico": (1.0, 0.0, "m²/s"),
        "Inglés": (1.0 / _FT_PER_M**2, 0.0, "ft²/s"),
    },
    # Fase 3.1 (ciclos). Coherentes con la energía específica de cada sistema,
    # así potencia = caudal · trabajo específico sin factores de conversión:
    # kg/s · kJ/kg = kW y lb/s · Btu/lb = Btu/s (0.45359237 · 2326 = 1055.0559 W).
    "mass_flow": {
        "SI": (1.0, 0.0, "kg/s"),
        "Técnico": (1.0, 0.0, "kg/s"),
        "Inglés": (1.0 / _LB_PER_KG, 0.0, "lb/s"),
    },
    "power": {
        "SI": (1.0, 0.0, "W"),
        "Técnico": (1.0e-3, 0.0, "kW"),
        "Inglés": (1.0 / _BTU_IT_J, 0.0, "Btu/s"),
    },
    # Fase 3.2 (refrigeración): caudal volumétrico en la aspiración del compresor.
    # Coherente con el caudal másico: kg/s · m³/kg = m³/s y lb/s · ft³/lb = ft³/s.
    "volume_flow": {
        "SI": (1.0, 0.0, "m³/s"),
        "Técnico": (1.0, 0.0, "m³/s"),
        "Inglés": (1.0 / _FT_PER_M**3, 0.0, "ft³/s"),
    },
    # Fase 5 (combustión): por mol, como las tablas de entalpías de formación.
    # J/mol = kJ/kmol; 1 Btu/lbmol = 2,326 J/mol (como 1 Btu/lb = 2326 J/kg).
    "molar_enthalpy": {
        "SI": (1.0, 0.0, "J/mol"),
        "Técnico": (1.0, 0.0, "kJ/kmol"),
        "Inglés": (1000.0 / _BTU_PER_LB_J_PER_KG, 0.0, "Btu/lbmol"),
    },
    "molar_entropy": {
        "SI": (1.0, 0.0, "J/(mol·K)"),
        "Técnico": (1.0, 0.0, "kJ/(kmol·K)"),
        "Inglés": (1000.0 / _BTU_PER_LB_R_J_PER_KG_K, 0.0, "Btu/(lbmol·°R)"),
    },
    # Fase 7 (exergía): una masa (sistema cerrado), su exergía y la altura de
    # la energía potencial. Coherentes con la energía específica de cada
    # sistema: kg · kJ/kg = kJ y lb · Btu/lb = Btu (0,45359237 · 2326 J = 1 Btu).
    "mass": {
        "SI": (1.0, 0.0, "kg"),
        "Técnico": (1.0, 0.0, "kg"),
        "Inglés": (1.0 / _LB_PER_KG, 0.0, "lb"),
    },
    "energy": {
        "SI": (1.0, 0.0, "J"),
        "Técnico": (1.0e-3, 0.0, "kJ"),
        "Inglés": (1.0 / _BTU_IT_J, 0.0, "Btu"),
    },
    "length": {
        "SI": (1.0, 0.0, "m"),
        "Técnico": (1.0, 0.0, "m"),
        "Inglés": (1.0 / _FT_PER_M, 0.0, "ft"),
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
