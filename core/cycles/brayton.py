"""Turbina de gas: ciclo Brayton abierto, con combustión o con aire estándar — Fase 3.4.

El aire ambiente entra al **compresor** (1 → 2), se calienta en la **cámara
de combustión** quemando el combustible (2 → 3) y se expande en la
**turbina** (3 → 4); los gases de escape salen a la presión ambiente (a la
caldera de recuperación en un ciclo combinado). Cengel & Boles §9-8.

- El aire y los gases son **mezclas de gases ideales** (:mod:`core.ideal_gas`),
  con las propiedades de cada componente variables con T: es el mismo cálculo
  que con las tablas de gas ideal (Cengel A-17 a A-23).
- Compresión y expansión: la isoentrópica sale de la función s°,
  s°(T₂s) = s°(T₁) + R·ln(p₂/p₁), y el rendimiento isoentrópico da el estado
  real (vademecum §10.4).
- **Combustión** completa del combustible (metano, o un gas con los
  componentes de ISO 6976:2016) con el aire: el balance de la cámara
  adiabática, con las entalpías medidas desde 25 °C y el poder calorífico
  inferior de ISO 6976 a 25 °C, da la relación combustible/aire f:
  h_a(T₂) + f·PCI = (1 + f)·h_g(T₃). El exceso de aire λ y la composición
  de los gases salen de la estequiometría (vademecum §16).
- **Aire estándar** (``fuel=None``, Cengel §9-3): el calor entra al aire en
  un intercambiador y por la turbina pasa aire. Es el modelo de los ejemplos
  del libro.

El cálculo es directo (sin TESPy). :func:`brayton_tespy` resuelve la misma
turbina con TESPy (``Compressor`` + ``DiabaticCombustionChamber`` +
``Turbine``, como su tutorial de turbina de gas) como control cruzado.
Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Literal

from scipy.optimize import brentq

from core.combustion.iso6976 import ISO6976Tables, load_tables
from core.ideal_gas import AIR_DRY, T_GAS_MAX_K, FlueGas, combustion_products

__all__ = [
    "METHANE",
    "BraytonInputs",
    "BraytonResult",
    "BraytonTespy",
    "Fuel",
    "GasState",
    "TsLine",
    "brayton_notes",
    "brayton_ts_lines",
    "brayton_tespy",
    "solve_brayton",
    "validate_brayton_inputs",
]

#: Temperatura del combustible y de referencia de los poderes caloríficos (25 °C).
T_FUEL_K = 298.15
_ISO_T_COMBUSTION_C = 25.0
_NOBLE_GASES = frozenset({"helium", "neon", "argon"})


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


@cache
def _iso_tables() -> ISO6976Tables:
    return load_tables()


# ---------------------------------------------------------------------
# Combustible
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Fuel:
    """Combustible gaseoso: composición molar con los componentes de ISO 6976:2016.

    Solo componentes con C, H, N y O (sin azufre ni gases nobles), que se
    queman por completo a CO₂ y H₂O; el N₂ y el CO₂ del combustible pasan a
    los gases sin reaccionar.
    """

    composition: tuple[tuple[str, float], ...]

    @classmethod
    def from_fractions(cls, fractions: Mapping[str, float]) -> Fuel:
        """Combustible a partir de fracciones molares (se normalizan).

        Raises
        ------
        ValueError
            Si un componente no está en ISO 6976, tiene azufre o es un gas
            noble, o si las fracciones son negativas o suman cero.
        """
        tables = _iso_tables()
        unknown = [n for n in fractions if n not in tables.components]
        if unknown:
            raise ValueError(f"Componentes que no están en ISO 6976:2016: {unknown}.")
        not_supported = [n for n in fractions if tables.components[n].e > 0 or n in _NOBLE_GASES]
        if not_supported:
            raise ValueError(
                f"Componentes no soportados (azufre o gases nobles): {not_supported}. La "
                "turbina de gas quema combustibles con C, H, N y O."
            )
        if any(x < 0.0 or not math.isfinite(x) for x in fractions.values()):
            raise ValueError("Las fracciones del combustible no pueden ser negativas.")
        total = sum(fractions.values())
        if total <= 0.0:
            raise ValueError("La composición del combustible está vacía.")
        comp = tuple((n, x / total) for n, x in fractions.items() if x > 0.0)
        fuel = cls(comp)
        if fuel.lhv_J_per_kg <= 0.0:
            raise ValueError("El combustible no tiene componentes que se quemen.")
        return fuel

    @property
    def atoms(self) -> tuple[float, float, float, float]:
        """Átomos (C, H, N, O) por mol de combustible."""
        rows = _iso_tables().components
        a = sum(x * rows[n].a for n, x in self.composition)
        b = sum(x * rows[n].b for n, x in self.composition)
        c = sum(x * rows[n].c for n, x in self.composition)
        d = sum(x * rows[n].d for n, x in self.composition)
        return a, b, c, d

    @property
    def M_kg_per_mol(self) -> float:
        rows = _iso_tables().components
        return sum(x * rows[n].M_kg_per_kmol for n, x in self.composition) / 1e3

    @property
    def hhv_molar_J_per_mol(self) -> float:
        """Poder calorífico superior molar de gas ideal a 25 °C (ISO 6976:2016, tabla 3)."""
        tables = _iso_tables()
        return 1e3 * sum(
            x * tables.calorific_values[n][_ISO_T_COMBUSTION_C] for n, x in self.composition
        )

    @property
    def lhv_molar_J_per_mol(self) -> float:
        """Poder calorífico inferior molar a 25 °C: PCS − Σ xⱼ·(bⱼ/2)·L₀ (ISO 6976:2016)."""
        tables = _iso_tables()
        L0 = 1e3 * tables.L0_water_by_T[_ISO_T_COMBUSTION_C]
        b = self.atoms[1]
        return self.hhv_molar_J_per_mol - b / 2.0 * L0

    @property
    def lhv_J_per_kg(self) -> float:
        """Poder calorífico inferior por kilogramo (PCI), J/kg."""
        return self.lhv_molar_J_per_mol / self.M_kg_per_mol

    @property
    def hhv_J_per_kg(self) -> float:
        return self.hhv_molar_J_per_mol / self.M_kg_per_mol

    @property
    def name(self) -> str:
        if len(self.composition) == 1:
            return self.composition[0][0]
        return "gas natural"


#: Metano puro.
METHANE = Fuel.from_fractions({"methane": 1.0})


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BraytonInputs:
    """Datos de la turbina de gas, en SI.

    ``dp_combustor`` es la caída de presión en la cámara como fracción de la
    presión de entrada (p₃ = p₂·(1 − Δp)). ``fuel=None`` es el modelo de aire
    estándar (Cengel §9-3): el calor entra al aire y no hay combustible. El
    combustible entra a 25 °C (la referencia de su poder calorífico).
    """

    pressure_ratio: float
    T_turbine_in_K: float
    eta_compressor: float = 1.0
    eta_turbine: float = 1.0
    T_amb_K: float = 288.15
    p_amb_Pa: float = 101_325.0
    dp_combustor: float = 0.0
    air: FlueGas = AIR_DRY
    fuel: Fuel | None = METHANE
    m_air_kg_s: float = 1.0

    @property
    def air_standard(self) -> bool:
        return self.fuel is None

    @property
    def is_ideal(self) -> bool:
        return self.eta_compressor == 1.0 and self.eta_turbine == 1.0 and self.dp_combustor == 0.0


@dataclass(frozen=True)
class GasState:
    """Un estado del aire o de los gases: T, p, h (desde 25 °C) y s absoluta (SI)."""

    label: str
    T_K: float
    P_Pa: float
    h_J_per_kg: float
    s_J_per_kg_K: float
    medium: Literal["aire", "gases"]


@dataclass(frozen=True)
class BraytonResult:
    """Estados, caudales y balances de la turbina de gas, en SI.

    ``states`` son 1, 2s, 2, 3, 4s y 4 (salida del compresor isoentrópica y
    real, salida de la turbina isoentrópica y real). Las energías específicas
    van por kilogramo de **aire** (como las potencias, que se escalan con
    ``m_air_kg_s``); la turbina mueve 1 + f kg de gases por kg de aire.
    """

    inputs: BraytonInputs
    states: tuple[GasState, ...]
    gas: FlueGas
    fuel_air_ratio: float
    excess_air: float | None
    air_per_mol_fuel: float | None

    def state(self, label: str) -> GasState:
        return next(s for s in self.states if s.label == label)

    @property
    def w_compressor_J_per_kg(self) -> float:
        return self.state("2").h_J_per_kg - self.state("1").h_J_per_kg

    @property
    def w_turbine_J_per_kg(self) -> float:
        """Trabajo de la turbina por kg de aire: (1 + f)·(h₃ − h₄)."""
        return (1.0 + self.fuel_air_ratio) * (
            self.state("3").h_J_per_kg - self.state("4").h_J_per_kg
        )

    @property
    def w_net_J_per_kg(self) -> float:
        return self.w_turbine_J_per_kg - self.w_compressor_J_per_kg

    @property
    def q_in_J_per_kg(self) -> float:
        """Calor que entra por kg de aire: f·PCI (combustión) o h₃ − h₂ (aire estándar)."""
        fuel = self.inputs.fuel
        if fuel is None:
            return self.state("3").h_J_per_kg - self.state("2").h_J_per_kg
        return self.fuel_air_ratio * fuel.lhv_J_per_kg

    @property
    def eta_th(self) -> float:
        return self.w_net_J_per_kg / self.q_in_J_per_kg

    @property
    def back_work_ratio(self) -> float:
        """Relación de trabajo de retroceso: w_C / w_T (Cengel §9-8)."""
        return self.w_compressor_J_per_kg / self.w_turbine_J_per_kg

    @property
    def heat_rate_kJ_per_kWh(self) -> float:
        """Consumo específico de calor: 3600 / η (kJ por kWh producido)."""
        return 3600.0 / self.eta_th

    @property
    def T_exhaust_K(self) -> float:
        return self.state("4").T_K

    # --- caudales y potencias (W) ---------------------------------------
    @property
    def m_air_kg_s(self) -> float:
        return self.inputs.m_air_kg_s

    @property
    def m_fuel_kg_s(self) -> float:
        return self.m_air_kg_s * self.fuel_air_ratio

    @property
    def m_gas_kg_s(self) -> float:
        return self.m_air_kg_s + self.m_fuel_kg_s

    @property
    def W_compressor_W(self) -> float:
        return self.m_air_kg_s * self.w_compressor_J_per_kg

    @property
    def W_turbine_W(self) -> float:
        return self.m_air_kg_s * self.w_turbine_J_per_kg

    @property
    def W_net_W(self) -> float:
        return self.m_air_kg_s * self.w_net_J_per_kg

    @property
    def Q_in_W(self) -> float:
        return self.m_air_kg_s * self.q_in_J_per_kg


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


def validate_brayton_inputs(inputs: BraytonInputs) -> None:
    """Verifica que la turbina de gas tenga sentido físico antes de calcular.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno: qué dato está fuera de rango y por qué.
    """
    if not (math.isfinite(inputs.pressure_ratio) and inputs.pressure_ratio > 1.0):
        raise ValueError(
            f"La relación de presiones ({inputs.pressure_ratio:g}) tiene que ser mayor que 1: "
            "el compresor sube la presión."
        )
    if inputs.pressure_ratio > 60.0:
        raise ValueError(
            f"La relación de presiones ({inputs.pressure_ratio:g}) es demasiado alta: las "
            "turbinas de gas trabajan hasta unos 40, y por encima el aire deja de comportarse "
            "como gas ideal."
        )
    for name, eta in (
        ("del compresor", inputs.eta_compressor),
        ("de la turbina", inputs.eta_turbine),
    ):
        if not 0.3 <= eta <= 1.0:
            raise ValueError(
                f"El rendimiento isoentrópico {name} ({eta:g}) tiene que estar entre 0,3 y 1 "
                "(1 = isoentrópico; vademecum §10.4)."
            )
    if not 0.0 <= inputs.dp_combustor < 0.2:
        raise ValueError(
            "La caída de presión en la cámara de combustión tiene que estar entre 0 y 20 % "
            "(típico: 3 a 5 %)."
        )
    if not 223.15 <= inputs.T_amb_K <= 333.15:
        raise ValueError(
            f"La temperatura ambiente ({_degC(inputs.T_amb_K)}) tiene que estar entre −50 y 60 °C."
        )
    if not 50e3 <= inputs.p_amb_Pa <= 110e3:
        raise ValueError(
            "La presión ambiente tiene que estar entre 0,5 y 1,1 bar (del nivel del mar a "
            "unos 5000 m de altura)."
        )
    if not inputs.T_turbine_in_K <= T_GAS_MAX_K:
        raise ValueError(
            f"La temperatura de entrada a la turbina ({_degC(inputs.T_turbine_in_K)}) supera "
            f"el rango del modelo ({_degC(T_GAS_MAX_K)}); las turbinas más modernas llegan a "
            "unos 1600 °C."
        )
    if not (math.isfinite(inputs.m_air_kg_s) and inputs.m_air_kg_s > 0.0):
        raise ValueError("El caudal de aire tiene que ser positivo.")


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _products(inputs: BraytonInputs, f: float) -> tuple[FlueGas, float, float]:
    """Gases con relación combustible/aire f: (gases, λ, moles de aire por mol de combustible)."""
    fuel = inputs.fuel
    assert fuel is not None
    air = inputs.air
    _, n_air_st = combustion_products(air, fuel.atoms, 1.0)
    f_st = fuel.M_kg_per_mol / (n_air_st * air.M_kg_per_mol)
    lam = f_st / f
    gas, n_air = combustion_products(air, fuel.atoms, lam)
    return gas, lam, n_air


def _stoichiometric_fuel_air_ratio(inputs: BraytonInputs) -> float:
    fuel = inputs.fuel
    assert fuel is not None
    _, n_air_st = combustion_products(inputs.air, fuel.atoms, 1.0)
    return fuel.M_kg_per_mol / (n_air_st * inputs.air.M_kg_per_mol)


def solve_brayton(inputs: BraytonInputs) -> BraytonResult:
    """Turbina de gas: compresor, cámara de combustión (o aire estándar) y turbina.

    Raises
    ------
    ValueError
        Si los datos no tienen sentido (:func:`validate_brayton_inputs`), la TIT
        no supera la salida del compresor o pediría menos aire que el
        estequiométrico.
    """
    validate_brayton_inputs(inputs)
    air = inputs.air
    p1 = inputs.p_amb_Pa
    p2 = p1 * inputs.pressure_ratio
    p3 = p2 * (1.0 - inputs.dp_combustor)
    p4 = p1
    T1 = inputs.T_amb_K
    T3 = inputs.T_turbine_in_K

    # Compresor (Cengel §9-8): s°(T₂s) = s°(T₁) + R·ln(p₂/p₁).
    h1 = air.h(T1)
    T2s = air.T_isentropic(T1, p1, p2)
    h2s = air.h(T2s)
    h2 = h1 + (h2s - h1) / inputs.eta_compressor
    T2 = air.T_from_h(h2)
    if T3 <= T2 + 1.0:
        raise ValueError(
            f"La temperatura de entrada a la turbina ({_degC(T3)}) tiene que superar la del "
            f"aire que sale del compresor ({_degC(T2)}): la cámara calienta el aire. Subí la "
            "TIT o bajá la relación de presiones."
        )

    # Cámara de combustión: h_a(T₂) + f·PCI = (1 + f)·h_g(T₃) (entalpías desde 25 °C).
    fuel = inputs.fuel
    if fuel is None:
        gas = air
        f = 0.0
        lam: float | None = None
        n_air: float | None = None
    else:
        lhv = fuel.lhv_J_per_kg
        f_st = _stoichiometric_fuel_air_ratio(inputs)

        def residual(f_: float) -> float:
            gas_, _, _ = _products(inputs, f_)
            return h2 + f_ * lhv - (1.0 + f_) * gas_.h(T3)

        if residual(f_st) <= 0.0:
            raise ValueError(
                f"Con la TIT de {_degC(T3)} haría falta quemar más combustible que el que "
                "admite el aire (λ < 1, combustión incompleta). Bajá la TIT."
            )
        f = float(brentq(residual, 1e-9 * f_st, f_st, xtol=1e-14, rtol=1e-13))
        gas, lam, n_air = _products(inputs, f)

    # Turbina: s°(T₄s) = s°(T₃) − R·ln(p₃/p₄).
    gas = FlueGas(gas.y, p4)
    h3 = gas.h(T3)
    T4s = gas.T_isentropic(T3, p3, p4)
    h4s = gas.h(T4s)
    h4 = h3 - inputs.eta_turbine * (h3 - h4s)
    T4 = gas.T_from_h(h4)

    def state(label: str, medium: Literal["aire", "gases"], T: float, p: float, h: float):
        mix = air if medium == "aire" else gas
        return GasState(label, T, p, h, mix.s(T, p), medium)

    states = (
        state("1", "aire", T1, p1, h1),
        state("2s", "aire", T2s, p2, h2s),
        state("2", "aire", T2, p2, h2),
        state("3", "gases", T3, p3, h3),
        state("4s", "gases", T4s, p4, h4s),
        state("4", "gases", T4, p4, h4),
    )
    return BraytonResult(
        inputs=inputs,
        states=states,
        gas=gas,
        fuel_air_ratio=f,
        excess_air=lam,
        air_per_mol_fuel=n_air,
    )


# ---------------------------------------------------------------------
# Diagrama T–s
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TsLine:
    """Una línea del diagrama T–s de la turbina de gas (SI).

    ``kind``: ``"isobar"`` (referencia), ``"heat"`` (calor a presión constante:
    cámara o caldera de recuperación), ``"isentropic"`` (2s o 4s),
    ``"actual"`` (compresión o expansión real, solo de referencia) o
    ``"composition"`` (el salto de la cámara: cambia la composición).
    """

    kind: Literal["isobar", "heat", "isentropic", "actual", "composition"]
    medium: Literal["aire", "gases"]
    p_Pa: float
    T_K: tuple[float, ...]
    s_J_per_kg_K: tuple[float, ...]


def brayton_ts_lines(
    result: BraytonResult, T_stack_K: float | None = None, n: int = 30
) -> list[TsLine]:
    """Isobaras y procesos para dibujar la turbina de gas en un diagrama T–s.

    Con entropías absolutas (:meth:`FlueGas.s`), el aire (1 → 2) y los gases
    (3 → 4) quedan en la misma escala aunque cambie la composición. Con
    ``T_stack_K`` se suma el enfriamiento de los gases en la caldera de
    recuperación (4 → chimenea, a la presión del escape).
    """
    air, gas = result.inputs.air, result.gas
    s1, s2s, s2 = result.state("1"), result.state("2s"), result.state("2")
    s3, s4s, s4 = result.state("3"), result.state("4s"), result.state("4")

    def isobar(
        kind: Literal["isobar", "heat"],
        medium: Literal["aire", "gases"],
        p: float,
        T_lo: float,
        T_hi: float,
    ) -> TsLine:
        mix = air if medium == "aire" else gas
        T_lo = max(T_lo, mix.T_min_K + 1.0)
        temps = [T_lo + (T_hi - T_lo) * k / (n - 1) for k in range(n)]
        return TsLine(kind, medium, p, tuple(temps), tuple(mix.s(T, p) for T in temps))

    def segment(
        kind: Literal["isentropic", "actual", "composition"],
        medium: Literal["aire", "gases"],
        a: GasState,
        b_T: float,
        b_s: float,
    ) -> TsLine:
        return TsLine(kind, medium, a.P_Pa, (a.T_K, b_T), (a.s_J_per_kg_K, b_s))

    T_low = min(s1.T_K, T_stack_K or s1.T_K) - 20.0
    lines = [
        isobar("isobar", "aire", s1.P_Pa, T_low, s2.T_K + 60.0),
        isobar("isobar", "aire", s2.P_Pa, s1.T_K - 20.0, s2.T_K + 60.0),
        isobar("isobar", "gases", s4.P_Pa, min(s4s.T_K, T_stack_K or s4s.T_K) - 40.0, s3.T_K),
        segment("isentropic", "aire", s1, s2s.T_K, s2s.s_J_per_kg_K),
        segment("actual", "aire", s1, s2.T_K, s2.s_J_per_kg_K),
        segment("composition", "gases", s2, s2.T_K, gas.s(s2.T_K, s3.P_Pa)),
        isobar("heat", "gases", s3.P_Pa, s2.T_K, s3.T_K),
        segment("isentropic", "gases", s3, s4s.T_K, s4s.s_J_per_kg_K),
        segment("actual", "gases", s3, s4.T_K, s4.s_J_per_kg_K),
    ]
    if T_stack_K is not None:
        lines.append(isobar("heat", "gases", s4.P_Pa, T_stack_K, s4.T_K))
    return lines


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def brayton_notes(result: BraytonResult) -> list[str]:
    """Observaciones didácticas sobre la turbina de gas calculada (markdown)."""
    notes: list[str] = []
    rbw = result.back_work_ratio
    if rbw > 0.5:
        notes.append(
            f"El compresor se lleva el {rbw * 100:.0f} % del trabajo de la turbina (relación de "
            "trabajo de retroceso): por eso en una turbina de gas los rendimientos del "
            "compresor y de la turbina pesan tanto (Cengel §9-8). En un ciclo de vapor la bomba "
            "consume menos del 2 %."
        )
    if result.excess_air is not None:
        o2 = result.gas.mole_fractions["O2"]
        notes.append(
            f"La turbina quema con un exceso de aire λ = {result.excess_air:.3g}: el aire de más "
            "baja la temperatura de los gases hasta la que soportan los álabes. Por eso el "
            f"escape todavía tiene {o2 * 100:.3g} % de O₂ (alcanza para quemadores "
            "suplementarios en la caldera de recuperación)."
        )
    else:
        notes.append(
            "Modelo de aire estándar (Cengel §9-3): la combustión se reemplaza por un "
            "intercambiador que le pasa calor al aire, y por la turbina pasa aire. Es una "
            "simplificación: no hay combustible ni cambia la composición."
        )
    return notes


# ---------------------------------------------------------------------
# Control con TESPy
# ---------------------------------------------------------------------

_TESPY_FUEL_NAMES: dict[str, str] = {
    "methane": "CH4",
    "ethane": "C2H6",
    "propane": "C3H8",
    "nitrogen": "N2",
    "carbon dioxide": "CO2",
    "hydrogen": "H2",
}


@dataclass(frozen=True)
class BraytonTespy:
    """La misma turbina de gas resuelta con TESPy (por kg de aire, SI)."""

    T2_K: float
    T4_K: float
    fuel_air_ratio: float
    w_compressor_J_per_kg: float
    w_turbine_J_per_kg: float
    q_in_J_per_kg: float
    excess_air: float | None

    @property
    def w_net_J_per_kg(self) -> float:
        return self.w_turbine_J_per_kg - self.w_compressor_J_per_kg

    @property
    def eta_th(self) -> float:
        return self.w_net_J_per_kg / self.q_in_J_per_kg


def brayton_tespy(result: BraytonResult) -> BraytonTespy:
    """Resuelve la misma turbina de gas con TESPy, como control cruzado.

    Combustión: ``Compressor`` + ``DiabaticCombustionChamber`` + ``Turbine``,
    como el tutorial de turbina de gas de TESPy 0.11; aire estándar: la cámara
    es un ``SimpleHeatExchanger``. TESPy evalúa cada componente a su presión
    parcial con la ecuación de estado real, así que difiere unas décimas de %
    del cálculo con gases ideales. Arranca de los valores del cálculo directo
    (sin ellos, el método de Newton puede no converger).

    Raises
    ------
    ValueError
        Si el combustible tiene componentes que TESPy no quema o el cálculo no
        converge.
    """
    from tespy.components import (
        Compressor,
        DiabaticCombustionChamber,
        SimpleHeatExchanger,
        Sink,
        Source,
        Turbine,
    )
    from tespy.connections import Connection, Ref

    from core.cycles.tespy_utils import new_network, solve

    inputs = result.inputs
    fuel = inputs.fuel
    nw = new_network()
    compressor = Compressor("compresor")
    turbine = Turbine("turbina")
    air_in, exhaust = Source("aire"), Sink("escape")
    c1 = Connection(air_in, "out1", compressor, "in1", label="1")
    if fuel is None:
        heater = SimpleHeatExchanger("calentador")
        c2 = Connection(compressor, "out1", heater, "in1", label="2")
        c3 = Connection(heater, "out1", turbine, "in1", label="3")
        conns = [c1, c2, c3]
        heater.set_attr(pr=1.0 - inputs.dp_combustor)
    else:
        missing = [n for n, _ in fuel.composition if n not in _TESPY_FUEL_NAMES]
        if missing:
            raise ValueError(f"TESPy no tiene en su modelo de combustión: {missing}.")
        chamber = DiabaticCombustionChamber("cámara de combustión")
        fuel_in = Source("combustible")
        c2 = Connection(compressor, "out1", chamber, "in1", label="2")
        c3 = Connection(chamber, "out1", turbine, "in1", label="3")
        c5 = Connection(fuel_in, "out1", chamber, "in2", label="5")
        conns = [c1, c2, c3, c5]
        chamber.set_attr(pr=1.0 - inputs.dp_combustor, eta=1.0)
        rows = _iso_tables().components
        M = fuel.M_kg_per_mol * 1e3
        w_fuel = {_TESPY_FUEL_NAMES[n]: x * rows[n].M_kg_per_kmol / M for n, x in fuel.composition}
        c5.set_attr(T=T_FUEL_K, fluid=w_fuel, p=Ref(c2, 1.05, 0), m0=result.fuel_air_ratio)
    c4 = Connection(turbine, "out1", exhaust, "in1", label="4")
    nw.add_conns(*conns, c4)
    c1.set_attr(
        T=inputs.T_amb_K,
        p=inputs.p_amb_Pa,
        m=1.0,
        fluid={s: w for s, w in inputs.air.mass_fractions.items() if w > 0.0},
    )
    compressor.set_attr(pr=inputs.pressure_ratio, eta_s=inputs.eta_compressor)
    turbine.set_attr(eta_s=inputs.eta_turbine)
    c3.set_attr(T=inputs.T_turbine_in_K)
    c4.set_attr(p=inputs.p_amb_Pa, T0=result.T_exhaust_K)
    c2.set_attr(T0=result.state("2").T_K)
    solve(nw, what="la turbina de gas con TESPy")
    w_c = c2.h.val_SI - c1.h.val_SI
    w_t = c3.m.val_SI * (c3.h.val_SI - c4.h.val_SI)
    if fuel is None:
        f = 0.0
        q_in = c3.h.val_SI - c2.h.val_SI
        lam = None
    else:
        f = c5.m.val_SI
        q_in = chamber.ti.val_SI
        lam = float(chamber.lamb.val_SI)
    return BraytonTespy(
        T2_K=c2.T.val_SI,
        T4_K=c4.T.val_SI,
        fuel_air_ratio=f,
        w_compressor_J_per_kg=w_c,
        w_turbine_J_per_kg=w_t,
        q_in_J_per_kg=q_in,
        excess_air=lam,
    )
