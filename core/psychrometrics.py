"""Aire húmedo (psicrometría) — Fase 4.

Modelo del **vademecum §14** (Çengel & Boles, *Termodinámica*, §14-1 a §14-5):
el aire seco y el vapor de agua son **gases ideales con calores específicos
constantes**, y el vapor nunca supera su presión de saturación. Todo se expresa
**por kg de aire seco**, que es lo que se conserva en los procesos:

- humedad absoluta ω = 0,622·p_v/(p − p_v) (§14.2);
- humedad relativa φ = p_v/p_vs(T) (§14.3) y grado de saturación μ = ω/ω_s
  (§14.4);
- entalpía h = c_p,a·t + ω·(r₀ + c_p,v·t), con t en °C (§14.5);
- volumen específico v = R_a·T/p_a (§14.7), densidad, constante y entropía
  (§14.8 a §14.10) y exergía de flujo ψ = ψ_tm + ψ_qu (§14.11);
- temperatura de bulbo húmedo termodinámica = temperatura de saturación
  adiabática (§14.12.5): T_bh tal que h₁ + (ω_s(T_bh) − ω₁)·h_w(T_bh) =
  h(T_bh, ω_s(T_bh));
- temperatura de punto de rocío: la de saturación a la presión parcial del
  vapor (bajo 0 °C, de escarcha).

**Presión de saturación**: la de IAPWS. Sobre el agua líquida (desde el punto
triple), la de IAPWS-95 de CoolProp, que es la de la tabla A-4 de Cengel;
sobre el hielo, la ecuación de sublimación de IAPWS (2011), que es la de la
tabla A-8. La entalpía del agua líquida y del vapor que entran o salen en los
procesos también es la de las tablas (h_w = h_f(T_w), como Cengel y el
vademecum); la del hielo, la de ASHRAE (−333,4 + 2,1·t kJ/kg).

**Referencias** (las de ASHRAE, *Handbook—Fundamentals*, 2017, cap. 1): h = 0
para el aire seco a 0 °C y para el agua líquida en el punto triple; s = 0 para
el aire seco a 0 °C y 101,325 kPa y para el agua líquida en el punto triple.

CoolProp trae además el modelo de gas real de ASHRAE RP-1485 (Herrmann,
Kretzschmar y Gatley, 2009), con el factor de mejora y los coeficientes
viriales de la mezcla: :func:`coolprop_comparison` lo usa para mostrar cuánto
se aparta el modelo ideal. Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Literal

import CoolProp
import numpy as np
from CoolProp.CoolProp import HAPropsSI
from scipy.optimize import brentq

from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "CHART_PHI_VALUES",
    "COMFORT_PHI",
    "COMFORT_T_K",
    "C_P_AIR",
    "C_P_VAPOR",
    "EPSILON",
    "H_VAPOR_0",
    "MOLAR_RATIO",
    "PAIRS",
    "PAIR_LABELS",
    "P_MAX_PA",
    "P_MIN_PA",
    "P_REF_AIR_PA",
    "P_REF_VAPOR_PA",
    "P_SEA_LEVEL_PA",
    "P_TRIPLE_PA",
    "R_AIR",
    "R_VAPOR",
    "S_VAPOR_0",
    "T_MAX_K",
    "T_MIN_K",
    "T_REF_K",
    "T_TRIPLE_K",
    "VARIABLE_LABELS",
    "Z_MAX_M",
    "Z_MIN_M",
    "ChartLine",
    "ChartWindow",
    "ComparisonRow",
    "CoolPropComparison",
    "DeadState",
    "InputName",
    "MoistAirState",
    "WaterPhase",
    "altitude_from_pressure",
    "chart_grid",
    "chart_lines",
    "chart_window",
    "coolprop_comparison",
    "dew_point",
    "dry_bulb_from_h_omega",
    "humidity_ratio",
    "moist_air_state",
    "moist_enthalpy",
    "omega_from_wet_bulb",
    "pressure_from_altitude",
    "saturation_humidity_ratio",
    "saturation_pressure",
    "saturation_temperature",
    "state_from_T_omega",
    "state_from_T_phi",
    "vapor_pressure",
    "water_enthalpy",
    "water_entropy",
    "water_exergy",
    "wet_bulb_temperature",
]

# ---------------------------------------------------------------------
# Constantes del modelo (vademecum §14; Cengel tabla A-1)
# ---------------------------------------------------------------------

#: Calor específico del aire seco, J/(kg·K) (vademecum §14.5).
C_P_AIR = 1005.0
#: Calor específico del vapor de agua, J/(kg·K) (vademecum §14.5; ASHRAE: 1,86).
C_P_VAPOR = 1864.0
#: r₀: entalpía del vapor a 0 °C medida desde el líquido en el punto triple, J/kg.
H_VAPOR_0 = 2501.0e3
#: Constante del aire seco, J/(kg·K).
R_AIR = 287.0
#: M_a/M_v = 28,97/18,015 (vademecum §14.7).
MOLAR_RATIO = 1.608
#: Constante del vapor de agua, J/(kg·K): R_a·M_a/M_v = 461,5 (Cengel A-1: 0,4615;
#: el vademecum la redondea a 0,462). Con este valor la entropía (§14.10) y la
#: exergía (§14.11) son coherentes: ψ = Σ (h − h₀) − T₀·(s − s₀) componente a
#: componente.
R_VAPOR = R_AIR * MOLAR_RATIO
#: M_v/M_a (vademecum §14.2).
EPSILON = 0.622
#: s₀: entropía del vapor saturado en el punto triple (desde el líquido), J/(kg·K).
S_VAPOR_0 = 9156.0
#: Temperatura de referencia de h y s (0 °C; el punto triple es 0,01 °C).
T_REF_K = 273.15
#: Presión de referencia de la entropía del aire seco.
P_REF_AIR_PA = 101325.0
#: Presión de referencia de la entropía del vapor (la del punto triple).
P_REF_VAPOR_PA = 611.657
#: Punto triple del agua (IAPWS).
T_TRIPLE_K = 273.16
P_TRIPLE_PA = 611.657
#: Presión atmosférica normal (nivel del mar).
P_SEA_LEVEL_PA = 101325.0

#: Entalpía del hielo: h = −333,4 + 2,1·t kJ/kg (ASHRAE HoF 2017, cap. 1).
_H_ICE_0 = -333.4e3
_C_ICE = 2100.0

#: Rangos de la página: de −50 a 200 °C; de 0,4 bar (≈ 7200 m de altura) a 10 bar.
T_MIN_K = 223.15
T_MAX_K = 473.15
P_MIN_PA = 40.0e3
P_MAX_PA = 1.0e6
#: Alturas para la presión atmosférica (la ecuación de ASHRAE vale en la tropósfera).
Z_MIN_M = -500.0
Z_MAX_M = 7000.0

#: Zona de confort de Cengel §14-6: 22–27 °C y φ de 40 a 60 %.
COMFORT_T_K: tuple[float, float] = (295.15, 300.15)
COMFORT_PHI: tuple[float, float] = (0.40, 0.60)
#: Curvas de humedad relativa de la carta.
CHART_PHI_VALUES: tuple[float, ...] = tuple(k / 10 for k in range(1, 11))

#: Por debajo de esta humedad absoluta el aire se trata como seco (sin punto de rocío).
_OMEGA_DRY = 1e-12

# Ecuación de sublimación de IAPWS R14-08(2011): ln(p/p_t) = θ⁻¹·Σ aᵢ·θ^bᵢ.
_SUBLIMATION_A = (-0.212144006e2, 0.273203819e2, -0.610598130e1)
_SUBLIMATION_B = (0.333333333e-2, 0.120666667e1, 0.170333333e1)

WaterPhase = Literal["liquid", "vapor", "ice"]
InputName = Literal["T", "phi", "T_wb", "T_dp", "omega", "h"]

# ---------------------------------------------------------------------
# Agua: saturación y propiedades (IAPWS)
# ---------------------------------------------------------------------

_LOCAL = threading.local()


def _water() -> CoolProp.AbstractState:
    """AbstractState de agua de este hilo (Streamlit corre cada sesión en un hilo)."""
    state = getattr(_LOCAL, "water", None)
    if state is None:
        state = CoolProp.AbstractState("HEOS", "Water")
        _LOCAL.water = state
    return state


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _bar(p_Pa: float) -> str:
    return f"{p_Pa / 1e5:.4g} bar"


def _pct(fraction: float) -> str:
    return f"{100 * fraction:.4g} %"


def over_ice(T_K: float) -> bool:
    """Si a esa temperatura la saturación es sobre hielo (debajo del punto triple)."""
    return T_K < T_TRIPLE_K


def saturation_pressure(T_K: float) -> float:
    """Presión de saturación del vapor de agua p_vs(T), Pa (vademecum §14.2).

    Sobre el agua líquida desde el punto triple (IAPWS-95, Wagner y Pruß, 2002,
    con CoolProp: la tabla A-4 de Cengel); por debajo, sobre el hielo con la
    ecuación de sublimación de IAPWS R14-08(2011) (la tabla A-8).
    """
    if T_K >= T_TRIPLE_K:
        water = _water()
        water.update(CoolProp.QT_INPUTS, 0.0, T_K)
        return float(water.p())
    theta = T_K / T_TRIPLE_K
    exponent = sum(a * theta**b for a, b in zip(_SUBLIMATION_A, _SUBLIMATION_B, strict=True))
    return P_TRIPLE_PA * math.exp(exponent / theta)


def saturation_temperature(p_v_Pa: float) -> float:
    """Temperatura a la que el vapor a ``p_v_Pa`` satura (sobre agua o hielo), K."""
    if p_v_Pa <= 0.0:
        raise ValueError("La presión del vapor tiene que ser positiva.")
    if p_v_Pa >= P_TRIPLE_PA:
        water = _water()
        water.update(CoolProp.PQ_INPUTS, p_v_Pa, 0.0)
        return float(water.T())
    # La ecuación de sublimación vale desde 50 K; más abajo la presión es despreciable.
    return float(brentq(lambda T: saturation_pressure(T) - p_v_Pa, 30.0, T_TRIPLE_K, xtol=1e-12))


def water_enthalpy(T_K: float, phase: WaterPhase = "liquid") -> float:
    """Entalpía del agua que entra o sale de un proceso, J/kg (h = 0: líquido en el punto triple).

    Líquida o vapor saturado: tablas (IAPWS-95, CoolProp), como Cengel y el
    vademecum §14.12 (h_w = h_f(T_w) o h_g(T_w)). Hielo: −333,4 + 2,1·t kJ/kg
    (ASHRAE HoF 2017, cap. 1).
    """
    if phase == "ice":
        return _H_ICE_0 + _C_ICE * (T_K - T_REF_K)
    if T_K < T_TRIPLE_K:
        raise ValueError(
            f"A {_degC(T_K)} el agua no está líquida: está por debajo del punto triple (0,01 °C)."
        )
    water = _water()
    water.update(CoolProp.QT_INPUTS, 0.0 if phase == "liquid" else 1.0, T_K)
    return float(water.hmass())


def water_entropy(T_K: float, phase: WaterPhase = "liquid") -> float:
    """Entropía del agua líquida o del vapor saturado (tablas), J/(kg·K)."""
    if phase == "ice" or T_K < T_TRIPLE_K:
        raise ValueError("La entropía del agua se evalúa como líquido o vapor, desde 0,01 °C.")
    water = _water()
    water.update(CoolProp.QT_INPUTS, 0.0 if phase == "liquid" else 1.0, T_K)
    return float(water.smass())


def _wet_bulb_water(T_K: float) -> float:
    """h del agua del bulbo húmedo: líquida, o hielo debajo del punto triple."""
    return water_enthalpy(T_K, "ice" if over_ice(T_K) else "liquid")


# ---------------------------------------------------------------------
# Relaciones básicas (vademecum §14.2 a §14.5)
# ---------------------------------------------------------------------


def pressure_from_altitude(z_m: float) -> float:
    """Presión atmosférica normal a la altura ``z_m`` (ASHRAE HoF 2017, cap. 1, ec. 3).

    p = 101,325·(1 − 2,25577×10⁻⁵·Z)^5,2559 kPa, con Z en metros: la atmósfera
    estándar en la tropósfera.
    """
    return P_SEA_LEVEL_PA * (1.0 - 2.25577e-5 * z_m) ** 5.2559


def altitude_from_pressure(p_Pa: float) -> float:
    """Altura de la atmósfera estándar donde la presión es ``p_Pa`` (inversa de la anterior)."""
    return (1.0 - (p_Pa / P_SEA_LEVEL_PA) ** (1.0 / 5.2559)) / 2.25577e-5


def humidity_ratio(p_v_Pa: float, p_Pa: float) -> float:
    """ω = 0,622·p_v/(p − p_v) (vademecum §14.2)."""
    return EPSILON * p_v_Pa / (p_Pa - p_v_Pa)


def vapor_pressure(omega: float, p_Pa: float) -> float:
    """p_v = ω·p/(0,622 + ω) (vademecum §14.2)."""
    return omega * p_Pa / (EPSILON + omega)


def saturation_humidity_ratio(T_K: float, p_Pa: float) -> float | None:
    """ω_s(T, p) (vademecum §14.2); ``None`` si a esa T el agua hierve a la presión p."""
    p_vs = saturation_pressure(T_K)
    if p_vs >= p_Pa:
        return None
    return humidity_ratio(p_vs, p_Pa)


def moist_enthalpy(T_K: float, omega: float) -> float:
    """h = c_p,a·t + ω·(r₀ + c_p,v·t), J/kg de aire seco, con t = T − 273,15 K (§14.5)."""
    t = T_K - T_REF_K
    return C_P_AIR * t + omega * (H_VAPOR_0 + C_P_VAPOR * t)


def dry_bulb_from_h_omega(h: float, omega: float) -> float:
    """T de bulbo seco con h y ω: t = (h − ω·r₀)/(c_p,a + ω·c_p,v) (despeje de §14.5)."""
    return T_REF_K + (h - omega * H_VAPOR_0) / (C_P_AIR + omega * C_P_VAPOR)


def dew_point(omega: float, p_Pa: float) -> float | None:
    """Temperatura de punto de rocío (de escarcha bajo 0 °C); ``None`` para el aire seco."""
    if omega <= _OMEGA_DRY:
        return None
    return saturation_temperature(vapor_pressure(omega, p_Pa))


def omega_from_wet_bulb(T_K: float, T_wb_K: float, p_Pa: float) -> float:
    """ω del aire con su T de bulbo seco y de bulbo húmedo (saturación adiabática, §14.12.5).

    Del balance h₁ + (ω_s* − ω₁)·h_w* = h* (Cengel ec. 14-14):
    ω₁ = [c_p,a·(t* − t₁) + ω_s*·(r₀ + c_p,v·t* − h_w*)] / (r₀ + c_p,v·t₁ − h_w*),
    con * en T_bh y h_w* la del agua (hielo debajo del punto triple).
    """
    omega_s = saturation_humidity_ratio(T_wb_K, p_Pa)
    if omega_s is None:
        raise ValueError(
            f"Con T_bh = {_degC(T_wb_K)} el agua hierve a {_bar(p_Pa)}: el bulbo húmedo "
            "no puede estar tan caliente."
        )
    t, t_wb = T_K - T_REF_K, T_wb_K - T_REF_K
    h_w = _wet_bulb_water(T_wb_K)
    num = C_P_AIR * (t_wb - t) + omega_s * (H_VAPOR_0 + C_P_VAPOR * t_wb - h_w)
    return num / (H_VAPOR_0 + C_P_VAPOR * t - h_w)


def _dry_bulb_from_wet_bulb(T_wb_K: float, omega: float, p_Pa: float) -> float:
    """T de bulbo seco con T_bh y ω (despeje de :func:`omega_from_wet_bulb`)."""
    omega_s = saturation_humidity_ratio(T_wb_K, p_Pa)
    if omega_s is None:
        raise ValueError(
            f"Con T_bh = {_degC(T_wb_K)} el agua hierve a {_bar(p_Pa)}: el bulbo húmedo "
            "no puede estar tan caliente."
        )
    t_wb = T_wb_K - T_REF_K
    h_w = _wet_bulb_water(T_wb_K)
    num = C_P_AIR * t_wb + omega_s * (H_VAPOR_0 + C_P_VAPOR * t_wb - h_w)
    num -= omega * (H_VAPOR_0 - h_w)
    return T_REF_K + num / (C_P_AIR + omega * C_P_VAPOR)


def _upper_saturation_T(p_Pa: float) -> float:
    """Temperatura máxima con ω_s finita: la de ebullición a la presión total, con margen."""
    return min(T_MAX_K, saturation_temperature(p_Pa) - 1e-6)


def wet_bulb_temperature(T_K: float, omega: float, p_Pa: float) -> float:
    """Temperatura de bulbo húmedo termodinámica = de saturación adiabática (§14.12.5), K.

    Ecuación implícita h₁ + [ω_s(T_bh) − ω₁]·h_w(T_bh) = h(T_bh, ω_s(T_bh)),
    con el agua líquida (o hielo, debajo del punto triple: bulbo de hielo, como
    ASHRAE). Se resuelve con ``brentq`` entre el punto de rocío y la T de bulbo
    seco. Con aire a presión atmosférica coincide con la del psicrómetro (Lewis).
    """
    h1 = moist_enthalpy(T_K, omega)

    def residual(T_wb: float) -> float:
        omega_s = saturation_humidity_ratio(T_wb, p_Pa)
        assert omega_s is not None
        return h1 + (omega_s - omega) * _wet_bulb_water(T_wb) - moist_enthalpy(T_wb, omega_s)

    hi = min(T_K, _upper_saturation_T(p_Pa))
    T_dp = dew_point(omega, p_Pa)
    lo = T_dp if T_dp is not None else max(120.0, T_K - 150.0)
    if hi - lo < 1e-9:
        return hi
    f_hi = residual(hi)
    if f_hi >= 0.0:
        return hi
    return float(brentq(residual, lo, hi, xtol=1e-10))


# ---------------------------------------------------------------------
# Estado muerto (ambiente) y exergía del agua
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DeadState:
    """Ambiente para la exergía (vademecum §14.11): T₀, φ₀ y p₀."""

    T0_K: float
    phi0: float
    p0_Pa: float = P_SEA_LEVEL_PA

    def __post_init__(self) -> None:
        if not (T_MIN_K <= self.T0_K <= 333.15):
            raise ValueError("La temperatura del ambiente tiene que estar entre −50 y 60 °C.")
        if not (0.0 < self.phi0 <= 1.0):
            raise ValueError(
                "La humedad relativa del ambiente tiene que estar entre 0 y 100 % (sin 0: con "
                "un ambiente totalmente seco la exergía química del vapor sería infinita)."
            )
        if not (P_MIN_PA <= self.p0_Pa <= P_MAX_PA):
            raise ValueError("La presión del ambiente está fuera de rango (0,4 a 10 bar).")
        if self.phi0 * saturation_pressure(self.T0_K) >= self.p0_Pa:
            raise ValueError("Con esos datos el vapor del ambiente superaría la presión total.")

    @property
    def p_v0_Pa(self) -> float:
        """Presión parcial del vapor en el ambiente."""
        return self.phi0 * saturation_pressure(self.T0_K)

    @property
    def omega0(self) -> float:
        """Humedad absoluta del ambiente."""
        return humidity_ratio(self.p_v0_Pa, self.p0_Pa)

    @property
    def h_vapor0(self) -> float:
        """h del vapor en el ambiente (modelo: r₀ + c_p,v·t₀)."""
        return H_VAPOR_0 + C_P_VAPOR * (self.T0_K - T_REF_K)

    @property
    def s_vapor0(self) -> float:
        """s del vapor a T₀ y a su presión parcial en el ambiente (§14.10)."""
        return (
            S_VAPOR_0
            + C_P_VAPOR * math.log(self.T0_K / T_REF_K)
            - R_VAPOR * math.log(self.p_v0_Pa / P_REF_VAPOR_PA)
        )


def water_exergy(T_K: float, phase: WaterPhase, dead: DeadState) -> float:
    """Exergía de flujo del agua (líquida o vapor saturado) a T, J/kg (Wepfer et al., 1979).

    Contra el vapor del ambiente (a T₀ y su presión parcial p_v0):
    ψ_w = (h_w − h_g0) − T₀·(s_w − s_g0) + R_v·T₀·ln(1/φ₀), con h_g0 y s_g0 las
    del vapor saturado a T₀ (las del modelo) y h_w, s_w de tablas. El último
    término es la exergía química: el agua líquida en un ambiente no saturado
    puede evaporarse. Wepfer, Gaggioli y Obert, «Proper evaluation of available
    energy for HVAC», ASHRAE Transactions 85(1), 1979.
    """
    h_w = water_enthalpy(T_K, phase)
    s_w = water_entropy(T_K, phase)
    return (h_w - dead.h_vapor0) - dead.T0_K * (s_w - dead.s_vapor0)


# ---------------------------------------------------------------------
# Estado del aire húmedo
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MoistAirState:
    """Estado del aire húmedo por kg de aire seco (vademecum §14).

    Lo fijan p, T y ω; el bulbo húmedo y el rocío se calculan al armarlo
    (:func:`state_from_T_omega`).
    """

    p_Pa: float
    T_K: float
    omega: float
    p_vs_Pa: float
    T_wb_K: float
    T_dp_K: float | None

    @property
    def p_v_Pa(self) -> float:
        """Presión parcial del vapor (§14.2)."""
        return vapor_pressure(self.omega, self.p_Pa)

    @property
    def p_a_Pa(self) -> float:
        """Presión parcial del aire seco: p_a = p − p_v (§14.1)."""
        return self.p_Pa - self.p_v_Pa

    @property
    def phi(self) -> float:
        """Humedad relativa φ = p_v/p_vs(T) (§14.3), como fracción."""
        return self.p_v_Pa / self.p_vs_Pa

    @property
    def omega_s(self) -> float | None:
        """Humedad absoluta de saturación a T y p (``None`` si a esa T el agua hierve)."""
        if self.p_vs_Pa >= self.p_Pa:
            return None
        return humidity_ratio(self.p_vs_Pa, self.p_Pa)

    @property
    def mu(self) -> float | None:
        """Grado de saturación μ = ω/ω_s (§14.4)."""
        omega_s = self.omega_s
        return None if omega_s is None else self.omega / omega_s

    @property
    def t_C(self) -> float:
        """t = T − 273,15 K: la temperatura medida desde la referencia de h."""
        return self.T_K - T_REF_K

    @property
    def h_air(self) -> float:
        """h_a = c_p,a·t (§14.5)."""
        return C_P_AIR * self.t_C

    @property
    def h_vapor(self) -> float:
        """h_v = r₀ + c_p,v·t (§14.5)."""
        return H_VAPOR_0 + C_P_VAPOR * self.t_C

    @property
    def h_J_per_kg(self) -> float:
        """Entalpía por kg de aire seco: h = h_a + ω·h_v (§14.5)."""
        return self.h_air + self.omega * self.h_vapor

    @property
    def cp_J_per_kg_K(self) -> float:
        """c_p,ah = c_p,a + ω·c_p,v, por kg de aire seco (§14.6)."""
        return C_P_AIR + self.omega * C_P_VAPOR

    @property
    def v_m3_per_kg(self) -> float:
        """Volumen por kg de aire seco: v = R_a·T/p_a (§14.7)."""
        return R_AIR * self.T_K / self.p_a_Pa

    @property
    def v_moist_m3_per_kg(self) -> float:
        """Volumen por kg de aire húmedo: v_ah = v/(1 + ω) (§14.9)."""
        return self.v_m3_per_kg / (1.0 + self.omega)

    @property
    def rho_kg_per_m3(self) -> float:
        """Densidad del aire húmedo: ρ = (1 + ω)/v (§14.8)."""
        return (1.0 + self.omega) / self.v_m3_per_kg

    @property
    def R_J_per_kg_K(self) -> float:
        """Constante del aire húmedo: R_ah = R_a·(1 + 1,608·ω)/(1 + ω) (§14.9)."""
        return R_AIR * (1.0 + MOLAR_RATIO * self.omega) / (1.0 + self.omega)

    @property
    def s_air(self) -> float:
        """s_a = c_p,a·ln(T/T_ref) − R_a·ln(p_a/p_ref,a) (§14.10)."""
        return C_P_AIR * math.log(self.T_K / T_REF_K) - R_AIR * math.log(self.p_a_Pa / P_REF_AIR_PA)

    @property
    def s_vapor(self) -> float | None:
        """s_v = s₀ + c_p,v·ln(T/T_ref) − R_v·ln(p_v/p_ref,v) (§14.10); ``None`` sin vapor."""
        if self.omega <= 0.0:
            return None
        return (
            S_VAPOR_0
            + C_P_VAPOR * math.log(self.T_K / T_REF_K)
            - R_VAPOR * math.log(self.p_v_Pa / P_REF_VAPOR_PA)
        )

    @property
    def s_J_per_kg_K(self) -> float:
        """Entropía por kg de aire seco: s = s_a + ω·s_v (§14.10)."""
        s_v = self.s_vapor
        return self.s_air + (self.omega * s_v if s_v is not None else 0.0)

    @property
    def dew_over_ice(self) -> bool:
        """Si el punto de rocío es de escarcha (saturación sobre hielo)."""
        return self.T_dp_K is not None and over_ice(self.T_dp_K)

    @property
    def wet_bulb_over_ice(self) -> bool:
        """Si el bulbo húmedo es de hielo (debajo del punto triple)."""
        return over_ice(self.T_wb_K)

    @property
    def saturated(self) -> bool:
        """Si está saturado (φ = 100 %)."""
        return self.phi >= 1.0 - 1e-9

    def psi_parts(self, dead: DeadState) -> tuple[float, float]:
        """(ψ_tm, ψ_qu): exergía termomecánica y química por kg de aire seco (§14.11)."""
        T, T0, w, w0 = self.T_K, dead.T0_K, self.omega, dead.omega0
        thermo = (C_P_AIR + w * C_P_VAPOR) * ((T - T0) - T0 * math.log(T / T0))
        thermo += (1.0 + MOLAR_RATIO * w) * R_AIR * T0 * math.log(self.p_Pa / dead.p0_Pa)
        mixing = (1.0 + MOLAR_RATIO * w) * math.log(
            (1.0 + MOLAR_RATIO * w0) / (1.0 + MOLAR_RATIO * w)
        )
        if w > 0.0:
            mixing += MOLAR_RATIO * w * math.log(w / w0)
        return thermo, R_AIR * T0 * mixing

    def psi(self, dead: DeadState) -> float:
        """Exergía de flujo por kg de aire seco: ψ = ψ_tm + ψ_qu (§14.11), J/kg."""
        thermo, mixing = self.psi_parts(dead)
        return thermo + mixing


def _check_pressure(p_Pa: float) -> None:
    if not (P_MIN_PA <= p_Pa <= P_MAX_PA):
        raise ValueError(
            f"La presión total tiene que estar entre 0,4 y 10 bar (pusiste {_bar(p_Pa)}). "
            "A nivel del mar es 1,01325 bar; a 7000 m de altura, unos 0,41 bar."
        )


def _check_T(T_K: float, what: str = "La temperatura de bulbo seco") -> None:
    if not (T_MIN_K - 1e-9 <= T_K <= T_MAX_K + 1e-9):
        raise ValueError(f"{what} tiene que estar entre −50 y 200 °C (pusiste {_degC(T_K)}).")


#: El rocío y el bulbo húmedo del aire muy seco pueden quedar bajo −50 °C.
_T_MIN_MOISTURE_K = 173.15


def _check_T_moisture(T_K: float, what: str) -> None:
    if not (_T_MIN_MOISTURE_K <= T_K <= T_MAX_K + 1e-9):
        raise ValueError(f"{what} tiene que estar entre −100 y 200 °C (pusiste {_degC(T_K)}).")


def state_from_T_omega(p_Pa: float, T_K: float, omega: float) -> MoistAirState:
    """Estado con p, T y ω (las variables con que se arman todos los demás)."""
    _check_pressure(p_Pa)
    _check_T(T_K)
    if omega < 0.0:
        raise ValueError(f"La humedad absoluta no puede ser negativa (ω = {omega:.4g}).")
    if omega <= _OMEGA_DRY:
        omega = 0.0
    p_v = vapor_pressure(omega, p_Pa)
    p_vs = saturation_pressure(T_K)
    if p_v > p_vs * (1.0 + 1e-9):
        raise ValueError(
            f"Con ω = {omega:.4g} kg/kg a {_degC(T_K)} el aire estaría sobresaturado "
            f"(φ = {_pct(p_v / p_vs)}): el vapor de más condensa como niebla. Como mucho, "
            f"ω_s = {_omega_s_text(T_K, p_Pa)}."
        )
    T_wb = wet_bulb_temperature(T_K, omega, p_Pa)
    T_dp = dew_point(omega, p_Pa)
    if T_dp is not None:
        T_dp = min(T_dp, T_K)
    return MoistAirState(p_Pa, T_K, omega, p_vs, min(T_wb, T_K), T_dp)


def _omega_s_text(T_K: float, p_Pa: float) -> str:
    omega_s = saturation_humidity_ratio(T_K, p_Pa)
    return "∞ (el agua hierve)" if omega_s is None else f"{omega_s:.4g} kg/kg"


def state_from_T_phi(p_Pa: float, T_K: float, phi: float) -> MoistAirState:
    """Estado con p, T y φ (el dato más común)."""
    return moist_air_state(p_Pa, ("T", "phi"), T_K, phi)


#: Pares de datos que acepta :func:`moist_air_state` (además de la presión).
PAIRS: tuple[tuple[InputName, InputName], ...] = (
    ("T", "phi"),
    ("T", "T_wb"),
    ("T", "T_dp"),
    ("T", "omega"),
    ("T", "h"),
    ("h", "phi"),
    ("T_wb", "phi"),
    ("omega", "phi"),
    ("T_dp", "phi"),
    ("h", "omega"),
    ("T_wb", "omega"),
    ("h", "T_dp"),
    ("T_wb", "T_dp"),
)

#: Nombre de cada dato (para mensajes y rótulos).
VARIABLE_LABELS: dict[InputName, str] = {
    "T": "temperatura de bulbo seco T",
    "phi": "humedad relativa φ",
    "T_wb": "temperatura de bulbo húmedo T_bh",
    "T_dp": "temperatura de punto de rocío T_pr",
    "omega": "humedad absoluta ω",
    "h": "entalpía h",
}

#: Rótulo de cada par para el selector de la página.
PAIR_LABELS: dict[tuple[InputName, InputName], str] = {
    ("T", "phi"): "T y φ (bulbo seco y humedad relativa)",
    ("T", "T_wb"): "T y T_bh (psicrómetro: bulbo seco y húmedo)",
    ("T", "T_dp"): "T y T_pr (bulbo seco y punto de rocío)",
    ("T", "omega"): "T y ω (bulbo seco y humedad absoluta)",
    ("T", "h"): "T y h (bulbo seco y entalpía)",
    ("h", "phi"): "h y φ",
    ("T_wb", "phi"): "T_bh y φ",
    ("omega", "phi"): "ω y φ",
    ("T_dp", "phi"): "T_pr y φ",
    ("h", "omega"): "h y ω",
    ("T_wb", "omega"): "T_bh y ω",
    ("h", "T_dp"): "h y T_pr",
    ("T_wb", "T_dp"): "T_bh y T_pr",
}


def _check_phi(phi: float) -> None:
    if not (0.0 <= phi <= 1.0 + 1e-12):
        raise ValueError(
            f"La humedad relativa tiene que estar entre 0 y 100 % (pusiste {_pct(phi)}): con más "
            "de 100 % el aire tendría más vapor del que admite a esa temperatura (niebla)."
        )


def _omega_from_T_and(p_Pa: float, T_K: float, name: InputName, value: float) -> float:
    """ω con la T de bulbo seco y el otro dato (fórmulas cerradas)."""
    if name == "phi":
        _check_phi(value)
        p_v = min(value, 1.0) * saturation_pressure(T_K)
        if p_v >= p_Pa:
            raise ValueError(
                f"A {_bar(p_Pa)} el agua hierve a {_degC(saturation_temperature(p_Pa))}: con "
                f"T = {_degC(T_K)} y φ = {_pct(value)} el vapor superaría la presión total. "
                "Bajá φ o la temperatura."
            )
        return humidity_ratio(p_v, p_Pa)
    if name == "T_wb":
        _check_T_moisture(value, "La temperatura de bulbo húmedo")
        if value > T_K + 1e-9:
            raise ValueError(
                f"La temperatura de bulbo húmedo ({_degC(value)}) no puede superar a la de bulbo "
                f"seco ({_degC(T_K)}): el agua que se evapora del bulbo lo enfría."
            )
        omega = omega_from_wet_bulb(T_K, value, p_Pa)
        if omega < -1e-12:
            dry = wet_bulb_temperature(T_K, 0.0, p_Pa)
            raise ValueError(
                f"Con T = {_degC(T_K)}, el aire más seco posible (ω = 0) tiene "
                f"T_bh = {_degC(dry)}: un bulbo húmedo de {_degC(value)} pediría humedad "
                "negativa."
            )
        return max(omega, 0.0)
    if name == "T_dp":
        _check_T_moisture(value, "La temperatura de punto de rocío")
        if value > T_K + 1e-9:
            raise ValueError(
                f"El punto de rocío ({_degC(value)}) no puede superar a la temperatura de bulbo "
                f"seco ({_degC(T_K)}): el aire estaría sobresaturado."
            )
        p_v = saturation_pressure(value)
        if p_v >= p_Pa:
            raise ValueError(f"Con un rocío de {_degC(value)} el vapor superaría {_bar(p_Pa)}.")
        return humidity_ratio(p_v, p_Pa)
    if name == "omega":
        return value
    if name == "h":
        t = T_K - T_REF_K
        omega = (value - C_P_AIR * t) / (H_VAPOR_0 + C_P_VAPOR * t)
        if omega < -1e-12:
            raise ValueError(
                f"A {_degC(T_K)} el aire seco ya tiene h = {C_P_AIR * t / 1e3:.4g} kJ/kg: con "
                f"h = {value / 1e3:.4g} kJ/kg la humedad daría negativa."
            )
        return max(omega, 0.0)
    raise ValueError(f"Dato desconocido: {name!r}.")


def _omega_from_moisture(p_Pa: float, name: InputName, value: float) -> float:
    """ω con el dato de humedad que no depende de T (ω o T_pr)."""
    if name == "omega":
        if value < 0.0:
            raise ValueError(f"La humedad absoluta no puede ser negativa (ω = {value:.4g}).")
        return value
    _check_T_moisture(value, "La temperatura de punto de rocío")
    p_v = saturation_pressure(value)
    if p_v >= p_Pa:
        raise ValueError(f"Con un rocío de {_degC(value)} el vapor superaría {_bar(p_Pa)}.")
    return humidity_ratio(p_v, p_Pa)


def _T_from_h_phi(p_Pa: float, h: float, phi: float) -> float:
    """T con h y φ: h(T, ω(T, φ)) crece con T; se busca con ``brentq``."""
    _check_phi(phi)
    if phi <= 0.0:
        return dry_bulb_from_h_omega(h, 0.0)
    hi = T_MAX_K
    if phi * saturation_pressure(hi) >= p_Pa:
        hi = brentq(lambda T: phi * saturation_pressure(T) - 0.999 * p_Pa, T_MIN_K, T_MAX_K)

    def residual(T: float) -> float:
        p_v = phi * saturation_pressure(T)
        return moist_enthalpy(T, humidity_ratio(p_v, p_Pa)) - h

    if residual(T_MIN_K) > 0.0 or residual(hi) < 0.0:
        raise ValueError(
            f"No hay aire húmedo entre −50 y 200 °C con h = {h / 1e3:.4g} kJ/kg y φ = "
            f"{_pct(phi)} a {_bar(p_Pa)}."
        )
    return float(brentq(residual, T_MIN_K, hi, xtol=1e-10))


def _T_from_wb_phi(p_Pa: float, T_wb: float, phi: float) -> float:
    """T con T_bh y φ: φ baja de 100 % (T = T_bh) a 0 (aire seco); se busca con ``brentq``."""
    _check_phi(phi)
    _check_T_moisture(T_wb, "La temperatura de bulbo húmedo")
    if phi >= 1.0:
        return T_wb
    T_dry = _dry_bulb_from_wet_bulb(T_wb, 0.0, p_Pa)

    def residual(T: float) -> float:
        omega = max(omega_from_wet_bulb(T, T_wb, p_Pa), 0.0)
        return vapor_pressure(omega, p_Pa) / saturation_pressure(T) - phi

    hi = min(T_dry, T_MAX_K)
    if residual(hi) > 0.0:
        raise ValueError(
            f"Con T_bh = {_degC(T_wb)} y φ = {_pct(phi)} la temperatura de bulbo seco pasaría "
            "de 200 °C."
        )
    return float(brentq(residual, T_wb, hi, xtol=1e-10))


def _canonical(pair: tuple[str, str]) -> tuple[InputName, InputName]:
    for known in PAIRS:
        if tuple(pair) == known:
            return known
        if tuple(reversed(pair)) == known:
            return known
    raise ValueError(
        f"El par {pair!r} no sirve para fijar el estado. ω y T_pr son la misma información "
        "(con la presión), y h con T_bh son casi paralelas en la carta."
    )


def moist_air_state(
    p_Pa: float, pair: tuple[str, str], first: float, second: float
) -> MoistAirState:
    """Estado del aire húmedo con la presión total y dos datos (vademecum §14).

    ``pair`` es uno de :data:`PAIRS` (en cualquier orden, con ``first`` y
    ``second`` en ese mismo orden). Con T entre los datos todo sale con
    fórmulas cerradas; ω y T_pr fijan p_v; con h y φ, o T_bh y φ, la T de bulbo
    seco se busca con ``brentq``. Los mensajes de error explican qué dato no
    tiene sentido físico.
    """
    _check_pressure(p_Pa)
    canonical = _canonical(pair)
    values = dict(zip(pair, (first, second), strict=True))
    a, b = canonical
    va, vb = values[a], values[b]
    if a == "T":
        _check_T(va)
        omega = _omega_from_T_and(p_Pa, va, b, vb)
        return state_from_T_omega(p_Pa, va, omega)
    if b == "phi":
        if a == "h":
            T = _T_from_h_phi(p_Pa, va, vb)
            omega = humidity_ratio(min(vb, 1.0) * saturation_pressure(T), p_Pa)
        elif a == "T_wb":
            T = _T_from_wb_phi(p_Pa, va, vb)
            omega = max(omega_from_wet_bulb(T, va, p_Pa), 0.0)
        else:
            _check_phi(vb)
            if vb <= 0.0:
                raise ValueError(
                    "Con φ = 0 el aire es seco: no hay vapor que fije la temperatura. Usá T y φ."
                )
            omega = _omega_from_moisture(p_Pa, a, va)
            p_vs = vapor_pressure(omega, p_Pa) / min(vb, 1.0)
            if p_vs <= 0.0:
                raise ValueError("Con ω = 0 (aire seco) φ vale 0 a cualquier temperatura.")
            T = saturation_temperature(p_vs)
            if T > T_MAX_K:
                raise ValueError(
                    f"Con esa humedad y φ = {_pct(vb)} la temperatura pasaría de 200 °C."
                )
            if T < T_MIN_K:
                raise ValueError(
                    f"Con esa humedad y φ = {_pct(vb)} la temperatura quedaría bajo −50 °C."
                )
        _check_T(T)
        return state_from_T_omega(p_Pa, T, omega)
    # Los demás pares fijan ω (con ω o T_pr) y una variable que da T en forma cerrada.
    omega = _omega_from_moisture(p_Pa, b, vb)
    if a == "h":
        T = dry_bulb_from_h_omega(va, omega)
    else:  # a == "T_wb"
        _check_T_moisture(va, "La temperatura de bulbo húmedo")
        T = _dry_bulb_from_wet_bulb(va, omega, p_Pa)
        T_dp = dew_point(omega, p_Pa)
        if T_dp is not None and va < T_dp - 1e-9:
            raise ValueError(
                f"El bulbo húmedo ({_degC(va)}) no puede estar más frío que el punto de rocío "
                f"({_degC(T_dp)}): T_pr ≤ T_bh ≤ T."
            )
    _check_T(T)
    return state_from_T_omega(p_Pa, T, omega)


# ---------------------------------------------------------------------
# Comparación con el modelo de gas real de CoolProp (ASHRAE RP-1485)
# ---------------------------------------------------------------------

_HA_KEYS: dict[InputName, str] = {
    "T": "T",
    "phi": "R",
    "T_wb": "B",
    "T_dp": "D",
    "omega": "W",
    "h": "H",
}


@dataclass(frozen=True)
class ComparisonRow:
    """Una propiedad con el modelo ideal y con CoolProp."""

    key: str
    label: str
    model: float | None
    coolprop: float | None
    is_temperature: bool = False

    @property
    def difference(self) -> float | None:
        """Diferencia: en K para temperaturas; relativa (fracción) para el resto."""
        if self.model is None or self.coolprop is None:
            return None
        if self.is_temperature:
            return self.coolprop - self.model
        if self.model == 0.0:
            return None
        return self.coolprop / self.model - 1.0


@dataclass(frozen=True)
class CoolPropComparison:
    """El mismo estado con el modelo del vademecum y con ``HAPropsSI`` (RP-1485)."""

    rows: tuple[ComparisonRow, ...]
    error: str | None = None


def coolprop_comparison(
    state: MoistAirState, pair: tuple[str, str], first: float, second: float
) -> CoolPropComparison:
    """Mismos datos al modelo de gas real de CoolProp (Herrmann et al., 2009; RP-1485).

    Ese modelo tiene en cuenta el factor de mejora (el aire «ayuda» a que entre
    un poco más de vapor: ω_s sube ~0,4 % a 1 atm) y que la mezcla no es un gas
    ideal. Se le pasan los mismos datos que al modelo ideal y se comparan las
    demás propiedades. La h de CoolProp usa la misma referencia de ASHRAE.
    """
    p = state.p_Pa
    a, b = pair
    try:
        inputs = (_HA_KEYS[a], first, _HA_KEYS[b], second, "P", p)  # type: ignore[index]

        def ha(key: str) -> float | None:
            if key == "D" and state.omega <= 0.0:
                return None
            try:
                return float(HAPropsSI(key, *inputs))
            except ValueError:
                return None

        W = ha("W")
        V = ha("V")
        rho = (1.0 + W) / V if W is not None and V is not None else None
        rows = (
            ComparisonRow("T", "T", state.T_K, ha("T"), is_temperature=True),
            ComparisonRow("omega", "ω", state.omega, W),
            ComparisonRow("phi", "φ", state.phi, ha("R")),
            ComparisonRow("h", "h", state.h_J_per_kg, ha("H")),
            ComparisonRow("T_wb", "T_bh", state.T_wb_K, ha("B"), is_temperature=True),
            ComparisonRow("T_dp", "T_pr", state.T_dp_K, ha("D"), is_temperature=True),
            ComparisonRow("v", "v", state.v_m3_per_kg, V),
            ComparisonRow("rho", "ρ", state.rho_kg_per_m3, rho),
        )
    except (ValueError, KeyError) as exc:  # pragma: no cover - HAPropsSI fuera de rango
        return CoolPropComparison((), f"CoolProp no pudo evaluar el estado: {exc}")
    return CoolPropComparison(tuple(row for row in rows if row.key not in pair))


# ---------------------------------------------------------------------
# Carta psicrométrica (líneas y grilla; el dibujo vive en ui/)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ChartWindow:
    """Rango de la carta: T de bulbo seco (K) y ω (kg/kg)."""

    T_min_K: float
    T_max_K: float
    omega_max: float


@dataclass(frozen=True)
class ChartLine:
    """Una línea de la carta: familia, valor (SI) y puntos (T en K, ω)."""

    family: Literal["phi", "h", "v", "T_wb"]
    value: float
    T_K: tuple[float, ...]
    omega: tuple[float, ...]


def chart_window(
    p_Pa: float, states: tuple[MoistAirState, ...] = (), *, extra_T_K: tuple[float, ...] = ()
) -> ChartWindow:
    """Rango de la carta: 0–50 °C y ω hasta 0,030 a 1 atm (Cengel y ASHRAE), que se agranda.

    ω_máx escala con 1/p (a menor presión entra más vapor) y crece hasta
    abarcar los estados; las temperaturas, hasta cubrir los estados con un
    margen.
    """
    temps = [s.T_K for s in states] + list(extra_T_K)
    T_lo = min([273.15, *[T - 3.0 for T in temps]])
    T_hi = max([323.15, *[T + 5.0 for T in temps]])
    T_lo = max(T_MIN_K, 5.0 * math.floor(T_lo / 5.0 - 273.15 / 5.0) + 273.15)
    T_hi = min(T_MAX_K, 5.0 * math.ceil(T_hi / 5.0 - 273.15 / 5.0) + 273.15)
    omega_max = 0.030 * P_SEA_LEVEL_PA / p_Pa
    if states:
        omega_max = max(omega_max, 1.12 * max(s.omega for s in states))
    omega_top = saturation_humidity_ratio(min(T_hi, _upper_saturation_T(p_Pa)), p_Pa)
    if omega_top is not None:
        omega_max = min(omega_max, max(1.05 * omega_top, 1e-4))
    return ChartWindow(T_lo, T_hi, omega_max)


def _phi_line(p_Pa: float, phi: float, window: ChartWindow, n: int) -> ChartLine:
    T_top = window.T_max_K
    if phi * saturation_pressure(T_top) >= 0.999 * p_Pa:
        T_top = brentq(lambda T: phi * saturation_pressure(T) - 0.999 * p_Pa, window.T_min_K, T_top)
    temps: list[float] = []
    omegas: list[float] = []
    for T in np.linspace(window.T_min_K, T_top, n):
        omega = humidity_ratio(phi * saturation_pressure(float(T)), p_Pa)
        temps.append(float(T))
        omegas.append(omega)
        if omega > window.omega_max:
            break
    return ChartLine("phi", phi, tuple(temps), tuple(omegas))


def _h_line(p_Pa: float, h: float, window: ChartWindow, n: int) -> ChartLine | None:
    """Recta de h constante: de la saturación hasta ω = 0 (o el borde de la carta)."""

    def on_saturation(T: float) -> float:
        omega_s = saturation_humidity_ratio(T, p_Pa)
        assert omega_s is not None
        return moist_enthalpy(T, omega_s) - h

    T_hi_sat = _upper_saturation_T(p_Pa)
    if on_saturation(T_MIN_K - 30.0) > 0.0 or on_saturation(T_hi_sat) < 0.0:
        return None
    T_sat = brentq(on_saturation, T_MIN_K - 30.0, T_hi_sat)
    T_dry = dry_bulb_from_h_omega(h, 0.0)
    temps = np.linspace(T_sat, T_dry, n)
    omegas = (h - C_P_AIR * (temps - T_REF_K)) / (H_VAPOR_0 + C_P_VAPOR * (temps - T_REF_K))
    keep = (temps >= window.T_min_K - 1e-9) & (temps <= window.T_max_K + 1e-9)
    keep &= omegas <= window.omega_max * 1.0001
    if keep.sum() < 2:
        return None
    return ChartLine("h", h, tuple(map(float, temps[keep])), tuple(map(float, omegas[keep])))


def _v_line(p_Pa: float, v: float, window: ChartWindow, n: int) -> ChartLine | None:
    """Línea de v constante: p_v = p − R_a·T/v, desde la saturación hasta ω = 0."""
    T_dry = v * p_Pa / R_AIR

    def on_saturation(T: float) -> float:
        return (p_Pa - R_AIR * T / v) - saturation_pressure(T)

    lo = max(T_MIN_K - 30.0, 150.0)
    if on_saturation(lo) < 0.0 or on_saturation(T_dry) > 0.0:
        return None
    T_sat = brentq(on_saturation, lo, T_dry)
    temps = np.linspace(T_sat, T_dry, n)
    p_v = np.maximum(p_Pa - R_AIR * temps / v, 0.0)
    omegas = EPSILON * p_v / (p_Pa - p_v)
    keep = (temps >= window.T_min_K - 1e-9) & (temps <= window.T_max_K + 1e-9)
    keep &= omegas <= window.omega_max * 1.0001
    if keep.sum() < 2:
        return None
    return ChartLine("v", v, tuple(map(float, temps[keep])), tuple(map(float, omegas[keep])))


def _wb_line(p_Pa: float, T_wb: float, window: ChartWindow, n: int) -> ChartLine | None:
    """Línea de T_bh constante: de la saturación (T = T_bh) hasta el aire seco."""
    if saturation_humidity_ratio(T_wb, p_Pa) is None:
        return None
    T_dry = _dry_bulb_from_wet_bulb(T_wb, 0.0, p_Pa)
    temps = np.linspace(T_wb, T_dry, n)
    omegas = np.array([max(omega_from_wet_bulb(float(T), T_wb, p_Pa), 0.0) for T in temps])
    keep = (temps >= window.T_min_K - 1e-9) & (temps <= window.T_max_K + 1e-9)
    keep &= omegas <= window.omega_max * 1.0001
    if keep.sum() < 2:
        return None
    return ChartLine("T_wb", T_wb, tuple(map(float, temps[keep])), tuple(map(float, omegas[keep])))


def chart_lines(
    p_Pa: float,
    window: ChartWindow,
    *,
    h_values: tuple[float, ...] = (),
    v_values: tuple[float, ...] = (),
    T_wb_values: tuple[float, ...] = (),
    phi_values: tuple[float, ...] = CHART_PHI_VALUES,
    n: int = 120,
) -> tuple[ChartLine, ...]:
    """Líneas de la carta a la presión ``p_Pa``: φ, h, v y T_bh constantes (SI).

    Los valores de h, v y T_bh los elige la página (redondos en las unidades
    de cada sistema); las líneas que no cruzan la ventana se omiten.
    """
    lines: list[ChartLine] = [_phi_line(p_Pa, phi, window, n) for phi in phi_values]
    for h in h_values:
        line = _h_line(p_Pa, h, window, 24)
        if line is not None:
            lines.append(line)
    for v in v_values:
        line = _v_line(p_Pa, v, window, 24)
        if line is not None:
            lines.append(line)
    for T_wb in T_wb_values:
        line = _wb_line(p_Pa, T_wb, window, 16)
        if line is not None:
            lines.append(line)
    return tuple(lines)


def chart_grid(
    p_Pa: float, window: ChartWindow, *, n_T: int = 61, n_omega: int = 41
) -> tuple[tuple[float, float], ...]:
    """Puntos (T, ω) de una grilla debajo de la saturación: los que se pueden tocar en la carta."""
    points: list[tuple[float, float]] = []
    for T in np.linspace(window.T_min_K, window.T_max_K, n_T):
        omega_s = saturation_humidity_ratio(float(T), p_Pa)
        top = window.omega_max if omega_s is None else min(window.omega_max, omega_s)
        for omega in np.linspace(0.0, window.omega_max, n_omega):
            if omega <= top * (1.0 - 1e-9):
                points.append((float(T), float(omega)))
    return tuple(points)


# ---------------------------------------------------------------------
# Ejemplos, masas en un recinto, barrido de la altura y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StateExample:
    """Datos de un ejemplo del estado (la presión sale de ``p_Pa`` o de la altura)."""

    pair: tuple[InputName, InputName]
    first: float
    second: float
    p_Pa: float = P_SEA_LEVEL_PA
    altitude_m: float | None = None
    room_volume_m3: float | None = None
    T0_K: float = 298.15
    phi0: float = 0.5

    @property
    def pressure_Pa(self) -> float:
        """La presión total del ejemplo (por altura, si la tiene)."""
        if self.altitude_m is not None:
            return pressure_from_altitude(self.altitude_m)
        return self.p_Pa


#: Ejemplos del estado (Cengel, cap. 14, y dos de Argentina).
STATE_EXAMPLES: dict[str, StateExample] = {
    "Cengel 14-1: el vapor de agua de una habitación": StateExample(
        ("T", "phi"), 298.15, 0.75, p_Pa=100.0e3, room_volume_m3=75.0
    ),
    "Cengel 14-2: el empañado de las ventanas": StateExample(("T", "phi"), 293.15, 0.75),
    "Cengel 14-3: el psicrómetro (bulbo seco y húmedo)": StateExample(
        ("T", "T_wb"), 298.15, 288.15
    ),
    "Cengel 14-4: la lectura de la carta": StateExample(("T", "phi"), 308.15, 0.40),
    "Buenos Aires: verano húmedo": StateExample(("T", "phi"), 305.15, 0.70, T0_K=303.15),
    "La Quiaca, a 3440 m de altura": StateExample(
        ("T", "phi"), 293.15, 0.30, altitude_m=3440.0, T0_K=288.15, phi0=0.4
    ),
}

#: Qué muestra cada ejemplo (y lo que da el libro).
STATE_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 14-1: el vapor de agua de una habitación": (
        "Una habitación de 5 m × 5 m × 3 m con aire a 25 °C, 100 kPa y 75 %. Cengel da "
        "p_a = 97,62 kPa, ω = 0,0152, h = 63,8 kJ/kg, m_a = 85,61 kg y m_v = 1,30 kg: usa la "
        "h_g de la tabla (2546,5 kJ/kg) en lugar de r₀ + c_p,v·t y T = 298 K."
    ),
    "Cengel 14-2: el empañado de las ventanas": (
        "Con el aire de la casa a 20 °C y 75 %, la ventana se empaña si su superficie está por "
        "debajo del punto de rocío: 15,4 °C según Cengel."
    ),
    "Cengel 14-3: el psicrómetro (bulbo seco y húmedo)": (
        "Un psicrómetro mide 25 °C de bulbo seco y 15 °C de bulbo húmedo a 1 atm. Cengel da "
        "ω = 0,00653, φ = 33,2 % y h = 41,8 kJ/kg."
    ),
    "Cengel 14-4: la lectura de la carta": (
        "Aire a 1 atm, 35 °C y 40 %. En la carta Cengel lee ω = 0,0142, h = 71,5 kJ/kg, "
        "T_bh = 24 °C, T_pr = 19,4 °C y v = 0,893 m³/kg: compará con las fórmulas."
    ),
    "Buenos Aires: verano húmedo": (
        "Un día de verano a 32 °C y 70 %: el punto de rocío pasa de 25 °C y el bulbo húmedo "
        "está cerca de 27 °C, por eso cuesta tanto refrescarse transpirando."
    ),
    "La Quiaca, a 3440 m de altura": (
        "A 3440 m la presión baja a unos 0,66 bar. Con la misma T y φ que al nivel del mar, el "
        "aire lleva más vapor por kg de aire seco (ω sube), es menos denso y su bulbo húmedo "
        "es más bajo. Probá cambiar la altura."
    ),
}


def room_masses(state: MoistAirState, volume_m3: float) -> tuple[float, float]:
    """Masas de aire seco y de vapor en un recinto: m_a = p_a·V/(R_a·T) y m_v = ω·m_a."""
    if volume_m3 <= 0.0:
        raise ValueError("El volumen del recinto tiene que ser positivo.")
    m_a = state.p_a_Pa * volume_m3 / (R_AIR * state.T_K)
    return m_a, state.omega * m_a


@dataclass(frozen=True)
class AltitudePoint:
    """Un punto del barrido de la altura (mismas T y φ)."""

    altitude_m: float
    state: MoistAirState


def altitude_sweep(T_K: float, phi: float, values: tuple[float, ...]) -> list[AltitudePoint]:
    """El mismo aire (T y φ) a distintas alturas; omite los puntos imposibles."""
    points: list[AltitudePoint] = []
    for z in values:
        try:
            state = state_from_T_phi(pressure_from_altitude(z), T_K, phi)
        except ValueError:
            continue
        points.append(AltitudePoint(float(z), state))
    return points


def default_altitude_values() -> tuple[float, ...]:
    """Alturas del barrido: del nivel del mar a 5000 m."""
    return tuple(float(z) for z in np.linspace(0.0, 5000.0, 26))


def moist_air_to_dict(
    state: MoistAirState,
    system: str,
    *,
    pair: tuple[str, str] | None = None,
    given: tuple[float, float] | None = None,
    dead: DeadState | None = None,
    altitude_m: float | None = None,
    room_volume_m3: float | None = None,
) -> dict[str, object]:
    """Estado serializable a JSON (valores en ``system``)."""
    sys_: UnitSystem = system  # type: ignore[assignment]

    def value(v: float | None, kind: QuantityKind) -> dict[str, object] | None:
        if v is None:
            return None
        return {"valor": convert_from_si(v, kind, sys_), "unidad": unit_label(kind, sys_)}

    data: dict[str, object] = {
        "modelo": "gas ideal con c_p constante (vademecum §14)",
        "datos": None if pair is None else [VARIABLE_LABELS[name] for name in pair],  # type: ignore[index]
        "altura_m": altitude_m,
        "p": value(state.p_Pa, "pressure"),
        "T": value(state.T_K, "temperature"),
        "phi": state.phi,
        "omega": state.omega,
        "omega_s": state.omega_s,
        "mu": state.mu,
        "p_v": value(state.p_v_Pa, "pressure"),
        "p_a": value(state.p_a_Pa, "pressure"),
        "p_vs": value(state.p_vs_Pa, "pressure"),
        "h": value(state.h_J_per_kg, "specific_enthalpy"),
        "cp": value(state.cp_J_per_kg_K, "specific_heat"),
        "v": value(state.v_m3_per_kg, "specific_volume"),
        "v_aire_humedo": value(state.v_moist_m3_per_kg, "specific_volume"),
        "rho": value(state.rho_kg_per_m3, "density"),
        "R_aire_humedo": value(state.R_J_per_kg_K, "specific_heat"),
        "s": value(state.s_J_per_kg_K, "specific_entropy"),
        "T_bulbo_humedo": value(state.T_wb_K, "temperature"),
        "T_punto_de_rocio": value(state.T_dp_K, "temperature"),
        "rocio_sobre_hielo": state.dew_over_ice,
    }
    if given is not None and pair is not None:
        data["valores_dados_SI"] = dict(zip(pair, given, strict=True))
    if dead is not None:
        thermo, mixing = state.psi_parts(dead)
        data["ambiente"] = {
            "T0": value(dead.T0_K, "temperature"),
            "phi0": dead.phi0,
            "omega0": dead.omega0,
        }
        data["psi_termomecanica"] = value(thermo, "specific_enthalpy")
        data["psi_quimica"] = value(mixing, "specific_enthalpy")
        data["psi"] = value(thermo + mixing, "specific_enthalpy")
    if room_volume_m3 is not None:
        m_a, m_v = room_masses(state, room_volume_m3)
        data["recinto"] = {"volumen_m3": room_volume_m3, "m_aire_seco_kg": m_a, "m_vapor_kg": m_v}
    return data


def moist_air_notes(state: MoistAirState) -> list[str]:
    """Observaciones sobre el estado (markdown): saturación, escarcha, confort, presión."""
    notes: list[str] = []
    p = state.p_Pa
    if state.saturated:
        notes.append(
            "Aire **saturado**: T = T_bh = T_pr. Si se enfría un poco, el vapor condensa "
            "(niebla o rocío)."
        )
    elif state.phi < 0.2:
        notes.append(
            f"Aire **muy seco** (φ = {_pct(state.phi)}): el agua se evapora rápido; el bulbo "
            f"húmedo queda {state.T_K - state.T_wb_K:.3g} K por debajo de la temperatura del "
            "aire. Sirve para enfriar por evaporación."
        )
    if state.dew_over_ice and state.T_dp_K is not None:
        notes.append(
            f"El punto de rocío está bajo 0 °C ({_degC(state.T_dp_K)}): es de **escarcha**, el "
            "vapor pasa directo a hielo (presión de saturación sobre hielo, tabla A-8)."
        )
    if state.omega_s is None:
        notes.append(
            f"A {_degC(state.T_K)} el agua hierve a la presión total ({_bar(p)}): el aire no se "
            f"puede saturar; como mucho, φ = p/p_vs = {_pct(p / state.p_vs_Pa)}."
        )
    T_lo, T_hi = COMFORT_T_K
    phi_lo, phi_hi = COMFORT_PHI
    if T_lo <= state.T_K <= T_hi and phi_lo <= state.phi <= phi_hi:
        notes.append(
            "Está en la **zona de confort** de Cengel §14-6 (22 a 27 °C y φ de 40 a 60 %)."
        )
    if abs(p - P_SEA_LEVEL_PA) > 0.02 * P_SEA_LEVEL_PA and state.omega_s is not None:
        sea = saturation_humidity_ratio(state.T_K, P_SEA_LEVEL_PA)
        if sea is not None:
            notes.append(
                f"A {_bar(p)} (nivel del mar: {_bar(P_SEA_LEVEL_PA)}), con la misma T y φ el "
                f"aire lleva {state.omega_s / sea:.3g} veces el vapor por kg de aire seco que al "
                "nivel del "
                "mar: ω_s = 0,622·p_vs/(p − p_vs) depende de la presión total."
            )
    if p > 2.0e5:
        notes.append(
            "A presión alta el aire húmedo se aparta del gas ideal (el aire «ayuda» a que entre "
            "más vapor: factor de mejora). Mirá la comparación con CoolProp."
        )
    return notes
