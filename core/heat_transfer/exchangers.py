"""Intercambiadores de calor: ε-NTU, LMTD con F, U global y exergía (Fase 8.2).

Cengel y Ghajar (2015), cap. 11 (§11-2 U y ensuciamiento, §11-4 LMTD con el
factor F, §11-5 ε-NTU); Incropera et al. (2007), §11.2 a §11.4.

Con las capacidades caloríficas C = ṁ·c_p de cada corriente:

- Q̇ = C_c·(T_c,sal − T_c,ent) = C_h·(T_h,ent − T_h,sal) = U·A·F·ΔT_ml;
- Q̇_máx = C_mín·(T_h,ent − T_c,ent), ε = Q̇/Q̇_máx, NTU = U·A/C_mín y
  C_r = C_mín/C_máx; ε(NTU, C_r) depende del tipo (Kays y London, 1984).

El factor F de cualquier arreglo es el cociente de los NTU de un
contracorriente y del arreglo para el mismo ε y C_r: F = NTU_cc/NTU (los
gráficos de Bowman, Mueller y Nagle, 1940, son esa misma cuenta). Una corriente
que condensa o evapora a temperatura constante tiene C → ∞ y C_r = 0: entonces
ε = 1 − e^(−NTU) en cualquier tipo.

La exergía sigue al vademecum: Δs = c·ln(T₂/T₁) (§13.2), Ẋ_dest = T₀·Ṡ_gen y el
rendimiento exergético del intercambiador adiabático (§11.10),
η_ex = ṁ_C·(ψ₄ − ψ₃)/(ṁ_H·(ψ₁ − ψ₂)).

Todo en SI. «hot» es el fluido caliente (H) y «cold» el frío (C).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc

from core.fluids import (
    fluid_limits,
    fluid_state_from_pair,
    fluid_with_article,
    saturation_at_pressure,
    saturation_at_temperature,
)
from core.units_system import UnitSystem, format_quantity

__all__ = [
    "EXCHANGER_KINDS",
    "FOULING_FACTORS",
    "NTU_MAX",
    "RATING_EXAMPLES",
    "SIZING_EXAMPLES",
    "TEST_EXAMPLES",
    "U_EXAMPLES",
    "Arrangement",
    "ExchangerExample",
    "ExchangerKind",
    "ExchangerResult",
    "ExergyBalance",
    "OverallUInputs",
    "OverallUResult",
    "ProfilePoint",
    "RatingInputs",
    "Side",
    "SizingInputs",
    "SizingTarget",
    "Stream",
    "StreamResult",
    "FourTemperatureInputs",
    "TypeComparison",
    "UResistance",
    "WallGeometry",
    "correction_factor",
    "effectiveness",
    "effectiveness_curves",
    "eps_counter",
    "eps_cross_cmax_mixed",
    "eps_cross_cmin_mixed",
    "eps_cross_unmixed",
    "eps_cross_unmixed_approx",
    "eps_parallel",
    "eps_shell",
    "exchanger_notes",
    "exchanger_to_dict",
    "f_curves",
    "lmtd",
    "max_effectiveness",
    "ntu_from_effectiveness",
    "overall_u",
    "overall_u_notes",
    "overall_u_to_dict",
    "solve_rating",
    "solve_sizing",
    "solve_four_temperatures",
    "temperature_profile",
    "type_comparison",
]

ExchangerKind = Literal["parallel", "counter", "shell", "cross_unmixed", "cross_mixed"]
Side = Literal["hot", "cold"]
SizingTarget = Literal["T_hot_out", "T_cold_out", "Q"]
WallGeometry = Literal["tube", "plane"]

EXCHANGER_KINDS: dict[ExchangerKind, str] = {
    "parallel": "Doble tubo, flujo paralelo",
    "counter": "Doble tubo, contracorriente",
    "shell": "Casco y tubos",
    "cross_unmixed": "Flujo cruzado, los dos fluidos sin mezclar",
    "cross_mixed": "Flujo cruzado, un fluido mezclado",
}

#: Factores de ensuciamiento representativos R''_f [m²·K/W] (TEMA, 1978, según
#: Incropera et al., 2007, tabla 11.1; Cengel y Ghajar, tabla 11-2).
FOULING_FACTORS: dict[str, float] = {
    "Limpio (sin ensuciamiento)": 0.0,
    "Agua de mar o de caldera tratada, hasta 50 °C": 0.0001,
    "Agua de mar o de caldera tratada, más de 50 °C": 0.0002,
    "Agua de río, hasta 50 °C (0,0002 a 0,001)": 0.0002,
    "Fuel oil": 0.0009,
    "Refrigerante líquido": 0.0002,
    "Vapor de agua sin aceite": 0.0001,
}

#: Más allá de este NTU el intercambiador no es razonable (y ε ya no cambia).
NTU_MAX = 1000.0
#: F por debajo de este valor: el arreglo desperdicia área (Cengel y Ghajar, §11-4).
F_MIN_RECOMMENDED = 0.75
#: Para que C_r = 1 use las fórmulas del límite.
_CR_ONE = 1e-9

_SIDE_NAMES: dict[Side, str] = {"hot": "caliente", "cold": "frío"}


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


# ---------------------------------------------------------------------
# Relaciones ε-NTU (Incropera, tabla 11.3; Cengel y Ghajar, tabla 11-4)
# ---------------------------------------------------------------------


def eps_parallel(ntu: float, cr: float) -> float:
    """Doble tubo en flujo paralelo: ε = [1 − e^(−NTU(1+C_r))]/(1 + C_r).

    Incropera et al. (2007), tabla 11.3; Cengel y Ghajar (2015), tabla 11-4.
    """
    return -math.expm1(-ntu * (1.0 + cr)) / (1.0 + cr)


def eps_counter(ntu: float, cr: float) -> float:
    """Doble tubo en contracorriente.

    ε = [1 − e^(−NTU(1−C_r))]/[1 − C_r·e^(−NTU(1−C_r))]; con C_r = 1,
    ε = NTU/(1 + NTU) (Incropera et al., 2007, tabla 11.3).
    """
    if abs(1.0 - cr) < _CR_ONE:
        return ntu / (1.0 + ntu)
    e = math.exp(-ntu * (1.0 - cr))
    return (1.0 - e) / (1.0 - cr * e)


def _eps_shell1(ntu: float, cr: float) -> float:
    s = math.sqrt(1.0 + cr * cr)
    e = math.exp(-ntu * s)
    if e == 0.0:
        return 2.0 / (1.0 + cr + s)
    return 2.0 / (1.0 + cr + s * (1.0 + e) / (1.0 - e))


def eps_shell(ntu: float, cr: float, shell_passes: int = 1) -> float:
    """Casco y tubos con n pasos de casco (2n, 4n, … pasos de tubo).

    Un casco: ε₁ = 2/{1 + C_r + √(1+C_r²)·[1 + e^(−NTU√(1+C_r²))]/[1 − e^(−NTU√(1+C_r²))]}.
    n cascos en contracorriente, cada uno con NTU/n:
    ε = {[(1 − ε₁C_r)/(1 − ε₁)]ⁿ − 1}/{[(1 − ε₁C_r)/(1 − ε₁)]ⁿ − C_r}; con
    C_r = 1, ε = nε₁/[1 + (n − 1)ε₁] (Incropera et al., 2007, tabla 11.3;
    Kays y London, 1984).
    """
    if shell_passes == 1:
        return _eps_shell1(ntu, cr)
    e1 = _eps_shell1(ntu / shell_passes, cr)
    if abs(1.0 - cr) < _CR_ONE:
        return shell_passes * e1 / (1.0 + (shell_passes - 1) * e1)
    a = ((1.0 - e1 * cr) / (1.0 - e1)) ** shell_passes
    return (a - 1.0) / (a - cr)


def eps_cross_unmixed(ntu: float, cr: float) -> float:
    """Flujo cruzado de un paso con los dos fluidos sin mezclar: la serie exacta.

    ε = 1/(C_r·NTU)·Σₙ Pₙ(NTU)·Pₙ(C_r·NTU), con Pₙ(x) = 1 − e^(−x)·Σ_{m≤n} xᵐ/m!
    (la función gamma incompleta regularizada). Mason (1955); Shah y Sekulić
    (2003), tabla 3.3. Las tablas de los libros traen la aproximación de
    :func:`eps_cross_unmixed_approx`, que se aparta hasta ~4 %.
    """
    if cr <= 0.0:
        return -math.expm1(-ntu)
    a, b = ntu, cr * ntu
    n = np.arange(1.0, int(a + 12.0 * math.sqrt(a) + 60.0) + 1.0)
    return float(np.sum(gammainc(n, a) * gammainc(n, b)) / b)


def eps_cross_unmixed_approx(ntu: float, cr: float) -> float:
    """La aproximación de los libros para los dos fluidos sin mezclar.

    ε ≈ 1 − exp{(1/C_r)·NTU^0,22·[e^(−C_r·NTU^0,78) − 1]} (Incropera et al., 2007,
    tabla 11.3; Cengel y Ghajar, 2015, tabla 11-4).
    """
    if cr <= 0.0:
        return -math.expm1(-ntu)
    return 1.0 - math.exp((1.0 / cr) * ntu**0.22 * math.expm1(-cr * ntu**0.78))


def eps_cross_cmax_mixed(ntu: float, cr: float) -> float:
    """Flujo cruzado con C_máx mezclado y C_mín sin mezclar.

    ε = (1/C_r)·{1 − exp[−C_r·(1 − e^(−NTU))]} (Incropera et al., 2007, tabla 11.3).
    """
    if cr <= 0.0:
        return -math.expm1(-ntu)
    inner = -math.expm1(-ntu)  # 1 − e^(−NTU)
    return -math.expm1(-cr * inner) / cr


def eps_cross_cmin_mixed(ntu: float, cr: float) -> float:
    """Flujo cruzado con C_mín mezclado y C_máx sin mezclar.

    ε = 1 − exp{−(1/C_r)·[1 − e^(−C_r·NTU)]} (Incropera et al., 2007, tabla 11.3).
    """
    if cr <= 0.0:
        return -math.expm1(-ntu)
    inner = -math.expm1(-cr * ntu)  # 1 − e^(−C_r·NTU)
    return -math.expm1(-inner / cr)


# ---------------------------------------------------------------------
# El arreglo
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Arrangement:
    """El tipo de intercambiador.

    ``shell_passes`` es la cantidad de pasos de casco (solo ``"shell"``);
    ``mixed`` es el fluido mezclado (solo ``"cross_mixed"``) y ``tubes`` el que
    va por los tubos, para P y R como Cengel y Ghajar (t: el fluido de los tubos).
    """

    kind: ExchangerKind
    shell_passes: int = 1
    mixed: Side = "hot"
    tubes: Side = "hot"

    @property
    def name(self) -> str:
        if self.kind == "shell":
            n = self.shell_passes
            tubes = f"{2 * n}, {4 * n}, … pasos de tubo"
            passes = "1 paso de casco" if n == 1 else f"{n} pasos de casco"
            return f"Casco y tubos: {passes} ({tubes})"
        if self.kind == "cross_mixed":
            return f"Flujo cruzado, el fluido {_SIDE_NAMES[self.mixed]} mezclado"
        return EXCHANGER_KINDS[self.kind]


def _mixed_is_cmin(arr: Arrangement, C_hot: float, C_cold: float) -> bool:
    """Si el fluido mezclado (flujo cruzado) es el de menor capacidad."""
    hot_is_min = C_hot <= C_cold
    return hot_is_min if arr.mixed == "hot" else not hot_is_min


def _eps_function(arr: Arrangement, mixed_is_cmin: bool) -> Callable[[float, float], float]:
    if arr.kind == "parallel":
        return eps_parallel
    if arr.kind == "counter":
        return eps_counter
    if arr.kind == "shell":
        n = arr.shell_passes
        return lambda ntu, cr: eps_shell(ntu, cr, n)
    if arr.kind == "cross_unmixed":
        return eps_cross_unmixed
    return eps_cross_cmin_mixed if mixed_is_cmin else eps_cross_cmax_mixed


def effectiveness(arr: Arrangement, ntu: float, cr: float, mixed_is_cmin: bool = False) -> float:
    """ε(NTU, C_r) del arreglo; con C_r = 0 (cambio de fase), 1 − e^(−NTU)."""
    if ntu <= 0.0:
        return 0.0
    if cr <= 0.0:
        return -math.expm1(-ntu)
    return _eps_function(arr, mixed_is_cmin)(ntu, cr)


def max_effectiveness(arr: Arrangement, cr: float, mixed_is_cmin: bool = False) -> float:
    """ε con NTU → ∞: el máximo que puede dar ese tipo con ese C_r."""
    if cr <= 0.0:
        return 1.0
    if arr.kind == "parallel":
        return 1.0 / (1.0 + cr)
    if arr.kind in ("counter", "cross_unmixed"):
        return 1.0
    if arr.kind == "shell":
        e1 = 2.0 / (1.0 + cr + math.sqrt(1.0 + cr * cr))
        n = arr.shell_passes
        if n == 1:
            return e1
        if abs(1.0 - cr) < _CR_ONE:
            return n * e1 / (1.0 + (n - 1) * e1)
        a = ((1.0 - e1 * cr) / (1.0 - e1)) ** n
        return (a - 1.0) / (a - cr)
    if mixed_is_cmin:
        return -math.expm1(-1.0 / cr)
    return -math.expm1(-cr) / cr


def _ntu_counter(eps: float, cr: float) -> float:
    if abs(1.0 - cr) < _CR_ONE:
        return eps / (1.0 - eps)
    return math.log((eps - 1.0) / (eps * cr - 1.0)) / (cr - 1.0)


def _ntu_shell1(eps: float, cr: float) -> float:
    s = math.sqrt(1.0 + cr * cr)
    E = (2.0 / eps - (1.0 + cr)) / s
    return -math.log((E - 1.0) / (E + 1.0)) / s


def ntu_from_effectiveness(
    arr: Arrangement, eps: float, cr: float, mixed_is_cmin: bool = False
) -> float:
    """NTU(ε, C_r): la inversa (Incropera et al., 2007, tabla 11.4).

    Cerrada salvo en el flujo cruzado sin mezclar, que se invierte con
    ``brentq``.

    Raises
    ------
    ValueError
        Si ε no es alcanzable con ese tipo (ε ≥ ε_máx) o pide un NTU mayor que
        :data:`NTU_MAX`.
    """
    if eps <= 0.0:
        return 0.0
    eps_max = max_effectiveness(arr, cr, mixed_is_cmin)
    if eps >= eps_max * (1.0 - 1e-12):
        raise ValueError(
            f"Con un intercambiador «{arr.name.lower()}» y C_r = {_num(cr, 3)} la "
            f"efectividad no puede pasar de {_num(eps_max, 4)} ni con un área infinita, y "
            f"hace falta ε = {_num(eps, 4)}. Probá con contracorriente o con más pasos de "
            "casco, o pedí menos calor."
        )
    if cr <= 0.0:
        ntu = -math.log1p(-eps)
    elif arr.kind == "parallel":
        ntu = -math.log1p(-eps * (1.0 + cr)) / (1.0 + cr)
    elif arr.kind == "counter":
        ntu = _ntu_counter(eps, cr)
    elif arr.kind == "shell" and arr.shell_passes == 1:
        ntu = _ntu_shell1(eps, cr)
    elif arr.kind == "shell":
        n = arr.shell_passes
        if abs(1.0 - cr) < _CR_ONE:
            e1 = eps / (n - (n - 1) * eps)
        else:
            F = ((eps * cr - 1.0) / (eps - 1.0)) ** (1.0 / n)
            e1 = (F - 1.0) / (F - cr)
        ntu = n * _ntu_shell1(e1, cr)
    elif arr.kind == "cross_mixed" and not mixed_is_cmin:
        ntu = -math.log1p(math.log1p(-eps * cr) / cr)
    elif arr.kind == "cross_mixed":
        ntu = -math.log1p(cr * math.log1p(-eps)) / cr
    else:
        f = _eps_function(arr, mixed_is_cmin)
        if f(NTU_MAX, cr) <= eps:
            ntu = math.inf
        else:
            ntu = brentq(lambda x: f(x, cr) - eps, 1e-12, NTU_MAX, xtol=1e-14, rtol=1e-14)
    if not ntu < NTU_MAX:
        raise ValueError(
            f"Para ε = {_num(eps, 4)} haría falta un NTU de más de {_num(NTU_MAX)}: un "
            "intercambiador así no es razonable. Pedí menos calor o cambiá el tipo."
        )
    return ntu


def correction_factor(
    arr: Arrangement, eps: float, cr: float, mixed_is_cmin: bool = False
) -> float:
    """F = NTU_contracorriente/NTU_arreglo para el mismo ε y C_r.

    Es el factor de corrección de la LMTD (Cengel y Ghajar, §11-4; Bowman,
    Mueller y Nagle, 1940): Q̇ = U·A·F·ΔT_ml,cc. Vale 1 en contracorriente y con
    C_r = 0. En flujo paralelo no se usa (la ΔT_ml es la del paralelo).
    """
    if eps <= 0.0 or cr <= 0.0 or arr.kind in ("counter", "parallel"):
        return 1.0
    return _ntu_counter(eps, cr) / ntu_from_effectiveness(arr, eps, cr, mixed_is_cmin)


def lmtd(dT1: float, dT2: float) -> float:
    """ΔT_ml = (ΔT₁ − ΔT₂)/ln(ΔT₁/ΔT₂); con ΔT₁ = ΔT₂, la misma ΔT."""
    if dT1 <= 0.0 or dT2 <= 0.0:
        raise ValueError(
            "Una de las diferencias de temperatura de los extremos es cero o negativa: "
            "las temperaturas se cruzan y ese intercambiador no puede funcionar así."
        )
    if math.isclose(dT1, dT2, rel_tol=1e-9):
        return 0.5 * (dT1 + dT2)
    return (dT1 - dT2) / math.log(dT1 / dT2)


# ---------------------------------------------------------------------
# Corrientes
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    """Una corriente: el caudal, la temperatura de entrada y el c_p.

    Con ``cp_J_per_kgK`` dado se usa ese (como los libros); si no, el de
    CoolProp para ``fluid`` a ``p_Pa`` y a la temperatura media de la corriente.
    Con ``phase_change`` condensa o evapora a ``T_in_K`` (la de saturación): su
    temperatura no cambia (C → ∞), el caudal sale de Q̇/h_fg y h_fg es el dado o el
    de CoolProp a esa temperatura.
    """

    name: str
    T_in_K: float
    m_dot_kg_s: float = 0.0
    cp_J_per_kgK: float | None = None
    fluid: str | None = None
    p_Pa: float = 101_325.0
    phase_change: bool = False
    h_fg_J_per_kg: float | None = None


@dataclass(frozen=True)
class StreamResult:
    """Una corriente resuelta."""

    stream: Stream
    T_out_K: float
    m_dot_kg_s: float
    cp_J_per_kgK: float | None
    C_W_per_K: float
    h_fg_J_per_kg: float | None = None
    p_sat_Pa: float | None = None

    @property
    def name(self) -> str:
        return self.stream.name

    @property
    def T_in_K(self) -> float:
        return self.stream.T_in_K

    @property
    def phase_change(self) -> bool:
        return self.stream.phase_change

    @property
    def T_mean_K(self) -> float:
        return 0.5 * (self.stream.T_in_K + self.T_out_K)

    @property
    def cp_from_coolprop(self) -> bool:
        return self.stream.cp_J_per_kgK is None and not self.stream.phase_change

    @property
    def dT_K(self) -> float:
        return abs(self.T_out_K - self.stream.T_in_K)


def _check_stream(s: Stream, label: str) -> None:
    if s.T_in_K <= 0.0:
        raise ValueError(f"La temperatura del fluido {label} tiene que ser absoluta positiva.")
    if s.phase_change:
        if s.fluid is None and (s.h_fg_J_per_kg is None or s.h_fg_J_per_kg <= 0.0):
            raise ValueError(
                f"El fluido {label} cambia de fase: falta su calor latente h_fg (o elegí el "
                "fluido para sacarlo de CoolProp)."
            )
        if s.fluid is not None and not fluid_limits(s.fluid).is_pure:
            raise ValueError(
                f"{fluid_with_article(s.fluid).capitalize()} tiene deslizamiento de temperatura "
                "en la campana: no condensa ni evapora a temperatura constante. Elegí un "
                "fluido puro o tratalo sin cambio de fase."
            )
        return
    if s.cp_J_per_kgK is None and s.fluid is None:
        raise ValueError(f"Al fluido {label} le falta el c_p (o elegí un fluido de CoolProp).")
    if s.cp_J_per_kgK is not None and s.cp_J_per_kgK <= 0.0:
        raise ValueError(f"El c_p del fluido {label} tiene que ser positivo.")


def _cp(s: Stream, T_K: float, label: str) -> float:
    """c_p de CoolProp a (T, p), con el fluido en una sola fase."""
    assert s.fluid is not None
    try:
        state = fluid_state_from_pair(s.fluid, "TP", t=T_K, p=s.p_Pa)
    except ValueError as exc:
        raise ValueError(f"El fluido {label} ({s.name}) a {_degC(T_K)}: {exc}") from exc
    if state.region == "saturated_mixture" or state.cp_J_per_kg_K is None:
        raise ValueError(
            f"El fluido {label} ({s.name}) está cambiando de fase a {_degC(T_K)} y "
            f"{_num(s.p_Pa / 1e5)} bar. Subí o bajá la presión, o marcá que cambia de fase."
        )
    return float(state.cp_J_per_kg_K)


def _check_single_phase(s: Stream, T_out_K: float, label: str) -> None:
    """El fluido no puede cruzar la saturación entre la entrada y la salida."""
    if s.fluid is None or s.phase_change:
        return
    limits = fluid_limits(s.fluid)
    if not (limits.P_triple_Pa <= s.p_Pa < limits.P_crit_Pa):
        return
    T_sat = saturation_at_pressure(s.fluid, s.p_Pa).T_sat_K
    lo, hi = sorted((s.T_in_K, T_out_K))
    if lo < T_sat < hi:
        boils = s.T_in_K < T_out_K
        hint = "Subí la presión" if boils else "Bajá la presión"
        raise ValueError(
            f"{fluid_with_article(s.fluid).capitalize()} (fluido {label}) va de "
            f"{_degC(s.T_in_K)} a {_degC(T_out_K)}, y a {_num(s.p_Pa / 1e5)} bar "
            f"{'hierve' if boils else 'condensa'} a {_degC(T_sat)}: cambia de fase en el "
            f"intercambiador. {hint}, o marcá que cambia de fase."
        )


def _latent(s: Stream, label: str) -> tuple[float, float | None]:
    """h_fg y la presión de saturación de una corriente que cambia de fase."""
    if s.h_fg_J_per_kg is not None:
        p = saturation_at_temperature(s.fluid, s.T_in_K).P_sat_Pa if s.fluid else None
        return s.h_fg_J_per_kg, p
    assert s.fluid is not None
    try:
        sat = saturation_at_temperature(s.fluid, s.T_in_K)
    except ValueError as exc:
        raise ValueError(f"El fluido {label} ({s.name}): {exc}") from exc
    return sat.h_fg_J_per_kg, sat.P_sat_Pa


def _capacity(s: Stream, T_out_K: float, label: str) -> tuple[float, float | None]:
    """(C, c_p) de una corriente sin cambio de fase, con c_p a la T media."""
    if s.phase_change:
        return math.inf, None
    if s.m_dot_kg_s <= 0.0:
        raise ValueError(f"El caudal del fluido {label} tiene que ser positivo.")
    cp = s.cp_J_per_kgK
    if cp is None:
        cp = _cp(s, 0.5 * (s.T_in_K + T_out_K), label)
    return s.m_dot_kg_s * cp, cp


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ExergyBalance:
    """Exergía del intercambiador (vademecum §11.10 y §13.2)."""

    T0_K: float
    dS_hot_W_per_K: float
    dS_cold_W_per_K: float
    dPsi_hot_W: float
    dPsi_cold_W: float

    @property
    def S_gen_W_per_K(self) -> float:
        return self.dS_hot_W_per_K + self.dS_cold_W_per_K

    @property
    def X_dest_W(self) -> float:
        return self.T0_K * self.S_gen_W_per_K

    @property
    def efficiency(self) -> float | None:
        """η_ex = ΔΨ̇ del frío/(−ΔΨ̇ del caliente); None si no tiene sentido."""
        if self.dPsi_cold_W <= 0.0 or self.dPsi_hot_W >= 0.0:
            return None
        return self.dPsi_cold_W / -self.dPsi_hot_W


@dataclass(frozen=True)
class RatingInputs:
    """Verificación (ε-NTU): el intercambiador es conocido (U y A)."""

    arrangement: Arrangement
    hot: Stream
    cold: Stream
    U_W_per_m2K: float
    A_m2: float
    T0_K: float = 298.15


@dataclass(frozen=True)
class SizingInputs:
    """Dimensionamiento: lo que tiene que hacer → el área (y el largo del tubo)."""

    arrangement: Arrangement
    hot: Stream
    cold: Stream
    U_W_per_m2K: float
    target: SizingTarget
    target_value: float
    tube_D_m: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class FourTemperatureInputs:
    """Ensayo: las cuatro temperaturas y el área; U dado o un caudal dado.

    Con ``U_W_per_m2K`` sale Q̇ (y los caudales); si es ``None``, el caudal de
    ``known_flow`` (con su c_p, o su h_fg si cambia de fase) da Q̇ y sale U.
    """

    arrangement: Arrangement
    hot: Stream
    cold: Stream
    T_hot_out_K: float
    T_cold_out_K: float
    A_m2: float
    U_W_per_m2K: float | None = None
    known_flow: Side = "hot"
    T0_K: float = 298.15


@dataclass(frozen=True)
class ExchangerResult:
    """Un intercambiador resuelto (cualquiera de los tres problemas)."""

    inputs: RatingInputs | SizingInputs | FourTemperatureInputs
    hot: StreamResult
    cold: StreamResult
    Q_W: float
    effectiveness: float
    NTU: float
    UA_W_per_K: float
    A_m2: float
    U_W_per_m2K: float
    F: float
    dT_lm_K: float
    mixed_is_cmin: bool
    exergy: ExergyBalance
    warnings: tuple[str, ...] = ()

    @property
    def arrangement(self) -> Arrangement:
        return self.inputs.arrangement

    @property
    def mode(self) -> Literal["rating", "sizing", "test"]:
        if isinstance(self.inputs, RatingInputs):
            return "rating"
        if isinstance(self.inputs, SizingInputs):
            return "sizing"
        return "test"

    @property
    def C_min_W_per_K(self) -> float:
        return min(self.hot.C_W_per_K, self.cold.C_W_per_K)

    @property
    def C_max_W_per_K(self) -> float:
        return max(self.hot.C_W_per_K, self.cold.C_W_per_K)

    @property
    def C_r(self) -> float:
        return 0.0 if math.isinf(self.C_max_W_per_K) else self.C_min_W_per_K / self.C_max_W_per_K

    @property
    def cmin_side(self) -> Side:
        return "hot" if self.hot.C_W_per_K <= self.cold.C_W_per_K else "cold"

    @property
    def dT_max_K(self) -> float:
        return self.hot.T_in_K - self.cold.T_in_K

    @property
    def Q_max_W(self) -> float:
        return self.C_min_W_per_K * self.dT_max_K

    @property
    def eps_max(self) -> float:
        return max_effectiveness(self.arrangement, self.C_r, self.mixed_is_cmin)

    @property
    def end_differences_K(self) -> tuple[float, float]:
        """(ΔT₁, ΔT₂) de la LMTD: los extremos del paralelo o del contracorriente."""
        if self.arrangement.kind == "parallel":
            return (self.hot.T_in_K - self.cold.T_in_K, self.hot.T_out_K - self.cold.T_out_K)
        return (self.hot.T_in_K - self.cold.T_out_K, self.hot.T_out_K - self.cold.T_in_K)

    @property
    def P_R(self) -> tuple[float, float] | None:
        """(P, R) de Cengel y Ghajar con t el fluido de los tubos; None con cambio de fase."""
        if self.hot.phase_change or self.cold.phase_change:
            return None
        t, s = (self.hot, self.cold) if self.arrangement.tubes == "hot" else (self.cold, self.hot)
        dt = t.T_out_K - t.T_in_K
        return dt / (s.T_in_K - t.T_in_K), (s.T_in_K - s.T_out_K) / dt

    @property
    def tube_length_m(self) -> float | None:
        D = getattr(self.inputs, "tube_D_m", None)
        if not D:
            return None
        return self.A_m2 / (math.pi * D)

    @property
    def eps_approx(self) -> float | None:
        """ε de la fórmula aproximada de los libros (flujo cruzado sin mezclar)."""
        if self.arrangement.kind != "cross_unmixed" or self.C_r <= 0.0:
            return None
        return eps_cross_unmixed_approx(self.NTU, self.C_r)


def _exergy(hot: StreamResult, cold: StreamResult, Q: float, T0: float) -> ExergyBalance:
    def parts(r: StreamResult, sign: float) -> tuple[float, float]:
        if r.phase_change:
            dS = sign * Q / r.T_in_K
            return dS, sign * Q - T0 * dS
        dS = r.C_W_per_K * math.log(r.T_out_K / r.T_in_K)
        return dS, r.C_W_per_K * (r.T_out_K - r.T_in_K) - T0 * dS

    dS_h, dPsi_h = parts(hot, -1.0)
    dS_c, dPsi_c = parts(cold, +1.0)
    return ExergyBalance(T0, dS_h, dS_c, dPsi_h, dPsi_c)


def _validate_common(
    arr: Arrangement, hot: Stream, cold: Stream, T0_K: float, U: float | None
) -> None:
    if arr.kind not in EXCHANGER_KINDS:
        raise ValueError(f"Tipo de intercambiador desconocido: {arr.kind!r}.")
    if arr.kind == "shell" and not 1 <= arr.shell_passes <= 8:
        raise ValueError("Los pasos de casco van de 1 a 8.")
    _check_stream(hot, "caliente")
    _check_stream(cold, "frío")
    if hot.phase_change and cold.phase_change:
        raise ValueError(
            "Los dos fluidos cambian de fase: ninguno cambia de temperatura y estos métodos "
            "no se aplican (Q̇ = U·A·(T_h − T_c) directamente)."
        )
    if hot.T_in_K <= cold.T_in_K:
        raise ValueError(
            f"El fluido caliente entra a {_degC(hot.T_in_K)} y el frío a "
            f"{_degC(cold.T_in_K)}: el caliente tiene que entrar más caliente que el frío."
        )
    if T0_K <= 0.0:
        raise ValueError("La temperatura del ambiente T₀ tiene que ser absoluta positiva.")
    if U is not None and U <= 0.0:
        raise ValueError("El coeficiente global U tiene que ser positivo.")


def _warnings(r: ExchangerResult) -> tuple[str, ...]:
    out: list[str] = []
    arr = r.arrangement
    if arr.kind not in ("counter", "parallel") and r.C_r > 0.0 and r.F < F_MIN_RECOMMENDED:
        out.append(
            f"F = {_num(r.F, 3)} es menor que {_num(F_MIN_RECOMMENDED, 2)}: el arreglo "
            "desperdicia área (pequeños cambios hacen caer mucho a F). Conviene sumar pasos "
            "de casco o ir a contracorriente (Cengel y Ghajar, §11-4)."
        )
    if r.mode == "rating" and r.effectiveness > 0.97 * r.eps_max:
        out.append(
            f"ε = {_num(r.effectiveness, 3)} está muy cerca del máximo de este tipo "
            f"({_num(r.eps_max, 3)}): agregar área casi no suma calor."
        )
    return tuple(out)


def _build(
    inputs: RatingInputs | SizingInputs | FourTemperatureInputs,
    hot: StreamResult,
    cold: StreamResult,
    Q: float,
    UA: float,
    A: float,
) -> ExchangerResult:
    arr = inputs.arrangement
    C_min = min(hot.C_W_per_K, cold.C_W_per_K)
    C_max = max(hot.C_W_per_K, cold.C_W_per_K)
    cr = 0.0 if math.isinf(C_max) else C_min / C_max
    mixed_is_cmin = _mixed_is_cmin(arr, hot.C_W_per_K, cold.C_W_per_K)
    eps = Q / (C_min * (hot.T_in_K - cold.T_in_K))
    ntu = UA / C_min
    F = correction_factor(arr, eps, cr, mixed_is_cmin)
    probe = ExchangerResult(
        inputs,
        hot,
        cold,
        Q,
        eps,
        ntu,
        UA,
        A,
        UA / A,
        F,
        0.0,
        mixed_is_cmin,
        _exergy(hot, cold, Q, inputs.T0_K),
    )
    dT1, dT2 = probe.end_differences_K
    result = replace(probe, dT_lm_K=lmtd(dT1, dT2))
    return replace(result, warnings=_warnings(result))


# ---------------------------------------------------------------------
# Los tres problemas
# ---------------------------------------------------------------------


def _stream_result(
    s: Stream, T_out: float, C: float, cp: float | None, Q: float, label: str
) -> StreamResult:
    if s.phase_change:
        h_fg, p_sat = _latent(s, label)
        return StreamResult(s, s.T_in_K, Q / h_fg, None, math.inf, h_fg, p_sat)
    return StreamResult(s, T_out, s.m_dot_kg_s, cp, C)


def solve_rating(inputs: RatingInputs) -> ExchangerResult:
    """Verificación: Q̇ y las salidas con ε-NTU (Cengel y Ghajar, §11-5).

    Con c_p de CoolProp se itera la temperatura media de cada corriente.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    arr, hot, cold = inputs.arrangement, inputs.hot, inputs.cold
    _validate_common(arr, hot, cold, inputs.T0_K, inputs.U_W_per_m2K)
    if inputs.A_m2 <= 0.0:
        raise ValueError("El área de transferencia A tiene que ser positiva.")
    UA = inputs.U_W_per_m2K * inputs.A_m2
    T_h_out, T_c_out = hot.T_in_K, cold.T_in_K
    for _ in range(200):
        C_h, cp_h = _capacity(hot, T_h_out, "caliente")
        C_c, cp_c = _capacity(cold, T_c_out, "frío")
        C_min, C_max = min(C_h, C_c), max(C_h, C_c)
        cr = 0.0 if math.isinf(C_max) else C_min / C_max
        eps = effectiveness(arr, UA / C_min, cr, _mixed_is_cmin(arr, C_h, C_c))
        Q = eps * C_min * (hot.T_in_K - cold.T_in_K)
        new_h = hot.T_in_K if hot.phase_change else hot.T_in_K - Q / C_h
        new_c = cold.T_in_K if cold.phase_change else cold.T_in_K + Q / C_c
        done = abs(new_h - T_h_out) < 1e-10 and abs(new_c - T_c_out) < 1e-10
        T_h_out, T_c_out = new_h, new_c
        if done:
            break
    _check_single_phase(hot, T_h_out, "caliente")
    _check_single_phase(cold, T_c_out, "frío")
    h = _stream_result(hot, T_h_out, C_h, cp_h, Q, "caliente")
    c = _stream_result(cold, T_c_out, C_c, cp_c, Q, "frío")
    return _build(inputs, h, c, Q, UA, inputs.A_m2)


def solve_sizing(inputs: SizingInputs) -> ExchangerResult:
    """Dimensionamiento: el área para una salida o un Q̇ dados.

    Q̇ sale del objetivo; la otra salida, del balance; NTU = NTU(ε, C_r) y
    A = NTU·C_mín/U. Es lo mismo que A = Q̇/(U·F·ΔT_ml) (Cengel y Ghajar, §11-4).

    Raises
    ------
    ValueError
        Si el objetivo no se puede alcanzar (con un mensaje que dice por qué).
    """
    arr, hot, cold = inputs.arrangement, inputs.hot, inputs.cold
    _validate_common(arr, hot, cold, inputs.T0_K, inputs.U_W_per_m2K)
    if inputs.tube_D_m is not None and inputs.tube_D_m <= 0.0:
        raise ValueError("El diámetro del tubo tiene que ser positivo.")
    target, value = inputs.target, inputs.target_value
    dT_max = hot.T_in_K - cold.T_in_K
    C_h = C_c = math.inf
    cp_h = cp_c = None
    if target == "T_hot_out":
        if hot.phase_change:
            raise ValueError("El fluido caliente cambia de fase: su salida no es un objetivo.")
        if not cold.T_in_K < value < hot.T_in_K:
            raise ValueError(
                f"La salida del caliente tiene que quedar entre la entrada del frío "
                f"({_degC(cold.T_in_K)}) y la suya ({_degC(hot.T_in_K)})."
            )
        C_h, cp_h = _capacity(hot, value, "caliente")
        Q = C_h * (hot.T_in_K - value)
    elif target == "T_cold_out":
        if cold.phase_change:
            raise ValueError("El fluido frío cambia de fase: su salida no es un objetivo.")
        if not cold.T_in_K < value < hot.T_in_K:
            raise ValueError(
                f"La salida del frío tiene que quedar entre su entrada ({_degC(cold.T_in_K)}) "
                f"y la del caliente ({_degC(hot.T_in_K)})."
            )
        C_c, cp_c = _capacity(cold, value, "frío")
        Q = C_c * (value - cold.T_in_K)
    else:
        if value <= 0.0:
            raise ValueError("El calor pedido tiene que ser positivo.")
        Q = value
    # la otra corriente (o las dos, si se pidió Q̇): c_p a su T media, iterando
    T_h_out = value if target == "T_hot_out" else hot.T_in_K
    T_c_out = value if target == "T_cold_out" else cold.T_in_K
    for _ in range(200):
        if target != "T_hot_out":
            C_h, cp_h = _capacity(hot, T_h_out, "caliente")
        if target != "T_cold_out":
            C_c, cp_c = _capacity(cold, T_c_out, "frío")
        new_h = hot.T_in_K if hot.phase_change else hot.T_in_K - Q / C_h
        new_c = cold.T_in_K if cold.phase_change else cold.T_in_K + Q / C_c
        if new_h < cold.T_in_K or new_c > hot.T_in_K:
            Q_max = min(C_h, C_c) * dT_max
            raise ValueError(
                f"Para transferir {_num(Q / 1e3, 4)} kW el caliente tendría que salir a "
                f"{_degC(new_h)} y el frío a {_degC(new_c)}: las temperaturas se cruzan. "
                f"Lo máximo posible es Q̇_máx = C_mín·ΔT_máx = {_num(Q_max / 1e3, 4)} kW."
            )
        done = abs(new_h - T_h_out) < 1e-10 and abs(new_c - T_c_out) < 1e-10
        T_h_out, T_c_out = new_h, new_c
        if done:
            break
    _check_single_phase(hot, T_h_out, "caliente")
    _check_single_phase(cold, T_c_out, "frío")
    C_min, C_max = min(C_h, C_c), max(C_h, C_c)
    cr = 0.0 if math.isinf(C_max) else C_min / C_max
    eps = Q / (C_min * dT_max)
    if eps >= 1.0:
        raise ValueError(
            f"Para ese objetivo haría falta ε = {_num(eps, 4)} ≥ 1: se pide más que el "
            f"máximo posible, Q̇_máx = {_num(C_min * dT_max / 1e3, 4)} kW."
        )
    if arr.kind == "parallel" and T_c_out >= T_h_out:
        raise ValueError(
            f"En flujo paralelo el frío no puede salir más caliente que el caliente "
            f"({_degC(T_c_out)} contra {_degC(T_h_out)}): las dos corrientes se acercan a una "
            "misma temperatura. Usá contracorriente."
        )
    ntu = ntu_from_effectiveness(arr, eps, cr, _mixed_is_cmin(arr, C_h, C_c))
    UA = ntu * C_min
    h = _stream_result(hot, T_h_out, C_h, cp_h, Q, "caliente")
    c = _stream_result(cold, T_c_out, C_c, cp_c, Q, "frío")
    return _build(inputs, h, c, Q, UA, UA / inputs.U_W_per_m2K)


def solve_four_temperatures(inputs: FourTemperatureInputs) -> ExchangerResult:
    """Ensayo con las cuatro temperaturas (Cengel y Ghajar, ejemplos 11-3, 11-5 y 11-6).

    Las temperaturas dan C_c/C_h = ΔT_h/ΔT_c, ε y C_r; NTU = NTU(ε, C_r) y F.
    Con U dado: U·A = NTU·C_mín da C_mín y Q̇. Con un caudal dado: Q̇ = C·ΔT de esa
    corriente y U = Q̇/(A·F·ΔT_ml).

    Raises
    ------
    ValueError
        Si las temperaturas no son posibles en ese intercambiador.
    """
    arr, hot, cold = inputs.arrangement, inputs.hot, inputs.cold
    _validate_common(arr, hot, cold, inputs.T0_K, inputs.U_W_per_m2K)
    if inputs.A_m2 <= 0.0:
        raise ValueError("El área de transferencia A tiene que ser positiva.")
    T_h_out = hot.T_in_K if hot.phase_change else inputs.T_hot_out_K
    T_c_out = cold.T_in_K if cold.phase_change else inputs.T_cold_out_K
    if not hot.phase_change and not cold.T_in_K < T_h_out < hot.T_in_K:
        raise ValueError(
            f"El caliente tiene que salir más frío que como entra ({_degC(hot.T_in_K)}) y "
            f"más caliente que la entrada del frío ({_degC(cold.T_in_K)})."
        )
    if not cold.phase_change and not cold.T_in_K < T_c_out < hot.T_in_K:
        raise ValueError(
            f"El frío tiene que salir más caliente que como entra ({_degC(cold.T_in_K)}) y "
            f"más frío que la entrada del caliente ({_degC(hot.T_in_K)})."
        )
    dT_h = hot.T_in_K - T_h_out
    dT_c = T_c_out - cold.T_in_K
    dT_max = hot.T_in_K - cold.T_in_K
    # C_mín es el fluido que más cambia de temperatura
    if hot.phase_change or cold.phase_change:
        cr = 0.0
        dT_big = dT_c if hot.phase_change else dT_h
        hot_is_min = cold.phase_change
    else:
        hot_is_min = dT_h >= dT_c
        dT_big, dT_small = (dT_h, dT_c) if hot_is_min else (dT_c, dT_h)
        cr = dT_small / dT_big
    eps = dT_big / dT_max
    mixed_is_cmin = hot_is_min if arr.mixed == "hot" else not hot_is_min
    if arr.kind == "parallel" and not (hot.phase_change or cold.phase_change):
        if T_c_out >= T_h_out:
            raise ValueError(
                "En flujo paralelo el frío no puede salir más caliente que el caliente: "
                "revisá las temperaturas o usá contracorriente."
            )
    ntu = ntu_from_effectiveness(arr, eps, cr, mixed_is_cmin)
    if inputs.U_W_per_m2K is not None:
        UA = inputs.U_W_per_m2K * inputs.A_m2
        C_min = UA / ntu
        Q = C_min * dT_big
    else:
        side = hot if inputs.known_flow == "hot" else cold
        label = _SIDE_NAMES[inputs.known_flow]
        if side.m_dot_kg_s <= 0.0:
            raise ValueError(f"Falta el caudal del fluido {label} (o el U).")
        if side.phase_change:
            h_fg, _ = _latent(side, label)
            Q = side.m_dot_kg_s * h_fg
        else:
            T_out = T_h_out if inputs.known_flow == "hot" else T_c_out
            C, _ = _capacity(side, T_out, label)
            Q = C * (dT_h if inputs.known_flow == "hot" else dT_c)
        C_min = Q / dT_big
        UA = ntu * C_min
    C_h = math.inf if hot.phase_change else Q / dT_h
    C_c = math.inf if cold.phase_change else Q / dT_c
    _check_single_phase(hot, T_h_out, "caliente")
    _check_single_phase(cold, T_c_out, "frío")
    h = _test_stream(hot, T_h_out, C_h, Q, "caliente")
    c = _test_stream(cold, T_c_out, C_c, Q, "frío")
    return _build(inputs, h, c, Q, UA, inputs.A_m2)


def _test_stream(s: Stream, T_out: float, C: float, Q: float, label: str) -> StreamResult:
    """En el ensayo el caudal sale de C (con el c_p de la corriente, si se puede)."""
    if s.phase_change:
        return _stream_result(s, T_out, C, None, Q, label)
    cp = s.cp_J_per_kgK
    if cp is None and s.fluid is not None:
        cp = _cp(s, 0.5 * (s.T_in_K + T_out), label)
    m_dot = C / cp if cp else 0.0
    return StreamResult(replace(s, m_dot_kg_s=m_dot), T_out, m_dot, cp, C)


# ---------------------------------------------------------------------
# Comparación de tipos, curvas y perfil
# ---------------------------------------------------------------------


_COMPARED: tuple[Arrangement, ...] = (
    Arrangement("parallel"),
    Arrangement("counter"),
    Arrangement("shell", 1),
    Arrangement("shell", 2),
    Arrangement("cross_unmixed"),
    Arrangement("cross_mixed", mixed="hot"),
    Arrangement("cross_mixed", mixed="cold"),
)


@dataclass(frozen=True)
class TypeComparison:
    """El mismo problema con otro tipo de intercambiador."""

    arrangement: Arrangement
    Q_W: float | None
    A_m2: float | None
    effectiveness: float | None
    F: float | None
    error: str | None = None


def type_comparison(result: ExchangerResult) -> list[TypeComparison]:
    """Cada tipo con los mismos datos: el calor (verificación) o el área (dimensionamiento).

    En el ensayo se compara el área para el mismo trabajo (como un dimensionamiento).
    """
    inputs = result.inputs
    out: list[TypeComparison] = []
    for arr in _COMPARED:
        arr = replace(arr, tubes=result.arrangement.tubes)
        try:
            if isinstance(inputs, RatingInputs):
                r = solve_rating(replace(inputs, arrangement=arr))
            else:
                sizing = SizingInputs(
                    arr,
                    replace(result.hot.stream, m_dot_kg_s=result.hot.m_dot_kg_s),
                    replace(result.cold.stream, m_dot_kg_s=result.cold.m_dot_kg_s),
                    result.U_W_per_m2K,
                    "Q",
                    result.Q_W,
                    getattr(inputs, "tube_D_m", None),
                    inputs.T0_K,
                )
                if result.hot.phase_change:
                    sizing = replace(sizing, hot=result.hot.stream)
                if result.cold.phase_change:
                    sizing = replace(sizing, cold=result.cold.stream)
                r = solve_sizing(sizing)
        except ValueError as exc:
            out.append(TypeComparison(arr, None, None, None, None, str(exc)))
            continue
        out.append(TypeComparison(arr, r.Q_W, r.A_m2, r.effectiveness, r.F))
    return out


def effectiveness_curves(
    arr: Arrangement,
    cr_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ntu_max: float = 5.0,
    n: int = 81,
    mixed_is_cmin: bool = False,
) -> dict[float, tuple[list[float], list[float]]]:
    """ε(NTU) para varios C_r: el gráfico de los libros (Cengel y Ghajar, fig. 11-26)."""
    xs = [ntu_max * j / (n - 1) for j in range(n)]
    return {cr: (xs, [effectiveness(arr, x, cr, mixed_is_cmin) for x in xs]) for cr in cr_values}


def f_curves(
    arr: Arrangement,
    R_values: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0),
    n: int = 60,
    F_floor: float = 0.5,
) -> dict[float, tuple[list[float], list[float]]]:
    """F(P) para varios R (Bowman, Mueller y Nagle, 1940; Cengel y Ghajar, fig. 11-18).

    P = (t₂ − t₁)/(T₁ − t₁) y R = (T₁ − T₂)/(t₂ − t₁) = C_t/C_s, con t el fluido de
    los tubos. Cada curva llega hasta donde F cae a ``F_floor``.
    """
    out: dict[float, tuple[list[float], list[float]]] = {}
    for R in R_values:
        # el fluido de los tubos es C_mín si R ≤ 1
        cr = R if R <= 1.0 else 1.0 / R
        tubes_min = R <= 1.0
        mixed_is_cmin = (arr.mixed == arr.tubes) == tubes_min
        eps_max = max_effectiveness(arr, cr, mixed_is_cmin)
        P_max = eps_max if tubes_min else eps_max / R
        Ps, Fs = [], []
        for j in range(1, n):
            P = P_max * j / n
            eps = P if tubes_min else P * R
            try:
                F = correction_factor(arr, eps, cr, mixed_is_cmin)
            except ValueError:
                break
            if F < F_floor:
                break
            Ps.append(P)
            Fs.append(F)
        out[R] = (Ps, Fs)
    return out


@dataclass(frozen=True)
class ProfilePoint:
    """Un punto del perfil de un doble tubo: fracción del área y temperaturas."""

    x: float
    T_hot_K: float
    T_cold_K: float


def temperature_profile(result: ExchangerResult, n: int = 61) -> list[ProfilePoint] | None:
    """T_h(x) y T_c(x) a lo largo de un doble tubo (x = fracción del área desde la
    entrada del caliente). None para casco y tubos o flujo cruzado (no son 1D)."""
    kind = result.arrangement.kind
    if kind not in ("parallel", "counter"):
        return None
    UA = result.UA_W_per_K
    inv_h = 0.0 if result.hot.phase_change else 1.0 / result.hot.C_W_per_K
    inv_c = 0.0 if result.cold.phase_change else 1.0 / result.cold.C_W_per_K
    T_hi = result.hot.T_in_K
    if kind == "parallel":
        dT0, k = T_hi - result.cold.T_in_K, UA * (inv_h + inv_c)
    else:
        dT0, k = T_hi - result.cold.T_out_K, UA * (inv_h - inv_c)
    pts: list[ProfilePoint] = []
    for j in range(n):
        x = j / (n - 1)
        q = UA * dT0 * x if abs(k) < 1e-12 else UA * dT0 * -math.expm1(-k * x) / k
        T_h = T_hi - q * inv_h
        if kind == "parallel":
            T_c = result.cold.T_in_K + q * inv_c
        else:
            T_c = result.cold.T_out_K - q * inv_c
        pts.append(ProfilePoint(x, T_h, T_c))
    return pts


# ---------------------------------------------------------------------
# Coeficiente global U (Cengel y Ghajar, §11-2; Incropera, §11.2)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class OverallUInputs:
    """La pared de un intercambiador, con convección y ensuciamiento de cada lado.

    ``tube``: un tubo de diámetros D_i y D_o y largo L (lado i = adentro).
    ``plane``: una pared plana de espesor t y área A (la de un intercambiador de
    placas). R''_f son los factores de ensuciamiento (por unidad de área).
    """

    geometry: WallGeometry
    h_i_W_per_m2K: float
    h_o_W_per_m2K: float
    k_wall_W_per_mK: float
    D_i_m: float = 0.0
    D_o_m: float = 0.0
    L_m: float = 1.0
    thickness_m: float = 0.0
    A_m2: float = 1.0
    R_f_i_m2K_per_W: float = 0.0
    R_f_o_m2K_per_W: float = 0.0


@dataclass(frozen=True)
class UResistance:
    """Una resistencia de la pared del intercambiador."""

    label: str
    R_K_per_W: float
    kind: Literal["convección", "ensuciamiento", "pared"]


@dataclass(frozen=True)
class OverallUResult:
    inputs: OverallUInputs
    resistances: tuple[UResistance, ...]
    A_i_m2: float
    A_o_m2: float

    @property
    def R_total_K_per_W(self) -> float:
        return sum(r.R_K_per_W for r in self.resistances)

    @property
    def UA_W_per_K(self) -> float:
        return 1.0 / self.R_total_K_per_W

    @property
    def U_i_W_per_m2K(self) -> float:
        return self.UA_W_per_K / self.A_i_m2

    @property
    def U_o_W_per_m2K(self) -> float:
        return self.UA_W_per_K / self.A_o_m2

    @property
    def R_clean_K_per_W(self) -> float:
        return sum(r.R_K_per_W for r in self.resistances if r.kind != "ensuciamiento")

    @property
    def U_i_clean_W_per_m2K(self) -> float:
        return 1.0 / (self.R_clean_K_per_W * self.A_i_m2)

    def share(self, r: UResistance) -> float:
        return r.R_K_per_W / self.R_total_K_per_W


def overall_u(inputs: OverallUInputs) -> OverallUResult:
    """1/(U·A) = 1/(h_i·A_i) + R''_f,i/A_i + R_pared + R''_f,o/A_o + 1/(h_o·A_o).

    Pared de un tubo: R = ln(D_o/D_i)/(2π·k·L); plana: t/(k·A) (Cengel y Ghajar,
    §11-2; Incropera et al., 2007, §11.2).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    i = inputs
    if i.h_i_W_per_m2K <= 0.0 or i.h_o_W_per_m2K <= 0.0:
        raise ValueError("Los coeficientes de convección h_i y h_o tienen que ser positivos.")
    if i.k_wall_W_per_mK <= 0.0:
        raise ValueError("La conductividad k de la pared tiene que ser positiva.")
    if i.R_f_i_m2K_per_W < 0.0 or i.R_f_o_m2K_per_W < 0.0:
        raise ValueError("Los factores de ensuciamiento no pueden ser negativos.")
    if i.geometry == "tube":
        if i.D_i_m <= 0.0 or i.L_m <= 0.0:
            raise ValueError("El diámetro interior y el largo del tubo tienen que ser positivos.")
        if i.D_o_m <= i.D_i_m:
            raise ValueError(
                "El diámetro exterior tiene que ser mayor que el interior (la pared tiene espesor)."
            )
        A_i, A_o = math.pi * i.D_i_m * i.L_m, math.pi * i.D_o_m * i.L_m
        R_wall = math.log(i.D_o_m / i.D_i_m) / (2.0 * math.pi * i.k_wall_W_per_mK * i.L_m)
        inner, outer = "adentro del tubo", "afuera del tubo"
    elif i.geometry == "plane":
        if i.thickness_m <= 0.0 or i.A_m2 <= 0.0:
            raise ValueError("El espesor y el área de la pared tienen que ser positivos.")
        A_i = A_o = i.A_m2
        R_wall = i.thickness_m / (i.k_wall_W_per_mK * i.A_m2)
        inner, outer = "lado 1", "lado 2"
    else:
        raise ValueError(f"Geometría desconocida: {i.geometry!r}.")
    rows = [UResistance(f"Convección {inner}", 1.0 / (i.h_i_W_per_m2K * A_i), "convección")]
    if i.R_f_i_m2K_per_W > 0.0:
        rows.append(UResistance(f"Ensuciamiento {inner}", i.R_f_i_m2K_per_W / A_i, "ensuciamiento"))
    rows.append(UResistance("Pared", R_wall, "pared"))
    if i.R_f_o_m2K_per_W > 0.0:
        rows.append(UResistance(f"Ensuciamiento {outer}", i.R_f_o_m2K_per_W / A_o, "ensuciamiento"))
    rows.append(UResistance(f"Convección {outer}", 1.0 / (i.h_o_W_per_m2K * A_o), "convección"))
    return OverallUResult(i, tuple(rows), A_i, A_o)


def overall_u_notes(result: OverallUResult) -> list[str]:
    """Interpretación física del U (markdown)."""
    notes: list[str] = []
    top = max(result.resistances, key=lambda r: r.R_K_per_W)
    notes.append(
        f"La resistencia que manda es **{top.label.lower()}** ({_pct(result.share(top))} del "
        "total): para subir U hay que actuar sobre ella (más velocidad, aletas de ese lado)."
    )
    i = result.inputs
    h_lo, h_hi = sorted((i.h_i_W_per_m2K, i.h_o_W_per_m2K))
    if h_hi > 10.0 * h_lo:
        notes.append(
            f"Un h es más de diez veces el otro ({_num(h_hi)} contra {_num(h_lo)} W/(m²·K)): "
            "U queda cerca del menor. Por eso los intercambiadores gas–líquido llevan aletas del "
            "lado del gas."
        )
    fouling = sum(r.R_K_per_W for r in result.resistances if r.kind == "ensuciamiento")
    if fouling > 0.0:
        drop = 1.0 - result.U_i_W_per_m2K / result.U_i_clean_W_per_m2K
        notes.append(
            f"El ensuciamiento baja U un {_pct(drop)} (de {_num(result.U_i_clean_W_per_m2K, 4)} "
            f"a {_num(result.U_i_W_per_m2K, 4)} W/(m²·K) referido a A_i): por eso se "
            "sobredimensiona el área y se limpian los tubos."
        )
    wall = next(r for r in result.resistances if r.kind == "pared")
    if result.share(wall) < 0.02:
        notes.append(
            f"La pared casi no pesa ({_pct(result.share(wall))}): con una pared delgada y "
            "metálica, U ≈ 1/(1/h_i + 1/h_o)."
        )
    if i.geometry == "tube":
        notes.append(
            "U depende del área a la que se refiere: U_i·A_i = U_o·A_o = UA. Con un tubo de "
            "pared delgada U_i ≈ U_o."
        )
    return notes


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def exchanger_notes(result: ExchangerResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = list(result.warnings)
    r = result
    cmin = r.hot if r.cmin_side == "hot" else r.cold
    notes.append(
        f"ε = {_num(r.effectiveness, 4)}: transfiere el {_pct(r.effectiveness)} del máximo "
        f"posible, Q̇_máx = C_mín·(T_h,ent − T_c,ent) = {_num(r.Q_max_W / 1e3, 4)} kW (el que "
        f"daría un contracorriente infinito)."
    )
    if r.C_r > 0.0:
        notes.append(
            f"El fluido de menor capacidad (C_mín) es el {_SIDE_NAMES[r.cmin_side]} "
            f"({cmin.name}): es el que más cambia de temperatura ({_num(cmin.dT_K, 3)} K)."
        )
    else:
        pc = r.hot if r.hot.phase_change else r.cold
        notes.append(
            f"El {pc.name} {'condensa' if pc is r.hot else 'evapora'} a temperatura constante: "
            "C → ∞ y C_r = 0, así que ε = 1 − e^(−NTU) en cualquier tipo de intercambiador "
            f"(se {'condensan' if pc is r.hot else 'evaporan'} {_num(pc.m_dot_kg_s, 4)} kg/s)."
        )
    if r.NTU > 3.0:
        notes.append(
            f"NTU = {_num(r.NTU, 3)}: pasado NTU ≈ 3 la curva ε(NTU) se aplana y más área suma "
            "poco calor."
        )
    kind = r.arrangement.kind
    if kind not in ("counter", "parallel") and r.C_r > 0.0 and r.F < 1.0:
        notes.append(
            f"F = {_num(r.F, 4)}: para el mismo trabajo este arreglo necesita {_num(1.0 / r.F, 3)} "
            "veces el área de un contracorriente (F = NTU_cc/NTU)."
        )
    if kind == "parallel":
        notes.append(
            "En flujo paralelo las dos corrientes se acercan a una misma temperatura: el frío "
            "nunca sale más caliente que el caliente, y ε no pasa de 1/(1 + C_r)."
        )
    if kind == "shell" and r.C_r > 0.0 and r.cold.T_out_K > r.hot.T_out_K:
        notes.append(
            "El frío sale más caliente que el caliente (cruce de temperaturas): en casco y tubos "
            "se puede, pero F cae rápido; con más pasos de casco el arreglo se parece a un "
            "contracorriente."
        )
    approx = r.eps_approx
    if approx is not None:
        notes.append(
            f"La fórmula aproximada de las tablas daría ε = {_num(approx, 4)} "
            f"({_pct(approx / r.effectiveness - 1.0)}); acá se usa la serie exacta (Mason, 1955)."
        )
    ex = r.exergy
    eta = ex.efficiency
    notes.append(
        f"Se destruyen Ẋ_dest = T₀·Ṡ_gen = {_num(ex.X_dest_W / 1e3, 4)} kW de exergía: el "
        "calor pasa con una diferencia finita de temperatura (en promedio ΔT_ml = "
        f"{_num(r.dT_lm_K, 3)} K). Más área (menos ΔT) destruye menos."
        + (f" Rendimiento exergético: η_ex = {_pct(eta)}." if eta is not None else "")
    )
    if eta is None:
        notes.append(
            "El η_ex del vademecum (§11.10) no se define acá: una corriente está por debajo de "
            "T₀ y calentarla hacia el ambiente no le suma exergía."
        )
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangerExample:
    """Un ejemplo con su nota."""

    inputs: RatingInputs | SizingInputs | FourTemperatureInputs | OverallUInputs
    note: str = ""


_K0 = 273.15

RATING_EXAMPLES: dict[str, ExchangerExample] = {
    "Aceite enfriado con agua: 1 casco y 8 pasos de tubo (Cengel y Ghajar 11-9)": ExchangerExample(
        RatingInputs(
            Arrangement("shell", 1, tubes="cold"),
            Stream("aceite", 150.0 + _K0, 0.3, cp_J_per_kgK=2130.0),
            Stream("agua", 20.0 + _K0, 0.2, cp_J_per_kgK=4180.0),
            310.0,
            8 * math.pi * 0.014 * 5.0,
        ),
        "El libro lee ε = 0,47 en el gráfico y da 39,1 kW; con la fórmula, ε = 0,462 y 38,4 kW.",
    ),
    "Agua calentada con agua geotérmica, contracorriente (Cengel y Ghajar 11-8)": ExchangerExample(
        RatingInputs(
            Arrangement("counter", tubes="cold"),
            Stream("agua geotérmica", 160.0 + _K0, 2.0, cp_J_per_kgK=4310.0),
            Stream("agua", 20.0 + _K0, 1.2, cp_J_per_kgK=4180.0),
            640.0,
            5.113,
        ),
        "Con el área del 11-4 (L = 108,5 m de un tubo de 1,5 cm) el agua sale a 80 °C.",
    ),
    "El mismo en flujo paralelo": ExchangerExample(
        RatingInputs(
            Arrangement("parallel", tubes="cold"),
            Stream("agua geotérmica", 160.0 + _K0, 2.0, cp_J_per_kgK=4310.0),
            Stream("agua", 20.0 + _K0, 1.2, cp_J_per_kgK=4180.0),
            640.0,
            5.113,
        ),
        "Con la misma área, el paralelo transfiere menos calor que el contracorriente.",
    ),
    "Condensador de una central, con CoolProp (basado en Cengel y Ghajar 11-3)": ExchangerExample(
        RatingInputs(
            Arrangement("shell", 1, tubes="cold"),
            Stream("vapor", 30.0 + _K0, fluid="Water", phase_change=True),
            Stream("agua de enfriamiento", 14.0 + _K0, 32.6, fluid="Water", p_Pa=2.0e5),
            2100.0,
            45.0,
        ),
        "El vapor condensa a 30 °C; con C_r = 0 cualquier tipo da lo mismo.",
    ),
    "Radiador de un auto, flujo cruzado sin mezclar (basado en Cengel y Ghajar 11-6)": (
        ExchangerExample(
            RatingInputs(
                Arrangement("cross_unmixed", tubes="hot"),
                Stream("agua", 90.0 + _K0, 0.6, cp_J_per_kgK=4195.0),
                Stream("aire", 20.0 + _K0, 3.131, cp_J_per_kgK=1005.0),
                3347.0,
                40 * math.pi * 0.005 * 0.65,
            ),
            "Con el U del ensayo del 11-6, el agua sale a 65 °C y el aire a 40 °C.",
        )
    ),
    "Recuperador de aire con gases de escape, aire mezclado (con CoolProp)": ExchangerExample(
        RatingInputs(
            Arrangement("cross_mixed", mixed="cold", tubes="hot"),
            Stream("gases (aire)", 400.0 + _K0, 1.2, fluid="Air"),
            Stream("aire de combustión", 25.0 + _K0, 1.0, fluid="Air"),
            60.0,
            40.0,
        ),
        "Los gases se modelan como aire; c_p sale de CoolProp a la temperatura media de cada "
        "corriente.",
    ),
}

SIZING_EXAMPLES: dict[str, ExchangerExample] = {
    "Agua calentada con agua geotérmica, contracorriente (Cengel y Ghajar 11-4)": ExchangerExample(
        SizingInputs(
            Arrangement("counter", tubes="cold"),
            Stream("agua geotérmica", 160.0 + _K0, 2.0, cp_J_per_kgK=4310.0),
            Stream("agua", 20.0 + _K0, 1.2, cp_J_per_kgK=4180.0),
            640.0,
            "T_cold_out",
            80.0 + _K0,
            tube_D_m=0.015,
        ),
        "El libro: ΔT_ml = 91,9 °C, A = 5,11 m² y L = 109 m.",
    ),
    "Glicerina calentada con agua: 2 cascos y 4 pasos (basado en Cengel y Ghajar 11-5)": (
        ExchangerExample(
            SizingInputs(
                Arrangement("shell", 2, tubes="hot"),
                # C_agua/C_glicerina = 30/40, como las temperaturas del libro
                Stream("agua", 80.0 + _K0, 45.8 / 4187.0, cp_J_per_kgK=4187.0),
                Stream("glicerina", 20.0 + _K0, 45.8 / 0.75 / 2410.0, cp_J_per_kgK=2410.0),
                1.0 / (1.0 / 160.0 + 1.0 / 25.0),
                "T_cold_out",
                50.0 + _K0,
                tube_D_m=0.02,
            ),
            "Caudales elegidos para que el agua salga a 40 °C como en el libro: A = 3,77 m² "
            "(60 m de tubo de 2 cm) con F = 0,91. Con un solo paso de casco no se llega.",
        )
    ),
    "Enfriador de aceite de doble tubo (basado en Incropera 11.1)": ExchangerExample(
        SizingInputs(
            Arrangement("counter", tubes="cold"),
            Stream("aceite", 100.0 + _K0, 0.1, cp_J_per_kgK=2131.0),
            Stream("agua", 30.0 + _K0, 0.2, cp_J_per_kgK=4178.0),
            38.1,
            "T_hot_out",
            60.0 + _K0,
            tube_D_m=0.025,
        ),
        "El aceite tiene que salir a 60 °C; U = 38,1 W/(m²·K) lo fija el lado del aceite.",
    ),
    "Evaporador de amoníaco que enfría agua (con CoolProp)": ExchangerExample(
        SizingInputs(
            Arrangement("shell", 1, tubes="hot"),
            Stream("agua", 12.0 + _K0, 2.0, fluid="Water", p_Pa=3.0e5),
            Stream("amoníaco", 2.0 + _K0, fluid="Ammonia", phase_change=True),
            900.0,
            "T_hot_out",
            7.0 + _K0,
        ),
        "El amoníaco evapora a 2 °C (C_r = 0): el caudal evaporado sale de Q̇/h_fg.",
    ),
}

TEST_EXAMPLES: dict[str, ExchangerExample] = {
    "Radiador de un auto: el U de un ensayo (Cengel y Ghajar 11-6)": ExchangerExample(
        FourTemperatureInputs(
            Arrangement("cross_unmixed", tubes="hot"),
            Stream("agua", 90.0 + _K0, 0.6, cp_J_per_kgK=4195.0),
            Stream("aire", 20.0 + _K0, cp_J_per_kgK=1005.0),
            65.0 + _K0,
            40.0 + _K0,
            40 * math.pi * 0.005 * 0.65,
            None,
            "hot",
        ),
        "El libro lee F = 0,97 en el gráfico (la serie exacta da 0,970).",
    ),
    "Condensador de una central (Cengel y Ghajar 11-3)": ExchangerExample(
        FourTemperatureInputs(
            Arrangement("shell", 1, tubes="cold"),
            Stream("vapor", 30.0 + _K0, phase_change=True, h_fg_J_per_kg=2431.0e3),
            Stream("agua de enfriamiento", 14.0 + _K0, cp_J_per_kgK=4184.0),
            30.0 + _K0,
            22.0 + _K0,
            45.0,
            2100.0,
        ),
        "El libro: ΔT_ml = 11,5 °C, Q̇ = 1,09 MW, 32,5 kg/s de agua y 0,45 kg/s de vapor.",
    ),
    "Glicerina con agua: 2 cascos y 4 pasos (Cengel y Ghajar 11-5)": ExchangerExample(
        FourTemperatureInputs(
            Arrangement("shell", 2, tubes="hot"),
            Stream("agua", 80.0 + _K0, cp_J_per_kgK=4187.0),
            Stream("glicerina", 20.0 + _K0, cp_J_per_kgK=2410.0),
            40.0 + _K0,
            50.0 + _K0,
            math.pi * 0.02 * 60.0,
            1.0 / (1.0 / 160.0 + 1.0 / 25.0),
        ),
        "El libro: P = 0,67, R = 0,75, F = 0,91 y Q̇ = 1,83 kW (1,81 kW con R''_f = 0,0006).",
    ),
}

U_EXAMPLES: dict[str, ExchangerExample] = {
    "Doble tubo de acero inoxidable con ensuciamiento (Cengel y Ghajar 11-2)": ExchangerExample(
        OverallUInputs(
            "tube",
            800.0,
            1200.0,
            15.1,
            D_i_m=0.015,
            D_o_m=0.019,
            L_m=1.0,
            R_f_i_m2K_per_W=0.0004,
            R_f_o_m2K_per_W=0.0001,
        ),
        "El libro: R = 0,0532 °C/W por metro, U_i = 399 y U_o = 315 W/(m²·K).",
    ),
    "Agua y aire en un tubo de cobre: el lado del gas manda": ExchangerExample(
        OverallUInputs("tube", 4000.0, 60.0, 385.0, D_i_m=0.02, D_o_m=0.022, L_m=1.0),
        "Con h del aire 60 veces menor, U ≈ h_aire: las aletas van de ese lado.",
    ),
    "Placa de un intercambiador de placas (acero inoxidable, 0,6 mm)": ExchangerExample(
        OverallUInputs(
            "plane",
            4500.0,
            3500.0,
            16.0,
            thickness_m=0.0006,
            A_m2=1.0,
            R_f_i_m2K_per_W=0.0001,
            R_f_o_m2K_per_W=0.0001,
        ),
        "Placas delgadas y h altos de los dos lados: el ensuciamiento pesa mucho.",
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def exchanger_to_dict(result: ExchangerResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""

    def T(x: float) -> str:
        return format_quantity(x, "temperature", system)

    def stream(sr: StreamResult) -> dict[str, Any]:
        d: dict[str, Any] = {
            "nombre": sr.name,
            "T_entrada": T(sr.T_in_K),
            "T_salida": T(sr.T_out_K),
            "caudal": format_quantity(sr.m_dot_kg_s, "mass_flow", system),
        }
        if sr.phase_change:
            d["cambia_de_fase"] = True
            if sr.h_fg_J_per_kg is not None:
                d["h_fg"] = format_quantity(sr.h_fg_J_per_kg, "specific_enthalpy", system)
        else:
            assert sr.cp_J_per_kgK is not None
            d["cp"] = format_quantity(sr.cp_J_per_kgK, "specific_heat", system)
            d["C"] = format_quantity(sr.C_W_per_K, "heat_capacity_rate", system)
        return d

    r = result
    data: dict[str, Any] = {
        "problema": {"rating": "verificación", "sizing": "dimensionamiento", "test": "ensayo"}[
            r.mode
        ],
        "tipo": r.arrangement.name,
        "caliente": stream(r.hot),
        "frio": stream(r.cold),
        "Q": format_quantity(r.Q_W, "heat_rate", system),
        "Q_max": format_quantity(r.Q_max_W, "heat_rate", system),
        "efectividad": r.effectiveness,
        "NTU": r.NTU,
        "C_r": r.C_r,
        "UA": format_quantity(r.UA_W_per_K, "heat_capacity_rate", system),
        "U": format_quantity(r.U_W_per_m2K, "heat_transfer_coefficient", system),
        "A": format_quantity(r.A_m2, "area", system),
        "F": r.F,
        "dT_ml": format_quantity(r.dT_lm_K, "temperature_difference", system),
        "exergia": {
            "T0": T(r.exergy.T0_K),
            "S_gen": format_quantity(r.exergy.S_gen_W_per_K, "entropy_rate", system),
            "X_dest": format_quantity(r.exergy.X_dest_W, "heat_rate", system),
            "eta_ex": r.exergy.efficiency,
        },
    }
    if r.P_R is not None:
        data["P"], data["R"] = r.P_R
    if r.tube_length_m is not None:
        data["largo_del_tubo"] = format_quantity(r.tube_length_m, "length", system)
    return data


def overall_u_to_dict(result: OverallUResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado del U → dict serializable."""
    return {
        "geometria": "tubo" if result.inputs.geometry == "tube" else "pared plana",
        "resistencias": [
            {
                "elemento": r.label,
                "R": format_quantity(r.R_K_per_W, "thermal_resistance", system),
                "fraccion": result.share(r),
            }
            for r in result.resistances
        ],
        "R_total": format_quantity(result.R_total_K_per_W, "thermal_resistance", system),
        "UA": format_quantity(result.UA_W_per_K, "heat_capacity_rate", system),
        "U_i": format_quantity(result.U_i_W_per_m2K, "heat_transfer_coefficient", system),
        "U_o": format_quantity(result.U_o_W_per_m2K, "heat_transfer_coefficient", system),
    }
