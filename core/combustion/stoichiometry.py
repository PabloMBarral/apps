"""Estequiometría de la combustión: aire, exceso, productos y humos — Fase 5.

Sigue el vademecum §16.1–16.5:

- **Aire técnico** (§16.1): 21 % O₂ y 79 % N₂ (3,762 moles de N₂ por mol de
  O₂, M = 28,85 kg/kmol); también aire seco (con Ar y CO₂, el de las tablas de
  Cengel) u oxígeno puro, y con humedad (φ a la T del aire, con
  :mod:`core.psychrometrics`, como Cengel 15-3).
- **Oxígeno teórico** por unidad de combustible: n_C + n_H/4 + n_S − n_O/2,
  y el aire estequiométrico a_s = O₂,teórico / y_O₂ (§16.2).
- **Exceso**: λ = a/a_s = 1 + e = 1/φ (§16.2); también el % de aire teórico,
  la relación aire–combustible AC = λ·AC_s (§16.4) o el O₂ medido en los
  humos secos.
- **Productos** (§16.2, reacción real): con λ ≥ 1, combustión completa o con
  una fracción del carbono a CO (Cengel 15-6); con λ < 1, el hidrógeno se
  quema a H₂O y el carbono reparte el O₂ que queda entre CO₂ y CO (Cengel
  15-8 c).
- **Humos** (§16.3–16.5): fracciones en base húmeda y seca, M_g, GC = 1 + AC,
  volumen normal, punto de rocío del agua y el **ácido** (Verhoff & Banchero,
  1974) y el análisis inverso: λ desde el O₂ medido o desde un **Orsat**
  (Cengel 15-4).

Las cantidades van por unidad de combustible: por mol (combustibles con
especies) o por kg (análisis elemental). Todo en SI. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from scipy.optimize import brentq

from core.combustion.fuels import Fuel
from core.combustion.thermo import R_U, species
from core.psychrometrics import saturation_pressure, saturation_temperature

__all__ = [
    "AIR_SPEC_KINDS",
    "OXIDIZER_KINDS",
    "PRODUCT_SPECIES",
    "AirSpec",
    "AirSpecKind",
    "OrsatResult",
    "Oxidizer",
    "OxidizerKind",
    "Stoichiometry",
    "acid_dew_point",
    "dry_gas_curves",
    "lambda_from_o2",
    "orsat_analysis",
    "products",
    "solve_stoichiometry",
]

#: Especies de los productos (gases).
PRODUCT_SPECIES: tuple[str, ...] = ("CO2", "CO", "H2O", "H2", "SO2", "O2", "N2", "Ar")

OxidizerKind = Literal["technical", "dry", "oxygen"]
OXIDIZER_KINDS: dict[OxidizerKind, str] = {
    "technical": "Aire técnico (21 % O₂ y 79 % N₂)",
    "dry": "Aire seco (con Ar y CO₂)",
    "oxygen": "Oxígeno puro",
}
_DRY_COMPOSITION: dict[OxidizerKind, dict[str, float]] = {
    "technical": {"O2": 0.21, "N2": 0.79},
    # El aire seco de core.ideal_gas.AIR_DRY (tablas de Cengel).
    "dry": {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004},
    "oxygen": {"O2": 1.0},
}

AirSpecKind = Literal["lambda", "excess", "theoretical", "equivalence", "AC", "o2_dry"]
AIR_SPEC_KINDS: dict[AirSpecKind, str] = {
    "lambda": "λ (relación de aire)",
    "excess": "Exceso de aire e",
    "theoretical": "Aire teórico",
    "equivalence": "Relación de equivalencia φ",
    "AC": "Relación aire–combustible AC",
    "o2_dry": "O₂ medido en humos secos",
}

#: Condiciones normales del volumen de los humos (0 °C y 1 atm).
T_NORMAL_K = 273.15
P_NORMAL_PA = 101_325.0

# λ máximo que se acepta (más aire no tiene sentido en un hogar).
_LAMBDA_MAX = 20.0


def _pct(x: float) -> str:
    return f"{100 * x:.4g} %"


# ---------------------------------------------------------------------
# Comburente
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Oxidizer:
    """Aire (técnico o seco) u oxígeno, a T y con humedad relativa φ."""

    kind: OxidizerKind = "technical"
    T_K: float = 298.15
    phi: float = 0.0

    @property
    def label(self) -> str:
        return OXIDIZER_KINDS[self.kind]

    def check(self, p_Pa: float) -> None:
        if self.kind not in _DRY_COMPOSITION:
            raise ValueError(f"Comburente desconocido: {self.kind!r}.")
        if not (math.isfinite(self.T_K) and 223.15 <= self.T_K <= 1573.15):
            raise ValueError(
                "La temperatura del comburente tiene que estar entre −50 °C y 1300 °C "
                "(precalentado)."
            )
        if not (math.isfinite(self.phi) and 0.0 <= self.phi <= 1.0):
            raise ValueError("La humedad relativa del aire tiene que estar entre 0 y 100 %.")
        if self.phi > 0.0 and self.kind != "oxygen":
            p_v = self.phi * saturation_pressure(self.T_K) if self.T_K < 647.0 else math.inf
            if p_v >= 0.5 * p_Pa:
                raise ValueError(
                    f"Con φ = {_pct(self.phi)} a {self.T_K - 273.15:.0f} °C el vapor tendría "
                    "más de la mitad de la presión: no es aire húmedo. Bajá la humedad o la "
                    "temperatura."
                )

    def vapor_fraction(self, p_Pa: float) -> float:
        """Fracción molar del vapor de agua: y_v = φ·p_vs(T)/p (vademecum §14.1)."""
        if self.phi <= 0.0 or self.kind == "oxygen":
            return 0.0
        return self.phi * saturation_pressure(self.T_K) / p_Pa

    def composition(self, p_Pa: float) -> dict[str, float]:
        """Fracciones molares del comburente, con el vapor de agua."""
        y_v = self.vapor_fraction(p_Pa)
        out = {k: v * (1.0 - y_v) for k, v in _DRY_COMPOSITION[self.kind].items()}
        if y_v > 0.0:
            out["H2O"] = y_v
        return out

    @property
    def dry_composition(self) -> dict[str, float]:
        return dict(_DRY_COMPOSITION[self.kind])


def _molar_mass(fractions: Mapping[str, float]) -> float:
    return sum(y * species(k).M_kg_per_mol for k, y in fractions.items())


# ---------------------------------------------------------------------
# Productos
# ---------------------------------------------------------------------


def _soot_lambda(fuel: Fuel) -> float:
    """λ mínimo de la regla de Cengel: el O₂ justo para el H → H₂O y el C → CO."""
    e = fuel.elements()
    need = e["C"] / 2.0 + e["H"] / 4.0 + e["S"] - e["O"] / 2.0
    return need / fuel.theoretical_oxygen()


def products(
    fuel: Fuel, air: Mapping[str, float], air_moles: float, co_fraction: float = 0.0
) -> tuple[dict[str, float], bool]:
    """Moles de cada producto por unidad de combustible y si hubo defecto de aire.

    Con oxígeno de sobra (λ ≥ 1): el C pasa a CO₂, salvo la fracción
    ``co_fraction`` que sale como CO; el H a H₂O, el S a SO₂ y el N a N₂ (el
    vademecum §16.2 desprecia el SO₃ y los NOx). Con defecto (λ < 1), como
    Cengel 15-8 c: el H y el S se queman primero y el C reparte el O₂ que queda
    entre CO₂ y CO; si ni así alcanza, un error (se formaría hollín).

    Raises
    ------
    ValueError
        Si con defecto de aire el O₂ no alcanza ni para pasar el C a CO.
    """
    e = fuel.elements()
    out = dict.fromkeys(PRODUCT_SPECIES, 0.0)
    for k, y in air.items():
        out[k] = out.get(k, 0.0) + air_moles * y
    o2_in = out["O2"]
    out["O2"] = 0.0
    out["H2O"] += fuel.water_formed()
    out["SO2"] += e["S"]
    out["N2"] += e["N"] / 2.0
    o2_th = fuel.theoretical_oxygen()
    rich = o2_in < o2_th * (1.0 - 1e-12)
    if not rich:
        co = co_fraction * e["C"]
        out["CO"] += co
        out["CO2"] += e["C"] - co
        out["O2"] = o2_in - (o2_th - co / 2.0)
        return {k: v for k, v in out.items() if v > 0.0}, False
    # Defecto de aire: H → H₂O y S → SO₂ primero; el C reparte el resto.
    o2_left = o2_in + e["O"] / 2.0 - e["S"] - e["H"] / 4.0
    if e["C"] <= 0.0:
        if o2_left < 0.0:  # combustible sin carbono (H₂): sobra hidrógeno
            burnt = e["H"] / 2.0 + 2.0 * o2_left
            out["H2O"] += burnt - e["H"] / 2.0
            out["H2"] += e["H"] / 2.0 - burnt
        return {k: v for k, v in out.items() if v > 0.0}, True
    co2 = 2.0 * o2_left - e["C"]
    if o2_left < e["C"] / 2.0 * (1.0 - 1e-12):
        lam = o2_in / o2_th
        raise ValueError(
            f"Con λ = {lam:.3g} el oxígeno no alcanza ni para pasar todo el carbono a CO: se "
            f"formaría hollín. Con este modelo (Cengel 15-8 c) λ tiene que ser al menos "
            f"{_soot_lambda(fuel):.3g}."
        )
    co2 = max(0.0, min(co2, e["C"]))
    out["CO2"] += co2
    out["CO"] += e["C"] - co2
    return {k: v for k, v in out.items() if v > 0.0}, True


def _dry(moles: Mapping[str, float]) -> dict[str, float]:
    return {k: v for k, v in moles.items() if k != "H2O" and v > 0.0}


def _fractions(moles: Mapping[str, float]) -> dict[str, float]:
    total = sum(moles.values())
    return {k: v / total for k, v in moles.items()} if total > 0.0 else {}


# ---------------------------------------------------------------------
# Rocíos
# ---------------------------------------------------------------------


def acid_dew_point(p_H2O_Pa: float, p_SO3_Pa: float) -> float:
    """Punto de rocío ácido (H₂SO₄) de los humos, K.

    Verhoff, F. H. & Banchero, J. T. (1974), «Predicting dew points of flue
    gases», *Chemical Engineering Progress* 70(8), 71–72:
    1000/T = 2,276 − 0,0294·ln p_H₂O − 0,0858·ln p_SO₃ + 0,0062·ln p_H₂O·ln p_SO₃,
    con las presiones parciales en mmHg (correlación de datos de H₂SO₄–agua).
    """
    if p_H2O_Pa <= 0.0 or p_SO3_Pa <= 0.0:
        raise ValueError("Hace falta vapor de agua y SO₃ en los humos para el rocío ácido.")
    mmhg = 101_325.0 / 760.0
    ln_w = math.log(p_H2O_Pa / mmhg)
    ln_s = math.log(p_SO3_Pa / mmhg)
    return 1000.0 / (2.276 - 0.0294 * ln_w - 0.0858 * ln_s + 0.0062 * ln_w * ln_s)


# ---------------------------------------------------------------------
# Estequiometría completa
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class AirSpec:
    """Cómo se da la cantidad de comburente (todo como fracción, no en %).

    ``lambda``: λ; ``excess``: e (0,2 = 20 %); ``theoretical``: aire teórico
    (1,2 = 120 %); ``equivalence``: φ = 1/λ; ``AC``: kg de aire seco por kg de
    combustible; ``o2_dry``: O₂ en los humos secos (0,03 = 3 %).
    """

    kind: AirSpecKind = "lambda"
    value: float = 1.0


@dataclass(frozen=True)
class Stoichiometry:
    """La estequiometría resuelta, por unidad de combustible (mol o kg)."""

    fuel: Fuel
    oxidizer: Oxidizer
    p_Pa: float
    air: dict[str, float]
    lam: float
    co_fraction: float
    o2_theoretical: float
    air_moles: float
    products: dict[str, float] = field(default_factory=dict)
    rich: bool = False

    # --- aire ---------------------------------------------------------------
    @property
    def excess(self) -> float:
        """Exceso de aire e = λ − 1 (vademecum §16.2)."""
        return self.lam - 1.0

    @property
    def equivalence_ratio(self) -> float:
        """φ = 1/λ."""
        return 1.0 / self.lam

    @property
    def air_theoretical(self) -> float:
        """Moles de comburente estequiométrico (con su humedad) por unidad: a_s."""
        return self.o2_theoretical / self.air["O2"]

    @property
    def dry_air_moles(self) -> float:
        return self.air_moles * (1.0 - self.air.get("H2O", 0.0))

    @property
    def M_dry_air(self) -> float:
        return _molar_mass(self.oxidizer.dry_composition)

    @property
    def AC_s(self) -> float:
        """Relación aire–combustible estequiométrica, aire seco (kg/kg; vademecum §16.4)."""
        dry = self.air_theoretical * (1.0 - self.air.get("H2O", 0.0))
        return dry * self.M_dry_air / self.fuel.mass_per_basis

    @property
    def AC(self) -> float:
        """Relación aire–combustible real, aire seco (kg/kg): λ·AC_s."""
        return self.dry_air_moles * self.M_dry_air / self.fuel.mass_per_basis

    @property
    def air_vapor_moles(self) -> float:
        return self.air_moles * self.air.get("H2O", 0.0)

    # --- humos ------------------------------------------------------------
    @property
    def n_gas(self) -> float:
        return sum(self.products.values())

    @property
    def dry_products(self) -> dict[str, float]:
        return _dry(self.products)

    @property
    def n_dry(self) -> float:
        return sum(self.dry_products.values())

    @property
    def wet_fractions(self) -> dict[str, float]:
        return _fractions(self.products)

    @property
    def dry_fractions(self) -> dict[str, float]:
        return _fractions(self.dry_products)

    @property
    def mass_fractions(self) -> dict[str, float]:
        masses = {k: n * species(k).M_kg_per_mol for k, n in self.products.items()}
        total = sum(masses.values())
        return {k: m / total for k, m in masses.items()}

    @property
    def M_gas(self) -> float:
        """Masa molar de los humos húmedos, M_g = Σ y_i·M_i (vademecum §16.3)."""
        return _molar_mass(self.wet_fractions)

    @property
    def M_dry_gas(self) -> float:
        return _molar_mass(self.dry_fractions)

    @property
    def gas_mass(self) -> float:
        """Masa de humos por unidad de combustible (kg)."""
        return self.n_gas * self.M_gas

    @property
    def GC(self) -> float:
        """Relación gases–combustible (kg/kg): masa de humos húmedos por kg (vademecum §16.5)."""
        return self.gas_mass / self.fuel.mass_per_basis

    @property
    def GC_dry(self) -> float:
        return self.n_dry * self.M_dry_gas / self.fuel.mass_per_basis

    @property
    def normal_volume(self) -> float:
        """Volumen normal de humos húmedos (m³ a 0 °C y 1 atm por kg de combustible)."""
        return self.n_gas * R_U * T_NORMAL_K / P_NORMAL_PA / self.fuel.mass_per_basis

    @property
    def normal_volume_dry(self) -> float:
        return self.n_dry * R_U * T_NORMAL_K / P_NORMAL_PA / self.fuel.mass_per_basis

    @property
    def p_H2O_Pa(self) -> float:
        return self.wet_fractions.get("H2O", 0.0) * self.p_Pa

    @property
    def dew_point_K(self) -> float | None:
        """Rocío del agua de los humos: T_sat(p_H₂O) (Cengel 15-2). ``None`` sin agua."""
        p_w = self.p_H2O_Pa
        if p_w <= 1.0:
            return None
        return saturation_temperature(p_w)

    def acid_dew_point_K(self, so3_conversion: float) -> float | None:
        """Rocío ácido con una fracción del SO₂ pasada a SO₃ (``None`` sin azufre o sin agua)."""
        y = self.wet_fractions
        if y.get("SO2", 0.0) <= 0.0 or y.get("H2O", 0.0) <= 0.0 or so3_conversion <= 0.0:
            return None
        return acid_dew_point(y["H2O"] * self.p_Pa, so3_conversion * y["SO2"] * self.p_Pa)

    @property
    def co2_max(self) -> float:
        """CO₂ en humos secos con λ = 1 y combustión completa (el máximo posible)."""
        stoich, _ = products(self.fuel, self.air, self.air_theoretical)
        return _fractions(_dry(stoich)).get("CO2", 0.0)


def _air_moles_for_lambda(fuel: Fuel, air: Mapping[str, float], lam: float) -> float:
    return lam * fuel.theoretical_oxygen() / air["O2"]


def lambda_from_o2(
    fuel: Fuel,
    air: Mapping[str, float],
    o2_dry: float,
    co_fraction: float = 0.0,
) -> float:
    """λ que da ``o2_dry`` de O₂ en los humos secos (con λ ≥ 1).

    Raises
    ------
    ValueError
        Si ese O₂ no se puede alcanzar (más que el del aire, o menos que el
        que deja el CO con λ = 1).
    """

    def y_o2(lam: float) -> float:
        prods, _ = products(fuel, air, _air_moles_for_lambda(fuel, air, lam), co_fraction)
        return _fractions(_dry(prods)).get("O2", 0.0)

    y_air = air["O2"] / (1.0 - air.get("H2O", 0.0))
    if not (math.isfinite(o2_dry) and 0.0 <= o2_dry < y_air):
        raise ValueError(
            f"El O₂ de los humos secos tiene que estar entre 0 y {_pct(y_air)} (el O₂ del "
            "comburente seco: con más λ los humos se parecen al aire)."
        )
    low, high = y_o2(1.0), y_o2(_LAMBDA_MAX)
    if o2_dry < low:
        raise ValueError(
            f"Con esa fracción de CO los humos secos tienen al menos {_pct(low)} de O₂ con "
            "λ = 1: no se puede medir menos."
        )
    if o2_dry > high:
        raise ValueError(
            f"Ese O₂ corresponde a λ > {_LAMBDA_MAX:g}: un exceso de aire fuera de rango."
        )
    if o2_dry == low:
        return 1.0
    return float(brentq(lambda lam: y_o2(lam) - o2_dry, 1.0, _LAMBDA_MAX, xtol=1e-13))


def solve_stoichiometry(
    fuel: Fuel,
    oxidizer: Oxidizer,
    spec: AirSpec,
    *,
    p_Pa: float = 101_325.0,
    co_fraction: float = 0.0,
) -> Stoichiometry:
    """Aire, exceso y productos (vademecum §16.2–16.5).

    Raises
    ------
    ValueError
        Con mensajes para el alumno: λ fuera de rango, O₂ imposible, fracción
        de CO fuera de [0, 1), defecto de aire que formaría hollín.
    """
    if not (math.isfinite(p_Pa) and 0.1e5 <= p_Pa <= 100e5):
        raise ValueError("La presión tiene que estar entre 0,1 y 100 bar.")
    if not (math.isfinite(co_fraction) and 0.0 <= co_fraction < 1.0):
        raise ValueError("La fracción del carbono que sale como CO tiene que estar entre 0 y 1.")
    oxidizer.check(p_Pa)
    air = oxidizer.composition(p_Pa)
    o2_th = fuel.theoretical_oxygen()
    value = spec.value
    if not math.isfinite(value):
        raise ValueError("La cantidad de aire tiene que ser un número.")
    if spec.kind == "lambda":
        lam = value
    elif spec.kind == "excess":
        lam = 1.0 + value
    elif spec.kind == "theoretical":
        lam = value
    elif spec.kind == "equivalence":
        if value <= 0.0:
            raise ValueError("La relación de equivalencia φ tiene que ser positiva.")
        lam = 1.0 / value
    elif spec.kind == "AC":
        dry_air_s = o2_th / air["O2"] * (1.0 - air.get("H2O", 0.0))
        ac_s = dry_air_s * _molar_mass(oxidizer.dry_composition) / fuel.mass_per_basis
        lam = value / ac_s
    elif spec.kind == "o2_dry":
        lam = lambda_from_o2(fuel, air, value, co_fraction)
    else:
        raise ValueError(f"Dato de aire desconocido: {spec.kind!r}.")
    if not (0.3 <= lam <= _LAMBDA_MAX):
        raise ValueError(
            f"λ = {lam:.4g} queda fuera de rango (entre 0,3 y {_LAMBDA_MAX:g}): con menos aire "
            "la llama no se sostiene y con más no tiene sentido."
        )
    air_moles = _air_moles_for_lambda(fuel, air, lam)
    prods, rich = products(fuel, air, air_moles, co_fraction)
    return Stoichiometry(
        fuel=fuel,
        oxidizer=oxidizer,
        p_Pa=p_Pa,
        air=air,
        lam=lam,
        co_fraction=0.0 if rich else co_fraction,
        o2_theoretical=o2_th,
        air_moles=air_moles,
        products=prods,
        rich=rich,
    )


def dry_gas_curves(
    fuel: Fuel, oxidizer: Oxidizer, lambdas: Sequence[float], *, p_Pa: float = 101_325.0
) -> dict[str, list[float]]:
    """CO₂, O₂ y CO en humos secos en función de λ: el diagrama de combustión.

    Con λ ≥ 1 combustión completa; con λ < 1 la regla de Cengel (CO). Los
    puntos que formarían hollín se omiten.
    """
    air = oxidizer.composition(p_Pa)
    out: dict[str, list[float]] = {"lambda": [], "CO2": [], "O2": [], "CO": []}
    for lam in lambdas:
        try:
            prods, _ = products(fuel, air, _air_moles_for_lambda(fuel, air, lam))
        except ValueError:
            continue
        y = _fractions(_dry(prods))
        out["lambda"].append(float(lam))
        for k in ("CO2", "O2", "CO"):
            out[k].append(y.get(k, 0.0))
    return out


# ---------------------------------------------------------------------
# Análisis inverso: Orsat (Cengel 15-4)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class OrsatResult:
    """El análisis de humos secos resuelto (cada 100 mol de humos secos)."""

    fuel: Fuel
    air: dict[str, float]
    measured: dict[str, float]
    fuel_per_100: float
    air_per_100: float
    water_per_100: float
    oxygen_residual: float
    lam: float

    @property
    def products_per_fuel(self) -> dict[str, float]:
        """Productos por unidad de combustible (los medidos más el agua)."""
        out = {k: 100.0 * v / self.fuel_per_100 for k, v in self.measured.items() if v > 0}
        out["H2O"] = self.water_per_100 / self.fuel_per_100
        return out

    @property
    def oxygen_residual_relative(self) -> float:
        """Residuo del balance de O sobre el O que sale (0 = medición coherente)."""
        m = self.measured
        o_out = 100.0 * (2 * m.get("CO2", 0.0) + m.get("CO", 0.0) + 2 * m.get("O2", 0.0))
        o_out += self.water_per_100
        return self.oxygen_residual / o_out


def orsat_analysis(
    fuel: Fuel, oxidizer: Oxidizer, co2: float, co: float, o2: float, *, p_Pa: float = 101_325.0
) -> OrsatResult:
    """λ desde los humos secos medidos (CO₂, CO y O₂; el N₂ por diferencia).

    Cengel 15-4: el balance de C da el combustible cada 100 mol de humos
    secos; el de N₂ (con el Ar, que el Orsat cuenta como N₂), el aire; el de H,
    el agua; y el de O queda para verificar la medición. El CO₂ del Orsat
    incluye el SO₂ (los dos se absorben en la misma pipeta).

    Raises
    ------
    ValueError
        Si las fracciones no tienen sentido o el combustible no tiene carbono.
    """
    values = {"CO2": co2, "CO": co, "O2": o2}
    if any(not math.isfinite(v) or v < 0.0 for v in values.values()):
        raise ValueError("Las fracciones medidas no pueden ser negativas.")
    n2 = 1.0 - co2 - co - o2
    if n2 <= 0.0:
        raise ValueError("CO₂ + CO + O₂ no pueden sumar 100 %: falta el N₂ (por diferencia).")
    if co2 + co <= 0.0:
        raise ValueError("Sin CO₂ ni CO en los humos no se puede hacer el balance de carbono.")
    oxidizer.check(p_Pa)
    if oxidizer.kind == "oxygen":
        raise ValueError("Con oxígeno puro no hay N₂ en los humos: el Orsat no aplica.")
    air = oxidizer.composition(p_Pa)
    e = fuel.elements()
    if e["C"] <= 0.0:
        raise ValueError("El combustible no tiene carbono: el Orsat necesita CO₂ en los humos.")
    # Dos balances lineales (cada 100 mol de humos secos) en x (combustible) y a (aire):
    #   C (+S, que el Orsat cuenta con el CO₂): (n_C + n_S)·x + y_CO₂,a·a = CO₂ + CO
    #   N₂ (+Ar):                                (n_N/2)·x + (y_N₂,a + y_Ar,a)·a = N₂
    a11, a12, b1 = e["C"] + e["S"], air.get("CO2", 0.0), 100.0 * (co2 + co)
    a21, a22, b2 = e["N"] / 2.0, air.get("N2", 0.0) + air.get("Ar", 0.0), 100.0 * n2
    det = a11 * a22 - a12 * a21
    x = (b1 * a22 - a12 * b2) / det
    a = (a11 * b2 - a21 * b1) / det
    if x <= 0.0 or a <= 0.0:
        raise ValueError("Con esos humos los balances de C y N₂ no cierran: revisá la medición.")
    water = fuel.water_formed() * x + a * air.get("H2O", 0.0)
    o_in = e["O"] * x + fuel.moisture_mol * x
    o_in += a * (2 * air["O2"] + 2 * air.get("CO2", 0.0) + air.get("H2O", 0.0))
    o_out = 100.0 * (2 * co2 + co + 2 * o2) + water
    lam = a * air["O2"] / (fuel.theoretical_oxygen() * x)
    return OrsatResult(
        fuel=fuel,
        air=air,
        measured={"CO2": co2, "CO": co, "O2": o2, "N2": n2},
        fuel_per_100=x,
        air_per_100=a,
        water_per_100=water,
        oxygen_residual=o_in - o_out,
        lam=lam,
    )
