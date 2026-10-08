"""Combustión: estequiometría, llama adiabática, calor y segundo principio — Fase 5.

Junta las piezas de :mod:`core.combustion` en lo que resuelve la página
``/Combustion``:

- **Estequiometría y humos** (:mod:`core.combustion.stoichiometry`, vademecum
  §16.1–16.5).
- **Primer principio** (vademecum §16.7): con la entalpía estándar
  h = h_f + Δh (§16.6) de cada reactivo y producto,
  q − w = Σ ν_p·h_p(T_p) − Σ ν_r·h_r(T_r); a volumen constante (la bomba,
  Cengel 15-7) con u = h − R·T para los gases.
- **Temperatura adiabática de llama** (§16.8): Σ ν_p·h_p(T_ad) = Σ ν_r·h_r,
  con combustión completa (como el vademecum y Cengel 15-8) y con la
  disociación de los productos (equilibrio químico,
  :mod:`core.combustion.equilibrium`), verificada con las constantes de
  equilibrio K_p (Cengel §16-2).
- **Calor con los humos a una temperatura dada** (Cengel 15-6 y 15-11): si
  los humos se enfrían por debajo del punto de rocío condensa una parte del
  agua; el rendimiento sobre PCI y PCS y el reparto de la energía (calor
  útil, humos, inquemados, latente recuperado).
- **Segundo principio** (vademecum §16.11 y §16.13; Cengel 15-10 y 15-11):
  S_gen y X_dest = T₀·S_gen de la combustión adiabática y con el calor
  entregado a T_b; la exergía del combustible ≈ PCI.
- **Análisis de humos**: λ desde el O₂ medido (y el CO) o desde un Orsat
  (Cengel 15-4).

Todo en SI y por unidad de combustible (mol o kg). No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from scipy.optimize import brentq

from core.combustion.equilibrium import adiabatic_equilibrium
from core.combustion.fuels import FUELS, Fuel, component_entropy
from core.combustion.stoichiometry import (
    AirSpec,
    OrsatResult,
    Oxidizer,
    Stoichiometry,
    lambda_from_o2,
    orsat_analysis,
    solve_stoichiometry,
)
from core.combustion.thermo import (
    P_STANDARD_PA,
    R_U,
    T_MAX_K,
    T_REF_K,
    gas_mixture_entropy,
    mixture_enthalpy,
    species,
)
from core.psychrometrics import saturation_pressure
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = [
    "COMBUSTION_EXAMPLES",
    "COMBUSTION_EXAMPLE_NOTES",
    "FLUE_GAS_EXAMPLES",
    "FLUE_GAS_EXAMPLE_NOTES",
    "CombustionInputs",
    "CombustionResult",
    "EnergySplit",
    "FlameResult",
    "FlueGasInputs",
    "FlueGasResult",
    "HeatResult",
    "KpCheck",
    "ProcessKind",
    "SecondLaw",
    "air_temperature_sweep",
    "combustion_notes",
    "combustion_to_dict",
    "default_air_temperature_values",
    "default_lambda_values",
    "default_products_temperature_values",
    "flue_gas_notes",
    "flue_gas_to_dict",
    "lambda_sweep",
    "products_temperature_sweep",
    "solve_combustion",
    "solve_flue_gas",
]

ProcessKind = Literal["p", "v"]
#: Temperatura más baja de la búsqueda de la llama (los polinomios llegan a 200 K).
_T_LOW_K = 250.0
#: Calor que libera cada inquemado al quemarse (J/mol): CO → CO₂ y H₂ → H₂O(g).
_UNBURNED = ("CO", "H2")


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _pct(x: float) -> str:
    return f"{100 * x:.3g} %"


# ---------------------------------------------------------------------
# Datos y resultados
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CombustionInputs:
    """Datos de una combustión (todo en SI).

    ``T_products_K``: la temperatura a la que salen los humos (para el calor
    y el rendimiento); ``None`` calcula solo la llama. ``T_sink_K``: la
    temperatura a la que se entrega ese calor (para la exergía; ``None`` = T₀).
    ``so3_conversion``: la fracción del SO₂ que pasa a SO₃ (rocío ácido).
    """

    fuel: Fuel
    oxidizer: Oxidizer = field(default_factory=Oxidizer)
    air: AirSpec = field(default_factory=AirSpec)
    co_fraction: float = 0.0
    p_Pa: float = 101_325.0
    T_fuel_K: float = T_REF_K
    process: ProcessKind = "p"
    T_products_K: float | None = None
    T_sink_K: float | None = None
    T0_K: float = T_REF_K
    m_fuel_kg_s: float | None = None
    so3_conversion: float = 0.02
    equilibrium: bool = True


@dataclass(frozen=True)
class FlameResult:
    """La llama adiabática: temperatura, composición y presión final."""

    T_K: float
    products: dict[str, float]
    p_Pa: float
    method: Literal["complete", "equilibrium"]

    @property
    def fractions(self) -> dict[str, float]:
        total = sum(self.products.values())
        return {k: v / total for k, v in self.products.items() if v > 0.0}


@dataclass(frozen=True)
class KpCheck:
    """Verificación de una reacción de disociación con su constante de equilibrio.

    ln K_p = −Δg°/(R·T) con los g° de los polinomios (p° = 1 bar), contra
    K_p = Π (yᵢ·p/p°)^νᵢ de la composición calculada (Cengel §16-2).
    """

    key: str
    reaction: str
    ln_Kp_tables: float
    ln_Kp_composition: float
    delta_g_J_per_mol: float


@dataclass(frozen=True)
class EnergySplit:
    """Reparto de la energía por unidad de combustible (J), a presión constante.

    Q_sal = PCI + calor sensible de los reactivos − inquemados − humos
    (sensible, con el agua como vapor) + latente recuperado.
    """

    lhv: float
    hhv: float
    sensible_reactants: float
    unburned: float
    stack_sensible: float
    latent_recovered: float

    @property
    def Q_out(self) -> float:
        return (
            self.lhv
            + self.sensible_reactants
            - self.unburned
            - self.stack_sensible
            + self.latent_recovered
        )


@dataclass(frozen=True)
class HeatResult:
    """Los humos a T_s: calor que sale, agua que condensa y rendimiento."""

    T_K: float
    gas: dict[str, float]
    condensed: float
    Q_out: float
    p_Pa: float
    split: EnergySplit | None

    @property
    def eta_lhv(self) -> float | None:
        return self.Q_out / self.split.lhv if self.split else None

    @property
    def eta_hhv(self) -> float | None:
        return self.Q_out / self.split.hhv if self.split else None


@dataclass(frozen=True)
class SecondLaw:
    """Entropía generada y exergía destruida (por unidad de combustible)."""

    T0_K: float
    S_reactants: float
    S_products_adiabatic: float
    lhv: float
    T_sink_K: float | None = None
    S_products_heat: float | None = None
    Q_out: float | None = None

    @property
    def S_gen_adiabatic(self) -> float:
        """S_gen = S_p(T_ad) − S_r (Cengel 15-10)."""
        return self.S_products_adiabatic - self.S_reactants

    @property
    def X_dest_adiabatic(self) -> float:
        return self.T0_K * self.S_gen_adiabatic

    @property
    def S_gen_heat(self) -> float | None:
        """S_gen = S_p(T_s) − S_r + Q_sal/T_b (Cengel 15-11)."""
        if self.S_products_heat is None or self.Q_out is None or self.T_sink_K is None:
            return None
        return self.S_products_heat - self.S_reactants + self.Q_out / self.T_sink_K

    @property
    def X_dest_heat(self) -> float | None:
        s = self.S_gen_heat
        return None if s is None else self.T0_K * s

    @property
    def X_heat(self) -> float | None:
        """Exergía del calor entregado a T_b: Q·(1 − T₀/T_b)."""
        if self.Q_out is None or self.T_sink_K is None:
            return None
        return self.Q_out * (1.0 - self.T0_K / self.T_sink_K)


@dataclass(frozen=True)
class CombustionResult:
    """La combustión resuelta (por unidad de combustible: mol o kg)."""

    inputs: CombustionInputs
    stoich: Stoichiometry
    H_reactants: float
    H_reactants_ref: float
    reactant_gas_moles: float
    RT_reactants: float
    flame: FlameResult
    flame_eq: FlameResult | None
    eq_error: str | None
    kp_checks: tuple[KpCheck, ...]
    heat: HeatResult | None
    second_law: SecondLaw | None

    @property
    def fuel(self) -> Fuel:
        return self.inputs.fuel

    @property
    def U_reactants(self) -> float:
        return self.H_reactants - self.RT_reactants

    @property
    def volume(self) -> float:
        """Volumen de los reactivos gaseosos (m³ por unidad): el de la bomba."""
        return self.RT_reactants / self.inputs.p_Pa

    @property
    def lhv(self) -> float:
        return self.fuel.lhv_per_basis

    @property
    def hhv(self) -> float:
        return self.fuel.hhv_per_basis

    @property
    def per_kg(self) -> float:
        """Factor de «por unidad» a «por kg de combustible»."""
        return 1.0 / self.fuel.mass_per_basis

    @property
    def dissociation_drop_K(self) -> float | None:
        return None if self.flame_eq is None else self.flame.T_K - self.flame_eq.T_K

    def total(self, per_basis: float) -> float | None:
        """Lo que vale por unidad, multiplicado por el caudal (por segundo)."""
        m = self.inputs.m_fuel_kg_s
        return None if m is None else per_basis * self.per_kg * m


# ---------------------------------------------------------------------
# Energía de reactivos y productos
# ---------------------------------------------------------------------


def _air_moles(stoich: Stoichiometry) -> dict[str, float]:
    return {k: stoich.air_moles * y for k, y in stoich.air.items()}


def _fuel_gas_moles(fuel: Fuel) -> float:
    """Moles de gas que aporta el combustible por unidad (1 si es un gas, 0 si no)."""
    return 1.0 if fuel.kind == "gas" else 0.0


def _elements_of_reactants(stoich: Stoichiometry) -> dict[str, float]:
    """Átomos de cada elemento que entran (combustible, su humedad y el aire)."""
    out = dict(stoich.fuel.elements())
    w = stoich.fuel.moisture_mol
    out["H"] += 2.0 * w
    out["O"] += w
    for k, n in _air_moles(stoich).items():
        for el, a in species(k).formula.items():
            out[el] = out.get(el, 0.0) + n * a
    return {k: v for k, v in out.items() if v > 0.0}


def _gas_energy(moles: Mapping[str, float], T_K: float, process: ProcessKind) -> float:
    """H (a p constante) o U = H − R·T·n (a V constante) de una mezcla de gases."""
    H = mixture_enthalpy(moles, T_K)
    if process == "p":
        return H
    return H - R_U * T_K * sum(moles.values())


def _complete_flame(products: Mapping[str, float], target: float, process: ProcessKind) -> float:
    """T a la que los productos tienen la energía de los reactivos (combustión completa)."""

    def f(T: float) -> float:
        return _gas_energy(products, T, process) - target

    if f(_T_LOW_K) > 0.0:
        raise ValueError(
            "Los productos tienen más energía que los reactivos aun fríos: no hay llama "
            "(revisá el poder calorífico o la humedad del combustible)."
        )
    if f(T_MAX_K) < 0.0:
        raise ValueError(
            f"Con combustión completa la llama superaría {T_MAX_K:.0f} K, el límite de los "
            "polinomios NASA: a esas temperaturas la combustión completa no tiene sentido (la "
            "disociación manda). Con más exceso de oxígeno, o con aire, la llama baja."
        )
    return float(brentq(f, _T_LOW_K, T_MAX_K, xtol=1e-9, rtol=1e-14))


def _split_water(
    products: Mapping[str, float], T_K: float, *, p_Pa: float | None, V_m3: float | None
) -> tuple[dict[str, float], float]:
    """Gas y agua condensada a T: el vapor no puede pasar su presión de saturación."""
    gas = dict(products)
    n_w = gas.get("H2O", 0.0)
    if n_w <= 0.0 or T_K >= 647.0:
        return gas, 0.0
    p_sat = saturation_pressure(T_K)
    if V_m3 is not None:
        n_v_max = p_sat * V_m3 / (R_U * T_K)
    else:
        assert p_Pa is not None
        if p_sat >= p_Pa:
            return gas, 0.0
        n_dry = sum(v for k, v in gas.items() if k != "H2O")
        n_v_max = p_sat / p_Pa * n_dry / (1.0 - p_sat / p_Pa)
    if n_w <= n_v_max:
        return gas, 0.0
    gas["H2O"] = n_v_max
    return gas, n_w - n_v_max


def _kp_checks(flame: FlameResult) -> tuple[KpCheck, ...]:
    """K_p de tablas contra K_p de la composición de equilibrio (Cengel §16-2)."""
    reactions = (
        (
            "CO2",
            r"\mathrm{CO_2 \rightleftharpoons CO + \tfrac{1}{2}O_2}",
            {"CO2": -1, "CO": 1, "O2": 0.5},
        ),
        (
            "H2O",
            r"\mathrm{H_2O \rightleftharpoons H_2 + \tfrac{1}{2}O_2}",
            {"H2O": -1, "H2": 1, "O2": 0.5},
        ),
        (
            "NO",
            r"\mathrm{\tfrac{1}{2}N_2 + \tfrac{1}{2}O_2 \rightleftharpoons NO}",
            {"N2": -0.5, "O2": -0.5, "NO": 1},
        ),
    )
    y = flame.fractions
    T = flame.T_K
    out = []
    for key, label, nu in reactions:
        if any(y.get(k, 0.0) <= 0.0 for k in nu):
            continue
        dg = sum(v * species(k).g0(T) for k, v in nu.items())
        ln_tab = -dg / (R_U * T)
        ln_comp = sum(v * math.log(y[k] * flame.p_Pa / P_STANDARD_PA) for k, v in nu.items())
        out.append(KpCheck(key, label, ln_tab, ln_comp, dg))
    return tuple(out)


# ---------------------------------------------------------------------
# Resolución
# ---------------------------------------------------------------------


def solve_combustion(inputs: CombustionInputs) -> CombustionResult:
    """Estequiometría, llama (completa y con disociación), calor y segundo principio.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno: datos fuera de rango, combustible fuera
        de sus tablas, humos más calientes que la llama, calor entregado a una
        temperatura que viola el segundo principio.
    """
    fuel = inputs.fuel
    fuel.check_T(inputs.T_fuel_K)
    if inputs.process not in ("p", "v"):
        raise ValueError(f"Proceso desconocido: {inputs.process!r}.")
    if not (math.isfinite(inputs.T0_K) and 223.15 <= inputs.T0_K <= 333.15):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −50 °C y 60 °C.")
    if not (math.isfinite(inputs.so3_conversion) and 0.0 <= inputs.so3_conversion <= 0.2):
        raise ValueError("La conversión de SO₂ a SO₃ tiene que estar entre 0 y 20 %.")
    if inputs.m_fuel_kg_s is not None and not (
        math.isfinite(inputs.m_fuel_kg_s) and inputs.m_fuel_kg_s > 0.0
    ):
        raise ValueError("El caudal de combustible tiene que ser positivo.")
    stoich = solve_stoichiometry(
        fuel, inputs.oxidizer, inputs.air, p_Pa=inputs.p_Pa, co_fraction=inputs.co_fraction
    )
    air = _air_moles(stoich)
    T_air = inputs.oxidizer.T_K
    H_r = fuel.enthalpy(inputs.T_fuel_K) + mixture_enthalpy(air, T_air)
    H_r_ref = fuel.enthalpy(T_REF_K) + mixture_enthalpy(air, T_REF_K)
    n_fuel_gas = _fuel_gas_moles(fuel)
    RT_r = R_U * (n_fuel_gas * inputs.T_fuel_K + stoich.air_moles * T_air)
    n_gas_r = n_fuel_gas + stoich.air_moles
    process = inputs.process
    V = RT_r / inputs.p_Pa
    target = H_r if process == "p" else H_r - RT_r

    # --- llama con combustión completa (o con el CO dado) ------------------
    T_ad = _complete_flame(stoich.products, target, process)
    p_ad = inputs.p_Pa if process == "p" else sum(stoich.products.values()) * R_U * T_ad / V
    flame = FlameResult(T_ad, dict(stoich.products), p_ad, "complete")

    # --- llama con disociación ---------------------------------------------
    flame_eq: FlameResult | None = None
    eq_error: str | None = None
    kp: tuple[KpCheck, ...] = ()
    if inputs.equilibrium:
        try:
            T_eq, x_eq = adiabatic_equilibrium(
                _elements_of_reactants(stoich),
                target,
                process=process,
                p_Pa=inputs.p_Pa if process == "p" else None,
                V_m3=V if process == "v" else None,
                guess=stoich.products,
                T_low_K=_T_LOW_K + 50.0,
            )
            p_eq = inputs.p_Pa if process == "p" else sum(x_eq.values()) * R_U * T_eq / V
            flame_eq = FlameResult(T_eq, x_eq, p_eq, "equilibrium")
            kp = _kp_checks(flame_eq)
        except ValueError as exc:
            eq_error = str(exc)

    # --- calor con los humos a T_s -------------------------------------------
    heat: HeatResult | None = None
    if inputs.T_products_K is not None:
        T_out = inputs.T_products_K
        if not (math.isfinite(T_out) and 273.16 <= T_out <= 3000.0):
            raise ValueError(
                "La temperatura de salida de los humos tiene que estar entre 0,01 °C y 2727 °C."
            )
        if T_out >= T_ad:
            raise ValueError(
                f"Los humos no pueden salir a {_degC(T_out)}: la llama adiabática llega a "
                f"{_degC(T_ad)}, así que no habría calor que sacar (habría que calentarlos)."
            )
        gas, condensed = _split_water(
            stoich.products,
            T_out,
            p_Pa=inputs.p_Pa if process == "p" else None,
            V_m3=V if process == "v" else None,
        )
        h_l = species("H2O(l)").h(T_out) if condensed > 0.0 else 0.0
        if process == "p":
            Q = H_r - mixture_enthalpy(gas, T_out) - condensed * h_l
            p_out = inputs.p_Pa
            prods = stoich.products
            unburned = sum(
                prods.get(k, 0.0)
                * (species(k).hf_J_per_mol - species("CO2" if k == "CO" else "H2O").hf_J_per_mol)
                for k in _UNBURNED
            )
            stack = sum(n * species(k).delta_h(T_out) for k, n in prods.items())
            latent = condensed * (species("H2O").h(T_out) - h_l)
            split = EnergySplit(
                lhv=fuel.lhv_per_basis,
                hhv=fuel.hhv_per_basis,
                sensible_reactants=H_r - H_r_ref,
                unburned=unburned,
                stack_sensible=stack,
                latent_recovered=latent,
            )
        else:
            U_p = _gas_energy(gas, T_out, "v") + condensed * h_l
            Q = (H_r - RT_r) - U_p
            p_out = sum(gas.values()) * R_U * T_out / V
            split = None
        heat = HeatResult(T_out, gas, condensed, Q, p_out, split)

    # --- segundo principio (solo con especies: el análisis no tiene s°) --------
    second: SecondLaw | None = None
    if fuel.per_mol:
        second = _second_law(inputs, stoich, flame, heat, V)

    return CombustionResult(
        inputs=inputs,
        stoich=stoich,
        H_reactants=H_r,
        H_reactants_ref=H_r_ref,
        reactant_gas_moles=n_gas_r,
        RT_reactants=RT_r,
        flame=flame,
        flame_eq=flame_eq,
        eq_error=eq_error,
        kp_checks=kp,
        heat=heat,
        second_law=second,
    )


def _reactant_entropy(inputs: CombustionInputs, stoich: Stoichiometry) -> float:
    """Entropía de los reactivos: combustible y aire como corrientes separadas a p.

    En la bomba (V constante) los gases están mezclados: cada uno a su presión
    parcial en la mezcla de combustible y aire.
    """
    fuel = inputs.fuel
    p = inputs.p_Pa
    air = _air_moles(stoich)
    T_f, T_a = inputs.T_fuel_K, inputs.oxidizer.T_K
    if inputs.process == "p":
        s_fuel = fuel.entropy(T_f, p)
        assert s_fuel is not None
        return s_fuel + gas_mixture_entropy(air, T_a, p)
    n_gas = _fuel_gas_moles(fuel) + stoich.air_moles
    s = 0.0
    for k, y in fuel.components:
        if fuel.kind == "gas":
            s += y * component_entropy(k, T_f, y / n_gas * p)
        else:
            s += y * component_entropy(k, T_f, p)
    for k, n in air.items():
        s += n * (species(k).s0(T_a) - R_U * math.log(n / n_gas * p / P_STANDARD_PA))
    return s


def _second_law(
    inputs: CombustionInputs,
    stoich: Stoichiometry,
    flame: FlameResult,
    heat: HeatResult | None,
    V: float,
) -> SecondLaw:
    S_r = _reactant_entropy(inputs, stoich)
    S_ad = gas_mixture_entropy(flame.products, flame.T_K, flame.p_Pa)
    T0 = inputs.T0_K
    if heat is None:
        return SecondLaw(T0, S_r, S_ad, inputs.fuel.lhv_per_basis)
    T_b = inputs.T_sink_K if inputs.T_sink_K is not None else T0
    if not (math.isfinite(T_b) and 200.0 <= T_b < flame.T_K):
        raise ValueError(
            "La temperatura a la que se entrega el calor tiene que estar entre −73 °C y la "
            "llama adiabática."
        )
    gas_p = heat.p_Pa
    S_heat = gas_mixture_entropy(heat.gas, heat.T_K, gas_p)
    if heat.condensed > 0.0:
        S_heat += heat.condensed * species("H2O(l)").s0(heat.T_K)
    second = SecondLaw(T0, S_r, S_ad, inputs.fuel.lhv_per_basis, T_b, S_heat, heat.Q_out)
    s_gen = second.S_gen_heat
    if s_gen is not None and s_gen < 0.0:
        raise ValueError(
            f"Con el calor entregado a {_degC(T_b)} la entropía generada da negativa: el calor "
            "iría de los humos a un medio más caliente que ellos. Bajá la temperatura T_b."
        )
    return second


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def combustion_notes(result: CombustionResult) -> list[str]:
    """Notas para el alumno sobre el resultado."""
    notes: list[str] = []
    s = result.stoich
    inp = result.inputs
    fuel = result.fuel
    if s.rich:
        lost = _unburned_fraction(result)
        notes.append(
            f"Con defecto de aire (λ = {s.lam:.3g}) quedan CO"
            + (" y H₂" if s.products.get("H2", 0.0) > 0 else "")
            + f" en los humos: se pierde el {_pct(lost)} del PCI sin quemar (Cengel 15-8 c). "
            "En la llama de equilibrio el CO y el H₂ se reparten por la reacción de gas de agua "
            "(CO + H₂O ⇌ CO₂ + H₂)."
        )
    elif inp.co_fraction > 0.0:
        notes.append(
            f"El {_pct(inp.co_fraction)} del carbono sale como CO: se pierde el "
            f"{_pct(_unburned_fraction(result))} del PCI (Cengel 15-6). El CO es tóxico: una "
            "caldera bien regulada no pasa de unas decenas de ppm."
        )
    if s.lam > 2.5 and inp.oxidizer.kind != "oxygen":
        notes.append(
            f"Con λ = {s.lam:.3g} sobra mucho aire: la llama se enfría y por la chimenea se va "
            "el calor de todo ese aire. En una caldera se trabaja con λ ≈ 1,05–1,3."
        )
    if result.flame_eq is not None:
        drop = result.dissociation_drop_K or 0.0
        if drop < -5.0:
            notes.append(
                f"La llama de equilibrio ({_degC(result.flame_eq.T_K)}) sale más caliente que "
                f"la del modelo ({_degC(result.flame.T_K)}): con el CO que diste el carbono "
                "quema incompleto, y en equilibrio (con oxígeno de sobra) casi todo el CO pasa "
                "a CO₂ y libera ese calor. El CO real es cosa de la cinética (mezcla, tiempo, "
                "superficies frías)."
            )
        if drop > 20.0:
            notes.append(
                f"La disociación baja la llama {drop:.0f} K (de {_degC(result.flame.T_K)} a "
                f"{_degC(result.flame_eq.T_K)}): a esas temperaturas parte del CO₂ y del H₂O se "
                "separa en CO, H₂, OH, O y H, y eso absorbe energía (vademecum §16.8)."
            )
        y_no = result.flame_eq.fractions.get("NO", 0.0)
        if y_no > 1e-4:
            notes.append(
                f"En equilibrio a la temperatura de llama hay {1e6 * y_no:.0f} ppm de NO: es el "
                "techo del NO térmico. El que sale de verdad depende de la cinética (mecanismo "
                "de Zeldovich) y del tiempo de residencia, y cae mucho con menos temperatura."
            )
    if result.eq_error:
        notes.append(f"No se pudo calcular la llama con disociación: {result.eq_error}")
    if inp.oxidizer.kind == "oxygen":
        notes.append(
            "Con oxígeno puro no hay N₂ que absorba calor: la llama con combustión completa da "
            "temperaturas irreales y solo la de equilibrio (con la disociación) tiene sentido."
        )
    if inp.oxidizer.phi > 0.0 and inp.oxidizer.kind != "oxygen":
        notes.append(
            f"El aire tiene {_pct(inp.oxidizer.phi)} de humedad relativa a "
            f"{_degC(inp.oxidizer.T_phi_K)}: ese vapor pasa a los humos y sube el punto de "
            "rocío (Cengel 15-3)."
        )
    if fuel.elements()["S"] > 0.0 and s.products.get("H2O", 0.0) > 0.0:
        acid = s.acid_dew_point_K(inp.so3_conversion)
        if acid is not None:
            notes.append(
                f"El azufre pasa a SO₂ y una parte ({_pct(inp.so3_conversion)}) a SO₃, que con el "
                f"agua forma ácido sulfúrico: el rocío ácido es {_degC(acid)}, muy por encima "
                "del rocío del agua. Los humos (y las superficies frías) tienen que quedar por "
                "encima para no corroer la caldera."
            )
    if not fuel.per_mol and fuel.analysis is not None and fuel.analysis.W > 0.2:
        notes.append(
            f"El combustible tiene {_pct(fuel.analysis.W)} de humedad: evaporarla se lleva parte "
            "del calor, por eso el PCI queda tan por debajo del PCS."
        )
    heat = result.heat
    if heat is not None and heat.split is not None and inp.oxidizer.T_K > inp.T0_K + 50.0:
        notes.append(
            f"El aire entra precalentado ({_degC(inp.oxidizer.T_K)}) y el rendimiento cuenta ese "
            "calor sensible como un aporte de afuera. Si lo calientan los mismos humos (un "
            "precalentador de aire dentro de la caldera), el rendimiento se calcula con el aire "
            "a la temperatura ambiente."
        )
    if heat is not None:
        if heat.condensed > 0.0:
            total_w = s.products.get("H2O", 0.0)
            notes.append(
                f"A {_degC(heat.T_K)} los humos están por debajo del punto de rocío: condensa el "
                f"{_pct(heat.condensed / total_w)} del agua y se recupera su calor latente."
            )
            if heat.eta_lhv is not None and heat.eta_lhv > 1.0:
                notes.append(
                    f"El rendimiento sobre el PCI da {_pct(heat.eta_lhv)}: no es un error. El "
                    "PCI supone que el agua se va como vapor; al condensarla se recupera parte "
                    "del latente. Sobre el PCS el rendimiento es "
                    f"{_pct(heat.eta_hhv or 0.0)} (caldera de condensación)."
                )
        if fuel.elements()["S"] > 0.0:
            acid = s.acid_dew_point_K(inp.so3_conversion)
            if acid is not None and heat.T_K < acid:
                notes.append(
                    f"¡Ojo! Los humos salen a {_degC(heat.T_K)}, por debajo del rocío ácido "
                    f"({_degC(acid)}): el ácido sulfúrico condensa y corroe la chimenea y el "
                    "economizador."
                )
    if inp.process == "v":
        notes.append(
            f"A volumen constante la llama es más caliente ({_degC(result.flame.T_K)}): no se "
            "gasta trabajo de flujo (p·v), y la presión sube a "
            f"{result.flame.p_Pa / 1e5:.3g} bar (Cengel 15-7)."
        )
    if fuel.kind == "liquid":
        notes.append(
            "El combustible entra líquido: parte del calor se usa en evaporarlo (su h_f es más "
            "negativa que la del vapor)."
        )
    second = result.second_law
    if second is not None:
        frac = second.X_dest_adiabatic / second.lhv
        notes.append(
            f"La combustión adiabática destruye el {_pct(frac)} de la exergía del combustible "
            "(≈ PCI, vademecum §16.13): es la mayor irreversibilidad de una planta térmica "
            "(Cengel 15-10)."
        )
    return notes


def _unburned_fraction(result: CombustionResult) -> float:
    prods = result.stoich.products
    q = prods.get("CO", 0.0) * (
        species("CO").hf_J_per_mol - species("CO2").hf_J_per_mol
    ) + prods.get("H2", 0.0) * (-species("H2O").hf_J_per_mol)
    return q / result.lhv


# ---------------------------------------------------------------------
# Análisis de humos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FlueGasInputs:
    """Humos secos medidos: solo O₂ (y CO) o un Orsat (CO₂, CO y O₂).

    Fracciones (no %); ``T_cool_K``: la temperatura a la que se enfrían los
    humos para ver cuánta agua condensa (Cengel 15-4 c).
    """

    fuel: Fuel
    oxidizer: Oxidizer = field(default_factory=Oxidizer)
    method: Literal["o2", "orsat"] = "o2"
    o2_dry: float = 0.03
    co_dry: float = 0.0
    co2_dry: float = 0.0
    p_Pa: float = 101_325.0
    T_cool_K: float = T_REF_K


@dataclass(frozen=True)
class FlueGasResult:
    """λ y los humos que corresponden a la medición."""

    inputs: FlueGasInputs
    stoich: Stoichiometry
    orsat: OrsatResult | None
    co_fraction: float
    unburned_fraction: float
    condensed: float
    gas_cooled: dict[str, float]

    @property
    def lam(self) -> float:
        return self.stoich.lam


def solve_flue_gas(inputs: FlueGasInputs) -> FlueGasResult:
    """λ, el aire, el agua y las pérdidas desde los humos secos medidos.

    Con solo O₂: la λ que da ese O₂ con combustión completa, más el CO medido
    (iterando la fracción del carbono que pasa a CO). Con el Orsat: los
    balances de C, N₂ y H (Cengel 15-4).

    Raises
    ------
    ValueError
        Si la medición no tiene sentido para ese combustible.
    """
    fuel = inputs.fuel
    if not (math.isfinite(inputs.T_cool_K) and 273.16 <= inputs.T_cool_K <= 600.0):
        raise ValueError("La temperatura de enfriamiento tiene que estar entre 0,01 °C y 327 °C.")
    if not (math.isfinite(inputs.co_dry) and 0.0 <= inputs.co_dry < 0.1):
        raise ValueError("El CO medido tiene que estar entre 0 y 10 % (100 000 ppm).")
    e = fuel.elements()
    orsat: OrsatResult | None = None
    if inputs.method == "orsat":
        orsat = orsat_analysis(
            fuel,
            inputs.oxidizer,
            inputs.co2_dry,
            inputs.co_dry,
            inputs.o2_dry,
            p_Pa=inputs.p_Pa,
        )
        x_co = 100.0 * inputs.co_dry / (e["C"] * orsat.fuel_per_100)
        x_co = min(max(x_co, 0.0), 0.999)
        lam = orsat.lam
    else:
        inputs.oxidizer.check(inputs.p_Pa)
        air = inputs.oxidizer.composition(inputs.p_Pa)
        x_co = 0.0
        lam = 1.0
        for _ in range(60):
            lam = lambda_from_o2(fuel, air, inputs.o2_dry, x_co)
            st = solve_stoichiometry(
                fuel, inputs.oxidizer, AirSpec("lambda", lam), p_Pa=inputs.p_Pa, co_fraction=x_co
            )
            if e["C"] <= 0.0:
                break
            new = inputs.co_dry * st.n_dry / e["C"]
            if new >= 1.0:
                raise ValueError("Con tanto CO casi todo el carbono quedaría sin quemar.")
            if abs(new - x_co) < 1e-14:
                x_co = new
                break
            x_co = new
    if lam < 1.0:
        raise ValueError(
            f"La medición da λ = {lam:.3g} < 1 (defecto de aire): revisá el análisis; con O₂ "
            "en los humos el aire no puede faltar."
        )
    stoich = solve_stoichiometry(
        fuel, inputs.oxidizer, AirSpec("lambda", lam), p_Pa=inputs.p_Pa, co_fraction=x_co
    )
    co = stoich.products.get("CO", 0.0)
    unburned = co * (species("CO").hf_J_per_mol - species("CO2").hf_J_per_mol) / fuel.lhv_per_basis
    gas, condensed = _split_water(stoich.products, inputs.T_cool_K, p_Pa=inputs.p_Pa, V_m3=None)
    return FlueGasResult(
        inputs=inputs,
        stoich=stoich,
        orsat=orsat,
        co_fraction=x_co,
        unburned_fraction=unburned,
        condensed=condensed,
        gas_cooled=gas,
    )


def flue_gas_notes(result: FlueGasResult) -> list[str]:
    """Notas para el alumno sobre el análisis de humos."""
    notes: list[str] = []
    s = result.stoich
    if result.orsat is not None:
        rel = result.orsat.oxygen_residual_relative
        if abs(rel) > 0.02:
            notes.append(
                f"El balance de oxígeno no cierra ({_pct(rel)} del O que sale): la medición no "
                "es coherente con ese combustible (o el analizador está mal calibrado)."
            )
        else:
            notes.append(
                f"El balance de oxígeno cierra a {_pct(abs(rel))}: la medición es coherente con "
                "el combustible."
            )
    if s.lam > 1.5:
        notes.append(
            f"λ = {s.lam:.3g}: demasiado aire. Bajarlo reduce la pérdida por chimenea (cada "
            "punto de O₂ de más cuesta alrededor de medio punto de rendimiento)."
        )
    elif s.lam < 1.05:
        notes.append(f"λ = {s.lam:.3g}: muy poco exceso; con tan poco aire suele aparecer CO.")
    if result.unburned_fraction > 0.002:
        notes.append(
            f"El CO se lleva el {_pct(result.unburned_fraction)} del PCI sin quemar: hay que "
            "revisar la mezcla en el quemador."
        )
    if result.condensed > 0.0:
        total = s.products.get("H2O", 0.0)
        notes.append(
            f"Al enfriar los humos a {_degC(result.inputs.T_cool_K)} condensa el "
            f"{_pct(result.condensed / total)} del agua (Cengel 15-4 c): por eso el análisis se "
            "da en base seca."
        )
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

_GN_CENGEL = Fuel.mixture(
    "Gas natural (Cengel 15-3)", {"CH4": 0.72, "H2": 0.09, "N2": 0.14, "O2": 0.02, "CO2": 0.03}
)

COMBUSTION_EXAMPLES: dict[str, CombustionInputs] = {
    "Cengel 15-1: octano con 20 kmol de O₂": CombustionInputs(
        FUELS["n-Octano (líquido)"], air=AirSpec("lambda", 20.0 / 12.5)
    ),
    "Cengel 15-2: etano con 20 % de exceso de aire": CombustionInputs(
        Fuel.mixture("Etano", {"C2H6": 1.0}), air=AirSpec("excess", 0.2), p_Pa=100e3
    ),
    "Cengel 15-3: gas natural con aire húmedo": CombustionInputs(
        _GN_CENGEL, oxidizer=Oxidizer("technical", 293.15, 0.8), T0_K=293.15
    ),
    "Cengel 15-6: propano líquido con CO en los humos": CombustionInputs(
        FUELS["Propano líquido"],
        oxidizer=Oxidizer("technical", 280.15),
        air=AirSpec("excess", 0.5),
        co_fraction=0.1,
        T_products_K=1500.0,
        T0_K=280.15,
        m_fuel_kg_s=0.05 / 60.0,
    ),
    "Cengel 15-7: metano y oxígeno en un recipiente rígido": CombustionInputs(
        FUELS["Metano"],
        oxidizer=Oxidizer("oxygen"),
        air=AirSpec("lambda", 1.5),
        process="v",
        T_products_K=1000.0,
    ),
    "Cengel 15-8: llama adiabática del octano": CombustionInputs(
        FUELS["n-Octano (líquido)"], air=AirSpec("theoretical", 1.0)
    ),
    "Cengel 15-10 y 15-11: metano con 50 % de exceso de aire": CombustionInputs(
        FUELS["Metano"], air=AirSpec("excess", 0.5), T_products_K=T_REF_K, T_sink_K=T_REF_K
    ),
    "Caldera de gas natural (3 % de O₂, humos a 150 °C)": CombustionInputs(
        FUELS["Gas natural típico"],
        oxidizer=Oxidizer("technical", 293.15, 0.6),
        air=AirSpec("o2_dry", 0.03),
        T_products_K=423.15,
        T_sink_K=453.15,
        T0_K=293.15,
        m_fuel_kg_s=0.05,
    ),
    "Caldera de condensación (humos a 45 °C)": CombustionInputs(
        FUELS["Gas natural típico"],
        oxidizer=Oxidizer("technical", 293.15, 0.6),
        air=AirSpec("o2_dry", 0.03),
        T_products_K=318.15,
        T_sink_K=313.15,
        T0_K=293.15,
        m_fuel_kg_s=0.002,
    ),
    "Fueloil en una caldera (rocío ácido)": CombustionInputs(
        FUELS["Fueloil"],
        air=AirSpec("excess", 0.15),
        T_fuel_K=383.15,
        T_products_K=453.15,
        T_sink_K=473.15,
        m_fuel_kg_s=0.5,
    ),
    "Bagazo de caña en la caldera de un ingenio": CombustionInputs(
        FUELS["Bagazo de caña (50 % de humedad)"],
        oxidizer=Oxidizer("technical", 473.15),
        air=AirSpec("excess", 0.4),
        T_products_K=473.15,
        T_sink_K=523.15,
        m_fuel_kg_s=10.0,
    ),
    "Hidrógeno con 20 % de exceso de aire": CombustionInputs(
        FUELS["Hidrógeno"], air=AirSpec("excess", 0.2)
    ),
    "Motor a gas: propano a volumen constante": CombustionInputs(
        FUELS["Propano"],
        oxidizer=Oxidizer("technical", 650.0),
        air=AirSpec("lambda", 1.0),
        p_Pa=15e5,
        T_fuel_K=650.0,
        process="v",
    ),
}

COMBUSTION_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 15-1: octano con 20 kmol de O₂": (
        "Cengel da AC = 24,2 kg/kg con 4,76 kmol de aire de 29 kg/kmol por kmol de O₂; con el "
        "aire técnico del vademecum (M = 28,85 kg/kmol) da 24,05."
    ),
    "Cengel 15-2: etano con 20 % de exceso de aire": (
        "Cengel: AC = 19,3 kg/kg y punto de rocío de 52,3 °C a 100 kPa."
    ),
    "Cengel 15-3: gas natural con aire húmedo": (
        "Cengel: punto de rocío de los productos de 60,9 °C (el aire entra a 20 °C y 80 %)."
    ),
    "Cengel 15-6: propano líquido con CO en los humos": (
        "Cengel: 1,18 kg/min de aire (con M = 29) y 413 kJ/min de calor que sale (363 880 kJ "
        "por kmol de propano). El propano líquido sale del gas de NASA menos el h_fg de "
        "CoolProp a 25 °C."
    ),
    "Cengel 15-7: metano y oxígeno en un recipiente rígido": (
        "Cengel (en unidades inglesas): 308 730 Btu de calor que sale y 3,35 atm al final, "
        "con los productos a 1800 R (1000 K)."
    ),
    "Cengel 15-8: llama adiabática del octano": (
        "Cengel: 2395 K con 100 % de aire teórico y 962 K con 400 %. Con 90 % (5,5 CO₂ + 2,5 CO) "
        "el libro da 2236 K, pero con esos productos el balance cierra en 2284 K: revisá tu "
        "edición. Con disociación, 2264 K."
    ),
    "Cengel 15-10 y 15-11: metano con 50 % de exceso de aire": (
        "Cengel 15-10: llama de 1789 K. 15-11 (enfriado a 25 °C, condensa parte del agua): "
        "871 400 kJ/kmol de calor, S_gen = 2746 kJ/(kmol·K) y X_dest = 818 000 kJ/kmol."
    ),
    "Caldera de gas natural (3 % de O₂, humos a 150 °C)": (
        "Un analizador mide 3 % de O₂ en humos secos; el agua de la caldera, a 180 °C."
    ),
    "Caldera de condensación (humos a 45 °C)": (
        "Los humos salen por debajo del punto de rocío: parte del agua condensa y el "
        "rendimiento sobre el PCI pasa el 100 %."
    ),
    "Fueloil en una caldera (rocío ácido)": (
        "Fueloil pesado con 2,5 % de azufre, precalentado a 110 °C para pulverizarlo; PCS de "
        "42,4 MJ/kg (Channiwala & Parikh, 2002)."
    ),
    "Bagazo de caña en la caldera de un ingenio": (
        "Bagazo con 50 % de humedad (Hugot, 1986: PCS seco ≈ 19 250 kJ/kg) y aire precalentado "
        "a 200 °C con los humos."
    ),
    "Hidrógeno con 20 % de exceso de aire": (
        "Sin carbono: los humos son agua, N₂ y O₂. El PCI por kg es casi 2,4 veces el del metano."
    ),
    "Motor a gas: propano a volumen constante": (
        "La mezcla, comprimida a 15 bar y 377 °C, se quema a volumen constante (el ciclo Otto "
        "ideal): la presión se triplica."
    ),
}

FLUE_GAS_EXAMPLES: dict[str, FlueGasInputs] = {
    "Cengel 15-4: Orsat del octano": FlueGasInputs(
        FUELS["n-Octano (líquido)"],
        method="orsat",
        co2_dry=0.1002,
        co_dry=0.0088,
        o2_dry=0.0562,
        p_Pa=100e3,
    ),
    "Caldera de gas natural: 3,2 % de O₂ y 40 ppm de CO": FlueGasInputs(
        FUELS["Gas natural típico"], o2_dry=0.032, co_dry=40e-6, T_cool_K=313.15
    ),
    "Caldera de fueloil (Orsat)": FlueGasInputs(
        FUELS["Fueloil"], method="orsat", co2_dry=0.1289, co_dry=0.0004, o2_dry=0.0441
    ),
}

FLUE_GAS_EXAMPLE_NOTES: dict[str, str] = {
    "Cengel 15-4: Orsat del octano": (
        "Cengel: 131 % de aire teórico y 6,59 kmol de agua que condensa por kmol de octano al "
        "enfriar a 25 °C y 100 kPa (con aire de 3,76 N₂/O₂; acá 3,762)."
    ),
    "Caldera de gas natural: 3,2 % de O₂ y 40 ppm de CO": (
        "Lo que mide un analizador portátil en la chimenea de una caldera."
    ),
    "Caldera de fueloil (Orsat)": "El CO₂ del Orsat incluye el SO₂ (se absorben juntos).",
}


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------


def default_lambda_values(inputs: CombustionInputs, n: int = 41) -> list[float]:
    """λ desde cerca del mínimo (sin hollín) hasta 3."""
    from core.combustion.stoichiometry import _soot_lambda

    lo = max(0.6, _soot_lambda(inputs.fuel) + 0.02) if inputs.fuel.elements()["C"] > 0 else 0.6
    lo = min(lo, 0.95)
    return [lo + (3.0 - lo) * k / (n - 1) for k in range(n)]


def lambda_sweep(
    inputs: CombustionInputs, values: Sequence[float], *, equilibrium: bool = True
) -> dict[str, list[float | None]]:
    """La llama, AC, los humos secos, el rocío y el rendimiento en función de λ."""
    out: dict[str, list[float | None]] = {
        k: []
        for k in (
            "lambda",
            "T_ad",
            "T_ad_eq",
            "AC",
            "CO2_dry",
            "O2_dry",
            "CO_dry",
            "T_dew",
            "eta_lhv",
        )
    }
    for lam in values:
        try:
            r = solve_combustion(
                replace(inputs, air=AirSpec("lambda", float(lam)), equilibrium=equilibrium)
            )
        except ValueError:
            continue
        y = r.stoich.dry_fractions
        out["lambda"].append(float(lam))
        out["T_ad"].append(r.flame.T_K)
        out["T_ad_eq"].append(r.flame_eq.T_K if r.flame_eq else None)
        out["AC"].append(r.stoich.AC)
        out["CO2_dry"].append(y.get("CO2", 0.0))
        out["O2_dry"].append(y.get("O2", 0.0))
        out["CO_dry"].append(y.get("CO", 0.0))
        out["T_dew"].append(r.stoich.dew_point_K)
        out["eta_lhv"].append(r.heat.eta_lhv if r.heat else None)
    return out


def default_air_temperature_values(n: int = 25) -> list[float]:
    """De 0 °C a 900 °C (aire precalentado)."""
    return [273.15 + 900.0 * k / (n - 1) for k in range(n)]


def air_temperature_sweep(
    inputs: CombustionInputs, values: Sequence[float], *, equilibrium: bool = True
) -> dict[str, list[float | None]]:
    """La llama en función de la temperatura del aire (precalentamiento)."""
    out: dict[str, list[float | None]] = {"T_air": [], "T_ad": [], "T_ad_eq": []}
    for T in values:
        try:
            # Precalentar no cambia el vapor del aire: φ queda dada a la T del ambiente.
            ox = replace(inputs.oxidizer, T_K=float(T), T_humidity_K=inputs.oxidizer.T_phi_K)
            r = solve_combustion(
                replace(inputs, oxidizer=ox, T_products_K=None, equilibrium=equilibrium)
            )
        except ValueError:
            continue
        out["T_air"].append(float(T))
        out["T_ad"].append(r.flame.T_K)
        out["T_ad_eq"].append(r.flame_eq.T_K if r.flame_eq else None)
    return out


def default_products_temperature_values(inputs: CombustionInputs, n: int = 40) -> list[float]:
    """De 30 °C a 400 °C de salida de los humos, con la mitad de los puntos bajo 100 °C.

    Ahí está el punto de rocío y la condensación cambia rápido el rendimiento.
    """
    low = n // 2
    cold = [303.15 + 70.0 * k / low for k in range(low)]
    hot = [373.15 + 300.0 * k / (n - low - 1) for k in range(n - low)]
    return cold + hot


def products_temperature_sweep(
    inputs: CombustionInputs, values: Sequence[float]
) -> dict[str, list[float | None]]:
    """Rendimiento sobre PCI y PCS y agua condensada en función de la T de los humos."""
    out: dict[str, list[float | None]] = {
        "T_out": [],
        "eta_lhv": [],
        "eta_hhv": [],
        "condensed_fraction": [],
        "stack_loss": [],
    }
    base = replace(inputs, equilibrium=False, T_sink_K=None)
    for T in values:
        try:
            r = solve_combustion(replace(base, T_products_K=float(T)))
        except ValueError:
            continue
        h = r.heat
        assert h is not None
        water = r.stoich.products.get("H2O", 0.0)
        out["T_out"].append(float(T))
        out["eta_lhv"].append(h.eta_lhv)
        out["eta_hhv"].append(h.eta_hhv)
        out["condensed_fraction"].append(h.condensed / water if water > 0 else 0.0)
        out["stack_loss"].append(
            (h.split.stack_sensible - h.split.latent_recovered) / h.split.lhv if h.split else None
        )
    return out


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _value(value_si: float | None, kind: QuantityKind, system: UnitSystem) -> Any:
    if value_si is None:
        return None
    return {"valor": convert_from_si(value_si, kind, system), "unidad": unit_label(kind, system)}


def _fuel_dict(fuel: Fuel) -> dict[str, Any]:
    out: dict[str, Any] = {"nombre": fuel.name, "clase": fuel.kind}
    if fuel.per_mol:
        out["composicion_molar"] = dict(fuel.components)
    else:
        assert fuel.analysis is not None
        out["analisis_elemental"] = fuel.analysis.as_dict()
    return out


def combustion_to_dict(result: CombustionResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado serializable a JSON (valores en ``system``)."""
    s = result.stoich
    inp = result.inputs
    per = "kmol de combustible" if result.fuel.per_mol else "kg de combustible"
    mh: QuantityKind = "molar_enthalpy"
    out: dict[str, Any] = {
        "modelo": "polinomios NASA-9 (McBride et al., 2002); vademecum §16",
        "base": f"por {per}",
        "combustible": _fuel_dict(result.fuel),
        "comburente": {
            "tipo": inp.oxidizer.label,
            "T": _value(inp.oxidizer.T_K, "temperature", system),
            "phi": inp.oxidizer.phi,
            "T_de_phi": _value(inp.oxidizer.T_phi_K, "temperature", system),
            "composicion_molar": s.air,
        },
        "p": _value(inp.p_Pa, "pressure", system),
        "proceso": "presión constante" if inp.process == "p" else "volumen constante",
        "estequiometria": {
            "O2_teorico": s.o2_theoretical,
            "aire_teorico": s.air_theoretical,
            "lambda": s.lam,
            "exceso": s.excess,
            "phi_equivalencia": s.equivalence_ratio,
            "AC_s": s.AC_s,
            "AC": s.AC,
            "GC": s.GC,
        },
        "productos": {
            "moles": s.products,
            "fraccion_humeda": s.wet_fractions,
            "fraccion_seca": s.dry_fractions,
            "fraccion_masica": s.mass_fractions,
            "M_g": s.M_gas * 1000.0,
            "volumen_normal_humedo_m3_por_kg": s.normal_volume,
            "volumen_normal_seco_m3_por_kg": s.normal_volume_dry,
            "CO2_max_seco": s.co2_max,
            "T_rocio": _value(s.dew_point_K, "temperature", system),
            "T_rocio_acido": _value(s.acid_dew_point_K(inp.so3_conversion), "temperature", system),
        },
        "poder_calorifico": {
            "PCS": _value(result.hhv * result.per_kg, "specific_enthalpy", system),
            "PCI": _value(result.lhv * result.per_kg, "specific_enthalpy", system),
        },
        "llama_completa": {
            "T": _value(result.flame.T_K, "temperature", system),
            "p": _value(result.flame.p_Pa, "pressure", system),
        },
    }
    if result.fuel.per_mol:
        out["poder_calorifico"]["PCS_molar"] = _value(result.hhv, mh, system)
        out["poder_calorifico"]["PCI_molar"] = _value(result.lhv, mh, system)
    if result.flame_eq is not None:
        out["llama_con_disociacion"] = {
            "T": _value(result.flame_eq.T_K, "temperature", system),
            "p": _value(result.flame_eq.p_Pa, "pressure", system),
            "fraccion_molar": result.flame_eq.fractions,
            "Kp": {
                k.key: {"ln_Kp_tablas": k.ln_Kp_tables, "ln_Kp_composicion": k.ln_Kp_composition}
                for k in result.kp_checks
            },
        }
    if result.heat is not None:
        h = result.heat
        out["calor"] = {
            "T_humos": _value(h.T_K, "temperature", system),
            "Q_sale_por_kg": _value(h.Q_out * result.per_kg, "specific_enthalpy", system),
            "agua_condensada": h.condensed,
            "rendimiento_PCI": h.eta_lhv,
            "rendimiento_PCS": h.eta_hhv,
            "Q_total": _value(result.total(h.Q_out), "power", system),
        }
        if h.split is not None:
            sp = h.split
            out["calor"]["reparto_por_kg"] = {
                k: _value(v * result.per_kg, "specific_enthalpy", system)
                for k, v in (
                    ("PCI", sp.lhv),
                    ("sensible_reactivos", sp.sensible_reactants),
                    ("inquemados", sp.unburned),
                    ("humos", sp.stack_sensible),
                    ("latente_recuperado", sp.latent_recovered),
                )
            }
    if result.second_law is not None:
        sl = result.second_law
        ms: QuantityKind = "molar_entropy"
        out["segundo_principio"] = {
            "T0": _value(sl.T0_K, "temperature", system),
            "S_gen_adiabatica": _value(sl.S_gen_adiabatic, ms, system),
            "X_destruida_adiabatica": _value(sl.X_dest_adiabatic, mh, system),
            "fraccion_del_PCI": sl.X_dest_adiabatic / sl.lhv,
            "T_b": _value(sl.T_sink_K, "temperature", system),
            "S_gen_con_calor": _value(sl.S_gen_heat, ms, system),
            "X_destruida_con_calor": _value(sl.X_dest_heat, mh, system),
            "X_del_calor": _value(sl.X_heat, mh, system),
        }
    if inp.m_fuel_kg_s is not None:
        out["caudales"] = {
            "combustible": _value(inp.m_fuel_kg_s, "mass_flow", system),
            "aire_seco": _value(s.AC * inp.m_fuel_kg_s, "mass_flow", system),
            "humos": _value(s.GC * inp.m_fuel_kg_s, "mass_flow", system),
        }
    return out


def flue_gas_to_dict(result: FlueGasResult, system: UnitSystem) -> dict[str, Any]:
    """Análisis de humos serializable a JSON (valores en ``system``)."""
    inp = result.inputs
    s = result.stoich
    out: dict[str, Any] = {
        "combustible": _fuel_dict(inp.fuel),
        "metodo": "solo O₂" if inp.method == "o2" else "Orsat",
        "medido_seco": {"O2": inp.o2_dry, "CO": inp.co_dry, "CO2": inp.co2_dry or None},
        "lambda": s.lam,
        "aire_teorico": s.lam,
        "AC": s.AC,
        "fraccion_C_a_CO": result.co_fraction,
        "perdida_por_inquemados": result.unburned_fraction,
        "productos_por_unidad": s.products,
        "fraccion_seca_calculada": s.dry_fractions,
        "T_enfriamiento": _value(inp.T_cool_K, "temperature", system),
        "agua_condensada": result.condensed,
    }
    if result.orsat is not None:
        o = result.orsat
        out["orsat"] = {
            "combustible_cada_100_mol": o.fuel_per_100,
            "aire_cada_100_mol": o.air_per_100,
            "agua_cada_100_mol": o.water_per_100,
            "residuo_O": o.oxygen_residual,
            "residuo_O_relativo": o.oxygen_residual_relative,
        }
    return out
