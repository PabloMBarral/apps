"""Un gas ideal entre dos estados (vademecum §4 y §10.5).

El gas sigue p·v = R·T con R = R_u/M (vademecum §4.2). Sus calores específicos
salen de los polinomios NASA de 9 coeficientes (McBride, Zehe y Gordon, 2002),
los mismos de /Combustion, y Δu, Δh y Δs se calculan con tres modelos para ver
cuánto pesa suponer c_p constante (Çengel, *Termodinámica*, §7-9):

- ``constant``: c_p a 25 °C (la tabla del vademecum §4.6, que coincide con la
  NASA al 0,05 %) y c_v = c_p − R (relación de Mayer, §4.5);
- ``mean``: c_p a la temperatura media del proceso;
- ``variable``: las integrales exactas con el polinomio (§4.7 y §10.5.2), h(T)
  y s°(T) con h = s° = 0 a 25 °C, así que p_r(25 °C) = 1.

El aire es la mezcla seca de cuatro componentes (N₂, O₂, Ar y CO₂, la de
``core.ideal_gas.AIR_DRY``) y el helio, monoatómico: c_p = 5/2·R exacto (no está
en el archivo de la NASA).

Todo en SI: Pa, K, m³/kg, J/kg y J/(kg·K).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import CoolProp.CoolProp as CP

from core.combustion import thermo

__all__ = [
    "HEAT_MODELS",
    "IDEAL_GASES",
    "P_REF_PA",
    "R_U",
    "STATE_EXAMPLES",
    "T_REF_K",
    "VADEMECUM_GASES",
    "GasModel",
    "GasState",
    "HeatModel",
    "IdealGas",
    "PairKind",
    "PropertyChange",
    "StateChange",
    "StateChangeInputs",
    "StateExample",
    "StatePair",
    "gas",
    "gas_state",
    "ideal_gas_check",
    "model_cp",
    "model_h",
    "model_s",
    "model_s0",
    "model_u",
    "solve_state_change",
    "state_from_pair",
    "state_change_to_dict",
]

R_U: float = thermo.R_U
#: Temperatura de referencia: h = u = s° = 0 a 25 °C (y p_r = 1).
T_REF_K: float = 298.15

HeatModel = Literal["constant", "mean", "variable"]
HEAT_MODELS: dict[HeatModel, str] = {
    "constant": "c_p constante a 25 °C",
    "mean": "c_p a la temperatura media",
    "variable": "c_p variable (polinomios NASA)",
}

PairKind = Literal["pT", "pv", "Tv"]

_HE_M_KG_PER_MOL = 4.002602e-3


def _num(x: float, fmt: str = ".3g") -> str:
    """Un número con coma decimal, para los textos."""
    return f"{x:{fmt}}".replace(".", ",")


def _pct(x: float, decimals: int = 1) -> str:
    return f"{100.0 * x:.{decimals}f} %".replace(".", ",")


def _temp(T_K: float) -> str:
    """«300,0 K (26,9 °C)»."""
    return f"{T_K:.1f} K ({T_K - 273.15:.1f} °C)".replace(".", ",")


# ---------------------------------------------------------------------
# El gas
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IdealGas:
    """Un gas ideal: una especie de la NASA, una mezcla fija (el aire) o el helio.

    Las propiedades son por kg y en SI; h, u y s° valen 0 a 25 °C.
    """

    key: str
    name: str
    formula: str
    coolprop: str
    composition: tuple[tuple[str, float], ...] = ()
    monatomic: bool = False

    @property
    def M(self) -> float:
        """Masa molar (kg/mol): Σ y_i·M_i (vademecum §5.2)."""
        if self.monatomic:
            return _HE_M_KG_PER_MOL
        return sum(y * thermo.species(k).M_kg_per_mol for k, y in self.composition)

    @property
    def R(self) -> float:
        """Constante del gas (J/(kg·K)): R = R_u/M (vademecum §4.2)."""
        return R_U / self.M

    @property
    def T_min_K(self) -> float:
        if self.monatomic:
            return 200.0
        return max(thermo.species(k).T_min_K for k, _ in self.composition)

    @property
    def T_max_K(self) -> float:
        if self.monatomic:
            return 6000.0
        return min(thermo.species(k).T_max_K for k, _ in self.composition)

    def check_T(self, T_K: float, what: str = "La temperatura") -> None:
        """Error para el alumno si T está fuera del rango de los polinomios."""
        if not (math.isfinite(T_K) and self.T_min_K - 1e-9 <= T_K <= self.T_max_K + 1e-9):
            raise ValueError(
                f"{what}, {_temp(T_K)}, está fuera del rango de los polinomios NASA del "
                f"{self.name.lower()}: de {self.T_min_K:.0f} K ({self.T_min_K - 273.15:.0f} °C) "
                f"a {self.T_max_K:.0f} K."
            )

    def _molar(self, fn: Callable[[Any, float], float], T_K: float) -> float:
        return sum(y * fn(thermo.species(k), T_K) for k, y in self.composition)

    def cp(self, T_K: float) -> float:
        """c_p (J/(kg·K)) a T."""
        if self.monatomic:
            return 2.5 * self.R
        return self._molar(lambda s, T: s.cp(T), T_K) / self.M

    def cv(self, T_K: float) -> float:
        """c_v = c_p − R (J/(kg·K)), la relación de Mayer (vademecum §4.5)."""
        return self.cp(T_K) - self.R

    def k(self, T_K: float) -> float:
        """k = c_p/c_v."""
        return self.cp(T_K) / self.cv(T_K)

    def h(self, T_K: float) -> float:
        """h(T) − h(25 °C) (J/kg): ∫c_p dT desde 25 °C (vademecum §4.7)."""
        if self.monatomic:
            return 2.5 * self.R * (T_K - T_REF_K)
        return self._molar(lambda s, T: s.delta_h(T), T_K) / self.M

    def u(self, T_K: float) -> float:
        """u(T) − u(25 °C) (J/kg): h − R·(T − 25 °C), porque h = u + R·T (§4.3)."""
        return self.h(T_K) - self.R * (T_K - T_REF_K)

    def s0(self, T_K: float) -> float:
        """s°(T) (J/(kg·K)): ∫c_p dT/T desde 25 °C (vademecum §10.5.2)."""
        if self.monatomic:
            return 2.5 * self.R * math.log(T_K / T_REF_K)
        return (
            self._molar(lambda s, T: s.s0(T), T_K) - self._molar(lambda s, T: s.s0(T), T_REF_K)
        ) / self.M

    def pr(self, T_K: float) -> float:
        """Presión relativa p_r = exp(s°/R), con p_r(25 °C) = 1 (vademecum §10.5.3)."""
        return math.exp(self.s0(T_K) / self.R)

    def vr(self, T_K: float) -> float:
        """Volumen relativo v_r = T/p_r (K) (vademecum §10.5.3)."""
        return T_K / self.pr(T_K)

    @property
    def cp_ref(self) -> float:
        """El c_p constante: a 25 °C, el de la tabla del vademecum §4.6."""
        return self.cp(T_REF_K)


_AIR = (("N2", 0.7808), ("O2", 0.2095), ("Ar", 0.0093), ("CO2", 0.0004))

#: Los gases de la página: los 9 de la tabla del vademecum §4.6 y algunos más.
IDEAL_GASES: dict[str, IdealGas] = {
    g.key: g
    for g in (
        IdealGas("air", "Aire", "—", "Air", _AIR),
        IdealGas("Ar", "Argón", "Ar", "Argon", (("Ar", 1.0),)),
        IdealGas("CO2", "Dióxido de carbono", "CO₂", "CarbonDioxide", (("CO2", 1.0),)),
        IdealGas("He", "Helio", "He", "Helium", monatomic=True),
        IdealGas("H2", "Hidrógeno", "H₂", "Hydrogen", (("H2", 1.0),)),
        IdealGas("CH4", "Metano", "CH₄", "Methane", (("CH4", 1.0),)),
        IdealGas("N2", "Nitrógeno", "N₂", "Nitrogen", (("N2", 1.0),)),
        IdealGas("O2", "Oxígeno", "O₂", "Oxygen", (("O2", 1.0),)),
        IdealGas("H2O", "Vapor de agua", "H₂O", "Water", (("H2O", 1.0),)),
        IdealGas("CO", "Monóxido de carbono", "CO", "CarbonMonoxide", (("CO", 1.0),)),
        IdealGas("NH3", "Amoníaco", "NH₃", "Ammonia", (("NH3", 1.0),)),
        IdealGas("SO2", "Dióxido de azufre", "SO₂", "SulfurDioxide", (("SO2", 1.0),)),
        IdealGas("C2H6", "Etano", "C₂H₆", "Ethane", (("C2H6", 1.0),)),
        IdealGas("C3H8", "Propano", "C₃H₈", "Propane", (("C3H8", 1.0),)),
        IdealGas("n-C4H10", "n-Butano", "C₄H₁₀", "n-Butane", (("n-C4H10", 1.0),)),
    )
}
#: Los de la tabla del vademecum §4.6, en su orden.
VADEMECUM_GASES: tuple[str, ...] = ("air", "Ar", "CO2", "He", "H2", "CH4", "N2", "O2", "H2O")

#: Presión de referencia de la entropía: s(T, p°) = s°(T).
P_REF_PA: float = 1.0e5

#: Los modelos de los procesos y las mezclas: c_p a 25 °C o variable.
GasModel = Literal["constant", "variable"]


def model_cp(g: IdealGas, T_K: float, model: GasModel) -> float:
    """c_p (J/(kg·K)) con el modelo: a 25 °C o a T."""
    return g.cp_ref if model == "constant" else g.cp(T_K)


def model_h(g: IdealGas, T_K: float, model: GasModel) -> float:
    """h − h(25 °C) (J/kg) con el modelo."""
    return g.cp_ref * (T_K - T_REF_K) if model == "constant" else g.h(T_K)


def model_u(g: IdealGas, T_K: float, model: GasModel) -> float:
    """u − u(25 °C) = h − R·(T − 25 °C) (J/kg) con el modelo."""
    return model_h(g, T_K, model) - g.R * (T_K - T_REF_K)


def model_s0(g: IdealGas, T_K: float, model: GasModel) -> float:
    """s°(T) (J/(kg·K)) con el modelo: c_p·ln(T/298,15 K) o el polinomio."""
    return g.cp_ref * math.log(T_K / T_REF_K) if model == "constant" else g.s0(T_K)


def model_s(g: IdealGas, T_K: float, p_Pa: float, model: GasModel) -> float:
    """s(T, p) = s°(T) − R·ln(p/p°) (J/(kg·K)), con s = 0 a 25 °C y 1 bar (§10.5)."""
    return model_s0(g, T_K, model) - g.R * math.log(p_Pa / P_REF_PA)


def gas(key: str) -> IdealGas:
    """El gas ``key`` (``"air"``, ``"CO2"``…)."""
    if key not in IDEAL_GASES:
        raise ValueError(f"El gas {key!r} no está en la lista: {', '.join(IDEAL_GASES)}.")
    return IDEAL_GASES[key]


# ---------------------------------------------------------------------
# Estados
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GasState:
    """Un estado del gas ideal: p, T y v con p·v = R·T."""

    p_Pa: float
    T_K: float
    v_m3_per_kg: float


@dataclass(frozen=True)
class StatePair:
    """Cómo se da un estado: dos de p, T y v (``a`` es el primero del par)."""

    pair: PairKind
    a: float
    b: float


def gas_state(
    g: IdealGas,
    *,
    p_Pa: float | None = None,
    T_K: float | None = None,
    v_m3_per_kg: float | None = None,
    label: str = "",
) -> GasState:
    """El estado con dos de p, T y v (vademecum §4.2: p·v = R·T).

    Raises
    ------
    ValueError
        Si no se dan exactamente dos datos, si alguno no es positivo o si T queda
        fuera del rango de los polinomios.
    """
    given = {k: v for k, v in (("p", p_Pa), ("T", T_K), ("v", v_m3_per_kg)) if v is not None}
    where = f" del estado {label}" if label else ""
    if len(given) != 2:
        raise ValueError(f"Para el estado{where} hacen falta exactamente dos de p, T y v.")
    names = {"p": "La presión", "T": "La temperatura", "v": "El volumen específico"}
    for k, value in given.items():
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{names[k]}{where} tiene que ser positiva (absoluta).")
    R = g.R
    if p_Pa is not None and T_K is not None:
        p, T = p_Pa, T_K
    elif p_Pa is not None and v_m3_per_kg is not None:
        p, T = p_Pa, p_Pa * v_m3_per_kg / R
    else:
        assert T_K is not None and v_m3_per_kg is not None
        p, T = R * T_K / v_m3_per_kg, T_K
    g.check_T(T, f"La temperatura{where}")
    return GasState(p_Pa=p, T_K=T, v_m3_per_kg=R * T / p)


def state_from_pair(g: IdealGas, pair: StatePair, label: str = "") -> GasState:
    """El estado a partir de un :class:`StatePair`."""
    if pair.pair == "pT":
        return gas_state(g, p_Pa=pair.a, T_K=pair.b, label=label)
    if pair.pair == "pv":
        return gas_state(g, p_Pa=pair.a, v_m3_per_kg=pair.b, label=label)
    return gas_state(g, T_K=pair.a, v_m3_per_kg=pair.b, label=label)


def ideal_gas_check(g: IdealGas, p_Pa: float, T_K: float) -> tuple[float | None, str]:
    """Z = p·v/(R·T) del gas real (CoolProp) y su fase, para ver si vale el ideal.

    Devuelve ``(None, motivo)`` si CoolProp no lo puede evaluar (p. ej. el CO₂
    bajo su punto triple).
    """
    try:
        z = float(CP.PropsSI("Z", "T", T_K, "P", p_Pa, g.coolprop))
        phase = str(CP.PhaseSI("T", T_K, "P", p_Pa, g.coolprop))
    except ValueError as exc:
        return None, str(exc).split("\n")[0][:120]
    return z, phase


# ---------------------------------------------------------------------
# Entre dos estados
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyChange:
    """Δu, Δh y Δs con un modelo de c_p.

    ``cp`` es el que usa el modelo; con c_p variable es el medio, Δh/ΔT.
    """

    model: HeatModel
    cp: float
    cv: float
    du: float
    dh: float
    ds: float


@dataclass(frozen=True)
class StateChangeInputs:
    """El gas y sus dos estados."""

    gas_key: str
    state1: StatePair
    state2: StatePair


@dataclass(frozen=True)
class StateChange:
    """Un gas ideal entre dos estados, con los tres modelos de c_p."""

    inputs: StateChangeInputs
    gas: IdealGas
    state1: GasState
    state2: GasState
    T_mean_K: float
    constant: PropertyChange
    mean: PropertyChange
    variable: PropertyChange
    cp1: float
    cp2: float
    s01: float
    s02: float
    pr1: float
    pr2: float
    vr1: float
    vr2: float
    Z1: float | None
    Z2: float | None
    phase1: str
    phase2: str
    notes: tuple[str, ...]

    @property
    def changes(self) -> tuple[PropertyChange, PropertyChange, PropertyChange]:
        return (self.constant, self.mean, self.variable)

    def error(self, model: HeatModel, what: Literal["du", "dh", "ds"]) -> float | None:
        """Error relativo de un modelo contra el de c_p variable (None si el exacto es 0)."""
        exact = getattr(self.variable, what)
        approx = getattr(getattr(self, model), what)
        if abs(exact) < 1e-9 * max(1.0, abs(approx)):
            return None
        return approx / exact - 1.0


def _constant_change(
    model: HeatModel, g: IdealGas, cp: float, s1: GasState, s2: GasState
) -> PropertyChange:
    cv = cp - g.R
    dT = s2.T_K - s1.T_K
    ds = cp * math.log(s2.T_K / s1.T_K) - g.R * math.log(s2.p_Pa / s1.p_Pa)
    return PropertyChange(model=model, cp=cp, cv=cv, du=cv * dT, dh=cp * dT, ds=ds)


def _z_note(g: IdealGas, label: str, z: float | None, phase: str) -> str | None:
    if z is None:
        return (
            f"En el estado {label}, CoolProp no puede evaluar el {g.name.lower()} real "
            f"({phase}): revisá que tenga sentido como gas a esa p y T."
        )
    if phase in ("liquid", "twophase"):
        return (
            f"En el estado {label}, el {g.name.lower()} real a esa p y T es líquido (CoolProp): "
            "el modelo de gas ideal no sirve, salvo como vapor a su presión parcial en una "
            "mezcla (como en el aire húmedo)."
        )
    if abs(z - 1.0) > 0.05:
        return (
            f"En el estado {label}, el gas real se aparta {_pct(abs(z - 1.0), 0)} del ideal "
            f"(Z = p·v/(R·T) = {_num(z, '.3f')}, CoolProp): el modelo de gas ideal da números "
            "aproximados."
        )
    return None


def solve_state_change(inputs: StateChangeInputs) -> StateChange:
    """Δu, Δh y Δs entre dos estados con los tres modelos (Çengel §7-9).

    - c_p constante: Δh = c_p·ΔT, Δu = c_v·ΔT, Δs = c_p·ln(T₂/T₁) − R·ln(p₂/p₁)
      (vademecum §4.5 y §10.5.1);
    - c_p variable: Δh = h(T₂) − h(T₁), Δu = Δh − R·ΔT, Δs = s°₂ − s°₁ − R·ln(p₂/p₁)
      (§4.7 y §10.5.2).
    """
    g = gas(inputs.gas_key)
    s1 = state_from_pair(g, inputs.state1, "1")
    s2 = state_from_pair(g, inputs.state2, "2")
    T_mean = 0.5 * (s1.T_K + s2.T_K)
    constant = _constant_change("constant", g, g.cp_ref, s1, s2)
    mean = _constant_change("mean", g, g.cp(T_mean), s1, s2)
    dT = s2.T_K - s1.T_K
    dh = g.h(s2.T_K) - g.h(s1.T_K)
    du = dh - g.R * dT
    s01, s02 = g.s0(s1.T_K), g.s0(s2.T_K)
    ds = s02 - s01 - g.R * math.log(s2.p_Pa / s1.p_Pa)
    cp_eff = dh / dT if abs(dT) > 1e-9 else g.cp(s1.T_K)
    variable = PropertyChange(model="variable", cp=cp_eff, cv=cp_eff - g.R, du=du, dh=dh, ds=ds)
    z1, phase1 = ideal_gas_check(g, s1.p_Pa, s1.T_K)
    z2, phase2 = ideal_gas_check(g, s2.p_Pa, s2.T_K)

    notes: list[str] = []
    if g.monatomic:
        notes.append(
            "El helio es monoatómico: c_p = 5/2·R no cambia con la temperatura, así que los "
            "tres modelos dan lo mismo."
        )
    elif abs(dT) < 1e-9:
        notes.append(
            "Con T₁ = T₂, Δu = Δh = 0 en un gas ideal (u y h dependen solo de T, vademecum "
            "§4.3): solo cambia la entropía, Δs = −R·ln(p₂/p₁)."
        )
    else:
        err_c = constant.dh / dh - 1.0
        err_m = mean.dh / dh - 1.0
        rise = g.cp(max(s1.T_K, s2.T_K)) / g.cp(min(s1.T_K, s2.T_K)) - 1.0
        notes.append(
            f"El c_p del {g.name.lower()} cambia {_pct(abs(rise))} entre "
            f"{min(s1.T_K, s2.T_K):.0f} K y {max(s1.T_K, s2.T_K):.0f} K. Con c_p constante a "
            f"25 °C, Δh sale {_pct(abs(err_c))} {'más bajo' if err_c < 0 else 'más alto'} que con "
            f"c_p variable; con el c_p a la temperatura media, {_pct(abs(err_m), 2)}."
        )
    for label, z, phase in (("1", z1, phase1), ("2", z2, phase2)):
        note = _z_note(g, label, z, phase)
        if note:
            notes.append(note)

    return StateChange(
        inputs=inputs,
        gas=g,
        state1=s1,
        state2=s2,
        T_mean_K=T_mean,
        constant=constant,
        mean=mean,
        variable=variable,
        cp1=g.cp(s1.T_K),
        cp2=g.cp(s2.T_K),
        s01=s01,
        s02=s02,
        pr1=g.pr(s1.T_K),
        pr2=g.pr(s2.T_K),
        vr1=g.vr(s1.T_K),
        vr2=g.vr(s2.T_K),
        Z1=z1,
        Z2=z2,
        phase1=phase1,
        phase2=phase2,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class StateExample:
    """Un ejemplo precargado, con lo que muestra."""

    inputs: StateChangeInputs
    note: str = ""


_ATM = 101_325.0
C = 273.15

STATE_EXAMPLES: dict[str, StateExample] = {
    "Aire calentado de 300 K a 1500 K a 1 atm": StateExample(
        StateChangeInputs("air", StatePair("pT", _ATM, 300.0), StatePair("pT", _ATM, 1500.0)),
        "Cuánto pesa suponer c_p constante cuando la temperatura cambia mucho.",
    ),
    "Aire comprimido de 100 kPa y 17 °C a 600 kPa y 57 °C (Çengel §7-9)": StateExample(
        StateChangeInputs(
            "air", StatePair("pT", 100e3, 17.0 + C), StatePair("pT", 600e3, 57.0 + C)
        ),
        "Çengel da Δs = −0,3842 kJ/(kg·K) con la tabla del aire y con el c_p medio.",
    ),
    "Nitrógeno calentado en un tanque rígido de 300 K a 600 K": StateExample(
        StateChangeInputs("N2", StatePair("pT", 200e3, 300.0), StatePair("pT", 400e3, 600.0)),
        "Con v constante la presión se duplica al duplicarse T (p·v = R·T).",
    ),
    "Vapor de agua de los humos, de 400 K a 1200 K a 10 kPa": StateExample(
        StateChangeInputs("H2O", StatePair("pT", 10e3, 400.0), StatePair("pT", 10e3, 1200.0)),
        "A su presión parcial el vapor se comporta como gas ideal.",
    ),
    "Dióxido de carbono que se enfría de 1500 K a 400 K a 1 atm": StateExample(
        StateChangeInputs("CO2", StatePair("pT", _ATM, 1500.0), StatePair("pT", _ATM, 400.0)),
    ),
    "Hidrógeno comprimido de 1 bar y 300 K a 50 bar y 320 K": StateExample(
        StateChangeInputs("H2", StatePair("pT", 1e5, 300.0), StatePair("pT", 50e5, 320.0)),
    ),
    "Helio de 300 K a 1000 K a 100 kPa": StateExample(
        StateChangeInputs("He", StatePair("pT", 100e3, 300.0), StatePair("pT", 100e3, 1000.0)),
        "Monoatómico: c_p constante exacto.",
    ),
}


def state_change_to_dict(result: StateChange) -> dict[str, Any]:
    """El resultado para exportar (SI)."""
    g = result.gas
    out: dict[str, Any] = {
        "gas": g.name,
        "M_kg_per_kmol": g.M * 1000.0,
        "R_J_per_kgK": g.R,
        "cp_25C_J_per_kgK": g.cp_ref,
    }
    for label, st, cp, s0, pr, vr, z in (
        ("1", result.state1, result.cp1, result.s01, result.pr1, result.vr1, result.Z1),
        ("2", result.state2, result.cp2, result.s02, result.pr2, result.vr2, result.Z2),
    ):
        out[f"estado_{label}"] = {
            "p_Pa": st.p_Pa,
            "T_K": st.T_K,
            "v_m3_per_kg": st.v_m3_per_kg,
            "cp_J_per_kgK": cp,
            "s0_J_per_kgK": s0,
            "pr": pr,
            "vr_K": vr,
            "Z_coolprop": z,
        }
    for change in result.changes:
        out[change.model] = {
            "cp_J_per_kgK": change.cp,
            "cv_J_per_kgK": change.cv,
            "du_J_per_kg": change.du,
            "dh_J_per_kg": change.dh,
            "ds_J_per_kgK": change.ds,
        }
    out["notas"] = list(result.notes)
    return out
