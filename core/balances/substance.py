"""Sustancias para los balances: fluido real, gas ideal o incompresible.

Tres modelos con la misma interfaz, para que un mismo balance se resuelva con cualquiera:

- **fluido real**: la ecuación de estado de CoolProp (Bell et al., 2014) de los fluidos
  de la app, con la región de cada estado (:mod:`core.fluids`);
- **gas ideal** (vademecum §4 y §10.5): los gases de :mod:`core.gases.ideal`, con c_p
  constante a 25 °C o variable (polinomios NASA-9 de McBride et al., 2002);
- **incompresible** (vademecum §13): v y c constantes; c y ρ a 300 K de Incropera et
  al. (2007), tablas A.1 (metales), A.3 (no metales) y A.5–A.6 (líquidos), o a
  elección.

Un estado se da con dos propiedades (el par, en el orden de su código: ``"PH"`` es p y
h). En los tres modelos h = u + p·v, así que los balances con caudal (la entalpía que
entra a un tanque contra su energía interna) son coherentes:

- fluido real: la referencia de CoolProp para cada fluido;
- gas ideal: u = 0 a 25 °C y h = u + R·T; s = 0 a 25 °C y 1 bar (s = s°(T) − R·ln(p/p°));
- incompresible: u = c·(T − 25 °C), h = u + p·v y s = c·ln(T/298,15 K).

Las diferencias (lo único que entra en un balance de energía o de entropía de una
sustancia) no dependen de la referencia. Todo en SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import CoolProp
from scipy.optimize import brentq

from core import fluids as fl
from core.gases import ideal as gi

__all__ = [
    "INCOMPRESSIBLES",
    "KINDS",
    "PAIR_NAMES",
    "PairCode",
    "Incompressible",
    "Substance",
    "SubstanceKind",
    "ThermoState",
    "allowed_pairs",
    "fluid_substance",
    "balance_state",
    "of",
    "ideal_gas_substance",
    "incompressible_substance",
    "saturation_dome",
    "state",
]

SubstanceKind = Literal["fluid", "ideal_gas", "incompressible"]
KINDS: dict[SubstanceKind, str] = {
    "fluid": "Fluido real (ecuación de estado)",
    "ideal_gas": "Gas ideal",
    "incompressible": "Líquido o sólido incompresible",
}

#: Los pares de un estado; las letras van en el orden de los dos datos.
PairCode = Literal["TP", "PV", "TV", "PH", "PS", "TS", "PU", "PX", "TX", "VU", "VS", "VX"]
PAIR_NAMES: dict[PairCode, str] = {
    "TP": "T y p",
    "PV": "p y v",
    "TV": "T y v",
    "PH": "p y h",
    "PS": "p y s",
    "TS": "T y s",
    "PU": "p y u",
    "PX": "p y x",
    "TX": "T y x",
    "VU": "v y u",
    "VS": "v y s",
    "VX": "v y x",
}

T_REF_K: float = gi.T_REF_K  # 25 °C
_P_ATM: float = 101_325.0


def _num(x: float, fmt: str = ".4g") -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")


# ---------------------------------------------------------------------
# Incompresibles
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Incompressible:
    """Un líquido o un sólido con c y ρ constantes (vademecum §13)."""

    key: str
    name: str
    c: float  # J/(kg·K)
    rho: float  # kg/m³
    liquid: bool
    noun: str  # con el artículo, para las oraciones

    @property
    def v(self) -> float:
        """Volumen específico (m³/kg)."""
        return 1.0 / self.rho


#: c y ρ a 300 K: Incropera et al. (2007), tablas A.1, A.3, A.5 y A.6.
INCOMPRESSIBLES: dict[str, Incompressible] = {
    m.key: m
    for m in (
        Incompressible("water", "Agua líquida", 4179.0, 997.0, True, "el agua"),
        Incompressible("oil", "Aceite de motor", 1909.0, 884.1, True, "el aceite"),
        Incompressible("glycol", "Etilenglicol", 2415.0, 1114.4, True, "el etilenglicol"),
        Incompressible("glycerin", "Glicerina", 2427.0, 1259.9, True, "la glicerina"),
        Incompressible("mercury", "Mercurio", 139.3, 13529.0, True, "el mercurio"),
        Incompressible("iron", "Hierro", 447.0, 7870.0, False, "el hierro"),
        Incompressible("carbon_steel", "Acero al carbono", 434.0, 7854.0, False, "el acero"),
        Incompressible("stainless", "Acero inoxidable AISI 304", 477.0, 7900.0, False, "el acero"),
        Incompressible("aluminum", "Aluminio", 903.0, 2702.0, False, "el aluminio"),
        Incompressible("copper", "Cobre", 385.0, 8933.0, False, "el cobre"),
        Incompressible("lead", "Plomo", 129.0, 11340.0, False, "el plomo"),
        Incompressible("silver", "Plata", 235.0, 10500.0, False, "la plata"),
        Incompressible("brick", "Ladrillo común", 835.0, 1920.0, False, "el ladrillo"),
        Incompressible("concrete", "Hormigón", 880.0, 2300.0, False, "el hormigón"),
        Incompressible("glass", "Vidrio", 750.0, 2500.0, False, "el vidrio"),
        Incompressible("granite", "Granito", 775.0, 2630.0, False, "el granito"),
        Incompressible("ice", "Hielo (a 0 °C)", 2040.0, 920.0, False, "el hielo"),
        Incompressible("oak", "Madera de roble", 2385.0, 545.0, False, "la madera"),
    )
}


# ---------------------------------------------------------------------
# Sustancia
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Substance:
    """Qué se calcula: un fluido real, un gas ideal o un incompresible.

    ``key`` es el nombre de CoolProp (fluido real), la clave de
    :data:`core.gases.ideal.IDEAL_GASES` (gas ideal) o la de :data:`INCOMPRESSIBLES`
    (``"custom"`` con ``c`` y ``rho`` a elección).
    """

    kind: SubstanceKind
    key: str
    model: gi.GasModel = "variable"  # gas ideal: c_p constante a 25 °C o variable
    c: float | None = None  # incompresible a elección (J/(kg·K))
    rho: float | None = None  # incompresible a elección (kg/m³)

    @property
    def gas(self) -> gi.IdealGas:
        return gi.gas(self.key)

    @property
    def material(self) -> Incompressible:
        if self.key == "custom":
            assert self.c is not None and self.rho is not None
            return Incompressible(
                "custom", "Material a elección", self.c, self.rho, False, "el material"
            )
        base = INCOMPRESSIBLES[self.key]
        if self.c is None and self.rho is None:
            return base
        return Incompressible(
            base.key,
            base.name,
            self.c if self.c is not None else base.c,
            self.rho if self.rho is not None else base.rho,
            base.liquid,
            base.noun,
        )

    @property
    def label(self) -> str:
        if self.kind == "fluid":
            return fl.FLUID_NAMES_ES.get(self.key, self.key)
        if self.kind == "ideal_gas":
            how = "c_p constante" if self.model == "constant" else "c_p variable"
            return f"{self.gas.name} (gas ideal, {how})"
        return self.material.name

    @property
    def noun(self) -> str:
        """Con el artículo: «el agua», «el aire», «el hierro»."""
        if self.kind == "fluid":
            return fl.fluid_with_article(self.key)
        if self.kind == "ideal_gas":
            return f"el {self.gas.name.lower()}"  # todos los gases de la lista son masculinos
        return self.material.noun


def fluid_substance(key: str) -> Substance:
    """Un fluido real de la app (nombre de CoolProp)."""
    if key not in fl.SUPPORTED_FLUIDS:
        options = ", ".join(fl.SUPPORTED_FLUIDS)
        raise ValueError(f"El fluido {key!r} no está en la lista: {options}.")
    return Substance("fluid", key)


def ideal_gas_substance(key: str, model: gi.GasModel = "variable") -> Substance:
    """Un gas ideal de :data:`core.gases.ideal.IDEAL_GASES`."""
    gi.gas(key)  # valida
    return Substance("ideal_gas", key, model=model)


def incompressible_substance(
    key: str, c: float | None = None, rho: float | None = None
) -> Substance:
    """Un incompresible de la tabla (con c o ρ cambiados si se dan) o ``"custom"``.

    Raises
    ------
    ValueError
        Si el material no existe o c o ρ no son positivos.
    """
    if key != "custom" and key not in INCOMPRESSIBLES:
        raise ValueError(f"El material {key!r} no está en la lista: {', '.join(INCOMPRESSIBLES)}.")
    if key == "custom" and (c is None or rho is None):
        raise ValueError("Para un material a elección hacen falta su c y su densidad.")
    for name, value in (("El calor específico c", c), ("La densidad", rho)):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} tiene que ser positivo.")
    return Substance("incompressible", key, c=c, rho=rho)


def allowed_pairs(sub: Substance) -> tuple[PairCode, ...]:
    """Los pares con que se puede dar un estado de ``sub``."""
    if sub.kind == "fluid":
        return ("TP", "PV", "TV", "PH", "PS", "TS", "PU", "PX", "TX", "VU", "VS", "VX")
    if sub.kind == "ideal_gas":
        return ("TP", "PV", "TV", "PH", "PS", "TS", "PU", "VU", "VS")
    return ("TP", "PH", "PS", "PU")


# ---------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------

REGION_GAS = "ideal_gas"
REGION_LIQUID = "incompressible_liquid"
REGION_SOLID = "incompressible_solid"
_REGION_ES = {
    REGION_GAS: "Gas ideal",
    REGION_LIQUID: "Líquido incompresible",
    REGION_SOLID: "Sólido incompresible",
}


@dataclass(frozen=True)
class ThermoState:
    """Un estado: p, T, v, u, h, s (SI), el título (solo en la campana) y la región."""

    p: float
    T: float
    v: float
    u: float
    h: float
    s: float
    x: float | None
    region: str  # la de core.fluids (fluido real) o la del modelo
    label: str = ""

    @property
    def region_es(self) -> str:
        return fl.REGION_LABELS_ES.get(self.region, _REGION_ES.get(self.region, self.region))

    @property
    def is_two_phase(self) -> bool:
        return self.region in ("saturated_liquid", "saturated_mixture", "saturated_vapor")

    def relabel(self, label: str) -> ThermoState:
        return ThermoState(
            self.p, self.T, self.v, self.u, self.h, self.s, self.x, self.region, label
        )


def state(sub: Substance, pair: PairCode, a: float, b: float, label: str = "") -> ThermoState:
    """El estado de ``sub`` con el par ``pair`` = (``a``, ``b``) en SI.

    Raises
    ------
    ValueError
        Si el par no vale para la sustancia, un dato no es positivo o el estado queda
        fuera del rango del modelo, con un mensaje para el alumno.
    """
    if pair not in allowed_pairs(sub):
        raise ValueError(_pair_error(sub, pair))
    for letter, value in zip(pair, (a, b), strict=True):
        if not math.isfinite(value):
            raise ValueError(f"Falta un dato del estado{_where(label)}.")
        if letter in "TPV" and value <= 0.0:
            names = {"T": "La temperatura", "P": "La presión", "V": "El volumen específico"}
            raise ValueError(f"{names[letter]}{_where(label)} tiene que ser positiva (absoluta).")
        if letter == "X" and not 0.0 <= value <= 1.0:
            raise ValueError(f"El título{_where(label)} tiene que estar entre 0 y 1.")
    if sub.kind == "fluid":
        return _fluid_state(sub, pair, a, b, label)
    if sub.kind == "ideal_gas":
        return _gas_state(sub, pair, a, b, label)
    return _incompressible_state(sub, pair, a, b, label)


def _where(label: str) -> str:
    return f" del estado {label}" if label else ""


def of(sub: Substance) -> str:
    """«del agua», «del hierro», «de la glicerina»: «de» con el artículo contraído."""
    noun = sub.noun
    return f"del {noun[3:]}" if noun.startswith("el ") else f"de {noun}"


def _pair_error(sub: Substance, pair: str) -> str:
    if sub.kind == "ideal_gas" and "X" in pair:
        return "El título x es de un fluido dentro de la campana: un gas ideal no tiene."
    if sub.kind == "incompressible":
        if "X" in pair:
            return "Un incompresible no cambia de fase en este modelo: no tiene título."
        if "V" in pair:
            return (
                f"El volumen específico {of(sub)} es fijo (v = 1/ρ, vademecum §13): no sirve "
                "como dato del estado. Usá T o p con h, s o u."
            )
        if pair == "TS":
            return f"La entropía {of(sub)} depende solo de T (s = c·ln T/T_ref): con T y s falta p."
    return f"El par {PAIR_NAMES.get(pair, pair)} no sirve para {sub.noun}."  # type: ignore[call-overload]


# --- fluido real ---------------------------------------------------------

_FLUIDS_PAIR = {"TP", "PV", "TV", "PH", "PS", "TS", "PU", "PX", "TX"}
_COOLPROP_EXTRA = {
    "VU": CoolProp.DmassUmass_INPUTS,
    "VS": CoolProp.DmassSmass_INPUTS,
    "VX": CoolProp.DmassQ_INPUTS,
}


def _fluid_state(sub: Substance, pair: PairCode, a: float, b: float, label: str) -> ThermoState:
    if pair in _FLUIDS_PAIR:
        k1, k2 = fl.PAIR_KWARGS[pair]
        try:
            fs = fl.fluid_state_from_pair(sub.key, pair, **{k1: a, k2: b})  # type: ignore[arg-type]
        except ValueError as exc:
            if not label:
                raise
            raise ValueError(f"Estado {label}: {exc}") from exc
        return ThermoState(
            fs.P_Pa,
            fs.T_K,
            fs.v_m3_per_kg,
            fs.u_J_per_kg,
            fs.h_J_per_kg,
            fs.s_J_per_kg_K,
            fs.x,
            fs.region,
            label,
        )
    # v y u, v y s, v y x: la API de bajo nivel (core.fluids no tiene estos pares)
    st = CoolProp.AbstractState("HEOS", sub.key)
    limits = fl.fluid_limits(sub.key)
    rho = 1.0 / a
    try:
        st.update(_COOLPROP_EXTRA[pair], rho, b)
        T, p = float(st.T()), float(st.p())
    except ValueError as exc:
        raise ValueError(
            f"No hay un estado {of(sub)} con {PAIR_NAMES[pair]} dados{_where(label)} "
            "dentro del rango de la ecuación de estado. Revisá los datos."
        ) from exc
    if not (limits.T_min_K * 0.999 <= T <= limits.T_max_K * 1.001 and p <= limits.P_max_Pa * 1.001):
        raise ValueError(
            f"El estado{_where(label)} {of(sub)} queda fuera del rango de la ecuación de "
            f"estado (T = {_num(T)} K, p = {_num(p / 1e5)} bar)."
        )
    region, x = fl.classify_region(int(st.phase()), float(st.Q()))
    return ThermoState(
        p,
        T,
        1.0 / float(st.rhomass()),
        float(st.umass()),
        float(st.hmass()),
        float(st.smass()),
        x,
        region,
        label,
    )


# --- gas ideal -----------------------------------------------------------


def _gas_T(sub: Substance, fn: str, target: float, label: str) -> float:
    """T con u, h (absoluta, h = u + R·T) o s° dados, en el rango de los polinomios."""
    g, model = sub.gas, sub.model
    R = g.R

    def value(T: float) -> float:
        if fn == "u":
            return gi.model_u(g, T, model)
        if fn == "h":
            return gi.model_u(g, T, model) + R * T
        return gi.model_s0(g, T, model)

    lo, hi = g.T_min_K, g.T_max_K
    f_lo, f_hi = value(lo) - target, value(hi) - target
    if f_lo > 0 or f_hi < 0:
        what = {"u": "la energía interna", "h": "la entalpía", "s": "la entropía"}[fn]
        raise ValueError(
            f"Con {what} dada, la temperatura{_where(label)} del {g.name.lower()} queda fuera "
            f"del rango de los polinomios ({lo:.0f} K a {hi:.0f} K)."
        )
    return float(brentq(lambda T: value(T) - target, lo, hi, xtol=1e-10, rtol=1e-13))


def _gas_state(sub: Substance, pair: PairCode, a: float, b: float, label: str) -> ThermoState:
    g, model = sub.gas, sub.model
    R = g.R
    p0 = gi.P_REF_PA
    if pair == "TP":
        T, p = a, b
    elif pair == "PV":
        p, T = a, a * b / R
    elif pair == "TV":
        T, p = a, R * a / b
    elif pair == "PH":
        p, T = a, _gas_T(sub, "h", b, label)
    elif pair == "PU":
        p, T = a, _gas_T(sub, "u", b, label)
    elif pair == "PS":
        p = a
        T = _gas_T(sub, "s", b + R * math.log(p / p0), label)
    elif pair == "TS":
        T = a
        g.check_T(T, f"La temperatura{_where(label)}")
        p = p0 * math.exp((gi.model_s0(g, T, model) - b) / R)
    elif pair == "VU":
        T = _gas_T(sub, "u", b, label)
        p = R * T / a
    else:  # VS: s = s°(T) − R·ln(R·T/(v·p°)), creciente en T
        v, s = a, b
        lo, hi = g.T_min_K, g.T_max_K

        def f(T: float) -> float:
            return gi.model_s0(g, T, model) - R * math.log(R * T / (v * p0)) - s

        if f(lo) > 0 or f(hi) < 0:
            raise ValueError(
                f"Con v y s dados, la temperatura{_where(label)} queda fuera del rango de los "
                "polinomios."
            )
        T = float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-13))
        p = R * T / v
    g.check_T(T, f"La temperatura{_where(label)}")
    u = gi.model_u(g, T, model)
    return ThermoState(
        p, T, R * T / p, u, u + R * T, gi.model_s(g, T, p, model), None, REGION_GAS, label
    )


# --- incompresible -------------------------------------------------------


def _incompressible_state(
    sub: Substance, pair: PairCode, a: float, b: float, label: str
) -> ThermoState:
    m = sub.material
    v = m.v
    if pair == "TP":
        T, p = a, b
    elif pair == "PH":
        p = a
        T = T_REF_K + (b - v * p) / m.c
    elif pair == "PU":
        p = a
        T = T_REF_K + b / m.c
    else:  # PS
        p = a
        T = T_REF_K * math.exp(b / m.c)
    if T <= 0.0:
        raise ValueError(
            f"Con esos datos la temperatura{_where(label)} saldría negativa (absoluta)."
        )
    u = m.c * (T - T_REF_K)
    region = REGION_LIQUID if m.liquid else REGION_SOLID
    return ThermoState(p, T, v, u, u + p * v, m.c * math.log(T / T_REF_K), None, region, label)


# --- estado que sale de un balance ------------------------------------------

_BALANCE_PROPS = {"H": ("h", "kJ/kg"), "U": ("u", "kJ/kg"), "S": ("s", "kJ/(kg·K)")}


def balance_state(
    sub: Substance, pair: PairCode, a: float, b: float, label: str, hint: str
) -> ThermoState:
    """Un estado cuya segunda propiedad (h, u o s) sale de un balance.

    Si no existe, el mensaje dice que es el balance el que no cierra con un estado real (y
    ``hint`` qué dato revisar), en vez del detalle de la ecuación de estado.
    """
    try:
        return state(sub, pair, a, b, label)
    except ValueError as exc:
        sym, unit = _BALANCE_PROPS.get(pair[1], (pair[1].lower(), ""))
        where = "la ecuación de estado" if sub.kind == "fluid" else "del modelo"
        raise ValueError(
            f"El balance de energía da {sym} = {_num(b / 1e3)} {unit} en el estado {label}, y "
            f"con ese valor no hay un estado posible {of(sub)} dentro del rango de {where}. "
            f"{hint}"
        ) from exc


# --- campana de saturación ------------------------------------------------


def saturation_dome(sub: Substance, points: int = 60) -> tuple[ThermoState, ...]:
    """La campana de un fluido real para los diagramas p–v y T–s: el líquido saturado del
    punto triple al crítico y el vapor saturado de vuelta (un solo trazo). Vacía si no es
    un fluido real.
    """
    if sub.kind != "fluid":
        return ()
    limits = fl.fluid_limits(sub.key)
    st = CoolProp.AbstractState("HEOS", sub.key)
    T_lo = max(limits.T_triple_K, limits.T_min_K) * 1.0005
    T_hi = limits.T_crit_K * 0.9995
    # más puntos cerca del crítico, donde la campana se cierra
    temps = [T_hi - (T_hi - T_lo) * (1.0 - k / (points - 1)) ** 1.6 for k in range(points)]
    liquid, vapor = [], []
    for T in temps:
        for q, out in ((0.0, liquid), (1.0, vapor)):
            try:
                st.update(CoolProp.QT_INPUTS, q, T)
            except ValueError:
                continue
            out.append(
                ThermoState(
                    float(st.p()),
                    T,
                    1.0 / float(st.rhomass()),
                    float(st.umass()),
                    float(st.hmass()),
                    float(st.smass()),
                    q,
                    "saturated_liquid" if q == 0.0 else "saturated_vapor",
                )
            )
    return tuple(liquid + vapor[::-1])
