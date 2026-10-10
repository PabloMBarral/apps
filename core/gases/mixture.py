"""Mezclas de gases ideales y su mezcla adiabática (vademecum §5 y §11).

**Composición** (vademecum §5.1 a §5.6): x_i es la fracción másica e y_i la molar
(la notación del vademecum, al revés que Çengel); M_M = Σ y_i·M_i, R_M = R_u/M_M =
Σ x_i·R_i, c_p,M = Σ x_i·c_p,i, la presión parcial de Dalton p_i = y_i·p y el
volumen parcial de Amagat V_i = y_i·V.

**Entropía** (§5.7): s_M = Σ x_i·s_i(T, p_i) = Σ x_i·[s_i(T, p) − R_i·ln y_i], con
s_i(T, p) = s°_i(T) − R_i·ln(p/p°), s° = 0 a 25 °C y p° = 1 bar.

**Mezcla adiabática** de 2 a 4 corrientes (Çengel, *Termodinámica*, §13-3):
- en un tanque rígido dividido que se abre: U = cte y V = ΣV_i;
- en una cámara de flujo permanente: H = cte y la presión de salida es un dato.
La entropía generada se parte en lo que cuesta llevar cada gas a la T y la p
finales y lo que cuesta mezclarlos (−R_i·ln y_i); dos corrientes del mismo gas
no suman entropía de mezcla. X_dest = T₀·S_gen (Gouy–Stodola, vademecum §11.8).

Con c_p constante (a 25 °C) o variable (polinomios NASA), como ``core.gases.ideal``.
Todo en SI.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from scipy.optimize import brentq

from core.gases.ideal import (
    P_REF_PA,
    R_U,
    GasModel,
    IdealGas,
    gas,
)
from core.gases.ideal import model_h as _h
from core.gases.ideal import model_s as _s
from core.gases.ideal import model_u as _u

__all__ = [
    "COMPOSITION_EXAMPLES",
    "MIXING_EXAMPLES",
    "P_REF_PA",
    "Component",
    "ComponentResult",
    "CompositionBasis",
    "MixingInputs",
    "MixingKind",
    "MixingResult",
    "MixingStream",
    "MixtureExample",
    "MixtureInputs",
    "MixtureResult",
    "MixingExample",
    "StreamResult",
    "mixing_to_dict",
    "mixture_to_dict",
    "solve_mixing",
    "solve_mixture",
]

CompositionBasis = Literal["mass", "moles", "mass_fraction", "mole_fraction"]
MixingKind = Literal["tank", "flow"]


def _num(x: float, fmt: str = ".3g") -> str:
    return f"{x:{fmt}}".replace(".", ",")


def _pct(x: float, decimals: int = 1) -> str:
    return f"{100.0 * x:.{decimals}f} %".replace(".", ",")


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


# ---------------------------------------------------------------------
# Composición
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """Un gas de la mezcla y cuánto hay: kg, mol o fracción, según la base."""

    gas_key: str
    amount: float


@dataclass(frozen=True)
class MixtureInputs:
    """La composición en una base y el estado (T, p) de la mezcla."""

    components: tuple[Component, ...]
    basis: CompositionBasis
    T_K: float
    p_Pa: float


@dataclass(frozen=True)
class ComponentResult:
    """Un componente de la mezcla."""

    gas: IdealGas
    x: float
    y: float
    m_kg: float | None
    n_mol: float | None
    cp: float
    cv: float
    p_i: float
    V_i: float | None
    s_i: float


@dataclass(frozen=True)
class MixtureResult:
    """Las propiedades de la mezcla a (T, p)."""

    inputs: MixtureInputs
    components: tuple[ComponentResult, ...]
    M: float
    R: float
    cp: float
    cv: float
    k: float
    v_m3_per_kg: float
    m_kg: float | None
    n_mol: float | None
    V_m3: float | None
    s: float
    s_mixing: float
    notes: tuple[str, ...]


_BASIS_NAMES: dict[CompositionBasis, str] = {
    "mass": "masa",
    "moles": "moles",
    "mass_fraction": "fracción másica",
    "mole_fraction": "fracción molar",
}


def _check_components(components: tuple[Component, ...]) -> list[IdealGas]:
    if len(components) < 2:
        raise ValueError("Una mezcla necesita al menos dos gases.")
    keys = [c.gas_key for c in components]
    if len(set(keys)) != len(keys):
        raise ValueError("Cada gas va una sola vez en la mezcla: sumá las cantidades repetidas.")
    gases = [gas(k) for k in keys]
    for g, c in zip(gases, components, strict=True):
        if not (math.isfinite(c.amount) and c.amount >= 0.0):
            raise ValueError(f"La cantidad de {g.name.lower()} no puede ser negativa.")
    return gases


def solve_mixture(inputs: MixtureInputs) -> MixtureResult:
    """Fracciones, propiedades, Dalton, Amagat y entropía de una mezcla (vademecum §5).

    Raises
    ------
    ValueError
        Si hay menos de dos gases, alguno repetido, una cantidad negativa, las
        fracciones no suman 1 (con 0,1 % de tolerancia) o T y p no tienen sentido.
    """
    gases = _check_components(inputs.components)
    amounts = [c.amount for c in inputs.components]
    if sum(amounts) <= 0.0:
        raise ValueError("La mezcla no tiene nada: poné alguna cantidad positiva.")
    if inputs.basis in ("mass_fraction", "mole_fraction"):
        total = sum(amounts)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Las fracciones ({_BASIS_NAMES[inputs.basis]}) suman {_num(total, '.4f')}: "
                "tienen que sumar 1."
            )
        amounts = [a / total for a in amounts]
    if not (math.isfinite(inputs.p_Pa) and inputs.p_Pa > 0.0):
        raise ValueError("La presión de la mezcla tiene que ser positiva (absoluta).")
    for g in gases:
        g.check_T(inputs.T_K, "La temperatura de la mezcla")

    T, p = inputs.T_K, inputs.p_Pa
    Ms = [g.M for g in gases]
    if inputs.basis in ("mass", "mass_fraction"):
        mass = amounts
        moles = [m / M for m, M in zip(mass, Ms, strict=True)]
    else:
        moles = amounts
        mass = [n * M for n, M in zip(moles, Ms, strict=True)]
    m_tot, n_tot = sum(mass), sum(moles)
    xs = [m / m_tot for m in mass]
    ys = [n / n_tot for n in moles]
    M_mix = sum(y * M for y, M in zip(ys, Ms, strict=True))
    R_mix = R_U / M_mix
    absolute = inputs.basis in ("mass", "moles")
    V = m_tot * R_mix * T / p if absolute else None

    comps: list[ComponentResult] = []
    for g, x, y, m, n in zip(gases, xs, ys, mass, moles, strict=True):
        p_i = y * p
        s_i = _s(g, T, p_i, "variable") if y > 0.0 else 0.0
        comps.append(
            ComponentResult(
                gas=g,
                x=x,
                y=y,
                m_kg=m if absolute else None,
                n_mol=n if absolute else None,
                cp=g.cp(T),
                cv=g.cv(T),
                p_i=p_i,
                V_i=y * V if V is not None else None,
                s_i=s_i,
            )
        )
    cp_mix = sum(c.x * c.cp for c in comps)
    cv_mix = cp_mix - R_mix
    s_mix = sum(c.x * c.s_i for c in comps if c.y > 0.0)
    s_mixing = -sum(c.x * c.gas.R * math.log(c.y) for c in comps if c.y > 0.0)

    notes: list[str] = []
    heavy = max(comps, key=lambda c: c.x)
    many = max(comps, key=lambda c: c.y)
    if heavy.gas.key != many.gas.key:
        notes.append(
            f"En masa domina el {heavy.gas.name.lower()} ({_pct(heavy.x)}) y en moles, el "
            f"{many.gas.name.lower()} ({_pct(many.y)}): las fracciones másica y molar difieren "
            "porque los gases pesan distinto (x_i = y_i·M_i/M_M)."
        )
    water = next((c for c in comps if c.gas.key == "H2O" and c.y > 0.0), None)
    if water is not None:
        try:
            from CoolProp.CoolProp import PropsSI

            p_sat = float(PropsSI("P", "T", T, "Q", 0, "Water"))
        except ValueError:
            p_sat = math.inf
        if water.p_i > p_sat:
            notes.append(
                f"La presión parcial del vapor de agua ({_num(water.p_i / 1e3)} kPa) supera la de "
                f"saturación a {_degC(T)} ({_num(p_sat / 1e3)} kPa): parte del agua condensa y la "
                "mezcla ya no es solo de gases."
            )
    return MixtureResult(
        inputs=inputs,
        components=tuple(comps),
        M=M_mix,
        R=R_mix,
        cp=cp_mix,
        cv=cv_mix,
        k=cp_mix / cv_mix,
        v_m3_per_kg=R_mix * T / p,
        m_kg=m_tot if absolute else None,
        n_mol=n_tot if absolute else None,
        V_m3=V,
        s=s_mix,
        s_mixing=s_mixing,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------
# Mezcla adiabática
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MixingStream:
    """Un gas que entra a la mezcla: kg (tanque) o kg/s (flujo), a T y p."""

    gas_key: str
    amount: float
    T_K: float
    p_Pa: float


@dataclass(frozen=True)
class MixingInputs:
    """La mezcla: el tipo, las corrientes, el modelo de c_p y el ambiente T₀.

    ``p_out_Pa`` es la presión de salida de la cámara (solo para ``flow``).
    """

    kind: MixingKind
    streams: tuple[MixingStream, ...]
    model: GasModel = "variable"
    T0_K: float = 298.15
    p_out_Pa: float | None = None


@dataclass(frozen=True)
class StreamResult:
    """Lo que le pasa a cada corriente."""

    gas: IdealGas
    amount: float
    T_K: float
    p_Pa: float
    V_m3: float | None
    y: float
    p_final_Pa: float
    ds_TP: float
    ds_mix: float


@dataclass(frozen=True)
class MixingResult:
    """El estado final de la mezcla y su entropía generada."""

    inputs: MixingInputs
    streams: tuple[StreamResult, ...]
    T_K: float
    p_Pa: float
    V_m3: float | None
    amount: float
    S_gen_TP: float
    S_gen_mix: float
    X_dest: float
    mixture: MixtureResult
    notes: tuple[str, ...]

    @property
    def S_gen(self) -> float:
        """S_gen total: J/K en el tanque o W/K en el flujo."""
        return self.S_gen_TP + self.S_gen_mix


def solve_mixing(inputs: MixingInputs) -> MixingResult:
    """La mezcla adiabática de gases ideales (Çengel §13-3; vademecum §5 y §11).

    - Tanque rígido: Σm_i·u_i(T_i) = Σm_i·u_i(T), V = Σm_i·R_i·T_i/p_i y
      p = (Σm_i·R_i)·T/V.
    - Flujo permanente: Σṁ_i·h_i(T_i) = Σṁ_i·h_i(T) a la presión de salida dada.
    - S_gen = Σm_i·[s_i(T, y_i·p) − s_i(T_i, p_i)], partida en
      Σm_i·[s_i(T, p) − s_i(T_i, p_i)] y −Σm_i·R_i·ln y_i; X_dest = T₀·S_gen.

    Raises
    ------
    ValueError
        Con menos de dos corrientes, una cantidad o una presión que no es
        positiva, una temperatura fuera de rango, o una presión de salida que
        falta o es mayor que la de la entrada más baja.
    """
    streams = inputs.streams
    if len(streams) < 2:
        raise ValueError("Para mezclar hacen falta al menos dos corrientes.")
    if not (math.isfinite(inputs.T0_K) and 200.0 <= inputs.T0_K <= 400.0):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −73 °C y 127 °C.")
    unit = "kg" if inputs.kind == "tank" else "kg/s"
    gases: list[IdealGas] = []
    for j, st in enumerate(streams, start=1):
        g = gas(st.gas_key)
        if not (math.isfinite(st.amount) and st.amount > 0.0):
            raise ValueError(f"La cantidad de la corriente {j} ({unit}) tiene que ser positiva.")
        if not (math.isfinite(st.p_Pa) and st.p_Pa > 0.0):
            raise ValueError(f"La presión de la corriente {j} tiene que ser positiva (absoluta).")
        g.check_T(st.T_K, f"La temperatura de la corriente {j}")
        gases.append(g)
    model = inputs.model
    masses = [st.amount for st in streams]
    T_lo = max(g.T_min_K for g in gases)
    T_hi = min(g.T_max_K for g in gases)

    if inputs.kind == "tank":
        V = sum(m * g.R * st.T_K / st.p_Pa for m, g, st in zip(masses, gases, streams, strict=True))
        U0 = sum(m * _u(g, st.T_K, model) for m, g, st in zip(masses, gases, streams, strict=True))

        def balance(T: float) -> float:
            return sum(m * _u(g, T, model) for m, g in zip(masses, gases, strict=True)) - U0

    else:
        V = None
        H0 = sum(m * _h(g, st.T_K, model) for m, g, st in zip(masses, gases, streams, strict=True))

        def balance(T: float) -> float:
            return sum(m * _h(g, T, model) for m, g in zip(masses, gases, strict=True)) - H0

    T_min = min(st.T_K for st in streams)
    T_max = max(st.T_K for st in streams)
    if T_max - T_min < 1e-9:
        T = T_min
    else:
        T = brentq(balance, max(T_lo, T_min - 1e-6), min(T_hi, T_max + 1e-6), xtol=1e-10)

    m_R = sum(m * g.R for m, g in zip(masses, gases, strict=True))
    if inputs.kind == "tank":
        assert V is not None
        p = m_R * T / V
    else:
        if inputs.p_out_Pa is None:
            raise ValueError("Falta la presión de salida de la cámara de mezcla.")
        p_min = min(st.p_Pa for st in streams)
        if not (inputs.p_out_Pa > 0.0 and inputs.p_out_Pa <= p_min * (1 + 1e-12)):
            raise ValueError(
                f"La presión de salida ({_num(inputs.p_out_Pa / 1e3)} kPa) no puede ser mayor que "
                f"la de la entrada más baja ({_num(p_min / 1e3)} kPa): una cámara de mezcla no "
                "comprime."
            )
        p = inputs.p_out_Pa

    # fracciones molares por gas: dos corrientes del mismo gas no se «mezclan»
    moles_by_gas: dict[str, float] = defaultdict(float)
    for m, g in zip(masses, gases, strict=True):
        moles_by_gas[g.key] += m / g.M
    n_tot = sum(moles_by_gas.values())

    results: list[StreamResult] = []
    S_TP = S_mix = 0.0
    for m, g, st in zip(masses, gases, streams, strict=True):
        y = moles_by_gas[g.key] / n_tot
        ds_TP = _s(g, T, p, model) - _s(g, st.T_K, st.p_Pa, model)
        if abs(ds_TP) < 1e-9:  # misma T y misma p: lo que queda es redondeo (~1e-13)
            ds_TP = 0.0
        ds_mix = -g.R * math.log(y)
        S_TP += m * ds_TP
        S_mix += m * ds_mix
        results.append(
            StreamResult(
                gas=g,
                amount=m,
                T_K=st.T_K,
                p_Pa=st.p_Pa,
                V_m3=m * g.R * st.T_K / st.p_Pa if inputs.kind == "tank" else None,
                y=y,
                p_final_Pa=y * p,
                ds_TP=ds_TP,
                ds_mix=ds_mix,
            )
        )

    keys = list(moles_by_gas)
    mass_by_gas = {
        k: sum(m for m, g in zip(masses, gases, strict=True) if g.key == k) for k in keys
    }
    if len(keys) >= 2:
        mixture = solve_mixture(
            MixtureInputs(tuple(Component(k, mass_by_gas[k]) for k in keys), "mass", T, p)
        )
    else:
        # un solo gas: la «mezcla» es ese gas (fracción 1)
        g = gas(keys[0])
        mixture = MixtureResult(
            inputs=MixtureInputs((Component(keys[0], mass_by_gas[keys[0]]),), "mass", T, p),
            components=(
                ComponentResult(
                    gas=g,
                    x=1.0,
                    y=1.0,
                    m_kg=mass_by_gas[keys[0]],
                    n_mol=mass_by_gas[keys[0]] / g.M,
                    cp=g.cp(T),
                    cv=g.cv(T),
                    p_i=p,
                    V_i=None,
                    s_i=_s(g, T, p, "variable"),
                ),
            ),
            M=g.M,
            R=g.R,
            cp=g.cp(T),
            cv=g.cv(T),
            k=g.k(T),
            v_m3_per_kg=g.R * T / p,
            m_kg=mass_by_gas[keys[0]],
            n_mol=mass_by_gas[keys[0]] / g.M,
            V_m3=None,
            s=_s(g, T, p, "variable"),
            s_mixing=0.0,
            notes=(),
        )

    S_gen = S_TP + S_mix
    notes: list[str] = []
    if len(keys) == 1:
        notes.append(
            "Todas las corrientes son del mismo gas: no hay entropía de mezcla (mezclar un gas "
            "consigo mismo no cuesta nada); toda la S_gen es por igualar temperaturas y presiones."
        )
    elif S_gen > 0.0 and S_mix / S_gen > 0.99:
        notes.append(
            "Los gases entran a la misma temperatura y presión: toda la entropía generada es de "
            "mezcla, −Σm_i·R_i·ln y_i. Separarlos de nuevo costaría por lo menos T₀·S_gen de "
            "trabajo."
        )
    elif S_gen > 0.0:
        notes.append(
            f"De la entropía generada, {_pct(S_TP / S_gen, 0)} es por igualar las temperaturas y "
            f"las presiones y {_pct(S_mix / S_gen, 0)} por mezclar gases distintos."
        )
    return MixingResult(
        inputs=inputs,
        streams=tuple(results),
        T_K=T,
        p_Pa=p,
        V_m3=V,
        amount=sum(masses),
        S_gen_TP=S_TP,
        S_gen_mix=S_mix,
        X_dest=inputs.T0_K * S_gen,
        mixture=mixture,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------

C = 273.15
_ATM = 101_325.0


@dataclass(frozen=True)
class MixtureExample:
    inputs: MixtureInputs
    note: str = ""


@dataclass(frozen=True)
class MixingExample:
    inputs: MixingInputs
    note: str = ""


COMPOSITION_EXAMPLES: dict[str, MixtureExample] = {
    "3 kg de O₂, 5 kg de N₂ y 12 kg de CH₄ (Çengel §13-1)": MixtureExample(
        MixtureInputs(
            (Component("O2", 3.0), Component("N2", 5.0), Component("CH4", 12.0)),
            "mass",
            25.0 + C,
            _ATM,
        ),
        "Çengel: x = 0,15, 0,25 y 0,60; y = 0,092, 0,175 y 0,733; M = 19,6 kg/kmol.",
    ),
    "El aire seco (fracciones molares)": MixtureExample(
        MixtureInputs(
            (
                Component("N2", 0.7808),
                Component("O2", 0.2095),
                Component("Ar", 0.0093),
                Component("CO2", 0.0004),
            ),
            "mole_fraction",
            25.0 + C,
            _ATM,
        ),
        "De ahí salen M = 28,97 kg/kmol y R = 0,287 kJ/(kg·K) del aire.",
    ),
    "Humos del metano con 10 % de exceso de aire, a 1000 K": MixtureExample(
        MixtureInputs(
            (
                Component("CO2", 0.0870),
                Component("H2O", 0.1739),
                Component("O2", 0.0174),
                Component("N2", 0.7217),
            ),
            "mole_fraction",
            1000.0,
            _ATM,
        ),
    ),
    "Gas natural a 30 °C (fracciones molares)": MixtureExample(
        MixtureInputs(
            (
                Component("CH4", 0.90),
                Component("C2H6", 0.05),
                Component("C3H8", 0.02),
                Component("N2", 0.02),
                Component("CO2", 0.01),
            ),
            "mole_fraction",
            30.0 + C,
            _ATM,
        ),
    ),
}

MIXING_EXAMPLES: dict[str, MixingExample] = {
    "Tanque: 7 kg de O₂ a 40 °C y 100 kPa con 4 kg de N₂ a 20 °C y 150 kPa (Çengel §13-3)": (
        MixingExample(
            MixingInputs(
                "tank",
                (
                    MixingStream("O2", 7.0, 40.0 + C, 100e3),
                    MixingStream("N2", 4.0, 20.0 + C, 150e3),
                ),
                model="constant",
                T0_K=25.0 + C,
            ),
            "Çengel, con c_v constante: 32,2 °C y 114,5 kPa.",
        )
    ),
    "O₂ y CO₂ a 25 °C y 200 kPa: 3 kmol y 5 kmol (Çengel §13-3)": MixingExample(
        MixingInputs(
            "tank",
            (
                MixingStream("O2", 3000.0 * gas("O2").M, 25.0 + C, 200e3),
                MixingStream("CO2", 5000.0 * gas("CO2").M, 25.0 + C, 200e3),
            ),
            model="constant",
            T0_K=25.0 + C,
        ),
        "Çengel: S_gen = 44,0 kJ/K y X_dest = 13 100 kJ, toda de mezcla.",
    ),
    "Cámara de mezcla: aire a 20 °C y a 80 °C": MixingExample(
        MixingInputs(
            "flow",
            (
                MixingStream("air", 1.0, 20.0 + C, 100e3),
                MixingStream("air", 2.0, 80.0 + C, 100e3),
            ),
            model="variable",
            T0_K=20.0 + C,
            p_out_Pa=100e3,
        ),
        "El mismo gas: sin entropía de mezcla.",
    ),
    "Premezcla de metano y aire precalentado": MixingExample(
        MixingInputs(
            "flow",
            (
                MixingStream("CH4", 1.0, 25.0 + C, 120e3),
                MixingStream("air", 18.9, 400.0 + C, 110e3),
            ),
            model="variable",
            T0_K=25.0 + C,
            p_out_Pa=105e3,
        ),
    ),
}


def mixture_to_dict(result: MixtureResult) -> dict[str, Any]:
    """La mezcla para exportar (SI)."""
    return {
        "T_K": result.inputs.T_K,
        "p_Pa": result.inputs.p_Pa,
        "M_kg_per_kmol": result.M * 1000.0,
        "R_J_per_kgK": result.R,
        "cp_J_per_kgK": result.cp,
        "cv_J_per_kgK": result.cv,
        "k": result.k,
        "v_m3_per_kg": result.v_m3_per_kg,
        "m_kg": result.m_kg,
        "n_mol": result.n_mol,
        "V_m3": result.V_m3,
        "s_J_per_kgK": result.s,
        "s_mezcla_J_per_kgK": result.s_mixing,
        "componentes": [
            {
                "gas": c.gas.name,
                "x": c.x,
                "y": c.y,
                "m_kg": c.m_kg,
                "n_mol": c.n_mol,
                "p_parcial_Pa": c.p_i,
                "V_parcial_m3": c.V_i,
                "cp_J_per_kgK": c.cp,
            }
            for c in result.components
        ],
        "notas": list(result.notes),
    }


def mixing_to_dict(result: MixingResult) -> dict[str, Any]:
    """La mezcla adiabática para exportar (SI)."""
    tank = result.inputs.kind == "tank"
    return {
        "tipo": "tanque" if tank else "cámara de flujo",
        "modelo": result.inputs.model,
        "T0_K": result.inputs.T0_K,
        "T_final_K": result.T_K,
        "p_final_Pa": result.p_Pa,
        "V_m3": result.V_m3,
        "S_gen_TP": result.S_gen_TP,
        "S_gen_mezcla": result.S_gen_mix,
        "S_gen": result.S_gen,
        "unidad_S_gen": "J/K" if tank else "W/K",
        "X_dest": result.X_dest,
        "unidad_X_dest": "J" if tank else "W",
        "corrientes": [
            {
                "gas": s.gas.name,
                "cantidad": s.amount,
                "T_K": s.T_K,
                "p_Pa": s.p_Pa,
                "y_final": s.y,
                "p_parcial_final_Pa": s.p_final_Pa,
                "ds_TP_J_per_kgK": s.ds_TP,
                "ds_mezcla_J_per_kgK": s.ds_mix,
            }
            for s in result.streams
        ],
        "mezcla": mixture_to_dict(result.mixture),
        "notas": list(result.notes),
    }
