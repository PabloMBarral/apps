"""Equilibrio químico de los productos: la disociación en la llama — Fase 5.

A las temperaturas de llama el CO₂ y el H₂O se disocian (CO₂ ⇌ CO + ½O₂,
H₂O ⇌ H₂ + ½O₂, H₂O ⇌ OH + ½H₂…) y aparece NO: la temperatura adiabática real
queda por debajo de la de combustión completa (nota del vademecum §16.8;
Cengel cap. 16).

La composición de equilibrio minimiza la energía libre de Gibbs de la mezcla
de gases ideales con los átomos de cada elemento fijos. Se resuelve con el
método de los potenciales de los elementos y la iteración de Newton de NASA CEA
(Gordon, S. & McBride, B. J. (1994), *Computer Program for Calculation of
Complex Chemical Equilibrium Compositions and Applications*, NASA RP-1311,
ecs. 2.24–2.26 y la amortiguación de las ecs. 3.1–3.3), con los mismos
polinomios NASA-9 de :mod:`core.combustion.thermo`.

- A T y p: μ_j/RT = g°_j/RT + ln(n_j/n) + ln(p/p°).
- A T y V (la bomba calorimétrica): μ_j/RT = g°_j/RT + ln(n_j·R·T/(V·p°)).
- Adiabática: la T a la que la entalpía (o la energía interna, a V
  constante) de los productos en equilibrio iguala la de los reactivos.

Todo en SI y por mol. No importa Streamlit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from core.combustion.thermo import ELEMENTS, P_STANDARD_PA, R_U, T_MAX_K, species

__all__ = [
    "EQUILIBRIUM_SPECIES",
    "adiabatic_equilibrium",
    "equilibrium_composition",
]

#: Especies de los productos en equilibrio (gases).
EQUILIBRIUM_SPECIES: tuple[str, ...] = (
    "CO2",
    "CO",
    "H2O",
    "H2",
    "O2",
    "N2",
    "OH",
    "H",
    "O",
    "NO",
    "N",
    "Ar",
    "SO2",
)

# Convergencia de CEA (RP-1311 §3.3): Σ n_j·|Δln n_j| / Σ n_j ≤ 0,5e-5.
_TOL = 0.5e-5
_MAX_ITER = 400
# Una especie traza tiene n_j/n < 1e-8 (CEA, ec. 3.2).
_TRACE = 1e-8
_LN_TRACE = math.log(1e-4)  # −9,21: el tope de la amortiguación de las trazas (ec. 3.2)


def _system(elements: Mapping[str, float]) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    present = tuple(el for el in ELEMENTS if elements.get(el, 0.0) > 0.0)
    names = tuple(k for k in EQUILIBRIUM_SPECIES if all(el in present for el in species(k).formula))
    A = np.zeros((len(present), len(names)))
    for j, k in enumerate(names):
        for el, n in species(k).formula.items():
            A[present.index(el), j] = n
    return present, names, A


def equilibrium_composition(
    elements: Mapping[str, float],
    T_K: float,
    *,
    p_Pa: float | None = None,
    V_m3: float | None = None,
    guess: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Moles de cada especie en equilibrio a T y p (o a T y V).

    ``elements`` son los moles de átomos de cada elemento (C, H, O, N, S, Ar);
    ``guess``, una composición de partida (la de combustión completa sirve).

    Raises
    ------
    ValueError
        Si la iteración no converge (con un mensaje para el alumno).
    """
    if (p_Pa is None) == (V_m3 is None):
        raise ValueError("El equilibrio se calcula a T y p o a T y V (uno de los dos).")
    present, names, A = _system(elements)
    b0 = np.array([elements[el] for el in present])
    g = np.array([species(k).g0(T_K) / (R_U * T_K) for k in names])
    if V_m3 is not None:
        g = g + math.log(R_U * T_K / (V_m3 * P_STANDARD_PA))
    ns = len(names)
    l = len(present)  # noqa: E741 (la cantidad de elementos, como en CEA)
    if guess:
        n_j = np.array([max(float(guess.get(k, 0.0)), 0.0) for k in names])
        floor = 1e-6 * max(float(n_j.sum()), 1e-300)
        n_j = np.maximum(n_j, floor)
    else:
        n_j = np.full(ns, 0.1 / ns)
    ln_nj = np.log(n_j)
    ln_n = math.log(float(n_j.sum()))
    fixed_p = p_Pa is not None
    ln_p = math.log(p_Pa / P_STANDARD_PA) if fixed_p else 0.0
    for _ in range(_MAX_ITER):
        n_j = np.exp(ln_nj)
        n = math.exp(ln_n)
        mu = g + ln_nj + ((ln_p - ln_n) if fixed_p else 0.0)
        An = A * n_j
        if fixed_p:
            G = np.zeros((l + 1, l + 1))
            G[:l, :l] = An @ A.T
            G[:l, l] = A @ n_j
            G[l, :l] = A @ n_j
            G[l, l] = n_j.sum() - n
            rhs = np.empty(l + 1)
            rhs[:l] = b0 - A @ n_j + An @ mu
            rhs[l] = n - n_j.sum() + n_j @ mu
        else:
            G = An @ A.T
            rhs = b0 - A @ n_j + An @ mu
        try:
            sol = np.linalg.solve(G, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "El equilibrio químico no se pudo resolver (sistema singular)."
            ) from exc
        pi = sol[:l]
        d_ln_n = float(sol[l]) if fixed_p else 0.0
        d_ln_nj = -mu + A.T @ pi + d_ln_n
        # Amortiguación de CEA: los mayoritarios no cambian más que e² por paso y
        # las trazas no suben más allá de 1e-4 del total (ecs. 3.1–3.3).
        frac = n_j / n_j.sum()
        major = frac > _TRACE
        big = 5.0 * abs(d_ln_n)
        if major.any():
            big = max(big, float(np.max(np.abs(d_ln_nj[major]))))
        lam = 2.0 / big if big > 2.0 else 1.0
        rising = (~major) & (d_ln_nj - d_ln_n > 0.0)
        if rising.any():
            denom = d_ln_nj[rising] - d_ln_n
            cand = np.abs((-(ln_nj[rising] - ln_n) + _LN_TRACE) / denom)
            lam = min(lam, float(np.min(cand)))
        lam = min(lam, 1.0)
        ln_nj = ln_nj + lam * d_ln_nj
        if fixed_p:
            ln_n = ln_n + lam * d_ln_n
        else:
            ln_n = math.log(float(np.exp(ln_nj).sum()))
        total = float(n_j.sum())
        if (n_j @ np.abs(d_ln_nj)) / total < _TOL and (
            not fixed_p or n * abs(d_ln_n) / total < _TOL
        ):
            n_j = np.exp(ln_nj)
            if np.max(np.abs(A @ n_j - b0)) <= 1e-7 * float(b0.max()):
                break
    else:
        raise ValueError(
            f"El equilibrio químico no convergió a {T_K:.0f} K: probá con otro exceso de aire."
        )
    n_j = np.exp(ln_nj)
    n_j[n_j / n_j.sum() < 1e-14] = 0.0
    return {k: float(v) for k, v in zip(names, n_j, strict=True)}


def _enthalpy(moles: Mapping[str, float], T_K: float) -> float:
    return sum(n * species(k).h(T_K) for k, n in moles.items() if n > 0.0)


def adiabatic_equilibrium(
    elements: Mapping[str, float],
    energy_target: float,
    *,
    process: Literal["p", "v"] = "p",
    p_Pa: float | None = None,
    V_m3: float | None = None,
    guess: Mapping[str, float] | None = None,
    T_low_K: float = 300.0,
    T_high_K: float = T_MAX_K,
) -> tuple[float, dict[str, float]]:
    """Temperatura adiabática y composición de equilibrio de los productos.

    A presión constante (flujo): H_productos(T) = ``energy_target`` (la
    entalpía de los reactivos). A volumen constante (bomba): U_productos(T) =
    ``energy_target`` (la energía interna de los reactivos), con
    U = H − R·T·n_gas.

    Raises
    ------
    ValueError
        Si la energía no alcanza (o sobra) para el rango de los datos.
    """
    if process == "p" and p_Pa is None:
        raise ValueError("A presión constante hace falta la presión.")
    if process == "v" and V_m3 is None:
        raise ValueError("A volumen constante hace falta el volumen.")
    state: dict[str, dict[str, float]] = {"x": dict(guess) if guess else {}}

    def residual(T: float) -> float:
        if process == "p":
            x = equilibrium_composition(elements, T, p_Pa=p_Pa, guess=state["x"] or None)
            value = _enthalpy(x, T)
        else:
            x = equilibrium_composition(elements, T, V_m3=V_m3, guess=state["x"] or None)
            value = _enthalpy(x, T) - R_U * T * sum(x.values())
        state["x"] = x
        return value - energy_target

    lo, hi = residual(T_low_K), residual(T_high_K)
    if lo > 0.0:
        raise ValueError(
            "Los reactivos tienen menos energía que los productos en equilibrio a "
            f"{T_low_K - 273.15:.0f} °C: no hay llama."
        )
    if hi < 0.0:
        raise ValueError(
            f"La llama superaría {T_high_K:.0f} K, el límite de los datos (polinomios NASA)."
        )
    T = float(brentq(residual, T_low_K, T_high_K, xtol=1e-6, rtol=1e-12))
    residual(T)
    return T, state["x"]
