"""Interpolación lineal simple y de doble entrada (bilineal) sobre tablas.

Diseñado para uso didáctico: además del valor interpolado, cada función
devuelve un :class:`InterpolationResult` que incluye los pasos
intermedios, la fórmula aplicada en LaTeX y una explicación en español
plano. La UI puede renderizar ese procedimiento para que el alumno vea
cómo se llegó al resultado.

Convenciones
------------
- Todas las funciones son puras y sin estado.
- Los nodos ``xs`` (y ``ys`` en bilineal) deben estar **estrictamente
  ordenados de menor a mayor**. Si llegan en orden decreciente, la
  función levanta ``ValueError`` con la sugerencia de invertir filas.
- Extrapolación prohibida por defecto: si el valor de consulta cae
  fuera del rango de la tabla y no se pasó ``allow_extrapolation=True``,
  se levanta ``ValueError``.
- En el bilineal, ``zs[i][j] = z(xs[i], ys[j])``. Convención de
  esquinas: ``z00 = z(x0, y0)``, ``z01 = z(x0, y1)``,
  ``z10 = z(x1, y0)``, ``z11 = z(x1, y1)``.

Cita
----
Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
(2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.,
§3.6 "Interpolation on a grid in multidimensions"). Cambridge University
Press.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

InterpolationKind = Literal["linear", "bilinear"]

ZsInput = pd.DataFrame | np.ndarray | Sequence[Sequence[float]]


@dataclass(frozen=True)
class InterpolationSteps:
    """Pasos didácticos de una interpolación.

    Pensado para ser renderizado en la UI (LaTeX + texto en español
    plano). Discriminado por :attr:`kind`. Los campos opcionales solo
    están poblados cuando aplica al tipo de interpolación.
    """

    kind: InterpolationKind
    formula_latex: str
    substituted_latex: str
    narrative_es: str
    exact_node: bool = False

    # Lineal: x_query es la consulta; (x0, f0), (x1, f1) son los nodos vecinos.
    # Bilineal: x_query, x0, x1 son la coordenada-x; ver y_query, y0, y1 abajo.
    x_query: float | None = None
    x0: float | None = None
    x1: float | None = None
    f0: float | None = None
    f1: float | None = None

    # Solo bilineal.
    y_query: float | None = None
    y0: float | None = None
    y1: float | None = None
    z00: float | None = None
    z01: float | None = None
    z10: float | None = None
    z11: float | None = None
    # Pasos intermedios: primero interpolo en x sobre las dos líneas y = y0, y = y1.
    z_query_y0: float | None = None
    z_query_y1: float | None = None


@dataclass(frozen=True)
class InterpolationResult:
    """Resultado de una interpolación: valor + procedimiento."""

    value: float
    steps: InterpolationSteps


# ---------------------------------------------------------------------
# Lineal
# ---------------------------------------------------------------------


def linear(
    x: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    allow_extrapolation: bool = False,
) -> InterpolationResult:
    """Interpolación lineal entre dos nodos ``(x0, y0)`` y ``(x1, y1)``.

    Parámetros
    ----------
    x : float
        Valor de consulta.
    x0, x1 : float
        Coordenadas-x de los dos nodos. Deben ser distintos.
    y0, y1 : float
        Valores ``f(x0)`` y ``f(x1)``.
    allow_extrapolation : bool, opcional
        Si es ``False`` (default), levanta ``ValueError`` cuando ``x``
        cae fuera del intervalo ``[min(x0, x1), max(x0, x1)]``.

    Devuelve
    --------
    InterpolationResult
        Valor interpolado y procedimiento didáctico.
    """
    x_f, x0_f, x1_f, y0_f, y1_f = float(x), float(x0), float(x1), float(y0), float(y1)

    if x0_f == x1_f:
        raise ValueError(
            f"Los nodos x0 y x1 deben ser distintos para interpolar; recibí x0 = x1 = {x0_f}."
        )

    x_min, x_max = (x0_f, x1_f) if x0_f < x1_f else (x1_f, x0_f)
    if not (x_min <= x_f <= x_max) and not allow_extrapolation:
        raise ValueError(
            f"El valor x = {x_f} cae fuera del intervalo "
            f"[{x_min}, {x_max}] definido por los nodos. Para extrapolar "
            f"explícitamente, pasá `allow_extrapolation=True`."
        )

    # Coincidencia exacta con un nodo: sin interpolación.
    if x_f == x0_f:
        return _exact_node_linear(x_f, x0_f, y0_f, other_x=x1_f, other_y=y1_f)
    if x_f == x1_f:
        return _exact_node_linear(x_f, x1_f, y1_f, other_x=x0_f, other_y=y0_f)

    slope = (y1_f - y0_f) / (x1_f - x0_f)
    frac = (x_f - x0_f) / (x1_f - x0_f)
    y = y0_f + slope * (x_f - x0_f)

    formula = r"y = y_0 + (y_1 - y_0) \cdot \frac{x - x_0}{x_1 - x_0}"
    substituted = (
        rf"y = {y0_f:.6g} + ({y1_f:.6g} - {y0_f:.6g}) \cdot "
        rf"\frac{{{x_f:.6g} - {x0_f:.6g}}}{{{x1_f:.6g} - {x0_f:.6g}}}"
        rf" = {y:.6g}"
    )
    narrative = (
        f"Interpolación lineal entre los nodos (x₀, y₀) = "
        f"({x0_f:g}, {y0_f:g}) y (x₁, y₁) = ({x1_f:g}, {y1_f:g}). "
        f"El valor x = {x_f:g} representa una fracción "
        f"(x − x₀)/(x₁ − x₀) = {frac:.4f} del intervalo, así que se avanza "
        f"esa misma fracción del salto (y₁ − y₀) = {y1_f - y0_f:g} "
        f"a partir de y₀."
    )

    return InterpolationResult(
        value=y,
        steps=InterpolationSteps(
            kind="linear",
            formula_latex=formula,
            substituted_latex=substituted,
            narrative_es=narrative,
            x_query=x_f,
            x0=x0_f,
            x1=x1_f,
            f0=y0_f,
            f1=y1_f,
        ),
    )


def linear_from_table(
    x: float,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    allow_extrapolation: bool = False,
) -> InterpolationResult:
    """Interpolación lineal sobre una tabla de nodos.

    Parámetros
    ----------
    x : float
        Valor de consulta.
    xs : Sequence[float]
        Coordenadas-x, **estrictamente crecientes**.
    ys : Sequence[float]
        Valores asociados ``f(xs[i])``. Mismo tamaño que ``xs``.
    allow_extrapolation : bool, opcional
        Permite consultar fuera del rango de la tabla.

    Devuelve
    --------
    InterpolationResult
    """
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    if xs_arr.ndim != 1 or ys_arr.ndim != 1:
        raise ValueError(
            f"`xs` e `ys` deben ser secuencias unidimensionales. "
            f"Recibí xs.ndim = {xs_arr.ndim}, ys.ndim = {ys_arr.ndim}."
        )
    if xs_arr.shape != ys_arr.shape:
        raise ValueError(
            f"`xs` y `ys` deben tener el mismo tamaño. "
            f"Recibí len(xs) = {xs_arr.size}, len(ys) = {ys_arr.size}."
        )
    if xs_arr.size < 2:
        raise ValueError(
            f"La tabla necesita al menos 2 nodos para interpolar; recibí {xs_arr.size}."
        )

    _validate_strictly_increasing(xs_arr, name="xs")

    x_f = float(x)
    if (x_f < xs_arr[0] or x_f > xs_arr[-1]) and not allow_extrapolation:
        raise ValueError(
            f"El valor x = {x_f} cae fuera del rango de la tabla "
            f"[{xs_arr[0]}, {xs_arr[-1]}]. Para extrapolar explícitamente, "
            f"pasá `allow_extrapolation=True`."
        )

    i0, i1 = _bracket_indices(xs_arr, x_f)
    return linear(
        x_f,
        float(xs_arr[i0]),
        float(xs_arr[i1]),
        float(ys_arr[i0]),
        float(ys_arr[i1]),
        allow_extrapolation=allow_extrapolation,
    )


def _exact_node_linear(
    x_query: float,
    x_node: float,
    y_node: float,
    *,
    other_x: float,
    other_y: float,
) -> InterpolationResult:
    narrative = (
        f"El valor de consulta x = {x_query:g} coincide exactamente con un "
        f"nodo de la tabla (x = {x_node:g}). Coincidencia exacta con nodo, "
        f"sin interpolación: y = {y_node:g}."
    )
    return InterpolationResult(
        value=float(y_node),
        steps=InterpolationSteps(
            kind="linear",
            formula_latex=r"y = y_i \quad \text{(coincidencia exacta con nodo)}",
            substituted_latex=rf"y = {y_node:g}",
            narrative_es=narrative,
            exact_node=True,
            x_query=float(x_query),
            x0=float(min(x_node, other_x)),
            x1=float(max(x_node, other_x)),
            f0=float(other_y if x_node > other_x else y_node),
            f1=float(y_node if x_node > other_x else other_y),
        ),
    )


# ---------------------------------------------------------------------
# Bilineal
# ---------------------------------------------------------------------


def bilinear(
    x: float,
    y: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z00: float,
    z01: float,
    z10: float,
    z11: float,
    *,
    allow_extrapolation: bool = False,
) -> InterpolationResult:
    """Interpolación bilineal sobre la celda rectangular de cuatro vértices.

    Convención: ``z00 = z(x0, y0)``, ``z01 = z(x0, y1)``,
    ``z10 = z(x1, y0)``, ``z11 = z(x1, y1)``.

    Procedimiento (didáctico, en dos etapas):
    1. Dos interpolaciones lineales en ``x`` para obtener ``z`` sobre
       las líneas ``y = y0`` y ``y = y1``.
    2. Una tercera interpolación lineal en ``y`` entre esos dos valores
       intermedios.
    """
    x_f, y_f = float(x), float(y)
    x0_f, x1_f, y0_f, y1_f = float(x0), float(x1), float(y0), float(y1)
    z00_f, z01_f = float(z00), float(z01)
    z10_f, z11_f = float(z10), float(z11)

    if x0_f == x1_f:
        raise ValueError(f"Los nodos x0 y x1 deben ser distintos; recibí x0 = x1 = {x0_f}.")
    if y0_f == y1_f:
        raise ValueError(f"Los nodos y0 e y1 deben ser distintos; recibí y0 = y1 = {y0_f}.")

    x_min, x_max = (x0_f, x1_f) if x0_f < x1_f else (x1_f, x0_f)
    y_min, y_max = (y0_f, y1_f) if y0_f < y1_f else (y1_f, y0_f)
    if not allow_extrapolation:
        if not (x_min <= x_f <= x_max):
            raise ValueError(
                f"El valor x = {x_f} cae fuera del intervalo "
                f"[{x_min}, {x_max}]. Para extrapolar explícitamente, "
                f"pasá `allow_extrapolation=True`."
            )
        if not (y_min <= y_f <= y_max):
            raise ValueError(
                f"El valor y = {y_f} cae fuera del intervalo "
                f"[{y_min}, {y_max}]. Para extrapolar explícitamente, "
                f"pasá `allow_extrapolation=True`."
            )

    # Coincidencia exacta con un vértice.
    if x_f in (x0_f, x1_f) and y_f in (y0_f, y1_f):
        z_node = {
            (x0_f, y0_f): z00_f,
            (x0_f, y1_f): z01_f,
            (x1_f, y0_f): z10_f,
            (x1_f, y1_f): z11_f,
        }[(x_f, y_f)]
        narrative = (
            f"El punto de consulta (x, y) = ({x_f:g}, {y_f:g}) coincide "
            f"exactamente con un vértice de la celda. Coincidencia exacta "
            f"con nodo, sin interpolación: z = {z_node:g}."
        )
        return InterpolationResult(
            value=z_node,
            steps=InterpolationSteps(
                kind="bilinear",
                formula_latex=r"z = z_{ij} \quad \text{(coincidencia exacta con vértice)}",
                substituted_latex=rf"z = {z_node:g}",
                narrative_es=narrative,
                exact_node=True,
                x_query=x_f,
                x0=x0_f,
                x1=x1_f,
                y_query=y_f,
                y0=y0_f,
                y1=y1_f,
                z00=z00_f,
                z01=z01_f,
                z10=z10_f,
                z11=z11_f,
            ),
        )

    frac_x = (x_f - x0_f) / (x1_f - x0_f)
    frac_y = (y_f - y0_f) / (y1_f - y0_f)
    z_query_y0 = z00_f + (z10_f - z00_f) * frac_x
    z_query_y1 = z01_f + (z11_f - z01_f) * frac_x
    z = z_query_y0 + (z_query_y1 - z_query_y0) * frac_y

    formula = (
        r"\begin{aligned}"
        r"z(x, y_0) &= z_{00} + (z_{10} - z_{00}) \cdot \tfrac{x - x_0}{x_1 - x_0} \\"
        r"z(x, y_1) &= z_{01} + (z_{11} - z_{01}) \cdot \tfrac{x - x_0}{x_1 - x_0} \\"
        r"z(x, y)   &= z(x, y_0) + \bigl[z(x, y_1) - z(x, y_0)\bigr] \cdot "
        r"\tfrac{y - y_0}{y_1 - y_0}"
        r"\end{aligned}"
    )
    substituted = (
        r"\begin{aligned}"
        rf"z(x, y_0) &= {z00_f:.6g} + ({z10_f:.6g} - {z00_f:.6g}) \cdot "
        rf"\tfrac{{{x_f:.6g} - {x0_f:.6g}}}{{{x1_f:.6g} - {x0_f:.6g}}}"
        rf" = {z_query_y0:.6g} \\"
        rf"z(x, y_1) &= {z01_f:.6g} + ({z11_f:.6g} - {z01_f:.6g}) \cdot "
        rf"\tfrac{{{x_f:.6g} - {x0_f:.6g}}}{{{x1_f:.6g} - {x0_f:.6g}}}"
        rf" = {z_query_y1:.6g} \\"
        rf"z(x, y)   &= {z_query_y0:.6g} + ({z_query_y1:.6g} - {z_query_y0:.6g}) "
        rf"\cdot \tfrac{{{y_f:.6g} - {y0_f:.6g}}}{{{y1_f:.6g} - {y0_f:.6g}}}"
        rf" = {z:.6g}"
        r"\end{aligned}"
    )
    narrative = (
        f"Interpolación bilineal sobre la celda rectangular con vértices "
        f"(x₀, y₀) = ({x0_f:g}, {y0_f:g}), (x₁, y₀) = ({x1_f:g}, {y0_f:g}), "
        f"(x₀, y₁) = ({x0_f:g}, {y1_f:g}) y (x₁, y₁) = ({x1_f:g}, {y1_f:g}). "
        f"Se hace en dos pasos. (1) Primero dos interpolaciones lineales en "
        f"la dirección x, una sobre cada borde horizontal: "
        f"z({x_f:g}, {y0_f:g}) = {z_query_y0:.6g} y "
        f"z({x_f:g}, {y1_f:g}) = {z_query_y1:.6g}. "
        f"(2) Después una tercera lineal en la dirección y, entre esos dos "
        f"valores intermedios, para llegar al resultado final "
        f"z({x_f:g}, {y_f:g}) = {z:.6g}."
    )

    return InterpolationResult(
        value=z,
        steps=InterpolationSteps(
            kind="bilinear",
            formula_latex=formula,
            substituted_latex=substituted,
            narrative_es=narrative,
            x_query=x_f,
            x0=x0_f,
            x1=x1_f,
            y_query=y_f,
            y0=y0_f,
            y1=y1_f,
            z00=z00_f,
            z01=z01_f,
            z10=z10_f,
            z11=z11_f,
            z_query_y0=z_query_y0,
            z_query_y1=z_query_y1,
        ),
    )


def bilinear_from_table(
    x: float,
    y: float,
    xs: Sequence[float],
    ys: Sequence[float],
    zs: ZsInput,
    *,
    allow_extrapolation: bool = False,
) -> InterpolationResult:
    """Interpolación bilineal sobre una tabla 2D.

    Parámetros
    ----------
    x, y : float
        Valores de consulta.
    xs : Sequence[float]
        Coordenadas-x, estrictamente crecientes.
    ys : Sequence[float]
        Coordenadas-y, estrictamente crecientes.
    zs : DataFrame | ndarray | Sequence[Sequence[float]]
        Matriz con ``zs[i][j] = z(xs[i], ys[j])``. Si se pasa un
        :class:`pandas.DataFrame`, se usa su ``.to_numpy()`` (el índice
        y las columnas se ignoran; pasá los ``xs``/``ys`` por separado).
    allow_extrapolation : bool
        Permite consultar fuera del rango de la tabla.
    """
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    if isinstance(zs, pd.DataFrame):
        zs_arr = zs.to_numpy(dtype=float)
    else:
        zs_arr = np.asarray(zs, dtype=float)

    if xs_arr.ndim != 1 or ys_arr.ndim != 1:
        raise ValueError(
            f"`xs` e `ys` deben ser secuencias unidimensionales. "
            f"Recibí xs.ndim = {xs_arr.ndim}, ys.ndim = {ys_arr.ndim}."
        )
    expected_shape = (xs_arr.size, ys_arr.size)
    if zs_arr.shape != expected_shape:
        raise ValueError(
            f"La matriz `zs` debe tener shape (len(xs), len(ys)) = "
            f"{expected_shape}, pero recibí {zs_arr.shape}."
        )
    if xs_arr.size < 2 or ys_arr.size < 2:
        raise ValueError(
            f"La tabla necesita al menos 2 nodos en cada eje; recibí "
            f"len(xs) = {xs_arr.size}, len(ys) = {ys_arr.size}."
        )

    _validate_strictly_increasing(xs_arr, name="xs")
    _validate_strictly_increasing(ys_arr, name="ys")

    x_f, y_f = float(x), float(y)
    if not allow_extrapolation:
        if x_f < xs_arr[0] or x_f > xs_arr[-1]:
            raise ValueError(
                f"El valor x = {x_f} cae fuera del rango de la tabla en x "
                f"[{xs_arr[0]}, {xs_arr[-1]}]. Para extrapolar explícitamente, "
                f"pasá `allow_extrapolation=True`."
            )
        if y_f < ys_arr[0] or y_f > ys_arr[-1]:
            raise ValueError(
                f"El valor y = {y_f} cae fuera del rango de la tabla en y "
                f"[{ys_arr[0]}, {ys_arr[-1]}]. Para extrapolar explícitamente, "
                f"pasá `allow_extrapolation=True`."
            )

    i0, i1 = _bracket_indices(xs_arr, x_f)
    j0, j1 = _bracket_indices(ys_arr, y_f)

    return bilinear(
        x_f,
        y_f,
        float(xs_arr[i0]),
        float(xs_arr[i1]),
        float(ys_arr[j0]),
        float(ys_arr[j1]),
        float(zs_arr[i0, j0]),
        float(zs_arr[i0, j1]),
        float(zs_arr[i1, j0]),
        float(zs_arr[i1, j1]),
        allow_extrapolation=allow_extrapolation,
    )


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------


def _validate_strictly_increasing(arr: np.ndarray, *, name: str) -> None:
    """Valida que ``arr`` sea estrictamente creciente; mensajes didácticos."""
    diffs = np.diff(arr)
    if np.all(diffs > 0):
        return
    if np.all(diffs < 0):
        raise ValueError(
            f"La columna `{name}` está en orden decreciente. Invertí las "
            f"filas de la tabla (orden de menor a mayor) y volvé a intentar."
        )
    # Primer índice donde falla la monotonía estricta.
    bad_idx = int(np.argmax(diffs <= 0))
    raise ValueError(
        f"La columna `{name}` debe ser estrictamente creciente (sin "
        f"duplicados ni saltos hacia atrás). Encontré "
        f"{name}[{bad_idx + 1}] = {arr[bad_idx + 1]} ≤ "
        f"{name}[{bad_idx}] = {arr[bad_idx]}."
    )


def _bracket_indices(xs_arr: np.ndarray, x: float) -> tuple[int, int]:
    """Devuelve ``(i0, i1)`` tal que ``xs[i0] ≤ x ≤ xs[i1]`` cuando ``x``
    está en el rango. Si ``x`` está fuera, devuelve los dos índices del
    extremo más cercano (sirve para extrapolación cuando está habilitada).
    """
    n = xs_arr.size
    idx = int(np.searchsorted(xs_arr, x, side="left"))
    if idx <= 0:
        return 0, 1
    if idx >= n:
        return n - 2, n - 1
    return idx - 1, idx
