"""Conducción transitoria (Fase 11.1).

Çengel y Ghajar (2015), cap. 4; Incropera et al. (2007), cap. 5. Cuatro problemas:

- **Capacidad concentrada** (§4-1): con Bi = h·L_c/k < 0,1 la temperatura del cuerpo es
  uniforme y θ = (T − T∞)/(T_i − T∞) = exp(−b·t), con b = h·A/(ρ·V·c) y L_c = V/A.
- **Pared plana, cilindro largo y esfera** con convección (§4-2): la serie de Fourier
  θ = Σ A_n·exp(−λ_n²·τ)·X(λ_n·ξ), con τ = α·t/L² y λ_n las raíces de la ecuación
  característica (λ·tan λ = Bi en la pared; λ·J₁(λ)/J₀(λ) = Bi en el cilindro;
  1 − λ·cot λ = Bi en la esfera). El primer término alcanza con τ > 0,2: es la solución
  de los gráficos de Heisler (1947) y de las tablas de λ₁ y A₁.
- **Sólido semi-infinito** (§4-3): con la superficie a T_s, un flujo constante q̇_s o
  convección, con erf y erfc de η = x/(2·√(α·t)) (Carslaw y Jaeger, 1959); y el contacto
  de dos sólidos, cuya superficie queda a una temperatura constante que pesa cada uno
  con √(k·ρ·c).
- **Multidimensional** (§4-4): la solución de un sólido que es la intersección de otros
  de una dimensión es el producto de sus θ; Q/Q_máx se combina con la regla de Langston.

Todo en SI: temperaturas en K, tiempos en s, longitudes en m. L es la mitad del espesor
de una pared que se enfría por las dos caras (o el espesor, con la otra cara aislada).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfc, erfcx, erfinv, j0, j1, jn_zeros

from core.units_system import UnitSystem, format_quantity

__all__ = [
    "BODIES",
    "BODY_DIMENSIONS",
    "CONTACT_EXAMPLES",
    "GEOMETRIES",
    "LUMPED_EXAMPLES",
    "ONE_DIM_EXAMPLES",
    "PRODUCT_EXAMPLES",
    "PRODUCT_SHAPES",
    "SEMI_BOUNDARIES",
    "SEMI_INFINITE_EXAMPLES",
    "SOLIDS",
    "TAU_ONE_TERM",
    "Body",
    "ContactInputs",
    "ContactResult",
    "Factor",
    "FactorKind",
    "FactorResult",
    "Geometry1D",
    "LumpedInputs",
    "LumpedResult",
    "OneDimInputs",
    "OneDimResult",
    "ProductInputs",
    "ProductResult",
    "ProductShape",
    "SemiBoundary",
    "SemiInfiniteInputs",
    "SemiInfiniteResult",
    "Solid",
    "TransientExample",
    "body_volume_area",
    "center_history",
    "coefficient",
    "contact_to_dict",
    "eigenvalues",
    "heat_ratio",
    "lumped_curve",
    "lumped_to_dict",
    "one_dim_to_dict",
    "product_factors",
    "product_map",
    "product_to_dict",
    "semi_infinite_profiles",
    "semi_infinite_theta",
    "semi_infinite_to_dict",
    "shape_function",
    "solve_contact",
    "solve_lumped",
    "solve_one_dimensional",
    "solve_product",
    "solve_semi_infinite",
    "theta",
    "time_profiles",
]

C = 273.15
#: Desde este τ el primer término de la serie alcanza (error < 2 %, Çengel y Ghajar §4-2).
TAU_ONE_TERM = 0.2
#: Hasta este Bi la capacidad concentrada vale (Çengel y Ghajar §4-1).
BI_LUMPED = 0.1
#: La serie se corta cuando el término siguiente pesa menos que esto.
_SERIES_TOL = 1e-12
_SERIES_MAX = 400
#: η en el que el semi-infinito cambió el 1 %: erfc(η) = 0,01.
_ETA_1PCT = float(erfinv(0.99))


def _degC(T_K: float) -> str:
    return f"{T_K - C:.1f} °C".replace(".", ",").replace("-", "−")


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",").replace("-", "−").replace("e−0", "e−")


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


def _len_text(x_m: float) -> str:
    """Una longitud en mm, cm o m según su tamaño."""
    if x_m >= 1.0:
        return f"{_num(x_m, 3)} m"
    if x_m >= 0.01:
        return f"{_num(100.0 * x_m, 3)} cm"
    return f"{_num(1000.0 * x_m, 3)} mm"


def _dT_text(dT: float) -> str:
    """Una diferencia de temperatura chica, sin ruido de redondeo."""
    return "menos de 0,01 K" if dT < 0.01 else f"{dT:.2f} K".replace(".", ",")


def _time_text(t_s: float) -> str:
    """Un tiempo con la unidad que se lee mejor (s, min, h o días)."""
    if t_s < 120.0:
        return f"{_num(t_s, 3)} s"
    if t_s < 2.0 * 3600.0:
        return f"{_num(t_s / 60.0, 3)} min"
    if t_s < 3.0 * 86400.0:
        return f"{_num(t_s / 3600.0, 3)} h"
    return f"{_num(t_s / 86400.0, 3)} días"


# ---------------------------------------------------------------------
# Sólidos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Solid:
    """Un sólido con k, ρ y c constantes."""

    name: str
    k: float  # W/(m·K)
    rho: float  # kg/m³
    c: float  # J/(kg·K)

    @property
    def alpha(self) -> float:
        """Difusividad térmica α = k/(ρ·c) [m²/s]."""
        return self.k / (self.rho * self.c)

    @property
    def effusivity(self) -> float:
        """√(k·ρ·c) [W·s^½/(m²·K)]: manda en el contacto de dos sólidos (§4-3)."""
        return math.sqrt(self.k * self.rho * self.c)


#: k, ρ y c a 300 K: Incropera et al. (2007), tablas A.1 (metales) y A.3 (no metales).
SOLIDS: dict[str, Solid] = {
    "aluminum": Solid("Aluminio puro", 237.0, 2702.0, 903.0),
    "copper": Solid("Cobre puro", 401.0, 8933.0, 385.0),
    "brass": Solid("Latón (70 Cu–30 Zn)", 110.0, 8530.0, 380.0),
    "iron": Solid("Hierro puro", 80.2, 7870.0, 447.0),
    "carbon_steel": Solid("Acero al carbono (AISI 1010)", 63.9, 7832.0, 434.0),
    "stainless": Solid("Acero inoxidable AISI 304", 14.9, 7900.0, 477.0),
    "lead": Solid("Plomo", 35.3, 11340.0, 129.0),
    "silver": Solid("Plata", 429.0, 10500.0, 235.0),
    "glass": Solid("Vidrio de ventana", 1.4, 2500.0, 750.0),
    "concrete": Solid("Hormigón", 1.4, 2300.0, 880.0),
    "brick": Solid("Ladrillo común", 0.72, 1920.0, 835.0),
    "granite": Solid("Granito", 2.79, 2630.0, 775.0),
    "oak": Solid("Madera de roble (a través de la veta)", 0.17, 545.0, 2385.0),
    "soil": Solid("Suelo", 0.52, 2050.0, 1840.0),
    "sand": Solid("Arena", 0.27, 1515.0, 800.0),
    "ice": Solid("Hielo (a 0 °C)", 1.88, 920.0, 2040.0),
}


def _check_solid(solid: Solid, what: str = "El sólido") -> None:
    if not (solid.k > 0.0 and solid.rho > 0.0 and solid.c > 0.0):
        raise ValueError(
            f"{what}: la conductividad k, la densidad ρ y el calor específico c tienen que ser "
            "positivos."
        )


def _check_temperatures(T_i: float, T_inf: float, fluid: str = "la del fluido") -> None:
    if not (T_i > 0.0 and T_inf > 0.0):
        raise ValueError("Las temperaturas tienen que estar por encima del cero absoluto.")
    if abs(T_i - T_inf) < 1e-9:
        raise ValueError(
            f"La temperatura inicial y {fluid} son iguales: no hay transitorio (el cuerpo ya "
            "está en equilibrio)."
        )


def _check_target(T_target: float, T_i: float, T_inf: float, what: str) -> float:
    """θ de una temperatura buscada; tiene que estar estrictamente entre T_i y T∞."""
    theta_t = (T_target - T_inf) / (T_i - T_inf)
    if not 0.0 < theta_t < 1.0:
        raise ValueError(
            f"{what} ({_degC(T_target)}) tiene que estar entre la temperatura inicial "
            f"({_degC(T_i)}) y la del fluido ({_degC(T_inf)}): el cuerpo va de una a la otra "
            "y nunca pasa ninguna de las dos."
        )
    return theta_t


# ---------------------------------------------------------------------
# Serie de Fourier: pared, cilindro y esfera (Çengel y Ghajar §4-2)
# ---------------------------------------------------------------------

Geometry1D = Literal["wall", "cylinder", "sphere"]
GEOMETRIES: dict[Geometry1D, str] = {
    "wall": "Pared plana",
    "cylinder": "Cilindro largo",
    "sphere": "Esfera",
}


@lru_cache(maxsize=4096)
def eigenvalues(geometry: Geometry1D, Bi: float, n: int = 1) -> tuple[float, ...]:
    """Las n primeras raíces λ de la ecuación característica (Çengel y Ghajar §4-2).

    - pared: λ·tan λ = Bi (una raíz en cada intervalo ((k)π, (k + ½)π));
    - cilindro: λ·J₁(λ) − Bi·J₀(λ) = 0 (una entre cada cero de J₁ y el siguiente de J₀);
    - esfera: 1 − λ·cot λ = Bi (una en cada ((k)π, (k + 1)π)).
    """
    if not Bi > 0.0:
        raise ValueError("El número de Biot tiene que ser positivo (h > 0).")
    roots: list[float] = []
    if geometry == "wall":

        def f(x: float) -> float:
            return x * math.sin(x) - Bi * math.cos(x)

        for k in range(n):
            roots.append(brentq(f, k * math.pi + 1e-14, k * math.pi + math.pi / 2.0, xtol=1e-15))
    elif geometry == "sphere":

        def f(x: float) -> float:
            return x * math.cos(x) - (1.0 - Bi) * math.sin(x)

        for k in range(n):
            a = k * math.pi + (1e-9 if k == 0 else 0.0)
            b = (k + 1) * math.pi
            if k > 0:
                a += 1e-12
            roots.append(brentq(f, a, b - 1e-12, xtol=1e-15))
    elif geometry == "cylinder":

        def f(x: float) -> float:
            return float(x * j1(x) - Bi * j0(x))

        zeros0 = jn_zeros(0, n)
        zeros1 = jn_zeros(1, max(n - 1, 1))
        for k in range(n):
            a = 1e-14 if k == 0 else float(zeros1[k - 1])
            roots.append(brentq(f, a, float(zeros0[k]), xtol=1e-15))
    else:
        raise ValueError(f"Geometría desconocida: {geometry!r}.")
    return tuple(roots)


def coefficient(geometry: Geometry1D, lam: float) -> float:
    """A_n de la serie (Çengel y Ghajar, tabla 4-1)."""
    if geometry == "wall":
        return 4.0 * math.sin(lam) / (2.0 * lam + math.sin(2.0 * lam))
    if geometry == "cylinder":
        return 2.0 / lam * float(j1(lam)) / (float(j0(lam)) ** 2 + float(j1(lam)) ** 2)
    if lam < 1e-3:  # sin λ − λ·cos λ ≈ λ³/3 − λ⁵/30 y 2λ − sin 2λ ≈ 4λ³/3 − 4λ⁵/15
        return (1.0 - lam**2 / 10.0) / (1.0 - lam**2 / 5.0)
    return 4.0 * (math.sin(lam) - lam * math.cos(lam)) / (2.0 * lam - math.sin(2.0 * lam))


def shape_function(geometry: Geometry1D, lam: float, xi: float) -> float:
    """X(λ·ξ): cos (pared), J₀ (cilindro) o sin(λξ)/(λξ) (esfera), con ξ = x/L o r/r₀."""
    if geometry == "wall":
        return math.cos(lam * xi)
    if geometry == "cylinder":
        return float(j0(lam * xi))
    z = lam * xi
    return 1.0 if z < 1e-8 else math.sin(z) / z


def _heat_weight(geometry: Geometry1D, lam: float) -> float:
    """El peso de cada término en Q/Q_máx = 1 − Σ A_n·e^(−λ²τ)·w_n (tabla 4-1)."""
    if geometry == "wall":
        return math.sin(lam) / lam
    if geometry == "cylinder":
        return 2.0 * float(j1(lam)) / lam
    if lam < 1e-3:
        return 1.0 - lam**2 / 10.0
    return 3.0 * (math.sin(lam) - lam * math.cos(lam)) / lam**3


def _n_terms(tau: float) -> int:
    """Cuántos términos hacen falta: e^(−λ_N²·τ) < 10⁻¹² con λ_N ≈ (N − 1)·π."""
    if tau <= 0.0:
        return _SERIES_MAX
    need = math.sqrt(-math.log(_SERIES_TOL) / tau) / math.pi + 2.0
    return int(min(max(need, 1.0), _SERIES_MAX))


def theta(
    geometry: Geometry1D, Bi: float, tau: float, xi: float, terms: int | None = None
) -> float:
    """θ = (T − T∞)/(T_i − T∞) con la serie (``terms=None``: los que hagan falta) o con
    ``terms`` términos (1 = la solución de un término)."""
    n = terms if terms is not None else _n_terms(tau)
    total = 0.0
    for lam in eigenvalues(geometry, Bi, n):
        total += (
            coefficient(geometry, lam)
            * math.exp(-lam * lam * tau)
            * shape_function(geometry, lam, xi)
        )
    return total


def heat_ratio(geometry: Geometry1D, Bi: float, tau: float, terms: int | None = None) -> float:
    """Q/Q_máx: la fracción de la energía que el cuerpo puede intercambiar (tabla 4-1)."""
    n = terms if terms is not None else _n_terms(tau)
    s = 0.0
    for lam in eigenvalues(geometry, Bi, n):
        s += coefficient(geometry, lam) * math.exp(-lam * lam * tau) * _heat_weight(geometry, lam)
    return 1.0 - s


def _solve_tau(fn: Callable[[float], float], target: float) -> float:
    """El τ en el que la función decreciente de τ vale ``target``."""
    lo, hi = 1e-10, 1.0
    while fn(hi) > target:
        hi *= 2.0
        if hi > 1e7:
            raise ValueError("No se alcanza esa temperatura en un tiempo razonable.")
    return brentq(lambda t: fn(t) - target, lo, hi, xtol=1e-14, rtol=1e-12)


# ---------------------------------------------------------------------
# Capacidad concentrada (Çengel y Ghajar §4-1)
# ---------------------------------------------------------------------

Body = Literal["sphere", "cylinder", "long_cylinder", "plate", "cube", "box", "custom"]
BODIES: dict[Body, str] = {
    "sphere": "Esfera",
    "cylinder": "Cilindro (con las tapas)",
    "long_cylinder": "Cilindro largo (sin las tapas)",
    "plate": "Placa (por las dos caras)",
    "cube": "Cubo",
    "box": "Prisma rectangular",
    "custom": "V y A a elección",
}
#: Las medidas de cada cuerpo, en el orden de ``LumpedInputs.dims``.
BODY_DIMENSIONS: dict[Body, tuple[str, ...]] = {
    "sphere": ("D",),
    "cylinder": ("D", "L"),
    "long_cylinder": ("D",),
    "plate": ("e",),
    "cube": ("a",),
    "box": ("a", "b", "c"),
    "custom": ("V", "A"),
}


def body_volume_area(body: Body, dims: tuple[float, ...]) -> tuple[float, float]:
    """V y A del cuerpo; el cilindro largo y la placa, por metro de largo o por m² de cara."""
    need = BODY_DIMENSIONS[body]
    if len(dims) != len(need) or not all(d > 0.0 for d in dims):
        raise ValueError(
            f"{BODIES[body]}: hacen falta {len(need)} medidas positivas ({', '.join(need)})."
        )
    if body == "sphere":
        D = dims[0]
        return math.pi * D**3 / 6.0, math.pi * D**2
    if body == "cylinder":
        D, L = dims
        return math.pi * D**2 / 4.0 * L, math.pi * D * L + 2.0 * math.pi * D**2 / 4.0
    if body == "long_cylinder":
        D = dims[0]
        return math.pi * D**2 / 4.0, math.pi * D
    if body == "plate":
        return dims[0], 2.0  # por m² de cara: el volumen e·1 y las dos caras
    if body == "cube":
        a = dims[0]
        return a**3, 6.0 * a**2
    if body == "box":
        a, b, c = dims
        return a * b * c, 2.0 * (a * b + b * c + c * a)
    return dims[0], dims[1]


@dataclass(frozen=True)
class LumpedInputs:
    """Un cuerpo de temperatura uniforme en un fluido a T∞ con h constante.

    Con ``time_s`` da la temperatura; con ``T_target_K``, el tiempo que tarda en llegar.
    """

    body: Body
    dims: tuple[float, ...]
    solid: Solid
    h: float
    T_i: float
    T_inf: float
    time_s: float | None = None
    T_target_K: float | None = None


@dataclass(frozen=True)
class LumpedResult:
    inputs: LumpedInputs
    V: float
    A: float
    time_s: float
    exact_mean_K: float | None  # la temperatura media con la solución exacta (§4-2 y §4-4)
    exact_center_K: float | None

    @property
    def Lc(self) -> float:
        """Longitud característica L_c = V/A."""
        return self.V / self.A

    @property
    def Bi(self) -> float:
        return self.inputs.h * self.Lc / self.inputs.solid.k

    @property
    def b(self) -> float:
        """b = h·A/(ρ·V·c) [1/s]."""
        s = self.inputs.solid
        return self.inputs.h * self.A / (s.rho * self.V * s.c)

    @property
    def time_constant_s(self) -> float:
        """La constante de tiempo 1/b: en ella se recorre el 63,2 % de la diferencia."""
        return 1.0 / self.b

    @property
    def theta(self) -> float:
        return math.exp(-self.b * self.time_s)

    @property
    def T_K(self) -> float:
        i = self.inputs
        return i.T_inf + (i.T_i - i.T_inf) * self.theta

    @property
    def mass(self) -> float:
        return self.inputs.solid.rho * self.V

    @property
    def Q_max_J(self) -> float:
        """Q_máx = m·c·(T_i − T∞): positiva si el cuerpo se enfría (la energía que entrega)."""
        i = self.inputs
        return self.mass * i.solid.c * (i.T_i - i.T_inf)

    @property
    def Q_J(self) -> float:
        """La energía que el cuerpo entregó al fluido hasta t (negativa si la recibió)."""
        i = self.inputs
        return self.mass * i.solid.c * (i.T_i - self.T_K)

    @property
    def heat_rate_W(self) -> float:
        """El calor que entrega en el instante t: h·A·(T − T∞)."""
        return self.inputs.h * self.A * (self.T_K - self.inputs.T_inf)

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(_lumped_notes(self))


def _exact_for_lumped(inp: LumpedInputs, t: float) -> tuple[float, float] | None:
    """La temperatura media y la del centro con la solución exacta (si el cuerpo la tiene)."""
    s, h = inp.solid, inp.h
    if inp.body == "sphere":
        factors = [("sphere", inp.dims[0] / 2.0)]
    elif inp.body == "long_cylinder":
        factors = [("cylinder", inp.dims[0] / 2.0)]
    elif inp.body == "plate":
        factors = [("wall", inp.dims[0] / 2.0)]
    elif inp.body == "cylinder":
        factors = [("cylinder", inp.dims[0] / 2.0), ("wall", inp.dims[1] / 2.0)]
    elif inp.body == "cube":
        factors = [("wall", inp.dims[0] / 2.0)] * 3
    elif inp.body == "box":
        factors = [("wall", d / 2.0) for d in inp.dims]
    else:
        return None
    if t <= 0.0:
        return inp.T_i, inp.T_i
    centers, ratios = [], []
    for geom, size in factors:
        Bi = h * size / s.k
        tau = s.alpha * t / size**2
        centers.append(theta(geom, Bi, tau, 0.0))  # type: ignore[arg-type]
        ratios.append(heat_ratio(geom, Bi, tau))  # type: ignore[arg-type]
    q = _combine_ratios(ratios)
    mean = inp.T_inf + (inp.T_i - inp.T_inf) * (1.0 - q)
    center = inp.T_inf + (inp.T_i - inp.T_inf) * math.prod(centers)
    return mean, center


def solve_lumped(inputs: LumpedInputs) -> LumpedResult:
    """Capacidad concentrada (Çengel y Ghajar §4-1; Incropera §5.1 y §5.2)."""
    _check_solid(inputs.solid)
    if not inputs.h > 0.0:
        raise ValueError(
            "El coeficiente h tiene que ser positivo: sin convección no hay transitorio."
        )
    _check_temperatures(inputs.T_i, inputs.T_inf)
    V, A = body_volume_area(inputs.body, inputs.dims)
    b = inputs.h * A / (inputs.solid.rho * V * inputs.solid.c)
    if (inputs.time_s is None) == (inputs.T_target_K is None):
        raise ValueError("Dá el tiempo (para la temperatura) o la temperatura (para el tiempo).")
    if inputs.time_s is not None:
        if inputs.time_s < 0.0:
            raise ValueError("El tiempo no puede ser negativo.")
        t = inputs.time_s
    else:
        th = _check_target(
            inputs.T_target_K or 0.0, inputs.T_i, inputs.T_inf, "La temperatura buscada"
        )
        t = -math.log(th) / b
    exact = _exact_for_lumped(inputs, t)
    return LumpedResult(
        inputs=inputs,
        V=V,
        A=A,
        time_s=t,
        exact_mean_K=exact[0] if exact else None,
        exact_center_K=exact[1] if exact else None,
    )


def _lumped_notes(r: LumpedResult) -> list[str]:
    i = r.inputs
    notes: list[str] = []
    if r.Bi <= BI_LUMPED:
        notes.append(
            f"Bi = {_num(r.Bi)} ≤ 0,1: la conducción dentro del cuerpo es mucho más rápida que "
            "la convección afuera, así que la temperatura es prácticamente uniforme y la "
            "capacidad concentrada vale (Çengel y Ghajar, §4-1)."
        )
    else:
        text = (
            f"Bi = {_num(r.Bi)} > 0,1: la temperatura **no** es uniforme y la capacidad "
            "concentrada es solo una estimación."
        )
        if r.exact_mean_K is not None and r.exact_center_K is not None:
            text += (
                f" Con la solución exacta, a {_time_text(r.time_s)} la temperatura media sería "
                f"{_degC(r.exact_mean_K)} y el centro estaría a {_degC(r.exact_center_K)} "
                f"(la capacidad concentrada da {_degC(r.T_K)})."
            )
        notes.append(text)
    notes.append(
        f"La constante de tiempo es 1/b = {_time_text(r.time_constant_s)}: en ese tiempo el "
        "cuerpo recorre el 63,2 % de la diferencia con el fluido, y en cinco constantes el "
        "99,3 %. Con más h, o un cuerpo más chico (menos V/A), responde más rápido."
    )
    if i.T_target_K is None and r.theta < 0.01:
        notes.append(
            "Ya recorrió más del 99 % de la diferencia: está prácticamente a la temperatura "
            "del fluido."
        )
    return notes


def lumped_curve(r: LumpedResult, points: int = 160) -> tuple[list[float], list[float]]:
    """(t, T) desde 0 hasta un poco después del instante pedido (o de 4 constantes)."""
    t_end = max(1.25 * r.time_s, 4.0 * r.time_constant_s if r.time_s == 0.0 else 0.0)
    t_end = max(t_end, 1e-9)
    ts = [t_end * k / (points - 1) for k in range(points)]
    i = r.inputs
    return ts, [i.T_inf + (i.T_i - i.T_inf) * math.exp(-r.b * t) for t in ts]


# ---------------------------------------------------------------------
# Pared, cilindro y esfera (Çengel y Ghajar §4-2)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class OneDimInputs:
    """Una pared (de espesor 2L, o L con la otra cara aislada), un cilindro largo o una
    esfera de radio r₀, en un fluido a T∞ con h constante.

    ``position`` es ξ = x/L o r/r₀ (0 el centro, 1 la superficie). Con ``time_s`` da las
    temperaturas; con ``T_center_target_K``, el tiempo en que el centro llega a esa
    temperatura.
    """

    geometry: Geometry1D
    size: float  # L o r₀
    solid: Solid
    h: float
    T_i: float
    T_inf: float
    position: float = 0.0
    time_s: float | None = None
    T_center_target_K: float | None = None


@dataclass(frozen=True)
class OneDimResult:
    inputs: OneDimInputs
    time_s: float
    time_one_term_s: float | None  # el tiempo con la solución de un término (si se buscó t)

    @property
    def Bi(self) -> float:
        return self.inputs.h * self.inputs.size / self.inputs.solid.k

    @property
    def tau(self) -> float:
        """τ = α·t/L² (número de Fourier)."""
        return self.inputs.solid.alpha * self.time_s / self.inputs.size**2

    @property
    def lambda1(self) -> float:
        return eigenvalues(self.inputs.geometry, self.Bi, 1)[0]

    @property
    def A1(self) -> float:
        return coefficient(self.inputs.geometry, self.lambda1)

    @property
    def X1(self) -> float:
        """X(λ₁·ξ) en la posición pedida."""
        return shape_function(self.inputs.geometry, self.lambda1, self.inputs.position)

    @property
    def theta0_one(self) -> float:
        return self.A1 * math.exp(-(self.lambda1**2) * self.tau)

    @property
    def theta_one(self) -> float:
        return self.theta0_one * self.X1

    @property
    def theta0(self) -> float:
        """θ del centro con la serie completa."""
        return theta(self.inputs.geometry, self.Bi, self.tau, 0.0)

    @property
    def theta(self) -> float:
        """θ en la posición pedida con la serie completa."""
        return theta(self.inputs.geometry, self.Bi, self.tau, self.inputs.position)

    def T(self, th: float) -> float:
        i = self.inputs
        return i.T_inf + (i.T_i - i.T_inf) * th

    @property
    def T_center_K(self) -> float:
        return self.T(self.theta0)

    @property
    def T_K(self) -> float:
        return self.T(self.theta)

    @property
    def T_center_one_K(self) -> float:
        return self.T(self.theta0_one)

    @property
    def T_one_K(self) -> float:
        return self.T(self.theta_one)

    @property
    def T_surface_K(self) -> float:
        return self.T(theta(self.inputs.geometry, self.Bi, self.tau, 1.0))

    @property
    def heat_ratio_one(self) -> float:
        g = self.inputs.geometry
        return 1.0 - self.theta0_one * _heat_weight(g, self.lambda1)

    @property
    def heat_ratio(self) -> float:
        return heat_ratio(self.inputs.geometry, self.Bi, self.tau)

    @property
    def mass_measure(self) -> float:
        """La masa por m² de cara (pared, toda la placa de espesor 2L), por metro
        (cilindro) o total (esfera)."""
        i = self.inputs
        L = i.size
        if i.geometry == "wall":
            return i.solid.rho * 2.0 * L
        if i.geometry == "cylinder":
            return i.solid.rho * math.pi * L**2
        return i.solid.rho * 4.0 / 3.0 * math.pi * L**3

    @property
    def Q_max(self) -> float:
        """m·c·(T_i − T∞), por m² de cara (pared), por metro (cilindro) o total (esfera)."""
        i = self.inputs
        return self.mass_measure * i.solid.c * (i.T_i - i.T_inf)

    @property
    def Q(self) -> float:
        return self.heat_ratio * self.Q_max

    @property
    def Q_one(self) -> float:
        return self.heat_ratio_one * self.Q_max

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(_one_dim_notes(self))


def _check_one_dim(i: OneDimInputs) -> None:
    if i.geometry not in GEOMETRIES:
        raise ValueError(f"Geometría desconocida: {i.geometry!r}.")
    _check_solid(i.solid)
    if not i.size > 0.0:
        what = "El semiespesor L" if i.geometry == "wall" else "El radio r₀"
        raise ValueError(f"{what} tiene que ser positivo.")
    if not i.h > 0.0:
        raise ValueError(
            "El coeficiente h tiene que ser positivo: sin convección no hay transitorio."
        )
    _check_temperatures(i.T_i, i.T_inf)
    if not 0.0 <= i.position <= 1.0:
        raise ValueError(
            "La posición x/L (o r/r₀) tiene que estar entre 0 (el centro) y 1 (la superficie)."
        )


def solve_one_dimensional(inputs: OneDimInputs) -> OneDimResult:
    """Pared plana, cilindro largo o esfera con convección (Çengel y Ghajar §4-2)."""
    _check_one_dim(inputs)
    if (inputs.time_s is None) == (inputs.T_center_target_K is None):
        raise ValueError(
            "Dá el tiempo (para las temperaturas) o la temperatura del centro (para el tiempo)."
        )
    g, alpha, L = inputs.geometry, inputs.solid.alpha, inputs.size
    Bi = inputs.h * L / inputs.solid.k
    if inputs.time_s is not None:
        if not inputs.time_s > 0.0:
            raise ValueError("El tiempo tiene que ser positivo.")
        return OneDimResult(inputs, inputs.time_s, None)
    th = _check_target(
        inputs.T_center_target_K or 0.0, inputs.T_i, inputs.T_inf, "La temperatura del centro"
    )
    tau = _solve_tau(lambda t: theta(g, Bi, t, 0.0), th)
    lam = eigenvalues(g, Bi, 1)[0]
    A1 = coefficient(g, lam)
    tau_one = -math.log(th / A1) / lam**2 if th < A1 else 0.0
    return OneDimResult(inputs, tau * L**2 / alpha, tau_one * L**2 / alpha)


def _one_dim_notes(r: OneDimResult) -> list[str]:
    i = r.inputs
    notes: list[str] = []
    diff = abs(r.T_center_one_K - r.T_center_K)
    if r.tau < TAU_ONE_TERM:
        notes.append(
            f"τ = {_num(r.tau)} < 0,2: el primer término de la serie no alcanza (en el centro da "
            f"{_degC(r.T_center_one_K)} contra {_degC(r.T_center_K)} de la serie). Al "
            "principio el calor todavía no llegó al centro y hacen falta más términos; "
            "los gráficos de Heisler tampoco sirven ahí."
        )
    else:
        notes.append(
            f"τ = {_num(r.tau)} ≥ 0,2: la solución de un término (la de las tablas de λ₁ y A₁) "
            f"se aparta de la serie completa en {_dT_text(diff)} en el centro."
        )
    if r.Bi <= BI_LUMPED:
        exponent = {"wall": "Bi·τ", "cylinder": "2·Bi·τ", "sphere": "3·Bi·τ"}[i.geometry]
        lumped_b = {"wall": 1.0, "cylinder": 2.0, "sphere": 3.0}[i.geometry]
        th = math.exp(-lumped_b * r.Bi * r.tau)
        notes.append(
            f"Bi = {_num(r.Bi)} ≤ 0,1: la capacidad concentrada (θ = e^(−{exponent})) habría "
            f"dado {_degC(r.T(th))} para todo el cuerpo; la temperatura media exacta es "
            f"{_degC(r.T(1.0 - r.heat_ratio))}."
        )
    if r.Bi > 100.0:
        notes.append(
            f"Bi = {_num(r.Bi)}: la convección es tanto más rápida que la conducción que la "
            "superficie queda prácticamente a T∞ desde el principio (h → ∞)."
        )
    if r.heat_ratio > 0.99:
        notes.append(
            f"Ya intercambió el {_pct(r.heat_ratio)} de Q_máx: el cuerpo está prácticamente en "
            "equilibrio con el fluido."
        )
    if i.geometry == "wall":
        notes.append(
            "La pared tiene espesor 2L y se enfría por las dos caras (x = 0 es el plano medio); "
            "con una cara aislada vale lo mismo con espesor L, y x = 0 en la aislada."
        )
    return notes


def time_profiles(
    r: OneDimResult, count: int = 5, points: int = 61
) -> list[tuple[float, list[float], list[float]]]:
    """Perfiles T(ξ) a ``count`` tiempos hasta el pedido: [(t, ξ, T)]."""
    i = r.inputs
    out = []
    for k in range(1, count + 1):
        t = r.time_s * k / count
        tau = i.solid.alpha * t / i.size**2
        xs = [j / (points - 1) for j in range(points)]
        out.append((t, xs, [r.T(theta(i.geometry, r.Bi, tau, x)) for x in xs]))
    return out


def center_history(
    r: OneDimResult, points: int = 120
) -> tuple[list[float], list[float], list[float]]:
    """(t, T del centro con la serie, T del centro con un término) hasta 1,5·t."""
    i = r.inputs
    t_end = 1.5 * r.time_s
    ts = [t_end * k / (points - 1) for k in range(1, points)]
    series, one = [], []
    for t in ts:
        tau = i.solid.alpha * t / i.size**2
        series.append(r.T(theta(i.geometry, r.Bi, tau, 0.0)))
        one.append(r.T(min(r.A1 * math.exp(-(r.lambda1**2) * tau), 1.5)))
    return ts, series, one


# ---------------------------------------------------------------------
# Sólido semi-infinito (Çengel y Ghajar §4-3)
# ---------------------------------------------------------------------

SemiBoundary = Literal["temperature", "flux", "convection"]
SEMI_BOUNDARIES: dict[SemiBoundary, str] = {
    "temperature": "Superficie a temperatura constante",
    "flux": "Flujo de calor constante en la superficie",
    "convection": "Convección en la superficie",
}


@dataclass(frozen=True)
class SemiInfiniteInputs:
    """Un sólido semi-infinito a T_i con la superficie a T_s, con un flujo q̇_s que entra o
    con convección (h, T∞).

    De ``depth_m``, ``time_s`` y ``T_target_K`` se dan dos: la profundidad y el tiempo dan
    la temperatura; el tiempo y una temperatura, la profundidad; la profundidad y una
    temperatura, el tiempo.
    """

    boundary: SemiBoundary
    solid: Solid
    T_i: float
    T_s: float | None = None
    q_s: float | None = None
    h: float | None = None
    T_inf: float | None = None
    depth_m: float | None = None
    time_s: float | None = None
    T_target_K: float | None = None


def semi_infinite_theta(inp: SemiInfiniteInputs, x: float, t: float) -> float:
    """(T − T_i) de la solución, en K, a la profundidad x y el tiempo t (§4-3)."""
    a = inp.solid.alpha
    k = inp.solid.k
    if t <= 0.0:
        return 0.0
    sq = math.sqrt(a * t)
    eta = x / (2.0 * sq)
    if inp.boundary == "temperature":
        return ((inp.T_s or 0.0) - inp.T_i) * float(erfc(eta))
    if inp.boundary == "flux":
        q = inp.q_s or 0.0
        return 2.0 * q * sq / (math.sqrt(math.pi) * k) * math.exp(-eta * eta) - q * x / k * float(
            erfc(eta)
        )
    h = inp.h or 0.0
    beta = h * sq / k
    # e^(h·x/k + β²)·erfc(η + β) = e^(−η²)·erfcx(η + β): sin desbordes
    ratio = float(erfc(eta)) - math.exp(-eta * eta) * float(erfcx(eta + beta))
    return ((inp.T_inf or 0.0) - inp.T_i) * ratio


def _surface_flux(inp: SemiInfiniteInputs, t: float) -> float:
    """q̇ en la superficie (W/m²), positivo si entra al sólido."""
    k, a = inp.solid.k, inp.solid.alpha
    if inp.boundary == "temperature":
        return k * ((inp.T_s or 0.0) - inp.T_i) / math.sqrt(math.pi * a * t)
    if inp.boundary == "flux":
        return inp.q_s or 0.0
    T_s = inp.T_i + semi_infinite_theta(inp, 0.0, t)
    return (inp.h or 0.0) * ((inp.T_inf or 0.0) - T_s)


def _energy_per_area(inp: SemiInfiniteInputs, t: float) -> float:
    """La energía que entró por m² hasta t (J/m²)."""
    k, a = inp.solid.k, inp.solid.alpha
    if inp.boundary == "temperature":
        return 2.0 * k * ((inp.T_s or 0.0) - inp.T_i) * math.sqrt(t / (math.pi * a))
    if inp.boundary == "flux":
        return (inp.q_s or 0.0) * t
    value, _ = quad(lambda s: _surface_flux(inp, s), 0.0, t, limit=200)
    return value


@dataclass(frozen=True)
class SemiInfiniteResult:
    inputs: SemiInfiniteInputs
    depth_m: float
    time_s: float

    @property
    def sqrt_alpha_t(self) -> float:
        return math.sqrt(self.inputs.solid.alpha * self.time_s)

    @property
    def eta(self) -> float:
        """η = x/(2·√(α·t))."""
        return self.depth_m / (2.0 * self.sqrt_alpha_t)

    @property
    def beta(self) -> float | None:
        """h·√(α·t)/k, el otro parámetro de la convección."""
        i = self.inputs
        if i.boundary != "convection":
            return None
        return (i.h or 0.0) * self.sqrt_alpha_t / i.solid.k

    @property
    def T_K(self) -> float:
        return self.inputs.T_i + semi_infinite_theta(self.inputs, self.depth_m, self.time_s)

    @property
    def T_surface_K(self) -> float:
        return self.inputs.T_i + semi_infinite_theta(self.inputs, 0.0, self.time_s)

    @property
    def surface_flux(self) -> float:
        return _surface_flux(self.inputs, self.time_s)

    @property
    def energy_per_area(self) -> float:
        return _energy_per_area(self.inputs, self.time_s)

    @property
    def penetration_m(self) -> float:
        """Hasta dónde llegó la perturbación: donde cambió el 1 % (erfc(η) = 0,01)."""
        return 2.0 * _ETA_1PCT * self.sqrt_alpha_t

    @property
    def notes(self) -> tuple[str, ...]:
        return tuple(_semi_notes(self))


def _check_semi(i: SemiInfiniteInputs) -> None:
    if i.boundary not in SEMI_BOUNDARIES:
        raise ValueError(f"Condición de borde desconocida: {i.boundary!r}.")
    _check_solid(i.solid)
    if i.boundary == "temperature":
        if i.T_s is None:
            raise ValueError("Falta la temperatura de la superficie T_s.")
        _check_temperatures(i.T_i, i.T_s, "la de la superficie")
    elif i.boundary == "flux":
        if i.q_s is None or i.q_s == 0.0:
            raise ValueError("Falta el flujo de calor q̇_s (distinto de cero).")
        if not i.T_i > 0.0:
            raise ValueError("La temperatura inicial tiene que estar por encima del cero absoluto.")
    else:
        if i.h is None or not i.h > 0.0 or i.T_inf is None:
            raise ValueError("Faltan h (positivo) y la temperatura del fluido T∞.")
        _check_temperatures(i.T_i, i.T_inf)
    given = sum(v is not None for v in (i.depth_m, i.time_s, i.T_target_K))
    if given != 2:
        raise ValueError(
            "Dá dos de tres: la profundidad, el tiempo y una temperatura (con la profundidad y "
            "el tiempo sale la temperatura; con una temperatura, lo que falte)."
        )
    if i.depth_m is not None and i.depth_m < 0.0:
        raise ValueError("La profundidad no puede ser negativa (x = 0 es la superficie).")
    if i.time_s is not None and not i.time_s > 0.0:
        raise ValueError("El tiempo tiene que ser positivo.")


def _target_bounds(i: SemiInfiniteInputs) -> tuple[float, float]:
    """(T_i, el extremo al que tiende la superficie) para validar una temperatura buscada."""
    if i.boundary == "temperature":
        return i.T_i, i.T_s or 0.0
    if i.boundary == "convection":
        return i.T_i, i.T_inf or 0.0
    return i.T_i, math.inf if (i.q_s or 0.0) > 0.0 else -math.inf


def solve_semi_infinite(inputs: SemiInfiniteInputs) -> SemiInfiniteResult:
    """Sólido semi-infinito (Çengel y Ghajar §4-3; Incropera §5.7)."""
    _check_semi(inputs)
    i = inputs
    if i.depth_m is not None and i.time_s is not None:
        return SemiInfiniteResult(i, i.depth_m, i.time_s)
    lo, hi = _target_bounds(i)
    T_t = i.T_target_K or 0.0
    inside = min(lo, hi) < T_t < max(lo, hi)
    if not inside:
        end = "la de la superficie" if i.boundary == "temperature" else "la del fluido"
        if i.boundary == "flux":
            end = "la que impone el flujo (sube si entra calor, baja si sale)"
        raise ValueError(
            f"La temperatura buscada ({_degC(T_t)}) tiene que estar entre la inicial "
            f"({_degC(i.T_i)}) y {end}: el sólido va de una hacia la otra."
        )
    dT = T_t - i.T_i
    if i.time_s is not None:  # la profundidad a la que llega T_t
        t = i.time_s

        def at(x: float) -> float:
            return semi_infinite_theta(i, x, t) - dT

        if at(0.0) * dT <= 0.0:  # todavía no llegó ni a la superficie
            raise ValueError(
                f"A {_time_text(t)} ni la superficie llegó a {_degC(T_t)}: dale más tiempo."
            )
        x_hi = 2.0 * math.sqrt(i.solid.alpha * t)
        while at(x_hi) * dT > 0.0:
            x_hi *= 2.0
        x = brentq(at, 0.0, x_hi, xtol=1e-12)
        return SemiInfiniteResult(i, x, t)
    x = i.depth_m or 0.0  # el tiempo que tarda en llegar a la profundidad x

    def when(s: float) -> float:
        return semi_infinite_theta(i, x, s) - dT

    t_hi = max(x * x / i.solid.alpha, 1.0)
    while when(t_hi) * dT < 0.0:
        t_hi *= 2.0
        if t_hi > 1e13:
            raise ValueError("No se alcanza esa temperatura en un tiempo razonable.")
    t_lo = t_hi * 1e-12
    t = brentq(when, t_lo, t_hi, xtol=1e-9, rtol=1e-12)
    return SemiInfiniteResult(i, x, t)


def _semi_notes(r: SemiInfiniteResult) -> list[str]:
    notes = [
        f"A {_time_text(r.time_s)} la perturbación llegó a unos {_len_text(r.penetration_m)} "
        "(donde la temperatura cambió el 1 %: δ = 3,64·√(α·t)). Un cuerpo más grueso que eso "
        "se comporta como semi-infinito; uno más delgado ya «siente» la otra cara."
    ]
    if r.inputs.boundary == "convection":
        notes.append(
            "Con convección la superficie no salta a T∞: arranca en T_i y se acerca a T∞ más "
            "rápido cuanto mayor es h·√(α·t)/k (con h → ∞ es el caso de la superficie a T_s)."
        )
    if r.inputs.boundary == "flux":
        notes.append(
            "Con un flujo constante la superficie sube sin parar, como √t: el sólido no llega a "
            "un equilibrio mientras le entre calor."
        )
    return notes


def semi_infinite_profiles(
    r: SemiInfiniteResult, count: int = 4, points: int = 80
) -> list[tuple[float, list[float], list[float]]]:
    """Perfiles T(x) a ``count`` tiempos hasta el pedido: [(t, x, T)]."""
    i = r.inputs
    x_end = max(1.2 * r.penetration_m, 1.5 * r.depth_m, 1e-6)
    out = []
    for k in range(1, count + 1):
        t = r.time_s * k / count
        xs = [x_end * j / (points - 1) for j in range(points)]
        out.append((t, xs, [i.T_i + semi_infinite_theta(i, x, t) for x in xs]))
    return out


@dataclass(frozen=True)
class ContactInputs:
    """Dos sólidos semi-infinitos que se tocan (§4-3): A a T_A y B a T_B."""

    solid_a: Solid
    T_a: float
    solid_b: Solid
    T_b: float
    time_s: float | None = None


@dataclass(frozen=True)
class ContactResult:
    inputs: ContactInputs

    @property
    def T_contact_K(self) -> float:
        """T_s = (√(kρc)_A·T_A + √(kρc)_B·T_B)/(√(kρc)_A + √(kρc)_B)."""
        i = self.inputs
        ea, eb = i.solid_a.effusivity, i.solid_b.effusivity
        return (ea * i.T_a + eb * i.T_b) / (ea + eb)

    @property
    def share_a(self) -> float:
        """El peso de A en la temperatura de contacto."""
        i = self.inputs
        ea, eb = i.solid_a.effusivity, i.solid_b.effusivity
        return ea / (ea + eb)

    @property
    def flux_W_m2(self) -> float | None:
        """El flujo que pasa de A a B a ``time_s``: k_A·(T_A − T_s)/√(π·α_A·t)."""
        i = self.inputs
        if i.time_s is None:
            return None
        a = i.solid_a
        return a.k * (i.T_a - self.T_contact_K) / math.sqrt(math.pi * a.alpha * i.time_s)

    @property
    def notes(self) -> tuple[str, ...]:
        i = self.inputs
        hot = i.solid_a if self.share_a >= 0.5 else i.solid_b
        return (
            f"La temperatura de contacto queda más cerca de la de {hot.name.lower()}, el de mayor "
            "√(k·ρ·c). Por eso un metal «se siente» frío y la madera no, aunque los dos estén a "
            "la misma temperatura: lo que siente la mano es la temperatura del contacto.",
            "La temperatura de contacto no cambia con el tiempo mientras los dos se comporten "
            "como semi-infinitos.",
        )


def solve_contact(inputs: ContactInputs) -> ContactResult:
    """Contacto de dos sólidos semi-infinitos (Çengel y Ghajar §4-3)."""
    _check_solid(inputs.solid_a, "El sólido A")
    _check_solid(inputs.solid_b, "El sólido B")
    if not (inputs.T_a > 0.0 and inputs.T_b > 0.0):
        raise ValueError("Las temperaturas tienen que estar por encima del cero absoluto.")
    if inputs.time_s is not None and not inputs.time_s > 0.0:
        raise ValueError("El tiempo tiene que ser positivo.")
    return ContactResult(inputs)


# ---------------------------------------------------------------------
# Multidimensional (Çengel y Ghajar §4-4)
# ---------------------------------------------------------------------

FactorKind = Literal["wall", "cylinder", "semi"]
ProductShape = Literal[
    "short_cylinder", "semi_cylinder", "bar", "box", "semi_plate", "quarter", "corner"
]
PRODUCT_SHAPES: dict[ProductShape, str] = {
    "short_cylinder": "Cilindro corto",
    "semi_cylinder": "Cilindro semi-infinito",
    "bar": "Barra rectangular larga",
    "box": "Paralelepípedo",
    "semi_plate": "Placa semi-infinita",
    "quarter": "Medio cuarto-infinito (arista)",
    "corner": "Esquina de un medio infinito",
}


@dataclass(frozen=True)
class Factor:
    """Un factor del producto: una pared (L), un cilindro (r₀) o un semi-infinito.

    ``position`` es ξ = x/L o r/r₀ en la pared y el cilindro, y la profundidad x (m) en el
    semi-infinito.
    """

    kind: FactorKind
    size: float | None
    position: float
    label: str = ""


@dataclass(frozen=True)
class ProductInputs:
    """Un sólido que es la intersección de otros de una dimensión (§4-4)."""

    shape: ProductShape
    factors: tuple[Factor, ...]
    solid: Solid
    h: float
    T_i: float
    T_inf: float
    time_s: float


@dataclass(frozen=True)
class FactorResult:
    factor: Factor
    Bi: float | None
    tau: float | None
    theta: float
    heat_ratio: float | None


@dataclass(frozen=True)
class ProductResult:
    inputs: ProductInputs
    factors: tuple[FactorResult, ...]

    @property
    def theta(self) -> float:
        return math.prod(f.theta for f in self.factors)

    @property
    def T_K(self) -> float:
        i = self.inputs
        return i.T_inf + (i.T_i - i.T_inf) * self.theta

    @property
    def heat_ratio(self) -> float | None:
        """Q/Q_máx con la regla de Langston (solo si ningún factor es semi-infinito)."""
        ratios = [f.heat_ratio for f in self.factors]
        if any(q is None for q in ratios):
            return None
        return _combine_ratios([q for q in ratios if q is not None])

    @property
    def volume(self) -> float | None:
        sizes = []
        for f in self.inputs.factors:
            if f.kind == "semi" or f.size is None:
                return None
            sizes.append(f.size)
        if self.inputs.shape == "short_cylinder":
            r0 = next(f.size for f in self.inputs.factors if f.kind == "cylinder")
            L = next(f.size for f in self.inputs.factors if f.kind == "wall")
            return math.pi * (r0 or 0.0) ** 2 * 2.0 * (L or 0.0)
        if self.inputs.shape == "box":
            return math.prod(2.0 * s for s in sizes)
        return None

    @property
    def Q_max(self) -> float | None:
        V = self.volume
        if V is None:
            return None
        i = self.inputs
        return i.solid.rho * V * i.solid.c * (i.T_i - i.T_inf)

    @property
    def Q(self) -> float | None:
        if self.Q_max is None or self.heat_ratio is None:
            return None
        return self.heat_ratio * self.Q_max

    @property
    def notes(self) -> tuple[str, ...]:
        notes = [
            "La solución es el producto de las de una dimensión porque el sólido es la "
            "intersección de ellas: cada factor lleva su Bi y su τ con su propia longitud "
            "(Çengel y Ghajar, §4-4)."
        ]
        if self.heat_ratio is None:
            notes.append(
                "Con un factor semi-infinito no hay Q_máx (el sólido no tiene fin): solo se "
                "calcula la temperatura."
            )
        return tuple(notes)


def _combine_ratios(ratios: list[float]) -> float:
    """La regla de Langston (1982): Q/Q_máx de la intersección de dos o tres sólidos."""
    total = 0.0
    rest = 1.0
    for q in ratios:
        total += q * rest
        rest *= 1.0 - q
    return total


def product_factors(
    shape: ProductShape, dims: tuple[float, ...], point: tuple[float, ...]
) -> tuple[Factor, ...]:
    """Los factores de cada forma con sus medidas y el punto (ξ o profundidad).

    - cilindro corto: (D, H) y (r/r₀, x/L) con x desde el plano medio;
    - cilindro semi-infinito: (D,) y (r/r₀, profundidad desde la cara);
    - barra: (a, b) y (x/L₁, y/L₂); paralelepípedo: (a, b, c) y tres ξ;
    - placa semi-infinita: (e,) y (x/L, profundidad desde el borde);
    - arista y esquina: () y dos o tres profundidades.
    """
    if shape == "short_cylinder":
        D, H = dims
        return (
            Factor("cylinder", D / 2.0, point[0], "cilindro"),
            Factor("wall", H / 2.0, point[1], "pared"),
        )
    if shape == "semi_cylinder":
        return (
            Factor("cylinder", dims[0] / 2.0, point[0], "cilindro"),
            Factor("semi", None, point[1], "semi-infinito"),
        )
    if shape == "bar":
        return (
            Factor("wall", dims[0] / 2.0, point[0], "pared 1"),
            Factor("wall", dims[1] / 2.0, point[1], "pared 2"),
        )
    if shape == "box":
        return tuple(
            Factor("wall", d / 2.0, p, f"pared {k}")
            for k, (d, p) in enumerate(zip(dims, point, strict=True), 1)
        )
    if shape == "semi_plate":
        return (
            Factor("wall", dims[0] / 2.0, point[0], "pared"),
            Factor("semi", None, point[1], "semi-infinito"),
        )
    if shape == "quarter":
        return tuple(
            Factor("semi", None, p, f"semi-infinito {k}") for k, p in enumerate(point[:2], 1)
        )
    if shape == "corner":
        return tuple(
            Factor("semi", None, p, f"semi-infinito {k}") for k, p in enumerate(point[:3], 1)
        )
    raise ValueError(f"Forma desconocida: {shape!r}.")


def _factor_result(f: Factor, i: ProductInputs) -> FactorResult:
    s = i.solid
    if f.kind == "semi":
        semi = SemiInfiniteInputs(
            "convection", s, i.T_i, h=i.h, T_inf=i.T_inf, depth_m=f.position, time_s=i.time_s
        )
        th = 1.0 - semi_infinite_theta(semi, f.position, i.time_s) / (i.T_inf - i.T_i)
        return FactorResult(f, None, None, th, None)
    size = f.size or 0.0
    Bi = i.h * size / s.k
    tau = s.alpha * i.time_s / size**2
    g: Geometry1D = "wall" if f.kind == "wall" else "cylinder"
    return FactorResult(f, Bi, tau, theta(g, Bi, tau, f.position), heat_ratio(g, Bi, tau))


def solve_product(inputs: ProductInputs) -> ProductResult:
    """Producto de soluciones de una dimensión (Çengel y Ghajar §4-4; Incropera §5.8)."""
    _check_solid(inputs.solid)
    if not inputs.h > 0.0:
        raise ValueError("El coeficiente h tiene que ser positivo.")
    _check_temperatures(inputs.T_i, inputs.T_inf)
    if not inputs.time_s > 0.0:
        raise ValueError("El tiempo tiene que ser positivo.")
    if not inputs.factors:
        raise ValueError("Hace falta al menos un factor.")
    for f in inputs.factors:
        if f.kind == "semi":
            if f.position < 0.0:
                raise ValueError("La profundidad no puede ser negativa (0 es la cara expuesta).")
        else:
            if f.size is None or not f.size > 0.0:
                raise ValueError("Las medidas tienen que ser positivas.")
            if not 0.0 <= f.position <= 1.0:
                raise ValueError(
                    "Cada posición relativa (x/L o r/r₀) tiene que estar entre 0 (el centro) y 1 "
                    "(la superficie)."
                )
    return ProductResult(inputs, tuple(_factor_result(f, inputs) for f in inputs.factors))


def product_map(
    r: ProductResult, points: int = 41
) -> tuple[list[float], list[float], list[list[float]]] | None:
    """El mapa de temperatura en el plano de los dos primeros factores (en m), con el resto
    de los factores en el punto pedido: (eje 1, eje 2, T[eje 2][eje 1])."""
    i = r.inputs
    if len(i.factors) < 2:
        return None
    f1, f2 = i.factors[0], i.factors[1]
    rest = math.prod(fr.theta for fr in r.factors[2:])

    def axis(f: Factor) -> list[float]:
        if f.kind == "semi":
            depth = max(4.0 * math.sqrt(i.solid.alpha * i.time_s), 2.0 * f.position, 1e-6)
            return [depth * k / (points - 1) for k in range(points)]
        return [(f.size or 0.0) * k / (points - 1) for k in range(points)]

    def th(f: Factor, x_m: float) -> float:
        pos = x_m if f.kind == "semi" else x_m / (f.size or 1.0)
        return _factor_result(Factor(f.kind, f.size, pos), i).theta

    xs, ys = axis(f1), axis(f2)
    t1 = [th(f1, x) for x in xs]
    t2 = [th(f2, y) for y in ys]
    grid = [[i.T_inf + (i.T_i - i.T_inf) * a * b * rest for a in t1] for b in t2]
    return xs, ys, grid


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TransientExample:
    description: str
    inputs: Any


def _custom(name: str, k: float, alpha: float, rho: float = 1000.0) -> Solid:
    """Un sólido con k y α del enunciado: ρ·c = k/α (se elige ρ y sale c)."""
    return Solid(name, k, rho, k / (alpha * rho))


LUMPED_EXAMPLES: dict[str, TransientExample] = {
    "Termocupla en un gas (Çengel y Ghajar §4-1)": TransientExample(
        "La junta, una esfera de 1 mm (k = 35 W/(m·K), ρ = 8500 kg/m³, c = 320 J/(kg·K)), "
        "entra a un gas con h = 210 W/(m²·K). El libro: tarda 10 s en marcar el 99 % de la "
        "diferencia (acá, de 25 a 198,25 °C con el gas a 200 °C).",
        LumpedInputs(
            "sphere",
            (0.001,),
            Solid("Junta de termocupla", 35.0, 8500.0, 320.0),
            210.0,
            25.0 + C,
            200.0 + C,
            T_target_K=200.0 - 0.01 * 175.0 + C,
        ),
    ),
    "La hora de la muerte (Çengel y Ghajar §4-1)": TransientExample(
        "Un cuerpo (cilindro de 30 cm por 1,70 m, con las propiedades del agua) se encuentra a "
        "25 °C en un cuarto a 20 °C, con h = 8 W/(m²·K). El libro: murió unas 12,2 h antes. "
        "Con Bi = 0,89 la capacidad concentrada es solo una estimación.",
        LumpedInputs(
            "cylinder",
            (0.30, 1.70),
            Solid("Cuerpo humano (como agua)", 0.617, 996.0, 4178.0),
            8.0,
            37.0 + C,
            20.0 + C,
            T_target_K=25.0 + C,
        ),
    ),
    "Termocupla con constante de tiempo de 1 s (Incropera §5.2)": TransientExample(
        "Una junta esférica (k = 20 W/(m·K), c = 400 J/(kg·K), ρ = 8500 kg/m³) con h = 400 "
        "W/(m²·K): para 1 s de constante de tiempo, D = 0,706 mm. El libro: de 25 °C, en un gas "
        "a 200 °C, llega a 199 °C en 5,2 s.",
        LumpedInputs(
            "sphere",
            (6.0 * 400.0 * 1.0 / (8500.0 * 400.0),),
            Solid("Junta de termocupla", 20.0, 8500.0, 400.0),
            400.0,
            25.0 + C,
            200.0 + C,
            T_target_K=199.0 + C,
        ),
    ),
    "Bolas de acero en un horno (ejemplo propio)": TransientExample(
        "Bolas de acero al carbono de 2 cm a 20 °C entran a un horno a 800 °C con h = 100 "
        "W/(m²·K): la temperatura a los 5 min.",
        LumpedInputs(
            "sphere", (0.02,), SOLIDS["carbon_steel"], 100.0, 20.0 + C, 800.0 + C, time_s=300.0
        ),
    ),
}

ONE_DIM_EXAMPLES: dict[str, TransientExample] = {
    "Huevo hervido (Çengel y Ghajar §4-2)": TransientExample(
        "Un huevo (esfera de 5 cm con k = 0,6 W/(m·K) y α = 0,14·10⁻⁶ m²/s) a 5 °C entra a agua "
        "hirviendo a 95 °C con h = 1200 W/(m²·K). El libro: el centro llega a 70 °C en 933 s "
        "(redondea θ₀ a 0,278).",
        OneDimInputs(
            "sphere",
            0.025,
            _custom("Huevo (como agua)", 0.6, 0.14e-6),
            1200.0,
            5.0 + C,
            95.0 + C,
            T_center_target_K=70.0 + C,
        ),
    ),
    "Placas de bronce en un horno (Çengel y Ghajar §4-2)": TransientExample(
        "Placas de latón de 4 cm a 20 °C pasan 7 min en un horno a 500 °C con h = 120 W/(m²·K) "
        "(convección y radiación). El libro: salen con la superficie a 279 °C.",
        OneDimInputs(
            "wall", 0.02, SOLIDS["brass"], 120.0, 20.0 + C, 500.0 + C, position=1.0, time_s=420.0
        ),
    ),
    "Eje de acero inoxidable (Çengel y Ghajar §4-2)": TransientExample(
        "Un eje de 20 cm de AISI 304 sale del horno a 600 °C y se enfría en una cámara a 200 °C "
        "con h = 80 W/(m²·K). El libro: a los 45 min el centro está a 364 °C y entregó 30 120 kJ "
        "por metro.",
        OneDimInputs(
            "cylinder", 0.10, SOLIDS["stainless"], 80.0, 600.0 + C, 200.0 + C, time_s=45.0 * 60.0
        ),
    ),
    "Bifes que no se tienen que escarchar (Çengel y Ghajar §4-2)": TransientExample(
        "Bifes de 1 in a 75 °F en una heladera a 5 °F: el centro tiene que bajar de 45 °F sin "
        "que la superficie pase de 35 °F. El libro busca el h: unos 4 Btu/(h·ft²·°F). Con ese h, "
        "cuando el centro llega a 45 °F la superficie está a 35 °F.",
        OneDimInputs(
            "wall",
            0.5 * 0.0254,
            Solid("Carne", 0.26 * 1.730735, 74.9 * 16.01846, 0.98 * 4186.8),
            3.98 * 5.678263,
            (75.0 - 32.0) / 1.8 + C,
            (5.0 - 32.0) / 1.8 + C,
            position=1.0,
            T_center_target_K=(45.0 - 32.0) / 1.8 + C,
        ),
    ),
}

SEMI_INFINITE_EXAMPLES: dict[str, TransientExample] = {
    "Caños de agua enterrados (Çengel y Ghajar §4-3)": TransientExample(
        "El suelo (k = 0,4 W/(m·K), α = 0,15·10⁻⁶ m²/s) a 15 °C queda tres meses bajo nieve a "
        "−10 °C. El libro: para que no se congele el agua, los caños van a 0,80 m.",
        SemiInfiniteInputs(
            "temperature",
            _custom("Suelo del ejemplo", 0.4, 0.15e-6, rho=2000.0),
            15.0 + C,
            T_s=-10.0 + C,
            time_s=90.0 * 86400.0,
            T_target_K=0.0 + C,
        ),
    ),
    "Bloque de madera al sol (Çengel y Ghajar §4-3)": TransientExample(
        "Un bloque grueso de madera pintada de negro a 20 °C recibe 1250 W/m² del sol. El libro: "
        "a los 20 min la superficie está a 149 °C.",
        SemiInfiniteInputs(
            "flux",
            _custom("Madera del ejemplo", 0.159, 1.75e-7, rho=545.0),
            20.0 + C,
            q_s=1250.0,
            depth_m=0.0,
            time_s=1200.0,
        ),
    ),
    "Bloque de aluminio al sol (Çengel y Ghajar §4-3)": TransientExample(
        "El mismo sol sobre un bloque de aluminio: el libro da 22 °C, porque el aluminio se "
        "lleva el calor hacia adentro.",
        SemiInfiniteInputs(
            "flux", SOLIDS["aluminum"], 20.0 + C, q_s=1250.0, depth_m=0.0, time_s=1200.0
        ),
    ),
    "Caño de agua en el invierno (Incropera §5.7)": TransientExample(
        "Suelo (k = 0,52 W/(m·K), α = 0,138·10⁻⁶ m²/s) a 20 °C con la superficie a −15 °C "
        "durante 60 días. El libro: la profundidad mínima es 0,68 m.",
        SemiInfiniteInputs(
            "temperature",
            _custom("Suelo del ejemplo", 0.52, 0.138e-6, rho=2050.0),
            20.0 + C,
            T_s=-15.0 + C,
            time_s=60.0 * 86400.0,
            T_target_K=0.0 + C,
        ),
    ),
    "Losa de hormigón con aire caliente (ejemplo propio)": TransientExample(
        "Una losa gruesa de hormigón a 15 °C queda expuesta a aire a 300 °C con h = 25 "
        "W/(m²·K): la temperatura a 2 cm de la cara a la hora.",
        SemiInfiniteInputs(
            "convection",
            SOLIDS["concrete"],
            15.0 + C,
            h=25.0,
            T_inf=300.0 + C,
            depth_m=0.02,
            time_s=3600.0,
        ),
    ),
}

_SKIN = Solid("Piel (tejido vivo)", 0.37, 1100.0, 3400.0)

CONTACT_EXAMPLES: dict[str, TransientExample] = {
    "La mano en aluminio (ejemplo propio)": TransientExample(
        "Una mano a 37 °C toca un bloque de aluminio a 20 °C: el contacto queda casi a la "
        "temperatura del aluminio y por eso se siente frío.",
        ContactInputs(_SKIN, 37.0 + C, SOLIDS["aluminum"], 20.0 + C, time_s=1.0),
    ),
    "La mano en madera (ejemplo propio)": TransientExample(
        "La misma mano sobre madera de roble a 20 °C: el contacto queda cerca de la piel y "
        "se siente tibio.",
        ContactInputs(_SKIN, 37.0 + C, SOLIDS["oak"], 20.0 + C, time_s=1.0),
    ),
}

PRODUCT_EXAMPLES: dict[str, TransientExample] = {
    "Cilindro corto de bronce (Çengel y Ghajar §4-4)": TransientExample(
        "Un cilindro de latón de 10 cm de diámetro y 12 cm de alto a 120 °C se enfría en aire "
        "a 25 °C con h = 60 W/(m²·K): el centro a los 15 min.",
        ProductInputs(
            "short_cylinder",
            product_factors("short_cylinder", (0.10, 0.12), (0.0, 0.0)),
            SOLIDS["brass"],
            60.0,
            120.0 + C,
            25.0 + C,
            15.0 * 60.0,
        ),
    ),
    "Cilindro semi-infinito de aluminio en agua (Çengel y Ghajar §4-4)": TransientExample(
        "Un cilindro largo de aluminio de 20 cm a 200 °C entra a agua a 15 °C con h = 120 "
        "W/(m²·K): el eje a 15 cm de la punta a los 5 min.",
        ProductInputs(
            "semi_cylinder",
            product_factors("semi_cylinder", (0.20,), (0.0, 0.15)),
            SOLIDS["aluminum"],
            120.0,
            200.0 + C,
            15.0 + C,
            300.0,
        ),
    ),
    "Ladrillo que se enfría (ejemplo propio)": TransientExample(
        "Un ladrillo de 20 × 10 × 6 cm a 300 °C se enfría en aire a 25 °C con h = 15 "
        "W/(m²·K): el centro a la hora.",
        ProductInputs(
            "box",
            product_factors("box", (0.20, 0.10, 0.06), (0.0, 0.0, 0.0)),
            SOLIDS["brick"],
            15.0,
            300.0 + C,
            25.0 + C,
            3600.0,
        ),
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _solid_dict(s: Solid, system: UnitSystem) -> dict[str, Any]:
    return {
        "material": s.name,
        "k": format_quantity(s.k, "thermal_conductivity", system),
        "rho": format_quantity(s.rho, "density", system),
        "c": format_quantity(s.c, "specific_heat", system),
        "alpha": format_quantity(s.alpha, "diffusivity", system),
    }


def lumped_to_dict(r: LumpedResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    i = r.inputs

    def T(x: float) -> str:
        return format_quantity(x, "temperature", system)

    data: dict[str, Any] = {
        "cuerpo": BODIES[i.body],
        "medidas_m": list(i.dims),
        **_solid_dict(i.solid, system),
        "h": format_quantity(i.h, "heat_transfer_coefficient", system),
        "T_inicial": T(i.T_i),
        "T_fluido": T(i.T_inf),
        "L_c_m": r.Lc,
        "Bi": r.Bi,
        "b_1_s": r.b,
        "constante_de_tiempo_s": r.time_constant_s,
        "tiempo_s": r.time_s,
        "T": T(r.T_K),
        "Q": format_quantity(r.Q_J, "energy", system),
        "Q_max": format_quantity(r.Q_max_J, "energy", system),
    }
    if r.exact_mean_K is not None:
        data["T_media_exacta"] = T(r.exact_mean_K)
        data["T_centro_exacta"] = T(r.exact_center_K or 0.0)
    return data


def _q_kind(geometry: Geometry1D) -> str:
    return {"wall": "energy_per_area", "cylinder": "energy_per_length", "sphere": "energy"}[
        geometry
    ]


def one_dim_to_dict(r: OneDimResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    i = r.inputs

    def T(x: float) -> str:
        return format_quantity(x, "temperature", system)

    kind = _q_kind(i.geometry)
    return {
        "geometria": GEOMETRIES[i.geometry],
        "L_o_r0_m": i.size,
        **_solid_dict(i.solid, system),
        "h": format_quantity(i.h, "heat_transfer_coefficient", system),
        "T_inicial": T(i.T_i),
        "T_fluido": T(i.T_inf),
        "tiempo_s": r.time_s,
        "tiempo_un_termino_s": r.time_one_term_s,
        "Bi": r.Bi,
        "tau": r.tau,
        "lambda_1": r.lambda1,
        "A_1": r.A1,
        "posicion": i.position,
        "T_centro": T(r.T_center_K),
        "T_centro_un_termino": T(r.T_center_one_K),
        "T_posicion": T(r.T_K),
        "T_superficie": T(r.T_surface_K),
        "Q_sobre_Qmax": r.heat_ratio,
        "Q": format_quantity(r.Q, kind, system),  # type: ignore[arg-type]
        "Q_max": format_quantity(r.Q_max, kind, system),  # type: ignore[arg-type]
    }


def semi_infinite_to_dict(r: SemiInfiniteResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    i = r.inputs

    def T(x: float) -> str:
        return format_quantity(x, "temperature", system)

    data: dict[str, Any] = {
        "condicion": SEMI_BOUNDARIES[i.boundary],
        **_solid_dict(i.solid, system),
        "T_inicial": T(i.T_i),
        "profundidad_m": r.depth_m,
        "tiempo_s": r.time_s,
        "eta": r.eta,
        "T": T(r.T_K),
        "T_superficie": T(r.T_surface_K),
        "flujo_superficie": format_quantity(r.surface_flux, "heat_flux", system),
        "energia_por_area": format_quantity(r.energy_per_area, "energy_per_area", system),
        "penetracion_m": r.penetration_m,
    }
    if i.T_s is not None:
        data["T_s"] = T(i.T_s)
    if i.q_s is not None:
        data["q_s"] = format_quantity(i.q_s, "heat_flux", system)
    if i.h is not None:
        data["h"] = format_quantity(i.h, "heat_transfer_coefficient", system)
        data["T_fluido"] = T(i.T_inf or 0.0)
    return data


def contact_to_dict(r: ContactResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    i = r.inputs
    return {
        "A": {**_solid_dict(i.solid_a, system), "T": format_quantity(i.T_a, "temperature", system)},
        "B": {**_solid_dict(i.solid_b, system), "T": format_quantity(i.T_b, "temperature", system)},
        "T_contacto": format_quantity(r.T_contact_K, "temperature", system),
        "peso_de_A": r.share_a,
        **(
            {"flujo": format_quantity(r.flux_W_m2, "heat_flux", system), "tiempo_s": i.time_s}
            if r.flux_W_m2 is not None
            else {}
        ),
    }


def product_to_dict(r: ProductResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    i = r.inputs
    data: dict[str, Any] = {
        "forma": PRODUCT_SHAPES[i.shape],
        **_solid_dict(i.solid, system),
        "h": format_quantity(i.h, "heat_transfer_coefficient", system),
        "T_inicial": format_quantity(i.T_i, "temperature", system),
        "T_fluido": format_quantity(i.T_inf, "temperature", system),
        "tiempo_s": i.time_s,
        "factores": [
            {
                "factor": f.factor.label or f.factor.kind,
                "tipo": f.factor.kind,
                "medida_m": f.factor.size,
                "posicion": f.factor.position,
                "Bi": f.Bi,
                "tau": f.tau,
                "theta": f.theta,
                "Q_sobre_Qmax": f.heat_ratio,
            }
            for f in r.factors
        ],
        "theta": r.theta,
        "T": format_quantity(r.T_K, "temperature", system),
    }
    if r.heat_ratio is not None:
        data["Q_sobre_Qmax"] = r.heat_ratio
    if r.Q is not None:
        data["Q"] = format_quantity(r.Q, "energy", system)
    return data
