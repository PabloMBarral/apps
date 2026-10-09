"""Aletas: recta, de aguja y anular de espesor constante; arreglos (Fase 8.1).

Incropera et al. (2007), §3.6; Cengel y Ghajar (2015), §3-6.

Con la temperatura uniforme en la sección (Bi pequeño), θ = T − T∞ cumple
d²θ/dx² = m²·θ en una aleta de sección constante, con

    m = √(h·P/(k·A_c)),   M = √(h·P·k·A_c)·θ_b.

Las cuatro condiciones de la punta de la tabla 3.4 de Incropera:

- **convectiva**: q = M·[senh mL + (h/mk)·cosh mL] / [cosh mL + (h/mk)·senh mL];
- **adiabática**: q = M·tanh mL;
- **temperatura dada** θ_L: q = M·(cosh mL − θ_L/θ_b)/senh mL;
- **infinita**: q = M.

Más la **longitud corregida** (Cengel y Ghajar, §3-6): la punta
convectiva se aproxima con una adiabática de L_c = L + t/2 (recta) o
L + D/4 (aguja).

En la aleta **anular** d/dr(r·dθ/dr) = m²·r·θ con m = √(2h/(k·t)); la solución
es θ = C₁·I₀(mr) + C₂·K₀(mr) (Incropera, §3.6.4). Las funciones de Bessel van
escaladas (``scipy.special.i0e``…) para no desbordar con m·r grande.

Eficiencia η = q/(h·A_aleta·θ_b) y efectividad ε = q/(h·A_c·θ_b) (Incropera,
§3.6.3). En un arreglo, la eficiencia global
η_o = 1 − (N·A_f/A_t)·(1 − η) (§3.6.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.special import i0e, i1e, k0e, k1e

from core.units_system import UnitSystem, format_quantity

__all__ = [
    "FIN_EXAMPLES",
    "FIN_SHAPES",
    "TIP_CONDITIONS",
    "FinArrayInputs",
    "FinArrayResult",
    "FinExample",
    "FinInputs",
    "FinResult",
    "FinShape",
    "TipCondition",
    "efficiency_curve",
    "fin_array_notes",
    "fin_notes",
    "fin_profile",
    "fin_to_dict",
    "solve_fin",
    "solve_fin_array",
    "tip_comparison",
]

FinShape = Literal["straight", "pin", "annular"]
TipCondition = Literal["convective", "adiabatic", "temperature", "infinite", "corrected"]

FIN_SHAPES: dict[FinShape, str] = {
    "straight": "Recta (rectangular)",
    "pin": "De aguja (cilíndrica)",
    "annular": "Anular (circular)",
}
TIP_CONDITIONS: dict[TipCondition, str] = {
    "convective": "Convectiva (exacta)",
    "adiabatic": "Adiabática",
    "corrected": "Longitud corregida L_c",
    "temperature": "Temperatura dada",
    "infinite": "Aleta infinita",
}
#: mL a partir del cual la aleta se comporta como infinita: tanh(2,65) = 0,99.
INFINITE_ML = 2.65
#: Biot límite para tratar la aleta como unidimensional (Incropera, §3.6).
BIOT_1D_MAX = 0.1


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


# ---------------------------------------------------------------------
# Datos y resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FinInputs:
    """Una aleta y su base.

    ``length_m`` es L: el largo de la recta o de la aguja, o el largo radial
    r₂ − r₁ de la anular. ``thickness_m`` es t (recta y anular), ``width_m`` el
    ancho w de la recta, ``diameter_m`` el D de la aguja y ``r_base_m`` el r₁
    de la anular (el radio exterior del caño).
    """

    shape: FinShape
    k_W_per_mK: float
    h_W_per_m2K: float
    T_base_K: float
    T_inf_K: float
    length_m: float
    thickness_m: float = 0.0
    width_m: float = 1.0
    diameter_m: float = 0.0
    r_base_m: float = 0.0
    tip: TipCondition = "convective"
    T_tip_K: float | None = None

    @property
    def theta_b(self) -> float:
        return self.T_base_K - self.T_inf_K


@dataclass(frozen=True)
class FinResult:
    """El calor de una aleta, su eficiencia, su efectividad y su punta."""

    inputs: FinInputs
    m_per_m: float
    M_W: float
    Q_W: float
    A_fin_m2: float
    A_base_m2: float
    L_c_m: float
    T_tip_K: float

    @property
    def efficiency(self) -> float | None:
        """η = q/(h·A_aleta·θ_b): contra una aleta ideal toda a T_b (la infinita no tiene)."""
        if self.inputs.tip == "infinite":
            return None
        return self.Q_W / (self.inputs.h_W_per_m2K * self.A_fin_m2 * self.inputs.theta_b)

    @property
    def effectiveness(self) -> float:
        """ε = q/(h·A_c·θ_b): cuánto más transfiere la base con la aleta que sin ella."""
        return self.Q_W / (self.inputs.h_W_per_m2K * self.A_base_m2 * self.inputs.theta_b)

    @property
    def mL(self) -> float:
        return self.m_per_m * self.L_c_m

    @property
    def L_infinite_m(self) -> float:
        """Largo a partir del cual la aleta ya se comporta como infinita (mL = 2,65)."""
        return INFINITE_ML / self.m_per_m

    @property
    def biot(self) -> float:
        """Bi = h·δ/k, con δ = t/2 (recta y anular) o D/4 (aguja)."""
        i = self.inputs
        delta = i.diameter_m / 4.0 if i.shape == "pin" else i.thickness_m / 2.0
        return i.h_W_per_m2K * delta / i.k_W_per_mK

    @property
    def resistance_K_per_W(self) -> float:
        """La resistencia térmica de la aleta, θ_b/q."""
        return self.inputs.theta_b / self.Q_W


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _section(i: FinInputs) -> tuple[float, float]:
    """(A_c, P) de una aleta recta o de aguja."""
    if i.shape == "pin":
        return math.pi * i.diameter_m**2 / 4.0, math.pi * i.diameter_m
    return i.width_m * i.thickness_m, 2.0 * (i.width_m + i.thickness_m)


def _validate(i: FinInputs) -> None:
    if i.shape not in FIN_SHAPES:
        raise ValueError(f"Forma de aleta desconocida: {i.shape!r}.")
    if i.tip not in TIP_CONDITIONS:
        raise ValueError(f"Condición de la punta desconocida: {i.tip!r}.")
    if not i.k_W_per_mK > 0.0:
        raise ValueError("La conductividad k de la aleta tiene que ser positiva.")
    if not i.h_W_per_m2K > 0.0:
        raise ValueError("El coeficiente de convección h tiene que ser positivo.")
    if not (i.T_base_K > 0.0 and i.T_inf_K > 0.0):
        raise ValueError("Las temperaturas tienen que ser absolutas positivas.")
    if i.T_base_K == i.T_inf_K:
        raise ValueError(
            "La base y el fluido están a la misma temperatura: la aleta no transfiere calor."
        )
    if not i.length_m > 0.0:
        raise ValueError("El largo L de la aleta tiene que ser positivo.")
    if i.shape == "pin":
        if not i.diameter_m > 0.0:
            raise ValueError("El diámetro de la aguja tiene que ser positivo.")
    else:
        if not i.thickness_m > 0.0:
            raise ValueError("El espesor t de la aleta tiene que ser positivo.")
    if i.shape == "straight" and not i.width_m > 0.0:
        raise ValueError("El ancho w de la aleta recta tiene que ser positivo.")
    if i.shape == "annular" and not i.r_base_m > 0.0:
        raise ValueError(
            "El radio de la base r₁ (el radio exterior del caño) tiene que ser positivo."
        )
    if i.tip == "temperature":
        if i.T_tip_K is None or not i.T_tip_K > 0.0:
            raise ValueError("Con la punta a temperatura dada, falta esa temperatura.")


def _sinh_ratio(a: float, b: float) -> float:
    """senh(a)/senh(b) sin desbordar (0 ≤ a ≤ b)."""
    if b < 20.0:
        return math.sinh(a) / math.sinh(b)
    return math.exp(a - b) * (1.0 - math.exp(-2.0 * a)) / (1.0 - math.exp(-2.0 * b))


def _uniform_profile(i: FinInputs, m: float, L: float, x: float, tip: TipCondition) -> float:
    """θ(x)/θ_b de una aleta de sección constante (Incropera, tabla 3.4), sin desbordar."""
    if tip == "infinite":
        return math.exp(-m * x)
    if tip == "temperature":
        ratio = (i.T_tip_K - i.T_inf_K) / i.theta_b  # type: ignore[operator]
        return ratio * _sinh_ratio(m * x, m * L) + _sinh_ratio(m * (L - x), m * L)
    beta = i.h_W_per_m2K / (m * i.k_W_per_mK) if tip == "convective" else 0.0
    # [cosh m(L−x) + β·senh m(L−x)] / [cosh mL + β·senh mL], dividido por e^{mL}
    num = (1.0 + beta) * math.exp(-m * x) + (1.0 - beta) * math.exp(-m * (2.0 * L - x))
    den = (1.0 + beta) + (1.0 - beta) * math.exp(-2.0 * m * L)
    return num / den


def _uniform(i: FinInputs) -> FinResult:
    A_c, P = _section(i)
    k, h, theta_b = i.k_W_per_mK, i.h_W_per_m2K, i.theta_b
    m = math.sqrt(h * P / (k * A_c))
    M = math.sqrt(h * P * k * A_c) * theta_b
    L = i.length_m
    tip = i.tip
    if tip == "corrected":
        L = L + (i.diameter_m / 4.0 if i.shape == "pin" else i.thickness_m / 2.0)
    mL = m * L
    if tip == "convective":
        beta = h / (m * k)
        t = math.tanh(mL)
        Q = M * (t + beta) / (1.0 + beta * t)
        A_fin = P * L + A_c
    elif tip in ("adiabatic", "corrected"):
        Q = M * math.tanh(mL)
        A_fin = P * L
    elif tip == "temperature":
        ratio = (i.T_tip_K - i.T_inf_K) / theta_b  # type: ignore[operator]
        coth = 1.0 / math.tanh(mL)
        inv_sinh = 0.0 if mL > 700.0 else 1.0 / math.sinh(mL)
        Q = M * (coth - ratio * inv_sinh)
        A_fin = P * L
    else:  # infinita: el área es la del largo dado (para la eficiencia no se usa)
        Q = M
        A_fin = P * L
    T_tip = i.T_inf_K + theta_b * _uniform_profile(i, m, L, L, tip)
    return FinResult(i, m, M, Q, A_fin, A_c, L, T_tip)


def _annular_coeffs(i: FinInputs, m: float, r1: float, rE: float, tip: TipCondition) -> np.ndarray:
    """(A, B) de θ(r) = A·i0e(mr)·e^{m(r−rE)} + B·k0e(mr)·e^{−m(r−r1)}."""
    theta_b = i.theta_b
    k, h = i.k_W_per_mK, i.h_W_per_m2K
    x1, xE = m * r1, m * rE
    eE = math.exp(-m * (rE - r1))
    row1 = [i0e(x1) * math.exp(m * (r1 - rE)), k0e(x1)]
    if tip in ("adiabatic", "corrected"):
        row2 = [i1e(xE), -k1e(xE) * eE]
        rhs2 = 0.0
    elif tip == "convective":
        row2 = [k * m * i1e(xE) + h * i0e(xE), (-k * m * k1e(xE) + h * k0e(xE)) * eE]
        rhs2 = 0.0
    else:  # temperatura dada en la punta
        row2 = [i0e(xE), k0e(xE) * eE]
        rhs2 = (i.T_tip_K or i.T_inf_K) - i.T_inf_K
    return np.linalg.solve(np.array([row1, row2]), np.array([theta_b, rhs2]))


def _annular_theta(i: FinInputs, m: float, r1: float, rE: float, coeffs: Any, r: float) -> float:
    if i.tip == "infinite":
        return float(coeffs[1] * k0e(m * r) * math.exp(-m * (r - r1)))
    A, B = coeffs
    return float(A * i0e(m * r) * math.exp(m * (r - rE)) + B * k0e(m * r) * math.exp(-m * (r - r1)))


def _annular(i: FinInputs) -> FinResult:
    k, h, t = i.k_W_per_mK, i.h_W_per_m2K, i.thickness_m
    m = math.sqrt(2.0 * h / (k * t))
    r1 = i.r_base_m
    r2 = r1 + i.length_m
    rE = r2 + t / 2.0 if i.tip == "corrected" else r2
    A_c = 2.0 * math.pi * r1 * t
    if i.tip == "infinite":
        B = i.theta_b / k0e(m * r1)
        coeffs: Any = (0.0, B)
        dtheta = -m * B * k1e(m * r1)
    else:
        coeffs = _annular_coeffs(i, m, r1, rE, i.tip)
        A, B = coeffs
        dtheta = m * (A * i1e(m * r1) * math.exp(m * (r1 - rE)) - B * k1e(m * r1))
    Q = -k * A_c * float(dtheta)
    A_fin = 2.0 * math.pi * (rE**2 - r1**2)
    if i.tip == "convective":
        A_fin += 2.0 * math.pi * r2 * t
    # M de una aleta recta con la sección y el perímetro (las dos caras) de la base: k·A_c·m·θ_b
    M = k * A_c * m * i.theta_b
    T_tip = i.T_inf_K + _annular_theta(i, m, r1, rE, coeffs, rE)
    return FinResult(i, m, M, Q, A_fin, A_c, rE - r1, T_tip)


def solve_fin(inputs: FinInputs) -> FinResult:
    """El calor de una aleta (Incropera, tabla 3.4 y §3.6.4).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    _validate(inputs)
    return _annular(inputs) if inputs.shape == "annular" else _uniform(inputs)


def fin_profile(result: FinResult, n: int = 81) -> tuple[list[float], list[float]]:
    """(x en m desde la base, T en K) a lo largo de la aleta (hasta L, o L_c si es corregida)."""
    i = result.inputs
    L = result.L_c_m
    xs = [L * j / (n - 1) for j in range(n)]
    if i.shape == "annular":
        m = result.m_per_m
        r1 = i.r_base_m
        rE = r1 + L
        if i.tip == "infinite":
            coeffs: Any = (0.0, i.theta_b / k0e(m * r1))
        else:
            coeffs = _annular_coeffs(i, m, r1, rE, i.tip)
        Ts = [i.T_inf_K + _annular_theta(i, m, r1, rE, coeffs, r1 + x) for x in xs]
    else:
        tip: TipCondition = "adiabatic" if i.tip == "corrected" else i.tip
        Ts = [i.T_inf_K + i.theta_b * _uniform_profile(i, result.m_per_m, L, x, tip) for x in xs]
    return xs, Ts


def tip_comparison(inputs: FinInputs) -> dict[TipCondition, FinResult]:
    """La misma aleta con cada condición de la punta (menos la temperatura dada)."""
    out: dict[TipCondition, FinResult] = {}
    for tip in ("convective", "adiabatic", "corrected", "infinite"):
        out[tip] = solve_fin(replace(inputs, tip=tip))  # type: ignore[arg-type]
    return out


def efficiency_curve(result: FinResult, n: int = 80) -> tuple[list[float], list[float]]:
    """η contra m·L_c para esta forma (la curva del libro), con L_c variable.

    En la recta y la aguja η = tanh(mL_c)/(mL_c); en la anular depende además
    de r₂c/r₁ y se calcula con la misma relación de radios que la aleta.
    """
    i = result.inputs
    x_max = max(3.0, 1.5 * result.mL)
    xs = [x_max * (j + 1) / n for j in range(n)]
    if i.shape != "annular":
        return xs, [math.tanh(x) / x for x in xs]
    ratio = (i.r_base_m + result.L_c_m) / i.r_base_m
    etas: list[float] = []
    m = result.m_per_m
    for x in xs:
        # m fijo y L_c = x/m; el radio de la base cambia para mantener r₂c/r₁
        Lc = x / m
        r1 = Lc / (ratio - 1.0)
        trial = replace(
            i, r_base_m=r1, length_m=max(Lc - i.thickness_m / 2.0, 1e-9), tip="corrected"
        )
        etas.append(_annular(trial).efficiency or 0.0)
    return xs, etas


# ---------------------------------------------------------------------
# Arreglo de aletas
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class FinArrayInputs:
    """N aletas iguales sobre una superficie de área A_base (antes de poner las aletas)."""

    fin: FinInputs
    n_fins: int
    base_area_m2: float


@dataclass(frozen=True)
class FinArrayResult:
    """El arreglo: η_o, el calor con y sin aletas y la efectividad del conjunto."""

    inputs: FinArrayInputs
    fin: FinResult
    A_unfinned_m2: float
    Q_fins_W: float
    Q_unfinned_W: float
    Q_no_fins_W: float

    @property
    def Q_total_W(self) -> float:
        return self.Q_fins_W + self.Q_unfinned_W

    @property
    def A_total_m2(self) -> float:
        return self.inputs.n_fins * self.fin.A_fin_m2 + self.A_unfinned_m2

    @property
    def overall_efficiency(self) -> float:
        """η_o = Q̇_total/(h·A_t·θ_b) = 1 − (N·A_f/A_t)·(1 − η) (Incropera, §3.6.5)."""
        i = self.inputs.fin
        return self.Q_total_W / (i.h_W_per_m2K * self.A_total_m2 * i.theta_b)

    @property
    def effectiveness(self) -> float:
        """Q̇ con aletas / Q̇ sin aletas."""
        return self.Q_total_W / self.Q_no_fins_W


def solve_fin_array(inputs: FinArrayInputs) -> FinArrayResult:
    """Un arreglo de N aletas (Cengel y Ghajar, §3-6; Incropera, §3.6.5)."""
    fin = solve_fin(inputs.fin)
    if inputs.n_fins < 1:
        raise ValueError("El arreglo necesita al menos una aleta.")
    if not inputs.base_area_m2 > 0.0:
        raise ValueError("El área de la superficie base tiene que ser positiva.")
    occupied = inputs.n_fins * fin.A_base_m2
    if occupied >= inputs.base_area_m2:
        raise ValueError(
            f"Las {inputs.n_fins} aletas ocupan {_num(occupied)} m² de la base, y la superficie "
            f"tiene {_num(inputs.base_area_m2)} m²: no entran (o no dejan nada sin aletas)."
        )
    h, theta_b = inputs.fin.h_W_per_m2K, inputs.fin.theta_b
    A_unf = inputs.base_area_m2 - occupied
    return FinArrayResult(
        inputs,
        fin,
        A_unf,
        inputs.n_fins * fin.Q_W,
        h * A_unf * theta_b,
        h * inputs.base_area_m2 * theta_b,
    )


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def fin_notes(result: FinResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = []
    i = result.inputs
    if result.biot > BIOT_1D_MAX:
        notes.append(
            f"Bi = h·δ/k = {_num(result.biot)}: la aleta es gruesa para su conductividad y la "
            "temperatura cambia también en el espesor; el modelo unidimensional pide Bi < 0,1."
        )
    eps = result.effectiveness
    if eps < 2.0:
        notes.append(
            f"Con ε = {_num(eps)} la aleta casi no ayuda: no se justifica con ε < 2 (Incropera, "
            "§3.6.3). Las aletas sirven donde h es bajo (un gas en convección natural), no con "
            "un líquido o en ebullición."
        )
    if i.tip != "infinite" and result.mL >= INFINITE_ML:
        notes.append(
            f"mL = {_num(result.mL)}: la aleta se comporta como infinita. Más allá de "
            f"L∞ = 2,65/m = {_num(1000.0 * result.L_infinite_m)} mm no suma calor y la punta "
            "está prácticamente a la temperatura del fluido: se puede acortar."
        )
    eta = result.efficiency
    if eta is not None and eta > 0.9:
        notes.append(
            f"η = {_num(eta)}: la aleta está casi toda a la temperatura de la base (es corta o "
            "muy conductora); podría ser más larga o más fina."
        )
    if i.tip == "infinite":
        notes.append(
            f"Aleta infinita: q = M. Una aleta de largo finito transfiere lo mismo (al 1 %) desde "
            f"L∞ = 2,65/m = {_num(1000.0 * result.L_infinite_m)} mm."
        )
    if i.tip == "corrected":
        exact = solve_fin(replace(i, tip="convective"))
        diff = result.Q_W / exact.Q_W - 1.0
        notes.append(
            f"La longitud corregida L_c = {_num(1000.0 * result.L_c_m)} mm reemplaza la punta "
            f"convectiva por una adiabática: da {_num(100.0 * diff, 2)} % respecto de la exacta."
        )
    if i.shape == "annular":
        notes.append(
            "La eficiencia de la aleta anular sale de las funciones de Bessel modificadas I₀, I₁, "
            "K₀ y K₁ (Incropera, tabla 3.5); los libros la leen en un gráfico de η contra "
            "L_c^{3/2}·(h/(k·A_p))^{1/2}."
        )
    return notes


def fin_array_notes(result: FinArrayResult) -> list[str]:
    eff = result.effectiveness
    return [
        f"Con las {result.inputs.n_fins} aletas la superficie transfiere {_num(eff)} veces lo que "
        f"transfería lisa (η_o = {_num(result.overall_efficiency)}). Sin aletas: "
        f"{_num(result.Q_no_fins_W, 4)} W; con aletas: {_num(result.Q_total_W, 4)} W."
    ]


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

C = 273.15


@dataclass(frozen=True)
class FinExample:
    """Una aleta de ejemplo, con el arreglo si lo tiene y una nota."""

    fin: FinInputs
    n_fins: int | None = None
    base_area_m2: float | None = None
    note: str = ""


def _rod(k: float) -> FinInputs:
    return FinInputs(
        "pin", k, 100.0, 100.0 + C, 25.0 + C, length_m=0.25, diameter_m=0.005, tip="infinite"
    )


FIN_EXAMPLES: dict[str, FinExample] = {
    "Varilla muy larga de cobre (Incropera ej. 3.9: D = 5 mm, h = 100, base a 100 °C)": (
        FinExample(_rod(398.0), note="Incropera: 8,3 W y L∞ ≈ 0,19 m.")
    ),
    "Varilla muy larga de aluminio 2024 (Incropera ej. 3.9)": FinExample(
        _rod(180.0), note="Incropera: 5,6 W y L∞ ≈ 0,13 m."
    ),
    "Varilla muy larga de acero inoxidable 316 (Incropera ej. 3.9)": FinExample(
        _rod(14.0), note="Incropera: 1,6 W y L∞ ≈ 0,04 m: el acero conduce poco."
    ),
    "Aguja de aluminio de 3 cm con punta convectiva (D = 5 mm, h = 100)": FinExample(
        FinInputs(
            "pin",
            180.0,
            100.0,
            100.0 + C,
            25.0 + C,
            length_m=0.03,
            diameter_m=0.005,
            tip="convective",
        ),
        note="La exacta coincide con la solución numérica de la ecuación de la aleta.",
    ),
    "Aleta recta de un disipador (aluminio, 1,5 mm × 50 mm, 3 cm de largo)": FinExample(
        FinInputs(
            "straight",
            200.0,
            25.0,
            80.0 + C,
            25.0 + C,
            length_m=0.03,
            thickness_m=0.0015,
            width_m=0.05,
            tip="corrected",
        ),
        n_fins=12,
        base_area_m2=0.05 * 0.06,
        note="12 aletas sobre una base de 5 × 6 cm, en aire quieto (h ≈ 25 W/(m²·K)).",
    ),
    "Caño de vapor con aletas anulares (basado en Cengel y Ghajar: 250 aletas por metro)": (
        FinExample(
            FinInputs(
                "annular",
                186.0,
                40.0,
                180.0 + C,
                25.0 + C,
                length_m=0.005,
                thickness_m=0.001,
                r_base_m=0.025,
                tip="corrected",
            ),
            n_fins=250,
            base_area_m2=math.pi * 0.05 * 1.0,
            note=(
                "Caño de 5 cm, aletas de aluminio 2024-T6 de 6 cm de diámetro y 1 mm de "
                "espesor, cada 3 mm. El libro lee η del gráfico; acá sale exacta de Bessel."
            ),
        )
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def fin_to_dict(
    result: FinResult, system: UnitSystem, array: FinArrayResult | None = None
) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    i = result.inputs

    def L(x: float) -> str:
        return format_quantity(x, "small_length", system)

    data: dict[str, Any] = {
        "forma": FIN_SHAPES[i.shape],
        "punta": TIP_CONDITIONS[i.tip],
        "k": format_quantity(i.k_W_per_mK, "thermal_conductivity", system),
        "h": format_quantity(i.h_W_per_m2K, "heat_transfer_coefficient", system),
        "T_base": format_quantity(i.T_base_K, "temperature", system),
        "T_fluido": format_quantity(i.T_inf_K, "temperature", system),
        "L": L(i.length_m),
        "L_c": L(result.L_c_m),
        "m_1_por_m": result.m_per_m,
        "mL": result.mL,
        "Q": format_quantity(result.Q_W, "heat_rate", system),
        "eficiencia": result.efficiency,
        "efectividad": result.effectiveness,
        "T_punta": format_quantity(result.T_tip_K, "temperature", system),
        "Biot": result.biot,
        "A_aleta": format_quantity(result.A_fin_m2, "area", system),
    }
    if array is not None:
        data["arreglo"] = {
            "N": array.inputs.n_fins,
            "A_base": format_quantity(array.inputs.base_area_m2, "area", system),
            "Q_total": format_quantity(array.Q_total_W, "heat_rate", system),
            "Q_sin_aletas": format_quantity(array.Q_no_fins_W, "heat_rate", system),
            "eficiencia_global": array.overall_efficiency,
            "efectividad": array.effectiveness,
        }
    return data
