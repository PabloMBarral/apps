"""Radiación térmica: cuerpo negro, factores de forma y recintos (Fase 8.2).

Cengel y Ghajar (2015), cap. 12 (§12-3 cuerpo negro, §12-5 propiedades) y
cap. 13 (§13-1 y §13-2 factor de forma, §13-3 superficies negras, §13-4 grises,
§13-5 pantallas y medición de temperatura); Incropera et al. (2007), §12.4 a
§12.8 y §13.1 a §13.4.

- Cuerpo negro: Planck, E_bλ = C₁/{λ⁵·[exp(C₂/λT) − 1]}; Wien, λ_máx·T = C₃;
  Stefan–Boltzmann, E_b = σ·T⁴; la fracción emitida por debajo de λ es f(λT)
  (Chang y Rhee, 1984). Constantes de CODATA 2018 (Tiesinga et al., 2021).
- Superficies reales: ε(λ) escalonada en bandas da ε(T) = Σ εᵢ·Δfᵢ(T) y la
  absortividad para la radiación de un cuerpo negro a otra temperatura.
- Factor de forma: fórmulas cerradas (Incropera, tablas 13.1 y 13.2), con
  reciprocidad A_i·F_ij = A_j·F_ji y la regla de la suma.
- Intercambio entre superficies grises y difusas: la red de resistencias
  (1 − ε)/(A·ε) de cada superficie y 1/(A_i·F_ij) del espacio; con pantallas,
  en serie; en un recinto de N superficies, el método de las radiosidades.
- Radiación y convección juntas: h_rad = ε·σ·(T_s + T_alr)·(T_s² + T_alr²), el
  balance de una superficie (con el sol) y el error de una termocupla.

Todo en SI: λ en m, E_bλ en W/m³ (por metro de longitud de onda).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq

from core.units_system import UnitSystem, format_quantity

__all__ = [
    "BLACKBODY_EXAMPLES",
    "C1",
    "C2",
    "C3",
    "ENCLOSURE_EXAMPLES",
    "SIGMA",
    "SURFACE_EXAMPLES",
    "THERMOCOUPLE_EXAMPLES",
    "TWO_SURFACE_EXAMPLES",
    "TWO_SURFACE_GEOMETRIES",
    "VIEW_FACTORS",
    "VIEW_FACTOR_EXAMPLES",
    "BlackbodyInputs",
    "BlackbodyResult",
    "EnclosureInputs",
    "EnclosureResult",
    "EnclosureSurface",
    "NetworkResistance",
    "RadiationExample",
    "Shield",
    "SpectralBand",
    "SurfaceBalanceInputs",
    "SurfaceBalanceResult",
    "ThermocoupleInputs",
    "ThermocoupleResult",
    "TwoSurfaceGeometry",
    "TwoSurfaceInputs",
    "TwoSurfaceResult",
    "ViewFactor",
    "ViewFactorInputs",
    "ViewFactorResult",
    "ViewGeometry",
    "blackbody_fraction",
    "blackbody_notes",
    "blackbody_to_dict",
    "cylindrical_furnace",
    "enclosure_notes",
    "enclosure_to_dict",
    "fraction_inverse",
    "open_cylindrical_cavity",
    "parallel_plates_with_surroundings",
    "planck",
    "solve_blackbody",
    "solve_enclosure",
    "solve_surface_balance",
    "solve_thermocouple",
    "solve_two_surface",
    "solve_view_factor",
    "surface_balance_notes",
    "surface_to_dict",
    "thermocouple_notes",
    "thermocouple_to_dict",
    "triangular_duct",
    "two_surface_notes",
    "two_surface_to_dict",
    "view_factor_curve",
    "view_factor_notes",
    "view_factor_to_dict",
]

#: Constante de Stefan–Boltzmann [W/(m²·K⁴)] (CODATA 2018).
SIGMA = 5.670374419e-8
#: Primera constante de radiación C₁ = 2π·h·c² [W·m²] (CODATA 2018).
C1 = 3.741771852e-16
#: Segunda constante de radiación C₂ = h·c/k [m·K] (CODATA 2018).
C2 = 1.438776877e-2
#: Constante de Wien C₃ = λ_máx·T [m·K] (CODATA 2018).
C3 = 2.897771955e-3

_K0 = 273.15


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


def _degC(T_K: float) -> str:
    return f"{T_K - _K0:.1f} °C".replace(".", ",")


def _um(x_m: float) -> str:
    return f"{_num(x_m * 1e6, 4)} μm"


# ---------------------------------------------------------------------
# Cuerpo negro (Cengel y Ghajar, §12-3; Incropera, §12.4)
# ---------------------------------------------------------------------


def planck(lambda_m: float, T_K: float) -> float:
    """Poder emisivo espectral del cuerpo negro E_bλ [W/m³] (Planck, 1901).

    E_bλ = C₁/{λ⁵·[exp(C₂/(λT)) − 1]} (Cengel y Ghajar, §12-3).
    """
    x = C2 / (lambda_m * T_K)
    if x > 700.0:
        return 0.0
    return C1 / (lambda_m**5 * math.expm1(x))


def blackbody_fraction(lambda_T_mK: float) -> float:
    """f(λT): fracción de la emisión del cuerpo negro por debajo de λ.

    Con ζ = C₂/(λT): para ζ ≥ 1 la serie exponencial
    f = (15/π⁴)·Σₙ e^(−nζ)/n·(ζ³ + 3ζ²/n + 6ζ/n² + 6/n³), y para ζ < 1 la serie de
    potencias f = 1 − (15/π⁴)·ζ³·(1/3 − ζ/8 + ζ²/60 − ζ⁴/5040 + ζ⁶/272160 −
    ζ⁸/13305600) (Chang y Rhee, 1984; Siegel y Howell). Es la tabla 12-2 de
    Cengel y Ghajar y la 12.1 de Incropera.
    """
    if lambda_T_mK <= 0.0:
        return 0.0
    z = C2 / lambda_T_mK
    if z >= 1.0:
        total = 0.0
        for n in range(1, 400):
            term = math.exp(-n * z) / n * (z**3 + 3.0 * z**2 / n + 6.0 * z / n**2 + 6.0 / n**3)
            total += term
            if term < 1e-18 * total:
                break
        return 15.0 / math.pi**4 * total
    series = 1.0 / 3.0 - z / 8.0 + z**2 / 60.0 - z**4 / 5040.0 + z**6 / 272160.0 - z**8 / 13305600.0
    return 1.0 - 15.0 / math.pi**4 * z**3 * series


def fraction_inverse(fraction: float) -> float:
    """El λT [m·K] por debajo del cual el cuerpo negro emite ``fraction``."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("La fracción tiene que estar entre 0 y 1 (sin incluirlos).")
    return float(brentq(lambda x: blackbody_fraction(x) - fraction, 1e-5, 10.0, xtol=1e-14))


@dataclass(frozen=True)
class SpectralBand:
    """Una banda de una ε(λ) escalonada: ε hasta ``upper_m`` (None: hasta ∞)."""

    emissivity: float
    upper_m: float | None = None


@dataclass(frozen=True)
class BlackbodyInputs:
    """Un cuerpo negro a T y, opcionalmente, una superficie real con ε(λ) en bandas.

    ``lambda_point_m``: una λ para E_bλ; ``band_*_m``: la banda de la que se pide la
    fracción (por ejemplo, el visible); ``fraction_target``: el λ por debajo del
    cual se emite esa fracción;
    ``bands``: ε(λ) escalonada (vacía: cuerpo negro); ``T_source_K``: la
    temperatura de un cuerpo negro que irradia a la superficie (el sol, 5800 K),
    para su absortividad.
    """

    T_K: float
    band_lower_m: float | None = None
    band_upper_m: float | None = None
    fraction_target: float | None = None
    bands: tuple[SpectralBand, ...] = ()
    T_source_K: float | None = None
    lambda_point_m: float | None = None


@dataclass(frozen=True)
class BandRow:
    """Una banda de ε(λ) a una temperatura: [λ_inf, λ_sup), su fracción y su aporte."""

    lower_m: float
    upper_m: float | None
    emissivity: float
    f_lower: float
    f_upper: float

    @property
    def fraction(self) -> float:
        return self.f_upper - self.f_lower

    @property
    def contribution(self) -> float:
        return self.emissivity * self.fraction


@dataclass(frozen=True)
class BlackbodyResult:
    inputs: BlackbodyInputs
    E_b_W_per_m2: float
    lambda_max_m: float
    E_blambda_max_W_per_m3: float
    band_fraction: float | None
    lambda_target_m: float | None
    rows: tuple[BandRow, ...]
    source_rows: tuple[BandRow, ...]

    @property
    def emissivity(self) -> float | None:
        if not self.rows:
            return None
        return sum(r.contribution for r in self.rows)

    @property
    def absorptivity(self) -> float | None:
        if not self.source_rows:
            return None
        return sum(r.contribution for r in self.source_rows)

    @property
    def E_W_per_m2(self) -> float:
        eps = self.emissivity
        return self.E_b_W_per_m2 * (1.0 if eps is None else eps)

    @property
    def E_blambda_point_W_per_m3(self) -> float | None:
        lam = self.inputs.lambda_point_m
        return None if lam is None else planck(lam, self.inputs.T_K)

    @property
    def band_power_W_per_m2(self) -> float | None:
        if self.band_fraction is None:
            return None
        return self.band_fraction * self.E_b_W_per_m2


def _band_rows(bands: tuple[SpectralBand, ...], T_K: float) -> tuple[BandRow, ...]:
    rows: list[BandRow] = []
    lower = 0.0
    f_lower = 0.0
    for band in bands:
        f_upper = 1.0 if band.upper_m is None else blackbody_fraction(band.upper_m * T_K)
        rows.append(BandRow(lower, band.upper_m, band.emissivity, f_lower, f_upper))
        if band.upper_m is None:
            break
        lower, f_lower = band.upper_m, f_upper
    return tuple(rows)


def _validate_bands(bands: tuple[SpectralBand, ...]) -> None:
    last = 0.0
    for k, band in enumerate(bands):
        if not 0.0 <= band.emissivity <= 1.0:
            raise ValueError(f"La ε de la banda {k + 1} tiene que estar entre 0 y 1.")
        if band.upper_m is None:
            if k != len(bands) - 1:
                raise ValueError("Solo la última banda puede llegar hasta λ → ∞.")
            continue
        if band.upper_m <= last:
            raise ValueError(
                f"Los cortes de las bandas tienen que crecer: la banda {k + 1} termina en "
                f"{_um(band.upper_m)} y la anterior en {_um(last)}."
            )
        last = band.upper_m
    if bands and bands[-1].upper_m is not None:
        raise ValueError("La última banda tiene que llegar hasta λ → ∞.")


def solve_blackbody(inputs: BlackbodyInputs) -> BlackbodyResult:
    """Cuerpo negro a T y la superficie real de ε(λ) escalonada.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    i = inputs
    if i.T_K <= 0.0:
        raise ValueError("La temperatura tiene que ser absoluta positiva.")
    band_fraction = None
    if i.band_lower_m is not None or i.band_upper_m is not None:
        lo = i.band_lower_m or 0.0
        hi = i.band_upper_m
        if lo < 0.0 or (hi is not None and hi <= lo):
            raise ValueError("La banda tiene que ir de una λ menor a una mayor.")
        f_hi = 1.0 if hi is None else blackbody_fraction(hi * i.T_K)
        band_fraction = f_hi - blackbody_fraction(lo * i.T_K)
    target = None
    if i.fraction_target is not None:
        target = fraction_inverse(i.fraction_target) / i.T_K
    _validate_bands(i.bands)
    if i.lambda_point_m is not None and i.lambda_point_m <= 0.0:
        raise ValueError("La longitud de onda tiene que ser positiva.")
    if i.T_source_K is not None and i.T_source_K <= 0.0:
        raise ValueError("La temperatura de la fuente tiene que ser absoluta positiva.")
    rows = _band_rows(i.bands, i.T_K)
    source = _band_rows(i.bands, i.T_source_K) if i.T_source_K else ()
    lam_max = C3 / i.T_K
    return BlackbodyResult(
        i, SIGMA * i.T_K**4, lam_max, planck(lam_max, i.T_K), band_fraction, target, rows, source
    )


def blackbody_notes(result: BlackbodyResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = []
    T = result.inputs.T_K
    lam = result.lambda_max_m
    if lam < 0.40e-6:
        where = "en el ultravioleta"
    elif lam <= 0.76e-6:
        where = "en el visible (0,40 a 0,76 μm)"
    else:
        where = "en el infrarrojo"
    notes.append(
        f"A {_num(T, 4)} K el máximo de E_bλ está en λ = {_um(lam)}, {where}: cuanto más "
        "caliente el cuerpo, más corta la longitud de onda (Wien)."
    )
    visible = blackbody_fraction(0.76e-6 * T) - blackbody_fraction(0.40e-6 * T)
    notes.append(
        f"En el visible (0,40 a 0,76 μm) cae el {_pct(visible)} de la emisión"
        + (": por eso una lámpara incandescente es más estufa que lámpara." if T < 3500 else ".")
    )
    eps, alpha = result.emissivity, result.absorptivity
    if eps is not None and alpha is not None and result.inputs.T_source_K:
        if abs(alpha - eps) > 0.1:
            kind = "absorbedor selectivo" if alpha > eps else "reflector selectivo"
            notes.append(
                f"α = {_num(alpha, 3)} para la radiación de un cuerpo a "
                f"{_num(result.inputs.T_source_K, 4)} K y ε = {_num(eps, 3)} a su temperatura: "
                f"Kirchhoff (α = ε) vale para la misma distribución espectral, no para fuentes "
                f"a otra temperatura. Es un {kind}."
            )
    return notes


# ---------------------------------------------------------------------
# Factores de forma (Incropera, tablas 13.1 y 13.2; Cengel y Ghajar, §13-1)
# ---------------------------------------------------------------------

ViewGeometry = Literal[
    "parallel_rectangles",
    "coaxial_disks",
    "perpendicular_rectangles",
    "parallel_plates_2d",
    "perpendicular_plates_2d",
    "three_sided_2d",
    "parallel_cylinders_2d",
    "tube_row_2d",
]


@dataclass(frozen=True)
class ViewFactor:
    """Una geometría de factor de forma: nombre, dimensiones y fórmula."""

    name: str
    dims: tuple[str, ...]
    labels: tuple[str, ...]
    latex: str
    source: str
    two_d: bool = False


def _f_parallel_rectangles(X: float, Y: float, L: float) -> float:
    x, y = X / L, Y / L
    t1 = 0.5 * math.log((1 + x * x) * (1 + y * y) / (1 + x * x + y * y))
    t2 = x * math.sqrt(1 + y * y) * math.atan(x / math.sqrt(1 + y * y))
    t3 = y * math.sqrt(1 + x * x) * math.atan(y / math.sqrt(1 + x * x))
    return 2.0 / (math.pi * x * y) * (t1 + t2 + t3 - x * math.atan(x) - y * math.atan(y))


def _f_coaxial_disks(ri: float, rj: float, L: float) -> float:
    Ri, Rj = ri / L, rj / L
    S = 1.0 + (1.0 + Rj * Rj) / (Ri * Ri)
    return 0.5 * (S - math.sqrt(S * S - 4.0 * (rj / ri) ** 2))


def _f_perpendicular_rectangles(X: float, Y: float, Z: float) -> float:
    H, W = Z / X, Y / X
    a = (1 + W * W) * (1 + H * H) / (1 + W * W + H * H)
    b = W * W * (1 + W * W + H * H) / ((1 + W * W) * (W * W + H * H))
    c = H * H * (1 + H * H + W * W) / ((1 + H * H) * (H * H + W * W))
    s = math.sqrt(H * H + W * W)
    log_term = math.log(a) + W * W * math.log(b) + H * H * math.log(c)
    return (
        W * math.atan(1 / W) + H * math.atan(1 / H) - s * math.atan(1 / s) + 0.25 * log_term
    ) / (math.pi * W)


def _f_parallel_plates_2d(wi: float, wj: float, L: float) -> float:
    Wi, Wj = wi / L, wj / L
    return (math.sqrt((Wi + Wj) ** 2 + 4.0) - math.sqrt((Wj - Wi) ** 2 + 4.0)) / (2.0 * Wi)


def _f_perpendicular_plates_2d(wi: float, wj: float) -> float:
    r = wj / wi
    return 0.5 * (1.0 + r - math.sqrt(1.0 + r * r))


def _f_three_sided_2d(wi: float, wj: float, wk: float) -> float:
    return (wi + wj - wk) / (2.0 * wi)


def _f_parallel_cylinders_2d(r: float, s: float) -> float:
    X = 1.0 + s / (2.0 * r)
    return (math.sqrt(X * X - 1.0) + math.asin(1.0 / X) - X) / math.pi


def _f_tube_row_2d(D: float, s: float) -> float:
    d = D / s
    return 1.0 - math.sqrt(1.0 - d * d) + d * math.atan(math.sqrt((s * s - D * D) / (D * D)))


VIEW_FACTORS: dict[ViewGeometry, ViewFactor] = {
    "parallel_rectangles": ViewFactor(
        "Rectángulos paralelos alineados",
        ("X", "Y", "L"),
        ("Ancho X", "Largo Y", "Separación L"),
        r"F_{ij} = \frac{2}{\pi \bar{X} \bar{Y}} \left\{ \ln\left[\frac{(1+\bar{X}^2)(1+\bar{Y}^2)}"
        r"{1+\bar{X}^2+\bar{Y}^2}\right]^{1/2} + \cdots \right\}",
        "Incropera, tabla 13.2",
    ),
    "coaxial_disks": ViewFactor(
        "Discos coaxiales paralelos",
        ("r_i", "r_j", "L"),
        ("Radio r_i", "Radio r_j", "Separación L"),
        r"F_{ij} = \frac{1}{2}\left\{S - \left[S^2 - 4\left(\frac{r_j}{r_i}\right)^2\right]^{1/2}"
        r"\right\}",
        "Incropera, tabla 13.2",
    ),
    "perpendicular_rectangles": ViewFactor(
        "Rectángulos perpendiculares con un borde común",
        ("X", "Y", "Z"),
        ("Borde común X", "Ancho Y (de i)", "Alto Z (de j)"),
        r"F_{ij} = \frac{1}{\pi W}\left(W\tan^{-1}\frac{1}{W} + H\tan^{-1}\frac{1}{H} - \cdots"
        r"\right)",
        "Incropera, tabla 13.2",
    ),
    "parallel_plates_2d": ViewFactor(
        "Placas paralelas largas (centradas)",
        ("w_i", "w_j", "L"),
        ("Ancho w_i", "Ancho w_j", "Separación L"),
        r"F_{ij} = \frac{\left[(W_i+W_j)^2+4\right]^{1/2} - \left[(W_j-W_i)^2+4\right]^{1/2}}"
        r"{2W_i}",
        "Incropera, tabla 13.1",
        True,
    ),
    "perpendicular_plates_2d": ViewFactor(
        "Placas largas perpendiculares con un borde común",
        ("w_i", "w_j"),
        ("Ancho w_i", "Ancho w_j"),
        r"F_{ij} = \frac{1 + \frac{w_j}{w_i} - \left[1 + \left(\frac{w_j}{w_i}\right)^2"
        r"\right]^{1/2}}{2}",
        "Incropera, tabla 13.1",
        True,
    ),
    "three_sided_2d": ViewFactor(
        "Recinto largo de tres lados",
        ("w_i", "w_j", "w_k"),
        ("Lado w_i", "Lado w_j", "Lado w_k"),
        r"F_{ij} = \frac{w_i + w_j - w_k}{2 w_i}",
        "Incropera, tabla 13.1 (cuerdas cruzadas de Hottel)",
        True,
    ),
    "parallel_cylinders_2d": ViewFactor(
        "Dos cilindros paralelos iguales",
        ("r", "s"),
        ("Radio r", "Separación s (entre superficies)"),
        r"F_{ij} = \frac{1}{\pi}\left[\left(X^2-1\right)^{1/2} + \sin^{-1}\frac{1}{X} - X\right]",
        "Cengel y Ghajar, tabla 13-1",
        True,
    ),
    "tube_row_2d": ViewFactor(
        "Un plano y una hilera de tubos",
        ("D", "s"),
        ("Diámetro D", "Paso s"),
        r"F_{ij} = 1 - \left[1-\left(\frac{D}{s}\right)^2\right]^{1/2} + \frac{D}{s}"
        r"\tan^{-1}\left(\frac{s^2-D^2}{D^2}\right)^{1/2}",
        "Incropera, tabla 13.1",
        True,
    ),
}


@dataclass(frozen=True)
class ViewFactorInputs:
    """Una geometría y sus dimensiones (en m, en el orden de ``VIEW_FACTORS[g].dims``)."""

    geometry: ViewGeometry
    a: float
    b: float
    c: float = 0.0


@dataclass(frozen=True)
class ViewFactorResult:
    inputs: ViewFactorInputs
    F_ij: float
    A_i: float
    A_j: float

    @property
    def F_ji(self) -> float:
        """Reciprocidad: A_i·F_ij = A_j·F_ji."""
        return self.A_i * self.F_ij / self.A_j

    @property
    def F_i_rest(self) -> float:
        """Regla de la suma: lo que sale de i y no llega a j (F_ii = 0 si i es plana o convexa)."""
        return 1.0 - self.F_ij

    @property
    def ratios(self) -> dict[str, float]:
        """Los parámetros adimensionales de la fórmula."""
        g, a, b, c = self.inputs.geometry, self.inputs.a, self.inputs.b, self.inputs.c
        if g == "parallel_rectangles":
            return {"X̄": a / c, "Ȳ": b / c}
        if g == "coaxial_disks":
            Ri, Rj = a / c, b / c
            return {"R_i": Ri, "R_j": Rj, "S": 1.0 + (1.0 + Rj * Rj) / (Ri * Ri)}
        if g == "perpendicular_rectangles":
            return {"H": c / a, "W": b / a}
        if g == "parallel_plates_2d":
            return {"W_i": a / c, "W_j": b / c}
        if g == "parallel_cylinders_2d":
            return {"X": 1.0 + b / (2.0 * a)}
        if g == "tube_row_2d":
            return {"D/s": a / b}
        return {}


def _view_factor_areas(i: ViewFactorInputs) -> tuple[float, float]:
    g, a, b, c = i.geometry, i.a, i.b, i.c
    if g == "parallel_rectangles":
        return a * b, a * b
    if g == "coaxial_disks":
        return math.pi * a * a, math.pi * b * b
    if g == "perpendicular_rectangles":
        return a * b, a * c
    if g == "parallel_plates_2d":
        return a, b
    if g == "perpendicular_plates_2d":
        return a, b
    if g == "three_sided_2d":
        return a, b
    if g == "parallel_cylinders_2d":
        return 2.0 * math.pi * a, 2.0 * math.pi * a
    # tube_row_2d: un paso del plano y el perímetro de un tubo, por metro de largo
    return b, math.pi * a


def solve_view_factor(inputs: ViewFactorInputs) -> ViewFactorResult:
    """F_ij de una geometría (fórmulas cerradas de Incropera, tablas 13.1 y 13.2).

    En las bidimensionales (superficies muy largas) las áreas son por metro de largo.

    Raises
    ------
    ValueError
        Si una dimensión no tiene sentido (con un mensaje para el alumno).
    """
    i = inputs
    if i.geometry not in VIEW_FACTORS:
        raise ValueError(f"Geometría desconocida: {i.geometry!r}.")
    vf = VIEW_FACTORS[i.geometry]
    values = (i.a, i.b, i.c)[: len(vf.dims)]
    for label, value in zip(vf.labels, values, strict=True):
        if value <= 0.0 and not (i.geometry == "parallel_cylinders_2d" and label.startswith("Sep")):
            raise ValueError(f"{label} tiene que ser positivo.")
    g = i.geometry
    if g == "parallel_rectangles":
        F = _f_parallel_rectangles(i.a, i.b, i.c)
    elif g == "coaxial_disks":
        F = _f_coaxial_disks(i.a, i.b, i.c)
    elif g == "perpendicular_rectangles":
        F = _f_perpendicular_rectangles(i.a, i.b, i.c)
    elif g == "parallel_plates_2d":
        F = _f_parallel_plates_2d(i.a, i.b, i.c)
    elif g == "perpendicular_plates_2d":
        F = _f_perpendicular_plates_2d(i.a, i.b)
    elif g == "three_sided_2d":
        sides = sorted((i.a, i.b, i.c))
        if sides[2] >= sides[0] + sides[1]:
            raise ValueError(
                "Con esos tres lados no se cierra un triángulo: cada lado tiene que ser menor "
                "que la suma de los otros dos."
            )
        F = _f_three_sided_2d(i.a, i.b, i.c)
    elif g == "parallel_cylinders_2d":
        if i.b < 0.0:
            raise ValueError("La separación entre los cilindros no puede ser negativa.")
        F = _f_parallel_cylinders_2d(i.a, i.b)
    else:
        if i.a > i.b:
            raise ValueError(
                "El diámetro de los tubos no puede ser mayor que el paso de la hilera."
            )
        F = _f_tube_row_2d(i.a, i.b)
    A_i, A_j = _view_factor_areas(i)
    return ViewFactorResult(i, F, A_i, A_j)


def view_factor_curve(
    inputs: ViewFactorInputs, n: int = 60
) -> tuple[str, list[float], list[float]]:
    """F_ij contra la separación (o la dimensión que la representa), con lo demás fijo."""
    g = inputs.geometry
    field, label = {
        "parallel_rectangles": ("c", "L"),
        "coaxial_disks": ("c", "L"),
        "perpendicular_rectangles": ("c", "Z"),
        "parallel_plates_2d": ("c", "L"),
        "perpendicular_plates_2d": ("b", "w_j"),
        "three_sided_2d": ("c", "w_k"),
        "parallel_cylinders_2d": ("b", "s"),
        "tube_row_2d": ("b", "s"),
    }[g]
    x0 = getattr(inputs, field)
    if g == "three_sided_2d":
        lo, hi = abs(inputs.a - inputs.b) * 1.001 + 1e-9, (inputs.a + inputs.b) * 0.999
    elif g == "tube_row_2d":
        lo, hi = inputs.a * 1.0001, max(10.0 * inputs.a, 2.0 * x0)
    elif g == "parallel_cylinders_2d":
        lo, hi = 0.0, max(10.0 * inputs.a, 2.0 * x0)
    else:
        lo, hi = x0 / 20.0, x0 * 20.0
    xs: list[float] = []
    if g in ("three_sided_2d", "parallel_cylinders_2d", "tube_row_2d"):
        xs = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    else:
        xs = [lo * (hi / lo) ** (k / (n - 1)) for k in range(n)]
    xs = sorted({*xs, x0})
    Fs = [solve_view_factor(replace(inputs, **{field: x})).F_ij for x in xs]
    return label, xs, Fs


def view_factor_notes(result: ViewFactorResult) -> list[str]:
    """Interpretación física (markdown)."""
    vf = VIEW_FACTORS[result.inputs.geometry]
    unit = "por metro de largo" if vf.two_d else ""
    notes = [
        f"De toda la radiación que sale de i, el {_pct(result.F_ij)} llega a j; el resto "
        f"({_pct(result.F_i_rest)}) va a otras superficies (regla de la suma).",
        "Reciprocidad: A_i·F_ij = A_j·F_ji, así que F_ji = "
        f"{_num(result.F_ji, 4)} {('(áreas ' + unit + ')') if unit else ''}".rstrip(),
    ]
    if result.F_ij < 0.05:
        notes.append(
            "Casi no se ven: el factor de forma depende solo de la geometría (no de las "
            "temperaturas ni de las propiedades de las superficies)."
        )
    return notes


# ---------------------------------------------------------------------
# Dos superficies y pantallas (Cengel y Ghajar, §13-4 y §13-5)
# ---------------------------------------------------------------------

TwoSurfaceGeometry = Literal[
    "parallel_plates", "concentric_cylinders", "concentric_spheres", "small_object", "general"
]

TWO_SURFACE_GEOMETRIES: dict[TwoSurfaceGeometry, str] = {
    "parallel_plates": "Placas paralelas infinitas",
    "concentric_cylinders": "Cilindros concéntricos largos",
    "concentric_spheres": "Esferas concéntricas",
    "small_object": "Un objeto chico en un recinto grande",
    "general": "Dos superficies cualesquiera (A₁, A₂ y F₁₂)",
}


@dataclass(frozen=True)
class Shield:
    """Una pantalla de radiación: la ε de cada cara y su radio (cilindros y esferas)."""

    eps_1: float
    eps_2: float
    r_m: float = 0.0


@dataclass(frozen=True)
class TwoSurfaceInputs:
    """Dos superficies grises y difusas que forman un recinto (con pantallas opcionales).

    Placas: ``A1_m2`` es el área (1 m² da el flujo q''). Cilindros: radios y largo
    ``L_m``. Esferas: radios. Objeto chico: su área ``A1_m2``. General: ``A1_m2``,
    ``A2_m2`` y ``F12``. Lado 1 es el interior en cilindros y esferas.
    """

    geometry: TwoSurfaceGeometry
    T1_K: float
    T2_K: float
    eps1: float
    eps2: float
    A1_m2: float = 1.0
    A2_m2: float = 0.0
    F12: float = 1.0
    r1_m: float = 0.0
    r2_m: float = 0.0
    L_m: float = 1.0
    shields: tuple[Shield, ...] = ()


@dataclass(frozen=True)
class NetworkResistance:
    """Una resistencia de la red de radiación: de superficie o de espacio."""

    label: str
    R_per_m2: float
    kind: Literal["superficie", "espacio"]
    area_m2: float


@dataclass(frozen=True)
class TwoSurfaceResult:
    inputs: TwoSurfaceInputs
    resistances: tuple[NetworkResistance, ...]
    Q_W: float
    Q_no_shields_W: float
    shield_T_K: tuple[float, ...]
    radiosities: tuple[float, ...]

    @property
    def R_total(self) -> float:
        return sum(r.R_per_m2 for r in self.resistances)

    def share(self, r: NetworkResistance) -> float:
        return r.R_per_m2 / self.R_total

    @property
    def reduction(self) -> float:
        """Q̇ con pantallas / Q̇ sin pantallas."""
        return self.Q_W / self.Q_no_shields_W if self.Q_no_shields_W else 1.0


def _two_surface_areas(i: TwoSurfaceInputs) -> tuple[float, float, list[float]]:
    """(A₁, A₂, áreas de las pantallas)."""
    g = i.geometry
    if g == "parallel_plates":
        return i.A1_m2, i.A1_m2, [i.A1_m2] * len(i.shields)
    if g == "concentric_cylinders":

        def area(r: float) -> float:
            return 2.0 * math.pi * r * i.L_m

    elif g == "concentric_spheres":

        def area(r: float) -> float:
            return 4.0 * math.pi * r * r

    else:
        return i.A1_m2, (math.inf if g == "small_object" else i.A2_m2), []
    return area(i.r1_m), area(i.r2_m), [area(s.r_m) for s in i.shields]


def _check_eps(eps: float, what: str) -> None:
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"La emisividad {what} tiene que estar entre 0 (sin incluir) y 1.")


def solve_two_surface(inputs: TwoSurfaceInputs) -> TwoSurfaceResult:
    """Q̇₁₂ = σ(T₁⁴ − T₂⁴)/ΣR con la red de resistencias (Cengel y Ghajar, §13-4 y §13-5).

    Superficie: (1 − ε)/(A·ε); espacio: 1/(A_i·F_ij). Cada pantalla suma dos
    resistencias de superficie (una por cara) y un espacio.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    i = inputs
    if i.geometry not in TWO_SURFACE_GEOMETRIES:
        raise ValueError(f"Geometría desconocida: {i.geometry!r}.")
    if i.T1_K <= 0.0 or i.T2_K <= 0.0:
        raise ValueError("Las temperaturas tienen que ser absolutas positivas.")
    _check_eps(i.eps1, "de la superficie 1")
    _check_eps(i.eps2, "de la superficie 2")
    for k, s in enumerate(i.shields, 1):
        _check_eps(s.eps_1, f"de la pantalla {k}")
        _check_eps(s.eps_2, f"de la pantalla {k}")
    if i.shields and i.geometry in ("small_object", "general"):
        raise ValueError(
            "Las pantallas se resuelven entre placas paralelas, cilindros concéntricos o "
            "esferas concéntricas."
        )
    if i.geometry in ("concentric_cylinders", "concentric_spheres"):
        radii = [i.r1_m, *(s.r_m for s in i.shields), i.r2_m]
        if i.r1_m <= 0.0 or any(b <= a for a, b in zip(radii, radii[1:], strict=False)):
            raise ValueError(
                "Los radios tienen que crecer de adentro hacia afuera: la superficie 1, las "
                "pantallas en orden y la superficie 2."
            )
        if i.geometry == "concentric_cylinders" and i.L_m <= 0.0:
            raise ValueError("El largo de los cilindros tiene que ser positivo.")
    elif i.A1_m2 <= 0.0:
        raise ValueError("El área de la superficie 1 tiene que ser positiva.")
    if i.geometry == "general":
        if i.A2_m2 <= 0.0:
            raise ValueError("El área de la superficie 2 tiene que ser positiva.")
        if not 0.0 < i.F12 <= 1.0:
            raise ValueError("El factor de forma F₁₂ tiene que estar entre 0 (sin incluir) y 1.")
        if i.A1_m2 * i.F12 > i.A2_m2 * (1.0 + 1e-9):
            raise ValueError(
                "Con esas áreas F₂₁ = A₁·F₁₂/A₂ sería mayor que 1: no es posible (reciprocidad)."
            )
    A1, A2, A_sh = _two_surface_areas(i)
    rows: list[NetworkResistance] = []

    def surface(label: str, eps: float, A: float) -> None:
        if math.isinf(A):
            rows.append(NetworkResistance(label, 0.0, "superficie", A))
        else:
            rows.append(NetworkResistance(label, (1.0 - eps) / (A * eps), "superficie", A))

    surface("Superficie 1", i.eps1, A1)
    F_first = i.F12 if i.geometry == "general" else 1.0
    prev_A = A1
    for k, (s, A) in enumerate(zip(i.shields, A_sh, strict=True), 1):
        rows.append(
            NetworkResistance(f"Espacio hasta la pantalla {k}", 1.0 / prev_A, "espacio", prev_A)
        )
        surface(f"Pantalla {k}, cara 1", s.eps_1, A)
        surface(f"Pantalla {k}, cara 2", s.eps_2, A)
        prev_A = A
    label = "Espacio hasta la superficie 2"
    rows.append(NetworkResistance(label, 1.0 / (prev_A * F_first), "espacio", prev_A))
    surface("Superficie 2", i.eps2, A2)
    R_total = sum(r.R_per_m2 for r in rows)
    E1, E2 = SIGMA * i.T1_K**4, SIGMA * i.T2_K**4
    Q = (E1 - E2) / R_total
    # sin pantallas
    R0 = rows[0].R_per_m2 + 1.0 / (A1 * F_first) + rows[-1].R_per_m2
    Q0 = (E1 - E2) / R0
    # potenciales de la red: E_b1, J1, (J, E_b, J de cada pantalla), J2, E_b2
    potentials = [E1]
    for r in rows:
        potentials.append(potentials[-1] - Q * r.R_per_m2)
    # potentials[k + 1] es el nodo después de la resistencia k
    shield_T: list[float] = []
    for k in range(len(i.shields)):
        # filas: S1, [esp, cara1, cara2]*k, ... → el E_b de la pantalla k está después de su cara 1
        idx = 1 + 3 * k + 2
        shield_T.append((potentials[idx] / SIGMA) ** 0.25)
    radiosities = (potentials[1], potentials[-2])
    return TwoSurfaceResult(i, tuple(rows), Q, Q0, tuple(shield_T), radiosities)


def two_surface_notes(result: TwoSurfaceResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = []
    top = max(result.resistances, key=lambda r: r.R_per_m2)
    notes.append(
        f"La resistencia que manda es **{top.label.lower()}** ({_pct(result.share(top))} del "
        "total): una superficie de ε chica (brillante) frena la radiación aunque esté caliente."
    )
    if result.inputs.shields:
        n = len(result.inputs.shields)
        notes.append(
            f"Con {n} pantalla{'s' if n > 1 else ''} el calor baja al {_pct(result.reduction)} "
            f"del que pasaba sin ellas. Con todas las ε iguales y placas paralelas, N pantallas "
            "lo dividen por N + 1."
        )
        temps = ", ".join(f"{_num(T, 4)} K" for T in result.shield_T_K)
        notes.append(f"Temperaturas de las pantallas: {temps}.")
    if result.inputs.geometry == "small_object":
        notes.append(
            "Con un recinto mucho más grande, A₁/A₂ → 0: Q̇ = ε₁·A₁·σ·(T₁⁴ − T₂⁴), y la "
            "emisividad del recinto no importa (el recinto se comporta como un cuerpo negro)."
        )
    if result.Q_W < 0.0:
        notes.append("Q̇ < 0: la superficie 2 está más caliente y el calor va de 2 a 1.")
    return notes


# ---------------------------------------------------------------------
# Recinto de N superficies (radiosidades; Cengel y Ghajar, §13-4)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class EnclosureSurface:
    """Una superficie del recinto: a temperatura dada o con un calor dado.

    ``Q_W = 0`` es una superficie rerradiante (aislada); su ε no importa.
    """

    name: str
    area_m2: float
    emissivity: float
    T_K: float | None = None
    Q_W: float | None = None

    @property
    def reradiating(self) -> bool:
        return self.T_K is None and self.Q_W == 0.0


@dataclass(frozen=True)
class EnclosureInputs:
    """Las superficies y la matriz de factores de forma F[i][j]."""

    surfaces: tuple[EnclosureSurface, ...]
    F: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class EnclosureResult:
    inputs: EnclosureInputs
    J_W_per_m2: tuple[float, ...]
    Q_W: tuple[float, ...]
    T_K: tuple[float, ...]

    @property
    def E_b_W_per_m2(self) -> tuple[float, ...]:
        return tuple(SIGMA * T**4 for T in self.T_K)

    def Q_between(self, i: int, j: int) -> float:
        """Q̇_ij = A_i·F_ij·(J_i − J_j): lo que va neto de i a j."""
        s = self.inputs.surfaces[i]
        return s.area_m2 * self.inputs.F[i][j] * (self.J_W_per_m2[i] - self.J_W_per_m2[j])


def _check_enclosure(inputs: EnclosureInputs) -> None:
    S, F = inputs.surfaces, inputs.F
    n = len(S)
    if n < 2:
        raise ValueError("Un recinto necesita al menos dos superficies.")
    if len(F) != n or any(len(row) != n for row in F):
        raise ValueError(f"La matriz de factores de forma tiene que ser de {n} × {n}.")
    for k, s in enumerate(S):
        if s.area_m2 <= 0.0:
            raise ValueError(f"El área de «{s.name}» tiene que ser positiva.")
        if (s.T_K is None) == (s.Q_W is None):
            raise ValueError(f"De «{s.name}» hay que dar la temperatura o el calor (uno solo).")
        if s.T_K is not None and s.T_K <= 0.0:
            raise ValueError(f"La temperatura de «{s.name}» tiene que ser absoluta positiva.")
        if not s.reradiating:
            _check_eps(s.emissivity, f"de «{s.name}»")
        row = F[k]
        if any(f < -1e-12 for f in row):
            raise ValueError(f"Hay un factor de forma negativo en la fila de «{s.name}».")
        if not math.isclose(sum(row), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Los factores de forma de «{s.name}» suman {_num(sum(row), 6)}: en un recinto "
                "cerrado tienen que sumar 1 (regla de la suma)."
            )
        for j in range(n):
            a, b = s.area_m2 * F[k][j], S[j].area_m2 * F[j][k]
            if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(
                    f"No se cumple la reciprocidad entre «{s.name}» y «{S[j].name}»: "
                    f"A_i·F_ij = {_num(a, 5)} y A_j·F_ji = {_num(b, 5)}."
                )
    if all(s.T_K is None for s in S):
        raise ValueError("Al menos una superficie tiene que tener la temperatura dada.")


def solve_enclosure(inputs: EnclosureInputs) -> EnclosureResult:
    """El método de las radiosidades (Cengel y Ghajar, §13-4; Incropera, §13.3).

    Para cada superficie a T dada: (E_bi − J_i)·ε_i/(1 − ε_i) = Σ_j F_ij·(J_i − J_j)
    (o J_i = E_bi si es negra); con el calor dado: Q̇_i = A_i·Σ_j F_ij·(J_i − J_j).

    Raises
    ------
    ValueError
        Si los datos no forman un recinto (suma, reciprocidad) o falta algo.
    """
    _check_enclosure(inputs)
    S, F = inputs.surfaces, inputs.F
    n = len(S)
    M = np.zeros((n, n))
    b = np.zeros(n)
    for i, s in enumerate(S):
        if s.T_K is not None:
            Eb = SIGMA * s.T_K**4
            if s.emissivity >= 1.0:
                M[i, i] = 1.0
                b[i] = Eb
                continue
            k = s.emissivity / (1.0 - s.emissivity)
            M[i, i] += k
            b[i] += k * Eb
            for j in range(n):
                M[i, i] += F[i][j]
                M[i, j] -= F[i][j]
        else:
            for j in range(n):
                M[i, i] += s.area_m2 * F[i][j]
                M[i, j] -= s.area_m2 * F[i][j]
            b[i] = float(s.Q_W)  # type: ignore[arg-type]
    try:
        J = np.linalg.solve(M, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "El sistema de radiosidades no tiene solución: revisá que las superficies con "
            "calor dado puedan intercambiar con alguna a temperatura dada."
        ) from exc
    Q: list[float] = []
    T: list[float] = []
    for i, s in enumerate(S):
        qi = s.area_m2 * sum(F[i][j] * (J[i] - J[j]) for j in range(n))
        Q.append(float(qi))
        if s.T_K is not None:
            T.append(s.T_K)
        else:
            if s.reradiating or s.emissivity >= 1.0:
                Eb = J[i]
            else:
                Eb = J[i] + qi * (1.0 - s.emissivity) / (s.area_m2 * s.emissivity)
            if Eb <= 0.0:
                raise ValueError(
                    f"Con ese calor «{s.name}» tendría que estar bajo el cero absoluto: le "
                    "sacan más calor del que puede recibir. Revisá el signo y el valor."
                )
            T.append(float((Eb / SIGMA) ** 0.25))
    return EnclosureResult(inputs, tuple(float(x) for x in J), tuple(Q), tuple(T))


def triangular_duct(
    w: tuple[float, float, float],
    names: tuple[str, str, str],
    eps: tuple[float, float, float],
    T: tuple[float | None, float | None, float | None],
    Q: tuple[float | None, float | None, float | None] = (None, None, None),
) -> EnclosureInputs:
    """Un ducto largo de tres lados (por metro de largo), con F de cuerdas cruzadas."""
    F = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if i != j:
                k = 3 - i - j
                F[i][j] = _f_three_sided_2d(w[i], w[j], w[k])
    surfaces = tuple(EnclosureSurface(names[i], w[i], eps[i], T[i], Q[i]) for i in range(3))
    return EnclosureInputs(surfaces, tuple(tuple(row) for row in F))


def cylindrical_furnace(
    r: float,
    H: float,
    eps: tuple[float, float, float],
    T: tuple[float | None, float | None, float | None],
    Q: tuple[float | None, float | None, float | None] = (None, None, None),
    names: tuple[str, str, str] = ("Tapa", "Base", "Lateral"),
) -> EnclosureInputs:
    """Un horno cilíndrico: tapa (1), base (2) y lateral (3)."""
    F12 = _f_coaxial_disks(r, r, H)
    A1 = math.pi * r * r
    A3 = 2.0 * math.pi * r * H
    F13 = 1.0 - F12
    F31 = A1 * F13 / A3
    F = ((0.0, F12, F13), (F12, 0.0, F13), (F31, F31, 1.0 - 2.0 * F31))
    areas = (A1, A1, A3)
    surfaces = tuple(EnclosureSurface(names[i], areas[i], eps[i], T[i], Q[i]) for i in range(3))
    return EnclosureInputs(surfaces, F)


def open_cylindrical_cavity(
    r: float, L: float, eps_side: float, eps_bottom: float, T_side: float, T_bottom: float,
    T_surr: float,
) -> EnclosureInputs:  # fmt: skip
    """Una cavidad cilíndrica abierta: lateral, fondo y la boca como superficie negra a la
    temperatura de los alrededores (Incropera, ejemplo 13.2)."""
    A_side, A_disk = 2.0 * math.pi * r * L, math.pi * r * r
    F_bm = _f_coaxial_disks(r, r, L)  # fondo → boca
    F_bs = 1.0 - F_bm
    F_sb = A_disk * F_bs / A_side
    F = ((1.0 - 2.0 * F_sb, F_sb, F_sb), (F_bs, 0.0, F_bm), (F_bs, F_bm, 0.0))
    surfaces = (
        EnclosureSurface("Lateral", A_side, eps_side, T_side),
        EnclosureSurface("Fondo", A_disk, eps_bottom, T_bottom),
        EnclosureSurface("Boca (alrededores)", A_disk, 1.0, T_surr),
    )
    return EnclosureInputs(surfaces, F)


def parallel_plates_with_surroundings(
    X: float, Y: float, L: float, eps1: float, eps2: float, T1: float, T2: float, T_surr: float
) -> EnclosureInputs:
    """Dos rectángulos paralelos frente a los alrededores (las aberturas laterales como una
    superficie negra a la temperatura de los alrededores)."""
    F12 = _f_parallel_rectangles(X, Y, L)
    A = X * Y
    A3 = 2.0 * L * (X + Y)
    F13 = 1.0 - F12
    F31 = A * F13 / A3
    F = ((0.0, F12, F13), (F12, 0.0, F13), (F31, F31, 1.0 - 2.0 * F31))
    surfaces = (
        EnclosureSurface("Placa 1", A, eps1, T1),
        EnclosureSurface("Placa 2", A, eps2, T2),
        EnclosureSurface("Alrededores", A3, 1.0, T_surr),
    )
    return EnclosureInputs(surfaces, F)


def enclosure_notes(result: EnclosureResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = []
    S = result.inputs.surfaces
    total = sum(result.Q_W)
    scale = max(abs(q) for q in result.Q_W) or 1.0
    notes.append(
        f"Balance: ΣQ̇ = {_num(total / scale * 100.0, 2)}·10⁻² % del mayor: lo que entregan unas "
        "superficies lo reciben las otras (el recinto está cerrado)."
        if abs(total) > 1e-9 * scale
        else "Balance: ΣQ̇ = 0 (lo que entregan unas superficies lo reciben las otras)."
    )
    for s, T in zip(S, result.T_K, strict=True):
        if s.reradiating:
            notes.append(
                f"«{s.name}» es rerradiante: no gana ni pierde calor neto, devuelve todo lo que "
                f"recibe y queda a {_num(T, 4)} K, entre las otras temperaturas (su J = E_b)."
            )
    hot = max(range(len(S)), key=lambda k: result.Q_W[k])
    notes.append(
        f"La que más calor entrega es «{S[hot].name}» ({_num(result.Q_W[hot] / 1e3, 4)} kW); "
        "las de Q̇ < 0 lo reciben."
    )
    return notes


# ---------------------------------------------------------------------
# Radiación y convección juntas (Cengel y Ghajar, §13-5; Incropera, §13.4)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceBalanceInputs:
    """Una superficie que pierde calor por convección y radiación (y gana sol).

    Con ``T_s_K`` dada sale el calor; con ``Q_in_W`` (lo que se le entrega: una
    resistencia, el calor que conduce desde adentro) sale la T_s de equilibrio.
    """

    emissivity: float
    h_W_per_m2K: float
    T_inf_K: float
    T_surr_K: float
    area_m2: float = 1.0
    T_s_K: float | None = None
    Q_in_W: float | None = None
    alpha_solar: float = 0.0
    G_solar_W_per_m2: float = 0.0


@dataclass(frozen=True)
class SurfaceBalanceResult:
    inputs: SurfaceBalanceInputs
    T_s_K: float

    @property
    def Q_conv_W(self) -> float:
        i = self.inputs
        return i.h_W_per_m2K * i.area_m2 * (self.T_s_K - i.T_inf_K)

    @property
    def Q_rad_W(self) -> float:
        i = self.inputs
        return i.emissivity * SIGMA * i.area_m2 * (self.T_s_K**4 - i.T_surr_K**4)

    @property
    def Q_solar_W(self) -> float:
        i = self.inputs
        return i.alpha_solar * i.G_solar_W_per_m2 * i.area_m2

    @property
    def Q_lost_W(self) -> float:
        """Lo que pierde neto la superficie: Q̇_conv + Q̇_rad − Q̇_sol."""
        return self.Q_conv_W + self.Q_rad_W - self.Q_solar_W

    @property
    def h_rad_W_per_m2K(self) -> float:
        i = self.inputs
        Ts, Tr = self.T_s_K, i.T_surr_K
        return i.emissivity * SIGMA * (Ts + Tr) * (Ts * Ts + Tr * Tr)

    @property
    def radiation_share(self) -> float:
        total = abs(self.Q_conv_W) + abs(self.Q_rad_W)
        return abs(self.Q_rad_W) / total if total else 0.0


def solve_surface_balance(inputs: SurfaceBalanceInputs) -> SurfaceBalanceResult:
    """Q̇ = h·A·(T_s − T∞) + ε·σ·A·(T_s⁴ − T_alr⁴) − α_s·G·A, o la T_s que lo anula.

    Incropera et al. (2007), §13.4 (ejemplo 1.2); Cengel y Ghajar, §12-6.

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    i = inputs
    _check_eps(i.emissivity, "de la superficie")
    if i.h_W_per_m2K < 0.0:
        raise ValueError("El coeficiente de convección h no puede ser negativo.")
    if i.T_inf_K <= 0.0 or i.T_surr_K <= 0.0:
        raise ValueError("Las temperaturas tienen que ser absolutas positivas.")
    if i.area_m2 <= 0.0:
        raise ValueError("El área tiene que ser positiva.")
    if not 0.0 <= i.alpha_solar <= 1.0:
        raise ValueError("La absortividad solar tiene que estar entre 0 y 1.")
    if i.G_solar_W_per_m2 < 0.0:
        raise ValueError("La radiación solar incidente no puede ser negativa.")
    if (i.T_s_K is None) == (i.Q_in_W is None):
        raise ValueError("Hay que dar la temperatura de la superficie o el calor que recibe.")
    if i.T_s_K is not None:
        if i.T_s_K <= 0.0:
            raise ValueError("La temperatura de la superficie tiene que ser absoluta positiva.")
        return SurfaceBalanceResult(i, i.T_s_K)
    q_in = float(i.Q_in_W)  # type: ignore[arg-type]

    def balance(T: float) -> float:
        return SurfaceBalanceResult(i, T).Q_lost_W - q_in

    lo, hi = 1.0, 5000.0
    if balance(lo) > 0.0 or balance(hi) < 0.0:
        raise ValueError(
            "No hay una temperatura de equilibrio entre 1 y 5000 K con esos datos: revisá el "
            "calor que recibe la superficie."
        )
    return SurfaceBalanceResult(i, float(brentq(balance, lo, hi, xtol=1e-12)))


def surface_balance_notes(result: SurfaceBalanceResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes = [
        f"La radiación es el {_pct(result.radiation_share)} del calor que intercambia la "
        f"superficie con el aire y los alrededores; h_rad = {_num(result.h_rad_W_per_m2K, 3)} "
        f"W/(m²·K) contra h = {_num(result.inputs.h_W_per_m2K, 3)} W/(m²·K).",
    ]
    if result.inputs.h_W_per_m2K < 10.0:
        notes.append(
            "Con convección natural (h chico) la radiación pesa tanto como la convección aun a "
            "temperaturas moderadas: no se puede despreciar."
        )
    if result.Q_solar_W > 0.0:
        notes.append(
            f"Absorbe {_num(result.Q_solar_W, 4)} W del sol (α_s·G·A): con una superficie "
            "selectiva (α_s alta y ε baja) se calienta más que con una gris."
        )
    if result.inputs.Q_in_W is None and result.Q_lost_W < 0.0:
        notes.append(
            f"La superficie gana {_num(-result.Q_lost_W, 4)} W netos: para mantenerla a esa "
            "temperatura hay que sacarle ese calor (o se calienta)."
        )
    return notes


@dataclass(frozen=True)
class ThermocoupleInputs:
    """Una termocupla en un gas, con las paredes del ducto a otra temperatura."""

    T_reading_K: float
    T_wall_K: float
    emissivity: float
    h_W_per_m2K: float


@dataclass(frozen=True)
class ThermocoupleResult:
    inputs: ThermocoupleInputs

    @property
    def T_gas_K(self) -> float:
        """T_f = T_th + ε·σ·(T_th⁴ − T_w⁴)/h (Cengel y Ghajar, §13-5)."""
        i = self.inputs
        return (
            i.T_reading_K
            + i.emissivity * SIGMA * (i.T_reading_K**4 - i.T_wall_K**4) / i.h_W_per_m2K
        )

    @property
    def error_K(self) -> float:
        return self.T_gas_K - self.inputs.T_reading_K

    @property
    def q_rad_W_per_m2(self) -> float:
        i = self.inputs
        return i.emissivity * SIGMA * (i.T_reading_K**4 - i.T_wall_K**4)


def solve_thermocouple(inputs: ThermocoupleInputs) -> ThermocoupleResult:
    """La temperatura real del gas a partir de la lectura (balance de la junta).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido.
    """
    i = inputs
    if i.T_reading_K <= 0.0 or i.T_wall_K <= 0.0:
        raise ValueError("Las temperaturas tienen que ser absolutas positivas.")
    _check_eps(i.emissivity, "de la junta")
    if i.h_W_per_m2K <= 0.0:
        raise ValueError("El coeficiente de convección h tiene que ser positivo.")
    return ThermocoupleResult(i)


def thermocouple_notes(result: ThermocoupleResult) -> list[str]:
    """Interpretación física (markdown)."""
    i = result.inputs
    err = result.error_K
    notes = [
        f"La termocupla marca {_num(abs(err), 3)} K {'menos' if err > 0 else 'más'} que el gas: "
        f"la junta {'pierde' if err > 0 else 'gana'} por radiación con las paredes "
        f"(a {_num(i.T_wall_K, 4)} K) lo que {'gana' if err > 0 else 'pierde'} por convección.",
        "El error baja con un h mayor (gas más rápido), una junta de ε chica o una pantalla "
        "de radiación alrededor de la junta.",
    ]
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RadiationExample:
    inputs: Any
    note: str = ""


BLACKBODY_EXAMPLES: dict[str, RadiationExample] = {
    "Bola negra a 800 K (Cengel y Ghajar 12-1)": RadiationExample(
        BlackbodyInputs(800.0, lambda_point_m=3.0e-6),
        "El libro: E_b = 23,2 kW/m² y E_bλ(3 μm) = 3846 W/(m²·μm).",
    ),
    "Filamento de una lámpara a 2500 K: la fracción visible": RadiationExample(
        BlackbodyInputs(2500.0, band_lower_m=0.40e-6, band_upper_m=0.76e-6),
        "Solo un 5 % de la emisión cae en el visible.",
    ),
    "El sol a 5800 K: el visible y dónde está el máximo": RadiationExample(
        BlackbodyInputs(5800.0, band_lower_m=0.40e-6, band_upper_m=0.76e-6),
        "El máximo cae en el visible: nuestros ojos están hechos para el sol.",
    ),
    "Recinto a 2000 K: dónde se emite el 10 % (basado en Incropera)": RadiationExample(
        BlackbodyInputs(2000.0, fraction_target=0.10),
        "El 10 % de la emisión está por debajo de 1,10 μm; el 90 %, por debajo de 4,69 μm.",
    ),
    "Superficie con ε por bandas a 800 K (Cengel y Ghajar 12-4)": RadiationExample(
        BlackbodyInputs(
            800.0,
            bands=(SpectralBand(0.3, 3.0e-6), SpectralBand(0.8, 7.0e-6), SpectralBand(0.1)),
        ),
        "El libro: ε = 0,521 y E = 12,1 kW/m².",
    ),
    "Absorbedor solar selectivo a 350 K (α del sol y ε propia)": RadiationExample(
        BlackbodyInputs(
            350.0, bands=(SpectralBand(0.95, 3.0e-6), SpectralBand(0.08)), T_source_K=5800.0
        ),
        "Absorbe casi todo el sol y emite poco en el infrarrojo: α_s ≫ ε.",
    ),
}

VIEW_FACTOR_EXAMPLES: dict[str, RadiationExample] = {
    "Techo y piso de un horno cúbico de 4 m": RadiationExample(
        ViewFactorInputs("parallel_rectangles", 4.0, 4.0, 4.0),
        "F = 0,1998: cada cara del cubo ve un 20 % de lo que emite en la de enfrente.",
    ),
    "El fondo y la boca de una cavidad cilíndrica (Incropera 13.2)": RadiationExample(
        ViewFactorInputs("coaxial_disks", 0.0375, 0.0375, 0.15),
        "Discos de 37,5 mm a 150 mm: F = 0,0557.",
    ),
    "Pared y piso de una habitación (rectángulos perpendiculares)": RadiationExample(
        ViewFactorInputs("perpendicular_rectangles", 4.0, 3.0, 2.6),
        "El piso (4 × 3 m) y una pared de 4 m × 2,6 m de alto.",
    ),
    "Dos placas largas enfrentadas": RadiationExample(
        ViewFactorInputs("parallel_plates_2d", 0.4, 0.4, 0.2),
        "Placas de 40 cm a 20 cm: la mitad de lo que emite cada una se escapa por los costados.",
    ),
    "Ducto triangular equilátero": RadiationExample(
        ViewFactorInputs("three_sided_2d", 1.0, 1.0, 1.0),
        "Por simetría y la regla de la suma, F = 1/2 entre dos lados cualesquiera.",
    ),
    "Pared de una caldera y una hilera de tubos": RadiationExample(
        ViewFactorInputs("tube_row_2d", 0.05, 0.10),
        "Tubos de 50 mm con un paso de 100 mm: la pared ve los tubos un 66 % del total.",
    ),
}

TWO_SURFACE_EXAMPLES: dict[str, RadiationExample] = {
    "Placas paralelas a 800 y 500 K (Cengel y Ghajar)": RadiationExample(
        TwoSurfaceInputs("parallel_plates", 800.0, 500.0, 0.2, 0.7),
        "El libro: 3625 W/m².",
    ),
    "Las mismas con una pantalla de aluminio (Cengel y Ghajar)": RadiationExample(
        TwoSurfaceInputs("parallel_plates", 800.0, 500.0, 0.2, 0.7, shields=(Shield(0.1, 0.1),)),
        "El libro: 806 W/m², un 22 % de lo que pasaba sin la pantalla.",
    ),
    "Tanque esférico de nitrógeno líquido con vacío alrededor": RadiationExample(
        TwoSurfaceInputs("concentric_spheres", 77.0, 300.0, 0.02, 0.05, r1_m=0.25, r2_m=0.275),
        "Q̇ < 0: el calor entra al tanque y evapora el nitrógeno; las superficies brillantes "
        "(ε chica) son la aislación.",
    ),
    "Caño de vapor dentro de una camisa (cilindros concéntricos)": RadiationExample(
        TwoSurfaceInputs(
            "concentric_cylinders", 473.15, 303.15, 0.8, 0.9, r1_m=0.035, r2_m=0.08, L_m=1.0
        ),
        "Por metro de caño.",
    ),
    "Caño de vapor desnudo en un cuarto grande (basado en Incropera 1.2)": RadiationExample(
        TwoSurfaceInputs("small_object", 473.15, 298.15, 0.8, 1.0, A1_m2=math.pi * 0.07),
        "El libro: 421 W por metro (más 577 W por convección).",
    ),
}

ENCLOSURE_EXAMPLES: dict[str, RadiationExample] = {
    "Horno de pintura: ducto triangular con una pared rerradiante (basado en Incropera)": (
        RadiationExample(
            triangular_duct(
                (1.0, 1.0, 1.0),
                ("Calefactor", "Paneles pintados", "Pared aislada"),
                (0.8, 0.4, 0.8),
                (1200.0, 500.0, None),
                (None, None, 0.0),
            ),
            "Por metro de largo: hay que entregar 37 kW/m y la pared aislada queda a 1102 K.",
        )
    ),
    "Ducto triangular con calor dado en la base (Cengel y Ghajar)": RadiationExample(
        triangular_duct(
            (1.0, 1.0, 1.0),
            ("Base", "Lado izquierdo", "Lado derecho"),
            (0.8, 0.5, 0.5),
            (None, 500.0, 500.0),
            (800.0, None, None),
        ),
        "El libro: la base queda a 543 K.",
    ),
    "Cavidad cilíndrica abierta: la potencia del horno (Incropera 13.2)": RadiationExample(
        open_cylindrical_cavity(0.0375, 0.15, 1.0, 1.0, 1350.0 + _K0, 1650.0 + _K0, 300.0),
        "El libro: 1830 W. La boca es una superficie negra a la temperatura del ambiente.",
    ),
    "Horno cilíndrico de tres superficies grises": RadiationExample(
        cylindrical_furnace(1.0, 1.0, (0.8, 0.4, 1.0), (700.0, 500.0, 400.0)),
        "r = H = 1 m: la tapa entrega calor y la base y el lateral lo reciben.",
    ),
    "Dos placas frente a los alrededores": RadiationExample(
        parallel_plates_with_surroundings(1.0, 1.0, 0.5, 0.8, 0.6, 900.0, 400.0, 300.0),
        "Placas de 1 × 1 m a 0,5 m; lo que se escapa por los costados va a los alrededores.",
    ),
}

SURFACE_EXAMPLES: dict[str, RadiationExample] = {
    "Caño de vapor desnudo en un cuarto (Incropera 1.2)": RadiationExample(
        SurfaceBalanceInputs(0.8, 15.0, 298.15, 298.15, area_m2=math.pi * 0.07, T_s_K=473.15),
        "Por metro: 577 W por convección y 421 W por radiación (998 W).",
    ),
    "Absorbedor gris al sol (Cengel y Ghajar)": RadiationExample(
        SurfaceBalanceInputs(
            0.9,
            0.0,
            300.0,
            260.0,
            T_s_K=320.0,
            alpha_solar=0.9,
            G_solar_W_per_m2=400.0 * math.cos(math.radians(20.0)) + 300.0,
        ),  # fmt: skip
        "α_s = ε = 0,9 a 320 K con el cielo a 260 K: gana 306 W/m² (el libro: 306).",
    ),
    "Absorbedor selectivo al sol (Cengel y Ghajar)": RadiationExample(
        SurfaceBalanceInputs(
            0.1,
            0.0,
            300.0,
            260.0,
            T_s_K=320.0,
            alpha_solar=0.9,
            G_solar_W_per_m2=400.0 * math.cos(math.radians(20.0)) + 300.0,
        ),  # fmt: skip
        "α_s = 0,9 y ε = 0,1: gana 575 W/m² (el libro: 575).",
    ),
    "Resistencia de 1 kW en un cuarto: ¿a qué temperatura queda?": RadiationExample(
        SurfaceBalanceInputs(0.9, 10.0, 293.15, 293.15, area_m2=0.25, Q_in_W=1000.0),
        "La temperatura de equilibrio sale del balance (una ecuación con T⁴).",
    ),
}

THERMOCOUPLE_EXAMPLES: dict[str, RadiationExample] = {
    "Termocupla en un ducto de aire caliente (Cengel y Ghajar)": RadiationExample(
        ThermocoupleInputs(650.0, 400.0, 0.6, 80.0),
        "El libro: el aire está a 715 K aunque la termocupla marca 650 K.",
    ),
    "La misma con una junta brillante (ε = 0,1)": RadiationExample(
        ThermocoupleInputs(650.0, 400.0, 0.1, 80.0),
        "Con ε chica el error baja a un sexto.",
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def blackbody_to_dict(result: BlackbodyResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable."""
    data: dict[str, Any] = {
        "T": format_quantity(result.inputs.T_K, "absolute_temperature", system),
        "E_b": format_quantity(result.E_b_W_per_m2, "heat_flux", system),
        "lambda_max": format_quantity(result.lambda_max_m, "wavelength", system),
        "E_blambda_max": format_quantity(
            result.E_blambda_max_W_per_m3, "spectral_emissive_power", system
        ),
    }
    if result.E_blambda_point_W_per_m3 is not None:
        data["E_blambda"] = format_quantity(
            result.E_blambda_point_W_per_m3, "spectral_emissive_power", system
        )
    if result.band_fraction is not None:
        data["fraccion_en_la_banda"] = result.band_fraction
    if result.lambda_target_m is not None:
        data["lambda_de_la_fraccion"] = format_quantity(
            result.lambda_target_m, "wavelength", system
        )
    if result.emissivity is not None:
        data["emisividad"] = result.emissivity
        data["E"] = format_quantity(result.E_W_per_m2, "heat_flux", system)
    if result.absorptivity is not None:
        data["absortividad"] = result.absorptivity
    return data


def view_factor_to_dict(result: ViewFactorResult, system: UnitSystem) -> dict[str, Any]:
    vf = VIEW_FACTORS[result.inputs.geometry]
    values = (result.inputs.a, result.inputs.b, result.inputs.c)[: len(vf.dims)]
    return {
        "geometria": vf.name,
        "dimensiones": {
            d: format_quantity(v, "length", system) for d, v in zip(vf.dims, values, strict=True)
        },
        "F_ij": result.F_ij,
        "F_ji": result.F_ji,
        "F_i_resto": result.F_i_rest,
    }


def two_surface_to_dict(result: TwoSurfaceResult, system: UnitSystem) -> dict[str, Any]:
    i = result.inputs
    return {
        "geometria": TWO_SURFACE_GEOMETRIES[i.geometry],
        "T1": format_quantity(i.T1_K, "absolute_temperature", system),
        "T2": format_quantity(i.T2_K, "absolute_temperature", system),
        "eps1": i.eps1,
        "eps2": i.eps2,
        "Q": format_quantity(result.Q_W, "heat_rate", system),
        "Q_sin_pantallas": format_quantity(result.Q_no_shields_W, "heat_rate", system),
        "pantallas": [
            {
                "eps_1": s.eps_1,
                "eps_2": s.eps_2,
                "T": format_quantity(T, "absolute_temperature", system),
            }
            for s, T in zip(i.shields, result.shield_T_K, strict=True)
        ],
    }


def enclosure_to_dict(result: EnclosureResult, system: UnitSystem) -> dict[str, Any]:
    return {
        "superficies": [
            {
                "nombre": s.name,
                "area": format_quantity(s.area_m2, "area", system),
                "emisividad": s.emissivity,
                "T": format_quantity(T, "absolute_temperature", system),
                "J": format_quantity(J, "heat_flux", system),
                "Q": format_quantity(Q, "heat_rate", system),
            }
            for s, T, J, Q in zip(
                result.inputs.surfaces, result.T_K, result.J_W_per_m2, result.Q_W, strict=True
            )
        ],
        "F": [list(row) for row in result.inputs.F],
    }


def surface_to_dict(result: SurfaceBalanceResult, system: UnitSystem) -> dict[str, Any]:
    return {
        "T_s": format_quantity(result.T_s_K, "temperature", system),
        "Q_conveccion": format_quantity(result.Q_conv_W, "heat_rate", system),
        "Q_radiacion": format_quantity(result.Q_rad_W, "heat_rate", system),
        "Q_sol": format_quantity(result.Q_solar_W, "heat_rate", system),
        "Q_neto_perdido": format_quantity(result.Q_lost_W, "heat_rate", system),
        "h_rad": format_quantity(result.h_rad_W_per_m2K, "heat_transfer_coefficient", system),
    }


def thermocouple_to_dict(result: ThermocoupleResult, system: UnitSystem) -> dict[str, Any]:
    return {
        "T_lectura": format_quantity(result.inputs.T_reading_K, "temperature", system),
        "T_gas": format_quantity(result.T_gas_K, "temperature", system),
        "error": format_quantity(result.error_K, "temperature_difference", system),
    }
