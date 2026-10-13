"""ISO 6976:2016 — poder calorífico, densidad y Wobbe para gas combustible.

Implementación didáctica de la norma ISO 6976:2016. Calcula, a partir
de la composición molar de una mezcla gaseosa combustible y de las
condiciones de referencia (combustión y medición):

- masa molar M y factor de sumación s,
- factor de compresibilidad Z = 1 − s² y volumen molar V_m,
- calor de combustión molar bruto H_c,G y neto H_c,N,
- calor específico másico (H_m,G / H_m,N) y volumétrico (H_v,G / H_v,N),
- densidad y densidad relativa al aire,
- índice de Wobbe bruto y neto,

junto con la propagación de incertidumbre estándar con dos matrices de
correlación de la composición:

- ``identity``: las u(xⱼ) se tratan como independientes. Es conservadora: la
  norma (§D.4.3.2) la llama una sobre-estimación segura (*safe overestimate*).
- ``normalization``: la composición se normalizó a Σx = 1, así que las
  fracciones están correlacionadas (si una sube, las demás bajan). La norma
  calcula esa covarianza con ISO 14912:2003 desde las incertidumbres del
  análisis crudo; el Anexo D publica solo las u(xⱼ) ya normalizadas, y
  :func:`normalization_fit` reconstruye las crudas (ver las notas al pie).

El módulo no importa Streamlit. Las tablas (4 CSV en ``data/``) se
cargan con :func:`load_tables`; la página de UI puede envolver esa
carga con ``@st.cache_data``.

Cita
----
International Organization for Standardization. (2016). *ISO 6976:2016
Natural gas — Calculation of calorific values, density, relative
density and Wobbe indices from composition*.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import nnls

# ---------------------------------------------------------------------
# Constantes del módulo
# ---------------------------------------------------------------------

_SUPPORTED_T_COMBUSTION: tuple[float, ...] = (0.0, 15.0, 15.55, 20.0, 25.0)
_SUPPORTED_T_METERING: tuple[float, ...] = (0.0, 15.0, 15.55, 20.0)
_TOL_SUM = 1.0e-6
_NUMERICAL_H_REL = 1.0e-6
_NUMERICAL_H_ABS = 1.0e-12
# Matriz de normalización: las u(x) que reconstruye el modelo pueden apartarse de las
# dadas un 1 % más 10⁻³ de la mayor (las u se publican redondeadas a una o dos cifras).
_NORM_TOL_REL = 1.0e-2
_NORM_TOL_ABS = 1.0e-3

# Mapa de label °C → kelvin exacto. La etiqueta "15.55" es el redondeo de
# 60 °F = (60 − 32)·5/9 = 15.5555… °C, **no** un decimal exacto. La norma
# y las tablas usan ese valor preciso aunque escriban "15.55".
_T_CELSIUS_TO_KELVIN: dict[float, float] = {
    0.0: 273.15,
    15.0: 288.15,
    15.55: 273.15 + (60.0 - 32.0) * 5.0 / 9.0,  # 60 °F = 288.7055555555556 K
    20.0: 293.15,
    25.0: 298.15,
}

_NOBLE_GASES: frozenset[str] = frozenset({"helium", "neon", "argon"})
_NON_COMBUSTIBLES: frozenset[str] = frozenset(
    {
        "argon",
        "carbon dioxide",
        "helium",
        "neon",
        "nitrogen",
        "oxygen",
        "sulfur dioxide",
    }
)

# Por defecto: <repo_root>/data
_DATA_DIR_DEFAULT: Path = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------
# Dataclasses públicas
# ---------------------------------------------------------------------

CorrelationMatrix = Literal["identity", "normalization"]
#: De qué son las u de la composición con la matriz de normalización: de las
#: fracciones ya normalizadas (como el Anexo D) o del análisis crudo, antes de
#: normalizar (como las da un cromatógrafo calibrado).
UncertaintyBasis = Literal["normalized", "raw"]


@dataclass(frozen=True)
class GasComponent:
    """Componente de la mezcla con fracción molar e incertidumbre absoluta."""

    name: str
    x: float
    u_x: float = 0.0


@dataclass(frozen=True)
class ReferenceCondition:
    """Condición de referencia (combustión o medición), en °C y kPa."""

    T_celsius: float
    P_kPa: float = 101.325


@dataclass(frozen=True)
class ISO6976Inputs:
    """Entradas completas del cálculo según ISO 6976:2016."""

    composition: list[GasComponent]
    combustion_reference: ReferenceCondition
    metering_reference: ReferenceCondition
    correlation_matrix: CorrelationMatrix
    #: Solo con la matriz de normalización (con la identidad, las u son las de x).
    uncertainty_basis: UncertaintyBasis = "normalized"


@dataclass(frozen=True)
class Quantity:
    """Valor escalar con su propagación de incertidumbre.

    ``u`` es la incertidumbre estándar combinada (k=1).
    ``U_k2 = 2·u`` es la expandida con factor de cobertura 2
    (aprox. 95 % bajo distribución normal).
    """

    value: float
    u: float
    U_k2: float


@dataclass(frozen=True)
class ISO6976Steps:
    """Pasos didácticos: fórmula en LaTeX, sustitución y narrativa.

    ``uncertainty_latex`` es la propagación de la incertidumbre de la composición
    con la matriz elegida (una ecuación por elemento).
    """

    formula_latex: str
    substituted_latex: str
    narrative_es: str
    uncertainty_latex: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizationFit:
    """Las incertidumbres del análisis crudo y de la composición normalizada.

    ``u_given`` son las que se cargaron (de x o de y, según ``basis``), ``u_raw`` las
    u(yⱼ) de las fracciones medidas antes de normalizar, ``u_normalized`` las u(xⱼ) de
    las fracciones normalizadas (iguales a las dadas, dentro de la tolerancia, si
    ``basis`` es ``"normalized"``) y ``correlation`` los coeficientes r(xᵢ, xⱼ).
    """

    names: tuple[str, ...]
    x: tuple[float, ...]
    basis: UncertaintyBasis
    u_given: tuple[float, ...]
    u_raw: tuple[float, ...]
    u_normalized: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class ISO6976Result:
    """Resultado completo: 4 intermedios + 10 finales + bloque didáctico."""

    inputs: ISO6976Inputs
    # Intermedios
    molar_mass_kg_per_kmol: Quantity
    summation_factor: Quantity
    compression_factor: Quantity
    molar_volume_m3_per_mol: Quantity
    # Calores molares
    Hc_G_molar_kJ_per_mol: Quantity
    Hc_N_molar_kJ_per_mol: Quantity
    # Específicos en masa
    Hm_G_mass_MJ_per_kg: Quantity
    Hm_N_mass_MJ_per_kg: Quantity
    # Volumétricos
    Hv_G_volume_MJ_per_m3: Quantity
    Hv_N_volume_MJ_per_m3: Quantity
    # Otras propiedades
    density_kg_per_m3: Quantity
    relative_density: Quantity
    Wobbe_gross_MJ_per_m3: Quantity
    Wobbe_net_MJ_per_m3: Quantity
    # Didáctico
    steps: ISO6976Steps
    # Matriz de normalización: las u del análisis crudo (None con la identidad)
    normalization: NormalizationFit | None = None


# ---------------------------------------------------------------------
# Tablas — estructura semi-pública para inyectar en tests
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentRow:
    """Fila de la Tabla 1: símbolo (nombre), j, M y conteos atómicos."""

    j: int
    name: str
    M_kg_per_kmol: float
    a: int  # C
    b: int  # H
    c: int  # N
    d: int  # O
    e: int  # S


@dataclass(frozen=True)
class ISO6976Tables:
    """Conjunto consolidado de tablas de ISO 6976:2016."""

    components: dict[str, ComponentRow]
    summation_factors: dict[str, dict[float, float]]
    u_summation_factors: dict[str, float]
    calorific_values: dict[str, dict[float, float]]
    u_calorific_values: dict[str, float]
    atomic_weights: dict[str, float]
    u_atomic_weights: dict[str, float]
    R_J_per_mol_K: float
    M_air_kg_per_kmol: float
    u_M_air_kg_per_kmol: float
    Z_air_by_T: dict[float, float]
    u_Z_air: float
    L0_water_by_T: dict[float, float]
    u_L0_water: float


# ---------------------------------------------------------------------
# Carga de tablas desde CSV
# ---------------------------------------------------------------------


def _parse_T_from_ref(ref_str: Any) -> float | None:
    """``"15_celsius_101.325_kPa"`` → ``15.0``. ``NaN`` o vacío → ``None``."""
    if ref_str is None or (isinstance(ref_str, float) and pd.isna(ref_str)):
        return None
    s = str(ref_str).strip()
    if not s:
        return None
    parts = s.split("_")
    return float(parts[0])


def load_tables(*, data_dir: Path | None = None) -> ISO6976Tables:
    """Carga las 4 tablas de ISO 6976:2016 desde el directorio ``data/``.

    Devuelve una estructura inmutable :class:`ISO6976Tables` con dicts
    indexados por nombre de componente o por temperatura de referencia.
    """
    base = data_dir if data_dir is not None else _DATA_DIR_DEFAULT

    # Tabla 1 — components
    comps_df = pd.read_csv(base / "iso6976_components.csv")
    components: dict[str, ComponentRow] = {}
    for _, row in comps_df.iterrows():
        name = str(row["name"])
        components[name] = ComponentRow(
            j=int(row["j"]),
            name=name,
            M_kg_per_kmol=float(row["molar_mass_kg_per_kmol"]),
            a=int(row["a"]),
            b=int(row["b"]),
            c=int(row["c"]),
            d=int(row["d"]),
            e=int(row["e"]),
        )

    # Tabla 2 — summation factors
    sf_df = pd.read_csv(base / "iso6976_summation_factors.csv")
    summation_factors: dict[str, dict[float, float]] = {}
    u_summation_factors: dict[str, float] = {}
    for _, row in sf_df.iterrows():
        name = str(row["name"])
        summation_factors[name] = {
            0.0: float(row["s_0C"]),
            15.0: float(row["s_15C"]),
            15.55: float(row["s_15.55C"]),
            20.0: float(row["s_20C"]),
        }
        u_summation_factors[name] = float(row["u_s"])

    # Tabla 3 — calorific values (53 combustibles)
    hc_df = pd.read_csv(base / "iso6976_calorific_values.csv")
    calorific_values: dict[str, dict[float, float]] = {}
    u_calorific_values: dict[str, float] = {}
    for _, row in hc_df.iterrows():
        name = str(row["name"])
        calorific_values[name] = {
            0.0: float(row["Hc_0C"]),
            15.0: float(row["Hc_15C"]),
            15.55: float(row["Hc_15.55C"]),
            20.0: float(row["Hc_20C"]),
            25.0: float(row["Hc_25C"]),
        }
        u_calorific_values[name] = float(row["u_Hc"])
    # Los no combustibles tienen Hc = 0 a todas las temperaturas, u_Hc = 0.
    for name in _NON_COMBUSTIBLES:
        if name in components and name not in calorific_values:
            calorific_values[name] = {T: 0.0 for T in _SUPPORTED_T_COMBUSTION}
            u_calorific_values[name] = 0.0

    # Annex A — constants
    const_df = pd.read_csv(base / "iso6976_constants.csv")
    atomic_weights: dict[str, float] = {}
    u_atomic_weights: dict[str, float] = {}
    R_value = 8.3144621
    M_air = 28.96546
    u_M_air = 0.0
    Z_air_by_T: dict[float, float] = {}
    u_Z_air = 0.0
    L0_water_by_T: dict[float, float] = {}
    u_L0_water = 0.0

    for _, row in const_df.iterrows():
        cat = str(row["category"])
        key = str(row["key"])
        val = float(row["value"])
        unc = float(row["uncertainty"]) if pd.notna(row.get("uncertainty")) else 0.0
        ref = row.get("reference_condition")

        if cat == "gas_constant":
            R_value = val
        elif cat == "atomic_weight":
            atomic_weights[key] = val
            u_atomic_weights[key] = unc
        elif cat == "air":
            if key == "M_air":
                M_air = val
                u_M_air = unc
            elif key == "Z_air":
                T_celsius = _parse_T_from_ref(ref)
                if T_celsius is not None:
                    Z_air_by_T[T_celsius] = val
                u_Z_air = unc  # idéntica para todas las T tabuladas
        elif cat == "water":
            if key == "L0":
                T_celsius = _parse_T_from_ref(ref)
                if T_celsius is not None:
                    L0_water_by_T[T_celsius] = val
                u_L0_water = unc

    return ISO6976Tables(
        components=components,
        summation_factors=summation_factors,
        u_summation_factors=u_summation_factors,
        calorific_values=calorific_values,
        u_calorific_values=u_calorific_values,
        atomic_weights=atomic_weights,
        u_atomic_weights=u_atomic_weights,
        R_J_per_mol_K=R_value,
        M_air_kg_per_kmol=M_air,
        u_M_air_kg_per_kmol=u_M_air,
        Z_air_by_T=Z_air_by_T,
        u_Z_air=u_Z_air,
        L0_water_by_T=L0_water_by_T,
        u_L0_water=u_L0_water,
    )


# ---------------------------------------------------------------------
# Validación de inputs
# ---------------------------------------------------------------------


def _validate_inputs(inputs: ISO6976Inputs, tables: ISO6976Tables) -> None:
    if not inputs.composition:
        raise ValueError("La composición no puede estar vacía.")

    known = set(tables.components.keys())
    for comp in inputs.composition:
        if comp.name not in known:
            raise ValueError(
                f"Componente desconocido: {comp.name!r}. Los nombres "
                f"aceptados son los 60 componentes de la Tabla 1 de "
                f"ISO 6976:2016, en inglés y con la ortografía exacta "
                f"de la norma (ej.: 'methane', 'n-butane', "
                f"'2,2-dimethylpropane', 'carbon dioxide')."
            )
        if not (0.0 <= comp.x <= 1.0):
            raise ValueError(
                f"La fracción molar de {comp.name!r} es x = {comp.x}, fuera del intervalo [0, 1]."
            )
        if comp.u_x < 0.0:
            raise ValueError(
                f"La incertidumbre u(x) de {comp.name!r} es {comp.u_x}, no puede ser negativa."
            )

    total = sum(c.x for c in inputs.composition)
    if abs(total - 1.0) > _TOL_SUM:
        raise ValueError(
            f"La composición no está normalizada: Σxⱼ = {total:.10f}. "
            f"Debe cumplir |Σxⱼ − 1| ≤ {_TOL_SUM:.0e}. Normalizá la "
            f"composición y volvé a intentar (la norma no permite "
            f"auto-normalización porque oculta errores de medición)."
        )

    T_c = inputs.combustion_reference.T_celsius
    if T_c not in _SUPPORTED_T_COMBUSTION:
        raise ValueError(
            f"Temperatura de combustión T_c = {T_c} °C no soportada. "
            f"La Tabla 3 de ISO 6976:2016 tabula valores solo a "
            f"{list(_SUPPORTED_T_COMBUSTION)} °C. Elegí una de esas."
        )
    T_m = inputs.metering_reference.T_celsius
    if T_m not in _SUPPORTED_T_METERING:
        raise ValueError(
            f"Temperatura de medición T_m = {T_m} °C no soportada. "
            f"La Tabla 2 y los factores Z_air del Annex A se tabulan "
            f"solo a {list(_SUPPORTED_T_METERING)} °C."
        )

    if inputs.combustion_reference.P_kPa <= 0.0:
        raise ValueError(
            f"La presión de referencia de combustión debe ser positiva. "
            f"Recibí P_c = {inputs.combustion_reference.P_kPa} kPa."
        )
    if inputs.metering_reference.P_kPa <= 0.0:
        raise ValueError(
            f"La presión de referencia de medición debe ser positiva. "
            f"Recibí P_m = {inputs.metering_reference.P_kPa} kPa."
        )

    if inputs.correlation_matrix not in ("identity", "normalization"):
        raise ValueError(
            f"correlation_matrix debe ser 'identity' o 'normalization'. "
            f"Recibí: {inputs.correlation_matrix!r}."
        )
    if inputs.uncertainty_basis not in ("normalized", "raw"):
        raise ValueError(
            f"uncertainty_basis debe ser 'normalized' (las u de x, ya normalizada) o "
            f"'raw' (las del análisis crudo). Recibí: {inputs.uncertainty_basis!r}."
        )


# ---------------------------------------------------------------------
# Cómputo de los 14 valores centrales
# ---------------------------------------------------------------------


def _compute_all_quantities(p: dict[str, Any]) -> dict[str, float]:
    """Calcula los 14 outputs dado un dict de parámetros completo."""
    x: np.ndarray = p["x"]
    a: np.ndarray = p["a"]
    b: np.ndarray = p["b"]
    c: np.ndarray = p["c"]
    d: np.ndarray = p["d"]
    e: np.ndarray = p["e"]
    is_He: np.ndarray = p["is_He"]
    is_Ne: np.ndarray = p["is_Ne"]
    is_Ar: np.ndarray = p["is_Ar"]
    A_C: float = p["A_C"]
    A_H: float = p["A_H"]
    A_N: float = p["A_N"]
    A_O: float = p["A_O"]
    A_S: float = p["A_S"]
    A_He: float = p["A_He"]
    A_Ne: float = p["A_Ne"]
    A_Ar: float = p["A_Ar"]
    s_vec: np.ndarray = p["s"]
    Hc_vec: np.ndarray = p["Hc"]
    L0: float = p["L0"]
    M_air: float = p["M_air"]
    Z_air: float = p["Z_air"]
    R: float = p["R"]
    T_K: float = p["T_K"]
    P_Pa: float = p["P_Pa"]

    # Masa molar por componente.
    M_per = a * A_C + b * A_H + c * A_N + d * A_O + e * A_S
    if np.any(is_He):
        M_per = np.where(is_He, A_He, M_per)
    if np.any(is_Ne):
        M_per = np.where(is_Ne, A_Ne, M_per)
    if np.any(is_Ar):
        M_per = np.where(is_Ar, A_Ar, M_per)

    M = float(np.sum(x * M_per))
    s_total = float(np.sum(x * s_vec))
    Z = 1.0 - s_total * s_total
    V_m = Z * R * T_K / P_Pa

    Hc_G_molar = float(np.sum(x * Hc_vec))
    sum_xb_2 = float(np.sum(x * b) / 2.0)
    Hc_N_molar = Hc_G_molar - sum_xb_2 * L0

    Hm_G = Hc_G_molar / M
    Hm_N = Hc_N_molar / M

    Hv_G = Hc_G_molar * 1.0e-3 / V_m
    Hv_N = Hc_N_molar * 1.0e-3 / V_m

    rho = M * 1.0e-3 / V_m
    d_rel = (M * Z_air) / (M_air * Z)
    sqrt_d = float(np.sqrt(d_rel))
    W_G = Hv_G / sqrt_d
    W_N = Hv_N / sqrt_d

    return {
        "M": M,
        "s": s_total,
        "Z": Z,
        "V_m": V_m,
        "Hc_G_molar": Hc_G_molar,
        "Hc_N_molar": Hc_N_molar,
        "Hm_G": Hm_G,
        "Hm_N": Hm_N,
        "Hv_G": Hv_G,
        "Hv_N": Hv_N,
        "rho": rho,
        "d": d_rel,
        "W_G": W_G,
        "W_N": W_N,
    }


# ---------------------------------------------------------------------
# Propagación numérica de incertidumbre
# ---------------------------------------------------------------------


def _step_scalar(value: float) -> float:
    return max(_NUMERICAL_H_REL * abs(value), _NUMERICAL_H_ABS)


def _grad_vec(
    f: Callable[[dict[str, Any]], float],
    params: dict[str, Any],
    key: str,
) -> np.ndarray:
    """Gradient de ``f`` respecto a un vector ``params[key]`` por diferencia central."""
    base: np.ndarray = params[key]
    grad = np.zeros_like(base, dtype=float)
    for i in range(base.size):
        h = _step_scalar(float(base[i]))
        x_plus = base.copy()
        x_plus[i] = base[i] + h
        x_minus = base.copy()
        x_minus[i] = base[i] - h
        params_plus = {**params, key: x_plus}
        params_minus = {**params, key: x_minus}
        grad[i] = (f(params_plus) - f(params_minus)) / (2.0 * h)
    return grad


def _grad_scalar(
    f: Callable[[dict[str, Any]], float],
    params: dict[str, Any],
    key: str,
) -> float:
    val = float(params[key])
    h = _step_scalar(val)
    return (f({**params, key: val + h}) - f({**params, key: val - h})) / (2.0 * h)


def _raw_variances(x: np.ndarray, u_x: np.ndarray) -> np.ndarray:
    """Las u²(yⱼ) ≥ 0 del análisis crudo que mejor reproducen las u²(xⱼ) normalizadas.

    La diagonal de la covarianza de la normalización es lineal en las u²(y):
    u²(xᵢ) = (1 − 2xᵢ)·u²(yᵢ) + xᵢ²·Σₖ u²(yₖ). Se resuelve con mínimos cuadrados no
    negativos (``scipy.optimize.nnls``, el algoritmo de Lawson y Hanson, 1974),
    escalado por la mayor u² para que el problema quede bien condicionado.
    """
    u2 = u_x**2
    scale = float(u2.max(initial=0.0))
    if scale <= 0.0:
        return np.zeros_like(u2)
    a = np.diag(1.0 - 2.0 * x) + np.outer(x**2, np.ones_like(x))
    v, _residual = nnls(a, u2 / scale)
    return v * scale


def _normalization_covariance(x: np.ndarray, raw_u2: np.ndarray) -> np.ndarray:
    """Covarianza de xᵢ = yᵢ/Σyₖ con las yₖ independientes, a primer orden (Σy = 1).

    Con J = ∂x/∂y (Jᵢⱼ = δᵢⱼ − xᵢ), Cov(x) = J·diag(u²(y))·Jᵀ:

        u(xᵢ, xⱼ) = u²(yᵢ)·δᵢⱼ − xⱼ·u²(yᵢ) − xᵢ·u²(yⱼ) + xᵢ·xⱼ·Σₖ u²(yₖ)

    Cada fila suma 0 (Σx = 1 no tiene incertidumbre) y la matriz es semidefinida
    positiva por construcción. Es la propagación que ISO 6976:2016 (Anexo D) pide con
    ISO 14912:2003.
    """
    jac = np.eye(x.size) - np.outer(x, np.ones_like(x))
    cov = jac @ np.diag(raw_u2) @ jac.T
    return 0.5 * (cov + cov.T)  # simétrica también con el redondeo


def _normalization_message(
    names: tuple[str, ...],
    x: np.ndarray,
    u_x: np.ndarray,
    u_fit: np.ndarray,
    worst: int,
) -> str:
    """Por qué unas u(x) no pueden salir de normalizar una composición."""
    if len(names) == 1:
        return (
            "Con la matriz de normalización y un solo componente, la fracción normalizada "
            f"vale 1 exacto: u(x) tiene que ser 0 y cargaste {u_x[0]:.3g}. Poné u(x) = 0 o "
            "usá la matriz identidad."
        )
    if len(names) == 2:
        return (
            "Con la matriz de normalización y dos componentes, x₁ + x₂ = 1 obliga a que las "
            f"dos u(x) sean iguales: cargaste {u_x[0]:.3g} para {names[0]!r} y "
            f"{u_x[1]:.3g} para {names[1]!r}. Igualalas o usá la matriz identidad."
        )
    return (
        "Con la matriz de normalización, las u(x) tienen que ser las de una composición "
        "normalizada (xᵢ = yᵢ/Σyₖ): u²(xᵢ) = (1 − xᵢ)²·u²(yᵢ) + xᵢ²·Σₖ≠ᵢ u²(yₖ), así "
        "que ninguna fracción puede tener menos incertidumbre que xᵢ·√(Σₖ≠ᵢ u²(yₖ)): "
        "al normalizar, la de los demás componentes se le pasa. Con las que cargaste no "
        f"hay incertidumbres del análisis crudo que las reproduzcan: a {names[worst]!r} "
        f"(x = {x[worst]:.6g}) le tocaría u(x) ≈ {u_fit[worst]:.2g} y cargaste "
        f"{u_x[worst]:.2g}. Revisá las u(x), cargá las del análisis crudo (antes de "
        "normalizar) o usá la matriz identidad, que no supone nada sobre cómo se "
        "obtuvieron."
    )


def normalization_fit(
    composition: Sequence[GasComponent], basis: UncertaintyBasis = "normalized"
) -> NormalizationFit:
    """Las incertidumbres del análisis crudo y de la composición normalizada.

    ISO 6976:2016 (Anexo D) calcula la matriz de normalización con ISO 14912:2003 a
    partir de las incertidumbres del análisis crudo: las fracciones medidas yⱼ
    (independientes) se normalizan, xⱼ = yⱼ/Σyₖ, y se propaga a primer orden
    (:func:`_normalization_covariance`).

    - ``basis="raw"``: las u de la composición son las u(yⱼ) del análisis crudo y las
      u(xⱼ) normalizadas salen de la diagonal de la covarianza.
    - ``basis="normalized"``: son las u(xⱼ) ya normalizadas (el Anexo D publica solo
      esas) y las u(yⱼ) salen de invertir la diagonal,

          u²(xᵢ) = (1 − xᵢ)²·u²(yᵢ) + xᵢ²·Σ_{k≠i} u²(yₖ),

      con u²(y) ≥ 0. Con el ejemplo D.4 (11 componentes) la reconstrucción es exacta
      y las u de los casos D.4.3.2 y D.4.4.2 salen dentro de las tolerancias de la
      norma.

    Raises
    ------
    ValueError
        Con ``basis="normalized"``, si ninguna u(y) ≥ 0 reproduce las u(x) dadas
        (dentro de 1 % más 10⁻³ de la mayor): por ejemplo, el componente principal con
        menos incertidumbre que la que le pasan los demás al normalizar, o dos
        componentes con u(x) distintas.
    """
    names = tuple(c.name for c in composition)
    x = np.array([c.x for c in composition], dtype=float)
    u_given = np.array([c.u_x for c in composition], dtype=float)
    raw_u2 = u_given**2 if basis == "raw" else _raw_variances(x, u_given)
    cov = _normalization_covariance(x, raw_u2)
    u_norm = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if basis == "normalized":
        tol = _NORM_TOL_REL * u_given + _NORM_TOL_ABS * float(u_given.max(initial=0.0))
        gap = np.abs(u_norm - u_given)
        if np.any(gap > tol):
            worst = int(np.argmax(gap / np.where(tol > 0.0, tol, 1.0)))
            raise ValueError(_normalization_message(names, x, u_given, u_norm, worst))
    sd = np.where(u_norm > 0.0, u_norm, 1.0)
    corr = cov / np.outer(sd, sd)
    corr[u_norm == 0.0, :] = 0.0
    corr[:, u_norm == 0.0] = 0.0
    np.fill_diagonal(corr, np.where(u_norm > 0.0, 1.0, 0.0))
    return NormalizationFit(
        names=names,
        x=tuple(float(v) for v in x),
        basis=basis,
        u_given=tuple(float(v) for v in u_given),
        u_raw=tuple(float(v) for v in np.sqrt(raw_u2)),
        u_normalized=tuple(float(v) for v in u_norm),
        correlation=tuple(tuple(float(v) for v in row) for row in corr),
    )


def _build_composition_covariance(
    u_x: np.ndarray,
    x: np.ndarray,
    correlation_matrix: CorrelationMatrix,
    fit: NormalizationFit | None = None,
) -> np.ndarray:
    """Devuelve la matriz de covarianza n×n de la composición.

    ``identity``: mediciones independientes, ``Cov_ij = u(xᵢ)²·δᵢⱼ``.
    ``normalization``: la de normalizar el análisis crudo, con las u(yⱼ) de
    :func:`normalization_fit` (``fit``; si no viene, se reconstruyen de las u(x) como
    si fueran las normalizadas, sin nombres).
    """
    if correlation_matrix == "identity":
        return np.diag(u_x**2)
    if fit is None:
        fit = normalization_fit(
            [
                GasComponent(f"componente {j + 1}", float(xj), float(uj))
                for j, (xj, uj) in enumerate(zip(x, u_x, strict=True))
            ]
        )
    return _normalization_covariance(x, np.array(fit.u_raw) ** 2)


# Lista de "fuentes" de incertidumbre para iteración determinista.
_SCALAR_UNC_SOURCES: tuple[tuple[str, str], ...] = (
    ("A_C", "u_A_C"),
    ("A_H", "u_A_H"),
    ("A_N", "u_A_N"),
    ("A_O", "u_A_O"),
    ("A_S", "u_A_S"),
    ("A_He", "u_A_He"),
    ("A_Ne", "u_A_Ne"),
    ("A_Ar", "u_A_Ar"),
    ("L0", "u_L0"),
    ("M_air", "u_M_air"),
    ("Z_air", "u_Z_air"),
)

_VEC_UNC_SOURCES: tuple[tuple[str, str], ...] = (
    ("s", "u_s"),
    ("Hc", "u_Hc"),
)


def _propagate(
    qty_name: str,
    params: dict[str, Any],
    u_dict: dict[str, Any],
    cov_x: np.ndarray,
) -> float:
    """Varianza propagada de un output, combinando todas las fuentes."""

    def f(p: dict[str, Any]) -> float:
        return _compute_all_quantities(p)[qty_name]

    # Composición (covarianza completa, identity o normalization).
    grad_x = _grad_vec(f, params, "x")
    var = float(grad_x @ cov_x @ grad_x)

    # Otros vectores: independientes entre componentes (diagonal).
    for key, u_key in _VEC_UNC_SOURCES:
        grad = _grad_vec(f, params, key)
        u_vec: np.ndarray = u_dict[u_key]
        var += float(np.sum((grad * u_vec) ** 2))

    # Escalares.
    for key, u_key in _SCALAR_UNC_SOURCES:
        u_val = float(u_dict[u_key])
        if u_val == 0.0:
            continue
        grad = _grad_scalar(f, params, key)
        var += (grad * u_val) ** 2

    # Pequeñas perturbaciones numéricas pueden dejar var ligeramente <0.
    if var < 0.0 and var > -1.0e-14:
        var = 0.0
    return float(np.sqrt(var))


# ---------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------


def calculate(
    inputs: ISO6976Inputs,
    *,
    tables: ISO6976Tables | None = None,
) -> ISO6976Result:
    """Calcula los 14 outputs de ISO 6976:2016 con propagación de incertidumbre.

    Parámetros
    ----------
    inputs : ISO6976Inputs
        Composición, referencias y matriz de correlación.
    tables : ISO6976Tables, opcional
        Tablas pre-cargadas. Si ``None``, se cargan desde ``data/``.
    """
    tabs = tables if tables is not None else load_tables()
    _validate_inputs(inputs, tabs)

    n = len(inputs.composition)
    names = [c.name for c in inputs.composition]
    x = np.array([c.x for c in inputs.composition], dtype=float)
    u_x = np.array([c.u_x for c in inputs.composition], dtype=float)

    T_c = inputs.combustion_reference.T_celsius
    T_m = inputs.metering_reference.T_celsius
    P_m_Pa = inputs.metering_reference.P_kPa * 1.0e3
    # IMPORTANTE: la etiqueta "15.55" es 60 °F = 288.7056 K, no 288.70 K.
    T_m_K = _T_CELSIUS_TO_KELVIN[T_m]

    a_vec = np.array([tabs.components[nm].a for nm in names], dtype=float)
    b_vec = np.array([tabs.components[nm].b for nm in names], dtype=float)
    c_vec = np.array([tabs.components[nm].c for nm in names], dtype=float)
    d_vec = np.array([tabs.components[nm].d for nm in names], dtype=float)
    e_vec = np.array([tabs.components[nm].e for nm in names], dtype=float)

    is_He = np.array([nm == "helium" for nm in names], dtype=bool)
    is_Ne = np.array([nm == "neon" for nm in names], dtype=bool)
    is_Ar = np.array([nm == "argon" for nm in names], dtype=bool)

    s_vec = np.array([tabs.summation_factors[nm][T_m] for nm in names], dtype=float)
    u_s_vec = np.array([tabs.u_summation_factors[nm] for nm in names], dtype=float)

    Hc_vec = np.array([tabs.calorific_values[nm][T_c] for nm in names], dtype=float)
    u_Hc_vec = np.array([tabs.u_calorific_values[nm] for nm in names], dtype=float)

    L0 = tabs.L0_water_by_T.get(T_c, 0.0)

    params: dict[str, Any] = {
        "x": x,
        "a": a_vec,
        "b": b_vec,
        "c": c_vec,
        "d": d_vec,
        "e": e_vec,
        "is_He": is_He,
        "is_Ne": is_Ne,
        "is_Ar": is_Ar,
        "A_C": tabs.atomic_weights.get("C", 0.0),
        "A_H": tabs.atomic_weights.get("H", 0.0),
        "A_N": tabs.atomic_weights.get("N", 0.0),
        "A_O": tabs.atomic_weights.get("O", 0.0),
        "A_S": tabs.atomic_weights.get("S", 0.0),
        "A_He": tabs.atomic_weights.get("He", 4.002602),
        "A_Ne": tabs.atomic_weights.get("Ne", 20.1797),
        "A_Ar": tabs.atomic_weights.get("Ar", 39.948),
        "s": s_vec,
        "Hc": Hc_vec,
        "L0": L0,
        "M_air": tabs.M_air_kg_per_kmol,
        "Z_air": tabs.Z_air_by_T[T_m],
        "R": tabs.R_J_per_mol_K,
        "T_K": T_m_K,
        "P_Pa": P_m_Pa,
    }

    u_dict: dict[str, Any] = {
        "u_s": u_s_vec,
        "u_Hc": u_Hc_vec,
        "u_A_C": tabs.u_atomic_weights.get("C", 0.0),
        "u_A_H": tabs.u_atomic_weights.get("H", 0.0),
        "u_A_N": tabs.u_atomic_weights.get("N", 0.0),
        "u_A_O": tabs.u_atomic_weights.get("O", 0.0),
        "u_A_S": tabs.u_atomic_weights.get("S", 0.0),
        "u_A_He": tabs.u_atomic_weights.get("He", 0.0),
        "u_A_Ne": tabs.u_atomic_weights.get("Ne", 0.0),
        "u_A_Ar": tabs.u_atomic_weights.get("Ar", 0.0),
        "u_L0": tabs.u_L0_water,
        "u_M_air": tabs.u_M_air_kg_per_kmol,
        "u_Z_air": tabs.u_Z_air,
    }

    fit = (
        normalization_fit(inputs.composition, inputs.uncertainty_basis)
        if inputs.correlation_matrix == "normalization"
        else None
    )
    cov_x = _build_composition_covariance(u_x, x, inputs.correlation_matrix, fit)

    values = _compute_all_quantities(params)

    quantities: dict[str, Quantity] = {}
    for qty in (
        "M",
        "s",
        "Z",
        "V_m",
        "Hc_G_molar",
        "Hc_N_molar",
        "Hm_G",
        "Hm_N",
        "Hv_G",
        "Hv_N",
        "rho",
        "d",
        "W_G",
        "W_N",
    ):
        val = values[qty]
        u_val = _propagate(qty, params, u_dict, cov_x)
        quantities[qty] = Quantity(value=val, u=u_val, U_k2=2.0 * u_val)

    return ISO6976Result(
        inputs=inputs,
        molar_mass_kg_per_kmol=quantities["M"],
        summation_factor=quantities["s"],
        compression_factor=quantities["Z"],
        molar_volume_m3_per_mol=quantities["V_m"],
        Hc_G_molar_kJ_per_mol=quantities["Hc_G_molar"],
        Hc_N_molar_kJ_per_mol=quantities["Hc_N_molar"],
        Hm_G_mass_MJ_per_kg=quantities["Hm_G"],
        Hm_N_mass_MJ_per_kg=quantities["Hm_N"],
        Hv_G_volume_MJ_per_m3=quantities["Hv_G"],
        Hv_N_volume_MJ_per_m3=quantities["Hv_N"],
        density_kg_per_m3=quantities["rho"],
        relative_density=quantities["d"],
        Wobbe_gross_MJ_per_m3=quantities["W_G"],
        Wobbe_net_MJ_per_m3=quantities["W_N"],
        steps=_build_steps(inputs, values, n),
        normalization=fit,
    )


# ---------------------------------------------------------------------
# Construcción del bloque didáctico
# ---------------------------------------------------------------------


def _build_steps(inputs: ISO6976Inputs, values: dict[str, float], n: int) -> ISO6976Steps:
    T_c = inputs.combustion_reference.T_celsius
    T_m = inputs.metering_reference.T_celsius

    formula = (
        r"\begin{aligned}"
        r"M &= \sum_j x_j\, M_j \\"
        r"s &= \sum_j x_j\, s_j(T_m) \\"
        r"Z &= 1 - s^2 \\"
        r"V_m &= \dfrac{Z\, R\, T_m}{P_m} \\"
        r"H_{c,G}^{\mathrm{mol}} &= \sum_j x_j\, H_{c,j}(T_c) \\"
        r"H_{c,N}^{\mathrm{mol}} &= H_{c,G}^{\mathrm{mol}} "
        r"- \left(\sum_j x_j\,\dfrac{b_j}{2}\right) L_0(T_c) \\"
        r"H_m &= \dfrac{H_c^{\mathrm{mol}}}{M} \\"
        r"H_v &= \dfrac{H_c^{\mathrm{mol}}}{V_m}\cdot 10^{-3} \\"
        r"\rho &= \dfrac{M\cdot 10^{-3}}{V_m} \\"
        r"d &= \dfrac{M\, Z_{\mathrm{air}}}{M_{\mathrm{air}}\, Z} \\"
        r"W &= \dfrac{H_v}{\sqrt{d}}"
        r"\end{aligned}"
    )

    substituted = (
        r"\begin{aligned}"
        rf"M &= {values['M']:.6f}~\text{{kg/kmol}} \\"
        rf"s &= {values['s']:.6f} \\"
        rf"Z &= {values['Z']:.8f} \\"
        rf"V_m &= {values['V_m']:.8f}~\text{{m}}^3/\text{{mol}} \\"
        rf"H_{{c,G}}^{{\mathrm{{mol}}}} &= {values['Hc_G_molar']:.6f}~\text{{kJ/mol}} \\"
        rf"H_{{c,N}}^{{\mathrm{{mol}}}} &= {values['Hc_N_molar']:.6f}~\text{{kJ/mol}} \\"
        rf"H_{{m,G}} &= {values['Hm_G']:.6f}~\text{{MJ/kg}} \\"
        rf"H_{{v,G}} &= {values['Hv_G']:.6f}~\text{{MJ/m}}^3 \\"
        rf"\rho &= {values['rho']:.6f}~\text{{kg/m}}^3 \\"
        rf"d &= {values['d']:.6f} \\"
        rf"W_G &= {values['W_G']:.6f}~\text{{MJ/m}}^3"
        r"\end{aligned}"
    )

    narrative = (
        f"Cálculo ISO 6976:2016 con {n} componente(s). Referencia de "
        f"combustión: T_c = {T_c:g} °C, P_c = "
        f"{inputs.combustion_reference.P_kPa:g} kPa. Referencia de "
        f"medición: T_m = {T_m:g} °C, P_m = "
        f"{inputs.metering_reference.P_kPa:g} kPa. Matriz de correlación "
        f"de la composición: {_CORRELATION_NAMES[inputs.correlation_matrix]}. "
        f"(1) Masa molar de la mezcla M = Σⱼ xⱼ·Mⱼ = "
        f"{values['M']:.6f} kg/kmol. "
        f"(2) Factor de sumación s = Σⱼ xⱼ·sⱼ(T_m) = "
        f"{values['s']:.6f}. "
        f"(3) Factor de compresibilidad Z = 1 − s² = "
        f"{values['Z']:.8f} (cerca de 1 ⇒ gas casi ideal). "
        f"(4) Volumen molar V_m = Z·R·T_m/P_m = "
        f"{values['V_m']:.8f} m³/mol. "
        f"(5) Calor de combustión bruto (molar) H_c,G = Σⱼ xⱼ·Hc(j,T_c) = "
        f"{values['Hc_G_molar']:.6f} kJ/mol. "
        f"(6) Calor de combustión neto (molar) H_c,N = H_c,G − "
        f"(Σⱼ xⱼ·bⱼ/2)·L₀(T_c) = {values['Hc_N_molar']:.6f} kJ/mol "
        f"(resta el calor latente del H₂O formado por la combustión). "
        f"(7) Volumétrico bruto H_v,G = H_c,G·10⁻³/V_m = "
        f"{values['Hv_G']:.6f} MJ/m³. "
        f"(8) Densidad ρ = M·10⁻³/V_m = {values['rho']:.6f} kg/m³. "
        f"(9) Densidad relativa al aire d = M·Z_air/(M_air·Z) = "
        f"{values['d']:.6f}. "
        f"(10) Índice de Wobbe bruto W_G = H_v,G/√d = "
        f"{values['W_G']:.6f} MJ/m³. "
        f"(11) {_UNCERTAINTY_NARRATIVE[_narrative_key(inputs)]}"
    )
    return ISO6976Steps(
        formula_latex=formula,
        substituted_latex=substituted,
        narrative_es=narrative,
        uncertainty_latex=UNCERTAINTY_LATEX[inputs.correlation_matrix],
    )


_CORRELATION_NAMES: dict[str, str] = {
    "identity": "identidad (las u(xⱼ) se tratan como independientes)",
    "normalization": (
        "normalización (la composición se normalizó a Σx = 1 y las fracciones quedan "
        "correlacionadas)"
    ),
}

_UNCERTAINTY_NARRATIVE: dict[str, str] = {
    "identity": (
        "Incertidumbre: cada salida Y suma los aportes cⱼ²·u²(xⱼ) de la composición, con "
        "cⱼ = ∂Y/∂xⱼ, y los de los datos de las tablas (sⱼ, H_c,j, las masas atómicas, "
        "L₀, M_aire y Z_aire). Tratar las u(xⱼ) como independientes ignora que la "
        "composición suma 1: la norma (§D.4.3.2) dice que es una sobre-estimación segura."
    ),
    "normalized": (
        "Incertidumbre: la composición se normalizó (xⱼ = yⱼ/Σyₖ), así que si una fracción "
        "sube, las demás bajan. La covarianza u(xᵢ, xⱼ) sale de las u(yⱼ) del análisis "
        "crudo, que se reconstruyen de las u(xⱼ) dadas, y cada salida suma Σᵢ Σⱼ "
        "cᵢ·cⱼ·u(xᵢ, xⱼ) más los aportes de los datos de las tablas. Las correlaciones "
        "negativas se compensan y la incertidumbre baja respecto de la matriz identidad."
    ),
    "raw": (
        "Incertidumbre: las u cargadas son las u(yⱼ) del análisis crudo y la composición "
        "se normalizó (xⱼ = yⱼ/Σyₖ), así que si una fracción sube, las demás bajan. La "
        "covarianza u(xᵢ, xⱼ) sale de propagar la normalización y cada salida suma Σᵢ Σⱼ "
        "cᵢ·cⱼ·u(xᵢ, xⱼ) más los aportes de los datos de las tablas."
    ),
}


def _narrative_key(inputs: ISO6976Inputs) -> str:
    if inputs.correlation_matrix == "identity":
        return "identity"
    return inputs.uncertainty_basis


#: La propagación de la incertidumbre con cada matriz (cada elemento, un ``st.latex``).
UNCERTAINTY_LATEX: dict[str, tuple[str, ...]] = {
    "identity": (
        r"\begin{aligned}u_c^2(Y) &= \sum_j c_j^2\,u^2(x_j) \\"
        r"&\quad + \sum_p c_p^2\,u^2(p) \\"
        r"c_j &= \frac{\partial Y}{\partial x_j}\end{aligned}",
    ),
    "normalization": (
        r"\begin{aligned}u_c^2(Y) &= \sum_i \sum_j c_i\,c_j\,u(x_i, x_j) \\"
        r"&\quad + \sum_p c_p^2\,u^2(p)\end{aligned}",
        r"\begin{aligned}u(x_i, x_j) &= u^2(y_i)\,\delta_{ij} \\"
        r"&\quad - x_j\,u^2(y_i) - x_i\,u^2(y_j) \\"
        r"&\quad + x_i\,x_j \sum_k u^2(y_k)\end{aligned}",
        r"\begin{aligned}u^2(x_i) &= (1 - x_i)^2\,u^2(y_i) \\"
        r"&\quad + x_i^2 \sum_{k \neq i} u^2(y_k)\end{aligned}",
    ),
}


# ---------------------------------------------------------------------
# Notas sobre la matriz de correlación de normalización
# ---------------------------------------------------------------------
#
# Fase 2.3 dejó ``correlation_matrix="normalization"`` pendiente: la norma calcula
# la matriz con ISO 14912:2003 a partir del análisis crudo, y las dos derivaciones
# que se probaron con las u(x) publicadas no reproducían el Anexo D (caso D.4.3.2):
#
# (A) Proyección sobre Σx = 1: Cov = diag(u²) − (u²)·(u²)ᵀ/Σu² → u_Hv_G = 0.016971 y
#     u_ρ = 0.000221 (esperado 0.016316 y 0.000277).
# (B) Renormalización con las u(x) dadas como si fueran las crudas:
#     Cov = diag(u²) − u²·xᵀ − x·(u²)ᵀ + x·xᵀ·Σu² → u_Hv_G = 0.016120, u_ρ = 0.000269.
#
# (B) es la propagación correcta, pero de las u(y) del análisis crudo, no de las
# u(x) ya normalizadas: la diagonal de (B) no devuelve las u(x) (al normalizar, a
# cada fracción se le suma la incertidumbre de las demás). La 0.30.0 reconstruye
# las u(y) invirtiendo esa diagonal con mínimos cuadrados no negativos
# (:func:`normalization_fit`): con las u(x) del ejemplo D.4 la reconstrucción es
# exacta (error relativo 6·10⁻¹⁵) y los casos D.4.3.2 y D.4.4.2 salen dentro de las
# tolerancias de la norma (u_Hv_G = 0.0163122 contra 0.016316; u_ρ = 0.0002762
# contra 0.000277; las diferencias son del orden del redondeo de los datos
# publicados, que tienen una o dos cifras).
#
# No todas las u(x) pueden salir de una normalización: ninguna fracción puede tener
# menos incertidumbre que xᵢ·√(Σ_{k≠i} u²(yₖ)), con dos componentes las dos u(x)
# son iguales y con uno, u(x) = 0. Con u(x) inventadas, la mayoría de las
# composiciones no cumple (en un ensayo con 3000 composiciones al azar, el 83 %):
# ``normalization_fit`` lo explica en vez de calcular con una matriz que no
# corresponde. La identidad sigue siendo la opción por defecto y, como dice la
# propia norma (§D.4.3.2), da una sobre-estimación segura (*safe overestimate*).


# ---------------------------------------------------------------------
# Exportación (Fase 1.7)
# ---------------------------------------------------------------------

# (atributo del resultado, símbolo, unidad) — mismas unidades que la página.
_EXPORT_INTERMEDIATES: tuple[tuple[str, str, str], ...] = (
    ("molar_mass_kg_per_kmol", "M (masa molar)", "kg/kmol"),
    ("summation_factor", "s (factor de sumación)", "-"),
    ("compression_factor", "Z (compresibilidad)", "-"),
    ("molar_volume_m3_per_mol", "V_m (volumen molar)", "m³/mol"),
)
_EXPORT_RESULTS: tuple[tuple[str, str, str], ...] = (
    ("Hc_G_molar_kJ_per_mol", "H_c,G (bruto, molar)", "kJ/mol"),
    ("Hc_N_molar_kJ_per_mol", "H_c,N (neto, molar)", "kJ/mol"),
    ("Hm_G_mass_MJ_per_kg", "H_m,G (bruto, másico)", "MJ/kg"),
    ("Hm_N_mass_MJ_per_kg", "H_m,N (neto, másico)", "MJ/kg"),
    ("Hv_G_volume_MJ_per_m3", "H_v,G (bruto, volumétrico)", "MJ/m³"),
    ("Hv_N_volume_MJ_per_m3", "H_v,N (neto, volumétrico)", "MJ/m³"),
    ("density_kg_per_m3", "ρ (densidad)", "kg/m³"),
    ("relative_density", "d (densidad relativa al aire)", "-"),
    ("Wobbe_gross_MJ_per_m3", "W_G (Wobbe bruto)", "MJ/m³"),
    ("Wobbe_net_MJ_per_m3", "W_N (Wobbe neto)", "MJ/m³"),
)


def iso6976_to_dict(result: ISO6976Result) -> dict[str, Any]:
    """Resultado de ISO 6976:2016 serializable a JSON (unidades de la norma).

    Cada magnitud lleva su valor, la incertidumbre estándar ``u`` (k = 1)
    y la expandida ``U`` (k = 2). Con la matriz de normalización suma las u(y) del
    análisis crudo que se reconstruyeron.
    """

    def quantity(attr: str, symbol: str, unit: str) -> dict[str, Any]:
        q: Quantity = getattr(result, attr)
        return {"magnitud": symbol, "valor": q.value, "u_k1": q.u, "U_k2": q.U_k2, "unidad": unit}

    inp = result.inputs
    return {
        "norma": "ISO 6976:2016",
        "composicion": [{"componente": c.name, "x": c.x, "u_x": c.u_x} for c in inp.composition],
        "referencia_combustion": {
            "T_C": inp.combustion_reference.T_celsius,
            "P_kPa": inp.combustion_reference.P_kPa,
        },
        "referencia_medicion": {
            "T_C": inp.metering_reference.T_celsius,
            "P_kPa": inp.metering_reference.P_kPa,
        },
        "matriz_de_correlacion": inp.correlation_matrix,
        **(
            {}
            if result.normalization is None
            else {
                "base_de_las_u": result.normalization.basis,
                "analisis_crudo": [
                    {"componente": name, "x": x, "u_y_crudo": u_y, "u_x_normalizada": u_x}
                    for name, x, u_y, u_x in zip(
                        result.normalization.names,
                        result.normalization.x,
                        result.normalization.u_raw,
                        result.normalization.u_normalized,
                        strict=True,
                    )
                ],
            }
        ),
        "intermedios": [quantity(*spec) for spec in _EXPORT_INTERMEDIATES],
        "resultados": [quantity(*spec) for spec in _EXPORT_RESULTS],
    }
