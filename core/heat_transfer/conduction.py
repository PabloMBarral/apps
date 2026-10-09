"""Conducción unidimensional en régimen permanente: la red de resistencias (Fase 8.1).

Cengel y Ghajar (2015), cap. 3; Incropera et al. (2007), §3.1 a §3.4.

Cada elemento entre dos temperaturas es una resistencia térmica, R = ΔT/Q̇:

- pared plana: R = L/(k·A);
- cilindro: R = ln(r₂/r₁)/(2π·k·L);
- esfera: R = (r₂ − r₁)/(4π·k·r₁·r₂);
- convección en una superficie: R = 1/(h·A);
- contacto entre dos capas: R = R''_c/A.

En serie se suman y Q̇ = (T₁ − T₂)/ΣR. Las partes de una capa plana compuesta
(el ladrillo y el revoque de una hilada) van en paralelo con las caras
isotérmicas, la aproximación de Cengel y Ghajar (§3-3): 1/R = Σ 1/R_i.

Un borde puede ser un fluido (T∞ y h; el h puede incluir la radiación), una
superficie a temperatura dada o un calor dado (el de un alambre que se
calienta por efecto Joule). El radio crítico de aislación (§3-5) es
r_cr = k/h en un cilindro y 2k/h en una esfera.

Todo en SI. Lado 1 es la izquierda de una pared o el interior de un caño o de
una esfera; Q̇ > 0 va del lado 1 al lado 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

from core.units_system import UnitSystem, format_quantity

__all__ = [
    "CONDUCTION_EXAMPLES",
    "CONDUCTION_EXAMPLE_NOTES",
    "GEOMETRY_NAMES",
    "INSULATION_K_MAX",
    "Boundary",
    "BoundaryKind",
    "ConductionInputs",
    "ConductionResult",
    "Geometry",
    "Layer",
    "ParallelPart",
    "PartFlow",
    "ProfileSegment",
    "Resistance",
    "ResistanceKind",
    "SweepPoint",
    "conduction_notes",
    "conduction_to_dict",
    "default_insulation_radii",
    "insulation_sweep",
    "solve_conduction",
    "temperature_profile",
]

Geometry = Literal["plane", "cylinder", "sphere"]
BoundaryKind = Literal["fluid", "surface", "heat"]
ResistanceKind = Literal["convección", "conducción", "contacto"]

GEOMETRY_NAMES: dict[Geometry, str] = {
    "plane": "Pared plana",
    "cylinder": "Cilindro (caño, alambre)",
    "sphere": "Esfera (tanque)",
}

#: Fracciones de una capa compuesta: tolerancia para la suma (los widgets redondean).
_FRACTION_TOL = 1e-3
#: Hasta esta k [W/(m·K)] la capa exterior se trata como aislación (radio crítico).
INSULATION_K_MAX = 1.0


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _len(x_m: float) -> str:
    """Un radio o espesor en mm (o en m desde 1 m), con coma decimal."""
    if x_m >= 1.0:
        return f"{_num(x_m)} m"
    return f"{_num(1000.0 * x_m)} mm"


# ---------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelPart:
    """Una parte de una capa plana compuesta, con su fracción del área."""

    name: str
    k_W_per_mK: float
    area_fraction: float


@dataclass(frozen=True)
class Layer:
    """Una capa: espesor y conductividad, o partes en paralelo (solo en la pared plana).

    ``contact_m2K_per_W`` es la resistencia de contacto por unidad de área, R''_c,
    entre esta capa y la siguiente (Cengel y Ghajar, §3-2).
    """

    name: str
    thickness_m: float
    k_W_per_mK: float = 0.0
    parts: tuple[ParallelPart, ...] = ()
    contact_m2K_per_W: float = 0.0

    @property
    def is_composite(self) -> bool:
        return bool(self.parts)

    @property
    def k_effective(self) -> float:
        """La k de una capa compuesta con las caras isotérmicas: Σ f_i·k_i."""
        if not self.parts:
            return self.k_W_per_mK
        return sum(p.area_fraction * p.k_W_per_mK for p in self.parts)


@dataclass(frozen=True)
class Boundary:
    """Un borde: un fluido (T∞ y h), una superficie (T) o un calor dado (Q̇).

    Con ``kind="heat"``, ``Q_W`` es el calor que entra al sólido por ese lado.
    """

    kind: BoundaryKind
    T_K: float | None = None
    h_W_per_m2K: float | None = None
    Q_W: float | None = None


@dataclass(frozen=True)
class ConductionInputs:
    """Una pared plana, un cilindro o una esfera de una o más capas.

    ``area_m2`` es el área de la pared plana; ``length_m`` el largo del
    cilindro; ``r_inner_m`` el radio interior de la primera capa (cilindro y
    esfera).
    """

    geometry: Geometry
    layers: tuple[Layer, ...]
    inner: Boundary
    outer: Boundary
    area_m2: float = 1.0
    length_m: float = 1.0
    r_inner_m: float = 0.01


@dataclass(frozen=True)
class PartFlow:
    """El calor por una de las partes en paralelo de una capa compuesta."""

    name: str
    R_K_per_W: float
    Q_W: float


@dataclass(frozen=True)
class Resistance:
    """Una resistencia de la red, con las temperaturas y posiciones de sus extremos.

    Las de convección y contacto no tienen espesor (x_from = x_to).
    """

    label: str
    kind: ResistanceKind
    R_K_per_W: float
    T_from_K: float
    T_to_K: float
    x_from_m: float
    x_to_m: float
    layer: int | None = None
    parts: tuple[PartFlow, ...] = ()

    @property
    def dT_K(self) -> float:
        return self.T_from_K - self.T_to_K


@dataclass(frozen=True)
class ProfileSegment:
    """El perfil de temperaturas dentro de una capa (x o r en m, T en K)."""

    label: str
    layer: int
    x_m: tuple[float, ...]
    T_K: tuple[float, ...]


@dataclass(frozen=True)
class ConductionResult:
    """El resultado: la red de resistencias, Q̇ y las temperaturas."""

    inputs: ConductionInputs
    resistances: tuple[Resistance, ...]
    Q_W: float
    faces_m: tuple[float, ...]

    @property
    def R_total_K_per_W(self) -> float:
        return sum(r.R_K_per_W for r in self.resistances)

    @property
    def T_side1_K(self) -> float:
        """La temperatura del extremo 1 de la red (el fluido o la superficie)."""
        return self.resistances[0].T_from_K

    @property
    def T_side2_K(self) -> float:
        return self.resistances[-1].T_to_K

    @property
    def A_inner_m2(self) -> float:
        return _area(self.inputs, self.faces_m[0])

    @property
    def A_outer_m2(self) -> float:
        return _area(self.inputs, self.faces_m[-1])

    @property
    def U_inner_W_per_m2K(self) -> float:
        """U referido a la cara 1: Q̇ = U·A₁·(T₁ − T₂) (Cengel y Ghajar, §3-1)."""
        return 1.0 / (self.R_total_K_per_W * self.A_inner_m2)

    @property
    def U_outer_W_per_m2K(self) -> float:
        return 1.0 / (self.R_total_K_per_W * self.A_outer_m2)

    @property
    def q_inner_W_per_m2(self) -> float:
        return self.Q_W / self.A_inner_m2

    @property
    def q_outer_W_per_m2(self) -> float:
        return self.Q_W / self.A_outer_m2

    @property
    def Q_per_length_W_per_m(self) -> float | None:
        """Q̇ por metro de caño (cilindro)."""
        if self.inputs.geometry != "cylinder":
            return None
        return self.Q_W / self.inputs.length_m

    @property
    def surface_temperatures_K(self) -> tuple[float, float]:
        """La temperatura de las dos superficies del sólido (cara 1 y cara 2)."""
        cond = [r for r in self.resistances if r.kind != "convección"]
        return cond[0].T_from_K, cond[-1].T_to_K

    @property
    def critical_radius_m(self) -> float | None:
        """r_cr = k/h (cilindro) o 2k/h (esfera) de la capa exterior (Cengel y Ghajar, §3-5)."""
        outer = self.inputs.outer
        if self.inputs.geometry == "plane" or outer.kind != "fluid":
            return None
        k = self.inputs.layers[-1].k_W_per_mK
        h = outer.h_W_per_m2K or 0.0
        return (k if self.inputs.geometry == "cylinder" else 2.0 * k) / h

    def share(self, r: Resistance) -> float:
        """La fracción de la resistencia total (y de la caída de temperatura)."""
        return r.R_K_per_W / self.R_total_K_per_W


# ---------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------


def _area(inputs: ConductionInputs, x: float) -> float:
    """El área de transferencia en la posición x (o r)."""
    if inputs.geometry == "plane":
        return inputs.area_m2
    if inputs.geometry == "cylinder":
        return 2.0 * math.pi * x * inputs.length_m
    return 4.0 * math.pi * x * x


def _faces(inputs: ConductionInputs) -> list[float]:
    x = 0.0 if inputs.geometry == "plane" else inputs.r_inner_m
    faces = [x]
    for layer in inputs.layers:
        x += layer.thickness_m
        faces.append(x)
    return faces


def _layer_R(inputs: ConductionInputs, layer: Layer, x1: float, x2: float) -> float:
    """R de conducción de una capa (Cengel y Ghajar, §3-1 y §3-4)."""
    if inputs.geometry == "plane":
        return layer.thickness_m / (layer.k_effective * inputs.area_m2)
    k = layer.k_W_per_mK
    if inputs.geometry == "cylinder":
        return math.log(x2 / x1) / (2.0 * math.pi * k * inputs.length_m)
    return (x2 - x1) / (4.0 * math.pi * k * x1 * x2)


def _validate(inputs: ConductionInputs) -> None:
    if inputs.geometry not in GEOMETRY_NAMES:
        raise ValueError(f"Geometría desconocida: {inputs.geometry!r}.")
    if not inputs.layers:
        raise ValueError("Hace falta al menos una capa.")
    if inputs.geometry == "plane" and not inputs.area_m2 > 0.0:
        raise ValueError("El área de la pared tiene que ser positiva.")
    if inputs.geometry == "cylinder" and not inputs.length_m > 0.0:
        raise ValueError("El largo del cilindro tiene que ser positivo.")
    if inputs.geometry != "plane" and not inputs.r_inner_m > 0.0:
        raise ValueError(
            "El radio interior tiene que ser positivo: en r = 0 la resistencia del cilindro "
            "o de la esfera es infinita (para un alambre macizo, el radio interior de la "
            "aislación es el del alambre)."
        )
    for i, layer in enumerate(inputs.layers, start=1):
        name = f"La capa {i} ({layer.name})"
        if not layer.thickness_m > 0.0:
            raise ValueError(f"{name} tiene que tener un espesor positivo.")
        if layer.parts:
            if inputs.geometry != "plane":
                raise ValueError(
                    f"{name}: las partes en paralelo solo se resuelven en una pared plana."
                )
            total = sum(p.area_fraction for p in layer.parts)
            if abs(total - 1.0) > _FRACTION_TOL:
                raise ValueError(
                    f"{name}: las fracciones de área de sus partes suman "
                    f"{_num(total, 4)}; tienen que sumar 1."
                )
            for p in layer.parts:
                if not p.k_W_per_mK > 0.0 or not p.area_fraction > 0.0:
                    raise ValueError(
                        f"{name}: cada parte necesita una k y una fracción de área positivas "
                        f"({p.name})."
                    )
        elif not layer.k_W_per_mK > 0.0:
            raise ValueError(f"{name} necesita una conductividad k positiva.")
        if layer.contact_m2K_per_W < 0.0:
            raise ValueError(f"{name}: la resistencia de contacto no puede ser negativa.")
    if inputs.layers[-1].contact_m2K_per_W > 0.0:
        raise ValueError(
            "La última capa no tiene una capa siguiente: la resistencia de contacto va entre "
            "dos capas."
        )
    for side, b in (("interior (lado 1)", inputs.inner), ("exterior (lado 2)", inputs.outer)):
        if b.kind not in ("fluid", "surface", "heat"):
            raise ValueError(f"Borde {side}: tipo desconocido {b.kind!r}.")
        if b.kind in ("fluid", "surface"):
            if b.T_K is None or not b.T_K > 0.0:
                raise ValueError(f"Borde {side}: la temperatura tiene que ser absoluta positiva.")
        if b.kind == "fluid" and not (b.h_W_per_m2K or 0.0) > 0.0:
            raise ValueError(
                f"Borde {side}: el coeficiente de convección h tiene que ser positivo (con h → ∞ "
                "la superficie queda a la temperatura del fluido: usá «superficie»)."
            )
        if b.kind == "heat" and b.Q_W is None:
            raise ValueError(f"Borde {side}: falta el calor que entra.")
    if inputs.inner.kind == "heat" and inputs.outer.kind == "heat":
        raise ValueError(
            "Con calor dado en los dos lados no hay ninguna temperatura de referencia: uno de "
            "los bordes tiene que ser un fluido o una superficie a temperatura dada."
        )


def solve_conduction(inputs: ConductionInputs) -> ConductionResult:
    """Resuelve la red de resistencias en serie (Cengel y Ghajar, §3-1 a §3-4).

    Raises
    ------
    ValueError
        Con un mensaje para el alumno si un dato no tiene sentido (espesor,
        conductividad o h no positivos, fracciones que no suman 1, contacto
        después de la última capa, calor dado en los dos lados).
    """
    _validate(inputs)
    faces = _faces(inputs)
    # (rótulo, tipo, R, x_desde, x_hasta, capa, partes)
    items: list[tuple[str, ResistanceKind, float, float, float, int | None, tuple]] = []
    if inputs.inner.kind == "fluid":
        A = _area(inputs, faces[0])
        items.append(
            (
                "Convección (lado 1)",
                "convección",
                1.0 / (inputs.inner.h_W_per_m2K * A),  # type: ignore[operator]
                faces[0],
                faces[0],
                None,
                (),
            )
        )
    for i, layer in enumerate(inputs.layers):
        x1, x2 = faces[i], faces[i + 1]
        R = _layer_R(inputs, layer, x1, x2)
        parts: tuple = ()
        if layer.parts:
            parts = tuple(
                (p.name, layer.thickness_m / (p.k_W_per_mK * p.area_fraction * inputs.area_m2))
                for p in layer.parts
            )
        items.append((f"Capa {i + 1}: {layer.name}", "conducción", R, x1, x2, i, parts))
        if layer.contact_m2K_per_W > 0.0:
            items.append(
                (
                    f"Contacto {i + 1}–{i + 2}",
                    "contacto",
                    layer.contact_m2K_per_W / _area(inputs, x2),
                    x2,
                    x2,
                    None,
                    (),
                )
            )
    if inputs.outer.kind == "fluid":
        A = _area(inputs, faces[-1])
        items.append(
            (
                "Convección (lado 2)",
                "convección",
                1.0 / (inputs.outer.h_W_per_m2K * A),  # type: ignore[operator]
                faces[-1],
                faces[-1],
                None,
                (),
            )
        )
    R_total = sum(it[2] for it in items)
    # Q̇ y la temperatura de un extremo conocido
    if inputs.inner.kind == "heat":
        Q = float(inputs.inner.Q_W)  # type: ignore[arg-type]
        T_end = float(inputs.outer.T_K)  # type: ignore[arg-type]
        T_start = T_end + Q * R_total
    elif inputs.outer.kind == "heat":
        Q = -float(inputs.outer.Q_W)  # type: ignore[arg-type]
        T_start = float(inputs.inner.T_K)  # type: ignore[arg-type]
    else:
        T_start = float(inputs.inner.T_K)  # type: ignore[arg-type]
        Q = (T_start - float(inputs.outer.T_K)) / R_total  # type: ignore[arg-type]
    resistances: list[Resistance] = []
    T = T_start
    for label, kind, R, x1, x2, layer_i, parts in items:
        T_next = T - Q * R
        flows = tuple(PartFlow(name, Rp, (T - T_next) / Rp) for name, Rp in parts)
        resistances.append(Resistance(label, kind, R, T, T_next, x1, x2, layer_i, flows))
        T = T_next
    return ConductionResult(inputs, tuple(resistances), Q, tuple(faces))


def temperature_profile(result: ConductionResult, n: int = 41) -> list[ProfileSegment]:
    """El perfil dentro de cada capa: recto en la pared, ln r en el cilindro, 1/r en la esfera.

    Incropera et al. (2007), §3.1 y §3.3.
    """
    geo = result.inputs.geometry
    segments: list[ProfileSegment] = []
    for r in result.resistances:
        if r.kind != "conducción" or r.layer is None:
            continue
        x1, x2, T1, T2 = r.x_from_m, r.x_to_m, r.T_from_K, r.T_to_K
        xs = [x1 + (x2 - x1) * j / (n - 1) for j in range(n)]
        if geo == "plane":
            Ts = [T1 + (T2 - T1) * (x - x1) / (x2 - x1) for x in xs]
        elif geo == "cylinder":
            Ts = [T1 + (T2 - T1) * math.log(x / x1) / math.log(x2 / x1) for x in xs]
        else:
            Ts = [T1 + (T2 - T1) * (1.0 / x1 - 1.0 / x) / (1.0 / x1 - 1.0 / x2) for x in xs]
        segments.append(ProfileSegment(r.label, r.layer, tuple(xs), tuple(Ts)))
    return segments


# ---------------------------------------------------------------------
# Radio crítico de aislación
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SweepPoint:
    """Un punto del barrido del espesor de la capa exterior."""

    r_outer_m: float
    thickness_m: float
    Q_W: float
    T_side1_K: float


def insulation_sweep(inputs: ConductionInputs, radii_m: list[float]) -> list[SweepPoint]:
    """Q̇ (o la T del lado 1, si el calor es dato) contra el radio exterior de la última capa.

    Con la temperatura dada en los dos lados, Q̇ tiene un máximo en r_cr; con
    el calor dado (un alambre), la T del lado 1 tiene un mínimo ahí (Cengel y
    Ghajar, §3-5).
    """
    if inputs.geometry == "plane":
        raise ValueError("El radio crítico es de un cilindro o de una esfera.")
    r_last_in = _faces(inputs)[-2]
    points: list[SweepPoint] = []
    for r in radii_m:
        t = r - r_last_in
        if t <= 0.0:
            continue
        layers = (*inputs.layers[:-1], replace(inputs.layers[-1], thickness_m=t))
        res = solve_conduction(replace(inputs, layers=layers))
        points.append(SweepPoint(r, t, res.Q_W, res.T_side1_K))
    return points


def default_insulation_radii(inputs: ConductionInputs, n: int = 80) -> list[float]:
    """Radios para el barrido: desde la cara interior de la capa exterior hasta
    pasar bien el radio crítico y el radio actual."""
    if inputs.geometry == "plane":
        return []
    faces = _faces(inputs)
    r0, r_now = faces[-2], faces[-1]
    res = solve_conduction(inputs)
    r_cr = res.critical_radius_m or r_now
    r_max = max(2.0 * r_now, 2.5 * r_cr, 1.5 * r0)
    return [r0 + (r_max - r0) * (j + 1) / n for j in range(n)]


# ---------------------------------------------------------------------
# Notas
# ---------------------------------------------------------------------


def conduction_notes(result: ConductionResult) -> list[str]:
    """Interpretación física (markdown)."""
    notes: list[str] = []
    inputs = result.inputs
    biggest = max(result.resistances, key=lambda r: r.R_K_per_W)
    share = result.share(biggest)
    if share >= 0.5 and len(result.resistances) > 1:
        notes.append(
            f"La resistencia que manda es **{biggest.label.lower()}** ({_pct(share)} del total): "
            "ahí está la mayor caída de temperatura, y para cambiar Q̇ hay que actuar sobre ella."
        )
    for r in result.resistances:
        if r.kind == "conducción" and len(result.resistances) > 2 and result.share(r) < 0.005:
            notes.append(
                f"{r.label} casi no ofrece resistencia ({_num(100.0 * result.share(r), 2)} %): "
                "la caída de "
                f"temperatura en ella es de {_num(abs(r.dT_K))} K. En la práctica se desprecia "
                "(como la pared metálica de un caño)."
            )
    for i, layer in enumerate(inputs.layers):
        if layer.contact_m2K_per_W > 0.0:
            L_eq = layer.k_effective * layer.contact_m2K_per_W
            notes.append(
                f"El contacto entre las capas {i + 1} y {i + 2} (R''_c = "
                f"{_num(layer.contact_m2K_per_W)} m²·K/W) equivale a "
                f"{_num(1000.0 * L_eq)} mm de {layer.name.lower()} (L = k·R''_c): las "
                "superficies reales se tocan solo en sus rugosidades y el aire atrapado conduce "
                "mal (Cengel y Ghajar, §3-2)."
            )
        if layer.parts:
            notes.append(
                f"La capa {i + 1} se resuelve con sus partes en paralelo y las caras isotérmicas "
                "(Cengel y Ghajar, §3-3): es una aproximación unidimensional; el calor que pasa "
                "por cada parte está en la tabla."
            )
    r_cr = result.critical_radius_m
    # el radio crítico es una idea de aislaciones: con una capa exterior metálica
    # (el acero de un tanque) es cierto pero no viene al caso
    if r_cr is not None and inputs.layers[-1].k_W_per_mK < INSULATION_K_MAX:
        r_out = result.faces_m[-1]
        heat_given = inputs.inner.kind == "heat"
        if r_out < r_cr:
            effect = "baja la temperatura del lado 1" if heat_given else "aumenta Q̇"
            notes.append(
                f"El radio exterior ({_len(r_out)}) es **menor que el radio crítico** "
                f"r_cr = {_len(r_cr)}: agregar más de la capa exterior {effect} hasta "
                "r_cr, porque gana más superficie de convección que resistencia de conducción. "
                "Por eso se aíslan los cables eléctricos finos (Cengel y Ghajar, §3-5)."
            )
        else:
            effect = "sube la temperatura del lado 1" if heat_given else "baja Q̇"
            notes.append(
                f"El radio exterior ({_len(r_out)}) supera el radio crítico "
                f"(r_cr = {_len(r_cr)}): más aislación {effect}."
            )
    if result.Q_W < 0.0:
        notes.append(
            "Q̇ sale negativo: el calor va del lado 2 al lado 1 (por ejemplo, del ambiente al "
            "interior de un tanque frío)."
        )
    return notes


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------

C = 273.15

CONDUCTION_EXAMPLES: dict[str, ConductionInputs] = {
    "Pared de ladrillo (Cengel y Ghajar: 3 × 5 m, 30 cm, k = 0,9; caras a 16 y 2 °C)": (
        ConductionInputs(
            geometry="plane",
            layers=(Layer("ladrillo", 0.30, 0.9),),
            inner=Boundary("surface", T_K=16.0 + C),
            outer=Boundary("surface", T_K=2.0 + C),
            area_m2=15.0,
        )
    ),
    "Ventana de un vidrio (Cengel y Ghajar: 0,8 × 1,5 m, 8 mm; 20 y −10 °C)": ConductionInputs(
        geometry="plane",
        layers=(Layer("vidrio", 0.008, 0.78),),
        inner=Boundary("fluid", T_K=20.0 + C, h_W_per_m2K=10.0),
        outer=Boundary("fluid", T_K=-10.0 + C, h_W_per_m2K=40.0),
        area_m2=1.2,
    ),
    "Ventana de doble vidrio (Cengel y Ghajar: 4 mm + 10 mm de aire + 4 mm)": ConductionInputs(
        geometry="plane",
        layers=(
            Layer("vidrio", 0.004, 0.78),
            Layer("aire quieto", 0.010, 0.026),
            Layer("vidrio", 0.004, 0.78),
        ),
        inner=Boundary("fluid", T_K=20.0 + C, h_W_per_m2K=10.0),
        outer=Boundary("fluid", T_K=-10.0 + C, h_W_per_m2K=40.0),
        area_m2=1.2,
    ),
    "Pared de ladrillos con revoque en paralelo (Cengel y Ghajar: 3 × 5 m)": ConductionInputs(
        geometry="plane",
        layers=(
            Layer("espuma rígida", 0.03, 0.026),
            Layer("revoque", 0.02, 0.22),
            Layer(
                "ladrillos y revoque",
                0.16,
                parts=(
                    ParallelPart("revoque (arriba)", 0.22, 0.06),
                    ParallelPart("ladrillo", 0.72, 0.88),
                    ParallelPart("revoque (abajo)", 0.22, 0.06),
                ),
            ),
            Layer("revoque", 0.02, 0.22),
        ),
        inner=Boundary("fluid", T_K=20.0 + C, h_W_per_m2K=10.0),
        outer=Boundary("fluid", T_K=-10.0 + C, h_W_per_m2K=25.0),
        area_m2=15.0,
    ),
    "Dos placas de aluminio apretadas: el contacto (Cengel y Ghajar: h_c = 11 000)": (
        ConductionInputs(
            geometry="plane",
            layers=(
                Layer("aluminio", 0.01, 237.0, contact_m2K_per_W=1.0 / 11_000.0),
                Layer("aluminio", 0.01, 237.0),
            ),
            inner=Boundary("surface", T_K=80.0 + C),
            outer=Boundary("surface", T_K=20.0 + C),
            area_m2=1.0,
        )
    ),
    "Caño de vapor aislado (Cengel y Ghajar: fundición y lana de vidrio, 320 °C)": (
        ConductionInputs(
            geometry="cylinder",
            layers=(Layer("fundición", 0.0025, 80.0), Layer("lana de vidrio", 0.03, 0.05)),
            inner=Boundary("fluid", T_K=320.0 + C, h_W_per_m2K=60.0),
            outer=Boundary("fluid", T_K=5.0 + C, h_W_per_m2K=18.0),
            length_m=1.0,
            r_inner_m=0.025,
        )
    ),
    "Alambre eléctrico con cobertura plástica: radio crítico (Cengel y Ghajar)": (
        ConductionInputs(
            geometry="cylinder",
            layers=(Layer("plástico", 0.002, 0.15),),
            inner=Boundary("heat", Q_W=80.0),
            outer=Boundary("fluid", T_K=30.0 + C, h_W_per_m2K=12.0),
            length_m=5.0,
            r_inner_m=0.0015,
        )
    ),
    "Tanque esférico de agua helada (basado en Cengel y Ghajar: acero de 2 cm)": (
        ConductionInputs(
            geometry="sphere",
            layers=(Layer("acero inoxidable", 0.02, 15.0),),
            inner=Boundary("fluid", T_K=0.0 + C, h_W_per_m2K=80.0),
            outer=Boundary("fluid", T_K=22.0 + C, h_W_per_m2K=15.34),
            r_inner_m=1.5,
        )
    ),
}

CONDUCTION_EXAMPLE_NOTES: dict[str, str] = {
    next(iter(CONDUCTION_EXAMPLES)): ("Las dos caras a temperatura dada: Q̇ = k·A·ΔT/L = 630 W."),
    list(CONDUCTION_EXAMPLES)[1]: (
        "El libro: 266 W y la cara interior del vidrio a −2,2 °C (por eso se empaña o se escarcha)."
    ),
    list(CONDUCTION_EXAMPLES)[2]: (
        "El libro: 69,2 W y la cara interior a 14,2 °C. El aire quieto del medio es casi toda la "
        "resistencia."
    ),
    list(CONDUCTION_EXAMPLES)[3]: (
        "El libro resuelve una sección de 0,25 m de alto (R = 6,87 K/W, 4,37 W) y la escala a "
        "la pared: 262 W. Acá la hilada de ladrillos y revoque es una capa con partes en paralelo."
    ),
    list(CONDUCTION_EXAMPLES)[4]: (
        "El libro calcula el espesor de aluminio equivalente al contacto: L = k/h_c = 2,15 cm."
    ),
    list(CONDUCTION_EXAMPLES)[5]: (
        "El libro: 121 W por metro de caño; la caída en el caño es de 0,02 °C y en la aislación "
        "de 284 °C."
    ),
    list(CONDUCTION_EXAMPLES)[6]: (
        "10 A y 8 V: 80 W generados en el alambre. El libro: la interfaz alambre–plástico a "
        "105 °C y, como r₂ < r_cr = 12,5 mm, duplicar el plástico la baja."
    ),
    list(CONDUCTION_EXAMPLES)[7]: (
        "El h exterior de 15,34 W/(m²·K) es el combinado del libro (convección natural y "
        "radiación). El calor entra al tanque: Q̇ sale negativo."
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def conduction_to_dict(result: ConductionResult, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""
    inputs = result.inputs

    def T(x: float) -> str:
        return format_quantity(x, "temperature", system)

    data: dict[str, Any] = {
        "geometria": GEOMETRY_NAMES[inputs.geometry],
        "capas": [
            {
                "nombre": layer.name,
                "espesor": format_quantity(layer.thickness_m, "small_length", system),
                "k": format_quantity(layer.k_effective, "thermal_conductivity", system),
                **(
                    {"partes": {p.name: p.area_fraction for p in layer.parts}}
                    if layer.parts
                    else {}
                ),
                **(
                    {"R_contacto_m2K_W": layer.contact_m2K_per_W} if layer.contact_m2K_per_W else {}
                ),
            }
            for layer in inputs.layers
        ],
        "Q": format_quantity(result.Q_W, "heat_rate", system),
        "R_total": format_quantity(result.R_total_K_per_W, "thermal_resistance", system),
        "U_lado1": format_quantity(result.U_inner_W_per_m2K, "heat_transfer_coefficient", system),
        "U_lado2": format_quantity(result.U_outer_W_per_m2K, "heat_transfer_coefficient", system),
        "q_lado1": format_quantity(result.q_inner_W_per_m2, "heat_flux", system),
        "q_lado2": format_quantity(result.q_outer_W_per_m2, "heat_flux", system),
        "resistencias": [
            {
                "elemento": r.label,
                "R": format_quantity(r.R_K_per_W, "thermal_resistance", system),
                "fraccion": result.share(r),
                "T_desde": T(r.T_from_K),
                "T_hasta": T(r.T_to_K),
            }
            for r in result.resistances
        ],
    }
    if result.Q_per_length_W_per_m is not None:
        data["Q_por_metro"] = format_quantity(
            result.Q_per_length_W_per_m, "linear_heat_rate", system
        )
    if result.critical_radius_m is not None:
        data["radio_critico"] = format_quantity(result.critical_radius_m, "small_length", system)
    return data
