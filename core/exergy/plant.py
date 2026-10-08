"""Exergía por componente de una planta y las filas del diagrama de Grassmann (Fase 7).

Cada componente k tiene (Bejan, Tsatsaronis y Moran, 1996, cap. 3; vademecum §11.9
y §11.10):

- **F_k** («combustible»): la exergía que consume el componente;
- **P_k** («producto»): la que produce, lo que se quiere de él;
- **L_k** (pérdida): la que sale de la planta sin usarse (escape, chimenea);
- **D_k** = T₀·Ṡ_gen,k: la destruida por las irreversibilidades.

Se cumple F_k = P_k + D_k + L_k y, para la planta, Ẋ_entra = Ẋ_producto + ΣẊ_D + ΣẊ_L.
D_k se calcula aparte, con el balance de entropía, como control (los tests verifican
F − P − L = D). Las definiciones de F y P por tipo de componente:

- turbina: F = Σṁψ_ent − Σṁψ_sal, P = Ẇ; bomba y compresor: F = Ẇ, P = ΔẊ;
- caldera o calentador con una fuente a T: F = Q̇·(1 − T₀/T), P = ΔẊ del fluido;
- intercambiadores y mezclas adiabáticas: F = lo que pierden las corrientes que
  pierden exergía, P = lo que ganan las que la ganan (vale también por debajo de
  T₀, donde la corriente que se enfría es la que gana exergía);
- disipativos (válvula, condensador, interenfriador, cañería): P = 0.

**Convención** (Cengel 10-8): el calor que va al ambiente a T₀ (condensador,
interenfriadores) no lleva exergía, así que lo que pierde el fluido ahí se
destruye en ese componente; lo que sale con una corriente (escape, chimenea) es
pérdida. Con un sumidero a T_L > T₀, la exergía de ese calor, Q̇·(1 − T₀/T_L), es
pérdida.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from core.cycles.combined_multi import MultiCombinedResult, bottoming_exergy
from core.cycles.gas_turbine import GasTurbineResult, from_brayton, solve_gas_turbine
from core.cycles.hrsg_multi import hrsg_exergy
from core.cycles.rankine import RankineResult
from core.cycles.rankine_layout import CycleComponent, Port
from core.cycles.refrigeration import (
    RefrigerationResult,
    Reservoirs,
    check_reservoirs,
    default_reservoirs,
)
from core.exergy.chemical import chemical_species, mixture_chemical_exergy
from core.exergy.physical import Ambient, _dead_state
from core.fluids import FLUID_NAMES_ES, FluidState
from core.ideal_gas import FlueGas
from core.units_system import UnitSystem, format_quantity

__all__ = [
    "KIND_EXPLANATIONS",
    "ComponentExergy",
    "FuelModel",
    "GrassmannRow",
    "PlantExergy",
    "PlantKind",
    "StreamExergy",
    "combined_plant_exergy",
    "default_rankine_source_K",
    "gas_turbine_plant_exergy",
    "grassmann_rows",
    "plant_exergy_to_dict",
    "plant_notes",
    "rankine_exergy",
    "refrigeration_exergy",
]

PlantKind = Literal["rankine", "refrigeration", "gas_turbine", "combined"]
FuelModel = Literal["pci", "szargut"]

#: Una fuente de calor para el Rankine de agua: el hogar de Cengel 10-8.
WATER_SOURCE_K = 1600.0
#: Con un fluido orgánico (ORC), la fuente queda esta cantidad arriba del vapor más caliente.
ORC_SOURCE_MARGIN_K = 50.0


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# ---------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentExergy:
    """Balance de exergía de un componente (W).

    ``destroyed_W`` es T₀·Ṡ_gen (o, en una cámara de combustión, lo que falta del
    balance). ``dissipative`` marca los componentes sin producto (válvula,
    condensador, interenfriador, cañería): su F se destruye o se pierde.
    ``is_loss`` marca una pérdida de la planta (escape, chimenea), que se
    muestra como un componente más.
    """

    name: str
    kind: str
    fuel_W: float
    product_W: float
    destroyed_W: float
    loss_W: float = 0.0
    dissipative: bool = False
    is_loss: bool = False

    @property
    def efficiency(self) -> float | None:
        """ε_k = P/F (vademecum §11.10); ``None`` en los disipativos y las pérdidas."""
        if self.dissipative or self.is_loss or self.fuel_W <= 0.0:
            return None
        return self.product_W / self.fuel_W

    @property
    def residual_W(self) -> float:
        """F − P − D − L (tiene que dar ~0)."""
        return self.fuel_W - self.product_W - self.destroyed_W - self.loss_W


@dataclass(frozen=True)
class StreamExergy:
    """Una corriente (estado) con su caudal y su exergía física y química, por kg."""

    label: str
    m_kg_s: float
    psi_J_per_kg: float
    e_ch_J_per_kg: float = 0.0
    T_K: float | None = None
    p_Pa: float | None = None

    @property
    def x_J_per_kg(self) -> float:
        return self.psi_J_per_kg + self.e_ch_J_per_kg

    @property
    def X_W(self) -> float:
        return self.m_kg_s * self.x_J_per_kg


@dataclass(frozen=True)
class PlantExergy:
    """Exergía de una planta por componente (W), con el ambiente a T₀, p₀.

    ``inputs`` es la exergía que entra (combustible, calor de la fuente o
    trabajo) y ``products`` lo que se quiere (trabajo neto, el frío o el calor).
    ``components`` sigue el orden del flujo (el del diagrama de Grassmann).
    ``settings`` guarda los datos del análisis (T₀, T_H, T_L, el modelo del
    combustible…) para el procedimiento y el export.
    """

    kind: PlantKind
    title: str
    T0_K: float
    p0_Pa: float
    inputs: tuple[tuple[str, float], ...]
    products: tuple[tuple[str, float], ...]
    components: tuple[ComponentExergy, ...]
    streams: tuple[StreamExergy, ...] = ()
    fuel_model: str = ""
    settings: tuple[tuple[str, float], ...] = ()

    @property
    def fuel_W(self) -> float:
        return sum(x for _, x in self.inputs)

    @property
    def product_W(self) -> float:
        return sum(x for _, x in self.products)

    @property
    def destroyed_W(self) -> float:
        return sum(c.destroyed_W for c in self.components)

    @property
    def loss_W(self) -> float:
        return sum(c.loss_W for c in self.components)

    @property
    def efficiency(self) -> float:
        """η_II = Ẋ_producto / Ẋ_entra (vademecum §11.10)."""
        return self.product_W / self.fuel_W

    @property
    def residual_W(self) -> float:
        """Ẋ_entra − Ẋ_producto − ΣẊ_D − ΣẊ_L (tiene que dar ~0)."""
        return self.fuel_W - self.product_W - self.destroyed_W - self.loss_W

    def y_D(self, comp: ComponentExergy) -> float:
        """Destrucción relativa a la exergía que entra a la planta."""
        return comp.destroyed_W / self.fuel_W

    def y_star_D(self, comp: ComponentExergy) -> float:
        """Destrucción relativa a la destrucción total."""
        total = self.destroyed_W
        return comp.destroyed_W / total if total > 0.0 else 0.0

    def setting(self, key: str) -> float | None:
        return dict(self.settings).get(key)


# ---------------------------------------------------------------------
# Herramientas comunes
# ---------------------------------------------------------------------


def _exchanger(paths: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """(F, P) de un intercambiador o una mezcla adiabática, desde los ΔẊ de cada tramo.

    ``paths`` son los ΔẊ = ṁ·(ψ_sal − ψ_ent) de cada corriente: las que pierden
    exergía son el combustible y las que ganan, el producto.
    """
    fuel = sum(-dx for dx in paths if dx < 0.0)
    product = sum(dx for dx in paths if dx > 0.0)
    return fuel, product


def _zero_if_noise(x: float, scale: float) -> float:
    """Un componente ideal da T₀·ΔS ~ 1e-10: es 0."""
    return 0.0 if abs(x) < 1e-9 * max(scale, 1.0) else x


# ---------------------------------------------------------------------
# Rankine
# ---------------------------------------------------------------------

_RANKINE_KIND: dict[str, str] = {
    "boiler": "caldera",
    "reheater": "recalentador",
    "turbine": "turbina",
    "pump": "bomba",
    "condenser": "condensador",
    "open_heater": "calentador abierto",
    "closed_heater": "calentador cerrado",
    "mixer": "mezcla",
    "valve": "válvula",
    "pipe": "cañería",
    "recuperator": "recuperador",
}


def default_rankine_source_K(result: RankineResult) -> float:
    """La fuente de calor por defecto: el hogar de Cengel 10-8 (1600 K) con agua; con un
    fluido orgánico, 50 K arriba del vapor más caliente (una fuente de calor residual)."""
    if result.inputs.fluid == "Water":
        return WATER_SOURCE_K
    return max(s.T_K for s in result.states) + ORC_SOURCE_MARGIN_K


def _rankine_paths(comp: CycleComponent) -> list[tuple[Port, Port]]:
    """Los tramos (entrada, salida) de un componente del Rankine."""
    if comp.kind == "closed_heater":
        paths = [(comp.port("fw_in"), comp.port("fw_out"))]
        hot_out = comp.port("drain_out")
        paths += [(p, hot_out) for p in comp.ports if p.role in ("bleed", "drain_in")]
        return paths
    if comp.kind == "recuperator":
        return [(comp.port("fw_in"), comp.port("fw_out")), (comp.port("in"), comp.port("out"))]
    if comp.kind in ("open_heater", "mixer"):
        out = comp.outlets[0]
        return [(p, out) for p in comp.inlets]
    return [(p, comp.outlets[0]) for p in comp.inlets]


def rankine_exergy(
    result: RankineResult,
    *,
    ambient: Ambient | None = None,
    T_source_K: float | None = None,
    T_sink_K: float | None = None,
) -> PlantExergy:
    """Exergía por componente de un ciclo de Rankine (Cengel 10-8, cualquier configuración).

    El calor entra en la caldera y el recalentador desde una fuente a
    ``T_source_K`` (por defecto, :func:`default_rankine_source_K`) y sale en el
    condensador hacia un sumidero a ``T_sink_K`` (por defecto, T₀). Las
    cañerías con pérdida de calor lo entregan al ambiente.

    Raises
    ------
    ValueError
        Si la fuente no está más caliente que el vapor, si el sumidero está más
        frío que el ambiente o más caliente que el fluido en el condensador.
    """
    ambient = ambient or Ambient(290.0, 100_000.0)
    T0 = ambient.T0_K
    T_H = default_rankine_source_K(result) if T_source_K is None else T_source_K
    T_L = T0 if T_sink_K is None else T_sink_K
    st = result.states
    T_max = max(
        st[c.outlets[0].state].T_K for c in result.components if c.kind in ("boiler", "reheater")
    )
    if not (math.isfinite(T_H) and T_H > T_max):
        raise ValueError(
            f"La fuente de calor (T_H = {_degC(T_H)}) tiene que estar más caliente que el "
            f"vapor más caliente del ciclo ({_degC(T_max)}): si no, el calor no podría pasar de "
            "la fuente al fluido."
        )
    cond = next(c for c in result.components if c.kind == "condenser")
    T_cond_min = min(st[p.state].T_K for p in cond.ports)
    if T_L < T0 - 1e-9:
        raise ValueError(
            f"El sumidero del condensador (T_L = {_degC(T_L)}) no puede estar más frío que el "
            f"ambiente (T₀ = {_degC(T0)}): en este análisis el ambiente es lo más frío que hay."
        )
    if T_L > T_cond_min:
        raise ValueError(
            f"El sumidero (T_L = {_degC(T_L)}) tiene que estar más frío que el fluido en el "
            f"condensador ({_degC(T_cond_min)}): si no, el calor no saldría del condensador."
        )
    dead = _dead_state(result.inputs.fluid, ambient)
    h0, s0 = dead.h_J_per_kg, dead.s_J_per_kg_K
    m = result.m_dot_kg_s

    def psi(i: int) -> float:
        return (st[i].h_J_per_kg - h0) - T0 * (st[i].s_J_per_kg_K - s0)

    def flow(ports: Sequence[Port], f: Callable[[int], float]) -> float:
        return m * sum(p.fraction * f(p.state) for p in ports)

    def h(i: int) -> float:
        return st[i].h_J_per_kg

    def s(i: int) -> float:
        return st[i].s_J_per_kg_K

    # orden del flujo: desde la caldera
    comps = list(result.components)
    start = next(i for i, c in enumerate(comps) if c.kind == "boiler")
    ordered = comps[start:] + comps[:start]
    scale = m * abs(result.w_net_J_per_kg)
    components: list[ComponentExergy] = []
    heat_in = 0.0
    for c in ordered:
        H_in, H_out = flow(c.inlets, h), flow(c.outlets, h)
        S_in, S_out = flow(c.inlets, s), flow(c.outlets, s)
        X_in, X_out = flow(c.inlets, psi), flow(c.outlets, psi)
        Q = H_out - H_in  # calor que recibe (los que no tienen trabajo)
        kind = _RANKINE_KIND[c.kind]
        name = _cap(c.label)
        if c.kind in ("boiler", "reheater"):
            XQ = Q * (1.0 - T0 / T_H)
            heat_in += XQ
            D = T0 * (S_out - S_in - Q / T_H)
            components.append(ComponentExergy(name, kind, XQ, X_out - X_in, D))
        elif c.kind == "turbine":
            D = _zero_if_noise(T0 * (S_out - S_in), scale)
            components.append(ComponentExergy(name, kind, X_in - X_out, H_in - H_out, D))
        elif c.kind == "pump":
            D = _zero_if_noise(T0 * (S_out - S_in), scale)
            components.append(ComponentExergy(name, kind, H_out - H_in, X_out - X_in, D))
        elif c.kind == "condenser":
            L = -Q * (1.0 - T0 / T_L)
            D = T0 * (S_out - S_in - Q / T_L)
            components.append(
                ComponentExergy(name, kind, X_in - X_out, 0.0, D, L, dissipative=True)
            )
        elif c.kind == "pipe":
            D = T0 * (S_out - S_in - Q / T0)
            components.append(ComponentExergy(name, kind, X_in - X_out, 0.0, D, dissipative=True))
        elif c.kind == "valve":
            D = T0 * (S_out - S_in)
            components.append(ComponentExergy(name, kind, X_in - X_out, 0.0, D, dissipative=True))
        else:  # calentadores, mezcla, recuperador: adiabáticos
            paths = [m * a.fraction * (psi(b.state) - psi(a.state)) for a, b in _rankine_paths(c)]
            F, P = _exchanger(paths)
            D = _zero_if_noise(T0 * (S_out - S_in), scale)
            components.append(ComponentExergy(name, kind, F, P, D))
    streams = _rankine_streams(result, psi)
    fluid_name = FLUID_NAMES_ES.get(result.inputs.fluid, result.inputs.fluid)
    return PlantExergy(
        kind="rankine",
        title=f"Ciclo de Rankine ({fluid_name})",
        T0_K=T0,
        p0_Pa=ambient.p0_Pa,
        inputs=((f"Calor de la fuente a {_degC(T_H)}", heat_in),),
        products=(("Trabajo neto", m * result.w_net_J_per_kg),),
        components=tuple(components),
        streams=streams,
        fuel_model="calor",
        settings=(("T_H", T_H), ("T_L", T_L), ("T0", T0), ("p0", ambient.p0_Pa)),
    )


def _rankine_streams(
    result: RankineResult, psi: Callable[[int], float]
) -> tuple[StreamExergy, ...]:
    """Cada estado con su caudal (el del componente que lo produce)."""
    assert result.layout is not None
    flows: dict[int, float] = {}
    for c in result.components:
        for p in c.outlets:
            flows.setdefault(p.state, p.fraction * result.m_dot_kg_s)
    labels = result.layout.labeled()
    return tuple(
        StreamExergy(labels[i], flows.get(i, 0.0), psi(i), 0.0, st.T_K, st.P_Pa)
        for i, st in enumerate(result.states)
    )


# ---------------------------------------------------------------------
# Refrigeración
# ---------------------------------------------------------------------

_REFRIGERATION_KIND: dict[str, str] = {
    "compressor": "compresor",
    "condenser": "condensador",
    "evaporator": "evaporador",
    "valve": "válvula",
    "flash_tank": "cámara de evaporación",
    "mixer": "mezcla",
    "cascade_hx": "intercambiador de la cascada",
}


def refrigeration_exergy(
    result: RefrigerationResult, reservoirs: Reservoirs | None = None
) -> PlantExergy:
    """Exergía por componente de un refrigerador o una bomba de calor (Cengel §11-5).

    Las fuentes son las de ``reservoirs``, las del ciclo o, si no trae, las de
    :func:`core.cycles.refrigeration.default_reservoirs`. El ambiente T₀ es la
    fuente caliente del refrigerador (la fría de la bomba de calor). La exergía
    que entra es el trabajo de los compresores y el producto, la del efecto útil:
    Q̇_C·(T₀/T_C − 1) (el frío) o Q̇_H·(1 − T₀/T_H) (el calor).

    Raises
    ------
    ValueError
        Si las fuentes no son compatibles con el ciclo (el calor iría al revés).
    """
    res = reservoirs or default_reservoirs(result)
    check_reservoirs(result, res)
    heat_pump = result.inputs.heat_pump
    T0 = res.T_cold_K if heat_pump else res.T_hot_K
    p0 = 101_325.0
    ambient = Ambient(T0, p0)
    fluids = result.layout.fluids
    dead: dict[str, FluidState] = {}
    st, m = result.states, result.m_dot_kg_s

    def psi(i: int) -> float:
        f = fluids[i]
        if f not in dead:
            dead[f] = _dead_state(f, ambient)
        d = dead[f]
        return (st[i].h_J_per_kg - d.h_J_per_kg) - T0 * (st[i].s_J_per_kg_K - d.s_J_per_kg_K)

    scale = result.W_W
    components: list[ComponentExergy] = []
    Q_C, Q_H = result.Q_cold_W, result.Q_hot_W
    for c in result.components:
        X_in = sum(m[i] * psi(i) for i in c.inlets)
        X_out = sum(m[o] * psi(o) for o in c.outlets)
        dH = result.energy_out_minus_in_W(c)
        dS = result.entropy_out_minus_in_W_per_K(c)
        kind = _REFRIGERATION_KIND[c.kind]
        name = _cap(c.label)
        if c.kind == "compressor":
            D = _zero_if_noise(T0 * dS, scale)
            components.append(ComponentExergy(name, kind, dH, X_out - X_in, D))
        elif c.kind == "evaporator":
            D = T0 * (dS - Q_C / res.T_cold_K)
            if heat_pump:  # el calor viene del ambiente: no tiene exergía
                components.append(
                    ComponentExergy(name, kind, X_in - X_out, 0.0, D, dissipative=True)
                )
            else:
                P = Q_C * (T0 / res.T_cold_K - 1.0)
                components.append(ComponentExergy(name, kind, X_in - X_out, P, D))
        elif c.kind == "condenser":
            D = T0 * (dS + Q_H / res.T_hot_K)
            if heat_pump:
                P = Q_H * (1.0 - T0 / res.T_hot_K)
                components.append(ComponentExergy(name, kind, X_in - X_out, P, D))
            else:  # el calor va al ambiente
                components.append(
                    ComponentExergy(name, kind, X_in - X_out, 0.0, D, dissipative=True)
                )
        elif c.kind == "valve":
            components.append(
                ComponentExergy(name, kind, X_in - X_out, 0.0, T0 * dS, dissipative=True)
            )
        else:  # cámara, mezcla, intercambiador de la cascada
            # en la mezcla cada entrada lleva su caudal; en la cámara, cada salida el suyo
            paths = [(m[a] if c.kind == "mixer" else m[b]) * (psi(b) - psi(a)) for a, b in c.paths]
            F, P = _exchanger(paths)
            components.append(ComponentExergy(name, kind, F, P, _zero_if_noise(T0 * dS, scale)))
    if heat_pump:
        product = ("Calor útil (exergía)", Q_H * (1.0 - T0 / res.T_hot_K))
    else:
        product = ("Frío (exergía del efecto frigorífico)", Q_C * (T0 / res.T_cold_K - 1.0))
    labels = result.layout.labeled()
    streams = tuple(
        StreamExergy(labels[i], m[i], psi(i), 0.0, s_.T_K, s_.P_Pa) for i, s_ in enumerate(st)
    )
    return PlantExergy(
        kind="refrigeration",
        title="Bomba de calor" if heat_pump else "Refrigerador",
        T0_K=T0,
        p0_Pa=p0,
        inputs=(("Trabajo de los compresores", result.W_W),),
        products=(product,),
        components=tuple(components),
        streams=streams,
        fuel_model="trabajo",
        settings=(("T_C", res.T_cold_K), ("T_H", res.T_hot_K), ("T0", T0)),
    )


# ---------------------------------------------------------------------
# Turbina de gas y ciclo combinado
# ---------------------------------------------------------------------


def _gas_key_map() -> dict[str, str]:
    """Nombre de ISO 6976 → clave de la tabla de exergías químicas."""
    return {sp.iso6976_name: k for k, sp in chemical_species().items() if sp.iso6976_name}


def _mix_chemical_J_per_kg(mix: FlueGas) -> float:
    """Exergía química por kg de una mezcla de gases (con el agua que condensa a 25 °C)."""
    return mixture_chemical_exergy(mix.mole_fractions).e_J_per_kg


def gas_fuel_chemical_exergy(composition: Sequence[tuple[str, float]]) -> float:
    """Exergía química (J/kg) de un combustible gaseoso de ISO 6976 (Szargut et al., 1988).

    Raises
    ------
    ValueError
        Si un componente no está en la tabla de exergías químicas.
    """
    names = _gas_key_map()
    missing = [n for n, _ in composition if n not in names]
    if missing:
        raise ValueError(
            "No hay exergía química tabulada para: " + ", ".join(missing) + ". Usá el PCI."
        )
    mix = mixture_chemical_exergy({names[n]: x for n, x in composition}, condense=False)
    return mix.e_J_per_kg


@dataclass(frozen=True)
class _GasTurbineParts:
    components: list[ComponentExergy]
    inputs: list[tuple[str, float]]
    exhaust_physical_W: float
    exhaust_chemical_W: float
    streams: list[StreamExergy]


def _mass_factors(result: GasTurbineResult) -> dict[int, float]:
    """kg de cada corriente por kg de aire (1 en el aire, 1 + F en los gases).

    Los estados isoentrópicos (2s, 4s…) no son corrientes y no están.
    """
    f: dict[int, float] = {}
    for c in result.compressors:
        f[c.inlet] = f[c.outlet] = 1.0
    for ic in result.intercoolers:
        f[ic.inlet] = f[ic.outlet] = 1.0
    rg = result.regenerator
    if rg is not None:
        f[rg.air_in] = f[rg.air_out] = 1.0
    for cc, t in zip(result.combustors, result.turbines, strict=True):
        f.setdefault(cc.inlet, cc.m_in)
        f[cc.outlet] = cc.m_out
        f[t.inlet] = t.m
        f[t.outlet] = t.m
    if rg is not None:
        f[rg.gas_in] = f[rg.gas_out] = rg.m_gas
    f[result.exhaust] = 1.0 + result.fuel_air_ratio
    return f


def _gas_turbine_parts(result: GasTurbineResult, fuel_model: FuelModel) -> _GasTurbineParts:
    """Los componentes de la turbina de gas (W) en el orden del flujo, sin el escape.

    Compresores (con el interenfriador después de cada uno), cámaras y turbinas
    alternadas y, al final, el regenerador (por el lado de los gases).
    """
    inputs = result.inputs
    T0, p0 = inputs.T_amb_K, inputs.p_amb_Pa
    st = result.states
    m_a = result.m_air_kg_s
    chem = fuel_model == "szargut" and inputs.fuel is not None
    e_ch_cache: dict[int, float] = {}

    def e_ch(i: int) -> float:
        if not chem:
            return 0.0
        if i not in e_ch_cache:
            e_ch_cache[i] = _mix_chemical_J_per_kg(st[i].mix)
        return e_ch_cache[i]

    def psi(i: int) -> float:
        return st[i].psi(T0, p0)

    def ds(i: int, j: int) -> float:
        return st[j].s_J_per_kg_K - st[i].s_J_per_kg_K

    scale = abs(result.w_turbine_J_per_kg) * m_a
    comps: list[ComponentExergy] = []
    ic_after = {ic.inlet: ic for ic in result.intercoolers}
    for c in result.compressors:
        W = m_a * result.stage_work(c)
        dX = m_a * c.m * (psi(c.outlet) - psi(c.inlet))
        D = _zero_if_noise(m_a * c.m * T0 * ds(c.inlet, c.outlet), scale)
        comps.append(ComponentExergy(_cap(c.name), "compresor", W, dX, D))
        ic = ic_after.get(c.outlet)
        if ic is not None:
            F = m_a * (psi(ic.inlet) - psi(ic.outlet))
            comps.append(
                ComponentExergy(_cap(ic.name), "interenfriador", F, 0.0, F, dissipative=True)
            )
    fuel = inputs.fuel
    e_f = 0.0
    if fuel is not None:
        e_f = gas_fuel_chemical_exergy(fuel.composition) if chem else fuel.lhv_J_per_kg
    x_in_total = 0.0
    for cc, t in zip(result.combustors, result.turbines, strict=True):
        X_in = m_a * cc.m_in * (psi(cc.inlet) + e_ch(cc.inlet))
        X_out = m_a * cc.m_out * (psi(cc.outlet) + e_ch(cc.outlet))
        if fuel is None:
            F = m_a * cc.q * (1.0 - T0 / st[cc.outlet].T_K)
            kind = "calentador"
        else:
            F = m_a * cc.fuel * e_f
            kind = "cámara de combustión"
        x_in_total += F
        comps.append(ComponentExergy(_cap(cc.name), kind, F, X_out - X_in, F - (X_out - X_in)))
        W = m_a * result.stage_work(t)
        dX = m_a * t.m * (psi(t.inlet) - psi(t.outlet))
        D = _zero_if_noise(m_a * t.m * T0 * ds(t.inlet, t.outlet), scale)
        comps.append(ComponentExergy(_cap(t.name), "turbina", dX, W, D))
    rg = result.regenerator
    if rg is not None:
        dX_air = m_a * (psi(rg.air_out) - psi(rg.air_in))
        dX_gas = m_a * rg.m_gas * (psi(rg.gas_out) - psi(rg.gas_in))
        F, P = _exchanger([dX_air, dX_gas])
        D = _zero_if_noise(
            m_a * T0 * (ds(rg.air_in, rg.air_out) + rg.m_gas * ds(rg.gas_in, rg.gas_out)), scale
        )
        comps.append(ComponentExergy("Regenerador", "regenerador", F, P, D))
    entries: list[tuple[str, float]] = []
    if fuel is None:
        entries.append(("Calor de los calentadores", x_in_total))
    elif chem:
        entries.append(("Combustible (exergía química, Szargut)", x_in_total))
        entries.append(("Aire (exergía química)", m_a * e_ch(0)))
    else:
        entries.append(("Combustible (≈ PCI)", x_in_total))
    factors = _mass_factors(result)
    streams = [
        StreamExergy(
            f"{st[i].label} ({st[i].description})",
            m_a * factors[i],
            psi(i),
            e_ch(i),
            st[i].T_K,
            st[i].P_Pa,
        )
        for i in sorted(factors)
    ]
    m_exh = m_a * (1.0 + result.fuel_air_ratio)
    exh = result.exhaust
    return _GasTurbineParts(comps, entries, m_exh * psi(exh), m_exh * e_ch(exh), streams)


def gas_turbine_plant_exergy(
    result: GasTurbineResult, fuel_model: FuelModel = "szargut"
) -> PlantExergy:
    """Exergía por componente de la turbina de gas (Cengel §9-12; Bejan et al., 1996).

    Con ``fuel_model="pci"`` la exergía del combustible es su PCI (vademecum
    §16.13) y los números coinciden con
    :func:`core.cycles.gas_turbine.gas_turbine_exergy`. Con ``"szargut"`` entra
    su exergía química (Szargut et al., 1988) más la del aire, y el escape se
    lleva también la química de los gases. Con aire estándar entra la exergía
    del calor de cada calentador, Q·(1 − T₀/T) con T su temperatura de salida.
    """
    parts = _gas_turbine_parts(result, fuel_model)
    exh = result.states[result.exhaust]
    loss = parts.exhaust_physical_W + parts.exhaust_chemical_W
    comps = list(parts.components)
    comps.append(
        ComponentExergy(
            f"Escape (gases a {_degC(exh.T_K)})", "escape", loss, 0.0, 0.0, loss, is_loss=True
        )
    )
    model = "calor" if result.inputs.fuel is None else fuel_model
    return PlantExergy(
        kind="gas_turbine",
        title="Turbina de gas",
        T0_K=result.inputs.T_amb_K,
        p0_Pa=result.inputs.p_amb_Pa,
        inputs=tuple(parts.inputs),
        products=(("Trabajo neto", result.m_air_kg_s * result.w_net_J_per_kg),),
        components=tuple(comps),
        streams=tuple(parts.streams),
        fuel_model=model,
        settings=(
            ("T0", result.inputs.T_amb_K),
            ("p0", result.inputs.p_amb_Pa),
            ("escape_quimica", parts.exhaust_chemical_W),
        ),
    )


def combined_plant_exergy(
    result: MultiCombinedResult, fuel_model: FuelModel = "szargut"
) -> PlantExergy:
    """Exergía de toda la planta de ciclo combinado: turbina de gas, HRSG y ciclo de vapor.

    La turbina de gas se analiza como en :func:`gas_turbine_plant_exergy` (sin la
    pérdida del escape: los gases van a la HRSG); la HRSG, por sección
    (:func:`core.cycles.hrsg_multi.hrsg_exergy`), y el ciclo de vapor como
    :func:`core.cycles.combined_multi.bottoming_exergy`. Se pierde la exergía de
    los gases de la chimenea (física y, con Szargut, la química).
    """
    gt_res = solve_gas_turbine(from_brayton(result.inputs.gas_turbine))
    parts = _gas_turbine_parts(gt_res, fuel_model)
    hx = hrsg_exergy(result.hrsg)
    bt = bottoming_exergy(result)
    T0 = hx.T0_K
    cyc = result.steam
    # junto a las turbinas de vapor, la de gas se llama así
    comps = [
        replace(c, name="Turbina de gas") if c.name == "Turbina" else c for c in parts.components
    ]
    # HRSG, sección por sección
    gained = dict(hx.gained_W)
    for name, D in hx.destroyed_W:
        P = gained[name]
        comps.append(ComponentExergy(f"HRSG: {name}", "sección de la HRSG", P + D, P, D))
    destroyed = list(bt.destroyed_W)[len(hx.destroyed_W) :]
    # turbinas de vapor
    dead = _dead_state("Water", Ambient(T0, result.inputs.gas_turbine.p_amb_Pa))

    def psi(i: int) -> float:
        s_ = cyc.states[i]
        return (s_.h_J_per_kg - dead.h_J_per_kg) - T0 * (s_.s_J_per_kg_K - dead.s_J_per_kg_K)

    k = 0
    for t in cyc.turbines:
        name, D = destroyed[k]
        k += 1
        # «turbina de alta» → «Turbina de vapor de alta»; «turbina» → «Turbina de vapor»
        label = "Turbina de vapor" + name.removeprefix("turbina")
        comps.append(ComponentExergy(label, "turbina", t.W_W + D, t.W_W, D))
    for a in cyc.admissions:
        name, D = destroyed[k]
        k += 1
        paths = [
            a.m_turbine_kg_s * (psi(a.outlet) - psi(a.turbine)),
            a.m_steam_kg_s * (psi(a.outlet) - psi(a.steam)),
        ]
        F, P = _exchanger(paths)
        comps.append(ComponentExergy(_cap(name), "mezcla", F, P, D))
    if cyc.deaerator is not None:
        name, D = destroyed[k]
        k += 1
        ext = cyc.extraction
        assert ext is not None
        paths = [
            cyc.m_extraction_kg_s * (psi(cyc.deaerator) - psi(ext)),
            cyc.m_condenser_kg_s * (psi(cyc.deaerator) - psi(cyc.condensate_pump_out)),
        ]
        F, P = _exchanger(paths)
        comps.append(ComponentExergy("Desaireador", "desaireador", F, P, D))
    name, D = destroyed[k]
    W_pumps = result.W_pumps_W
    comps.append(ComponentExergy("Bombas", "bomba", W_pumps, W_pumps - D, D))
    comps.append(
        ComponentExergy(
            "Condensador", "condensador", bt.X_condenser_W, 0.0, bt.X_condenser_W, dissipative=True
        )
    )
    stack_chem = 0.0
    if fuel_model == "szargut" and result.inputs.gas_turbine.fuel is not None:
        stack_chem = parts.exhaust_chemical_W
    stack = bt.X_stack_W + stack_chem
    comps.append(
        ComponentExergy(
            f"Chimenea (gases a {_degC(result.hrsg.T_stack_K)})",
            "chimenea",
            stack,
            0.0,
            0.0,
            stack,
            is_loss=True,
        )
    )
    model = "calor" if result.inputs.gas_turbine.fuel is None else fuel_model
    return PlantExergy(
        kind="combined",
        title="Ciclo combinado",
        T0_K=T0,
        p0_Pa=result.inputs.gas_turbine.p_amb_Pa,
        inputs=tuple(parts.inputs),
        products=(
            ("Trabajo de la turbina de gas", result.W_gas_turbine_W),
            ("Trabajo neto del ciclo de vapor", result.W_steam_turbine_W),
        ),
        components=tuple(comps),
        streams=tuple(parts.streams),
        fuel_model=model,
        settings=(
            ("T0", T0),
            ("p0", result.inputs.gas_turbine.p_amb_Pa),
            ("escape_quimica", stack_chem),
        ),
    )


# ---------------------------------------------------------------------
# Diagrama de Grassmann
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GrassmannRow:
    """Una rama del diagrama: lo que se destruye (D) o se pierde (L) en un componente."""

    name: str
    kind: str
    destroyed_W: float
    loss_W: float
    merged: int = 1  # cuántos componentes junta («otros»)


def grassmann_rows(
    plant: PlantExergy, *, min_share: float = 0.005, max_rows: int = 10
) -> list[GrassmannRow]:
    """Las ramas del diagrama de Grassmann, en el orden del flujo.

    Los componentes sin destrucción ni pérdida no tienen rama; los que se llevan
    menos de ``min_share`` de la exergía que entra (o los más chicos, si hay
    más de ``max_rows``) se juntan en «Otros».
    """
    rows = [
        GrassmannRow(c.name, c.kind, max(c.destroyed_W, 0.0), max(c.loss_W, 0.0))
        for c in plant.components
        if c.destroyed_W + c.loss_W > 1e-9 * plant.fuel_W
    ]
    total = plant.fuel_W
    small = [r for r in rows if r.destroyed_W + r.loss_W < min_share * total]
    big = [r for r in rows if r not in small]
    if len(big) > max_rows - 1:
        big_sorted = sorted(big, key=lambda r: r.destroyed_W + r.loss_W, reverse=True)
        keep = set(id(r) for r in big_sorted[: max_rows - 1])
        small += [r for r in big if id(r) not in keep]
        big = [r for r in big if id(r) in keep]
    out = [r for r in rows if r in big]
    if small:
        D = sum(r.destroyed_W for r in small)
        L = sum(r.loss_W for r in small)
        out.append(GrassmannRow(f"Otros ({len(small)})", "otros", D, L, len(small)))
    return out


# ---------------------------------------------------------------------
# Notas: interpretación física
# ---------------------------------------------------------------------

#: Qué irreversibilidad destruye exergía en cada tipo de componente.
KIND_EXPLANATIONS: dict[str, str] = {
    "caldera": (
        "el calor pasa de la fuente, mucho más caliente, al agua y al vapor: es transferencia "
        "de calor con una diferencia de temperatura grande (en una caldera real se suma la "
        "combustión)"
    ),
    "recalentador": (
        "el calor pasa de la fuente caliente al vapor con una diferencia de temperatura grande"
    ),
    "turbina": (
        "la fricción y los torbellinos de la expansión (η < 1); una turbina "
        "isoentrópica no destruye exergía"
    ),
    "compresor": "la fricción en la compresión (η < 1)",
    "bomba": "la fricción en la bomba (η < 1); el trabajo de bombeo es chico",
    "condensador": (
        "el calor va al ambiente a T₀ y no lleva exergía: todo lo que el fluido trae se destruye "
        "en la transferencia de calor. Por eso conviene condensar a la menor temperatura posible"
    ),
    "calentador abierto": "la mezcla del vapor de la extracción con el agua más fría",
    "calentador cerrado": (
        "la transferencia de calor del vapor de la extracción al agua con una "
        "diferencia de temperatura"
    ),
    "mezcla": "la mezcla de corrientes a distinta temperatura",
    "desaireador": "la mezcla del vapor de la extracción con el condensado más frío",
    "válvula": (
        "el estrangulamiento: la presión cae sin hacer trabajo (h constante, la entropía crece)"
    ),
    "cañería": "el rozamiento y el calor que se pierde al ambiente",
    "recuperador": (
        "la transferencia de calor entre el escape y el líquido con una diferencia de temperatura"
    ),
    "evaporador": (
        "la diferencia de temperatura entre el espacio refrigerado y el refrigerante que evapora"
    ),
    "cámara de evaporación": "separar las fases a la misma temperatura, que es reversible",
    "intercambiador de la cascada": "la diferencia de temperatura entre los dos refrigerantes",
    "interenfriador": "el calor del aire va al agua de enfriamiento a T₀ y se destruye",
    "regenerador": (
        "la transferencia de calor de los gases al aire con una diferencia de temperatura"
    ),
    "cámara de combustión": (
        "la reacción química y la mezcla de los productos calientes con el aire, muy "
        "irreversibles: es la mayor destrucción de cualquier planta que quema un combustible"
    ),
    "calentador": "el calor entra de una fuente a temperatura finita",
    "sección de la HRSG": (
        "la diferencia de temperatura entre los gases y el agua (el pinch y los extremos)"
    ),
    "escape": "los gases salen calientes y se llevan su exergía",
    "chimenea": "los gases salen calientes y se llevan su exergía",
}


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f} %".replace(".", ",")


def plant_notes(plant: PlantExergy) -> list[str]:
    """Interpretación física de la exergía destruida y perdida (markdown)."""
    notes: list[str] = []
    ranked = sorted(
        (c for c in plant.components if c.destroyed_W + c.loss_W > 0.0),
        key=lambda c: c.destroyed_W + c.loss_W,
        reverse=True,
    )
    for i, c in enumerate(ranked[:3]):
        share = (c.destroyed_W + c.loss_W) / plant.fuel_W
        if share < 0.01:
            break
        what = "se pierde" if c.is_loss else "se destruye"
        lead = "La mayor parte" if i == 0 else "Le sigue: en"
        if i == 0:
            text = f"{lead} de la exergía que entra {what} en **{c.name}** ({_pct(share)})"
        else:
            text = f"{lead} **{c.name}** {what} el {_pct(share)}"
        expl = KIND_EXPLANATIONS.get(c.kind)
        notes.append(text + (f": {expl}." if expl else "."))
    if plant.kind in ("gas_turbine", "combined"):
        if plant.fuel_model == "pci":
            notes.append(
                "Con la exergía del combustible igual al PCI (vademecum §16.13), η_II coincide con "
                "el rendimiento térmico. Con la química de Szargut entra un poco más de exergía "
                "(para el metano, φ = e/PCI ≈ 1,04) y η_II baja en proporción."
            )
        elif plant.fuel_model == "szargut":
            chem = plant.setting("escape_quimica") or 0.0
            notes.append(
                "Con la exergía química del combustible (Szargut et al., 1988) se cuenta también "
                f"la química de los gases que salen: {_pct(chem / plant.fuel_W)} de lo que entra, "
                "por el CO₂ y el agua, más concentrados que en el aire de referencia."
            )
    if plant.kind == "refrigeration":
        notes.append(
            "η_II = Ẋ_producto/Ẇ = COP/COP_rev: lo que falta para el ciclo reversible es la suma "
            "de la exergía destruida en cada componente (Cengel §11-5)."
        )
    if plant.kind == "rankine":
        T_H = plant.setting("T_H")
        if T_H is not None:
            notes.append(
                f"Con la fuente a {_degC(T_H)} la exergía que entra es Q̇·(1 − T₀/T_H). Con una "
                "fuente más fría (más cerca del vapor) la caldera destruiría menos y η_II "
                "subiría con el mismo ciclo: la irreversibilidad está en la diferencia de "
                "temperatura, no en el ciclo."
            )
    return notes


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def plant_exergy_to_dict(plant: PlantExergy, system: UnitSystem) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""

    def w(x: float) -> str:
        return format_quantity(x, "power", system)

    comps: dict[str, Any] = {}
    for c in plant.components:
        comps[c.name] = {
            "tipo": c.kind,
            "F": w(c.fuel_W),
            "P": w(c.product_W),
            "D": w(c.destroyed_W),
            "L": w(c.loss_W),
            "epsilon": c.efficiency,
            "y_D": plant.y_D(c),
            "y_D_estrella": plant.y_star_D(c),
        }
    return {
        "planta": plant.title,
        "T0": format_quantity(plant.T0_K, "temperature", system),
        "p0": format_quantity(plant.p0_Pa, "pressure", system),
        "modelo_del_combustible": plant.fuel_model,
        "entra": {name: w(x) for name, x in plant.inputs},
        "producto": {name: w(x) for name, x in plant.products},
        "destruida_total": w(plant.destroyed_W),
        "perdida_total": w(plant.loss_W),
        "rendimiento_exergetico": plant.efficiency,
        "componentes": comps,
        "corrientes": {
            s.label: {
                "m": format_quantity(s.m_kg_s, "mass_flow", system),
                "psi": format_quantity(s.psi_J_per_kg, "specific_enthalpy", system),
                "e_ch": format_quantity(s.e_ch_J_per_kg, "specific_enthalpy", system),
                "X": w(s.X_W),
            }
            for s in plant.streams
        },
    }
