"""Exergía física: de un estado, de un calor y de una fuente finita (vademecum §11).

- **Estado muerto** (§11.1): el fluido en equilibrio térmico y mecánico con el
  ambiente, a T₀ y p₀.
- **Exergía de una masa** (§11.2), por kg:
  φ = (u − u₀) + p₀·(v − v₀) − T₀·(s − s₀); con la cinética y la potencial,
  x = φ + V²/2 + g·z.
- **Exergía de un flujo** (§11.3), por kg: ψ = (h − h₀) − T₀·(s − s₀); con
  la cinética y la potencial, x = ψ + V²/2 + g·z.
- **Parte térmica y mecánica** (Kotas, 1985): se enfría a p constante hasta T₀
  y después se expande a T₀ hasta p₀ (el estado intermedio es (T₀, p)):
  ψ_T = (h − h_i) − T₀·(s − s_i) y ψ_M = (h_i − h₀) − T₀·(s_i − s₀).
- **Exergía de un calor** (§11.5): X_Q = Q·(1 − T₀/T).
- **Fuente térmica finita** (§11.6): Φ = m·c·[(T − T₀) − T₀·ln(T/T₀)].

Las propiedades salen de CoolProp con :func:`core.fluids.fluid_state_from_pair`
(Bell et al., 2014), con la convención del proyecto.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from core.fluids import (
    FLUID_NAMES_ES,
    FluidState,
    PairCode,
    fluid_state_from_pair,
    saturation_at_temperature,
)
from core.units_system import UnitSystem, format_quantity

__all__ = [
    "AMBIENT_P_RANGE_PA",
    "AMBIENT_T_RANGE_K",
    "FINITE_SOURCE_EXAMPLES",
    "G_STANDARD",
    "HEAT_EXAMPLES",
    "PHYSICAL_EXAMPLES",
    "Ambient",
    "Amount",
    "FiniteSourceExample",
    "FiniteSourceExergy",
    "HeatExample",
    "HeatExergy",
    "PhysicalExample",
    "PhysicalExergy",
    "ProcessExergy",
    "finite_source_exergy",
    "finite_source_notes",
    "heat_exergy",
    "heat_notes",
    "physical_exergy",
    "physical_exergy_to_dict",
    "physical_notes",
]

#: Aceleración de la gravedad estándar (m/s²).
G_STANDARD = 9.80665
#: Rangos razonables del ambiente (los mismos que la psicrometría).
AMBIENT_T_RANGE_K = (223.15, 333.15)
AMBIENT_P_RANGE_PA = (40_000.0, 1_000_000.0)

Amount = Literal["por kg", "masa", "caudal"]


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.1f} °C".replace(".", ",")


@dataclass(frozen=True)
class Ambient:
    """El ambiente: el estado muerto T₀, p₀ (vademecum §11.1).

    Raises
    ------
    ValueError
        Si T₀ no está entre −50 y 60 °C o p₀ entre 0,4 y 10 bar.
    """

    T0_K: float = 298.15
    p0_Pa: float = 101_325.0

    def __post_init__(self) -> None:
        lo, hi = AMBIENT_T_RANGE_K
        if not (math.isfinite(self.T0_K) and lo <= self.T0_K <= hi):
            raise ValueError(
                f"La temperatura del ambiente T₀ = {_degC(self.T0_K)} está fuera de rango: tiene "
                "que estar entre −50 y 60 °C (el ambiente es el aire o el agua que nos rodea)."
            )
        plo, phi = AMBIENT_P_RANGE_PA
        if not (math.isfinite(self.p0_Pa) and plo <= self.p0_Pa <= phi):
            raise ValueError(
                "La presión del ambiente p₀ tiene que estar entre 0,4 y 10 bar (la atmosférica "
                "es 1,013 bar)."
            )


def _dead_state(fluid: str, ambient: Ambient) -> FluidState:
    try:
        return fluid_state_from_pair(fluid, "TP", t=ambient.T0_K, p=ambient.p0_Pa)
    except ValueError as exc:
        name = FLUID_NAMES_ES.get(fluid, fluid)
        raise ValueError(
            f"No se puede armar el estado muerto del {name} a T₀ = {_degC(ambient.T0_K)} y "
            f"p₀ = {ambient.p0_Pa / 1e5:.3g} bar: {exc}".replace(".", ",", 1)
        ) from exc


def _state_at_T0(fluid: str, p_Pa: float, ambient: Ambient, state: FluidState) -> FluidState:
    """El estado intermedio (T₀, p) para separar la parte térmica y la mecánica.

    Si p es justo la de saturación a T₀ (o el estado ya es una mezcla a T₀), el
    estado intermedio no es único; cualquier título da la misma ψ_T porque
    Δh = T₀·Δs a lo largo de la isoterma dentro de la campana: se usa el propio
    estado (si ya está a T₀) o el líquido saturado.
    """
    T0 = ambient.T0_K
    if state.is_two_phase and abs(state.T_K - T0) < 1e-6:
        return state
    sat = None
    try:
        sat = saturation_at_temperature(fluid, T0)
    except ValueError:
        sat = None
    if sat is not None and abs(p_Pa - sat.P_sat_Pa) <= 1e-9 * p_Pa:
        return fluid_state_from_pair(fluid, "PX", p=p_Pa, x=0.0)
    return fluid_state_from_pair(fluid, "TP", t=T0, p=p_Pa)


@dataclass(frozen=True)
class PhysicalExergy:
    """Exergía física de un estado (vademecum §11.2 y §11.3), en SI.

    ``dead`` es el fluido a (T₀, p₀) e ``intermediate`` a (T₀, p). ``mass_kg``
    (sistema cerrado) o ``m_dot_kg_s`` (flujo) dan los totales.
    """

    fluid: str
    state: FluidState
    dead: FluidState
    intermediate: FluidState
    ambient: Ambient
    speed_m_per_s: float = 0.0
    height_m: float = 0.0
    amount: Amount = "por kg"
    mass_kg: float | None = None
    m_dot_kg_s: float | None = None

    @property
    def T0_K(self) -> float:
        return self.ambient.T0_K

    @property
    def p0_Pa(self) -> float:
        return self.ambient.p0_Pa

    # --- flujo ----------------------------------------------------------
    @property
    def dh_J_per_kg(self) -> float:
        return self.state.h_J_per_kg - self.dead.h_J_per_kg

    @property
    def ds_J_per_kg_K(self) -> float:
        return self.state.s_J_per_kg_K - self.dead.s_J_per_kg_K

    @property
    def psi_J_per_kg(self) -> float:
        """ψ = (h − h₀) − T₀·(s − s₀) (§11.3)."""
        return self.dh_J_per_kg - self.T0_K * self.ds_J_per_kg_K

    @property
    def psi_thermal_J_per_kg(self) -> float:
        i = self.intermediate
        return (self.state.h_J_per_kg - i.h_J_per_kg) - self.T0_K * (
            self.state.s_J_per_kg_K - i.s_J_per_kg_K
        )

    @property
    def psi_mechanical_J_per_kg(self) -> float:
        i = self.intermediate
        return (i.h_J_per_kg - self.dead.h_J_per_kg) - self.T0_K * (
            i.s_J_per_kg_K - self.dead.s_J_per_kg_K
        )

    # --- masa -----------------------------------------------------------
    @property
    def du_J_per_kg(self) -> float:
        return self.state.u_J_per_kg - self.dead.u_J_per_kg

    @property
    def dv_m3_per_kg(self) -> float:
        return self.state.v_m3_per_kg - self.dead.v_m3_per_kg

    @property
    def phi_J_per_kg(self) -> float:
        """φ = (u − u₀) + p₀·(v − v₀) − T₀·(s − s₀) (§11.2), siempre ≥ 0."""
        return self.du_J_per_kg + self.p0_Pa * self.dv_m3_per_kg - self.T0_K * self.ds_J_per_kg_K

    # --- cinética y potencial -------------------------------------------
    @property
    def ke_J_per_kg(self) -> float:
        return 0.5 * self.speed_m_per_s**2

    @property
    def pe_J_per_kg(self) -> float:
        return G_STANDARD * self.height_m

    @property
    def x_flow_J_per_kg(self) -> float:
        """ψ + V²/2 + g·z: exergía de flujo total por kg."""
        return self.psi_J_per_kg + self.ke_J_per_kg + self.pe_J_per_kg

    @property
    def x_mass_J_per_kg(self) -> float:
        """φ + V²/2 + g·z: exergía de una masa por kg."""
        return self.phi_J_per_kg + self.ke_J_per_kg + self.pe_J_per_kg

    @property
    def anergy_J_per_kg(self) -> float:
        """T₀·(s − s₀): la parte de h − h₀ que no se puede convertir en trabajo."""
        return self.T0_K * self.ds_J_per_kg_K

    # --- totales --------------------------------------------------------
    @property
    def X_J(self) -> float | None:
        """Exergía de la masa (J), con ``mass_kg``."""
        return None if self.mass_kg is None else self.mass_kg * self.x_mass_J_per_kg

    @property
    def X_W(self) -> float | None:
        """Exergía del flujo (W), con ``m_dot_kg_s``."""
        return None if self.m_dot_kg_s is None else self.m_dot_kg_s * self.x_flow_J_per_kg

    @property
    def is_dead_state(self) -> bool:
        return (
            abs(self.state.T_K - self.T0_K) < 1e-6
            and abs(self.state.P_Pa - self.p0_Pa) < 1e-6 * self.p0_Pa
            and self.speed_m_per_s == 0.0
            and self.height_m == 0.0
        )


def physical_exergy(
    fluid: str,
    pair: PairCode,
    *,
    ambient: Ambient | None = None,
    speed_m_per_s: float = 0.0,
    height_m: float = 0.0,
    amount: Amount = "por kg",
    mass_kg: float | None = None,
    m_dot_kg_s: float | None = None,
    **values: float,
) -> PhysicalExergy:
    """Exergía física de ``fluid`` en el estado dado por ``pair`` (los kwargs en SI).

    Raises
    ------
    ValueError
        Si el estado o el estado muerto no se pueden calcular, si la velocidad o
        la altura no son razonables, o si la masa o el caudal no son positivos.
    """
    ambient = ambient or Ambient()
    if not (math.isfinite(speed_m_per_s) and 0.0 <= speed_m_per_s <= 1000.0):
        raise ValueError("La velocidad tiene que estar entre 0 y 1000 m/s.")
    if not (math.isfinite(height_m) and -1.0e4 <= height_m <= 1.0e5):
        raise ValueError(
            "La altura (sobre el nivel de referencia del ambiente) está fuera de rango."
        )
    if amount == "masa" and not (mass_kg is not None and mass_kg > 0.0):
        raise ValueError("La masa tiene que ser positiva.")
    if amount == "caudal" and not (m_dot_kg_s is not None and m_dot_kg_s > 0.0):
        raise ValueError("El caudal tiene que ser positivo.")
    state = fluid_state_from_pair(fluid, pair, **values)
    dead = _dead_state(fluid, ambient)
    intermediate = _state_at_T0(fluid, state.P_Pa, ambient, state)
    return PhysicalExergy(
        fluid=fluid,
        state=state,
        dead=dead,
        intermediate=intermediate,
        ambient=ambient,
        speed_m_per_s=speed_m_per_s,
        height_m=height_m,
        amount=amount,
        mass_kg=mass_kg if amount == "masa" else None,
        m_dot_kg_s=m_dot_kg_s if amount == "caudal" else None,
    )


@dataclass(frozen=True)
class ProcessExergy:
    """Dos estados del mismo fluido con el mismo ambiente: el proceso 1 → 2.

    En régimen permanente y sin calor (o con calor a T₀), el trabajo mínimo para
    llevar el flujo de 1 a 2 es Δψ = ψ₂ − ψ₁ (Cengel cap. 8): si es positivo,
    hay que entregar al menos ese trabajo (un compresor); si es negativo, es el
    máximo que se puede obtener (una turbina).
    """

    first: PhysicalExergy
    second: PhysicalExergy

    def __post_init__(self) -> None:
        if self.first.fluid != self.second.fluid or self.first.ambient != self.second.ambient:
            raise ValueError("Los dos estados tienen que ser del mismo fluido y ambiente.")

    @property
    def dpsi_J_per_kg(self) -> float:
        return self.second.x_flow_J_per_kg - self.first.x_flow_J_per_kg

    @property
    def dphi_J_per_kg(self) -> float:
        return self.second.x_mass_J_per_kg - self.first.x_mass_J_per_kg

    @property
    def dh_J_per_kg(self) -> float:
        return self.second.state.h_J_per_kg - self.first.state.h_J_per_kg


# ---------------------------------------------------------------------
# Calor y fuente finita
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HeatExergy:
    """Exergía de un calor que pasa a temperatura T (vademecum §11.5).

    ``Q_W`` es el calor que recibe el sistema desde la fuente a T (una
    potencia). X_Q = Q·(1 − T₀/T): con T > T₀ tiene el signo de Q; con T < T₀,
    el contrario (sacarle calor a una fuente fría genera exergía; dárselo, la
    destruye). La anergía es B_Q = Q·T₀/T.
    """

    Q_W: float
    T_K: float
    T0_K: float

    @property
    def carnot(self) -> float:
        return 1.0 - self.T0_K / self.T_K

    @property
    def X_W(self) -> float:
        return self.Q_W * self.carnot

    @property
    def anergy_W(self) -> float:
        return self.Q_W * self.T0_K / self.T_K


def heat_exergy(Q_W: float, T_K: float, T0_K: float) -> HeatExergy:
    """Exergía de un calor ``Q_W`` a ``T_K`` con el ambiente a ``T0_K``.

    Raises
    ------
    ValueError
        Si las temperaturas no son positivas o Q no es finito.
    """
    if not (math.isfinite(T_K) and T_K > 0.0):
        raise ValueError("La temperatura de la fuente tiene que ser positiva (en K).")
    lo, hi = AMBIENT_T_RANGE_K
    if not (math.isfinite(T0_K) and lo <= T0_K <= hi):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −50 y 60 °C.")
    if not math.isfinite(Q_W):
        raise ValueError("El calor tiene que ser un número.")
    return HeatExergy(Q_W, T_K, T0_K)


@dataclass(frozen=True)
class FiniteSourceExergy:
    """Exergía de un cuerpo de capacidad calorífica m·c a T (vademecum §11.6).

    Φ = m·c·[(T − T₀) − T₀·ln(T/T₀)] ≥ 0, tanto si está más caliente que el
    ambiente (se extrae trabajo enfriándolo hasta T₀) como si está más frío (se
    lo calienta con el ambiente como fuente caliente). Es la integral del
    factor de Carnot (1 − T₀/θ) a lo largo del enfriamiento.
    """

    m_kg: float
    c_J_per_kg_K: float
    T_K: float
    T0_K: float

    @property
    def mc_J_per_K(self) -> float:
        return self.m_kg * self.c_J_per_kg_K

    @property
    def Q_J(self) -> float:
        """Calor que entrega hasta llegar a T₀ (negativo si está más frío)."""
        return self.mc_J_per_K * (self.T_K - self.T0_K)

    @property
    def anergy_J(self) -> float:
        """T₀·ΔS: la parte del calor que no se puede convertir, m·c·T₀·ln(T/T₀)."""
        return self.mc_J_per_K * self.T0_K * math.log(self.T_K / self.T0_K)

    @property
    def Phi_J(self) -> float:
        return self.Q_J - self.anergy_J

    @property
    def fraction_of_heat(self) -> float | None:
        """Φ/|Q|: qué parte del calor se podría convertir en trabajo."""
        return None if self.Q_J == 0.0 else self.Phi_J / abs(self.Q_J)

    def curve(self, n: int = 61) -> list[tuple[float, float]]:
        """(T, Φ) entre ~0,5·T₀ y la mayor de 2·T₀ y 1,2·T, para el gráfico."""
        lo = 0.5 * self.T0_K
        hi = max(2.0 * self.T0_K, 1.2 * self.T_K)
        out = []
        for k in range(n):
            T = lo + (hi - lo) * k / (n - 1)
            out.append(
                (T, self.mc_J_per_K * ((T - self.T0_K) - self.T0_K * math.log(T / self.T0_K)))
            )
        return out


def finite_source_exergy(
    m_kg: float, c_J_per_kg_K: float, T_K: float, T0_K: float
) -> FiniteSourceExergy:
    """Exergía de un cuerpo (o un tanque de agua) a ``T_K`` con el ambiente a ``T0_K``.

    Raises
    ------
    ValueError
        Si la masa, el calor específico o las temperaturas no son positivos.
    """
    if not (math.isfinite(m_kg) and m_kg > 0.0):
        raise ValueError("La masa tiene que ser positiva.")
    if not (math.isfinite(c_J_per_kg_K) and c_J_per_kg_K > 0.0):
        raise ValueError("El calor específico tiene que ser positivo.")
    if not (math.isfinite(T_K) and T_K > 0.0):
        raise ValueError("La temperatura del cuerpo tiene que ser positiva (en K).")
    lo, hi = AMBIENT_T_RANGE_K
    if not (math.isfinite(T0_K) and lo <= T0_K <= hi):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −50 y 60 °C.")
    return FiniteSourceExergy(m_kg, c_J_per_kg_K, T_K, T0_K)


# ---------------------------------------------------------------------
# Notas didácticas
# ---------------------------------------------------------------------


def _kj(x: float) -> str:
    return f"{x / 1e3:.4g}".replace(".", ",")


def physical_notes(result: PhysicalExergy) -> list[str]:
    """Observaciones sobre la exergía de un estado (markdown)."""
    notes: list[str] = []
    st, dead = result.state, result.dead
    if result.is_dead_state:
        notes.append("El estado es el estado muerto: su exergía es cero (vademecum §11.2).")
        return notes
    if st.T_K < result.T0_K - 0.5:
        notes.append(
            f"El fluido está **más frío que el ambiente** ({_degC(st.T_K)} contra "
            f"{_degC(result.T0_K)}) y aun así tiene exergía positiva: se podría hacer funcionar "
            "una máquina térmica usando el ambiente como fuente caliente y el fluido como fría. "
            "Por eso el «frío» de una cámara frigorífica tiene valor."
        )
    if result.psi_mechanical_J_per_kg < 0.0:
        notes.append(
            "La parte **mecánica** es negativa: la presión está por debajo de p₀ y habría que "
            "gastar trabajo para llevar el fluido a la presión del ambiente. En un flujo, ψ "
            "puede ser negativa (vademecum §11.3); φ, la de una masa, nunca."
        )
    if result.psi_J_per_kg < 0.0:
        notes.append(
            "ψ < 0: un flujo en este estado necesita trabajo para salir al ambiente (por ejemplo, "
            "el vacío de un condensador). La exergía de la masa φ sigue siendo positiva porque "
            "suma p₀·(v − v₀), el trabajo que hace la atmósfera."
        )
    vacuum = result.p0_Pa * result.dv_m3_per_kg
    if vacuum > 0.5 * abs(result.phi_J_per_kg) and result.phi_J_per_kg > 2.0 * result.psi_J_per_kg:
        notes.append(
            f"φ ({_kj(result.phi_J_per_kg)} kJ/kg) es mucho mayor que ψ: a baja presión el fluido "
            f"ocupa mucho volumen y la atmósfera, al comprimirlo, haría el trabajo "
            f"p₀·(v − v₀) = {_kj(vacuum)} kJ/kg. Es la exergía de un espacio vacío (vademecum "
            "§11.4); en un flujo ese término no aparece porque el fluido ya empuja al que sigue."
        )
    if dead.region in ("compressed_liquid", "saturated_liquid") and st.region in (
        "superheated_vapor",
        "saturated_mixture",
        "saturated_vapor",
    ):
        notes.append(
            f"El estado muerto es **líquido** ({_degC(dead.T_K)}): para llegar al equilibrio con "
            "el ambiente el vapor tiene que condensar, y la mayor parte de h − h₀ es calor latente "
            "a una temperatura cercana a T₀, con poca exergía. Por eso ψ es mucho menor que "
            "h − h₀."
        )
    if result.speed_m_per_s > 0.0 or result.height_m != 0.0:
        notes.append(
            "La energía cinética y la potencial son **exergía pura**: se pueden convertir "
            "íntegramente en trabajo (una turbina eólica ideal, una central hidráulica)."
        )
    frac = result.psi_J_per_kg / result.dh_J_per_kg if abs(result.dh_J_per_kg) > 1.0 else None
    if frac is not None and 0.0 < frac < 1.0 and st.T_K > result.T0_K:
        notes.append(
            f"De h − h₀ = {_kj(result.dh_J_per_kg)} kJ/kg solo {100 * frac:.0f} % es exergía; el "
            f"resto, T₀·(s − s₀) = {_kj(result.anergy_J_per_kg)} kJ/kg, es anergía: energía que "
            "termina en el ambiente sin poder convertirse en trabajo."
        )
    return notes


def heat_notes(result: HeatExergy) -> list[str]:
    """Observaciones sobre la exergía de un calor (markdown)."""
    notes: list[str] = []
    if result.T_K < result.T0_K:
        notes.append(
            "La fuente está **más fría que el ambiente**: el factor de Carnot es negativo y la "
            "exergía va en sentido contrario al calor. Sacarle calor a una cámara fría le "
            "entrega exergía a la cámara (por eso cuesta trabajo); dárselo, la destruye."
        )
    elif result.carnot < 0.2:
        notes.append(
            "Con la fuente cerca de T₀ el factor de Carnot es chico: ese calor vale poco como "
            "trabajo aunque sea mucha energía (el calor de baja temperatura)."
        )
    if result.T_K > 1000.0:
        notes.append(
            "Un calor a muy alta temperatura (un hogar, una llama) es casi todo exergía: si se "
            "lo usa para calentar algo mucho más frío (el agua de una caldera) se destruye gran "
            "parte, por la diferencia de temperaturas."
        )
    return notes


def finite_source_notes(result: FiniteSourceExergy) -> list[str]:
    """Observaciones sobre la exergía de una fuente finita (markdown)."""
    notes: list[str] = []
    frac = result.fraction_of_heat
    if frac is not None:
        notes.append(
            f"Solo el {100 * frac:.1f} % del calor que el cuerpo intercambia hasta llegar a T₀ se "
            "podría convertir en trabajo: a medida que se enfría (o se calienta) su temperatura se "
            "acerca a T₀ y el factor de Carnot cae a cero."
        )
    if result.T_K < result.T0_K:
        notes.append(
            "El cuerpo está más frío que el ambiente y tiene exergía: el ambiente hace de fuente "
            "caliente de una máquina que lo calienta hasta T₀ (el hielo «guarda» trabajo)."
        )
    return notes


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalExample:
    """Un estado de ejemplo (valores en SI) y, opcional, un segundo estado (1 → 2)."""

    fluid: str
    pair: PairCode
    values: dict[str, float]
    ambient: Ambient = field(default_factory=Ambient)
    speed_m_per_s: float = 0.0
    height_m: float = 0.0
    amount: Amount = "por kg"
    mass_kg: float | None = None
    m_dot_kg_s: float | None = None
    second: tuple[PairCode, dict[str, float]] | None = None
    note: str = ""


_CENGEL_10_8 = Ambient(290.0, 100_000.0)

PHYSICAL_EXAMPLES: dict[str, PhysicalExample] = {
    "Vapor que entra a la turbina (Cengel 10-8: 3 MPa y 350 °C; ambiente a 290 K y 100 kPa)": (
        PhysicalExample(
            "Water",
            "TP",
            {"t": 623.15, "p": 3.0e6},
            _CENGEL_10_8,
            second=("PS", {"p": 75.0e3, "s": 6745.0}),
            note=(
                "Cengel 10-8 (el ciclo de 10-1): el vapor sale de la turbina isoentrópica a "
                "75 kPa con ψ₄ ≈ 449 kJ/kg; la diferencia es el trabajo de la turbina."
            ),
        )
    ),
    "Aire comprimido en un tanque (Cengel cap. 8: 200 m³ a 1 MPa y 300 K)": PhysicalExample(
        "Air",
        "TP",
        {"t": 300.0, "p": 1.0e6},
        Ambient(300.0, 100_000.0),
        amount="masa",
        mass_kg=2323.0,
        note=(
            "m = p·V/(R·T) = 2323 kg con gas ideal. Cengel obtiene ≈ 281 MJ con gas ideal; "
            "CoolProp trata al aire como gas real."
        ),
    ),
    "R-134a en un compresor (Cengel cap. 8: de 0,14 MPa y −10 °C a 0,8 MPa y 50 °C)": (
        PhysicalExample(
            "R134a",
            "TP",
            {"t": 263.15, "p": 0.14e6},
            Ambient(293.15, 95_000.0),
            second=("TP", {"t": 323.15, "p": 0.8e6}),
            note=(
                "Cengel obtiene Δψ ≈ 38,0 kJ/kg: es el trabajo mínimo del compresor (el real "
                "es h₂ − h₁ ≈ 40,3 kJ/kg)."
            ),
        )
    ),
    "Agua fría de un chiller (5 °C, 1 atm; ambiente a 25 °C)": PhysicalExample(
        "Water", "TP", {"t": 278.15, "p": 101_325.0}, amount="caudal", m_dot_kg_s=10.0
    ),
    "Viento sobre un rotor de 12 m (Cengel cap. 8: 10 m/s; ρ = 1,25 kg/m³)": PhysicalExample(
        "Air",
        "TP",
        {"t": 298.15, "p": 101_325.0},
        speed_m_per_s=10.0,
        amount="caudal",
        m_dot_kg_s=1414.0,
        note="ṁ = ρ·V·A = 1,25·10·π·6² ≈ 1414 kg/s; Cengel obtiene ≈ 70,7 kW.",
    ),
    "Vapor de baja presión de un condensador (10 kPa, x = 0,9)": PhysicalExample(
        "Water", "PX", {"p": 10.0e3, "x": 0.9}
    ),
}


@dataclass(frozen=True)
class HeatExample:
    Q_W: float
    T_K: float
    T0_K: float
    note: str = ""


HEAT_EXAMPLES: dict[str, HeatExample] = {
    "Hogar a 2000 R que entrega 3000 Btu/s (Cengel cap. 8; ambiente a 77 °F)": HeatExample(
        3000.0 * 1055.05585262,
        2000.0 * 5.0 / 9.0,
        298.15,
        "Cengel obtiene ≈ 2195 Btu/s.",
    ),
    "Calor de una fuente geotérmica a 150 °C (1 MW; ambiente a 25 °C)": HeatExample(
        1.0e6, 423.15, 298.15
    ),
    "Cámara frigorífica: sacar 1 kW a −18 °C (ambiente a 25 °C)": HeatExample(
        1.0e3, 255.15, 298.15
    ),
}


@dataclass(frozen=True)
class FiniteSourceExample:
    m_kg: float
    c_J_per_kg_K: float
    T_K: float
    T0_K: float
    note: str = ""


FINITE_SOURCE_EXAMPLES: dict[str, FiniteSourceExample] = {
    "Bloque de hierro a 200 °C (Cengel cap. 8: 500 kg; ambiente a 27 °C)": FiniteSourceExample(
        500.0, 450.0, 473.0, 300.0, "Cengel usa 473 K y 300 K y obtiene ≈ 8191 kJ."
    ),
    "Tanque de agua caliente (1000 kg a 80 °C; ambiente a 20 °C)": FiniteSourceExample(
        1000.0, 4180.0, 353.15, 293.15
    ),
    "Agua helada (1000 kg a 5 °C; ambiente a 25 °C)": FiniteSourceExample(
        1000.0, 4180.0, 278.15, 298.15
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def physical_exergy_to_dict(
    result: PhysicalExergy, system: UnitSystem, second: PhysicalExergy | None = None
) -> dict[str, Any]:
    """Resultado → dict serializable (export CSV/JSON)."""

    def h(x: float) -> str:
        return format_quantity(x, "specific_enthalpy", system)

    def state_dict(s: FluidState) -> dict[str, str]:
        return {
            "T": format_quantity(s.T_K, "temperature", system),
            "p": format_quantity(s.P_Pa, "pressure", system),
            "h": h(s.h_J_per_kg),
            "s": format_quantity(s.s_J_per_kg_K, "specific_entropy", system),
            "u": h(s.u_J_per_kg),
            "v": format_quantity(s.v_m3_per_kg, "specific_volume", system),
        }

    data: dict[str, Any] = {
        "fluido": FLUID_NAMES_ES.get(result.fluid, result.fluid),
        "ambiente": {
            "T0": format_quantity(result.T0_K, "temperature", system),
            "p0": format_quantity(result.p0_Pa, "pressure", system),
        },
        "estado": state_dict(result.state),
        "estado_muerto": state_dict(result.dead),
        "estado_intermedio_T0_p": state_dict(result.intermediate),
        "psi": h(result.psi_J_per_kg),
        "psi_termica": h(result.psi_thermal_J_per_kg),
        "psi_mecanica": h(result.psi_mechanical_J_per_kg),
        "phi": h(result.phi_J_per_kg),
        "cinetica": h(result.ke_J_per_kg),
        "potencial": h(result.pe_J_per_kg),
        "x_flujo": h(result.x_flow_J_per_kg),
        "x_masa": h(result.x_mass_J_per_kg),
    }
    if result.X_J is not None:
        data["masa"] = format_quantity(result.mass_kg or 0.0, "mass", system)
        data["X_masa"] = format_quantity(result.X_J, "energy", system)
    if result.X_W is not None:
        data["caudal"] = format_quantity(result.m_dot_kg_s or 0.0, "mass_flow", system)
        data["X_flujo"] = format_quantity(result.X_W, "power", system)
    if second is not None:
        proc = ProcessExergy(result, second)
        data["estado_2"] = state_dict(second.state)
        data["psi_2"] = h(second.psi_J_per_kg)
        data["delta_psi"] = h(proc.dpsi_J_per_kg)
    return data
