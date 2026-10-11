"""El segundo principio: máquinas, Clausius, el ciclo de Carnot y la entropía del calor.

Fase 10.2. Cuatro cálculos (vademecum §9, §10.2 y §11):

- :func:`solve_machine`: una máquina térmica, un refrigerador o una bomba de calor entre
  dos fuentes, con dos datos de Q_H, Q_C, W y el rendimiento (o el COP), o uno con la
  máquina reversible. El primer principio da lo que falta y el segundo, el veredicto: el
  rendimiento contra el de Carnot (§9.2 y §9.3), la desigualdad de Clausius (§9.1), la
  entropía generada (§10.2), la exergía destruida (§11.8) y el rendimiento exergético
  (§11.10).
- :func:`solve_clausius`: un ciclo que intercambia calor con N fuentes (§9.1).
- :func:`solve_carnot`: el ciclo de Carnot de un fluido real o de un gas ideal entre T_H y
  T_C, proceso por proceso, en un sistema cerrado o en flujo estacionario.
- :func:`solve_heat_path`: la entropía que genera un calor al pasar de una temperatura a
  otra: entre dos fuentes o a través de una pared con sus capas de aire (Çengel y Boles
  §7-2 y §7-13).

Convenciones del vademecum: en las máquinas Q_H, Q_C y W son magnitudes (positivas, §9.2 y
§9.3); en Clausius Q_k > 0 es el calor que el ciclo recibe de la fuente k y Q_k < 0 el que
le entrega (§9.1). Las temperaturas son absolutas. Los calores y el trabajo van por ciclo
(J) o por unidad de tiempo (W): las ecuaciones son las mismas. Todo en SI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from core.balances.common import Verdict, deg_c, num, snap_sgen, verdict
from core.balances.substance import (
    Substance,
    ThermoState,
    fluid_substance,
    ideal_gas_substance,
    state,
)

__all__ = [
    "CARNOT_EXAMPLES",
    "CARNOT_PROPS",
    "CLAUSIUS_EXAMPLES",
    "FLOWS",
    "HEAT_PATH_EXAMPLES",
    "MACHINES",
    "MACHINE_EXAMPLES",
    "PATH_POINTS",
    "PERFORMANCE_NAMES",
    "REVERSIBLE_NAMES",
    "T_DEFAULT_K",
    "CarnotExample",
    "CarnotFlow",
    "CarnotInputs",
    "CarnotProcess",
    "CarnotProp",
    "CarnotResult",
    "ClausiusExample",
    "ClausiusInputs",
    "ClausiusResult",
    "HeatPathExample",
    "HeatPathInputs",
    "HeatPathResult",
    "HeatZone",
    "MachineDatum",
    "MachineExample",
    "MachineInputs",
    "MachineKind",
    "MachineResult",
    "Reservoir",
    "carnot_performance",
    "carnot_to_dict",
    "clausius_to_dict",
    "heat_path_to_dict",
    "machine_to_dict",
    "solve_carnot",
    "solve_clausius",
    "solve_heat_path",
    "solve_machine",
]

#: El ambiente por defecto (25 °C) para la exergía destruida.
T_DEFAULT_K: float = 298.15
_C = 273.15


# ---------------------------------------------------------------------
# Máquina térmica, refrigerador o bomba de calor
# ---------------------------------------------------------------------

MachineKind = Literal["engine", "refrigerator", "heat_pump"]
MACHINES: dict[MachineKind, str] = {
    "engine": "Máquina térmica",
    "refrigerator": "Refrigerador",
    "heat_pump": "Bomba de calor",
}
#: El símbolo del rendimiento de cada máquina (vademecum §9.2 y §9.3).
PERFORMANCE_NAMES: dict[MachineKind, str] = {
    "engine": "η",
    "refrigerator": "COP_R",
    "heat_pump": "COP_B",
}
#: El de la máquina reversible entre las mismas fuentes.
REVERSIBLE_NAMES: dict[MachineKind, str] = {
    "engine": "η_Carnot",
    "refrigerator": "COP_R,rev",
    "heat_pump": "COP_B,rev",
}
MachineDatum = Literal["Q_H", "Q_C", "W", "performance", "reversible"]


def carnot_performance(kind: MachineKind, T_H: float, T_C: float) -> float:
    """El rendimiento o el COP de la máquina reversible entre T_H y T_C (K).

    Vademecum §9.2 (η_Carnot = 1 − T_C/T_H) y §9.3 (COP_R,rev = T_C/(T_H − T_C) y
    COP_B,rev = T_H/(T_H − T_C)); Çengel y Boles §6-10 y §6-11.
    """
    if kind == "engine":
        return 1.0 - T_C / T_H
    if kind == "refrigerator":
        return T_C / (T_H - T_C)
    return T_H / (T_H - T_C)


@dataclass(frozen=True)
class MachineInputs:
    """Una máquina entre una fuente caliente a T_H y una fría a T_C (K).

    Hacen falta dos datos: dos de Q_H, Q_C y W, o uno de ellos y el rendimiento (``η`` de
    la máquina térmica, ``COP`` del refrigerador o de la bomba de calor), o uno de ellos con
    ``reversible`` (la máquina de Carnot, cuyo rendimiento es el de Carnot). Q_H, Q_C y W
    son magnitudes, por ciclo (J) o por unidad de tiempo (W, con ``rates``). ``T0_K`` es el
    ambiente para la exergía; por defecto, la fuente que hace de ambiente (la fría de la
    máquina térmica y de la bomba de calor, la caliente del refrigerador).
    """

    kind: MachineKind
    T_H_K: float
    T_C_K: float
    Q_H: float | None = None
    Q_C: float | None = None
    W: float | None = None
    performance: float | None = None
    reversible: bool = False
    rates: bool = False
    T0_K: float | None = None

    @property
    def data(self) -> tuple[MachineDatum, ...]:
        """Los datos que se dieron, en orden."""
        given: list[MachineDatum] = []
        if self.Q_H is not None:
            given.append("Q_H")
        if self.Q_C is not None:
            given.append("Q_C")
        if self.W is not None:
            given.append("W")
        if self.performance is not None:
            given.append("performance")
        if self.reversible:
            given.append("reversible")
        return tuple(given)

    @property
    def T0(self) -> float:
        if self.T0_K is not None:
            return self.T0_K
        return self.T_H_K if self.kind == "refrigerator" else self.T_C_K


@dataclass(frozen=True)
class MachineResult:
    """Los tres términos del primer principio, el rendimiento contra el de Carnot y la
    entropía generada (en las fuentes: el ciclo vuelve a su estado)."""

    inputs: MachineInputs
    Q_H: float
    Q_C: float
    W: float
    performance: float
    performance_rev: float
    S_gen: float  # J/K por ciclo o W/K
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def kind(self) -> MachineKind:
        return self.inputs.kind

    @property
    def T0(self) -> float:
        return self.inputs.T0

    @property
    def clausius_sum(self) -> float:
        """ΣQ/T del ciclo con el signo del vademecum §9.1 (= −S_gen)."""
        return -self.S_gen

    @property
    def X_dest(self) -> float:
        """La exergía destruida: T₀·S_gen (vademecum §11.8)."""
        return self.T0 * self.S_gen

    @property
    def ratio(self) -> float:
        """El rendimiento sobre el de Carnot (el rendimiento de segundo principio cuando el
        ambiente es la fuente de afuera, Çengel §8-3)."""
        return self.performance / self.performance_rev

    @property
    def W_ideal(self) -> float:
        """El trabajo de la máquina reversible: el máximo con el mismo Q_H (máquina térmica)
        o el mínimo con el mismo Q_C (refrigerador) o Q_H (bomba de calor)."""
        if self.kind == "engine":
            return self.performance_rev * self.Q_H
        if self.kind == "refrigerator":
            return self.Q_C / self.performance_rev
        return self.Q_H / self.performance_rev

    @property
    def lost_work(self) -> float:
        """Lo que se pierde contra la reversible: W_ideal − W (máquina térmica) o
        W − W_ideal (las otras). Es T·S_gen con T la fuente que hace de ambiente."""
        if self.kind == "engine":
            return self.W_ideal - self.W
        return self.W - self.W_ideal

    @property
    def X_supplied(self) -> float:
        """La exergía que se le da a la máquina (vademecum §11.10)."""
        if self.kind == "engine":
            return self.Q_H * (1.0 - self.T0 / self.inputs.T_H_K)
        return self.W

    @property
    def X_recovered(self) -> float:
        """La exergía que entrega: W, Q_C·(T₀/T_C − 1) o Q_H·(1 − T₀/T_H) (§11.10)."""
        if self.kind == "engine":
            return self.W
        if self.kind == "refrigerator":
            return self.Q_C * (self.T0 / self.inputs.T_C_K - 1.0)
        return self.Q_H * (1.0 - self.T0 / self.inputs.T_H_K)

    @property
    def eta_ex(self) -> float:
        """El rendimiento exergético (vademecum §11.10)."""
        return self.X_recovered / self.X_supplied

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen)


def _check_temperature(T: float, name: str) -> None:
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError(f"La temperatura {name} tiene que ser positiva (absoluta).")


def _check_reservoirs(T_H: float, T_C: float) -> None:
    _check_temperature(T_H, "de la fuente caliente T_H")
    _check_temperature(T_C, "de la fuente fría T_C")
    if T_H <= T_C:
        raise ValueError(
            f"La fuente caliente tiene que estar más caliente que la fría: T_H = {deg_c(T_H)} y "
            f"T_C = {deg_c(T_C)}. Entre dos fuentes a la misma temperatura ninguna máquina "
            "funciona (η_Carnot = 0)."
        )


_DATA_NAMES = {
    "engine": "Q_H, Q_C, W y η",
    "refrigerator": "Q_H, Q_C, W y COP_R",
    "heat_pump": "Q_H, Q_C, W y COP_B",
}


def _check_data(inputs: MachineInputs) -> None:
    kind = inputs.kind
    if kind not in MACHINES:
        raise ValueError(f"La máquina {kind!r} no existe: {', '.join(MACHINES)}.")
    perf = PERFORMANCE_NAMES[kind]
    if inputs.reversible and inputs.performance is not None:
        raise ValueError(
            f"Si la máquina es reversible su {perf} es el de Carnot: no lo des como dato."
        )
    n = len(inputs.data)
    if n != 2:
        what = "uno" if n == 1 else str(n)
        raise ValueError(
            f"Hacen falta dos datos de {_DATA_NAMES[kind]} (o uno, si la máquina es "
            f"reversible): diste {what}."
        )
    for name in ("Q_H", "Q_C", "W", "performance"):
        value = getattr(inputs, name)
        if value is not None and not math.isfinite(value):
            raise ValueError(f"Falta el valor de {perf if name == 'performance' else name}.")
    if inputs.Q_H is not None and inputs.Q_H <= 0.0:
        raise ValueError(
            "Q_H, el calor que se intercambia con la fuente caliente, tiene que ser mayor que cero."
        )
    if inputs.Q_C is not None and inputs.Q_C < 0.0:
        raise ValueError(
            "Q_C no puede ser negativo: es una magnitud (el sentido lo da la máquina)."
        )
    if inputs.W is not None and inputs.W <= 0.0:
        verb = "entrega" if kind == "engine" else "consume"
        raise ValueError(f"W, el trabajo que la máquina {verb}, tiene que ser mayor que cero.")
    p = inputs.performance
    if p is None:
        return
    if kind == "engine":
        if p <= 0.0:
            raise ValueError("El rendimiento η tiene que ser mayor que cero.")
        if p > 1.0:
            raise ValueError(
                "Un rendimiento mayor que 1 (100 %) viola el primer principio: el trabajo "
                "W = Q_H − Q_C no puede superar al calor que entra (vademecum §9.2)."
            )
    elif kind == "refrigerator":
        if p <= 0.0:
            raise ValueError("El COP de un refrigerador tiene que ser mayor que cero.")
    elif p < 1.0:
        raise ValueError(
            "El COP de una bomba de calor es al menos 1: COP_B = COP_R + 1 (vademecum §9.3). "
            "Con COP_B = 1 es una resistencia eléctrica (todo el trabajo se vuelve calor)."
        )


def _energies(
    inputs: MachineInputs, p: float | None
) -> tuple[float, float, float]:  # (Q_H, Q_C, W)
    """El primer principio (W = Q_H − Q_C) con la definición del rendimiento o del COP."""
    kind = inputs.kind
    Q_H, Q_C, W = inputs.Q_H, inputs.Q_C, inputs.W
    if p is None:  # dos de los tres
        if Q_H is None:
            assert Q_C is not None and W is not None
            return Q_C + W, Q_C, W
        if Q_C is None:
            assert W is not None
            if W > Q_H:
                raise ValueError(
                    "W no puede ser mayor que Q_H: el calor cedido Q_C = Q_H − W saldría "
                    "negativo (primer principio)."
                )
            return Q_H, Q_H - W, W
        if Q_C >= Q_H:
            raise ValueError(
                "Q_C tiene que ser menor que Q_H: la diferencia es el trabajo (W = Q_H − Q_C, "
                "vademecum §9.2)."
            )
        return Q_H, Q_C, Q_H - Q_C
    if kind == "engine":  # η = W/Q_H
        if Q_H is not None:
            W = p * Q_H
            return Q_H, Q_H - W, W
        if W is not None:
            Q_H = W / p
            return Q_H, Q_H - W, W
        assert Q_C is not None
        if p >= 1.0:
            raise ValueError(
                "Con η = 1 la máquina no cede calor: Q_C = 0 no alcanza para despejar Q_H. "
                "Da Q_H o W."
            )
        Q_H = Q_C / (1.0 - p)
        return Q_H, Q_C, Q_H - Q_C
    if kind == "refrigerator":  # COP_R = Q_C/W
        if W is not None:
            return p * W + W, p * W, W
        if Q_C is not None:
            W = Q_C / p
            return Q_C + W, Q_C, W
        assert Q_H is not None
        W = Q_H / (1.0 + p)
        return Q_H, Q_H - W, W
    # bomba de calor: COP_B = Q_H/W
    if W is not None:
        return p * W, p * W - W, W
    if Q_H is not None:
        W = Q_H / p
        return Q_H, Q_H - W, W
    assert Q_C is not None
    if p <= 1.0:
        raise ValueError(
            "Con COP_B = 1 la bomba no toma calor de la fuente fría: Q_C = 0 no alcanza para "
            "despejar W. Da Q_H o W."
        )
    W = Q_C / (p - 1.0)
    return Q_C + W, Q_C, W


def solve_machine(inputs: MachineInputs) -> MachineResult:
    """Una máquina térmica, un refrigerador o una bomba de calor entre dos fuentes.

    El primer principio (W = Q_H − Q_C) da lo que falta; el segundo, la entropía generada en
    las fuentes, S_gen = −ΣQ/T (vademecum §9.1 y §10.2): Q_C/T_C − Q_H/T_H en la máquina
    térmica y Q_H/T_H − Q_C/T_C en las otras dos. S_gen < 0 es una máquina imposible: el
    resultado lo dice con la temperatura que haría falta (teorema de Carnot, §9.2.1).

    Raises
    ------
    ValueError
        Con temperaturas o datos que no tienen sentido, con el mensaje para el alumno.
    """
    _check_data(inputs)
    kind = inputs.kind
    T_H, T_C = inputs.T_H_K, inputs.T_C_K
    _check_reservoirs(T_H, T_C)
    _check_temperature(inputs.T0, "del ambiente T₀")
    p_rev = carnot_performance(kind, T_H, T_C)
    Q_H, Q_C, W = _energies(inputs, p_rev if inputs.reversible else inputs.performance)
    if Q_C < 0.0:  # solo por redondeo
        Q_C = 0.0
    performance = W / Q_H if kind == "engine" else (Q_C / W if kind == "refrigerator" else Q_H / W)
    if inputs.reversible:
        performance = p_rev
    raw = Q_C / T_C - Q_H / T_H if kind == "engine" else Q_H / T_H - Q_C / T_C
    S_gen = snap_sgen(raw, Q_H / T_H + Q_C / T_C)
    result = MachineResult(inputs, Q_H, Q_C, W, performance, p_rev, S_gen)
    violation = _machine_violation(result) if S_gen < 0.0 else None
    return MachineResult(
        inputs,
        Q_H,
        Q_C,
        W,
        performance,
        p_rev,
        S_gen,
        violation,
        _machine_notes(result, violation is not None),
    )


def _sgen_text(S_gen: float, rates: bool) -> str:
    unit = "kW/K" if rates else "kJ/K"
    return f"S_gen = {num(S_gen / 1e3)} {unit}"


def _machine_violation(r: MachineResult) -> str:
    """Por qué la máquina es imposible y qué temperatura haría falta."""
    inp = r.inputs
    T_H, T_C = inp.T_H_K, inp.T_C_K
    perf, p, p_rev = PERFORMANCE_NAMES[r.kind], r.performance, r.performance_rev
    rev = REVERSIBLE_NAMES[r.kind]
    clausius = (
        f"ΣQ/T = {num(r.clausius_sum / 1e3)} {'kW/K' if inp.rates else 'kJ/K'} > 0 viola la "
        "desigualdad de Clausius (vademecum §9.1)."
    )
    if r.kind == "engine":
        if r.Q_C == 0.0:
            return (
                "La máquina convierte todo el calor en trabajo (Q_C = 0, η = 1): viola el "
                "enunciado de Kelvin–Planck. Toda máquina térmica tiene que ceder calor a una "
                f"fuente fría (Çengel §6-3). {clausius}"
            )
        need = (
            f"Con η = {num(p, '.4g')} la fuente caliente tendría que estar a más de "
            f"T_C/(1 − η) = {deg_c(T_C / (1.0 - p))}, o la fría a menos de "
            f"T_H·(1 − η) = {deg_c(T_H * (1.0 - p))}."
        )
        theorem = "teorema de Carnot, vademecum §9.2.1"
    elif r.kind == "refrigerator":
        need = (
            f"Con COP_R = {num(p, '.4g')} el ambiente tendría que estar a menos de "
            f"T_C·(COP + 1)/COP = {deg_c(T_C * (p + 1.0) / p)}, o el espacio frío a más de "
            f"T_H·COP/(COP + 1) = {deg_c(T_H * p / (p + 1.0))}."
        )
        theorem = "vademecum §9.3"
    else:
        need = (
            f"Con COP_B = {num(p, '.4g')} la fuente fría tendría que estar a más de "
            f"T_H·(COP − 1)/COP = {deg_c(T_H * (p - 1.0) / p)}, o la caliente a menos de "
            f"T_C·COP/(COP − 1) = {deg_c(T_C * p / (p - 1.0))}."
        )
        theorem = "vademecum §9.3"
    return (
        f"{perf} = {num(p, '.4g')} > {rev} = {num(p_rev, '.4g')}: la máquina es imposible "
        f"({theorem}). {clausius} {need}"
    )


def _machine_notes(r: MachineResult, impossible: bool) -> tuple[str, ...]:
    inp = r.inputs
    notes: list[str] = []
    perf = PERFORMANCE_NAMES[r.kind]
    if inp.reversible:
        notes.append(
            f"Es la máquina de Carnot: {perf} = {REVERSIBLE_NAMES[r.kind]} y S_gen = 0. Ninguna "
            "máquina real "
            "entre las mismas fuentes la supera (teorema de Carnot, vademecum §9.2.1)."
        )
    elif not impossible:
        notes.append(
            f"{perf} es el {num(100.0 * r.ratio, '.3g')} % del de la máquina reversible: lo "
            f"que falta es el trabajo perdido, {_energy_text(r.lost_work, inp.rates)} "
            "(T·S_gen con T la fuente de afuera)."
        )
    if r.kind == "heat_pump" and math.isclose(r.performance, 1.0, rel_tol=1e-12):
        notes.append(
            "Con COP_B = 1 la «bomba» no toma calor de afuera: es una resistencia eléctrica, "
            "todo el trabajo se disipa en calor."
        )
    if r.kind != "engine":
        other = "COP_B" if r.kind == "refrigerator" else "COP_R"
        value = r.performance + 1.0 if r.kind == "refrigerator" else r.performance - 1.0
        as_other = "bomba de calor" if r.kind == "refrigerator" else "refrigerador"
        notes.append(
            f"Con las mismas fuentes y el mismo W, como {as_other} tendría {other} = "
            f"{num(value, '.4g')} (COP_B = COP_R + 1, vademecum §9.3)."
        )
    default_T0 = inp.T_H_K if r.kind == "refrigerator" else inp.T_C_K
    if not math.isclose(r.T0, default_T0, rel_tol=1e-12):
        notes.append(
            "El ambiente no es una de las fuentes: el calor que llega a la fuente que no es el "
            "ambiente se lleva exergía (no se destruye), así que η_ex no es el cociente con "
            "Carnot."
        )
    return tuple(notes)


def _energy_text(value: float, rates: bool) -> str:
    """En kJ o kW, sin notación exponencial (23 599 kW, no 2,36e+04)."""
    k = value / 1e3
    text = f"{k:,.0f}".replace(",", "\u2009") if abs(k) >= 1e4 else num(k, ".4g")
    return f"{text} {'kW' if rates else 'kJ'}"


# ---------------------------------------------------------------------
# Desigualdad de Clausius con N fuentes
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Reservoir:
    """Una fuente a T (K) y el calor que el ciclo recibe de ella (Q > 0) o le entrega
    (Q < 0), con el signo del vademecum §9.1."""

    T_K: float
    Q: float
    label: str = ""


@dataclass(frozen=True)
class ClausiusInputs:
    """Un ciclo que intercambia calor con varias fuentes (por ciclo o por unidad de tiempo)."""

    reservoirs: tuple[Reservoir, ...]
    rates: bool = False
    T0_K: float = T_DEFAULT_K


@dataclass(frozen=True)
class ClausiusResult:
    inputs: ClausiusInputs
    terms: tuple[float, ...]  # Q_k/T_k
    S_gen: float
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def clausius_sum(self) -> float:
        """ΣQ/T (= −S_gen: las fuentes ganan lo que el ciclo les da)."""
        return -self.S_gen

    @property
    def W(self) -> float:
        """El trabajo neto que entrega el ciclo: ΣQ (primer principio; < 0 si lo consume)."""
        return math.fsum(r.Q for r in self.inputs.reservoirs)

    @property
    def Q_in(self) -> float:
        return math.fsum(r.Q for r in self.inputs.reservoirs if r.Q > 0.0)

    @property
    def Q_out(self) -> float:
        return -math.fsum(r.Q for r in self.inputs.reservoirs if r.Q < 0.0)

    @property
    def efficiency(self) -> float | None:
        """η = W/ΣQ_entra si el ciclo entrega trabajo (una máquina térmica)."""
        if self.W > 0.0 and self.Q_in > 0.0:
            return self.W / self.Q_in
        return None

    @property
    def X_dest(self) -> float:
        return self.inputs.T0_K * self.S_gen

    @property
    def X_heat(self) -> float:
        """La exergía neta de los calores, ΣQ·(1 − T₀/T) (vademecum §11.5): el trabajo de
        un ciclo reversible con los mismos calores. Es W + X_dest."""
        T0 = self.inputs.T0_K
        return math.fsum(r.Q * (1.0 - T0 / r.T_K) for r in self.inputs.reservoirs)

    @property
    def carnot_limit(self) -> float:
        """1 − T_mín/T_máx: el rendimiento máximo de un ciclo entre esas fuentes."""
        temps = [r.T_K for r in self.inputs.reservoirs]
        return 1.0 - min(temps) / max(temps)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen)


def solve_clausius(inputs: ClausiusInputs) -> ClausiusResult:
    """La desigualdad de Clausius de un ciclo con N fuentes: ΣQ_k/T_k ≤ 0 (vademecum §9.1).

    S_gen = −ΣQ_k/T_k es la entropía que ganan las fuentes (el ciclo vuelve a su estado,
    §10.2) y el trabajo neto, W = ΣQ_k (primer principio de un ciclo).

    Raises
    ------
    ValueError
        Con menos de dos fuentes, una temperatura que no es positiva o sin calores.
    """
    res = inputs.reservoirs
    if len(res) < 2:
        raise ValueError("Un ciclo intercambia calor con al menos dos fuentes.")
    for k, r in enumerate(res, 1):
        name = r.label or f"la fuente {k}"
        _check_temperature(r.T_K, f"de {name}")
        if not math.isfinite(r.Q):
            raise ValueError(f"Falta el calor de {name}.")
    _check_temperature(inputs.T0_K, "del ambiente T₀")
    if all(r.Q == 0.0 for r in res):
        raise ValueError("Todos los calores son cero: el ciclo no intercambia calor.")
    terms = tuple(r.Q / r.T_K for r in res)
    S_gen = snap_sgen(-math.fsum(terms), math.fsum(abs(t) for t in terms))
    result = ClausiusResult(inputs, terms, S_gen)
    violation = None
    if S_gen < 0.0:
        unit = "kW/K" if inputs.rates else "kJ/K"
        violation = (
            f"ΣQ/T = {num(result.clausius_sum / 1e3)} {unit} > 0: el ciclo es imposible "
            "(desigualdad de Clausius, vademecum §9.1). Las fuentes perderían entropía: "
            f"{_sgen_text(S_gen, inputs.rates)} < 0."
        )
        eta = result.efficiency
        if eta is not None and eta > result.carnot_limit:
            violation += (
                f" Como máquina térmica, η = {num(eta, '.3g')} supera a 1 − T_mín/T_máx = "
                f"{num(result.carnot_limit, '.3g')}."
            )
    return ClausiusResult(inputs, terms, S_gen, violation, _clausius_notes(result))


def _clausius_notes(r: ClausiusResult) -> tuple[str, ...]:
    notes: list[str] = []
    eta = r.efficiency
    if eta is not None:
        notes.append(
            f"Entrega trabajo: es una máquina térmica con η = W/ΣQ_entra = {num(eta, '.3g')}. "
            f"Con estas fuentes ninguna puede superar 1 − T_mín/T_máx = "
            f"{num(r.carnot_limit, '.3g')}."
        )
    elif r.W < 0.0:
        notes.append(
            "Consume trabajo (W < 0): es un refrigerador o una bomba de calor, o un ciclo que "
            "disipa trabajo en calor."
        )
    if r.S_gen > 0.0:
        rates = r.inputs.rates
        X = r.X_heat
        if X > 0.0:
            notes.append(
                "Con los mismos calores, un ciclo reversible entregaría W + T₀·S_gen = "
                f"{_energy_text(X, rates)}: la exergía neta de los calores (vademecum §11.5)."
            )
        else:
            notes.append(
                "Con los mismos calores, un ciclo reversible consumiría solo "
                f"{_energy_text(-X, rates)} (−ΣQ·(1 − T₀/T), vademecum §11.5): el resto del "
                "trabajo se destruye, T₀·S_gen."
            )
    return tuple(notes)


# ---------------------------------------------------------------------
# Ciclo de Carnot
# ---------------------------------------------------------------------

CarnotProp = Literal["p", "x", "v"]
CARNOT_PROPS: dict[CarnotProp, str] = {"p": "presión p", "x": "título x", "v": "volumen v"}
CarnotFlow = Literal["closed", "steady"]
FLOWS: dict[CarnotFlow, str] = {
    "closed": "Sistema cerrado (cilindro y pistón)",
    "steady": "Flujo estacionario (cuatro equipos)",
}
#: Los puntos de cada tramo de los diagramas.
PATH_POINTS: int = 31
_PAIRS: dict[CarnotProp, str] = {"p": "TP", "x": "TX", "v": "TV"}


@dataclass(frozen=True)
class CarnotInputs:
    """El ciclo de Carnot de un fluido real o un gas ideal entre T_H y T_C (K).

    El estado 1 (donde empieza a recibir calor) y el 2 (donde termina) están a T_H y se
    dan con otra propiedad: la presión, el título (dentro de la campana) o el volumen
    específico. Los estados 3 y 4 salen de las isoentrópicas hasta T_C.
    """

    substance: Substance
    T_H_K: float
    T_C_K: float
    prop1: CarnotProp
    value1: float
    prop2: CarnotProp
    value2: float
    mass_kg: float = 1.0
    flow: CarnotFlow = "closed"


@dataclass(frozen=True)
class CarnotProcess:
    """Un proceso del ciclo: el calor que recibe y el trabajo que hace el fluido (J)."""

    name: str
    start: int
    end: int
    Q: float
    W: float
    dE: float  # ΔU (cerrado) o ΔH (flujo estacionario)
    path: tuple[ThermoState, ...]


@dataclass(frozen=True)
class CarnotResult:
    inputs: CarnotInputs
    states: tuple[ThermoState, ThermoState, ThermoState, ThermoState]
    processes: tuple[CarnotProcess, CarnotProcess, CarnotProcess, CarnotProcess]
    notes: tuple[str, ...] = ()

    @property
    def Q_H(self) -> float:
        return self.processes[0].Q

    @property
    def Q_C(self) -> float:
        """El calor que cede (magnitud)."""
        return -self.processes[2].Q

    @property
    def W_net(self) -> float:
        return math.fsum(p.W for p in self.processes)

    @property
    def eta(self) -> float:
        return self.W_net / self.Q_H

    @property
    def eta_carnot(self) -> float:
        return 1.0 - self.inputs.T_C_K / self.inputs.T_H_K

    @property
    def cop_refrigerator(self) -> float:
        """El COP del mismo ciclo recorrido al revés (refrigerador de Carnot)."""
        T_H, T_C = self.inputs.T_H_K, self.inputs.T_C_K
        return T_C / (T_H - T_C)

    @property
    def cop_heat_pump(self) -> float:
        T_H, T_C = self.inputs.T_H_K, self.inputs.T_C_K
        return T_H / (T_H - T_C)

    @property
    def W_out(self) -> float:
        """El trabajo que entregan los procesos que se expanden."""
        return math.fsum(p.W for p in self.processes if p.W > 0.0)

    @property
    def W_in(self) -> float:
        """El trabajo que consumen los que comprimen (magnitud)."""
        return -math.fsum(p.W for p in self.processes if p.W < 0.0)

    @property
    def back_work_ratio(self) -> float:
        """W_consumido/W_entregado (Çengel §9-8)."""
        return self.W_in / self.W_out

    @property
    def path(self) -> tuple[ThermoState, ...]:
        """El ciclo completo 1 → 2 → 3 → 4 → 1, para los diagramas."""
        out: list[ThermoState] = []
        for p in self.processes:
            out.extend(p.path if not out else p.path[1:])
        return tuple(out)


def _carnot_state(inp: CarnotInputs, prop: CarnotProp, value: float, label: str) -> ThermoState:
    sub = inp.substance
    if prop == "x" and sub.kind != "fluid":
        raise ValueError("El título x es de un fluido dentro de la campana: un gas ideal no tiene.")
    pair = _PAIRS[prop]
    return state(sub, pair, inp.T_H_K, value, label)  # type: ignore[arg-type]


def _ts_state(sub: Substance, T: float, s: float, label: str = "") -> ThermoState:
    return state(sub, "TS", T, s, label)


def _isotherm(sub: Substance, T: float, s_a: float, s_b: float, n: int) -> list[ThermoState]:
    return [_ts_state(sub, T, s_a + (s_b - s_a) * k / (n - 1)) for k in range(n)]


def _isentrope(sub: Substance, s: float, T_a: float, T_b: float, n: int) -> list[ThermoState]:
    return [_ts_state(sub, T_a + (T_b - T_a) * k / (n - 1), s) for k in range(n)]


def solve_carnot(inputs: CarnotInputs) -> CarnotResult:
    """El ciclo de Carnot: dos isotermas (T_H y T_C) y dos isoentrópicas.

    1 → 2 recibe Q_H = m·T_H·(s₂ − s₁), 2 → 3 se expande sin calor, 3 → 4 cede
    Q_C = m·T_C·(s₂ − s₁) y 4 → 1 se comprime sin calor. El trabajo de cada proceso sale del
    primer principio, W = Q − ΔU (sistema cerrado) o W = Q − ΔH (flujo estacionario); el
    neto es Q_H − Q_C y η = 1 − T_C/T_H con cualquier fluido (vademecum §9.2; Çengel y
    Boles §6-7 y §6-10).

    Raises
    ------
    ValueError
        Con un incompresible, T_H ≤ T_C, s₂ ≤ s₁ o un estado fuera del rango del modelo.
    """
    sub = inputs.substance
    if sub.kind == "incompressible":
        raise ValueError(
            "Un incompresible no sirve para el ciclo de Carnot: su isoentrópica es a T "
            "constante (s = c·ln T/T_ref, vademecum §13), así que no puede ir de T_H a T_C."
        )
    T_H, T_C = inputs.T_H_K, inputs.T_C_K
    _check_reservoirs(T_H, T_C)
    if not (math.isfinite(inputs.mass_kg) and inputs.mass_kg > 0.0):
        raise ValueError("La masa tiene que ser mayor que cero.")
    if inputs.flow not in FLOWS:
        raise ValueError(f"El tipo de sistema {inputs.flow!r} no existe: {', '.join(FLOWS)}.")
    s1 = _carnot_state(inputs, inputs.prop1, inputs.value1, "1")
    s2 = _carnot_state(inputs, inputs.prop2, inputs.value2, "2")
    if s2.s <= s1.s:
        raise ValueError(
            "En 1 → 2 el fluido recibe calor a T_H, así que su entropía sube: s₂ tiene que ser "
            f"mayor que s₁ (s₁ = {num(s1.s / 1e3)} y s₂ = {num(s2.s / 1e3)} kJ/(kg·K)). "
            + (
                "Con un gas, la presión del estado 2 tiene que ser menor que la del 1."
                if sub.kind == "ideal_gas"
                else "Revisá los datos de los estados 1 y 2."
            )
        )
    try:
        s3 = _ts_state(sub, T_C, s2.s, "3")
        s4 = _ts_state(sub, T_C, s1.s, "4")
    except ValueError as exc:
        raise ValueError(
            f"La expansión sin calor hasta T_C = {deg_c(T_C)} sale del rango del modelo: {exc}"
        ) from exc
    n = PATH_POINTS
    paths = (
        _isotherm(sub, T_H, s1.s, s2.s, n),
        _isentrope(sub, s2.s, T_H, T_C, n),
        _isotherm(sub, T_C, s2.s, s1.s, n),
        _isentrope(sub, s1.s, T_C, T_H, n),
    )
    states = (s1, s2, s3, s4)
    m = inputs.mass_kg
    closed = inputs.flow == "closed"
    names = (
        f"1 → 2: recibe calor a T_H = {deg_c(T_H)}",
        "2 → 3: se expande sin calor",
        f"3 → 4: cede calor a T_C = {deg_c(T_C)}",
        "4 → 1: se comprime sin calor",
    )
    q12 = m * T_H * (s2.s - s1.s)
    q34 = -m * T_C * (s2.s - s1.s)
    procs = []
    for k, (Q, a, b) in enumerate(((q12, s1, s2), (0.0, s2, s3), (q34, s3, s4), (0.0, s4, s1))):
        dE = m * ((b.u - a.u) if closed else (b.h - a.h))
        W = Q - dE
        if abs(W) <= 1e-9 * max(abs(Q), abs(dE)):
            W = 0.0  # la caldera y el condensador en la campana: ruido de redondeo
        ends = (k + 1, (k + 1) % 4 + 1)
        procs.append(CarnotProcess(names[k], ends[0], ends[1], Q, W, dE, tuple(paths[k])))
    result = CarnotResult(inputs, states, tuple(procs))  # type: ignore[arg-type]
    return CarnotResult(inputs, states, tuple(procs), _carnot_notes(result))  # type: ignore[arg-type]


def _carnot_notes(r: CarnotResult) -> tuple[str, ...]:
    inp = r.inputs
    sub = inp.substance
    s3, s4 = r.states[2], r.states[3]
    notes = [
        f"η = 1 − T_C/T_H = {num(r.eta_carnot, '.4g')} no depende del fluido: con las mismas "
        "temperaturas cualquier sustancia da lo mismo (vademecum §9.2).",
        f"Al revés (1 → 4 → 3 → 2 → 1) es el refrigerador de Carnot, con COP_R = "
        f"{num(r.cop_refrigerator, '.4g')}, o la bomba de calor, con COP_B = "
        f"{num(r.cop_heat_pump, '.4g')} (vademecum §9.3).",
    ]
    if sub.kind == "fluid" and inp.flow == "steady":
        if s4.is_two_phase:
            notes.append(
                f"El compresor (4 → 1) recibe una mezcla de líquido y vapor (x₄ = "
                f"{num(s4.x or 0.0, '.3g')}): comprimir una mezcla es impracticable. Por eso las "
                "centrales usan el ciclo de Rankine y los refrigeradores, el de compresión de "
                "vapor (Çengel §10-1 y §11-2)."
            )
        if s3.is_two_phase and (s3.x or 0.0) < 0.85:
            notes.append(
                f"La turbina (2 → 3) termina con x₃ = {num(s3.x or 0.0, '.3g')}: tanta humedad "
                "erosiona los álabes (Çengel §10-1)."
            )
    if sub.kind == "ideal_gas":
        notes.append(
            "Con un gas, recibir y ceder calor a T constante obliga a expandir y comprimir muy "
            "despacio: el ciclo de Carnot de gas es un límite teórico (Çengel §9-2)."
        )
    if inp.flow == "steady":
        notes.append(
            f"El compresor consume el {num(100.0 * r.back_work_ratio, '.3g')} % de lo que da la "
            "turbina (la relación de trabajos, Çengel §9-8)."
        )
    return tuple(notes)


# ---------------------------------------------------------------------
# La entropía que genera un calor
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HeatPathInputs:
    """Un calor Q que pasa por una serie de temperaturas (K), de la primera a la última.

    ``points`` nombra cada temperatura y ``zones`` cada tramo entre dos de ellas (la pared,
    una capa de aire). Por ciclo o por proceso (J) o por unidad de tiempo (W).
    """

    Q: float
    temperatures: tuple[float, ...]
    points: tuple[str, ...]
    zones: tuple[str, ...]
    rates: bool = False
    T0_K: float = T_DEFAULT_K


@dataclass(frozen=True)
class HeatZone:
    """Un tramo: el calor entra a T_in y sale a T_out, y genera Q·(1/T_out − 1/T_in)."""

    name: str
    T_in: float
    T_out: float
    S_gen: float


@dataclass(frozen=True)
class HeatPathResult:
    inputs: HeatPathInputs
    zones: tuple[HeatZone, ...]
    S_gen: float
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def X_dest(self) -> float:
        return self.inputs.T0_K * self.S_gen

    @property
    def exergies(self) -> tuple[float, ...]:
        """La exergía del calor a cada temperatura, Q·(1 − T₀/T) (vademecum §11.5)."""
        T0, Q = self.inputs.T0_K, self.inputs.Q
        return tuple(Q * (1.0 - T0 / T) for T in self.inputs.temperatures)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen)


def solve_heat_path(inputs: HeatPathInputs) -> HeatPathResult:
    """La entropía que genera un calor al pasar de una temperatura a otra.

    Cada tramo es un sistema que no cambia (el calor entra y sale igual): genera
    S_gen = Q/T_sal − Q/T_ent ≥ 0 (vademecum §10.3; Çengel y Boles §7-2 entre dos fuentes y
    §7-13 en una pared con sus capas de aire). El total es Q·(1/T_última − 1/T_primera) y la
    exergía destruida, T₀·S_gen (§11.8).

    Raises
    ------
    ValueError
        Con menos de dos temperaturas, nombres que no coinciden o Q ≤ 0.
    """
    temps = inputs.temperatures
    if len(temps) < 2:
        raise ValueError("El calor pasa de una temperatura a otra: hacen falta al menos dos.")
    if len(inputs.points) != len(temps) or len(inputs.zones) != len(temps) - 1:
        raise ValueError("Cada temperatura lleva su nombre y cada tramo entre dos, el suyo.")
    for name, T in zip(inputs.points, temps, strict=True):
        _check_temperature(T, f"de {name}")
    _check_temperature(inputs.T0_K, "del ambiente T₀")
    Q = inputs.Q
    if not (math.isfinite(Q) and Q > 0.0):
        raise ValueError(
            "El calor Q es el que pasa del primer punto al último: tiene que ser mayor que "
            "cero (si va al revés, invertí el orden de las temperaturas)."
        )
    zones = tuple(
        HeatZone(name, a, b, snap_sgen(Q / b - Q / a, Q / a + Q / b))
        for name, a, b in zip(inputs.zones, temps[:-1], temps[1:], strict=True)
    )
    S_gen = snap_sgen(math.fsum(z.S_gen for z in zones), Q / temps[0] + Q / temps[-1])
    result = HeatPathResult(inputs, zones, S_gen)
    violation = None
    bad = [z for z in zones if z.S_gen < 0.0]
    if bad:
        z = bad[0]
        unit = "W/K" if inputs.rates else "kJ/K"
        value = z.S_gen if inputs.rates else z.S_gen / 1e3
        violation = (
            f"En {z.name} el calor pasaría de {deg_c(z.T_in)} a {deg_c(z.T_out)}: de lo frío a "
            "lo caliente, solo, viola el enunciado de Clausius (vademecum §9.1). Genera "
            f"S_gen = {num(value)} {unit} < 0: imposible."
        )
    return HeatPathResult(inputs, zones, S_gen, violation, _heat_notes(result))


def _heat_notes(r: HeatPathResult) -> tuple[str, ...]:
    notes: list[str] = []
    if r.S_gen > 0.0 and len(r.zones) > 1:
        top = max(r.zones, key=lambda z: z.S_gen)
        share = 100.0 * top.S_gen / r.S_gen
        notes.append(
            f"{top.name[:1].upper()}{top.name[1:]} genera el {num(share, '.3g')} % de la "
            "entropía: es el tramo con el salto de temperatura más grande frente a la "
            "temperatura a la que ocurre."
        )
    X = r.exergies
    if X[0] > 0.0 and r.S_gen > 0.0:
        if abs(X[-1]) <= 1e-12 * X[0]:
            notes.append(
                "Al llegar a T₀ el calor ya no tiene exergía: toda la que traía, T₀·S_gen, se "
                "destruyó en el camino. La energía se conserva; su calidad, no."
            )
        else:
            notes.append(
                f"El calor termina con el {num(100.0 * X[-1] / X[0], '.3g')} % de la exergía "
                "con que empezó: lo que falta, T₀·S_gen, se destruyó en el camino. La energía "
                "se conserva; su calidad, no."
            )
    T0 = r.inputs.T0_K
    if any(T < T0 * (1.0 - 1e-12) for T in r.inputs.temperatures):
        notes.append(
            "Debajo de T₀ la exergía del calor cambia de signo (vademecum §11.5): llevar calor "
            "a algo más frío que el ambiente cuesta exergía."
        )
    return tuple(notes)


# ---------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MachineExample:
    description: str
    inputs: MachineInputs


@dataclass(frozen=True)
class ClausiusExample:
    description: str
    inputs: ClausiusInputs


@dataclass(frozen=True)
class CarnotExample:
    description: str
    inputs: CarnotInputs


@dataclass(frozen=True)
class HeatPathExample:
    description: str
    inputs: HeatPathInputs


def _f_to_k(t_f: float) -> float:
    return (t_f - 32.0) / 1.8 + _C


_KJ_MIN = 1e3 / 60.0  # 1 kJ/min en W
_KJ_H = 1e3 / 3600.0  # 1 kJ/h en W

MACHINE_EXAMPLES: dict[str, MachineExample] = {
    "Central térmica de 30 MW (Çengel §6-3)": MachineExample(
        "Un horno le da 80 MW y cede 50 MW a un río: el libro da Ẇ = 30 MW y η = 37,5 %. No "
        "da las temperaturas: acá, el horno a 600 °C y el río a 15 °C.",
        MachineInputs("engine", 600.0 + _C, 15.0 + _C, Q_H=80e6, Q_C=50e6, rates=True),
    ),
    "Refrigerador de una cocina (Çengel §6-4)": MachineExample(
        "El compartimiento a 4 °C pierde 360 kJ/min con 2 kW de potencia: el libro da "
        "COP_R = 3 y Q̇_H = 480 kJ/min. La cocina no figura: acá, a 25 °C.",
        MachineInputs("refrigerator", 25.0 + _C, 4.0 + _C, Q_C=360.0 * _KJ_MIN, W=2e3, rates=True),
    ),
    "Bomba de calor de una casa (Çengel §6-4)": MachineExample(
        "La casa a 20 °C pierde 80 000 kJ/h con −2 °C afuera y COP_B = 2,5: el libro da "
        "Ẇ = 32 000 kJ/h (8,9 kW) y Q̇_C = 48 000 kJ/h.",
        MachineInputs(
            "heat_pump", 20.0 + _C, -2.0 + _C, Q_H=80_000.0 * _KJ_H, performance=2.5, rates=True
        ),
    ),
    "Máquina de Carnot (Çengel §6-10)": MachineExample(
        "Recibe 500 kJ por ciclo de una fuente a 652 °C y cede calor a 30 °C: el libro da "
        "η = 67,2 % y Q_C = 164 kJ.",
        MachineInputs("engine", 652.0 + _C, 30.0 + _C, Q_H=500e3, reversible=True),
    ),
    "Un refrigerador demasiado bueno (Çengel §6-11)": MachineExample(
        "Un inventor dice que su refrigerador mantiene 35 °F en una cocina a 75 °F con "
        "COP_R = 13,5: el libro da COP_R,rev = 12,4, así que es imposible. Con 1 kW (el "
        "veredicto no depende de la potencia).",
        MachineInputs(
            "refrigerator", _f_to_k(75.0), _f_to_k(35.0), W=1e3, performance=13.5, rates=True
        ),
    ),
    "Bomba de calor de Carnot (Çengel §6-11)": MachineExample(
        "La casa a 21 °C pierde 135 000 kJ/h con −5 °C afuera: el libro da COP_B,rev = 11,3 "
        "y una potencia mínima de 3,32 kW.",
        MachineInputs(
            "heat_pump", 21.0 + _C, -5.0 + _C, Q_H=135_000.0 * _KJ_H, reversible=True, rates=True
        ),
    ),
    "Un motor que promete demasiado": MachineExample(
        "Un inventor dice que su máquina recibe 700 kJ de una fuente a 500 K y entrega 300 kJ "
        "de trabajo, con el sumidero a 290 K: ¿es razonable?",
        MachineInputs("engine", 500.0, 290.0, Q_H=700e3, W=300e3),
    ),
}

CLAUSIUS_EXAMPLES: dict[str, ClausiusExample] = {
    "Máquina con dos fuentes y un sumidero": ClausiusExample(
        "Recibe 1000 kJ a 1000 K y 500 kJ a 600 K, y cede 900 kJ a 300 K por ciclo.",
        ClausiusInputs(
            (
                Reservoir(1000.0, 1000e3, "la fuente a 1000 K"),
                Reservoir(600.0, 500e3, "la fuente a 600 K"),
                Reservoir(300.0, -900e3, "el sumidero"),
            ),
            T0_K=300.0,
        ),
    ),
    "Heladera con freezer": ClausiusExample(
        "Saca 100 kJ del freezer a −18 °C y 200 kJ de la heladera a 5 °C, y le entrega 420 kJ a "
        "la cocina a 25 °C: consume 120 kJ de trabajo.",
        ClausiusInputs(
            (
                Reservoir(-18.0 + _C, 100e3, "el freezer"),
                Reservoir(5.0 + _C, 200e3, "la heladera"),
                Reservoir(25.0 + _C, -420e3, "la cocina"),
            ),
            T0_K=25.0 + _C,
        ),
    ),
    "Ciclo de Carnot (la igualdad)": ClausiusExample(
        "La máquina de Carnot de Çengel §6-10: 500 kJ a 652 °C y lo justo cedido a 30 °C. "
        "ΣQ/T = 0.",
        ClausiusInputs(
            (
                Reservoir(652.0 + _C, 500e3, "la fuente"),
                Reservoir(30.0 + _C, -500e3 * (30.0 + _C) / (652.0 + _C), "el sumidero"),
            ),
            T0_K=30.0 + _C,
        ),
    ),
    "Un ciclo imposible": ClausiusExample(
        "Recibe 1000 kJ a 500 K y cede solo 300 kJ a 300 K por ciclo.",
        ClausiusInputs(
            (
                Reservoir(500.0, 1000e3, "la fuente"),
                Reservoir(300.0, -300e3, "el sumidero"),
            ),
            T0_K=300.0,
        ),
    ),
}

_WATER = fluid_substance("Water")
_R134A = fluid_substance("R134a")
_AIR = ideal_gas_substance("air", "variable")

CARNOT_EXAMPLES: dict[str, CarnotExample] = {
    "Vapor de agua en la campana": CarnotExample(
        "1 kg de agua pasa de líquido saturado a vapor saturado a 250 °C y cede calor a "
        "40 °C, en un cilindro con pistón.",
        CarnotInputs(_WATER, 250.0 + _C, 40.0 + _C, "x", 0.0, "x", 1.0),
    ),
    "Vapor de agua en cuatro equipos": CarnotExample(
        "El mismo ciclo en flujo estacionario (caldera, turbina, condensador y compresor): el "
        "trabajo neto es el mismo, pero se reparte distinto.",
        CarnotInputs(_WATER, 250.0 + _C, 40.0 + _C, "x", 0.0, "x", 1.0, flow="steady"),
    ),
    "Aire entre 1000 K y 300 K (gas ideal)": CarnotExample(
        "1 kg de aire se expande a 1000 K de 50 a 25 bar, sigue sin calor hasta 300 K, se "
        "comprime a 300 K y vuelve sin calor.",
        CarnotInputs(_AIR, 1000.0, 300.0, "p", 50e5, "p", 25e5),
    ),
    "Vapor sobrecalentado (fuera de la campana)": CarnotExample(
        "El vapor recibe calor a 500 °C mientras se expande de 100 a 20 bar (fuera de la "
        "campana) y lo cede a 40 °C ya en la campana.",
        CarnotInputs(_WATER, 500.0 + _C, 40.0 + _C, "p", 100e5, "p", 20e5),
    ),
    "R-134a en la campana": CarnotExample(
        "El R-134a pasa de líquido a vapor saturado a 40 °C y cede calor a −10 °C, en flujo "
        "estacionario: al revés, es el refrigerador de Carnot.",
        CarnotInputs(_R134A, 40.0 + _C, -10.0 + _C, "x", 0.0, "x", 1.0, flow="steady"),
    ),
}

HEAT_PATH_EXAMPLES: dict[str, HeatPathExample] = {
    "Calor entre dos fuentes: sumidero a 500 K (Çengel §7-2)": HeatPathExample(
        "Una fuente a 800 K le da 2000 kJ a un sumidero a 500 K: el libro da S_gen = 1,5 kJ/K.",
        HeatPathInputs(2000e3, (800.0, 500.0), ("la fuente", "el sumidero"), ("la transferencia",)),
    ),
    "Calor entre dos fuentes: sumidero a 750 K (Çengel §7-2)": HeatPathExample(
        "Los mismos 2000 kJ con el sumidero a 750 K: el libro da S_gen = 0,167 kJ/K (menos "
        "salto de temperatura, menos irreversible).",
        HeatPathInputs(2000e3, (800.0, 750.0), ("la fuente", "el sumidero"), ("la transferencia",)),
    ),
    "Pared de ladrillo (Çengel §7-13)": HeatPathExample(
        "Pasan 1035 W por una pared con las caras a 20 y 5 °C, la casa a 27 °C y afuera 0 °C: el "
        "libro da Ṡ_gen = 0,191 W/K en la pared y 0,341 W/K en total.",
        HeatPathInputs(
            1035.0,
            (27.0 + _C, 20.0 + _C, 5.0 + _C, 0.0 + _C),
            ("el aire de la casa", "la cara interior", "la cara exterior", "el aire de afuera"),
            ("la capa de aire interior", "la pared", "la capa de aire exterior"),
            rates=True,
            T0_K=0.0 + _C,
        ),
    ),
    "Caldera: de los gases a 1500 °C al agua a 250 °C": HeatPathExample(
        "Los gases de la combustión le dan 1000 kJ al agua que hierve a 250 °C: mucho salto de "
        "temperatura, mucha exergía destruida (Çengel §8-1).",
        HeatPathInputs(
            1000e3,
            (1500.0 + _C, 250.0 + _C),
            ("los gases", "el agua"),
            ("la caldera",),
        ),
    ),
}


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def _state_dict(s: ThermoState) -> dict[str, Any]:
    return {
        "p_Pa": s.p,
        "T_K": s.T,
        "v_m3_kg": s.v,
        "u_J_kg": s.u,
        "h_J_kg": s.h,
        "s_J_kgK": s.s,
        "x": s.x,
        "region": s.region_es,
    }


def machine_to_dict(r: MachineResult) -> dict[str, Any]:
    inp = r.inputs
    unit = "W" if inp.rates else "J"
    return {
        "maquina": MACHINES[inp.kind],
        "por_unidad_de_tiempo": inp.rates,
        "T_H_K": inp.T_H_K,
        "T_C_K": inp.T_C_K,
        "T0_K": r.T0,
        "datos": list(inp.data),
        f"Q_H_{unit}": r.Q_H,
        f"Q_C_{unit}": r.Q_C,
        f"W_{unit}": r.W,
        PERFORMANCE_NAMES[inp.kind]: r.performance,
        f"{PERFORMANCE_NAMES[inp.kind]}_rev": r.performance_rev,
        f"suma_Q_sobre_T_{unit}_K": r.clausius_sum,
        f"S_gen_{unit}_K": r.S_gen,
        f"X_dest_{unit}": r.X_dest,
        f"W_reversible_{unit}": r.W_ideal,
        "eta_ex": r.eta_ex,
        "veredicto": r.verdict,
        "violacion": r.violation,
        "notas": list(r.notes),
    }


def clausius_to_dict(r: ClausiusResult) -> dict[str, Any]:
    inp = r.inputs
    unit = "W" if inp.rates else "J"
    return {
        "por_unidad_de_tiempo": inp.rates,
        "fuentes": [
            {
                "fuente": res.label or f"fuente {k}",
                "T_K": res.T_K,
                f"Q_{unit}": res.Q,
                f"Q_sobre_T_{unit}_K": term,
            }
            for k, (res, term) in enumerate(zip(inp.reservoirs, r.terms, strict=True), 1)
        ],
        f"suma_Q_sobre_T_{unit}_K": r.clausius_sum,
        f"W_{unit}": r.W,
        f"S_gen_{unit}_K": r.S_gen,
        "T0_K": inp.T0_K,
        f"X_dest_{unit}": r.X_dest,
        f"exergia_de_los_calores_{unit}": r.X_heat,
        "rendimiento": r.efficiency,
        "veredicto": r.verdict,
        "violacion": r.violation,
        "notas": list(r.notes),
    }


def carnot_to_dict(r: CarnotResult) -> dict[str, Any]:
    inp = r.inputs
    return {
        "sustancia": inp.substance.label,
        "sistema": FLOWS[inp.flow],
        "masa_kg": inp.mass_kg,
        "T_H_K": inp.T_H_K,
        "T_C_K": inp.T_C_K,
        "estados": {f"estado_{k}": _state_dict(s) for k, s in enumerate(r.states, 1)},
        "procesos": [
            {
                "proceso": p.name,
                "Q_J": p.Q,
                "W_J": p.W,
                ("dU_J" if inp.flow == "closed" else "dH_J"): p.dE,
            }
            for p in r.processes
        ],
        "Q_H_J": r.Q_H,
        "Q_C_J": r.Q_C,
        "W_neto_J": r.W_net,
        "eta": r.eta,
        "eta_Carnot": r.eta_carnot,
        "notas": list(r.notes),
    }


def heat_path_to_dict(r: HeatPathResult) -> dict[str, Any]:
    inp = r.inputs
    unit = "W" if inp.rates else "J"
    return {
        f"Q_{unit}": inp.Q,
        "T0_K": inp.T0_K,
        "puntos": [
            {"punto": name, "T_K": T, f"exergia_del_calor_{unit}": X}
            for name, T, X in zip(inp.points, inp.temperatures, r.exergies, strict=True)
        ],
        "tramos": [
            {"tramo": z.name, "T_entra_K": z.T_in, "T_sale_K": z.T_out, f"S_gen_{unit}_K": z.S_gen}
            for z in r.zones
        ],
        f"S_gen_{unit}_K": r.S_gen,
        f"X_dest_{unit}": r.X_dest,
        "veredicto": r.verdict,
        "violacion": r.violation,
        "notas": list(r.notes),
    }
