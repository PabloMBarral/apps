"""Flujo estacionario: un dispositivo, una cámara de mezcla y un intercambiador.

Vademecum §3.3 (primer principio) y §10.3.2 (entropía); por unidad de masa de una
corriente, con la velocidad ω como en el vademecum:

    q − w = (h₂ − h₁) + (ω₂² − ω₁²)/2 + g·(z₂ − z₁)
    s_gen = s₂ − s₁ − q/T_b ≥ 0

q es el calor que recibe el fluido, w el trabajo que hace (positivo en una turbina,
negativo en un compresor, una bomba o una resistencia) y T_b la temperatura de la frontera
por donde pasa el calor (la del ambiente, a T₀, si no se da).

Los dispositivos de Çengel y Boles §5-4: toberas y difusores (w = 0), válvulas de
estrangulamiento (adiabáticas, h₂ = h₁), turbinas, compresores y bombas (con el rendimiento
isoentrópico del vademecum §10.4) y calentadores o enfriadores (caños y ductos, sin trabajo
salvo una resistencia o una paleta). Con dos corrientes, la cámara de mezcla y el
intercambiador sin mezcla (§5-4; la entropía, §7-13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from core.balances.common import (
    S_FLOOR,
    G,
    Verdict,
    deg_c,
    heat_direction_message,
    num,
    snap_sgen,
    verdict,
    violation_message,
)
from core.balances.substance import (
    PairCode,
    Substance,
    ThermoState,
    fluid_substance,
    ideal_gas_substance,
    incompressible_substance,
    state,
)

__all__ = [
    "ALLOWED_OUT",
    "DEVICES",
    "DEVICE_EXAMPLES",
    "EXCHANGER_EXAMPLES",
    "FLOW_NAMES",
    "MIXING_EXAMPLES",
    "OUT_NAMES",
    "DeviceExample",
    "DeviceInputs",
    "DeviceKind",
    "DeviceResult",
    "ExchangerExample",
    "ExchangerInputs",
    "ExchangerResult",
    "ExchangerStream",
    "FlowKind",
    "MixingExample",
    "MixingInputs",
    "MixingResult",
    "MixingStream",
    "OutKind",
    "allowed_devices",
    "allowed_out",
    "device_to_dict",
    "exchanger_to_dict",
    "mixing_to_dict",
    "solve_device",
    "solve_exchanger",
    "solve_mixing",
]

DeviceKind = Literal["nozzle", "diffuser", "valve", "turbine", "compressor", "pump", "heater"]
DEVICES: dict[DeviceKind, str] = {
    "nozzle": "Tobera",
    "diffuser": "Difusor",
    "valve": "Válvula (estrangulamiento)",
    "turbine": "Turbina",
    "compressor": "Compresor",
    "pump": "Bomba",
    "heater": "Calentador o enfriador (caño, ducto)",
}
_WORK_DEVICES = ("turbine", "compressor", "pump")
_ETA_DEVICES = ("turbine", "compressor", "pump", "nozzle")

OutKind = Literal["T", "x", "eta", "V", "w", "q"]
OUT_NAMES: dict[OutKind, str] = {
    "T": "la temperatura de salida T₂",
    "x": "el título de salida x₂",
    "eta": "el rendimiento isoentrópico η_s",
    "V": "la velocidad de salida ω₂",
    "w": "el trabajo w",
    "q": "el calor q",
}
#: El segundo dato de la salida (además de p₂); la válvula no lleva (h₂ = h₁).
ALLOWED_OUT: dict[DeviceKind, tuple[OutKind, ...]] = {
    "nozzle": ("V", "T", "x", "eta"),
    "diffuser": ("V", "T", "x"),
    "valve": (),
    "turbine": ("T", "x", "eta", "w"),
    "compressor": ("T", "eta", "w"),
    "pump": ("T", "eta", "w"),
    "heater": ("T", "x", "q"),
}

FlowKind = Literal["m", "Vdot", "inlet", "power"]
FLOW_NAMES: dict[FlowKind, str] = {
    "m": "Caudal másico ṁ",
    "Vdot": "Caudal volumétrico a la entrada V̇₁",
    "inlet": "Área de entrada A₁ (con ω₁)",
    "power": "Potencia del dispositivo (|Ẇ| o |Q̇|)",
}

_FLOW_NOUNS: dict[FlowKind, str] = {
    "m": "El caudal másico ṁ",
    "Vdot": "El caudal volumétrico V̇₁",
    "inlet": "El área de entrada A₁",
    "power": "La potencia",
}

_PATH_POINTS = 41


def allowed_devices(sub: Substance) -> tuple[DeviceKind, ...]:
    """Una bomba mueve líquidos y un compresor gases (Çengel §5-4)."""
    if sub.kind == "incompressible":
        return ("nozzle", "diffuser", "valve", "turbine", "pump", "heater")
    if sub.kind == "ideal_gas":
        return ("nozzle", "diffuser", "valve", "turbine", "compressor", "heater")
    return tuple(DEVICES)


def allowed_out(device: DeviceKind, sub: Substance) -> tuple[OutKind, ...]:
    """Los datos de salida para ``device`` y ``sub`` (el título, solo un fluido real)."""
    outs = ALLOWED_OUT[device]
    return outs if sub.kind == "fluid" else tuple(o for o in outs if o != "x")


def _kJkg(x: float) -> str:
    return f"{num(x / 1e3)} kJ/kg"


# ---------------------------------------------------------------------
# Un dispositivo
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceInputs:
    """Un dispositivo con una entrada y una salida (SI).

    La salida va con p₂ (``None``: p₂ = p₁) y un dato más (``out``, ``out_value``); el
    balance da lo que falta: w (turbina, compresor, bomba), q (calentador), ω₂ (tobera,
    difusor) o el estado 2 (con w, q u ω₂ dados).

    ``q`` y ``w`` son el calor y el trabajo que se conocen (por kg, o Q̇ y Ẇ en W con
    ``rates``); el que es incógnita se ignora. ``V2 = None`` es ω₂ = ω₁ (cuando ω₂ no es
    la incógnita). Con ``A1_m2`` y el caudal dado, ω₁ = ṁ·v₁/A₁.
    """

    device: DeviceKind
    substance: Substance
    pair1: PairCode
    a1: float
    b1: float
    p2_Pa: float | None = None
    out: OutKind = "T"
    out_value: float = math.nan
    flow: FlowKind = "m"
    flow_value: float = 1.0
    V1: float = 0.0
    A1_m2: float | None = None
    V2: float | None = None
    z1: float = 0.0
    z2: float = 0.0
    q: float = 0.0
    w: float = 0.0
    rates: bool = False
    T_b_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class DeviceResult:
    inputs: DeviceInputs
    state1: ThermoState
    state2: ThermoState
    state2s: ThermoState | None  # la salida isoentrópica (p₂, s₁)
    m_dot: float
    V1: float
    V2: float
    q: float  # calor que recibe (J/kg)
    w: float  # trabajo que hace (J/kg)
    T_b: float
    s_gen: float  # J/(kg·K)
    eta_s: float | None  # el de los datos o el que resulta (dispositivo adiabático)
    unknown: str  # lo que despejó el balance: "w", "q", "V" o "h"
    path: tuple[ThermoState, ...]
    path_s: tuple[ThermoState, ...] = ()  # la isoentrópica de 1 a 2s
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def dh(self) -> float:
        return self.state2.h - self.state1.h

    @property
    def dke(self) -> float:
        return (self.V2**2 - self.V1**2) / 2.0

    @property
    def dpe(self) -> float:
        return G * (self.inputs.z2 - self.inputs.z1)

    @property
    def Q_dot(self) -> float:
        return self.m_dot * self.q

    @property
    def W_dot(self) -> float:
        return self.m_dot * self.w

    @property
    def S_gen_dot(self) -> float:
        return self.m_dot * self.s_gen

    @property
    def X_dest_dot(self) -> float:
        """Exergía destruida por unidad de tiempo: T₀·Ṡ_gen (W)."""
        return self.inputs.T0_K * self.S_gen_dot

    @property
    def A1(self) -> float | None:
        return self.m_dot * self.state1.v / self.V1 if self.V1 > 0.0 else None

    @property
    def A2(self) -> float | None:
        return self.m_dot * self.state2.v / self.V2 if self.V2 > 0.0 else None

    @property
    def energy_residual(self) -> float:
        """q − w − (Δh + Δe_c + Δe_p), que cierra a cero (J/kg)."""
        return self.q - self.w - (self.dh + self.dke + self.dpe)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.s_gen)


def _positive(value: float | None, what: str) -> float:
    if value is None or not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{what} tiene que ser mayor que cero.")
    return value


def _check_device(inputs: DeviceInputs) -> None:
    dev, sub, out = inputs.device, inputs.substance, inputs.out
    if dev not in DEVICES:
        raise ValueError(f"Dispositivo desconocido: {dev!r}.")
    if dev not in allowed_devices(sub):
        if dev == "compressor":
            raise ValueError(
                f"Un compresor mueve un gas; {sub.noun} es un líquido: usá una bomba (Çengel §5-4)."
            )
        raise ValueError(
            f"Una bomba mueve un líquido; {sub.noun} es un gas: usá un compresor (Çengel §5-4)."
        )
    if dev != "valve" and out not in allowed_out(dev, sub):
        names = ", ".join(OUT_NAMES[o] for o in allowed_out(dev, sub))
        raise ValueError(f"En {DEVICES[dev].lower()} el dato de salida es uno de: {names}.")
    if dev != "valve" and not math.isfinite(inputs.out_value):
        raise ValueError(f"Falta {OUT_NAMES[out]}.")
    if dev != "valve" and out == "eta" and not 0.0 < inputs.out_value <= 1.0:
        raise ValueError("El rendimiento isoentrópico η_s tiene que estar entre 0 y 1.")
    if dev != "valve" and out == "eta" and inputs.q != 0.0:
        raise ValueError(
            "El rendimiento isoentrópico es de un dispositivo adiabático (vademecum §10.4): "
            "con η_s, el calor q tiene que ser 0."
        )
    for value, name in ((inputs.q, "El calor"), (inputs.w, "El trabajo")):
        if not math.isfinite(value):
            raise ValueError(f"{name} tiene que ser un número.")
    if dev in ("nozzle", "diffuser", "valve") and inputs.w != 0.0:
        raise ValueError(f"{DEVICES[dev]}: no hay trabajo (w = 0).")
    if dev == "valve" and inputs.q != 0.0:
        raise ValueError(
            "Una válvula de estrangulamiento es adiabática (Çengel §5-4): dejá q = 0. Para un "
            "caño con calor, usá el calentador o enfriador."
        )
    for value, name in ((inputs.V1, "La velocidad ω₁"), (inputs.V2, "La velocidad ω₂")):
        if value is not None and not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} no puede ser negativa.")
    for value, name in ((inputs.z1, "La altura z₁"), (inputs.z2, "La altura z₂")):
        if not math.isfinite(value):
            raise ValueError(f"{name} tiene que ser un número.")
    if inputs.flow not in FLOW_NAMES:
        raise ValueError(f"Caudal desconocido: {inputs.flow!r}.")
    _positive(inputs.flow_value, _FLOW_NOUNS[inputs.flow])
    if inputs.flow == "power":
        if _unknown(dev, out) not in ("w", "q"):
            raise ValueError(
                "La potencia sirve de dato del caudal cuando el balance da el trabajo (turbina, "
                "compresor o bomba con T₂, x₂ o η_s) o el calor (calentador con T₂ o x₂)."
            )
        if inputs.rates:
            raise ValueError("Con la potencia como dato, q y w van por kg (el caudal sale de ahí).")
        if inputs.A1_m2 is not None:
            raise ValueError("Con la potencia como dato, dá la velocidad ω₁ (no el área A₁).")
    if inputs.flow == "inlet":
        if not inputs.V1 > 0.0:
            raise ValueError("Con el área de entrada como dato, la velocidad ω₁ tiene que ser > 0.")
        if inputs.A1_m2 is not None:
            raise ValueError("El área de entrada ya es el dato del caudal.")
    if inputs.A1_m2 is not None:
        _positive(inputs.A1_m2, "El área A₁")
    if inputs.p2_Pa is not None:
        _positive(inputs.p2_Pa, "La presión de salida p₂")
    if inputs.T_b_K is not None and not (math.isfinite(inputs.T_b_K) and inputs.T_b_K > 0.0):
        raise ValueError("La temperatura de la fuente T_b tiene que ser positiva (absoluta).")
    if not (math.isfinite(inputs.T0_K) and 200.0 <= inputs.T0_K <= 350.0):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −73 y 77 °C.")


def _unknown(dev: DeviceKind, out: OutKind) -> str:
    """Lo que despeja el balance de energía."""
    if dev == "valve":
        return "h"
    if dev in _WORK_DEVICES:
        return "h" if out == "w" else "w"
    if dev == "heater":
        return "h" if out == "q" else "q"
    return "h" if out == "V" else "V"


def _check_inlet(dev: DeviceKind, sub: Substance, s1: ThermoState) -> None:
    if sub.kind != "fluid":
        return
    liquid = s1.region in ("compressed_liquid", "saturated_liquid")
    if dev == "pump" and not liquid:
        raise ValueError(
            f"Una bomba mueve un líquido, y el estado 1 es {s1.region_es.lower()}. Para un gas "
            "o un vapor, usá un compresor (Çengel §5-4)."
        )
    if dev == "compressor" and liquid:
        raise ValueError(
            "Un compresor mueve un gas o un vapor, y el estado 1 es un líquido: usá una bomba "
            "(Çengel §5-4)."
        )


def _check_pressures(dev: DeviceKind, p1: float, p2: float) -> None:
    if dev == "valve" and p2 >= p1:
        raise ValueError(
            "Una válvula de estrangulamiento baja la presión: p₂ tiene que ser menor que p₁."
        )
    if dev == "turbine" and p2 >= p1:
        raise ValueError("En una turbina el fluido se expande: p₂ tiene que ser menor que p₁.")
    if dev in ("compressor", "pump") and p2 <= p1:
        name = "un compresor" if dev == "compressor" else "una bomba"
        raise ValueError(f"En {name} la presión sube: p₂ tiene que ser mayor que p₁.")
    if dev == "nozzle" and p2 >= p1:
        raise ValueError(
            "En una tobera el fluido se acelera porque baja la presión: p₂ tiene que ser menor "
            "que p₁ (para frenarlo, elegí un difusor)."
        )
    if dev == "diffuser" and p2 <= p1:
        raise ValueError(
            "En un difusor el fluido se frena y la presión sube: p₂ tiene que ser mayor que p₁ "
            "(para acelerarlo, elegí una tobera)."
        )
    if dev == "heater" and p2 > p1:
        raise ValueError(
            "En un caño o un ducto la presión no sube sin una bomba: p₂ tiene que ser menor o "
            "igual que p₁."
        )


def _outlet(sub: Substance, out: OutKind, value: float, p2: float) -> ThermoState:
    if out == "T":
        return state(sub, "TP", value, p2, "2")
    return state(sub, "PX", p2, value, "2")


def _isentropic(sub: Substance, s1: ThermoState, p2: float) -> ThermoState | None:
    try:
        return state(sub, "PS", p2, s1.s, "2s")
    except ValueError:
        return None


def solve_device(inputs: DeviceInputs) -> DeviceResult:
    """Los balances de energía y de entropía de un dispositivo de flujo estacionario.

    Una s_gen < 0 no es un error: el proceso es imposible (``violation`` dice por qué).

    Raises
    ------
    ValueError
        Con datos que no sirven para el dispositivo (una válvula que comprime, una tobera
        que frena, una turbina que consume trabajo) o un estado fuera de rango.
    """
    _check_device(inputs)
    sub, dev, out = inputs.substance, inputs.device, inputs.out
    s1 = state(sub, inputs.pair1, inputs.a1, inputs.b1, "1")
    _check_inlet(dev, sub, s1)
    p2 = inputs.p2_Pa if inputs.p2_Pa is not None else s1.p
    _check_pressures(dev, s1.p, p2)
    unknown = _unknown(dev, out)

    m_dot: float | None = None
    V1 = inputs.V1
    if inputs.flow == "m":
        m_dot = inputs.flow_value
    elif inputs.flow == "Vdot":
        m_dot = inputs.flow_value / s1.v
    elif inputs.flow == "inlet":
        m_dot = inputs.flow_value * V1 / s1.v
    if inputs.A1_m2 is not None:
        assert m_dot is not None
        V1 = m_dot * s1.v / inputs.A1_m2
    q_known, w_known, value = inputs.q, inputs.w, inputs.out_value
    if inputs.rates:
        assert m_dot is not None
        q_known, w_known = q_known / m_dot, w_known / m_dot
        if out in ("q", "w"):
            value = value / m_dot
    dpe = G * (inputs.z2 - inputs.z1)
    V2 = inputs.V2 if inputs.V2 is not None else V1
    s2s: ThermoState | None = None

    if dev == "valve":
        V2 = V1
        q, w = 0.0, 0.0
        s2 = state(sub, "PH", p2, s1.h - dpe, "2")
    elif unknown in ("w", "q"):
        if out == "eta":
            s2s = state(sub, "PS", p2, s1.s, "2s")
            if dev == "turbine":
                h2 = s1.h - value * (s1.h - s2s.h)
            else:
                h2 = s1.h + (s2s.h - s1.h) / value
            s2 = state(sub, "PH", p2, h2, "2")
        else:
            s2 = _outlet(sub, out, value, p2)
        balance = s2.h - s1.h + (V2**2 - V1**2) / 2.0 + dpe
        if unknown == "w":
            q, w = q_known, q_known - balance
        else:
            q, w = w_known + balance, w_known
    elif unknown == "h":  # trabajo (turbina, compresor, bomba), calor (calentador) u ω₂ dados
        w = value if out == "w" else w_known
        q = value if out == "q" else q_known
        if out == "V":
            V2 = value
        h2 = s1.h + q - w - (V2**2 - V1**2) / 2.0 - dpe
        s2 = state(sub, "PH", p2, h2, "2")
    else:  # tobera o difusor: el balance da ω₂
        q, w = q_known, 0.0
        if out == "eta":
            s2s = state(sub, "PS", p2, s1.s, "2s")
            ke2s = V1**2 / 2.0 + s1.h - s2s.h - dpe
            ke2 = value * ke2s
            h2 = s1.h + V1**2 / 2.0 - ke2 - dpe
            s2 = state(sub, "PH", p2, h2, "2")
        else:
            s2 = _outlet(sub, out, value, p2)
            ke2 = V1**2 / 2.0 + q - (s2.h - s1.h) - dpe
        if ke2 < 0.0:
            raise ValueError(
                f"Con esa salida la energía no alcanza: ω₂²/2 = {_kJkg(ke2)} < 0. Revisá "
                f"{OUT_NAMES[out]} (con menos entalpía a la salida, el fluido gana velocidad)."
            )
        V2 = math.sqrt(2.0 * ke2)

    if dev == "nozzle" and V2 <= V1:
        raise ValueError(
            f"En una tobera el fluido se acelera, y con estos datos frena (ω₁ = {num(V1)} m/s, "
            f"ω₂ = {num(V2)} m/s). Para frenarlo, elegí un difusor."
        )
    if dev == "diffuser" and V2 >= V1:
        raise ValueError(
            f"En un difusor el fluido se frena, y con estos datos se acelera (ω₁ = {num(V1)} "
            f"m/s, ω₂ = {num(V2)} m/s). Para acelerarlo, elegí una tobera."
        )
    if dev == "turbine" and w <= 0.0:
        raise ValueError(
            f"Con estos datos la turbina no entrega trabajo (w = {_kJkg(w)}): revisá el estado "
            "de salida o el calor."
        )
    if dev in ("compressor", "pump") and w >= 0.0:
        raise ValueError(
            f"Con estos datos {DEVICES[dev].lower()} no consume trabajo (w = {_kJkg(w)}): "
            "revisá el estado de salida o el calor."
        )

    if inputs.flow == "power":
        term = w if unknown == "w" else q
        if term == 0.0:
            raise ValueError(
                "Con un trabajo (o un calor) nulo por kg, la potencia no da el caudal."
            )
        m_dot = inputs.flow_value / abs(term)
    assert m_dot is not None

    if s2s is None and dev in _ETA_DEVICES:
        s2s = _isentropic(sub, s1, p2)
    eta = _eta(dev, inputs, s1, s2, s2s, V1, V2, q, dpe)
    T_b = inputs.T_b_K if inputs.T_b_K is not None else inputs.T0_K
    ds = s2.s - s1.s
    s_gen = snap_sgen(ds - q / T_b, abs(s1.s) + abs(s2.s) + S_FLOOR + abs(q / T_b))
    T_lo, T_hi = min(s1.T, s2.T), max(s1.T, s2.T)
    violation = heat_direction_message(q, T_b, T_lo, T_hi, "el fluido") or (
        violation_message(
            f"s_gen = {num(s_gen * 1e-3)} kJ/(kg·K)",
            q,
            ds,
            T_b,
            min(s1.T, s2.T),
            max(s1.T, s2.T),
            "el fluido",
        )
        if s_gen < 0.0
        else None
    )
    result = DeviceResult(
        inputs=inputs,
        state1=s1,
        state2=s2,
        state2s=s2s,
        m_dot=m_dot,
        V1=V1,
        V2=V2,
        q=q,
        w=w,
        T_b=T_b,
        s_gen=s_gen,
        eta_s=eta,
        unknown=unknown,
        path=_device_path(sub, dev, s1, s2),
        path_s=_isentropic_path(sub, s1, s2s),
        violation=violation,
    )
    return replace(result, notes=_device_notes(result))


def _eta(
    dev: DeviceKind,
    inputs: DeviceInputs,
    s1: ThermoState,
    s2: ThermoState,
    s2s: ThermoState | None,
    V1: float,
    V2: float,
    q: float,
    dpe: float,
) -> float | None:
    """η_s del vademecum §10.4: el de los datos o el que resulta, si es adiabático."""
    if inputs.out == "eta" and dev != "valve":
        return inputs.out_value
    if s2s is None or q != 0.0 or dev not in _ETA_DEVICES:
        return None
    if dev == "turbine":
        den = s1.h - s2s.h
        return (s1.h - s2.h) / den if den > 0.0 else None
    if dev in ("compressor", "pump"):
        den = s2.h - s1.h
        return (s2s.h - s1.h) / den if den > 0.0 else None
    ke2s = V1**2 / 2.0 + s1.h - s2s.h - dpe  # tobera: η_N = ω₂²/ω₂s²
    return (V2**2 / 2.0) / ke2s if ke2s > 0.0 else None


def _device_path(
    sub: Substance, dev: DeviceKind, s1: ThermoState, s2: ThermoState
) -> tuple[ThermoState, ...]:
    """Isobara si p₂ = p₁ y entalpía casi constante en la válvula; si no, los dos extremos."""
    if math.isclose(s1.p, s2.p, rel_tol=1e-12):
        ps: Any = [s1.p] * _PATH_POINTS
    elif dev == "valve":
        ps = np.geomspace(s1.p, s2.p, _PATH_POINTS)
    else:
        return (s1, s2)
    out = []
    for p, h in zip(ps, np.linspace(s1.h, s2.h, _PATH_POINTS), strict=True):
        try:
            out.append(state(sub, "PH", float(p), float(h)))
        except ValueError:
            continue
    return tuple(out) if len(out) >= 2 else (s1, s2)


def _isentropic_path(
    sub: Substance, s1: ThermoState, s2s: ThermoState | None
) -> tuple[ThermoState, ...]:
    if s2s is None:
        return ()
    out = []
    for p in np.geomspace(s1.p, s2s.p, _PATH_POINTS):
        try:
            out.append(state(sub, "PS", float(p), s1.s))
        except ValueError:
            continue
    return tuple(out)


def _device_notes(r: DeviceResult) -> tuple[str, ...]:
    notes: list[str] = []
    inputs, s1, s2 = r.inputs, r.state1, r.state2
    sub, dev = inputs.substance, inputs.device
    if dev == "valve":
        if sub.kind == "ideal_gas":
            notes.append(
                "Con un gas ideal h depende solo de T: en la válvula T₂ = T₁, aunque la "
                "entropía sube (s₂ − s₁ = R·ln(p₁/p₂))."
            )
        elif sub.kind == "incompressible":
            notes.append(
                "Con un incompresible, h₂ = h₁ da c·(T₂ − T₁) = v·(p₁ − p₂): el líquido se "
                "calienta un poco al estrangularse."
            )
        else:
            notes.append(
                f"La entalpía no cambia pero la temperatura sí: de {deg_c(s1.T)} a "
                f"{deg_c(s2.T)} (efecto Joule–Thomson)."
            )
    if sub.kind == "fluid" and s2.is_two_phase:
        x2 = num(s2.x or 0.0, ".3g")
        notes.append(f"La salida es una mezcla de líquido y vapor (x₂ = {x2}).")
    if sub.kind == "fluid" and dev == "compressor" and s1.is_two_phase:
        notes.append(
            "Entra una mezcla al compresor: las gotas lo dañarían (en un ciclo real entra vapor "
            "saturado o sobrecalentado)."
        )
    if r.eta_s is not None and inputs.out != "eta":
        notes.append(
            f"Es adiabático: el rendimiento isoentrópico resulta η_s = {num(r.eta_s, '.3f')} "
            "(vademecum §10.4)."
        )
    if dev in _ETA_DEVICES and r.q != 0.0:
        notes.append("Con calor, el rendimiento isoentrópico no está definido (vademecum §10.4).")
    big = abs(r.dh) + abs(r.q) + abs(r.w)
    if (r.dke != 0.0 or r.dpe != 0.0) and dev not in ("nozzle", "diffuser"):
        if abs(r.dke) + abs(r.dpe) < 0.02 * big:
            notes.append(
                f"Δe_c = {_kJkg(r.dke)} y Δe_p = {_kJkg(r.dpe)} son chicas frente a "
                f"Δh = {_kJkg(r.dh)}: en estos dispositivos suelen despreciarse."
            )
    if r.violation is None:
        if r.s_gen == 0.0:
            notes.append("s_gen = 0: el proceso es reversible.")
        elif dev == "valve":
            notes.append(
                "La válvula no hace trabajo ni intercambia calor, pero genera entropía: la "
                "caída de presión se disipa."
            )
    return tuple(notes)


# ---------------------------------------------------------------------
# Cámara de mezcla
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MixingStream:
    """Una entrada de la cámara de mezcla; ``m_dot = None`` es la incógnita."""

    pair: PairCode
    a: float
    b: float
    m_dot: float | None
    label: str = ""


@dataclass(frozen=True)
class MixingInputs:
    """Dos corrientes de la misma sustancia que se mezclan (Çengel §5-4 y §7-13).

    O se da la salida (``out`` = T o x, a ``p_out_Pa``) y un caudal es la incógnita, o se
    dan los dos caudales y el balance da la salida (``out = "h"``). ``Q_dot_W`` es el calor
    que recibe la cámara (negativo si pierde).
    """

    substance: Substance
    inlet_1: MixingStream
    inlet_2: MixingStream
    out: Literal["T", "x", "h"] = "h"
    out_value: float = math.nan
    p_out_Pa: float | None = None  # None: la menor de las entradas
    Q_dot_W: float = 0.0
    T_b_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class MixingResult:
    inputs: MixingInputs
    inlets: tuple[ThermoState, ThermoState]
    outlet: ThermoState
    m_dots: tuple[float, float]
    T_b: float
    S_gen_dot: float  # W/K
    violation: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def m_out(self) -> float:
        return self.m_dots[0] + self.m_dots[1]

    @property
    def X_dest_dot(self) -> float:
        return self.inputs.T0_K * self.S_gen_dot

    @property
    def energy_residual(self) -> float:
        """Q̇ − (ṁ₃·h₃ − ṁ₁·h₁ − ṁ₂·h₂), que cierra a cero (W)."""
        (a, b), (m1, m2) = self.inlets, self.m_dots
        return self.inputs.Q_dot_W - (self.m_out * self.outlet.h - m1 * a.h - m2 * b.h)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen_dot)


def _stream_name(s: MixingStream | ExchangerStream, i: int) -> str:
    return s.label or f"la corriente {i}"


def solve_mixing(inputs: MixingInputs) -> MixingResult:
    """ṁ₁·h₁ + ṁ₂·h₂ + Q̇ = ṁ₃·h₃ y Ṡ_gen = ṁ₃·s₃ − ṁ₁·s₁ − ṁ₂·s₂ − Q̇/T_b.

    Raises
    ------
    ValueError
        Si falta o sobra un dato, la salida tiene más presión que las entradas o un caudal
        sale negativo (la salida fuera del rango de las entradas).
    """
    sub = inputs.substance
    st1, st2 = inputs.inlet_1, inputs.inlet_2
    unknowns = [s.m_dot is None for s in (st1, st2)] + [inputs.out == "h"]
    if sum(unknowns) != 1:
        raise ValueError(
            "La cámara de mezcla lleva una incógnita: un caudal de entrada (con la salida dada) "
            "o la salida (con los dos caudales dados)."
        )
    for i, s in enumerate((st1, st2), start=1):
        if s.m_dot is not None and not (math.isfinite(s.m_dot) and s.m_dot > 0.0):
            raise ValueError(f"El caudal de {_stream_name(s, i)} tiene que ser positivo.")
    if not math.isfinite(inputs.Q_dot_W):
        raise ValueError("El calor Q̇ tiene que ser un número.")
    _check_ambient(inputs.T_b_K, inputs.T0_K)
    a = state(sub, st1.pair, st1.a, st1.b, "1")
    b = state(sub, st2.pair, st2.a, st2.b, "2")
    p3 = inputs.p_out_Pa if inputs.p_out_Pa is not None else min(a.p, b.p)
    if not (math.isfinite(p3) and p3 > 0.0):
        raise ValueError("La presión de salida tiene que ser positiva.")
    if p3 > min(a.p, b.p) * (1.0 + 1e-9):
        raise ValueError(
            "La presión de salida no puede superar a la de las entradas: el fluido no sube "
            "solo de presión al mezclarse."
        )
    Q = inputs.Q_dot_W
    if inputs.out == "h":
        assert st1.m_dot is not None and st2.m_dot is not None
        m1, m2 = st1.m_dot, st2.m_dot
        c = state(sub, "PH", p3, (m1 * a.h + m2 * b.h + Q) / (m1 + m2), "3")
    else:
        if not math.isfinite(inputs.out_value):
            raise ValueError("Falta el dato de la salida.")
        if inputs.out == "T":
            c = state(sub, "TP", inputs.out_value, p3, "3")
        else:
            c = state(sub, "PX", p3, inputs.out_value, "3")
        if st1.m_dot is None:
            assert st2.m_dot is not None
            m2 = st2.m_dot
            m1 = _mixing_flow(m2, b.h, a.h, c.h, Q, _stream_name(st1, 1))
        else:
            m1 = st1.m_dot
            m2 = _mixing_flow(m1, a.h, b.h, c.h, Q, _stream_name(st2, 2))
    T_b = inputs.T_b_K if inputs.T_b_K is not None else inputs.T0_K
    flows = (m1 * (c.s - a.s), m2 * (c.s - b.s))
    scale = sum(m * (abs(x.s) + abs(c.s) + S_FLOOR) for m, x in ((m1, a), (m2, b)))
    S_gen = snap_sgen(sum(flows) - Q / T_b, scale + abs(Q / T_b))
    temps = (a.T, b.T, c.T)
    violation = heat_direction_message(Q, T_b, min(temps), max(temps), "el fluido") or (
        violation_message(
            f"Ṡ_gen = {num(S_gen)} W/K", Q, sum(flows), T_b, min(temps), max(temps), "el fluido"
        )
        if S_gen < 0.0
        else None
    )
    result = MixingResult(inputs, (a, b), c, (m1, m2), T_b, S_gen, violation)
    notes = []
    if Q == 0.0:
        notes.append(
            "Sin calor, la salida queda entre las dos entradas, y mezclar dos corrientes a "
            "distinta temperatura siempre genera entropía."
        )
    if sub.kind == "fluid" and c.is_two_phase:
        notes.append(f"La salida es una mezcla de líquido y vapor (x₃ = {num(c.x or 0.0, '.3g')}).")
    return replace(result, notes=tuple(notes))


def _mixing_flow(
    m_known: float, h_known: float, h_other: float, h3: float, Q: float, name: str
) -> float:
    """El caudal de la otra entrada: ṁ·(h − h₃) = ṁ_dato·(h₃ − h_dato) − Q̇."""
    den = h_other - h3
    if den == 0.0:
        raise ValueError(
            f"La salida tiene la misma entalpía que {name}: el caudal no se puede despejar."
        )
    m = (m_known * (h3 - h_known) - Q) / den
    if not m > 0.0:
        raise ValueError(
            f"Con esa salida el caudal de {name} sale negativo ({num(m)} kg/s): sin calor, la "
            "salida tiene que quedar entre las dos entradas."
        )
    return m


def _check_ambient(T_b: float | None, T0: float) -> None:
    if T_b is not None and not (math.isfinite(T_b) and T_b > 0.0):
        raise ValueError("La temperatura de la fuente T_b tiene que ser positiva (absoluta).")
    if not (math.isfinite(T0) and 200.0 <= T0 <= 350.0):
        raise ValueError("La temperatura del ambiente T₀ tiene que estar entre −73 y 77 °C.")


# ---------------------------------------------------------------------
# Intercambiador sin mezcla
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangerStream:
    """Una corriente del intercambiador; ``m_dot = None`` u ``out = "h"`` es la incógnita."""

    substance: Substance
    pair: PairCode
    a: float
    b: float
    m_dot: float | None
    out: Literal["T", "x", "h"] = "T"
    out_value: float = math.nan
    p_out_Pa: float | None = None  # None: sin pérdida de carga
    label: str = ""


@dataclass(frozen=True)
class ExchangerInputs:
    """Dos corrientes que intercambian calor sin mezclarse (Çengel §5-4).

    Una sola incógnita: un caudal o una salida. ``Q_dot_W`` es el calor que recibe el
    intercambiador desde afuera (negativo si pierde hacia el ambiente).
    """

    stream_a: ExchangerStream
    stream_b: ExchangerStream
    Q_dot_W: float = 0.0
    T_b_K: float | None = None
    T0_K: float = 298.15


@dataclass(frozen=True)
class ExchangerResult:
    inputs: ExchangerInputs
    inlets: tuple[ThermoState, ThermoState]
    outlets: tuple[ThermoState, ThermoState]
    m_dots: tuple[float, float]
    T_b: float
    S_gen_dot: float
    violation: str | None = None
    notes: tuple[str, ...] = ()

    def heat(self, i: int) -> float:
        """El calor que recibe la corriente ``i`` (0 o 1): ṁ·(h_sal − h_ent) (W)."""
        return self.m_dots[i] * (self.outlets[i].h - self.inlets[i].h)

    @property
    def Q_transfer(self) -> float:
        """El calor que pasa de la corriente caliente a la fría (W)."""
        return max(-self.heat(0), -self.heat(1))

    def entropy_change(self, i: int) -> float:
        return self.m_dots[i] * (self.outlets[i].s - self.inlets[i].s)

    @property
    def X_dest_dot(self) -> float:
        return self.inputs.T0_K * self.S_gen_dot

    @property
    def energy_residual(self) -> float:
        return self.inputs.Q_dot_W - self.heat(0) - self.heat(1)

    @property
    def verdict(self) -> Verdict:
        return "imposible" if self.violation else verdict(self.S_gen_dot)


def _exchanger_outlet(s: ExchangerStream, inlet: ThermoState, label: str) -> ThermoState:
    p = s.p_out_Pa if s.p_out_Pa is not None else inlet.p
    if not (math.isfinite(p) and p > 0.0):
        raise ValueError("La presión de salida tiene que ser positiva.")
    if p > inlet.p * (1.0 + 1e-9):
        raise ValueError("En un intercambiador la presión de cada corriente no sube.")
    if not math.isfinite(s.out_value):
        raise ValueError(f"Falta la salida de {s.label or 'una corriente'}.")
    if s.out == "T":
        return state(s.substance, "TP", s.out_value, p, label)
    return state(s.substance, "PX", p, s.out_value, label)


def solve_exchanger(inputs: ExchangerInputs) -> ExchangerResult:
    """Σ ṁ·(h_sal − h_ent) = Q̇ y Ṡ_gen = Σ ṁ·(s_sal − s_ent) − Q̇/T_b.

    Raises
    ------
    ValueError
        Si falta o sobra un dato o un caudal sale negativo.
    """
    sa, sb = inputs.stream_a, inputs.stream_b
    unknowns = [sa.m_dot is None, sb.m_dot is None, sa.out == "h", sb.out == "h"]
    if sum(unknowns) != 1:
        raise ValueError(
            "El intercambiador lleva una sola incógnita: un caudal o una temperatura de salida."
        )
    for i, s in enumerate((sa, sb), start=1):
        if s.m_dot is not None and not (math.isfinite(s.m_dot) and s.m_dot > 0.0):
            raise ValueError(f"El caudal de {_stream_name(s, i)} tiene que ser positivo.")
    if not math.isfinite(inputs.Q_dot_W):
        raise ValueError("El calor Q̇ tiene que ser un número.")
    _check_ambient(inputs.T_b_K, inputs.T0_K)
    ia = state(sa.substance, sa.pair, sa.a, sa.b, "A1")
    ib = state(sb.substance, sb.pair, sb.a, sb.b, "B1")
    Q = inputs.Q_dot_W
    if sa.out == "h" or sb.out == "h":
        known, other = (sb, sa) if sa.out == "h" else (sa, sb)
        i_known, i_other = (ib, ia) if sa.out == "h" else (ia, ib)
        o_known = _exchanger_outlet(known, i_known, "B2" if known is sb else "A2")
        assert known.m_dot is not None and other.m_dot is not None
        p = other.p_out_Pa if other.p_out_Pa is not None else i_other.p
        h = i_other.h + (Q - known.m_dot * (o_known.h - i_known.h)) / other.m_dot
        o_other = state(other.substance, "PH", p, h, "A2" if other is sa else "B2")
        oa, ob = (o_other, o_known) if other is sa else (o_known, o_other)
        m_a, m_b = sa.m_dot, sb.m_dot
    else:
        oa = _exchanger_outlet(sa, ia, "A2")
        ob = _exchanger_outlet(sb, ib, "B2")
        dha, dhb = oa.h - ia.h, ob.h - ib.h
        if sa.m_dot is None:
            assert sb.m_dot is not None
            m_b = sb.m_dot
            m_a = _exchanger_flow(Q - m_b * dhb, dha, _stream_name(sa, 1))
        else:
            m_a = sa.m_dot
            assert m_a is not None
            m_b = _exchanger_flow(Q - m_a * dha, dhb, _stream_name(sb, 2))
    assert m_a is not None and m_b is not None
    T_b = inputs.T_b_K if inputs.T_b_K is not None else inputs.T0_K
    ds = (m_a * (oa.s - ia.s), m_b * (ob.s - ib.s))
    scale = m_a * (abs(ia.s) + abs(oa.s) + S_FLOOR) + m_b * (abs(ib.s) + abs(ob.s) + S_FLOOR)
    S_gen = snap_sgen(sum(ds) - Q / T_b, scale + abs(Q / T_b))
    result = ExchangerResult(inputs, (ia, ib), (oa, ob), (m_a, m_b), T_b, S_gen)
    violation = _exchanger_violation(result)
    return replace(result, violation=violation, notes=_exchanger_notes(result))


def _exchanger_flow(rhs: float, dh: float, name: str) -> float:
    if dh == 0.0:
        raise ValueError(
            f"{name.capitalize()} no cambia de entalpía: su caudal no se puede despejar."
        )
    m = rhs / dh
    if not m > 0.0:
        raise ValueError(
            f"Con esas temperaturas el caudal de {name} sale negativo ({num(m)} kg/s): una "
            "corriente se tiene que enfriar y la otra calentar."
        )
    return m


def _exchanger_violation(r: ExchangerResult) -> str | None:
    """El calor pasa de la corriente caliente a la fría, y ninguna cruza la entrada de la otra."""
    hot = 0 if r.heat(0) < 0.0 else 1
    cold = 1 - hot
    if r.heat(hot) >= 0.0 or r.heat(cold) <= 0.0:
        return None  # un balance con calor de afuera en las dos: lo juzga S_gen
    names = [
        r.inputs.stream_a.label or "la corriente A",
        r.inputs.stream_b.label or "la corriente B",
    ]
    t_hot_in, t_cold_in = r.inlets[hot].T, r.inlets[cold].T
    base = "El proceso es imposible (vademecum §10.2): "
    if t_hot_in <= t_cold_in:
        return (
            f"{base}{names[hot]} se enfría y {names[cold]} se calienta, pero entra más fría "
            f"({deg_c(t_hot_in)} contra {deg_c(t_cold_in)}): el calor no pasa solo de frío a "
            "caliente."
        )
    if r.outlets[cold].T > t_hot_in + 1e-9:
        return (
            f"{base}{names[cold]} saldría a {deg_c(r.outlets[cold].T)}, más caliente que la "
            f"entrada de {names[hot]} ({deg_c(t_hot_in)}): ningún intercambiador lo logra."
        )
    if r.outlets[hot].T < t_cold_in - 1e-9:
        return (
            f"{base}{names[hot]} saldría a {deg_c(r.outlets[hot].T)}, más fría que la entrada "
            f"de {names[cold]} ({deg_c(t_cold_in)}): ningún intercambiador lo logra."
        )
    temps = [s.T for s in r.inlets + r.outlets]
    direction = heat_direction_message(
        r.inputs.Q_dot_W, r.T_b, min(temps), max(temps), "las corrientes"
    )
    if direction is not None:
        return direction
    if r.S_gen_dot < 0.0:
        return violation_message(
            f"Ṡ_gen = {num(r.S_gen_dot)} W/K",
            r.inputs.Q_dot_W,
            r.entropy_change(0) + r.entropy_change(1),
            r.T_b,
            min(s.T for s in r.inlets + r.outlets),
            max(s.T for s in r.inlets + r.outlets),
            "las corrientes",
        )
    return None


def _exchanger_notes(r: ExchangerResult) -> tuple[str, ...]:
    notes = [
        "El calor que entrega la corriente caliente lo recibe la fría (menos lo que se pierde "
        "al ambiente); la entropía que gana la fría es mayor que la que pierde la caliente, "
        "porque lo recibe a menor temperatura."
    ]
    for s, o in zip((r.inputs.stream_a, r.inputs.stream_b), r.outlets, strict=True):
        if s.substance.kind == "fluid" and o.is_two_phase:
            name = s.label or "una corriente"
            notes.append(f"{name.capitalize()} sale como mezcla (x = {num(o.x or 0.0, '.3g')}).")
    return tuple(notes)


# ---------------------------------------------------------------------
# Ejemplos y export
# ---------------------------------------------------------------------

C = 273.15
_PSIA = 6894.757293168
_LBM = 0.45359237
_BTU = 1055.05585262


def _F(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0 + C


@dataclass(frozen=True)
class DeviceExample:
    inputs: DeviceInputs
    description: str


@dataclass(frozen=True)
class MixingExample:
    inputs: MixingInputs
    description: str


@dataclass(frozen=True)
class ExchangerExample:
    inputs: ExchangerInputs
    description: str


_WATER = fluid_substance("Water")
_AIR = ideal_gas_substance("air", "variable")
_R134A = fluid_substance("R134a")

DEVICE_EXAMPLES: dict[str, DeviceExample] = {
    "Difusor de un motor a reacción (Çengel §5-4)": DeviceExample(
        DeviceInputs(
            "diffuser",
            _AIR,
            "TP",
            10.0 + C,
            80e3,
            p2_Pa=100e3,
            out="V",
            out_value=0.0,
            flow="inlet",
            flow_value=0.4,
            V1=200.0,
            T0_K=10.0 + C,
        ),
        "Aire a 10 °C y 80 kPa entra a 200 m/s por 0,4 m² y sale casi quieto: el libro da "
        "ṁ = 78,8 kg/s y T₂ = 303 K. No da p₂ (con un gas ideal T₂ no depende de ella): con "
        "100 kPa el difusor genera un poco de entropía.",
    ),
    "Tobera de vapor que pierde calor (Çengel §5-4)": DeviceExample(
        DeviceInputs(
            "nozzle",
            _WATER,
            "TP",
            400.0 + C,
            1.8e6,
            p2_Pa=1.4e6,
            out="V",
            out_value=275.0,
            flow="m",
            flow_value=5.0,
            A1_m2=0.02,
            q=-2.8e3,
            T0_K=25.0 + C,
        ),
        "Vapor a 1,8 MPa y 400 °C, 5 kg/s por 0,02 m², sale a 1,4 MPa y 275 m/s y pierde "
        "2,8 kJ/kg: el libro da ω₁ = 42,1 m/s y T₂ = 378,6 °C.",
    ),
    "Compresor de aire que pierde calor (Çengel §5-4)": DeviceExample(
        DeviceInputs(
            "compressor",
            _AIR,
            "TP",
            280.0,
            100e3,
            p2_Pa=600e3,
            out="T",
            out_value=400.0,
            flow="m",
            flow_value=0.02,
            q=-16e3,
            T0_K=280.0,
        ),
        "Aire de 100 kPa y 280 K a 600 kPa y 400 K, 0,02 kg/s, con una pérdida de 16 kJ/kg: "
        "el libro da 2,74 kW de potencia de entrada. El balance de entropía muestra que así es "
        "imposible: con tan poco calor el aire no puede salir tan frío (habría que cederlo a "
        "un medio a menos de 103 K).",
    ),
    "Turbina de vapor de 5 MW (Çengel §5-4)": DeviceExample(
        DeviceInputs(
            "turbine",
            _WATER,
            "TP",
            400.0 + C,
            2e6,
            p2_Pa=15e3,
            out="x",
            out_value=0.9,
            flow="power",
            flow_value=5e6,
            V1=50.0,
            V2=180.0,
            z1=10.0,
            z2=6.0,
            T0_K=25.0 + C,
        ),
        "Vapor a 2 MPa y 400 °C (50 m/s, 10 m) que sale a 15 kPa con x = 0,9 (180 m/s, 6 m) "
        "en una turbina adiabática de 5 MW: el libro da w = 872,5 kJ/kg y ṁ = 5,73 kg/s.",
    ),
    "Válvula de expansión de R-134a (Çengel §5-4)": DeviceExample(
        DeviceInputs(
            "valve",
            _R134A,
            "PX",
            0.8e6,
            0.0,
            p2_Pa=0.12e6,
            flow="m",
            flow_value=1.0,
            T0_K=25.0 + C,
        ),
        "R-134a líquido saturado a 0,8 MPa laminado hasta 0,12 MPa: el libro da x₂ = 0,340 y "
        "T₂ = −22,32 °C.",
    ),
    "Calefacción eléctrica de aire en un ducto (Çengel §5-4)": DeviceExample(
        DeviceInputs(
            "heater",
            _AIR,
            "TP",
            17.0 + C,
            100e3,
            out="q",
            out_value=-200.0,
            flow="Vdot",
            flow_value=150.0 / 60.0,
            w=-15e3,
            rates=True,
            T0_K=17.0 + C,
        ),
        "Una resistencia de 15 kW calienta 150 m³/min de aire a 100 kPa y 17 °C que pierde "
        "200 W en el ducto: el libro da T₂ = 21,9 °C.",
    ),
    "Rendimiento isoentrópico de una turbina de vapor (Çengel §7-12)": DeviceExample(
        DeviceInputs(
            "turbine",
            _WATER,
            "TP",
            400.0 + C,
            3e6,
            p2_Pa=50e3,
            out="T",
            out_value=100.0 + C,
            flow="power",
            flow_value=2e6,
            T0_K=25.0 + C,
        ),
        "Vapor de 3 MPa y 400 °C a 50 kPa y 100 °C en una turbina adiabática de 2 MW: el "
        "libro da η_s = 66,7 % y ṁ = 3,64 kg/s.",
    ),
    "Laminación de vapor (Çengel §7-13)": DeviceExample(
        DeviceInputs(
            "valve",
            _WATER,
            "TP",
            450.0 + C,
            7e6,
            p2_Pa=3e6,
            flow="m",
            flow_value=1.0,
            T0_K=25.0 + C,
        ),
        "Vapor a 7 MPa y 450 °C laminado hasta 3 MPa: el libro da s_gen = 0,369 kJ/(kg·K).",
    ),
    "Caldera: agua que sale como vapor": DeviceExample(
        DeviceInputs(
            "heater",
            _WATER,
            "TP",
            40.0 + C,
            5e6,
            out="T",
            out_value=500.0 + C,
            flow="m",
            flow_value=10.0,
            T_b_K=1200.0 + C,
            T0_K=25.0 + C,
        ),
        "10 kg/s de agua a 5 MPa y 40 °C que salen como vapor a 500 °C, calentados por gases "
        "a 1200 °C: casi toda la irreversibilidad de una central está acá.",
    ),
    "Bomba de agua (incompresible)": DeviceExample(
        DeviceInputs(
            "pump",
            incompressible_substance("water"),
            "TP",
            20.0 + C,
            100e3,
            p2_Pa=5e6,
            out="eta",
            out_value=0.75,
            flow="m",
            flow_value=10.0,
            T0_K=20.0 + C,
        ),
        "10 kg/s de agua de 100 kPa a 5 MPa con η_s = 75 %: w_ideal = v·Δp y lo que falta se "
        "disipa y calienta el agua.",
    ),
    "Tobera de aire con rendimiento isoentrópico": DeviceExample(
        DeviceInputs(
            "nozzle",
            _AIR,
            "TP",
            500.0,
            300e3,
            p2_Pa=200e3,
            out="eta",
            out_value=0.95,
            flow="m",
            flow_value=2.0,
            V1=30.0,
            T0_K=25.0 + C,
        ),
        "2 kg/s de aire a 300 kPa y 500 K (30 m/s) que se expanden hasta 200 kPa en una "
        "tobera con η_N = 95 % (sale a velocidad subsónica).",
    ),
}

MIXING_EXAMPLES: dict[str, MixingExample] = {
    "Ducha: agua caliente y fría (Çengel §5-4)": MixingExample(
        MixingInputs(
            _WATER,
            MixingStream("TP", _F(140.0), 20.0 * _PSIA, None, "el agua caliente"),
            MixingStream("TP", _F(50.0), 20.0 * _PSIA, 1.0 * _LBM, "el agua fría"),
            out="T",
            out_value=_F(110.0),
            T0_K=_F(70.0),
        ),
        "Agua caliente a 140 °F con agua fría a 50 °F para salir a 110 °F (20 psia): el libro "
        "da ṁ_caliente/ṁ_fría = 2,0 (acá, 1 lbm/s de agua fría).",
    ),
    "Cámara de mezcla que pierde calor (Çengel §7-13)": MixingExample(
        MixingInputs(
            _WATER,
            MixingStream("TP", _F(50.0), 20.0 * _PSIA, 300.0 * _LBM / 60.0, "el agua"),
            MixingStream("TP", _F(240.0), 20.0 * _PSIA, None, "el vapor"),
            out="T",
            out_value=_F(130.0),
            Q_dot_W=-180.0 * _BTU / 60.0,
            T0_K=_F(70.0),
        ),
        "300 lbm/min de agua a 50 °F con vapor a 240 °F (20 psia) salen a 130 °F y pierden "
        "180 Btu/min al aire a 70 °F: el libro da 22,7 lbm/min de vapor y "
        "Ṡ_gen = 8,65 Btu/(min·R).",
    ),
    "Mezcla de dos corrientes de aire": MixingExample(
        MixingInputs(
            _AIR,
            MixingStream("TP", 15.0 + C, 101325.0, 2.0, "el aire exterior"),
            MixingStream("TP", 35.0 + C, 101325.0, 1.0, "el aire de retorno"),
            T0_K=15.0 + C,
        ),
        "2 kg/s de aire exterior a 15 °C con 1 kg/s de aire de retorno a 35 °C, a 1 atm: la "
        "salida y la entropía que genera mezclarlas.",
    ),
}

EXCHANGER_EXAMPLES: dict[str, ExchangerExample] = {
    "Condensador de R-134a enfriado con agua (Çengel §5-4)": ExchangerExample(
        ExchangerInputs(
            ExchangerStream(
                _R134A, "TP", 70.0 + C, 1e6, 6.0 / 60.0, "T", 35.0 + C, label="el R-134a"
            ),
            ExchangerStream(_WATER, "TP", 15.0 + C, 300e3, None, "T", 25.0 + C, label="el agua"),
            T0_K=15.0 + C,
        ),
        "6 kg/min de R-134a a 1 MPa de 70 a 35 °C, enfriados con agua a 300 kPa de 15 a 25 °C: "
        "el libro da 29,1 kg/min de agua y 1218 kJ/min.",
    ),
    "Enfriador de aceite con agua": ExchangerExample(
        ExchangerInputs(
            ExchangerStream(
                incompressible_substance("oil"),
                "TP",
                150.0 + C,
                300e3,
                2.0,
                "h",
                label="el aceite",
            ),
            ExchangerStream(
                incompressible_substance("water"),
                "TP",
                20.0 + C,
                200e3,
                1.5,
                "T",
                70.0 + C,
                label="el agua",
            ),
            T0_K=20.0 + C,
        ),
        "2 kg/s de aceite a 150 °C enfriados con 1,5 kg/s de agua que pasa de 20 a 70 °C "
        "(incompresibles): ¿a qué temperatura sale el aceite?",
    ),
}


def _state_dict(s: ThermoState) -> dict[str, Any]:
    return {
        "p_Pa": s.p,
        "T_K": s.T,
        "v_m3_kg": s.v,
        "h_J_kg": s.h,
        "s_J_kgK": s.s,
        "x": s.x,
        "region": s.region_es,
    }


def device_to_dict(r: DeviceResult) -> dict[str, Any]:
    inp = r.inputs
    return {
        "dispositivo": DEVICES[inp.device],
        "sustancia": inp.substance.label,
        "estado_1": _state_dict(r.state1),
        "estado_2": _state_dict(r.state2),
        "estado_2s": _state_dict(r.state2s) if r.state2s is not None else None,
        "m_dot_kg_s": r.m_dot,
        "omega_1_m_s": r.V1,
        "omega_2_m_s": r.V2,
        "A1_m2": r.A1,
        "A2_m2": r.A2,
        "q_J_kg": r.q,
        "w_J_kg": r.w,
        "dh_J_kg": r.dh,
        "dec_J_kg": r.dke,
        "dep_J_kg": r.dpe,
        "Q_dot_W": r.Q_dot,
        "W_dot_W": r.W_dot,
        "eta_s": r.eta_s,
        "T_b_K": r.T_b,
        "s_gen_J_kgK": r.s_gen,
        "S_gen_dot_W_K": r.S_gen_dot,
        "X_dest_dot_W": r.X_dest_dot,
        "T0_K": inp.T0_K,
        "veredicto": r.verdict,
        "notas": list(r.notes),
    }


def mixing_to_dict(r: MixingResult) -> dict[str, Any]:
    names = [r.inputs.inlet_1.label or "entrada 1", r.inputs.inlet_2.label or "entrada 2"]
    return {
        "sustancia": r.inputs.substance.label,
        "entradas": [
            {"corriente": n, "m_dot_kg_s": m, **_state_dict(s)}
            for n, m, s in zip(names, r.m_dots, r.inlets, strict=True)
        ],
        "salida": {"m_dot_kg_s": r.m_out, **_state_dict(r.outlet)},
        "Q_dot_W": r.inputs.Q_dot_W,
        "T_b_K": r.T_b,
        "S_gen_dot_W_K": r.S_gen_dot,
        "X_dest_dot_W": r.X_dest_dot,
        "veredicto": r.verdict,
        "notas": list(r.notes),
    }


def exchanger_to_dict(r: ExchangerResult) -> dict[str, Any]:
    streams = (r.inputs.stream_a, r.inputs.stream_b)
    return {
        "corrientes": [
            {
                "corriente": s.label or f"corriente {i + 1}",
                "sustancia": s.substance.label,
                "m_dot_kg_s": r.m_dots[i],
                "entrada": _state_dict(r.inlets[i]),
                "salida": _state_dict(r.outlets[i]),
                "Q_W": r.heat(i),
                "dS_W_K": r.entropy_change(i),
            }
            for i, s in enumerate(streams)
        ],
        "Q_dot_ambiente_W": r.inputs.Q_dot_W,
        "S_gen_dot_W_K": r.S_gen_dot,
        "X_dest_dot_W": r.X_dest_dot,
        "veredicto": r.verdict,
        "notas": list(r.notes),
    }
