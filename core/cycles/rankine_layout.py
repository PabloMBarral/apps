"""Topología y numeración de estados del ciclo de Rankine — Fases 3.1b y 3.1c.

Describe la planta —bombas, calentadores de agua de alimentación, caldera,
recalentador, tramos de turbina, condensador y, en el ciclo real, las
cañerías con pérdidas y el recuperador del ORC— a partir de los datos, sin
calcular nada: no importa TESPy ni CoolProp. Sirve para numerar los estados
como Çengel & Boles (*Termodinámica*, cap. 10) **antes** de resolver (los
rótulos de la página) y como plano para la red de TESPy, el procedimiento y
el diagrama.

Calentadores de agua de alimentación (Cengel §10-6):

- **Abierto** (de contacto directo): la extracción se mezcla con el agua de
  alimentación y sale líquido saturado a la presión de extracción; después
  hace falta una bomba.
- **Cerrado**: el agua de alimentación pasa por los tubos y sale a
  T_sat(p_ext) − TTD (TTD = 0 en el calentador ideal); la extracción
  condensa y sale como líquido saturado (el drenaje). El drenaje va **en
  cascada hacia atrás** —por una válvula— al calentador de presión
  inmediatamente menor o al condensador, o se **bombea hacia adelante** a
  una cámara de mezcla en la línea de agua, justo después del calentador
  (en Cengel 10-6, el de mayor presión, antes de la caldera). Con un
  subenfriador de drenaje el drenaje sale más frío (DCA) y con un
  desrecalentador el agua puede salir por encima de T_sat (TTD < 0).

Numeración: primero la línea de agua de alimentación desde la salida del
condensador (con el recuperador, los drenajes bombeados, las mezclas y la
cañería hasta la caldera en su lugar), después el vapor desde la caldera
—la extracción comparte el número del estado del que sale— y al final los
drenajes en cascada. Así quedan los ejemplos 10-1 a 10-6 de Cengel (10-2,
el ciclo real, con sus cañerías: 1 a 6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

HeaterKind = Literal["open", "closed"]

ComponentKind = Literal[
    "pump",
    "boiler",
    "reheater",
    "turbine",
    "condenser",
    "open_heater",
    "closed_heater",
    "valve",
    "mixer",
    "pipe",
    "recuperator",
]

#: Rol de cada conexión de un componente lógico.
#: ``in``/``out``: entrada y salida simples (en el recuperador, el lado
#: caliente: el escape de la turbina); ``fw_in``/``fw_out``: agua de
#: alimentación (en el recuperador, el líquido que sale de la bomba);
#: ``bleed``: extracción de la turbina; ``drain_in``: drenaje que llega de
#: otro calentador; ``drain_out``: drenaje que sale.
PortRole = Literal["in", "out", "fw_in", "fw_out", "bleed", "drain_in", "drain_out"]

INLET_ROLES: frozenset[str] = frozenset({"in", "fw_in", "bleed", "drain_in"})

#: Máximo de calentadores (Cengel resuelve a mano hasta tres).
MAX_HEATERS = 3

# Dos presiones de evento (extracción y recalentamiento) a menos de esta
# tolerancia relativa son la misma: la extracción sale antes de recalentar.
_SAME_P_RTOL = 1.0e-4

_KIND_ES: dict[str, str] = {"open": "abierto", "closed": "cerrado"}
_ROMAN = ("I", "II", "III", "IV")


@dataclass(frozen=True)
class FeedwaterHeater:
    """Calentador de agua de alimentación (Cengel §10-6), en SI.

    ``p_Pa`` es la presión de extracción. Lo demás es solo para los
    cerrados:

    - ``ttd_K``: diferencia terminal; el agua de alimentación sale a
      T_sat(p_ext) − TTD (0 = calentador ideal). Negativa solo con
      desrecalentador.
    - ``dca_K``: con subenfriador de drenaje, el drenaje sale a
      T_entrada del agua + DCA (*drain cooler approach*) en vez de líquido
      saturado; ``None`` = sin subenfriador.
    - ``desuperheater``: zona de desrecalentamiento; la extracción
      sobrecalentada se enfría hasta vapor saturado calentando el agua que
      ya sale de la zona de condensación.
    - ``drain_forward``: el drenaje se bombea hacia adelante, a una cámara de
      mezcla en la línea de agua justo después del calentador.
    - ``dp_Pa``: caída de presión del agua en los tubos.
    """

    p_Pa: float
    kind: HeaterKind = "open"
    ttd_K: float = 0.0
    dca_K: float | None = None
    desuperheater: bool = False
    drain_forward: bool = False
    dp_Pa: float = 0.0


@dataclass(frozen=True)
class Port:
    """Conexión de un componente lógico con un estado del ciclo.

    ``state`` es el índice en la lista de estados (el número de estado
    menos 1) y ``fraction`` el caudal de esa corriente sobre el de la
    caldera, ṁ/ṁ_caldera (lo completa el cálculo).
    """

    role: PortRole
    state: int
    fraction: float = 1.0


@dataclass(frozen=True)
class CycleComponent:
    """Componente del ciclo tal como lo ve el alumno (no uno a uno con TESPy).

    ``casing`` agrupa los tramos de una misma turbina (de alta, de baja);
    ``heater`` es el índice del calentador asociado (0 = menor presión),
    también para su válvula de drenaje, su bomba de drenaje y la mezcla.
    """

    kind: ComponentKind
    label: str
    ports: tuple[Port, ...]
    casing: str = ""
    heater: int | None = None

    def port(self, role: PortRole) -> Port:
        """La primera conexión con ese rol."""
        return next(p for p in self.ports if p.role == role)

    def ports_with(self, role: PortRole) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role == role)

    @property
    def inlets(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role in INLET_ROLES)

    @property
    def outlets(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role not in INLET_ROLES)


@dataclass(frozen=True)
class PlantLayout:
    """Plano de la planta: estados numerados y componentes lógicos.

    Los índices son posiciones en ``labels`` (estado número = índice + 1).
    ``drain_forward`` dice si el calentador de mayor presión bombea su
    drenaje hacia adelante (como Cengel 10-6) y ``forward_drains`` lista
    todos los que lo hacen. ``boiler_in``/``boiler_out`` son la entrada y la
    salida de la caldera (con cañerías, distintas de la salida de la línea
    de agua y de la entrada a la turbina) y ``condenser_in`` lo que entra al
    condensador (con recuperador, su salida del lado caliente).
    """

    labels: tuple[str, ...]
    components: tuple[CycleComponent, ...]
    heaters: tuple[FeedwaterHeater, ...]
    heater_names: tuple[str, ...]
    drain_forward: bool
    turbine_inlet: int
    exhaust: int
    reheat_in: int | None
    reheat_out: int | None
    extraction_states: tuple[int, ...]
    forward_drains: tuple[int, ...]
    boiler_in: int
    boiler_out: int
    condenser_in: int
    boiler: str = "caldera"

    @property
    def n_states(self) -> int:
        return len(self.labels)

    def number(self, index: int) -> int:
        """Número de estado (como en el libro) de un índice."""
        return index + 1

    def labeled(self) -> tuple[str, ...]:
        """Etiquetas completas: ``"6 (extracción para el calentador abierto)"``."""
        return tuple(f"{i + 1} ({label})" for i, label in enumerate(self.labels))

    def of_kind(self, kind: ComponentKind) -> tuple[CycleComponent, ...]:
        return tuple(c for c in self.components if c.kind == kind)


def heater_names(heaters: Sequence[FeedwaterHeater]) -> tuple[str, ...]:
    """Nombre de cada calentador, de menor a mayor presión.

    Uno solo: «calentador abierto»; dos: «… de baja» y «… de alta»; tres:
    «… de baja», «… intermedio» y «… de alta».
    """
    n = len(heaters)
    if n == 1:
        return (f"calentador {_KIND_ES[heaters[0].kind]}",)
    positions = ("de baja", "de alta") if n == 2 else ("de baja", "intermedio", "de alta")
    return tuple(
        f"calentador {_KIND_ES[h.kind]} {pos}" for h, pos in zip(heaters, positions, strict=False)
    )


def _same_p(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_SAME_P_RTOL)


_MASCULINE = frozenset({"evaporador", "generador de vapor"})


def _of(name: str) -> str:
    """``"de la caldera"`` / ``"del evaporador"``."""
    return f"del {name}" if name in _MASCULINE else f"de la {name}"


def _to(name: str) -> str:
    """``"a la caldera"`` / ``"al evaporador"``."""
    return f"al {name}" if name in _MASCULINE else f"a la {name}"


def plant_layout(
    *,
    heaters: Sequence[FeedwaterHeater] = (),
    reheat_p_Pa: float | None = None,
    drain_forward: bool = False,
    feed_pipe: bool = False,
    steam_pipe: bool = False,
    recuperator: bool = False,
    boiler: str = "caldera",
) -> PlantLayout:
    """Arma el plano de la planta (Cengel §10-5 y §10-6).

    ``heaters`` va de menor a mayor presión (la validación de
    :mod:`core.cycles.rankine` lo exige antes de llegar acá).
    ``drain_forward`` bombea hacia adelante el drenaje del calentador de
    mayor presión, que tiene que ser cerrado (también se puede pedir para
    cada cerrado con :attr:`FeedwaterHeater.drain_forward`).
    ``feed_pipe`` y ``steam_pipe`` suman las cañerías con pérdidas de la línea
    de agua a la caldera y de la caldera a la turbina (Cengel 10-2);
    ``recuperator``, el recuperador del ORC entre el escape de la turbina y el
    líquido que sale de la bomba. ``boiler`` es el nombre del equipo donde
    entra el calor ("caldera"; en el ORC, "evaporador").

    Raises
    ------
    ValueError
        Si la configuración no se puede representar (más de tres
        calentadores, presiones fuera de orden o repetidas, drenaje hacia
        adelante desde un calentador abierto, o recuperador con
        calentadores).
    """
    heaters = tuple(heaters)
    n = len(heaters)
    if n > MAX_HEATERS:
        raise ValueError(f"Se admiten hasta {MAX_HEATERS} calentadores de agua de alimentación.")
    for low, high in zip(heaters, heaters[1:], strict=False):
        if not high.p_Pa > low.p_Pa or _same_p(low.p_Pa, high.p_Pa):
            raise ValueError(
                "Las presiones de extracción tienen que ir de menor a mayor y ser distintas "
                "(cada calentador a su propia presión)."
            )
    if drain_forward and n > 0 and heaters[-1].kind != "closed":
        raise ValueError(
            "Solo un calentador cerrado puede bombear su drenaje hacia adelante, y el de mayor "
            "presión es abierto."
        )
    names = heater_names(heaters)
    for name, h in zip(names, heaters, strict=True):
        if h.drain_forward and h.kind != "closed":
            raise ValueError(
                f"El {name} es abierto: solo un calentador cerrado puede bombear su drenaje "
                "hacia adelante."
            )
    if recuperator and n:
        raise ValueError(
            "El recuperador se arma en el ciclo sin calentadores de agua de alimentación."
        )
    top = n - 1
    forward = {i for i, h in enumerate(heaters) if h.drain_forward}
    if drain_forward and n > 0:
        forward.add(top)

    labels: list[str] = []

    def new_state(label: str) -> int:
        labels.append(label)
        return len(labels) - 1

    opens = [i for i, h in enumerate(heaters) if h.kind == "open"]
    n_stages = len(opens) + 1
    pump_names = ["bomba"] if n_stages == 1 else [f"bomba {_ROMAN[k]}" for k in range(n_stages)]

    def stage_of(i: int) -> int:
        return sum(1 for j in opens if j < i)

    def forward_suffix(i: int) -> str:
        """Con más de un drenaje bombeado, cada bomba y cada mezcla llevan el calentador."""
        return "" if len(forward) == 1 else f" del {names[i]}"

    # Lo que se arma al final, cuando ya están todos los números.
    pumps: list[CycleComponent] = []
    feed_order: list[tuple[str, int]] = []  # ("pump", k) / ("heater", i) / ("drain", i)
    closed_fw: dict[int, tuple[int, int]] = {}
    open_io: dict[int, tuple[int, int]] = {}
    forward_states: dict[int, tuple[int, int, int]] = {}  # drenaje, bomba, mezcla
    recuperator_cold: tuple[int, int] | None = None

    # 1. Línea de agua de alimentación, desde la salida del condensador.
    cond_out = new_state("salida del condensador" if n else "entrada a la bomba")
    current = cond_out
    for stage in range(n_stages):
        pump_out = new_state(f"salida de la {pump_names[stage]}")
        pumps.append(
            CycleComponent("pump", pump_names[stage], (Port("in", current), Port("out", pump_out)))
        )
        feed_order.append(("pump", stage))
        current = pump_out
        if recuperator and stage == 0:
            cold_out = new_state("salida del recuperador (líquido)")
            recuperator_cold = (current, cold_out)
            current = cold_out
        for i in (i for i, h in enumerate(heaters) if h.kind == "closed" and stage_of(i) == stage):
            fw_out = new_state(f"agua de alimentación a la salida del {names[i]}")
            closed_fw[i] = (current, fw_out)
            feed_order.append(("heater", i))
            current = fw_out
            if i in forward:
                drain = new_state(f"drenaje del {names[i]}")
                pumped = new_state(f"salida de la bomba del drenaje{forward_suffix(i)}")
                mixed = new_state(f"salida de la cámara de mezcla{forward_suffix(i)}")
                forward_states[i] = (drain, pumped, mixed)
                feed_order.append(("drain", i))
                current = mixed
        if stage < len(opens):
            i = opens[stage]
            out = new_state(f"salida del {names[i]}")
            open_io[i] = (current, out)
            feed_order.append(("heater", i))
            current = out
    feed_pipe_comp: CycleComponent | None = None
    if feed_pipe:
        pipe_out = new_state(f"entrada {_to(boiler)}")
        feed_pipe_comp = CycleComponent(
            "pipe", "cañería de alimentación", (Port("in", current), Port("out", pipe_out))
        )
        current = pipe_out
    boiler_in = current

    # 2. Camino del vapor: caldera, cañería, tramos de turbina y recalentador.
    reheat = reheat_p_Pa is not None
    inlet_label = "entrada a la turbina de alta" if reheat else "entrada a la turbina"
    steam_pipe_comp: CycleComponent | None = None
    if steam_pipe:
        boiler_out = new_state(f"salida {_of(boiler)}")
        turbine_in = new_state(inlet_label)
        steam_pipe_comp = CycleComponent(
            "pipe", "cañería de vapor", (Port("in", boiler_out), Port("out", turbine_in))
        )
    else:
        turbine_in = new_state(inlet_label)
        boiler_out = turbine_in
    events: list[tuple[float, list[int], bool]] = []
    for i, h in enumerate(heaters):
        for event in events:
            if _same_p(event[0], h.p_Pa):
                event[1].append(i)
                break
        else:
            events.append((h.p_Pa, [i], False))
    if reheat_p_Pa is not None:
        for k, event in enumerate(events):
            if _same_p(event[0], reheat_p_Pa):
                events[k] = (event[0], event[1], True)
                break
        else:
            events.append((reheat_p_Pa, [], True))
    events.sort(key=lambda e: -e[0])

    # Tramos de cada turbina (cada extracción corta uno), para el nombre «(tramo k)».
    if reheat_p_Pa is None:
        n_sections = {"turbina": len(events) + 1}
    else:
        above = sum(1 for p, _, rh in events if not rh and p > reheat_p_Pa)
        below = sum(1 for p, _, rh in events if not rh and p < reheat_p_Pa)
        n_sections = {"turbina de alta": above + 1, "turbina de baja": below + 1}

    def section_label(casing_name: str, k: int) -> str:
        return casing_name if n_sections[casing_name] == 1 else f"{casing_name} (tramo {k})"

    casing = "turbina de alta" if reheat else "turbina"
    steam_order: list[CycleComponent] = []
    bleed_state: dict[int, int] = {}
    reheat_in = reheat_out = None
    start = turbine_in
    section_k = 0
    for _p, extracted, is_reheat in events:
        section_k += 1
        if is_reheat:
            label = "salida de la turbina de alta"
            if extracted:
                label += f" y extracción para el {names[extracted[0]]}"
        else:
            label = f"extracción para el {names[extracted[0]]}"
        out = new_state(label)
        steam_order.append(
            CycleComponent(
                "turbine",
                section_label(casing, section_k),
                (Port("in", start), Port("out", out)),
                casing=casing,
            )
        )
        for i in extracted:
            bleed_state[i] = out
        if is_reheat:
            reheat_in = out
            reheat_out = new_state("entrada a la turbina de baja")
            steam_order.append(
                CycleComponent(
                    "reheater", "recalentador", (Port("in", out), Port("out", reheat_out))
                )
            )
            casing = "turbina de baja"
            section_k = 0
            start = reheat_out
        else:
            start = out
    exhaust = new_state("salida de la turbina de baja" if reheat else "salida de la turbina")
    steam_order.append(
        CycleComponent(
            "turbine",
            section_label(casing, section_k + 1),
            (Port("in", start), Port("out", exhaust)),
            casing=casing,
        )
    )
    condenser_in = exhaust
    recuperator_comp: CycleComponent | None = None
    if recuperator_cold is not None:
        hot_out = new_state("salida del recuperador (vapor)")
        recuperator_comp = CycleComponent(
            "recuperator",
            "recuperador",
            (
                Port("fw_in", recuperator_cold[0]),
                Port("fw_out", recuperator_cold[1]),
                Port("in", exhaust),
                Port("out", hot_out),
            ),
        )
        condenser_in = hot_out

    # 3. Drenajes en cascada, de mayor a menor presión.
    valves: list[CycleComponent] = []
    drains_into: dict[int | None, list[int]] = {}  # destino (None = condensador)
    back_drain: dict[int, int] = {}  # salida del drenaje de cada cerrado
    for i in reversed(range(n)):
        if heaters[i].kind != "closed" or i in forward:
            continue
        drain = new_state(f"drenaje del {names[i]}")
        after = new_state(f"drenaje del {names[i]}, tras la válvula")
        back_drain[i] = drain
        valves.append(
            CycleComponent(
                "valve",
                f"válvula del drenaje del {names[i]}",
                (Port("in", drain), Port("out", after)),
                heater=i,
            )
        )
        drains_into.setdefault(i - 1 if i > 0 else None, []).append(after)

    # Armado de los componentes con todas sus conexiones.
    feed: list[CycleComponent] = []
    for what, k in feed_order:
        if what == "pump":
            feed.append(pumps[k])
            continue
        i = k
        if what == "drain":
            drain, pumped, mixed = forward_states[i]
            fw_out = closed_fw[i][1]
            feed.append(
                CycleComponent(
                    "pump",
                    f"bomba del drenaje{forward_suffix(i)}",
                    (Port("in", drain), Port("out", pumped)),
                    heater=i,
                )
            )
            feed.append(
                CycleComponent(
                    "mixer",
                    f"cámara de mezcla{forward_suffix(i)}",
                    (Port("fw_in", fw_out), Port("drain_in", pumped), Port("out", mixed)),
                    heater=i,
                )
            )
            continue
        drains_in = tuple(Port("drain_in", s) for s in drains_into.get(i, []))
        if heaters[i].kind == "open":
            fw_in, out = open_io[i]
            ports = (
                Port("fw_in", fw_in),
                Port("bleed", bleed_state[i]),
                *drains_in,
                Port("out", out),
            )
            feed.append(CycleComponent("open_heater", names[i], ports, heater=i))
        else:
            fw_in, fw_out = closed_fw[i]
            drain_out = forward_states[i][0] if i in forward else back_drain[i]
            ports = (
                Port("fw_in", fw_in),
                Port("fw_out", fw_out),
                Port("bleed", bleed_state[i]),
                *drains_in,
                Port("drain_out", drain_out),
            )
            feed.append(CycleComponent("closed_heater", names[i], ports, heater=i))

    boiler_comp = CycleComponent("boiler", boiler, (Port("in", boiler_in), Port("out", boiler_out)))
    condenser = CycleComponent(
        "condenser",
        "condensador",
        (
            Port("in", condenser_in),
            *(Port("drain_in", s) for s in drains_into.get(None, [])),
            Port("out", cond_out),
        ),
    )
    components = (
        *feed,
        *((feed_pipe_comp,) if feed_pipe_comp else ()),
        boiler_comp,
        *((steam_pipe_comp,) if steam_pipe_comp else ()),
        *steam_order,
        *((recuperator_comp,) if recuperator_comp else ()),
        condenser,
        *valves,
    )
    return PlantLayout(
        labels=tuple(labels),
        components=components,
        heaters=heaters,
        heater_names=names,
        drain_forward=n > 0 and top in forward,
        turbine_inlet=turbine_in,
        exhaust=exhaust,
        reheat_in=reheat_in,
        reheat_out=reheat_out,
        extraction_states=tuple(bleed_state[i] for i in range(n)),
        forward_drains=tuple(sorted(forward)),
        boiler_in=boiler_in,
        boiler_out=boiler_out,
        condenser_in=condenser_in,
        boiler=boiler,
    )


#: Caudal de una conexión como combinación de fracciones de extracción:
#: ``{None: 1, 0: -1}`` es 1 − y₁ (``None`` es la unidad: el caudal de la caldera).
Flow = dict[int | None, int]


def _plus(a: Flow, b: Flow, sign: int = 1) -> Flow:
    out = dict(a)
    for key, coef in b.items():
        out[key] = out.get(key, 0) + sign * coef
    return {k: v for k, v in out.items() if v}


def port_flows(layout: PlantLayout) -> dict[tuple[int, int], Flow]:
    """Caudal (ṁ/ṁ_caldera) de cada conexión, en función de las fracciones y_k.

    Es el balance de masa que Cengel escribe sobre el esquema (1, y, 1 − y…):
    la extracción del calentador k es y_k; el vapor que sigue en la turbina,
    1 menos lo extraído antes; el drenaje de un cerrado, su extracción más los
    drenajes que recibe; por cada cámara de mezcla se reincorpora un drenaje
    bombeado. Las claves son ``(componente, conexión)`` con los índices de
    ``layout.components`` y de sus ``ports``.
    """
    comps = layout.components
    flows: dict[tuple[int, int], Flow] = {}
    one: Flow = {None: 1}

    # Vapor: caldera, cañerías, tramos de turbina y recalentador. Por las
    # cañerías pasa todo el caudal (la de alimentación está después de la
    # última bomba, cuando ya se juntó todo).
    current = dict(one)
    for ci, comp in enumerate(comps):
        if comp.kind in ("boiler", "pipe"):
            for pi in range(len(comp.ports)):
                flows[(ci, pi)] = dict(one)
        elif comp.kind in ("turbine", "reheater"):
            for pi in range(len(comp.ports)):
                flows[(ci, pi)] = dict(current)
            out = comp.port("out").state
            if comp.kind == "turbine" and out in layout.extraction_states:
                current = _plus(current, {layout.extraction_states.index(out): 1}, -1)
    exhaust = current

    # Drenajes de los cerrados, de mayor a menor presión.
    feeds: dict[int, int] = {}  # estado a la salida de cada válvula -> calentador
    for comp in comps:
        if comp.kind == "valve" and comp.heater is not None:
            feeds[comp.port("out").state] = comp.heater
    drain_out: dict[int, Flow] = {}
    closed = sorted((c for c in comps if c.kind == "closed_heater"), key=lambda c: -(c.heater or 0))
    for comp in closed:
        assert comp.heater is not None
        flow: Flow = {comp.heater: 1}
        for port in comp.ports_with("drain_in"):
            flow = _plus(flow, drain_out[feeds[port.state]])
        drain_out[comp.heater] = flow

    # Condensador.
    cond_out: Flow = dict(exhaust)
    for ci, comp in enumerate(comps):
        if comp.kind != "condenser":
            continue
        for pi, port in enumerate(comp.ports):
            if port.role == "in":
                flows[(ci, pi)] = dict(exhaust)
            elif port.role == "drain_in":
                flows[(ci, pi)] = drain_out[feeds[port.state]]
                cond_out = _plus(cond_out, flows[(ci, pi)])
        flows[(ci, [p.role for p in comp.ports].index("out"))] = cond_out

    # Recuperador (solo sin calentadores): el escape de un lado y, del otro, el
    # líquido que sale de la primera bomba.
    for ci, comp in enumerate(comps):
        if comp.kind == "recuperator":
            for pi, port in enumerate(comp.ports):
                flows[(ci, pi)] = dict(exhaust if port.role in ("in", "out") else cond_out)

    # Línea de agua de alimentación, desde el condensador hasta la caldera.
    line = dict(cond_out)
    for ci, comp in enumerate(comps):
        if comp.kind == "valve":
            assert comp.heater is not None
            for pi in range(len(comp.ports)):
                flows[(ci, pi)] = drain_out[comp.heater]
        elif comp.kind == "pump" and comp.heater is not None:  # bomba del drenaje
            for pi in range(len(comp.ports)):
                flows[(ci, pi)] = drain_out[comp.heater]
        elif comp.kind == "pump":
            for pi in range(len(comp.ports)):
                flows[(ci, pi)] = dict(line)
        elif comp.kind in ("open_heater", "closed_heater", "mixer"):
            for pi, port in enumerate(comp.ports):
                if port.role in ("fw_in", "fw_out"):
                    flows[(ci, pi)] = dict(line)
                elif port.role == "bleed":
                    assert comp.heater is not None
                    flows[(ci, pi)] = {comp.heater: 1}
                elif port.role == "drain_in":
                    source = feeds.get(port.state, comp.heater)
                    assert source is not None
                    flows[(ci, pi)] = drain_out[source]
                elif port.role == "drain_out":
                    assert comp.heater is not None
                    flows[(ci, pi)] = drain_out[comp.heater]
            if comp.kind != "closed_heater":
                out_flow = _plus(dict(line), {}, 1)
                for pi, port in enumerate(comp.ports):
                    if port.role in ("bleed", "drain_in"):
                        out_flow = _plus(out_flow, flows[(ci, pi)])
                flows[(ci, [p.role for p in comp.ports].index("out"))] = out_flow
                line = out_flow
    assert line == one, line  # a la caldera llega todo el caudal
    return flows
