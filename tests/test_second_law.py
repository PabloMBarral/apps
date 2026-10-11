"""Tests del segundo principio (Fase 10.2): máquinas, Clausius, Carnot y el calor.

Contra Çengel y Boles (§6-3, §6-4, §6-10, §6-11, §7-2 y §7-13) y contra las identidades:
W = Q_H − Q_C, S_gen = −ΣQ/T, el trabajo perdido = T·S_gen, η = 1 − T_C/T_H con
cualquier fluido y X_Q,entra − X_Q,sale = T₀·S_gen.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import replace

import pytest

from core.balances import second_law as sl
from core.balances.substance import fluid_substance, ideal_gas_substance, incompressible_substance

C = 273.15
KJ_MIN = 1e3 / 60.0
KJ_H = 1e3 / 3600.0


def _machine(name: str) -> sl.MachineResult:
    return sl.solve_machine(sl.MACHINE_EXAMPLES[name].inputs)


# ---------------------------------------------------------------------
# Máquinas: contra el libro
# ---------------------------------------------------------------------


def test_power_plant_of_cengel() -> None:
    r = _machine("Central térmica de 30 MW (Çengel §6-3)")
    assert r.W == pytest.approx(30e6)
    assert r.performance == pytest.approx(0.375)
    assert r.verdict == "irreversible"


def test_kitchen_refrigerator_of_cengel() -> None:
    r = _machine("Refrigerador de una cocina (Çengel §6-4)")
    assert r.performance == pytest.approx(3.0)
    assert r.Q_H == pytest.approx(480.0 * KJ_MIN)


def test_house_heat_pump_of_cengel() -> None:
    r = _machine("Bomba de calor de una casa (Çengel §6-4)")
    assert r.W == pytest.approx(32_000.0 * KJ_H)
    assert r.W / 1e3 == pytest.approx(8.9, abs=0.02)  # el libro redondea 8,89 kW
    assert r.Q_C == pytest.approx(48_000.0 * KJ_H)


def test_carnot_engine_of_cengel() -> None:
    r = _machine("Máquina de Carnot (Çengel §6-10)")
    assert r.performance == pytest.approx(0.672, abs=5e-4)
    assert r.Q_C / 1e3 == pytest.approx(164.0, abs=0.5)
    assert r.S_gen == 0.0 and r.verdict == "reversible"
    assert r.eta_ex == pytest.approx(1.0)


def test_questionable_refrigerator_of_cengel() -> None:
    """COP_R = 13,5 con 35 °F y 75 °F: el reversible da 12,4 → imposible."""
    r = _machine("Un refrigerador demasiado bueno (Çengel §6-11)")
    assert r.performance_rev == pytest.approx(12.4, abs=0.05)
    assert r.verdict == "imposible" and r.S_gen < 0.0
    assert r.violation is not None and "COP_R,rev" in r.violation


def test_carnot_heat_pump_of_cengel() -> None:
    r = _machine("Bomba de calor de Carnot (Çengel §6-11)")
    assert r.performance == pytest.approx(11.3, abs=0.05)
    assert r.W / 1e3 == pytest.approx(3.32, abs=0.01)
    assert r.verdict == "reversible"


def test_inventor_engine_is_impossible() -> None:
    r = _machine("Un motor que promete demasiado")
    assert r.performance == pytest.approx(3.0 / 7.0)
    assert r.performance_rev == pytest.approx(0.42)
    assert r.verdict == "imposible"
    assert "η_Carnot" in (r.violation or "")


# ---------------------------------------------------------------------
# Máquinas: identidades
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sl.MACHINE_EXAMPLES))
def test_machine_identities(name: str) -> None:
    r = _machine(name)
    inp = r.inputs
    assert r.W == pytest.approx(r.Q_H - r.Q_C, rel=1e-12)
    assert r.clausius_sum == -r.S_gen
    # el trabajo perdido es T·S_gen con T la fuente que hace de ambiente
    T_env = inp.T_H_K if r.kind == "refrigerator" else inp.T_C_K
    assert r.lost_work == pytest.approx(T_env * r.S_gen, rel=1e-9, abs=1e-6)
    # con el ambiente por defecto, η_ex es el cociente con el reversible y X_dest el perdido
    assert r.eta_ex == pytest.approx(r.ratio, rel=1e-12)
    assert r.X_dest == pytest.approx(r.lost_work, rel=1e-9, abs=1e-6)
    assert r.X_supplied - r.X_recovered == pytest.approx(r.X_dest, rel=1e-9, abs=1e-6)
    assert (r.verdict == "imposible") == (r.performance > r.performance_rev * (1 + 1e-12))
    pickle.loads(pickle.dumps(r))
    data = sl.machine_to_dict(r)
    assert data["veredicto"] == r.verdict


@pytest.mark.parametrize("kind", list(sl.MACHINES))
@pytest.mark.parametrize("pair", ["QH_QC", "QH_W", "QC_W", "QH_p", "QC_p", "W_p"])
def test_every_pair_of_data_gives_the_same_machine(kind: sl.MachineKind, pair: str) -> None:
    """Con cualquier par de datos sale la misma máquina."""
    base = sl.solve_machine(sl.MachineInputs(kind, 600.0, 300.0, Q_H=1000.0, Q_C=700.0))
    values = {"QH": ("Q_H", base.Q_H), "QC": ("Q_C", base.Q_C), "W": ("W", base.W)}
    values["p"] = ("performance", base.performance)
    a, b = pair.split("_")
    kwargs = dict((values[a], values[b]))
    r = sl.solve_machine(sl.MachineInputs(kind, 600.0, 300.0, **kwargs))
    for attr in ("Q_H", "Q_C", "W", "performance", "S_gen"):
        assert getattr(r, attr) == pytest.approx(getattr(base, attr), rel=1e-12)


@pytest.mark.parametrize("kind", list(sl.MACHINES))
def test_reversible_machine_has_no_entropy_generation(kind: sl.MachineKind) -> None:
    r = sl.solve_machine(sl.MachineInputs(kind, 500.0, 280.0, W=1.0, reversible=True))
    assert r.S_gen == 0.0 and r.verdict == "reversible"
    assert r.performance == pytest.approx(sl.carnot_performance(kind, 500.0, 280.0))


def test_cop_relation_between_refrigerator_and_heat_pump() -> None:
    """COP_B = COP_R + 1 con las mismas fuentes (vademecum §9.3)."""
    for T_H, T_C in ((300.0, 250.0), (320.0, 270.0)):
        r = sl.carnot_performance("refrigerator", T_H, T_C)
        b = sl.carnot_performance("heat_pump", T_H, T_C)
        assert b == pytest.approx(r + 1.0)


def test_kelvin_planck_engine() -> None:
    r = sl.solve_machine(sl.MachineInputs("engine", 600.0, 300.0, Q_H=100.0, Q_C=0.0))
    assert r.performance == 1.0 and r.verdict == "imposible"
    assert "Kelvin–Planck" in (r.violation or "")


def test_electric_resistance_is_a_heat_pump_with_cop_one() -> None:
    r = sl.solve_machine(sl.MachineInputs("heat_pump", 293.15, 273.15, W=1000.0, performance=1.0))
    assert r.Q_C == 0.0 and r.verdict == "irreversible"
    assert any("resistencia eléctrica" in n for n in r.notes)


def test_other_ambient_temperature() -> None:
    """Con T₀ distinta de la fuente de afuera el calor rechazado lleva exergía (pérdida)."""
    base = sl.MachineInputs("engine", 800.0, 350.0, Q_H=1000.0, W=300.0)
    r = sl.solve_machine(replace(base, T0_K=300.0))
    assert r.X_dest == pytest.approx(300.0 * r.S_gen)
    lost_with_heat = r.Q_C * (1.0 - 300.0 / 350.0)
    assert r.X_supplied - r.X_recovered == pytest.approx(r.X_dest + lost_with_heat)
    assert any("no es una de las fuentes" in n for n in r.notes)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (sl.MachineInputs("engine", 300.0, 300.0, Q_H=1.0, W=0.5), "más caliente"),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_H=1.0), "Hacen falta dos datos"),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_H=1.0, W=0.5, Q_C=0.5), "diste 3"),
        (
            sl.MachineInputs("engine", 600.0, 300.0, Q_H=1.0, performance=0.3, reversible=True),
            "no lo des",
        ),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_H=1.0, performance=1.2), "primer principio"),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_H=1.0, W=2.0), "no puede ser mayor"),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_H=1.0, Q_C=2.0), "menor que Q_H"),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_H=-1.0, W=0.5), "mayor que cero"),
        (sl.MachineInputs("refrigerator", 300.0, 250.0, W=1.0, performance=0.0), "mayor que cero"),
        (sl.MachineInputs("heat_pump", 300.0, 250.0, W=1.0, performance=0.8), "al menos 1"),
        (sl.MachineInputs("heat_pump", 300.0, 250.0, Q_C=1.0, performance=1.0), "Q_C = 0"),
        (sl.MachineInputs("engine", 600.0, 300.0, Q_C=1.0, performance=1.0), "Q_C = 0"),
    ],
)
def test_machine_errors(inputs: sl.MachineInputs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sl.solve_machine(inputs)


# ---------------------------------------------------------------------
# Clausius
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sl.CLAUSIUS_EXAMPLES))
def test_clausius_identities(name: str) -> None:
    r = sl.solve_clausius(sl.CLAUSIUS_EXAMPLES[name].inputs)
    assert r.W == pytest.approx(sum(res.Q for res in r.inputs.reservoirs))
    assert r.X_heat == pytest.approx(r.W + r.X_dest, rel=1e-12, abs=1e-6)
    assert (r.verdict == "imposible") == (r.clausius_sum > 0.0)
    assert sl.clausius_to_dict(r)["veredicto"] == r.verdict


def test_two_sources_engine() -> None:
    r = sl.solve_clausius(sl.CLAUSIUS_EXAMPLES["Máquina con dos fuentes y un sumidero"].inputs)
    assert r.clausius_sum / 1e3 == pytest.approx(1.0 + 500.0 / 600.0 - 3.0)
    assert r.W / 1e3 == pytest.approx(600.0)
    assert r.efficiency == pytest.approx(0.4)
    assert r.X_heat / 1e3 == pytest.approx(950.0)


def test_carnot_cycle_gives_the_equality() -> None:
    r = sl.solve_clausius(sl.CLAUSIUS_EXAMPLES["Ciclo de Carnot (la igualdad)"].inputs)
    assert r.S_gen == 0.0 and r.verdict == "reversible"


def test_impossible_cycle() -> None:
    r = sl.solve_clausius(sl.CLAUSIUS_EXAMPLES["Un ciclo imposible"].inputs)
    assert r.clausius_sum / 1e3 == pytest.approx(1.0)
    assert r.verdict == "imposible" and "Clausius" in (r.violation or "")


def test_clausius_matches_the_machine() -> None:
    """Dos fuentes: Clausius y la máquina dan la misma S_gen."""
    m = _machine("Central térmica de 30 MW (Çengel §6-3)")
    c = sl.solve_clausius(
        sl.ClausiusInputs(
            (sl.Reservoir(m.inputs.T_H_K, m.Q_H), sl.Reservoir(m.inputs.T_C_K, -m.Q_C)),
            rates=True,
        )
    )
    assert c.S_gen == pytest.approx(m.S_gen, rel=1e-12)


@pytest.mark.parametrize(
    ("reservoirs", "message"),
    [
        ((sl.Reservoir(300.0, 1.0),), "al menos dos"),
        ((sl.Reservoir(300.0, 1.0), sl.Reservoir(-5.0, -1.0)), "positiva"),
        ((sl.Reservoir(300.0, 0.0), sl.Reservoir(400.0, 0.0)), "cero"),
    ],
)
def test_clausius_errors(reservoirs: tuple[sl.Reservoir, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sl.solve_clausius(sl.ClausiusInputs(reservoirs))


# ---------------------------------------------------------------------
# Ciclo de Carnot
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(sl.CARNOT_EXAMPLES))
def test_carnot_efficiency_does_not_depend_on_the_fluid(name: str) -> None:
    r = sl.solve_carnot(sl.CARNOT_EXAMPLES[name].inputs)
    assert r.eta == pytest.approx(r.eta_carnot, rel=1e-9)
    assert r.W_net == pytest.approx(r.Q_H - r.Q_C, rel=1e-9)
    # el primer principio en cada proceso
    for p in r.processes:
        assert p.Q - p.W == pytest.approx(p.dE, rel=1e-12, abs=1e-6)
    # el ciclo cierra: la suma de ΔU (o ΔH) es cero
    assert sum(p.dE for p in r.processes) == pytest.approx(0.0, abs=1e-6 * r.Q_H)
    s1, s2, s3, s4 = r.states
    assert s3.s == pytest.approx(s2.s, rel=1e-9) and s4.s == pytest.approx(s1.s, rel=1e-9)
    first, last = r.path[0], r.path[-1]
    assert first.T == pytest.approx(s1.T) and first.s == pytest.approx(s1.s, rel=1e-9)
    assert last.s == pytest.approx(s1.s, rel=1e-9)  # el camino vuelve al estado 1
    assert len(r.path) == 4 * (sl.PATH_POINTS - 1) + 1
    assert sl.carnot_to_dict(r)["eta"] == pytest.approx(r.eta)


def test_water_in_the_dome() -> None:
    """De líquido a vapor saturado a 250 °C: Q_H = h_fg = T·s_fg."""
    r = sl.solve_carnot(sl.CARNOT_EXAMPLES["Vapor de agua en la campana"].inputs)
    s1, s2 = r.states[:2]
    assert r.Q_H == pytest.approx(s2.h - s1.h, rel=1e-9)  # a p constante, q = Δh
    assert r.Q_H / 1e3 == pytest.approx(1715.3, abs=0.5)  # h_fg a 250 °C (tabla A-4: 1715,3)
    assert r.eta == pytest.approx(1.0 - 313.15 / 523.15)


def test_closed_and_steady_flow_share_the_net_work() -> None:
    a = sl.solve_carnot(sl.CARNOT_EXAMPLES["Vapor de agua en la campana"].inputs)
    b = sl.solve_carnot(sl.CARNOT_EXAMPLES["Vapor de agua en cuatro equipos"].inputs)
    assert a.W_net == pytest.approx(b.W_net, rel=1e-12)
    assert b.processes[0].W == pytest.approx(0.0, abs=1e-6)  # la caldera no hace trabajo
    assert a.processes[0].W > 0.0  # el pistón sí: el vapor se expande
    assert any("Rankine" in n for n in b.notes)


@pytest.mark.parametrize("model", ["constant", "variable"])
def test_ideal_gas_carnot(model: str) -> None:
    air = ideal_gas_substance("air", model)  # type: ignore[arg-type]
    r = sl.solve_carnot(sl.CarnotInputs(air, 1000.0, 300.0, "p", 50e5, "p", 25e5))
    R = air.gas.R
    # isoterma: q = R·T·ln(p₁/p₂)
    assert r.Q_H == pytest.approx(1000.0 * R * math.log(2.0), rel=1e-9)
    assert r.processes[0].dE == pytest.approx(0.0, abs=1e-6)  # u depende solo de T
    assert r.eta == pytest.approx(0.7, rel=1e-9)
    if model == "constant":  # isoentrópica: p₂/p₃ = (T_H/T_C)^(k/(k−1))
        k = air.gas.cp(298.15) / (air.gas.cp(298.15) - R)
        ratio = r.states[1].p / r.states[2].p
        assert ratio == pytest.approx((1000.0 / 300.0) ** (k / (k - 1.0)), rel=1e-6)


def test_carnot_mass_scales_everything() -> None:
    base = sl.CARNOT_EXAMPLES["R-134a en la campana"].inputs
    a = sl.solve_carnot(base)
    b = sl.solve_carnot(replace(base, mass_kg=2.5))
    assert b.W_net == pytest.approx(2.5 * a.W_net, rel=1e-12)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            sl.CarnotInputs(incompressible_substance("water"), 400.0, 300.0, "p", 2e5, "p", 1e5),
            "incompresible",
        ),
        (
            sl.CarnotInputs(fluid_substance("Water"), 500.0, 600.0, "x", 0.0, "x", 1.0),
            "más caliente",
        ),
        (
            sl.CarnotInputs(fluid_substance("Water"), 500.0, 300.0, "x", 1.0, "x", 0.0),
            "s₂ tiene que ser",
        ),
        (
            sl.CarnotInputs(ideal_gas_substance("air"), 1000.0, 300.0, "p", 25e5, "p", 50e5),
            "presión del estado 2",
        ),
        (
            sl.CarnotInputs(ideal_gas_substance("air"), 1000.0, 300.0, "x", 0.0, "x", 1.0),
            "gas ideal no tiene",
        ),
        (
            sl.CarnotInputs(
                fluid_substance("Water"), 500.0, 300.0, "x", 0.0, "x", 1.0, mass_kg=0.0
            ),
            "masa",
        ),
    ],
)
def test_carnot_errors(inputs: sl.CarnotInputs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sl.solve_carnot(inputs)


# ---------------------------------------------------------------------
# La entropía del calor
# ---------------------------------------------------------------------


def test_heat_between_reservoirs_of_cengel() -> None:
    a = sl.solve_heat_path(
        sl.HEAT_PATH_EXAMPLES["Calor entre dos fuentes: sumidero a 500 K (Çengel §7-2)"].inputs
    )
    b = sl.solve_heat_path(
        sl.HEAT_PATH_EXAMPLES["Calor entre dos fuentes: sumidero a 750 K (Çengel §7-2)"].inputs
    )
    assert a.S_gen / 1e3 == pytest.approx(1.5)
    assert b.S_gen / 1e3 == pytest.approx(0.167, abs=5e-4)


def test_brick_wall_of_cengel() -> None:
    r = sl.solve_heat_path(sl.HEAT_PATH_EXAMPLES["Pared de ladrillo (Çengel §7-13)"].inputs)
    wall = r.zones[1]
    assert wall.name == "la pared"
    assert wall.S_gen == pytest.approx(0.191, abs=1e-3)  # W/K
    assert r.S_gen == pytest.approx(0.341, abs=1e-3)
    assert r.exergies[-1] == pytest.approx(0.0, abs=1e-12)  # llega al ambiente


@pytest.mark.parametrize("name", list(sl.HEAT_PATH_EXAMPLES))
def test_heat_path_identities(name: str) -> None:
    r = sl.solve_heat_path(sl.HEAT_PATH_EXAMPLES[name].inputs)
    X = r.exergies
    assert X[0] - X[-1] == pytest.approx(r.X_dest, rel=1e-12)
    assert r.S_gen == pytest.approx(sum(z.S_gen for z in r.zones), rel=1e-12)
    assert r.verdict == "irreversible"
    assert sl.heat_path_to_dict(r)["veredicto"] == "irreversible"


def test_heat_from_cold_to_hot_is_impossible() -> None:
    r = sl.solve_heat_path(
        sl.HeatPathInputs(1000.0, (300.0, 400.0), ("a", "b"), ("la transferencia",))
    )
    assert r.verdict == "imposible" and "Clausius" in (r.violation or "")


def test_heat_without_temperature_difference_is_reversible() -> None:
    r = sl.solve_heat_path(
        sl.HeatPathInputs(1000.0, (350.0, 350.0), ("a", "b"), ("la transferencia",))
    )
    assert r.S_gen == 0.0 and r.verdict == "reversible"


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (sl.HeatPathInputs(1.0, (300.0,), ("a",), ()), "al menos dos"),
        (sl.HeatPathInputs(1.0, (400.0, 300.0), ("a",), ("t",)), "nombre"),
        (sl.HeatPathInputs(-1.0, (400.0, 300.0), ("a", "b"), ("t",)), "invertí"),
        (sl.HeatPathInputs(1.0, (400.0, 0.0), ("a", "b"), ("t",)), "positiva"),
    ],
)
def test_heat_path_errors(inputs: sl.HeatPathInputs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sl.solve_heat_path(inputs)
