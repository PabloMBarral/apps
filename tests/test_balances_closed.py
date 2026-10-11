"""Tests de core.balances.closed: sistemas cerrados y equilibrio térmico (Fase 10.1)."""

from __future__ import annotations

import json
import math
import pickle

import pytest

from core.balances import closed as cl
from core.balances import substance as sb

C = 273.15
BTU = 1055.05585262
WATER = sb.fluid_substance("Water")
R134A = sb.fluid_substance("R134a")
AIR = sb.ideal_gas_substance("air", "variable")
AIR_C = sb.ideal_gas_substance("air", "constant")
WATER_INC = sb.incompressible_substance("water")


def _solve(name: str) -> cl.ClosedResult:
    return cl.solve_closed(cl.CLOSED_EXAMPLES[name].inputs)


# ---------------------------------------------------------------------
# Ejemplos del libro
# ---------------------------------------------------------------------


def test_cengel_constant_pressure_boundary_work() -> None:
    """Çengel §4-1: 10 lbm de vapor a 60 psia de 320 a 400 °F: W_b = 96,4 Btu."""
    r = _solve("Vapor de agua calentado a presión constante (Çengel §4-1)")
    assert r.W_b / BTU == pytest.approx(96.4, abs=0.1)
    assert r.Q == pytest.approx(r.mass * (r.state2.h - r.state1.h), rel=1e-9)


def test_cengel_electric_heating_at_constant_pressure() -> None:
    """Çengel §4-2: W_ent + Q = m·Δh lleva el vapor a 200 °C (CoolProp: 199,5 °C)."""
    r = _solve("Calentamiento eléctrico a presión constante (Çengel §4-2)")
    assert r.state2.T - C == pytest.approx(200.0, abs=1.0)
    assert r.Q + r.W_in == pytest.approx(r.mass * (r.state2.h - r.state1.h), rel=1e-9)
    assert r.W_in == 7.2e3 and r.S_gen > 0.0


def test_cengel_unrestrained_expansion() -> None:
    """Çengel §4-2: sin trabajo, p₂ = p_sat(25 °C) y Q = ΔU (0,25 kJ con u₁ ≈ u_f)."""
    r = _solve("Expansión libre del agua (Çengel §4-2)")
    assert r.W_b == 0.0 and not r.quasi_static
    assert r.state2.p / 1e3 == pytest.approx(3.169, abs=0.002)
    assert r.Q / 1e3 == pytest.approx(0.34, abs=0.01)
    assert any("líquido comprimido" in n for n in r.notes)


def test_cengel_r134a_rigid_tank_entropy() -> None:
    """Çengel §7-3: 5 kg de R-134a de 140 a 100 kPa a volumen constante: ΔS = −1,173 kJ/K."""
    r = _solve("R-134a que se enfría en un tanque rígido (Çengel §7-3)")
    assert r.dS / 1e3 == pytest.approx(-1.173, abs=0.003)
    assert r.state2.is_two_phase and r.W_b == 0.0


def test_cengel_isothermal_compression() -> None:
    """Çengel §4-1: 0,4 m³ de aire a 100 kPa y 80 °C hasta 0,1 m³: W_b = −55,5 kJ."""
    r = _solve("Compresión isotérmica de aire (Çengel §4-1)")
    assert r.W_b / 1e3 == pytest.approx(-55.5, abs=0.1)
    assert r.W_b == pytest.approx(1e5 * 0.4 * math.log(0.1 / 0.4), rel=1e-9)
    assert r.dU == pytest.approx(0.0, abs=1e-9) and r.S_gen == 0.0


def test_cengel_isothermal_heat_rejection() -> None:
    """Çengel §7-13: una mezcla a 100 °C cede 600 kJ al aire a 25 °C."""
    r = _solve("Mezcla saturada que cede calor al ambiente (Çengel §7-13)")
    assert r.dS / 1e3 == pytest.approx(-1.61, abs=0.005)
    assert r.S_gen / 1e3 == pytest.approx(0.40, abs=0.005)
    assert r.dS == pytest.approx(r.Q / r.state1.T, rel=1e-6)


def test_cengel_car_engine_isentropic_compression() -> None:
    """Çengel §7-9: aire a 22 °C y 95 kPa con V₁/V₂ = 8: T₂ = 662,7 K (tabla A-17)."""
    r = _solve("Compresión isoentrópica del aire en un motor (Çengel §7-9)")
    assert r.state2.T == pytest.approx(662.7, abs=0.5)
    assert r.state1.v / r.state2.v == pytest.approx(8.0, rel=1e-12)
    constant = cl.solve_closed(
        cl.ClosedInputs(
            AIR_C,
            "TP",
            22.0 + C,
            95e3,
            mass_kg=1.0,
            process="isentropic",
            end="v",
            end_value=r.state1.v / 8.0,
        )
    )
    assert constant.state2.T == pytest.approx(295.15 * 8.0**0.4, rel=2e-3)  # k = 1,4


def test_cengel_iron_block_in_water() -> None:
    """Çengel §4-5: 50 kg de hierro a 80 °C en 0,5 m³ de agua a 25 °C: 25,6 °C."""
    ex = cl.EQUILIBRIUM_EXAMPLES["Bloque de hierro en agua (Çengel §4-5)"]
    r = cl.solve_equilibrium(ex.inputs)
    assert r.T_final - C == pytest.approx(25.6, abs=0.05)
    assert sum(c.Q for c in r.changes) == pytest.approx(0.0, abs=1e-6)
    assert r.S_gen > 0.0 and r.reservoir_dS is None


def test_cengel_iron_block_in_a_lake() -> None:
    """Çengel §7-13: 50 kg de hierro a 500 K en un lago a 285 K: S_gen = 4,32 kJ/K."""
    ex = cl.EQUILIBRIUM_EXAMPLES["Bloque de hierro en un lago (Çengel §7-13)"]
    r = cl.solve_equilibrium(ex.inputs)
    assert r.T_final == 285.0
    assert r.changes[0].dS / 1e3 == pytest.approx(-12.65, abs=0.01)
    assert r.reservoir_dS is not None and r.reservoir_dS / 1e3 == pytest.approx(16.97, abs=0.01)
    assert r.S_gen / 1e3 == pytest.approx(4.32, abs=0.01)
    assert r.X_dest == pytest.approx(285.0 * r.S_gen)


# ---------------------------------------------------------------------
# Todos los ejemplos: los balances cierran
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(cl.CLOSED_EXAMPLES))
def test_examples_close_both_balances(name: str) -> None:
    r = _solve(name)
    scale = abs(r.Q) + abs(r.W_b) + abs(r.W_in) + abs(r.dU) + 1.0
    assert abs(r.energy_residual) <= 1e-9 * scale
    assert r.S_gen == pytest.approx(r.dS - r.Q / r.T_b, abs=1e-9 * (abs(r.dS) + 1.0))
    assert r.S_gen >= 0.0 and r.violation is None and r.verdict != "imposible"
    assert r.X_dest == pytest.approx(r.inputs.T0_K * r.S_gen)
    assert r.path[0].T == pytest.approx(r.state1.T, rel=1e-6)
    assert r.path[-1].T == pytest.approx(r.state2.T, rel=1e-6)
    data = cl.closed_to_dict(r)
    json.dumps(data)
    assert data["S_gen_J_K"] == r.S_gen and data["proceso"] == cl.PROCESSES[r.inputs.process]
    assert cl.closed_to_dict(pickle.loads(pickle.dumps(r))) == data  # los NaN no son iguales


@pytest.mark.parametrize("name", list(cl.EQUILIBRIUM_EXAMPLES))
def test_equilibrium_examples(name: str) -> None:
    r = cl.solve_equilibrium(cl.EQUILIBRIUM_EXAMPLES[name].inputs)
    assert r.S_gen > 0.0
    json.dumps(cl.equilibrium_to_dict(r))
    assert pickle.loads(pickle.dumps(r)) == r
    assert r.notes


# ---------------------------------------------------------------------
# Procesos
# ---------------------------------------------------------------------


def test_paths_follow_each_process() -> None:
    """Cada camino cumple su condición: v, p, T o s constantes, o p·vⁿ = cte."""
    cases = {
        "v_const": (lambda s: s.v, cl.ClosedInputs(WATER, "PX", 1e5, 0.5, mass_kg=1.0,
                                                   process="v_const", end="x", end_value=1.0)),
        "p_const": (lambda s: s.p, cl.ClosedInputs(WATER, "PX", 1e5, 0.2, mass_kg=1.0,
                                                   process="p_const", end="T", end_value=500.0)),
        "T_const": (lambda s: s.T, cl.ClosedInputs(WATER, "TX", 400.0, 0.5, mass_kg=1.0,
                                                   process="T_const", end="p", end_value=1e5)),
        "isentropic": (lambda s: s.s, cl.ClosedInputs(AIR, "TP", 300.0, 1e5, mass_kg=1.0,
                                                      process="isentropic", end="p",
                                                      end_value=1e6)),
        "polytropic": (lambda s: s.p * s.v**1.2, cl.ClosedInputs(
            AIR, "TP", 300.0, 1e5, mass_kg=1.0, process="polytropic", end="p",
            end_value=1e6, n=1.2)),
    }  # fmt: skip
    for proc, (fn, inputs) in cases.items():
        r = cl.solve_closed(inputs)
        assert len(r.path) == 41, proc
        values = [fn(s) for s in r.path]
        assert max(values) == pytest.approx(min(values), rel=1e-6), proc


def test_constant_volume_to_saturated_vapor() -> None:
    """Un tanque rígido con mezcla que se calienta hasta vapor saturado (v₂ = v_g)."""
    r = cl.solve_closed(
        cl.ClosedInputs(WATER, "PX", 1e5, 0.5, mass_kg=1.0, process="v_const", end="x",
                        end_value=1.0, T_b_K=600.0)
    )  # fmt: skip
    assert r.state2.x == pytest.approx(1.0) and r.state2.v == pytest.approx(r.state1.v)
    assert r.W_b == 0.0 and r.Q == pytest.approx(r.dU)


def test_isotherm_inside_the_dome_is_m_hfg_dx() -> None:
    sat_l = sb.state(WATER, "TX", 150.0 + C, 0.0)
    sat_v = sb.state(WATER, "TX", 150.0 + C, 1.0)
    r = cl.solve_closed(
        cl.ClosedInputs(WATER, "TX", 150.0 + C, 0.2, mass_kg=2.0, process="T_const", end="x",
                        end_value=0.9)
    )  # fmt: skip
    assert r.Q == pytest.approx(2.0 * (sat_v.h - sat_l.h) * 0.7, rel=1e-6)
    assert r.S_gen == 0.0
    assert any("isobara" in n for n in r.notes)


def test_constant_temperature_with_heat_given() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 400.0, 5e5, mass_kg=1.0, process="T_const", end="Q",
                        end_value=50e3)
    )  # fmt: skip
    assert r.Q == pytest.approx(50e3, rel=1e-9) and r.W_b == pytest.approx(50e3, rel=1e-9)
    assert r.state2.p == pytest.approx(5e5 * math.exp(-50e3 / (400.0 * AIR.gas.R)), rel=1e-9)


@pytest.mark.parametrize("sub", [AIR, AIR_C, R134A])
def test_polytropic_work_formula(sub: sb.Substance) -> None:
    T1 = 300.0 if sub.kind == "ideal_gas" else 10.0 + C
    r = cl.solve_closed(
        cl.ClosedInputs(sub, "TP", T1, 1e5, mass_kg=1.0, process="polytropic", end="p",
                        end_value=4e5, n=1.25, T_b_K=T1 - 20.0)
    )  # fmt: skip
    s1, s2 = r.state1, r.state2
    assert s2.p * s2.v**1.25 == pytest.approx(s1.p * s1.v**1.25, rel=1e-9)
    assert r.W_b == pytest.approx((s2.p * s2.v - s1.p * s1.v) / (1.0 - 1.25), rel=1e-12)


def test_polytropic_with_n_one_is_the_isotherm_for_an_ideal_gas() -> None:
    iso = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 350.0, 1e5, mass_kg=1.0, process="T_const", end="p",
                        end_value=3e5)
    )  # fmt: skip
    poly = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 350.0, 1e5, mass_kg=1.0, process="polytropic", end="p",
                        end_value=3e5, n=1.0, T_b_K=350.0)
    )  # fmt: skip
    assert poly.W_b == pytest.approx(iso.W_b, rel=1e-9)
    assert poly.state2.T == pytest.approx(350.0, rel=1e-9)


@pytest.mark.parametrize("end", ["T", "v", "V"])
def test_polytropic_with_other_end_data(end: str) -> None:
    ref = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 300.0, 1e5, mass_kg=0.5, process="polytropic", end="p",
                        end_value=6e5, n=1.3)
    )  # fmt: skip
    value = {"T": ref.state2.T, "v": ref.state2.v, "V": ref.V2}[end]
    r = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 300.0, 1e5, mass_kg=0.5, process="polytropic",
                        end=end, end_value=value, n=1.3)  # type: ignore[arg-type]
    )  # fmt: skip
    assert r.state2.p == pytest.approx(6e5, rel=1e-7)


def test_polytropic_cooling_while_compressing() -> None:
    """Con n < k un gas que se comprime cede calor (Q < 0) aunque se caliente."""
    r = _solve("Compresión politrópica de aire (n = 1,3)")
    assert r.Q < 0.0 and r.dU > 0.0 and r.W_b < 0.0


def test_against_external_pressure_ideal_gas_closed_form() -> None:
    """Contra p_ext, adiabático: c_v·(T₂ − T₁) = −p₂·(R·T₂/p₂ − R·T₁/p₁)."""
    r = _solve("Aire que se expande contra la atmósfera (no cuasiestático)")
    g = AIR_C.gas
    cv = g.cp(298.15) - g.R
    T2 = 300.0 * (cv + g.R / 3.0) / (cv + g.R)
    assert r.state2.T == pytest.approx(T2, rel=1e-9)
    assert r.state2.p == pytest.approx(1e5)
    assert r.reversible_W is not None and r.reversible_state is not None
    assert r.reversible_W > r.W_b > 0.0
    assert r.reversible_state.T < r.state2.T
    assert r.S_gen > 0.0 and not r.quasi_static
    assert any("No es cuasiestático" in n for n in r.notes)


def test_against_external_pressure_with_final_temperature() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 300.0, 3e5, mass_kg=1.0, process="p_ext", end="T",
                        end_value=300.0, p_ext_Pa=1e5)
    )  # fmt: skip
    assert r.W_b == pytest.approx(1e5 * (r.state2.v - r.state1.v), rel=1e-9)
    assert r.Q == pytest.approx(r.W_b, rel=1e-9)  # gas ideal a la misma T: ΔU = 0
    assert r.reversible_W is None


def test_general_with_heat_given() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(WATER, "TP", 200.0 + C, 1e6, mass_kg=1.0, process="general", end="Q",
                        end_value=-100e3, end_pair="TP", end_a=150.0 + C, end_b=5e5)
    )  # fmt: skip
    assert r.W_b == pytest.approx(-100e3 - r.dU, rel=1e-12)
    assert len(r.path) == 2


def test_incompressible_heating() -> None:
    """Vademecum §13: Q = m·c·ΔT y ΔS = m·c·ln(T₂/T₁), sin trabajo de frontera."""
    r = _solve("Agua caliente en un termo que se enfría")
    c = WATER_INC.material.c
    assert r.Q == pytest.approx(c * (60.0 - 90.0), rel=1e-12)
    assert r.dS == pytest.approx(c * math.log((60.0 + C) / (90.0 + C)), rel=1e-12)
    by_heat = cl.solve_closed(
        cl.ClosedInputs(WATER_INC, "TP", 90.0 + C, 101325.0, mass_kg=1.0, process="v_const",
                        end="Q", end_value=r.Q, T0_K=20.0 + C)
    )  # fmt: skip
    assert by_heat.state2.T == pytest.approx(60.0 + C, rel=1e-12)


def test_paddle_work_in_a_rigid_tank() -> None:
    """Una paleta en un tanque rígido aislado: W_ent = ΔU y S_gen = ΔS > 0."""
    r = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 300.0, 1e5, mass_kg=1.0, process="v_const", end="Q",
                        end_value=0.0, W_in_J=20e3)
    )  # fmt: skip
    assert r.dU == pytest.approx(20e3, rel=1e-9) and r.S_gen == pytest.approx(r.dS)
    assert r.verdict == "irreversible"
    assert any("resistencia o una paleta" in n for n in r.notes)


# ---------------------------------------------------------------------
# Segundo principio: el veredicto
# ---------------------------------------------------------------------


def test_heat_from_a_colder_source_is_impossible() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 300.0, 1e5, mass_kg=1.0, process="p_const", end="T",
                        end_value=500.0, T0_K=298.15)
    )  # fmt: skip
    assert r.verdict == "imposible" and r.S_gen < 0.0
    assert r.violation is not None and "Subí la temperatura de la fuente" in r.violation
    assert abs(r.energy_residual) < 1e-6  # el primer principio cierra igual


def test_heat_to_a_hotter_medium_is_impossible() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(WATER_INC, "TP", 60.0 + C, 1e5, mass_kg=1.0, process="v_const",
                        end="T", end_value=30.0 + C, T_b_K=50.0 + C)
    )  # fmt: skip
    assert r.verdict == "imposible"
    assert r.violation is not None and "Bajá la temperatura del medio" in r.violation


def test_adiabatic_entropy_decrease_is_impossible() -> None:
    r = cl.solve_closed(
        cl.ClosedInputs(AIR, "TP", 400.0, 1e5, mass_kg=1.0, process="general", end="Q",
                        end_value=0.0, end_pair="TP", end_a=400.0, end_b=2e5)
    )  # fmt: skip
    assert r.verdict == "imposible"
    assert r.violation is not None and "Sin calor" in r.violation


def test_verdicts() -> None:
    assert cl.verdict(0.0) == "reversible"
    assert cl.verdict(1e-3) == "irreversible"
    assert cl.verdict(-1e-3) == "imposible"
    assert cl.snap_sgen(1e-15, 1.0) == 0.0 and cl.snap_sgen(1e-6, 1.0) == 1e-6


# ---------------------------------------------------------------------
# Opciones por sustancia y mensajes al alumno
# ---------------------------------------------------------------------


def test_allowed_processes_and_ends() -> None:
    assert cl.allowed_processes(WATER_INC) == ("v_const",)
    assert cl.allowed_processes(AIR) == tuple(cl.PROCESSES)
    assert "x" in cl.allowed_ends("v_const", WATER)
    assert "x" not in cl.allowed_ends("v_const", AIR)
    assert cl.allowed_ends("polytropic", WATER_INC) == ("T", "Q")


def _inputs(**kw: object) -> cl.ClosedInputs:
    base: dict[str, object] = {
        "substance": AIR,
        "pair1": "TP",
        "a1": 300.0,
        "b1": 1e5,
        "mass_kg": 1.0,
        "process": "p_const",
        "end": "T",
        "end_value": 400.0,
    }
    base.update(kw)
    return cl.ClosedInputs(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"volume_m3": 1.0}, "uno de los dos"),
        ({"mass_kg": None}, "uno de los dos"),
        ({"mass_kg": -1.0}, "masa tiene que ser positiva"),
        ({"end": "Q", "process": "polytropic"}, "el dato final es uno de"),
        ({"substance": WATER_INC, "process": "T_const", "end": "p"}, "no cambia"),
        ({"substance": WATER_INC, "process": "v_const", "end": "p"}, "temperatura T₂ o el calor"),
        ({"process": "isentropic", "end": "p", "end_value": 2e5, "W_in_J": 1e3}, "irreversible"),
        ({"W_in_J": math.nan}, "tiene que ser un número"),
        ({"process": "general", "end": "Q", "end_value": 0.0}, "dos propiedades del estado 2"),
        ({"end_value": math.nan}, "Falta la temperatura final"),
        ({"process": "p_ext", "end": "Q", "end_value": 0.0}, "p_ext tiene que ser positiva"),
        ({"process": "polytropic", "end": "p", "end_value": 2e5, "n": 0.0}, "n de la polit"),
        ({"T_b_K": -5.0}, "fuente T_b tiene que ser positiva"),
        ({"T0_K": 500.0}, "ambiente T₀"),
        ({"process": "T_const", "end": "V", "end_value": -1.0}, "volumen final"),
        ({"process": "polytropic", "end": "T", "n": 1.0}, "n = 1 un gas ideal"),
        ({"process": "polytropic", "end": "T", "end_value": 1e4, "n": 1.3}, "no llega"),
        ({"process": "v_const", "end": "x", "end_value": 0.5}, "un gas ideal no tiene"),
    ],
)
def test_student_errors(kw: dict[str, object], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        cl.solve_closed(_inputs(**kw))


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"body_a": cl.Body(AIR, 1.0, 300.0)}, "incompresible"),
        ({"body_b": None}, "uno de los dos"),
        ({"T_reservoir_K": 300.0}, "uno de los dos"),
        ({"body_b": cl.Body(WATER_INC, 0.0, 300.0, "agua")}, r"masa \(agua\)"),
        ({"body_b": cl.Body(WATER_INC, 1.0, -3.0)}, "temperatura tiene que ser positiva"),
        ({"T0_K": 100.0}, "ambiente T₀"),
    ],
)
def test_equilibrium_errors(kw: dict[str, object], words: str) -> None:
    base: dict[str, object] = {
        "body_a": cl.Body(sb.incompressible_substance("iron"), 1.0, 400.0),
        "body_b": cl.Body(WATER_INC, 1.0, 300.0),
    }
    base.update(kw)
    with pytest.raises(ValueError, match=words):
        cl.solve_equilibrium(cl.EquilibriumInputs(**base))  # type: ignore[arg-type]


def test_equilibrium_with_a_reservoir_needs_a_positive_temperature() -> None:
    body = cl.Body(sb.incompressible_substance("iron"), 1.0, 400.0)
    with pytest.raises(ValueError, match="reservorio tiene que ser positiva"):
        cl.solve_equilibrium(cl.EquilibriumInputs(body, T_reservoir_K=-1.0))
