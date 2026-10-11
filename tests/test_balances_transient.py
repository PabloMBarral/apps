"""Tests de core.balances.transient: llenado y vaciado de tanques (Fase 10.1)."""

from __future__ import annotations

import json
import math
import pickle

import pytest

from core.balances import substance as sb
from core.balances import transient as tr

C = 273.15
WATER = sb.fluid_substance("Water")
AIR_REAL = sb.fluid_substance("Air")
AIR = sb.ideal_gas_substance("air", "variable")
AIR_C = sb.ideal_gas_substance("air", "constant")


def _charge(name: str) -> tr.ChargingResult:
    return tr.solve_charging(tr.CHARGING_EXAMPLES[name].inputs)


def _discharge(name: str) -> tr.DischargingResult:
    return tr.solve_discharging(tr.DISCHARGING_EXAMPLES[name].inputs)


# ---------------------------------------------------------------------
# Ejemplos del libro
# ---------------------------------------------------------------------


def test_cengel_evacuated_tank_charged_with_steam() -> None:
    """Çengel §5-5: tanque vacío desde una línea a 1 MPa y 300 °C: T₂ = 456,1 °C."""
    r = _charge("Tanque vacío que se llena con vapor (Çengel §5-5)")
    assert r.state2.T - C == pytest.approx(456.1, abs=0.5)
    assert r.state2.u == pytest.approx(r.line.h, rel=1e-9)  # u₂ = h_línea
    assert r.m1 == 0.0 and r.state1 is None and r.S_gen > 0.0


def test_cengel_tank_charged_and_cooled() -> None:
    """Problema del cap. 5 de Çengel: 9,58 kg de aire entran y salen 339 kJ."""
    r = _charge("Tanque con aire que se llena y se enfría (Çengel §5-5)")
    assert r.m_in == pytest.approx(9.58, abs=0.01)
    assert r.Q / 1e3 == pytest.approx(-339.0, abs=1.0)


def test_adiabatic_discharge_follows_the_isentrope() -> None:
    """1 m³ de aire de 10 a 2 bar: T₂ = 188,8 K; la integral ∫h·dm cierra el balance."""
    r = _discharge("Tanque de aire que se vacía sin calor")
    assert r.state2.T == pytest.approx(188.8, abs=0.1)
    assert r.state2.s == pytest.approx(r.state1.s, rel=1e-9)
    assert abs(r.integration_residual) < 1e-9 * r.H_out
    assert r.Q == 0.0 and r.S_gen == 0.0
    assert r.Q_uniform / 1e3 == pytest.approx(-47.7, abs=0.1)  # flujo uniforme: no cierra


def test_isothermal_discharge_of_an_ideal_gas() -> None:
    """Gas ideal a T constante: Q = V·(p₁ − p₂) = 800 kJ."""
    r = _discharge("Tanque de aire que se vacía a temperatura constante")
    assert r.Q == pytest.approx(1.0 * (10e5 - 2e5), rel=1e-9)
    assert r.Q_uniform == pytest.approx(r.Q, rel=1e-9)
    assert r.S_gen == 0.0  # el calor entra a la temperatura del tanque


def test_discharge_through_the_valve_generates_entropy() -> None:
    """Con c_p constante, S_gen = ∫R·ln(p/p_sal)·dm con p = p₁·(m/m₁)^k."""
    inputs = tr.DischargingInputs(
        AIR_C, 1.0, "TP", 300.0, 10e5, p2_Pa=3e5, p_out_Pa=101325.0, T0_K=300.0
    )
    r = tr.solve_discharging(inputs)
    g = AIR_C.gas
    R = g.R
    k = g.cp(298.15) / (g.cp(298.15) - R)
    m1, m2 = r.m1, r.m2
    exact = R * ((m1 - m2) * math.log(10e5 / 101325.0) + k * (-m1 - m2 * math.log(m2 / m1) + m2))
    assert r.S_gen == pytest.approx(exact, rel=1e-6)
    assert r.state2.T == pytest.approx(300.0 * 0.3 ** ((k - 1.0) / k), rel=1e-9)


def test_evacuated_tank_with_constant_cp_ends_at_k_times_the_line() -> None:
    """Çengel §5-5: con c_p constante, u₂ = h_línea da T₂ = k·T_línea."""
    r = tr.solve_charging(tr.ChargingInputs(AIR_C, 0.5, "TP", 300.0, 5e5, p2_Pa=5e5))
    g = AIR_C.gas
    k = g.cp(298.15) / (g.cp(298.15) - g.R)
    assert r.state2.T == pytest.approx(k * 300.0, rel=1e-9)
    assert any("k·T_línea" in n for n in r.notes)


# ---------------------------------------------------------------------
# Todos los ejemplos
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(tr.CHARGING_EXAMPLES))
def test_charging_examples(name: str) -> None:
    r = _charge(name)
    assert abs(r.energy_residual) <= 1e-9 * (abs(r.U2) + abs(r.H_in) + 1.0)
    assert r.S_gen > 0.0 and r.violation is None
    masses = [m for m, _ in r.path]
    pressures = [s.p for _, s in r.path]
    assert masses == sorted(masses) and pressures == sorted(pressures)
    assert r.path[-1][1].p == pytest.approx(r.state2.p, rel=1e-6)
    data = tr.charging_to_dict(r)
    json.dumps(data)
    assert tr.charging_to_dict(pickle.loads(pickle.dumps(r))) == data


@pytest.mark.parametrize("name", list(tr.DISCHARGING_EXAMPLES))
def test_discharging_examples(name: str) -> None:
    r = _discharge(name)
    assert abs(r.energy_residual) <= 1e-9 * (abs(r.H_out) + 1.0)
    assert r.S_gen >= 0.0 and r.violation is None
    pressures = [s.p for _, s in r.path]
    assert pressures == sorted(pressures, reverse=True)
    assert r.path[0][1].p == pytest.approx(r.state1.p) and len(r.path) == 41
    assert r.path[-1][1].p == pytest.approx(r.state2.p, rel=1e-9)
    data = tr.discharging_to_dict(r)
    json.dumps(data)
    assert tr.discharging_to_dict(pickle.loads(pickle.dumps(r))) == data


def test_charging_with_heat_given_matches_the_final_temperature() -> None:
    by_t = _charge("Tanque con aire que se llena y se enfría (Çengel §5-5)")
    inputs = tr.CHARGING_EXAMPLES["Tanque con aire que se llena sin calor"].inputs
    by_q = tr.solve_charging(
        tr.ChargingInputs(**{**inputs.__dict__, "Q_J": by_t.Q})  # type: ignore[arg-type]
    )
    assert by_q.state2.T == pytest.approx(77.0 + C, rel=1e-8)
    assert by_q.m2 == pytest.approx(by_t.m2, rel=1e-8)


def test_ideal_gas_against_the_real_air() -> None:
    """El vaciado adiabático del aire real y del gas ideal (c_p variable) a 10 → 3 bar."""
    real = tr.solve_discharging(tr.DischargingInputs(AIR_REAL, 1.0, "TP", 300.0, 10e5, 3e5))
    ideal = tr.solve_discharging(tr.DischargingInputs(AIR, 1.0, "TP", 300.0, 10e5, 3e5))
    assert ideal.state2.T == pytest.approx(real.state2.T, abs=1.0)
    assert ideal.m_out == pytest.approx(real.m_out, rel=0.01)
    # ∫h·dm depende de la referencia de h; ∫(h − h₂)·dm, no
    excess = [r.H_out - r.m_out * r.state2.h for r in (ideal, real)]
    assert excess[0] == pytest.approx(excess[1], rel=0.01)


def test_heating_a_charging_tank_from_a_cold_source_is_impossible() -> None:
    inputs = tr.CHARGING_EXAMPLES["Tanque con aire que se llena sin calor"].inputs
    kw = {**inputs.__dict__, "Q_J": 500e3, "T_b_K": 10.0 + C}
    r = tr.solve_charging(tr.ChargingInputs(**kw))  # type: ignore[arg-type]
    assert r.verdict == "imposible" and r.S_gen > 0.0  # la S_gen total no alcanza
    assert r.violation is not None and "siempre más caliente" in r.violation


# ---------------------------------------------------------------------
# Mensajes al alumno
# ---------------------------------------------------------------------


def _charging(**kw: object) -> tr.ChargingInputs:
    base: dict[str, object] = {
        "substance": AIR,
        "volume_m3": 1.0,
        "line_pair": "TP",
        "line_a": 300.0,
        "line_b": 5e5,
        "p2_Pa": 5e5,
        "initial_pair": "TP",
        "initial_a": 300.0,
        "initial_b": 1e5,
    }
    base.update(kw)
    return tr.ChargingInputs(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"substance": sb.incompressible_substance("water")}, "no cambia de densidad"),
        ({"volume_m3": 0.0}, "volumen del tanque"),
        ({"p2_Pa": 6e5}, "no puede superar la presión de la línea"),
        ({"p2_Pa": 0.5e5}, "tiene que ser mayor que la inicial"),
        ({"end": "T", "T2_K": 2000.0}, "cabe menos masa"),
        ({"end": "T", "T2_K": math.nan}, "T₂ tiene que ser positiva"),
        ({"Q_J": -1e9}, "no llega a esa presión"),
        ({"Q_J": math.nan}, "tiene que ser un número"),
        ({"T0_K": 400.0}, "ambiente T₀"),
        ({"T_b_K": -1.0}, "fuente T_b"),
    ],
)
def test_charging_errors(kw: dict[str, object], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        tr.solve_charging(_charging(**kw))


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"p2_Pa": 12e5}, "p₂ tiene que ser menor que p₁"),
        ({"p_out_Pa": 5e5}, "no mayor que la presión final"),
        ({"mode": "polytropic"}, "Modo desconocido"),
        ({"substance": sb.incompressible_substance("oil")}, "no cambia de densidad"),
    ],
)
def test_discharging_errors(kw: dict[str, object], words: str) -> None:
    base: dict[str, object] = {
        "substance": AIR,
        "volume_m3": 1.0,
        "pair1": "TP",
        "a1": 300.0,
        "b1": 10e5,
        "p2_Pa": 3e5,
    }
    base.update(kw)
    with pytest.raises(ValueError, match=words):
        tr.solve_discharging(tr.DischargingInputs(**base))  # type: ignore[arg-type]
