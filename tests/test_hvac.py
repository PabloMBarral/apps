"""Tests de los procesos de acondicionamiento y la torre — Fase 4 (vademecum §14.12).

Contra los ejemplos 14-5 a 14-9 de Cengel, los balances de masa, energía y
exergía de cada proceso, la numeración de los estados y los mensajes al alumno.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import replace

import numpy as np
import pytest

from core.hvac import (
    HVAC_EXAMPLE_NOTES,
    HVAC_EXAMPLES,
    MAX_PROCESSES,
    TOWER_EXAMPLE_NOTES,
    TOWER_EXAMPLES,
    AdiabaticHumidification,
    AdiabaticMixing,
    AirFlow,
    AirInlet,
    CoolingDehumidification,
    HeatingHumidification,
    HvacInputs,
    SensibleProcess,
    cooling_tower_notes,
    cooling_tower_sweep,
    cooling_tower_to_dict,
    default_hvac_sweep_values,
    default_source_temperature,
    default_tower_sweep_values,
    hvac_notes,
    hvac_sweep,
    hvac_to_dict,
    solve_cooling_tower,
    solve_hvac,
)
from core.psychrometrics import (
    P_SEA_LEVEL_PA,
    DeadState,
    state_from_T_phi,
    water_entropy,
)

C = 273.15
P = P_SEA_LEVEL_PA
NAMES = list(HVAC_EXAMPLES)
EX_14_5, EX_14_6, EX_14_7, EX_14_8, EX_SUMMER, EX_WINTER = NAMES
PER_MIN = 60.0


def _inlet(T_C: float, phi: float, V: float = 1.0) -> AirInlet:
    return AirInlet(T_C + C, phi, AirFlow(V))


# ---------------------------------------------------------------------
# Ejemplos de Cengel
# ---------------------------------------------------------------------


def test_cengel_14_5_heating_and_humidification() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_14_5])
    heating, humidifier = r.processes
    assert r.inlet.m_dry_air_kg_s * PER_MIN == pytest.approx(55.2, abs=0.05)
    # Cengel: 673 kJ/min con las h de la carta redondeadas (28,0 − 15,8); sin redondear, 668.
    assert heating.Q_W * PER_MIN / 1e3 == pytest.approx(673.0, rel=0.01)
    assert humidifier.m_water_kg_s * PER_MIN == pytest.approx(0.539, rel=0.002)
    # El libro no hace el balance del humidificador: con vapor saturado a 100 °C falta calor.
    assert humidifier.Q_W > 0
    assert any("hace falta además" in n for n in hvac_notes(r))


def test_cengel_14_6_cooling_and_dehumidification() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_14_6])
    (coil,) = r.processes
    assert coil.Q_W * PER_MIN / 1e3 == pytest.approx(-511.0, rel=0.002)
    assert -coil.m_water_kg_s * PER_MIN == pytest.approx(0.131, rel=0.005)
    assert r.water_removed_kg_s == pytest.approx(-coil.m_water_kg_s)
    assert r.outlet.state.saturated


def test_cengel_14_7_evaporative_cooler() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_14_7])
    s1, s2 = r.inlet.state, r.outlet.state
    assert (s2.T_K - C) * 1.8 + 32 == pytest.approx(70.0, abs=0.5)
    assert (s1.T_wb_K - C) * 1.8 + 32 == pytest.approx(66.0, abs=0.1)
    # casi sobre la línea de bulbo húmedo constante
    assert s2.T_wb_K == pytest.approx(s1.T_wb_K, abs=0.05)


def test_cengel_14_8_mixing() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_14_8])
    mix = r.outlet.state
    assert mix.omega == pytest.approx(0.0122, abs=1e-4)
    assert mix.phi == pytest.approx(0.89, abs=0.005)
    assert mix.T_K - C == pytest.approx(19.0, abs=0.1)
    assert r.outlet.V_m3_s * PER_MIN == pytest.approx(70.1, abs=0.15)
    # la mezcla cae sobre la recta 1–2 (regla de la palanca)
    s1, s2 = r.stream(1), r.stream(2)
    ratio = s1.m_dry_air_kg_s / s2.m_dry_air_kg_s
    w1, w2, w3 = s1.state.omega, s2.state.omega, mix.omega
    h1, h2, h3 = s1.state.h_J_per_kg, s2.state.h_J_per_kg, mix.h_J_per_kg
    assert ratio == pytest.approx((w2 - w3) / (w3 - w1))
    assert ratio == pytest.approx((h2 - h3) / (h3 - h1))


def test_cengel_14_9_cooling_tower() -> None:
    r = solve_cooling_tower(TOWER_EXAMPLES["Cengel 14-9: torre de una central"])
    assert r.makeup_kg_s == pytest.approx(1.80, rel=0.005)
    assert r.m_dry_air_kg_s == pytest.approx(97.5, abs=0.1)  # el libro, con la carta: 96,9
    assert r.V_air_in_m3_s == pytest.approx(82.1, abs=0.1)
    assert r.range_K == pytest.approx(13.0)
    assert r.approach_K == pytest.approx(22 + C - r.air_in.T_wb_K)
    assert 0 < r.effectiveness < 1
    # balance de energía de toda la torre
    lhs = r.inputs.m_water_in_kg_s * r.h_water_in + r.m_dry_air_kg_s * r.air_in.h_J_per_kg
    rhs = r.m_water_out_kg_s * r.h_water_out + r.m_dry_air_kg_s * r.air_out.h_J_per_kg
    assert lhs == pytest.approx(rhs, rel=1e-12)


# ---------------------------------------------------------------------
# Balances de cada proceso
# ---------------------------------------------------------------------


def _all_results():  # type: ignore[no-untyped-def]
    return [solve_hvac(HVAC_EXAMPLES[name]) for name in NAMES]


@pytest.mark.parametrize("name", NAMES)
def test_every_process_closes_mass_and_energy(name: str) -> None:
    r = solve_hvac(HVAC_EXAMPLES[name])
    for p in r.processes:
        m_in = p.inlet.m_dry_air_kg_s + (p.other.m_dry_air_kg_s if p.other else 0.0)
        assert p.outlet.m_dry_air_kg_s == pytest.approx(m_in, rel=1e-12)
        water_in = p.inlet.m_vapor_kg_s + (p.other.m_vapor_kg_s if p.other else 0.0)
        assert p.outlet.m_vapor_kg_s == pytest.approx(water_in + p.m_water_kg_s, rel=1e-12)
        H_in = p.inlet.m_dry_air_kg_s * p.inlet.state.h_J_per_kg
        if p.other:
            H_in += p.other.m_dry_air_kg_s * p.other.state.h_J_per_kg
        H_out = p.outlet.m_dry_air_kg_s * p.outlet.state.h_J_per_kg
        water = p.m_water_kg_s * (p.h_water_J_per_kg or 0.0)
        assert H_out == pytest.approx(H_in + p.Q_W + water, abs=1e-9 * abs(H_in) + 1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_destroyed_exergy_is_positive_and_equals_T0_Sgen(name: str) -> None:
    r = solve_hvac(HVAC_EXAMPLES[name])
    T0 = r.dead.T0_K
    for p in r.processes:
        assert p.X_destroyed_W > 0
        S_in = p.inlet.m_dry_air_kg_s * p.inlet.state.s_J_per_kg_K
        if p.other:
            S_in += p.other.m_dry_air_kg_s * p.other.state.s_J_per_kg_K
        S_out = p.outlet.m_dry_air_kg_s * p.outlet.state.s_J_per_kg_K
        if p.m_water_kg_s > 0:
            S_in += p.m_water_kg_s * water_entropy(p.T_water_K, p.water_phase)
        elif p.m_water_kg_s < 0:
            S_out -= p.m_water_kg_s * water_entropy(p.T_water_K, "liquid")
        S_gen = S_out - S_in - (p.Q_W / p.T_source_K if p.T_source_K else 0.0)
        # 0,622 y 1,608 son redondeos: con agua que entra o sale difieren en ~1e-4
        assert p.X_destroyed_W == pytest.approx(T0 * S_gen, rel=1e-3)


def test_cooling_tower_destroys_exergy() -> None:
    for inputs in TOWER_EXAMPLES.values():
        r = solve_cooling_tower(inputs)
        assert r.X_destroyed_W > 0


# ---------------------------------------------------------------------
# Numeración y nombres
# ---------------------------------------------------------------------


def test_numbering_follows_the_flow() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_SUMMER])
    assert [s.number for s in r.streams] == [1, 2, 3, 4, 5]
    labels = [p.label for p in r.processes]
    assert labels == [
        "Mezcla adiabática (1 + 2 → 3)",
        "Enfriamiento y deshumidificación (3 → 4)",
        "Calentamiento sensible (4 → 5)",
    ]
    assert r.stream(2).label == "corriente que se mezcla"
    with pytest.raises(KeyError):
        r.stream(9)


def test_process_names() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_14_7])
    assert r.processes[0].name == "Humidificación adiabática con agua"
    cooling = solve_hvac(HvacInputs(P, _inlet(40, 0.2), (SensibleProcess(25 + C),)))
    assert cooling.processes[0].name == "Enfriamiento sensible"
    assert cooling.cooling_W > 0 and cooling.heating_W == 0


# ---------------------------------------------------------------------
# Fuente de calor y ambiente
# ---------------------------------------------------------------------


def test_default_source_temperatures() -> None:
    s = state_from_T_phi(P, 20 + C, 0.5)
    assert default_source_temperature(SensibleProcess(30 + C), s) == pytest.approx(60 + C)
    assert default_source_temperature(SensibleProcess(80 + C), s) == pytest.approx(90 + C)
    # enfriamiento sensible: 5 K más frío que la salida, sin bajar del punto de rocío
    assert default_source_temperature(SensibleProcess(15 + C), s) == pytest.approx(
        max(10 + C, s.T_dp_K)
    )
    coil = CoolingDehumidification(12 + C, 0.95, 9 + C)
    assert default_source_temperature(coil, s) == pytest.approx(9 + C)
    assert default_source_temperature(AdiabaticHumidification(0.6), s) is None


def test_dead_state_defaults_to_the_inlet_air() -> None:
    inputs = HVAC_EXAMPLES[EX_14_6]
    assert inputs.dead is None
    r = solve_hvac(inputs)
    assert r.dead.T0_K == pytest.approx(30 + C)
    assert r.dead.phi0 == pytest.approx(0.8)
    assert r.inlet.state.psi(r.dead) == pytest.approx(0.0, abs=1e-9)


def test_hotter_source_destroys_more_exergy() -> None:
    def x_dest(T_b: float) -> float:
        inputs = HvacInputs(P, _inlet(10, 0.4), (SensibleProcess(30 + C, T_source_K=T_b),))
        return solve_hvac(inputs).processes[0].X_destroyed_W

    assert x_dest(40 + C) < x_dest(60 + C) < x_dest(120 + C)


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("processes", "match"),
    [
        ((SensibleProcess(5 + C),), "punto de rocío"),
        ((SensibleProcess(25 + C),), "igual a la de entrada"),
        ((SensibleProcess(40 + C, T_source_K=35 + C),), "más caliente"),
        ((SensibleProcess(20 + C, T_source_K=10 + C),), "condensaría"),
        ((CoolingDehumidification(22 + C, 0.95),), "no condensa"),
        ((CoolingDehumidification(30 + C, 0.95),), "más frío"),
        ((CoolingDehumidification(10 + C, 0.95, -2 + C),), "escarcha"),
        ((CoolingDehumidification(10 + C, 0.95, 12 + C),), "superficie fría"),
        ((HeatingHumidification(30 + C, 0.1),), "no hay agua que agregar"),
        ((HeatingHumidification(30 + C, 0.5, "vapor", 90 + C),), "presión mayor"),
        ((HeatingHumidification(30 + C, 0.5, "liquid", 105 + C),), "líquida"),
        ((AdiabaticHumidification(0.3),), "humedad relativa mayor"),
        (
            (
                AdiabaticMixing(-5 + C, 1.0, AirFlow(1.0)),
                AdiabaticMixing(40 + C, 1.0, AirFlow(1.0)),
            ),
            "niebla",
        ),
        ((), "al menos un proceso"),
        ((SensibleProcess(30 + C),) * (MAX_PROCESSES + 1), "Como mucho"),
    ],
)
def test_invalid_processes_explain_why(processes: tuple, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match=match):
        solve_hvac(HvacInputs(P, _inlet(25, 0.5), processes))


def test_errors_name_the_process() -> None:
    inputs = HvacInputs(
        P, _inlet(25, 0.5), (SensibleProcess(35 + C), CoolingDehumidification(30 + C, 0.95))
    )
    with pytest.raises(ValueError, match=r"^Proceso 2 \(enfriamiento y deshumidificación\)"):
        solve_hvac(inputs)


def test_negative_flow_is_rejected() -> None:
    with pytest.raises(ValueError, match="positivo"):
        AirFlow(0.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"T_water_out_K": 15 + C}, "bulbo húmedo"),
        ({"T_water_out_K": 36 + C}, "más fría"),
        ({"T_air_out_K": 40 + C}, "más caliente"),
        ({"T_air_out_K": 21 + C, "phi_air_out": 0.4}, "más humedad"),
        ({"m_water_in_kg_s": 0.0}, "positivo"),
        ({"T_water_in_K": 105 + C}, "ebullición"),
    ],
)
def test_invalid_tower_explains_why(changes: dict, match: str) -> None:  # type: ignore[type-arg]
    inputs = replace(TOWER_EXAMPLES["Cengel 14-9: torre de una central"], **changes)
    with pytest.raises(ValueError, match=match):
        solve_cooling_tower(inputs)


# ---------------------------------------------------------------------
# Barridos, notas, ejemplos y export
# ---------------------------------------------------------------------


def test_hvac_sweep_of_the_outdoor_temperature() -> None:
    inputs = HVAC_EXAMPLES[EX_14_6]
    values = default_hvac_sweep_values(inputs, "T_in")
    points = hvac_sweep(inputs, "T_in", values)
    assert len(points) > 10
    cooling = [pt.cooling_W for pt in points]
    assert all(np.diff(cooling) > 0)  # más calor afuera, más carga
    # el ambiente queda fijo (el de los datos), no sigue al barrido
    assert all(pt.X_destroyed_W > 0 for pt in points)


def test_hvac_sweep_of_humidity_skips_impossible_points() -> None:
    inputs = HVAC_EXAMPLES[EX_14_6]
    points = hvac_sweep(inputs, "phi_in", default_hvac_sweep_values(inputs, "phi_in"))
    # con aire muy seco el serpentín no condensa (ω₁ ≤ ω₂): esos puntos se omiten
    assert 0 < len(points) < 20
    omega_out = solve_hvac(inputs).outlet.state.omega
    for value in default_hvac_sweep_values(inputs, "phi_in"):
        kept = any(pt.value == pytest.approx(value) for pt in points)
        condenses = state_from_T_phi(P, inputs.inlet.T_K, float(value)).omega > omega_out
        assert kept == condenses


def test_tower_sweep_of_the_approach() -> None:
    inputs = TOWER_EXAMPLES["Cengel 14-9: torre de una central"]
    points = cooling_tower_sweep(
        inputs, "T_water_out", default_tower_sweep_values(inputs, "T_water_out")
    )
    assert len(points) > 10
    # con más aproximación hace falta menos aire, pero se enfría menos el agua
    approaches = [pt.approach_K for pt in points]
    assert all(np.diff(approaches) > 0)
    for param in ("T_air_in", "phi_air_in"):
        pts = cooling_tower_sweep(inputs, param, default_tower_sweep_values(inputs, param))
        assert pts


@pytest.mark.parametrize("name", NAMES)
def test_examples_solve_without_warnings(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_hvac(HVAC_EXAMPLES[name])
    assert HVAC_EXAMPLE_NOTES[name]
    pickle.loads(pickle.dumps(r))
    for system in ("SI", "Técnico", "Inglés"):
        json.dumps(hvac_to_dict(r, system), ensure_ascii=False)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", list(TOWER_EXAMPLES))
def test_tower_examples(name: str) -> None:
    r = solve_cooling_tower(TOWER_EXAMPLES[name])
    assert TOWER_EXAMPLE_NOTES[name]
    assert cooling_tower_notes(r)
    pickle.loads(pickle.dumps(r))
    data = cooling_tower_to_dict(r, "Técnico")
    json.dumps(data, ensure_ascii=False)
    assert data["agua_de_reposicion"]["valor"] == pytest.approx(r.makeup_kg_s)


def test_bariloche_uses_the_altitude_pressure() -> None:
    r = solve_hvac(HVAC_EXAMPLES[EX_WINTER])
    assert r.inputs.p_Pa / 1e3 == pytest.approx(91.0, abs=0.2)
    assert r.inputs.altitude_m == 900.0
    assert r.processes[0].kind == "heating"
    assert r.processes[1].T_water_K == pytest.approx(15 + C)


def test_dead_state_pressure_follows_the_train() -> None:
    """Un ambiente dado a otra presión se pasa a la del tren (mismo lugar)."""
    inputs = replace(HVAC_EXAMPLES[EX_14_6], dead=DeadState(303.15, 0.8, 90e3))
    r = solve_hvac(inputs)
    assert r.dead.p0_Pa == pytest.approx(P)
