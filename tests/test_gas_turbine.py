"""Tests de la turbina de gas con etapas y regenerador (:mod:`core.cycles.gas_turbine`) — Fase 3.7.

Referencias: Cengel & Boles 8.ª ed., ejemplos 9-5 (Brayton ideal), 9-6 (real),
9-7 (con regenerador) y 9-8 (dos etapas de compresión y de expansión, con y
sin regenerador ideal), de aire estándar con la tabla A-17; la turbina de gas
de la Fase 3.4 (:func:`core.cycles.brayton.solve_brayton`), que tiene que dar
idéntico con una etapa y sin regenerador; una red de TESPy con las mismas
etapas como control cruzado; y los balances de energía, masa y exergía, que
tienen que cerrar exactos.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import replace

import pytest

from core.cycles.brayton import METHANE, BraytonInputs, Fuel, solve_brayton
from core.cycles.gas_turbine import (
    DEFAULT_EFFECTIVENESS,
    GAS_TURBINE_EXAMPLE_NOTES,
    GAS_TURBINE_EXAMPLES,
    GAS_TURBINE_SWEEP_PARAMETERS,
    MAX_STAGES,
    GasTurbineInputs,
    default_gas_turbine_sweep_values,
    from_brayton,
    gas_turbine_exergy,
    gas_turbine_lines,
    gas_turbine_notes,
    gas_turbine_sweep,
    gas_turbine_tespy,
    gas_turbine_to_dict,
    improvement_comparison,
    of_component,
    solve_gas_turbine,
    stage_names,
)
from core.ideal_gas import AIR_TECHNICAL

C = 273.15
NAMES = list(GAS_TURBINE_EXAMPLES)
EX_9_5, EX_9_6, EX_9_7, EX_9_8A, EX_9_8B, EX_IND, EX_MICRO, EX_AERO, EX_SEQ = NAMES
CENGEL_9_5 = GAS_TURBINE_EXAMPLES[EX_9_5]
INDUSTRIAL = GAS_TURBINE_EXAMPLES[EX_IND]
FULL = GasTurbineInputs(
    30.0,
    1400 + C,
    0.88,
    0.90,
    dp_combustor=0.03,
    compressor_stages=2,
    turbine_stages=2,
    T_intercool_K=30 + C,
    dp_intercooler=0.02,
    regenerator=0.85,
    dp_regenerator=0.03,
)
#: Una grilla de configuraciones: aire estándar y combustión, etapas, regenerador, pérdidas.
GRID = {
    "simple aire": CENGEL_9_5,
    "simple metano": INDUSTRIAL,
    "regenerador aire": GAS_TURBINE_EXAMPLES[EX_9_7],
    "regenerador metano": GAS_TURBINE_EXAMPLES[EX_MICRO],
    "2+2 aire": GAS_TURBINE_EXAMPLES[EX_9_8A],
    "interenfriada metano": GAS_TURBINE_EXAMPLES[EX_AERO],
    "secuencial": GAS_TURBINE_EXAMPLES[EX_SEQ],
    "2+2 metano con regenerador": FULL,
    "3+3 metano con regenerador": replace(
        FULL, compressor_stages=3, turbine_stages=3, T_turbine_in_K=1300 + C, T_intercool_K=None
    ),
    "3+1 aire técnico": replace(INDUSTRIAL, compressor_stages=3, air=AIR_TECHNICAL),
}


def _labels(inputs: GasTurbineInputs) -> list[str]:
    return [s.label for s in solve_gas_turbine(inputs).states]


# ---------------------------------------------------------------------
# Cengel 9-5 a 9-8 (aire estándar)
# ---------------------------------------------------------------------


def test_cengel_9_5_ideal_brayton() -> None:
    r = solve_gas_turbine(CENGEL_9_5)
    assert r.state("2").T_K == pytest.approx(540.0, abs=0.5)
    assert r.state("4").T_K == pytest.approx(770.0, abs=0.6)
    assert r.back_work_ratio == pytest.approx(0.403, abs=1e-3)
    assert r.eta_th == pytest.approx(0.426, abs=1e-3)


def test_cengel_9_6_actual_gas_turbine() -> None:
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_6])
    assert r.back_work_ratio == pytest.approx(0.592, abs=1e-3)
    assert r.eta_th == pytest.approx(0.266, abs=1e-3)
    assert r.q_in_J_per_kg / 1e3 == pytest.approx(790.58, rel=2e-3)


def test_cengel_9_7_regenerator() -> None:
    """ε = 0,80: q_regen = 219,98 kJ/kg, q_ent = 570,60 kJ/kg y η = 36,9 %."""
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_7])
    rg = r.regenerator
    assert rg is not None
    assert rg.q / 1e3 == pytest.approx(219.98, rel=3e-3)
    assert r.q_in_J_per_kg / 1e3 == pytest.approx(570.60, rel=2e-3)
    assert r.eta_th == pytest.approx(0.369, abs=1e-3)
    # El trabajo neto no cambia con el regenerador (9-6: 210,41 kJ/kg).
    assert r.w_net_J_per_kg / 1e3 == pytest.approx(210.41, rel=2e-3)
    assert rg.q / rg.q_max == pytest.approx(0.80, rel=1e-12)
    # Numeración de la figura 9-38: 5 y 6 son las salidas del regenerador.
    assert [s.label for s in r.states] == ["1", "2s", "2", "3", "4s", "4", "5", "6"]
    assert r.states[rg.air_out].label == "5" and r.states[r.exhaust].label == "6"
    assert r.combustors[0].inlet == rg.air_out


def test_cengel_9_8_intercooling_and_reheating() -> None:
    """Dos etapas de compresión y de expansión, r_p = 8: r_bw = 0,304 y η = 35,8 %."""
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8A])
    assert r.back_work_ratio == pytest.approx(0.304, abs=1e-3)
    assert r.eta_th == pytest.approx(0.358, abs=1e-3)
    assert r.w_compressor_J_per_kg / 1e3 == pytest.approx(208.24, rel=2e-3)
    assert r.w_turbine_J_per_kg / 1e3 == pytest.approx(685.28, rel=2e-3)
    assert r.q_in_J_per_kg / 1e3 == pytest.approx(1334.30, rel=1e-3)
    T = {s.label: s.T_K for s in r.states}
    assert T["2"] == pytest.approx(403.3, abs=0.2) and T["4"] == pytest.approx(403.3, abs=0.2)
    assert T["6"] == pytest.approx(1006.4, abs=0.2) and T["8"] == pytest.approx(1006.4, abs=0.2)
    # Relaciones iguales: √8 por etapa.
    p = {s.label: s.P_Pa for s in r.states}
    assert p["2"] / p["1"] == pytest.approx(math.sqrt(8.0))
    assert p["5"] / p["6"] == pytest.approx(math.sqrt(8.0))


def test_cengel_9_8_with_an_ideal_regenerator() -> None:
    """Con un regenerador ideal η = 69,6 %: el aire sale a T₉ y el escape a T₄."""
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8B])
    assert r.eta_th == pytest.approx(0.696, abs=1e-3)
    assert r.back_work_ratio == pytest.approx(0.304, abs=1e-3)
    T = {s.label: s.T_K for s in r.states}
    assert T["5"] == pytest.approx(T["9"], abs=1e-6)
    assert T["10"] == pytest.approx(T["4"], abs=1e-6)
    assert [s.label for s in r.states] == [
        "1", "2s", "2", "3", "4s", "4", "5", "6", "7s", "7", "8", "9s", "9", "10",
    ]  # fmt: skip


def test_stage_names() -> None:
    assert stage_names("compresor", 1) == ["compresor"]
    assert stage_names("compresor", 2) == ["compresor de baja", "compresor de alta"]
    assert stage_names("turbina", 3) == ["turbina de alta", "turbina de media", "turbina de baja"]
    assert stage_names("turbina", 4) == [f"turbina {k}" for k in (1, 2, 3, 4)]
    assert of_component("escape") == "del escape"
    assert of_component("cámara de recalentamiento") == "de la cámara de recalentamiento"
    assert of_component("turbina de baja") == "de la turbina de baja"


@pytest.mark.parametrize(
    ("inputs", "labels"),
    [
        (CENGEL_9_5, ["1", "2s", "2", "3", "4s", "4"]),
        (
            replace(CENGEL_9_5, compressor_stages=2),
            ["1", "2s", "2", "3", "4s", "4", "5", "6s", "6"],
        ),
        (
            replace(CENGEL_9_5, compressor_stages=2, regenerator=0.8),
            ["1", "2s", "2", "3", "4s", "4", "5", "6", "7s", "7", "8"],
        ),
        (
            replace(CENGEL_9_5, turbine_stages=2, regenerator=0.8),
            ["1", "2s", "2", "3", "4", "5s", "5", "6", "7s", "7", "8"],
        ),
    ],
)
def test_numbering_follows_the_flow_with_stages(
    inputs: GasTurbineInputs, labels: list[str]
) -> None:
    assert _labels(inputs) == labels


def test_state_descriptions() -> None:
    r = solve_gas_turbine(FULL)
    d = {s.label: s.description for s in r.states}
    assert d["1"] == "aire ambiente (entrada del compresor de baja)"
    assert d["3"] == "salida del interenfriador (entrada del compresor de alta)"
    assert d["5"] == "aire que sale del regenerador"
    assert d["6"] == "salida de la cámara de combustión (entrada a la turbina de alta)"
    assert d["8"] == "salida del recalentamiento (entrada a la turbina de baja)"
    assert d["10"] == "escape (sale del regenerador)"
    air = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8B])
    assert {s.label: s.description for s in air.states}["6"].startswith("salida del calentador")
    assert [c.name for c in r.combustors] == ["cámara de combustión", "cámara de recalentamiento"]
    assert [c.name for c in air.combustors] == ["calentador", "recalentador"]
    assert [ic.name for ic in r.intercoolers] == ["interenfriador"]


# ---------------------------------------------------------------------
# Una etapa sin regenerador = la Fase 3.4
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        BraytonInputs(8.0, 1300.0, T_amb_K=300.0, p_amb_Pa=100e3, fuel=None),
        BraytonInputs(8.0, 1300.0, 0.80, 0.85, T_amb_K=300.0, p_amb_Pa=100e3, fuel=None),
        BraytonInputs(15.0, 1250 + C, 0.88, 0.90, dp_combustor=0.03, m_air_kg_s=500.0),
        BraytonInputs(17.0, 1400 + C, 0.90, 0.91, dp_combustor=0.03, air=AIR_TECHNICAL),
        BraytonInputs(
            12.0,
            1200 + C,
            0.86,
            0.89,
            fuel=Fuel.from_fractions({"methane": 0.9, "ethane": 0.06, "nitrogen": 0.04}),
        ),
    ],
)
def test_one_stage_without_regenerator_is_the_fase_3_4_gas_turbine(inputs: BraytonInputs) -> None:
    old = solve_brayton(inputs)
    new = solve_gas_turbine(from_brayton(inputs))
    assert [s.label for s in new.states] == [s.label for s in old.states]
    for a, b in zip(old.states, new.states, strict=True):
        assert (b.T_K, b.P_Pa, b.h_J_per_kg, b.s_J_per_kg_K) == pytest.approx(
            (a.T_K, a.P_Pa, a.h_J_per_kg, a.s_J_per_kg_K), rel=1e-12
        )
        # Con aire estándar la Fase 3.4 rotulaba «gases» la turbina; por ella pasa aire.
        assert b.medium == (a.medium if inputs.fuel is not None else "aire")
    assert new.fuel_air_ratio == pytest.approx(old.fuel_air_ratio, rel=1e-12)
    assert new.eta_th == pytest.approx(old.eta_th, rel=1e-12)
    assert new.W_net_W == pytest.approx(old.W_net_W, rel=1e-12)
    assert new.back_work_ratio == pytest.approx(old.back_work_ratio, rel=1e-12)
    assert (new.excess_air is None) == (old.excess_air is None)
    if old.excess_air is not None:
        assert new.excess_air == pytest.approx(old.excess_air, rel=1e-12)
    assert new.inputs.brayton() == inputs


# ---------------------------------------------------------------------
# Balances
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(GRID))
def test_energy_and_mass_balances_close(name: str) -> None:
    r = solve_gas_turbine(GRID[name])
    assert abs(r.energy_residual_J_per_kg) < 1e-8 * r.q_in_J_per_kg
    F = 0.0
    for cc, t in zip(r.combustors, r.turbines, strict=True):
        assert cc.F_in == pytest.approx(F, abs=1e-15)
        F += cc.fuel
        assert cc.F_out == pytest.approx(F, rel=1e-12)
        assert t.m == pytest.approx(1.0 + F, rel=1e-12)
    assert r.fuel_air_ratio == pytest.approx(F, rel=1e-12)
    rg = r.regenerator
    if rg is not None:
        # Lo que gana el aire es lo que pierden los gases.
        gained = r.h(rg.air_out) - r.h(rg.air_in)
        lost = rg.m_gas * (r.h(rg.gas_in) - r.h(rg.gas_out))
        assert gained == pytest.approx(lost, rel=1e-10)
        assert r.states[rg.gas_out].T_K >= r.states[rg.air_in].T_K - 1e-6
    assert r.W_net_W == pytest.approx(r.W_turbine_W - r.W_compressor_W)
    assert r.heat_rate_kJ_per_kWh == pytest.approx(3600.0 / r.eta_th)
    assert r.eta_th < r.eta_carnot


@pytest.mark.parametrize("name", list(GRID))
def test_exergy_balance_closes(name: str) -> None:
    r = solve_gas_turbine(GRID[name])
    e = gas_turbine_exergy(r)
    assert abs(e.residual_J_per_kg) < 1e-9 * e.x_in_J_per_kg
    for item in e.items:
        assert item.x_J_per_kg >= -1e-9 * e.x_in_J_per_kg, item
    assert e.total("útil") == pytest.approx(r.w_net_J_per_kg)
    assert 0.0 < e.efficiency < 1.0
    names = [i.name for i in e.items]
    assert names[0] == "trabajo neto" and names[-1] == "escape"
    assert len(names) == len(set(names))
    if r.inputs.fuel is not None:
        # La exergía del combustible ≈ PCI (vademecum §16.13): η_II = η.
        assert e.efficiency == pytest.approx(r.eta_th, rel=1e-12)


def test_exergy_by_component_is_physical() -> None:
    r = solve_gas_turbine(INDUSTRIAL)
    e = {i.name: i.x_J_per_kg for i in gas_turbine_exergy(r).items}
    x_in = gas_turbine_exergy(r).x_in_J_per_kg
    # La cámara de combustión es la gran destructora (Cengel §9-12).
    assert 0.25 < e["cámara de combustión"] / x_in < 0.31
    assert e["compresor"] == pytest.approx(
        r.inputs.T_amb_K * (r.state("2").s_J_per_kg_K - r.state("1").s_J_per_kg_K)
    )
    assert e["turbina"] > 0 and e["escape"] > 0
    ideal = gas_turbine_exergy(solve_gas_turbine(CENGEL_9_5))
    assert {i.name: i.x_J_per_kg for i in ideal.items}["compresor"] == 0.0  # isoentrópico


def test_exergy_with_an_ambient_below_zero() -> None:
    """El estado muerto de los gases (con agua) a −20 °C: el agua sigue como gas ideal."""
    r = solve_gas_turbine(replace(INDUSTRIAL, T_amb_K=-20 + C))
    e = gas_turbine_exergy(r)
    assert abs(e.residual_J_per_kg) < 1e-9 * e.x_in_J_per_kg
    assert e.T0_K == pytest.approx(-20 + C)


def test_air_standard_exergy_uses_the_heat_at_the_outlet_temperature() -> None:
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8A])
    e = gas_turbine_exergy(r)
    expected = sum(cc.q * (1.0 - 300.0 / r.states[cc.outlet].T_K) for cc in r.combustors)
    assert e.x_in_J_per_kg == pytest.approx(expected)
    assert e.efficiency > r.eta_th  # parte del calor ya era anergía


# ---------------------------------------------------------------------
# Combustión secuencial y regenerador con combustión
# ---------------------------------------------------------------------


def test_sequential_combustion() -> None:
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_SEQ])
    first, second = r.combustors
    assert first.fuel > second.fuel > 0
    # Cada cámara cambia la composición: hay menos O₂ después de la segunda.
    o2_mid = r.states[second.inlet].mix.mole_fractions["O2"]
    o2_end = r.gas.mole_fractions["O2"]
    assert o2_end < o2_mid < 0.2095
    f_st = 1.0 / 17.12  # metano con aire seco, aproximado
    assert r.excess_air == pytest.approx(f_st / r.fuel_air_ratio, rel=0.02)
    assert r.excess_air > 1.0
    # El recalentamiento a 18 bar sube el rendimiento frente a recalentar a la presión media.
    equal = solve_gas_turbine(replace(GAS_TURBINE_EXAMPLES[EX_SEQ], p_reheat_Pa=()))
    assert r.eta_th > equal.eta_th + 0.03
    assert equal.T_exhaust_K > r.T_exhaust_K + 150.0


def test_regenerator_with_combustion_is_consistent() -> None:
    """Con combustión, f depende de T₅ y T₄ de f: la iteración tiene que cerrar."""
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_MICRO])
    rg = r.regenerator
    assert rg is not None
    air = r.inputs.air
    T4 = r.states[rg.gas_in].T_K
    h2 = r.h(rg.air_in)
    assert r.h(rg.air_out) == pytest.approx(h2 + 0.87 * (air.h(T4) - h2), rel=1e-12)
    # Sin el regenerador hace falta el doble de combustible para la misma TIT. Sin pérdidas
    # de carga, el trabajo neto cambia poco: solo por los gases que agrega el combustible.
    plain = solve_gas_turbine(replace(r.inputs, regenerator=None))
    assert plain.fuel_air_ratio > 1.8 * r.fuel_air_ratio
    lossless = solve_gas_turbine(replace(r.inputs, dp_regenerator=0.0))
    assert lossless.w_net_J_per_kg < plain.w_net_J_per_kg < 1.05 * lossless.w_net_J_per_kg
    # La pérdida de carga del regenerador (3 % por lado) cuesta trabajo: con r_p = 4, mucho.
    assert r.w_net_J_per_kg < 0.92 * lossless.w_net_J_per_kg


def test_sizing_by_net_power() -> None:
    r = solve_gas_turbine(replace(INDUSTRIAL, W_net_W=150e6))
    assert r.W_net_W == pytest.approx(150e6, rel=1e-12)
    assert r.m_air_kg_s == pytest.approx(150e6 / r.w_net_J_per_kg)


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(CENGEL_9_5, pressure_ratio=1.0), "mayor que 1"),
        (replace(CENGEL_9_5, compressor_stages=0), "etapas de compresión"),
        (replace(CENGEL_9_5, turbine_stages=MAX_STAGES + 1), "etapas de expansión"),
        (replace(CENGEL_9_5, dp_intercooler=0.3), "cada interenfriador"),
        (replace(CENGEL_9_5, dp_regenerator=-0.1), "cada lado del regenerador"),
        (replace(CENGEL_9_5, regenerator=0.0), "efectividad del regenerador"),
        (replace(CENGEL_9_5, regenerator=1.2), "efectividad del regenerador"),
        (
            replace(CENGEL_9_5, compressor_stages=2, T_intercool_K=900.0),
            "después de interenfriar",
        ),
        (
            replace(CENGEL_9_5, compressor_stages=2, T_intercool_K=420.0),
            "no enfría",
        ),
        (
            replace(CENGEL_9_5, turbine_stages=2, T_reheat_K=2100.0),
            "supera el rango",
        ),
        (
            replace(CENGEL_9_5, turbine_stages=2, T_reheat_K=900.0),
            "no calienta",
        ),
        (
            replace(CENGEL_9_5, compressor_stages=2, p_intercool_Pa=(3e5, 4e5)),
            "hacen falta 1 presiones de interenfriamiento",
        ),
        (
            replace(CENGEL_9_5, compressor_stages=2, p_intercool_Pa=(9e5,)),
            "tienen que ir creciendo",
        ),
        (
            replace(CENGEL_9_5, turbine_stages=2, p_reheat_Pa=(9e5,)),
            "tienen que ir bajando",
        ),
        (
            replace(CENGEL_9_5, turbine_stages=3, p_reheat_Pa=(5e5,)),
            "hacen falta 2 presiones de recalentamiento",
        ),
        (replace(CENGEL_9_5, pressure_ratio=30.0, regenerator=0.8), "El regenerador no sirve"),
        (replace(CENGEL_9_5, T_turbine_in_K=500.0), "tiene que superar la del aire"),
        (replace(CENGEL_9_5, W_net_W=-1.0), "potencia neta"),
        (
            replace(INDUSTRIAL, pressure_ratio=30.0, T_turbine_in_K=1699 + C, turbine_stages=4),
            "λ < 1",
        ),
    ],
)
def test_invalid_inputs_are_explained(inputs: GasTurbineInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_gas_turbine(inputs)


def test_regenerator_message_names_both_temperatures() -> None:
    with pytest.raises(ValueError, match=r"T₄ > T₂ \(Cengel §9-9\)"):
        solve_gas_turbine(replace(INDUSTRIAL, pressure_ratio=30.0, regenerator=0.8))


def test_sequential_combustion_runs_out_of_oxygen() -> None:
    with pytest.raises(ValueError, match="Recalentamiento antes de la turbina 4"):
        solve_gas_turbine(
            replace(INDUSTRIAL, pressure_ratio=30.0, T_turbine_in_K=1699 + C, turbine_stages=4)
        )


# ---------------------------------------------------------------------
# Comparación de las mejoras
# ---------------------------------------------------------------------


def test_improvement_comparison_like_cengel() -> None:
    rows = improvement_comparison(CENGEL_9_5)
    assert [r.label for r in rows] == [
        "Simple",
        "Con regenerador",
        "Con interenfriamiento (2 etapas)",
        "Con recalentamiento (2 etapas)",
        "Interenfriamiento y recalentamiento",
        "Interenfriamiento, recalentamiento y regenerador",
    ]
    assert [r.current for r in rows] == [True, False, False, False, False, False]
    eta = {r.label: r.result.eta_th for r in rows if r.result is not None}
    simple = eta["Simple"]
    # Sin regenerador, interenfriar y recalentar bajan el rendimiento (Cengel §9-10)…
    assert eta["Interenfriamiento y recalentamiento"] < simple
    assert eta["Con interenfriamiento (2 etapas)"] < simple
    assert eta["Con recalentamiento (2 etapas)"] < simple
    # …y con regenerador lo suben más que el regenerador solo.
    assert eta["Interenfriamiento, recalentamiento y regenerador"] > eta["Con regenerador"]
    assert rows[1].inputs.regenerator == DEFAULT_EFFECTIVENESS
    # El 9-8 a) es la fila de interenfriamiento y recalentamiento.
    assert eta["Interenfriamiento y recalentamiento"] == pytest.approx(0.3573, abs=2e-4)


def test_improvement_comparison_marks_a_useless_regenerator() -> None:
    rows = improvement_comparison(GAS_TURBINE_EXAMPLES[EX_AERO])
    by = {r.label: r for r in rows}
    assert by["Con regenerador"].result is None
    assert "El regenerador no sirve" in by["Con regenerador"].error
    assert by["Con interenfriamiento (2 etapas)"].current
    # Las presiones dadas se usan con la misma cantidad de etapas.
    assert by["Con interenfriamiento (2 etapas)"].inputs.p_intercool_Pa == (3.5e5,)
    assert by["Simple"].inputs.p_intercool_Pa == ()


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------


def test_pressure_ratio_sweep() -> None:
    values = default_gas_turbine_sweep_values(CENGEL_9_5, "pressure_ratio")
    pts = gas_turbine_sweep(CENGEL_9_5, "pressure_ratio", values)
    assert len(pts) == len(values)
    etas = [p.eta for p in pts]
    assert etas == sorted(etas)  # el Brayton ideal mejora con r_p
    best = max(pts, key=lambda p: p.w_net_J_per_kg)
    assert 10.0 < best.value < 20.0  # w_neto máximo en el medio
    # Con regenerador η baja con r_p y la curva se corta cuando T₄ ≤ T₂.
    regen = GAS_TURBINE_EXAMPLES[EX_9_7]
    pts = gas_turbine_sweep(regen, "pressure_ratio", values)
    assert 10 < len(pts) < len(values)
    peak = max(pts, key=lambda p: p.eta)
    assert 3.0 < peak.value < 6.0
    assert pts[-1].value < 16.0


def test_intermediate_pressure_sweep_has_its_minimum_at_the_geometric_mean() -> None:
    """Mínimo trabajo de compresión con p_x = √(p₁·p₂) (vademecum §6.3)."""
    two = replace(CENGEL_9_5, compressor_stages=2)
    pts = gas_turbine_sweep(
        two, "p_intercool", default_gas_turbine_sweep_values(two, "p_intercool")
    )
    best = min(pts, key=lambda p: p.w_compressor_J_per_kg)
    assert best.value == pytest.approx(math.sqrt(100e3 * 800e3), rel=0.05)
    two_t = replace(CENGEL_9_5, turbine_stages=2)
    pts = gas_turbine_sweep(two_t, "p_reheat", default_gas_turbine_sweep_values(two_t, "p_reheat"))
    best = max(pts, key=lambda p: p.w_turbine_J_per_kg)
    assert best.value == pytest.approx(math.sqrt(100e3 * 800e3), rel=0.05)
    with pytest.raises(ValueError, match="dos etapas de compresión"):
        default_gas_turbine_sweep_values(CENGEL_9_5, "p_intercool")


def test_many_stages_approach_the_ericsson_cycle() -> None:
    """Con todo ideal y ε = 1, η → Carnot entre T₁ y T₃ (Cengel §9-10)."""
    ideal = replace(CENGEL_9_5, regenerator=1.0)
    pts = gas_turbine_sweep(ideal, "stages", [1, 2, 4, 8])
    etas = [p.eta for p in pts]
    assert etas == sorted(etas)
    carnot = 1.0 - 300.0 / 1300.0
    assert pts[-1].eta_carnot == pytest.approx(carnot)
    assert carnot - etas[-1] < 0.02
    assert etas[1] == pytest.approx(0.696, abs=1e-3)  # Cengel 9-8 b)


@pytest.mark.parametrize("parameter", ["T_turbine_in", "regenerator", "T_amb"])
def test_other_sweeps_move_the_right_way(parameter: str) -> None:
    inputs = GAS_TURBINE_EXAMPLES[EX_MICRO]
    values = default_gas_turbine_sweep_values(inputs, parameter, n=5)  # type: ignore[arg-type]
    pts = gas_turbine_sweep(inputs, parameter, values)  # type: ignore[arg-type]
    assert len(pts) == 5
    etas = [p.eta for p in pts]
    if parameter == "T_amb":
        assert etas == sorted(etas, reverse=True)
        works = [p.w_net_J_per_kg for p in pts]
        assert works == sorted(works, reverse=True)  # en verano la turbina entrega menos
    else:
        assert etas == sorted(etas)


def test_sweep_parameters_and_unknown() -> None:
    assert set(GAS_TURBINE_SWEEP_PARAMETERS) >= {"pressure_ratio", "stages", "p_intercool"}
    with pytest.raises(ValueError, match="desconocido"):
        default_gas_turbine_sweep_values(CENGEL_9_5, "pinch")  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Notas, ejemplos, diagrama, export
# ---------------------------------------------------------------------


def test_notes() -> None:
    notes = " ".join(gas_turbine_notes(solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8A])))
    assert "sin regenerador" in notes and "aire estándar" in notes
    notes = " ".join(gas_turbine_notes(solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8B])))
    assert "Ericsson" in notes and "área infinita" in notes
    notes = " ".join(gas_turbine_notes(solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_SEQ])))
    assert "combustión secuencial" in notes and "λ =" in notes
    notes = " ".join(gas_turbine_notes(solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_AERO])))
    assert "refrigeran los álabes" in notes


@pytest.mark.parametrize("name", NAMES)
def test_every_example_solves_without_warnings(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[name])
        gas_turbine_exergy(r)
        gas_turbine_lines(r)
        gas_turbine_notes(r)
        improvement_comparison(r.inputs)
    assert GAS_TURBINE_EXAMPLE_NOTES[name]
    restored = pickle.loads(pickle.dumps(r))
    assert restored.eta_th == r.eta_th


def test_examples_like_the_plan() -> None:
    expected = {
        EX_IND: (0.3927, 608.6),
        EX_MICRO: (0.3573, 270.6),
        EX_AERO: (0.4660, 524.0),
        EX_SEQ: (0.4450, 581.4),
    }
    for name, (eta, T_exh) in expected.items():
        r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[name])
        assert r.eta_th == pytest.approx(eta, abs=2e-4), name
        assert r.T_exhaust_K - C == pytest.approx(T_exh, abs=0.2), name


def test_cycle_lines() -> None:
    r = solve_gas_turbine(FULL)
    lines = gas_turbine_lines(r)
    kinds = [line.kind for line in lines]
    assert kinds.count("isobar") == 2
    assert kinds.count("isentropic") == 4  # dos compresores y dos turbinas
    assert kinds.count("actual") == 4
    assert kinds.count("cooling") == 1
    assert kinds.count("regenerator") == 2
    assert kinds.count("composition") == 2  # cada cámara cambia la composición
    assert kinds.count("heat") == 2
    assert kinds.count("exhaust") == 1
    for line in lines:
        assert all(math.isfinite(v) for v in (*line.T_K, *line.s_J_per_kg_K, *line.p_Pa))
    # La línea de cada cámara termina en el estado de entrada a la turbina.
    heats = [line for line in lines if line.kind == "heat"]
    for line, cc in zip(heats, r.combustors, strict=True):
        assert line.T_K[-1] == pytest.approx(r.states[cc.outlet].T_K)
        assert line.s_J_per_kg_K[-1] == pytest.approx(r.states[cc.outlet].s_J_per_kg_K)
    # Con aire estándar no cambia la composición.
    air = gas_turbine_lines(solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8B]))
    assert "composition" not in [line.kind for line in air]


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_json_serializable(system: str) -> None:
    r = solve_gas_turbine(FULL)
    d = gas_turbine_to_dict(r, system)  # type: ignore[arg-type]
    json.dumps(d)
    assert d["configuracion"] == "con interenfriamiento, recalentamiento y regenerador"
    assert len(d["estados"]) == len(r.states)
    assert [c["nombre"] for c in d["camaras"]] == [
        "cámara de combustión",
        "cámara de recalentamiento",
    ]
    assert d["exergia"]["rendimiento_exergetico"] == pytest.approx(r.eta_th)


def test_configuration_names() -> None:
    assert CENGEL_9_5.configuration == "simple"
    assert replace(CENGEL_9_5, regenerator=0.8).configuration == "con regenerador"
    assert (
        replace(CENGEL_9_5, compressor_stages=2, turbine_stages=2).configuration
        == "con interenfriamiento y recalentamiento"
    )


# ---------------------------------------------------------------------
# TESPy como control
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", [n for n in GRID if n not in ("simple aire",)] + ["9-8 b con ε = 0,99"]
)
def test_tespy_gives_the_same_gas_turbine(name: str) -> None:
    inputs = (
        replace(GAS_TURBINE_EXAMPLES[EX_9_8B], regenerator=0.99)
        if name == "9-8 b con ε = 0,99"
        else GRID[name]
    )
    r = solve_gas_turbine(inputs)
    t = gas_turbine_tespy(r)
    # TESPy usa la ecuación de estado real de cada componente: difiere unas décimas de %.
    assert t.eta_th == pytest.approx(r.eta_th, abs=1e-3)
    assert t.w_net_J_per_kg == pytest.approx(r.w_net_J_per_kg, rel=3e-3)
    assert t.fuel_air_ratio == pytest.approx(r.fuel_air_ratio, rel=4e-3, abs=1e-12)
    assert t.T_exhaust_K == pytest.approx(r.T_exhaust_K, abs=2.5)
    if r.regenerator is not None:
        assert t.T_regenerator_air_K == pytest.approx(r.states[r.regenerator.air_out].T_K, abs=2.5)


def test_tespy_explains_the_ideal_regenerator() -> None:
    r = solve_gas_turbine(GAS_TURBINE_EXAMPLES[EX_9_8B])
    with pytest.raises(ValueError, match="regenerador ideal"):
        gas_turbine_tespy(r)


def test_methane_is_the_default_fuel() -> None:
    assert GasTurbineInputs(10.0, 1400.0).fuel is METHANE
