"""Tests del ciclo combinado gas–vapor de una presión — Fase 3.4.

Referencias: Cengel & Boles 8.ª ed., ejemplo 10-9 (ṁ_v/ṁ_g = 0,131 y η = 48,7 %);
el balance de energía de toda la planta y la relación de Kehlhofer et al.
(2009), η_CC = η_TG + η_HRSG·η_TV·(1 − η_TG), que tienen que cumplirse exactas.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import pytest

from core.cycles.combined import (
    COMBINED_EXAMPLE_NOTES,
    COMBINED_EXAMPLES,
    SH_HOT_END_K,
    CombinedInputs,
    SteamCycle,
    combined_notes,
    combined_sweep,
    combined_to_dict,
    default_sweep_values,
    solve_combined,
)
from core.cycles.combined_procedure import combined_sections

C = 273.15
NAMES = list(COMBINED_EXAMPLES)
EX_TYPICAL, EX_CENGEL, EX_MODERN = NAMES
TYPICAL = COMBINED_EXAMPLES[EX_TYPICAL]


def test_cengel_10_9() -> None:
    r = solve_combined(COMBINED_EXAMPLES[EX_CENGEL])
    assert r.steam_gas_ratio == pytest.approx(0.131, abs=5e-4)
    assert r.eta_th == pytest.approx(0.487, abs=1e-3)
    assert r.eta_gas_turbine == pytest.approx(0.266, abs=1e-3)
    assert r.steam.w_net_J_per_kg / 1e3 == pytest.approx(1331.4, rel=1e-3)
    assert r.gas_turbine.T_exhaust_K == pytest.approx(853.0, abs=0.5)
    assert r.hrsg.T_stack_K == pytest.approx(450.0)


@pytest.mark.parametrize("name", NAMES)
def test_energy_balance_and_kehlhofer(name: str) -> None:
    r = solve_combined(COMBINED_EXAMPLES[name])
    # La HRSG le da al agua lo mismo que recibe la caldera del ciclo de vapor.
    assert r.hrsg.Q_W == pytest.approx(r.steam.Q_in_W, rel=1e-8)
    # Q̇_comb = Ẇ_TG + Ẇ_TV + Q̇_cond + Q̇_chim
    assert sum(r.energy_balance_W.values()) == pytest.approx(r.Q_fuel_W, rel=1e-9)
    # Lo que se va por la chimenea: los gases a T_chim contra el aire que entra.
    gt = r.gas_turbine
    stack = gt.m_gas_kg_s * r.hrsg.h_gas_J_per_kg[-1] - gt.m_air_kg_s * gt.state("1").h_J_per_kg
    assert r.Q_stack_W == pytest.approx(stack, rel=1e-7)
    kehlhofer = r.eta_gas_turbine + r.eta_hrsg * r.eta_steam * (1 - r.eta_gas_turbine)
    assert r.eta_th == pytest.approx(kehlhofer, rel=1e-9)
    assert r.eta_gas_turbine < r.eta_th < 1.0
    # el vapor entra a la turbina tal como sale de la HRSG
    turbine_in = r.steam.states[r.steam.of_kind("boiler")[0].port("out").state]
    assert turbine_in.h_J_per_kg == pytest.approx(r.hrsg.h_steam_J_per_kg, rel=1e-9)
    assert r.steam.m_dot_kg_s == pytest.approx(r.hrsg.m_steam_kg_s)


def test_typical_numbers() -> None:
    r = solve_combined(TYPICAL)
    assert r.eta_gas_turbine == pytest.approx(0.3927, abs=5e-4)
    assert r.eta_th == pytest.approx(0.5561, abs=5e-4)
    assert r.W_gas_turbine_W / 1e6 == pytest.approx(216.2, abs=0.2)
    assert r.W_steam_turbine_W / 1e6 == pytest.approx(90.0, abs=0.2)
    assert r.hrsg.inputs.T_feedwater_K - C == pytest.approx(112.3, abs=0.2)  # desaireador


def test_power_sizing() -> None:
    r = solve_combined(replace(TYPICAL, W_net_W=400e6))
    base = solve_combined(TYPICAL)
    assert r.W_net_W == pytest.approx(400e6, rel=1e-9)
    assert r.eta_th == pytest.approx(base.eta_th, rel=1e-12)
    k = 400e6 / base.W_net_W
    assert r.gas_turbine.m_air_kg_s == pytest.approx(500.0 * k)
    assert r.hrsg.m_steam_kg_s == pytest.approx(base.hrsg.m_steam_kg_s * k)
    assert r.steam.W_net_W == pytest.approx(base.steam.W_net_W * k)
    assert r.hrsg.T_stack_K == pytest.approx(base.hrsg.T_stack_K)


def test_deaerator_and_saturated_steam() -> None:
    no_da = solve_combined(replace(TYPICAL, steam=replace(TYPICAL.steam, deaerator_p_Pa=None)))
    da = solve_combined(TYPICAL)
    # sin desaireador el agua llega fría y la chimenea queda más fría
    assert no_da.hrsg.inputs.T_feedwater_K < da.hrsg.inputs.T_feedwater_K - 60.0
    assert no_da.hrsg.T_stack_K < da.hrsg.T_stack_K
    assert no_da.hrsg.m_steam_kg_s == pytest.approx(da.hrsg.m_steam_kg_s, rel=1e-9)
    saturated = solve_combined(replace(TYPICAL, T_steam_K=None))
    assert saturated.hrsg.m_steam_kg_s > da.hrsg.m_steam_kg_s
    assert not saturated.hrsg.inputs.superheated
    assert saturated.steam.inputs.T_turbine_in_K is None


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(TYPICAL, W_net_W=-1.0), "potencia neta tiene que ser positiva"),
        (
            replace(TYPICAL, steam=replace(TYPICAL.steam, deaerator_p_Pa=70e5)),
            "desaireador",
        ),
        (
            replace(TYPICAL, gas_turbine=replace(TYPICAL.gas_turbine, pressure_ratio=0.5)),
            "Turbina de gas: La relación de presiones",
        ),
        (replace(TYPICAL, T_steam_K=650 + C), "Caldera de recuperación: El vapor"),
        (
            replace(TYPICAL, steam=SteamCycle(80e5, 0.88, 0.80)),
            "Ciclo de vapor:",
        ),
    ],
)
def test_errors_name_the_part(inputs: CombinedInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_combined(inputs)


def test_pressure_ratio_sweep_has_an_optimum_below_the_gas_turbine_one() -> None:
    values = default_sweep_values(TYPICAL, "pressure_ratio")
    points = combined_sweep(TYPICAL, "pressure_ratio", values)
    assert len(points) == len(values)
    eta = [p.eta for p in points]
    eta_gt = [p.eta_gas_turbine for p in points]
    k = eta.index(max(eta))
    assert 0 < k < len(points) - 1  # máximo interior del ciclo combinado
    assert eta_gt == sorted(eta_gt)  # la turbina de gas sola sigue mejorando
    # con relaciones altas el escape se enfría y el vapor queda 25 K por debajo
    last = points[-1]
    assert last.T_steam_K == pytest.approx(last.T_exhaust_K - SH_HOT_END_K)
    assert points[0].T_steam_K == pytest.approx(TYPICAL.T_steam_K)


def test_other_sweeps() -> None:
    pinch = combined_sweep(TYPICAL, "pinch", default_sweep_values(TYPICAL, "pinch", n=5))
    assert [p.eta for p in pinch] == sorted((p.eta for p in pinch), reverse=True)
    tit = combined_sweep(TYPICAL, "T_turbine_in", default_sweep_values(TYPICAL, "T_turbine_in", 5))
    assert [p.eta for p in tit] == sorted(p.eta for p in tit)
    p_steam = combined_sweep(TYPICAL, "p_steam", default_sweep_values(TYPICAL, "p_steam", n=5))
    assert len(p_steam) == 5
    cengel = COMBINED_EXAMPLES[EX_CENGEL]
    stack = combined_sweep(cengel, "T_stack", default_sweep_values(cengel, "T_stack", n=5))
    assert [p.eta for p in stack] == sorted((p.eta for p in stack), reverse=True)


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("name", NAMES)
def test_export_and_procedure(name: str, system: str) -> None:
    r = solve_combined(COMBINED_EXAMPLES[name])
    data = combined_to_dict(r, system)  # type: ignore[arg-type]
    json.dumps(data, ensure_ascii=False)
    assert data["resultados"]["rendimiento_ciclo_combinado"] == pytest.approx(r.eta_th)
    assert len(data["turbina_de_gas"]["estados"]) == 6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sections = combined_sections(r, system)  # type: ignore[arg-type]
    assert [title for title, _ in sections] == [
        "Turbina de gas",
        "Caldera de recuperación",
        "Ciclo de vapor",
        "Ciclo combinado",
    ]
    latex = [t for _, steps in sections for s in steps for t in s.latex]
    assert all(t.count("{") == t.count("}") for t in latex)
    assert not any("- -" in t or "+ -" in t for t in latex)
    if system == "Técnico":
        assert any(f"{r.eta_th * 100:.4g}" in t for t in latex)


def test_notes_and_metadata() -> None:
    notes = combined_notes(solve_combined(TYPICAL))
    assert notes[0].startswith("El ciclo de vapor sube el rendimiento")
    assert set(COMBINED_EXAMPLE_NOTES) == set(COMBINED_EXAMPLES)
    assert COMBINED_EXAMPLES[EX_CENGEL].gas_turbine.air_standard
    assert COMBINED_EXAMPLES[EX_CENGEL].T_stack_K == pytest.approx(450.0)
