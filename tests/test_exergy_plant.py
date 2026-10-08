"""Tests de core.exergy.plant: exergía por componente y filas del diagrama de Grassmann."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import pytest

from core.cycles import combined_multi as cm
from core.cycles import gas_turbine as gtm
from core.cycles import rankine as rk
from core.cycles import refrigeration as rf
from core.cycles.hrsg_multi import hrsg_exergy
from core.exergy.chemical import species_exergy
from core.exergy.physical import Ambient
from core.exergy.plant import (
    KIND_EXPLANATIONS,
    PlantExergy,
    combined_plant_exergy,
    default_rankine_source_K,
    gas_fuel_chemical_exergy,
    gas_turbine_plant_exergy,
    grassmann_rows,
    plant_exergy_to_dict,
    plant_notes,
    rankine_exergy,
    refrigeration_exergy,
)


@cache
def _rk(name: str) -> rk.RankineResult:
    return rk.solve_rankine(rk.RANKINE_EXAMPLES[name])


@cache
def _rf(name: str) -> rf.RefrigerationResult:
    return rf.solve_refrigeration(rf.REFRIGERATION_EXAMPLES[name])


@cache
def _gt(name: str) -> gtm.GasTurbineResult:
    return gtm.solve_gas_turbine(gtm.GAS_TURBINE_EXAMPLES[name])


@cache
def _cc(name: str) -> cm.MultiCombinedResult:
    return cm.solve_combined_multi(cm.COMBINED_MULTI_EXAMPLES[name])


def _assert_closes(plant: PlantExergy, rel: float = 1e-8) -> None:
    """Ẋ_entra = Ẋ_producto + ΣẊ_D + ΣẊ_L y F = P + D + L en cada componente."""
    assert abs(plant.residual_W) <= rel * plant.fuel_W
    for c in plant.components:
        assert abs(c.residual_W) <= rel * plant.fuel_W, c.name
    assert 0.0 < plant.efficiency < 1.0


_CENGEL_10_1 = next(n for n in rk.RANKINE_EXAMPLES if n.startswith("Cengel 10-1"))


# ---------------------------------------------------------------------
# Rankine
# ---------------------------------------------------------------------


def test_cengel_10_8() -> None:
    """Cengel 10-8: fuente a 1600 K, sumidero a 290 K: caldera 1110, condensador 414, ciclo 1524."""
    plant = rankine_exergy(_rk(_CENGEL_10_1), ambient=Ambient(290.0, 100_000.0), T_source_K=1600.0)
    D = {c.kind: c.destroyed_W for c in plant.components}
    assert D["caldera"] == pytest.approx(1110.0e3, rel=2e-3)
    assert D["condensador"] == pytest.approx(414.0e3, rel=2e-3)
    assert D["turbina"] == 0.0 and D["bomba"] == 0.0
    assert plant.destroyed_W == pytest.approx(1524.0e3, rel=2e-3)
    psi4 = plant.streams[3]
    assert psi4.psi_J_per_kg == pytest.approx(449.0e3, rel=2e-3)
    assert plant.components[0].kind == "caldera"  # el orden del flujo arranca en la caldera
    _assert_closes(plant)


@pytest.mark.parametrize("name", list(rk.RANKINE_EXAMPLES))
def test_rankine_examples_close(name: str) -> None:
    plant = rankine_exergy(_rk(name))
    _assert_closes(plant)
    assert plant.product_W == pytest.approx(_rk(name).W_net_W, rel=1e-12)
    for c in plant.components:
        assert c.destroyed_W >= -1e-9 * plant.fuel_W, c.name


def test_rankine_default_sources() -> None:
    water = _rk(_CENGEL_10_1)
    assert default_rankine_source_K(water) == 1600.0
    orc = _rk(next(n for n in rk.RANKINE_EXAMPLES if "R-245fa" in n))
    assert default_rankine_source_K(orc) == pytest.approx(max(s.T_K for s in orc.states) + 50.0)


def test_rankine_warm_sink_is_a_loss() -> None:
    """Con el sumidero a T_L > T₀, la exergía del calor que sale es pérdida y no destrucción."""
    res = _rk(_CENGEL_10_1)
    cold = rankine_exergy(res, ambient=Ambient(290.0, 100_000.0))
    warm = rankine_exergy(res, ambient=Ambient(290.0, 100_000.0), T_sink_K=300.0)
    c_cold = next(c for c in cold.components if c.kind == "condensador")
    c_warm = next(c for c in warm.components if c.kind == "condensador")
    assert c_cold.loss_W == 0.0 and c_warm.loss_W > 0.0
    assert c_warm.loss_W + c_warm.destroyed_W == pytest.approx(c_cold.destroyed_W, rel=1e-9)
    _assert_closes(warm)


def test_rankine_validation_messages() -> None:
    res = _rk(_CENGEL_10_1)
    with pytest.raises(ValueError, match="más caliente que el"):
        rankine_exergy(res, T_source_K=500.0)
    with pytest.raises(ValueError, match="más frío que el ambiente"):
        rankine_exergy(res, ambient=Ambient(290.0, 1e5), T_sink_K=280.0)
    with pytest.raises(ValueError, match="condensador"):
        rankine_exergy(res, ambient=Ambient(290.0, 1e5), T_sink_K=400.0)


def test_rankine_heaters_have_fuel_and_product() -> None:
    plant = rankine_exergy(_rk(next(n for n in rk.RANKINE_EXAMPLES if n.startswith("Cengel 10-6"))))
    closed = next(c for c in plant.components if c.kind == "calentador cerrado")
    assert closed.product_W > 0.0 and closed.fuel_W > closed.product_W
    assert 0.0 < (closed.efficiency or 0.0) < 1.0
    assert all(s.m_kg_s > 0.0 for s in plant.streams)


# ---------------------------------------------------------------------
# Refrigeración
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(rf.REFRIGERATION_EXAMPLES))
def test_refrigeration_examples_close(name: str) -> None:
    res = _rf(name)
    plant = refrigeration_exergy(res)
    _assert_closes(plant)
    assert plant.fuel_W == pytest.approx(res.W_W)
    ex = res.exergy
    if ex is not None:  # coincide con el segundo principio de la Fase 3.2
        ref = dict(ex.destroyed)
        for c in plant.components:
            assert c.destroyed_W == pytest.approx(ref[c.name.lower()], abs=1e-9 * res.W_W)
        assert plant.product_W == pytest.approx(ex.W_min_W)
        assert plant.efficiency == pytest.approx(ex.eta_ex)


def test_refrigeration_default_and_bad_reservoirs() -> None:
    name = next(n for n in rf.REFRIGERATION_EXAMPLES if n.startswith("Cengel 11-1"))
    res = _rf(name)
    assert res.inputs.reservoirs is None
    d = rf.default_reservoirs(res)
    assert d.T_cold_K > res.states[res.evaporator.outlet].T_K
    assert d.T_hot_K < res.states[res.condenser.outlet].T_K
    with pytest.raises(ValueError, match="sale del condensador"):
        refrigeration_exergy(res, rf.Reservoirs(270.0, 310.0))
    with pytest.raises(ValueError, match="sale del evaporador"):
        refrigeration_exergy(res, rf.Reservoirs(230.0, 290.0))


def test_heat_pump_product_is_the_heat() -> None:
    name = next(n for n in rf.REFRIGERATION_EXAMPLES if "Bomba de calor" in n)
    plant = refrigeration_exergy(_rf(name))
    assert plant.title == "Bomba de calor"
    cond = next(c for c in plant.components if c.kind == "condensador")
    evap = next(c for c in plant.components if c.kind == "evaporador")
    assert cond.product_W > 0.0 and not cond.dissipative
    assert evap.dissipative and evap.product_W == 0.0


# ---------------------------------------------------------------------
# Turbina de gas
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(gtm.GAS_TURBINE_EXAMPLES))
def test_gas_turbine_lhv_model_matches_brayton_page(name: str) -> None:
    res = _gt(name)
    plant = gas_turbine_plant_exergy(res, "pci")
    ref = gtm.gas_turbine_exergy(res)
    _assert_closes(plant)
    assert plant.fuel_W == pytest.approx(ref.X_in_W, rel=1e-12)
    assert plant.efficiency == pytest.approx(ref.efficiency, rel=1e-12)
    lost = (ref.total("destruida") + ref.total("perdida")) * res.m_air_kg_s
    assert plant.destroyed_W + plant.loss_W == pytest.approx(lost, rel=1e-9)


@pytest.mark.parametrize("name", list(gtm.GAS_TURBINE_EXAMPLES))
def test_gas_turbine_szargut_model(name: str) -> None:
    res = _gt(name)
    plant = gas_turbine_plant_exergy(res, "szargut")
    _assert_closes(plant)
    if res.inputs.fuel is None:  # aire estándar: no hay combustible
        assert plant.fuel_model == "calor"
        return
    lhv = gas_turbine_plant_exergy(res, "pci")
    phi = species_exergy("CH4").ratio_lhv
    assert plant.fuel_model == "szargut"
    # entra el combustible con φ ≈ 1,036 y un poco de química del aire seco
    assert plant.inputs[0][1] == pytest.approx(phi * lhv.fuel_W, rel=1e-3)
    assert plant.efficiency < lhv.efficiency
    cc = next(c for c in plant.components if c.kind == "cámara de combustión")
    assert 0.2 < plant.y_D(cc) < 0.4  # ~30 % de lo que entra (Bejan et al., 1996)
    exhaust = next(c for c in plant.components if c.is_loss)
    assert exhaust.loss_W > next(c for c in lhv.components if c.is_loss).loss_W


def test_gas_fuel_chemical_exergy() -> None:
    assert gas_fuel_chemical_exergy((("methane", 1.0),)) == pytest.approx(
        species_exergy("CH4").e_J_per_kg, rel=1e-12
    )
    with pytest.raises(ValueError, match="No hay exergía química"):
        gas_fuel_chemical_exergy((("methanethiol", 1.0),))


def test_intercooler_and_regenerator_rows() -> None:
    inter = gas_turbine_plant_exergy(_gt(next(n for n in gtm.GAS_TURBINE_EXAMPLES if "Aerod" in n)))
    assert any(c.kind == "interenfriador" and c.dissipative for c in inter.components)
    regen = gas_turbine_plant_exergy(_gt(next(n for n in gtm.GAS_TURBINE_EXAMPLES if "Micro" in n)))
    kinds = [c.kind for c in regen.components]
    assert kinds.index("regenerador") > kinds.index("turbina")  # del lado de los gases
    streams = {s.label.split()[0]: s for s in regen.streams}
    assert streams["5"].m_kg_s == pytest.approx(regen_air := streams["1"].m_kg_s)
    assert streams["6"].m_kg_s > regen_air  # el escape lleva el combustible


# ---------------------------------------------------------------------
# Ciclo combinado
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(cm.COMBINED_MULTI_EXAMPLES))
@pytest.mark.parametrize("model", ["pci", "szargut"])
def test_combined_examples_close(name: str, model: str) -> None:
    res = _cc(name)
    plant = combined_plant_exergy(res, model)  # type: ignore[arg-type]
    _assert_closes(plant)
    assert plant.product_W == pytest.approx(res.W_net_W, rel=1e-9)
    if model == "pci":
        assert plant.efficiency == pytest.approx(res.eta_th, rel=1e-9)


def test_combined_bottoming_matches_fase_3_6() -> None:
    res = _cc(next(iter(cm.COMBINED_MULTI_EXAMPLES)))
    plant = combined_plant_exergy(res, "pci")
    bt = cm.bottoming_exergy(res)
    hx = hrsg_exergy(res.hrsg)
    assert sum(x for _, x in hx.gained_W) == pytest.approx(hx.X_water_W, rel=1e-12)
    steam_side = [
        c
        for c in plant.components
        if c.kind not in ("compresor", "interenfriador", "cámara de combustión", "turbina")
        or c.name.startswith("Turbina de vapor")
    ]
    D = sum(c.destroyed_W for c in steam_side if c.kind != "chimenea")
    assert D == pytest.approx(bt.X_destroyed_W + bt.X_condenser_W, rel=1e-9)
    stack = next(c for c in plant.components if c.kind == "chimenea")
    assert stack.loss_W == pytest.approx(bt.X_stack_W, rel=1e-12)


# ---------------------------------------------------------------------
# Grassmann, notas y export
# ---------------------------------------------------------------------


def test_grassmann_rows_add_up_and_merge() -> None:
    plant = combined_plant_exergy(_cc(next(iter(cm.COMBINED_MULTI_EXAMPLES))), "szargut")
    rows = grassmann_rows(plant)
    assert sum(r.destroyed_W + r.loss_W for r in rows) == pytest.approx(
        plant.destroyed_W + plant.loss_W, rel=1e-9
    )
    assert len(rows) <= 10
    assert rows[-1].name.startswith("Otros")
    assert rows[0].kind == "compresor"  # el orden del flujo


def test_grassmann_rows_skip_ideal_components() -> None:
    plant = rankine_exergy(_rk(_CENGEL_10_1))
    rows = grassmann_rows(plant)
    assert [r.kind for r in rows] == ["caldera", "condensador"]


def test_notes_explain_the_largest_destruction() -> None:
    notes = plant_notes(rankine_exergy(_rk(_CENGEL_10_1)))
    assert "Caldera" in notes[0] and "diferencia de temperatura" in notes[0]
    industrial = next(n for n in gtm.GAS_TURBINE_EXAMPLES if "Industrial" in n)
    gt_notes = plant_notes(gas_turbine_plant_exergy(_gt(industrial)))
    assert "Cámara de combustión" in gt_notes[0] and "reacción química" in gt_notes[0]
    assert any("química de los gases" in n for n in gt_notes)
    assert set(KIND_EXPLANATIONS) >= {"caldera", "turbina", "cámara de combustión", "válvula"}


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export(system: str) -> None:
    plant = refrigeration_exergy(_rf(next(iter(rf.REFRIGERATION_EXAMPLES))))
    data = plant_exergy_to_dict(plant, system)  # type: ignore[arg-type]
    assert data["planta"] == "Refrigerador"
    assert set(data["componentes"]) == {c.name for c in plant.components}
    assert len(data["corrientes"]) == len(plant.streams)


def test_y_d_definitions() -> None:
    plant = rankine_exergy(_rk(_CENGEL_10_1))
    assert sum(plant.y_star_D(c) for c in plant.components) == pytest.approx(1.0)
    assert sum(plant.y_D(c) for c in plant.components) == pytest.approx(
        plant.destroyed_W / plant.fuel_W
    )


def test_rankine_flow_rate_scales_everything() -> None:
    base = _rk(_CENGEL_10_1)
    double = replace(base, m_dot_kg_s=2.0 * base.m_dot_kg_s)
    a, b = rankine_exergy(base), rankine_exergy(double)
    assert b.fuel_W == pytest.approx(2.0 * a.fuel_W)
    assert b.efficiency == pytest.approx(a.efficiency)
