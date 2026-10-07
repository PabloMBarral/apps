"""Tests del ciclo combinado de varias presiones (:mod:`core.cycles.combined_multi`) — Fase 3.6.

Referencias: con un nivel y sin recalentamiento, el ciclo combinado de la Fase
3.4 (con el Rankine de TESPy), exacto; una red de TESPy de todo el lado
agua–vapor (gases por la HRSG con el recalentador en paralelo, domos, bombas,
turbinas con las admisiones y el desaireador), al 0,2 % (TESPy evalúa los gases
a su presión parcial); el recorrido del vapor a mano con PropsSI; y los
balances de energía y de exergía, que tienen que cerrar exactos.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI

from core.cycles.combined import COMBINED_EXAMPLES, SteamCycle, solve_combined
from core.cycles.combined_multi import (
    COMBINED_MULTI_EXAMPLE_NOTES,
    COMBINED_MULTI_EXAMPLES,
    MultiCombinedInputs,
    ReheatSpec,
    bottoming_exergy,
    combined_multi_notes,
    combined_multi_sweep,
    combined_multi_tespy,
    combined_multi_to_dict,
    configuration_comparison,
    default_multi_combined_sweep_values,
    from_combined,
    solve_combined_multi,
)
from core.cycles.hrsg_multi import PressureLevel

C = 273.15
NAMES = list(COMBINED_MULTI_EXAMPLES)
EX_3PRH, EX_2P, EX_2PRH, EX_1PRH = NAMES
THREE = COMBINED_MULTI_EXAMPLES[EX_3PRH]
TWO = COMBINED_MULTI_EXAMPLES[EX_2P]
WITH_DA = replace(THREE, steam=replace(THREE.steam, deaerator_p_Pa=1.5e5))


def _solve(inputs: MultiCombinedInputs):  # noqa: ANN202
    return solve_combined_multi(inputs)


# ---------------------------------------------------------------------
# Una presión = la Fase 3.4
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", [n for n, i in COMBINED_EXAMPLES.items() if i.T_stack_K is None])
@pytest.mark.parametrize("deaerator", [True, False])
def test_one_pressure_is_the_fase_3_4_cycle(name: str, deaerator: bool) -> None:
    inputs = COMBINED_EXAMPLES[name]
    if not deaerator:
        inputs = replace(inputs, steam=replace(inputs.steam, deaerator_p_Pa=None))
    old = solve_combined(inputs)
    new = _solve(from_combined(inputs))
    assert new.eta_th == pytest.approx(old.eta_th, rel=1e-9)
    assert new.W_steam_turbine_W == pytest.approx(old.W_steam_turbine_W, rel=1e-9)
    assert new.m_steam_kg_s == pytest.approx(old.hrsg.m_steam_kg_s, rel=1e-9)
    assert new.hrsg.T_stack_K == pytest.approx(old.hrsg.T_stack_K, rel=1e-9)
    assert new.Q_condenser_W == pytest.approx(old.Q_condenser_W, rel=1e-9)
    # misma numeración que el Rankine de Cengel: 1–4 (simple) o 1–7 (con desaireador)
    assert len(new.steam.states) == len(old.steam.states)
    for a, b in zip(new.steam.states, old.steam.states, strict=True):
        assert a.h_J_per_kg == pytest.approx(b.h_J_per_kg, rel=1e-9)


def test_from_combined_rejects_the_stack_design() -> None:
    by_stack = next(i for i in COMBINED_EXAMPLES.values() if i.T_stack_K is not None)
    with pytest.raises(ValueError, match="por pinch"):
        from_combined(by_stack)


# ---------------------------------------------------------------------
# TESPy, a mano y balances
# ---------------------------------------------------------------------

TESPY_CASES = {
    **{name: COMBINED_MULTI_EXAMPLES[name] for name in NAMES},
    "3PRH con desaireador": WITH_DA,
    "3PRH con Baumann": replace(THREE, baumann_alpha=1.0),
    "3PRH con desaireador y Baumann": replace(WITH_DA, baumann_alpha=1.0),
    "3PRH con media y baja sobrecalentadas": replace(
        THREE,
        levels=(
            THREE.levels[0],
            PressureLevel(25e5, 240.0 + C, 8.0, 5.0),
            PressureLevel(4e5, 180.0 + C, 8.0, 5.0),
        ),
    ),
    "3P sin recalentamiento": replace(THREE, reheat=None, baumann_alpha=1.0),
    "1P saturada con RH": replace(
        THREE,
        levels=(PressureLevel(40e5, None, 8.0, 5.0),),
        reheat=ReheatSpec(500.0 + C, 8e5),
    ),
}


@pytest.mark.parametrize("name", list(TESPY_CASES))
def test_tespy_gives_the_same_plant(name: str) -> None:
    result = _solve(TESPY_CASES[name])
    ref = combined_multi_tespy(result)
    flows = [lv.m_steam_kg_s for lv in result.hrsg.levels]
    assert list(ref.m_steam_kg_s) == pytest.approx(flows, rel=2.5e-3)
    assert ref.W_steam_W == pytest.approx(result.W_steam_turbine_W, rel=3e-4)
    assert ref.T_stack_K == pytest.approx(result.hrsg.T_stack_K, abs=0.15)


def test_steam_path_by_hand() -> None:
    """Las turbinas y las mezclas del ejemplo de tres presiones, a mano con PropsSI."""
    result = _solve(THREE)
    cyc = result.steam
    eta = THREE.steam.eta_turbine
    m_A, m_M, m_B = (lv.m_steam_kg_s for lv in result.hrsg.levels)

    def expand(h_in: float, p_in: float, p_out: float) -> float:
        s_in = PropsSI("S", "P", p_in, "H", h_in, "Water")
        return h_in - eta * (h_in - PropsSI("H", "P", p_out, "S", s_in, "Water"))

    h3 = PropsSI("H", "T", 565.0 + C, "P", 120e5, "Water")
    h4 = expand(h3, 120e5, 25e5)
    h5 = PropsSI("H", "P", 25e5, "Q", 1, "Water")
    h6 = (m_A * h4 + m_M * h5) / (m_A + m_M)
    h7 = PropsSI("H", "T", 565.0 + C, "P", 25e5, "Water")
    h8 = expand(h7, 25e5, 4e5)
    h9 = PropsSI("H", "P", 4e5, "Q", 1, "Water")
    h10 = ((m_A + m_M) * h8 + m_B * h9) / (m_A + m_M + m_B)
    h11 = expand(h10, 4e5, 0.06e5)
    hand = [h3, h4, h5, h6, h7, h8, h9, h10, h11]
    assert [s.h_J_per_kg for s in cyc.states[2:]] == pytest.approx(hand, rel=1e-9)
    W_T = m_A * (h3 - h4) + (m_A + m_M) * (h7 - h8) + (m_A + m_M + m_B) * (h10 - h11)
    assert cyc.W_turbines_W == pytest.approx(W_T, rel=1e-9)
    assert [t.name for t in cyc.turbines] == [
        "turbina de alta",
        "turbina de media",
        "turbina de baja",
    ]


@pytest.mark.parametrize("name", list(TESPY_CASES))
def test_energy_and_mass_balances(name: str) -> None:
    r = _solve(TESPY_CASES[name])
    cyc = r.steam
    assert sum(r.energy_balance_W.values()) == pytest.approx(r.Q_fuel_W, rel=1e-9)
    # ciclo de vapor: lo que entra en la HRSG y las bombas sale por las turbinas y el condensador
    assert r.Q_hrsg_W + r.W_pumps_W == pytest.approx(cyc.W_turbines_W + cyc.Q_condenser_W, rel=1e-9)
    assert cyc.m_condenser_kg_s + cyc.m_extraction_kg_s == pytest.approx(cyc.m_total_kg_s)
    assert cyc.m_total_kg_s == pytest.approx(r.m_steam_kg_s)
    # Kehlhofer: η_CC = η_TG + η_HRSG·η_TV·(1 − η_TG)
    kehlhofer = r.eta_gas_turbine + r.eta_hrsg * r.eta_steam * (1.0 - r.eta_gas_turbine)
    assert r.eta_th == pytest.approx(kehlhofer, rel=1e-12)
    # cada mezcla conserva masa y energía
    for a in cyc.admissions:
        st = cyc.states
        assert a.m_out_kg_s * st[a.outlet].h_J_per_kg == pytest.approx(
            a.m_turbine_kg_s * st[a.turbine].h_J_per_kg + a.m_steam_kg_s * st[a.steam].h_J_per_kg,
            rel=1e-9,
        )
    # desaireador: y·h_ext + (1 − y)·h_bomba = h_f(p_DA)
    if cyc.deaerator is not None:
        st, ext = cyc.states, cyc.extraction
        assert ext is not None
        y = cyc.y_deaerator
        assert y * st[ext].h_J_per_kg + (1 - y) * st[cyc.condensate_pump_out].h_J_per_kg == (
            pytest.approx(st[cyc.deaerator].h_J_per_kg, rel=1e-9)
        )


@pytest.mark.parametrize("name", list(TESPY_CASES))
def test_bottoming_exergy_balance_closes(name: str) -> None:
    r = _solve(TESPY_CASES[name])
    x = bottoming_exergy(r)
    assert abs(x.residual_W) < 1e-6 * x.X_gas_in_W
    assert all(d > -1e-6 for _, d in x.destroyed_W)  # el segundo principio
    assert x.X_condenser_W > 0.0
    assert 0.5 < x.efficiency < 0.8
    names = [n for n, _ in x.destroyed_W]
    assert "bombas" in names
    assert any(n.startswith("turbina") for n in names)
    assert ("desaireador" in names) == (r.steam.deaerator is not None)


# ---------------------------------------------------------------------
# Baumann, desaireador, numeración y escala
# ---------------------------------------------------------------------


def test_baumann_only_penalizes_the_wet_expansion() -> None:
    dry = _solve(THREE)
    wet = _solve(replace(THREE, baumann_alpha=1.0))
    for a, b in zip(dry.steam.turbines, wet.steam.turbines, strict=True):
        if b.wet is None:  # la expansión no llega a la campana: no cambia
            assert b.W_W == pytest.approx(a.W_W, rel=1e-12)
        else:
            assert b.W_W < a.W_W
            assert wet.steam.states[b.outlet].h_J_per_kg > dry.steam.states[a.outlet].h_J_per_kg
            w = b.wet
            assert w.eta_wet == pytest.approx(0.9 * (1 - (w.y_in + w.y_out) / 2), rel=1e-12)
            assert w.y_out == pytest.approx(1 - wet.steam.states[b.outlet].x, abs=1e-12)
            # el cruce con la curva de vapor saturado está sobre la línea seca
            h_g = PropsSI("H", "P", w.p_x_Pa, "Q", 1, "Water")
            assert w.h_x_J_per_kg == pytest.approx(h_g, rel=1e-9)
    assert [t.wet is not None for t in wet.steam.turbines] == [False, False, True]
    assert wet.eta_th < dry.eta_th
    assert wet.steam.x_exhaust > dry.steam.x_exhaust


def test_deaerator_extraction_is_on_the_expansion_line() -> None:
    for alpha in (0.0, 1.0):
        r = _solve(replace(WITH_DA, baumann_alpha=alpha))
        cyc = r.steam
        lp = cyc.turbines[-1]
        assert lp.extraction is not None and lp.m_extraction_kg_s > 0.0
        ext = cyc.states[lp.extraction]
        assert ext.P_Pa == pytest.approx(1.5e5)
        inlet = cyc.states[lp.inlet]
        if lp.wet is None or ext.P_Pa >= lp.wet.p_x_Pa:
            s_in = inlet.s_J_per_kg_K
            h_s = PropsSI("H", "P", 1.5e5, "S", s_in, "Water")
            assert ext.h_J_per_kg == pytest.approx(
                inlet.h_J_per_kg - 0.9 * (inlet.h_J_per_kg - h_s), rel=1e-9
            )
    # el desaireador calienta el agua de alimentación: la chimenea sale más caliente
    assert _solve(WITH_DA).hrsg.T_stack_K > _solve(THREE).hrsg.T_stack_K


@pytest.mark.parametrize(
    ("inputs", "count", "first_labels"),
    [
        (replace(THREE, levels=THREE.levels[:1], reheat=None), 4, ["salida del condensador"]),
        (replace(THREE, levels=THREE.levels[:1], reheat=ReheatSpec(565.0 + C, 25e5)), 6, []),
        (replace(WITH_DA, levels=WITH_DA.levels[:1], reheat=None), 7, []),
        (THREE, 11, []),
        (TWO, 7, []),
    ],
    ids=["1P (Cengel 1–4)", "1PRH (Cengel 1–6)", "1P + desaireador (1–7)", "3PRH", "2P"],
)
def test_state_numbering(inputs: MultiCombinedInputs, count: int, first_labels: list[str]) -> None:
    cyc = _solve(inputs).steam
    assert len(cyc.states) == len(cyc.labels) == count
    for label, expected in zip(cyc.labels, first_labels, strict=False):
        assert label.startswith(expected)
    assert cyc.labels[cyc.exhaust].endswith("(al condensador)")
    assert cyc.exhaust == count - 1


def test_three_pressure_reheat_states() -> None:
    cyc = _solve(THREE).steam
    assert cyc.labels[3] == "salida de la turbina de alta (recalentamiento frío)"
    assert cyc.labels[4] == "vapor de media"
    assert cyc.labels[5] == "mezcla (entrada al recalentador)"
    assert cyc.labels[6].startswith("recalentamiento caliente")
    assert (cyc.cold_reheat, cyc.reheat_inlet, cyc.hot_reheat) == (3, 5, 6)
    assert cyc.level_states == (2, 4, 8)


def test_net_power_scales_the_plant() -> None:
    r = _solve(replace(THREE, W_net_W=300e6))
    assert r.W_net_W == pytest.approx(300e6, rel=1e-9)
    assert r.eta_th == pytest.approx(_solve(THREE).eta_th, rel=1e-9)
    assert r.inputs.W_net_W == 300e6


# ---------------------------------------------------------------------
# Comparación de configuraciones
# ---------------------------------------------------------------------


def test_configuration_comparison() -> None:
    rows = configuration_comparison(THREE)
    labels = [row.label for row in rows]
    assert labels == [
        "1 presión",
        "1 presión + RH",
        "2 presiones",
        "2 presiones + RH",
        "3 presiones",
        "3 presiones + RH",
    ]
    assert all(row.result is not None and row.result_baumann is not None for row in rows)
    eta = {row.label: row.result.eta_th for row in rows}
    eta_b = {row.label: row.result_baumann.eta_th for row in rows}
    x = {row.label: row.result.steam.x_exhaust for row in rows}
    stack = {row.label: row.result.hrsg.T_stack_K for row in rows}
    # con Baumann, cada paso suma: 1P < 2P < 3P < 3P + RH
    assert eta_b["1 presión"] < eta_b["2 presiones"] < eta_b["3 presiones"]
    assert eta_b["3 presiones"] < eta_b["3 presiones + RH"]
    # más niveles enfrían la chimenea; el RH seca el vapor
    assert stack["2 presiones"] < stack["1 presión"]
    for n in ("1 presión", "2 presiones", "3 presiones"):
        assert x[f"{n} + RH"] > x[n]
    # con una sola presión el RH baja el rendimiento a η_T constante (la chimenea se calienta)
    assert eta["1 presión + RH"] < eta["1 presión"]
    assert stack["1 presión + RH"] > stack["1 presión"]
    # la última fila es el caso de los datos
    assert rows[-1].result.eta_th == pytest.approx(_solve(THREE).eta_th, rel=1e-12)


def test_comparison_with_fewer_levels() -> None:
    assert [r.label for r in configuration_comparison(TWO)] == ["1 presión", "2 presiones"]
    labels = [r.label for r in configuration_comparison(COMBINED_MULTI_EXAMPLES[EX_1PRH])]
    assert labels == ["1 presión", "1 presión + RH"]


# ---------------------------------------------------------------------
# Validaciones, notas, ejemplos, barridos y export
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(THREE, W_net_W=-1.0), "potencia neta"),
        (replace(THREE, baumann_alpha=3.0), "Baumann"),
        (replace(THREE, steam=SteamCycle(5e5, 0.9, 0.8)), "condensador"),
        (replace(THREE, steam=SteamCycle(0.06e5, 1.2, 0.8)), "rendimiento de la turbina"),
        (replace(THREE, steam=SteamCycle(0.06e5, 0.9, 0.8, 6e5)), "desaireador"),
        (replace(TWO, steam=SteamCycle(0.08e5, 0.88, 0.8, 5.5e5)), "más caliente que la salida"),
        (replace(THREE, reheat=ReheatSpec(565.0 + C, 30e5)), "presión de media"),
        (replace(TWO, reheat=ReheatSpec(565.0 + C)), "falta la presión"),
        (replace(TWO, reheat=ReheatSpec(540.0 + C, 90e5)), "entre la de baja"),
        (
            replace(TWO, reheat=ReheatSpec(540.0 + C, 20e5), steam=SteamCycle(0.08e5, 0.88, 0.8)),
            None,
        ),
    ],
    ids=[
        "potencia",
        "Baumann",
        "condensador",
        "η turbina",
        "desaireador fuera de rango",
        "desaireador caliente",
        "RH ≠ media",
        "RH sin presión",
        "RH fuera de rango",
        "válido",
    ],
)
def test_invalid_data_are_explained(inputs: MultiCombinedInputs, match: str | None) -> None:
    if match is None:
        assert _solve(inputs).eta_th > 0.5
        return
    with pytest.raises(ValueError, match=match):
        _solve(inputs)


def test_errors_name_the_part() -> None:
    bad_gt = replace(THREE, gas_turbine=replace(THREE.gas_turbine, T_turbine_in_K=500.0))
    with pytest.raises(ValueError, match="^Turbina de gas: "):
        _solve(bad_gt)
    hot_rh = replace(THREE, reheat=ReheatSpec(700.0 + C))
    with pytest.raises(ValueError, match="^Recalentador: ") as exc:
        _solve(hot_rh)
    assert "Caldera de recuperación" not in str(exc.value)
    too_high = replace(
        THREE, levels=(replace(THREE.levels[0], T_steam_K=700.0 + C), *THREE.levels[1:])
    )
    with pytest.raises(ValueError, match="^Caldera de recuperación: "):
        _solve(too_high)


def test_notes() -> None:
    def text(inputs: MultiCombinedInputs) -> str:
        return " ".join(combined_multi_notes(_solve(inputs)))

    wet = replace(THREE, reheat=None)
    assert "de humedad" in text(wet)
    assert "de humedad" not in text(THREE)
    assert "Con una sola presión el recalentamiento" in text(COMBINED_MULTI_EXAMPLES[EX_1PRH])
    assert "Regla de Baumann en la turbina de baja" in text(replace(THREE, baumann_alpha=1.0))
    assert "El desaireador calienta" in text(WITH_DA)


@pytest.mark.parametrize("name", NAMES)
def test_examples_compute_without_warnings(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = _solve(COMBINED_MULTI_EXAMPLES[name])
    assert 0.55 < r.eta_th < 0.65
    assert name in COMBINED_MULTI_EXAMPLE_NOTES
    pickle.loads(pickle.dumps(r))


@pytest.mark.parametrize(
    "parameter",
    ["p_high", "p_middle", "p_low", "T_reheat", "pinch", "pressure_ratio", "T_turbine_in"],
)
def test_sweeps(parameter: str) -> None:
    values = default_multi_combined_sweep_values(THREE, parameter, n=5)
    points = combined_multi_sweep(THREE, parameter, values)
    assert len(points) >= 4
    assert all(0.4 < p.eta < 0.7 and 0.0 < p.x_exhaust <= 1.0 for p in points)
    if parameter == "p_high":  # más presión de alta, más rendimiento (con RH sigue seco)
        etas = [p.eta for p in points]
        assert etas == sorted(etas)


def test_reheat_pressure_sweep_has_an_interior_optimum() -> None:
    inputs = COMBINED_MULTI_EXAMPLES[EX_2PRH]
    values = default_multi_combined_sweep_values(inputs, "p_reheat")
    etas = [p.eta for p in combined_multi_sweep(inputs, "p_reheat", values)]
    k = etas.index(max(etas))
    assert 0 < k < len(etas) - 1


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export(system: str) -> None:
    r = _solve(WITH_DA)
    data = combined_multi_to_dict(r, system)
    json.dumps(data)
    assert data["equipo"] == "ciclo combinado gas–vapor de 3 presiones con recalentamiento"
    assert len(data["ciclo_de_vapor"]["estados"]) == len(r.steam.states)
    assert data["resultados"]["rendimiento_ciclo_combinado"] == pytest.approx(r.eta_th)
    assert "recalentador" in data["resultados"]["exergia_ciclo_de_fondo"]["destruida"]


# ---------------------------------------------------------------------
# Procedimiento
# ---------------------------------------------------------------------

PROCEDURE_CASES = {
    "3PRH": THREE,
    "3PRH con desaireador y Baumann": replace(WITH_DA, baumann_alpha=1.0),
    "2P": TWO,
    "1PRH con desaireador": COMBINED_MULTI_EXAMPLES[EX_1PRH],
    "1P sin recalentamiento": replace(THREE, levels=THREE.levels[:1], reheat=None),
}


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("name", list(PROCEDURE_CASES))
def test_procedure(name: str, system: str) -> None:
    from core.cycles.combined_multi_procedure import combined_multi_sections

    r = _solve(PROCEDURE_CASES[name])
    sections = combined_multi_sections(r, system)
    assert [title for title, _ in sections] == [
        "Turbina de gas",
        "Caldera de recuperación",
        "Ciclo de vapor",
        "Ciclo combinado",
    ]
    steam = dict(sections)["Ciclo de vapor"]
    titles = [s.title for s in steam]
    cyc = r.steam
    for t in cyc.turbines:
        assert f"{t.name[0].upper()}{t.name[1:]} ({t.inlet + 1} → {t.outlet + 1})" in titles
    assert sum(t.startswith("Mezcla con el vapor") for t in titles) == len(cyc.admissions)
    assert ("Balance del desaireador" in titles) == (cyc.deaerator is not None)
    assert any(t.startswith("Recalentador (") for t in titles) == (cyc.hot_reheat is not None)
    baumann = any(r"\bar{y}" in tex for s in steam for tex in s.latex)
    assert baumann == any(t.wet is not None for t in cyc.turbines)
    for _, steps in sections:
        for step in steps:
            for tex in step.latex:
                assert tex.count("{") == tex.count("}"), (step.title, tex)
                assert "- -" not in tex and "+ -" not in tex
    combined = dict(sections)["Ciclo combinado"]
    assert [s.title for s in combined][-1] == "Exergía del ciclo de fondo"
