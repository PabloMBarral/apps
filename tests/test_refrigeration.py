"""Tests de la refrigeración por compresión de vapor — Fase 3.2.

Referencias: Çengel & Boles, *Termodinámica* (8.ª ed.), ejemplos 11-1 a 11-5,
y cálculos a mano con CoolProp (misma ecuación de estado; las h y s de CoolProp
difieren de las tablas del libro en una constante, por la referencia).
"""

from __future__ import annotations

import json
import math
import pickle
import warnings
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI

from core.cycles.refrigeration import (
    REFRIGERATION_EXAMPLE_NOTES,
    REFRIGERATION_EXAMPLES,
    REFRIGERATION_EXAMPLES_BY_TEMPERATURE,
    REFRIGERATION_FLUIDS,
    TON_OF_REFRIGERATION_W,
    Cascade,
    RefrigerationInputs,
    Reservoirs,
    condensation_pressure,
    condensation_temperature,
    default_flash_pressure,
    default_sweep_values,
    evaporation_pressure,
    evaporation_temperature,
    refrigeration_diagram_groups,
    refrigeration_isentropic_states,
    refrigeration_labeled_states,
    refrigeration_layout,
    refrigeration_notes,
    refrigeration_steps,
    refrigeration_sweep,
    refrigeration_to_dict,
    single_stage_equivalent,
    solve_refrigeration,
    suggested_refrigeration_inputs,
)
from core.fluids import textbook_reference_offset

C = 273.15
F = "R134a"
NAMES = list(REFRIGERATION_EXAMPLES)
EX_11_1, EX_11_2, EX_11_3, EX_11_4, EX_11_5 = NAMES[:5]


def _h(k1: str, v1: float, k2: str, v2: float, fluid: str = F) -> float:
    return PropsSI("H", k1, v1, k2, v2, fluid)


def _s(k1: str, v1: float, k2: str, v2: float, fluid: str = F) -> float:
    return PropsSI("S", k1, v1, k2, v2, fluid)


@pytest.fixture(scope="module")
def solved() -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return {
            name: solve_refrigeration(inputs) for name, inputs in REFRIGERATION_EXAMPLES.items()
        }


# ---------------------------------------------------------------------
# Ejemplos de Cengel contra el cálculo a mano y contra el libro
# ---------------------------------------------------------------------


def test_cengel_11_1_ideal_cycle(solved) -> None:
    r = solved[EX_11_1]
    h1 = _h("P", 0.14e6, "Q", 1)
    s1 = _s("P", 0.14e6, "Q", 1)
    h2 = _h("P", 0.8e6, "S", s1)
    h3 = _h("P", 0.8e6, "Q", 0)
    assert [s.h_J_per_kg for s in r.states] == pytest.approx([h1, h2, h3, h3], rel=1e-7)
    assert r.Q_cold_W == pytest.approx(0.05 * (h1 - h3), rel=1e-7)
    # Libro: Q̇_L = 7,18 kW, Ẇ = 1,81 kW, Q̇_H = 8,99 kW, COP_R = 3,97.
    assert r.Q_cold_W == pytest.approx(7.18e3, rel=2e-3)
    assert r.W_W == pytest.approx(1.81e3, rel=3e-3)
    assert r.Q_hot_W == pytest.approx(8.99e3, rel=2e-3)
    assert r.COP_R == pytest.approx(3.97, rel=2e-3)
    assert r.x_evaporator_in == pytest.approx(PropsSI("Q", "P", 0.14e6, "H", h3, F), rel=1e-6)
    assert r.layout.labeled()[0].startswith("1 (entrada al compresor")


def test_cengel_11_1_matches_the_book_tables_after_the_reference_shift(solved) -> None:
    # Tablas de Cengel (A-12 y A-13, 8.ª ed.): h₁ = 239,16; h₂ = 275,39; h₃ = 95,47 kJ/kg.
    r = solved[EX_11_1]
    dh, ds = textbook_reference_offset(F)  # type: ignore[misc]
    book = [239.16e3, 275.39e3, 95.47e3]
    assert [r.states[i].h_J_per_kg - dh for i in range(3)] == pytest.approx(book, abs=0.1e3)
    assert r.states[0].s_J_per_kg_K - ds == pytest.approx(944.56, abs=1.0)


def test_cengel_11_2_real_cycle(solved) -> None:
    r = solved[EX_11_2]
    st = r.states
    assert [s.P_Pa for s in st] == pytest.approx([0.14e6, 0.8e6, 0.72e6, 0.15e6], rel=1e-9)
    assert [st[i].T_K - C for i in range(3)] == pytest.approx([-10.0, 50.0, 26.0], abs=1e-6)
    h1 = _h("P", 0.14e6, "T", 263.15)
    h2 = _h("P", 0.8e6, "T", 323.15)
    h3 = _h("P", 0.72e6, "T", 299.15)
    assert r.Q_cold_W == pytest.approx(0.05 * (h1 - h3), rel=1e-7)
    assert r.W_W == pytest.approx(0.05 * (h2 - h1), rel=1e-7)
    # Libro: 7,93 kW, 2,02 kW, η_C = 0,939 (con sus tablas) y COP_R = 3,93.
    assert r.Q_cold_W == pytest.approx(7.93e3, rel=2e-3)
    assert r.W_W == pytest.approx(2.02e3, rel=3e-3)
    assert r.inputs.eta_compressor == pytest.approx(0.939, abs=3e-3)
    assert r.COP_R == pytest.approx(3.93, rel=2e-3)


def test_cengel_11_3_second_law(solved) -> None:
    r = solved[EX_11_3]
    m, T0, TL = 0.05, 300.15, 260.15
    s = [st.s_J_per_kg_K for st in r.states]
    QL, QH = r.Q_cold_W, r.Q_hot_W
    expected = {
        "compresor": m * T0 * (s[1] - s[0]),
        "condensador": T0 * (m * (s[2] - s[1]) + QH / T0),
        "válvula de expansión": m * T0 * (s[3] - s[2]),
        "evaporador": T0 * (m * (s[0] - s[3]) - QL / TL),
    }
    exergy = r.exergy
    assert exergy is not None
    assert dict(exergy.destroyed) == pytest.approx(expected, rel=1e-9)
    assert exergy.W_min_W == pytest.approx(QL * (T0 / TL - 1.0), rel=1e-12)
    assert exergy.W_min_W + exergy.X_destroyed_W == pytest.approx(r.W_W, rel=1e-9)
    assert exergy.eta_ex == pytest.approx(r.COP / r.COP_rev, rel=1e-12)
    # Valores del ejemplo (con las tablas del libro, a 3 cifras): COP 2,26;
    # X_dest 0,39, 0,43, 0,67 y 0,41 kW; Ẇ_mín 1,02 kW; η_II 34,8 %.
    assert r.COP_R == pytest.approx(2.26, abs=0.01)
    assert [x / 1e3 for _, x in exergy.destroyed] == pytest.approx(
        [0.394, 0.426, 0.673, 0.409], abs=3e-3
    )
    assert exergy.eta_ex == pytest.approx(0.348, abs=2e-3)


def test_cengel_11_4_cascade(solved) -> None:
    r = solved[EX_11_4]
    h1 = _h("P", 0.14e6, "Q", 1)
    h2 = _h("P", 0.32e6, "S", _s("P", 0.14e6, "Q", 1))
    h3 = _h("P", 0.32e6, "Q", 0)
    h5 = _h("P", 0.32e6, "Q", 1)
    h6 = _h("P", 0.8e6, "S", _s("P", 0.32e6, "Q", 1))
    h7 = _h("P", 0.8e6, "Q", 0)
    mB = 0.05 * (h5 - h7) / (h2 - h3)
    assert [s.h_J_per_kg for s in r.states] == pytest.approx(
        [h1, h2, h3, h3, h5, h6, h7, h7], rel=1e-7
    )
    assert r.m_evap_kg_s == pytest.approx(mB, rel=1e-7)
    assert r.mass_ratio == pytest.approx(mB / 0.05, rel=1e-7)
    # Libro: ṁ_B = 0,039 kg/s; Q̇_L = 7,17 kW; Ẇ = 1,61 kW; COP = 4,46.
    assert r.m_evap_kg_s == pytest.approx(0.0390, rel=2e-3)
    assert r.Q_cold_W == pytest.approx(7.17e3, rel=2e-3)
    assert r.W_W == pytest.approx(1.61e3, rel=6e-3)
    assert r.COP_R == pytest.approx(4.46, rel=4e-3)
    assert r.Q_cascade_W == pytest.approx(0.05 * (h5 - h7), rel=1e-7)
    assert dict(r.tespy_internal)["intercambiador de la cascada"] == pytest.approx(
        r.Q_cascade_W, rel=1e-6
    )


def test_cengel_11_5_flash_chamber(solved) -> None:
    r = solved[EX_11_5]
    h5 = _h("P", 0.8e6, "Q", 0)
    hf, hg = _h("P", 0.32e6, "Q", 0), _h("P", 0.32e6, "Q", 1)
    x6 = (h5 - hf) / (hg - hf)
    assert r.flash_fraction == pytest.approx(x6, rel=1e-7)
    h1 = _h("P", 0.14e6, "Q", 1)
    h2 = _h("P", 0.32e6, "S", _s("P", 0.14e6, "Q", 1))
    h9 = x6 * hg + (1 - x6) * h2
    h4 = _h("P", 0.8e6, "S", _s("P", 0.32e6, "H", h9))
    assert r.states[8].h_J_per_kg == pytest.approx(h9, rel=1e-7)
    assert r.states[3].h_J_per_kg == pytest.approx(h4, rel=1e-7)
    assert r.q_cold_J_per_kg == pytest.approx((1 - x6) * (h1 - hf), rel=1e-7)
    # Libro: x₆ = 0,2049; q_L = 146,3 kJ/kg; w = 32,71 kJ/kg; COP = 4,47.
    assert r.flash_fraction == pytest.approx(0.2049, abs=1e-4)
    assert r.q_cold_J_per_kg == pytest.approx(146.3e3, rel=1e-3)
    assert r.w_J_per_kg == pytest.approx(32.71e3, rel=2e-3)
    assert r.COP_R == pytest.approx(4.47, rel=2e-3)


def test_two_stage_cycles_beat_the_single_stage_of_example_11_1(solved) -> None:
    single = solved[EX_11_1].COP
    for name in (EX_11_4, EX_11_5):
        r = solved[name]
        assert r.single_stage_COP == pytest.approx(single, rel=1e-9)
        assert r.COP > single
        assert r.single_stage.pressure_ratio == pytest.approx(0.8 / 0.14, rel=1e-7)
        assert max(r.pressure_ratios) < r.single_stage.pressure_ratio


# ---------------------------------------------------------------------
# Propiedades generales
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_example_balances(solved, name: str) -> None:
    r = solved[name]
    assert r.Q_hot_W == pytest.approx(r.Q_cold_W + r.W_W, rel=1e-9)  # primer principio
    assert r.COP_B == pytest.approx(r.COP_R + 1.0, rel=1e-9)
    assert r.COP < r.COP_carnot
    tespy = dict(r.tespy_balances)
    assert tespy["evaporador"] == pytest.approx(r.Q_cold_W, rel=1e-6)
    assert tespy["condensador"] == pytest.approx(-r.Q_hot_W, rel=1e-6)
    assert sum(v for k, v in tespy.items() if k.startswith("compresor")) == pytest.approx(
        r.W_W, rel=1e-6
    )
    for comp in r.components:  # masa de cada componente
        m_in = sum(r.m_dot_kg_s[i] for i in comp.inlets)
        m_out = sum(r.m_dot_kg_s[o] for o in comp.outlets)
        assert m_in == pytest.approx(m_out, rel=1e-9)
    if r.exergy is not None:
        assert r.exergy.W_min_W + r.exergy.X_destroyed_W == pytest.approx(r.W_W, rel=1e-9)
        assert r.exergy.eta_ex == pytest.approx(r.COP / r.COP_rev, rel=1e-9)
        assert all(x > -1e-9 * r.W_W for _, x in r.exergy.destroyed)
    pickle.loads(pickle.dumps(r))


def test_heat_pump_uses_cop_b_and_its_capacity(solved) -> None:
    name = next(n for n in NAMES if "Bomba de calor" in n)
    r = solved[name]
    assert r.inputs.heat_pump
    assert r.COP == pytest.approx(r.COP_B)
    assert r.Q_hot_W == pytest.approx(10.0e3, rel=1e-9)  # la capacidad es Q̇_H
    assert r.useful_heat_W == pytest.approx(r.Q_hot_W)
    assert r.exergy is not None and r.exergy.T0_K == pytest.approx(280.15)  # T₀ = exterior
    assert r.COP_carnot == pytest.approx(r.T_cond_K / (r.T_cond_K - r.T_evap_K), rel=1e-12)


def test_mass_flow_and_capacity_give_the_same_cycle(solved) -> None:
    r = solved[EX_11_1]
    by_capacity = solve_refrigeration(replace(r.inputs, m_dot_kg_s=None, capacity_W=r.Q_cold_W))
    assert by_capacity.m_cond_kg_s == pytest.approx(0.05, rel=1e-9)
    assert by_capacity.COP == pytest.approx(r.COP, rel=1e-12)
    assert by_capacity.tons_of_refrigeration == pytest.approx(r.Q_cold_W / TON_OF_REFRIGERATION_W)
    assert TON_OF_REFRIGERATION_W == pytest.approx(211.0e3 / 60.0, rel=1e-3)


def test_suction_volume_flow(solved) -> None:
    r = solved[EX_11_1]
    v1 = 1.0 / PropsSI("D", "P", 0.14e6, "Q", 1, F)
    assert r.suction_volume_flow_m3_s == pytest.approx(0.05 * v1, rel=1e-6)


# ---------------------------------------------------------------------
# Fluidos: datos sugeridos, mezclas y compresión húmeda
# ---------------------------------------------------------------------


@pytest.mark.parametrize("fluid", REFRIGERATION_FLUIDS)
def test_suggested_inputs_solve_in_every_cycle(fluid: str) -> None:
    base = suggested_refrigeration_inputs(fluid)
    flash = replace(base, p_flash_Pa=default_flash_pressure(base.p_evap_Pa, base.p_cond_Pa))
    T_mid = 0.5 * (
        evaporation_temperature(fluid, base.p_evap_Pa)
        + condensation_temperature(fluid, base.p_cond_Pa)
    )
    cascade = replace(
        base,
        cascade=Cascade(condensation_pressure(fluid, T_mid), evaporation_pressure(fluid, T_mid)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = [solve_refrigeration(i) for i in (base, flash, cascade)]
    for r in results:
        assert 1.0 < r.COP < 10.0
        assert r.Q_cold_W == pytest.approx(10.0e3, rel=1e-9)
        for system in ("SI", "Técnico", "Inglés"):
            for step in refrigeration_steps(r, system):  # type: ignore[arg-type]
                for tex in step.latex:
                    assert tex.count("{") == tex.count("}"), (fluid, step.title, tex)


def test_pseudo_pure_r410a_uses_dew_and_bubble_points() -> None:
    T_ev, T_co = 268.15, 318.15
    p_ev = evaporation_pressure("R410A", T_ev)
    p_co = condensation_pressure("R410A", T_co)
    assert evaporation_temperature("R410A", p_ev) == pytest.approx(T_ev, abs=1e-6)
    assert condensation_temperature("R410A", p_co) == pytest.approx(T_co, abs=1e-6)
    # Rocío y burbuja a la misma presión difieren (deslizamiento ~0,1 K).
    assert evaporation_temperature("R410A", p_ev) > condensation_temperature("R410A", p_ev)
    r = solve_refrigeration(RefrigerationInputs(p_ev, p_co, fluid="R410A"))
    assert r.T_evap_K == pytest.approx(T_ev, abs=1e-6)
    assert r.states[0].T_K == pytest.approx(T_ev, abs=1e-3)  # sale vapor saturado (rocío)
    T4 = r.states[3].T_K
    assert condensation_temperature("R410A", p_ev) - 1e-6 < T4 < T_ev + 1e-6


def test_wet_compression_of_a_dry_fluid_is_noted() -> None:
    p_ev = evaporation_pressure("IsoButane", 248.15)
    p_co = condensation_pressure("IsoButane", 313.15)
    r = solve_refrigeration(RefrigerationInputs(p_ev, p_co, fluid="IsoButane"))
    assert r.wet_compressors and 0.9 < r.wet_compressors[0][1] < 1.0
    assert any("dentro de la campana" in n for n in refrigeration_notes(r, "Técnico"))
    dry = solve_refrigeration(RefrigerationInputs(p_ev, p_co, fluid="IsoButane", superheat_K=10.0))
    assert dry.wet_compressors == ()
    steps = refrigeration_steps(r, "Técnico")
    assert any(r"x_{2s}" in tex for s in steps for tex in s.latex)


def test_ammonia_notes_discharge_and_subatmospheric_evaporator() -> None:
    p_ev = evaporation_pressure("Ammonia", 233.15)  # −40 °C: 0,72 bar
    p_co = condensation_pressure("Ammonia", 308.15)
    r = solve_refrigeration(RefrigerationInputs(p_ev, p_co, 0.8, fluid="Ammonia"))
    notes = " ".join(refrigeration_notes(r, "Técnico"))
    assert "temperatura de descarga" in notes and "presión atmosférica" in notes
    assert "Cengel del" not in notes  # la nota de las tablas es solo para el R-134a


def test_r134a_reference_note() -> None:
    dh, ds = textbook_reference_offset(F)  # type: ignore[misc]
    assert dh == pytest.approx(148.14e3, abs=10.0)
    assert ds == pytest.approx(795.6, abs=0.1)
    r = solve_refrigeration(REFRIGERATION_EXAMPLES[EX_11_1])
    assert any("148.14 kJ/kg" in n for n in refrigeration_notes(r, "Técnico"))
    assert textbook_reference_offset("Ammonia") is None


def test_co2_ammonia_cascade() -> None:
    name = next(n for n in NAMES if "CO₂" in n)
    r = solve_refrigeration(REFRIGERATION_EXAMPLES[name])
    assert r.layout.fluids == ("CarbonDioxide",) * 4 + ("Ammonia",) * 4
    assert r.Q_cold_W == pytest.approx(100.0e3, rel=1e-9)
    groups = refrigeration_diagram_groups(r)
    assert [g[0] for g in groups] == ["CarbonDioxide", "Ammonia"]
    assert groups[0][1] == (0, 1, 2, 3) and groups[1][1] == (4, 5, 6, 7)
    assert r.single_stage is not None and r.single_stage.fluid == "Ammonia"
    assert r.single_stage.T_discharge_K > r.T_discharge_K  # la cascada descarga más frío
    data = refrigeration_to_dict(r, "Técnico")
    assert data["datos"]["cascada"]["fluido_ciclo_de_baja"] == "CarbonDioxide"


def test_cascade_compares_pressures_within_each_cycle() -> None:
    # CO₂ a −45 °C (8,3 bar) abajo y R-134a a 30 °C (7,7 bar) arriba: el evaporador
    # tiene más presión que el condensador, pero son fluidos y ciclos distintos.
    inputs = RefrigerationInputs(
        evaporation_pressure("CarbonDioxide", 228.15),
        condensation_pressure(F, 303.15),
        cascade=Cascade(
            condensation_pressure("CarbonDioxide", 268.15),
            evaporation_pressure(F, 263.15),
            fluid_low="CarbonDioxide",
        ),
    )
    assert inputs.p_evap_Pa > inputs.p_cond_Pa
    r = solve_refrigeration(inputs)
    assert r.COP_R > 1.0
    assert all(ratio > 1.0 for ratio in r.pressure_ratios)


def test_same_fluid_cascade_draws_a_single_diagram(solved) -> None:
    groups = refrigeration_diagram_groups(solved[EX_11_4])
    assert len(groups) == 1 and len(groups[0][2]) == 8


# ---------------------------------------------------------------------
# Numeración (sin TESPy)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cycle", "n", "segments"), [("simple", 4, 4), ("flash", 9, 10), ("cascade", 8, 8)]
)
def test_layout_numbering(cycle: str, n: int, segments: int) -> None:
    layout = refrigeration_layout(cycle)  # type: ignore[arg-type]
    assert layout.n_states == n and len(layout.labeled()) == n
    assert len(layout.segments) == segments
    # Cada estado sale de un componente y entra a otro.
    outs = sorted(o for c in layout.components for o in c.outlets)
    ins = sorted(i for c in layout.components for i in c.inlets)
    assert outs == list(range(n)) and ins == list(range(n))
    assert layout.one("evaporator").outlet == 0  # 1 = entrada al compresor (de baja)


def test_flash_numbering_follows_cengel_11_5() -> None:
    layout = refrigeration_layout("flash")
    assert layout.one("flash_tank").inlet + 1 == 6
    assert [o + 1 for o in layout.one("flash_tank").outlets] == [3, 7]
    assert layout.one("mixer").outlet + 1 == 9
    assert layout.one("condenser").inlet + 1 == 4


def test_isentropic_states_labels(solved) -> None:
    assert [lbl for lbl, _ in refrigeration_isentropic_states(solved[EX_11_5])] == ["2s", "4s"]
    assert [lbl for lbl, _ in refrigeration_isentropic_states(solved[EX_11_4])] == ["2s", "6s"]
    assert len(refrigeration_labeled_states(solved[EX_11_4])) == 8


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------

BASE = RefrigerationInputs(0.14e6, 0.8e6, m_dot_kg_s=0.05)


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(BASE, fluid="Water"), "no está disponible"),
        (replace(BASE, p_evap_Pa=0.9e6), "tiene que ser menor"),
        (replace(BASE, eta_compressor=1.2), "entre 0 y 1"),
        (replace(BASE, eta_compressor=0.0), "entre 0 y 1"),
        (replace(BASE, capacity_W=1.0e3), "uno solo"),
        (replace(BASE, m_dot_kg_s=None), "uno solo"),
        (replace(BASE, m_dot_kg_s=-1.0), "positivos"),
        (replace(BASE, superheat_K=-1.0), "no puede ser negativo"),
        (replace(BASE, dp_cond_Pa=-1.0), "no puede ser negativo"),
        (RefrigerationInputs(20e5, 80e5, fluid="CarbonDioxide"), "transcrítico"),
        (RefrigerationInputs(45e5, 60e5, fluid="R134a"), "supera la crítica"),
        (RefrigerationInputs(10.0, 8e5), "punto triple"),
        (replace(BASE, dp_cond_Pa=0.4e6, dp_evap_Pa=0.3e6), "solo puede bajar la presión"),
        (replace(BASE, subcooling_K=200.0), "se congelaría"),
        (replace(BASE, p_flash_Pa=0.1e6), "tiene que quedar entre"),
        (replace(BASE, p_flash_Pa=0.7e6, subcooling_K=6.0), "no se evaporaría nada"),
        (replace(BASE, p_flash_Pa=0.3e6, cascade=Cascade(0.32e6, 0.32e6)), "un solo tipo"),
        (replace(BASE, cascade=Cascade(0.3e6, 0.35e6)), "de lo frío a lo caliente"),
        (replace(BASE, cascade=Cascade(0.1e6, 0.32e6)), "su compresor tiene que subir"),
        (replace(BASE, cascade=Cascade(0.5e6, 0.85e6)), "su válvula tiene que bajar"),
        (replace(BASE, cascade=Cascade(0.32e6, 0.32e6, fluid_low="Water")), "no está disponible"),
        (replace(BASE, reservoirs=Reservoirs(250.0, 300.0)), "más caliente que la fuente fría"),
        (replace(BASE, reservoirs=Reservoirs(270.0, 310.0)), "más frío que la fuente caliente"),
        (replace(BASE, reservoirs=Reservoirs(300.0, 280.0)), "más fría que la caliente"),
        (replace(BASE, application="freezer"), "refrigerador o bomba de calor"),  # type: ignore[arg-type]
    ],
)
def test_invalid_inputs_are_explained(inputs: RefrigerationInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_refrigeration(inputs)


def test_reservoirs_at_the_limit_are_accepted() -> None:
    # Con el refrigerante justo a la temperatura de las fuentes (límite reversible).
    r = solve_refrigeration(BASE)
    res = Reservoirs(r.states[0].T_K, r.states[2].T_K)
    assert solve_refrigeration(replace(BASE, reservoirs=res)).exergy is not None


# ---------------------------------------------------------------------
# Barridos
# ---------------------------------------------------------------------


def test_flash_pressure_sweep_has_an_interior_optimum() -> None:
    inputs = REFRIGERATION_EXAMPLES[EX_11_5]
    points = refrigeration_sweep(inputs, "p_flash", default_sweep_values(inputs, "p_flash", n=9))
    cops = [p.COP for p in points]
    assert len(points) == 9
    best = cops.index(max(cops))
    assert 0 < best < 8
    geometric = default_flash_pressure(0.14e6, 0.8e6)
    assert points[best].value_si == pytest.approx(geometric, rel=0.3)


def test_cascade_temperature_sweep_has_an_interior_optimum() -> None:
    inputs = REFRIGERATION_EXAMPLES[EX_11_4]
    points = refrigeration_sweep(
        inputs, "T_cascade", default_sweep_values(inputs, "T_cascade", n=7)
    )
    cops = [p.COP for p in points]
    assert len(points) == 7 and 0 < cops.index(max(cops)) < 6


@pytest.mark.parametrize(
    ("parameter", "increasing"),
    [("T_evap", True), ("T_cond", False), ("eta_compressor", True), ("subcooling", True)],
)
def test_sweeps_follow_cengel(parameter: str, increasing: bool) -> None:
    inputs = REFRIGERATION_EXAMPLES[EX_11_2]
    values = default_sweep_values(inputs, parameter, n=5)  # type: ignore[arg-type]
    points = refrigeration_sweep(inputs, parameter, values)  # type: ignore[arg-type]
    assert len(points) == 5
    cops = [p.COP for p in points]
    assert cops == sorted(cops, reverse=not increasing)
    assert all(p.COP < p.COP_carnot for p in points)


def test_sweeps_without_their_device_are_empty() -> None:
    assert refrigeration_sweep(BASE, "p_flash", [0.3e6]) == []
    assert refrigeration_sweep(BASE, "T_cascade", [273.15]) == []
    assert default_sweep_values(BASE, "T_cascade") == []


def test_sweeps_ignore_the_reservoirs() -> None:
    inputs = REFRIGERATION_EXAMPLES[EX_11_3]
    values = default_sweep_values(inputs, "T_evap", n=5)
    assert len(refrigeration_sweep(inputs, "T_evap", values)) == 5


# ---------------------------------------------------------------------
# Export, procedimiento y comparación
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export_is_json_in_the_system(solved, name: str, system: str) -> None:
    r = solved[name]
    data = refrigeration_to_dict(r, system)  # type: ignore[arg-type]
    json.dumps(data, ensure_ascii=False)
    assert data["resultados"]["COP"] == pytest.approx(r.COP)
    assert len(data["estados"]) == r.layout.n_states
    assert (data["segundo_principio"] is None) == (r.inputs.reservoirs is None)
    if system == "Técnico":
        assert data["resultados"]["W"]["unidad"] == "kW"
        assert data["resultados"]["W"]["valor"] == pytest.approx(r.W_W / 1e3)


@pytest.mark.parametrize("name", NAMES)
def test_procedure_shows_the_result(solved, name: str) -> None:
    r = solved[name]
    for system in ("SI", "Técnico", "Inglés"):
        steps = refrigeration_steps(r, system)  # type: ignore[arg-type]
        latex = [tex for s in steps for tex in s.latex]
        assert all(tex.count("{") == tex.count("}") for tex in latex)
        assert not any("- -" in tex or "+ -" in tex for tex in latex)
        assert any(f"{r.COP_R:.5g}" in tex for tex in latex)
    titles = [s.title for s in refrigeration_steps(r, "Técnico")]
    assert titles[0].startswith("Estado 1")
    assert "Coeficiente de operación (COP)" in titles
    assert ("Comparación con el ciclo simple" in titles) == (r.single_stage is not None)
    assert any(t.startswith("Exergía destruida") for t in titles) == (r.exergy is not None)


def test_procedure_of_the_flash_and_cascade_cycles(solved) -> None:
    flash = refrigeration_steps(solved[EX_11_5], "Técnico")
    text = " ".join(s.text for s in flash)
    assert "Cengel A-12" in text and "Cengel A-13" in text and "x₆" in text
    assert any("x_6" in tex for s in flash for tex in s.latex)
    cascade = refrigeration_steps(solved[EX_11_4], "Técnico")
    hx = next(s for s in cascade if s.title.startswith("Intercambiador"))
    assert any(f"{solved[EX_11_4].mass_ratio:.5g}" in tex for tex in hx.latex)


def test_single_stage_equivalent() -> None:
    assert single_stage_equivalent(BASE) is None
    eq = single_stage_equivalent(REFRIGERATION_EXAMPLES[EX_11_5])
    assert eq is not None and eq.cycle == "simple" and eq.p_evap_Pa == 0.14e6


def test_example_metadata() -> None:
    assert set(REFRIGERATION_EXAMPLE_NOTES) == set(REFRIGERATION_EXAMPLES)
    assert REFRIGERATION_EXAMPLES_BY_TEMPERATURE <= set(REFRIGERATION_EXAMPLES)
    assert all(not n.startswith("Cengel") for n in REFRIGERATION_EXAMPLES_BY_TEMPERATURE)
    cycles = {i.cycle for i in REFRIGERATION_EXAMPLES.values()}
    assert cycles == {"simple", "flash", "cascade"}
    assert math.isclose(REFRIGERATION_EXAMPLES[EX_11_5].p_flash_Pa or 0.0, 0.32e6)


def test_diagram_segments_of_the_flash_cycle(solved) -> None:
    from core.diagrams import DiagramSpec, build_diagram, segment_between, segments_overlays

    r = solved[EX_11_5]
    spec = DiagramSpec(fluid=F, system="Técnico")
    diagram = build_diagram(spec)
    points = [s.to_state_point() for s in r.states]
    kinds = {
        (a + 1, b + 1): segment_between(diagram, spec, points[a], points[b])[0]
        for a, b in r.layout.segments
    }
    assert kinds[(1, 2)] == kinds[(9, 4)] == "isentropic"  # compresores ideales
    assert kinds[(5, 6)] == kinds[(7, 8)] == "isenthalpic"  # válvulas
    assert kinds[(6, 3)] == kinds[(6, 7)] == "isobaric"  # la cámara separa a p constante
    assert kinds[(2, 9)] == kinds[(3, 9)] == kinds[(4, 5)] == kinds[(8, 1)] == "isobaric"
    names = [
        o.name
        for o in segments_overlays(
            diagram, spec, [(points[a], points[b]) for a, b in r.layout.segments]
        )
    ]
    assert names == ["procesos a p o s constante", "válvulas (h constante)"]
