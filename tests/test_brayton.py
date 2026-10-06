"""Tests de la turbina de gas (ciclo Brayton abierto) — Fase 3.4.

Referencias: Cengel & Boles 8.ª ed., ejemplos 9-5 (Brayton ideal) y 9-6 (real),
de aire estándar; un cálculo a mano independiente (PropsSI y brentq) con
combustión de metano; ISO 6976:2016 para el PCI; y la misma turbina resuelta
con TESPy (``DiabaticCombustionChamber``) como control cruzado.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from core.cycles.brayton import (
    METHANE,
    BraytonInputs,
    Fuel,
    brayton_notes,
    brayton_tespy,
    solve_brayton,
)
from core.cycles.brayton_procedure import brayton_steps
from core.ideal_gas import AIR_TECHNICAL

C = 273.15
COOLPROP = {"N2": "Nitrogen", "O2": "Oxygen", "CO2": "CarbonDioxide", "H2O": "Water", "Ar": "Argon"}
CENGEL_9_5 = BraytonInputs(8.0, 1300.0, T_amb_K=300.0, p_amb_Pa=100e3, fuel=None)
CENGEL_9_6 = replace(CENGEL_9_5, eta_compressor=0.80, eta_turbine=0.85)
TYPICAL = BraytonInputs(15.0, 1250 + C, 0.88, 0.90, dp_combustor=0.03, m_air_kg_s=500.0)


# ---------------------------------------------------------------------
# Aire estándar: Cengel 9-5 y 9-6
# ---------------------------------------------------------------------


def test_cengel_9_5_ideal_brayton() -> None:
    r = solve_brayton(CENGEL_9_5)
    assert r.state("2").T_K == pytest.approx(540.0, abs=0.5)
    assert r.state("4").T_K == pytest.approx(770.0, abs=0.6)
    assert r.w_compressor_J_per_kg / 1e3 == pytest.approx(244.16, rel=2e-3)
    assert r.w_turbine_J_per_kg / 1e3 == pytest.approx(606.60, rel=2e-3)
    assert r.back_work_ratio == pytest.approx(0.403, abs=1e-3)
    assert r.eta_th == pytest.approx(0.426, abs=1e-3)
    assert r.fuel_air_ratio == 0.0 and r.excess_air is None and r.gas.y == r.inputs.air.y


def test_cengel_9_6_actual_gas_turbine() -> None:
    r = solve_brayton(CENGEL_9_6)
    assert r.back_work_ratio == pytest.approx(0.592, abs=1e-3)
    assert r.eta_th == pytest.approx(0.266, abs=1e-3)
    assert r.state("4").T_K == pytest.approx(853.0, abs=0.5)
    assert r.w_compressor_J_per_kg / 1e3 == pytest.approx(305.20, rel=2e-3)
    assert r.w_turbine_J_per_kg / 1e3 == pytest.approx(515.61, rel=2e-3)
    assert r.q_in_J_per_kg / 1e3 == pytest.approx(790.58, rel=2e-3)


# ---------------------------------------------------------------------
# Combustión: cálculo a mano independiente
# ---------------------------------------------------------------------


def _h(species: str, T: float) -> float:
    return PropsSI("H", "T", T, "P", 1.0, COOLPROP[species]) - PropsSI(
        "H", "T", 298.15, "P", 1.0, COOLPROP[species]
    )


def _s(species: str, T: float) -> float:
    return PropsSI("S", "T", T, "P", 1.0, COOLPROP[species])


def _mix(y: dict[str, float]) -> tuple[dict[str, float], float]:
    M = {s: PropsSI("M", COOLPROP[s]) for s in y}
    Mm = sum(y[s] * M[s] for s in y)
    return {s: y[s] * M[s] / Mm for s in y if y[s] > 0}, 8.314462618 / Mm


def _hand(inputs: BraytonInputs) -> dict[str, float]:
    """La turbina de gas a mano, sin core.ideal_gas: PropsSI, estequiometría y brentq."""
    y_air = inputs.air.mole_fractions
    w_air, R_air = _mix(y_air)

    def h_mix(w: dict[str, float], T: float) -> float:
        return sum(wi * _h(s, T) for s, wi in w.items())

    def s_mix(w: dict[str, float], T: float) -> float:
        return sum(wi * _s(s, T) for s, wi in w.items())

    def T_s(w: dict[str, float], R: float, T_in: float, p_in: float, p_out: float) -> float:
        target = s_mix(w, T_in) + R * math.log(p_out / p_in)
        lo, hi = (T_in, 1990.0) if p_out > p_in else (273.2, T_in)
        return brentq(lambda T: s_mix(w, T) - target, lo, hi, xtol=1e-10)

    p1 = inputs.p_amb_Pa
    p2 = p1 * inputs.pressure_ratio
    p3 = p2 * (1 - inputs.dp_combustor)
    T1, T3 = inputs.T_amb_K, inputs.T_turbine_in_K
    h1 = h_mix(w_air, T1)
    h2s = h_mix(w_air, T_s(w_air, R_air, T1, p1, p2))
    h2 = h1 + (h2s - h1) / inputs.eta_compressor
    # metano: PCI = (PCS − 2·L₀)/M con los valores de ISO 6976:2016 a 25 °C
    lhv = (890.58e3 - 2 * 44.013e3) / 16.04246e-3
    M_air = sum(y_air[s] * PropsSI("M", COOLPROP[s]) for s in y_air)

    def products(lam: float) -> tuple[dict[str, float], float]:
        n_air = lam * 2.0 / y_air["O2"]
        n = {s: n_air * y_air[s] for s in y_air}
        n["CO2"] += 1.0
        n["H2O"] += 2.0
        n["O2"] -= 2.0
        tot = sum(n.values())
        return {s: v / tot for s, v in n.items()}, 16.04246e-3 / (n_air * M_air)

    def resid(lam: float) -> float:
        y, f = products(lam)
        w, _ = _mix(y)
        return h2 + f * lhv - (1 + f) * h_mix(w, T3)

    lam = brentq(resid, 1.0, 50.0, xtol=1e-12)
    y_g, f = products(lam)
    w_g, R_g = _mix(y_g)
    h3 = h_mix(w_g, T3)
    h4s = h_mix(w_g, T_s(w_g, R_g, T3, p3, p1))
    h4 = h3 - inputs.eta_turbine * (h3 - h4s)
    w_net = (1 + f) * (h3 - h4) - (h2 - h1)
    return {"lam": lam, "f": f, "h2": h2, "h4": h4, "w_net": w_net, "eta": w_net / (f * lhv)}


@pytest.mark.parametrize(
    "inputs",
    [
        TYPICAL,
        replace(TYPICAL, air=AIR_TECHNICAL),
        BraytonInputs(17.0, 1400 + C, 0.90, 0.91, dp_combustor=0.03),
        BraytonInputs(8.0, 1300.0, 0.80, 0.85, T_amb_K=300.0, p_amb_Pa=100e3),
    ],
)
def test_combustion_matches_the_hand_calculation(inputs: BraytonInputs) -> None:
    r = solve_brayton(inputs)
    hand = _hand(inputs)
    assert r.excess_air == pytest.approx(hand["lam"], rel=1e-7)
    assert r.fuel_air_ratio == pytest.approx(hand["f"], rel=1e-7)
    assert r.state("2").h_J_per_kg == pytest.approx(hand["h2"], rel=1e-8)
    assert r.state("4").h_J_per_kg == pytest.approx(hand["h4"], rel=1e-7)
    assert r.w_net_J_per_kg == pytest.approx(hand["w_net"], rel=1e-7)
    assert r.eta_th == pytest.approx(hand["eta"], rel=1e-7)


def test_energy_balance_of_the_gas_turbine() -> None:
    """h₁ + f·PCI = (1 + f)·h₄ + w_neto: lo que no sale como trabajo sale con el escape."""
    r = solve_brayton(TYPICAL)
    f = r.fuel_air_ratio
    lhs = r.state("1").h_J_per_kg + f * METHANE.lhv_J_per_kg
    rhs = (1 + f) * r.state("4").h_J_per_kg + r.w_net_J_per_kg
    assert lhs == pytest.approx(rhs, rel=1e-10)
    assert r.m_gas_kg_s == pytest.approx(500.0 * (1 + f))
    assert r.W_net_W == pytest.approx(r.W_turbine_W - r.W_compressor_W)
    assert r.heat_rate_kJ_per_kWh == pytest.approx(3600 / r.eta_th)


def test_exhaust_composition_and_entropy() -> None:
    r = solve_brayton(TYPICAL)
    y = r.gas.mole_fractions
    assert 0.10 < y["O2"] < 0.15  # mucho aire de más
    assert r.excess_air == pytest.approx(2.63, abs=0.02)
    s = {st.label: st.s_J_per_kg_K for st in r.states}
    assert s["2s"] == pytest.approx(s["1"], abs=1e-6)
    assert s["4s"] == pytest.approx(s["3"], abs=1e-6)
    assert s["2"] > s["2s"] and s["4"] > s["4s"] and s["3"] > s["2"]  # irreversibilidades
    assert r.gas.p_Pa == pytest.approx(TYPICAL.p_amb_Pa)


# ---------------------------------------------------------------------
# Combustible (ISO 6976)
# ---------------------------------------------------------------------


def test_methane_heating_value() -> None:
    assert METHANE.lhv_J_per_kg / 1e6 == pytest.approx(50.027, abs=1e-3)  # ISO 6976 a 25 °C
    assert METHANE.lhv_J_per_kg / 1e3 == pytest.approx(50_050, rel=1e-3)  # Cengel A-27
    assert METHANE.hhv_J_per_kg / 1e3 == pytest.approx(55_530, rel=2e-3)  # Cengel A-27
    assert METHANE.atoms == (1.0, 4.0, 0.0, 0.0)


def test_natural_gas_mixture() -> None:
    fuel = Fuel.from_fractions({"methane": 0.9, "ethane": 0.06, "nitrogen": 0.04})
    lhv_m = 0.9 * (890.58 - 2 * 44.013) + 0.06 * (1560.69 - 3 * 44.013)  # kJ/mol
    M = 0.9 * 16.04246 + 0.06 * 30.06904 + 0.04 * 28.0134
    assert fuel.lhv_J_per_kg == pytest.approx(lhv_m / M * 1e6, rel=1e-4)
    assert fuel.atoms == pytest.approx((0.9 + 0.12, 3.6 + 0.36, 0.08, 0.0))
    r = solve_brayton(replace(TYPICAL, fuel=fuel))
    assert r.eta_th == pytest.approx(solve_brayton(TYPICAL).eta_th, abs=5e-3)


@pytest.mark.parametrize(
    ("fractions", "match"),
    [
        ({"metano": 1.0}, "no están en ISO 6976"),
        ({"methane": 0.9, "hydrogen sulfide": 0.1}, "azufre o gases nobles"),
        ({"methane": 0.9, "helium": 0.1}, "azufre o gases nobles"),
        ({"methane": -1.0}, "negativas"),
        ({"methane": 0.0}, "vacía"),
        ({"nitrogen": 1.0}, "no tiene componentes que se quemen"),
    ],
)
def test_invalid_fuels(fractions: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Fuel.from_fractions(fractions)


# ---------------------------------------------------------------------
# Validación (mensajes al alumno)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inputs", "match"),
    [
        (replace(TYPICAL, pressure_ratio=1.0), "mayor que 1"),
        (replace(TYPICAL, pressure_ratio=80.0), "demasiado alta"),
        (replace(TYPICAL, eta_compressor=0.2), "rendimiento isoentrópico del compresor"),
        (replace(TYPICAL, eta_turbine=1.1), "rendimiento isoentrópico de la turbina"),
        (replace(TYPICAL, dp_combustor=0.3), "caída de presión"),
        (replace(TYPICAL, T_amb_K=400.0), "temperatura ambiente"),
        (replace(TYPICAL, p_amb_Pa=20e3), "presión ambiente"),
        (replace(TYPICAL, T_turbine_in_K=2100.0), "supera el rango"),
        (replace(TYPICAL, T_turbine_in_K=600.0), "tiene que superar la del aire"),
        (replace(TYPICAL, m_air_kg_s=0.0), "caudal de aire"),
    ],
)
def test_invalid_inputs_are_explained(inputs: BraytonInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        solve_brayton(inputs)


# ---------------------------------------------------------------------
# TESPy como control
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs",
    [
        TYPICAL,
        BraytonInputs(17.0, 1400 + C, 0.90, 0.91, dp_combustor=0.03),
        BraytonInputs(8.0, 1300.0, 0.80, 0.85, T_amb_K=300.0, p_amb_Pa=100e3),
        CENGEL_9_6,
    ],
)
def test_tespy_gives_the_same_gas_turbine(inputs: BraytonInputs) -> None:
    r = solve_brayton(inputs)
    t = brayton_tespy(r)
    # TESPy usa la ecuación de estado real de cada componente: difiere unas décimas de %.
    assert t.eta_th == pytest.approx(r.eta_th, rel=3e-3)
    assert t.T4_K == pytest.approx(r.T_exhaust_K, abs=0.5)
    assert t.T2_K == pytest.approx(r.state("2").T_K, abs=1.0)
    if inputs.fuel is not None:
        assert t.fuel_air_ratio == pytest.approx(r.fuel_air_ratio, rel=3e-3)
        assert t.excess_air == pytest.approx(r.excess_air, rel=3e-3)


# ---------------------------------------------------------------------
# Notas y procedimiento
# ---------------------------------------------------------------------


def test_notes() -> None:
    notes = " ".join(brayton_notes(solve_brayton(CENGEL_9_6)))
    assert "59 %" in notes and "aire estándar" in notes
    notes = " ".join(brayton_notes(solve_brayton(TYPICAL)))
    assert "exceso de aire" in notes and "O₂" in notes


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
@pytest.mark.parametrize("inputs", [TYPICAL, CENGEL_9_6, CENGEL_9_5])
def test_procedure(inputs: BraytonInputs, system: str) -> None:
    r = solve_brayton(inputs)
    steps = brayton_steps(r, system)  # type: ignore[arg-type]
    titles = [s.title for s in steps]
    assert titles[0] == "Composición del aire"
    assert ("Cámara de combustión (2 → 3)" in titles) == (inputs.fuel is not None)
    assert any(t.startswith("Calentamiento") for t in titles) == (inputs.fuel is None)
    latex = [t for s in steps for t in s.latex]
    assert all(t.count("{") == t.count("}") for t in latex)
    assert not any("- -" in t or "+ -" in t for t in latex)
    if system == "Técnico":
        assert any(f"{r.eta_th * 100:.4g}" in t for t in latex)
    if inputs.is_ideal:
        assert not any(t.startswith("h_2 ") or "eta_C}" in t for t in latex)


def test_ts_lines() -> None:
    from core.cycles.brayton import brayton_ts_lines

    r = solve_brayton(TYPICAL)
    lines = brayton_ts_lines(r, T_stack_K=420.0)
    kinds = [line.kind for line in lines]
    assert kinds.count("isentropic") == 2 and kinds.count("actual") == 2
    assert kinds.count("heat") == 2  # cámara y caldera de recuperación
    s = {st.label: st.s_J_per_kg_K for st in r.states}
    vertical = [line for line in lines if line.kind == "isentropic"]
    assert all(line.s_J_per_kg_K[0] == pytest.approx(line.s_J_per_kg_K[1]) for line in vertical)
    hrsg = lines[-1]
    assert hrsg.T_K[0] == pytest.approx(420.0) and hrsg.T_K[-1] == pytest.approx(r.T_exhaust_K)
    assert hrsg.s_J_per_kg_K[-1] == pytest.approx(s["4"], rel=1e-9)
    combustion = next(line for line in lines if line.kind == "heat")
    assert combustion.s_J_per_kg_K[-1] == pytest.approx(s["3"], rel=1e-9)
    assert len(brayton_ts_lines(r)) == len(lines) - 1  # sin la caldera
