"""Tests de core.balances.steady_flow: dispositivos, mezcla e intercambiador (Fase 10.1)."""

from __future__ import annotations

import json
import math
import pickle

import pytest

from core.balances import steady_flow as sf
from core.balances import substance as sb

C = 273.15
LBM = 0.45359237
BTU = 1055.05585262
WATER = sb.fluid_substance("Water")
R134A = sb.fluid_substance("R134a")
AIR = sb.ideal_gas_substance("air", "variable")
AIR_C = sb.ideal_gas_substance("air", "constant")
WATER_INC = sb.incompressible_substance("water")


def _device(name: str) -> sf.DeviceResult:
    return sf.solve_device(sf.DEVICE_EXAMPLES[name].inputs)


# ---------------------------------------------------------------------
# Ejemplos del libro
# ---------------------------------------------------------------------


def test_cengel_diffuser() -> None:
    """Çengel §5-4: aire a 10 °C, 80 kPa y 200 m/s por 0,4 m²: 78,8 kg/s y 303 K."""
    r = _device("Difusor de un motor a reacción (Çengel §5-4)")
    assert r.m_dot == pytest.approx(78.8, abs=0.1)
    assert r.state2.T == pytest.approx(303.0, abs=0.3)
    assert r.V2 == 0.0 and r.A2 is None and r.A1 == pytest.approx(0.4)


def test_cengel_nozzle() -> None:
    """Çengel §5-4: vapor a 1,8 MPa y 400 °C por 0,02 m²: ω₁ = 42,1 m/s y T₂ = 378,6 °C."""
    r = _device("Tobera de vapor que pierde calor (Çengel §5-4)")
    assert r.V1 == pytest.approx(42.1, abs=0.05)
    assert r.state2.T - C == pytest.approx(378.6, abs=0.1)
    assert r.Q_dot == pytest.approx(-14e3) and r.eta_s is None


def test_cengel_compressor_closes_energy_but_violates_the_second_law() -> None:
    """Çengel §5-4: 2,74 kW; pero con 16 kJ/kg de pérdida el aire no puede salir a 400 K."""
    r = _device("Compresor de aire que pierde calor (Çengel §5-4)")
    assert r.W_dot / 1e3 == pytest.approx(-2.74, abs=0.01)
    assert abs(r.energy_residual) < 1e-6
    assert r.verdict == "imposible" and r.s_gen < 0.0
    assert r.violation is not None and "menos de −169,9 °C" in r.violation
    # T* = q/Δs: el medio tendría que estar a 103 K
    assert r.q / (r.state2.s - r.state1.s) == pytest.approx(103.2, abs=0.2)


def test_cengel_steam_turbine_with_kinetic_and_potential_energy() -> None:
    """Çengel §5-4: 5 MW: w = 872,5 kJ/kg y ṁ = 5,73 kg/s."""
    r = _device("Turbina de vapor de 5 MW (Çengel §5-4)")
    assert r.w / 1e3 == pytest.approx(872.5, abs=0.5)
    assert r.m_dot == pytest.approx(5.73, abs=0.01)
    assert r.dke / 1e3 == pytest.approx(14.95, abs=0.01)
    assert r.dpe / 1e3 == pytest.approx(-0.039, abs=0.001)
    assert r.W_dot == pytest.approx(5e6, rel=1e-12)


def test_cengel_r134a_expansion_valve() -> None:
    """Çengel §5-4: líquido saturado a 0,8 MPa hasta 0,12 MPa: x₂ = 0,340, −22,32 °C."""
    r = _device("Válvula de expansión de R-134a (Çengel §5-4)")
    assert r.state2.x == pytest.approx(0.340, abs=0.001)
    assert r.state2.T - C == pytest.approx(-22.32, abs=0.03)
    assert r.state2.h == pytest.approx(r.state1.h, rel=1e-9)
    assert all(s.h == pytest.approx(r.state1.h, rel=1e-6) for s in r.path)


def test_cengel_electric_duct_heater() -> None:
    """Çengel §5-4: 15 kW y 200 W de pérdida con 150 m³/min de aire: T₂ = 21,9 °C."""
    r = _device("Calefacción eléctrica de aire en un ducto (Çengel §5-4)")
    assert r.state2.T - C == pytest.approx(21.9, abs=0.05)
    assert r.W_dot == pytest.approx(-15e3) and r.Q_dot == pytest.approx(-200.0)
    assert len(r.path) == 41 and all(s.p == pytest.approx(100e3) for s in r.path)


def test_cengel_turbine_isentropic_efficiency() -> None:
    """Çengel §7-12: vapor de 3 MPa y 400 °C a 50 kPa y 100 °C, 2 MW: 66,7 % y 3,64 kg/s."""
    r = _device("Rendimiento isoentrópico de una turbina de vapor (Çengel §7-12)")
    assert r.eta_s == pytest.approx(0.667, abs=0.001)
    assert r.m_dot == pytest.approx(3.64, abs=0.01)
    assert r.state2s is not None and r.state2s.s == pytest.approx(r.state1.s, rel=1e-9)
    assert len(r.path_s) == 41


def test_cengel_steam_throttling_entropy() -> None:
    """Çengel §7-13: vapor de 7 MPa y 450 °C a 3 MPa: s_gen = 0,369 kJ/(kg·K)."""
    r = _device("Laminación de vapor (Çengel §7-13)")
    assert r.s_gen / 1e3 == pytest.approx(0.369, abs=0.003)


def test_cengel_shower_flow_ratio() -> None:
    """Çengel §5-4: agua a 140 y 50 °F para salir a 110 °F: ṁ_caliente/ṁ_fría = 2,0."""
    r = sf.solve_mixing(sf.MIXING_EXAMPLES["Ducha: agua caliente y fría (Çengel §5-4)"].inputs)
    assert r.m_dots[0] / r.m_dots[1] == pytest.approx(2.0, abs=0.01)
    assert r.S_gen_dot > 0.0


def test_cengel_mixing_chamber_entropy() -> None:
    """Çengel §7-13: 22,7 lbm/min de vapor y Ṡ_gen = 8,65 Btu/(min·R)."""
    ex = sf.MIXING_EXAMPLES["Cámara de mezcla que pierde calor (Çengel §7-13)"]
    r = sf.solve_mixing(ex.inputs)
    assert r.m_dots[1] / LBM * 60.0 == pytest.approx(22.7, abs=0.05)
    btu_min_r = r.S_gen_dot * 60.0 / BTU * (5.0 / 9.0)
    assert btu_min_r == pytest.approx(8.65, abs=0.02)


def test_cengel_r134a_condenser() -> None:
    """Çengel §5-4: 29,1 kg/min de agua y 1218 kJ/min."""
    ex = sf.EXCHANGER_EXAMPLES["Condensador de R-134a enfriado con agua (Çengel §5-4)"]
    r = sf.solve_exchanger(ex.inputs)
    assert r.m_dots[1] * 60.0 == pytest.approx(29.1, abs=0.05)
    assert r.Q_transfer * 60.0 / 1e3 == pytest.approx(1218.0, abs=1.0)
    assert r.heat(0) == pytest.approx(-r.heat(1), rel=1e-9)
    assert r.entropy_change(1) > -r.entropy_change(0) > 0.0


# ---------------------------------------------------------------------
# Todos los ejemplos
# ---------------------------------------------------------------------

_IMPOSSIBLE = {"Compresor de aire que pierde calor (Çengel §5-4)"}


@pytest.mark.parametrize("name", list(sf.DEVICE_EXAMPLES))
def test_device_examples_close_both_balances(name: str) -> None:
    r = _device(name)
    scale = abs(r.q) + abs(r.w) + abs(r.dh) + abs(r.dke) + 1.0
    assert abs(r.energy_residual) <= 1e-9 * scale
    ds = r.state2.s - r.state1.s
    assert r.s_gen == pytest.approx(ds - r.q / r.T_b, abs=1e-9 * (abs(ds) + 1.0))
    assert (r.verdict == "imposible") == (name in _IMPOSSIBLE)
    assert r.X_dest_dot == pytest.approx(r.inputs.T0_K * r.m_dot * r.s_gen)
    data = sf.device_to_dict(r)
    json.dumps(data)
    assert sf.device_to_dict(pickle.loads(pickle.dumps(r))) == data


@pytest.mark.parametrize("name", list(sf.MIXING_EXAMPLES))
def test_mixing_examples(name: str) -> None:
    r = sf.solve_mixing(sf.MIXING_EXAMPLES[name].inputs)
    assert abs(r.energy_residual) <= 1e-9 * (abs(r.m_out * r.outlet.h) + 1.0)
    assert r.S_gen_dot > 0.0 and r.verdict == "irreversible"
    json.dumps(sf.mixing_to_dict(r))
    assert sf.mixing_to_dict(pickle.loads(pickle.dumps(r))) == sf.mixing_to_dict(r)


@pytest.mark.parametrize("name", list(sf.EXCHANGER_EXAMPLES))
def test_exchanger_examples(name: str) -> None:
    r = sf.solve_exchanger(sf.EXCHANGER_EXAMPLES[name].inputs)
    assert abs(r.energy_residual) <= 1e-9 * (abs(r.heat(0)) + 1.0)
    assert r.S_gen_dot > 0.0 and r.violation is None
    json.dumps(sf.exchanger_to_dict(r))
    assert sf.exchanger_to_dict(pickle.loads(pickle.dumps(r))) == sf.exchanger_to_dict(r)


# ---------------------------------------------------------------------
# Dispositivos
# ---------------------------------------------------------------------


def _inputs(device: sf.DeviceKind, **kw: object) -> sf.DeviceInputs:
    base: dict[str, object] = {
        "device": device,
        "substance": AIR,
        "pair1": "TP",
        "a1": 500.0,
        "b1": 5e5,
        "p2_Pa": 1e5,
        "out": "eta",
        "out_value": 0.85,
    }
    base.update(kw)
    return sf.DeviceInputs(**base)  # type: ignore[arg-type]


def test_turbine_and_compressor_with_isentropic_efficiency() -> None:
    t = sf.solve_device(_inputs("turbine"))
    assert t.state2s is not None
    assert t.state1.h - t.state2.h == pytest.approx(0.85 * (t.state1.h - t.state2s.h), rel=1e-9)
    assert t.w == pytest.approx(t.state1.h - t.state2.h, rel=1e-12)
    c = sf.solve_device(_inputs("compressor", a1=300.0, b1=1e5, p2_Pa=8e5))
    assert c.state2s is not None
    assert c.state2.h - c.state1.h == pytest.approx((c.state2s.h - c.state1.h) / 0.85)
    assert c.w < 0.0 and c.s_gen > 0.0


def test_isentropic_device_has_no_entropy_generation() -> None:
    r = sf.solve_device(_inputs("turbine", out_value=1.0))
    assert r.s_gen == 0.0 and r.verdict == "reversible"
    assert any("reversible" in n for n in r.notes)


def test_incompressible_pump() -> None:
    """w_ideal = −v·Δp (vademecum §10.4.3) y lo que falta calienta el agua."""
    r = _device("Bomba de agua (incompresible)")
    v, c = WATER_INC.material.v, WATER_INC.material.c
    assert r.w == pytest.approx(-v * (5e6 - 1e5) / 0.75, rel=1e-12)
    assert r.state2.T - r.state1.T == pytest.approx(v * 4.9e6 * (1 / 0.75 - 1) / c, rel=1e-9)


def test_nozzle_efficiency_is_the_ratio_of_kinetic_energies() -> None:
    """Vademecum §10.4.4: η_N = ω₂²/ω₂s², con el ω₁ de la entrada en los dos."""
    r = _device("Tobera de aire con rendimiento isoentrópico")
    assert r.state2s is not None
    ke2s = r.V1**2 / 2.0 + r.state1.h - r.state2s.h
    assert r.V2**2 / 2.0 == pytest.approx(0.95 * ke2s, rel=1e-9)
    assert r.V2 > r.V1 and r.A2 is not None and r.A2 < r.A1  # type: ignore[operator]


@pytest.mark.parametrize("sub", [AIR, AIR_C, WATER])
def test_nozzle_round_trip_between_exit_data(sub: sb.Substance) -> None:
    T1 = 600.0 if sub.kind == "ideal_gas" else 300.0 + C
    by_v = sf.solve_device(
        sf.DeviceInputs("nozzle", sub, "TP", T1, 1e6, p2_Pa=5e5, out="V", out_value=400.0,
                        V1=20.0)
    )  # fmt: skip
    by_t = sf.solve_device(
        sf.DeviceInputs("nozzle", sub, "TP", T1, 1e6, p2_Pa=5e5, out="T",
                        out_value=by_v.state2.T, V1=20.0)
    )  # fmt: skip
    assert by_t.V2 == pytest.approx(400.0, rel=1e-6)
    assert by_t.unknown == "V" and by_v.unknown == "h"


def test_turbine_with_work_given() -> None:
    ref = sf.solve_device(_inputs("turbine", flow_value=2.0))
    r = sf.solve_device(_inputs("turbine", out="w", out_value=ref.w, flow_value=2.0))
    assert r.state2.T == pytest.approx(ref.state2.T, rel=1e-9)
    assert r.eta_s == pytest.approx(0.85, rel=1e-6)  # adiabático: el η que resulta


def test_heater_with_heat_given_and_rates() -> None:
    by_t = sf.solve_device(
        sf.DeviceInputs("heater", WATER, "TP", 20.0 + C, 2e5, out="T", out_value=80.0 + C,
                        flow_value=0.5, T_b_K=150.0 + C)
    )  # fmt: skip
    by_q = sf.solve_device(
        sf.DeviceInputs("heater", WATER, "TP", 20.0 + C, 2e5, out="q", out_value=by_t.Q_dot,
                        flow_value=0.5, rates=True, T_b_K=150.0 + C)
    )  # fmt: skip
    assert by_q.state2.T == pytest.approx(80.0 + C, rel=1e-9)
    assert by_t.unknown == "q" and by_q.unknown == "h"


def test_valve_with_an_ideal_gas_keeps_the_temperature() -> None:
    r = sf.solve_device(sf.DeviceInputs("valve", AIR, "TP", 300.0, 1e6, p2_Pa=2e5, flow_value=1.0))
    assert r.state2.T == pytest.approx(300.0, rel=1e-9)
    assert r.s_gen == pytest.approx(AIR.gas.R * math.log(5.0), rel=1e-9)
    assert any("T₂ = T₁" in n for n in r.notes)


def test_valve_with_an_incompressible_heats_the_liquid() -> None:
    r = sf.solve_device(
        sf.DeviceInputs("valve", WATER_INC, "TP", 300.0, 1e6, p2_Pa=1e5, flow_value=1.0)
    )
    m = WATER_INC.material
    assert r.state2.T - 300.0 == pytest.approx(m.v * 9e5 / m.c, rel=1e-9)


def test_flow_from_the_power() -> None:
    r = sf.solve_device(_inputs("turbine", flow="power", flow_value=1e6))
    assert r.m_dot == pytest.approx(1e6 / r.w, rel=1e-12)
    c = sf.solve_device(
        _inputs("compressor", a1=300.0, b1=1e5, p2_Pa=8e5, flow="power", flow_value=1e5)
    )
    assert c.m_dot == pytest.approx(1e5 / -c.w, rel=1e-12) and c.W_dot == pytest.approx(-1e5)


def test_flow_from_volume_and_area() -> None:
    s1 = sb.state(AIR, "TP", 500.0, 5e5)
    by_v = sf.solve_device(_inputs("turbine", flow="Vdot", flow_value=0.3))
    assert by_v.m_dot == pytest.approx(0.3 / s1.v, rel=1e-12)
    by_a = sf.solve_device(_inputs("turbine", flow="inlet", flow_value=0.01, V1=30.0))
    assert by_a.m_dot == pytest.approx(0.01 * 30.0 / s1.v, rel=1e-12)
    with_a1 = sf.solve_device(_inputs("turbine", flow_value=2.0, A1_m2=0.05))
    assert with_a1.V1 == pytest.approx(2.0 * s1.v / 0.05, rel=1e-12)


def test_allowed_devices_and_exit_data() -> None:
    assert "pump" not in sf.allowed_devices(AIR) and "compressor" in sf.allowed_devices(AIR)
    assert "compressor" not in sf.allowed_devices(WATER_INC)
    assert sf.allowed_devices(WATER) == tuple(sf.DEVICES)
    assert "x" in sf.allowed_out("turbine", WATER) and "x" not in sf.allowed_out("turbine", AIR)
    assert sf.allowed_out("valve", WATER) == ()


# ---------------------------------------------------------------------
# Mezcla e intercambiador
# ---------------------------------------------------------------------


def test_mixing_with_both_flows_given() -> None:
    r = sf.solve_mixing(sf.MIXING_EXAMPLES["Mezcla de dos corrientes de aire"].inputs)
    a, b = r.inlets
    assert r.outlet.h == pytest.approx((2.0 * a.h + 1.0 * b.h) / 3.0, rel=1e-9)
    assert a.T < r.outlet.T < b.T


def test_exchanger_with_an_unknown_outlet() -> None:
    r = sf.solve_exchanger(sf.EXCHANGER_EXAMPLES["Enfriador de aceite con agua"].inputs)
    oil, water = sb.incompressible_substance("oil").material, WATER_INC.material
    dT_oil = 1.5 * water.c * 50.0 / (2.0 * oil.c)
    assert r.outlets[0].T == pytest.approx(150.0 + C - dT_oil, abs=0.01)


def test_exchanger_with_a_crossing_outlet_is_impossible() -> None:
    hot = sf.ExchangerStream(WATER_INC, "TP", 80.0 + C, 2e5, 1.0, "T", 40.0 + C, label="el agua")
    cold = sf.ExchangerStream(AIR, "TP", 20.0 + C, 1e5, None, "T", 90.0 + C, label="el aire")
    r = sf.solve_exchanger(sf.ExchangerInputs(hot, cold))
    assert r.verdict == "imposible"
    assert r.violation is not None and "ningún intercambiador" in r.violation


def test_exchanger_with_the_hot_stream_entering_colder_is_impossible() -> None:
    a = sf.ExchangerStream(WATER_INC, "TP", 30.0 + C, 2e5, 1.0, "T", 20.0 + C, label="el agua")
    b = sf.ExchangerStream(AIR, "TP", 50.0 + C, 1e5, None, "T", 60.0 + C, label="el aire")
    r = sf.solve_exchanger(sf.ExchangerInputs(a, b))
    assert r.violation is not None and "entra más fría" in r.violation


# ---------------------------------------------------------------------
# Mensajes al alumno
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"device": "valve", "p2_Pa": 6e5, "out": "T"}, "válvula de estrangulamiento baja"),
        ({"p2_Pa": 6e5}, "turbina el fluido se expande"),
        ({"device": "compressor"}, "la presión sube"),
        ({"device": "nozzle", "p2_Pa": 6e5}, "elegí un difusor"),
        ({"device": "diffuser", "out": "V", "out_value": 0.0, "V1": 100.0}, "elegí una tobera"),
        ({"device": "heater", "p2_Pa": 6e5, "out": "T", "out_value": 600.0}, "no sube sin"),
        (
            {"device": "nozzle", "out": "V", "out_value": 10.0, "V1": 50.0},
            "frena",
        ),
        ({"device": "diffuser", "p2_Pa": 6e5, "out": "V", "out_value": 90.0, "V1": 50.0},
         "se acelera"),
        ({"out": "T", "out_value": 520.0}, "no entrega trabajo"),
        ({"device": "compressor", "a1": 300.0, "b1": 1e5, "p2_Pa": 8e5, "out": "T",
          "out_value": 290.0}, "no consume trabajo"),
        ({"device": "nozzle", "out": "T", "out_value": 520.0, "V1": 1.0}, "no alcanza"),
        ({"device": "pump"}, "Una bomba mueve un líquido"),
        ({"device": "compressor", "substance": WATER_INC, "out": "T"}, "usá una bomba"),
        ({"out_value": 1.2}, "entre 0 y 1"),
        ({"q": -1e3}, "adiabático"),
        ({"device": "nozzle", "w": 1.0}, "no hay trabajo"),
        ({"device": "valve", "q": 1.0}, "adiabática"),
        ({"V1": -1.0}, "no puede ser negativa"),
        ({"device": "nozzle", "out": "V", "out_value": 300.0, "flow": "power"}, "La potencia"),
        ({"flow": "power", "rates": True}, "q y w van por kg"),
        ({"flow": "power", "A1_m2": 0.1}, "no el área"),
        ({"flow": "inlet"}, "ω₁ tiene que ser > 0"),
        ({"out": "x", "out_value": 0.9}, "el dato de salida es uno de"),
        ({"out_value": math.nan}, "Falta el rendimiento"),
        ({"T_b_K": 0.0}, "fuente T_b"),
        ({"T0_K": 10.0}, "ambiente T₀"),
        ({"flow_value": 0.0}, "El caudal másico ṁ tiene que ser mayor que cero"),
    ],
)  # fmt: skip
def test_device_errors(kw: dict[str, object], words: str) -> None:
    device = kw.pop("device", "turbine")
    with pytest.raises(ValueError, match=words):
        sf.solve_device(_inputs(device, **kw))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("pair", "a", "b", "device", "words"),
    [
        ("TP", 200.0 + C, 1e5, "pump", "Una bomba mueve un líquido, y el estado 1"),
        ("TP", 20.0 + C, 1e5, "compressor", "Un compresor mueve un gas o un vapor"),
    ],
)
def test_inlet_phase_errors(pair: str, a: float, b: float, device: str, words: str) -> None:
    with pytest.raises(ValueError, match=words):
        sf.solve_device(
            sf.DeviceInputs(device, WATER, pair, a, b, p2_Pa=1e6, out="eta", out_value=0.8)  # type: ignore[arg-type]
        )


def _mixing(**kw: object) -> sf.MixingInputs:
    base: dict[str, object] = {
        "substance": WATER,
        "inlet_1": sf.MixingStream("TP", 20.0 + C, 2e5, 1.0),
        "inlet_2": sf.MixingStream("TP", 80.0 + C, 2e5, None),
        "out": "T",
        "out_value": 40.0 + C,
    }
    base.update(kw)
    return sf.MixingInputs(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kw", "words"),
    [
        ({"out": "h"}, "una incógnita"),
        ({"inlet_2": sf.MixingStream("TP", 80.0 + C, 2e5, 1.0)}, "una incógnita"),
        ({"out_value": 90.0 + C}, "sale negativo"),
        ({"p_out_Pa": 3e5}, "no puede superar"),
        ({"inlet_1": sf.MixingStream("TP", 20.0 + C, 2e5, -1.0)}, "tiene que ser positivo"),
        ({"Q_dot_W": math.nan}, "tiene que ser un número"),
    ],
)
def test_mixing_errors(kw: dict[str, object], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        sf.solve_mixing(_mixing(**kw))


@pytest.mark.parametrize(
    ("a_kw", "b_kw", "words"),
    [
        ({}, {"m_dot": 1.0}, "una sola incógnita"),
        ({"m_dot": None}, {}, "una sola incógnita"),
        ({}, {"out_value": 10.0 + C}, "sale negativo"),
        ({"p_out_Pa": 9e5}, {}, "no sube"),
    ],
)
def test_exchanger_errors(a_kw: dict[str, object], b_kw: dict[str, object], words: str) -> None:
    a: dict[str, object] = {
        "substance": WATER,
        "pair": "TP",
        "a": 90.0 + C,
        "b": 3e5,
        "m_dot": 1.0,
        "out": "T",
        "out_value": 50.0 + C,
    }
    b: dict[str, object] = {"substance": WATER, "pair": "TP", "a": 15.0 + C, "b": 3e5,
                            "m_dot": None, "out": "T", "out_value": 40.0 + C}  # fmt: skip
    a.update(a_kw)
    b.update(b_kw)
    with pytest.raises(ValueError, match=words):
        sf.solve_exchanger(
            sf.ExchangerInputs(sf.ExchangerStream(**a), sf.ExchangerStream(**b))  # type: ignore[arg-type]
        )
