"""Tests de core.gases.relations (Fase 9.2): funciones características y relaciones de Maxwell."""

from __future__ import annotations

import json
import pickle

import CoolProp
import pytest

from core.gases import real as rg
from core.gases import relations as rl

C = 273.15


def _steam() -> rl.RelationsResult:
    return rl.relations(
        rl.RELATIONS_EXAMPLES["Vapor de agua a 250 °C y 300 kPa (Çengel §12-2)"].inputs
    )


# ---------------------------------------------------------------------
# Relaciones de Maxwell
# ---------------------------------------------------------------------


def test_steam_example_like_cengel() -> None:
    """Çengel §12-2: con las tablas, (∂s/∂p)_T ≈ −0,00165 y −(∂v/∂T)_p ≈ −0,00159 m³/(kg·K)."""
    r = _steam()
    g = r.relation("g")
    assert g.left.key == "ds_dp_T" and g.right.key == "dv_dT_p" and g.sign == -1
    assert g.left.finite == pytest.approx(-0.00165, abs=5e-6)
    assert -g.right.finite == pytest.approx(-0.00159, abs=5e-6)
    assert g.gap_finite == pytest.approx(0.037, abs=1e-3)
    # los estados del libro: s a 200 y 400 kPa, v a 200 y 300 °C
    lo, hi = r.paths["isothermal"]
    assert (lo.p, hi.p) == pytest.approx((200e3, 400e3))
    assert (lo.s / 1e3, hi.s / 1e3) == pytest.approx((7.7100, 7.3804), abs=2e-4)
    lo, hi = r.paths["isobaric"]
    assert (lo.T - C, hi.T - C) == pytest.approx((200.0, 300.0))
    assert (lo.v, hi.v) == pytest.approx((0.71643, 0.87535), abs=2e-5)
    # la derivada exacta es la misma de los dos lados
    assert g.left.exact == pytest.approx(-g.right.exact, rel=1e-12)
    assert g.left.exact == pytest.approx(-0.0015868, rel=1e-4)


@pytest.mark.parametrize("name", list(rl.RELATIONS_EXAMPLES))
def test_exact_sides_agree(name: str) -> None:
    r = rl.relations(rl.RELATIONS_EXAMPLES[name].inputs)
    assert [m.potential for m in r.maxwell] == ["u", "h", "f", "g"]
    for m in r.maxwell:
        assert m.gap_exact == pytest.approx(0.0, abs=1e-10), m.potential


def test_finite_differences_converge_as_delta_squared() -> None:
    r = _steam()
    (f1, g1), (f2, g2), (f4, g4) = r.convergence
    assert (f1, f2, f4) == (1.0, 0.5, 0.25)
    for a, b, c in zip(g1, g2, g4, strict=True):
        assert abs(a / b) == pytest.approx(4.0, rel=0.1)
        assert abs(b / c) == pytest.approx(4.0, rel=0.05)
    for m in r.maxwell:  # cada lado por separado también se acerca al exacto
        assert abs(m.left.error) < 0.1 and abs(m.right.error) < 0.1


def test_each_side_by_hand() -> None:
    """Las cuatro diferencias de cada camino, a mano."""
    r = rl.relations(rl.RelationsInputs("R134a", 1e6, 50.0 + C, 5.0, 0.5e5))
    T_lo, T_hi = r.paths["isentropic"]
    assert T_lo.s == pytest.approx(r.state.s) and T_hi.s == pytest.approx(r.state.s)
    assert r.relation("h").left.finite == pytest.approx((T_hi.T - T_lo.T) / 1e5)
    assert r.relation("u").left.finite == pytest.approx((T_hi.T - T_lo.T) / (T_hi.v - T_lo.v))
    v_lo, v_hi = r.paths["isochoric"]
    assert v_lo.v == pytest.approx(r.state.v) and v_hi.v == pytest.approx(r.state.v)
    assert r.relation("f").right.finite == pytest.approx((v_hi.p - v_lo.p) / 10.0)
    assert r.relation("u").right.finite == pytest.approx((v_hi.p - v_lo.p) / (v_hi.s - v_lo.s))
    b_lo, b_hi = r.paths["isobaric"]
    assert r.relation("h").right.finite == pytest.approx((b_hi.v - b_lo.v) / (b_hi.s - b_lo.s))
    t_lo, t_hi = r.paths["isothermal"]
    assert r.relation("f").left.finite == pytest.approx((t_hi.s - t_lo.s) / (t_hi.v - t_lo.v))


def test_exact_derivatives_against_coolprop() -> None:
    r = _steam()
    state = CoolProp.AbstractState("HEOS", "Water")
    state.update(CoolProp.PT_INPUTS, 300e3, 250.0 + C)
    rho = state.rhomass()
    dv_dT = -state.first_partial_deriv(CoolProp.iDmass, CoolProp.iT, CoolProp.iP) / rho**2
    assert r.relation("g").right.exact == pytest.approx(dv_dT)
    assert r.alpha == pytest.approx(dv_dT / r.state.v)  # α = (1/v)·(∂v/∂T)_p
    dp_dT = state.first_partial_deriv(CoolProp.iP, CoolProp.iT, CoolProp.iDmass)
    assert r.relation("f").right.exact == pytest.approx(dp_dT)
    assert dp_dT == pytest.approx(r.alpha / r.kappa_T)  # (∂p/∂T)_v = α/κ_T


def test_natural_variables() -> None:
    """§15.2: T = (∂u/∂s)_v = (∂h/∂s)_p, −p = (∂u/∂v)_s = (∂f/∂v)_T, v y −s."""
    r = _steam()
    assert len(r.natural) == 8
    for n in r.natural:
        assert n.value == pytest.approx(n.expected, rel=1e-6), n.key
    symbols = {n.key: n.expected_symbol for n in r.natural}
    assert symbols == {
        "du_ds_v": "T",
        "du_dv_s": "-p",
        "dh_ds_p": "T",
        "dh_dp_s": "v",
        "df_dT_v": "-s",
        "df_dv_T": "-p",
        "dg_dT_p": "-s",
        "dg_dp_T": "v",
    }


def test_potentials() -> None:
    s = _steam().state
    assert s.h == pytest.approx(s.u + s.p * s.v)  # §15.1
    assert s.f == pytest.approx(s.u - s.T * s.s)
    assert s.g == pytest.approx(s.h - s.T * s.s)
    state = CoolProp.AbstractState("HEOS", "Water")
    state.update(CoolProp.PT_INPUTS, 300e3, 250.0 + C)
    assert s.g == pytest.approx(state.gibbsmass())
    assert s.f == pytest.approx(state.helmholtzmass())


# ---------------------------------------------------------------------
# α, κ_T, Mayer y Joule–Thomson
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(rl.RELATIONS_EXAMPLES))
def test_generalized_mayer(name: str) -> None:
    """c_p − c_v = T·v·α²/κ_T (vademecum §15.5) con CoolProp."""
    r = rl.relations(rl.RELATIONS_EXAMPLES[name].inputs)
    assert r.mayer == pytest.approx(r.cp - r.cv, rel=1e-9)
    assert r.mu_JT_formula == pytest.approx(r.mu_JT, rel=1e-9)
    assert r.mu_JT_finite == pytest.approx(r.mu_JT, rel=0.05)


def test_nearly_ideal_nitrogen() -> None:
    r = rl.relations(rl.RELATIONS_EXAMPLES["Nitrógeno a 300 K y 1 bar (casi ideal)"].inputs)
    s = r.state
    assert r.alpha * s.T == pytest.approx(1.0, abs=0.005)
    assert r.kappa_T * s.p == pytest.approx(1.0, abs=0.005)
    assert (r.cp - r.cv) / r.fluid.R == pytest.approx(1.0, abs=0.01)
    assert r.mu_JT > 0.0  # el aire y el N₂ se enfrían al estrangularlos
    assert r.mu_JT * 1e5 == pytest.approx(0.22, abs=0.02)  # ≈ 0,22 K/bar
    assert any("Contra el gas ideal" in n for n in r.notes)


def test_liquid_water() -> None:
    r = rl.relations(rl.RELATIONS_EXAMPLES["Agua líquida a 100 bar y 20 °C"].inputs)
    assert r.phase == "liquid"
    assert (r.cp - r.cv) / r.cp < 0.01
    assert r.mu_JT < 0.0  # un líquido se calienta al estrangularlo
    assert any(n.startswith("Es líquido") for n in r.notes)


def test_density_maximum_of_water() -> None:
    """Cerca de 4 °C α ≈ 0: c_p = c_v y las cuatro relaciones valen ≈ 0."""
    r = rl.relations(rl.RelationsInputs("Water", 1e5, 277.13, 0.5, 0.5e5))
    assert r.degenerate
    assert r.cp - r.cv == pytest.approx(0.0, abs=1e-3)
    assert any("α ≈ 0" in n for n in r.notes)


def _mu_is_zero_on(fluid: str, out: dict[str, list[float]]) -> None:
    state = CoolProp.AbstractState("HEOS", fluid)
    for p, T in list(zip(out["p"], out["T"], strict=True))[::5]:
        state.update(CoolProp.PT_INPUTS, p, T)
        mu = state.first_partial_deriv(CoolProp.iT, CoolProp.iP, CoolProp.iHmass)
        assert abs(mu) < 1e-9  # μ_JT = 0 sobre la curva (K/Pa)


def test_inversion_curve_of_nitrogen() -> None:
    """Con la ecuación de estado de CoolProp, la máxima temperatura de inversión del N₂ es
    608 K y la nariz está a 394 bar y 277 K."""
    out = rl.inversion_curve("Nitrogen")
    assert max(out["T"]) == pytest.approx(608.0, abs=2.0)
    k = max(range(len(out["p"])), key=out["p"].__getitem__)
    assert out["p"][k] / 1e5 == pytest.approx(394.5, abs=1.0)
    assert out["T"][k] == pytest.approx(277.0, abs=5.0)
    assert out["p_edge"] == out["T_edge"] == []  # la curva vuelve a p chica
    _mu_is_zero_on("Nitrogen", out)


def test_the_nose_does_not_depend_on_the_grid() -> None:
    """La punta sale de la presión de inversión a T fija: no la corta la grilla en p."""
    coarse, fine = rl.inversion_curve("Nitrogen", 30), rl.inversion_curve("Nitrogen", 80)
    assert max(coarse["p"]) == pytest.approx(max(fine["p"]), rel=2e-3)


def test_inversion_curve_cut_by_the_equation_of_state() -> None:
    """La ecuación de estado del metano llega a 625 K: la rama alta queda más arriba y la
    zona donde se enfría se cierra por T = T_máx."""
    fl = rg.real_fluid("Methane")
    out = rl.inversion_curve("Methane")
    assert out["T_edge"] == [fl.T_max, fl.T_max]
    assert out["p_edge"][0] == out["p"][-1] > 100e5  # la curva termina arriba, no en p chica
    assert out["p_edge"][1] == pytest.approx(0.01 * fl.p_cr)
    _mu_is_zero_on("Methane", out)


def test_isenthalps() -> None:
    out = rl.isenthalps("Nitrogen", (200.0, 400.0), 1e5, 300e5)
    state = CoolProp.AbstractState("HEOS", "Nitrogen")
    state.update(CoolProp.PT_INPUTS, 1e5, 200.0)
    h = state.hmass()
    ps, Ts = out[200.0]
    assert Ts[0] == pytest.approx(200.0)
    state.update(CoolProp.PT_INPUTS, ps[-1], Ts[-1])
    assert state.hmass() == pytest.approx(h, rel=1e-6)
    # a 200 K, bajar la presión enfría (dentro de la curva de inversión): T sube con p
    assert Ts[10] > Ts[0]


# ---------------------------------------------------------------------
# Clapeyron
# ---------------------------------------------------------------------


def test_clapeyron_r134a_like_cengel() -> None:
    """Çengel §12-3: h_fg del R-134a a 20 °C con p_sat a 16 y 24 °C, 17,70 kPa/K y
    182,40 kJ/kg (la tabla, 182,27)."""
    r = rl.clapeyron(rl.CLAPEYRON_EXAMPLES["R-134a a 20 °C (Çengel §12-3)"].inputs)
    assert (r.p_minus / 1e3, r.p_plus / 1e3) == pytest.approx((504.58, 646.18), abs=0.5)
    assert r.dpdT_finite / 1e3 == pytest.approx(17.70, abs=0.02)
    assert r.h_fg_clapeyron / 1e3 == pytest.approx(182.40, abs=0.1)
    assert r.h_fg / 1e3 == pytest.approx(182.27, abs=0.05)
    assert r.dpdT_exact == pytest.approx(r.h_fg / (r.T * r.v_fg), rel=1e-6)  # Clapeyron exacta
    assert r.s_fg == pytest.approx(r.h_fg / r.T, rel=1e-9)


def test_clapeyron_water_at_100_C() -> None:
    r = rl.clapeyron(rl.ClapeyronInputs("Water", 100.0 + C, 0.5))
    assert r.h_fg / 1e3 == pytest.approx(2256.4, abs=0.1)
    assert r.h_fg_clapeyron == pytest.approx(r.h_fg, rel=2e-4)
    # Clausius–Clapeyron se equivoca un 2 %: el vapor a 1 atm ya no es tan ideal
    assert r.h_fg_cc / r.h_fg - 1.0 == pytest.approx(0.02, abs=0.005)


def test_clausius_clapeyron_extrapolation_worsens_near_the_critical_point() -> None:
    far = rl.clapeyron(rl.CLAPEYRON_EXAMPLES["Agua a 100 °C"].inputs)
    near = rl.clapeyron(
        rl.CLAPEYRON_EXAMPLES["Nitrógeno a 100 K (cerca del punto crítico, 126 K)"].inputs
    )
    assert far.p_sat2 is not None and far.p_sat2_cc is not None
    assert near.p_sat2 is not None and near.p_sat2_cc is not None
    assert abs(far.p_sat2_cc / far.p_sat2 - 1) < 0.01
    assert abs(near.p_sat2_cc / near.p_sat2 - 1) > 0.1
    assert abs(near.h_fg_cc / near.h_fg - 1) > 0.2


# ---------------------------------------------------------------------
# Errores, export
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "inputs,match",
    [
        (rl.RelationsInputs("Water", -1e5, 400.0), "positiva"),
        (rl.RelationsInputs("Water", 1e5, 400.0, 0.0), "ΔT tiene que ser positivo"),
        (rl.RelationsInputs("Water", 1e5, 400.0, 10.0, 2e5), "menor que la presión"),
        (rl.RelationsInputs("Water", 1e5, 400.0, 250.0), "demasiado grande"),
        (rl.RelationsInputs("Water", 1e5, 380.0, 10.0), "campana"),  # 107 °C − 10 °C: líquido
        (rl.RelationsInputs("Water", 1e5, 293.15, 5.0, 0.5e5), "bar por kelvin"),
        (rl.RelationsInputs("Water", 101_417.98, 373.15), "justo la de saturación"),
    ],
)
def test_relations_errors(inputs: rl.RelationsInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rl.relations(inputs)


@pytest.mark.parametrize(
    "inputs,match",
    [
        (rl.ClapeyronInputs("Water", 700.0), "campana"),
        (rl.ClapeyronInputs("Water", 400.0, -1.0), "positivo"),
        (rl.ClapeyronInputs("Water", 400.0, 2.0, 1000.0), "T₂"),
    ],
)
def test_clapeyron_errors(inputs: rl.ClapeyronInputs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        rl.clapeyron(inputs)


def test_examples_and_export() -> None:
    for ex in rl.RELATIONS_EXAMPLES.values():
        r = rl.relations(ex.inputs)
        d = rl.relations_to_dict(r)
        json.dumps(d)
        assert set(d["maxwell"]) == {"u", "h", "f", "g"}
        assert pickle.loads(pickle.dumps(r)) == r
    for ex in rl.CLAPEYRON_EXAMPLES.values():
        c = rl.clapeyron(ex.inputs)
        json.dumps(rl.clapeyron_to_dict(c))
        assert pickle.loads(pickle.dumps(c)) == c


# ---------------------------------------------------------------------
# Joule–Thomson en un estado y Clausius–Clapeyron para el gráfico
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(rl.JT_EXAMPLES))
def test_joule_thomson_three_ways(name: str) -> None:
    r = rl.joule_thomson(rl.JT_EXAMPLES[name].inputs)
    assert r.mu_formula == pytest.approx(r.mu, rel=1e-9)
    assert r.mu_finite == pytest.approx(r.mu, rel=0.01)
    lo, hi = r.isenthalpic
    assert lo.h == pytest.approx(r.state.h) and hi.h == pytest.approx(r.state.h)
    json.dumps(rl.joule_thomson_to_dict(r))
    assert pickle.loads(pickle.dumps(r)) == r


def test_nitrogen_cools_and_hydrogen_heats() -> None:
    n2 = rl.joule_thomson(rl.JT_EXAMPLES["Nitrógeno a 300 K y 50 bar (se enfría)"].inputs)
    h2 = rl.joule_thomson(rl.JT_EXAMPLES["Hidrógeno a 300 K y 50 bar (se calienta)"].inputs)
    assert n2.mu > 0 and h2.mu < 0
    assert n2.T_inversion[0] < 300.0 < n2.T_inversion[-1]  # adentro de la curva
    assert h2.T_inversion[-1] < 300.0  # el H₂ invierte a ~200 K
    assert any("se enfría" in n for n in n2.notes)
    assert any("se calienta" in n for n in h2.notes)
    state = CoolProp.AbstractState("HEOS", "Nitrogen")
    for T in n2.T_inversion:  # μ_JT = 0 en cada temperatura de inversión
        state.update(CoolProp.PT_INPUTS, 50e5, T)
        assert abs(state.first_partial_deriv(CoolProp.iT, CoolProp.iP, CoolProp.iHmass)) < 1e-9


@pytest.mark.parametrize(
    ("args", "low", "high", "words"),
    [
        (("Methane", 50e5, 300.0), True, False, "temperatura máxima de la ecuación de estado"),
        (("Nitrogen", 5e5, 300.0), False, True, "abajo de ella el nitrógeno se enfría"),
        (("Methane", 5e5, 300.0), False, False, "se enfría al estrangularlo a cualquier"),
        (("Nitrogen", 450e5, 300.0), False, False, "se calienta al estrangularlo a cualquier"),
    ],
)
def test_one_or_no_inversion_temperature(
    args: tuple[str, float, float], low: bool, high: bool, words: str
) -> None:
    """Con una sola temperatura de inversión en el rango (o ninguna), la nota dice de qué
    lado se enfría."""
    r = rl.joule_thomson(rl.JouleThomsonInputs(*args))
    assert (r.T_inversion_low is not None, r.T_inversion_high is not None) == (low, high)
    assert any(words in n for n in r.notes)
    assert len(r.T_inversion) == low + high


def test_joule_thomson_errors() -> None:
    with pytest.raises(ValueError, match="menor que la presión"):
        rl.joule_thomson(rl.JouleThomsonInputs("Nitrogen", 1e5, 300.0, 2e5))
    with pytest.raises(ValueError, match="fuera del rango"):
        rl.joule_thomson(rl.JouleThomsonInputs("Nitrogen", 1e5, 5000.0))
    with pytest.raises(ValueError, match="campana"):
        # vapor apenas sobrecalentado: a p + Δp y h constante ya es mezcla
        rl.joule_thomson(rl.JouleThomsonInputs("Water", 1e5, 375.0, 0.5e5))


def test_clausius_curve() -> None:
    c = rl.clapeyron(rl.CLAPEYRON_EXAMPLES["Agua a 100 °C"].inputs)
    out = rl.clausius_curve(c)
    k = min(range(len(out["T"])), key=lambda i: abs(out["T"][i] - c.T))
    assert out["cc"][k] == pytest.approx(out["real"][k], rel=0.05)  # coinciden cerca de T₁
    assert out["cc"][-1] > out["real"][-1]  # y se separan cerca del punto crítico
    assert out["real"] == sorted(out["real"])
