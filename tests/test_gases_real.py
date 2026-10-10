"""Tests de core.gases.real (Fase 9.2): el factor de compresibilidad con seis modelos."""

from __future__ import annotations

import json
import math
import pickle
import time

import CoolProp
import numpy as np
import pytest

from core.gases import real as rg

C = 273.15
PSI = 6894.757293168361
FT3_LB = 0.3048**3 / 0.45359237

# Tabla del vademecum §7.5 (CoolProp): T_cr [K], p_cr [MPa] y Z_cr
VADEMECUM_TABLE = {
    "Air": (132.53, 3.786, 0.290),
    "Argon": (150.69, 4.863, 0.290),
    "CarbonDioxide": (304.13, 7.377, 0.275),
    "Helium": (5.20, 0.228, 0.304),
    "Hydrogen": (33.14, 1.296, 0.303),
    "Methane": (190.56, 4.599, 0.286),
    "Nitrogen": (126.19, 3.396, 0.289),
    "Oxygen": (154.60, 5.046, 0.288),
    "Water": (647.10, 22.064, 0.229),
}


def _solve(key: str, pair: str, a: float, b: float) -> rg.CompressibilityResult:
    return rg.compressibility(rg.CompressibilityInputs(key, pair, a, b))  # type: ignore[arg-type]


def _unknown(r: rg.CompressibilityResult, kind: str) -> float:
    m = r.model(kind)  # type: ignore[arg-type]
    value = {"v": m.v_m3_per_kg, "p": m.p_Pa, "T": m.T_K}[r.unknown]
    assert value is not None, m.note
    return value


# ---------------------------------------------------------------------
# Los fluidos
# ---------------------------------------------------------------------


@pytest.mark.parametrize("key", rg.VADEMECUM_FLUIDS)
def test_critical_constants_match_the_vademecum_table(key: str) -> None:
    fl = rg.real_fluid(key)
    T_cr, p_cr, Z_cr = VADEMECUM_TABLE[key]
    assert fl.T_cr == pytest.approx(T_cr, abs=0.006)
    assert fl.p_cr / 1e6 == pytest.approx(p_cr, abs=6e-4)
    if key == "Oxygen":
        # El punto crítico de la ecuación de estado (CoolProp) está a 427 kg/m³: Z_cr = 0,294.
        # El 0,288 de la tabla sale con la densidad crítica medida (436 kg/m³).
        assert fl.Z_cr == pytest.approx(0.294, abs=6e-4)
        state = CoolProp.AbstractState("HEOS", key)
        assert fl.p_cr / (state.rhomass_reducing() * fl.R * fl.T_cr) == pytest.approx(
            Z_cr, abs=6e-4
        )
    else:
        assert fl.Z_cr == pytest.approx(Z_cr, abs=6e-4)


def test_every_fluid_has_its_constants() -> None:
    for fl in rg.REAL_FLUIDS.values():
        assert fl.M > 0.0 and fl.R == pytest.approx(rg.R_U / fl.M)
        assert 0.22 < fl.Z_cr < 0.31
        assert -0.4 < fl.omega < 0.35
        assert fl.T_min < fl.T_cr < fl.T_max
        assert fl.p_max > fl.p_cr
        assert fl.label.startswith(fl.name)
    assert rg.real_fluid("Water").label == "Agua (H₂O)"
    assert rg.real_fluid("Air").label == "Aire (pseudo puro)"


def test_acentric_factor_is_pitzers_definition() -> None:
    """ω = −log₁₀(p_sat/p_cr)|_{T_R = 0,7} − 1 con la presión de vapor de CoolProp."""
    for key in ("Argon", "Methane", "Nitrogen", "Water", "R134a", "Ammonia", "n-Propane"):
        fl = rg.real_fluid(key)
        state = CoolProp.AbstractState("HEOS", key)
        state.update(CoolProp.QT_INPUTS, 0.0, 0.7 * fl.T_cr)
        assert -math.log10(state.p() / fl.p_cr) - 1.0 == pytest.approx(fl.omega, abs=2e-4)


def test_unknown_fluid() -> None:
    with pytest.raises(ValueError, match="no está en la lista"):
        rg.real_fluid("Kryptonite")


# ---------------------------------------------------------------------
# Lee y Kesler
# ---------------------------------------------------------------------


def test_lee_kesler_critical_point() -> None:
    """Cada fluido de Lee y Kesler tiene su punto crítico en T_R = p_R = 1."""
    for fluid in ("simple", "reference"):
        v = np.linspace(0.15, 0.6, 20001)
        p = rg.lk_z_of_vr(1.0, v, fluid) / v  # p_R = Z·T_R/v_R
        dp = np.gradient(p, v)
        k = int(np.argmax(dp))
        assert dp[k] == pytest.approx(0.0, abs=1e-4)  # la inflexión horizontal
        assert p[k] == pytest.approx(1.0, abs=1e-4)
        assert p[k] * v[k] == pytest.approx(rg._LK_ZC[fluid], abs=1e-3)
    assert rg._LK_ZC["simple"] == pytest.approx(0.2901, abs=5e-4)  # Lee y Kesler (1975)


@pytest.mark.parametrize("omega", [0.0, 0.1, 0.25, 0.3978])
def test_lee_kesler_vapor_pressure_follows_pitzer(omega: float) -> None:
    assert rg.lee_kesler_psat_reduced(0.7, omega) == pytest.approx(10 ** (-1.0 - omega), rel=1e-3)
    assert rg.lee_kesler_psat_reduced(1.0, omega) == pytest.approx(1.0, abs=1e-3)


def test_lee_kesler_mixes_the_two_fluids() -> None:
    lk = rg.lee_kesler(1.5, 2.0, 0.2)
    assert lk.Z == pytest.approx(lk.Z0 + 0.2 / rg.LK_OMEGA_REF * (lk.Z_ref - lk.Z0))
    assert lk.Z0 == pytest.approx(lk.pr * lk.vr0 / lk.Tr)
    assert lk.Z_ref == pytest.approx(lk.pr * lk.vr_ref / lk.Tr)
    # con ω = ω_r, Z es el del fluido de referencia; con ω = 0, el del simple
    assert rg.lee_kesler(1.5, 2.0, rg.LK_OMEGA_REF).Z == pytest.approx(lk.Z_ref)
    assert rg.lee_kesler(1.5, 2.0, 0.0).Z == pytest.approx(lk.Z0)


def test_lee_kesler_coefficients() -> None:
    B, C_, D = rg.lk_coefficients(2.0, "simple")
    assert B == pytest.approx(0.1181193 - 0.265728 / 2 - 0.154790 / 4 - 0.030323 / 8)
    assert C_ == pytest.approx(0.0236744 - 0.0186984 / 2)
    assert D == pytest.approx(0.155488e-4 + 0.623689e-4 / 2)


def test_lee_kesler_against_argon() -> None:
    """El argón es casi un fluido simple (ω = −0,002): Lee–Kesler lo sigue al 1,1 %."""
    worst = 0.0
    fl = rg.real_fluid("Argon")
    for Tr in (1.1, 1.3, 1.6, 2.0, 3.0):
        for pr in (0.5, 1.0, 2.0, 4.0, 6.0):
            r = _solve("Argon", "pT", pr * fl.p_cr, Tr * fl.T_cr)
            worst = max(worst, abs(r.model("lk").error or 0.0))
    assert worst < 0.011


def test_lee_kesler_phase_choice() -> None:
    """A T_R = 0,9 y p_R = 0,5 el fluido simple tiene tres raíces: la mayor es vapor y la
    menor, líquido."""
    vap = rg.lee_kesler(0.9, 0.5, 0.0, "vapor")
    liq = rg.lee_kesler(0.9, 0.5, 0.0, "liquid")
    assert vap.Z0 > rg._LK_ZC["simple"] > liq.Z0
    assert liq.vr0 < vap.vr0
    # a T_R = 0,95 y p_R = 0,5 el líquido ya no existe ni como metaestable
    with pytest.raises(ValueError, match="todavía es vapor"):
        rg.lee_kesler(0.95, 0.5, 0.0, "liquid")


# ---------------------------------------------------------------------
# Van der Waals y Peng–Robinson
# ---------------------------------------------------------------------


def test_van_der_waals_constants_of_nitrogen_like_cengel() -> None:
    """Çengel §3-8: a = 0,175 m⁶·kPa/kg² y b = 0,00138 m³/kg para el N₂."""
    a, b = rg.cubic_constants(rg.real_fluid("Nitrogen"), "vdw", 175.0)
    assert a / 1e3 == pytest.approx(0.175, rel=0.01)
    assert b == pytest.approx(0.00138, rel=0.01)
    # con las a y b redondeadas del libro, la ecuación da sus 9471 kPa
    R = 0.2968
    assert R * 175.0 / (0.00375 - 0.00138) - 0.175 / 0.00375**2 == pytest.approx(9471.0, abs=1.0)


def test_van_der_waals_critical_point() -> None:
    """En (T_cr, p_cr): A = 27/64, B = 1/8 y la cúbica tiene una raíz triple Z = 3/8 (§8.3)."""
    fl = rg.real_fluid("Nitrogen")
    A, B = rg.cubic_ab(fl, "vdw", fl.T_cr, fl.p_cr)
    assert (A, B) == pytest.approx((27 / 64, 1 / 8))
    c2, c1, c0 = rg.cubic_coefficients("vdw", A, B)
    # (Z − 3/8)³ = Z³ − 9/8·Z² + 27/64·Z − 27/512
    assert (c2, c1, c0) == pytest.approx((-9 / 8, 27 / 64, -27 / 512))
    a, b = rg.cubic_constants(fl, "vdw", fl.T_cr)
    assert a / (27 * b**2) == pytest.approx(fl.p_cr)
    assert 8 * a / (27 * fl.R * b) == pytest.approx(fl.T_cr)


def test_peng_robinson_critical_z() -> None:
    """Con los Ω exactos de Peng y Robinson (1976) la raíz triple en el punto crítico es 0,3074."""
    roots = np.roots(
        [1.0, *rg.cubic_coefficients("pr", 0.45723553, 0.07779607)]
    )  # Ω_a y Ω_b sin redondear
    assert np.allclose(roots.real, rg.CUBIC_ZC["pr"], atol=2e-3)
    assert rg.pr_kappa(0.0) == pytest.approx(0.37464)
    assert rg.pr_kappa(0.344) == pytest.approx(0.37464 + 1.54226 * 0.344 - 0.26992 * 0.344**2)


@pytest.mark.parametrize("Tr,vR", [(0.9, 0.6), (0.9, 2.0), (1.0, 1.0), (1.5, 0.8), (3.0, 5.0)])
def test_van_der_waals_reduced_form(Tr: float, vR: float) -> None:
    """(p_R + 3/v_R²)(3v_R − 1) = 8·T_R con v_R = v/(3b) (vademecum §8.4)."""
    fl = rg.real_fluid("CarbonDioxide")
    _, b = rg.cubic_constants(fl, "vdw", fl.T_cr)
    p = rg.cubic_p(fl, "vdw", Tr * fl.T_cr, vR * 3 * b)
    pR = p / fl.p_cr
    assert (pR + 3 / vR**2) * (3 * vR - 1) == pytest.approx(8 * Tr)


def test_cubic_roots_and_p_agree() -> None:
    fl = rg.real_fluid("R134a")
    T, p = 323.15, 1e6
    for model in ("vdw", "pr"):
        A, B = rg.cubic_ab(fl, model, T, p)
        roots = rg.cubic_z_roots(model, A, B)
        assert len(roots) == 3 and all(z > B for z in roots)
        for z in roots:  # cada raíz da la misma presión
            assert rg.cubic_p(fl, model, T, z * fl.R * T / p) == pytest.approx(p, rel=1e-8)


# ---------------------------------------------------------------------
# El factor de compresibilidad: los ejemplos de Çengel
# ---------------------------------------------------------------------


def test_refrigerant_134a_at_1_MPa_and_50_C() -> None:
    """Çengel §3-7: tablas 0,021796 m³/kg, gas ideal 0,026325 (20,8 %), carta Z = 0,84."""
    r = _solve("R134a", "pT", 1e6, 50.0 + C)
    assert r.unknown == "v" and r.phase == "vapor"
    assert r.v_m3_per_kg == pytest.approx(0.021796, rel=2e-5)
    assert _unknown(r, "ideal") == pytest.approx(0.026333, rel=1e-4)  # el libro: R = 0,0815
    assert r.model("ideal").error == pytest.approx(0.208, abs=1e-3)
    assert _unknown(r, "vdw") == pytest.approx(0.02329, rel=1e-3)
    assert _unknown(r, "pr") == pytest.approx(0.02196, rel=1e-3)
    assert _unknown(r, "lk0") == pytest.approx(0.02257, rel=1e-3)
    assert _unknown(r, "lk") == pytest.approx(0.02187, rel=1e-3)
    assert r.p_R == pytest.approx(0.246, abs=1e-3) and r.T_R == pytest.approx(0.864, abs=1e-3)
    assert len(r.model("vdw").roots) == 3  # vapor, líquido y la del medio
    assert "tres raíces" in r.model("vdw").note
    assert r.best is not None and r.best.model == "lk"


def test_nitrogen_at_175_K_and_0_00375_m3_per_kg() -> None:
    """Çengel §3-8: 10 000 kPa medidos, 13 851 como gas ideal, 9471 con Van der Waals."""
    r = _solve("Nitrogen", "Tv", 175.0, 0.00375)
    assert r.unknown == "p"
    assert r.p_Pa / 1e3 == pytest.approx(10_001, abs=1.0)
    assert _unknown(r, "ideal") / 1e3 == pytest.approx(13_851, abs=1.0)
    assert _unknown(r, "vdw") / 1e3 == pytest.approx(9511, abs=2.0)  # 9471 con a y b del libro
    assert _unknown(r, "pr") / 1e3 == pytest.approx(9812, abs=2.0)
    assert _unknown(r, "lk0") / 1e3 == pytest.approx(9860, abs=2.0)
    assert _unknown(r, "lk") / 1e3 == pytest.approx(9971, abs=2.0)
    assert r.model("ideal").error == pytest.approx(0.385, abs=1e-3)


def test_steam_at_600_F_and_0_51431_ft3_per_lb() -> None:
    """Çengel §3-7: 1000 psia de las tablas, 1228 como gas ideal y 1056 con la carta."""
    r = _solve("Water", "Tv", (600.0 + 459.67) * 5 / 9, 0.51431 * FT3_LB)
    assert r.p_Pa / PSI == pytest.approx(1000.0, abs=0.5)
    assert _unknown(r, "ideal") / PSI == pytest.approx(1227.4, abs=0.5)
    assert _unknown(r, "vdw") / PSI == pytest.approx(1056, abs=1.0)
    assert _unknown(r, "pr") / PSI == pytest.approx(1005, abs=1.0)
    assert _unknown(r, "lk0") / PSI == pytest.approx(1029, abs=1.0)
    assert _unknown(r, "lk") / PSI == pytest.approx(1005, abs=1.0)
    assert r.v_R_pseudo == pytest.approx(2.372, abs=2e-3)  # el v′_R del libro


def test_temperature_from_pressure_and_volume() -> None:
    r = _solve("Air", "pv", 200e5, 0.0044438)
    assert r.unknown == "T"
    assert r.T_K == pytest.approx(300.0, abs=0.01)
    assert _unknown(r, "ideal") == pytest.approx(200e5 * 0.0044438 / r.fluid.R)
    assert _unknown(r, "lk") == pytest.approx(300.0, abs=0.2)
    # Van der Waals con (p, v) es explícita: T = (p + a/v²)(v − b)/R
    a, b = rg.cubic_constants(r.fluid, "vdw", r.fluid.T_cr)
    assert _unknown(r, "vdw") == pytest.approx(
        (200e5 + a / 0.0044438**2) * (0.0044438 - b) / r.fluid.R
    )


def test_round_trips() -> None:
    """Lo que da cada modelo con (T, v) o (p, v) vuelve a dar v con (p, T)."""
    r_tv = _solve("Methane", "Tv", 250.0, 0.01)
    r_pv = _solve("Methane", "pv", 60e5, 0.01)
    for kind in ("ideal", "lk0", "lk", "vdw", "pr"):
        for r in (r_tv, r_pv):
            m = r.model(kind)  # type: ignore[arg-type]
            assert m.p_Pa is not None and m.T_K is not None and m.v_m3_per_kg is not None
            back = _solve("Methane", "pT", m.p_Pa, m.T_K).model(kind)  # type: ignore[arg-type]
            assert back.v_m3_per_kg == pytest.approx(0.01, rel=1e-6)
            assert m.Z == pytest.approx(m.p_Pa * m.v_m3_per_kg / (r.fluid.R * m.T_K))


def test_low_pressure_is_ideal() -> None:
    r = _solve("Nitrogen", "pT", 1e3, 300.0)
    for m in r.models:
        assert m.Z == pytest.approx(1.0, abs=2e-4)
    assert any("alcanza" in n and "p_R ≪ 1" in n for n in r.notes)


def test_coolprop_reference_matches_its_own_z() -> None:
    r = _solve("CarbonDioxide", "pT", 80e5, 35.0 + C)
    state = CoolProp.AbstractState("HEOS", "CarbonDioxide")
    state.update(CoolProp.PT_INPUTS, 80e5, 35.0 + C)
    # Z con R = R_u/M (CODATA); CoolProp usa la R de su ecuación de estado
    assert r.Z == pytest.approx(state.compressibility_factor(), rel=2e-5)
    assert r.model("coolprop").error is None
    assert r.v_R == pytest.approx(r.v_m3_per_kg / r.fluid.v_cr)
    assert r.v_R_pseudo == pytest.approx(r.fluid.Z_cr * r.v_R)  # §7.4


# ---------------------------------------------------------------------
# Líquidos y fases
# ---------------------------------------------------------------------


def test_liquid_water_takes_the_liquid_root() -> None:
    r = _solve("Water", "pT", 1e5, 20.0 + C)
    assert r.phase == "liquid"
    assert r.v_m3_per_kg == pytest.approx(0.0010018, rel=1e-4)
    vdw = r.model("vdw")
    assert vdw.Z == pytest.approx(vdw.roots[0])  # la menor de las tres
    assert abs(vdw.error or 0.0) > 0.9  # Van der Waals: el doble del volumen
    assert abs(r.model("lk").error or 0.0) < 0.2
    assert any(n.startswith("Es líquido") for n in r.notes)


def test_liquid_where_van_der_waals_only_has_vapor() -> None:
    """R-134a líquido a T_R = 0,95 y 2 % sobre p_sat: para Van der Waals todavía es vapor."""
    fl = rg.real_fluid("R134a")
    T = 0.95 * fl.T_cr
    state = CoolProp.AbstractState("HEOS", "R134a")
    state.update(CoolProp.QT_INPUTS, 0.0, T)
    r = _solve("R134a", "pT", 1.02 * state.p(), T)
    assert r.phase == "liquid"
    vdw = r.model("vdw")
    assert vdw.Z is None and "todavía es vapor" in vdw.note
    assert r.model("pr").Z is not None and r.model("lk").Z is not None


def test_liquid_volume_below_the_covolume() -> None:
    r = _solve("Water", "Tv", 20.0 + C, 0.0010)
    assert "covolumen" in r.model("vdw").note
    assert "covolumen" in r.model("pr").note
    assert any("delicado" in n for n in r.notes)


def test_quantum_gas_without_a_reference_vapor_root() -> None:
    fl = rg.real_fluid("Helium")
    r = _solve("Helium", "pT", 0.6 * fl.p_cr, 0.9 * fl.T_cr)
    assert r.phase == "vapor"
    assert r.model("lk0").Z is not None  # el fluido simple sí tiene vapor
    assert "fluido de referencia" in r.model("lk").note
    assert any("gas cuántico" in n for n in r.notes)


# ---------------------------------------------------------------------
# Notas y errores
# ---------------------------------------------------------------------


def test_notes() -> None:
    co2 = _solve("CarbonDioxide", "pT", 80e5, 35.0 + C)
    assert any("punto crítico" in n for n in co2.notes)
    h2 = _solve("Hydrogen", "pT", 700e5, 25.0 + C)
    assert any("Z > 1" in n for n in h2.notes)
    assert any("Fuera del rango de ajuste de Lee y Kesler" in n for n in h2.notes)
    water = _solve("Water", "Tv", (600.0 + 459.67) * 5 / 9, 0.51431 * FT3_LB)
    assert any("polar" in n for n in water.notes)
    assert any("Z < 1" in n for n in water.notes)
    air = _solve("Air", "pT", 101_325.0, 25.0 + C)
    assert air.notes[-1].startswith("El modelo que menos se equivoca acá es")
    methane = _solve("Methane", "pT", 70e5, 15.0 + C)
    assert "la carta generalizada (Z⁰): 0,02 %" in methane.notes[-1]


@pytest.mark.parametrize(
    "pair,a,b,match",
    [
        ("pT", -1e5, 300.0, "positiva"),
        ("pT", 1e5, 0.0, "positiva"),
        ("pT", 1e5, 5000.0, "fuera del rango"),
        ("pT", 5e9, 300.0, "supera la máxima"),
        ("Tv", 300.0, 0.03, "dentro de la campana"),  # agua a 27 °C: mezcla
        ("Tv", 300.0, float("nan"), "positivo"),
    ],
)
def test_errors_for_the_student(pair: str, a: float, b: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _solve("Water", pair, a, b)


# ---------------------------------------------------------------------
# Gráficos, export y robustez
# ---------------------------------------------------------------------


def test_generalized_chart() -> None:
    chart = rg.generalized_chart()
    iso = chart["isotherms"]
    assert set(iso) == {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0}
    for pr, z in iso.values():
        assert len(pr) == len(z) > 50
        assert z[0] == pytest.approx(1.0, abs=0.01)  # a p_R chica, gas ideal
        assert list(pr) == sorted(pr)
    # bajo T_R = 1 la isoterma salta del vapor al líquido en p_sat
    pr, z = iso[0.7]
    psat = rg.lee_kesler_psat_reduced(0.7)
    k = next(i for i, x in enumerate(pr) if x > psat)
    assert z[k - 1] > 0.85 and z[k] < 0.05
    # T_R = 2 casi no se aparta de Z = 1 (Çengel §3-7)
    assert all(abs(x - 1.0) < 0.06 for x in iso[2.0][1][: len(iso[2.0][1]) // 2])
    prs, zf, zg = chart["dome"]
    assert prs[-1] == 1.0 and zf[-1] == zg[-1] == rg._LK_ZC["simple"]
    assert all(f < g for f, g in zip(zf[:-1], zg[:-1], strict=True))


def test_z_isotherm_with_the_saturation_jump() -> None:
    out = rg.z_isotherm("R134a", 50.0 + C, 2e6)
    p, p_sat = out["p"], out["p_sat"]
    assert p_sat == pytest.approx(1.3179e6, rel=1e-3)
    assert p.count(p_sat) == 2
    k = p.index(p_sat)
    real = out["Z"]["coolprop"]
    state = CoolProp.AbstractState("HEOS", "R134a")
    state.update(CoolProp.QT_INPUTS, 1.0, 50.0 + C)
    assert real[k] == pytest.approx(p_sat / (state.rhomass() * rg.real_fluid("R134a").R * (50 + C)))
    assert real[k + 1] < 0.1  # el líquido saturado
    for kind, zs in out["Z"].items():
        assert len(zs) == len(p), kind
    assert out["Z"]["ideal"] == [1.0] * len(p)
    # sobre la temperatura crítica no hay saturación
    assert rg.z_isotherm("Nitrogen", 300.0)["p_sat"] is None


def test_examples_solve_and_export() -> None:
    for name, ex in rg.REAL_EXAMPLES.items():
        t0 = time.perf_counter()
        r = rg.compressibility(ex.inputs)
        assert time.perf_counter() - t0 < 1.0, name
        assert all(m.Z is not None for m in r.models if m.model in ("ideal", "coolprop")), name
        d = rg.compressibility_to_dict(r)
        json.dumps(d)
        assert d["modelos"]["coolprop"]["Z"] == pytest.approx(r.Z)
        assert set(d["modelos"]) == set(rg.MODELS)
        assert pickle.loads(pickle.dumps(r)) == r


def test_robustness_grid() -> None:
    """Cada estado de una fase o se resuelve o da un error en castellano; cada modelo, un valor o
    el porqué."""
    for key in ("Argon", "CarbonDioxide", "Water", "R134a", "Hydrogen"):
        fl = rg.real_fluid(key)
        for Tr in (0.6, 0.9, 1.05, 2.0):
            T = Tr * fl.T_cr
            if not fl.T_min <= T <= fl.T_max:
                continue
            for pr in (0.05, 0.5, 1.5, 5.0):
                state = CoolProp.AbstractState("HEOS", key)
                state.update(CoolProp.PT_INPUTS, pr * fl.p_cr, T)
                v = 1.0 / state.rhomass()
                for inputs in (
                    rg.CompressibilityInputs(key, "pT", pr * fl.p_cr, T),
                    rg.CompressibilityInputs(key, "Tv", T, v),
                    rg.CompressibilityInputs(key, "pv", pr * fl.p_cr, v),
                ):
                    try:
                        r = rg.compressibility(inputs)
                    except ValueError as exc:
                        assert any(ch in str(exc) for ch in "áéíóú"), exc  # castellano
                        continue
                    assert r.Z == pytest.approx(state.compressibility_factor(), rel=1e-3)
                    for m in r.models:
                        if m.Z is None:
                            assert m.note, m
