"""Tests del poder calorífico por correlaciones — Fase 6.

Contra las formas publicadas de cada correlación (y la de Boie en kcal/kg),
las 536 biomasas de Ghugare et al. (2014), los carbones de Argonne (Vorres,
1990), el PCS exacto de las sustancias puras (las h_f de NASA de la Fase 5)
y los combustibles por análisis de la biblioteca de la Fase 5.
"""

from __future__ import annotations

import json
import math
import pickle

import pytest

from core.combustion.fuels import FUELS, UltimateAnalysis
from core.combustion.heating_value import (
    CORRELATIONS,
    HEATING_VALUE_EXAMPLE_NOTES,
    HEATING_VALUE_EXAMPLES,
    DryProximate,
    DryUltimate,
    HeatingValueInputs,
    Reference,
    applicable_correlations,
    argonne_coals,
    argonne_deviations,
    basis_value,
    dataset_fit,
    default_moisture_values,
    dry_ultimate_from_analysis,
    element_heating_values,
    estimate_hhv_as_fired,
    ghugare_dataset,
    heating_value_to_dict,
    hhv_boie,
    hhv_channiwala_parikh,
    hhv_cordero,
    hhv_dulong,
    hhv_parikh,
    hydrogen_water_ratio,
    lhv_from_hhv,
    moisture_sweep,
    out_of_range,
    pure_substance_inputs,
    solve_heating_value,
    to_basis,
    water_hfg_J_per_kg,
    zero_lhv_moisture,
)
from core.export import dict_to_csv

KCAL = 4186.8  # J/kcal (tabla internacional)
EUCALYPTUS = DryUltimate(C=0.4818, H=0.0592, O=0.4418, N=0.0039, S=0.0001, A=0.0132)


def _solve(name: str):  # type: ignore[no-untyped-def]
    return solve_heating_value(HEATING_VALUE_EXAMPLES[name])


# ---------------------------------------------------------------------
# Las correlaciones en su forma publicada
# ---------------------------------------------------------------------


def test_channiwala_parikh_by_hand() -> None:
    hand = (
        0.3491 * 48.18
        + 1.1783 * 5.92
        + 0.1005 * 0.01
        - 0.1034 * 44.18
        - 0.0151 * 0.39
        - 0.0211 * 1.32
    )
    assert hhv_channiwala_parikh(EUCALYPTUS) == pytest.approx(hand * 1e6, rel=1e-12)


def test_boie_matches_its_kcal_form() -> None:
    """Boie (1953) en kcal/kg: 84,0·C + 277,65·H + 25·S + 15·N − 26,5·O (con %)."""
    for u in (EUCALYPTUS, DryUltimate(C=0.80, H=0.05, O=0.08, N=0.015, S=0.02, A=0.035)):
        c = u.pct()
        kcal = 84.0 * c["C"] + 277.65 * c["H"] + 25.0 * c["S"] + 15.0 * c["N"] - 26.5 * c["O"]
        assert hhv_boie(u) == pytest.approx(kcal * KCAL, rel=5e-4)


def test_dulong_matches_its_classic_kcal_form() -> None:
    """La forma clásica: 8080·C + 34 500·(H − O/8) + 2250·S kcal/kg con fracciones."""
    u = DryUltimate(C=0.8436, H=0.0189, O=0.0440, N=0.0063, S=0.0089, A=0.0783)
    kcal = 8080 * u.C + 34_500 * (u.H - u.O / 8) + 2250 * u.S
    assert hhv_dulong(u) == pytest.approx(kcal * KCAL, rel=1e-3)


def test_proximate_correlations_by_hand() -> None:
    p = DryProximate(VM=0.80, FC=0.17, A=0.03)
    assert hhv_parikh(p) == pytest.approx((0.3536 * 17 + 0.1559 * 80 - 0.0078 * 3) * 1e6)
    assert hhv_cordero(p) == pytest.approx((0.3543 * 17 + 0.1708 * 80) * 1e6)


def test_registry_is_consistent() -> None:
    assert list(CORRELATIONS) == ["channiwala_parikh", "boie", "dulong", "parikh", "cordero"]
    probe_u = DryUltimate(C=0.5, H=0.06, O=0.4, N=0.01, S=0.005, A=0.025)
    probe_p = DryProximate(VM=0.75, FC=0.20, A=0.05)
    for key, corr in CORRELATIONS.items():
        assert corr.key == key
        probe = probe_u if corr.analysis == "ultimate" else probe_p
        pct = probe.pct()
        by_coeffs = sum(coef * pct[var] for var, coef in corr.coefficients)
        assert corr.function(probe) == pytest.approx(by_coeffs * 1e6, rel=1e-12), key
        assert corr.reference
        assert corr.function.__doc__ and "Rango de validez" in corr.function.__doc__
        assert "Referencia" in corr.function.__doc__


def test_element_heating_values_are_dulong_physics() -> None:
    """Los PCS de C (grafito), H₂ y S desde las h_f de NASA: la base de Dulong."""
    hv = element_heating_values()
    assert hv["C"] == pytest.approx(32.763e6, rel=1e-4)
    assert hv["H"] == pytest.approx(141.79e6, rel=1e-4)
    assert hv["S"] == pytest.approx(9.2565e6, rel=1e-4)
    # Los coeficientes de Dulong (por % → por kg) están a menos de 3,5 % de esos.
    assert 33.83e6 / hv["C"] - 1 == pytest.approx(0.0326, abs=1e-3)
    assert 144.3e6 / hv["H"] - 1 == pytest.approx(0.0177, abs=1e-3)


def test_water_constants() -> None:
    assert water_hfg_J_per_kg() == pytest.approx(2442.6e3, rel=1e-4)  # vademecum §16.9: ≈ 2442
    assert hydrogen_water_ratio() == pytest.approx(8.9367, rel=1e-4)  # el «9·H»


# ---------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------


def test_bases_round_trip() -> None:
    """El mismo combustible cargado tal cual, seco o seco y sin cenizas da lo mismo."""
    W, A_d = 0.25, 0.08
    dry = {"C": 0.70, "H": 0.05, "O": 0.13, "N": 0.015, "S": 0.025}
    ref_d = 28.0e6
    results = []
    for basis in ("ar", "d", "daf"):
        conv = to_basis({**dry, "A": A_d}, basis, W, A_d)  # type: ignore[arg-type]
        ult = {k: 100 * conv[k] for k in dry}
        ash = 100 * A_d if basis == "daf" else 100 * conv["A"]
        inp = HeatingValueInputs.from_basis(
            basis,  # type: ignore[arg-type]
            ultimate_pct=ult,
            ash_pct=ash,
            moisture_pct=100 * W,
            reference_J_per_kg=basis_value(ref_d, basis, W, A_d),  # type: ignore[arg-type]
            reference_basis=basis,  # type: ignore[arg-type]
        )
        assert inp.ultimate is not None and inp.reference is not None
        results.append(inp)
        for k, v in dry.items():
            assert getattr(inp.ultimate, k) == pytest.approx(v, rel=1e-12)
        assert inp.ash_d == pytest.approx(A_d, rel=1e-12)
        assert inp.reference.hhv_d_J_per_kg == pytest.approx(ref_d, rel=1e-12)
    r = solve_heating_value(results[0])
    e = r.main
    assert e.hhv_ar == pytest.approx(e.hhv_d * (1 - W))
    assert e.hhv_daf == pytest.approx(e.hhv_d / (1 - A_d))


def test_by_difference() -> None:
    inp = HeatingValueInputs.from_basis(
        "ar",
        ultimate_pct={"C": 40.0, "H": 5.0, "N": 1.0, "S": 0.5},
        proximate_pct={"VM": 55.0},
        ash_pct=8.0,
        moisture_pct=10.0,
        o_by_difference=True,
        fc_by_difference=True,
    )
    assert inp.ultimate is not None and inp.proximate is not None
    assert inp.ultimate.O == pytest.approx(0.355 / 0.9)
    assert inp.proximate.FC == pytest.approx(0.27 / 0.9)
    with pytest.raises(ValueError, match="O por diferencia daría"):
        HeatingValueInputs.from_basis(
            "d", ultimate_pct={"C": 90.0, "H": 6.0, "N": 1.0}, ash_pct=5.0, o_by_difference=True
        )
    with pytest.raises(ValueError, match="carbono fijo por diferencia daría"):
        HeatingValueInputs.from_basis(
            "d", proximate_pct={"VM": 99.0}, ash_pct=5.0, fc_by_difference=True
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"basis": "d", "ultimate_pct": {"C": 50, "H": 6, "O": 40}, "ash_pct": 2},
            r"elemental en base seca suma 98\.00 %: C \+ H \+ O \+ N \+ S \+ cenizas",
        ),
        (
            {"basis": "ar", "ultimate_pct": {"C": 50, "H": 6, "O": 40}, "ash_pct": 2},
            r"elemental tal cual suma 98\.00 %: .* \+ cenizas \+ humedad",
        ),
        (
            {"basis": "daf", "proximate_pct": {"VM": 70, "FC": 20}, "ash_pct": 5},
            r"inmediato seco y sin cenizas suma 90\.00 %: MV \+ CF tiene",
        ),
        ({"basis": "d", "ultimate_pct": {"C": -1, "H": 6, "O": 95}}, "no pueden ser negativos"),
        ({"basis": "d", "ultimate_pct": {"C": 94, "H": 6}, "moisture_pct": 100}, "humedad"),
        ({"basis": "d"}, "elemental, el inmediato o los dos"),
        (
            {"basis": "d", "ultimate_pct": {"C": 94, "H": 6}, "reference_J_per_kg": -1.0},
            "referencia tiene que ser positivo",
        ),
        ({"basis": "d", "ultimate_pct": {"C": math.nan, "H": 6}}, "vacío o inválido"),
        (
            {"basis": "d", "ultimate_pct": {"C": 94, "H": 6}, "main": "parikh"},
            "usa el análisis inmediato",
        ),
    ],
)
def test_validation_messages(kwargs: dict, match: str) -> None:  # type: ignore[type-arg]
    basis = kwargs.pop("basis")
    with pytest.raises(ValueError, match=match):
        HeatingValueInputs.from_basis(basis, **kwargs)


def test_ashes_must_agree_between_analyses() -> None:
    inp = HeatingValueInputs(
        ultimate=DryUltimate(C=0.5, H=0.06, O=0.4, A=0.04),
        proximate=DryProximate(VM=0.7, FC=0.25, A=0.05),
    )
    with pytest.raises(ValueError, match="cenizas del análisis elemental"):
        inp.validate()


def test_rounding_is_tolerated_and_normalized() -> None:
    """Un análisis que suma 100,05 % (redondeo) se acepta y queda sumando 1."""
    inp = HeatingValueInputs.from_basis(
        "d", ultimate_pct={"C": 50.0, "H": 6.05, "O": 42.0, "N": 0.5, "S": 0.0}, ash_pct=1.5
    )
    assert inp.ultimate is not None
    assert sum(inp.ultimate.as_dict().values()) == pytest.approx(1.0, abs=1e-14)


# ---------------------------------------------------------------------
# PCI
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["Metano", "n-Octano (líquido)", "Etanol (líquido)", "Metanol (líquido)", "Hidrógeno"]
)
def test_pure_substances_pci_matches_phase_5(key: str) -> None:
    """Con el PCS exacto, PCS − h_fg·8,94·H es el PCI de la Fase 5 (de las h_f)."""
    fuel = FUELS[key]
    r = solve_heating_value(pure_substance_inputs(fuel, "liquid"))
    assert r.reference is not None
    assert r.reference.hhv_d == pytest.approx(fuel.hhv_per_kg, rel=1e-12)
    assert r.reference.lhv_d == pytest.approx(fuel.lhv_per_kg, rel=1e-9)
    assert r.reference.lhv_ar == pytest.approx(fuel.lhv_per_kg, rel=1e-9)


def test_pci_with_moisture_matches_phase_5_analysis_fuel() -> None:
    """El bagazo de la Fase 5 (50 % de humedad): mismo PCI que ``Fuel.lhv_per_kg``."""
    fuel = FUELS["Bagazo de caña (50 % de humedad)"]
    a = fuel.analysis
    assert a is not None and fuel.hhv_J_per_kg is not None
    dry = dry_ultimate_from_analysis(a)
    inp = HeatingValueInputs(
        ultimate=dry,
        moisture=a.W,
        reference=Reference(fuel.hhv_J_per_kg / (1 - a.W)),
    )
    r = solve_heating_value(inp)
    assert r.reference is not None
    assert r.reference.hhv_ar == pytest.approx(fuel.hhv_per_kg, rel=1e-12)
    assert r.reference.lhv_ar == pytest.approx(fuel.lhv_per_kg, rel=1e-9)


def test_zero_lhv_moisture() -> None:
    r = _solve("Bagazo de caña (50 % de humedad)")
    w_star = r.zero_lhv_moisture
    assert w_star is not None
    assert w_star == pytest.approx(0.881, abs=2e-3)
    lhv_d = r.main.lhv_d
    assert lhv_d is not None
    _, lhv_ar = lhv_from_hhv(r.main.hhv_d, r.inputs.hydrogen_d or 0.0, w_star)
    assert lhv_ar == pytest.approx(0.0, abs=1e-6)
    assert zero_lhv_moisture(-1.0) is None


def test_proximate_only_has_no_pci_until_h_is_given() -> None:
    r = _solve("Carbón bituminoso Pocahontas N.º 3 (solo inmediato)")
    assert r.main.key == "parikh"
    assert r.main.lhv_ar is None
    assert any("no se conoce el H" in n for n in r.notes)
    inp = HeatingValueInputs.from_basis(
        "d", proximate_pct={"VM": 80.0, "FC": 17.0}, ash_pct=3.0, moisture_pct=20, H_d_pct=6.0
    )
    r2 = solve_heating_value(inp)
    assert r2.main.lhv_d == pytest.approx(
        r2.main.hhv_d - water_hfg_J_per_kg() * hydrogen_water_ratio() * 0.06
    )


# ---------------------------------------------------------------------
# Validación: biomasa, carbones y sustancias puras
# ---------------------------------------------------------------------


def test_ghugare_dataset() -> None:
    data = ghugare_dataset()
    assert len(data) == 536
    assert sum(s.subset == "Training" for s in data) == 456
    assert sum(s.subset == "Testing" for s in data) == 80
    assert all(s.hhv_d > 0 for s in data)
    assert all(s.ultimate.A >= 0 for s in data)


@pytest.mark.parametrize(
    ("key", "aae", "abe", "within"),
    [
        ("channiwala_parikh", 0.0494, 0.0012, 0.901),
        ("boie", 0.0523, -0.0000, 0.890),
        ("dulong", 0.1146, -0.1015, 0.485),
    ],
)
def test_ghugare_metrics(key: str, aae: float, abe: float, within: float) -> None:
    fit = dataset_fit(key)
    assert fit.n == 536
    assert fit.aae == pytest.approx(aae, abs=1e-4)
    assert fit.abe == pytest.approx(abe, abs=1e-4)
    assert fit.within_10 == pytest.approx(within, abs=2e-3)
    assert len(fit.errors) == 536


def test_dulong_error_grows_with_oxygen() -> None:
    fit = dataset_fit("dulong")
    low = [e for e, o in zip(fit.errors, fit.oxygen, strict=True) if o < 0.10]
    high = [e for e, o in zip(fit.errors, fit.oxygen, strict=True) if o > 0.45]
    assert sum(low) / len(low) > 0.0
    assert sum(high) / len(high) < -0.15


def test_dataset_fit_rejects_proximate() -> None:
    with pytest.raises(ValueError, match="inmediato"):
        dataset_fit("parikh")


def test_argonne_coals() -> None:
    coals = {c.key: c for c in argonne_coals()}
    assert set(coals) == {"pittsburgh8", "illinois6", "wyodak", "beulahzap", "pocahontas3"}
    pitt = coals["pittsburgh8"]
    assert pitt.hhv_d == pytest.approx(31.70e6, rel=1e-3)  # 31 699 kJ/kg seco en el Handbook
    assert pitt.ultimate is not None
    assert pitt.ultimate.O == pytest.approx(0.0674, abs=1e-6)  # por diferencia, con el Cl (0,11 %)
    # Las cenizas del inmediato tal cual pasadas a seco coinciden con las del elemental.
    for c in coals.values():
        if c.ultimate is not None:
            assert abs(c.ultimate.A - c.proximate.A) < 1e-12
        assert sum(c.proximate.as_dict().values()) == pytest.approx(1.0)
        assert c.proximate.FC > 0


def test_argonne_deviations() -> None:
    """Las elementales aciertan a ±3,1 %; las del inmediato subestiman 7–23 %."""
    for row in argonne_deviations():
        dev = row["deviations"]
        for key in ("channiwala_parikh", "boie", "dulong"):
            if row["coal"].ultimate is not None:
                assert abs(dev[key]) < 0.031, (row["coal"].name, key)
            else:
                assert dev[key] is None
        assert -0.23 < dev["parikh"] < -0.10
        assert -0.21 < dev["cordero"] < -0.07


@pytest.mark.parametrize(
    ("name", "cp", "boie", "dulong"),
    [
        ("Metano (sustancia pura)", 0.004, 0.000, 0.110),
        ("n-Octano (sustancia pura)", 0.004, 0.003, 0.073),
        ("Etanol (sustancia pura)", 0.014, 0.002, 0.022),
        ("Metanol (sustancia pura)", 0.004, -0.018, -0.037),
        ("Hidrógeno (sustancia pura)", -0.169, -0.180, 0.018),
    ],
)
def test_pure_substances_against_exact(name: str, cp: float, boie: float, dulong: float) -> None:
    r = _solve(name)
    assert r.inputs.reference is not None and r.inputs.reference.kind == "exact"
    assert r.estimate("channiwala_parikh").deviation == pytest.approx(cp, abs=1e-3)
    assert r.estimate("boie").deviation == pytest.approx(boie, abs=1e-3)
    assert r.estimate("dulong").deviation == pytest.approx(dulong, abs=1e-3)


def test_phase5_library_fuels_reproduce_their_correlation() -> None:
    """El fueloil (Channiwala y Parikh) y el carbón de Pensilvania (Dulong) de la Fase 5."""
    fueloil = FUELS["Fueloil"]
    coal = FUELS["Carbón (Pensilvania)"]
    assert fueloil.analysis is not None and coal.analysis is not None
    assert estimate_hhv_as_fired(fueloil.analysis, "channiwala_parikh") == pytest.approx(
        fueloil.hhv_per_kg, rel=2e-3
    )
    assert estimate_hhv_as_fired(coal.analysis, "dulong") == pytest.approx(
        coal.hhv_per_kg, rel=3e-3
    )


def test_estimate_as_fired_uses_the_dry_basis() -> None:
    a = UltimateAnalysis(C=0.235, H=0.0325, O=0.22, N=0.0015, W=0.5, A=0.011)
    dry = dry_ultimate_from_analysis(a)
    assert dry.C == pytest.approx(0.47)
    assert estimate_hhv_as_fired(a, "boie") == pytest.approx(0.5 * hhv_boie(dry))
    with pytest.raises(ValueError, match="inmediato"):
        estimate_hhv_as_fired(a, "parikh")
    with pytest.raises(ValueError, match="100 % de humedad"):
        dry_ultimate_from_analysis(UltimateAnalysis(C=0.0, H=0.0, W=1.0))


# ---------------------------------------------------------------------
# Rangos, tipos y notas
# ---------------------------------------------------------------------


def test_out_of_range() -> None:
    inp = HEATING_VALUE_EXAMPLES["Eucalipto"]
    warn = out_of_range("dulong", inp)
    assert len(warn) == 1 and warn[0].startswith("O = 44,2 % (base seca)")
    assert out_of_range("channiwala_parikh", inp) == ()
    h2 = HEATING_VALUE_EXAMPLES["Hidrógeno (sustancia pura)"]
    assert any(w.startswith("H = 100 %") for w in out_of_range("channiwala_parikh", h2))


def test_fuel_type_flags() -> None:
    r = _solve("Eucalipto")
    flags = {e.key: e.in_type for e in r.estimates}
    assert flags == {"channiwala_parikh": True, "boie": True, "dulong": False}
    r = _solve("Carbón bituminoso Pittsburgh N.º 8")
    flags = {e.key: e.in_type for e in r.estimates}
    assert flags["cordero"] is False and flags["parikh"] is True and flags["dulong"] is True


def test_notes() -> None:
    bag = _solve("Bagazo de caña (50 % de humedad)").notes
    assert any("H disponible" in n for n in bag)
    assert any("W* = 88,1 %" in n for n in bag)
    assert not any(n.startswith("Dulong:") for n in bag)  # la nota del O ya lo explica
    coal = _solve("Carbón bituminoso Illinois N.º 6").notes
    assert any("análisis inmediato se ajustaron sobre todo con biomasa" in n for n in coal)
    methane = _solve("Metano (sustancia pura)").notes
    assert any("sustancia pura" in n for n in methane)
    assert any("h_f del combustible" in n for n in methane)
    h2 = _solve("Hidrógeno (sustancia pura)").notes
    assert any("no extrapolan" in n for n in h2)
    rice = _solve("Cáscara de arroz").notes
    assert any("cenizas en base seca" in n for n in rice)


def test_wet_fuel_with_negative_lhv() -> None:
    inp = HeatingValueInputs.from_basis(
        "d", ultimate_pct={"C": 47, "H": 6, "O": 44, "N": 0.5}, ash_pct=2.5, moisture_pct=92
    )
    r = solve_heating_value(inp)
    assert r.main.lhv_ar is not None and r.main.lhv_ar < 0
    assert any("no alcanza a evaporar su propia agua" in n for n in r.notes)


def test_negative_hhv_is_an_error() -> None:
    inp = HeatingValueInputs.from_basis(
        "d", ultimate_pct={"C": 1, "H": 0.5, "O": 3, "N": 0}, ash_pct=95.5
    )
    with pytest.raises(ValueError, match="PCS negativo"):
        solve_heating_value(inp)


# ---------------------------------------------------------------------
# Ejemplos, barrido, export
# ---------------------------------------------------------------------


def test_every_example_solves() -> None:
    assert set(HEATING_VALUE_EXAMPLE_NOTES) == set(HEATING_VALUE_EXAMPLES)
    for name, inp in HEATING_VALUE_EXAMPLES.items():
        r = solve_heating_value(inp)
        assert r.estimates, name
        assert r.reference is not None, name
        assert r.main.key in applicable_correlations(inp)
        for e in r.estimates:
            assert e.hhv_d > 0, (name, e.key)


def test_example_measurements_come_from_the_dataset() -> None:
    bag = HEATING_VALUE_EXAMPLES["Bagazo de caña (50 % de humedad)"]
    assert bag.reference is not None
    assert bag.reference.hhv_d_J_per_kg == pytest.approx(18.99e6)
    assert bag.moisture == 0.5
    assert bag.ultimate is not None and bag.ultimate.C == pytest.approx(0.4864, rel=1e-3)


def test_moisture_sweep() -> None:
    r = _solve("Bagazo de caña (50 % de humedad)")
    values = default_moisture_values()
    assert values[0] == 0.0 and values[-1] == pytest.approx(0.70)
    sweep = moisture_sweep(r, values)
    assert sweep["PCS"][0] == pytest.approx(r.main.hhv_d)
    assert sweep["PCI"][0] == pytest.approx(r.main.lhv_d)
    lhv = [v for v in sweep["PCI"] if v is not None]
    assert all(b < a for a, b in zip(lhv, lhv[1:], strict=False))
    with pytest.raises(ValueError, match="humedad del barrido"):
        moisture_sweep(r, [1.0])
    no_h = moisture_sweep(_solve("Carbón bituminoso Pocahontas N.º 3 (solo inmediato)"), [0.1])
    assert no_h["PCI"] == [None]


@pytest.mark.parametrize("system", ["SI", "Técnico", "Inglés"])
def test_export(system: str) -> None:
    for name in HEATING_VALUE_EXAMPLES:
        data = heating_value_to_dict(_solve(name), system)  # type: ignore[arg-type]
        json.dumps(data)
        assert "correlaciones" in data and data["correlaciones"]
        assert dict_to_csv(data).startswith("campo,valor")
    data = heating_value_to_dict(_solve("Lignito Beulah-Zap"), "Técnico")
    assert set(data["analisis_elemental_%"]) == {
        "tal cual (como se recibe)",
        "base seca",
        "seca y sin cenizas",
    }
    assert data["correlaciones"][0]["PCS_seca"]["unidad"] == "kJ/kg"


def test_results_pickle() -> None:
    r = _solve("Carbón bituminoso Pittsburgh N.º 8")
    back = pickle.loads(pickle.dumps(r))
    assert back.main.hhv_d == r.main.hhv_d
    assert back.notes == r.notes


def test_to_basis_keys() -> None:
    dry = {"C": 0.5, "A": 0.1}
    assert to_basis(dry, "ar", 0.2, 0.1) == pytest.approx({"C": 0.4, "A": 0.08, "W": 0.2})
    assert to_basis(dry, "daf", 0.2, 0.1) == pytest.approx({"C": 0.5 / 0.9})
    assert basis_value(10.0, "daf", 0.2, 0.5) == pytest.approx(20.0)
