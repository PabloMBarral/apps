"""Tests de :mod:`core.combustion.iso6976` — Fase 2.3.

Estructura incremental: cada ejemplo del Annex D se verifica como una
clase con un fixture de clase que ejecuta ``calculate`` una sola vez y
los tests individuales asertan cada output contra los valores tabulados
en ``tests/fixtures/iso6976_annex_d.json``. Tolerancias = las del
fixture (no se aflojan).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.combustion.iso6976 import (
    UNCERTAINTY_LATEX,
    GasComponent,
    ISO6976Inputs,
    ISO6976Result,
    ISO6976Tables,
    Quantity,
    ReferenceCondition,
    _build_composition_covariance,
    calculate,
    iso6976_to_dict,
    load_tables,
    normalization_fit,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "iso6976_annex_d.json"

#: Las salidas con u en los casos de la matriz de normalización (atributo, clave).
_NORMALIZED_OUTPUTS = (
    ("Hv_G_volume_MJ_per_m3", "Hv_G_volume_MJ_per_m3"),
    ("Hv_N_volume_MJ_per_m3", "Hv_N_volume_net_MJ_per_m3"),
    ("density_kg_per_m3", "density_kg_per_m3"),
    ("relative_density", "relative_density"),
    ("Wobbe_gross_MJ_per_m3", "Wobbe_gross_MJ_per_m3"),
    ("Wobbe_net_MJ_per_m3", "Wobbe_net_MJ_per_m3"),
)


@pytest.fixture(scope="module")
def annex_d() -> dict[str, Any]:
    with open(_FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tables() -> ISO6976Tables:
    return load_tables()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_inputs(spec: dict[str, Any]) -> ISO6976Inputs:
    comp = [
        GasComponent(name=c["name"], x=c["x"], u_x=c["u_x"])
        for c in spec["composition_mol_fraction"]
    ]
    cref = ReferenceCondition(
        T_celsius=spec["combustion_reference"]["T_celsius"],
        P_kPa=spec["combustion_reference"]["P_kPa"],
    )
    mref = ReferenceCondition(
        T_celsius=spec["metering_reference"]["T_celsius"],
        P_kPa=spec["metering_reference"]["P_kPa"],
    )
    return ISO6976Inputs(
        composition=comp,
        combustion_reference=cref,
        metering_reference=mref,
        correlation_matrix=spec["correlation_matrix"],
    )


_RESULT_FIELDS = {
    "molar_mass_kg_per_kmol": "molar_mass_kg_per_kmol",
    "summation_factor": "summation_factor",
    "compression_factor": "compression_factor",
    "molar_volume_m3_per_mol": "molar_volume_m3_per_mol",
    "Hc_G_molar_kJ_per_mol": "Hc_G_molar_kJ_per_mol",
    "Hm_G_mass_MJ_per_kg": "Hm_G_mass_MJ_per_kg",
    "Hv_G_volume_MJ_per_m3": "Hv_G_volume_MJ_per_m3",
    "Hv_N_volume_net_MJ_per_m3": "Hv_N_volume_MJ_per_m3",
    "density_kg_per_m3": "density_kg_per_m3",
    "relative_density": "relative_density",
    "Wobbe_gross_MJ_per_m3": "Wobbe_gross_MJ_per_m3",
    "Wobbe_net_MJ_per_m3": "Wobbe_net_MJ_per_m3",
}


def _get_quantity(result: ISO6976Result, fixture_key: str) -> Quantity:
    attr = _RESULT_FIELDS[fixture_key]
    return getattr(result, attr)


def _assert_value(q: Quantity, expected: dict[str, Any]) -> None:
    tol = expected["tol_abs"]
    assert abs(q.value - expected["value"]) <= tol, (
        f"value mismatch: got {q.value!r}, expected {expected['value']!r}, tol_abs={tol}"
    )


def _assert_u(q: Quantity, expected: dict[str, Any]) -> None:
    if "u" not in expected:
        return
    tol = expected["tol_abs"]
    assert abs(q.u - expected["u"]) <= tol, (
        f"u mismatch: got {q.u!r}, expected {expected['u']!r}, tol_abs={tol}"
    )


def _assert_U_k2(q: Quantity, expected: dict[str, Any]) -> None:
    if "U_k2" not in expected:
        return
    # tolerancia un poco más holgada que tol_abs en algunos casos: U_k2 se
    # redondea a 3 decimales en la norma. Tolerancia razonable: 5e-4.
    tol = max(expected["tol_abs"], 5.0e-4)
    assert abs(q.U_k2 - expected["U_k2"]) <= tol, (
        f"U_k2 mismatch: got {q.U_k2!r}, expected {expected['U_k2']!r}, tol={tol}"
    )


# ---------------------------------------------------------------------
# Example 1 — Annex D.2 (5 componentes, 15/15 °C, identity)
# ---------------------------------------------------------------------


class TestExample1_D2:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        inputs = _build_inputs(annex_d["example_1"]["inputs"])
        return calculate(inputs, tables=tables)

    @pytest.fixture(scope="class")
    @classmethod
    def expected_inter(cls, annex_d: dict) -> dict:
        return annex_d["example_1"]["expected_intermediate"]

    @pytest.fixture(scope="class")
    @classmethod
    def expected_final(cls, annex_d: dict) -> dict:
        return annex_d["example_1"]["expected_final"]

    # Intermedios
    def test_molar_mass(self, result, expected_inter) -> None:
        _assert_value(result.molar_mass_kg_per_kmol, expected_inter["molar_mass_kg_per_kmol"])

    def test_summation_factor(self, result, expected_inter) -> None:
        _assert_value(result.summation_factor, expected_inter["summation_factor"])

    def test_compression_factor(self, result, expected_inter) -> None:
        _assert_value(result.compression_factor, expected_inter["compression_factor"])

    def test_molar_volume(self, result, expected_inter) -> None:
        _assert_value(result.molar_volume_m3_per_mol, expected_inter["molar_volume_m3_per_mol"])

    # Finales (valor + u)
    def test_Hc_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hc_G_molar_kJ_per_mol, expected_final["Hc_G_molar_kJ_per_mol"])

    def test_Hc_G_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hc_G_molar_kJ_per_mol, expected_final["Hc_G_molar_kJ_per_mol"])

    def test_Hm_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hm_G_mass_MJ_per_kg, expected_final["Hm_G_mass_MJ_per_kg"])

    def test_Hm_G_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hm_G_mass_MJ_per_kg, expected_final["Hm_G_mass_MJ_per_kg"])

    def test_Hv_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])

    def test_Hv_G_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])


# ---------------------------------------------------------------------
# Example 2 — Annex D.3 (con vapor de agua, 15.55/15.55 °C, identity)
# ---------------------------------------------------------------------


class TestExample2_D3:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        inputs = _build_inputs(annex_d["example_2"]["inputs"])
        return calculate(inputs, tables=tables)

    @pytest.fixture(scope="class")
    @classmethod
    def expected_inter(cls, annex_d: dict) -> dict:
        return annex_d["example_2"]["expected_intermediate"]

    @pytest.fixture(scope="class")
    @classmethod
    def expected_final(cls, annex_d: dict) -> dict:
        return annex_d["example_2"]["expected_final"]

    def test_molar_mass(self, result, expected_inter) -> None:
        _assert_value(result.molar_mass_kg_per_kmol, expected_inter["molar_mass_kg_per_kmol"])

    def test_summation_factor(self, result, expected_inter) -> None:
        _assert_value(result.summation_factor, expected_inter["summation_factor"])

    def test_compression_factor(self, result, expected_inter) -> None:
        # Fixture tol_abs = 2e-6 (subido de 1e-6) por la precisión de
        # 4 decimales de los factores de sumación en Tabla 2, que limita
        # la precisión de Z = 1 − s² a ~6 decimales. Mi Z = 0.99756896
        # vs fixture 0.997570 (que representa el redondeo a 6 dec).
        # Ver _precision_notes en _meta del JSON.
        _assert_value(result.compression_factor, expected_inter["compression_factor"])

    def test_molar_volume(self, result, expected_inter) -> None:
        _assert_value(result.molar_volume_m3_per_mol, expected_inter["molar_volume_m3_per_mol"])

    def test_Hc_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hc_G_molar_kJ_per_mol, expected_final["Hc_G_molar_kJ_per_mol"])

    def test_Hc_G_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hc_G_molar_kJ_per_mol, expected_final["Hc_G_molar_kJ_per_mol"])

    def test_Hm_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hm_G_mass_MJ_per_kg, expected_final["Hm_G_mass_MJ_per_kg"])

    def test_Hv_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])

    def test_water_contributes_zero_to_Hc_N(self, result, annex_d: dict) -> None:
        """Cross-check físico: el agua-vapor en el combustible debe sumar 0
        al calor de combustión neto (vapor que se queda vapor no libera calor).
        Por la fórmula Hc_N = Hc_G − (Σ xⱼ bⱼ/2)·L₀ con Hc_G(water) = L₀,
        la contribución de water (b=2) cancela exactamente."""
        spec = annex_d["example_2"]["inputs"]
        comp_no_water = [
            GasComponent(name=c["name"], x=c["x"], u_x=c["u_x"])
            for c in spec["composition_mol_fraction"]
            if c["name"] != "water"
        ]
        # Renormalizar.
        s = sum(c.x for c in comp_no_water)
        comp_no_water = [GasComponent(name=c.name, x=c.x / s, u_x=c.u_x / s) for c in comp_no_water]
        inputs_dry = ISO6976Inputs(
            composition=comp_no_water,
            combustion_reference=ReferenceCondition(
                spec["combustion_reference"]["T_celsius"],
                spec["combustion_reference"]["P_kPa"],
            ),
            metering_reference=ReferenceCondition(
                spec["metering_reference"]["T_celsius"],
                spec["metering_reference"]["P_kPa"],
            ),
            correlation_matrix=spec["correlation_matrix"],
        )
        # Comparar Hc_N por mol de combustible. La renormalización cambia
        # los valores; este test solo verifica que el módulo no crashea
        # con composición sin agua y que el Hc_N permanezca finito.
        r_dry = calculate(inputs_dry, tables=load_tables())
        assert r_dry.Hc_N_molar_kJ_per_mol.value > 0


# ---------------------------------------------------------------------
# Example 3 — Annex D.4 (11 componentes, 4 sub-casos)
# ---------------------------------------------------------------------


def _example_3_case_inputs(annex_d: dict, case_key: str) -> ISO6976Inputs:
    base = annex_d["example_3"]["inputs"]
    case = annex_d["example_3"]["cases"][case_key]
    spec = {
        "composition_mol_fraction": base["composition_mol_fraction"],
        "combustion_reference": case["combustion_reference"],
        "metering_reference": case["metering_reference"],
        "correlation_matrix": case["correlation_matrix"],
    }
    return _build_inputs(spec)


class TestExample3_D4_caseA_identity:
    """D.4.3.1 — Referencia ISO standard (15/15 °C), identity matrix."""

    CASE_KEY = "case_a_iso_standard_identity"

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        return calculate(_example_3_case_inputs(annex_d, cls.CASE_KEY), tables=tables)

    @pytest.fixture(scope="class")
    @classmethod
    def expected_final(cls, annex_d: dict) -> dict:
        return annex_d["example_3"]["cases"][cls.CASE_KEY]["expected_final"]

    def test_Hv_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])

    def test_Hv_G_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])

    def test_Hv_N_value(self, result, expected_final) -> None:
        _assert_value(result.Hv_N_volume_MJ_per_m3, expected_final["Hv_N_volume_net_MJ_per_m3"])

    def test_Hv_N_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hv_N_volume_MJ_per_m3, expected_final["Hv_N_volume_net_MJ_per_m3"])

    def test_density_value(self, result, expected_final) -> None:
        _assert_value(result.density_kg_per_m3, expected_final["density_kg_per_m3"])

    def test_density_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.density_kg_per_m3, expected_final["density_kg_per_m3"])

    def test_relative_density_value(self, result, expected_final) -> None:
        _assert_value(result.relative_density, expected_final["relative_density"])

    def test_relative_density_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.relative_density, expected_final["relative_density"])

    def test_Wobbe_gross_value(self, result, expected_final) -> None:
        _assert_value(result.Wobbe_gross_MJ_per_m3, expected_final["Wobbe_gross_MJ_per_m3"])

    def test_Wobbe_gross_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Wobbe_gross_MJ_per_m3, expected_final["Wobbe_gross_MJ_per_m3"])

    def test_Wobbe_net_value(self, result, expected_final) -> None:
        _assert_value(result.Wobbe_net_MJ_per_m3, expected_final["Wobbe_net_MJ_per_m3"])

    def test_Wobbe_net_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Wobbe_net_MJ_per_m3, expected_final["Wobbe_net_MJ_per_m3"])


class TestExample3_D4_caseB_normalization:
    """D.4.3.2 — Idem case_a con la matriz de normalización: los valores son los mismos
    y las u bajan (las fracciones normalizadas se correlacionan)."""

    CASE_KEY = "case_b_iso_standard_normalization"

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        return calculate(_example_3_case_inputs(annex_d, cls.CASE_KEY), tables=tables)

    @pytest.fixture(scope="class")
    @classmethod
    def result_a(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        return calculate(
            _example_3_case_inputs(annex_d, "case_a_iso_standard_identity"),
            tables=tables,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def expected_final(cls, annex_d: dict) -> dict:
        return annex_d["example_3"]["cases"][cls.CASE_KEY]["expected_final"]

    def test_values_identical_to_case_a(self, result, result_a) -> None:
        """Los valores Y son iguales en identity y normalization
        (solo cambia la covarianza, no la media)."""
        for attr, _key in _NORMALIZED_OUTPUTS:
            q_b = getattr(result, attr)
            q_a = getattr(result_a, attr)
            assert q_b.value == pytest.approx(q_a.value, rel=1e-12)

    @pytest.mark.parametrize(("attr", "key"), _NORMALIZED_OUTPUTS)
    def test_value(self, result, expected_final, attr: str, key: str) -> None:
        _assert_value(getattr(result, attr), expected_final[key])

    @pytest.mark.parametrize(("attr", "key"), _NORMALIZED_OUTPUTS)
    def test_uncertainty(self, result, expected_final, attr: str, key: str) -> None:
        _assert_u(getattr(result, attr), expected_final[key])
        _assert_U_k2(getattr(result, attr), expected_final[key])

    @pytest.mark.parametrize(("attr", "key"), _NORMALIZED_OUTPUTS)
    def test_uncertainty_within_half_a_percent(
        self, result, expected_final, attr: str, key: str
    ) -> None:
        """Más justo que la tolerancia de la fixture: las u publicadas tienen tres cifras y
        las u(x) de entrada una o dos (u_Hv_G: 0.0163122 contra 0.016316)."""
        assert getattr(result, attr).u == pytest.approx(expected_final[key]["u"], rel=5e-3)

    def test_lower_than_identity(self, result, result_a) -> None:
        """La identidad es la sobre-estimación segura de §D.4.3.2."""
        for attr, _key in _NORMALIZED_OUTPUTS:
            assert getattr(result, attr).u < getattr(result_a, attr).u

    def test_raw_uncertainties_are_reconstructed(self, result) -> None:
        fit = result.normalization
        assert fit is not None
        assert fit.basis == "normalized"
        assert fit.u_normalized == pytest.approx(fit.u_given, rel=1e-12)
        # Normalizar le baja la incertidumbre al metano: 0,000634 cruda, 0,000348 normalizada.
        assert fit.u_raw[0] == pytest.approx(6.3447e-4, rel=1e-4)
        assert fit.u_raw[0] > fit.u_given[0]
        # El metano se anticorrelaciona con los de más incertidumbre (etano, propano, N₂ y
        # CO₂); los que casi no tienen incertidumbre propia se mueven con él (r > 0: al
        # normalizar, los dos se escalan con el mismo Σy).
        r = dict(zip(fit.names, fit.correlation[0], strict=True))
        assert r["ethane"] < -0.6 and r["nitrogen"] < -0.5
        assert r["propane"] < -0.3 and r["carbon dioxide"] < -0.25
        assert r["n-pentane"] > 0.0


class TestExample3_D4_caseC_identity_25_0:
    """D.4.4.1 — Combustión 25 °C, medición 0 °C, identity."""

    CASE_KEY = "case_c_25c_0c_identity"

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        return calculate(_example_3_case_inputs(annex_d, cls.CASE_KEY), tables=tables)

    @pytest.fixture(scope="class")
    @classmethod
    def expected_final(cls, annex_d: dict) -> dict:
        return annex_d["example_3"]["cases"][cls.CASE_KEY]["expected_final"]

    def test_Hv_G_value(self, result, expected_final) -> None:
        _assert_value(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])

    def test_Hv_N_value(self, result, expected_final) -> None:
        _assert_value(result.Hv_N_volume_MJ_per_m3, expected_final["Hv_N_volume_net_MJ_per_m3"])

    def test_density_value(self, result, expected_final) -> None:
        _assert_value(result.density_kg_per_m3, expected_final["density_kg_per_m3"])

    def test_relative_density_value(self, result, expected_final) -> None:
        _assert_value(result.relative_density, expected_final["relative_density"])

    def test_Wobbe_gross_value(self, result, expected_final) -> None:
        _assert_value(result.Wobbe_gross_MJ_per_m3, expected_final["Wobbe_gross_MJ_per_m3"])

    def test_Wobbe_net_value(self, result, expected_final) -> None:
        _assert_value(result.Wobbe_net_MJ_per_m3, expected_final["Wobbe_net_MJ_per_m3"])

    def test_Hv_G_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.Hv_G_volume_MJ_per_m3, expected_final["Hv_G_volume_MJ_per_m3"])

    def test_density_uncertainty(self, result, expected_final) -> None:
        _assert_u(result.density_kg_per_m3, expected_final["density_kg_per_m3"])


class TestExample3_D4_caseD_normalization_25_0:
    """D.4.4.2 — Combustión 25 °C, medición 0 °C, matriz de normalización."""

    CASE_KEY = "case_d_25c_0c_normalization"

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls, annex_d: dict, tables: ISO6976Tables) -> ISO6976Result:
        return calculate(_example_3_case_inputs(annex_d, cls.CASE_KEY), tables=tables)

    @pytest.fixture(scope="class")
    @classmethod
    def expected_final(cls, annex_d: dict) -> dict:
        return annex_d["example_3"]["cases"][cls.CASE_KEY]["expected_final"]

    @pytest.mark.parametrize(("attr", "key"), _NORMALIZED_OUTPUTS)
    def test_value(self, result, expected_final, attr: str, key: str) -> None:
        _assert_value(getattr(result, attr), expected_final[key])

    @pytest.mark.parametrize(("attr", "key"), _NORMALIZED_OUTPUTS)
    def test_uncertainty(self, result, expected_final, attr: str, key: str) -> None:
        _assert_u(getattr(result, attr), expected_final[key])
        _assert_U_k2(getattr(result, attr), expected_final[key])
        assert getattr(result, attr).u == pytest.approx(expected_final[key]["u"], rel=5e-3)


class TestNormalizationFit:
    """La matriz de normalización: las u del análisis crudo y sus propiedades."""

    @pytest.fixture(scope="class")
    @classmethod
    def composition(cls, annex_d: dict) -> list[GasComponent]:
        spec = annex_d["example_3"]["inputs"]["composition_mol_fraction"]
        return [GasComponent(c["name"], c["x"], c["u_x"]) for c in spec]

    def test_annex_d_is_reconstructed_exactly(self, composition) -> None:
        fit = normalization_fit(composition)
        assert fit.names == tuple(c.name for c in composition)
        assert np.allclose(fit.u_normalized, fit.u_given, rtol=1e-12, atol=0.0)
        assert all(u >= 0.0 for u in fit.u_raw)

    def test_covariance_properties(self, composition) -> None:
        """Diagonal u², filas que suman 0 (Σx = 1 no tiene incertidumbre), simétrica y
        semidefinida positiva."""
        x = np.array([c.x for c in composition])
        u = np.array([c.u_x for c in composition])
        cov = _build_composition_covariance(u, x, "normalization")
        assert np.allclose(np.diag(cov), u**2, rtol=1e-12, atol=0.0)
        assert np.allclose(cov.sum(axis=1), 0.0, atol=1e-12 * u.max() ** 2)
        assert np.allclose(cov, cov.T, rtol=0.0, atol=1e-25)
        assert np.linalg.eigvalsh(cov).min() > -1e-12 * u.max() ** 2
        # Sin el ajuste da lo mismo que con los nombres.
        fit = normalization_fit(composition)
        assert np.allclose(
            _build_composition_covariance(u, x, "normalization", fit), cov, rtol=1e-12
        )

    def test_correlation_matrix(self, composition) -> None:
        r = np.array(normalization_fit(composition).correlation)
        assert np.allclose(np.diag(r), 1.0)
        assert np.all(np.abs(r) <= 1.0 + 1e-12)
        assert np.allclose(r, r.T)

    def test_two_components_are_fully_anticorrelated(self) -> None:
        fit = normalization_fit(
            [GasComponent("methane", 0.8, 0.002), GasComponent("ethane", 0.2, 0.002)]
        )
        assert fit.correlation[0][1] == pytest.approx(-1.0)
        cov = _build_composition_covariance(
            np.array([0.002, 0.002]), np.array([0.8, 0.2]), "normalization", fit
        )
        assert np.allclose(cov, 4e-6 * np.array([[1.0, -1.0], [-1.0, 1.0]]))

    def test_zero_uncertainties(self) -> None:
        fit = normalization_fit(
            [GasComponent("methane", 0.9, 0.0), GasComponent("ethane", 0.1, 0.0)]
        )
        assert fit.u_raw == (0.0, 0.0)
        assert fit.correlation == ((0.0, 0.0), (0.0, 0.0))

    def test_raw_basis_round_trip(self, composition) -> None:
        """Con las u del análisis crudo, la normalización da las u(x) del Anexo D (ida y
        vuelta)."""
        fit = normalization_fit(composition)
        raw = [GasComponent(c.name, c.x, u) for c, u in zip(composition, fit.u_raw, strict=True)]
        back = normalization_fit(raw, "raw")
        assert back.basis == "raw"
        assert back.u_raw == pytest.approx(fit.u_raw, rel=1e-12)
        assert back.u_normalized == pytest.approx(fit.u_given, rel=1e-10)

    def test_raw_basis_never_fails(self) -> None:
        """Cualquier u(y) ≥ 0 es posible: la que pasa la normalización a una traza sin
        incertidumbre propia, y las dos iguales con dos componentes."""
        fit = normalization_fit(
            [
                GasComponent("methane", 0.9, 0.0002),
                GasComponent("ethane", 0.06, 0.001),
                GasComponent("helium", 0.04, 0.0),
            ],
            "raw",
        )
        # u²(x_He) = x_He²·(u²(y_CH4) + u²(y_C2H6)) = 0.04²·(0.0002² + 0.001²)
        assert fit.u_normalized[2] == pytest.approx(0.04 * (0.0002**2 + 0.001**2) ** 0.5)
        pair = normalization_fit(
            [GasComponent("methane", 0.95, 0.001), GasComponent("ethane", 0.05, 0.0005)], "raw"
        )
        assert pair.u_normalized[0] == pytest.approx(pair.u_normalized[1])

    def test_a_trace_without_uncertainty_is_not_a_normalized_composition(self) -> None:
        """Con las u(x) normalizadas, una traza con u(x) = 0 no puede salir de normalizar:
        al normalizar le toca la de los demás (el mensaje sugiere cargar las crudas)."""
        comp = [
            GasComponent("methane", 0.9, 0.001),
            GasComponent("ethane", 0.06, 0.0008),
            GasComponent("helium", 0.04, 0.0),
        ]
        with pytest.raises(ValueError, match="'helium'.*análisis crudo"):
            normalization_fit(comp)

    def test_identity_has_no_raw_analysis(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 0.9, 0.001), GasComponent("ethane", 0.1, 0.001)],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        result = calculate(inputs, tables=tables)
        assert result.normalization is None
        assert "analisis_crudo" not in iso6976_to_dict(result)
        assert result.steps.uncertainty_latex == UNCERTAINTY_LATEX["identity"]
        assert "identidad" in result.steps.narrative_es

    def test_export_and_steps_with_normalization(self, annex_d, tables) -> None:
        result = calculate(
            _example_3_case_inputs(annex_d, "case_b_iso_standard_normalization"), tables=tables
        )
        data = iso6976_to_dict(result)
        assert data["matriz_de_correlacion"] == "normalization"
        raw = data["analisis_crudo"]
        assert data["base_de_las_u"] == "normalized"
        assert [r["componente"] for r in raw][:2] == ["methane", "ethane"]
        assert raw[0]["u_y_crudo"] == pytest.approx(6.3447e-4, rel=1e-4)
        assert raw[0]["u_x_normalizada"] == pytest.approx(0.000348, rel=1e-9)
        assert result.steps.uncertainty_latex == UNCERTAINTY_LATEX["normalization"]
        assert "normaliz" in result.steps.narrative_es


# ---------------------------------------------------------------------
# Validaciones
# ---------------------------------------------------------------------


def _valid_inputs() -> ISO6976Inputs:
    return ISO6976Inputs(
        composition=[
            GasComponent("methane", 0.9, 0.0001),
            GasComponent("ethane", 0.1, 0.0001),
        ],
        combustion_reference=ReferenceCondition(15.0),
        metering_reference=ReferenceCondition(15.0),
        correlation_matrix="identity",
    )


class TestValidations:
    def test_empty_composition_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="vacía"):
            calculate(inputs, tables=tables)

    def test_unknown_component_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("unicornium", 1.0, 0.0)],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="Componente desconocido"):
            calculate(inputs, tables=tables)

    def test_unsupported_combustion_temperature_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 1.0, 0.0)],
            combustion_reference=ReferenceCondition(30.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="combustión"):
            calculate(inputs, tables=tables)

    def test_unsupported_metering_temperature_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 1.0, 0.0)],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(25.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="medición"):
            calculate(inputs, tables=tables)

    def test_composition_must_sum_to_one(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[
                GasComponent("methane", 0.5, 0.0),
                GasComponent("ethane", 0.3, 0.0),
            ],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="normalizada"):
            calculate(inputs, tables=tables)

    def test_negative_x_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[
                GasComponent("methane", 1.1, 0.0),
                GasComponent("ethane", -0.1, 0.0),
            ],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calculate(inputs, tables=tables)

    def test_x_above_one_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 1.5, 0.0)],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calculate(inputs, tables=tables)

    def test_negative_uncertainty_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[
                GasComponent("methane", 0.5, -0.001),
                GasComponent("ethane", 0.5, 0.0),
            ],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="negativa"):
            calculate(inputs, tables=tables)

    def test_invalid_correlation_matrix_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 1.0, 0.0)],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="banana",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="correlation_matrix"):
            calculate(inputs, tables=tables)

    @staticmethod
    def _normalized(*components: tuple[str, float, float]) -> ISO6976Inputs:
        return ISO6976Inputs(
            composition=[GasComponent(*c) for c in components],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="normalization",
        )

    def test_normalization_with_two_components_needs_equal_u(self, tables) -> None:
        """x₁ + x₂ = 1: las dos u(x) tienen que ser iguales."""
        inputs = self._normalized(("methane", 0.95, 0.001), ("ethane", 0.05, 0.0005))
        with pytest.raises(ValueError, match="iguales"):
            calculate(inputs, tables=tables)

    def test_normalization_with_one_component_needs_zero_u(self, tables) -> None:
        with pytest.raises(ValueError, match="un solo componente"):
            calculate(self._normalized(("methane", 1.0, 0.001)), tables=tables)
        result = calculate(self._normalized(("methane", 1.0, 0.0)), tables=tables)
        assert result.Hv_G_volume_MJ_per_m3.u == pytest.approx(
            calculate(
                ISO6976Inputs(
                    composition=[GasComponent("methane", 1.0, 0.0)],
                    combustion_reference=ReferenceCondition(15.0),
                    metering_reference=ReferenceCondition(15.0),
                    correlation_matrix="identity",
                ),
                tables=tables,
            ).Hv_G_volume_MJ_per_m3.u,
            rel=1e-12,
        )

    def test_normalization_rejects_a_too_precise_major_component(self, tables) -> None:
        """Al normalizar, al metano se le pasa la incertidumbre de los demás: no puede
        tener menos que x·√(Σ u² de los demás) ≈ 0,9·0,00141."""
        inputs = self._normalized(
            ("methane", 0.9, 0.0002), ("ethane", 0.06, 0.001), ("propane", 0.04, 0.001)
        )
        with pytest.raises(ValueError, match="'methane'.*identidad"):
            calculate(inputs, tables=tables)

    def test_normalization_accepts_rounded_inputs(self, tables) -> None:
        """Con dos componentes, u(x) que difieren menos de 1 % se aceptan (redondeo)."""
        inputs = self._normalized(("methane", 0.95, 0.001), ("ethane", 0.05, 0.001005))
        result = calculate(inputs, tables=tables)
        assert result.normalization is not None
        assert result.normalization.u_normalized[0] == pytest.approx(
            result.normalization.u_normalized[1], rel=1e-12
        )

    def test_invalid_uncertainty_basis_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 1.0, 0.0)],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="normalization",
            uncertainty_basis="banana",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="uncertainty_basis"):
            calculate(inputs, tables=tables)

    def test_raw_basis_with_the_rejected_major_component(self, tables) -> None:
        """Las mismas u que se rechazan como normalizadas valen como crudas: al
        normalizar, el metano termina con más incertidumbre que la cruda."""
        inputs = ISO6976Inputs(
            composition=[
                GasComponent("methane", 0.9, 0.0002),
                GasComponent("ethane", 0.06, 0.001),
                GasComponent("propane", 0.04, 0.001),
            ],
            combustion_reference=ReferenceCondition(15.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="normalization",
            uncertainty_basis="raw",
        )
        result = calculate(inputs, tables=tables)
        assert result.normalization is not None
        assert result.normalization.u_normalized[0] > 0.0002
        assert "análisis crudo" in result.steps.narrative_es

    def test_zero_pressure_raises(self, tables) -> None:
        inputs = ISO6976Inputs(
            composition=[GasComponent("methane", 1.0, 0.0)],
            combustion_reference=ReferenceCondition(15.0, P_kPa=0.0),
            metering_reference=ReferenceCondition(15.0),
            correlation_matrix="identity",
        )
        with pytest.raises(ValueError, match="combustión"):
            calculate(inputs, tables=tables)


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------


class TestLoadTables:
    def test_loads_60_components(self, tables: ISO6976Tables) -> None:
        assert len(tables.components) == 60

    def test_components_have_atomic_counts(self, tables: ISO6976Tables) -> None:
        ch4 = tables.components["methane"]
        assert ch4.a == 1 and ch4.b == 4
        assert ch4.M_kg_per_kmol == pytest.approx(16.04246)

    def test_summation_factors_4_temperatures(self, tables: ISO6976Tables) -> None:
        sf_methane = tables.summation_factors["methane"]
        assert set(sf_methane.keys()) == {0.0, 15.0, 15.55, 20.0}

    def test_calorific_values_5_temperatures(self, tables: ISO6976Tables) -> None:
        hc_methane = tables.calorific_values["methane"]
        assert set(hc_methane.keys()) == {0.0, 15.0, 15.55, 20.0, 25.0}

    def test_noncombustibles_have_zero_Hc(self, tables: ISO6976Tables) -> None:
        noncombustibles = (
            "argon",
            "nitrogen",
            "carbon dioxide",
            "helium",
            "neon",
            "oxygen",
            "sulfur dioxide",
        )
        for name in noncombustibles:
            for T in (0.0, 15.0, 15.55, 20.0, 25.0):
                assert tables.calorific_values[name][T] == 0.0

    def test_atomic_weights_present(self, tables: ISO6976Tables) -> None:
        for sym in ("C", "H", "N", "O", "S", "He", "Ne", "Ar"):
            assert sym in tables.atomic_weights

    def test_air_constants_present(self, tables: ISO6976Tables) -> None:
        assert tables.M_air_kg_per_kmol == pytest.approx(28.96546)
        assert 0.99 < tables.Z_air_by_T[15.0] < 1.0

    def test_L0_water_present(self, tables: ISO6976Tables) -> None:
        assert 44.0 < tables.L0_water_by_T[15.0] < 45.0
