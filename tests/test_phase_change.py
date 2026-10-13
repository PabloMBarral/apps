"""Tests de lo común de la ebullición y la condensación (Fase 11.2): fluidos y propiedades."""

from __future__ import annotations

import pytest
from CoolProp.CoolProp import PropsSI

from core.heat_transfer import phase_change as pc

C = 273.15


@pytest.mark.parametrize("fluid", list(pc.PHASE_CHANGE_FLUIDS))
def test_every_fluid_has_saturation_properties(fluid: str) -> None:
    p_tr, p_cr = pc.pressure_limits(fluid)
    for p in (max(3 * p_tr, 0.05 * p_cr), 0.5 * p_cr, 0.9 * p_cr):
        sat = pc.saturation_state(fluid, p)
        assert sat.rho_l > sat.rho_v > 0.0
        assert sat.h_fg > 0.0 and sat.sigma > 0.0 and sat.beta_l > 0.0
        assert sat.Pr_l > 0.0 and sat.nu_l > 0.0 and sat.alpha_l > 0.0
        assert sat.delta_rho == pytest.approx(sat.rho_l - sat.rho_v)
        assert pc.saturation_pressure(fluid, sat.T_sat_K) == pytest.approx(p, rel=1e-6)
    assert pc.fluid_name(fluid) and pc.with_article(fluid).startswith("el ")


def test_water_at_one_atmosphere() -> None:
    sat = pc.saturation_state("Water", 101_325.0)
    assert sat.T_sat_K == pytest.approx(373.124, abs=1e-3)
    assert sat.h_fg == pytest.approx(2256.5e3, rel=1e-3)
    assert sat.sigma == pytest.approx(0.0589, rel=2e-3)


def test_phase_is_imposed_near_saturation() -> None:
    """A 1 mK de la saturación cada fase sigue siendo la pedida."""
    sat = pc.saturation_state("Water", 101_325.0)
    vapor = pc.phase_properties("Water", sat.T_sat_K + 1e-3, sat.p_Pa, "vapor")
    liquid = pc.phase_properties("Water", sat.T_sat_K - 1e-3, sat.p_Pa, "liquid")
    assert vapor.rho == pytest.approx(sat.rho_v, rel=1e-3)
    assert liquid.rho == pytest.approx(sat.rho_l, rel=1e-4)
    assert liquid.Pr == pytest.approx(sat.Pr_l, rel=1e-3)
    assert vapor.nu == pytest.approx(vapor.mu / vapor.rho)
    film = pc.phase_properties("Water", 225.0 + C, 101_325.0, "vapor")
    assert film.rho == pytest.approx(0.444, rel=0.01)  # Çengel y Ghajar, tabla A-16
    assert film.k == pytest.approx(0.0351, rel=0.03)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: pc.saturation_state("Water", 250e5), "crítica"),
        (lambda: pc.saturation_state("Water", 100.0), "punto triple"),
        (lambda: pc.saturation_state("Water", 0.0), "positiva"),
        (lambda: pc.saturation_state("Air", 1e5), "lista"),
        (lambda: pc.saturation_pressure("Water", 400.0 + C), "punto crítico"),
        (
            lambda: pc.phase_properties("R134a", PropsSI("Tmax", "R134a") + 10.0, 1e5, "vapor"),
            "temperatura máxima",
        ),
    ],
)
def test_errors(call: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()  # type: ignore[operator]
