"""Tests de :mod:`core.diagrams` — Fase 1.5a.

Cubre:

- Mapeo ``FLUPRODIA_UNITS`` (incluyendo el quirk de pint en sistema
  Inglés: ``Btu/(lb*degR)``).
- :data:`DEFAULT_RANGES` cubre todos los fluidos soportados.
- :func:`build_diagram` corre para los 7 fluidos del proyecto.
- Procesos (isoentrópico, isobárico, isotérmico): devuelven dicts con
  las keys SI esperadas, no vacíos, y monotonía en la coordenada esperada.
- Validaciones (fluido, sistema, tipo de diagrama) lanzan ``ValueError``
  con mensajes claros.
- Round-trip de JSON.
- Conversión SI ↔ coordenadas del diagrama (overlay helpers).

Headless: configuramos matplotlib con backend ``Agg`` antes de importar
fluprodia para evitar dependencias de GUI en CI.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless — antes de cualquier import indirecto de pyplot

import numpy as np
import pytest

from core.diagrams import (  # noqa: E402
    AXIS_MAP,
    DEFAULT_RANGES,
    FLUPRODIA_UNITS,
    SUPPORTED_DIAGRAM_TYPES,
    DiagramSpec,
    FluidRange,
    ProcessOverlay,
    build_diagram,
    from_json,
    isentropic_process,
    isobaric_process,
    isothermal_process,
    linear_segment_overlay,
    process_to_diagram_coords,
    state_to_diagram_coords,
    to_json,
)
from core.fluids import SUPPORTED_FLUIDS, state_from_pair  # noqa: E402

# ---------------------------------------------------------------------
# Mapeo de unidades
# ---------------------------------------------------------------------


class TestFluprodiaUnitsMapping:
    """``FLUPRODIA_UNITS`` debe cubrir los 3 sistemas y respetar el quirk
    de pint en sistema Inglés para la entropía."""

    def test_si_keys(self) -> None:
        assert FLUPRODIA_UNITS["SI"] == {
            "T": "K",
            "p": "Pa",
            "h": "J/kg",
            "s": "J/kgK",
            "vol": "m^3/kg",
        }

    def test_tecnico_uses_bar_and_celsius(self) -> None:
        u = FLUPRODIA_UNITS["Técnico"]
        assert u["T"] == "°C"
        assert u["p"] == "bar"
        assert u["h"] == "kJ/kg"

    def test_ingles_entropy_uses_pint_safe_string(self) -> None:
        # Quirk documentado: Btu/lbR rompe pint; debe usarse Btu/(lb*degR).
        assert FLUPRODIA_UNITS["Inglés"]["s"] == "Btu/(lb*degR)"

    def test_all_three_systems_present(self) -> None:
        assert set(FLUPRODIA_UNITS) == {"SI", "Técnico", "Inglés"}


# ---------------------------------------------------------------------
# Rangos por defecto
# ---------------------------------------------------------------------


class TestDefaultRanges:
    """Hay un :class:`FluidRange` por cada fluido soportado, con valores
    físicamente plausibles."""

    def test_one_entry_per_supported_fluid(self) -> None:
        assert set(DEFAULT_RANGES) == set(SUPPORTED_FLUIDS)

    @pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
    def test_range_is_strictly_increasing(self, fluid: str) -> None:
        rng = DEFAULT_RANGES[fluid]
        assert isinstance(rng, FluidRange)
        assert rng.T_K_min < rng.T_K_max
        assert rng.p_Pa_min < rng.p_Pa_max
        assert rng.T_K_min > 0.0
        assert rng.p_Pa_min > 0.0


# ---------------------------------------------------------------------
# Construcción del diagrama — uno por fluido
# ---------------------------------------------------------------------


class TestBuildDiagram:
    """``build_diagram`` debe correr sin error para los 7 fluidos en
    sistema Técnico (cobertura mínima)."""

    @pytest.mark.parametrize("fluid", SUPPORTED_FLUIDS)
    def test_builds_for_each_fluid(self, fluid: str) -> None:
        spec = DiagramSpec(fluid=fluid, system="Técnico")
        diagram = build_diagram(spec)
        assert diagram is not None
        # Atributo público de fluprodia: nombre del fluido.
        # Es defensivo: si fluprodia cambia el atributo, este aserto guía.
        assert hasattr(diagram, "fluid")


class TestBuildDiagramValidations:
    """Validaciones de inputs antes de invocar a fluprodia."""

    def test_invalid_fluid_raises(self) -> None:
        with pytest.raises(ValueError, match="no soportado"):
            build_diagram(DiagramSpec(fluid="Helio4", system="Técnico"))

    def test_invalid_system_raises(self) -> None:
        with pytest.raises(ValueError, match="Sistema de unidades"):
            build_diagram(DiagramSpec(fluid="Water", system="MKS"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Procesos — fixtures de diagrama compartidos
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def water_diagram_tecnico():
    """Diagrama de agua en sistema Técnico, compartido por los tests
    de procesos para amortizar el costo de ``calc_isolines``."""
    return build_diagram(DiagramSpec(fluid="Water", system="Técnico"))


@pytest.fixture(scope="module")
def water_spec_tecnico() -> DiagramSpec:
    return DiagramSpec(fluid="Water", system="Técnico")


class TestIsentropicProcess:
    """Expansión isoentrópica de vapor: 80 bar, 500 °C → 0.1 bar."""

    def test_returns_dict_with_expected_keys(
        self, water_diagram_tecnico, water_spec_tecnico
    ) -> None:
        s_in = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5).s_J_per_kg_K
        proc = isentropic_process(
            water_diagram_tecnico,
            water_spec_tecnico,
            s_J_per_kg_K=s_in,
            p_start_Pa=80.0e5,
            p_end_Pa=0.1e5,
        )
        # Las keys son las de fluprodia.calc_individual_isoline:
        assert {"p", "T", "h", "s", "vol", "Q"} <= set(proc)

    def test_arrays_non_empty(self, water_diagram_tecnico, water_spec_tecnico) -> None:
        s_in = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5).s_J_per_kg_K
        proc = isentropic_process(
            water_diagram_tecnico,
            water_spec_tecnico,
            s_J_per_kg_K=s_in,
            p_start_Pa=80.0e5,
            p_end_Pa=0.1e5,
        )
        assert len(proc["p"]) > 1
        assert len(proc["h"]) == len(proc["p"])

    def test_pressure_monotonic_in_si(self, water_diagram_tecnico, water_spec_tecnico) -> None:
        s_in = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5).s_J_per_kg_K
        proc = isentropic_process(
            water_diagram_tecnico,
            water_spec_tecnico,
            s_J_per_kg_K=s_in,
            p_start_Pa=80.0e5,
            p_end_Pa=0.1e5,
        )
        p = proc["p"]
        # Expansión: la presión debe ir de 80 bar (8e6 Pa) a 0.1 bar (1e4 Pa),
        # estrictamente decreciente.
        assert p[0] > p[-1]
        assert np.all(np.diff(p) <= 0.0)

    def test_pressure_endpoints_in_si(self, water_diagram_tecnico, water_spec_tecnico) -> None:
        s_in = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5).s_J_per_kg_K
        proc = isentropic_process(
            water_diagram_tecnico,
            water_spec_tecnico,
            s_J_per_kg_K=s_in,
            p_start_Pa=80.0e5,
            p_end_Pa=0.1e5,
        )
        # Tolerancia laxa porque fluprodia muestrea ~100 puntos sobre la curva.
        assert proc["p"][0] == pytest.approx(80.0e5, rel=1e-3)
        assert proc["p"][-1] == pytest.approx(0.1e5, rel=1e-3)


class TestIsobaricProcess:
    """Calentamiento isobárico de agua a 10 bar, 100 °C → 300 °C."""

    def test_returns_si_arrays(self, water_diagram_tecnico, water_spec_tecnico) -> None:
        proc = isobaric_process(
            water_diagram_tecnico,
            water_spec_tecnico,
            p_Pa=10.0e5,
            t_start_K=100 + 273.15,
            t_end_K=300 + 273.15,
        )
        assert {"p", "T", "h", "s"} <= set(proc)
        # La presión SI debe rondar 10 bar = 1e6 Pa en todos los puntos.
        assert proc["p"][0] == pytest.approx(10.0e5, rel=1e-2)
        # La temperatura SI debe crecer (calentamiento).
        assert proc["T"][-1] > proc["T"][0]


class TestIsothermalProcess:
    """Compresión isotérmica de aire a 300 K, 1 bar → 10 bar."""

    def test_returns_si_arrays(self) -> None:
        spec = DiagramSpec(fluid="Air", system="Técnico")
        diagram = build_diagram(spec)
        proc = isothermal_process(
            diagram,
            spec,
            t_K=300.0,
            p_start_Pa=1.0e5,
            p_end_Pa=10.0e5,
        )
        assert {"p", "T", "h", "s"} <= set(proc)
        assert proc["p"][-1] > proc["p"][0]
        # La temperatura SI debe mantenerse ~300 K en todos los puntos.
        assert np.allclose(proc["T"], 300.0, rtol=1e-2)


# ---------------------------------------------------------------------
# Coordenadas de overlay
# ---------------------------------------------------------------------


class TestStateToDiagramCoords:
    """Conversión SI → coordenadas en unidades del sistema."""

    def test_tecnico_logph_returns_kJ_per_kg_and_bar(self) -> None:
        # Agua a 500 °C, 80 bar: h ≈ 3399.5 kJ/kg.
        state = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5)
        x, y = state_to_diagram_coords(state, "logph", "Técnico")
        # x = h en kJ/kg, y = p en bar.
        assert x == pytest.approx(3399.5, rel=5e-3)
        assert y == pytest.approx(80.0, rel=1e-3)

    def test_si_ts_returns_kelvin(self) -> None:
        state = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5)
        x, y = state_to_diagram_coords(state, "Ts", "SI")
        # y = T en K.
        assert y == pytest.approx(500 + 273.15, rel=1e-6)

    def test_invalid_diagram_type_raises(self) -> None:
        state = state_from_pair("Water", "TP", t=298.15, p=1.0e5)
        with pytest.raises(ValueError, match="Tipo de diagrama"):
            state_to_diagram_coords(state, "px", "Técnico")  # type: ignore[arg-type]


class TestProcessToDiagramCoords:
    """``process_to_diagram_coords`` debe devolver arrays alineados en
    las unidades del sistema."""

    def test_logph_tecnico(self, water_diagram_tecnico, water_spec_tecnico) -> None:
        s_in = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5).s_J_per_kg_K
        proc = isentropic_process(
            water_diagram_tecnico,
            water_spec_tecnico,
            s_J_per_kg_K=s_in,
            p_start_Pa=80.0e5,
            p_end_Pa=0.1e5,
        )
        x, y = process_to_diagram_coords(proc, "logph", "Técnico")
        assert x.shape == y.shape
        assert len(x) > 1
        # y = p en bar — los extremos deben rondar 80 y 0.1.
        assert y[0] == pytest.approx(80.0, rel=1e-2)
        assert y[-1] == pytest.approx(0.1, rel=1e-2)


# ---------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------


class TestLinearSegmentOverlay:
    """Helper que arma un overlay de 2 puntos a partir de StatePoints."""

    def test_overlay_keeps_si_values(self) -> None:
        s1 = state_from_pair("Water", "TP", t=500 + 273.15, p=80.0e5)
        s2 = state_from_pair("Water", "PX", p=0.1e5, x=0.9)
        ov = linear_segment_overlay(name="1→2", color="#ff0000", dash="dash", start=s1, end=s2)
        assert isinstance(ov, ProcessOverlay)
        assert ov.name == "1→2"
        assert len(ov.coords_si["p"]) == 2
        assert ov.coords_si["p"][0] == pytest.approx(80.0e5)
        assert ov.coords_si["p"][1] == pytest.approx(0.1e5)
        assert ov.coords_si["h"][0] == pytest.approx(s1.h_J_per_kg)


# ---------------------------------------------------------------------
# Tipos de diagrama
# ---------------------------------------------------------------------


class TestDiagramTypes:
    def test_supported_diagram_types_exact(self) -> None:
        assert SUPPORTED_DIAGRAM_TYPES == ("logph", "Ts", "hs", "plogv")

    def test_axis_map_covers_all_types(self) -> None:
        assert set(AXIS_MAP) == set(SUPPORTED_DIAGRAM_TYPES)


# ---------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------


class TestJsonRoundTrip:
    """``to_json`` / ``from_json`` deben preservar el diagrama de modo
    que se pueda dibujar / interrogar tras la recarga."""

    def test_roundtrip_water_tecnico(self, tmp_path, water_diagram_tecnico) -> None:
        path = tmp_path / "water_tecnico.json"
        to_json(water_diagram_tecnico, str(path))
        assert os.path.getsize(path) > 0
        reloaded = from_json(str(path))
        assert reloaded is not None
        assert hasattr(reloaded, "fluid")
