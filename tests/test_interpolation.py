"""Tests de :mod:`core.interpolation` — Fase 1.1.

Cubre:
- Lineal: casos triviales, nodos exactos, validación de inputs.
- Lineal sobre tabla: tabla de agua saturada (Cengel A-4) contra CoolProp
  con tolerancia 0.2 %.
- Bilineal: esquinas, centro, degeneración a lineal sobre un eje.
- Bilineal sobre tabla: numpy, pandas y lista de listas dan mismo
  resultado; vapor sobrecalentado contra CoolProp con tolerancia 1 %.
- Procedimiento: los pasos contienen LaTeX y narrativa no vacíos.
"""

from __future__ import annotations

import CoolProp.CoolProp as cp
import numpy as np
import pandas as pd
import pytest

from core.interpolation import (
    InterpolationResult,
    bilinear,
    bilinear_from_table,
    linear,
    linear_from_table,
)


class TestLinearBasics:
    def test_midpoint(self) -> None:
        result = linear(5.0, 0.0, 10.0, 0.0, 100.0)
        assert isinstance(result, InterpolationResult)
        assert result.value == pytest.approx(50.0)
        assert result.steps.kind == "linear"
        assert not result.steps.exact_node

    def test_exact_x0_returns_exact_node(self) -> None:
        result = linear(0.0, 0.0, 10.0, 5.0, 105.0)
        assert result.value == pytest.approx(5.0)
        assert result.steps.exact_node

    def test_exact_x1_returns_exact_node(self) -> None:
        result = linear(10.0, 0.0, 10.0, 5.0, 105.0)
        assert result.value == pytest.approx(105.0)
        assert result.steps.exact_node

    def test_negative_slope(self) -> None:
        result = linear(2.5, 0.0, 5.0, 100.0, 0.0)
        assert result.value == pytest.approx(50.0)

    def test_reversed_node_order(self) -> None:
        # x0 > x1 también debe funcionar (x_min/x_max se normaliza).
        result = linear(5.0, 10.0, 0.0, 100.0, 0.0)
        assert result.value == pytest.approx(50.0)

    def test_x0_equals_x1_raises(self) -> None:
        with pytest.raises(ValueError, match="distintos"):
            linear(5.0, 3.0, 3.0, 0.0, 100.0)

    def test_extrapolation_blocked_by_default(self) -> None:
        with pytest.raises(ValueError, match="extrapolar"):
            linear(15.0, 0.0, 10.0, 0.0, 100.0)

    def test_extrapolation_opt_in(self) -> None:
        result = linear(15.0, 0.0, 10.0, 0.0, 100.0, allow_extrapolation=True)
        assert result.value == pytest.approx(150.0)


class TestLinearFromTable:
    XS = [0.0, 10.0, 20.0, 30.0]
    YS_QUADRATIC = [0.0, 100.0, 400.0, 900.0]  # y = x^2

    def test_interpolate_between_nodes(self) -> None:
        result = linear_from_table(5.0, self.XS, self.YS_QUADRATIC)
        # Lineal entre (0,0) y (10,100): 50. No coincide con x^2=25 — eso es
        # el error de interpolación, no del algoritmo.
        assert result.value == pytest.approx(50.0)

    def test_at_first_node(self) -> None:
        result = linear_from_table(0.0, self.XS, self.YS_QUADRATIC)
        assert result.value == pytest.approx(0.0)
        assert result.steps.exact_node

    def test_at_intermediate_node(self) -> None:
        result = linear_from_table(20.0, self.XS, self.YS_QUADRATIC)
        assert result.value == pytest.approx(400.0)
        assert result.steps.exact_node

    def test_at_last_node(self) -> None:
        result = linear_from_table(30.0, self.XS, self.YS_QUADRATIC)
        assert result.value == pytest.approx(900.0)
        assert result.steps.exact_node

    def test_unsorted_xs_raises(self) -> None:
        with pytest.raises(ValueError, match="estrictamente creciente"):
            linear_from_table(5.0, [10.0, 0.0, 20.0], [100.0, 0.0, 400.0])

    def test_decreasing_xs_raises_with_invert_hint(self) -> None:
        with pytest.raises(ValueError, match="[Ii]nvertí"):
            linear_from_table(5.0, [30.0, 20.0, 10.0], [900.0, 400.0, 100.0])

    def test_duplicate_xs_raises(self) -> None:
        with pytest.raises(ValueError, match="estrictamente creciente"):
            linear_from_table(5.0, [0.0, 10.0, 10.0, 20.0], [0.0, 100.0, 100.0, 400.0])

    def test_extrapolation_blocked(self) -> None:
        with pytest.raises(ValueError, match="extrapolar"):
            linear_from_table(100.0, [0.0, 10.0], [0.0, 100.0])

    def test_extrapolation_opt_in_above(self) -> None:
        result = linear_from_table(15.0, [0.0, 10.0], [0.0, 100.0], allow_extrapolation=True)
        assert result.value == pytest.approx(150.0)

    def test_extrapolation_opt_in_below(self) -> None:
        result = linear_from_table(-5.0, [0.0, 10.0], [0.0, 100.0], allow_extrapolation=True)
        assert result.value == pytest.approx(-50.0)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="mismo tamaño"):
            linear_from_table(5.0, [0.0, 10.0, 20.0], [0.0, 100.0])

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="al menos 2 nodos"):
            linear_from_table(5.0, [10.0], [100.0])


class TestLinearVsSaturatedWaterTable:
    """h_f de agua saturada (valores de Cengel A-4)."""

    T_C = [100.0, 110.0, 120.0, 130.0]
    HF_KJ = [419.06, 461.42, 503.81, 546.38]

    def test_115C_close_to_coolprop(self) -> None:
        # Tabla con saltos de 20 °C; interpolo en el punto medio 115 °C.
        result = linear_from_table(115.0, [100.0, 120.0], [419.06, 503.81])
        T_K = 115.0 + 273.15
        coolprop_hf_kj = cp.PropsSI("H", "T", T_K, "Q", 0.0, "Water") / 1000
        rel_err = abs(result.value - coolprop_hf_kj) / coolprop_hf_kj
        assert rel_err < 0.002  # 0.2 %


class TestBilinearBasics:
    CORNERS = dict(z00=10.0, z01=20.0, z10=30.0, z11=40.0)

    def test_corner_z00(self) -> None:
        result = bilinear(0.0, 0.0, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx(10.0)
        assert result.steps.exact_node

    def test_corner_z01(self) -> None:
        result = bilinear(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx(20.0)

    def test_corner_z10(self) -> None:
        result = bilinear(1.0, 0.0, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx(30.0)

    def test_corner_z11(self) -> None:
        result = bilinear(1.0, 1.0, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx(40.0)

    def test_center_is_average_of_corners(self) -> None:
        result = bilinear(0.5, 0.5, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx((10 + 20 + 30 + 40) / 4)

    def test_degenerate_to_linear_in_y_at_x0(self) -> None:
        # En x = x0, debe degenerar a lineal en y entre z00 y z01.
        result = bilinear(0.0, 0.5, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx(15.0)

    def test_degenerate_to_linear_in_x_at_y0(self) -> None:
        # En y = y0, debe degenerar a lineal en x entre z00 y z10.
        result = bilinear(0.5, 0.0, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.value == pytest.approx(20.0)

    def test_intermediate_steps_recorded(self) -> None:
        result = bilinear(0.5, 0.5, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)
        assert result.steps.z_query_y0 == pytest.approx(20.0)  # entre z00=10, z10=30
        assert result.steps.z_query_y1 == pytest.approx(30.0)  # entre z01=20, z11=40

    def test_x0_equals_x1_raises(self) -> None:
        with pytest.raises(ValueError, match="x0 y x1"):
            bilinear(0.5, 0.5, 1.0, 1.0, 0.0, 1.0, **self.CORNERS)

    def test_y0_equals_y1_raises(self) -> None:
        with pytest.raises(ValueError, match="y0 e y1"):
            bilinear(0.5, 0.5, 0.0, 1.0, 1.0, 1.0, **self.CORNERS)

    def test_extrapolation_blocked(self) -> None:
        with pytest.raises(ValueError, match="extrapolar"):
            bilinear(5.0, 0.5, 0.0, 1.0, 0.0, 1.0, **self.CORNERS)


class TestBilinearFromTable:
    XS = [0.0, 10.0, 20.0]
    YS = [0.0, 5.0, 10.0]
    # z(x, y) = x + y → bilineal exacta en una grilla lineal.
    ZS_LIST = [
        [0.0, 5.0, 10.0],
        [10.0, 15.0, 20.0],
        [20.0, 25.0, 30.0],
    ]

    def test_at_node(self) -> None:
        result = bilinear_from_table(10.0, 5.0, self.XS, self.YS, self.ZS_LIST)
        assert result.value == pytest.approx(15.0)

    def test_interior_linear_field_exact(self) -> None:
        result = bilinear_from_table(5.0, 2.5, self.XS, self.YS, self.ZS_LIST)
        assert result.value == pytest.approx(7.5)

    def test_accepts_numpy(self) -> None:
        result = bilinear_from_table(5.0, 2.5, self.XS, self.YS, np.array(self.ZS_LIST))
        assert result.value == pytest.approx(7.5)

    def test_accepts_dataframe(self) -> None:
        df = pd.DataFrame(self.ZS_LIST, index=self.XS, columns=self.YS)
        result = bilinear_from_table(5.0, 2.5, self.XS, self.YS, df)
        assert result.value == pytest.approx(7.5)

    def test_shape_mismatch_raises(self) -> None:
        bad_zs = [[1.0, 2.0]]
        with pytest.raises(ValueError, match="shape"):
            bilinear_from_table(5.0, 2.5, self.XS, self.YS, bad_zs)

    def test_decreasing_ys_raises_invert_hint(self) -> None:
        with pytest.raises(ValueError, match="[Ii]nvertí"):
            bilinear_from_table(5.0, 5.0, self.XS, [10.0, 5.0, 0.0], self.ZS_LIST)

    def test_extrapolation_blocked_in_x(self) -> None:
        with pytest.raises(ValueError, match="extrapolar"):
            bilinear_from_table(100.0, 5.0, self.XS, self.YS, self.ZS_LIST)

    def test_extrapolation_blocked_in_y(self) -> None:
        with pytest.raises(ValueError, match="extrapolar"):
            bilinear_from_table(5.0, 100.0, self.XS, self.YS, self.ZS_LIST)


class TestBilinearVsCoolPropSuperheatedSteam:
    """Tabla 2×2 de vapor sobrecalentado de agua, error < 1 %."""

    def test_interp_h_at_250C_3bar(self) -> None:
        T_corners_C = [200.0, 300.0]
        P_corners_bar = [1.0, 5.0]
        zs = np.empty((2, 2))
        for i, T_C in enumerate(T_corners_C):
            for j, P_bar in enumerate(P_corners_bar):
                T_K = T_C + 273.15
                P_Pa = P_bar * 1e5
                zs[i, j] = cp.PropsSI("H", "T", T_K, "P", P_Pa, "Water") / 1000.0

        result = bilinear_from_table(250.0, 3.0, T_corners_C, P_corners_bar, zs)

        T_K_q = 250.0 + 273.15
        P_Pa_q = 3.0 * 1e5
        h_exact_kj = cp.PropsSI("H", "T", T_K_q, "P", P_Pa_q, "Water") / 1000.0

        rel_err = abs(result.value - h_exact_kj) / h_exact_kj
        assert rel_err < 0.01


class TestProcedureSteps:
    def test_linear_steps_have_latex_and_narrative(self) -> None:
        result = linear(5.0, 0.0, 10.0, 0.0, 100.0)
        assert "y_0" in result.steps.formula_latex
        assert "frac" in result.steps.formula_latex
        # El sustituido debe contener números.
        assert any(ch.isdigit() for ch in result.steps.substituted_latex)
        assert result.steps.narrative_es
        assert "lineal" in result.steps.narrative_es.lower()

    def test_bilinear_steps_have_intermediate_values_and_narrative(self) -> None:
        result = bilinear(0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 10.0, 20.0, 30.0, 40.0)
        assert result.steps.z_query_y0 is not None
        assert result.steps.z_query_y1 is not None
        # La narrativa explica los dos pasos.
        narrative_lower = result.steps.narrative_es.lower()
        assert "bilineal" in narrative_lower
        assert "dos pasos" in narrative_lower or "(1)" in result.steps.narrative_es

    def test_exact_node_narrative_indicates_no_interpolation(self) -> None:
        result = linear(0.0, 0.0, 10.0, 5.0, 105.0)
        assert result.steps.exact_node
        narrative_lower = result.steps.narrative_es.lower()
        assert "coincide" in narrative_lower or "sin interpolación" in narrative_lower
