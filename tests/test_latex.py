"""Tests de los helpers de :mod:`core.latex` para ecuaciones angostas.

En un celular hay ~320 px útiles y KaTeX no parte una ecuación en
renglones: las sustituciones con números se escriben con
:func:`latex_chain` (un renglón por igualdad) y los negativos que van
después de un signo, entre paréntesis (:func:`latex_paren`).
"""

from __future__ import annotations

import pytest

from core.fluids import state_from_pair
from core.interpolation import bilinear, linear_from_table
from core.isentropic import isentropic_steps, pump_direct, suggested_device_inputs
from core.latex import latex_chain, latex_is_wide, latex_paren


def test_wide_numbers_are_the_si_powers_and_the_negatives() -> None:
    # Los que alargan un renglón en el celular: ×10ⁿ (J/kg y Pa en SI) y negativos.
    assert latex_is_wide(r"3.5831\times 10^{6}", "1.9")
    assert latex_is_wide("2706.2", "-0.049779")
    assert not latex_is_wide("2706.2", "503.81", "0.5")
    assert not latex_is_wide()


def test_si_pump_and_lever_rule_split_the_subtraction() -> None:
    """Agua líquida bombeada en SI: h₂ = h₁ + (h₂s − h₁)/η_s con ×10ⁿ pasa a dos renglones."""
    state_in = state_from_pair("Water", "PX", p=10e3, x=0.0)
    result = pump_direct(fluid="Water", state_in=state_in, p_out_Pa=15e6, eta_s=0.85)
    tex = isentropic_steps(result, "SI").substituted_latex
    assert r"h_2 &= 1.9181\times 10^{5} \\ &\quad + \dfrac" in tex
    assert r"h_2 &= 191.81 + \dfrac" in isentropic_steps(result, "Técnico").substituted_latex


class TestLatexChain:
    def test_one_row_per_step(self) -> None:
        assert latex_chain("h", "a", "b", "c") == (
            r"\begin{aligned}h &= a \\ &= b \\ &= c\end{aligned}"
        )

    def test_relation_applies_to_the_first_step(self) -> None:
        tex = latex_chain("h", "a", "b", relation=r"\approx")
        assert tex == r"\begin{aligned}h &\approx a \\ &= b\end{aligned}"

    def test_without_steps_returns_lhs(self) -> None:
        assert latex_chain("x") == "x"


@pytest.mark.parametrize(
    ("tex", "expected"),
    [
        ("3", "3"),
        ("0", "0"),
        ("-3", "(-3)"),
        (r"-1.2\times 10^{-5}", r"(-1.2\times 10^{-5})"),
    ],
)
def test_latex_paren(tex: str, expected: str) -> None:
    assert latex_paren(tex) == expected


class TestNegativeNumbersInProcedures:
    """``a - -3`` es ambiguo para el alumno: se escribe ``a - (-3)``."""

    def test_linear_with_negative_nodes(self) -> None:
        # Tabla de refrigerante en °C bajo cero.
        tex = linear_from_table(-5.0, [-10.0, 0.0, 10.0], [-2.5, 1.0, 3.0]).steps.substituted_latex
        assert r"\frac{-5 - (-10)}{0 - (-10)}" in tex
        assert "(1 - (-2.5))" in tex
        assert "- -" not in tex

    def test_bilinear_with_negative_values(self) -> None:
        result = bilinear(-5.0, 1.5, -10.0, 0.0, 1.0, 2.0, -4.0, -3.0, 2.0, 3.0)
        tex = result.steps.substituted_latex
        assert "- (-10)" in tex and "- (-4)" in tex
        assert "- -" not in tex

    def test_negative_extrapolation_fraction(self) -> None:
        result = linear_from_table(95.0, [100.0, 110.0], [1.0, 2.0], allow_extrapolation=True)
        assert r"\cdot (-0.5)" in result.steps.substituted_latex

    def test_liquid_air_pump(self) -> None:
        # Referencia NBP del aire: h < 0 en el líquido subenfriado de la entrada.
        d = suggested_device_inputs("Air", "pump")
        state_in = state_from_pair("Air", d.pair_in, **d.inlet)
        result = pump_direct(fluid="Air", state_in=state_in, p_out_Pa=d.p_out_Pa, eta_s=d.eta_s)
        tex = isentropic_steps(result, "Técnico").substituted_latex
        assert "- (-1.3874)" in tex
        assert "- -" not in tex


def test_reference_state_noise_is_shown_as_zero() -> None:
    """Aire líquido saturado a 1 atm es el estado de referencia (h = 0):
    CoolProp devuelve ~1e-8 J/kg y el paso mostraba −1.2352×10⁻⁵ kJ/kg."""
    state_in = state_from_pair("Air", "PX", p=101_325.0, x=0.0)
    result = pump_direct(fluid="Air", state_in=state_in, p_out_Pa=2.0e6, eta_s=0.75)
    tex = isentropic_steps(result, "Técnico").substituted_latex
    assert r"h_1 &= 0\ \mathrm{kJ/kg}" in tex
    assert r"\times 10^{-5}" not in tex
