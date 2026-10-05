"""Helpers para escribir cantidades en LaTeX (KaTeX, el motor de ``st.latex``).

Compartidos por los módulos que arman procedimientos didácticos
(:mod:`core.state_report`, :mod:`core.isentropic`). Reciben valores en SI
y los muestran en el sistema de unidades pedido, así ningún paso queda
"hardcodeado" en un sistema. No importa Streamlit.

Ancho: una ecuación en modo display no se parte sola, y en un celular
(~320 px útiles) KaTeX la deja con scroll horizontal, que el alumno no
suele notar. Por eso las sustituciones con números se escriben con
:func:`latex_chain`, un renglón por igualdad.
"""

from __future__ import annotations

from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label


def latex_number(value: float, sig: int = 5) -> str:
    """Número para LaTeX: ``1.0142`` o ``1.0142\\times 10^{5}``."""
    if value == 0.0:
        return "0"
    s = f"{value:.{sig}g}"
    if "e" in s:
        mantissa, exponent = s.split("e")
        return rf"{mantissa}\times 10^{{{int(exponent)}}}"
    return s


def latex_unit(label: str) -> str:
    """Etiqueta de unidad (``'kJ/(kg·K)'``, ``'°C'``, ``'m³/kg'``) → LaTeX."""
    tex = (
        label.replace("·", r"\cdot ")
        .replace("°", r"{}^{\circ}")
        .replace("³", "^{3}")
        .replace("²", "^{2}")
    )
    return rf"\mathrm{{{tex}}}"


def latex_value(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Solo el número convertido a ``system`` (para sustituir en una fórmula)."""
    return latex_number(convert_from_si(value_si, kind, system), sig)


def latex_quantity(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Número convertido a ``system`` + su unidad, en LaTeX."""
    return rf"{latex_value(value_si, kind, system, sig)}\ {latex_unit(unit_label(kind, system))}"


def text_quantity(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Número convertido + unidad en texto plano (para narrativas en markdown)."""
    value = convert_from_si(value_si, kind, system)
    return f"{value:.{sig}g} {unit_label(kind, system)}"


def latex_paren(tex: str) -> str:
    """Encierra entre paréntesis un número negativo que va después de un signo.

    ``a - b`` con ``b = -3`` se escribe ``a - (-3)`` y no ``a - -3``.
    """
    return f"({tex})" if tex.startswith("-") else tex


def latex_chain(lhs: str, *steps: str, relation: str = "=") -> str:
    r"""Cadena ``lhs = paso₁ = paso₂ = …`` con un renglón por paso.

    Arma un ``aligned`` alineado en el signo::

        h &= h_f + x\,(h_g - h_f) \\
          &= 762.51 + 0.5\,(2777.1 - 762.51) \\
          &= 1769.8\ \mathrm{kJ/kg}

    ``relation`` es el signo del primer paso (``\approx``, ``\le``…);
    los siguientes son igualdades. Sin pasos devuelve ``lhs``.
    """
    if not steps:
        return lhs
    first, *rest = steps
    lines = [rf"{lhs} &{relation} {first}", *(rf"&= {step}" for step in rest)]
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"
