"""Helpers compartidos por los procedimientos LaTeX de transferencia de calor (Fase 8.1).

Reciben valores en SI y escriben los números en el sistema de unidades pedido
(:mod:`core.latex`); las magnitudes de :mod:`core.units_system` están elegidas
para que cada sustitución cierre sin factores en los tres sistemas
(``heat_capacity_rate``, ``inverse_length``, ``expansion_coefficient``…).
"""

from __future__ import annotations

from core.latex import latex_number, latex_paren, latex_quantity, latex_value
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

__all__ = ["dimensionless", "frac", "n", "numbered", "q", "sub", "sum_rows", "times"]


def q(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Número convertido + unidad."""
    return latex_quantity(value_si, kind, system, sig)


def n(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Solo el número convertido (para sustituir en una fórmula)."""
    return latex_value(value_si, kind, system, sig)


def dimensionless(x: float, sig: int = 4) -> str:
    """Un número sin unidad (Re, Nu, Pr, mL…)."""
    return latex_number(x, sig)


def frac(num: str, den: str) -> str:
    r"""``\frac{num}{den}``."""
    return rf"\frac{{{num}}}{{{den}}}"


def sub(a: str, b: str) -> str:
    """``a - b`` con ``b`` entre paréntesis si es negativo."""
    return rf"{a} - {latex_paren(b)}"


def times(*factors: str) -> str:
    r"""``a \cdot b \cdot c`` con los negativos entre paréntesis (salvo el primero)."""
    out = [factors[0], *(latex_paren(f) for f in factors[1:])]
    return r" \cdot ".join(out)


def sum_rows(terms: list[str], per_row: int = 3) -> str:
    r"""``a + b + c \\ &\quad + d + e``: una suma larga partida en renglones (para un
    paso de :func:`~core.latex.latex_chain`, que entra en el ancho de un celular).

    Con números ×10ⁿ (SI) van de a dos por renglón.
    """
    if any(r"\times" in t for t in terms):
        per_row = min(per_row, 2)
    rows = [terms[i : i + per_row] for i in range(0, len(terms), per_row)]
    parts = []
    for k, row in enumerate(rows):
        body = " + ".join(latex_paren(t) if j or k else t for j, t in enumerate(row))
        parts.append(body if k == 0 else rf"\\ &\quad + {body}")
    return " ".join(parts)


def numbered(steps: list[ProcedureStep]) -> list[ProcedureStep]:
    """Antepone el número de paso al título."""
    return [ProcedureStep(f"{k}. {s.title}", s.text, s.latex) for k, s in enumerate(steps, 1)]
