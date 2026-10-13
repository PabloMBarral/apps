"""Helpers compartidos por los procedimientos LaTeX de transferencia de calor (Fase 8.1).

Reciben valores en SI y escriben los números en el sistema de unidades pedido
(:mod:`core.latex`); las magnitudes de :mod:`core.units_system` están elegidas
para que cada sustitución cierre sin factores en los tres sistemas
(``heat_capacity_rate``, ``inverse_length``, ``expansion_coefficient``…).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.latex import latex_chain, latex_number, latex_paren, latex_quantity, latex_value
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem

__all__ = [
    "ROW",
    "Pack",
    "budget",
    "chain",
    "dimensionless",
    "flush",
    "frac",
    "n",
    "numbered",
    "pack",
    "prod",
    "q",
    "sub",
    "sum_rows",
    "times",
    "vis",
]


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


# ---------------------------------------------------------------------
# Ancho de los renglones (Fase 11.1; los usan también la ebullición y la condensación)
# ---------------------------------------------------------------------

#: Un renglón de ``latex_chain`` en «caracteres» de :func:`vis` (324 px son unos 33 dígitos
#: de 9,7 px, menos el signo = y un margen).
ROW = 29.0

_NARROW_CMDS = {",": 0.2, " ": 0.3, ";": 0.3, "quad": 1.0, "qquad": 2.0, "left": 0.0, "right": 0.0}
_FUNCS = {"sin", "cos", "tan", "cot", "ln", "exp"}


def _group(tex: str, i: int) -> tuple[str, int]:
    """El contenido del grupo ``{…}`` que empieza en ``i`` y la posición siguiente."""
    if i >= len(tex) or tex[i] != "{":
        return (tex[i] if i < len(tex) else ""), i + 1
    depth = 0
    for j in range(i, len(tex)):
        if tex[j] == "{":
            depth += 1
        elif tex[j] == "}":
            depth -= 1
            if depth == 0:
                return tex[i + 1 : j], j + 1
    return tex[i + 1 :], len(tex)


def vis(tex: str) -> float:
    """El ancho aproximado de un renglón de LaTeX, en «caracteres» (un dígito = 1).

    Alcanza para decidir si un renglón se parte: una fracción cuenta lo que su parte más
    ancha, un exponente o un subíndice el 70 %, y los signos +, − e = con sus espacios, 2.
    """
    total = 0.0
    i = 0
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            j = i + 1
            if j < len(tex) and not tex[j].isalpha():
                total += _NARROW_CMDS.get(tex[j], 0.5)
                i = j + 1
                continue
            while j < len(tex) and tex[j].isalpha():
                j += 1
            cmd = tex[i + 1 : j]
            if cmd in ("frac", "tfrac"):
                a, j = _group(tex, j)
                b, j = _group(tex, j)
                total += max(vis(a), vis(b)) * (0.75 if cmd == "tfrac" else 1.0) + 1.0
            elif cmd in ("text", "mathrm", "operatorname"):
                a, j = _group(tex, j)
                total += float(len(a))
            elif cmd == "sqrt":
                a, j = _group(tex, j)
                total += vis(a) + 1.0
            elif cmd == "times":
                total += 1.6
            elif cmd == "cdot":
                total += 1.4
            elif cmd in _FUNCS:
                total += len(cmd) + 0.5
            elif cmd in _NARROW_CMDS:
                total += _NARROW_CMDS[cmd]
            else:
                total += 1.0
            i = j
            continue
        if ch in "^_":
            a, i = _group(tex, i + 1)
            total += 0.7 * vis(a)
            continue
        if ch == "=":
            total += 2.7
        elif ch in "+-":
            total += 2.4
        elif ch in "()":
            total += 0.8
        elif ch in ".,":
            total += 0.55
        elif ch in "{}& ":
            pass
        else:
            total += 1.0
        i += 1
    return total


@dataclass(frozen=True)
class Pack:
    """Un renglón de números que :func:`chain` parte si no entra: ``first`` y después cada
    ``(operador, término)``."""

    first: str
    rest: tuple[tuple[str, str], ...] = ()


def pack(p: Pack, budget: float) -> str:
    r"""Junta los términos y pasa a otro renglón (``\\ &\quad``) cuando no entran."""
    out, width = p.first, vis(p.first)
    for op, term in p.rest:
        piece = f" {op} {term}"
        w = vis(piece)
        if width + w > budget:
            out += rf" \\ &\quad {op} {term}"
            width = 2.0 + w
        else:
            out += piece
            width += w
    return out


def budget(lhs: str) -> float:
    """Lo que queda para los números en un renglón con ese lado izquierdo."""
    return ROW - vis(lhs)


def chain(lhs: str, *steps: str | Pack) -> str:
    """:func:`~core.latex.latex_chain` con los renglones de números partidos si no entran (en
    un ``aligned`` el lado izquierdo se suma al ancho de cada renglón)."""
    room = budget(lhs)
    return latex_chain(lhs, *(st if isinstance(st, str) else pack(st, room) for st in steps))


def flush(lhs: str, *steps: str | Pack) -> str:
    """Como :func:`chain`, con los renglones contra el margen: para un lado izquierdo ancho
    (una fracción), que en un ``aligned`` se sumaría a cada renglón."""
    out = [st if isinstance(st, str) else pack(st, ROW) for st in steps]
    rows = [rf"&{lhs} = {out[0]}", *(rf"&= {st}" for st in out[1:])]
    return r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"


def prod(*factors: str) -> Pack:
    r"""``a \cdot b \cdot c`` (los negativos entre paréntesis), partido si no entra."""
    return Pack(factors[0], tuple((r"\cdot", latex_paren(f)) for f in factors[1:]))
