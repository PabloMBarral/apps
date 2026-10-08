"""Procedimiento de la combustión «como en el pizarrón» — Fase 5.

Arma los pasos del expansor 🔬 Procedimiento en LaTeX, en el sistema de
unidades activo, con la notación del vademecum §16 (Cengel §15-1 a §15-7 y
§16-2):

- **combustión** (:func:`combustion_steps`): los átomos del combustible, el
  comburente, el O₂ y el aire teóricos, el exceso, los productos por balance
  de cada elemento, las fracciones de los humos, los rocíos, el PCS y el PCI,
  la llama adiabática (completa y con disociación, con K_p), el calor con los
  humos a T y el segundo principio;
- **análisis de humos** (:func:`flue_gas_steps`): los balances del Orsat
  (Cengel 15-4) o la λ del O₂ medido, la pérdida por CO y el agua que
  condensa.

Las cantidades van por kmol de combustible (por mol en SI) o por kg para un
análisis elemental. Las entalpías molares se escriben enteras (como las
tablas); las ecuaciones con números van con :func:`core.latex.latex_chain`
(una igualdad por renglón) y se cortan antes de un signo cuando hace falta,
para que entren en un celular. No importa Streamlit.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from core.combustion.combustion import CombustionResult, FlueGasResult
from core.combustion.fuels import Fuel, component_info
from core.combustion.stoichiometry import (
    AIR_SPEC_KINDS,
    P_NORMAL_PA,
    T_NORMAL_K,
    Stoichiometry,
    products,
)
from core.combustion.thermo import (
    P_STANDARD_PA,
    R_U,
    atomic_mass,
    species,
    water_hfg_J_per_mol,
)
from core.latex import (
    latex_chain,
    latex_is_wide,
    latex_number,
    latex_paren,
    latex_quantity,
    latex_unit,
)
from core.psychrometrics import saturation_pressure
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si, unit_label

__all__ = ["combustion_steps", "flue_gas_steps", "species_latex"]

_EH: QuantityKind = "specific_enthalpy"
_MH: QuantityKind = "molar_enthalpy"
_MS: QuantityKind = "molar_entropy"
_T: QuantityKind = "temperature"
_P: QuantityKind = "pressure"

# Productos en el orden de las tablas.
_ORDER = ("CO2", "CO", "H2O", "H2", "SO2", "O2", "N2", "Ar", "OH", "H", "O", "NO", "N")


# ---------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------


def species_latex(key: str) -> str:
    r"""Fórmula de una especie en LaTeX: ``C8H18(l)`` → ``\mathrm{C_{8}H_{18}}(\ell)``."""
    liquid = key.endswith("(l)")
    body = key[:-3] if liquid else key
    prefix = ""
    if body[:2] in ("n-", "i-"):
        prefix, body = body[0] + "{-}", body[2:]
    body = re.sub(r"(\d+)", r"_{\1}", body)
    tex = rf"\mathrm{{{prefix}{body}}}"
    return tex + (r"(\ell)" if liquid else "")


def _sub(key: str) -> str:
    """Subíndice de una especie: ``n_{\\mathrm{CO_{2}}}``."""
    return species_latex(key)


def _c(value: float, sig: int = 4) -> str:
    """Un coeficiente (moles por unidad de combustible) o una fracción."""
    return latex_number(value, sig)


def _int(value: float) -> str:
    r"""Entero con separador de miles fino (``-393\,510``), como las tablas."""
    s = f"{abs(value):,.0f}".replace(",", r"\,")
    return ("-" if value < 0 else "") + s


def _hm(value_si: float, system: UnitSystem) -> str:
    """Una entalpía molar (sin unidad): entera si es grande."""
    v = convert_from_si(value_si, _MH, system)
    return _int(v) if abs(v) >= 1000.0 else latex_number(v, 4)


def _hmq(value_si: float, system: UnitSystem) -> str:
    return rf"{_hm(value_si, system)}\ {latex_unit(unit_label(_MH, system))}"


def _sm(value_si: float, system: UnitSystem) -> str:
    return latex_number(convert_from_si(value_si, _MS, system), 5)


def _smq(value_si: float, system: UnitSystem) -> str:
    return rf"{_sm(value_si, system)}\ {latex_unit(unit_label(_MS, system))}"


def _q(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    return latex_quantity(value_si, kind, system, sig)


def _degC(T_K: float) -> str:
    return f"{T_K - 273.15:.4g} °C"


def _pn(p_Pa: float, system: UnitSystem, sig: int = 5) -> str:
    """Una presión (sin unidad) en el sistema."""
    return latex_number(convert_from_si(p_Pa, _P, system), sig)


def _T_user(T_K: float, system: UnitSystem) -> str:
    """Una temperatura (sin unidad) en el sistema."""
    return latex_number(convert_from_si(T_K, _T, system), 5)


def _M_fuel(fuel: Fuel) -> str:
    """Masa molar del combustible en kg/kmol."""
    assert fuel.M_kg_per_mol is not None
    return latex_number(fuel.M_kg_per_mol * 1000, 5)


def _basis(fuel: Fuel, system: UnitSystem) -> str:
    """«por kmol de combustible» (o mol, lbmol, kg, lb) en texto."""
    if fuel.per_mol:
        unit = {"SI": "mol", "Técnico": "kmol", "Inglés": "lbmol"}[system]
    else:
        unit = "lb" if system == "Inglés" else "kg"
    return f"por {unit} de combustible"


def _scale(fuel: Fuel, system: UnitSystem) -> float:
    """Factor de los moles por unidad: por kg se muestran kmol/kg (o lbmol/lb), salvo en SI."""
    return 1.0 if fuel.per_mol or system == "SI" else 1e-3


def _mole_unit(fuel: Fuel, system: UnitSystem) -> str:
    if fuel.per_mol:
        return ""
    unit = {"SI": "mol/kg", "Técnico": "kmol/kg", "Inglés": "lbmol/lb"}[system]
    return rf"\ {latex_unit(unit)}"


def _n(value: float, fuel: Fuel, system: UnitSystem, sig: int = 4) -> str:
    """Moles por unidad de combustible en las unidades del sistema."""
    return _c(value * _scale(fuel, system), sig)


def _nq(value: float, fuel: Fuel, system: UnitSystem, sig: int = 4) -> str:
    return _n(value, fuel, system, sig) + _mole_unit(fuel, system)


def _E(value_per_basis: float, fuel: Fuel, system: UnitSystem) -> str:
    """Una energía por unidad de combustible: molar (especies) o por kg (análisis)."""
    if fuel.per_mol:
        return _hm(value_per_basis, system)
    return latex_number(convert_from_si(value_per_basis, _EH, system), 5)


def _Eq(value_per_basis: float, fuel: Fuel, system: UnitSystem) -> str:
    if fuel.per_mol:
        return _hmq(value_per_basis, system)
    return _q(value_per_basis, _EH, system)


def _lines(terms: Sequence[str], per_line: int = 2) -> str:
    r"""Suma de términos con ``per_line`` por renglón (``\\ &\quad +`` entre renglones)."""
    chunks = [" + ".join(terms[k : k + per_line]) for k in range(0, len(terms), per_line)]
    return r" \\ &\quad + ".join(chunks)


def _sum_chain(lhs: str, terms: Sequence[str], result: str, per_line: int = 2) -> str:
    """``lhs = t₁ + t₂ … = resultado`` con los términos en renglones cortos.

    Con números con ×10ⁿ van de a dos por renglón (como mucho).
    """
    if len(terms) == 1:
        return latex_chain(lhs, terms[0], result) if terms[0] != result else f"{lhs} = {result}"
    if any(r"\times" in t for t in terms):
        per_line = min(per_line, 2)
    return latex_chain(lhs, _lines(terms, per_line), result)


def _wrap(head: str, op: str, tail: str) -> str:
    r"""``head op tail``; con números anchos (×10ⁿ o negativos), ``op tail`` en otro renglón."""
    if latex_is_wide(head, tail):
        return rf"{head} \\ &\quad {op} {tail}"
    return f"{head} {op} {tail}"


def _ordered(moles: dict[str, float]) -> list[tuple[str, float]]:
    return [(k, moles[k]) for k in _ORDER if moles.get(k, 0.0) > 0.0]


def _stoich_dry_moles(s: Stoichiometry) -> float:
    """Humos secos con λ = 1 y combustión completa: n_s,t."""
    prods, _ = products(s.fuel, s.air, s.air_theoretical)
    return sum(v for k, v in prods.items() if k != "H2O")


def _mol_name(system: UnitSystem) -> str:
    return {"SI": "mol", "Técnico": "kmol", "Inglés": "lbmol"}[system]


# ---------------------------------------------------------------------
# Pasos compartidos
# ---------------------------------------------------------------------


def _fuel_step(fuel: Fuel, system: UnitSystem) -> ProcedureStep:
    e = fuel.elements()
    present = [el for el in ("C", "H", "O", "N", "S") if e[el] > 0.0]
    latex: list[str] = []
    if fuel.per_mol:
        comps = fuel.components
        if len(comps) == 1:
            key = comps[0][0]
            atoms = r",\ ".join(rf"n_\mathrm{{{el}}} = {_c(e[el])}" for el in present)
            latex.append(rf"\begin{{aligned}}&{species_latex(key)}: \\ &{atoms}\end{{aligned}}")
            info = component_info(key)
            latex.append(
                rf"M_\mathrm{{comb}} = {latex_number(info.M_kg_per_mol * 1000, 5)}"
                r"\ \mathrm{kg/kmol}"
            )
            text = (
                f"Átomos de cada elemento en un {_mol_name(system)} de {info.name} (vademecum "
                "§16.2):"
            )
        else:
            for el in present:
                terms = [
                    rf"{_c(y)} \cdot {_c(component_info(k).atoms[el])}"
                    for k, y in comps
                    if component_info(k).atoms.get(el, 0.0) > 0.0
                ]
                latex.append(_sum_chain(rf"n_\mathrm{{{el}}}", terms, _c(e[el]), per_line=3))
            terms = [
                rf"{_c(y)} \cdot {latex_number(component_info(k).M_kg_per_mol * 1000, 4)}"
                for k, y in comps
            ]
            latex.append(
                _sum_chain(
                    r"M_\mathrm{comb}",
                    terms,
                    rf"{latex_number(fuel.M_kg_per_mol * 1000, 5)}\ \mathrm{{kg/kmol}}",  # type: ignore[operator]
                    per_line=2,
                )
            )
            text = (
                "Mezcla de gases con fracciones molares yₖ: los átomos de cada elemento son "
                "nₑₗ = Σ yₖ·nₑₗ,ₖ y la masa molar M = Σ yₖ·Mₖ (vademecum §16.2 y §5)."
            )
    else:
        a = fuel.analysis
        assert a is not None
        unit = _mole_unit(fuel, system)
        for el in present:
            mass = getattr(a, el)
            latex.append(
                latex_chain(
                    rf"n_\mathrm{{{el}}}",
                    rf"\frac{{{_c(mass)}}}{{{latex_number(atomic_mass(el) * 1000, 5)}}}",
                    _n(e[el], fuel, system) + unit,
                )
            )
        if a.W > 0.0:
            latex.append(
                latex_chain(
                    r"n_W",
                    rf"\frac{{{_c(a.W)}}}{{{latex_number(species('H2O').M_kg_per_mol * 1000, 5)}}}",
                    _n(fuel.moisture_mol, fuel, system) + unit,
                )
            )
        text = (
            "Con el análisis elemental (fracciones en masa, tal cual se quema), cada elemento "
            "da sus moles de átomos por kg dividiendo por su masa atómica; la humedad W es agua "
            "líquida que se evapora y las cenizas no reaccionan."
        )
    return ProcedureStep("Combustible", text, tuple(latex))


def _oxidizer_step(s: Stoichiometry, system: UnitSystem) -> ProcedureStep:
    ox = s.oxidizer
    y = s.air
    latex: list[str] = []
    if ox.kind == "oxygen":
        text = "Oxígeno puro: todo el comburente reacciona y no hay N₂ en los humos."
        latex.append(r"y_{\mathrm{O_2}} = 1")
    else:
        dry = ox.dry_composition
        if ox.kind == "technical":
            text = (
                "Aire técnico del vademecum (§16.1): 21 % de O₂ y 79 % de N₂, o sea 3,762 "
                "moles de N₂ por mol de O₂."
            )
        else:
            text = "Aire seco (el de las tablas de Cengel): N₂, O₂, Ar y un poco de CO₂."
        dry_terms = [rf"{_c(v)}\,{species_latex(k)}" for k, v in dry.items()]
        latex.append(
            r"\begin{aligned}\text{aire seco} &= " + _lines(dry_terms, 2) + r"\end{aligned}"
        )
        M_a = s.M_dry_air * 1000
        latex.append(rf"M_a = \textstyle\sum y_i\,M_i = {latex_number(M_a, 5)}\ \mathrm{{kg/kmol}}")
        if y.get("H2O", 0.0) > 0.0:
            p_vs = saturation_pressure(ox.T_phi_K)
            yv = y["H2O"]
            text += (
                f" Con humedad: a {_degC(ox.T_phi_K)} la presión de saturación es la de la tabla "
                "A-4 y el vapor ocupa y_v = φ·p_vs/p de los moles (vademecum §14.1, Cengel 15-3)."
            )
            if ox.T_humidity_K is not None and abs(ox.T_K - ox.T_humidity_K) > 1e-9:
                text += (
                    f" La φ es la del aire ambiente: al llevarlo a {_degC(ox.T_K)} el vapor no "
                    "cambia (ω constante)."
                )
            latex.append(
                latex_chain(
                    r"y_v",
                    r"\frac{\phi\,p_{vs}}{p}",
                    rf"\frac{{{_c(ox.phi, 3)} \cdot {_pn(p_vs, system, 4)}}}"
                    rf"{{{_pn(s.p_Pa, system)}}}",
                    _c(yv),
                )
            )
            latex.append(
                latex_chain(
                    r"y_{\mathrm{O_2}}",
                    rf"{_c(dry['O2'])}\,(1 - y_v)",
                    _c(y["O2"]),
                )
            )
    return ProcedureStep("Comburente", text, tuple(latex))


def _theoretical_step(s: Stoichiometry, system: UnitSystem) -> ProcedureStep:
    fuel = s.fuel
    e = fuel.elements()
    terms_sym = [r"n_\mathrm{C}", r"\frac{n_\mathrm{H}}{4}"]
    terms_num = [_n(e["C"], fuel, system), rf"\frac{{{_n(e['H'], fuel, system)}}}{{4}}"]
    if e["S"] > 0:
        terms_sym.append(r"n_\mathrm{S}")
        terms_num.append(_n(e["S"], fuel, system))
    sym = " + ".join(terms_sym)
    if e["O"] > 0:
        sym += r" - \frac{n_\mathrm{O}}{2}"
    num = _lines(terms_num, 2)
    if e["O"] > 0:
        o_term = rf"\frac{{{_n(e['O'], fuel, system)}}}{{2}}"
        num += (r" \\ &\quad - " if len(terms_num) > 2 else " - ") + o_term
    unit = _mole_unit(fuel, system)
    latex = [
        latex_chain(r"\mathrm{O_{2,t}}", sym, num, _n(s.o2_theoretical, fuel, system) + unit),
        latex_chain(
            r"a_s",
            r"\frac{\mathrm{O_{2,t}}}{y_{\mathrm{O_2}}}",
            rf"\frac{{{_n(s.o2_theoretical, fuel, system)}}}{{{_c(s.air['O2'])}}}",
            _n(s.air_theoretical, fuel, system) + unit,
        ),
    ]
    yv = s.air.get("H2O", 0.0)
    dry = rf"(1 - {_c(yv)})\," if yv > 0 else ""
    M_a = latex_number(s.M_dry_air * 1000, 5)
    if fuel.per_mol:
        M_f = latex_number(fuel.M_kg_per_mol * 1000, 5)  # type: ignore[operator]
        latex.append(
            latex_chain(
                r"AC_s",
                r"\frac{a_s\,M_a}{M_\mathrm{comb}}",
                rf"\frac{{{_c(s.air_theoretical)}\,{dry}\cdot {M_a}}}{{{M_f}}}",
                rf"{latex_number(s.AC_s, 5)}\ \mathrm{{kg/kg}}",
            )
        )
    else:
        latex.append(
            latex_chain(
                r"AC_s",
                r"a_s\,M_a",
                rf"{_n(s.air_theoretical, fuel, system)}\,{dry}\cdot {M_a}",
                rf"{latex_number(s.AC_s, 5)}\ \mathrm{{kg/kg}}",
            )
        )
    text = (
        f"Oxígeno teórico {_basis(fuel, system)} para quemar todo el C a CO₂, el H a H₂O y el "
        "S a SO₂ (el O del combustible ya aporta), aire estequiométrico a_s y relación "
        "aire–combustible AC_s (en aire seco; vademecum §16.2 y §16.4)."
    )
    return ProcedureStep("Oxígeno y aire teóricos", text, tuple(latex))


def _excess_step(
    result_lam_spec: tuple[str, float], s: Stoichiometry, system: UnitSystem
) -> ProcedureStep:
    kind, value = result_lam_spec
    fuel = s.fuel
    lam = latex_number(s.lam, 5)
    latex: list[str] = []
    if kind == "lambda" or kind == "theoretical":
        latex.append(rf"\lambda = {lam}")
    elif kind == "excess":
        latex.append(latex_chain(r"\lambda", "1 + e", rf"1 + {_c(value)}", lam))
    elif kind == "equivalence":
        latex.append(latex_chain(r"\lambda", r"\frac{1}{\phi}", rf"\frac{{1}}{{{_c(value)}}}", lam))
    elif kind == "AC":
        latex.append(
            latex_chain(
                r"\lambda",
                r"\frac{AC}{AC_s}",
                rf"\frac{{{latex_number(value, 5)}}}{{{latex_number(s.AC_s, 5)}}}",
                lam,
            )
        )
    else:  # O₂ medido
        y_air = s.air["O2"] / (1.0 - s.air.get("H2O", 0.0))
        if s.co_fraction > 0.0:
            latex.append(rf"y_{{\mathrm{{O_2}},s}} = {_c(value)} \;\Rightarrow\; \lambda = {lam}")
        else:
            n_dry_s = _stoich_dry_moles(s)
            latex.append(
                latex_chain(
                    r"\lambda",
                    r"1 + \frac{y_{\mathrm{O_2},s}\,n_{s,t}}"
                    r"{\mathrm{O_{2,t}}\,(1 - y_{\mathrm{O_2},s}/y_{\mathrm{O_2},a})}",
                    rf"1 + \frac{{{_c(value)} \cdot {_n(n_dry_s, fuel, system)}}}"
                    rf"{{{_n(s.o2_theoretical, fuel, system)}\,(1 - {_c(value)}/{_c(y_air)})}}",
                    lam,
                )
            )
    latex.append(
        latex_chain(
            r"a",
            r"\lambda\,a_s",
            rf"{lam} \cdot {_n(s.air_theoretical, fuel, system)}",
            _n(s.air_moles, fuel, system) + _mole_unit(fuel, system),
        )
    )
    e_txt = latex_number(s.excess, 4)
    phi_txt = latex_number(s.equivalence_ratio, 4)
    latex.append(
        rf"\begin{{aligned}}e &= \lambda - 1 = {e_txt} \\ \phi &= 1/\lambda = {phi_txt}"
        r"\end{aligned}"
    )
    latex.append(
        latex_chain(
            "AC",
            r"\lambda\,AC_s",
            rf"{lam} \cdot {latex_number(s.AC_s, 5)}",
            rf"{latex_number(s.AC, 5)}\ \mathrm{{kg/kg}}",
        )
    )
    if kind == "o2_dry":
        text = (
            f"Con el {_c(value)} de O₂ medido en los humos secos se despeja λ: el O₂ que sobra, "
            "(λ − 1)·O₂,t, sobre los humos secos (los estequiométricos n_s,t más el aire de "
            "más)."
        )
    else:
        text = f"Cantidad de aire dada como {AIR_SPEC_KINDS[kind].lower()} (vademecum §16.2)."
    text += " El aire real es a = λ·a_s y AC = λ·AC_s (vademecum §16.4)."
    return ProcedureStep("Exceso de aire", text, tuple(latex))


def _products_step(s: Stoichiometry, system: UnitSystem) -> ProcedureStep:
    fuel = s.fuel
    e = fuel.elements()
    a = s.air_moles
    y = s.air
    p = s.products
    unit = _mole_unit(fuel, system)
    N = lambda v: _n(v, fuel, system)  # noqa: E731
    latex: list[str] = []
    x_co = s.co_fraction
    air_co2 = a * y.get("CO2", 0.0)
    if e["C"] > 0:
        if s.rich:
            o2_left = a * y["O2"] + e["O"] / 2 - e["S"] - e["H"] / 4
            latex.append(
                latex_chain(
                    r"\mathrm{O_{2,C}}",
                    r"a\,y_{\mathrm{O_2}} + \frac{n_\mathrm{O}}{2}"
                    r" \\ &\quad - n_\mathrm{S} - \frac{n_\mathrm{H}}{4}",
                    N(o2_left),
                )
            )
            latex.append(
                latex_chain(
                    rf"n_{{{_sub('CO2')}}}",
                    r"2\,\mathrm{O_{2,C}} - n_\mathrm{C}",
                    N(p.get("CO2", 0.0) - air_co2) + unit,
                )
            )
            latex.append(
                latex_chain(
                    rf"n_{{{_sub('CO')}}}",
                    rf"n_\mathrm{{C}} - n_{{{_sub('CO2')}}}",
                    N(p.get("CO", 0.0)) + unit,
                )
            )
        elif x_co > 0:
            latex.append(
                latex_chain(
                    rf"n_{{{_sub('CO')}}}",
                    r"x_\mathrm{CO}\,n_\mathrm{C}",
                    rf"{_c(x_co)} \cdot {N(e['C'])}",
                    N(p["CO"]) + unit,
                )
            )
            latex.append(
                latex_chain(
                    rf"n_{{{_sub('CO2')}}}",
                    r"(1 - x_\mathrm{CO})\,n_\mathrm{C}"
                    + (r" + a\,y_{\mathrm{CO_2}}" if air_co2 else ""),
                    N(p["CO2"]) + unit,
                )
            )
        else:
            sym = r"n_\mathrm{C}" + (r" + a\,y_{\mathrm{CO_2}}" if air_co2 else "")
            latex.append(latex_chain(rf"n_{{{_sub('CO2')}}}", sym, N(p["CO2"]) + unit))
    if p.get("H2O", 0.0) > 0:
        sym = r"\frac{n_\mathrm{H}}{2}"
        if fuel.moisture_mol > 0:
            sym += r" + n_W"
        if y.get("H2O", 0.0) > 0:
            sym += r" + a\,y_v"
        latex.append(latex_chain(rf"n_{{{_sub('H2O')}}}", sym, N(p["H2O"]) + unit))
    if p.get("H2", 0.0) > 0:
        latex.append(rf"n_{{{_sub('H2')}}} = {N(p['H2'])}{unit}")
    if e["S"] > 0:
        latex.append(latex_chain(rf"n_{{{_sub('SO2')}}}", r"n_\mathrm{S}", N(p["SO2"]) + unit))
    if p.get("N2", 0.0) > 0:
        sym = r"a\,y_{\mathrm{N_2}}" + (r" + \frac{n_\mathrm{N}}{2}" if e["N"] > 0 else "")
        latex.append(latex_chain(rf"n_{{{_sub('N2')}}}", sym, N(p["N2"]) + unit))
    if p.get("Ar", 0.0) > 0:
        latex.append(latex_chain(rf"n_{{{_sub('Ar')}}}", r"a\,y_{\mathrm{Ar}}", N(p["Ar"]) + unit))
    if not s.rich:
        used = s.o2_theoretical - x_co * e["C"] / 2
        latex.append(
            latex_chain(
                rf"n_{{{_sub('O2')}}}",
                r"a\,y_{\mathrm{O_2}} - \mathrm{O_{2,t}}"
                + (r" + \frac{n_\mathrm{CO}}{2}" if x_co else ""),
                rf"{N(a * y['O2'])} - {N(used)}",
                N(p.get("O2", 0.0)) + unit,
            )
        )
    latex.append(
        _sum_chain(
            r"n_g",
            [N(v) for _, v in _ordered(p)],
            N(s.n_gas) + unit,
            per_line=3,
        )
    )
    latex.append(
        latex_chain(
            r"n_{g,s}",
            rf"n_g - n_{{{_sub('H2O')}}}",
            N(s.n_dry) + unit,
        )
    )
    reaction = _reaction(s, system)
    if reaction:
        latex.insert(0, reaction)
    if s.rich:
        text = (
            "Con defecto de aire (λ < 1) no sobra O₂: el H se quema a H₂O y el S a SO₂, y el C "
            "reparte el O₂ que queda (O₂,C) entre CO₂ y CO (Cengel 15-8 c)."
        )
    else:
        text = (
            "Productos por balance de cada elemento (vademecum §16.2, reacción real): el C a "
            "CO₂"
            + (" (salvo la fracción x_CO que sale como CO)" if x_co else "")
            + ", el H a H₂O (con la humedad), el S a SO₂, el N a N₂, y el O₂ que sobra."
        )
    return ProcedureStep("Productos", text, tuple(latex))


def _reaction(s: Stoichiometry, system: UnitSystem) -> str | None:
    """La reacción real escrita, solo para un combustible de una especie (por mol)."""
    fuel = s.fuel
    if not fuel.per_mol or len(fuel.components) != 1:
        return None
    key = fuel.components[0][0]
    air_terms = [rf"{_c(v)}\,{species_latex(k)}" for k, v in s.air.items()]
    prods = [rf"{_c(v)}\,{species_latex(k)}" for k, v in _ordered(s.products)]
    head = rf"&{species_latex(key)} + {_c(s.air_moles)}\,\big("
    chunks = [" + ".join(air_terms[k : k + 2]) for k in range(0, len(air_terms), 2)]
    lines = [head, *(rf"&\quad {'+ ' if k else ''}{c}" for k, c in enumerate(chunks))]
    lines[-1] += r"\big)"
    lines.append(r"&\longrightarrow " + _lines(prods, 2))
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"


def _gas_step(s: Stoichiometry, system: UnitSystem) -> ProcedureStep:
    fuel = s.fuel
    N = lambda v: _n(v, fuel, system)  # noqa: E731
    yw, yd = s.wet_fractions, s.dry_fractions
    latex: list[str] = []
    for k in ("CO2", "H2O"):
        if s.products.get(k, 0.0) > 0:
            latex.append(
                latex_chain(
                    rf"y_{{{_sub(k)}}}",
                    rf"\frac{{n_{{{_sub(k)}}}}}{{n_g}}",
                    rf"\frac{{{N(s.products[k])}}}{{{N(s.n_gas)}}}",
                    _c(yw[k]),
                )
            )
    for k in ("CO2", "O2", "CO"):
        if s.products.get(k, 0.0) > 0:
            latex.append(
                latex_chain(
                    rf"y_{{{_sub(k)},s}}",
                    rf"\frac{{n_{{{_sub(k)}}}}}{{n_{{g,s}}}}",
                    rf"\frac{{{N(s.products[k])}}}{{{N(s.n_dry)}}}",
                    _c(yd[k]),
                )
            )
    latex.append(
        rf"M_g = \textstyle\sum y_i\,M_i = {latex_number(s.M_gas * 1000, 5)}\ \mathrm{{kg/kmol}}"
    )
    if fuel.per_mol:
        latex.append(
            latex_chain(
                r"GC",
                r"\frac{n_g\,M_g}{M_\mathrm{comb}}",
                rf"\frac{{{_c(s.n_gas)} \cdot {latex_number(s.M_gas * 1000, 5)}}}"
                rf"{{{latex_number(fuel.M_kg_per_mol * 1000, 5)}}}",  # type: ignore[operator]
                rf"{latex_number(s.GC, 5)}\ \mathrm{{kg/kg}}",
            )
        )
    else:
        latex.append(
            latex_chain(
                r"GC",
                r"n_g\,M_g",
                rf"{N(s.n_gas)} \cdot {latex_number(s.M_gas * 1000, 5)}",
                rf"{latex_number(s.GC, 5)}\ \mathrm{{kg/kg}}",
            )
        )
    v_mol = R_U * T_NORMAL_K / P_NORMAL_PA
    v_unit = unit_label("specific_volume", system)
    v_num = latex_number(convert_from_si(s.normal_volume, "specific_volume", system), 5)
    v_mol_num = latex_number(v_mol * 1000, 5)
    if fuel.per_mol:
        latex.append(
            latex_chain(
                r"V_N",
                r"\frac{n_g\,\bar{v}_N}{M_\mathrm{comb}}",
                rf"\frac{{{_c(s.n_gas)} \cdot {v_mol_num}}}{{{_M_fuel(fuel)}}}",
                rf"{v_num}\ {latex_unit(v_unit)}",
            )
        )
    else:
        latex.append(
            latex_chain(
                r"V_N",
                r"n_g\,\bar{v}_N",
                rf"{_c(s.n_gas * 1e-3)} \cdot {v_mol_num}",
                rf"{v_num}\ {latex_unit(v_unit)}",
            )
        )
    text = (
        "Fracciones en base húmeda (sobre n_g) y seca (sobre n_g,s, como mide un analizador), "
        "masa molar de los humos y relación gases–combustible (vademecum §16.3 y §16.5). "
        "V_N es el volumen de los humos húmedos en condiciones normales (0 °C y 1 atm, "
        "v̄_N = 22,414 m³/kmol)."
    )
    return ProcedureStep("Humos", text, tuple(latex))


def _dew_step(s: Stoichiometry, system: UnitSystem, so3: float) -> ProcedureStep | None:
    T_dp = s.dew_point_K
    if T_dp is None:
        return None
    y = s.wet_fractions["H2O"]
    p_w = s.p_H2O_Pa
    latex = [
        latex_chain(
            r"p_{\mathrm{H_2O}}",
            r"y_{\mathrm{H_2O}}\,p",
            rf"{_c(y)} \cdot {latex_number(convert_from_si(s.p_Pa, _P, system), 5)}",
            _q(p_w, _P, system, 4),
        ),
        latex_chain(r"T_{pr}", r"T_{sat}(p_{\mathrm{H_2O}})", _q(T_dp, _T, system)),
    ]
    text = (
        "Punto de rocío: la temperatura de saturación a la presión parcial del vapor (tabla "
        "A-5; Cengel 15-2). Si los humos se enfrían por debajo, condensa agua."
    )
    acid = s.acid_dew_point_K(so3)
    if acid is not None:
        mmhg = P_NORMAL_PA / 760.0
        pw = p_w / mmhg
        ps = so3 * s.wet_fractions["SO2"] * s.p_Pa / mmhg
        latex.append(
            latex_chain(
                r"p_{\mathrm{SO_3}}",
                r"x_{\mathrm{SO_3}}\,y_{\mathrm{SO_2}}\,p",
                rf"{latex_number(ps, 4)}\ \mathrm{{mmHg}}",
            )
        )
        lw_t = latex_number(math.log(pw), 4)
        ls_t = latex_paren(latex_number(math.log(ps), 4))
        latex.append(
            latex_chain(
                r"\frac{1000}{T_{pr,a}}",
                r"2.276 - 0.0294\,\ln p_w \\ &\quad - 0.0858\,\ln p_s"
                r" \\ &\quad + 0.0062\,\ln p_w\,\ln p_s",
                rf"2.276 - 0.0294 \cdot {lw_t} \\ &\quad - 0.0858 \cdot {ls_t}"
                rf" \\ &\quad + 0.0062 \cdot {lw_t} \cdot {ls_t}",
                latex_number(1000.0 / acid, 5),
            )
        )
        latex.append(rf"T_{{pr,a}} = {_q(acid, _T, system)}")
        text += (
            f" Con azufre, el {_c(100 * so3, 3)} % del SO₂ pasa a SO₃ y forma ácido sulfúrico: su "
            "rocío sale de la correlación de Verhoff & Banchero (1974), con las presiones en mmHg."
        )
    return ProcedureStep("Punto de rocío", text, tuple(latex))


def _heating_value_step(fuel: Fuel, system: UnitSystem) -> ProcedureStep:
    e = fuel.elements()
    w = fuel.water_formed()
    hfg = water_hfg_J_per_mol()
    latex: list[str] = []
    if fuel.per_mol:
        hf = fuel.hf_per_basis
        co2, h2o, h2o_l, so2 = (species(k).hf_J_per_mol for k in ("CO2", "H2O", "H2O(l)", "SO2"))
        sym = (
            r"\bar{h}_{f,\mathrm{comb}} - n_\mathrm{C}\,\bar{h}_{f,\mathrm{CO_2}}"
            r" \\ &\quad - \frac{n_\mathrm{H}}{2}\,\bar{h}_{f,\mathrm{H_2O}}"
        )
        num = (
            rf"{_hm(hf, system)} - {_c(e['C'])} \cdot {latex_paren(_hm(co2, system))}"
            rf" \\ &\quad - {_c(e['H'] / 2)} \cdot {latex_paren(_hm(h2o, system))}"
        )
        num_s = num.replace(_hm(h2o, system), _hm(h2o_l, system))
        if e["S"] > 0:
            sym += r" - n_\mathrm{S}\,\bar{h}_{f,\mathrm{SO_2}}"
            num += rf" - {_c(e['S'])} \cdot {latex_paren(_hm(so2, system))}"
            num_s += rf" - {_c(e['S'])} \cdot {latex_paren(_hm(so2, system))}"
        latex.append(latex_chain(r"\overline{PCI}", sym, num, _hmq(fuel.lhv_per_basis, system)))
        latex.append(
            latex_chain(
                r"\overline{PCS}",
                sym.replace(r"\mathrm{H_2O}}", r"\mathrm{H_2O}(\ell)}"),
                num_s,
                _hmq(fuel.hhv_per_basis, system),
            )
        )
        M = latex_number(fuel.M_kg_per_mol * 1000, 5)  # type: ignore[operator]
        latex.append(
            latex_chain(
                r"PCI",
                r"\frac{\overline{PCI}}{M_\mathrm{comb}}",
                rf"\frac{{{_hm(fuel.lhv_per_basis, system)}}}{{{M}}}",
                _q(fuel.lhv_per_kg, _EH, system),
            )
        )
        latex.append(rf"PCS = {_q(fuel.hhv_per_kg, _EH, system)}")
        text = (
            "Poder calorífico con las entalpías de formación (vademecum §16.9): reactivos y "
            "productos a 25 °C; el PCI con el agua como vapor y el PCS como líquida. El "
            "combustible es h̄_f,comb = Σ yₖ·h̄_f,k (el O₂ y el N₂ valen cero)."
        )
    else:
        latex.append(rf"PCS = {_q(fuel.hhv_per_kg, _EH, system)}\ \text{{(dato)}}")
        latex.append(
            latex_chain(
                r"PCI",
                r"PCS - n_w\,\bar{h}_{fg}",
                rf"{latex_number(convert_from_si(fuel.hhv_per_kg, _EH, system), 5)}"
                rf" \\ &\quad - {_n(w, fuel, system)} \cdot {_hm(hfg, system)}",
                _q(fuel.lhv_per_kg, _EH, system),
            )
        )
        text = (
            "El PCS de un combustible por su análisis es un dato (de laboratorio o de una "
            "correlación, Fase 6). El PCI resta el calor latente del agua que se forma más la "
            "humedad, n_w = n_H/2 + n_W, con h̄_fg = 44 004 kJ/kmol a 25 °C (vademecum §16.9)."
        )
    latex.append(
        latex_chain(
            r"PCS - PCI",
            r"n_w\,\bar{h}_{fg}",
            rf"{_n(w, fuel, system)} \cdot {_hm(hfg, system)}",
            _Eq(w * hfg, fuel, system),
        )
    )
    return ProcedureStep("Poder calorífico", text, tuple(latex))


def _reactants_line(r: CombustionResult, system: UnitSystem) -> str:
    inp = r.inputs
    fuel = r.fuel
    h_fuel = fuel.enthalpy(inp.T_fuel_K)
    h_air = r.H_reactants - h_fuel
    return latex_chain(
        r"H_r",
        r"\bar{h}_\mathrm{comb}(T_c) \\ &\quad + a\textstyle\sum y_i\,\bar{h}_i(T_a)",
        _wrap(_E(h_fuel, fuel, system), "+", latex_paren(_E(h_air, fuel, system))),
        _Eq(r.H_reactants, fuel, system),
    )


def _flame_step(r: CombustionResult, system: UnitSystem) -> ProcedureStep:
    fuel = r.fuel
    s = r.stoich
    T = r.flame.T_K
    prods = _ordered(s.products)
    hf_sum = sum(n * species(k).hf_J_per_mol for k, n in prods)
    v = r.inputs.process == "v"
    latex = [_reactants_line(r, system)]
    if v:
        latex.append(
            latex_chain(
                r"U_r",
                r"H_r - R_u\,(n_{c}\,T_c + a\,T_a)",
                _Eq(r.U_reactants, fuel, system),
            )
        )
        target = r.U_reactants
        latex.append(
            r"\textstyle\sum n_i\,[\bar{h}_{f,i} + \Delta\bar{h}_i(T_{ad}) - R_u T_{ad}] = U_r"
        )
        sensible = target - hf_sum + R_U * T * s.n_gas
        latex.append(
            latex_chain(
                r"\textstyle\sum n_i\,\Delta\bar{h}_i",
                r"U_r - \textstyle\sum n_i\,\bar{h}_{f,i} \\ &\quad + R_u\,T_{ad}\,n_g",
                _Eq(sensible, fuel, system),
            )
        )
    else:
        target = r.H_reactants
        latex.append(r"\textstyle\sum n_i\,[\bar{h}_{f,i} + \Delta\bar{h}_i(T_{ad})] = H_r")
        sensible = target - hf_sum
        latex.append(
            latex_chain(
                r"\textstyle\sum n_i\,\Delta\bar{h}_i",
                r"H_r - \textstyle\sum n_i\,\bar{h}_{f,i}",
                _Eq(sensible, fuel, system),
            )
        )
    terms = [rf"{_n(n, fuel, system)} \cdot {_hm(species(k).delta_h(T), system)}" for k, n in prods]
    latex.append(rf"T_{{ad}} = {_q(T, _T, system)}")
    latex.append(
        _sum_chain(
            r"\textstyle\sum n_i\,\Delta\bar{h}_i",
            terms,
            _Eq(sum(n * species(k).delta_h(T) for k, n in prods), fuel, system),
            per_line=1,
        )
    )
    if v:
        latex.append(
            latex_chain(
                r"p_2",
                r"\frac{n_g\,R_u\,T_{ad}}{V}",
                _q(r.flame.p_Pa, _P, system, 4),
            )
        )
    text = (
        "Temperatura adiabática de llama (vademecum §16.8, Cengel §15-5): sin calor ni "
        "trabajo, la entalpía de los productos a T_ad iguala la de los reactivos. Con "
        "combustión completa (o con el CO dado), cada producto suma su h̄_f más su Δh̄ hasta "
        "T_ad (tablas A-18 a A-25; acá con los polinomios NASA) y T_ad se busca hasta que "
        "cierra el balance."
    )
    if v:
        text = (
            "A volumen constante (la bomba, Cengel 15-7) se iguala la energía interna, "
            "ū = h̄ − R_u·T para los gases; " + text[0].lower() + text[1:]
        )
    return ProcedureStep("Temperatura adiabática de llama", text, tuple(latex))


def _dissociation_step(r: CombustionResult, system: UnitSystem) -> ProcedureStep | None:
    f = r.flame_eq
    if f is None:
        return None
    y = f.fractions
    minor = [k for k in ("CO", "H2", "OH", "NO", "O", "H") if y.get(k, 0.0) > 1e-6]
    latex = [rf"T_{{ad,eq}} = {_q(f.T_K, _T, system)}"]
    pairs = [rf"y_{{{_sub(k)}}} = {_c(y[k], 3)}" for k in minor]
    k = 0
    while k < len(pairs):
        if k + 1 < len(pairs) and not latex_is_wide(pairs[k], pairs[k + 1]):
            latex.append(rf"{pairs[k]},\quad {pairs[k + 1]}")
            k += 2
        else:
            latex.append(pairs[k])
            k += 1
    pp = f.p_Pa / P_STANDARD_PA
    for check in r.kp_checks:
        latex.append(check.reaction)
        latex.append(
            latex_chain(
                r"\ln K_p",
                r"-\frac{\Delta\bar{g}^\circ}{R_u\,T}",
                rf"-\frac{{{latex_paren(_hm(check.delta_g_J_per_mol, system))}}}"
                rf"{{{_ru(system)} \cdot {_T_abs(f.T_K, system)}}}",
                latex_number(check.ln_Kp_tables, 5),
            )
        )
        sym, num = _kp_expression(check.key, y)
        if check.key in ("CO2", "H2O"):
            sym += r"\,(p/p^\circ)^{1/2}"
            factor = rf"({latex_number(pp, 4)})^{{1/2}}"
            num += (r" \\ &\quad \cdot " if r"\times" in num else r"\,") + factor
        latex.append(
            latex_chain(
                "K_p",
                sym,
                num,
                rf"{_c(math.exp(check.ln_Kp_composition), 4)}"
                rf" = e^{{{latex_number(check.ln_Kp_composition, 5)}}}",
            )
        )
    text = (
        "Con disociación (equilibrio químico, Cengel cap. 16) la composición minimiza la "
        "energía libre de Gibbs: aparecen CO, H₂, OH, NO, O y H, y eso absorbe energía, así que "
        "la llama queda más fría. Cada reacción cumple su constante de equilibrio: el K_p de "
        "tablas (ln K_p = −Δḡ°/(R_u·T), a p° = 1 bar) coincide con el de la composición."
    )
    return ProcedureStep("Llama con disociación", text, tuple(latex))


def _ru(system: UnitSystem) -> str:
    """R_u en las unidades molares del sistema (8,3145 kJ/(kmol·K) o 1,9859 Btu/(lbmol·°R))."""
    return latex_number(convert_from_si(R_U, _MS, system), 5)


def _T_abs(T_K: float, system: UnitSystem) -> str:
    """Temperatura absoluta: K, o °R en el sistema inglés."""
    return latex_number(T_K * (1.8 if system == "Inglés" else 1.0), 5)


def _kp_expression(key: str, y: dict[str, float]) -> tuple[str, str]:
    """K_p de la composición: la fórmula y los números (Cengel §16-2)."""
    if key in ("CO2", "H2O"):
        prod, reac = ("CO", "CO2") if key == "CO2" else ("H2", "H2O")
        sym = rf"\frac{{y_{{{_sub(prod)}}}\,y_{{{_sub('O2')}}}^{{1/2}}}}{{y_{{{_sub(reac)}}}}}"
        num = rf"\frac{{{_c(y[prod], 3)}\,({_c(y['O2'], 3)})^{{1/2}}}}{{{_c(y[reac], 3)}}}"
        return sym, num
    sym = rf"\frac{{y_{{{_sub('NO')}}}}}{{y_{{{_sub('N2')}}}^{{1/2}}\,y_{{{_sub('O2')}}}^{{1/2}}}}"
    num = rf"\frac{{{_c(y['NO'], 3)}}}{{({_c(y['N2'], 3)})^{{1/2}}\,({_c(y['O2'], 3)})^{{1/2}}}}"
    return sym, num


def _heat_step(r: CombustionResult, system: UnitSystem) -> ProcedureStep | None:
    h = r.heat
    if h is None:
        return None
    fuel = r.fuel
    s = r.stoich
    T = h.T_K
    latex: list[str] = []
    text = (
        f"Los humos salen a {_degC(T)}: el calor que sale es lo que les falta para llegar a "
        "esa temperatura (vademecum §16.7, Cengel 15-6)."
    )
    if h.condensed > 0.0:
        p_sat = saturation_pressure(T)
        if r.inputs.process == "p":
            latex.append(
                latex_chain(
                    r"n_v",
                    r"\frac{p_{sat}}{p - p_{sat}}\,n_{g,s}",
                    rf"\frac{{{latex_number(convert_from_si(p_sat, _P, system), 4)}}}"
                    rf"{{{latex_number(convert_from_si(s.p_Pa - p_sat, _P, system), 5)}}}"
                    rf" \cdot {_n(s.n_dry, fuel, system)}",
                    _n(h.gas.get("H2O", 0.0), fuel, system),
                )
            )
        n_v = _n(h.gas.get("H2O", 0.0), fuel, system)
        latex.append(
            latex_chain(
                r"n_\ell",
                r"n_{\mathrm{H_2O}} - n_v",
                rf"{_n(s.products['H2O'], fuel, system)} - {n_v}",
                _n(h.condensed, fuel, system) + _mole_unit(fuel, system),
            )
        )
        text += (
            " Por debajo del rocío el vapor no puede pasar su presión de saturación: condensa "
            "el resto del agua, que sale líquida (Cengel 15-11)."
        )
    H_p = r.H_reactants - h.Q_out if r.inputs.process == "p" else None
    if H_p is not None:
        latex.append(
            latex_chain(
                r"Q_{sal}",
                r"H_r - H_p(T_s)",
                _wrap(_E(r.H_reactants, fuel, system), "-", latex_paren(_E(H_p, fuel, system))),
                _Eq(h.Q_out, fuel, system),
            )
        )
    else:
        latex.append(latex_chain(r"Q_{sal}", r"U_r - U_p(T_s)", _Eq(h.Q_out, fuel, system)))
        latex.append(rf"p_2 = {_q(h.p_Pa, _P, system, 4)}")
    if fuel.per_mol:
        latex.append(
            latex_chain(
                r"q_{sal}",
                r"\frac{Q_{sal}}{M_\mathrm{comb}}",
                _q(h.Q_out / fuel.M_kg_per_mol, _EH, system),  # type: ignore[operator]
            )
        )
    if h.split is not None:
        sp = h.split
        latex.append(
            latex_chain(
                r"\eta_{PCI}",
                r"\frac{Q_{sal}}{PCI}",
                rf"\frac{{{_E(h.Q_out, fuel, system)}}}{{{_E(sp.lhv, fuel, system)}}}",
                latex_number(h.eta_lhv, 4),  # type: ignore[arg-type]
            )
        )
        latex.append(
            latex_chain(
                r"\eta_{PCS}",
                r"\frac{Q_{sal}}{PCS}",
                rf"\frac{{{_E(h.Q_out, fuel, system)}}}{{{_E(sp.hhv, fuel, system)}}}",
                latex_number(h.eta_hhv, 4),  # type: ignore[arg-type]
            )
        )
        E = lambda v: latex_paren(_E(v, fuel, system))  # noqa: E731
        terms = (
            rf"{_E(sp.lhv, fuel, system)} + {E(sp.sensible_reactants)}"
            rf" \\ &\quad - {E(sp.unburned)} - {E(sp.stack_sensible)}"
            rf" \\ &\quad + {E(sp.latent_recovered)}"
        )
        latex.append(
            latex_chain(
                r"Q_{sal}",
                r"PCI + q_{reac} - q_{inq} \\ &\quad - q_{humos} + q_{lat}",
                terms,
                _Eq(sp.Q_out, fuel, system),
            )
        )
        text += (
            " Rendimiento sobre el PCI y el PCS, y el reparto: el PCI más el calor sensible de "
            "los reactivos (si entran calientes), menos lo que no se quemó (CO, H₂), menos el "
            "calor sensible de los humos (con el agua como vapor), más el latente del agua que "
            "condensa."
        )
    return ProcedureStep("Calor con los humos a T_s", text, tuple(latex))


def _second_law_step(r: CombustionResult, system: UnitSystem) -> ProcedureStep | None:
    sl = r.second_law
    if sl is None:
        return None
    T0 = latex_number(sl.T0_K, 5)
    latex = [
        latex_chain(
            r"S_r",
            r"s_\mathrm{comb} \\ &\quad + a\textstyle\sum y_i\,[\bar{s}^\circ_i"
            r" - R_u\ln\frac{y_i\,p}{p^\circ}]",
            _smq(sl.S_reactants, system),
        ),
        latex_chain(
            r"S_p",
            r"\textstyle\sum n_i\,[\bar{s}^\circ_i(T_{ad})"
            r" \\ &\quad - R_u\ln\frac{y_i\,p}{p^\circ}]",
            _smq(sl.S_products_adiabatic, system),
        ),
        latex_chain(
            r"S_{gen}",
            r"S_p - S_r",
            rf"{_sm(sl.S_products_adiabatic, system)} - {_sm(sl.S_reactants, system)}",
            _smq(sl.S_gen_adiabatic, system),
        ),
        latex_chain(
            r"X_{dest}",
            r"T_0\,S_{gen}",
            rf"{T0} \cdot {_sm(sl.S_gen_adiabatic, system)}",
            _hmq(sl.X_dest_adiabatic, system),
        ),
        latex_chain(
            r"\frac{X_{dest}}{PCI}",
            rf"\frac{{{_hm(sl.X_dest_adiabatic, system)}}}{{{_hm(sl.lhv, system)}}}",
            latex_number(sl.X_dest_adiabatic / sl.lhv, 4),
        ),
    ]
    text = (
        "Segundo principio (vademecum §16.11, Cengel 15-10): cada gas con su entropía absoluta "
        "(tabla A-26, a p° = 1 bar) a su presión parcial. La combustión adiabática genera "
        "entropía; la exergía destruida X_dest = T₀·S_gen, comparada con la del combustible "
        "(≈ PCI, vademecum §16.13)."
    )
    if sl.S_gen_heat is not None and r.heat is not None:
        T_b = latex_number(sl.T_sink_K, 5)  # type: ignore[arg-type]
        latex.append(rf"S_p(T_s) = {_smq(sl.S_products_heat, system)}")  # type: ignore[arg-type]
        latex.append(
            latex_chain(
                r"S_{gen}",
                r"S_p(T_s) - S_r + \frac{Q_{sal}}{T_b}",
                rf"{_sm(sl.S_products_heat, system)} - {_sm(sl.S_reactants, system)}"  # type: ignore[arg-type]
                rf" \\ &\quad + \frac{{{_hm(sl.Q_out, system)}}}{{{T_b}}}",  # type: ignore[arg-type]
                _smq(sl.S_gen_heat, system),
            )
        )
        latex.append(
            latex_chain(
                r"X_{dest}",
                r"T_0\,S_{gen}",
                rf"{T0} \cdot {_sm(sl.S_gen_heat, system)}",
                _hmq(sl.X_dest_heat, system),  # type: ignore[arg-type]
            )
        )
        latex.append(
            latex_chain(
                r"X_Q",
                r"Q_{sal}\,(1 - T_0/T_b)",
                _hmq(sl.X_heat, system),  # type: ignore[arg-type]
            )
        )
        text += (
            f" Con el calor entregado a T_b = {_degC(sl.T_sink_K)} (Cengel 15-11): S_gen suma "  # type: ignore[arg-type]
            "la entropía que el calor lleva al medio, y X_Q es la exergía del calor que sale."
        )
    return ProcedureStep("Segundo principio", text, tuple(latex))


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------


def combustion_steps(result: CombustionResult, system: UnitSystem) -> list[ProcedureStep]:
    """Pasos de la combustión, en el orden del vademecum §16."""
    s = result.stoich
    inp = result.inputs
    steps = [
        _fuel_step(result.fuel, system),
        _oxidizer_step(s, system),
        _theoretical_step(s, system),
        _excess_step((inp.air.kind, inp.air.value), s, system),
        _products_step(s, system),
        _gas_step(s, system),
    ]
    dew = _dew_step(s, system, inp.so3_conversion)
    if dew is not None:
        steps.append(dew)
    steps.append(_heating_value_step(result.fuel, system))
    steps.append(_flame_step(result, system))
    for extra in (
        _dissociation_step(result, system),
        _heat_step(result, system),
        _second_law_step(result, system),
    ):
        if extra is not None:
            steps.append(extra)
    return [
        ProcedureStep(f"{k}. {st.title}", st.text, st.latex) for k, st in enumerate(steps, start=1)
    ]


def flue_gas_steps(result: FlueGasResult, system: UnitSystem) -> list[ProcedureStep]:
    """Pasos del análisis de humos (Cengel 15-4)."""
    inp = result.inputs
    fuel = inp.fuel
    s = result.stoich
    e = fuel.elements()
    steps = [_fuel_step(fuel, system), _oxidizer_step(s, system)]
    o = result.orsat
    latex: list[str] = []
    if o is not None:
        m = o.measured
        latex.append(
            latex_chain(
                r"y_{\mathrm{N_2},s}",
                r"1 - y_{\mathrm{CO_2}} - y_{\mathrm{CO}} - y_{\mathrm{O_2}}",
                _c(m["N2"]),
            )
        )
        sC = "n_\\mathrm{C}" + (" + n_\\mathrm{S}" if e["S"] > 0 else "")
        n_cs = _n(e["C"] + e["S"], fuel, system)
        latex.append(
            latex_chain(
                r"x",
                rf"\frac{{100\,(y_{{\mathrm{{CO_2}}}} + y_{{\mathrm{{CO}}}})}}{{{sC}}}",
                rf"\frac{{100\,({_c(m['CO2'])} + {_c(m['CO'])})}}{{{n_cs}}}",
                _n(o.fuel_per_100, fuel, system, 5),
            )
            if s.air.get("CO2", 0.0) == 0.0
            else rf"x = {_n(o.fuel_per_100, fuel, system, 5)}"
        )
        latex.append(
            latex_chain(
                r"a",
                r"\frac{100\,y_{\mathrm{N_2},s} - x\,n_\mathrm{N}/2}{y_{\mathrm{N_2}}}",
                rf"{_c(o.air_per_100, 5)}",
            )
        )
        o2t = _n(fuel.theoretical_oxygen(), fuel, system)
        latex.append(
            latex_chain(
                r"\lambda",
                r"\frac{a\,y_{\mathrm{O_2}}}{x\,\mathrm{O_{2,t}}}",
                rf"\frac{{{_c(o.air_per_100, 5)} \cdot {_c(s.air['O2'])}}}"
                rf"{{{_n(o.fuel_per_100, fuel, system, 5)} \cdot {o2t}}}",
                latex_number(o.lam, 5),
            )
        )
        latex.append(
            latex_chain(
                r"n_{\mathrm{H_2O}}",
                r"x\,(n_\mathrm{H}/2 + n_W) + a\,y_v",
                _c(o.water_per_100, 5),
            )
        )
        rel = o.oxygen_residual_relative
        latex.append(
            latex_chain(
                r"\Delta O",
                r"O_{entra} - O_{sale}",
                r"\approx 0"
                if abs(rel) < 1e-6
                else rf"{_c(o.oxygen_residual, 3)}\ ({_c(100 * rel, 3)}\,\%)",
            )
        )
        text = (
            "Cada 100 kmol de humos secos (Cengel 15-4): el balance de C da los kmol de "
            "combustible x (el Orsat mide el SO₂ junto con el CO₂), el de N₂ (con el Ar) el "
            "aire a, y el de H el agua. El balance de O no se usa: queda para verificar la "
            "medición (ΔO ≈ 0)."
        )
    else:
        y_air = s.air["O2"] / (1.0 - s.air.get("H2O", 0.0))
        o2_t, co_t = _c(inp.o2_dry), _c(inp.co_dry, 3)
        latex.append(rf"y_{{\mathrm{{O_2}},s}} = {o2_t},\quad y_{{\mathrm{{CO}},s}} = {co_t}")
        n_dry_s = _stoich_dry_moles(s)
        x_co_t = _c(result.co_fraction, 3)
        latex.append(
            latex_chain(
                r"\lambda",
                r"1 + \frac{y_{\mathrm{O_2},s}\,n_{s,t}}"
                r"{\mathrm{O_{2,t}}\,(1 - y_{\mathrm{O_2},s}/y_{\mathrm{O_2},a})}",
                rf"1 + \frac{{{o2_t} \cdot {_n(n_dry_s, fuel, system)}}}"
                rf"{{{_n(fuel.theoretical_oxygen(), fuel, system)}\,(1 - {o2_t}/{_c(y_air)})}}",
                latex_number(s.lam, 5),
            )
            if inp.co_dry == 0.0
            else rf"\lambda = {latex_number(s.lam, 5)},\quad x_\mathrm{{CO}} = {x_co_t}"
        )
        text = (
            "Con el O₂ de los humos secos se despeja λ: el O₂ que sobra, (λ − 1)·O₂,t, sobre "
            "los humos secos (los estequiométricos n_s,t más el aire de más). Con CO se itera: "
            "la fracción del C que pasa a CO sale del CO medido."
        )
    steps.append(ProcedureStep("Análisis de los humos", text, tuple(latex)))
    co = s.products.get("CO", 0.0)
    latex2 = [
        latex_chain(
            "AC",
            r"\lambda\,AC_s",
            rf"{latex_number(s.lam, 5)} \cdot {latex_number(s.AC_s, 5)}",
            rf"{latex_number(s.AC, 5)}\ \mathrm{{kg/kg}}",
        )
    ]
    if co > 0:
        dh = species("CO").hf_J_per_mol - species("CO2").hf_J_per_mol
        latex2.append(
            latex_chain(
                r"\frac{q_{inq}}{PCI}",
                r"\frac{n_\mathrm{CO}\,(\bar{h}_{f,\mathrm{CO}} - \bar{h}_{f,\mathrm{CO_2}})}{PCI}",
                rf"\frac{{{_n(co, fuel, system)} \cdot {_hm(dh, system)}}}"
                rf"{{{_E(fuel.lhv_per_basis, fuel, system)}}}",
                latex_number(result.unburned_fraction, 3),
            )
        )
    if result.condensed > 0.0:
        T = inp.T_cool_K
        p_sat = saturation_pressure(T)
        latex2.append(
            latex_chain(
                r"n_v",
                r"\frac{p_{sat}}{p - p_{sat}}\,n_{g,s}",
                rf"\frac{{{latex_number(convert_from_si(p_sat, _P, system), 4)}}}"
                rf"{{{latex_number(convert_from_si(s.p_Pa - p_sat, _P, system), 5)}}}"
                rf" \cdot {_n(s.n_dry, fuel, system)}",
                _n(result.gas_cooled.get("H2O", 0.0), fuel, system),
            )
        )
        latex2.append(
            latex_chain(
                r"n_\ell",
                r"n_{\mathrm{H_2O}} - n_v",
                _n(result.condensed, fuel, system) + _mole_unit(fuel, system),
            )
        )
    steps.append(
        ProcedureStep(
            "Aire, inquemados y agua",
            "La relación aire–combustible real, la pérdida por el CO que no se quemó (el calor "
            "que daría al pasar a CO₂) y el agua que condensa al enfriar los humos a "
            f"{_degC(inp.T_cool_K)} (Cengel 15-4 c).",
            tuple(latex2),
        )
    )
    return [
        ProcedureStep(f"{k}. {st.title}", st.text, st.latex) for k, st in enumerate(steps, start=1)
    ]
