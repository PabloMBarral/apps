"""Procedimiento didáctico de la exergía (LaTeX), en el sistema de unidades activo.

Tres partes, una por modo de la página:

- :func:`physical_steps`, :func:`heat_steps` y :func:`finite_source_steps`:
  vademecum §11.1 a §11.6;
- :func:`species_steps`, :func:`mixture_steps` y :func:`fuel_steps`: la exergía
  química (Szargut et al., 1988; Szargut y Styrylska, 1964);
- :func:`plant_steps`: F, P, D y L de cada componente y el balance de la planta
  (Bejan, Tsatsaronis y Moran, 1996; vademecum §11.9 y §11.10).

Cada igualdad va en su renglón (:func:`core.latex.latex_chain`) y, con números
×10ⁿ (SI), las restas se parten: así todo entra en el ancho de un celular.
"""

from __future__ import annotations

import math

from core.combustion.thermo import R_U
from core.cycles.rankine_procedure import _n, _q, _wrap
from core.exergy.chemical import (
    MODELS,
    T_STANDARD_K,
    FuelExergy,
    MixtureExergy,
    SpeciesExergy,
    chemical_species,
)
from core.exergy.physical import (
    G_STANDARD,
    FiniteSourceExergy,
    HeatExergy,
    PhysicalExergy,
    ProcessExergy,
)
from core.exergy.plant import ComponentExergy, PlantExergy
from core.latex import latex_chain, latex_number, latex_paren
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si

__all__ = [
    "finite_source_steps",
    "fuel_steps",
    "heat_steps",
    "mixture_steps",
    "physical_steps",
    "plant_steps",
    "species_steps",
]

_EH: QuantityKind = "specific_enthalpy"
_ES: QuantityKind = "specific_entropy"
_EM: QuantityKind = "molar_enthalpy"
_W: QuantityKind = "power"


def _numbered(steps: list[ProcedureStep]) -> list[ProcedureStep]:
    return [ProcedureStep(f"{k}. {s.title}", s.text, s.latex) for k, s in enumerate(steps, 1)]


def _absolute_T(T_K: float, system: UnitSystem) -> str:
    """Temperatura absoluta: K (SI y Técnico) o °R (Inglés)."""
    if system == "Inglés":
        return rf"{latex_number(T_K * 1.8, 5)}\ \mathrm{{R}}"
    return rf"{latex_number(T_K, 5)}\ \mathrm{{K}}"


def _T_abs_number(T_K: float, system: UnitSystem) -> str:
    return latex_number(T_K * 1.8 if system == "Inglés" else T_K, 5)


def _sub(a: str, b: str) -> str:
    return rf"{a} - {latex_paren(b)}"


def _lookup(lhs: str, args: str, value: str) -> str:
    r"""``h_0 = h(T_0, p_0) \\ = …`` en dos renglones (la unidad del inglés es larga)."""
    return latex_chain(lhs, rf"{lhs[0]}({args})", value)


# ---------------------------------------------------------------------
# Exergía física
# ---------------------------------------------------------------------


def _dead_state_step(r: PhysicalExergy, system: UnitSystem, mass: bool) -> ProcedureStep:
    d = r.dead
    text = (
        f"El **estado muerto** es el fluido en equilibrio con el ambiente, a T₀ y p₀ (vademecum "
        f"§11.1): {d.T_K - 273.15:.4g} °C y {d.P_Pa / 1e5:.4g} bar. Sus propiedades salen de la "
        "misma ecuación de estado que las del estado (CoolProp), así las referencias se cancelan."
    ).replace(".", ",", 0)
    args = "T_0, p_0"
    latex = [
        rf"T_0 = {_absolute_T(r.T0_K, system)}",
        rf"p_0 = {_q(r.p0_Pa, 'pressure', system)}",
        _lookup("h_0", args, _q(d.h_J_per_kg, _EH, system)),
        _lookup("s_0", args, _q(d.s_J_per_kg_K, _ES, system)),
    ]
    if mass:
        latex.append(_lookup("u_0", args, _q(d.u_J_per_kg, _EH, system)))
        latex.append(_lookup("v_0", args, _q(d.v_m3_per_kg, "specific_volume", system)))
    return ProcedureStep("Estado muerto", text, tuple(latex))


def _flow_step(r: PhysicalExergy, system: UnitSystem) -> ProcedureStep:
    st, d = r.state, r.dead
    text = (
        "La **exergía de flujo** (vademecum §11.3) es el trabajo máximo que se obtiene llevando "
        "la corriente al estado muerto: lo que baja la entalpía menos lo que hay que entregarle "
        "al ambiente para que la entropía no disminuya."
    )
    latex = [
        latex_chain(
            r"h - h_0",
            _wrap(_n(st.h_J_per_kg, _EH, system), "-", latex_paren(_n(d.h_J_per_kg, _EH, system))),
            _q(r.dh_J_per_kg, _EH, system),
        ),
        latex_chain(
            r"s - s_0",
            _wrap(
                _n(st.s_J_per_kg_K, _ES, system), "-", latex_paren(_n(d.s_J_per_kg_K, _ES, system))
            ),
            _q(r.ds_J_per_kg_K, _ES, system),
        ),
        latex_chain(
            r"\psi",
            r"(h - h_0) - T_0\,(s - s_0)",
            _wrap(
                _n(r.dh_J_per_kg, _EH, system),
                "-",
                rf"{_T_abs_number(r.T0_K, system)}\cdot "
                f"{latex_paren(_n(r.ds_J_per_kg_K, _ES, system))}",
            ),
            _q(r.psi_J_per_kg, _EH, system),
        ),
    ]
    return ProcedureStep("Exergía de flujo ψ", text, tuple(latex))


def _split_step(r: PhysicalExergy, system: UnitSystem) -> ProcedureStep:
    i = r.intermediate
    text = (
        "Separada en dos partes (Kotas, 1985): la **térmica**, de enfriarlo (o calentarlo) a "
        "presión constante hasta T₀, y la **mecánica**, de llevarlo después de p a p₀ a T₀. El "
        "estado intermedio es el fluido a (T₀, p)."
    )
    args = "T_0, p"
    psi_T = r.psi_thermal_J_per_kg
    psi_M = r.psi_mechanical_J_per_kg
    latex = [
        _lookup("h_i", args, _q(i.h_J_per_kg, _EH, system)),
        _lookup("s_i", args, _q(i.s_J_per_kg_K, _ES, system)),
        latex_chain(
            r"\psi_T",
            r"(h - h_i) - T_0\,(s - s_i)",
            _q(psi_T, _EH, system),
        ),
        latex_chain(
            r"\psi_M",
            r"(h_i - h_0) - T_0\,(s_i - s_0)",
            _q(psi_M, _EH, system),
        ),
        latex_chain(
            r"\psi_T + \psi_M",
            _wrap(_n(psi_T, _EH, system), "+", latex_paren(_n(psi_M, _EH, system))),
            _q(psi_T + psi_M, _EH, system),
        ),
    ]
    return ProcedureStep("Parte térmica y mecánica", text, tuple(latex))


def _mass_step(r: PhysicalExergy, system: UnitSystem) -> ProcedureStep:
    st, d = r.state, r.dead
    text = (
        "La **exergía de una masa** (vademecum §11.2) suma p₀·(v − v₀): el trabajo que hace (o "
        "recibe) la atmósfera cuando el sistema cambia de volumen. Es siempre ≥ 0."
    )
    p0_dv = r.p0_Pa * r.dv_m3_per_kg
    latex = [
        latex_chain(
            r"u - u_0",
            _wrap(_n(st.u_J_per_kg, _EH, system), "-", latex_paren(_n(d.u_J_per_kg, _EH, system))),
            _q(r.du_J_per_kg, _EH, system),
        ),
        latex_chain(
            r"p_0\,(v - v_0)",
            _wrap(
                _n(r.p0_Pa, "pressure", system),
                r"\cdot",
                latex_paren(_n(r.dv_m3_per_kg, "specific_volume", system)),
            ),
            _q(p0_dv, _EH, system),
        ),
        latex_chain(
            r"\phi",
            r"(u - u_0) + p_0\,(v - v_0) \\ &\quad - T_0\,(s - s_0)",
            _q(r.phi_J_per_kg, _EH, system),
        ),
    ]
    return ProcedureStep("Exergía de una masa φ", text, tuple(latex))


def _ke_pe_lines(r: PhysicalExergy, system: UnitSystem) -> list[str]:
    lines: list[str] = []
    if r.speed_m_per_s > 0.0:
        V = _q(r.speed_m_per_s, "speed", system)
        lines.append(
            latex_chain(r"\frac{V^2}{2}", rf"\frac{{({V})^2}}{{2}}", _q(r.ke_J_per_kg, _EH, system))
        )
    if r.height_m != 0.0:
        z = _q(r.height_m, "length", system)
        g = (
            r"32.174\ \mathrm{ft/s^2}"
            if system == "Inglés"
            else rf"{latex_number(G_STANDARD, 6)}\ \mathrm{{m/s^2}}"
        )
        lines.append(latex_chain(r"g\,z", rf"{g}\cdot {z}", _q(r.pe_J_per_kg, _EH, system)))
    return lines


def _total_step(r: PhysicalExergy, system: UnitSystem) -> ProcedureStep | None:
    lines = _ke_pe_lines(r, system)
    if r.amount == "masa" and r.X_J is not None:
        lines.append(
            latex_chain(
                r"X",
                r"m\,\left(\phi + \frac{V^2}{2} + g\,z\right)",
                rf"{_n(r.mass_kg or 0.0, 'mass', system)}\cdot "
                f"{latex_paren(_n(r.x_mass_J_per_kg, _EH, system))}",
                _q(r.X_J, "energy", system),
            )
        )
    elif r.amount == "caudal" and r.X_W is not None:
        lines.append(
            latex_chain(
                r"\dot X",
                r"\dot m\,\left(\psi + \frac{V^2}{2} + g\,z\right)",
                rf"{_n(r.m_dot_kg_s or 0.0, 'mass_flow', system)}\cdot "
                f"{latex_paren(_n(r.x_flow_J_per_kg, _EH, system))}",
                _q(r.X_W, _W, system),
            )
        )
    if not lines:
        return None
    text = (
        "La energía cinética y la potencial son **exergía pura** y se suman tal cual "
        "(vademecum §11.2 y §11.3)."
    )
    if r.amount != "por kg":
        text += " El total es por la masa del sistema o por el caudal."
    return ProcedureStep("Cinética, potencial y total", text, tuple(lines))


def _process_step(proc: ProcessExergy, system: UnitSystem) -> ProcedureStep:
    a, b = proc.first, proc.second
    dpsi = proc.dpsi_J_per_kg
    text = (
        "Entre dos estados con el mismo ambiente, en régimen permanente y sin calor (o con calor "
        "a T₀), el **trabajo reversible** es la diferencia de exergías (Cengel cap. 8). "
        + (
            "Δψ > 0: hay que entregar al menos ese trabajo (un compresor, una bomba)."
            if dpsi > 0
            else "Δψ < 0: es lo máximo que se puede obtener (una turbina)."
        )
    )
    latex = [
        latex_chain(
            r"\psi_2",
            r"(h_2 - h_0) - T_0\,(s_2 - s_0)",
            _q(b.psi_J_per_kg, _EH, system),
        ),
        latex_chain(
            r"\Delta\psi",
            r"\psi_2 - \psi_1",
            _wrap(
                _n(b.x_flow_J_per_kg, _EH, system),
                "-",
                latex_paren(_n(a.x_flow_J_per_kg, _EH, system)),
            ),
            _q(dpsi, _EH, system),
        ),
        latex_chain(r"h_2 - h_1", _q(proc.dh_J_per_kg, _EH, system)),
    ]
    return ProcedureStep("Proceso 1 → 2", text, tuple(latex))


def physical_steps(
    result: PhysicalExergy, system: UnitSystem, second: PhysicalExergy | None = None
) -> list[ProcedureStep]:
    """Los pasos de la exergía física de un estado (y del proceso 1 → 2, si hay)."""
    mass = result.amount == "masa"
    steps = [_dead_state_step(result, system, mass), _flow_step(result, system)]
    steps.append(_split_step(result, system))
    if mass or result.dv_m3_per_kg != 0.0:
        steps.append(_mass_step(result, system))
    total = _total_step(result, system)
    if total is not None:
        steps.append(total)
    if second is not None:
        steps.append(_process_step(ProcessExergy(result, second), system))
    return _numbered(steps)


def heat_steps(result: HeatExergy, system: UnitSystem) -> list[ProcedureStep]:
    """La exergía de un calor (vademecum §11.5)."""
    text = (
        "El calor Q̇ que pasa a la temperatura T lleva la exergía Q̇·(1 − T₀/T): el trabajo que "
        "daría una máquina de Carnot entre T y el ambiente. El resto, Q̇·T₀/T, es anergía."
    )
    if result.T_K < result.T0_K:
        text += " Con T < T₀ el factor de Carnot es negativo: la exergía va contra el calor."
    T = _T_abs_number(result.T_K, system)
    T0 = _T_abs_number(result.T0_K, system)
    latex = [
        latex_chain(
            r"1 - \frac{T_0}{T}",
            rf"1 - \frac{{{T0}}}{{{T}}}",
            latex_number(result.carnot, 5),
        ),
        latex_chain(
            r"\dot X_Q",
            r"\dot Q\,\left(1 - \frac{T_0}{T}\right)",
            rf"{_n(result.Q_W, _W, system)}\cdot {latex_paren(latex_number(result.carnot, 5))}",
            _q(result.X_W, _W, system),
        ),
        latex_chain(
            r"\dot B_Q",
            r"\dot Q\,\frac{T_0}{T}",
            _q(result.anergy_W, _W, system),
        ),
    ]
    return _numbered([ProcedureStep("Exergía de un calor", text, tuple(latex))])


def finite_source_steps(result: FiniteSourceExergy, system: UnitSystem) -> list[ProcedureStep]:
    """La exergía de una fuente térmica finita (vademecum §11.6)."""
    text = (
        "Al enfriarse (o calentarse) hasta T₀, el cuerpo entrega el calor m·c·(T − T₀), pero "
        "su temperatura va bajando y el factor de Carnot con ella: la exergía es la integral, "
        "m·c·[(T − T₀) − T₀·ln(T/T₀)]."
    )
    T = _T_abs_number(result.T_K, system)
    T0 = _T_abs_number(result.T0_K, system)
    # m·c en J/K, kJ/K o Btu/R: el producto de la masa y el calor específico del sistema
    mc = convert_from_si(result.m_kg, "mass", system) * convert_from_si(
        result.c_J_per_kg_K, "specific_heat", system
    )
    mc_unit = {"SI": r"\mathrm{J/K}", "Técnico": r"\mathrm{kJ/K}", "Inglés": r"\mathrm{Btu/R}"}[
        system
    ]
    dT = result.T_K - result.T0_K
    dT_num = latex_number(dT * 1.8 if system == "Inglés" else dT, 5)
    latex = [
        latex_chain(
            r"m\,c",
            rf"{_n(result.m_kg, 'mass', system)}\cdot "
            rf"{_n(result.c_J_per_kg_K, 'specific_heat', system)}",
            rf"{latex_number(mc, 5)}\ {mc_unit}",
        ),
        latex_chain(
            r"T_0\,\ln\frac{T}{T_0}",
            rf"{T0}\cdot \ln\frac{{{T}}}{{{T0}}}",
            latex_number(
                (result.T0_K * 1.8 if system == "Inglés" else result.T0_K)
                * math.log(result.T_K / result.T0_K),
                5,
            ),
        ),
        latex_chain(
            r"\Phi",
            r"m\,c\left[(T - T_0) - T_0\,\ln\frac{T}{T_0}\right]",
            _q(result.Phi_J, "energy", system),
        ),
        latex_chain(
            r"Q",
            r"m\,c\,(T - T_0)",
            rf"{latex_number(mc, 5)}\cdot {latex_paren(dT_num)}",
            _q(result.Q_J, "energy", system),
        ),
    ]
    return _numbered([ProcedureStep("Exergía de una fuente finita", text, tuple(latex))])


# ---------------------------------------------------------------------
# Exergía química
# ---------------------------------------------------------------------


def _tex_formula(key: str) -> str:
    """``CH4`` → ``\\mathrm{CH_4}``; ``C8H18(l)`` → ``\\mathrm{C_8H_{18}}(\\ell)``."""
    sp = chemical_species()[key]
    body = "".join(
        el + ("" if n == 1 else (f"_{int(n)}" if n < 10 else f"_{{{int(n)}}}"))
        for el, n in sp.formula
    )
    phase = {"l": r"(\ell)", "s": r"(s)", "g": ""}[sp.phase]
    return rf"\mathrm{{{body}}}{phase}"


def _molar(value_J_per_mol: float, system: UnitSystem) -> str:
    return _q(value_J_per_mol, _EM, system)


def _molar_n(value_J_per_mol: float, system: UnitSystem) -> str:
    return _n(value_J_per_mol, _EM, system)


def species_steps(result: SpeciesExergy, system: UnitSystem) -> list[ProcedureStep]:
    """La exergía química de una sustancia: la tabla y el método de Szargut."""
    sp = result.species
    f = _tex_formula(sp.key)
    steps: list[ProcedureStep] = []
    M = sp.M_kg_per_mol
    M_txt = (
        rf"{latex_number(M * 1e3, 5)}\ \mathrm{{{'lb/lbmol' if system == 'Inglés' else 'kg/kmol'}}}"
    )
    if result.source == "tabla":
        text = (
            f"La exergía química estándar del {sp.name} (25 °C, 1 atm) sale de la tabla del "
            f"{MODELS[result.model]}."
        )
    else:
        text = (
            f"El {MODELS[result.model]} no tabula el {sp.name}: se calcula con el método de "
            "Szargut (el paso siguiente)."
        )
    latex = [
        rf"\bar e^{{ch}}_{{{f}}} = {_molar(result.e_J_per_mol, system)}",
        latex_chain(
            r"e^{ch}",
            r"\frac{\bar e^{ch}}{M}",
            rf"\frac{{{_molar_n(result.e_J_per_mol, system)}}}{{{latex_number(M * 1e3, 5)}}}",
            _q(result.e_J_per_kg, _EH, system),
        ),
        rf"M = {M_txt}",
    ]
    steps.append(ProcedureStep("Exergía química estándar", text, tuple(latex)))
    m = result.method
    if m is not None:
        text = (
            "**Método de Szargut**: la sustancia se forma desde sus elementos (Δg_f) y cada "
            "elemento lleva la exergía química de su sustancia de referencia: "
            "ē = Δḡ_f + Σν·ē_ref. Las h_f y s° salen de los polinomios NASA del proyecto."
        )
        if result.source == "tabla":
            dev = result.method_deviation or 0.0
            text += f" Da {100 * dev:+.2f} % respecto de la tabla.".replace(".", ",", 1)
        T0 = _T_abs_number(T_STANDARD_K, system)
        rows = [
            rf"\Delta\bar h_f &= {_molar(m.hf_J_per_mol, system)}",
            rf"\bar s^\circ &= {_q(m.s0_J_per_mol_K, 'molar_entropy', system)}",
            rf"\textstyle\sum \nu\,\bar s^\circ_{{ref}} &= "
            rf"{_q(m.s0_elements_J_per_mol_K, 'molar_entropy', system)}",
        ]
        latex = [
            r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}",
            latex_chain(
                r"\Delta\bar g_f",
                r"\Delta\bar h_f - T_0\,\Delta\bar s_f",
                _wrap(
                    _molar_n(m.hf_J_per_mol, system),
                    "-",
                    rf"{T0}\cdot {latex_paren(_n(m.ds_f_J_per_mol_K, 'molar_entropy', system))}",
                ),
                _molar(m.dg_f_J_per_mol, system),
            ),
        ]
        terms = [
            rf"{latex_number(nu, 4)}\cdot {_molar_n(e, system)}" for _, nu, _, e, _ in m.elements
        ]
        rows = [r"\textstyle\sum \nu\,\bar e_{ref} &= " + terms[0]]
        rows += [rf"&\quad + {t}" for t in terms[1:]]
        rows.append(rf"&= {_molar(m.elements_J_per_mol, system)}")
        latex.append(r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}")
        latex.append(
            latex_chain(
                r"\bar e^{ch}",
                r"\Delta\bar g_f + \textstyle\sum \nu\,\bar e_{ref}",
                _molar(m.e_J_per_mol, system),
            )
        )
        steps.append(ProcedureStep("Método de Szargut (control)", text, tuple(latex)))
    if result.x_reference is not None:
        text = (
            "Los gases del aire tienen exergía química porque en el ambiente de referencia están "
            "diluidos: es el trabajo de llevarlos desde su presión parcial en el aire hasta p₀. "
            "Despejando, su fracción molar de referencia es x⁰⁰ = exp(−ē/R̄T₀)."
        )
        RT0 = R_U * T_STANDARD_K
        latex = [
            latex_chain(
                r"x^{00}",
                r"\exp\left(-\frac{\bar e^{ch}}{\bar R\,T_0}\right)",
                rf"\exp\left(-\frac{{{latex_number(result.e_J_per_mol, 5)}}}"
                rf"{{{latex_number(RT0, 5)}}}\right)",
                latex_number(result.x_reference, 4),
            )
        ]
        steps.append(ProcedureStep("Presión parcial de referencia", text, tuple(latex)))
    if result.lhv_J_per_mol is not None and result.hhv_J_per_mol is not None:
        text = (
            "Comparada con el poder calorífico (de las h_f, vademecum §16.9): φ = e/PCI. El "
            "vademecum aproxima e ≈ PCI (§16.13); para los hidrocarburos φ ≈ 1,04–1,07 y para el "
            "H₂ y el CO queda por debajo de 1 (su reacción baja la entropía)."
        )
        latex = [
            rf"\overline{{PCI}} = {_molar(result.lhv_J_per_mol, system)}",
            latex_chain(
                r"\varphi",
                r"\frac{\bar e^{ch}}{\overline{PCI}}",
                latex_number(result.ratio_lhv or 0.0, 4),
            ),
            latex_chain(
                r"\frac{\bar e^{ch}}{\overline{PCS}}", latex_number(result.ratio_hhv or 0.0, 4)
            ),
        ]
        steps.append(ProcedureStep("Exergía y poder calorífico", text, tuple(latex)))
    return _numbered(steps)


def mixture_steps(result: MixtureExergy, system: UnitSystem) -> list[ProcedureStep]:
    """La exergía química de una mezcla de gases (con el agua que condensa)."""
    steps: list[ProcedureStep] = []
    RT0 = R_U * T_STANDARD_K
    if result.condenses:
        text = (
            "A 25 °C y 1 atm el vapor de agua de la mezcla supera su presión de saturación: una "
            "parte condensa. El gas queda saturado (x_v = p_sat/p₀) y el agua líquida entra con "
            "su propia exergía química (Bejan et al., 1996)."
        )
        y_w = next(r.x_in for r in result.rows if r.key == "H2O")
        latex = [
            latex_chain(
                r"x_{v,sat}",
                r"\frac{p_{sat}(25\,^\circ\mathrm{C})}{p_0}",
                latex_number(result.x_sat, 4),
            ),
            latex_chain(
                r"n_g",
                r"\frac{1 - y_{\mathrm{H_2O}}}{1 - x_{v,sat}}",
                rf"\frac{{1 - {latex_number(y_w, 4)}}}{{1 - {latex_number(result.x_sat, 4)}}}",
                latex_number(result.n_gas, 5),
            ),
            latex_chain(r"n_\ell", r"1 - n_g", latex_number(result.n_liquid, 5)),
        ]
        steps.append(ProcedureStep("Agua que condensa", text, tuple(latex)))
    text = (
        "Para una mezcla de gases ideales (Szargut et al., 1988): ē = Σ x·ē + R̄T₀·Σ x·ln x. El "
        "primer término es la exergía de cada componente puro; el segundo, negativo, la que se "
        "pierde por estar mezclados. Un renglón por componente: x·ē + R̄T₀·x·ln x."
    )
    rows = []
    for r in result.rows:
        f = _tex_formula(r.key)
        rows.append(
            rf"{f}:\quad {latex_number(r.x_gas, 4)}\cdot {_molar_n(r.e_J_per_mol, system)} "
            rf"&{'+' if r.mixing_J_per_mol >= 0 else '-'} "
            rf"{_molar_n(abs(r.mixing_J_per_mol), system)}"
        )
    latex = [
        rf"\bar R\,T_0 = {latex_number(RT0, 5)}\ \mathrm{{J/mol}}",
        r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}",
        latex_chain(r"\textstyle\sum x\,\bar e", _molar(result.sum_x_e_J_per_mol, system)),
        latex_chain(r"\bar R T_0 \textstyle\sum x \ln x", _molar(result.mixing_J_per_mol, system)),
    ]
    if result.condenses:
        latex.append(
            latex_chain(
                r"\bar e^{ch}",
                r"n_g\,\bar e_g + n_\ell\,\bar e_{\mathrm{H_2O}(\ell)}",
                _molar(result.e_J_per_mol, system),
            )
        )
    else:
        latex.append(latex_chain(r"\bar e^{ch}", _molar(result.e_J_per_mol, system)))
    latex.append(
        latex_chain(
            r"e^{ch}",
            r"\frac{\bar e^{ch}}{M}",
            _q(result.e_J_per_kg, _EH, system),
        )
    )
    steps.append(ProcedureStep("Exergía química de la mezcla", text, tuple(latex)))
    return _numbered(steps)


def fuel_steps(result: FuelExergy, system: UnitSystem) -> list[ProcedureStep]:
    """La exergía química de un combustible por su análisis elemental (Szargut y Styrylska)."""
    u = result.ultimate
    steps: list[ProcedureStep] = []
    text = (
        "La correlación usa las relaciones **en masa** entre los elementos y el carbono, en base "
        "seca (las mismas en cualquier base)."
    )
    latex = [
        latex_chain(
            r"\frac{h}{c}",
            rf"\frac{{{latex_number(100 * u.H, 4)}}}{{{latex_number(100 * u.C, 4)}}}",
            latex_number(result.h_c, 4),
        ),
        latex_chain(
            r"\frac{o}{c}",
            rf"\frac{{{latex_number(100 * u.O, 4)}}}{{{latex_number(100 * u.C, 4)}}}",
            latex_number(result.o_c, 4),
        ),
        rf"\frac{{n}}{{c}} = {latex_number(result.n_c, 4)} \qquad "
        rf"\frac{{s}}{{c}} = {latex_number(result.s_c, 4)}",
    ]
    steps.append(ProcedureStep("Relaciones en masa", text, tuple(latex)))
    hc, oc, nc, sc = (latex_number(x, 4) for x in (result.h_c, result.o_c, result.n_c, result.s_c))
    if result.branch == "carbón":
        text = "Con o/c ≤ 0,667 (carbones, lignitos, coque, turba), Szargut y Styrylska (1964):"
        formula = (
            r"1.0437 + 0.1882\,\tfrac{h}{c} \\ &\quad + 0.0610\,\tfrac{o}{c}"
            r" \\ &\quad + 0.0404\,\tfrac{n}{c}"
        )
        numbers = (
            rf"1.0437 + 0.1882\cdot {hc} \\ &\quad + 0.0610\cdot {oc}"
            rf" \\ &\quad + 0.0404\cdot {nc}"
        )
    elif result.branch == "madera":
        text = (
            "Con 0,667 < o/c ≤ 2,67 (madera y biomasa), Szargut y Styrylska (1964). Con o/c → 0 da "
            "casi la forma de los carbones."
        )
        formula = (
            r"\big[1.0438 + 0.1882\,\tfrac{h}{c} \\ &\quad - 0.2509\,\tfrac{o}{c}\,"
            r"(1 + 0.7256\,\tfrac{h}{c}) \\ &\quad + 0.0383\,\tfrac{n}{c}\big] \\ &\quad"
            r" \div (1 - 0.3035\,\tfrac{o}{c})"
        )
        numbers = (
            rf"\big[1.0438 + 0.1882\cdot {hc} \\ &\quad - 0.2509\cdot {oc} \\ &\qquad"
            rf" \cdot (1 + 0.7256\cdot {hc}) \\ &\quad + 0.0383\cdot {nc}\big] \\ &\quad"
            rf" \div (1 - 0.3035\cdot {oc})"
        )
    else:
        text = "Combustibles líquidos, Szargut y Styrylska (1964):"
        formula = (
            r"1.0401 + 0.1728\,\tfrac{h}{c} \\ &\quad + 0.0432\,\tfrac{o}{c}"
            r" \\ &\quad + 0.2169\,\tfrac{s}{c} \\ &\qquad \cdot (1 - 2.0628\,\tfrac{h}{c})"
        )
        numbers = (
            rf"1.0401 + 0.1728\cdot {hc} \\ &\quad + 0.0432\cdot {oc}"
            rf" \\ &\quad + 0.2169\cdot {sc} \\ &\qquad \cdot (1 - 2.0628\cdot {hc})"
        )
    latex = [latex_chain(r"\beta", formula, numbers, latex_number(result.beta, 5))]
    steps.append(ProcedureStep("β = e/PCI de la materia seca", text, tuple(latex)))
    text = (
        "Por kg tal cual (Kotas, 1985): la materia orgánica vale β·PCI*, donde "
        "PCI* = PCI + W·h_fg es el PCI de la parte seca; el azufre de un sólido suma su exergía "
        "química menos el calor que ya contó el PCI (S → SO₂), y la humedad, la del agua líquida."
    )
    W = result.moisture
    latex = [
        latex_chain(
            r"PCI^{*}",
            r"PCI + W\,h_{fg}",
            rf"{_n(result.lhv_ar_J_per_kg, _EH, system)} \\ &\quad + "
            rf"{latex_number(W, 4)}\cdot {_n(result.h_fg_J_per_kg, _EH, system)}",
            _q(result.lhv_plus_water_J_per_kg, _EH, system),
        ),
        latex_chain(
            r"e_{org}",
            r"\beta\,PCI^{*}",
            _q(result.e_organic_J_per_kg, _EH, system),
        ),
    ]
    if result.e_sulfur_J_per_kg != 0.0:
        latex.append(
            latex_chain(
                r"e_S",
                r"(e^{ch}_S - PCI_S)\,S",
                rf"({_n(result.e_S_J_per_kg, _EH, system)} - "
                rf"{_n(result.lhv_S_J_per_kg, _EH, system)}) \\ &\quad "
                rf"\cdot {latex_number(result.S_ar, 4)}",
                _q(result.e_sulfur_J_per_kg, _EH, system),
            )
        )
    if result.e_water_J_per_kg != 0.0:
        latex.append(
            latex_chain(
                r"e_w",
                r"e^{ch}_{\mathrm{H_2O}(\ell)}\,W",
                _q(result.e_water_J_per_kg, _EH, system),
            )
        )
    latex.append(
        latex_chain(
            r"e^{ch}",
            r"e_{org} + e_S + e_w",
            _q(result.e_J_per_kg, _EH, system),
        )
    )
    if result.ratio_lhv is not None:
        latex.append(latex_chain(r"\frac{e^{ch}}{PCI}", latex_number(result.ratio_lhv, 4)))
    latex.append(latex_chain(r"\frac{e^{ch}}{PCS}", latex_number(result.ratio_hhv, 4)))
    steps.append(ProcedureStep("Exergía química por kg tal cual", text, tuple(latex)))
    return _numbered(steps)


# ---------------------------------------------------------------------
# Planta
# ---------------------------------------------------------------------

#: (F, P) de cada tipo de componente, para el procedimiento.
_DEFINITIONS: dict[str, tuple[str, str]] = {
    "turbina": (r"\textstyle\sum \dot m\,\psi_{ent} - \sum \dot m\,\psi_{sal}", r"\dot W"),
    "compresor": (r"\dot W", r"\textstyle\sum \dot m\,\psi_{sal} - \sum \dot m\,\psi_{ent}"),
    "bomba": (r"\dot W", r"\textstyle\sum \dot m\,\psi_{sal} - \sum \dot m\,\psi_{ent}"),
    "caldera": (r"\dot Q\,(1 - T_0/T_H)", r"\dot m\,(\psi_{sal} - \psi_{ent})"),
    "recalentador": (r"\dot Q\,(1 - T_0/T_H)", r"\dot m\,(\psi_{sal} - \psi_{ent})"),
    "calentador": (r"\dot Q\,(1 - T_0/T)", r"\dot m\,(\psi_{sal} - \psi_{ent})"),
    "cámara de combustión": (r"\dot m_c\,e_c", r"\dot m_g\,x_g - \dot m_a\,x_a"),
    "evaporador": (r"\dot m\,(\psi_{ent} - \psi_{sal})", r"\dot Q_C\,(T_0/T_C - 1)"),
    "escape": (r"\dot m\,(\psi + e^{ch})", "0"),
    "chimenea": (r"\dot m\,(\psi + e^{ch})", "0"),
}
_EXCHANGER = (
    r"\textstyle\sum (\dot m\,\Delta\psi)_{\text{las que pierden}}",
    r"\textstyle\sum (\dot m\,\Delta\psi)_{\text{las que ganan}}",
)
_DISSIPATIVE = (r"\textstyle\sum \dot m\,\psi_{ent} - \sum \dot m\,\psi_{sal}", "0")


def _definition(c: ComponentExergy) -> tuple[str, str]:
    if c.kind == "condensador" and not c.dissipative:  # bomba de calor
        return (r"\dot m\,(\psi_{ent} - \psi_{sal})", r"\dot Q_H\,(1 - T_0/T_H)")
    if c.kind == "evaporador" and c.dissipative:
        return _DISSIPATIVE
    if c.kind in _DEFINITIONS:
        return _DEFINITIONS[c.kind]
    if c.dissipative:
        return _DISSIPATIVE
    return _EXCHANGER


def _component_step(plant: PlantExergy, c: ComponentExergy, system: UnitSystem) -> ProcedureStep:
    from core.exergy.plant import KIND_EXPLANATIONS

    F_def, P_def = _definition(c)
    lines = [
        latex_chain(r"\dot X_F", F_def, _q(c.fuel_W, _W, system)),
        latex_chain(r"\dot X_P", P_def, _q(c.product_W, _W, system)),
    ]
    if c.loss_W:
        lines.append(rf"\dot X_L = {_q(c.loss_W, _W, system)}")
    if not c.is_loss:
        lines.append(
            latex_chain(
                r"\dot X_D",
                r"\dot X_F - \dot X_P - \dot X_L" if c.loss_W else r"\dot X_F - \dot X_P",
                _q(c.destroyed_W, _W, system),
            )
        )
    if c.efficiency is not None:
        lines.append(
            latex_chain(
                r"\varepsilon",
                r"\frac{\dot X_P}{\dot X_F}",
                rf"{latex_number(100 * c.efficiency, 4)}\,\%",
            )
        )
    lines.append(
        latex_chain(
            r"y_D" if not c.is_loss else r"y_L",
            r"\frac{\dot X_D}{\dot X_{entra}}"
            if not c.is_loss
            else r"\frac{\dot X_L}{\dot X_{entra}}",
            rf"{latex_number(100 * (c.loss_W if c.is_loss else c.destroyed_W) / plant.fuel_W, 4)}"
            r"\,\%",
        )
    )
    expl = KIND_EXPLANATIONS.get(c.kind, "")
    if c.is_loss:
        text = f"Pérdida: {expl}." if expl else "Pérdida de la planta."
    elif c.dissipative:
        text = (
            "Componente disipativo (no tiene producto): lo que gasta se destruye"
            + (" o se pierde" if c.loss_W else "")
            + (f". Lo que destruye exergía: {expl}." if expl else ".")
        )
    else:
        text = f"Lo que destruye exergía: {expl}." if expl else ""
    if c.destroyed_W != 0.0 and not c.is_loss and c.kind not in ("cámara de combustión",):
        text += " ẊD se calcula también como T₀·Ṡ_gen (balance de entropía) y coincide."
    return ProcedureStep(c.name, text, tuple(lines))


def _input_symbol(name: str) -> str:
    """Subíndice corto de una entrada de exergía: Q, W, comb o aire."""
    if name.startswith("Calor"):
        return "Q"
    if name.startswith("Trabajo"):
        return "W"
    if name.startswith("Aire"):
        return "aire"
    return "comb"


def plant_steps(plant: PlantExergy, system: UnitSystem) -> list[ProcedureStep]:
    """El balance de exergía de la planta, componente por componente."""
    steps: list[ProcedureStep] = []
    T0 = _absolute_T(plant.T0_K, system)
    text = (
        "Cada corriente lleva su exergía de flujo ψ = (h − h₀) − T₀·(s − s₀) (vademecum §11.3), "
        "con el estado muerto del ambiente (la tabla de corrientes trae cada ψ). Para cada "
        "componente: Ẋ_F es la exergía que gasta y Ẋ_P la que produce; la diferencia se destruye "
        "o se pierde (Bejan, Tsatsaronis y Moran, 1996)."
    )
    lines = [rf"T_0 = {T0}", rf"p_0 = {_q(plant.p0_Pa, 'pressure', system)}"]
    lines.append(r"\dot X_F = \dot X_P + \dot X_D + \dot X_L")
    steps.append(ProcedureStep("Ambiente y definiciones", text, tuple(lines)))
    # exergía que entra: un símbolo corto por entrada (el nombre va en el texto)
    symbols = [_input_symbol(name) for name, _ in plant.inputs]
    text = {
        "calor": "Entra el calor de la fuente con su exergía Q̇·(1 − T₀/T) (vademecum §11.5).",
        "trabajo": "Entra el trabajo de los compresores: es exergía pura.",
        "pci": "Entra el combustible con su exergía ≈ PCI (vademecum §16.13).",
        "szargut": (
            "Entra el combustible con su exergía química (Szargut et al., 1988) y el aire con la "
            "suya, chica (el aire seco está un poco más «seco» que el de referencia)."
        ),
    }.get(plant.fuel_model, "")
    text += (
        " "
        + "; ".join(
            f"Ẋ_{sym}: {name.lower()}" for sym, (name, _) in zip(symbols, plant.inputs, strict=True)
        )
        + "."
    )
    latex = [
        rf"\dot X_{{\mathrm{{{sym}}}}} = {_q(x, _W, system)}"
        for sym, (_, x) in zip(symbols, plant.inputs, strict=True)
    ]
    latex.append(latex_chain(r"\dot X_{entra}", _q(plant.fuel_W, _W, system)))
    steps.append(ProcedureStep("Exergía que entra", text, tuple(latex)))
    for c in plant.components:
        steps.append(_component_step(plant, c, system))
    # balance
    text = (
        "La exergía que entra termina como producto, destruida en los componentes o perdida "
        "(vademecum §11.9). El rendimiento exergético es η_II = Ẋ_producto/Ẋ_entra (§11.10)."
    )
    latex = [
        latex_chain(r"\dot X_{producto}", _q(plant.product_W, _W, system)),
        latex_chain(r"\textstyle\sum \dot X_D", _q(plant.destroyed_W, _W, system)),
        latex_chain(r"\textstyle\sum \dot X_L", _q(plant.loss_W, _W, system)),
        latex_chain(
            r"\dot X_{entra}",
            r"\dot X_{producto} + \textstyle\sum \dot X_D + \sum \dot X_L",
            _q(plant.product_W + plant.destroyed_W + plant.loss_W, _W, system),
        ),
        latex_chain(
            r"\eta_{II}",
            r"\frac{\dot X_{producto}}{\dot X_{entra}}",
            rf"{latex_number(100 * plant.efficiency, 4)}\,\%",
        ),
    ]
    steps.append(ProcedureStep("Balance de la planta", text, tuple(latex)))
    return _numbered(steps)
