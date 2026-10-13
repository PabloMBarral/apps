"""Procedimiento didáctico de la ebullición y la condensación (LaTeX), en el sistema activo.

Çengel y Ghajar (2015), §10-2 y §10-5 a §10-7; Incropera et al. (2007), cap. 10:

- :func:`boiling_steps`: las propiedades a T_sat, el flujo crítico con L* y C_cr, la
  ebullición nucleada con Rohsenow (q̇ = q̇₀·ψ³ o, al revés, ψ = (q̇/q̇₀)^⅓), la convección
  natural, la transición, la película con la radiación, y el calor y el caudal evaporado;
- :func:`condensation_steps`: las propiedades (el líquido a T_f), h*_fg, el régimen de la
  película con su Re, h según la geometría, el calor, el caudal condensado, el espesor y
  las gotas.

Factores de unidades a la vista: con h_fg en kJ/kg (Técnico) los flujos llevan 1000; con
k y q̇ por hora y μ por segundo (Inglés), 3600. En Bromley y Nusselt el factor va adentro de
la raíz cuarta. En el Inglés σ va en lbf/ft y g·Δρ/σ lleva g_c = 32,174 lbm·ft/(lbf·s²).
"""

from __future__ import annotations

import math

from core.heat_transfer.boiling import (
    CRITICAL_CASES,
    HEATERS,
    PoolBoilingResult,
)
from core.heat_transfer.condensation import RE_VAPOR_CHATO, CondensationResult
from core.heat_transfer.convection_procedure import big, correlation_latex
from core.heat_transfer.phase_change import G, with_article
from core.heat_transfer.procedure_common import (
    Pack,
    chain,
    dimensionless,
    flush,
    frac,
    n,
    numbered,
    prod,
    q,
    sub,
    times,
)
from core.heat_transfer.radiation import SIGMA
from core.latex import text_quantity
from core.state_report import ProcedureStep
from core.units_system import QuantityKind, UnitSystem, convert_from_si

__all__ = ["boiling_steps", "condensation_steps"]

_T = "temperature"
_TA = "absolute_temperature"
_DT = "temperature_difference"
_P = "pressure"
_RHO = "density"
_HFG = "specific_enthalpy"
_SIG = "surface_tension"
_MU = "dynamic_viscosity"
_CP = "specific_heat"
_K = "thermal_conductivity"
_NU = "diffusivity"
_BETA = "expansion_coefficient"
_G = "acceleration"
_L = "length"
_A = "area"
_W = "heat_rate"
_QF = "heat_flux"
_H = "heat_transfer_coefficient"
_M = "mass_flow"
_IL = "inverse_length"
_V = "speed"

#: El factor de unidades de los flujos (vacío en el SI).
_FACTOR: dict[str, str] = {"SI": "", "Técnico": "1000", "Inglés": "3600"}
_HEATER_ARTICLE = {"plate": "una placa", "cylinder": "un cilindro", "sphere": "una esfera"}
#: g_c del Inglés, con σ en lbf/ft.
_GC = "32.174"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _d(x: float, sig: int = 4) -> str:
    return dimensionless(x, sig)


def _pw(base: str, exp: str) -> str:
    """``base^{exp}`` con paréntesis si la base lleva ×10ⁿ o es negativa."""
    if r"\times" in base or base.startswith("-"):
        return rf"({base})^{{{exp}}}"
    return rf"{base}^{{{exp}}}"


def _pr_n(Pr: float, n_exp: float) -> str:
    """Pr^n (sin exponente con n = 1, como el agua)."""
    return _d(Pr) if n_exp == 1.0 else _pw(_d(Pr), _d(n_exp, 3))


def _paren_pow(tex: str, exp: str) -> str:
    r"""``\left(tex\right)^{exp}``."""
    return rf"\left({tex}\right)^{{{exp}}}"


def _with_factor(system: UnitSystem, *factors: str) -> Pack:
    """El factor del sistema adelante (nada en el SI) y los factores, partidos si no entran."""
    f = _FACTOR[system]
    return prod(f, *factors) if f else prod(*factors)


def _f_times(system: UnitSystem, tex: str) -> str:
    r"""``1000 \cdot tex`` (``tex`` en el SI)."""
    f = _FACTOR[system]
    return rf"{f} \cdot {tex}" if f else tex


def _sigma_sym(system: UnitSystem) -> str:
    r"""``\sigma`` o ``g_c\,\sigma`` (Inglés)."""
    return r"g_c\,\sigma" if system == "Inglés" else r"\sigma"


def _sigma_num(sigma: float, system: UnitSystem) -> str:
    """σ con g_c adelante en el Inglés."""
    s = n(sigma, _SIG, system)
    return times(_GC, s) if system == "Inglés" else s


def _factor_text(system: UnitSystem, root: bool = False) -> str:
    where = " (adentro de la raíz cuarta)" if root else ""
    if system == "Técnico":
        return f" Con h_fg en kJ/kg, el 1000{where} pasa el resultado a W."
    if system == "Inglés":
        return (
            f" Con μ en lb/(ft·s) y los flujos por hora, el 3600{where} pasa de segundos a horas."
        )
    return ""


def _gc_text(system: UnitSystem) -> str:
    if system == "Inglés":
        return " Con σ en lbf/ft, g_c = 32,174 lbm·ft/(lbf·s²) deja √(g·Δρ/(g_c·σ)) en 1/ft."
    return ""


def _sigma_sb(system: UnitSystem) -> str:
    """σ de Stefan–Boltzmann con su unidad."""
    value = convert_from_si(SIGMA, _QF, system) / convert_from_si(1.0, _TA, system) ** 4
    if system == "Inglés":
        unit = r"\mathrm{Btu/(h\cdot ft^{2}\cdot {}^{\circ}R^{4})}"
    else:
        unit = r"\mathrm{W/(m^{2}\cdot K^{4})}"
    return rf"{dimensionless(value, 4)}\ {unit}"


def _props(rows: list[tuple[str, str]]) -> str:
    """Un bloque ``aligned`` con una propiedad por renglón."""
    return r"\begin{aligned}" + r" \\ ".join(rf"{a} &= {b}" for a, b in rows) + r"\end{aligned}"


# ---------------------------------------------------------------------
# Ebullición en pileta (§10-2)
# ---------------------------------------------------------------------


def _boiling_properties(r: PoolBoilingResult, system: UnitSystem) -> ProcedureStep:
    i, s = r.inputs, r.sat
    rows = [
        (r"T_{\mathrm{sat}}", q(s.T_sat_K, _T, system)),
        (r"\rho_\ell", q(s.rho_l, _RHO, system)),
        (r"\rho_v", q(s.rho_v, _RHO, system)),
        (r"h_{fg}", q(s.h_fg, _HFG, system)),
        (r"\sigma", q(s.sigma, _SIG, system)),
        (r"\mu_\ell", q(s.mu_l, _MU, system)),
        (r"c_{p\ell}", q(s.cp_l, _CP, system)),
        (r"\mathrm{Pr}_\ell", _d(s.Pr_l)),
    ]
    if r.regime == "natural":
        rows += [
            (r"k_\ell", q(s.k_l, _K, system)),
            (r"\nu_\ell", q(s.nu_l, _NU, system)),
            (r"\alpha_\ell", q(s.alpha_l, _NU, system)),
            (r"\beta_\ell", q(s.beta_l, _BETA, system)),
        ]
    latex = [
        _props(rows[:4]),
        _props(rows[4:8]),
    ]
    if len(rows) > 8:
        latex.append(_props(rows[8:]))
    text = (
        f"Saturación {_del(i.fluid)} a p = {_txt(i.p_Pa, _P, system)}: el líquido y el vapor a "
        "T_sat, de CoolProp (los libros las leen de las tablas de saturación)."
    )
    return ProcedureStep("Propiedades a la saturación", text, tuple(latex))


def _txt(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Una cantidad para el texto: «1,0133 bar»."""
    return text_quantity(value_si, kind, system, sig).replace(".", ",")


def _root_term(r: PoolBoilingResult, system: UnitSystem) -> tuple[str, str, str]:
    r"""(símbolo, sustitución, valor) de √(g·Δρ/σ), en 1/m o 1/ft."""
    s = r.sat
    sym = rf"\sqrt{{\frac{{g\,(\rho_\ell - \rho_v)}}{{{_sigma_sym(system)}}}}}"
    g_drho = times(n(G, _G, system), n(s.delta_rho, _RHO, system))
    sub_ = rf"\sqrt{{{frac(g_drho, _sigma_num(s.sigma, system))}}}"
    value = math.sqrt(G * s.delta_rho / s.sigma)
    return sym, sub_, q(value, _IL, system)


def _critical_step(r: PoolBoilingResult, system: UnitSystem) -> ProcedureStep:
    i, s = r.inputs, r.sat
    L = r.L_star / math.sqrt(G * s.delta_rho / s.sigma)
    sym, sub_, root = _root_term(r, system)
    latex = [chain(sym, sub_, root)]
    L_tex = "D" if i.heater == "plate" else r"\frac{D}{2}"
    latex.append(chain("L", L_tex, q(L, _L, system)))
    latex.append(
        chain(
            "L^*",
            r"L\,\sqrt{g\,\Delta\rho/(g_c\,\sigma)}"
            if system == "Inglés"
            else r"L\,\sqrt{g\,\Delta\rho/\sigma}",
            times(n(L, _L, system), n(math.sqrt(G * s.delta_rho / s.sigma), _IL, system)),
            _d(r.L_star),
        )
    )
    name, rng = CRITICAL_CASES[r.critical_case]
    if r.critical_case == "cylinder_small":
        latex.append(
            chain(
                r"C_{cr}",
                r"0.12\,L^{*-1/4}",
                rf"0.12 \cdot {_pw(_d(r.L_star), '-1/4')}",
                _d(r.C_cr),
            )
        )
    elif r.critical_case == "sphere_small":
        latex.append(
            chain(
                r"C_{cr}",
                r"0.227\,L^{*-1/2}",
                rf"0.227 \cdot {_pw(_d(r.L_star), '-1/2')}",
                _d(r.C_cr),
            )
        )
    else:
        latex.append(rf"C_{{cr}} = {_d(r.C_cr, 3)}")
    sig = _sigma_sym(system)
    bracket_sys = (
        float(convert_from_si(s.sigma, _SIG, system))
        * (32.17404855643 if system == "Inglés" else 1.0)
        * float(convert_from_si(G, _G, system))
        * float(convert_from_si(s.rho_v, _RHO, system)) ** 2
        * float(convert_from_si(s.delta_rho, _RHO, system))
    )
    latex.append(
        chain(
            r"Z",
            rf"{sig}\,g\,\rho_v^2\,(\rho_\ell - \rho_v)",
            prod(
                _sigma_num(s.sigma, system),
                n(G, _G, system),
                _pw(n(s.rho_v, _RHO, system), "2"),
                n(s.delta_rho, _RHO, system),
            ),
            _d(bracket_sys),
        )
    )
    latex.append(
        chain(
            r"\dot q_{\text{máx}}",
            r"C_{cr}\,h_{fg}\,Z^{1/4}",
            _with_factor(system, _d(r.C_cr), n(s.h_fg, _HFG, system), _pw(_d(bracket_sys), "1/4")),
            q(r.q_max, _QF, system),
        )
    )
    text = (
        "El flujo crítico (Zuber, 1958) con la constante de Lienhard y Dhir (1973), que depende "
        f"de la geometría y de su tamaño adimensional L* (L sobre la longitud capilar). "
        f"{HEATERS[i.heater]}: {name.lower()} ({rng}). Z es el corchete de Zuber."
        + _gc_text(system)
        + _factor_text(system)
    )
    return ProcedureStep("Flujo crítico (Zuber, Lienhard y Dhir)", text, tuple(latex))


def _q0(r: PoolBoilingResult) -> float:
    s = r.sat
    return s.mu_l * s.h_fg * math.sqrt(G * s.delta_rho / s.sigma)


def _q0_latex(r: PoolBoilingResult, system: UnitSystem) -> str:
    s = r.sat
    root = n(math.sqrt(G * s.delta_rho / s.sigma), _IL, system)
    return chain(
        r"\dot q_0",
        r"\mu_\ell\,h_{fg}\,\sqrt{g\,\Delta\rho/\sigma}",
        _with_factor(system, n(s.mu_l, _MU, system), n(s.h_fg, _HFG, system), root),
        q(_q0(r), _QF, system),
    )


def _psi_formula() -> str:
    return r"\frac{c_{p\ell}\,\Delta T_e}{C_{sf}\,h_{fg}\,\mathrm{Pr}_\ell^{\,n}}"


def _nucleate_step(r: PoolBoilingResult, system: UnitSystem, dT: float) -> ProcedureStep:
    """Rohsenow con ΔT dado: q̇ = q̇₀·ψ³."""
    i, s = r.inputs, r.sat
    psi = s.cp_l * dT / (i.C_sf * s.h_fg * s.Pr_l**i.n)
    q_nb = _q0(r) * psi**3
    latex = [
        _q0_latex(r, system),
        flush(
            r"\psi",
            _psi_formula(),
            frac(
                times(n(s.cp_l, _CP, system), n(dT, _DT, system)),
                times(_d(i.C_sf), n(s.h_fg, _HFG, system), _pr_n(s.Pr_l, i.n)),
            ),
            _d(psi),
        ),
        chain(
            r"\dot q",
            r"\dot q_0\,\psi^3",
            times(n(_q0(r), _QF, system), _pw(_d(psi), "3")),
            q(q_nb, _QF, system),
        ),
    ]
    text = (
        "Ebullición nucleada con Rohsenow (1952): q̇ = μ_ℓ·h_fg·√(g·Δρ/σ)·ψ³, con "
        f"C_sf = {_comma(i.C_sf)} y n = {_comma(i.n)} del par fluido–superficie. q̇₀ es lo que "
        "multiplica a ψ³ y ψ, el corchete adimensional." + _gc_text(system) + _factor_text(system)
    )
    return ProcedureStep("Ebullición nucleada (Rohsenow)", text, tuple(latex))


def _del(fluid: str) -> str:
    """«del agua», «del n-pentano», …"""
    return "del " + with_article(fluid)[3:]


def _comma(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _inverse_nucleate_step(
    r: PoolBoilingResult, system: UnitSystem, q_value: float, title: str, what: str
) -> ProcedureStep:
    """Rohsenow al revés: ψ = (q̇/q̇₀)^⅓ y ΔT_e = ψ·C_sf·h_fg·Pr^n/c_pℓ."""
    i, s = r.inputs, r.sat
    psi = (q_value / _q0(r)) ** (1.0 / 3.0)
    dT = psi * i.C_sf * s.h_fg * s.Pr_l**i.n / s.cp_l
    latex = [
        _q0_latex(r, system),
        chain(
            r"\psi",
            r"(\dot q/\dot q_0)^{1/3}",
            _paren_pow(frac(n(q_value, _QF, system), n(_q0(r), _QF, system)), "1/3"),
            _d(psi),
        ),
        flush(
            r"\Theta",
            r"\frac{C_{sf}\,h_{fg}\,\mathrm{Pr}_\ell^{\,n}}{c_{p\ell}}",
            frac(
                times(_d(i.C_sf), n(s.h_fg, _HFG, system), _pr_n(s.Pr_l, i.n)),
                n(s.cp_l, _CP, system),
            ),
            q(dT / psi, _DT, system),
        ),
        chain(
            r"\Delta T_e",
            r"\psi\,\Theta",
            times(_d(psi), n(dT / psi, _DT, system)),
            q(dT, _DT, system),
        ),
        chain(
            "T_s",
            r"T_{\mathrm{sat}} + \Delta T_e",
            rf"{n(s.T_sat_K, _T, system)} + {n(dT, _DT, system)}",
            q(s.T_sat_K + dT, _T, system),
        ),
    ]
    text = (
        f"{what} Rohsenow despejado: ψ = (q̇/q̇₀)^⅓ y ΔT_e = ψ·Θ, con Θ = C_sf·h_fg·Pr^n/c_pℓ, "
        f"C_sf = {_comma(i.C_sf)} y n = {_comma(i.n)}. Con q̇ dado el error de Rohsenow baja a "
        "±30 % en ΔT (el cubo lo achica)." + _gc_text(system) + _factor_text(system)
    )
    return ProcedureStep(title, text, tuple(latex))


def _natural_step(r: PoolBoilingResult, system: UnitSystem) -> ProcedureStep:
    i, s, nc = r.inputs, r.sat, r.natural
    L_tex = {"plate": r"\frac{A}{P} = \frac{D}{4}", "cylinder": "D", "sphere": "D"}[i.heater]
    latex = [
        chain("L_c", L_tex, q(nc.L_m, _L, system)),
        chain(
            "Ra",
            r"\frac{g\,\beta_\ell\,\Delta T_e}{\nu_\ell\,\alpha_\ell}\,L_c^3",
            Pack(
                frac(
                    times(n(G, _G, system), n(s.beta_l, _BETA, system), n(r.dT_K, _DT, system)),
                    times(n(s.nu_l, _NU, system), n(s.alpha_l, _NU, system)),
                ),
                ((r"\cdot", _pw(n(nc.L_m, _L, system), "3")),),
            ),
            big(nc.Ra),
        ),
    ]
    key = {
        "plate": "horizontal_upper",
        "cylinder": "horizontal_cylinder_churchill_chu",
        "sphere": "sphere_churchill",
    }[i.heater]
    latex += correlation_latex(key, {"Ra": nc.Ra, "Pr": s.Pr_l}, nc.Nu)
    latex.append(
        chain(
            "h",
            r"\frac{\mathrm{Nu}\,k_\ell}{L_c}",
            frac(times(_d(nc.Nu), n(s.k_l, _K, system)), n(nc.L_m, _L, system)),
            q(nc.h, _H, system),
        )
    )
    latex.append(
        chain(
            r"\dot q",
            r"h\,\Delta T_e",
            times(n(nc.h, _H, system), n(r.dT_K, _DT, system)),
            q(nc.q, _QF, system),
        )
    )
    text = (
        f"Con ΔT_e chico todavía no hay burbujas: el calor sale por convección natural del "
        f"líquido ({nc.correlation}, con las propiedades del líquido saturado). Rohsenow daría "
        f"{_txt(r.nucleate_q, _QF, system, 3)}, menos que esto; la nucleada arranca hacia "
        f"ΔT_e ≈ {_txt(r.dT_onset, _DT, system, 3)}."
    )
    if r.inputs.mode == "flux":
        text += " Con q̇ dado, ΔT_e sale de iterar h(ΔT_e)·ΔT_e = q̇ (h depende de ΔT_e por Ra)."
    return ProcedureStep("Convección natural", text, tuple(latex))


def _minimum_latex(r: PoolBoilingResult, system: UnitSystem) -> list[str]:
    s = r.sat
    sig = _sigma_sym(system)
    W = (
        float(convert_from_si(s.sigma, _SIG, system))
        * (32.17404855643 if system == "Inglés" else 1.0)
        * float(convert_from_si(G, _G, system))
        * float(convert_from_si(s.delta_rho, _RHO, system))
        / float(convert_from_si(s.rho_l + s.rho_v, _RHO, system)) ** 2
    )
    return [
        flush(
            "Y",
            rf"\frac{{{sig}\,g\,(\rho_\ell - \rho_v)}}{{(\rho_\ell + \rho_v)^2}}",
            frac(
                times(_sigma_num(s.sigma, system), n(G, _G, system), n(s.delta_rho, _RHO, system)),
                _pw(n(s.rho_l + s.rho_v, _RHO, system), "2"),
            ),
            _d(W),
        ),
        chain(
            r"\dot q_{\text{mín}}",
            r"0.09\,\rho_v\,h_{fg}\,Y^{1/4}",
            _with_factor(
                system, "0.09", n(s.rho_v, _RHO, system), n(s.h_fg, _HFG, system), _pw(_d(W), "1/4")
            ),
            q(r.q_min, _QF, system),
        ),
    ]


def _transition_step(r: PoolBoilingResult, system: UnitSystem) -> ProcedureStep:
    t = math.log(r.dT_K / r.dT_max) / math.log(r.dT_min / r.dT_max)
    latex = _minimum_latex(r, system)
    latex += [
        flush(
            "t",
            frac(
                r"\ln(\Delta T_e/\Delta T_{\text{máx}})",
                r"\ln(\Delta T_{\text{mín}}/\Delta T_{\text{máx}})",
            ),
            frac(
                rf"\ln({frac(n(r.dT_K, _DT, system), n(r.dT_max, _DT, system))})",
                rf"\ln({frac(n(r.dT_min, _DT, system), n(r.dT_max, _DT, system))})",
            ),
            _d(t),
        ),
        chain(
            r"\frac{\dot q_{\text{mín}}}{\dot q_{\text{máx}}}",
            frac(n(r.q_min, _QF, system), n(r.q_max, _QF, system)),
            _d(r.q_min / r.q_max),
        ),
        chain(
            r"\dot q",
            r"\dot q_{\text{máx}}\,(\dot q_{\text{mín}}/\dot q_{\text{máx}})^{t}",
            times(n(r.q_max, _QF, system), _pw(_d(r.q_min / r.q_max), _d(t))),
            q(r.q, _QF, system),
        ),
    ]
    text = (
        "El flujo mínimo, el punto de Leidenfrost (Zuber, 1958, con el 0,09 de Berenson, 1961)."
        " Para la ebullición de transición no hay una correlación confiable: la app interpola "
        f"en log–log entre el flujo crítico (ΔT_máx = {_txt(r.dT_max, _DT, system, 4)}, de "
        f"Rohsenow con q̇_máx) y el mínimo (ΔT_mín = {_txt(r.dT_min, _DT, system, 4)}, donde la "
        "película de Bromley baja a q̇_mín; sale numérico)."
        + _gc_text(system)
        + _factor_text(system)
    )
    return ProcedureStep("Transición (interpolada)", text, tuple(latex))


def _film_step(r: PoolBoilingResult, system: UnitSystem) -> ProcedureStep:
    i, s, f = r.inputs, r.sat, r.film
    assert f is not None
    v = f.vapor
    dT = r.dT_K
    h_prime = s.h_fg + 0.4 * v.cp * dT
    latex = [
        chain(
            "T_f",
            r"\frac{T_s + T_{\mathrm{sat}}}{2}",
            frac(f"{n(r.T_s_K, _T, system)} + {n(s.T_sat_K, _T, system)}", "2"),
            q(v.T_K, _T, system),
        ),
        _props(
            [
                (r"\rho_v", q(v.rho, _RHO, system)),
                (r"c_{pv}", q(v.cp, _CP, system)),
                (r"\mu_v", q(v.mu, _MU, system)),
                (r"k_v", q(v.k, _K, system)),
            ]
        ),
        chain(
            "h'_{fg}",
            r"h_{fg} + 0.4\,c_{pv}\,\Delta T_e",
            Pack(
                n(s.h_fg, _HFG, system),
                (("+", times("0.4", n(v.cp, _CP, system), n(dT, _DT, system))),),
            ),
            q(h_prime, _HFG, system),
        ),
    ]
    g_drho = times(n(G, _G, system), n(s.delta_rho, _RHO, system))
    if i.heater == "plate":
        latex.append(
            chain(
                r"L_b",
                rf"\sqrt{{\frac{{{_sigma_sym(system)}}}{{g\,(\rho_\ell - \rho_v)}}}}",
                rf"\sqrt{{{frac(_sigma_num(s.sigma, system), g_drho)}}}",
                q(f.length_m, _L, system),
            )
        )
        D_sym = "L_b"
    else:
        latex.append(rf"D = {q(f.length_m, _L, system)}")
        D_sym = "D"
    num_sys = (
        float(convert_from_si(G, _G, system))
        * float(convert_from_si(v.k, _K, system)) ** 3
        * float(convert_from_si(v.rho, _RHO, system))
        * float(convert_from_si(s.delta_rho, _RHO, system))
        * float(convert_from_si(h_prime, _HFG, system))
    )
    den_sys = (
        float(convert_from_si(v.mu, _MU, system))
        * float(convert_from_si(f.length_m, _L, system))
        * float(convert_from_si(dT, _DT, system))
    )
    ratio = frac(_d(num_sys), _d(den_sys))
    Ts4 = _pw(n(r.T_s_K, _TA, system), "4")
    Tsat4 = _pw(n(s.T_sat_K, _TA, system), "4")
    latex += [
        chain(
            "N",
            r"g\,k_v^3\,\rho_v\,(\rho_\ell - \rho_v)\,h'_{fg}",
            prod(
                n(G, _G, system),
                _pw(n(v.k, _K, system), "3"),
                n(v.rho, _RHO, system),
                n(s.delta_rho, _RHO, system),
                n(h_prime, _HFG, system),
            ),
            _d(num_sys),
        ),
        chain(
            "M",
            rf"\mu_v\,{D_sym}\,\Delta T_e",
            prod(n(v.mu, _MU, system), n(f.length_m, _L, system), n(dT, _DT, system)),
            _d(den_sys),
        ),
        chain(
            r"\dot q_{\text{pel}}",
            rf"C\,\left({_f_times(system, 'N')}/M\right)^{{1/4}}\,\Delta T_e",
            Pack(
                rf"{_d(f.C_film, 3)} \cdot {_paren_pow(_f_times(system, ratio), '1/4')}",
                ((r"\cdot", n(dT, _DT, system)),),
            ),
            q(f.q_film, _QF, system),
        ),
    ]
    if i.emissivity > 0.0:
        latex.append(
            chain(
                r"\dot q_{\text{rad}}",
                r"\varepsilon\,\sigma_{\text{SB}}\,(T_s^4 - T_{\mathrm{sat}}^4)",
                Pack(
                    times(_d(i.emissivity, 3), _sigma_sb(system).split(r"\ ")[0]),
                    (
                        (
                            r"\cdot",
                            f"({sub(Ts4, Tsat4)})",
                        ),
                    ),
                ),
                q(f.q_rad, _QF, system),
            )
        )
        latex.append(
            chain(
                r"\dot q",
                r"\dot q_{\text{pel}} + \tfrac{3}{4}\,\dot q_{\text{rad}}",
                Pack(
                    n(f.q_film, _QF, system),
                    (("+", times(r"\tfrac{3}{4}", n(f.q_rad, _QF, system))),),
                ),
                q(f.q_total, _QF, system),
            )
        )
    C_text = (
        "Berenson (1961) en una placa: C = 0,425 y la longitud capilar L_b en vez de D"
        if i.heater == "plate"
        else f"Bromley (1950): C = {_comma(f.C_film)} en {_HEATER_ARTICLE[i.heater]}"
    )
    text = (
        f"Ebullición en película: el vapor de la película a T_f (de CoolProp) y {C_text}. "
        "h′_fg suma el calor sensible del vapor."
    )
    if i.emissivity > 0.0:
        text += (
            f" La radiación de la superficie (ε = {_comma(i.emissivity)}, σ_SB = "
            f"{_sigma_sb_text(system)}, T absolutas) se suma con ¾ (Bromley)."
        )
    if r.burnout:
        text = (
            "Con q̇ mayor que el flujo crítico la superficie salta a la película: ΔT_e sale de "
            "resolver q̇_película(ΔT_e) + ¾·q̇_rad(ΔT_e) = q̇ (numérico). " + text
        )
    text += _factor_text(system, root=True)
    return ProcedureStep("Ebullición en película (Bromley)", text, tuple(latex))


def _sigma_sb_text(system: UnitSystem) -> str:
    """σ de Stefan–Boltzmann para el texto: 5,670·10⁻⁸ W/(m²·K⁴) o 0,1712·10⁻⁸ Btu/(h·ft²·°R⁴)."""
    value = convert_from_si(SIGMA, _QF, system) / convert_from_si(1.0, _TA, system) ** 4
    if system == "Inglés":
        return f"{value * 1e8:.4f}".replace(".", ",") + "·10⁻⁸ Btu/(h·ft²·°R⁴)"
    return f"{value * 1e8:.3f}".replace(".", ",") + "·10⁻⁸ W/(m²·K⁴)"


def _boiling_totals(r: PoolBoilingResult, system: UnitSystem) -> ProcedureStep:
    i, s = r.inputs, r.sat
    D = n(i.D_m, _L, system)
    if i.heater == "plate":
        area = chain(
            "A",
            r"\frac{\pi D^2}{4}",
            frac(rf"\pi \cdot {_pw(D, '2')}", "4"),
            q(r.area_m2, _A, system),
        )
    elif i.heater == "cylinder":
        area = chain(
            "A", r"\pi D L", times(r"\pi", D, n(i.L_m, _L, system)), q(r.area_m2, _A, system)
        )
    else:
        area = chain("A", r"\pi D^2", rf"\pi \cdot {_pw(D, '2')}", q(r.area_m2, _A, system))
    f = _FACTOR[system]
    m_den = times(f, n(s.h_fg, _HFG, system)) if f else n(s.h_fg, _HFG, system)
    latex = [
        area,
        chain(
            r"\dot Q",
            r"\dot q\,A",
            times(n(r.q, _QF, system), n(r.area_m2, _A, system)),
            q(r.Q_W, _W, system),
        ),
        chain(
            r"\dot m",
            r"\frac{\dot Q}{h_{fg}}",
            frac(n(r.Q_W, _W, system), m_den),
            q(r.m_dot, _M, system),
        ),
        chain(
            "h",
            r"\frac{\dot q}{\Delta T_e}",
            frac(n(r.q, _QF, system), n(r.dT_K, _DT, system)),
            q(r.h, _H, system),
        ),
    ]
    text = "El calor del calentador, el líquido que se evapora y el coeficiente equivalente."
    if system == "Técnico":
        text += " Con Q̇ en W y h_fg en kJ/kg, ṁ lleva el 1000."
    elif system == "Inglés":
        text += " Con Q̇ en Btu/h, ṁ sale en lb/h: dividido por 3600, en lb/s."
    return ProcedureStep("Calor y caudal evaporado", text, tuple(latex))


def boiling_steps(r: PoolBoilingResult, system: UnitSystem) -> list[ProcedureStep]:
    """El procedimiento de la ebullición en pileta, paso a paso (Çengel y Ghajar §10-2)."""
    i, s = r.inputs, r.sat
    steps = [_boiling_properties(r, system)]
    if i.mode == "temperature":
        steps.append(
            ProcedureStep(
                "Exceso de temperatura",
                "La fuerza impulsora de la ebullición es el exceso de temperatura de la superficie "
                "sobre la saturación.",
                (
                    chain(
                        r"\Delta T_e",
                        r"T_s - T_{\mathrm{sat}}",
                        sub(n(r.T_s_K, _T, system), n(s.T_sat_K, _T, system)),
                        q(r.dT_K, _DT, system),
                    ),
                ),
            )
        )
    steps.append(_critical_step(r, system))
    if i.mode == "critical":
        steps.append(
            _inverse_nucleate_step(
                r,
                system,
                r.q_max,
                "Temperatura en el flujo crítico (Rohsenow)",
                "La superficie en el flujo crítico: ",
            )
        )
    elif r.regime == "nucleate" and i.mode == "flux":
        steps.append(
            _inverse_nucleate_step(
                r, system, r.q, "Temperatura de la superficie (Rohsenow)", "Con el flujo dado: "
            )
        )
    elif r.regime == "nucleate":
        steps.append(_nucleate_step(r, system, r.dT_K))
    elif r.regime == "natural":
        steps.append(_natural_step(r, system))
    elif r.regime == "transition":
        steps.append(_transition_step(r, system))
    else:
        steps.append(_film_step(r, system))
    steps.append(_boiling_totals(r, system))
    return numbered(steps)


# ---------------------------------------------------------------------
# Condensación en película (§10-5 a §10-7)
# ---------------------------------------------------------------------


def _condensation_properties(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    i, s, liq = r.inputs, r.sat, r.liquid
    latex = [
        _props(
            [
                (r"T_{\mathrm{sat}}", q(s.T_sat_K, _T, system)),
                (r"h_{fg}", q(s.h_fg, _HFG, system)),
                (r"\rho_v", q(s.rho_v, _RHO, system)),
            ]
        ),
        chain(
            "T_f",
            r"\frac{T_{\mathrm{sat}} + T_s}{2}",
            frac(f"{n(s.T_sat_K, _T, system)} + {n(i.T_s_K, _T, system)}", "2"),
            q(liq.T_K, _T, system),
        ),
        _props(
            [
                (r"\rho_\ell", q(liq.rho, _RHO, system)),
                (r"\mu_\ell", q(liq.mu, _MU, system)),
                (r"\nu_\ell", q(liq.nu, _NU, system)),
            ]
        ),
        _props(
            [
                (r"c_{p\ell}", q(liq.cp, _CP, system)),
                (r"k_\ell", q(liq.k, _K, system)),
                (r"\mathrm{Pr}_\ell", _d(liq.Pr)),
            ]
        ),
    ]
    if r.vapor is not None:
        latex.append(rf"c_{{pv}} = {q(r.vapor.cp, _CP, system)}")
    text = (
        f"Saturación {_del(i.fluid)} a p = {_txt(i.p_Pa, _P, system)}: h_fg y ρ_v a T_sat; el "
        "líquido de la película, a la temperatura media T_f (de CoolProp)."
    )
    if r.vapor is not None:
        text += " c_pv es la del vapor entre T_sat y T_v."
    return ProcedureStep("Propiedades", text, tuple(latex))


def _h_star_step(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    i, s, liq = r.inputs, r.sat, r.liquid
    latex = [
        chain(
            r"\Delta T",
            r"T_{\mathrm{sat}} - T_s",
            sub(n(s.T_sat_K, _T, system), n(i.T_s_K, _T, system)),
            q(r.dT_K, _DT, system),
        )
    ]
    rest: tuple[tuple[str, str], ...] = (
        ("+", times("0.68", n(liq.cp, _CP, system), n(r.dT_K, _DT, system))),
    )
    formula = r"h_{fg} + 0.68\,c_{p\ell}\,\Delta T"
    if r.vapor is not None:
        rest += (("+", times(n(r.vapor.cp, _CP, system), n(r.superheat_K, _DT, system))),)
        formula += r" \\ &\quad + c_{pv}\,(T_v - T_{\mathrm{sat}})"
    latex.append(
        chain(
            r"h^*_{fg}", formula, Pack(n(s.h_fg, _HFG, system), rest), q(r.h_fg_star, _HFG, system)
        )
    )
    text = (
        "El calor latente corregido (Rohsenow, 1956) suma el subenfriamiento del condensado "
        "(0,68·c_pℓ·ΔT)"
        + (" y el sobrecalentamiento del vapor" if r.vapor is not None else "")
        + ": es lo que entrega cada kilogramo que condensa."
    )
    return ProcedureStep("Calor latente corregido", text, tuple(latex))


def _g_nu(r: CondensationResult) -> float:
    return (G / r.liquid.nu**2) ** (1.0 / 3.0)


def _g_nu_latex(r: CondensationResult, system: UnitSystem) -> str:
    return chain(
        r"\left(\frac{g}{\nu_\ell^2}\right)^{1/3}",
        rf"\left({frac(n(G, _G, system), _pw(n(r.liquid.nu, _NU, system), '2'))}\right)^{{1/3}}",
        q(_g_nu(r), _IL, system),
    )


def _x_value(r: CondensationResult) -> float:
    liq, i = r.liquid, r.inputs
    return i.L_m * liq.k * r.dT_K / (liq.mu * r.h_fg_star) * _g_nu(r)


def _x_latex(r: CondensationResult, system: UnitSystem) -> str:
    i, liq = r.inputs, r.liquid
    f = _FACTOR[system]
    den = times(*([f] if f else []), n(liq.mu, _MU, system), n(r.h_fg_star, _HFG, system))
    num = times(n(i.L_m, _L, system), n(liq.k, _K, system), n(r.dT_K, _DT, system))
    return flush(
        "X",
        r"\frac{L\,k_\ell\,\Delta T}{\mu_\ell\,h^*_{fg}}\left(\frac{g}{\nu_\ell^2}\right)^{1/3}",
        Pack(frac(num, den), ((r"\cdot", n(_g_nu(r), _IL, system)),)),
        big(_x_value(r)),
    )


def _nusselt_latex(
    r: CondensationResult,
    system: UnitSystem,
    coef: str,
    length_sym: str,
    length: float,
    h_value: float,
    latent: float,
    latent_sym: str = r"h^*_{fg}",
    lhs: str = "h",
) -> list[str]:
    """h = C·[F·g·ρ_ℓ·Δρ·h*·k³/(μ·ΔT·L)]^¼ en tres renglones: el numerador N, el denominador M
    y la raíz."""
    s, liq = r.sat, r.liquid
    num_sys = (
        float(convert_from_si(G, _G, system))
        * float(convert_from_si(liq.rho, _RHO, system))
        * float(convert_from_si(liq.rho - s.rho_v, _RHO, system))
        * float(convert_from_si(latent, _HFG, system))
        * float(convert_from_si(liq.k, _K, system)) ** 3
    )
    den_sys = (
        float(convert_from_si(liq.mu, _MU, system))
        * float(convert_from_si(r.dT_K, _DT, system))
        * float(convert_from_si(length, _L, system))
    )
    return [
        chain(
            "N",
            rf"g\,\rho_\ell\,(\rho_\ell - \rho_v)\,{latent_sym}\,k_\ell^3",
            prod(
                n(G, _G, system),
                n(liq.rho, _RHO, system),
                n(liq.rho - s.rho_v, _RHO, system),
                n(latent, _HFG, system),
                _pw(n(liq.k, _K, system), "3"),
            ),
            _d(num_sys),
        ),
        chain(
            "M",
            rf"\mu_\ell\,\Delta T\,{length_sym}",
            prod(n(liq.mu, _MU, system), n(r.dT_K, _DT, system), n(length, _L, system)),
            _d(den_sys),
        ),
        chain(
            lhs,
            rf"{coef}\,\left({_f_times(system, 'N')}/M\right)^{{1/4}}",
            rf"{coef} \cdot {_paren_pow(_f_times(system, frac(_d(num_sys), _d(den_sys))), '1/4')}",
            q(h_value, _H, system),
        ),
    ]


def _re_balance_latex(r: CondensationResult, system: UnitSystem, h: float, Re: float) -> str:
    i, liq = r.inputs, r.liquid
    f = _FACTOR[system]
    den = times(*([f] if f else []), n(liq.mu, _MU, system), n(r.h_fg_star, _HFG, system))
    return flush(
        "Re",
        r"\frac{4\,L\,h\,\Delta T}{\mu_\ell\,h^*_{fg}}",
        frac(times("4", n(i.L_m, _L, system), n(h, _H, system), n(r.dT_K, _DT, system)), den),
        _d(Re),
    )


def _vertical_step(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    i, liq = r.inputs, r.liquid
    assert r.Re is not None
    X = _x_value(r)
    Re_w = (4.81 + 3.70 * X) ** 0.820
    latex = [_g_nu_latex(r, system), _x_latex(r, system)]
    latex.append(
        chain(
            r"\mathrm{Re}_{\text{ondas}}",
            r"(4.81 + 3.70\,X)^{0.820}",
            rf"(4.81 + 3.70 \cdot {big(X)})^{{0.820}}",
            big(Re_w),
        )
    )
    factor = math.cos(math.radians(i.angle_deg)) ** 0.25 if i.geometry == "inclined_plate" else 1.0
    h_vert = r.h / factor
    Re_vert = r.Re / factor
    if r.regime == "laminar":
        latex += _nusselt_latex(r, system, "0.943", "L", i.L_m, h_vert, r.h_fg_star)
        latex.append(_re_balance_latex(r, system, h_vert, Re_vert))
        text = (
            f"Se supone la película con ondas: Re = {_comma(round(Re_w, 1))} < 30, así que es "
            "laminar y lisa: Nusselt (1916), "
            "h = 0,943·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg·k_ℓ³/(μ_ℓ·ΔT·L)]^¼, "
            "en tres renglones (N, M y la raíz). Re sale del balance, 4·L·h·ΔT/(μ_ℓ·h*_fg)."
        )
        text += _factor_text(system, root=True)
    elif r.regime == "wavy":
        den = rf"1.08 \cdot {_pw(big(Re_w), '1.22')} - 5.2"
        latex.append(
            flush(
                "h",
                frac(r"\mathrm{Re}\,k_\ell", r"1.08\,\mathrm{Re}^{1.22} - 5.2")
                + r"\left(\frac{g}{\nu_\ell^2}\right)^{1/3}",
                Pack(
                    frac(times(big(Re_w), n(liq.k, _K, system)), den),
                    ((r"\cdot", n(_g_nu(r), _IL, system)),),
                ),
                q(h_vert, _H, system),
            )
        )
        text = (
            f"Se supone la película con ondas (Kutateladze, 1963): Re = {_comma(round(Re_w))} está "
            "entre 30 y 1800, así que vale; h sale de Re."
        )
        text += _x_text(system)
    else:
        Pr = liq.Pr
        a = (0.0690 * X - 151.0) * Pr**0.5 + 253.0
        b = 8750.0 + 58.0 * Pr**-0.5 * (Re_vert**0.75 - 253.0)
        latex += [
            chain(
                "a",
                r"(0.0690\,X - 151)\,\mathrm{Pr}_\ell^{1/2} + 253",
                Pack(
                    rf"(0.0690 \cdot {big(X)} - 151)",
                    ((r"\cdot", rf"{_pw(_d(Pr), '1/2')} + 253"),),
                ),
                _d(a),
            ),
            chain(r"\mathrm{Re}", "a^{4/3}", _pw(_d(a), "4/3"), big(Re_vert)),
            chain(
                "b",
                r"8750 + 58\,\mathrm{Pr}_\ell^{-1/2}\,(\mathrm{Re}^{3/4} - 253)",
                Pack(
                    rf"8750 + 58 \cdot {_pw(_d(Pr), '-1/2')}",
                    ((r"\cdot", rf"({_pw(big(Re_vert), '3/4')} - 253)"),),
                ),
                big(b),
            ),
            chain(
                "h",
                r"\frac{\mathrm{Re}\,k_\ell}{b}\left(\frac{g}{\nu_\ell^2}\right)^{1/3}",
                Pack(
                    frac(times(big(Re_vert), n(liq.k, _K, system)), big(b)),
                    ((r"\cdot", n(_g_nu(r), _IL, system)),),
                ),
                q(h_vert, _H, system),
            ),
        ]
        text = (
            f"Se supone la película con ondas: Re = {_comma(round(Re_w))} > 1800, así que es "
            "turbulenta (Labuntsov, 1957): Re sale de su forma explícita y h, de Re."
        )
        text += _x_text(system)
    if i.geometry == "inclined_plate":
        latex.append(
            chain(
                r"h_\theta",
                r"h\,(\cos\theta)^{1/4}",
                rf"{n(h_vert, _H, system)} \cdot (\cos {_d(i.angle_deg, 3)}^\circ)^{{1/4}}",
                q(r.h, _H, system),
            )
        )
        text += (
            " Inclinada θ de la vertical, la gravedad que hace escurrir la película es g·cos θ: "
            "h baja como (cos θ)^¼."
        )
    if i.geometry == "vertical_tube":
        text += " El tubo vertical es una placa de ancho π·D (la película es mucho más fina que D)."
    title = {
        "laminar": "Película laminar (Nusselt)",
        "wavy": "Película con ondas (Kutateladze)",
        "turbulent": "Película turbulenta (Labuntsov)",
    }[r.regime]
    return ProcedureStep(title, text, tuple(latex))


def _x_text(system: UnitSystem) -> str:
    if system == "Técnico":
        return " X lleva el 1000 de h*_fg en kJ/kg."
    if system == "Inglés":
        return " X lleva el 3600 (k por hora y μ por segundo)."
    return ""


def _round_step(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    if i.geometry in ("horizontal_tube", "tube_bank"):
        h1 = r.h_single if r.h_single is not None else r.h
        latex = _nusselt_latex(
            r,
            system,
            "0.729",
            "D",
            i.D_m,
            h1,
            r.h_fg_star,
            lhs="h_1" if i.geometry == "tube_bank" else "h",
        )
        text = (
            "Por fuera de un tubo horizontal (Nusselt): h = 0,729·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg·k_ℓ³/"
            "(μ_ℓ·ΔT·D)]^¼, en tres renglones (N, M y la raíz). La película es laminar: el "
            "recorrido es medio perímetro."
        )
        if i.geometry == "tube_bank":
            latex.append(
                chain(
                    "h_N",
                    r"h_1\,N^{-1/4}",
                    rf"{n(h1, _H, system)} \cdot {i.rows}^{{-1/4}}",
                    q(r.h, _H, system),
                )
            )
            text += (
                f" En el banco, con N = {i.rows} tubos por columna, el condensado de cada tubo cae "
                "sobre el de abajo y engrosa su película: h_N = h_1·N^−¼."
            )
        title = "Tubo horizontal (Nusselt)" if i.geometry == "horizontal_tube" else "Banco de tubos"
    elif i.geometry == "sphere":
        latex = _nusselt_latex(r, system, "0.815", "D", i.D_m, r.h, r.h_fg_star)
        text = "Sobre una esfera: como el tubo horizontal con 0,815 en vez de 0,729."
        title = "Esfera"
    else:
        s, liq = r.sat, r.liquid
        superheat = r.vapor.cp * r.superheat_K if r.vapor is not None else 0.0
        h_prime = s.h_fg + superheat + 3.0 / 8.0 * liq.cp * r.dT_K
        rest: tuple[tuple[str, str], ...] = (
            ("+", times(r"\tfrac{3}{8}", n(liq.cp, _CP, system), n(r.dT_K, _DT, system))),
        )
        if r.vapor is not None:
            rest += (("+", times(n(r.vapor.cp, _CP, system), n(r.superheat_K, _DT, system))),)
        latex = [
            chain(
                "h'_{fg}",
                r"h_{fg} + \tfrac{3}{8}\,c_{p\ell}\,\Delta T",
                Pack(n(s.h_fg, _HFG, system), rest),
                q(h_prime, _HFG, system),
            )
        ]
        latex += _nusselt_latex(r, system, "0.555", "D", i.D_m, r.h, h_prime, latent_sym="h'_{fg}")
        text = (
            "Adentro de un tubo horizontal con poco vapor (Chato, 1962): "
            "h = 0,555·[g·ρ_ℓ·(ρ_ℓ − ρ_v)·"
            "k_ℓ³·h′_fg/(μ_ℓ·ΔT·D)]^¼, con su propio calor latente h′_fg = h_fg + ⅜·c_pℓ·ΔT."
        )
        if r.Re_vapor is not None:
            mu_v = s.rho_v * i.V_v_m_s * i.D_m / r.Re_vapor if i.V_v_m_s else 0.0
            latex.append(
                chain(
                    r"\mathrm{Re}_v",
                    r"\frac{\rho_v\,V_v\,D}{\mu_v}",
                    frac(
                        times(
                            n(s.rho_v, _RHO, system),
                            n(i.V_v_m_s or 0.0, _V, system),
                            n(i.D_m, _L, system),
                        ),
                        n(mu_v, _MU, system),
                    ),
                    _d(r.Re_vapor),
                )
            )
            verdict = "menor" if r.Re_vapor < RE_VAPOR_CHATO else "mayor"
            text += f" Vale con Re_v < 35 000 a la entrada: acá es {verdict}."
        title = "Adentro del tubo (Chato)"
    text += _factor_text(system, root=True)
    return ProcedureStep(title, text, tuple(latex))


def _condensation_totals(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    i = r.inputs
    D, L = n(i.D_m, _L, system), n(i.L_m, _L, system)
    if i.geometry in ("vertical_plate", "inclined_plate"):
        area = chain("A", "L\\,W", times(L, n(i.W_m, _L, system)), q(r.area_m2, _A, system))
    elif i.geometry == "sphere":
        area = chain("A", r"\pi D^2", rf"\pi \cdot {_pw(D, '2')}", q(r.area_m2, _A, system))
    elif i.geometry == "tube_bank":
        area = chain(
            "A",
            r"N_t\,\pi D L",
            times(str(r.tubes), r"\pi", D, L),
            q(r.area_m2, _A, system),
        )
    else:
        area = chain("A", r"\pi D L", times(r"\pi", D, L), q(r.area_m2, _A, system))
    f = _FACTOR[system]
    m_den = times(f, n(r.h_fg_star, _HFG, system)) if f else n(r.h_fg_star, _HFG, system)
    latex = [
        area,
        chain(
            r"\dot Q",
            r"h\,A\,\Delta T",
            prod(n(r.h, _H, system), n(r.area_m2, _A, system), n(r.dT_K, _DT, system)),
            q(r.Q_W, _W, system),
        ),
        chain(
            r"\dot m",
            r"\frac{\dot Q}{h^*_{fg}}",
            frac(n(r.Q_W, _W, system), m_den),
            q(r.m_dot, _M, system),
        ),
    ]
    text = "El calor que recibe la pared y el vapor que condensa."
    if i.geometry == "tube_bank":
        text += f" N_t = {r.tubes} tubos ({i.rows} × {i.columns})."
    if system == "Técnico":
        text += " Con Q̇ en W y h*_fg en kJ/kg, ṁ lleva el 1000."
    elif system == "Inglés":
        text += " Con Q̇ en Btu/h, ṁ sale en lb/h: dividido por 3600, en lb/s."
    return ProcedureStep("Calor y caudal condensado", text, tuple(latex))


def _thickness_step(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    i, s, liq = r.inputs, r.sat, r.liquid
    assert r.delta_m is not None
    inclined = i.geometry == "inclined_plate"
    g_sym = r"g\cos\theta" if inclined else "g"
    g_eff = G * math.cos(math.radians(i.angle_deg)) if inclined else G
    num_sys = (
        4.0
        * float(convert_from_si(liq.mu, _MU, system))
        * float(convert_from_si(liq.k, _K, system))
        * float(convert_from_si(r.dT_K, _DT, system))
        * float(convert_from_si(i.L_m, _L, system))
    )
    den_sys = (
        float(convert_from_si(g_eff, _G, system))
        * float(convert_from_si(liq.rho, _RHO, system))
        * float(convert_from_si(liq.rho - s.rho_v, _RHO, system))
        * float(convert_from_si(r.h_fg_star, _HFG, system))
    )
    f = _FACTOR[system]
    latex = [
        chain(
            "P",
            r"4\,\mu_\ell\,k_\ell\,\Delta T\,L",
            prod(
                "4",
                n(liq.mu, _MU, system),
                n(liq.k, _K, system),
                n(r.dT_K, _DT, system),
                n(i.L_m, _L, system),
            ),
            _d(num_sys),
        ),
        chain(
            "S",
            rf"{g_sym}\,\rho_\ell\,(\rho_\ell - \rho_v)\,h^*_{{fg}}",
            prod(
                n(g_eff, _G, system),
                n(liq.rho, _RHO, system),
                n(liq.rho - s.rho_v, _RHO, system),
                n(r.h_fg_star, _HFG, system),
            ),
            _d(den_sys),
        ),
        chain(
            r"\delta_L",
            rf"\left(\frac{{P}}{{{_f_times(system, 'S')}}}\right)^{{1/4}}",
            _paren_pow(frac(_d(num_sys), times(f, _d(den_sys)) if f else _d(den_sys)), "1/4"),
            q(r.delta_m, "small_length", system),
        ),
    ]
    text = (
        "El espesor de la película al pie, con el perfil laminar de Nusselt: "
        "δ = [4·μ_ℓ·k_ℓ·ΔT·L/(g·ρ_ℓ·(ρ_ℓ − ρ_v)·h*_fg)]^¼."
    )
    if r.regime != "laminar":
        text += " Con ondas o turbulencia es una estimación."
    if f:
        text += f" El {f} va abajo, adentro de la raíz."
    return ProcedureStep("Espesor de la película", text, tuple(latex))


def _dropwise_step(r: CondensationResult, system: UnitSystem) -> ProcedureStep:
    assert r.h_dropwise is not None
    t = r.sat.T_sat_K - 273.15
    if t <= 100.0:
        formula = r"51\,104 + 2044\,T_{\mathrm{sat}}"
        subst = rf"51\,104 + 2044 \cdot {_d(t, 5)}"
    else:
        formula = r"255\,510"
        subst = r"255\,510"
    value = rf"{dimensionless(r.h_dropwise, 5)}\ \mathrm{{W/(m^{{2}}\cdot K)}}"
    latex = [chain(r"h_{\text{gotas}}", formula, subst, value)]
    if system == "Inglés":
        latex.append(rf"h_{{\text{{gotas}}}} = {q(r.h_dropwise, _H, system)}")
    text = (
        "Si el agua no moja la pared (un promotor), condensa en gotas: Griffith (1983) da h en "
        "W/(m²·K) con T_sat en °C (22 a 100 °C; arriba, constante). Es "
        f"{_comma(round(r.h_dropwise / r.h, 1))} veces la película, pero los promotores se gastan "
        "y el diseño supone película (Çengel y Ghajar §10-7)."
    )
    return ProcedureStep("Condensación en gotas (Griffith)", text, tuple(latex))


def condensation_steps(r: CondensationResult, system: UnitSystem) -> list[ProcedureStep]:
    """El procedimiento de la condensación en película, paso a paso (Çengel y Ghajar §10-5)."""
    i = r.inputs
    steps = [_condensation_properties(r, system), _h_star_step(r, system)]
    if i.geometry in ("vertical_plate", "vertical_tube", "inclined_plate"):
        steps.append(_vertical_step(r, system))
    else:
        steps.append(_round_step(r, system))
    steps.append(_condensation_totals(r, system))
    if r.delta_m is not None:
        steps.append(_thickness_step(r, system))
    if r.h_dropwise is not None:
        steps.append(_dropwise_step(r, system))
    return numbered(steps)
