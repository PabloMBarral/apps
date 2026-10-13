"""El lado del casco con el método de Bell–Delaware (Fase 12.2).

Bell (1963) estudió en la Universidad de Delaware cómo se reparte el flujo del lado del
casco; Taborek (1983) llevó el método a ecuaciones en el *Heat Exchanger Design Handbook*
(§3.3), y así lo presentan Thome (*Wolverine Engineering Data Book III*, 2004, cap. 3) y
Serth (2014, cap. 6). El coeficiente de un banco de tubos ideal (todo el fluido cruzando el
haz) se corrige con cinco factores, uno por cada corriente que no cruza o cruza mal:

- J_c, el corte de los bafles: en las ventanas el fluido corre paralelo a los tubos;
- J_l, las fugas entre los tubos y los agujeros del bafle (S_tb) y entre el bafle y el
  casco (S_sb) — las corrientes A y E de Tinker;
- J_b, el bypass por el espacio entre el haz y el casco (S_b, la corriente C), que las
  tiras de sellado cortan;
- J_s, las separaciones distintas en la entrada y en la salida;
- J_r, el gradiente adverso de temperatura en flujo laminar (Re < 100).

h_o = h_ideal·J_c·J_l·J_b·J_s·J_r. La caída de presión va por zonas: el flujo cruzado de
los N_b − 1 compartimientos centrales (con R_l y R_b), las N_b ventanas (con R_l) y los dos
extremos (con R_b y R_s).

Todo en SI. Las holguras diametrales por defecto son las de TEMA (2007, 9.ª ed.): RCB-4.3 la
del casco con el bafle y RCB-4.2 la de los agujeros de los tubos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from core.heat_transfer.design_common import StreamProps

__all__ = [
    "BAFFLE_CUT_RANGE",
    "BUNDLE_CLEARANCE_HINTS",
    "IDEAL_BANK",
    "LAYOUT_ANGLE",
    "BDAreas",
    "BDFactors",
    "BellDelawareGeometry",
    "BellDelawareSide",
    "IdealBank",
    "IdealBankCoefficients",
    "bd_areas",
    "bd_factors",
    "ideal_bank",
    "solve_bell_delaware",
    "tema_shell_baffle_clearance",
    "tema_tube_hole_clearance",
]

Angle = Literal[30, 45, 90]

#: El ángulo del arreglo de cada disposición de los tubos (Taborek: 30°, 45° y 90°).
LAYOUT_ANGLE: dict[str, Angle] = {"triangular": 30, "rotated_square": 45, "square": 90}

#: Los cortes de bafle de las correlaciones (Taborek, 1983): del 15 al 45 % de D_s.
BAFFLE_CUT_RANGE = (0.15, 0.45)

#: El orden de magnitud de la holgura diametral entre el haz y el casco, L_bb, según el
#: cabezal (en m, mínimo y máximo): es un dato del equipo, esto es solo una guía.
BUNDLE_CLEARANCE_HINTS: dict[str, tuple[float, float]] = {
    "Placas fijas o tubos en U": (0.010, 0.015),
    "Cabezal flotante con anillo partido": (0.030, 0.050),
    "Cabezal flotante extraíble": (0.080, 0.100),
}

_IN = 0.0254


# ---------------------------------------------------------------------
# El banco ideal (Taborek)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IdealBankCoefficients:
    """Los coeficientes de Taborek de un arreglo: a₃, a₄, b₃ y b₄, y por tramo de Re
    (de mayor a menor) (Re_mín, a₁, a₂, b₁, b₂)."""

    a3: float
    a4: float
    b3: float
    b4: float
    rows: tuple[tuple[float, float, float, float, float], ...]


#: j_i y f_i del banco ideal (Taborek, 1983, HEDH §3.3; Thome, Data Book III, cap. 3; Serth,
#: 2014, cap. 6). La tabla original salta ~4 % en j a 45° con Re = 10³ y ~5 % a 90° con
#: Re = 10⁴; en los demás cambios de tramo es continua dentro del 1 %.
IDEAL_BANK: dict[int, IdealBankCoefficients] = {
    30: IdealBankCoefficients(
        1.450,
        0.519,
        7.00,
        0.500,
        (
            (1.0e4, 0.321, -0.388, 0.372, -0.123),
            (1.0e3, 0.321, -0.388, 0.486, -0.152),
            (1.0e2, 0.593, -0.477, 4.570, -0.476),
            (1.0e1, 1.360, -0.657, 45.100, -0.973),
            (0.0, 1.400, -0.667, 48.000, -1.000),
        ),
    ),
    45: IdealBankCoefficients(
        1.930,
        0.500,
        6.59,
        0.520,
        (
            (1.0e4, 0.370, -0.396, 0.303, -0.126),
            (1.0e3, 0.370, -0.396, 0.333, -0.136),
            (1.0e2, 0.730, -0.500, 3.500, -0.476),
            (1.0e1, 1.498, -0.656, 26.200, -0.913),
            (0.0, 1.550, -0.667, 32.000, -1.000),
        ),
    ),
    90: IdealBankCoefficients(
        1.187,
        0.370,
        6.30,
        0.378,
        (
            (1.0e4, 0.370, -0.395, 0.391, -0.148),
            (1.0e3, 0.107, -0.266, 0.0815, 0.022),
            (1.0e2, 0.408, -0.460, 6.090, -0.602),
            (1.0e1, 0.900, -0.631, 32.100, -0.963),
            (0.0, 0.970, -0.667, 35.000, -1.000),
        ),
    ),
}

#: El rango de Re de la tabla de Taborek.
RE_IDEAL_BANK = (1.0, 1.0e5)


@dataclass(frozen=True)
class IdealBank:
    """j_i y f_i del banco ideal, con los coeficientes que se usaron."""

    Re: float
    angle: Angle
    pitch_ratio: float
    a1: float
    a2: float
    a3: float
    a4: float
    b1: float
    b2: float
    b3: float
    b4: float
    a: float
    b: float
    j: float
    f: float


def ideal_bank(Re: float, angle: Angle, pitch_ratio: float) -> IdealBank:
    """El factor de Colburn j_i y el de fricción f_i de un banco de tubos ideal (Taborek,
    1983): j_i = a₁·(1,33/(P_T/d_o))^a·Re^a₂ con a = a₃/(1 + 0,14·Re^a₄), y f_i igual con
    los b. Re = d_o·ṁ/(μ·S_m)."""
    if angle not in IDEAL_BANK:
        raise ValueError(f"Ángulo de arreglo desconocido: {angle} (30°, 45° o 90°).")
    c = IDEAL_BANK[angle]
    row = next(r for r in c.rows if Re >= r[0])
    _, a1, a2, b1, b2 = row
    a = c.a3 / (1.0 + 0.14 * Re**c.a4)
    b = c.b3 / (1.0 + 0.14 * Re**c.b4)
    j = a1 * (1.33 / pitch_ratio) ** a * Re**a2
    f = b1 * (1.33 / pitch_ratio) ** b * Re**b2
    return IdealBank(Re, angle, pitch_ratio, a1, a2, c.a3, c.a4, b1, b2, c.b3, c.b4, a, b, j, f)


# ---------------------------------------------------------------------
# Holguras de TEMA
# ---------------------------------------------------------------------

#: TEMA (2007), RCB-4.3: el diámetro interior del casco menos el de los bafles, según el
#: tamaño nominal del casco (6–17", 18–39", 40–54", 55–69", 70–84" y 85–100").
_TEMA_SHELL_BAFFLE: tuple[tuple[float, float], ...] = (
    (18 * _IN, 0.125 * _IN),
    (40 * _IN, 0.1875 * _IN),
    (55 * _IN, 0.25 * _IN),
    (70 * _IN, 0.3125 * _IN),
    (85 * _IN, 0.375 * _IN),
)


def tema_shell_baffle_clearance(D_s_m: float) -> float:
    """L_sb, la holgura diametral entre el casco y los bafles (TEMA, 2007, RCB-4.3): de ⅛"
    en los cascos de menos de 18" a 7/16" en los de más de 85"."""
    for D_max, L_sb in _TEMA_SHELL_BAFFLE:
        if D_s_m < D_max:
            return L_sb
    return 0.4375 * _IN


def tema_tube_hole_clearance(d_o_m: float, unsupported_m: float) -> float:
    """L_tb, la holgura diametral entre los tubos y los agujeros de los bafles (TEMA, 2007,
    RCB-4.2): 1/32" (0,79 mm), o 1/64" (0,40 mm) si el tubo es de 1¼" o menos y el tramo sin
    apoyo pasa de 36" (914 mm)."""
    if d_o_m <= 1.25 * _IN and unsupported_m > 36 * _IN:
        return _IN / 64.0
    return _IN / 32.0


# ---------------------------------------------------------------------
# La geometría
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BellDelawareGeometry:
    """Lo que Bell–Delaware pide además de lo de Kern.

    - ``L_bb_m``: la holgura diametral entre el haz y el casco, D_s − D_otl (un dato del
      equipo; ver :data:`BUNDLE_CLEARANCE_HINTS`).
    - ``baffle_cut``: el corte de los bafles, como fracción de D_s.
    - ``L_sb_m`` y ``L_tb_m``: las holguras diametrales casco–bafle y tubo–bafle; ``None``
      toma las de TEMA.
    - ``sealing_strips``: los pares de tiras de sellado.
    - ``L_pl_m``: el ancho del carril de las divisiones de los pasos paralelo al flujo
      (0 si no hay o si va perpendicular).
    - ``B_in_m`` y ``B_out_m``: las separaciones de los bafles de la entrada y de la salida;
      ``None`` las iguala a la central.
    """

    L_bb_m: float
    baffle_cut: float = 0.25
    L_sb_m: float | None = None
    L_tb_m: float | None = None
    sealing_strips: int = 0
    L_pl_m: float = 0.0
    B_in_m: float | None = None
    B_out_m: float | None = None


@dataclass(frozen=True)
class BDAreas:
    """La geometría que usa el método (Taborek, 1983; Thome, Data Book III, §3.3)."""

    angle: Angle
    D_otl_m: float
    D_ctl_m: float
    L_pp_m: float
    L_tp_eff_m: float
    theta_ds: float
    theta_ctl: float
    F_w: float
    F_c: float
    S_m_m2: float
    S_wg_m2: float
    S_wt_m2: float
    S_w_m2: float
    D_w_m: float
    N_tcc: float
    N_tcw: float
    L_sb_m: float
    L_tb_m: float
    S_sb_m2: float
    S_tb_m2: float
    S_b_m2: float
    F_sbp: float
    r_ss: float
    r_s: float
    r_lm: float
    B_in_m: float
    B_out_m: float


def bd_areas(
    D_s_m: float,
    d_o_m: float,
    P_T_m: float,
    angle: Angle,
    B_m: float,
    N_t: int,
    g: BellDelawareGeometry,
) -> BDAreas:
    """Las áreas y las hileras de Bell–Delaware (Taborek, 1983; Thome, Data Book III, §3.3):

    - D_otl = D_s − L_bb, D_ctl = D_otl − d_o;
    - el paso paralelo al flujo L_pp (0,866·P_T a 30°, 0,707·P_T a 45°, P_T a 90°) y el
      efectivo normal L_tp,eff (P_T a 30° y 90°, 0,707·P_T a 45°);
    - S_m = B·[L_bb + (D_ctl/L_tp,eff)·(P_T − d_o)], el área de flujo cruzado;
    - θ_ds = 2·acos(1 − 2·B_c) y θ_ctl = 2·acos[D_s·(1 − 2·B_c)/D_ctl];
      F_w = (θ_ctl − sen θ_ctl)/(2π), F_c = 1 − 2·F_w;
    - S_w = S_wg − S_wt, con S_wg = (π/4)·D_s²·(θ_ds − sen θ_ds)/(2π) y
      S_wt = N_t·F_w·π·d_o²/4; D_w = 4·S_w/(π·d_o·N_t·F_w + θ_ds·D_s);
    - N_tcc = (D_s/L_pp)·(1 − 2·B_c) y N_tcw = (0,8/L_pp)·[D_s·B_c − (D_s − D_ctl)/2];
    - las fugas S_sb = D_s·L_sb·(2π − θ_ds)/4 y S_tb = (π/4)·[(d_o + L_tb)² − d_o²]·N_t·(1 − F_w);
    - el bypass S_b = B·(L_bb + L_pl), F_sbp = S_b/S_m y r_ss = N_ss/N_tcc;
    - r_s = S_sb/(S_sb + S_tb) y r_lm = (S_sb + S_tb)/S_m.
    """
    if angle not in IDEAL_BANK:
        raise ValueError(f"Ángulo de arreglo desconocido: {angle} (30°, 45° o 90°).")
    B_c = g.baffle_cut
    if not 0.0 < B_c < 0.5:
        raise ValueError(
            f"El corte de los bafles ({_pct(B_c)}) tiene que estar entre 0 y 50 % de D_s: "
            "con la mitad o más, los bafles no se superponen."
        )
    if g.L_bb_m < 0.0:
        raise ValueError("La holgura entre el haz y el casco, L_bb, no puede ser negativa.")
    D_otl = D_s_m - g.L_bb_m
    D_ctl = D_otl - d_o_m
    if D_ctl <= 0.0:
        raise ValueError(
            f"Con L_bb = {_mm(g.L_bb_m)} el haz de tubos no entra en un casco de {_mm(D_s_m)}: "
            "achicá la holgura entre el haz y el casco."
        )
    L_sb = g.L_sb_m if g.L_sb_m is not None else tema_shell_baffle_clearance(D_s_m)
    L_tb = g.L_tb_m if g.L_tb_m is not None else tema_tube_hole_clearance(d_o_m, 2.0 * B_m)
    if L_sb < 0.0 or L_tb < 0.0:
        raise ValueError("Las holguras casco–bafle y tubo–bafle no pueden ser negativas.")
    if g.sealing_strips < 0:
        raise ValueError("Los pares de tiras de sellado no pueden ser negativos.")
    if g.L_pl_m < 0.0:
        raise ValueError("El ancho del carril de los pasos no puede ser negativo.")
    B_in = g.B_in_m if g.B_in_m is not None else B_m
    B_out = g.B_out_m if g.B_out_m is not None else B_m
    if not (B_in > 0.0 and B_out > 0.0):
        raise ValueError("Las separaciones de entrada y de salida tienen que ser positivas.")
    L_pp = {30: 0.866, 45: 0.707, 90: 1.0}[angle] * P_T_m
    L_tp_eff = {30: 1.0, 45: 0.707, 90: 1.0}[angle] * P_T_m
    S_m = B_m * (g.L_bb_m + D_ctl / L_tp_eff * (P_T_m - d_o_m))
    theta_ds = 2.0 * math.acos(1.0 - 2.0 * B_c)
    cut_line = D_s_m * (1.0 - 2.0 * B_c) / D_ctl
    theta_ctl = 2.0 * math.acos(cut_line) if cut_line < 1.0 else 0.0
    F_w = (theta_ctl - math.sin(theta_ctl)) / (2.0 * math.pi)
    F_c = 1.0 - 2.0 * F_w
    S_wg = math.pi / 4.0 * D_s_m**2 * (theta_ds - math.sin(theta_ds)) / (2.0 * math.pi)
    S_wt = N_t * F_w * math.pi * d_o_m**2 / 4.0
    S_w = S_wg - S_wt
    D_w = 4.0 * S_w / (math.pi * d_o_m * N_t * F_w + theta_ds * D_s_m)
    N_tcc = D_s_m / L_pp * (1.0 - 2.0 * B_c)
    N_tcw = max(0.0, 0.8 / L_pp * (D_s_m * B_c - (D_s_m - D_ctl) / 2.0))
    S_sb = D_s_m * L_sb * (2.0 * math.pi - theta_ds) / 4.0
    S_tb = math.pi / 4.0 * ((d_o_m + L_tb) ** 2 - d_o_m**2) * N_t * (1.0 - F_w)
    S_b = B_m * (g.L_bb_m + g.L_pl_m)
    leak = S_sb + S_tb
    return BDAreas(
        angle=angle,
        D_otl_m=D_otl,
        D_ctl_m=D_ctl,
        L_pp_m=L_pp,
        L_tp_eff_m=L_tp_eff,
        theta_ds=theta_ds,
        theta_ctl=theta_ctl,
        F_w=F_w,
        F_c=F_c,
        S_m_m2=S_m,
        S_wg_m2=S_wg,
        S_wt_m2=S_wt,
        S_w_m2=S_w,
        D_w_m=D_w,
        N_tcc=N_tcc,
        N_tcw=N_tcw,
        L_sb_m=L_sb,
        L_tb_m=L_tb,
        S_sb_m2=S_sb,
        S_tb_m2=S_tb,
        S_b_m2=S_b,
        F_sbp=S_b / S_m,
        r_ss=g.sealing_strips / N_tcc,
        r_s=S_sb / leak if leak > 0.0 else 0.0,
        r_lm=leak / S_m,
        B_in_m=B_in,
        B_out_m=B_out,
    )


# ---------------------------------------------------------------------
# Los factores
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BDFactors:
    """Los factores de corrección del coeficiente (J) y de la caída de presión (R)."""

    laminar: bool
    J_c: float
    J_l: float
    J_b: float
    J_s: float
    J_r: float
    J_r_star: float
    N_c: float
    C_bh: float
    n: float
    R_l: float
    p: float
    R_b: float
    C_bp: float
    R_s_in: float
    R_s_out: float
    n_prime: float

    @property
    def J_total(self) -> float:
        return self.J_c * self.J_l * self.J_b * self.J_s * self.J_r


def bd_factors(a: BDAreas, Re: float, B_m: float, baffles: int) -> BDFactors:
    """Los factores de Bell–Delaware con las formas de Taborek (1983, HEDH; Serth, 2014):

    - J_c = 0,55 + 0,72·F_c;
    - J_l = 0,44·(1 − r_s) + [1 − 0,44·(1 − r_s)]·exp(−2,2·r_lm);
    - J_b = exp[−C_bh·F_sbp·(1 − (2·r_ss)^⅓)] (1 si r_ss ≥ ½), C_bh = 1,25 (1,35 con Re < 100);
    - J_s = [(N_b − 1) + L_ent*^(1−n) + L_sal*^(1−n)]/[(N_b − 1) + L_ent* + L_sal*], con
      L* = B_ext/B y n = 0,6 (⅓ con Re < 100);
    - J_r = 1 con Re ≥ 100; (10/N_c)^0,18 con Re ≤ 20 (N_c = (N_tcc + N_tcw)·(N_b + 1), las
      hileras que cruza todo el fluido), interpolado en el medio y nunca menos de 0,4;
    - R_l = exp[−1,33·(1 + r_s)·r_lm^p], p = −0,15·(1 + r_s) + 0,8;
    - R_b = exp[−C_bp·F_sbp·(1 − (2·r_ss)^⅓)] (1 si r_ss ≥ ½), C_bp = 3,7 (4,5 con Re < 100);
    - R_s de cada extremo = (B/B_ext)^(2−n′), n′ = 0,2 (1 con Re < 100).
    """
    laminar = Re < 100.0
    J_c = 0.55 + 0.72 * a.F_c
    J_l = 0.44 * (1.0 - a.r_s) + (1.0 - 0.44 * (1.0 - a.r_s)) * math.exp(-2.2 * a.r_lm)
    strips = 1.0 - (2.0 * a.r_ss) ** (1.0 / 3.0) if a.r_ss < 0.5 else 0.0
    C_bh = 1.35 if laminar else 1.25
    J_b = math.exp(-C_bh * a.F_sbp * strips)
    n = 1.0 / 3.0 if laminar else 0.6
    L_in, L_out = a.B_in_m / B_m, a.B_out_m / B_m
    J_s = ((baffles - 1) + L_in ** (1.0 - n) + L_out ** (1.0 - n)) / ((baffles - 1) + L_in + L_out)
    N_c = (a.N_tcc + a.N_tcw) * (baffles + 1)
    J_r_star = (10.0 / N_c) ** 0.18
    if Re >= 100.0:
        J_r = 1.0
    elif Re <= 20.0:
        J_r = max(J_r_star, 0.4)
    else:
        J_r = max(J_r_star + (20.0 - Re) / 80.0 * (J_r_star - 1.0), 0.4)
    p = -0.15 * (1.0 + a.r_s) + 0.8
    R_l = math.exp(-1.33 * (1.0 + a.r_s) * a.r_lm**p)
    C_bp = 4.5 if laminar else 3.7
    R_b = math.exp(-C_bp * a.F_sbp * strips)
    n_prime = 1.0 if laminar else 0.2
    R_s_in = (B_m / a.B_in_m) ** (2.0 - n_prime)
    R_s_out = (B_m / a.B_out_m) ** (2.0 - n_prime)
    return BDFactors(
        laminar=laminar,
        J_c=J_c,
        J_l=J_l,
        J_b=J_b,
        J_s=J_s,
        J_r=J_r,
        J_r_star=J_r_star,
        N_c=N_c,
        C_bh=C_bh,
        n=n,
        R_l=R_l,
        p=p,
        R_b=R_b,
        C_bp=C_bp,
        R_s_in=R_s_in,
        R_s_out=R_s_out,
        n_prime=n_prime,
    )


# ---------------------------------------------------------------------
# El lado del casco
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BellDelawareSide:
    """El lado del casco con Bell–Delaware."""

    geometry: BellDelawareGeometry
    areas: BDAreas
    B_m: float
    baffles: int
    G_s: float
    Re: float
    Pr: float
    phi: float
    ideal: IdealBank
    h_ideal_W_per_m2K: float
    factors: BDFactors
    h_W_per_m2K: float
    dp_bi_Pa: float
    dp_wi_Pa: float
    dp_cross_Pa: float
    dp_window_Pa: float
    dp_ends_Pa: float
    dp_Pa: float
    warnings: tuple[str, ...]

    @property
    def dp_cross_ideal_Pa(self) -> float:
        """El flujo cruzado sin fugas ni bypass (R_l = R_b = 1)."""
        return (self.baffles - 1) * self.dp_bi_Pa

    @property
    def dp_window_ideal_Pa(self) -> float:
        """Las ventanas sin fugas (R_l = 1)."""
        return self.baffles * self.dp_wi_Pa

    @property
    def dp_ends_ideal_Pa(self) -> float:
        """Los extremos sin bypass (R_b = 1; R_s queda: es la separación)."""
        a, fc = self.areas, self.factors
        return self.dp_bi_Pa * (1.0 + a.N_tcw / a.N_tcc) * (fc.R_s_in + fc.R_s_out)


def solve_bell_delaware(
    props: StreamProps,
    m_dot_kg_s: float,
    D_s_m: float,
    d_o_m: float,
    P_T_m: float,
    angle: Angle,
    B_m: float,
    baffles: int,
    N_t: int,
    geometry: BellDelawareGeometry,
    mu_wall_Pa_s: float | None,
) -> BellDelawareSide:
    """h_o y Δp del lado del casco con Bell–Delaware (Taborek, 1983; Thome, Data Book III,
    §3.3; Serth, 2014):

    - Re = d_o·ṁ/(μ·S_m); h_ideal = j_i·c_p·(ṁ/S_m)·Pr^(−⅔)·φ, φ = (μ/μ_p)^0,14;
    - h_o = h_ideal·J_c·J_l·J_b·J_s·J_r;
    - Δp_bi = 2·f_i·N_tcc·(ṁ/S_m)²/(ρ·φ), un compartimiento ideal;
    - Δp_wi = (2 + 0,6·N_tcw)·ṁ²/(2·ρ·S_m·S_w), una ventana (turbulento); con Re < 100,
      26·μ·ṁ/(ρ·√(S_m·S_w))·[N_tcw/(P_T − d_o) + B/D_w²] + ṁ²/(ρ·S_m·S_w);
    - Δp = (N_b − 1)·Δp_bi·R_b·R_l + N_b·Δp_wi·R_l
      + Δp_bi·(1 + N_tcw/N_tcc)·R_b·(R_s,ent + R_s,sal).
    """
    if baffles < 1:
        raise ValueError("Hace falta al menos un bafle (Bell–Delaware es con bafles).")
    a = bd_areas(D_s_m, d_o_m, P_T_m, angle, B_m, N_t, geometry)
    rho, mu = props.rho_kg_per_m3, props.mu_Pa_s
    G = m_dot_kg_s / a.S_m_m2
    Re = d_o_m * G / mu
    Pr = props.Pr
    phi = (mu / mu_wall_Pa_s) ** 0.14 if mu_wall_Pa_s else 1.0
    ib = ideal_bank(Re, angle, P_T_m / d_o_m)
    h_ideal = ib.j * props.cp_J_per_kgK * G * Pr ** (-2.0 / 3.0) * phi
    fc = bd_factors(a, Re, B_m, baffles)
    h = h_ideal * fc.J_total
    dp_bi = 2.0 * ib.f * a.N_tcc * G**2 / (rho * phi)
    if fc.laminar:
        dp_wi = 26.0 * mu * m_dot_kg_s / (rho * math.sqrt(a.S_m_m2 * a.S_w_m2)) * (
            a.N_tcw / (P_T_m - d_o_m) + B_m / a.D_w_m**2
        ) + m_dot_kg_s**2 / (rho * a.S_m_m2 * a.S_w_m2)
    else:
        dp_wi = (2.0 + 0.6 * a.N_tcw) * m_dot_kg_s**2 / (2.0 * rho * a.S_m_m2 * a.S_w_m2)
    dp_cross = (baffles - 1) * dp_bi * fc.R_b * fc.R_l
    dp_window = baffles * dp_wi * fc.R_l
    dp_ends = dp_bi * (1.0 + a.N_tcw / a.N_tcc) * fc.R_b * (fc.R_s_in + fc.R_s_out)
    warnings: list[str] = []
    lo, hi = BAFFLE_CUT_RANGE
    if not lo <= geometry.baffle_cut <= hi:
        warnings.append(
            f"El corte de los bafles ({_pct(geometry.baffle_cut)}) está fuera de las "
            "correlaciones de Bell–Delaware (15 a 45 % de D_s)."
        )
    if not RE_IDEAL_BANK[0] <= Re <= RE_IDEAL_BANK[1]:
        warnings.append(
            f"Re_s = {_fmt(Re)} está fuera de la tabla de Taborek (1 a 10⁵): se extrapola el "
            "último tramo."
        )
    if a.r_lm > 0.8:
        warnings.append(
            f"Las fugas son muy grandes (r_lm = {_num(a.r_lm)}): las formas de J_l y R_l se "
            "ajustaron hasta r_lm ≈ 0,8."
        )
    if a.N_tcw == 0.0:
        warnings.append(
            "El corte de los bafles no llega a los tubos (la cuerda queda en el espacio entre el "
            "haz y el casco): no hay tubos en las ventanas."
        )
    return BellDelawareSide(
        geometry=geometry,
        areas=a,
        B_m=B_m,
        baffles=baffles,
        G_s=G,
        Re=Re,
        Pr=Pr,
        phi=phi,
        ideal=ib,
        h_ideal_W_per_m2K=h_ideal,
        factors=fc,
        h_W_per_m2K=h,
        dp_bi_Pa=dp_bi,
        dp_wi_Pa=dp_wi,
        dp_cross_Pa=dp_cross,
        dp_window_Pa=dp_window,
        dp_ends_Pa=dp_ends,
        dp_Pa=dp_cross + dp_window + dp_ends,
        warnings=tuple(warnings),
    )


def _num(x: float, sig: int = 3) -> str:
    return f"{x:.{sig}g}".replace(".", ",")


def _fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x) >= 100.0 else _num(x)


def _pct(x: float) -> str:
    return f"{_num(100.0 * x)} %"


def _mm(x: float) -> str:
    return f"{_num(1000.0 * x)} mm"
