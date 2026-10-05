"""Reporte didáctico de un estado termodinámico — Fase 1.6.

A partir de un :class:`core.fluids.FluidState` arma todo lo que la página
de Propiedades le muestra al alumno, ya convertido al sistema de unidades
activo:

- :func:`property_rows` — tabla con todas las propiedades del estado.
- :func:`saturation_rows` — fila de la tabla de saturación (a la p o a
  la T del estado), como en Cengel A-4/A-5.
- :func:`region_summary` — una oración con la región y su "distancia" a
  la saturación (sobrecalentamiento, subenfriamiento o título).
- :func:`state_notes` — advertencias y observaciones didácticas.
- :func:`build_procedure` — cómo se resuelve el estado "a mano" con las
  tablas: saturación a la variable ancla, comparación con f y g, regla
  de la palanca, aproximación de líquido incompresible, comparación con
  gas ideal y verificación h = u + p·v; en LaTeX con los valores
  reemplazados.
- :func:`state_to_dict` / :func:`state_to_csv` — exportación.

No importa Streamlit (regla de arquitectura del proyecto).

Referencias
-----------
Cengel, Y. A., & Boles, M. A. (2015). *Thermodynamics: An Engineering
Approach* (8th ed.), §3-5 "Property tables" (uso de las tablas de
saturación, vapor sobrecalentado y líquido comprimido; aproximación
h ≈ h_f@T + v_f@T·(P − P_sat@T)) y §3-7/3-8 (factor de compresibilidad).

Barral, P. M. *Vademecum de Termodinámica* (vademecum-termo): §3.2
(entalpía), §4.2 (gas ideal), §7.3 (factor de compresibilidad), §12
(vapor húmedo) y §13 (líquidos incompresibles).
DOI: 10.5281/zenodo.20092635
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Any, Literal

from core.fluids import (
    FLUID_NAMES_ES,
    PAIR_KWARGS,
    REGION_LABELS_ES,
    FluidState,
    PairCode,
    SaturatedPhase,
    SaturationProperties,
)
from core.latex import latex_chain, latex_number, latex_paren, latex_unit
from core.units_system import QuantityKind, UnitSystem, convert_from_si, convert_to_si, unit_label

# ---------------------------------------------------------------------
# Metadatos de los pares y de las variables de entrada
# ---------------------------------------------------------------------

#: Orden didáctico de los pares en el selector de la página.
PAIR_ORDER: tuple[PairCode, ...] = ("TP", "PX", "TX", "PH", "PS", "TS", "HS", "TV", "PV", "PU")

PAIR_LABELS_ES: dict[str, str] = {
    "TP": "Temperatura y presión (T, p)",
    "PX": "Presión y título (p, x)",
    "TX": "Temperatura y título (T, x)",
    "PH": "Presión y entalpía (p, h)",
    "PS": "Presión y entropía (p, s)",
    "TS": "Temperatura y entropía (T, s)",
    "HS": "Entalpía y entropía (h, s) — Mollier",
    "TV": "Temperatura y volumen específico (T, v)",
    "PV": "Presión y volumen específico (p, v)",
    "PU": "Presión y energía interna (p, u)",
}


@dataclass(frozen=True)
class InputSpec:
    """Cómo se presenta una variable de entrada (``kind=None``: adimensional)."""

    name: str
    symbol: str
    kind: QuantityKind | None


INPUT_SPECS: dict[str, InputSpec] = {
    "t": InputSpec("Temperatura", "T", "temperature"),
    "p": InputSpec("Presión", "p", "pressure"),
    "h": InputSpec("Entalpía específica", "h", "specific_enthalpy"),
    "s": InputSpec("Entropía específica", "s", "specific_entropy"),
    "x": InputSpec("Título", "x", None),
    "v": InputSpec("Volumen específico", "v", "specific_volume"),
    "u": InputSpec("Energía interna específica", "u", "specific_enthalpy"),
}

# Propiedad "y" que acompaña a la variable ancla, y su atributo en SaturatedPhase.
_Y_KIND: dict[str, QuantityKind] = {
    "h": "specific_enthalpy",
    "s": "specific_entropy",
    "v": "specific_volume",
    "u": "specific_enthalpy",
}
_Y_ATTR: dict[str, str] = {
    "h": "h_J_per_kg",
    "s": "s_J_per_kg_K",
    "v": "v_m3_per_kg",
    "u": "u_J_per_kg",
}


def input_value_si(state: FluidState, kw: str) -> float:
    """Valor en SI de la variable ``kw`` (``'t','p','h','s','x','v','u'``) del estado."""
    values = {
        "t": state.T_K,
        "p": state.P_Pa,
        "h": state.h_J_per_kg,
        "s": state.s_J_per_kg_K,
        "v": state.v_m3_per_kg,
        "u": state.u_J_per_kg,
    }
    if kw == "x":
        if state.x is None:
            raise ValueError("El estado está fuera de la campana: el título no está definido.")
        return state.x
    if kw not in values:
        raise ValueError(f"Variable '{kw}' desconocida. Usá una de: {sorted(INPUT_SPECS)}.")
    return values[kw]


# ---------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyRow:
    """Una fila de tabla ya convertida al sistema de unidades activo.

    ``value`` es ``None`` cuando la propiedad no está definida en el
    estado; ``note`` explica por qué.
    """

    group: str
    name: str
    symbol: str
    value: float | None
    unit: str
    note: str = ""


_DIMENSIONLESS = "-"
_MIXTURE_NOTE = "no definida dentro de la campana (mezcla bifásica)"
_UNAVAILABLE_NOTE = "CoolProp no la tiene para este estado"


def _row(
    group: str,
    name: str,
    symbol: str,
    value_si: float | None,
    kind: QuantityKind | None,
    system: UnitSystem,
    note: str = "",
) -> PropertyRow:
    if kind is None:
        return PropertyRow(group, name, symbol, value_si, _DIMENSIONLESS, note)
    value = None if value_si is None else convert_from_si(value_si, kind, system)
    return PropertyRow(group, name, symbol, value, unit_label(kind, system), note)


def property_rows(state: FluidState, system: UnitSystem) -> list[PropertyRow]:
    """Todas las propiedades del estado, agrupadas, en el sistema ``system``."""
    in_mixture = state.region == "saturated_mixture"
    g_state = "Estado"
    g_sat = "Respecto de la saturación"
    g_cal = "Calores específicos"
    g_tr = "Transporte (para convección)"

    # (grupo, nombre, símbolo, valor SI, magnitud o None si es adimensional)
    specs: list[tuple[str, str, str, float | None, QuantityKind | None]] = [
        (g_state, "Temperatura", "T", state.T_K, "temperature"),
        (g_state, "Presión", "p", state.P_Pa, "pressure"),
        (g_state, "Volumen específico", "v", state.v_m3_per_kg, "specific_volume"),
        (g_state, "Densidad", "ρ = 1/v", state.rho_kg_per_m3, "density"),
        (g_state, "Energía interna específica", "u", state.u_J_per_kg, "specific_enthalpy"),
        (g_state, "Entalpía específica", "h", state.h_J_per_kg, "specific_enthalpy"),
        (g_state, "Entropía específica", "s", state.s_J_per_kg_K, "specific_entropy"),
        (g_state, "Título", "x", state.x, None),
    ]
    if state.superheat_K is not None:
        specs.append(
            (
                g_sat,
                "Grado de sobrecalentamiento",
                "T − T_sat(p)",
                state.superheat_K,
                "temperature_difference",
            )
        )
    if state.subcooling_K is not None:
        specs.append(
            (
                g_sat,
                "Grado de subenfriamiento",
                "T_sat(p) − T",
                state.subcooling_K,
                "temperature_difference",
            )
        )
    specs += [
        (g_sat, "Factor de compresibilidad", "Z = pv/(RT)", state.Z, None),
        (
            g_cal,
            "Calor específico a presión constante",
            "c_p",
            state.cp_J_per_kg_K,
            "specific_heat",
        ),
        (
            g_cal,
            "Calor específico a volumen constante",
            "c_v",
            state.cv_J_per_kg_K,
            "specific_heat",
        ),
        (g_cal, "Relación de calores específicos", "γ = c_p/c_v", state.gamma, None),
        (g_cal, "Velocidad del sonido", "w", state.speed_of_sound_m_per_s, "speed"),
        (g_tr, "Viscosidad dinámica", "μ", state.viscosity_Pa_s, "dynamic_viscosity"),
        (g_tr, "Conductividad térmica", "k", state.conductivity_W_per_m_K, "thermal_conductivity"),
        (
            g_tr,
            "Viscosidad cinemática",
            "ν = μ/ρ",
            state.kinematic_viscosity_m2_per_s,
            "diffusivity",
        ),
        (
            g_tr,
            "Difusividad térmica",
            "α = k/(ρ·c_p)",
            state.thermal_diffusivity_m2_per_s,
            "diffusivity",
        ),
        (g_tr, "Número de Prandtl", "Pr = μ·c_p/k", state.prandtl, None),
    ]

    def note_for(symbol: str, value: float | None) -> str:
        if value is not None:
            return ""
        if symbol == "x":
            return "solo se define dentro de la campana"
        return _MIXTURE_NOTE if in_mixture else _UNAVAILABLE_NOTE

    return [
        _row(group, name, symbol, value, kind, system, note_for(symbol, value))
        for group, name, symbol, value, kind in specs
    ]


def saturation_rows(sat: SaturationProperties, system: UnitSystem) -> list[PropertyRow]:
    """Fila de la tabla de saturación (Cengel A-4 si ``basis='T'``, A-5 si ``'P'``).

    Siempre trae T y p de saturación (una de las dos es el dato) y, para
    v, u, h y s, los valores f, fg y g; así las filas a p y a T del mismo
    estado se pueden poner lado a lado. Si el fluido tiene glide
    (pseudo-puro), la variable que no es dato se muestra por separado en
    el punto de burbuja (f) y en el de rocío (g).
    """
    group = "Saturación a p" if sat.basis == "P" else "Saturación a T"
    rows: list[PropertyRow] = []
    if sat.basis == "P" and abs(sat.glide_K) > 1e-6:
        rows.append(
            _row(group, "Temperatura de burbuja", "T_f", sat.liquid.T_K, "temperature", system)
        )
        rows.append(
            _row(group, "Temperatura de rocío", "T_g", sat.vapor.T_K, "temperature", system)
        )
    else:
        rows.append(
            _row(group, "Temperatura de saturación", "T_sat", sat.T_sat_K, "temperature", system)
        )
    if sat.basis == "T" and not math.isclose(sat.liquid.P_Pa, sat.vapor.P_Pa, rel_tol=1e-9):
        rows.append(_row(group, "Presión de burbuja", "p_f", sat.liquid.P_Pa, "pressure", system))
        rows.append(_row(group, "Presión de rocío", "p_g", sat.vapor.P_Pa, "pressure", system))
    else:
        rows.append(_row(group, "Presión de saturación", "p_sat", sat.P_sat_Pa, "pressure", system))

    specs: tuple[tuple[str, str, str, QuantityKind, float], ...] = (
        ("Volumen específico", "v", "v_m3_per_kg", "specific_volume", sat.v_fg_m3_per_kg),
        ("Energía interna", "u", "u_J_per_kg", "specific_enthalpy", sat.u_fg_J_per_kg),
        ("Entalpía", "h", "h_J_per_kg", "specific_enthalpy", sat.h_fg_J_per_kg),
        ("Entropía", "s", "s_J_per_kg_K", "specific_entropy", sat.s_fg_J_per_kg_K),
    )
    for name, sym, attr, kind, fg in specs:
        liquid, vapor = getattr(sat.liquid, attr), getattr(sat.vapor, attr)
        rows.append(_row(group, f"{name}, líquido saturado", f"{sym}_f", liquid, kind, system))
        rows.append(_row(group, f"{name} de vaporización (g − f)", f"{sym}_fg", fg, kind, system))
        rows.append(_row(group, f"{name}, vapor saturado", f"{sym}_g", vapor, kind, system))
    return rows


def format_value(value: float | None, sig: int = 6) -> str:
    """Número con ``sig`` cifras significativas, o ``"—"`` si no está definido.

    ``NaN`` cuenta como no definido (pandas convierte ``None`` en ``NaN``
    en las columnas numéricas).
    """
    if value is None or math.isnan(value):
        return "—"
    if value == 0.0:
        return "0"
    return f"{value:.{sig}g}"


# ---------------------------------------------------------------------
# Resumen de región y notas didácticas
# ---------------------------------------------------------------------


def _fmt(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    return (
        f"{format_value(convert_from_si(value_si, kind, system), sig)} {unit_label(kind, system)}"
    )


def region_summary(state: FluidState, system: UnitSystem) -> str:
    """Una oración (markdown) con la región y qué tan lejos está de saturación."""
    label = REGION_LABELS_ES[state.region]
    if state.region == "saturated_mixture" and state.x is not None:
        return (
            f"**{label}** — título x = {state.x:.4f}: el {100 * state.x:.1f} % de la masa "
            f"es vapor saturado y el resto, líquido saturado."
        )
    if state.region == "saturated_liquid":
        return f"**{label}** (x = 0): justo sobre la curva de líquido saturado."
    if state.region == "saturated_vapor":
        return f"**{label}** (x = 1): justo sobre la curva de vapor saturado."
    if state.region == "superheated_vapor":
        if state.superheat_K is not None and state.sat_at_P is not None:
            return (
                f"**{label}** — {_fmt(state.superheat_K, 'temperature_difference', system)} por "
                f"encima de la saturación (T_sat = "
                f"{_fmt(state.sat_at_P.vapor.T_K, 'temperature', system)} a esa presión)."
            )
        if state.T_K >= state.limits.T_crit_K:
            return f"**{label}** (gas): T supera la temperatura crítica."
        return f"**{label}**: por debajo de la presión del punto triple."
    if state.region == "compressed_liquid":
        if state.subcooling_K is not None and state.sat_at_P is not None:
            return (
                f"**{label}** — {_fmt(state.subcooling_K, 'temperature_difference', system)} por "
                f"debajo de la saturación (T_sat = "
                f"{_fmt(state.sat_at_P.liquid.T_K, 'temperature', system)} a esa presión)."
            )
        return f"**{label}** — la presión supera la crítica, pero T < T_c."
    return f"**{label}** — p y T por encima de las del punto crítico."


NoteKind = Literal["warning", "info"]


@dataclass(frozen=True)
class StateNote:
    """Observación para mostrar junto al resultado (markdown)."""

    kind: NoteKind
    text: str


# Margen para avisar que un par T-p cae sobre la curva de saturación.
_SATURATION_MARGIN_K = 0.5


def state_notes(state: FluidState, pair: PairCode, system: UnitSystem) -> list[StateNote]:
    """Advertencias y observaciones didácticas sobre el estado calculado."""
    notes: list[StateNote] = []
    lim = state.limits
    sat_p = state.sat_at_P

    if (
        pair == "TP"
        and sat_p is not None
        and state.region in ("compressed_liquid", "superheated_vapor")
    ):
        T_ref = sat_p.vapor.T_K if state.region == "superheated_vapor" else sat_p.liquid.T_K
        if abs(state.T_K - T_ref) < _SATURATION_MARGIN_K:
            notes.append(
                StateNote(
                    "warning",
                    "La temperatura está a menos de "
                    f"{_fmt(_SATURATION_MARGIN_K, 'temperature_difference', system)} de la de "
                    f"saturación a esa presión (T_sat = {_fmt(T_ref, 'temperature', system)}). "
                    "**Dentro de la campana T y p no son independientes**: con ese par no se "
                    "puede fijar un estado bifásico. CoolProp lo resolvió como "
                    f"*{REGION_LABELS_ES[state.region].lower()}*; si el fluido es una mezcla "
                    "líquido–vapor, usá p y x (o T y x).",
                )
            )

    near_T = abs(state.T_K / lim.T_crit_K - 1.0) < 0.02
    near_p = abs(state.P_Pa / lim.P_crit_Pa - 1.0) < 0.10
    if near_T and near_p:
        notes.append(
            StateNote(
                "warning",
                "Estás muy cerca del **punto crítico** (T_c = "
                f"{_fmt(lim.T_crit_K, 'temperature', system)}, p_c = "
                f"{_fmt(lim.P_crit_Pa, 'pressure', system)}): las propiedades cambian muy "
                "rápido, c_p y la compresibilidad divergen, y las tablas tienen pocos datos. "
                "Interpolar acá da errores grandes.",
            )
        )

    if state.region == "supercritical":
        notes.append(
            StateNote(
                "info",
                "Por encima de la presión **y** de la temperatura críticas no hay distinción "
                "entre líquido y vapor: no existe la campana ni el título.",
            )
        )
    elif state.region == "compressed_liquid" and state.P_Pa >= lim.P_crit_Pa:
        notes.append(
            StateNote(
                "info",
                "La presión supera la crítica pero T < T_c: siguiendo a Cengel se lo trata "
                "como líquido comprimido (en rigor es un fluido a presión supercrítica).",
            )
        )
    elif state.region == "superheated_vapor" and state.T_K >= lim.T_crit_K:
        notes.append(
            StateNote(
                "info",
                "T supera la temperatura crítica: es un gas que **no se puede licuar** "
                "comprimiéndolo a temperatura constante.",
            )
        )

    if state.region == "superheated_vapor":
        error_pct = abs(state.Z - 1.0) * 100.0
        if error_pct < 1.0:
            notes.append(
                StateNote(
                    "info",
                    f"Z = {state.Z:.4f} ≈ 1: a estas condiciones el vapor se comporta casi "
                    f"como gas ideal (pv = RT con un error de {error_pct:.2f} %).",
                )
            )
        elif error_pct > 5.0:
            notes.append(
                StateNote(
                    "info",
                    f"Z = {state.Z:.4f}: tratarlo como gas ideal daría un error de "
                    f"{error_pct:.1f} % en el volumen específico. Usá tablas o la ecuación "
                    "de estado.",
                )
            )

    if not lim.is_pure:
        glide = sat_p.glide_K if sat_p is not None else None
        glide_txt = (
            f" (deslizamiento de {_fmt(glide, 'temperature_difference', system)} entre el "
            "punto de burbuja y el de rocío a esta presión)"
            if glide
            else ""
        )
        notes.append(
            StateNote(
                "info",
                f"{FLUID_NAMES_ES.get(state.fluid, state.fluid)} es un fluido pseudo-puro: "
                "a presión constante el cambio de fase no ocurre a una única temperatura"
                f"{glide_txt}.",
            )
        )

    if state.T_K < lim.T_triple_K + 1.0:
        notes.append(
            StateNote(
                "info",
                "Estás a menos de 1 K del punto triple "
                f"({_fmt(lim.T_triple_K, 'temperature', system)}): por debajo, el fluido "
                "puede solidificarse y CoolProp ya no lo modela.",
            )
        )
    return notes


# ---------------------------------------------------------------------
# Procedimiento paso a paso (modo didáctico)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProcedureStep:
    """Un paso del procedimiento: título, explicación (markdown) y ecuaciones LaTeX."""

    title: str
    text: str = ""
    latex: tuple[str, ...] = ()


def _q(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Cantidad en LaTeX: número convertido + unidad."""
    value = convert_from_si(value_si, kind, system)
    return rf"{latex_number(value, sig)}\ {latex_unit(unit_label(kind, system))}"


def _n(value_si: float, kind: QuantityKind, system: UnitSystem, sig: int = 5) -> str:
    """Solo el número convertido (para sustituciones dentro de una fórmula)."""
    return latex_number(convert_from_si(value_si, kind, system), sig)


def _sym(kw: str) -> str:
    return INPUT_SPECS[kw].symbol


def _sat_value(phase: SaturatedPhase, y: str) -> float:
    return float(getattr(phase, _Y_ATTR[y]))


def _lever_line(y: str, x: float, sat: SaturationProperties, system: UnitSystem) -> str:
    """``y = y_f + x (y_g − y_f) = …`` con números."""
    kind = _Y_KIND[y]
    yf = _sat_value(sat.liquid, y)
    yg = _sat_value(sat.vapor, y)
    result = yf + x * (yg - yf)
    # y_f y el salto x·(y_g − y_f) en renglones separados: juntos no entran
    # en un celular (v_f tiene muchas cifras: 0.0011272).
    return latex_chain(
        y,
        rf"{y}_f + x\,({y}_g - {y}_f)",
        rf"{_n(yf, kind, system)} \\ &\quad + "
        rf"{latex_number(x, 5)}\,({_n(yg, kind, system)} - {latex_paren(_n(yf, kind, system))})",
        _q(result, kind, system),
    )


_GLIDE_TXT = (
    " Al ser un pseudo-puro, la temperatura no es constante dentro de la campana: va de "
    "la de burbuja (T_f, x = 0) a la de rocío (T_g, x = 1) a la misma presión."
)


def _two_phase_T_line(state: FluidState, sat: SaturationProperties, system: UnitSystem) -> str:
    """T dentro de la campana a p dada: T_sat, o entre burbuja y rocío si hay glide."""
    T = _q(state.T_K, "temperature", system)
    if abs(sat.glide_K) > 1e-6:
        # Pseudo-puro: la T va de la de burbuja (x = 0) a la de rocío (x = 1).
        return (
            rf"\begin{{aligned}}T_f &= {_q(sat.liquid.T_K, 'temperature', system)} \\ "
            rf"&\le T = {T} \\ &\le T_g = {_q(sat.vapor.T_K, 'temperature', system)}"
            r"\end{aligned}"
        )
    return rf"T = T_{{\mathrm{{sat}}}} = {T}"


def _sat_table_lines(
    sat: SaturationProperties, system: UnitSystem, ys: tuple[str, ...] = ("v", "u", "h", "s")
) -> list[str]:
    lines: list[str] = []
    if sat.basis == "P" and abs(sat.glide_K) > 1e-6:
        # Pseudo-puro: a esa presión hay una T de burbuja (f) y una de rocío (g).
        lines.append(
            rf"\begin{{aligned}}T_f &= {_q(sat.liquid.T_K, 'temperature', system)} \\ "
            rf"T_g &= {_q(sat.vapor.T_K, 'temperature', system)}\end{{aligned}}"
        )
    elif sat.basis == "P":
        lines.append(rf"T_{{\mathrm{{sat}}}}(p) = {_q(sat.T_sat_K, 'temperature', system)}")
    elif abs(sat.vapor.P_Pa - sat.liquid.P_Pa) > 1e-9 * sat.liquid.P_Pa:
        # Pseudo-puro a T dada: presión de burbuja (f) y de rocío (g).
        lines.append(
            rf"\begin{{aligned}}p_f &= {_q(sat.liquid.P_Pa, 'pressure', system)} \\ "
            rf"p_g &= {_q(sat.vapor.P_Pa, 'pressure', system)}\end{{aligned}}"
        )
    else:
        lines.append(rf"p_{{\mathrm{{sat}}}}(T) = {_q(sat.P_sat_Pa, 'pressure', system)}")
    if len(ys) == 1:
        y = ys[0]
        kind = _Y_KIND[y]
        lines.append(
            rf"\begin{{aligned}}{y}_f &= {_q(_sat_value(sat.liquid, y), kind, system)} \\ "
            rf"{y}_g &= {_q(_sat_value(sat.vapor, y), kind, system)}\end{{aligned}}"
        )
        return lines
    # Como en las tablas de saturación: una fila por propiedad, columnas f y
    # g. Las unidades van en el texto del paso (sat_units_text): una columna
    # más no entra en el ancho de un celular.
    rows = [
        rf"{y} & {_n(_sat_value(sat.liquid, y), _Y_KIND[y], system)} & "
        rf"{_n(_sat_value(sat.vapor, y), _Y_KIND[y], system)}"
        for y in ys
    ]
    lines.append(
        r"\begin{array}{c|rr} & \text{líquido } (f) & \text{vapor } (g) \\ \hline "
        + r" \\ ".join(rows)
        + r"\end{array}"
    )
    return lines


def _sat_units_text(system: UnitSystem, ys: tuple[str, ...] = ("v", "u", "h", "s")) -> str:
    """``"Unidades: v en m³/kg; u y h en kJ/kg; s en kJ/(kg·K)."`` para la tabla f | g."""
    groups: dict[str, list[str]] = {}
    for y in ys:
        groups.setdefault(unit_label(_Y_KIND[y], system), []).append(y)
    parts = [f"{' y '.join(names)} en {unit}" for unit, names in groups.items()]
    return "Unidades: " + "; ".join(parts) + "."


def _table_name(sat_basis: Literal["P", "T"]) -> str:
    return "por presión (Cengel A-5)" if sat_basis == "P" else "por temperatura (Cengel A-4)"


def _single_phase_table(region: str) -> str:
    if region == "superheated_vapor":
        return "la tabla de **vapor sobrecalentado** (Cengel A-6)"
    if region == "compressed_liquid":
        return "la tabla de **líquido comprimido** (Cengel A-7)"
    return "la tabla de **vapor sobrecalentado / supercrítico** (Cengel A-6)"


def _result_lines(state: FluidState, system: UnitSystem, skip: tuple[str, ...]) -> list[str]:
    """Propiedades resultantes que no fueron dato."""
    lines: list[str] = []
    for kw, kind in (
        ("t", "temperature"),
        ("p", "pressure"),
        ("v", "specific_volume"),
        ("u", "specific_enthalpy"),
        ("h", "specific_enthalpy"),
        ("s", "specific_entropy"),
    ):
        if kw in skip:
            continue
        lines.append(rf"{_sym(kw)} = {_q(input_value_si(state, kw), kind, system)}")  # type: ignore[arg-type]
    return lines


def build_procedure(state: FluidState, pair: PairCode, system: UnitSystem) -> list[ProcedureStep]:
    """Pasos para resolver el estado como se haría con las tablas.

    Sigue la secuencia de Cengel §3-5: con la variable "ancla" (p o T)
    se leen las propiedades de saturación, se compara la segunda
    propiedad con las de líquido (f) y vapor (g) saturados para ubicar
    la región y, si es vapor húmedo, se aplica la regla de la palanca
    (vademecum §12.2). Cierra con la aproximación de líquido
    incompresible (§13) o la comparación con gas ideal (§4.2, §7.3), y
    la verificación h = u + p·v (§3.2). Los valores se muestran en el
    sistema ``system``.
    """
    if pair not in PAIR_KWARGS:
        raise ValueError(f"Par '{pair}' no soportado.")
    steps = [_step_data(state, pair, system)]
    k1, k2 = PAIR_KWARGS[pair]
    if pair in ("PX", "TX"):
        steps += _steps_quality_given(state, pair, system)
    elif pair == "TP":
        steps += _steps_tp(state, system)
    elif pair == "HS":
        steps += _steps_hs(state, system)
    else:
        steps += _steps_anchor(state, anchor=k1, y=k2, system=system)

    if state.region == "compressed_liquid" and state.sat_at_T is not None:
        steps.append(_step_incompressible(state, system))
    if state.region == "superheated_vapor":
        steps.append(_step_ideal_gas(state, system))
    steps.append(_step_consistency(state, system))
    return steps


def _step_data(state: FluidState, pair: PairCode, system: UnitSystem) -> ProcedureStep:
    parts = []
    for kw in PAIR_KWARGS[pair]:
        spec = INPUT_SPECS[kw]
        if spec.kind is None:
            parts.append(rf"{spec.symbol} = {latex_number(input_value_si(state, kw), 4)}")
        else:
            parts.append(rf"{spec.symbol} = {_q(input_value_si(state, kw), spec.kind, system)}")
    name = FLUID_NAMES_ES.get(state.fluid, state.fluid)
    return ProcedureStep(
        title="Datos",
        text=(
            f"Fluido: **{name}**. Para una sustancia pura simple compresible, dos propiedades "
            "intensivas independientes fijan el estado (postulado de estado)."
        ),
        latex=tuple(parts),
    )


def _steps_quality_given(
    state: FluidState, pair: PairCode, system: UnitSystem
) -> list[ProcedureStep]:
    sat = state.sat_at_P if pair == "PX" else state.sat_at_T
    assert sat is not None and state.x is not None  # garantizado por la validación de entrada
    anchor = "la presión" if pair == "PX" else "la temperatura"
    steps = [
        ProcedureStep(
            title="Propiedades de saturación",
            text=(
                f"Con {anchor} se entra a la tabla de saturación {_table_name(sat.basis)} y "
                "se leen las propiedades del líquido saturado (f) y del vapor saturado (g). "
                f"{_sat_units_text(system)}"
            ),
            latex=tuple(_sat_table_lines(sat, system)),
        )
    ]
    if state.region == "saturated_liquid":
        region_txt = "Con x = 0 el estado es **líquido saturado**: sus propiedades son las de f."
    elif state.region == "saturated_vapor":
        region_txt = "Con x = 1 el estado es **vapor saturado seco**: sus propiedades son las de g."
    else:
        region_txt = (
            "Es **vapor húmedo**: cada propiedad específica es el promedio de las de f y g "
            "pesado con el título (vademecum §12.2)."
        )
        if abs(sat.glide_K) > 1e-6:
            region_txt += _GLIDE_TXT
    other_line = (
        _two_phase_T_line(state, sat, system)
        if pair == "PX"
        else rf"p = p_{{\mathrm{{sat}}}} = {_q(state.P_Pa, 'pressure', system)}"
    )
    steps.append(
        ProcedureStep(
            title="Regla de la palanca",
            text=region_txt,
            latex=(
                other_line,
                *(_lever_line(y, state.x, sat, system) for y in ("v", "u", "h", "s")),
            ),
        )
    )
    return steps


def _beyond_saturation_step(state: FluidState, anchor: str, system: UnitSystem) -> ProcedureStep:
    lim = state.limits
    if anchor == "p" and state.P_Pa >= lim.P_crit_Pa:
        line = (
            rf"p = {_q(state.P_Pa, 'pressure', system)} \ge p_c = "
            rf"{_q(lim.P_crit_Pa, 'pressure', system)}"
        )
        why = "Por encima de la presión crítica no hay cambio de fase: no existe la campana."
    elif anchor == "t" and state.T_K >= lim.T_crit_K:
        line = (
            rf"T = {_q(state.T_K, 'temperature', system)} \ge T_c = "
            rf"{_q(lim.T_crit_K, 'temperature', system)}"
        )
        why = "Por encima de la temperatura crítica no puede haber líquido y vapor en equilibrio."
    else:
        p_t = _q(lim.P_triple_Pa, "pressure", system)
        line = rf"p = {_q(state.P_Pa, 'pressure', system)} < p_t = {p_t}"
        why = "Por debajo del punto triple no existe equilibrio líquido–vapor."
    return ProcedureStep(
        title="Fuera del rango de saturación",
        text=(
            f"{why} El estado es **{REGION_LABELS_ES[state.region].lower()}** y se lee en "
            f"{_single_phase_table(state.region)} o, como acá, con la ecuación de estado:"
        ),
        latex=(line, *_result_lines(state, system, skip=())),
    )


def _steps_anchor(
    state: FluidState, anchor: str, y: str, system: UnitSystem
) -> list[ProcedureStep]:
    """Pares p-h, p-s, p-v, p-u, T-s y T-v: ancla + propiedad ``y``."""
    sat = state.sat_at_P if anchor == "p" else state.sat_at_T
    if sat is None:
        return [_beyond_saturation_step(state, anchor, system)]

    kind = _Y_KIND[y]
    y_val = input_value_si(state, y)
    yf = _sat_value(sat.liquid, y)
    yg = _sat_value(sat.vapor, y)
    anchor_txt = "la presión" if anchor == "p" else "la temperatura"
    steps = [
        ProcedureStep(
            title=f"Saturación a {anchor_txt} dada",
            text=(
                f"Con {anchor_txt} se entra a la tabla de saturación {_table_name(sat.basis)} "
                f"y se buscan {y}_f y {y}_g:"
            ),
            latex=tuple(_sat_table_lines(sat, system, ys=(y,))),
        )
    ]

    y_tex = rf"{y} = {_n(y_val, kind, system)}"
    if state.region == "compressed_liquid":
        compare = rf"{y_tex} < {y}_f = {_n(yf, kind, system)}"
        where = "a la izquierda de la campana: **líquido comprimido** (subenfriado)."
    elif state.region == "superheated_vapor":
        compare = rf"{y_tex} > {y}_g = {_n(yg, kind, system)}"
        where = "a la derecha de la campana: **vapor sobrecalentado**."
    else:
        compare = (
            rf"\begin{{aligned}}{y}_f = {_n(yf, kind, system)} &\le {y_tex} \\ "
            rf"&\le {y}_g = {_n(yg, kind, system)}\end{{aligned}}"
        )
        where = "dentro de la campana: **vapor húmedo**."
    steps.append(
        ProcedureStep(
            title="Ubicación del estado",
            text=f"Se compara {y} con {y}_f y {y}_g: el estado está {where}",
            latex=(compare,),
        )
    )

    if state.is_two_phase and state.x is not None:
        other_line = (
            _two_phase_T_line(state, sat, system)
            if anchor == "p"
            else rf"p = p_{{\mathrm{{sat}}}} = {_q(state.P_Pa, 'pressure', system)}"
        )
        quality = latex_chain(
            "x",
            rf"\frac{{{y} - {y}_f}}{{{y}_g - {y}_f}}",
            rf"\frac{{{_n(y_val, kind, system)} - {latex_paren(_n(yf, kind, system))}}}"
            rf"{{{_n(yg, kind, system)} - {latex_paren(_n(yf, kind, system))}}}",
            latex_number(state.x, 5),
        )
        others = tuple(_lever_line(z, state.x, sat, system) for z in ("v", "u", "h", "s") if z != y)
        steps.append(
            ProcedureStep(
                title="Título y demás propiedades (regla de la palanca)",
                text=(
                    "Se despeja el título de la regla de la palanca (vademecum §12.2) y con "
                    "él se calcula el resto:" + (_GLIDE_TXT if abs(sat.glide_K) > 1e-6 else "")
                ),
                latex=(quality, other_line, *others),
            )
        )
        return steps

    skip = (anchor, y)
    lines = _result_lines(state, system, skip=skip)
    if anchor == "p" and state.superheat_K is not None:
        lines.append(
            rf"\Delta T_{{\mathrm{{sob}}}} = T - T_{{\mathrm{{sat}}}}(p) = "
            rf"{_q(state.superheat_K, 'temperature_difference', system)}"
        )
    if anchor == "p" and state.subcooling_K is not None:
        lines.append(
            rf"\Delta T_{{\mathrm{{sub}}}} = T_{{\mathrm{{sat}}}}(p) - T = "
            rf"{_q(state.subcooling_K, 'temperature_difference', system)}"
        )
    anchor_sym = "p" if anchor == "p" else "T"
    steps.append(
        ProcedureStep(
            title="Propiedades en la región monofásica",
            text=(
                f"Con {anchor_sym} y {y} se entra a {_single_phase_table(state.region)} "
                "interpolando entre las filas que encierran el dato (ver la página "
                "**Interpolación**). La ecuación de estado da directamente:"
            ),
            latex=tuple(lines),
        )
    )
    return steps


def _steps_tp(state: FluidState, system: UnitSystem) -> list[ProcedureStep]:
    sat_T = state.sat_at_T
    if sat_T is None:
        return [_beyond_saturation_step(state, "t", system)]

    p_sat = sat_T.P_sat_Pa
    p_tex = _q(state.P_Pa, "pressure", system)
    lines = [rf"p_{{\mathrm{{sat}}}}(T) = {_q(p_sat, 'pressure', system)}"]
    if state.region == "compressed_liquid":
        lines.append(rf"p = {p_tex} > p_{{\mathrm{{sat}}}}(T)")
        where = (
            "A esa temperatura el fluido recién hierve a p_sat(T); a una presión mayor sigue "
            "siendo **líquido comprimido** (subenfriado)."
        )
    elif state.region == "superheated_vapor":
        lines.append(rf"p = {p_tex} < p_{{\mathrm{{sat}}}}(T)")
        where = (
            "A una presión menor que p_sat(T) el líquido ya se evaporó por completo: es "
            "**vapor sobrecalentado**."
        )
    else:
        lines.append(rf"p = {p_tex} \approx p_{{\mathrm{{sat}}}}(T)")
        where = (
            "T y p corresponden a la saturación: dentro de la campana no son independientes "
            "y no alcanzan para fijar el estado (hace falta el título)."
        )
    sat_P = state.sat_at_P
    if sat_P is not None and state.region in ("compressed_liquid", "superheated_vapor"):
        T_tex = _q(state.T_K, "temperature", system)
        if state.region == "compressed_liquid":
            T_sat = _q(sat_P.liquid.T_K, "temperature", system)
            lines.append(rf"T = {T_tex} < T_{{\mathrm{{sat}}}}(p) = {T_sat}")
        else:
            T_sat = _q(sat_P.vapor.T_K, "temperature", system)
            lines.append(rf"T = {T_tex} > T_{{\mathrm{{sat}}}}(p) = {T_sat}")
    steps = [
        ProcedureStep(
            title="Comparación con la saturación",
            text=(
                "Con T se busca la presión de saturación en la tabla por temperatura "
                f"(Cengel A-4) y se la compara con la dada. {where}"
            ),
            latex=tuple(lines),
        )
    ]
    result = _result_lines(state, system, skip=("t", "p"))
    if state.superheat_K is not None:
        result.append(
            rf"\Delta T_{{\mathrm{{sob}}}} = T - T_{{\mathrm{{sat}}}}(p) = "
            rf"{_q(state.superheat_K, 'temperature_difference', system)}"
        )
    if state.subcooling_K is not None:
        result.append(
            rf"\Delta T_{{\mathrm{{sub}}}} = T_{{\mathrm{{sat}}}}(p) - T = "
            rf"{_q(state.subcooling_K, 'temperature_difference', system)}"
        )
    steps.append(
        ProcedureStep(
            title="Propiedades del estado",
            text=(
                f"Con T y p se entra a {_single_phase_table(state.region)}, interpolando si "
                "hace falta. La ecuación de estado da directamente:"
            ),
            latex=tuple(result),
        )
    )
    return steps


def _steps_hs(state: FluidState, system: UnitSystem) -> list[ProcedureStep]:
    steps = [
        ProcedureStep(
            title="Diagrama de Mollier",
            text=(
                "Con h y s no hay una variable «ancla» (p o T) para entrar directo a las "
                "tablas de saturación. Gráficamente se resuelve en el diagrama de Mollier "
                "(h–s); con tablas hay que iterar en p hasta que el estado con esa h tenga "
                "la s dada. La ecuación de estado lo resuelve numéricamente:"
            ),
            latex=(
                rf"p = {_q(state.P_Pa, 'pressure', system)}",
                rf"T = {_q(state.T_K, 'temperature', system)}",
            ),
        )
    ]
    sat = state.sat_at_P
    if sat is None:
        return steps
    h, s = state.h_J_per_kg, state.s_J_per_kg_K
    lines = list(_sat_table_lines(sat, system, ys=("h", "s")))
    if state.is_two_phase and state.x is not None:
        x_h = (h - sat.liquid.h_J_per_kg) / sat.h_fg_J_per_kg
        x_s = (s - sat.liquid.s_J_per_kg_K) / sat.s_fg_J_per_kg_K
        lines += [
            rf"x = \frac{{h - h_f}}{{h_g - h_f}} = {latex_number(x_h, 5)}",
            rf"x = \frac{{s - s_f}}{{s_g - s_f}} = {latex_number(x_s, 5)}",
        ]
        text = (
            "Verificación con la tabla de saturación a esa presión: h y s caen entre f y g "
            "(**vapor húmedo**) y el título que se despeja con cualquiera de las dos es el "
            f"mismo. {_sat_units_text(system, ('h', 's'))}"
        )
    else:
        text = (
            "Verificación con la tabla de saturación a esa presión: el estado queda fuera de "
            f"la campana (**{REGION_LABELS_ES[state.region].lower()}**). "
            f"{_sat_units_text(system, ('h', 's'))}"
        )
    steps.append(
        ProcedureStep(title="Verificación con la saturación", text=text, latex=tuple(lines))
    )
    return steps


def _pct(approx: float, exact: float) -> str:
    if abs(exact) < 1e-12:
        return ""
    return rf"\quad (\text{{dif.}}\ {abs(approx - exact) / abs(exact) * 100:.2f}\,\%)"


def _step_incompressible(state: FluidState, system: UnitSystem) -> ProcedureStep:
    sat = state.sat_at_T
    assert sat is not None
    vf = sat.liquid.v_m3_per_kg
    correction = vf * (state.P_Pa - sat.P_sat_Pa)
    h_approx = sat.liquid.h_J_per_kg + correction
    eh = "specific_enthalpy"
    approx = r"\approx"
    lines = (
        latex_chain(
            "v",
            "v_f(T)",
            rf"{_q(vf, 'specific_volume', system)}{_pct(vf, state.v_m3_per_kg)}",
            relation=approx,
        ),
        latex_chain(
            "u",
            "u_f(T)",
            rf"{_q(sat.liquid.u_J_per_kg, eh, system)}"
            rf"{_pct(sat.liquid.u_J_per_kg, state.u_J_per_kg)}",
            relation=approx,
        ),
        latex_chain(
            "h",
            r"h_f(T) + v_f(T)\,[p - p_{\mathrm{sat}}(T)]",
            rf"{_n(sat.liquid.h_J_per_kg, eh, system)} + {_n(correction, eh, system)}",
            rf"{_q(h_approx, eh, system)}{_pct(h_approx, state.h_J_per_kg)}",
            relation=approx,
        ),
        latex_chain(
            "s",
            "s_f(T)",
            rf"{_q(sat.liquid.s_J_per_kg_K, 'specific_entropy', system)}"
            rf"{_pct(sat.liquid.s_J_per_kg_K, state.s_J_per_kg_K)}",
            relation=approx,
        ),
    )
    return ProcedureStep(
        title="Aproximación de líquido incompresible",
        text=(
            "Si no hay tabla de líquido comprimido para esa presión, se aproxima con el "
            "líquido saturado **a la misma temperatura** (Cengel §3-5; vademecum §13). La "
            "entalpía admite la corrección v_f·(p − p_sat). Entre paréntesis, la diferencia "
            "con el valor exacto de la ecuación de estado:"
        ),
        latex=lines,
    )


def _step_ideal_gas(state: FluidState, system: UnitSystem) -> ProcedureStep:
    R = state.limits.R_J_per_kg_K
    v_ig = R * state.T_K / state.P_Pa
    error_pct = abs(state.v_m3_per_kg - v_ig) / state.v_m3_per_kg * 100.0
    if error_pct < 1.0:
        verdict = "Z ≈ 1: el modelo de gas ideal es una buena aproximación."
    elif error_pct < 5.0:
        verdict = "El error es chico: el gas ideal sirve para estimaciones."
    else:
        verdict = (
            "El vapor está lejos del comportamiento de gas ideal: hay que usar tablas o la "
            "ecuación de estado."
        )
    return ProcedureStep(
        title="¿Gas ideal?",
        text=(
            "Se compara con la ecuación de estado del gas ideal pv = RT (vademecum §4.2) "
            "mediante el factor de compresibilidad Z (vademecum §7.3), con T absoluta. "
            f"Usar pv = RT erraría el volumen específico en {error_pct:.2f} %. {verdict}"
        ),
        latex=(
            rf"R = \frac{{R_u}}{{M}} = {_q(R, 'specific_entropy', system)}",
            rf"v_{{\mathrm{{gi}}}} = \frac{{R\,T}}{{p}} = {_q(v_ig, 'specific_volume', system)}",
            rf"Z = \frac{{p\,v}}{{R\,T}} = \frac{{v}}{{v_{{\mathrm{{gi}}}}}} = "
            rf"{latex_number(state.Z, 5)}",
        ),
    )


def pv_energy_factor(system: UnitSystem) -> float:
    """Cuánta energía específica (en unidades de ``system``) es 1 [p]·1 [v].

    Técnico: 1 bar·m³/kg = 100 kJ/kg. SI: 1 Pa·m³/kg = 1 J/kg. Inglés:
    1 psia·ft³/lb ≈ 0.18505 Btu/lb.
    """
    pv_si = convert_to_si(1.0, "pressure", system) * convert_to_si(1.0, "specific_volume", system)
    return convert_from_si(pv_si, "specific_enthalpy", system)


def _step_consistency(state: FluidState, system: UnitSystem) -> ProcedureStep:
    eh = "specific_enthalpy"
    pv_si = state.P_Pa * state.v_m3_per_kg
    factor = pv_energy_factor(system)
    p_unit = unit_label("pressure", system)
    v_unit = unit_label("specific_volume", system)
    e_unit = unit_label(eh, system)
    product = (
        rf"({_q(state.P_Pa, 'pressure', system)})\,"
        rf"({_q(state.v_m3_per_kg, 'specific_volume', system)})"
    )
    if math.isclose(factor, 1.0):
        pv_line = latex_chain(r"p\,v", product, _q(pv_si, eh, system))
        units_txt = ""
    else:
        pv_line = latex_chain(
            r"p\,v",
            rf"{product} \\ &\quad \cdot {latex_number(factor, 5)}\,"
            rf"\frac{{{latex_unit(e_unit)}}}{{{latex_unit(p_unit + '·' + v_unit)}}}",
            _q(pv_si, eh, system),
        )
        units_txt = (
            f" Ojo con las unidades de p·v: 1 {p_unit}·{v_unit} = "
            f"{format_value(factor, 5)} {e_unit}."
        )
    total = state.u_J_per_kg + pv_si
    return ProcedureStep(
        title="Verificación: h = u + p·v",
        text=f"La entalpía es h = u + p·v (vademecum §3.2).{units_txt}",
        latex=(
            pv_line,
            latex_chain(
                r"u + p\,v",
                rf"{_n(state.u_J_per_kg, eh, system)} + {_n(pv_si, eh, system)}",
                rf"{_q(total, eh, system)} = h\ \checkmark",
            ),
        ),
    )


# ---------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------


def _rows_as_dicts(rows: list[PropertyRow]) -> list[dict[str, Any]]:
    return [
        {
            "grupo": r.group,
            "propiedad": r.name,
            "simbolo": r.symbol,
            "valor": r.value,
            "unidad": r.unit,
            "nota": r.note,
        }
        for r in rows
    ]


def state_to_dict(state: FluidState, pair: PairCode, system: UnitSystem) -> dict[str, Any]:
    """Estado completo serializable a JSON (valores en el sistema activo + SI crudo)."""
    data: dict[str, Any] = {}
    for kw in PAIR_KWARGS[pair]:
        spec = INPUT_SPECS[kw]
        value_si = input_value_si(state, kw)
        if spec.kind is None:
            data[spec.symbol] = {"valor": value_si, "unidad": _DIMENSIONLESS}
        else:
            data[spec.symbol] = {
                "valor": convert_from_si(value_si, spec.kind, system),
                "unidad": unit_label(spec.kind, system),
            }
    return {
        "fluido": state.fluid,
        "fluido_es": FLUID_NAMES_ES.get(state.fluid, state.fluid),
        "par": pair,
        "datos": data,
        "sistema_de_unidades": system,
        "region": state.region,
        "region_es": REGION_LABELS_ES[state.region],
        "propiedades": _rows_as_dicts(property_rows(state, system)),
        "saturacion_a_p": (
            _rows_as_dicts(saturation_rows(state.sat_at_P, system)) if state.sat_at_P else None
        ),
        "saturacion_a_T": (
            _rows_as_dicts(saturation_rows(state.sat_at_T, system)) if state.sat_at_T else None
        ),
        "si": {
            "T_K": state.T_K,
            "P_Pa": state.P_Pa,
            "v_m3_per_kg": state.v_m3_per_kg,
            "u_J_per_kg": state.u_J_per_kg,
            "h_J_per_kg": state.h_J_per_kg,
            "s_J_per_kg_K": state.s_J_per_kg_K,
            "x": state.x,
        },
        "fuente": "CoolProp (HEOS; IAPWS-95 para el agua)",
    }


def state_to_csv(state: FluidState, pair: PairCode, system: UnitSystem) -> str:
    """CSV con la tabla de propiedades (y las de saturación, si existen)."""
    rows = property_rows(state, system)
    for sat in (state.sat_at_P, state.sat_at_T):
        if sat is not None:
            rows += saturation_rows(sat, system)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    name = FLUID_NAMES_ES.get(state.fluid, state.fluid)
    writer.writerow(["# fluido", name, "par", PAIR_LABELS_ES[pair], "sistema", system])
    writer.writerow(["grupo", "propiedad", "simbolo", "valor", "unidad", "nota"])
    for r in rows:
        writer.writerow(
            [r.group, r.name, r.symbol, "" if r.value is None else repr(r.value), r.unit, r.note]
        )
    return buffer.getvalue()


# ---------------------------------------------------------------------
# Tabla de varios estados (para resolver ciclos)
# ---------------------------------------------------------------------


def states_table(
    labeled_states: list[tuple[str, FluidState]], system: UnitSystem
) -> list[dict[str, Any]]:
    """Una fila por estado con T, p, v, u, h, s, x y región, en ``system``.

    Las claves llevan la unidad (``"h [kJ/kg]"``) para poder armar la
    tabla o el CSV directamente. ``x`` es ``None`` fuera de la campana.
    """
    columns: tuple[tuple[str, str, QuantityKind], ...] = (
        ("T", "T_K", "temperature"),
        ("p", "P_Pa", "pressure"),
        ("v", "v_m3_per_kg", "specific_volume"),
        ("u", "u_J_per_kg", "specific_enthalpy"),
        ("h", "h_J_per_kg", "specific_enthalpy"),
        ("s", "s_J_per_kg_K", "specific_entropy"),
    )
    rows: list[dict[str, Any]] = []
    for label, state in labeled_states:
        row: dict[str, Any] = {
            "Estado": label,
            "Fluido": FLUID_NAMES_ES.get(state.fluid, state.fluid),
        }
        for symbol, attr, kind in columns:
            header = f"{symbol} [{unit_label(kind, system)}]"
            row[header] = convert_from_si(getattr(state, attr), kind, system)
        row["x [-]"] = state.x
        row["Región"] = REGION_LABELS_ES[state.region]
        rows.append(row)
    return rows


def states_table_csv(labeled_states: list[tuple[str, FluidState]], system: UnitSystem) -> str:
    """CSV de :func:`states_table` (celdas vacías para propiedades no definidas)."""
    rows = states_table(labeled_states, system)
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return buffer.getvalue()
