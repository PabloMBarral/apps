"""Capa de UI para diagramas de propiedades — Fase 1.5a.

Vive en ``ui/`` porque importa Streamlit y plotly. Envuelve a
:mod:`core.diagrams` con un cache ``@st.cache_resource`` keyed por
``(fluido, sistema)`` para evitar recalcular isolíneas en cada rerun
de Streamlit.

Helpers
-------
- :func:`get_diagram` — ``@st.cache_resource`` sobre
  :func:`core.diagrams.build_diagram`.
- :func:`diagram_type_selector` — selectbox con etiquetas amigables
  (``"log p–h"``, ``"T–s"``, ``"h–s (Mollier)"``, ``"p–log v"``).
- :func:`render_diagram_plotly` — dibuja isolíneas + overlays (puntos
  y curvas conceptuales) usando ``st.plotly_chart``.

Convención
----------
Los puntos (``DiagramPoint``) y procesos (``ProcessOverlay`` de
``core.diagrams``) viajan en **SI**. La capa UI se encarga de
convertirlos al sistema activo del usuario al momento de plottear.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from fluprodia import FluidPropertyDiagram

from core.diagrams import (
    AXIS_MAP,
    DEFAULT_RANGES,
    SUPPORTED_DIAGRAM_TYPES,
    DiagramSpec,
    DiagramType,
    ProcessOverlay,
    build_diagram,
    process_to_diagram_coords,
    state_to_diagram_coords,
)
from core.fluids import StatePoint
from core.units_system import UnitSystem, convert_from_si, unit_label

# ---------------------------------------------------------------------
# Labels amigables
# ---------------------------------------------------------------------

DIAGRAM_TYPE_LABELS: dict[DiagramType, str] = {
    "logph": "log p–h",
    "Ts": "T–s",
    "hs": "h–s (Mollier)",
    "plogv": "p–log v",
}


# Para cada axis-property, qué QuantityKind aplica.
_PROP_TO_KIND = {
    "h": "specific_enthalpy",
    "T": "temperature",
    "s": "specific_entropy",
    "p": "pressure",
    "vol": "specific_volume",
}


# ---------------------------------------------------------------------
# DiagramPoint — punto de estado para overlay
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DiagramPoint:
    """Punto de estado para plotear como marker sobre el diagrama.

    Atributos
    ---------
    state : StatePoint
        Estado termodinámico en SI.
    label : str
        Etiqueta corta (ej. ``"1"``, ``"2s"``, ``"3"``).
    color : str
        Color CSS del marker.
    """

    state: StatePoint
    label: str
    color: str = "#1f77b4"


# ---------------------------------------------------------------------
# Cache de diagramas
# ---------------------------------------------------------------------


@st.cache_resource(show_spinner="Calculando isolíneas…")
def get_diagram(fluid: str, system: UnitSystem) -> FluidPropertyDiagram:
    """Construye (con cache) el :class:`FluidPropertyDiagram` para
    ``(fluido, sistema)``.

    El cache es por (fluido, sistema) — calc_isolines() puede tardar
    1.1 s para Water; cachear evita el costo en cada rerun. El cache
    es global del worker de Streamlit (no por sesión).
    """
    spec = DiagramSpec(fluid=fluid, system=system)
    return build_diagram(spec)


# ---------------------------------------------------------------------
# Selector de tipo de diagrama
# ---------------------------------------------------------------------


def diagram_type_selector(
    *,
    key: str,
    default: DiagramType = "logph",
    available: Sequence[DiagramType] | None = None,
    label: str = "Tipo de diagrama",
) -> DiagramType:
    """Selectbox de tipo de diagrama con etiquetas amigables.

    Parámetros
    ----------
    key : str
        Clave de ``st.session_state`` para preservar la elección.
    default : DiagramType
        Tipo inicial.
    available : sequence of DiagramType, opcional
        Subset de tipos a ofrecer. Si es ``None``, ofrece los 4.
    label : str
        Texto del selectbox.

    Devuelve
    --------
    DiagramType
    """
    options: list[DiagramType] = (
        list(available) if available is not None else list(SUPPORTED_DIAGRAM_TYPES)
    )
    if default not in options:
        default = options[0]
    return st.selectbox(  # type: ignore[no-any-return]
        label,
        options,
        index=options.index(default),
        format_func=lambda dt: DIAGRAM_TYPE_LABELS.get(dt, dt),
        key=key,
    )


# ---------------------------------------------------------------------
# Renderizado plotly
# ---------------------------------------------------------------------


def _axis_window_in_user_units(
    fluid: str, diagram_type: DiagramType, system: UnitSystem
) -> tuple[float, float, float, float]:
    """Calcula ``(x_min, x_max, y_min, y_max)`` para el draw, en unidades
    del sistema activo.

    Para los ejes ``h`` y ``s`` no tenemos un rango cerrado SI a priori,
    así que dejamos ``None`` y plotly autoescala — pero fluprodia
    requiere los 4 floats. La estrategia: para ``p`` y ``T`` usamos
    ``DEFAULT_RANGES``; para ``h`` y ``s`` calculamos extremos típicos a
    partir de la ventana de presión/temperatura del fluido vía CoolProp
    sería costoso → usamos defaults razonables convertidos al sistema.
    """
    rng = DEFAULT_RANGES[fluid]
    prop_x, prop_y, _x_log, _y_log = AXIS_MAP[diagram_type]

    # Defaults SI por propiedad. Para h y s, usamos rangos amplios típicos.
    si_window: dict[str, tuple[float, float]] = {
        "T": (rng.T_K_min, rng.T_K_max),
        "p": (rng.p_Pa_min, rng.p_Pa_max),
        "h": (0.0, 4.0e6),  # 0 a 4000 kJ/kg cubre agua/refrigerantes
        "s": (0.0, 1.0e4),  # 0 a 10 kJ/(kg·K)
        "vol": (1.0e-4, 1.0e1),  # 1e-4 a 10 m³/kg
    }

    x_lo_si, x_hi_si = si_window[prop_x]
    y_lo_si, y_hi_si = si_window[prop_y]

    kind_x = _PROP_TO_KIND[prop_x]
    kind_y = _PROP_TO_KIND[prop_y]

    x_lo = convert_from_si(x_lo_si, kind_x, system)  # type: ignore[arg-type]
    x_hi = convert_from_si(x_hi_si, kind_x, system)  # type: ignore[arg-type]
    y_lo = convert_from_si(y_lo_si, kind_y, system)  # type: ignore[arg-type]
    y_hi = convert_from_si(y_hi_si, kind_y, system)  # type: ignore[arg-type]

    # Para ejes log, evitar valores ≤ 0 (la conversión de °C/°F puede
    # darnos negativos en T cuando el rango incluye temperaturas bajas).
    if _x_log and x_lo <= 0:
        x_lo = max(x_lo, 1e-3)
    if _y_log and y_lo <= 0:
        y_lo = max(y_lo, 1e-3)

    # Si por alguna razón el orden se invierte (no debería con factores >0),
    # restaurarlo.
    if x_lo > x_hi:
        x_lo, x_hi = x_hi, x_lo
    if y_lo > y_hi:
        y_lo, y_hi = y_hi, y_lo

    return x_lo, x_hi, y_lo, y_hi


def _axis_labels(diagram_type: DiagramType, system: UnitSystem) -> tuple[str, str]:
    """Etiquetas de ejes ``x`` e ``y`` con su unidad."""
    prop_x, prop_y, _x_log, _y_log = AXIS_MAP[diagram_type]
    kind_x = _PROP_TO_KIND[prop_x]
    kind_y = _PROP_TO_KIND[prop_y]
    x_label = f"{prop_x} [{unit_label(kind_x, system)}]"  # type: ignore[arg-type]
    y_label = f"{prop_y} [{unit_label(kind_y, system)}]"  # type: ignore[arg-type]
    return x_label, y_label


def render_diagram_plotly(
    *,
    fluid: str,
    diagram_type: DiagramType,
    system: UnitSystem,
    points: Sequence[DiagramPoint] | None = None,
    overlays: Sequence[ProcessOverlay] | None = None,
    height: int = 520,
    chart_key: str | None = None,
) -> None:
    """Dibuja el diagrama con isolíneas + puntos + overlays.

    Parámetros
    ----------
    fluid, diagram_type, system :
        Definen el diagrama a construir/recuperar del cache.
    points : sequence of DiagramPoint, opcional
        Markers numerados a superponer.
    overlays : sequence of ProcessOverlay, opcional
        Curvas conceptuales (procesos) a superponer.
    height : int
        Alto del chart en píxeles.
    chart_key : str, opcional
        Key para ``st.plotly_chart`` (necesario si se renderizan varios
        diagramas en una sola página).
    """
    diagram = get_diagram(fluid, system)
    x_min, x_max, y_min, y_max = _axis_window_in_user_units(fluid, diagram_type, system)

    fig = diagram.draw_isolines_plotly(
        diagram_type,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    # Overlays (procesos)
    if overlays:
        for ov in overlays:
            x, y = process_to_diagram_coords(ov.coords_si, diagram_type, system)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=ov.name,
                    line=dict(color=ov.color, dash=ov.dash, width=2.5),
                    hovertemplate=f"{ov.name}<br>x=%{{x:.4g}}<br>y=%{{y:.4g}}<extra></extra>",
                )
            )

    # Markers (puntos de estado)
    if points:
        for pt in points:
            x, y = state_to_diagram_coords(pt.state, diagram_type, system)
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    name=pt.label,
                    text=[pt.label],
                    textposition="top center",
                    textfont=dict(size=13, color=pt.color),
                    marker=dict(
                        size=10,
                        color=pt.color,
                        line=dict(width=1.5, color="white"),
                        symbol="circle",
                    ),
                    hovertemplate=(
                        f"<b>{pt.label}</b><br>x=%{{x:.4g}}<br>y=%{{y:.4g}}<extra></extra>"
                    ),
                )
            )

    x_label, y_label = _axis_labels(diagram_type, system)
    fig.update_layout(
        height=height,
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=60, r=20, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


# ---------------------------------------------------------------------
# Helpers extra (numeración por defecto, colores)
# ---------------------------------------------------------------------

#: Paleta consistente para los puntos numerados de las páginas que llaman
#: a :func:`render_diagram_plotly`. Pensada para ser legible sobre las
#: isolíneas claras de fluprodia.
DEFAULT_POINT_COLORS: tuple[str, ...] = (
    "#1f77b4",  # azul
    "#ff7f0e",  # naranja
    "#2ca02c",  # verde
    "#d62728",  # rojo
    "#9467bd",  # violeta
    "#8c564b",  # marrón
    "#e377c2",  # rosa
    "#7f7f7f",  # gris
)


def auto_point_colors(n: int) -> list[str]:
    """Devuelve ``n`` colores cíclicos de :data:`DEFAULT_POINT_COLORS`."""
    if n <= 0:
        return []
    palette = DEFAULT_POINT_COLORS
    return [palette[i % len(palette)] for i in range(n)]


# Silenciar warning de import no usado de numpy (se usa indirectamente).
_ = np
