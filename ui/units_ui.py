"""Selector global de unidades + helpers de UI — Fase 1.4.

Vive en ``ui/`` porque importa Streamlit (regla del proyecto: ``core/``
no importa Streamlit). El estado del sistema seleccionado vive en
``st.session_state["units_system"]``; el default es ``"Técnico"``.

Helpers principales:

- :func:`render_units_selector` — coloca el selectbox en el sidebar.
  Llamar después de :func:`ui.branding.sidebar_credits` en cada página.
- :func:`get_current_system` — devuelve el sistema actual (con fallback
  al default).
- :func:`number_input_si` — wrapper sobre ``st.number_input`` que muestra
  label + unidad del sistema actual y devuelve el valor en SI.
- :func:`quantity_label` — string con el valor en el sistema actual y su
  unidad. Útil para encabezados ``"Resultados a {…}"``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from core.units_system import (
    DEFAULT_SYSTEM,
    SUPPORTED_SYSTEMS,
    QuantityKind,
    UnitSystem,
    convert_from_si,
    convert_to_si,
    format_quantity,
    unit_label,
)

_SESSION_KEY = "units_system"


# ---------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------


def get_current_system() -> UnitSystem:
    """Devuelve el sistema seleccionado en session_state, o el default."""
    return st.session_state.get(_SESSION_KEY, DEFAULT_SYSTEM)  # type: ignore[no-any-return]


def render_units_selector() -> None:
    """Renderiza el selector global en el sidebar.

    Idempotente: se puede llamar una vez por página. Streamlit usa el
    ``key`` para conservar la selección entre páginas via session_state.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sistema de unidades")
    current = get_current_system()
    options = list(SUPPORTED_SYSTEMS)
    st.sidebar.selectbox(
        "Sistema de unidades",
        options,
        index=options.index(current),
        key=_SESSION_KEY,
        label_visibility="collapsed",
        help=(
            "SI: K, Pa, J/kg, J/(kg·K).  "
            "Técnico: °C, bar, kJ/kg, kJ/(kg·K).  "
            "Inglés: °F, psia, Btu/lb, Btu/(lb·°R)."
        ),
    )


# ---------------------------------------------------------------------
# Widget helper: number_input que opera en SI internamente
# ---------------------------------------------------------------------


def number_input_si(
    *,
    label: str,
    kind: QuantityKind,
    default_si: float,
    key: str,
    format: str = "%.4f",  # noqa: A002 — replicamos la API de st.number_input
    step: float | None = None,
    min_value_si: float | None = None,
    max_value_si: float | None = None,
    help: str | None = None,  # noqa: A002
) -> float:
    """``st.number_input`` con labels/unidades del sistema actual; devuelve SI.

    Parámetros relevantes
    ---------------------
    label : str
        Texto del label sin la unidad. Ej.: ``"Temperatura"``. La unidad
        se concatena automáticamente: ``"Temperatura [°C]"``.
    kind : QuantityKind
        Tipo de la magnitud (``"temperature"``, ``"pressure"``,
        ``"specific_enthalpy"``, …).
    default_si : float
        Valor por defecto en SI. Se convierte al sistema actual para
        mostrar.
    min_value_si, max_value_si : float, opcionales
        Límites en SI. Se convierten al sistema actual antes de pasarlos
        a ``st.number_input``.
    step : float, opcional
        Step del widget, **en unidades del sistema actual** (Streamlit
        no lo escala automáticamente). Si es ``None``, Streamlit elige.
    """
    system = get_current_system()
    full_label = f"{label} [{unit_label(kind, system)}]"
    default_user = convert_from_si(default_si, kind, system)

    kwargs: dict[str, Any] = {
        "label": full_label,
        "value": float(default_user),
        "format": format,
        "key": key,
    }
    if step is not None:
        kwargs["step"] = step
    if min_value_si is not None:
        kwargs["min_value"] = float(convert_from_si(min_value_si, kind, system))
    if max_value_si is not None:
        kwargs["max_value"] = float(convert_from_si(max_value_si, kind, system))
    if help is not None:
        kwargs["help"] = help

    user_value = st.number_input(**kwargs)
    return convert_to_si(float(user_value), kind, system)


# ---------------------------------------------------------------------
# Formato de salida
# ---------------------------------------------------------------------


def quantity_label(
    value_si: float,
    kind: QuantityKind,
    *,
    precision: int = 4,
) -> str:
    """``format_quantity`` con el sistema actual. Ej.: ``'25.00 °C'``."""
    return format_quantity(value_si, kind, get_current_system(), precision=precision)


def current_unit_label(kind: QuantityKind) -> str:
    """Etiqueta de la unidad del sistema actual para usar en tablas/headers."""
    return unit_label(kind, get_current_system())
