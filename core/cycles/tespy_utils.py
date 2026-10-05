"""Helpers comunes para los ciclos que se resuelven con TESPy — Fase 3.1.

TESPy (Witte & Tuschy, 2020, *JOSS* 5(49), 2178) arma la planta como una
red de componentes y conexiones y resuelve el sistema de ecuaciones con
Newton. Estos helpers fijan la convención del proyecto (todo en SI dentro
de ``core/``) y traducen al castellano los casos en que el cálculo no
converge, para que el alumno sepa qué revisar. No importan Streamlit.

Basado en el tutorial oficial de TESPy 0.11 (``tutorial/basics/rankine.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from tespy.networks import Network

# Unidades SI explícitas. En TESPy 0.11 fijar "pressure" sin
# "pressure_difference" emite un FutureWarning (en 0.12 dejarán de ir juntas).
_SI_UNITS: dict[str, str] = {
    "temperature": "K",
    "pressure": "Pa",
    "pressure_difference": "Pa",
    "enthalpy": "J/kg",
    "entropy": "J/kg/K",
    "power": "W",
    "heat": "W",
    "mass_flow": "kg/s",
}

# Nombre del logger de TESPy (tespy.tools.logger.TESPY_LOGGER_ID).
_TESPY_LOGGER = "TESPyLogger"


def new_network() -> Network:
    """Red de TESPy en unidades SI, sin informe de iteraciones en consola."""
    network = Network(iterinfo=False)
    network.units.set_defaults(**_SI_UNITS)
    return network


@contextmanager
def _quiet_tespy() -> Iterator[None]:
    """Silencia los avisos de TESPy en consola mientras resuelve.

    Los problemas que importan al alumno se informan con el ``ValueError``
    de :func:`solve`; el resto (p. ej. "Invalid value for P") es ruido.
    """
    logger = logging.getLogger(_TESPY_LOGGER)
    previous = logger.level
    logger.setLevel(logging.ERROR + 1)
    try:
        yield
    finally:
        logger.setLevel(previous)


# Códigos de ``Network.status`` (TESPy 0.11): 0 convergió; 1 convergió con
# parámetros fuera de rango (p. ej. una turbina que comprime); 2 no avanzó o
# no cumplió las especificaciones; 3 sistema singular; 99 error inesperado.
_STATUS_ES: dict[int, str] = {
    1: (
        "la solución quedó fuera del rango físico de algún componente (por ejemplo, "
        "una turbina que tendría que comprimir o una bomba que entregaría trabajo)"
    ),
    2: "el cálculo no convergió con esos datos",
    3: "los datos no alcanzan para determinar todos los estados (sistema singular)",
    99: "el cálculo se interrumpió por un error numérico",
}


def solve(network: Network, *, what: str) -> None:
    """Resuelve ``network`` en modo diseño o explica en castellano por qué no.

    Parámetros
    ----------
    what :
        Qué se está calculando, para el mensaje (p. ej. ``"el ciclo de Rankine"``).

    Raises
    ------
    ValueError
        Si TESPy no llega a una solución válida (``status`` distinto de 0).
    """
    try:
        with _quiet_tespy():
            network.solve("design")
    except Exception as exc:  # TESPy puede lanzar errores de CoolProp o propios
        raise ValueError(
            f"No se pudo calcular {what}: {exc}. Revisá que los datos sean coherentes."
        ) from exc
    status = int(getattr(network, "status", 99))
    if status != 0:
        reason = _STATUS_ES.get(status, f"TESPy terminó con estado {status}")
        raise ValueError(
            f"No se pudo calcular {what}: {reason}. Revisá que los datos sean coherentes "
            "(presiones, temperaturas y rendimientos)."
        )
