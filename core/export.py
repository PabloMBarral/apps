"""Exportación genérica de resultados — Fase 1.7.

Los módulos de cálculo arman un ``dict`` serializable a JSON con su
resultado (por ejemplo :func:`core.interpolation.interpolation_to_dict`);
este módulo lo aplana a un CSV de dos columnas (``campo``, ``valor``)
para abrirlo en una planilla. No importa Streamlit.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Any


def flatten(data: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Aplana diccionarios y listas anidadas a pares ``(campo, valor)``.

    - ``{"a": {"b": 1}}`` → ``[("a.b", 1)]``
    - ``{"a": [{"b": 1}, {"b": 2}]}`` → ``[("a[1].b", 1), ("a[2].b", 2)]``
    - listas de valores simples se unen con ``"; "``.
    """
    rows: list[tuple[str, Any]] = []
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows += flatten(value, name)
        elif (
            isinstance(value, list | tuple)
            and value
            and all(isinstance(item, Mapping) for item in value)
        ):
            for i, item in enumerate(value, start=1):
                rows += flatten(item, f"{name}[{i}]")
        elif isinstance(value, list | tuple):
            rows.append((name, "; ".join(str(item) for item in value)))
        else:
            rows.append((name, value))
    return rows


def dict_to_csv(data: Mapping[str, Any]) -> str:
    """CSV ``campo,valor`` de :func:`flatten` (``None`` → celda vacía)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["campo", "valor"])
    for name, value in flatten(data):
        writer.writerow([name, "" if value is None else value])
    return buffer.getvalue()
