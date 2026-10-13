"""Muestrea los gráficos de Zukauskas → ``data/zukauskas_tube_bank_friction.csv``.

Fuente: Zukauskas, A. (1972). Heat transfer from tubes in crossflow. *Advances in
Heat Transfer* 8, 93–160. Son los gráficos del factor de fricción f y de la
corrección χ que reproducen Incropera et al. (2007), figs. 7.13 y 7.14, y Çengel y
Ghajar (2015), fig. 7-27: Δp = N_L·χ·f·ρ·V_máx²/2.

No tienen una fórmula. La digitalización es la de la biblioteca ht 1.2.0 (C. Bell,
``ht.conv_tube_bank``, licencia MIT: ``data/LICENSE-ht.txt``), que guarda cada
familia de curvas como un spline bicúbico. Este script evalúa cada curva sobre su
rango (el del gráfico, según la digitalización independiente del paquete tubebank
1.0) y guarda los puntos; la app interpola entre ellos como se lee un gráfico.

Curvas (columna ``chart``):

- ``f_aligned``: f del arreglo alineado cuadrado, una curva por P_L = S_L/D;
- ``f_staggered``: f del tresbolillo equilátero, una curva por P_T = S_T/D;
- ``chi_aligned``: χ contra (P_T − 1)/(P_L − 1), una curva por Re;
- ``chi_staggered``: χ contra P_T/P_L, una curva por Re.

Uso (ht no es dependencia de la app)::

    pip install ht==1.2.0
    python scripts/sample_zukauskas.py data/zukauskas_tube_bank_friction.csv
"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np

#: (chart, valor de la curva, x mínimo, x máximo, puntos por década de x)
CURVES: tuple[tuple[str, float, float, float, int], ...] = (
    ("f_aligned", 1.25, 30.0, 1.8e6, 20),
    ("f_aligned", 1.5, 30.0, 1.8e6, 20),
    ("f_aligned", 2.0, 1550.0, 1.8e6, 20),
    ("f_aligned", 2.5, 570.0, 1.8e6, 20),
    ("f_staggered", 1.25, 10.0, 2.4e6, 20),
    ("f_staggered", 1.5, 10.0, 2.4e6, 20),
    ("f_staggered", 2.0, 120.0, 2.4e6, 20),
    ("f_staggered", 2.5, 120.0, 2.4e6, 20),
    ("chi_aligned", 1.0e3, 0.1, 5.7, 40),
    ("chi_aligned", 1.0e4, 0.1, 5.7, 40),
    ("chi_aligned", 1.0e5, 0.1, 5.7, 40),
    ("chi_aligned", 1.0e6, 0.1, 5.7, 40),
    ("chi_staggered", 1.0e2, 1.155, 3.26, 40),
    ("chi_staggered", 1.0e3, 0.50, 3.54, 40),
    ("chi_staggered", 1.0e4, 0.48, 3.54, 40),
    ("chi_staggered", 1.0e5, 0.48, 3.28, 40),
)

HEADER = (
    "# Gráficos del banco de tubos de Zukauskas (1972), Advances in Heat Transfer 8, "
    "93-160 (Incropera et al., 2007, figs. 7.13 y 7.14; Çengel y Ghajar, 2015, fig. "
    "7-27): Δp = N_L·χ·f·ρ·V_máx²/2. chart = f_aligned (curva = P_L = S_L/D, x = Re), "
    "f_staggered (curva = P_T = S_T/D, x = Re), chi_aligned (curva = Re, "
    "x = (P_T − 1)/(P_L − 1)) o chi_staggered (curva = Re, x = P_T/P_L). Puntos "
    "muestreados de la digitalización de ht 1.2.0 (C. Bell, licencia MIT, "
    "data/LICENSE-ht.txt) con scripts/sample_zukauskas.py."
)


def _value(chart: str, curve: float, x: float) -> float:
    from fluids.numerics import bisplev
    from ht.conv_tube_bank import (
        dP_inline_correction_tck,
        dP_inline_f_tck,
        dP_staggered_correction_tck,
        dP_staggered_f_tck,
    )

    if chart == "f_aligned":
        return float(bisplev(x, curve, dP_inline_f_tck))
    if chart == "f_staggered":
        return float(bisplev(x, curve, dP_staggered_f_tck))
    if chart == "chi_aligned":
        return float(bisplev(x, curve, dP_inline_correction_tck))
    return float(bisplev(x, curve, dP_staggered_correction_tck))


def sample() -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for chart, curve, lo, hi, per_decade in CURVES:
        n = max(int(math.ceil(math.log10(hi / lo) * per_decade)), 2) + 1
        for x in np.geomspace(lo, hi, n):
            rows.append((chart, curve, float(x), _value(chart, curve, float(x))))
    return rows


def main(path: str) -> None:
    rows = sample()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(HEADER + "\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("chart", "curve", "x", "y"))
        for chart, curve, x, y in rows:
            writer.writerow((chart, f"{curve:g}", f"{x:.6g}", f"{y:.6g}"))
    print(f"{len(rows)} puntos → {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/zukauskas_tube_bank_friction.csv")
