"""Extrae de la base de NASA CEA las especies que usa la combustión → ``data/nasa9_thermo.csv``.

Fuente: los polinomios de 9 coeficientes de McBride, Zehe & Gordon (2002),
*NASA Glenn Coefficients for Calculating Thermodynamic Properties of
Individual Species*, NASA/TP-2002-211556 (la fuente que cita el vademecum
§16.12). El archivo ``thermo.inp`` está en el repositorio oficial de NASA CEA:
https://github.com/nasa/cea (``data/thermo.inp``).

Uso::

    python scripts/extract_nasa9.py ruta/a/thermo.inp

Formato de ``thermo.inp`` (TP-2002-211556, apéndice A): por especie, un
renglón con el nombre (columnas 1–18); otro con la cantidad de tramos, la
fórmula (5 pares elemento–cantidad), la fase (0 gas), M y h_f a 298,15 K
(J/mol); y por tramo, T_mín, T_máx y los coeficientes a1…a7, b1 y b2 en dos
renglones (formato Fortran ``D``).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "nasa9_thermo.csv"

#: (clave de la app, nombre en thermo.inp, nombre en castellano, fase)
SPECIES: tuple[tuple[str, str, str, str], ...] = (
    ("CO2", "CO2", "dióxido de carbono", "g"),
    ("CO", "CO", "monóxido de carbono", "g"),
    ("H2O", "H2O", "agua (vapor)", "g"),
    ("H2", "H2", "hidrógeno", "g"),
    ("O2", "O2", "oxígeno", "g"),
    ("N2", "N2", "nitrógeno", "g"),
    ("Ar", "Ar", "argón", "g"),
    ("OH", "OH", "radical hidroxilo", "g"),
    ("H", "H", "hidrógeno atómico", "g"),
    ("O", "O", "oxígeno atómico", "g"),
    ("NO", "NO", "óxido nítrico", "g"),
    ("N", "N", "nitrógeno atómico", "g"),
    ("SO2", "SO2", "dióxido de azufre", "g"),
    ("CH4", "CH4", "metano", "g"),
    ("C2H6", "C2H6", "etano", "g"),
    ("C3H8", "C3H8", "propano", "g"),
    ("n-C4H10", "C4H10,n-butane", "n-butano", "g"),
    ("i-C4H10", "C4H10,isobutane", "isobutano", "g"),
    ("n-C5H12", "C5H12,n-pentane", "n-pentano", "g"),
    ("C2H4", "C2H4", "etileno", "g"),
    ("C2H2", "C2H2,acetylene", "acetileno", "g"),
    ("H2S", "H2S", "sulfuro de hidrógeno", "g"),
    ("NH3", "NH3", "amoníaco", "g"),
    ("CH3OH", "CH3OH", "metanol (vapor)", "g"),
    ("C2H5OH", "C2H5OH", "etanol (vapor)", "g"),
    ("C8H18", "C8H18,n-octane", "n-octano (vapor)", "g"),
    ("C12H23", "Jet-A(g)", "querosén Jet-A (vapor)", "g"),
    ("H2O(l)", "H2O(L)", "agua líquida", "l"),
    ("CH3OH(l)", "CH3OH(L)", "metanol", "l"),
    ("C2H5OH(l)", "C2H5OH(L)", "etanol", "l"),
    ("C8H18(l)", "C8H18(L),n-octa", "n-octano", "l"),
    ("C12H23(l)", "Jet-A(L)", "querosén Jet-A", "l"),
)

ELEMENTS = ("C", "H", "O", "N", "S", "Ar")
COLUMNS = ["key", "nasa_name", "name", "phase", *ELEMENTS, "M_g_per_mol", "hf_J_per_mol"] + [
    "T_lo_K",
    "T_hi_K",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "b1",
    "b2",
    "ref",
]


def _number(text: str) -> float:
    return float(text.replace("D", "E"))


def parse(path: Path) -> dict[str, dict]:
    """Registros de ``thermo.inp`` por nombre (solo los que tienen polinomios)."""
    lines = path.read_text(encoding="latin-1").splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("thermo")) + 2
    records: dict[str, dict] = {}
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.startswith(("!", "END")):
            i += 1
            continue
        name = line[:18].strip()
        head = lines[i + 1]
        n_intervals = int(head[0:2])
        formula: dict[str, float] = {}
        for k in range(5):
            element = head[10 + 8 * k : 12 + 8 * k].strip().capitalize()
            count = head[12 + 8 * k : 18 + 8 * k].strip()
            if element and count and float(count) != 0.0:
                formula[element] = float(count)
        record = {
            "ref": head[3:9].strip(),
            "formula": formula,
            "M": float(head[52:65]),
            "hf": float(head[65:80]),
            "intervals": [],
        }
        i += 2
        if n_intervals == 0:  # reactivo con la entalpía a una sola temperatura
            i += 1
        for _ in range(n_intervals):
            t_line, c1, c2 = lines[i], lines[i + 1], lines[i + 2]
            coeffs = [_number(c1[16 * k : 16 * (k + 1)]) for k in range(5)]
            coeffs += [_number(c2[0:16]), _number(c2[16:32])]
            coeffs += [_number(c2[48:64]), _number(c2[64:80])]
            record["intervals"].append((float(t_line[0:11]), float(t_line[11:21]), coeffs))
            i += 3
        records[name] = record
    return records


def main(thermo_inp: str) -> None:
    records = parse(Path(thermo_inp))
    rows = []
    for key, nasa_name, name, phase in SPECIES:
        rec = records[nasa_name]
        for T_lo, T_hi, coeffs in rec["intervals"]:
            if T_lo >= 6000.0:  # más allá de la combustión
                continue
            atoms = [rec["formula"].get(el, 0.0) for el in ELEMENTS]
            rows.append(
                [key, nasa_name, name, phase, *(f"{a:g}" for a in atoms)]
                + [f"{rec['M']:.6f}", f"{rec['hf']:.3f}", f"{T_lo:.3f}", f"{T_hi:.3f}"]
                + [f"{c:.9e}" for c in coeffs]
                + [rec["ref"]]
            )
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# Polinomios NASA de 9 coeficientes (McBride, Zehe & Gordon 2002, "
            "NASA/TP-2002-211556), de thermo.inp de NASA CEA (github.com/nasa/cea). "
            "Generado con scripts/extract_nasa9.py.\n"
        )
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"{len(rows)} tramos de {len(SPECIES)} especies → {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python scripts/extract_nasa9.py ruta/a/thermo.inp")
    main(sys.argv[1])
