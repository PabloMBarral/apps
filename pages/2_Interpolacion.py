"""Página 2 — Interpolación lineal simple y de doble entrada.

Dos pestañas: lineal sobre tablas de dos columnas y bilineal sobre
tablas 2D. Para cada modo, además del resultado se muestra el
procedimiento didáctico (fórmula en LaTeX, valores reemplazados y
explicación en español plano) y, cuando la tabla es reconocida como
una tabla termodinámica clásica (saturación o vapor sobrecalentado),
una comparación contra CoolProp con el error relativo.

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Expansor `📖 Fórmulas teóricas` con link al apartado correspondiente
  de vademecum-termo.
- Botón de exportar resultados (CSV / JSON).
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd
import streamlit as st

from core.fluids import state_from_pair
from core.interpolation import (
    InterpolationResult,
    bilinear_from_table,
    linear_from_table,
)

# ---------------------------------------------------------------------
# Constantes y datos de ejemplo
# ---------------------------------------------------------------------

SUPPORTED_FLUIDS = ["Water", "R134a", "R410A", "Ammonia", "CarbonDioxide", "Air"]

EXAMPLE_LINEAR_CSV = """T [°C],h_f [kJ/kg]
100,419.06
110,461.42
120,503.81
130,546.38
140,589.16"""

# Tabla 2x4 de vapor de agua sobrecalentado generada con CoolProp
# (T en °C, P en bar, h en kJ/kg). Aproximaciones a 4 cifras.
EXAMPLE_BILINEAR_CSV = """T [°C],1.0,3.0,5.0,10.0
200,2875.5,2865.6,2855.8,2828.3
250,2974.5,2967.7,2960.7,2942.6
300,3074.5,3069.6,3064.6,3051.7
350,3175.6,3171.7,3167.8,3157.7"""

# Propiedad saturada → (símbolo CoolProp, Q)
_SAT_PROPS: dict[str, tuple[str, float, str]] = {
    "hf": ("H", 0.0, "h_f"),
    "hg": ("H", 1.0, "h_g"),
    "sf": ("S", 0.0, "s_f"),
    "sg": ("S", 1.0, "s_g"),
}


# ---------------------------------------------------------------------
# Helpers de parsing y unidades
# ---------------------------------------------------------------------


def _parse_header(name: str) -> tuple[str, str]:
    """Devuelve ``(base, unit)``: símbolo canónico (lower, sin guión bajo) y unidad."""
    s = str(name).strip()
    m = re.match(r"^(.+?)(?:\s*[\[\(](.+?)[\]\)])?\s*$", s)
    if not m:
        return s.lower(), ""
    base = m.group(1).strip().lower().replace("_", "")
    unit = (m.group(2) or "").strip().lower()
    return base, unit


def _t_to_kelvin(value: float, unit: str) -> float | None:
    u = unit.strip().lower()
    if u in ("", "°c", "ºc", "c", "celsius"):
        return value + 273.15
    if u in ("k", "kelvin"):
        return value
    return None


def _p_to_pascal(value: float, unit: str) -> float | None:
    u = unit.strip().lower()
    if u in ("", "bar", "bar(a)", "bara"):
        return value * 1.0e5
    if u == "pa":
        return value
    if u == "kpa":
        return value * 1.0e3
    if u == "mpa":
        return value * 1.0e6
    return None


def _specific_from_si(value_si: float, unit: str) -> float | None:
    """Convierte un valor SI (J/kg ó J/(kg·K)) a la unidad del header."""
    u = unit.strip().lower().replace(" ", "")
    if u in (
        "",
        "kj/kg",
        "kj/(kg·k)",
        "kj/(kg.k)",
        "kj/(kgk)",
        "kj/kgk",
        "kj/kg/k",
    ):
        return value_si / 1.0e3
    if u in ("j/kg", "j/(kg·k)", "j/(kg.k)", "j/(kgk)", "j/kgk", "j/kg/k"):
        return value_si
    return None


@st.cache_data(show_spinner=False)
def _parse_csv_text(csv_text: str, *, decimal_comma: bool) -> pd.DataFrame:
    decimal = "," if decimal_comma else "."
    try:
        df = pd.read_csv(
            StringIO(csv_text),
            sep=None,
            engine="python",
            decimal=decimal,
        )
    except Exception as exc:
        raise ValueError(
            f"No pude parsear la tabla. Revisá el formato (separador, "
            f"decimal, encabezados). Detalle: {exc}"
        ) from exc
    return df


def _parse_linear_csv(csv_text: str, *, decimal_comma: bool) -> pd.DataFrame:
    df = _parse_csv_text(csv_text, decimal_comma=decimal_comma)
    if df.shape[1] < 2:
        raise ValueError(f"La tabla lineal necesita al menos 2 columnas; encontré {df.shape[1]}.")
    # Quedate solo con las primeras dos columnas; el resto se ignora.
    df = df.iloc[:, :2].copy()
    # Convertí todo a numérico; las celdas no numéricas dejan NaN y las saco.
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df = df.dropna()
    if df.empty:
        raise ValueError("La tabla quedó vacía después de descartar filas no numéricas.")
    return df


def _parse_bilinear_csv(csv_text: str, *, decimal_comma: bool) -> pd.DataFrame:
    df = _parse_csv_text(csv_text, decimal_comma=decimal_comma)
    if df.shape[1] < 2 or df.shape[0] < 1:
        raise ValueError(
            f"La tabla bilineal necesita al menos 1 fila y 2 columnas "
            f"(una de índice + al menos una de y). Recibí shape {df.shape}."
        )
    index_header = str(df.columns[0])
    df = df.set_index(df.columns[0])
    # Convertí columnas (ys) e índice (xs) a numérico; el cuerpo también.
    try:
        df.columns = pd.to_numeric(df.columns, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Los headers de las columnas (los ys) deben ser numéricos. Detalle: {exc}"
        ) from exc
    try:
        df.index = pd.to_numeric(df.index, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"La primera columna (los xs) debe ser numérica. Detalle: {exc}") from exc
    df = df.astype(float)
    df.index.name = index_header
    return df


# ---------------------------------------------------------------------
# Helpers de comparación con CoolProp
# ---------------------------------------------------------------------


def _coolprop_for_linear_saturation(
    *,
    indep_base: str,
    indep_unit: str,
    indep_value: float,
    dep_base: str,
    dep_unit: str,
    fluid: str,
) -> tuple[float, str] | None:
    """Para tabla de saturación: devuelve (valor_en_unidad_dependiente, label_dep).

    Soporta indep = T [°C/K] y dep ∈ {hf, hg, sf, sg}.
    """
    if indep_base != "t":
        return None
    if dep_base not in _SAT_PROPS:
        return None
    out_sym, q, dep_label = _SAT_PROPS[dep_base]

    t_K = _t_to_kelvin(indep_value, indep_unit)
    if t_K is None:
        return None

    sp = state_from_pair(fluid, "TX", t=t_K, x=q)
    si_value = sp.h_J_per_kg if out_sym == "H" else sp.s_J_per_kg_K
    converted = _specific_from_si(si_value, dep_unit)
    if converted is None:
        return None
    return converted, dep_label


def _coolprop_for_bilinear_superheated(
    *,
    t_value: float,
    t_unit: str,
    p_value: float,
    p_unit: str,
    z_kind: str,
    z_unit: str,
    fluid: str,
) -> float | None:
    """Para tabla T×P → h ó s. Devuelve valor en la unidad de z, o ``None``."""
    t_K = _t_to_kelvin(t_value, t_unit)
    p_Pa = _p_to_pascal(p_value, p_unit)
    if t_K is None or p_Pa is None:
        return None

    sp = state_from_pair(fluid, "TP", t=t_K, p=p_Pa)
    si_value = sp.h_J_per_kg if z_kind == "h" else sp.s_J_per_kg_K
    return _specific_from_si(si_value, z_unit)


# ---------------------------------------------------------------------
# Bloques de UI reutilizables
# ---------------------------------------------------------------------


def _render_procedure(result: InterpolationResult) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.markdown("**Fórmula aplicada:**")
        st.latex(result.steps.formula_latex)
        st.markdown("**Con los valores de la tabla:**")
        st.latex(result.steps.substituted_latex)
        st.markdown("**En palabras:**")
        st.write(result.steps.narrative_es)


def _table_input(*, tab_key: str, example_csv: str) -> tuple[str, bool] | None:
    """Devuelve ``(csv_text, decimal_comma)`` o ``None`` si todavía no hay tabla."""
    source = st.radio(
        "Origen de la tabla",
        ("Pegar CSV", "Subir archivo"),
        horizontal=True,
        key=f"{tab_key}_source",
    )
    decimal_comma = st.checkbox(
        "Mi tabla usa coma como separador decimal (formato local AR)",
        value=False,
        key=f"{tab_key}_decimal",
    )
    if source == "Pegar CSV":
        csv_text = st.text_area(
            "Pegá el CSV acá",
            value=example_csv,
            height=180,
            key=f"{tab_key}_text",
        )
        if not csv_text.strip():
            return None
        return csv_text, decimal_comma

    upload = st.file_uploader("Subí un archivo CSV", type=["csv", "txt"], key=f"{tab_key}_upload")
    if upload is None:
        return None
    try:
        csv_text = upload.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        csv_text = upload.getvalue().decode("latin-1")
    return csv_text, decimal_comma


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.subheader("Tecnología del Calor")
st.title("📈 Interpolación")
st.markdown(
    "Interpolación lineal simple sobre tablas de dos columnas y bilineal "
    "(doble entrada) sobre tablas 2D, con procedimiento didáctico y "
    "comparación opcional contra CoolProp."
)
st.markdown("---")

st.sidebar.write("Desarrollado por Pablo M. Barral para **Tecnología del Calor**.")
st.sidebar.write("Versión: 0.3.0 — Fase 1.1.")
st.sidebar.write("Contacto: pbarral@fi.uba.ar.")
st.sidebar.write("Powered by CoolProp.")
st.sidebar.markdown("[Readme.md](https://github.com/PabloMBarral/apps/blob/main/README.md)")

with st.expander("📋 Cómo formatear tu tabla", expanded=False):
    st.markdown(
        """
        **Lineal simple (dos columnas)**

        - Primera fila: headers con nombre y unidad opcional entre corchetes.
        - Primera columna: la variable independiente (``x``).
        - Segunda columna: la variable dependiente (``y``).

        Ejemplo:
        ```
        T [°C], h_f [kJ/kg]
        100, 419.06
        110, 461.42
        120, 503.81
        ```

        ---

        **Doble entrada (tabla 2D)**

        - Primera celda (esquina superior-izquierda): puede quedar vacía o
          tener un label (ej. ``T [°C]``); se usa como nombre del eje x.
        - Resto de la primera fila: los valores de la **segunda variable
          independiente** (``ys``), numéricos.
        - Resto de la primera columna: los valores de la **primera
          variable independiente** (``xs``), numéricos.
        - Celdas internas: los valores de ``z(xs[i], ys[j])``.

        Ejemplo (vapor sobrecalentado, T en °C, P en bar, h en kJ/kg):
        ```
        T [°C], 1.0, 3.0, 5.0, 10.0
        200, 2875.5, 2865.6, 2855.8, 2828.3
        250, 2974.5, 2967.7, 2960.7, 2942.6
        300, 3074.5, 3069.6, 3064.6, 3051.7
        ```

        **Convenciones**

        - Separador: coma (``,``) o punto y coma (``;``). El parser los
          detecta automáticamente.
        - Decimal: punto (``419.06``) por defecto. Marcá la opción
          "coma decimal" si tu tabla usa formato local (``419,06``).
        - ``xs`` (y ``ys`` en bilineal) deben estar **estrictamente
          ordenados de menor a mayor**.
        - Extrapolación: prohibida por defecto; activala con el checkbox
          si necesitás consultar fuera del rango de la tabla.

        **Comparación con CoolProp**

        Se ofrece automáticamente cuando los headers se reconocen como
        tablas didácticas clásicas:

        - **Lineal**: tabla de saturación. Columna x es ``T``; columna y
          es ``h_f``, ``h_g``, ``s_f`` o ``s_g``.
        - **Bilineal**: tabla de vapor sobrecalentado. Eje x es ``T``;
          el eje y representa la presión ``P`` (lo confirmás vos).
        """
    )

tab_linear, tab_bilinear = st.tabs(["📈 Lineal simple", "🔢 Doble entrada"])


# =====================================================================
# Tab 1 — Lineal simple
# =====================================================================
with tab_linear:
    st.markdown("### Lineal simple")
    table_input = _table_input(tab_key="lin", example_csv=EXAMPLE_LINEAR_CSV)

    if table_input is not None:
        csv_text, decimal_comma = table_input
        try:
            df = _parse_linear_csv(csv_text, decimal_comma=decimal_comma)
        except ValueError as exc:
            st.error(f"Error de parsing: {exc}")
        else:
            st.markdown("**Tabla cargada:**")
            st.dataframe(df, use_container_width=True)

            xs = df.iloc[:, 0].to_numpy(dtype=float)
            ys = df.iloc[:, 1].to_numpy(dtype=float)
            x_header = str(df.columns[0])
            y_header = str(df.columns[1])
            x_base, x_unit = _parse_header(x_header)
            y_base, y_unit = _parse_header(y_header)

            allow_extra = st.checkbox(
                "Permitir extrapolación fuera del rango de la tabla",
                value=False,
                key="lin_extrap",
            )
            x_default = float((xs.min() + xs.max()) / 2)
            if allow_extra:
                x_query = st.number_input(
                    f"Valor de {x_header}",
                    value=x_default,
                    format="%.6g",
                    key="lin_x",
                )
            else:
                x_query = st.number_input(
                    f"Valor de {x_header} (dentro del rango [{xs.min():g}, {xs.max():g}])",
                    value=x_default,
                    min_value=float(xs.min()),
                    max_value=float(xs.max()),
                    format="%.6g",
                    key="lin_x",
                )

            if st.button("Interpolar", key="lin_btn"):
                try:
                    result = linear_from_table(x_query, xs, ys, allow_extrapolation=allow_extra)
                except ValueError as exc:
                    st.error(f"Error en la interpolación: {exc}")
                else:
                    st.success(f"**Resultado:** {y_header} = {result.value:.6g}")
                    _render_procedure(result)

                    # Comparación contra CoolProp (sólo tablas de saturación).
                    if x_base == "t" and y_base in _SAT_PROPS:
                        with st.expander("🆚 Comparar con CoolProp", expanded=True):
                            fluid = st.selectbox(
                                "Fluido",
                                SUPPORTED_FLUIDS,
                                index=0,
                                key="lin_fluid",
                            )
                            try:
                                pair = _coolprop_for_linear_saturation(
                                    indep_base=x_base,
                                    indep_unit=x_unit,
                                    indep_value=x_query,
                                    dep_base=y_base,
                                    dep_unit=y_unit,
                                    fluid=fluid,
                                )
                            except ValueError as exc:
                                st.warning(f"CoolProp no pudo resolver el estado: {exc}")
                                pair = None
                            if pair is None:
                                st.info(
                                    "No se pudo armar la comparación con las "
                                    "unidades detectadas. Revisá el header."
                                )
                            else:
                                cp_value, dep_label = pair
                                err_pct = (
                                    abs(result.value - cp_value) / abs(cp_value) * 100.0
                                    if cp_value
                                    else float("nan")
                                )
                                c1, c2, c3 = st.columns(3)
                                interp_help = (
                                    f"Valor de {dep_label} obtenido por interpolación lineal."
                                )
                                cp_help = (
                                    f"Valor de {dep_label} calculado con la ecuación de estado."
                                )
                                c1.metric(
                                    "Interpolación",
                                    f"{result.value:.6g}",
                                    help=interp_help,
                                )
                                c2.metric(
                                    f"CoolProp ({fluid})",
                                    f"{cp_value:.6g}",
                                    help=cp_help,
                                )
                                c3.metric(
                                    "Error relativo",
                                    f"{err_pct:.3f} %",
                                )
                                st.caption(
                                    "El error muestra cuánto se aleja la "
                                    "interpolación lineal del valor 'exacto' "
                                    "que devuelve CoolProp."
                                )


# =====================================================================
# Tab 2 — Doble entrada (bilineal)
# =====================================================================
with tab_bilinear:
    st.markdown("### Doble entrada (bilineal)")
    table_input = _table_input(tab_key="bil", example_csv=EXAMPLE_BILINEAR_CSV)

    if table_input is not None:
        csv_text, decimal_comma = table_input
        try:
            df = _parse_bilinear_csv(csv_text, decimal_comma=decimal_comma)
        except ValueError as exc:
            st.error(f"Error de parsing: {exc}")
        else:
            st.markdown("**Tabla cargada:**")
            st.dataframe(df, use_container_width=True)

            xs = df.index.to_numpy(dtype=float)
            ys = df.columns.to_numpy(dtype=float)
            zs = df.to_numpy(dtype=float)
            x_header = str(df.index.name) if df.index.name else "x"
            x_base, x_unit = _parse_header(x_header)

            allow_extra = st.checkbox(
                "Permitir extrapolación fuera del rango de la tabla",
                value=False,
                key="bil_extrap",
            )
            x_default = float((xs.min() + xs.max()) / 2)
            y_default = float((ys.min() + ys.max()) / 2)
            col_x, col_y = st.columns(2)
            with col_x:
                if allow_extra:
                    x_query = st.number_input(
                        f"Valor de {x_header}",
                        value=x_default,
                        format="%.6g",
                        key="bil_x",
                    )
                else:
                    x_query = st.number_input(
                        f"{x_header} ∈ [{xs.min():g}, {xs.max():g}]",
                        value=x_default,
                        min_value=float(xs.min()),
                        max_value=float(xs.max()),
                        format="%.6g",
                        key="bil_x",
                    )
            with col_y:
                if allow_extra:
                    y_query = st.number_input(
                        "Valor de y",
                        value=y_default,
                        format="%.6g",
                        key="bil_y",
                    )
                else:
                    y_query = st.number_input(
                        f"y ∈ [{ys.min():g}, {ys.max():g}]",
                        value=y_default,
                        min_value=float(ys.min()),
                        max_value=float(ys.max()),
                        format="%.6g",
                        key="bil_y",
                    )

            if st.button("Interpolar", key="bil_btn"):
                try:
                    result = bilinear_from_table(
                        x_query,
                        y_query,
                        xs,
                        ys,
                        zs,
                        allow_extrapolation=allow_extra,
                    )
                except ValueError as exc:
                    st.error(f"Error en la interpolación: {exc}")
                else:
                    st.success(f"**Resultado:** z = {result.value:.6g}")
                    _render_procedure(result)

                    # Comparación CoolProp solo si el eje x es T.
                    if x_base == "t":
                        with st.expander("🆚 Comparar con CoolProp", expanded=True):
                            st.caption(
                                "Decile a CoolProp qué representan los ejes y "
                                "el cuerpo de la tabla. Solo se soporta el "
                                "caso T×P → {h, s} para Fase 1.1."
                            )
                            c1, c2 = st.columns(2)
                            with c1:
                                p_unit_label = st.selectbox(
                                    "Unidad de y (P)",
                                    ["bar", "Pa", "kPa", "MPa"],
                                    index=0,
                                    key="bil_p_unit",
                                )
                            with c2:
                                z_choice = st.selectbox(
                                    "z representa",
                                    [
                                        "h [kJ/kg]",
                                        "h [J/kg]",
                                        "s [kJ/(kg·K)]",
                                        "s [J/(kg·K)]",
                                    ],
                                    index=0,
                                    key="bil_z_kind",
                                )
                            fluid = st.selectbox(
                                "Fluido",
                                SUPPORTED_FLUIDS,
                                index=0,
                                key="bil_fluid",
                            )
                            z_base, z_unit = _parse_header(z_choice)
                            try:
                                cp_value = _coolprop_for_bilinear_superheated(
                                    t_value=x_query,
                                    t_unit=x_unit,
                                    p_value=y_query,
                                    p_unit=p_unit_label,
                                    z_kind=z_base,
                                    z_unit=z_unit,
                                    fluid=fluid,
                                )
                            except ValueError as exc:
                                st.warning(f"CoolProp no pudo resolver el estado: {exc}")
                                cp_value = None
                            if cp_value is None:
                                st.info(
                                    "No se pudo armar la comparación con las unidades elegidas."
                                )
                            else:
                                err_pct = (
                                    abs(result.value - cp_value) / abs(cp_value) * 100.0
                                    if cp_value
                                    else float("nan")
                                )
                                c1, c2, c3 = st.columns(3)
                                c1.metric(
                                    "Interpolación bilineal",
                                    f"{result.value:.6g}",
                                )
                                c2.metric(
                                    f"CoolProp ({fluid})",
                                    f"{cp_value:.6g}",
                                )
                                c3.metric(
                                    "Error relativo",
                                    f"{err_pct:.3f} %",
                                )
                                st.caption(
                                    "El error muestra cuánto se aleja la "
                                    "interpolación bilineal del valor 'exacto' "
                                    "que devuelve CoolProp. Suele ser mayor "
                                    "que el lineal porque tablas 2D son "
                                    "típicamente menos densas."
                                )
