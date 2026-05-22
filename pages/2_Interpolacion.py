"""Página 2 — Interpolación lineal simple y de doble entrada.

Dos pestañas:

- **Lineal simple**: tabla de dos columnas pegada o subida como CSV;
  ``text_area`` define los headers (estructura), ``st.data_editor``
  permite editar valores in-place.
- **Doble entrada (bilineal)**: tabla 2D con xs como índice y ys como
  columnas numéricas. ``text_area`` define la estructura completa
  (headers, ys y xs); ``st.data_editor`` solo permite editar el cuerpo
  numérico.

Para cada modo se muestra el resultado, el procedimiento didáctico
(LaTeX + narrativa en español) y, **bajo botón explícito**, una
comparación contra CoolProp con autodetección de tipo de tabla.

TODO (próximas fases, ver CLAUDE.md §"Páginas Streamlit"):
- Expansor `📖 Fórmulas teóricas` con link al apartado correspondiente
  de vademecum-termo.
- Botón de exportar resultados (CSV / JSON).
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
import streamlit as st

from core.fluids import SUPPORTED_FLUIDS, state_from_pair
from core.interpolation import (
    InterpolationResult,
    bilinear_from_table,
    linear_from_table,
)
from core.units import (
    parse_header,
    pressure_to_pascal,
    specific_from_si,
    temperature_to_kelvin,
)
from ui.branding import SUBJECT, sidebar_credits

PAGE_VERSION = "0.6.0"

EXAMPLE_LINEAR_CSV = """T [°C],h_f [kJ/kg]
100,419.06
110,461.42
120,503.81
130,546.38"""

# Bilineal: (0,0) vacía, primera fila = ys (P [bar]), primera col = xs (T [°C]).
# Cuerpo = h [kJ/kg]. El alumno puede sobreescribir (0,0) con "T [°C]" si
# quiere que la comparación CoolProp detecte la unidad automáticamente.
EXAMPLE_BILINEAR_CSV = """,1.0,3.0,5.0,10.0
200,2875.5,2865.6,2855.8,2828.3
250,2974.5,2967.7,2960.7,2942.6
300,3074.5,3069.6,3064.6,3051.7
350,3175.6,3171.7,3167.8,3157.7"""

# Propiedad saturada autodetectada en headers de tabla lineal →
# (símbolo CoolProp, valor de Q, label legible).
_SAT_PROPS: dict[str, tuple[str, float, str]] = {
    "hf": ("H", 0.0, "h_f"),
    "hg": ("H", 1.0, "h_g"),
    "sf": ("S", 0.0, "s_f"),
    "sg": ("S", 1.0, "s_g"),
}


# ---------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------


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
    df = df.iloc[:, :2].copy()
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
    first_col = df.columns[0]
    # pandas asigna "Unnamed: 0" cuando la celda (0,0) está vacía.
    index_header = "" if str(first_col).startswith("Unnamed:") else str(first_col)
    df = df.set_index(first_col)
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
# CoolProp helpers — usan los normalizadores de core.units
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
    """Comparación contra CoolProp para tabla de saturación T → hf/hg/sf/sg."""
    if indep_base != "t":
        return None
    if dep_base not in _SAT_PROPS:
        return None
    out_sym, q, dep_label = _SAT_PROPS[dep_base]

    t_K = temperature_to_kelvin(indep_value, indep_unit)
    if t_K is None:
        return None

    sp = state_from_pair(fluid, "TX", t=t_K, x=q)
    si_value = sp.h_J_per_kg if out_sym == "H" else sp.s_J_per_kg_K
    converted = specific_from_si(si_value, dep_unit)
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
    """Comparación contra CoolProp para tabla T×P → h ó s."""
    t_K = temperature_to_kelvin(t_value, t_unit)
    p_Pa = pressure_to_pascal(p_value, p_unit)
    if t_K is None or p_Pa is None:
        return None

    sp = state_from_pair(fluid, "TP", t=t_K, p=p_Pa)
    si_value = sp.h_J_per_kg if z_kind == "h" else sp.s_J_per_kg_K
    return specific_from_si(si_value, z_unit)


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------


def _render_procedure(result: InterpolationResult) -> None:
    with st.expander("🔬 Procedimiento", expanded=True):
        st.markdown("**Fórmula aplicada:**")
        st.latex(result.steps.formula_latex)
        st.markdown("**Con los valores de la tabla:**")
        st.latex(result.steps.substituted_latex)
        st.markdown("**En palabras:**")
        st.write(result.steps.narrative_es)


def _render_header_help() -> None:
    with st.expander("ℹ️ Cómo formatear headers", expanded=False):
        st.markdown(
            """
            Cada header se interpreta como **símbolo + unidad opcional
            entre corchetes o paréntesis**. Las unidades aceptadas son
            (case-insensitive, tolera espacios y notación variada):

            **Temperatura**: `°C`, `ºC`, `C`, `Celsius`, `K`, `Kelvin`.

            **Presión**: `bar`, `bar(a)`, `Pa`, `kPa`, `MPa`.

            **Entalpía / energía específica**: `J/kg`, `kJ/kg`,
            `J kg^-1`, `kJ kg^-1`.

            **Entropía / calor específico**: `J/(kg·K)`, `kJ/(kg·K)`,
            `J/(kg K)`, `J/(kg.K)`, `J/kg-K`, `J/kg/K`,
            `J kg^-1 K^-1`, y las variantes con `kJ`.

            Ejemplos válidos: `T [°C]`, `T [K]`, `P [bar]`, `h_f [kJ/kg]`,
            `s [kJ/(kg·K)]`, `s [kJ/kg-K]`, `h [kJ kg^-1]`.
            """
        )


def _table_input(*, tab_key: str, example_csv: str) -> tuple[str, bool] | None:
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
            "Pegá el CSV acá (define la estructura: headers, xs y ys)",
            value=example_csv,
            height=200,
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


def _clean_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return cleaned.dropna()


def _format_metric(value: float | None, *, fmt: str = "%.6g") -> str:
    if value is None:
        return "—"
    return fmt % value


def _safe_relative_error(interp: float, exact: float) -> float | None:
    if exact == 0:
        return None
    return abs(interp - exact) / abs(exact) * 100.0


def _clear_result_state(prefix: str) -> None:
    for k in (
        f"{prefix}_result",
        f"{prefix}_compare_active",
        f"{prefix}_x_query",
        f"{prefix}_y_query",
        f"{prefix}_x_header",
    ):
        st.session_state.pop(k, None)


def _render_linear_comparison_panel(
    *,
    result: InterpolationResult,
    x_query: float,
    x_base: str,
    x_unit: str,
    y_base: str,
    y_unit: str,
) -> None:
    st.markdown("#### 🆚 Comparación con CoolProp")
    fluid = st.selectbox("Fluido", SUPPORTED_FLUIDS, index=0, key="lin_fluid")
    try:
        pair: Any = _coolprop_for_linear_saturation(
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
            "No se pudo armar la comparación con las unidades detectadas. "
            "Revisá el header (`T` o `t` + `h_f`/`h_g`/`s_f`/`s_g`)."
        )
        return
    cp_value, dep_label = pair
    err_pct = _safe_relative_error(result.value, cp_value)
    c1, c2, c3 = st.columns(3)
    interp_help = f"Valor de {dep_label} por interpolación lineal."
    cp_help = f"Valor de {dep_label} con la ecuación de estado de CoolProp."
    c1.metric("Interpolación", f"{result.value:.6g}", help=interp_help)
    c2.metric(f"CoolProp ({fluid})", f"{cp_value:.6g}", help=cp_help)
    c3.metric("Error relativo", _format_metric(err_pct, fmt="%.3f %%"))
    st.caption(
        "El error muestra cuánto se aleja la interpolación lineal del "
        "valor 'exacto' que devuelve CoolProp."
    )


def _render_bilinear_comparison_panel(
    *,
    result: InterpolationResult,
    x_query: float,
    y_query: float,
    x_unit_detected: str,
    y_unit_detected: str = "",
) -> None:
    st.markdown("#### 🆚 Comparación con CoolProp")
    st.caption(
        "Decile a CoolProp qué representan los ejes y el cuerpo. Por ahora "
        "solo se soporta el caso T × P → {h, s}."
    )

    # Unidad de T: usa la autodetectada del header de filas. Si vino vacía,
    # default °C con opción a K.
    if x_unit_detected:
        t_unit_label = x_unit_detected
        st.caption(f"Unidad de T detectada del header: `{t_unit_label}`.")
    else:
        t_unit_label = st.selectbox(
            "Unidad de las filas (T)",
            ["°C", "K"],
            index=0,
            key="bil_t_unit",
        )

    # Unidad de P: usa la autodetectada del header de columnas si vino, si no
    # pedila por dropdown.
    if y_unit_detected:
        p_unit_label = y_unit_detected
        st.caption(f"Unidad de P detectada del header: `{p_unit_label}`.")
        col_z = st.container()
    else:
        c1, c2 = st.columns(2)
        with c1:
            p_unit_label = st.selectbox(
                "Unidad de las columnas (P)",
                ["bar", "Pa", "kPa", "MPa"],
                index=0,
                key="bil_p_unit",
            )
        col_z = c2

    with col_z:
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
    fluid = st.selectbox("Fluido", SUPPORTED_FLUIDS, index=0, key="bil_fluid")

    z_base, z_unit = parse_header(z_choice)
    try:
        cp_value = _coolprop_for_bilinear_superheated(
            t_value=x_query,
            t_unit=t_unit_label,
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
        st.info("No se pudo armar la comparación con las unidades elegidas.")
        return

    err_pct = _safe_relative_error(result.value, cp_value)
    c1, c2, c3 = st.columns(3)
    c1.metric("Interpolación bilineal", f"{result.value:.6g}")
    c2.metric(f"CoolProp ({fluid})", f"{cp_value:.6g}")
    c3.metric("Error relativo", _format_metric(err_pct, fmt="%.3f %%"))
    st.caption(
        "El error muestra cuánto se aleja la interpolación bilineal del "
        "valor 'exacto' que devuelve CoolProp. Suele ser mayor que el "
        "lineal porque las tablas 2D son típicamente menos densas."
    )


# ---------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------

st.set_page_config(page_title="Interpolación", page_icon="📐", layout="centered")

st.subheader(SUBJECT)
st.title("📐 Interpolación")
st.markdown(
    "Interpolación lineal simple sobre tablas de dos columnas y bilineal "
    "(doble entrada) sobre tablas 2D, con procedimiento didáctico y "
    "comparación **opcional** contra CoolProp."
)
st.markdown("---")

sidebar_credits(version=PAGE_VERSION, page_name="Interpolación")

with st.expander("📋 Cómo formatear tu tabla", expanded=False):
    st.markdown(
        """
        **Lineal simple (dos columnas)**

        - Primera fila: headers con nombre y unidad opcional entre
          corchetes (ej. `T [°C]`).
        - Primera columna: variable independiente (``x``).
        - Segunda columna: variable dependiente (``y``).

        Ejemplo:
        ```
        T [°C], h_f [kJ/kg]
        100, 419.06
        110, 461.42
        120, 503.81
        ```

        ---

        **Doble entrada (tabla 2D)**

        - Celda (0,0): vacía (o con label del eje x si querés que la
          comparación CoolProp detecte automáticamente la unidad de T).
        - Resto de la primera fila: valores de la segunda variable
          independiente (``ys``), numéricos.
        - Primera columna: valores de la primera variable
          independiente (``xs``), numéricos.
        - Celdas internas: valores de ``z(xs[i], ys[j])``.

        Ejemplo (vapor sobrecalentado, T en °C, P en bar, h en kJ/kg):
        ```
        , 1.0, 3.0, 5.0, 10.0
        200, 2875.5, 2865.6, 2855.8, 2828.3
        250, 2974.5, 2967.7, 2960.7, 2942.6
        300, 3074.5, 3069.6, 3064.6, 3051.7
        ```

        (Filas = T [°C], columnas = P [bar], cuerpo = h [kJ/kg].)

        ---

        **Convenciones**

        - Separador: coma o punto y coma. Auto-detectado.
        - Decimal: punto por defecto. Marcá la opción "coma decimal" si
          tu tabla usa formato local (``419,06``).
        - ``xs`` (y ``ys`` en bilineal) deben estar estrictamente
          ordenados de menor a mayor.
        - Extrapolación: prohibida por defecto; activala con el
          checkbox si necesitás consultar fuera del rango.
        - Después de parsear el CSV, podés editar **los valores del
          cuerpo** in-place en la tabla. Los headers vienen del CSV.
        """
    )

tab_linear, tab_bilinear = st.tabs(["📈 Lineal simple", "🔢 Doble entrada"])


# =====================================================================
# Tab 1 — Lineal simple
# =====================================================================
with tab_linear:
    st.markdown("### Lineal simple")
    _render_header_help()
    table_input = _table_input(tab_key="lin", example_csv=EXAMPLE_LINEAR_CSV)

    if table_input is not None:
        csv_text, decimal_comma = table_input
        try:
            df_parsed = _parse_linear_csv(csv_text, decimal_comma=decimal_comma)
        except ValueError as exc:
            st.error(f"Error de parsing: {exc}")
            _clear_result_state("lin")
        else:
            # text_area = base de la estructura. Si el CSV cambia, reseteamos
            # los overrides de headers, los edits y el resultado.
            csv_signature = str(hash(csv_text) ^ hash(decimal_comma))
            default_x_header_lin = str(df_parsed.columns[0])
            default_y_header_lin = str(df_parsed.columns[1])
            if st.session_state.get("lin_csv_signature") != csv_signature:
                st.session_state["lin_csv_signature"] = csv_signature
                st.session_state["lin_x_header_override"] = default_x_header_lin
                st.session_state["lin_y_header_override"] = default_y_header_lin
                _clear_result_state("lin")

            st.markdown("**Headers (editables — sobrescriben los del CSV):**")
            hcol1, hcol2 = st.columns(2)
            x_header = hcol1.text_input(
                "Header columna 1 (x)",
                key="lin_x_header_override",
                help="Ej.: `T [°C]`. Se usa para la detección de unidades.",
            )
            y_header = hcol2.text_input(
                "Header columna 2 (y)",
                key="lin_y_header_override",
                help="Ej.: `h_f [kJ/kg]`. Define el label del resultado.",
            )

            st.markdown("**Tabla cargada (editá valores in-place):**")
            # Mantenemos el schema (column names) estable y solo cambiamos el
            # label visible vía column_config: así editar el header no resetea
            # los edits del cuerpo en el data_editor.
            edited = st.data_editor(
                df_parsed,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    df_parsed.columns[0]: st.column_config.Column(label=x_header or "x"),
                    df_parsed.columns[1]: st.column_config.Column(label=y_header or "y"),
                },
                key=f"lin_editor_{csv_signature}",
            )
            edited_clean = _clean_numeric_dataframe(edited)
            if edited_clean.empty or edited_clean.shape[0] < 2:
                st.warning(
                    "La tabla quedó con menos de 2 filas numéricas válidas; "
                    "agregá filas o revisá los valores para poder interpolar."
                )
            else:
                xs = edited_clean.iloc[:, 0].to_numpy(dtype=float)
                ys = edited_clean.iloc[:, 1].to_numpy(dtype=float)
                # Los headers vienen de los text_inputs editables, no del CSV
                # original ni de las columnas del data_editor.
                x_base, x_unit = parse_header(x_header)
                y_base, y_unit = parse_header(y_header)

                allow_extra = st.checkbox(
                    "Permitir extrapolación fuera del rango de la tabla",
                    value=False,
                    key="lin_extrap",
                )
                x_default = float((xs.min() + xs.max()) / 2)
                if allow_extra:
                    x_query: float = st.number_input(
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

                if st.button("Interpolar", key="lin_btn", type="primary"):
                    try:
                        result = linear_from_table(x_query, xs, ys, allow_extrapolation=allow_extra)
                    except ValueError as exc:
                        st.error(f"Error en la interpolación: {exc}")
                        _clear_result_state("lin")
                    else:
                        st.session_state["lin_result"] = result
                        st.session_state["lin_x_query"] = x_query

                # Render persistente del resultado (sobrevive a re-renders
                # disparados por widgets de la comparación). FIX del bug
                # Fase 1.1: la comparación ya no vive dentro de `if button:`.
                if "lin_result" in st.session_state:
                    result = st.session_state["lin_result"]
                    st.success(f"**Resultado:** {y_header} = {result.value:.6g}")
                    _render_procedure(result)

                    can_compare = x_base == "t" and y_base in _SAT_PROPS
                    if can_compare:
                        if st.button(
                            "🆚 Comparar con CoolProp",
                            key="lin_compare_btn",
                        ):
                            st.session_state["lin_compare_active"] = True

                        if st.session_state.get("lin_compare_active", False):
                            _render_linear_comparison_panel(
                                result=result,
                                x_query=st.session_state["lin_x_query"],
                                x_base=x_base,
                                x_unit=x_unit,
                                y_base=y_base,
                                y_unit=y_unit,
                            )
                    else:
                        st.caption(
                            "💡 La comparación con CoolProp se ofrece cuando "
                            "el header se reconoce como tabla de saturación "
                            "(``T`` + ``h_f``/``h_g``/``s_f``/``s_g``)."
                        )


# =====================================================================
# Tab 2 — Doble entrada (bilineal)
# =====================================================================
with tab_bilinear:
    st.markdown("### Doble entrada (bilineal)")
    _render_header_help()
    table_input = _table_input(tab_key="bil", example_csv=EXAMPLE_BILINEAR_CSV)

    if table_input is not None:
        csv_text, decimal_comma = table_input
        try:
            df_parsed = _parse_bilinear_csv(csv_text, decimal_comma=decimal_comma)
        except ValueError as exc:
            st.error(f"Error de parsing: {exc}")
            _clear_result_state("bil")
        else:
            csv_signature = str(hash(csv_text) ^ hash(decimal_comma))
            default_x_header_bil = df_parsed.index.name or ""
            if st.session_state.get("bil_csv_signature") != csv_signature:
                st.session_state["bil_csv_signature"] = csv_signature
                st.session_state["bil_x_header_override"] = default_x_header_bil
                # y-axis header no viene del CSV (los ys son numéricos);
                # lo dejamos vacío para que el alumno lo escriba.
                st.session_state["bil_y_header_override"] = ""
                _clear_result_state("bil")

            xs_initial = df_parsed.index.to_numpy(dtype=float)
            ys_initial = df_parsed.columns.to_numpy(dtype=float)
            body_initial = df_parsed.to_numpy(dtype=float)

            st.markdown("**Headers de los ejes (editables):**")
            hcol_x, hcol_y = st.columns(2)
            x_header_raw = hcol_x.text_input(
                "Eje x (filas)",
                key="bil_x_header_override",
                help=(
                    "Ej.: `T [°C]`. Se usa para la detección automática de "
                    "la unidad de temperatura en la comparación CoolProp. "
                    "Sobrescribe la celda (0,0) del CSV."
                ),
            )
            y_header_raw = hcol_y.text_input(
                "Eje y (columnas)",
                key="bil_y_header_override",
                help=(
                    "Ej.: `P [bar]`. El CSV solo trae los valores numéricos "
                    "de los ys; este campo le pone nombre al eje y se usa "
                    "para autodetectar la unidad de presión en la "
                    "comparación CoolProp."
                ),
            )
            x_label_for_editor = x_header_raw if x_header_raw else "x"
            if y_header_raw:
                ys_str = ", ".join(f"{v:g}" for v in ys_initial)
                st.caption(f"Columnas (`{y_header_raw}`): {ys_str}")

            # Schema interno estable: una columna sentinel para xs (con label
            # custom vía column_config) + las ys numéricas del CSV. Eso
            # preserva edits del body cuando el alumno cambia el header.
            _XS_COL = "__xs__"
            flat = pd.DataFrame(body_initial, columns=ys_initial)
            flat.insert(0, _XS_COL, xs_initial)

            st.markdown(
                "**Tabla cargada (editá solo el cuerpo numérico — los `ys` y "
                "los `xs` vienen del CSV):**"
            )
            edited_flat = st.data_editor(
                flat,
                num_rows="fixed",
                use_container_width=True,
                disabled=[_XS_COL],
                column_config={
                    _XS_COL: st.column_config.Column(label=x_label_for_editor),
                },
                key=f"bil_editor_{csv_signature}",
            )

            xs = edited_flat[_XS_COL].to_numpy(dtype=float)
            try:
                body = edited_flat.iloc[:, 1:].astype(float).to_numpy()
                body_ok = True
            except (ValueError, TypeError):
                body_ok = False

            if not body_ok:
                st.error("Hay celdas no numéricas en el cuerpo de la tabla. Revisá los valores.")
            else:
                allow_extra = st.checkbox(
                    "Permitir extrapolación fuera del rango de la tabla",
                    value=False,
                    key="bil_extrap",
                )
                x_default = float((xs.min() + xs.max()) / 2)
                y_default = float((ys_initial.min() + ys_initial.max()) / 2)
                col_x, col_y = st.columns(2)
                with col_x:
                    if allow_extra:
                        x_query_b: float = st.number_input(
                            f"Valor de {x_label_for_editor}",
                            value=x_default,
                            format="%.6g",
                            key="bil_x",
                        )
                    else:
                        x_query_b = st.number_input(
                            f"{x_label_for_editor} ∈ [{xs.min():g}, {xs.max():g}]",
                            value=x_default,
                            min_value=float(xs.min()),
                            max_value=float(xs.max()),
                            format="%.6g",
                            key="bil_x",
                        )
                with col_y:
                    if allow_extra:
                        y_query_b: float = st.number_input(
                            "Valor de y",
                            value=y_default,
                            format="%.6g",
                            key="bil_y",
                        )
                    else:
                        y_query_b = st.number_input(
                            f"y ∈ [{ys_initial.min():g}, {ys_initial.max():g}]",
                            value=y_default,
                            min_value=float(ys_initial.min()),
                            max_value=float(ys_initial.max()),
                            format="%.6g",
                            key="bil_y",
                        )

                if st.button("Interpolar", key="bil_btn", type="primary"):
                    try:
                        result_b = bilinear_from_table(
                            x_query_b,
                            y_query_b,
                            xs,
                            ys_initial,
                            body,
                            allow_extrapolation=allow_extra,
                        )
                    except ValueError as exc:
                        st.error(f"Error en la interpolación: {exc}")
                        _clear_result_state("bil")
                    else:
                        st.session_state["bil_result"] = result_b
                        st.session_state["bil_x_query"] = x_query_b
                        st.session_state["bil_y_query"] = y_query_b
                        st.session_state["bil_x_header"] = x_header_raw
                        st.session_state["bil_y_header"] = y_header_raw

                # Render persistente del resultado y la comparación.
                if "bil_result" in st.session_state:
                    result_b = st.session_state["bil_result"]
                    st.success(f"**Resultado:** z = {result_b.value:.6g}")
                    _render_procedure(result_b)

                    x_base_b, x_unit_b = parse_header(
                        st.session_state.get("bil_x_header", "") or ""
                    )
                    y_base_b, y_unit_b = parse_header(
                        st.session_state.get("bil_y_header", "") or ""
                    )

                    if st.button(
                        "🆚 Comparar con CoolProp",
                        key="bil_compare_btn",
                    ):
                        st.session_state["bil_compare_active"] = True

                    if st.session_state.get("bil_compare_active", False):
                        _render_bilinear_comparison_panel(
                            result=result_b,
                            x_query=st.session_state["bil_x_query"],
                            y_query=st.session_state["bil_y_query"],
                            x_unit_detected=x_unit_b if x_base_b == "t" else "",
                            y_unit_detected=y_unit_b if y_base_b == "p" else "",
                        )
