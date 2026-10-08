"""Entry point — TA216 Apps.

Define la navegación explícita con :func:`st.navigation` y :class:`st.Page`
(API introducida en Streamlit 1.36). Las páginas viven en ``app_pages/`` y
se referencian por path con un label e ícono custom.

La carpeta **no** se llama ``pages/`` a propósito: si existe una carpeta
``pages/`` junto a este script, Streamlit arranca en el modo multipágina
viejo (una bandera global del proceso) hasta que este script llama a
``st.navigation``. Así, si el primer visitante después de un reinicio
entraba por un link directo (p. ej. ``/Propiedades``), la página corría
sin pasar por acá y el menú mostraba los nombres de archivo
("streamlit app", "Interpolacion" sin tilde) hasta que alguien entrara
por la raíz.
"""

from __future__ import annotations

import streamlit as st

from ui.branding import SUBJECT, sidebar_credits
from ui.units_ui import render_units_selector

PAGE_VERSION = "0.21.0"


def _home_page() -> None:
    st.set_page_config(
        page_title="TA216 Apps",
        page_icon="🏠",
        layout="centered",
    )
    st.subheader("Facultad de Ingeniería — UBA")
    st.title("🏠 TA216 — Apps")
    st.markdown("---")

    st.markdown(
        f"""
        Suite de herramientas didácticas de ingeniería térmica para la
        materia **{SUBJECT}** (FIUBA).

        👈 Elegí una página del menú lateral para empezar.

        Las **fórmulas teóricas** que sustentan estos cálculos viven en el
        repo hermano
        [**vademecum-termo**](https://github.com/PabloMBarral/vademecum-termo).
        Pensalos como un único material: el vademecum explica, este repo
        calcula.

        ---

        ### Módulos disponibles

        - 💧 **Propiedades** — estado termodinámico del agua y otros
          fluidos (refrigerantes como R-134a, R-410A, R-32, amoníaco, CO₂,
          propano o isobutano, aire y fluidos de ORC) a
          partir de cualquier par de propiedades independientes (T-p, p-x,
          T-x, p-h, p-s, T-s, h-s, T-v, p-v, p-u). Región, todas las
          propiedades (incluidas las de transporte), tablas de saturación,
          el procedimiento para resolverlo con las tablas y una tabla de
          estados para armar ciclos sobre el diagrama.
        - 📐 **Interpolación** — lineal simple y doble entrada (bilineal)
          sobre tablas, con procedimiento didáctico paso a paso y
          comparación opcional contra CoolProp.
        - ⚙️ **Isoentrópicos** — turbina, compresor, bomba y compresor
          multietapa con intercooler, en modo directo (calcular el
          estado real dado η_s) o inverso (calcular η_s dados los dos
          estados). Con **diagramas T–s / log p–h / h–s** del proceso.
        - 🔥 **ISO 6976** — poder calorífico (bruto y neto), densidad,
          densidad relativa al aire e índice de Wobbe de mezclas
          gaseosas combustibles, con propagación de incertidumbre y los
          ejemplos del Annex D precargados (matriz identidad; matriz
          de normalización deferida a fase futura por requerir
          ISO 14912:2003).
        - ♨️ **Rankine** — ciclo de potencia de vapor simple, con
          recalentamiento o con regeneración (calentadores abiertos y
          cerrados, con subenfriador de drenaje y desrecalentador), ideal
          o real (con pérdidas de carga y de calor), con agua o con un
          fluido orgánico (ORC, con recuperador), resuelto con
          [TESPy](https://tespy.readthedocs.io): estados, fracciones de
          extracción, ciclo sobre el diagrama, procedimiento con las tablas,
          agua de enfriamiento y barridos para ver cómo mejora el
          rendimiento (ejemplos de Cengel cap. 10).
        - ❄️ **Refrigeración** — ciclo de compresión de vapor simple, de
          dos etapas con cámara de evaporación instantánea o en cascada
          (con uno o dos refrigerantes), ideal o real, como refrigerador o
          bomba de calor: COP, estados, ciclo sobre el diagrama log p–h,
          procedimiento con las tablas, exergía destruida en cada
          componente y barridos (ejemplos de Cengel cap. 11).
        - 🌫️ **Psicrometría** — aire húmedo con el modelo del vademecum
          §14: el estado con dos datos cualquiera (T, φ, bulbo húmedo,
          punto de rocío, ω o h) a nivel del mar, por altura o a otra
          presión, con todas sus propiedades, la exergía y la comparación
          con el modelo de gas real de CoolProp; una **carta psicrométrica
          interactiva** (tocás un punto y lo carga); un tren de **procesos
          de acondicionamiento** (calentar, enfriar y secar, humidificar,
          mezclar) con el calor, el agua y la exergía destruida en cada
          uno, y la **torre de enfriamiento** (ejemplos de Cengel cap. 14).
        - 🕯️ **Combustión** — un combustible gaseoso (gas natural, GLP,
          hidrógeno, biogás o una mezcla a medida), líquido (octano,
          alcoholes, querosén, propano líquido) o sólido por su análisis
          elemental (carbón, bagazo, fueloil) con aire técnico, aire seco u
          oxígeno, húmedo y precalentado: el aire (λ, exceso, AC o el O₂
          medido en los humos), los **humos** en base húmeda y seca, los
          **puntos de rocío** del agua y ácido, el PCS y el PCI, la
          **temperatura adiabática de llama** con combustión completa y con
          **disociación** (equilibrio químico, con los polinomios NASA), el
          **calor** con los humos a una temperatura dada (con condensación y
          rendimiento) y la exergía destruida; y al revés, el **análisis de
          humos** (el O₂ de un analizador o un Orsat) con el diagrama de
          combustión (ejemplos de Cengel cap. 15).
        - 🌀 **Brayton** — turbina de gas con las tres mejoras del ciclo
          Brayton: compresión en etapas con interenfriamiento, expansión en
          etapas con recalentamiento (con metano, una segunda cámara de
          combustión) y regenerador, con combustión de metano o aire
          estándar: rendimiento, trabajo y relación de trabajo de
          retroceso, diagrama T–s, **cuánto suma cada mejora**, la
          **exergía destruida en cada componente**, procedimiento con las
          tablas de gas ideal y barridos (ejemplos de Cengel cap. 9).
        - 🏭 **HRSG** — caldera de recuperación de una, dos o tres
          presiones: con los gases (temperatura, caudal y composición, o el
          escape de una turbina de gas a metano con exceso de aire λ), el
          agua de alimentación y, para cada nivel, la presión, el pinch, el
          approach y el vapor sobrecalentado o saturado, arma el
          **diagrama T–Q**: caudales de vapor, temperaturas de los gases
          entre secciones y de chimenea, punto de rocío, la **exergía**
          (cuánto gana el agua, cuánto se destruye en cada sección y cuánto
          se va por la chimenea), la comparación entre 1, 2 y 3 presiones,
          procedimiento y barridos.
        - ⚡ **Ciclo combinado** — turbina de gas (compresor, cámara de
          combustión con metano o aire estándar y turbina, con calores
          específicos variables y TESPy como control) + caldera de
          recuperación de una, dos o tres presiones, con recalentamiento
          si querés + ciclo de vapor con admisiones y desaireador:
          rendimiento de cada parte y del conjunto, a dónde va la energía
          (diagrama de Sankey), diagramas T–s y T–Q, **cuánto ganás con
          más presiones y recalentamiento** (con la regla de Baumann para
          la humedad), la **exergía del ciclo de fondo**, procedimiento y
          barridos (ejemplos de Cengel cap. 9 y 10 y de Kehlhofer cap. 5).
        - 📈 **Diagramas** — Propiedades, Isoentrópicos, Rankine,
          Refrigeración y Ciclo combinado incluyen
          gráficos interactivos con isolíneas (log p–h, T–s, h–s,
          p–log v) y overlay del estado / proceso calculado, vía
          [fluprodia](https://github.com/fwitte/fluprodia).

        Cada página trae las fórmulas del vademecum (📖 *Fórmulas
        teóricas*), el procedimiento con los valores reemplazados
        (🔬 *Procedimiento*) y la descarga de resultados en CSV / JSON.
        Está pensada para usarse también desde el celular.

        ### En desarrollo

        Poder calorífico por correlaciones, exergía química y de toda la
        planta. Ver el
        [README](https://github.com/PabloMBarral/apps#m%C3%B3dulos) para
        el roadmap completo.

        ---

        ### Sistema de unidades

        En el sidebar (después de los créditos; en el celular se abre con
        la flecha **»** de arriba a la izquierda) podés elegir entre **SI**
        (K, Pa, J/kg, J/(kg·K)), **Técnico** (°C, bar, kJ/kg, kJ/(kg·K))
        — default, alineado con Cengel — o **Inglés** (°F, psia, Btu/lb,
        Btu/(lb·°R)). La selección persiste entre páginas y aplica a
        Propiedades, Isoentrópicos, Rankine, Refrigeración, Psicrometría,
        Combustión, Brayton, HRSG y Ciclo combinado.
        Interpolación respeta las unidades del CSV original; ISO 6976 usa
        las unidades de la norma.
        """
    )

    sidebar_credits(version=PAGE_VERSION, page_name="Home")
    render_units_selector()


pages = [
    st.Page(_home_page, title="Home", icon="🏠", default=True),
    st.Page("app_pages/1_Propiedades.py", title="Propiedades", icon="💧"),
    st.Page("app_pages/2_Interpolacion.py", title="Interpolación", icon="📐"),
    st.Page("app_pages/3_Isoentropicos.py", title="Isoentrópicos", icon="⚙️"),
    st.Page("app_pages/4_ISO6976.py", title="ISO 6976", icon="🔥"),
    st.Page("app_pages/5_Rankine.py", title="Rankine", icon="♨️"),
    st.Page("app_pages/6_Refrigeracion.py", title="Refrigeración", icon="❄️"),
    st.Page("app_pages/7_Psicrometria.py", title="Psicrometría", icon="🌫️"),
    st.Page("app_pages/8_Combustion.py", title="Combustión", icon="🕯️"),
    st.Page("app_pages/13_Brayton.py", title="Brayton", icon="🌀"),
    st.Page("app_pages/10_HRSG.py", title="HRSG", icon="🏭"),
    st.Page("app_pages/12_Ciclo_Combinado.py", title="Ciclo combinado", icon="⚡"),
]
pg = st.navigation(pages)
pg.run()
