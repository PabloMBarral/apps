# CLAUDE.md

> Este archivo da contexto al agente Claude Code. Mantenelo actualizado:
> cualquier cambio de arquitectura, dependencia o convención debería
> reflejarse acá.

## Proyecto

Suite de herramientas didácticas de ingeniería térmica para la materia
**TA216 — Tecnología de Calor Avanzada** (FIUBA). Streamlit + CoolProp + TESPy + fluprodia.

Repo hermano de fórmulas teóricas:
[PabloMBarral/vademecum-termo](https://github.com/PabloMBarral/vademecum-termo).
Cada página de la app debe linkear la sección correspondiente del vademecum.

Autor: Pablo M. Barral (pbarral@fi.uba.ar, ORCID 0000-0003-1125-4199).
Licencia: MIT.

## Stack

- Python 3.11+
- **Streamlit** — UI (multipágina con `st.navigation`; páginas en `app_pages/`)
- **CoolProp** — propiedades termofísicas punto a punto
- **TESPy** — simulación de ciclos termodinámicos
- **fluprodia** — diagramas de propiedades de fluidos
- **NumPy, SciPy, pandas, matplotlib** — utilitarios numéricos y plots base
- **pytest** — tests
- **ruff** — lint y format

Toda dependencia nueva debe agregarse a `requirements.txt` Y a
`CITATION.cff` (con autoría/cita correspondiente si es académica).

## Arquitectura

Separación estricta entre lógica de cálculo (testeable, sin Streamlit)
y UI (Streamlit).

```
apps/
├── streamlit_app.py           # Home / landing
├── app_pages/                 # Una página por módulo (numeradas). NO se llama
│                              # pages/: ver «Navegación» más abajo.
│   ├── 1_Propiedades.py       # ✅ Fase 1.6 — Estado completo del agua (y otros
│   │                          # fluidos): región, tablas, procedimiento,
│   │                          # diagrama, tabla de estados para ciclos.
│   ├── 2_Interpolacion.py     # ✅ Fase 1.1 (+ teoría y export, Fase 1.7)
│   ├── 3_Isoentropicos.py     # ✅ Fase 1.3 + 1.5a (diagrama del proceso) +
│   │                          # 1.5b y validaciones / defaults por fluido (1.7)
│   ├── 5_Rankine.py           # ✅ Fase 3.1a — Rankine simple / con
│   │                          # recalentamiento (TESPy), Carnot, barridos de η.
│   │                          # Fase 3.1b: regeneración (hasta 3 calentadores),
│   │                          # rótulos con la numeración del layout, agua de
│   │                          # enfriamiento, barrido de la presión de extracción.
│   ├── 6_Refrigeracion.py
│   ├── 7_Psicrometria.py
│   ├── 8_Combustion.py
│   ├── 9_Poder_Calorifico.py
│   ├── 4_ISO6976.py           # ✅ Fase 2.3 (matriz identidad; teoría y export 1.7)
│   ├── 11_Exergia.py
│   └── 99_Acerca.py           # Créditos, licencias, citas
├── core/                      # Lógica pura, sin dependencia de Streamlit
│   ├── __init__.py
│   ├── units.py               # Conversiones simples + normalizador de
│   │                          # headers (Fase 1.2). NO importa Streamlit.
│   ├── units_system.py        # ✅ Fase 1.4 — Sistema global SI/Técnico/Inglés.
│   │                          # Tabla (kind, system) → (factor, offset, label),
│   │                          # API format_quantity / convert_*_si / unit_label.
│   │                          # Fase 1.6: + ΔT, ρ, velocidad, μ, k, difusividad.
│   │                          # Fase 3.1a: + caudal másico y potencia.
│   ├── fluids.py              # ✅ Fase 1.6 — Wrappers sobre CoolProp:
│   │                          # StatePoint/state_from_pair (cálculo) y
│   │                          # FluidState/fluid_state_from_pair (estado completo:
│   │                          # región, saturación, transporte, validación con
│   │                          # mensajes al alumno, suggested_inputs).
│   ├── state_report.py        # ✅ Fase 1.6 — Presentación pura de un FluidState:
│   │                          # tablas, notas, procedimiento LaTeX por sistema
│   │                          # de unidades, export JSON/CSV, tabla de estados.
│   ├── latex.py               # ✅ Fase 1.7 — latex_number/unit/quantity,
│   │                          # latex_chain (una igualdad por renglón) y
│   │                          # latex_paren (negativos tras un signo).
│   ├── export.py              # ✅ Fase 1.7 — flatten / dict_to_csv: el dict de
│   │                          # un resultado → CSV (campo, valor) y JSON.
│   ├── interpolation.py       # ✅ Fase 1.1 — Interpolación lineal y doble entrada
│   │                          # (+ interpolation_to_dict, Fase 1.7).
│   ├── isentropic.py          # ✅ Fase 1.3 — Turbina / compresor / bomba; multietapa.
│   │                          # Fase 1.7: validaciones al alumno, DeviceDefaults /
│   │                          # suggested_device_inputs, pasos por sistema (1.5b),
│   │                          # tabla de estados y *_to_dict para exportar.
│   ├── diagrams.py            # ✅ Fase 1.5a — Wrappers tipados sobre fluprodia.
│   │                          # FLUPRODIA_UNITS, DEFAULT_RANGES, build_diagram,
│   │                          # isentropic/isobaric/isothermal_process, overlays,
│   │                          # axis_window_si, cycle_overlays (Fase 1.6).
│   ├── exergy.py              # Exergía física y química
│   ├── combustion/
│   │   ├── __init__.py
│   │   ├── fuels.py           # Modelos de combustible (sólido/líquido/gas)
│   │   ├── stoichiometry.py   # Reacciones, exceso aire, productos
│   │   ├── heating_value.py   # PCI/PCS por correlaciones (último/próximo)
│   │   └── iso6976.py         # ✅ Fase 2.3 — ISO 6976:2016 (matriz identidad;
│   │                          # iso6976_to_dict, Fase 1.7)
│   │                          # Normalization matrix: pendiente, requiere
│   │                          # ISO 14912:2003 Formula (69) — deferido.
│   ├── cycles/
│   │   ├── tespy_utils.py     # ✅ Fase 3.1a — new_network() en SI y
│   │   │                      # solve() (status de TESPy → ValueError en castellano).
│   │   ├── rankine.py         # ✅ Fase 3.1a — RankineInputs / solve_rankine /
│   │   │                      # RankineResult, validaciones, barridos y
│   │   │                      # rankine_to_dict. Fase 3.1b: red genérica desde el
│   │   │                      # layout (calentadores), componentes con fracciones,
│   │   │                      # agua de enfriamiento, barrido de extracción.
│   │   ├── rankine_layout.py  # ✅ Fase 3.1b — Topología y numeración tipo Cengel
│   │   │                      # (sin TESPy): FeedwaterHeater, plant_layout,
│   │   │                      # port_flows (caudales 1 − y… simbólicos).
│   │   ├── rankine_procedure.py # ✅ Fase 3.1b — rankine_steps (se mudó; rankine.py
│   │   │                      # lo reexporta): simple, recalentamiento y regenerativo.
│   │   ├── refrigeration.py
│   │   ├── brayton.py
│   │   └── combined.py
│   └── plots.py               # fluprodia + matplotlib helpers
├── ui/                        # Helpers de UI que sí importan Streamlit
│   ├── branding.py            # Bloque de créditos compartido (sidebar)
│   ├── units_ui.py            # ✅ Fase 1.4 — Selector global + number_input_si
│   │                          # (key real f"{key}@{sistema}": el valor físico
│   │                          # sobrevive al cambio de unidades, Fase 1.6).
│   └── diagrams.py            # ✅ Fase 1.5a — Cache de FluidPropertyDiagram
│                              # (@st.cache_resource), render_diagram_plotly
│                              # con overlays de puntos / procesos.
├── tests/                     # pytest: tests/test_<modulo>.py; páginas con
│                              # streamlit.testing (tests/test_page_<pagina>.py)
├── data/                      # Tablas, propiedades por componente, etc.
│   ├── iso6976_components.csv # Valores tabulados por componente puro
│   └── szargut_chemical_exergy.csv
├── requirements.txt
├── CITATION.cff
├── LICENSE
├── README.md
└── CLAUDE.md
```

## Convenciones

### Unidades

- Por defecto: **bar(a)**, **°C**, **kJ/kg**, **kJ/(kg·K)**, título
  adimensional, fracciones másicas/molares como decimales (no porcentajes).
- Cada función pura en `core/` recibe valores en **SI** internamente
  (Pa, K, J/kg). La conversión vive en `core/units.py` y en la UI.
- Hay un selector global de sistema de unidades (SI / técnico / inglés)
  en sidebar; las páginas leen del `st.session_state`.

### Estilo

- **Type hints obligatorios** en todo `core/`.
- Funciones puras, sin estado global. Resultados como `dataclass`
  cuando hay varios valores (`StatePoint`, `CycleResult`, etc.).
- Cache de CoolProp con `@st.cache_data` en los wrappers de Streamlit
  (`app_pages/`) que envuelven funciones de `core.fluids`. `core/` no
  importa Streamlit.
- Idioma de la UI: **español rioplatense**. Los identificadores de
  código en inglés, comentarios y docstrings en español.
- Cada función académicamente relevante incluye en su docstring una
  cita corta a la fuente (libro de texto, paper, norma).

### Navegación

- `streamlit_app.py` arma el menú con `st.navigation` + `st.Page` y es
  el único entry point. Las páginas viven en `app_pages/` y la URL de
  cada una sale del nombre de archivo sin el número (`1_Propiedades.py`
  → `/Propiedades`).
- **No crear una carpeta `pages/`** en la raíz: Streamlit la detecta y
  arranca en el modo multipágina viejo (bandera global del proceso)
  hasta que corre `st.navigation`; los links directos a una página
  después de un reinicio mostraban el menú con nombres de archivo. Un
  test (`tests/test_navigation.py`) lo vigila.

### Páginas Streamlit

Toda página debe tener, mínimo:

1. Título y descripción breve del módulo.
2. Un expansor `📖 Fórmulas teóricas` con link al apartado del vademecum.
3. Inputs validados con rangos razonables y mensajes claros.
4. Resultado principal destacado + tabla con todos los estados.
5. Si aplica, un diagrama con fluprodia / matplotlib.
6. Un expansor `🔬 Procedimiento` con las ecuaciones aplicadas en LaTeX
   y los valores reemplazados (modo didáctico).
7. Botón de exportar resultados (CSV / JSON).

Notas de implementación (aprendidas en la Fase 1.6):

- Guardar el resultado en `st.session_state` y renderizarlo en cada
  corrida: si se muestra solo dentro de `if submit:`, desaparece al tocar
  cualquier otro widget (p. ej. el selector de diagrama).
- Los links al vademecum salen de las constantes `VADEMECUM_*` de
  `ui/branding.py`; citar la sección (§N) además del link.
- Pensar en el celular: tablas con símbolo, valor y unidad primero; la
  unidad de `st.metric` en el rótulo; selectores en el cuerpo de la
  página (el sidebar queda oculto).
- Testear la página con `streamlit.testing.v1.AppTest` (ver
  `tests/test_page_propiedades.py`) y validar el LaTeX nuevo con KaTeX,
  que es el motor de `st.latex`.

Notas de la Fase 1.7:

- **Ancho de las ecuaciones**: KaTeX no parte una ecuación en renglones
  y en un celular de 390 px hay ~324 px útiles (dentro de un expansor).
  Las sustituciones con números se escriben con `core.latex.latex_chain`
  (una igualdad por renglón, `aligned`); si un renglón sigue largo, se
  corta antes de un `+`/`−` con `\\ &\quad +`. Dos ecuaciones en la
  misma línea (`\qquad`) solo si son cortas; si no, van en `st.latex`
  separados o en un `aligned`. Los negativos después de un signo, con
  `latex_paren`. Medir el ancho real renderizando con KaTeX en un
  navegador (p. ej. Chromium con Playwright) en los tres sistemas de
  unidades; en SI, los números con ×10ⁿ J/kg pueden quedar más anchos
  (se deslizan).
- **Isolíneas de fluprodia**: `set_isolines` interpreta los valores en
  las unidades activas del diagrama (`set_unit_system`). Generarlas en
  las unidades de cada sistema (`core.diagrams._isoline_grid(fluid,
  system)`); hasta la 0.9.0 iban siempre en °C/bar y en SI quedaban
  isobaras de 0,01–1000 Pa.
- Un diagrama que falla no debe tumbar la página: envolverlo en
  `try/except` y mostrar `st.warning` (como Propiedades e Isoentrópicos).
- Los valores por defecto de cada página tienen que ser calculables para
  todos los fluidos (`suggested_inputs`, `suggested_device_inputs`) y
  hay tests que los recorren.
- En el estado de referencia de cada fluido (agua: u = s = 0 en el punto
  triple; aire: h = s = 0 líquido saturado a 1 atm) CoolProp devuelve
  ruido (−6.6×10⁻⁸ J/kg); `core.fluids` lo pasa a 0 (`_zero_if_noise`)
  en `StatePoint`, `FluidState` y la saturación. Si se lee CoolProp
  directo, aplicar lo mismo.
- **Pseudo-puros** (aire, R410A; `FluidLimits.is_pure` False): tienen
  deslizamiento de temperatura en la campana. Con p y (h, s, v o u) el
  estado se calcula con (p, x), con x de la regla de la palanca (el
  flash de CoolProp falla cerca de la línea de burbuja); T-x con
  0 < x < 1 no está definido. En los procedimientos, T_f ≤ T ≤ T_g en vez
  de T = T_sat.

Notas de la Fase 3.1a (TESPy 0.11):

- Armar la red como el tutorial oficial (`tutorial/basics/rankine.py`:
  `CycleCloser` + `SimpleHeatExchanger` + `Turbine` + `Pump`) con
  `core.cycles.tespy_utils.new_network()`, que fija unidades SI (incluida
  `pressure_difference`: si falta, TESPy 0.11 avisa con un FutureWarning).
- Resolver con `tespy_utils.solve(network, what=...)`: exige
  `network.status == 0` (1 = convergió con parámetros fuera de rango, p. ej.
  una turbina que comprime; 2 = no convergió; 3 = singular; 99 = error) y
  explica el problema en castellano.
- TESPy acepta datos imposibles sin quejarse (entrada a la turbina por
  debajo de la saturación, un "recalentamiento" que enfría): validar en
  `core` antes de resolver y revisar el resultado después.
- De TESPy leer solo p y h (`.val_SI`) y reconstruir cada estado con
  `core.fluids.fluid_state_from_pair` (p, h): región, título y saturación
  salen con la convención del proyecto (no usar el x de TESPy).
- Cachear el resultado (dataclasses picklables) con `st.cache_data`, nunca
  la `Network`. `import tespy` tarda ~3,6 s en frío: queda dentro de
  `core/cycles/`. Un Rankine se resuelve en ~60 ms; un barrido, una red por
  punto.
- Ecuaciones: el factor de unidades de v·Δp y los términos con ×10ⁿ (SI)
  van en un renglón de continuación (`\\ &\quad`); la regla de la palanca
  usa h_fg y s_fg, como Cengel.

Notas de la Fase 3.1b (regeneración):

- Calentador cerrado = `Condenser` (`ttd_u` se mide contra T_sat a la
  presión de la extracción; `ttd_u = 0` es el ideal de Cengel y la salida
  caliente queda en líquido saturado); abierto = `Merge` con x = 0 a la
  salida; extracción = `Splitter`; drenaje en cascada = `Valve` + `Merge`
  (antes del condensador o de la entrada caliente del cerrado de menor
  presión); drenaje hacia adelante = bomba + `Merge`. Todo como los ejemplos
  oficiales (tutorial de optimización de una central, modelo SEGS de sus
  tests, CCPP).
- La presión se fija una sola vez: entrada de la turbina y salida de cada
  tramo; bombas, válvulas y `Merge` la toman de la red.
- Topologías: hasta 3 calentadores; solo el cerrado de mayor presión
  bombea el drenaje hacia adelante. Así cada fracción se despeja en orden
  (de mayor a menor presión) en el procedimiento.
- η_T se mide desde la entrada de cada turbina (soluciones de Cengel): los
  tramos llevan el `eta_s` local que reproduce esa línea de expansión.
- TESPy no avisa una extracción negativa hacia un abierto (un `Merge` no
  tiene límites): `solve_rankine` revisa cada fracción después de resolver.
  Antes de resolver, cada calentador tiene que poder calentar (incluido el
  calentamiento en la bomba).
- La numeración sale de `core.cycles.rankine_layout` antes de resolver: la
  página la usa para los rótulos (lee de `session_state` los widgets que
  están más abajo). Las opciones de los radios no llevan números: cambiar
  las opciones reinicia el widget.
- Con regeneración η < 1 − T_C/T̄_H (los calentadores generan entropía).
- "Cómo aumentar el rendimiento" es Cengel §10-4 (ediciones 7.ª a 9.ª);
  §10-6 es la regeneración.

### Citas y licencias

- **Toda librería externa de cálculo** (no UI) que se sume al proyecto
  exige actualizar `CITATION.cff` con su referencia formal y `README.md`
  con su BibTeX.
- **Normas técnicas** (ISO, ASHRAE, IRAM): citar siempre versión y año.
- En la página `99_Acerca.py` se muestra dinámicamente el contenido de
  `CITATION.cff` y la licencia.

## Comandos

```bash
# Correr localmente
streamlit run streamlit_app.py

# Tests
pytest
pytest tests/test_combustion.py -v

# Lint + format
ruff check .
ruff format .

# Cobertura
pytest --cov=core --cov-report=term-missing
```

## Reglas para Claude Code

- **No** mezclar UI (Streamlit) con cálculo en el mismo archivo. Si
  encontrás `st.*` dentro de `core/`, refactorizá.
- **No** hardcodear unidades dentro de funciones de cálculo en `core/`.
- **No** agregar dependencias sin actualizar `requirements.txt` Y
  `CITATION.cff`.
- **No** romper la API pública existente sin migración explícita y
  tests que cubran el caso viejo.
- Antes de implementar un módulo nuevo, **proponer un plan**
  (estructura de archivos, firmas de funciones, casos de test) y
  esperar OK. Usar Plan Mode (`Shift+Tab`) si la tarea es grande.
- Cada PR / commit nuevo debe incluir: código + tests + actualización
  de README si es módulo nuevo + actualización de `CITATION.cff` si
  cambian dependencias.
- Cuando integres TESPy, basarse en los ejemplos canónicos de la doc
  oficial (`tespy.readthedocs.io`), no inventar la API.
- Para ISO 6976, los valores por componente puro deben venir de
  `data/iso6976_components.csv` extraídos de la norma; no hardcodear
  en código. Tests obligatorios contra los ejemplos del anexo de la norma.
- Para correlaciones de PCI (Dulong, Boie, Channiwala-Parikh), cada
  función lleva en el docstring la referencia exacta al paper original
  y el rango de validez (tipo de combustible).
- Mensajes de error orientados al alumno: explicar qué entrada está
  fuera de rango y por qué, no solo "ValueError".

## Notas didácticas

Este software apunta a estudiantes de grado de ingeniería. Priorizar
**transparencia del cálculo** sobre performance:

- Mostrar pasos intermedios siempre que sea pedagógicamente útil.
- Permitir que el alumno vea la diferencia entre interpolar tablas y
  usar la ecuación de estado de CoolProp.
- Permitir comparar correlaciones de PCI entre sí para el mismo
  combustible.
- Mostrar destrucción exergética con interpretación física, no solo
  número.
