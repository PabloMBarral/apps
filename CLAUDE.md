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
│   │                          # Fase 3.1c: fluido de trabajo (ORC), pérdidas,
│   │                          # calentadores reales, recuperador, barrido de ε.
│   ├── 6_Refrigeracion.py     # ✅ Fase 3.2 — Compresión de vapor: simple, cámara
│   │                          # de evaporación instantánea y cascada (1 o 2
│   │                          # refrigerantes), refrigerador o bomba de calor,
│   │                          # niveles por p o por T_sat, ciclo real, segundo
│   │                          # principio (barras), pestañas por fluido, barridos.
│   ├── 7_Psicrometria.py
│   ├── 8_Combustion.py
│   ├── 9_Poder_Calorifico.py
│   ├── 4_ISO6976.py           # ✅ Fase 2.3 (matriz identidad; teoría y export 1.7)
│   ├── 10_HRSG.py             # ✅ Fase 3.3 — HRSG de una presión: diagrama T–Q
│   │                          # (plotly) con vapor sobrecalentado o saturado,
│   │                          # composición con λ o cargada, secciones, estados,
│   │                          # comparación saturado/sobrecalentado, barridos.
│   ├── 11_Exergia.py
│   ├── 12_Ciclo_Combinado.py  # ✅ Fase 3.4 — Turbina de gas (metano o aire estándar)
│   │                          # + HRSG de una presión (por pinch o por chimenea) +
│   │                          # Rankine con desaireador: métricas, Sankey, T–s de
│   │                          # la TG, T–Q, ciclo de vapor, control con TESPy,
│   │                          # procedimiento por partes, barridos.
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
│   │                          # Fase 3.2: + caudal volumétrico (volume_flow).
│   ├── fluids.py              # ✅ Fase 1.6 — Wrappers sobre CoolProp:
│   │                          # StatePoint/state_from_pair (cálculo) y
│   │                          # FluidState/fluid_state_from_pair (estado completo:
│   │                          # región, saturación, transporte, validación con
│   │                          # mensajes al alumno, suggested_inputs).
│   │                          # Fase 3.1c: + R-245fa, R-1233zd(E), isopentano, tolueno.
│   │                          # Fase 3.2: + R-32, propano, isobutano;
│   │                          # textbook_reference_offset (tablas del R-134a).
│   ├── state_report.py        # ✅ Fase 1.6 — Presentación pura de un FluidState:
│   │                          # tablas, notas, procedimiento LaTeX por sistema
│   │                          # de unidades, export JSON/CSV, tabla de estados.
│   │                          # Fase 3.2: textbook_reference_note (R-134a).
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
│   │                          # Fase 3.1c: con fricción, casi isobárica.
│   │                          # Fase 3.2: válvulas a h constante (isenthalpic).
│   ├── ideal_gas.py           # ✅ Fase 3.4 — Mezclas de gases ideales (se mudó de
│   │                          # hrsg.py, que lo reexporta): FlueGas con h, cp, s°(T),
│   │                          # s(T, p) absoluta (NIST-JANAF), T_isentropic, T_from_h;
│   │                          # AIR_DRY / AIR_TECHNICAL, exhaust_composition(λ),
│   │                          # combustion_products(aire, átomos, λ).
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
│   │   │                      # Fase 3.1c: Losses/PipeLoss (ciclo real), DCA,
│   │   │                      # desrecalentador, bombeados intermedios, fluid
│   │   │                      # (ORC), Recuperator, fluid_behavior,
│   │   │                      # suggested_rankine_inputs.
│   │   ├── rankine_layout.py  # ✅ Fase 3.1b — Topología y numeración tipo Cengel
│   │   │                      # (sin TESPy): FeedwaterHeater, plant_layout,
│   │   │                      # port_flows (caudales 1 − y… simbólicos).
│   │   │                      # Fase 3.1c: cañerías, recuperador, bombeados en
│   │   │                      # cualquier cerrado, "evaporador".
│   │   ├── rankine_procedure.py # ✅ Fase 3.1b — rankine_steps (se mudó; rankine.py
│   │   │                      # lo reexporta): simple, recalentamiento y regenerativo.
│   │   │                      # Fase 3.1c: ciclo real, recuperador, calentadores
│   │   │                      # reales, sistema acoplado, textos por fluido.
│   │   ├── refrigeration.py   # ✅ Fase 3.2 — RefrigerationInputs / solve_refrigeration /
│   │   │                      # RefrigerationResult: layout fijo por ciclo (numeración
│   │   │                      # de Cengel), red de TESPy, validaciones, ExergyAnalysis,
│   │   │                      # SingleStage, notas, ejemplos, barridos y export.
│   │   ├── refrigeration_procedure.py # ✅ Fase 3.2 — refrigeration_steps (reexportado):
│   │   │                      # estados, cámara, mezcla, cascada, COP, Carnot,
│   │   │                      # potencias y exergía destruida por componente.
│   │   ├── hrsg.py            # ✅ Fase 3.3 — FlueGas (mezcla de gases ideales:
│   │   │                      # M, w, R, h_g(T), T(h), rocío), exhaust_composition(λ),
│   │   │                      # HRSGInputs / solve_hrsg / HRSGResult (balances por
│   │   │                      # sección), hrsg_tq_profile, notas, ejemplos,
│   │   │                      # other_steam_option, barridos y hrsg_to_dict. Sin TESPy.
│   │   │                      # Fase 3.4: el gas pasa a core/ideal_gas.py; diseño
│   │   │                      # por temperatura de chimenea (T_stack_K, pinch_K).
│   │   ├── hrsg_procedure.py  # ✅ Fase 3.3 — hrsg_steps: composición, entalpía de
│   │   │                      # los gases, agua, caudal de vapor, secciones, total
│   │   │                      # y aprovechamiento, punto de rocío. Fase 3.4: modo
│   │   │                      # chimenea, times_diff / factor_times_diff públicos.
│   │   ├── brayton.py         # ✅ Fase 3.4 — Fuel (ISO 6976), BraytonInputs /
│   │   │                      # solve_brayton / BraytonResult, validaciones, notas,
│   │   │                      # brayton_ts_lines y brayton_tespy (control).
│   │   ├── brayton_procedure.py # ✅ Fase 3.4 — brayton_steps: aire, compresor (s°),
│   │   │                      # cámara (PCI, aire teórico, f, λ) o aire estándar,
│   │   │                      # gases, turbina, rendimiento y potencias.
│   │   ├── combined.py        # ✅ Fase 3.4 — SteamCycle, CombinedInputs /
│   │   │                      # solve_combined / CombinedResult (Kehlhofer, balance),
│   │   │                      # notas, ejemplos, barridos y combined_to_dict.
│   │   └── combined_procedure.py # ✅ Fase 3.4 — combined_sections: TG, HRSG,
│   │                          # ciclo de vapor y el acople con el rendimiento.
│   └── plots.py               # fluprodia + matplotlib helpers
├── ui/                        # Helpers de UI que sí importan Streamlit
│   ├── branding.py            # Bloque de créditos compartido (sidebar)
│   ├── units_ui.py            # ✅ Fase 1.4 — Selector global + number_input_si
│   │                          # (key real f"{key}@{sistema}": el valor físico
│   │                          # sobrevive al cambio de unidades, Fase 1.6).
│   ├── diagrams.py            # ✅ Fase 1.5a — Cache de FluidPropertyDiagram
│   │                          # (@st.cache_resource), render_diagram_plotly
│   │                          # con overlays de puntos / procesos.
│   └── cycle_charts.py        # ✅ Fase 3.4 — tq_figure (de /HRSG),
│                              # render_rankine_diagram (de /Rankine),
│                              # gas_turbine_ts_figure y energy_sankey_figure.
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
- Numeración del cap. 10 de Cengel (ediciones 7.ª a 9.ª): §10-2 Rankine
  ideal, §10-3 desviaciones del ciclo real (pérdidas de carga y de calor,
  ejemplo 10-2), §10-4 cómo aumentar el rendimiento, §10-5 recalentamiento,
  §10-6 regeneración, §10-9 ciclos combinados. El autor confirmó §10-3 y
  §10-5: hasta la 0.15.0 el ciclo real citaba §10-5 por error, y un test
  (`test_procedure_cites_the_textbook_sections`) lo vigila.

Notas de la Fase 3.1c (ciclo real, calentadores reales y ORC):

- Ciclo real (`Losses`, `PipeLoss`): los datos siguen siendo los de la
  turbina (p y T de entrada, presión de escape, presión de recalentamiento
  a la salida de la de alta) y la bomba compensa todas las caídas: cada
  componente lleva su `dp` y TESPy resuelve la presión de la bomba.
  Cañerías = `Pipe` con `dp` y la T de salida (la de alimentación, con
  `Ref(entrada, 1, −ΔT)`); sin ΔT, `Q = 0`. Subenfriamiento con
  `td_bubble` a la salida del condensador. Primer principio:
  w_neto = q_H − q_C − q_pérd. `T_low_K` es la de condensación (T_sat a
  la presión de escape), no la del condensado subenfriado.
- Calentador cerrado real: subenfriador de drenaje = `Condenser` con
  `subcooling=True` y `ttd_l` = DCA (el ejemplo de la doc de TESPy);
  desrecalentador = `Desuperheater` en serie (la extracción sale como vapor
  saturado) y el TTD se fija como la T del agua a la salida: el `ttd_u` de
  TESPy tiene mínimo 0 y un TTD < 0 en un `Condenser` da status 1.
- Un drenaje bombeado desde un cerrado que no es el de mayor presión
  acopla las fracciones (la mezcla cambia la h que entra al calentador
  siguiente): el procedimiento muestra el sistema, la solución y la
  verificación de cada balance en vez de despejar y de a una.
- Antes de resolver se calculan la línea de expansión (`_expansion_line`)
  y la de agua (`_feedwater_line`, `_line_pressures`) para validar:
  extracción sobrecalentada para el desrecalentador, escape sobrecalentado
  para el recuperador, calentador que no puede calentar, presión de la
  bomba.
- ORC: `RankineInputs.fluid` (`RANKINE_FLUIDS`). TESPy recibe `water` para
  el agua (como el tutorial) y el nombre de CoolProp para el resto. En la
  página las keys de los widgets llevan el fluido, salvo el agua (las de la
  0.12.0 no cambian); los valores por defecto de cada fluido salen de
  `suggested_rankine_inputs` (un test los recorre). Recuperador =
  `HeatExchanger` con `eff_max` = ε (q/q_máx), solo sin calentadores.
  Seco, húmedo o casi isoentrópico (`fluid_behavior`): T·(ds_g/dT)/s_fg a
  0,8·T_c, con umbral ±0,25 (Chen, Goswami y Stefanakos, 2010).
- Textos según el fluido: tablas de Cengel para el agua (A-4 a A-7) y el
  R-134a (A-11 a A-13); si no, "ecuación de estado". En el LaTeX, los
  subíndices con tilde van en `\text{}`: KaTeX estricto avisa con
  `\mathrm{pérd}`.
- Diagramas: un intercambiador o una cañería con caída de presión
  (presiones a menos de 1/4 una de otra) se dibuja con p lineal en h, no
  como recta.
- Streamlit no recarga los módulos de `core/` al editarlos: reiniciar el
  servidor antes del smoke test (si no, el síntoma es un diagrama viejo).
- Regresión: un snapshot de la 0.12.0 (24 casos: estados, red de TESPy con
  todas sus especificaciones, pasos en los tres sistemas, export y
  barridos) queda idéntico.

Notas de la Fase 3.2 (refrigeración por compresión de vapor):

- Red como el tutorial de bomba de calor de TESPy 0.11
  (`tutorial/basics/heat_pump.py`): `CycleCloser` + `SimpleHeatExchanger`
  + `Compressor` + `Valve`; sobrecalentamiento con `td_dew` y
  subenfriamiento con `td_bubble` (`tutorial/advanced/stepwise.py`). Cámara
  de evaporación instantánea = `DropletSeparator` (out1 líquido, out2
  vapor) + `Merge`; cascada = `HeatExchanger` entre dos lazos, cada uno con
  su `CycleCloser` y su fluido. Con ΔT = 0 en el intercambiador (ejemplo
  11-4) TESPy da `kA = nan` sin avisar: no importa (no se usa).
- Base de cálculo: 1 kg/s por el condensador (la de Cengel en los cinco
  ejemplos) y después se escala al caudal o a la capacidad.
- `p_evap_Pa` y `p_cond_Pa` son la aspiración y la descarga del compresor;
  la T de evaporación es la de rocío ahí y la de condensación la de
  burbuja (el R-410A tiene deslizamiento). En la página, los niveles por
  temperatura se convierten con esas mismas definiciones.
- En una cascada con dos fluidos, la presión del evaporador puede superar
  a la del condensador (CO₂ abajo, R-134a arriba): las presiones se
  comparan dentro de cada ciclo, nunca entre ciclos.
- Notación del vademecum (§9.3): Q_C sale de la fuente fría (Cengel: Q_L).
  Segundo principio: T₀ es el ambiente (la fuente caliente del
  refrigerador, la fría de la bomba de calor); X_dest = T₀·S_gen por
  componente y Ẇ = Ẇ_mín + ΣX_dest cierra (test). La cámara no destruye
  exergía (separar fases a la misma T es reversible).
- Cengel 8.ª ed.: §11-5 (segundo principio) es nuevo, así que los
  ejemplos de cascada y cámara son 11-4 y 11-5 (11-3 y 11-4 en la 7.ª).
- Tablas del R-134a (A-11 a A-13): h = s = 0 en el líquido saturado a
  −40 °C; CoolProp usa la del IIR. No se cambia la referencia de CoolProp
  (es global del proceso y afecta a TESPy): `textbook_reference_note` lo
  explica con las constantes (148,14 kJ/kg y 0,7956 kJ/(kg·K)) donde se
  citan esas tablas (Propiedades, Rankine con R-134a y Refrigeración).
- Con un fluido «seco» (isobutano) la compresión isoentrópica desde vapor
  saturado termina dentro de la campana: es una nota, no un error.
- La comparación con el ciclo simple usa el refrigerante del condensador a
  la misma temperatura de evaporación; en la cascada CO₂/NH₃ el COP casi
  no cambia, pero la relación de presiones y la descarga sí (la nota lo
  dice).
- Diagramas: una válvula (h igual, p distinta) se dibuja rayada sobre su
  línea de h constante («estrangulamiento (h constante)»), también en los
  drenajes del Rankine y en una cañería sin pérdida de calor.
- Smoke test: la primera carga de la página puede seguir dibujando
  después de que aparece la segunda fila de métricas; esperar ~3 s antes
  de leer el texto.
- Regresión: el snapshot de la 0.13.0 (29 casos) queda idéntico salvo la
  nota del R-134a en el ORC.

Notas de la Fase 3.3 (HRSG de una presión):

- Cálculo directo en `core/cycles/hrsg.py` (los balances que se hacen a
  mano, sin TESPy: rápido y transparente). TESPy es el control cruzado en
  los tests: `HeatExchanger` en serie con los gases como mezcla (como su
  tutorial de turbina de gas); coincide al 0,02 % (TESPy evalúa cada
  componente a su presión parcial). Sirve de base para el ciclo combinado.
- Gases = mezcla de gases ideales: la h de cada componente sale de CoolProp
  a 1 Pa (`AbstractState`, ~16 µs por punto; `PropsSI` es 8 veces más
  lento), todos a la misma presión, con h_g = 0 a 25 °C. Con la presión
  parcial el H₂O cae en la campana a temperaturas bajas. `T_from_h` busca
  entre el punto triple del agua y 1720 °C.
- Diseño (Kehlhofer et al., 2009): los gases salen del evaporador a
  T_sat + pinch y el agua del economizador a T_sat − approach. El caudal
  de vapor sale del tramo entre la entrada de los gases y el pinch; el
  economizador fija la chimenea (por eso T_alim no cambia ṁ_v).
- Diagrama T–Q con la convención del domo: el agua sube vertical de
  T_sat − approach a T_sat al entrar al evaporador, y los tubos del
  evaporador ven agua a T_sat (su ΔT frío es el pinch). Así el mínimo ΔT
  del perfil es exactamente el pinch (test).
- Errores al alumno: cruce de temperaturas en el economizador o dentro de
  la caldera, chimenea bajo el punto de rocío (el modelo no condensa),
  gases que no alcanzan T_sat + pinch, vapor más caliente que los gases,
  presión supercrítica. Notas: agua de alimentación bajo el punto de rocío,
  ΔT de un extremo menor que el pinch, approach 0.
- El barrido de la presión no es monótono: cerca de la crítica h_fg se
  achica y el caudal vuelve a subir. Se limita a 1–160 bar.
- La página recalcula sola en cada cambio (~0,2 s con la comparación,
  `st.cache_data`); los barridos van con botón. En el diagrama, el pinch y
  el approach (unos pocos kelvin) se señalan con flechas que vienen de
  zonas libres del evaporador, y los nombres de las secciones van arriba;
  medido en 1280 y 390 px.
- Los textos usan °C y bar (como el Rankine); el LaTeX sigue el sistema.
  En SI las restas con ×10ⁿ se parten (`_times_diff`) y el calor total va
  un sumando por renglón: máximo 304 px.

Notas de la Fase 3.4 (turbina de gas y ciclo combinado de una presión):

- Gases ideales en `core/ideal_gas.py` (`hrsg.py` lo reexporta: la API no
  cambió). h y s°(T) desde 25 °C; la entropía absoluta suma la S° de
  NIST-JANAF a 25 °C y 1 bar, −R·ln(p/p°) y el término de mezcla, así el
  aire y los gases quedan en la misma escala del T–s. La isoentrópica sale
  de s°(T₂s) = s°(T₁) + R·ln(p₂/p₁) (método de la A-17 de Cengel). Cada
  especie tiene la T mínima de CoolProp (el CO₂ del aire seco limita a
  216,6 K; con agua, 273,17 K).
- Aire seco (N₂ 0,7808, O₂ 0,2095, Ar 0,0093, CO₂ 0,0004): reproduce las
  tablas de Cengel (Δh̄ de A-18 a A-23 al 0,1 %; ejemplos 9-5, 9-6 y
  10-9). El técnico (21/79) es el del vademecum §16.1.
- Cámara, por kg de aire y con h desde 25 °C (la referencia del PCI):
  h_a(T₂) + f·PCI = (1 + f)·h_g(T₃). La composición depende de f: `brentq`
  entre ~0 y el estequiométrico (si no alcanza, λ < 1 y error). PCI de
  ISO 6976:2016 a 25 °C: PCS_m − (b/2)·L₀ (metano: 50,027 MJ/kg). Aire
  estándar = `fuel=None`. TIT ≤ T₂ se valida antes (sin eso `brentq` falla
  sin explicación).
- TESPy de control (`brayton_tespy`, como su tutorial de turbina de gas):
  `Compressor` + `DiabaticCombustionChamber` (combustible a 25 °C con
  `p=Ref(c2, 1.05, 0)`) + `Turbine`; aire estándar con
  `SimpleHeatExchanger`. Sin `T0`/`m0` del cálculo directo, Newton pasa por
  63 K y termina en status 99. Coincide al 0,3 % (gas real a r_p alta).
- HRSG por chimenea (Cengel 10-9): ṁ_v del balance de toda la caldera; el
  pinch es un resultado (≤ 0: cruce de temperaturas, con mensaje).
- Acople: el Rankine se resuelve por kg (bomba → agua de alimentación de
  la HRSG) y se escala con el ṁ_v de la HRSG (`dataclasses.replace`).
  η_HRSG = Q̇_HRSG / (Q̇_comb − Ẇ_TG): con esa definición la relación de
  Kehlhofer y el balance Q̇_comb = Ẇ_TG + Ẇ_TV + Q̇_cond + Q̇_chim (la
  chimenea contra el aire a T₁) cierran exactos (tests). Los errores llevan
  el nombre de la parte («Turbina de gas: », «Caldera de recuperación: »,
  «Ciclo de vapor: »).
- Barridos: en los de r_p y TIT el vapor se sobrecalienta como mucho hasta
  T₄ − 25 K (`SH_HOT_END_K`); los puntos sin sentido físico se omiten. En
  el ejemplo típico el óptimo del ciclo combinado está en r_p ≈ 20 y el de
  la turbina de gas sola, en el extremo del barrido (40).
- Página: botón «Calcular» como el Rankine; el TESPy de control se
  cachea (devuelve el resultado o el texto del error). El diseño por
  chimenea arranca con la del diseño por pinch del ejemplo redondeada hacia
  arriba a 5 °C (con 150 °C el típico cruza). Los rótulos del Sankey van
  en varios renglones (`<br>`): en uno solo se pisaban a 390 px. En el T–s,
  el tramo gris horizontal en 2 es el cambio de composición de la cámara.
- Procedimiento con la notación de Cengel (s°₁, s°₂s; s°_g,3 para los
  gases). En SI la turbina calcula primero Δh_s y w_T usa
  `factor_times_diff`: máximo 306 px en los tres sistemas (2592
  ecuaciones). Los pasos del Rankine en SI se siguen deslizando (hasta
  384 px), igual que en /Rankine.
- `ui/cycle_charts.py` junta los gráficos de los ciclos: /HRSG y /Rankine
  importan de ahí el T–Q y el diagrama del Rankine, sin cambios visibles.

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
