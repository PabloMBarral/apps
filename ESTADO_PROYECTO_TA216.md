# Estado del proyecto — TA216 Apps

> Bitácora viva de qué está cerrado, qué quedó deferido y qué viene.
> Se actualiza al cierre de cada fase como parte del checklist (junto
> con el bump de versión, el README y el `CITATION.cff`). Generada a
> partir del `git log` y la documentación interna del proyecto.
>
> **Versión actual**: `0.15.0` — Fase 3.3 cerrada (2026-10-06).

---

## Hitos cerrados

### Fase 0 — Migración a arquitectura multipágina
- **Commit**: `40abd58` (2026-05-19) — `refactor: migrar a arquitectura multipágina (Fase 0)`
- **Scope**: separación estricta `core/` (lógica pura, sin Streamlit) +
  `pages/` (UI Streamlit) + `ui/` (helpers de presentación). Navegación
  vía `st.navigation` + `st.Page`. Sidebar de créditos centralizado en
  `ui/branding.py`.

### Fase 1.1 — Interpolación lineal y doble entrada
- **Commit**: `8570cd3` (2026-05-20) — `feat: módulo de interpolación lineal y doble entrada (Fase 1.1)`
- **Scope**: `core/interpolation.py` con interpolación lineal simple y
  bilineal sobre tablas. Procedimiento didáctico paso a paso en LaTeX +
  narrativa en español. Comparación contra CoolProp como opt-in.

### Fase 1.2 — Polish de Interpolación
- **Commit**: `3b3f7ed` (2026-05-20) — `feat: pulido del módulo de interpolación (Fase 1.2)`
- **Scope**: fix de bug "crash al elegir s" (era UX collapse, resuelto
  con `st.session_state`). Normalizador de unidades tolerante a notación
  variada (`kJ/(kg·K)`, `kJ/kg-K`, `kJ kg^-1 K^-1`). Comparación CoolProp
  como opt-in con selector de fluido (Water, R134a, R410A, R1234yf, NH₃,
  CO₂, Air). Tabla editable in-place (`st.data_editor`). Rename de
  materia. `st.navigation` + sidebar branding compartido.

### Fase 1.3 — Rendimientos isoentrópicos
- **Commit**: `1d0519a` (2026-05-20) — `feat: rendimientos isoentrópicos (Fase 1.3)`
- **Scope**: `core/isentropic.py` con turbina, compresor, bomba y
  compresor multietapa (politrópico con intercooler). Modo directo
  (dado η_s y P_out → estado real) e inverso (dados los estados →
  recuperar η_s). Validación de fase líquida del inlet de bomba vía
  `CoolProp.PhaseSI`. Comparación opt-in de la bomba contra el modelo
  incompresible `w_p ≈ v_1·Δp/η_s`. Diagrama T–s del proceso: **deferido
  a Fase 1.5**.

### Fase 2.3 — ISO 6976:2016
- **Commits**: `e399b78` (datos Tabla 1 + Annex A),
  `ca8452e` (Tabla 2, Tabla 3, fixtures Annex D),
  `6af2cf7` (2026-05-21) — `feat: implementación de ISO 6976:2016 (Fase 2.3, matriz identidad)`,
  `222fc00` — `fix(iso6976): subir tol_abs de compression_factor Ex 2 a 2e-6`.
- **Scope**: poder calorífico bruto/neto (molar, másico, volumétrico),
  densidad, densidad relativa al aire e índices de Wobbe G y N, con
  propagación de incertidumbre estándar (matriz identidad). Composición
  editable in-place con autocompletado de los 60 componentes de la
  norma. Ejemplos del Annex D (D.2, D.3, D.4) verificados contra los
  valores tabulados.
- **Deferido**: matriz de normalización completa — requiere
  implementar ISO 14912:2003 Formula (69). Pseudocódigo de dos
  formulaciones (`Cov_proj` y `Cov_renorm`) documentado al pie de
  `core/combustion/iso6976.py` para retomar en fase futura.

### Fase 1.4 — Multifluido + selector global de unidades
- **Commit**: `b0f25c9` (2026-05-21) — `feat: multifluido + selector global de unidades (Fase 1.4)`
- **Scope**: `core/units_system.py` con tabla central
  `(QuantityKind, UnitSystem) → (factor, offset, label)` y API
  `format_quantity / convert_*_si / unit_label`. Tres sistemas:
  **SI**, **Técnico** (default, alineado con Cengel) e **Inglés**.
  `ui/units_ui.py` con selector global en sidebar + `number_input_si`
  que opera en SI internamente. La página de Propiedades habilita los
  7 fluidos del proyecto. Los pasos didácticos LaTeX de Isoentrópicos
  quedan en Técnico hardcoded — su conversión al sistema activo entra
  en Fase 1.5b.

### Fase 1.5a — Diagramas con fluprodia
- **Versión**: `0.8.0` (2026-05-23)
- **Scope**: `core/diagrams.py` (wrappers tipados sobre fluprodia 4.2)
  + `ui/diagrams.py` (cache `@st.cache_resource` keyed por
  `(fluido, sistema)`, render via plotly). Los 4 tipos de diagrama del
  roadmap (log p–h, T–s, h–s, p–log v) disponibles para los 7 fluidos
  del proyecto y los 3 sistemas de unidades.
- **Integraciones**:
  - Página **Propiedades**: expansor `📈 Diagrama del fluido` con el
    estado calculado superpuesto.
  - Página **Isoentrópicos**: expansor `📈 Diagrama del proceso` en los
    4 tabs (turbina, compresor, bomba, multietapa). Para single-stage
    se traza la isoentrópica real 1→2s vía
    `calc_individual_isoline(s=cte)`; los segmentos 2s→2 y 1→2 quedan
    como referencias visuales (línea recta, con caption didáctico que
    aclara que **no** representan la trayectoria termodinámica real).
    Para multietapa, cada etapa con su intercooler isobárico.
- **Tests**: 35 tests en `tests/test_diagrams.py` (mapeo de unidades,
  rangos por fluido, build_diagram para los 7 fluidos, procesos,
  conversión SI ↔ coords del diagrama, JSON round-trip).
- **Dependencias**: `fluprodia>=4.2`, `plotly>=5.0` agregadas a
  `requirements.txt` y `CITATION.cff`.

### Fase 1.6 — Calculador de estado del agua para los alumnos
- **Versión**: `0.9.0` (2026-10-03). Rama `claude/water-state-analyzer-f4fiev`.
- **Commits**: `e38e9d3` (chore: `use_container_width` → `width="stretch"`,
  Streamlit ≥ 1.51), `9dd6579` (fix: p–log v, título x = 10000 con h-s,
  ventana de ejes por fluido), `3d13be8` (feat: magnitudes nuevas en
  `units_system`), `ed948f2` (feat: `FluidState` en `core/fluids.py`),
  `d5716d8` (feat: `core/state_report.py`), `e149e55` (fix: valor físico
  de los inputs al cambiar de sistema), `872ac7b` (feat: reescritura de la
  página), `e3d356b` (feat: tabla de estados y ciclos en el diagrama), y
  el commit de docs que cierra la fase.
- **Bugs corregidos** (todos reproducidos antes de arreglarlos):
  - El primer cálculo que veía el alumno fallaba: "t y p" con 0 °C y 1 bar
    está debajo del punto triple del agua (CoolProp: "below Tmelt"). Los
    defaults de p-h y h-s (h = 0, s = 0) tampoco existían.
  - Los resultados de Propiedades desaparecían al cambiar el tipo de
    diagrama (se mostraban solo dentro de `if submit:`), y el selector
    volvía a log p–h: en la práctica solo se podía ver un diagrama.
  - p–log v lanzaba `ValueError` en Propiedades e Isoentrópicos
    (`StatePoint` no tenía volumen); además `AXIS_MAP` declaraba log el
    eje p, que fluprodia dibuja lineal.
  - Con el par h-s, CoolProp devuelve x = 10000 fuera de la campana y la
    página lo mostraba.
  - Pares como T-s "resolvían" estados absurdos sin aviso (R134a a
    44 549 bar).
  - Al cambiar el sistema de unidades, `number_input_si` conservaba el
    número con la unidad nueva (400 °C pasaba a 400 K) — afectaba también
    a Isoentrópicos.
  - La ventana fija de los diagramas dejaba fuera el vapor de agua a baja
    presión en p–log v y achicaba la campana de los refrigerantes.
- **Scope**: estado completo (`FluidState`): región, u, ρ, cp, cv, γ, w,
  μ, k, ν, α, Pr, Z, sobrecalentamiento/subenfriamiento y saturación a p
  y a T; pares nuevos T-v, p-v, p-u; validación con mensajes en
  castellano; procedimiento "como con las tablas" en LaTeX en el sistema
  activo; tablas pensadas para el celular; tabla de estados para armar
  ciclos con isobáricas e isoentrópicas reales sobre el diagrama;
  exportación CSV/JSON; expansor de fórmulas con link al vademecum
  (§3.2, §7.3, §12, §13).
- **Tests**: de 381 a 791 (+410). Valores contra Cengel A-4, A-5, A-6,
  A-7 y Cengel & Ghajar A-9; AppTest de la página (defaults de los 10
  pares, persistencia, errores, ciclo Rankine armado por la UI). El LaTeX
  generado se validó con KaTeX (3192 expresiones, 0 errores). Revisado en
  Chromium a 1280 px y 390 px de ancho.
- **Referencias**: IAPWS-95 (Wagner & Pruß, 2002) agregada a
  `CITATION.cff` y al README. Sin dependencias nuevas.

### Fase 1.7 — Isoentrópicos, fórmulas y export en todas las páginas, celular
- **Versión**: `0.10.0` (2026-10-03). Rama `claude/water-state-analyzer-f4fiev`.
- **Commits**: `3df17ae` (fix: `pages/` → `app_pages/`), `704bd1b` (feat:
  Isoentrópicos con validaciones, defaults por fluido, pasos por sistema
  — Fase 1.5b — y export), `dd05941` (feat: fórmulas teóricas y export en
  Interpolación e ISO 6976), `5ad2890` (fix: isolíneas de los diagramas en
  SI e Inglés), `2ef9cfe` (fix: ecuaciones que entran en un celular),
  `d829767` (docs), `205ed6f` (fix: pseudo-puros dentro de la campana y
  ruido del estado de referencia) y el commit de docs que lo registra.
- **Bugs corregidos** (todos reproducidos antes de arreglarlos):
  - Links directos (`/Propiedades`) después de un reinicio mostraban el
    menú automático con nombres de archivo: la carpeta `pages/` activa el
    modo multipágina viejo con una bandera global del proceso.
  - Isoentrópicos: con R410A y CO₂ la bomba fallaba con los valores
    iniciales (errores en inglés); con R134a la turbina "calculaba" a
    400 °C, por encima del máximo de su ecuación de estado (182 °C), sin
    aviso; al cambiar de fluido seguía el resultado del anterior (y el
    diagrama mezclaba isolíneas de un fluido con estados de otro); la
    tabla mostraba x = −1 fuera de la campana.
  - Diagramas en SI: las isolíneas se generaban en °C/bar pero fluprodia
    las leía en K/Pa (isobaras de 0,01–1000 Pa, isotermas de 0–600 K); con
    aire, CoolProp fallaba y la página de Isoentrópicos se caía. En
    Inglés el diagrama del agua quedaba recortado a 315 °C y 69 bar.
  - Ecuaciones más anchas que la pantalla del celular: en Técnico, 148 de
    289 superaban los ~324 px útiles a 390 px de ancho (hasta 684 px) y
    KaTeX las cortaba con un scroll horizontal que no se ve.
  - Restas con negativos sin paréntesis (`a − −3`) en interpolación (tablas
    en °C bajo cero) e Isoentrópicos (aire), y la bomba de aire líquido
    arrancaba en el estado de referencia de CoolProp (h₁ = −1,2×10⁻⁵ kJ/kg,
    "líquido comprimido" para un líquido saturado).
  - Aire y R410A (pseudo-puros): cerca de la línea de burbuja, p con h, s,
    v o u daba "líquido comprimido" en el borde y, apenas adentro
    (x ≈ 1e-4 … 1e-2), un error que decía que el estado no existía. Ahora el
    título sale de la regla de la palanca (lineal en el modelo de CoolProp)
    y se calcula con (p, x); p-x con 0 < x < 1 se acepta (T-x no: a T fija
    la presión cambia con el título). El procedimiento muestra
    T_f ≤ T ≤ T_g en vez de "T = T_sat".
  - Ruido en el estado de referencia: el agua en el punto triple mostraba
    u_f = −6,6048×10⁻¹¹ kJ/kg (Cengel A-4: 0,000) y el aire líquido a 1 atm
    h_f = −1,2×10⁻⁵ kJ/kg; ahora valen 0.
- **Scope**: Isoentrópicos con las validaciones de `fluid_state_from_pair`
  (mensajes en castellano: entrada líquida a un compresor, vapor en la
  bomba, interenfriador que condensaría), valores iniciales calculables
  para los 7 fluidos (`suggested_device_inputs`), tabla de estados con la
  región, pasos LaTeX en el sistema activo (Fase 1.5b, con
  `core/latex.py`) y comparación de la bomba con el modelo incompresible.
  Las cuatro páginas cumplen el mínimo de CLAUDE.md: expansor `📖
  Fórmulas teóricas` con la sección del vademecum (§2.1/§2.2, §10.4,
  §3.3, §6.3, §16.9) y descarga CSV/JSON (`core/export.py`). Ecuaciones
  en renglones cortos (`latex_chain`): en Técnico ya no hay ninguna más
  ancha que el celular.
- **Tests**: de 791 a 952 (+161): AppTest de Isoentrópicos (7 fluidos ×
  4 dispositivos, modo inverso, errores, unidades, aire en SI e Inglés),
  de Interpolación y de ISO 6976; navegación por las cuatro páginas;
  isolíneas por sistema (7 fluidos × 3 sistemas); export y helpers de
  LaTeX; pseudo-puros a lo largo de la campana; ruido de referencia. Sin
  warnings de pytest (fixtures de clase con `@classmethod`). Las 1013
  ecuaciones distintas de las páginas 1–4 (28 estados de Propiedades,
  7 fluidos × 4 dispositivos, 3 sistemas) validan con KaTeX en modo
  estricto y se midió su ancho renderizado en Chromium.
- Sin dependencias nuevas.

### Fase 3.1a — Ciclo de Rankine con TESPy
- **Versión**: `0.11.0` (2026-10-05). Rama `claude/water-state-analyzer-f4fiev`.
- **Commits**: `c2bda6c` (feat: caudal másico y potencia en
  `units_system`), `549f2a3` (feat: `core/cycles` con TESPy — Rankine
  simple y con recalentamiento), `175ac5d` (feat: página `/Rankine`) y el
  commit de docs que cierra la fase.
- **Scope**: `core/cycles/tespy_utils.py` (red de TESPy en SI y `solve`,
  que exige `status == 0` y explica en castellano por qué no convergió) y
  `core/cycles/rankine.py`: Rankine de agua simple o con recalentamiento
  (una etapa), ideal o real (η_T y η_B), con entrada a la turbina
  sobrecalentada o como vapor saturado seco, y caudal másico o potencia
  neta como dato. La red se arma como el tutorial oficial de TESPy 0.11
  (`CycleCloser` + `SimpleHeatExchanger` + `Turbine` + `Pump`) y cada
  estado se reconstruye con `fluid_state_from_pair` (p, h). Resultados:
  q_H, q_C, w_T, w_B, w_neto, η, BWR, título a la salida de la turbina
  (aviso por debajo de 0,88), η de Carnot y temperatura media de aporte de
  calor, potencias. Procedimiento "como con las tablas" en LaTeX y en el
  sistema activo (bomba con v·Δp, regla de la palanca con h_fg y s_fg,
  balances); barridos de η y del título vs. presión de caldera,
  temperatura de entrada a la turbina y presión del condensador (Cengel
  §10-6); export CSV/JSON. `units_system`: caudal másico y potencia.
  Página `5_Rankine.py` (`/Rankine`) con el checklist de CLAUDE.md:
  ejemplos de Cengel (10-1, 10-3 a/b/c, 10-4 y uno basado en 10-2),
  diagrama del ciclo (T–s por defecto) con la expansión isoentrópica de
  referencia cuando η_T < 1, expansor "Cómo lo resuelve TESPy" (red y
  balances por componente) y fórmulas del vademecum (§3.3, §9.2, §10.1,
  §10.4, §12, §13).
- **Validación**: Cengel 10-1 (η = 26,0 %, x₄ = 0,886), 10-3 a/b/c
  (33,4 / 37,3 / 43,0 %) y 10-4 (45,0 %, x₆ = 0,896); TESPy contra el
  cálculo a mano con CoolProp, estado por estado, en 6 casos (ideal, real,
  recalentamiento, vapor saturado seco, supercrítico).
- **Mensajes al alumno**: entrada a la turbina que sería líquida o justo
  saturada, condensador por encima de la caldera o por debajo del punto
  triple, recalentamiento fuera de rango o que enfriaría el vapor,
  rendimientos fuera de (0, 1], falta el caudal o la potencia; TESPy acepta
  algunos de estos datos sin quejarse, por eso se validan antes y después.
- **Tests**: de 952 a 1070 (+118): `tests/test_rankine.py` (74: Cengel
  10-1, 10-3 a/b/c y 10-4, TESPy contra CoolProp en 6 casos, balances,
  η = 1 − T_C/T̄_H, validaciones, barridos, pasos en los tres sistemas,
  export), `tests/test_page_rankine.py` (AppTest, 18: ejemplos,
  recalentamiento, rótulos, vapor saturado seco, potencia neta, errores,
  unidades, diagrama, barridos), caudal y potencia en `units_system` (+25)
  y la navegación a `/Rankine`. Sin warnings de pytest. El LaTeX de Rankine (604
  expresiones distintas: 6 ejemplos y 4 casos borde × 3 sistemas) valida
  con KaTeX; en Técnico e Inglés ninguna supera el ancho de un celular
  (máx. 272 y 296 px); en SI quedan 2 (h de la turbina real, ≤ 347 px).
  Smoke test en Chromium por link directo a 1280 y 390 px.
- **Dependencias**: `tespy>=0.11.2,<0.12` (ya estaba; se acota a la API
  de unidades de la 0.11).

### Fase 3.1b — Rankine regenerativo y agua de enfriamiento
- **Versión**: `0.12.0` (2026-10-05). Rama `claude/water-state-analyzer-f4fiev`.
- **Commits**: `287c9e0` (feat: plano de la planta y numeración de Cengel),
  `d8b9b1e` (feat: Rankine regenerativo con TESPy, procedimiento y agua de
  enfriamiento), `45975d1` (feat: regeneración en la página) y el commit de
  docs que cierra la fase.
- **Scope**:
  - `core/cycles/rankine_layout.py` (nuevo, sin TESPy): topología y
    numeración de estados como Cengel (10-5: 1–7; 10-6: 1–13; sin
    calentadores, la de la 0.11.0), componentes lógicos con conexiones por
    rol y `port_flows`, el caudal de cada corriente como 1 − y… La página
    lo usa para numerar los rótulos antes de resolver.
  - `core/cycles/rankine.py`: hasta 3 calentadores de agua de alimentación,
    abiertos o cerrados (TTD; drenaje en cascada hacia atrás, o bombeado
    hacia adelante en el cerrado de mayor presión), combinables con el
    recalentamiento. Red genérica de TESPy desde el layout, como los
    ejemplos oficiales: cerrado = `Condenser`, abierto = `Merge` con x = 0,
    extracción = `Splitter`, cascada = `Valve` + `Merge`. η_T se mide desde
    la entrada de cada turbina (tramos con el `eta_s` local que reproduce
    la línea de expansión). Componentes con la fracción de cada corriente;
    las propiedades (w_T, w_B, q_H, q_C, T̄_H…) salen de ahí. Agua de
    enfriamiento del condensador; barrido de la presión de extracción;
    export con calentadores, fracciones y agua de enfriamiento.
  - `core/cycles/rankine_procedure.py` (nuevo): el procedimiento se mudó
    (sin calentadores, la salida es idéntica a la de la 0.11.0) y suma el
    regenerativo en el orden de Cengel: estados de la línea de
    alimentación, turbinas, balances de los calentadores de mayor a menor
    presión con la fracción despejada, mezcla, calores y trabajos con
    (1 − y), T̄_H, potencias y agua de enfriamiento.
  - `core/diagrams.py`: `segments_overlays` (une pares de estados; la usa
    `cycle_overlays`). `ui/diagrams.py`: `point_legend` agrupa los puntos de
    un color en una sola entrada de la leyenda.
  - Página `/Rankine`: bloque de regeneración, rótulos con la numeración del
    layout (las opciones de los radios ya no llevan números), tabla de
    extracciones, agua de enfriamiento, diagrama con extracciones, drenajes,
    válvulas y mezcla, barrido de la presión de extracción, teoría de §10-6
    y ejemplos 10-5, 10-6 (con la nota de la errata) y un cerrado con
    drenaje al condensador.
- **Correcciones de la 3.1a**:
  - "Cómo aumentar el rendimiento", el límite de humedad y T̄_H son Cengel
    §10-4 (estaba §10-6, que es la regeneración).
  - η = 1 − T_C/T̄_H solo vale sin regeneración y con salida húmeda
    (docstring y teoría).
- **Validación**:
  - Cengel 10-5: y = 0,2271 y η = 46,31 % (libro: 0,2270 y 46,3 %); los
    estados coinciden con CoolProp a 10⁻⁶.
  - Cengel 10-6: y = 0,1729, z = 0,1313 y η = 48,98 %, igual que el
    cálculo a mano. El libro da 0,1766 / 0,1306 / 49,2 %, números que
    salen de una bomba II hasta 4 MPa (h₄ = 643,9 kJ/kg); el ejemplo lo
    aclara en pantalla.
  - Cerrado que drena al condensador, cascadas y dos abiertos: iguales a
    los balances a mano.
  - Ciclo real con 3 calentadores y recalentamiento: estados sobre la línea
    de expansión de cada turbina a 10⁻⁹; el balance cierra a 10⁻⁸.
- **Mensajes al alumno**:
  - Presiones de extracción fuera de rango, desordenadas, repetidas o por
    encima de la crítica.
  - TTD negativo, o en un abierto.
  - Drenaje hacia adelante con el calentador de mayor presión abierto.
  - Calentador que no puede calentar el agua (incluido el calentamiento en
    la bomba).
  - Extracción negativa hacia un abierto (TESPy no la avisa).
  - Agua de enfriamiento más caliente que el vapor que condensa.
- **Tests**: de 1070 a 1265 (+195), sin warnings de pytest.
  - `tests/test_rankine_layout.py` (79, sin TESPy): numeración, cascadas
    y una grilla de configuraciones.
  - `tests/test_rankine_regen.py` (101): 10-5, 10-6, cerrado al
    condensador, cascadas, dos abiertos, ciclo real con 3 calentadores,
    balances por componente, validaciones, caudales simbólicos iguales a
    los de TESPy, procedimiento en los tres sistemas, agua de enfriamiento,
    barridos, export, tramos del diagrama y una grilla de robustez de 42
    ciclos reales.
  - AppTest de la página (+12) y los ejemplos nuevos en
    `tests/test_rankine.py` (+3).
  - Regresión: sin calentadores, los resultados, la red de TESPy, los
    pasos y el export son idénticos a los de la 0.11.0 (snapshot de 12
    casos; solo cambian las citas corregidas).
  - LaTeX: validan con KaTeX las 1576 expresiones distintas del
    regenerativo (con agua de enfriamiento) y las 953 de la página. En
    Técnico e Inglés ninguna supera el ancho de un celular (máx. 308 px);
    en SI se deslizan las sumas con ×10ⁿ (≤ 387 px).
  - Smoke test en Chromium a 1280 y 390 px: todas las páginas por link
    directo y /Rankine con el ejemplo 10-6.
- **Dependencias**: ninguna nueva.

### Fase 3.1c — Ciclo real, calentadores cerrados reales y ORC
- **Versión**: `0.13.0` (2026-10-05). Rama `claude/water-state-analyzer-f4fiev`.
- **Commits**: `492f4fc` (feat: fluidos de ORC en `core/fluids` y
  `core/diagrams`), `000ff6e` (feat: plano de la planta con cañerías,
  recuperador y drenajes bombeados desde cualquier cerrado), `4400272`
  (feat: ciclo real, calentadores reales y ORC en `core/cycles`), `9997147`
  (feat: página `/Rankine`) y el commit de docs que cierra la fase.
- **Scope**:
  - **Fluidos**: R-245fa, R-1233zd(E), isopentano y tolueno se suman a
    `SUPPORTED_FLUIDS` (Propiedades, Interpolación, Isoentrópicos y ciclos),
    con su rango de diagrama hasta la T máx. de su ecuación de estado.
  - **Ciclo real** (Cengel §10-5): `Losses` con caídas de presión en la
    caldera, el recalentador, el condensador y los cerrados (lado del agua),
    subenfriamiento del condensado y cañerías de alimentación y de vapor
    (`PipeLoss`: Δp y ΔT). La bomba compensa todas las caídas; q_pérd y
    w_neto = q_H − q_C − q_pérd. En TESPy: `dp` en cada componente, `Pipe`,
    `Ref` y `td_bubble`.
  - **Calentadores cerrados reales**: subenfriador de drenaje (DCA;
    `Condenser` con `subcooling` y `ttd_l`), desrecalentador (`Desuperheater`
    en serie, TTD < 0) y drenaje bombeado hacia adelante desde cualquier
    cerrado. Si el bombeado no es el de mayor presión las fracciones quedan
    acopladas y el procedimiento muestra el sistema, su solución y la
    verificación de cada balance.
  - **ORC**: `RankineInputs.fluid` (agua, R-245fa, R-1233zd(E), isopentano,
    tolueno, R-134a, R-1234yf, amoníaco), "evaporador" en lugar de
    "caldera", recuperador (`HeatExchanger` con `eff_max` = ε) en el ciclo
    sin calentadores, clasificación del fluido en seco, húmedo o casi
    isoentrópico (`fluid_behavior`; Chen, Goswami y Stefanakos, 2010) y
    valores por defecto calculables para cada fluido
    (`suggested_rankine_inputs`; Quoilin et al., 2013). Barridos por
    fluido y de la efectividad del recuperador.
  - **Procedimiento**: textos según el fluido (tablas de Cengel para el
    agua y el R-134a; si no, ecuación de estado) y pasos nuevos (condensado
    subenfriado, presión de cada bomba con ΣΔp, cañerías, recuperador,
    calentadores con DCA y desrecalentador —con la temperatura entre
    zonas—, primer principio con q_pérd).
  - **Diagramas**: un intercambiador o una cañería con fricción se dibuja
    casi sobre la isobara (antes, una recta que cruzaba la campana).
  - **Página `/Rankine`**: selector de fluido (keys por fluido; las del agua
    no cambian), bloques de pérdidas y recuperador, casillas por cerrado
    (desrecalentador, subenfriador, bombeado, Δp) y teoría de §10-5 y del
    ORC. Ejemplos nuevos: Cengel 10-2 completo, una planta con
    calentadores reales y ORC con R-245fa, tolueno y R-134a.
- **Validación**:
  - Cengel 10-2 con los estados de su figura: η = 36,10 % y
    Ẇ_neto = 18,87 MW (libro: 36,1 % y 18,9 MW); estados contra CoolProp a
    10⁻⁶.
  - DCA y desrecalentador contra los balances a mano; el sistema acoplado
    de un cerrado bombeado intermedio contra `fsolve` (10⁻⁶).
  - ORC con R-245fa contra CoolProp; recuperador con ε = q/q_máx.
  - Sin pérdidas, sin calentadores reales y con agua, los 24 casos del
    snapshot de la 0.12.0 (estados, red de TESPy, pasos, export y
    barridos) quedan idénticos.
- **Mensajes al alumno**:
  - Pérdidas negativas, condensado bajo el punto triple o que se
    congelaría, bomba fuera del rango de la ecuación de estado.
  - Cañería de vapor que pide a la caldera algo que no es vapor;
    extracción dentro del recalentador; caída en el recalentador sin
    recalentamiento.
  - DCA sin nada que enfriar; desrecalentador con extracción húmeda o con
    un TTD que dejaría el agua más caliente que la extracción; opciones de
    cerrado en un abierto.
  - Recuperador con escape húmedo, con calentadores o con ε fuera de
    (0, 1); fluido no disponible y límites de su ecuación de estado.
- **Tests**: de 1265 a 1740 (+475), sin warnings de pytest.
  - `tests/test_rankine_layout.py` (+172): numeración de 10-2, ORC con
    recuperador, bombeados intermedios y balances de masa simbólicos en
    una grilla de plantas reales.
  - `tests/test_rankine_losses.py` (35), `tests/test_rankine_heaters_real.py`
    (95, con una grilla de 78 plantas) y `tests/test_rankine_orc.py` (44).
  - AppTest de la página (42; 8 nuevos o cambiados) y los fluidos nuevos
    en los tests que recorren todos los fluidos.
  - LaTeX: validan con KaTeX estricto las 2427 expresiones distintas del
    procedimiento nuevo y las de la teoría. En Técnico e Inglés ninguna
    supera el ancho de un celular (máx. 308 y 321 px); en SI se deslizan
    las de ×10ⁿ (≤ 393 px).
  - Smoke test en Chromium a 1280 y 390 px: todas las páginas por link
    directo, Propiedades con tolueno y /Rankine con 10-2, calentadores
    reales, los tres ORC y 10-6.
- **Dependencias**: ninguna nueva.

### Fase 3.2 — Refrigeración por compresión de vapor
- **Versión**: `0.14.0` (2026-10-06). Rama `claude/water-state-analyzer-f4fiev`.
- **Commits**: `cc97baf` (feat: refrigeración en `core/cycles` y lo
  compartido en `core/`), `12999b5` (feat: página `/Refrigeracion` y
  navegación) y el commit que ajusta el ancho de las ecuaciones y cierra la
  fase con las docs.
- **Scope**:
  - **Ciclos** (Cengel 8.ª ed., cap. 11): simple ideal (§11-3) y real
    (§11-4: η_C, sobrecalentamiento, subenfriamiento y caídas de presión),
    de dos etapas con cámara de evaporación instantánea («economizador
    abierto», §11-8, ej. 11-5) y en cascada de dos etapas con uno o dos
    refrigerantes (§11-8, ej. 11-4). Como refrigerador o bomba de calor
    (§11-7). Red de TESPy como el tutorial de bomba de calor
    (`CycleCloser` + `SimpleHeatExchanger` + `Compressor` + `Valve`;
    `td_dew` / `td_bubble`); cámara = `DropletSeparator` + `Merge`;
    cascada = `HeatExchanger` entre dos lazos.
  - **Segundo principio** (Cengel §11-5; vademecum §11.5, §11.8 y §11.10):
    con las temperaturas de las fuentes, X_dest = T₀·S_gen de cada
    componente, trabajo mínimo, rendimiento exergético y COP reversible; el
    balance Ẇ = Ẇ_mín + ΣX_dest cierra. La página lo muestra en tabla y
    barras, con la interpretación física de cada componente en el
    procedimiento.
  - **Refrigerantes**: R-134a, R-1234yf, R-410A (pseudo-puro: rocío en el
    evaporador, burbuja en el condensador), amoníaco, CO₂ (solo subcrítico)
    y, nuevos en `core.fluids` (y en Propiedades, Interpolación e
    Isoentrópicos), **R-32, propano (R-290) e isobutano (R-600a)**.
  - **Tablas del R-134a**: Cengel usa h = s = 0 en el líquido saturado a
    −40 °C y CoolProp la referencia del IIR; `textbook_reference_offset` /
    `textbook_reference_note` lo avisan con las constantes (148,14 kJ/kg y
    0,7956 kJ/(kg·K)) en Propiedades, en el ORC con R-134a y en
    Refrigeración. La referencia de CoolProp no se toca (es global).
  - **Comparación** de la cámara y la cascada con el ciclo simple
    equivalente (COP, relación de presiones y descarga) y notas
    didácticas: compresión húmeda con fluidos secos, descarga muy caliente,
    evaporador bajo la presión atmosférica.
  - **Diagramas**: las válvulas se dibujan rayadas sobre su línea de h
    constante (también los drenajes del Rankine); una pestaña por
    refrigerante en la cascada; log p–h por defecto.
  - **Página `/Refrigeracion`**: niveles por presión o por temperatura de
    saturación, rótulos con la numeración de Cengel, ciclo real, tamaño
    por caudal o capacidad (con toneladas de refrigeración), segundo
    principio, TESPy, export y barridos (COP y descarga contra T_evap,
    T_cond, η_C, subenfriamiento y sobrecalentamiento; presión de la cámara
    y temperatura intermedia de la cascada, con su óptimo). Nueva magnitud
    `volume_flow` (caudal volumétrico aspirado).
  - **Ejemplos**: Cengel 11-1 a 11-5 y propios (bomba de calor aire–agua
    con R-410A, cascada CO₂/amoníaco a −45 °C, amoníaco en dos etapas con
    cámara, heladera con isobutano).
- **Validación** (contra el cálculo a mano con CoolProp a 10⁻⁷ y contra el
  libro):
  - 11-1: COP 3,968, Q̇_L 7,184 kW, Ẇ 1,811 kW (libro: 3,97; 7,18; 1,81);
    restando la referencia, h₁, h₂ y h₃ coinciden con las tablas A-12/A-13.
  - 11-2 con los estados del libro: COP 3,930 y η_C 0,937 (libro: 3,93 y
    0,939 con sus tablas).
  - 11-3: X_dest de compresor, condensador, válvula y evaporador 0,394,
    0,426, 0,673 y 0,409 kW; η_II 34,8 %.
  - 11-4: ṁ_B 0,03896 kg/s, COP 4,473 (libro: 0,0390 y 4,46).
  - 11-5: x₆ 0,2049, q_L 146,28 kJ/kg, w 32,69 kJ/kg, COP 4,475 (libro:
    0,2049; 146,3; 32,71; 4,47).
  - El Rankine queda idéntico al snapshot de la 0.13.0 (29 casos) salvo la
    nota del R-134a en el ORC.
- **Mensajes al alumno**: refrigerante no disponible; presiones
  invertidas, sobre el punto crítico (con la explicación del CO₂
  transcrítico) o bajo el punto triple; sobrecalentamiento, subenfriamiento
  o Δp negativos; líquido que se congelaría; válvula que tendría que subir
  la presión; cámara fuera de rango o sin vapor que separar; cascada con el
  calor al revés o con un ciclo desordenado; fuentes incompatibles con las
  temperaturas del refrigerante; cámara y cascada juntas.
- **Tests**: de 1740 a 1967 (+227), sin warnings de pytest.
  - `tests/test_refrigeration.py` (113): ejemplos contra el libro y a mano,
    balances de energía, masa y exergía de todos los ejemplos, bomba de
    calor, caudal o capacidad, cada refrigerante en los tres ciclos,
    R-410A, compresión húmeda, notas, numeración, 25 validaciones,
    barridos con óptimo interior, export, procedimiento y diagrama.
  - `tests/test_page_refrigeracion.py` (26, AppTest), navegación y los
    fluidos nuevos en los tests que recorren todos los fluidos.
  - LaTeX: validan con KaTeX estricto las 3287 expresiones distintas
    (30 casos × 3 sistemas, más la teoría); **ninguna supera el ancho de un
    celular en ningún sistema, SI incluido** (máx. 321 px: las restas con
    ×10ⁿ se parten antes del signo).
  - Smoke test en Chromium a 1280 y 390 px: todas las páginas por link
    directo, Propiedades con R-134a e isobutano, y /Refrigeracion con
    11-1, 11-3, 11-4, 11-5, la bomba de calor y la cascada CO₂/NH₃.
- **Dependencias**: ninguna nueva.

### Fase 3.3 — Caldera de recuperación (HRSG) de una presión
- **Versión**: `0.15.0` (2026-10-06). Rama `claude/water-state-analyzer-f4fiev`
  (pedido del autor: «el diagrama T–Q de una HRSG de una presión […] y lo
  mismo si producís vapor saturado»).
- **Scope**:
  - **`core/cycles/hrsg.py`** (sin TESPy): `FlueGas`, mezcla de gases
    ideales con N₂, O₂, CO₂, H₂O y Ar en fracción molar o másica
    (vademecum §5: M, wᵢ, R, h_g(T) desde 25 °C con la h de gas ideal de
    CoolProp, T(h), presión parcial y punto de rocío);
    `exhaust_composition(λ)`, los gases del metano con el aire técnico del
    vademecum (§16.1–§16.2); `HRSGInputs` / `solve_hrsg` / `HRSGResult`, con
    el balance de energía sección por sección de Kehlhofer et al. (2009):
    pinch a la salida de gases del evaporador, approach a la salida del
    economizador, caudal de vapor del tramo entre la entrada de los gases
    y el pinch, T de los gases tras el sobrecalentador y de chimenea, calor
    de cada sección, aprovechamiento contra 15 °C y c_p medio.
  - **Diagrama T–Q** (`hrsg_tq_profile`): curva de los gases (no recta: c_p
    crece con T) y del agua con el escalón del domo, de modo que el mínimo
    ΔT es el pinch; límites de las secciones.
  - **Vapor sobrecalentado o saturado** (sin sobrecalentador), con la
    comparación entre las dos opciones (`other_steam_option`: el
    sobrecalentado a 25 K bajo los gases, hasta 565 °C).
  - **`core/cycles/hrsg_procedure.py`**: composición, entalpía de los
    gases, estados del agua (A-5, A-6, A-7 y h_f(T)), caudal de vapor
    (Q̇ arriba del pinch), cada sección con la T de los gases que sale,
    calor total y aprovechamiento, y punto de rocío.
  - **Página `/HRSG`** (`app_pages/10_HRSG.py`, 🏭): gases (T, caudal,
    composición con λ o cargada, presión), agua y vapor, pinch y approach;
    se recalcula sola. Métricas, diagrama T–Q (plotly) con el pinch y el
    approach señalados, tablas de secciones (ΔT de cada extremo) y de
    estados del agua y de los gases, comparación saturado/sobrecalentado,
    procedimiento, export y barridos (ṁ_v y chimenea contra pinch,
    approach, presión, T del vapor, T del agua de alimentación y T de los
    gases, con la explicación de cada compromiso).
  - **Ejemplos**: escape típico de una turbina de gas a gas natural (600 °C,
    100 kg/s; vapor a 60 bar y 540 °C), vapor saturado para proceso
    (gases de metano con λ = 3 a 500 °C; 10 bar) y pinch chico (5 K).
- **Validación**:
  - Cálculo a mano independiente en los tests (PropsSI + brentq): igual a
    10⁻⁹.
  - **TESPy** como control cruzado (`HeatExchanger` en serie, gases como
    mezcla, como su tutorial de turbina de gas): ṁ_v, T tras el
    sobrecalentador y chimenea al 0,02 % con vapor sobrecalentado y
    saturado (turbina de gas: 15,385 kg/s, 503,55 °C y 153,05 °C; TESPy
    15,387 kg/s, 503,55 °C y 153,08 °C).
  - Balances de cada sección, mínimo del perfil igual al pinch, saturado
    contra sobrecalentado (más vapor y chimenea más fría), pinch chico
    contra grande.
- **Mensajes al alumno**: caudal o presión fuera de rango (supercrítica:
  sin domo), pinch no positivo, approach negativo, agua de alimentación
  más caliente que la salida del economizador, gases que no alcanzan
  T_sat + pinch, vapor más frío que la saturación o más caliente que los
  gases, cruce de temperaturas en el economizador o dentro de la caldera,
  chimenea bajo el punto de rocío, composición vacía o negativa. Notas:
  agua de alimentación bajo el punto de rocío, extremos con ΔT menor que
  el pinch, approach 0.
- **Tests**: de 1967 a 2038 (+71), sin warnings de pytest.
  - `tests/test_hrsg.py` (49): composición contra CoolProp, preset con λ,
    rocío, ejemplos a mano y con TESPy, balances, perfil T–Q, 11
    validaciones, notas, barridos, comparación, export y procedimiento.
  - `tests/test_page_hrsg.py` (21, AppTest): ejemplos, vapor saturado,
    λ, fracciones másicas (y el cambio de base, que conserva la
    composición cargada), normalización, errores, notas, unidades, teoría
    y barridos; la navegación recorre la página nueva.
  - LaTeX: validan con KaTeX estricto las 726 expresiones distintas
    (13 casos × 3 sistemas y la teoría); ninguna supera el ancho de un
    celular (máx. 304 px en SI y Técnico, 293 px en Inglés).
  - Smoke test en Chromium a 1280 y 390 px por link directo: los tres
    ejemplos y la turbina de gas con vapor saturado, con el procedimiento
    y la teoría abiertos; sin ecuaciones con scroll ni desborde.
- **Dependencias**: ninguna nueva.

---

## Pendientes / próximas fases

- **Detectado en Fase 1.7, sin resolver**:
  - En sistema SI (J/kg con ×10ⁿ) algunas sustituciones siguen siendo más
    anchas que el celular (regla de la palanca, h₂ de Isoentrópicos: hasta
    475 px; en Rankine, la h de la turbina real, la bomba de los ORC y las
    sumas del ciclo regenerativo: ≤ 393 px) y se deslizan;
    en Inglés, 5 apenas pasadas (≤ 339 px). Se resolvería mostrando las
    energías en kJ/kg dentro del SI o con otro formato de número (decisión
    del autor: hoy el SI es J/kg).
  - Pseudo-puros con T y (s o v): dentro de la campana CoolProp los
    resuelve a una sola presión entre la de burbuja y la de rocío (es su
    modelo; el título no coincide con la palanca a T constante), y justo
    en el borde los clasifica como líquido comprimido / vapor
    sobrecalentado. Con p ya está resuelto; con T es un caso de borde raro.
- **Fase 2.3 (continuación)** — Matriz de normalización ISO 6976
  cuando se incorpore ISO 14912:2003 Formula (69).
- **HRSG (continuación)**: dos y tres presiones, recalentamiento,
  quemadores suplementarios, pérdidas de carga y purga, diseño de las
  superficies (UA, NTU), condensación ácida, y el ciclo combinado completo
  sobre esta base y la del Rankine.
- **Fase 3.x** — Brayton; ciclo combinado; exergía de los ciclos de
  potencia. Evaluar `tespy.tools.get_plotting_data` para los diagramas
  (hoy el ciclo se dibuja con `core.diagrams.segments_overlays`).
- **Refrigeración (continuación)**: ciclo transcrítico de CO₂,
  intercambiador líquido–vapor, economizador cerrado (subenfriador),
  refrigeración por gas y por absorción.
- **Fase 4** — Psicrometría (carta interactiva, procesos HVAC).
- **Fase 5** — Combustión: estequiometría, exceso de aire, productos
  de combustión, temperatura adiabática de llama.
- **Fase 6** — Poder calorífico por composición última (Dulong, Boie,
  Channiwala-Parikh) y próxima (Parikh).
- **Fase 7** — Exergía física y química (Szargut), diagrama de
  Grassmann por componente.
- **Fase 8** — Transferencia de calor: conducción, aletas, convección,
  radiación, intercambiadores (LMTD, ε-NTU).

---

## Convenciones del workflow

- **Versionado**: bump al cierre de cada fase. Sincronizar
  `CITATION.cff`, `streamlit_app.py:PAGE_VERSION` y la version label
  de cada `app_pages/N_*.py` afectada.
- **Checklist de cierre de fase** (todas obligatorias):
  1. `ruff check .` + `ruff format .` limpios.
  2. `pytest -v` en verde, incluyendo tests nuevos del módulo.
  3. Smoke test con `streamlit run streamlit_app.py` (esperar OK
     explícito del usuario antes del commit).
  4. Bump de versión + actualización de `CITATION.cff`.
  5. Actualización de `README.md` (mover el módulo a "Estables" o
     refinar la descripción) y `CLAUDE.md` (marcar archivos con ✅).
  6. **Actualización de este archivo** (`ESTADO_PROYECTO_TA216.md`)
     con la nueva entrada de la fase.
  7. Commit con mensaje convencional `feat:` / `fix:` / `refactor:` /
     `chore:` y `Co-Authored-By: Claude <noreply@anthropic.com>`.
  8. `git push origin main` (esperar OK explícito del usuario).
- **Regla de arquitectura**: `core/` no importa Streamlit. Si un
  helper necesita Streamlit, vive en `ui/`.
- **Unidades**: dentro de `core/` todo es SI. La conversión a
  Técnico/Inglés vive en la frontera UI (`ui/units_ui.py` para
  inputs, `format_quantity` para outputs).
