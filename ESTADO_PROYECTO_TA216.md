# Estado del proyecto — TA216 Apps

> Bitácora viva de qué está cerrado, qué quedó deferido y qué viene.
> Se actualiza al cierre de cada fase como parte del checklist (junto
> con el bump de versión, el README y el `CITATION.cff`). Generada a
> partir del `git log` y la documentación interna del proyecto.
>
> **Versión actual**: `0.25.0` — Fase 8.2 cerrada (2026-10-09).

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
  - **Ciclo real** (Cengel §10-3): `Losses` con caídas de presión en la
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
    (desrecalentador, subenfriador, bombeado, Δp) y teoría de §10-3 y del
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
- **Corrección en el Rankine** (confirmada por el autor): el ciclo real
  citaba Cengel §10-5, que es el recalentamiento; el ciclo real con sus
  pérdidas es el §10-3. Se corrigió en `core/`, la página, el README y
  este archivo, y el recalentamiento ahora cita §10-5 en la teoría, en la
  leyenda del procedimiento y en el paso del recalentador. Tests nuevos
  en `tests/test_rankine.py` y `tests/test_page_rankine.py`.
- **Dependencias**: ninguna nueva.

### Fase 3.4 — Turbina de gas y ciclo combinado de una presión
- **Versión**: `0.16.0` (2026-10-06). Rama `claude/water-state-analyzer-f4fiev`
  (pedido del autor: el ciclo combinado antes que la HRSG de varias
  presiones, «la hrsg esa dejala para despues, primero lo otro»).
- **Scope**:
  - **`core/ideal_gas.py`**: el modelo de mezclas de gases ideales se mudó
    desde `hrsg.py` (que lo reexporta) y suma la función s°(T), la entropía
    absoluta con las S° de NIST-JANAF a 25 °C y 1 bar (para dibujar el aire
    y los gases en el mismo T–s), la T de una compresión o expansión
    isoentrópica (s°₂ = s°₁ + R·ln(p₂/p₁), como la tabla A-17 de Cengel),
    el aire seco (con Ar y CO₂, el de las tablas de Cengel) y el técnico
    (vademecum §16.1), y los productos de combustión con cualquier aire.
  - **`core/cycles/brayton.py`**: turbina de gas (compresor, cámara y
    turbina) con calores específicos variables. Combustión de metano (o de
    una mezcla de componentes de ISO 6976) con el PCI de ISO 6976:2016 a
    25 °C; el balance de la cámara da f, λ y la composición de los gases.
    Modo aire estándar (Cengel §9-3). Validaciones, notas, líneas del T–s
    y la misma turbina resuelta con **TESPy** (`Compressor`,
    `DiabaticCombustionChamber`, `Turbine`) como control.
  - **HRSG por temperatura de chimenea** (Cengel 10-9): el caudal de vapor
    sale del balance de toda la caldera y el pinch es un resultado.
  - **`core/cycles/combined.py`**: turbina de gas + HRSG + el Rankine de la
    app (con desaireador opcional; el agua de alimentación es la que sale
    de la bomba). Rendimientos de cada parte y del conjunto, η_HRSG,
    relación de Kehlhofer, balance de energía, heat rate, dimensionado por
    caudal de aire o por potencia neta, notas, tres ejemplos, barridos y
    export.
  - **Procedimientos**: `brayton_procedure.py` (aire, compresor con s°,
    cámara con el PCI, el aire teórico, f y λ, gases, turbina, rendimiento
    y potencias) y `combined_procedure.py` (turbina de gas, HRSG, ciclo de
    vapor y el acople con el rendimiento del ciclo combinado).
  - **Página `/Ciclo_Combinado`** (`app_pages/12_Ciclo_Combinado.py`, ⚡):
    datos de la turbina de gas (metano o aire estándar, aire seco o
    técnico), de la HRSG (sobrecalentado o saturado, por pinch o por
    chimenea) y del ciclo de vapor; botón «Calcular». Métricas, notas,
    diagrama de **Sankey** de la energía, **T–s** de la turbina de gas con
    el enfriamiento en la HRSG, **T–Q** de la HRSG, diagrama y estados del
    ciclo de vapor, procedimiento por partes, control con TESPy, export y
    barridos (relación de presiones, TIT, presión del vapor, pinch o
    chimenea).
  - **`ui/cycle_charts.py`**: el T–Q de /HRSG y el diagrama de /Rankine se
    mudaron ahí (las dos páginas no cambian) junto con los gráficos nuevos.
  - **Ejemplos**: planta típica (r_p 15, TIT 1250 °C, 500 kg/s de aire;
    vapor a 60 bar y 540 °C, desaireador a 1,5 bar): η_TG = 39,27 %,
    η_TV = 36,73 %, **η_CC = 55,61 %** (216 + 90 MW); Cengel 10-9 (aire
    estándar, chimenea a 450 K): y = 0,1313 y η = 48,70 % (libro: 0,131 y
    48,7 %); turbina moderna (r_p 17, TIT 1400 °C, 100 bar y 565 °C):
    η_CC = 60,52 %.
- **Validación**:
  - Gas ideal contra Cengel A-18 a A-23 (Δh̄ de 298 a 1000 K, ≤ 0,085 %) y
    s° de NIST-JANAF a 1000 K (±0,02 J/(mol·K)).
  - Cengel 9-5: T₂ = 539,8 K, T₄ = 770,4 K, r_bw = 0,402, η = 42,5 %
    (libro: 540 K, 770 K, 0,403, 42,6 %); 9-6: r_bw = 0,592, η = 26,61 %,
    T₄ = 853,0 K (libro: 0,592, 26,6 %, 853 K).
  - Turbina de gas contra un cálculo a mano independiente (10⁻⁷) y contra
    TESPy (≤ 0,3 %: el gas real a presiones altas). PCI del metano:
    50,027 MJ/kg (Cengel A-27: 50,05).
  - Ciclo combinado: el balance de energía y la relación de Kehlhofer
    cierran a 10⁻⁹; el barrido de r_p tiene un máximo interior (r_p ≈ 20 en
    el típico) mientras la turbina de gas sola sigue mejorando.
- **Mensajes al alumno** (con el nombre de la parte): TIT por debajo de la
  salida del compresor, combustión que pediría λ < 1, relación de
  presiones, rendimientos, caída de presión, ambiente o caudal fuera de
  rango, combustible que no está en ISO 6976 o que no se quema, cruce de
  temperaturas en la HRSG por chimenea, vapor más caliente que el escape,
  desaireador fuera del rango de presiones, potencia no positiva.
- **Tests**: de 2056 a **2169** passed (10 skipped), sin warnings de pytest.
  - `tests/test_ideal_gas.py` (15), `tests/test_brayton.py` (41),
    `tests/test_combined.py` (24) y 9 nuevos en `tests/test_hrsg.py`
    (modo chimenea).
  - `tests/test_page_ciclo_combinado.py` (23, AppTest): ejemplos, botón,
    aire estándar y técnico, vapor saturado, sin desaireador, diseño por
    chimenea, potencia neta, errores, unidades, control con TESPy, teoría,
    export y barridos; la navegación recorre la página nueva.
  - LaTeX: KaTeX estricto valida las 2592 expresiones distintas (8
    turbinas de gas y 13 ciclos combinados × 3 sistemas, la HRSG por
    chimenea y la teoría); lo nuevo entra en 306 px como máximo. Los pasos
    del ciclo de vapor en SI son los del Rankine y se siguen deslizando
    (≤ 384 px, como en /Rankine).
  - Smoke test en Chromium a 1280 y 390 px por link directo: los tres
    ejemplos y el típico en SI, con el procedimiento, la teoría, el control
    con TESPy y el diagrama del ciclo de vapor abiertos; sin desborde de la
    página. El Sankey lleva los rótulos en varios renglones para que no se
    pisen en el celular.
- **Dependencias**: ninguna nueva. `CITATION.cff` y el README suman NIST-JANAF
  (Chase, 1998) y Kehlhofer et al. (2009) como referencias. A la referencia
  de ISO 6976 del `CITATION.cff` le faltaba `authors` (obligatorio en CFF
  1.2.0): ahora el archivo valida contra el esquema.

### Fase 3.5 — HRSG de dos y tres presiones (y anchos en SI)
- **Versión**: `0.17.0` (2026-10-07). Rama `claude/water-state-analyzer-f4fiev`
  (el autor la había dejado para después del ciclo combinado; «dale mergea
  y seguí»).
- **Antes, ecuaciones en SI al ancho del celular** (`7b84e37`, con el OK del
  autor): ver «Detectado en Fase 1.7» abajo. Rankine, Propiedades e
  Isoentrópicos entran en 324 px en los tres sistemas; Técnico e Inglés del
  Rankine idénticos.
- **Scope**:
  - **`core/cycles/hrsg_multi.py`**: niveles de presión (`PressureLevel`, de
    alta a baja) en **cascada**: los gases los recorren de alta a baja; el
    economizador de baja calienta toda el agua y cada domo manda el líquido
    saturado que no evapora a la bomba (isoentrópica) del nivel siguiente.
    Los caudales salen nivel por nivel del balance hasta cada pinch, con el
    calor del domo del agua que sube (Kehlhofer et al., 2009, cap. 5).
    `from_single` pasa la caldera de una presión al modelo nuevo (mismos
    números).
  - **Diagrama T–Q** con la curva del agua en serrucho (`multi_tq_profile`),
    **exergía** (`hrsg_exergy`: la de los gases = la que gana el agua + la
    destruida en cada sección + la de la chimenea; η_II), **comparación**
    de la misma caldera con 1, 2 y 3 niveles (`level_comparison`), notas,
    cuatro ejemplos, barridos (presión de baja y de alta, pinch, agua de
    alimentación) y export.
  - **`core/cycles/hrsg_multi_procedure.py`**: por nivel, los estados del
    agua (1 a 5, con el nivel de subíndice), el caudal (con el calor del
    domo), sobrecalentador, evaporador y economizador; el total, la exergía
    y el punto de rocío. `exergy_step` también se suma al procedimiento de
    la caldera de una presión.
  - **Página `/HRSG`**: radio **«Niveles de presión»** (1, 2 o 3). Con 1 todo
    queda como antes, más la exergía (métricas y barras por sección). Con 2
    o 3: un bloque por nivel (presión, vapor sobrecalentado o saturado,
    pinch y approach), métricas por nivel, T–Q (un color por nivel y el
    pinch de cada uno), secciones, **«¿Cuánto ganás con más presiones?»**
    (tabla y barras de exergía de 1, 2 y 3 niveles), exergía destruida por
    sección, estados, procedimiento, export, barridos y teoría (cascada y
    exergía; vademecum §11).
  - **Ejemplos**: dos presiones (80 bar y 540 °C + 6 bar y 200 °C), tres
    (100 / 20 / 4 bar) y una turbina moderna (escape a 640 °C) con dos y
    con tres presiones.
- **Validación** (escape de 600 °C y 100 kg/s, agua a 60 °C):

  | Caldera | Vapor (kg/s) | Chimenea | Aprovechamiento | Ẋ destruida | η_II |
  |---|---|---|---|---|---|
  | 1 presión (60 bar) | 15,38 | 153 °C | 77,5 % | 4,31 MW | 76,6 % |
  | 2 presiones (80/6 bar) | 15,27 + 2,33 | 103 °C | 85,7 % | 3,73 MW | 83,5 % |
  | 3 presiones (100/20/4 bar) | 15,22 + 1,73 + 1,06 | 97 °C | 86,6 % | 3,19 MW | 85,8 % |

  - Un nivel = la caldera de la Fase 3.3, exacto.
  - Cálculo a mano independiente (PropsSI + brentq) a 10⁻⁷.
  - **TESPy** (`HeatExchanger` en serie; cada domo, un `DropletSeparator` y
    una `Pump`): al 0,12 % en caudales y 0,04 K en la chimenea.
  - Balances de energía de cada sección y de exergía (cierre a 10⁻⁹).
- **Mensajes al alumno** que nombran el nivel: presiones que no bajan o
  demasiado cerca, gases que ya no alcanzan T_sat + pinch (o solo alcanzan
  para el domo), vapor más caliente que los gases que le llegan (el límite
  de la cascada), cruces en un economizador, punto de rocío.
- **Tests**: de 2185 a 2253 passed (10 skipped), sin warnings.
  - `tests/test_hrsg_multi.py` (51) y `tests/test_page_hrsg_multi.py` (17,
    AppTest); `tests/test_page_hrsg.py` cuenta el gráfico de exergía.
  - LaTeX: las 1212 expresiones distintas de la Fase 3.5 validan con KaTeX
    estricto y entran en 313 px como máximo.
  - Smoke test en Chromium a 1280 y 390 px: 1 presión, los cuatro ejemplos
    de 2 y 3 presiones y 3 presiones en SI, con el procedimiento y la teoría
    abiertos; sin ecuaciones con scroll ni desborde.
- **Dependencias**: ninguna nueva.

### Fase 3.6 — Ciclo combinado de dos y tres presiones con recalentamiento
- **Versión**: `0.18.0` (2026-10-07). Rama `claude/water-state-analyzer-f4fiev`
  (plan enviado y aprobado: «dale mergea y seguí»; la regla de Baumann va
  apagada por defecto, como se propuso).
- **Scope**:
  - **`core/cycles/hrsg_multi.py`**: recalentador opcional (`Reheater`) **en
    paralelo** con el sobrecalentador de alta (los dos bancos ven los gases de
    entrada y salen a una T común). Con tres niveles el vapor de media se suma
    al recalentamiento frío y los caudales de alta y media salen de un
    **sistema lineal de 2×2** (`reheat_system`, Cramer). T–Q con los dos
    bancos en el mismo tramo, exergía por banco, validaciones que nombran el
    recalentador y rendimiento de las bombas entre niveles (`eta_pump`). Sin
    recalentador, idéntico a la 0.17.0.
  - **`core/cycles/combined_multi.py`** (nuevo): turbina de gas + HRSG en
    cascada + **turbina de vapor con admisiones** (el vapor de cada nivel se
    mezcla a la presión de su domo), recalentamiento, desaireador opcional y
    **regla de Baumann** (η_húmedo = η_T·(1 − α·ȳ) en la parte húmeda).
    Estados numerados como Cengel (1–4, 1–6 con recalentamiento, 1–7 con
    desaireador). `configuration_comparison` (1/2/3 presiones, con y sin
    recalentamiento, con η_T constante y con Baumann), `bottoming_exergy`
    (exergía del ciclo de fondo por componente), notas, siete ejemplos (uno o
    dos por combinación), barridos (presiones de alta, media, baja y de
    recalentamiento, T de recalentamiento, pinch, r_p y TIT), export y
    `combined_multi_tespy` (control con TESPy de todo el lado agua–vapor).
  - **`core/cycles/combined_multi_procedure.py`**: por partes (turbina de gas,
    HRSG con el recalentador y el 2×2, ciclo de vapor con cada turbina, las
    mezclas, el recalentador, Baumann, el desaireador, el condensador y las
    potencias; el acople, el rendimiento de Kehlhofer, el balance de energía y
    la exergía del ciclo de fondo).
  - **Página `/Ciclo_Combinado`**: radio **«Niveles de presión»** y casilla
    **«Recalentamiento»**; con una presión y sin recalentar, la 0.17.0 sin
    cambios. Si no: un bloque por nivel, el recalentamiento, la casilla de
    Baumann; métricas por nivel y el título a la salida de la turbina,
    Sankey, T–s de la TG, T–Q con el recalentador, **T–s del ciclo de vapor
    con las admisiones**, tablas de estados y de turbinas, **«¿Cuánto ganás
    con más presiones y recalentamiento?»** (tabla y gráfico de puntos),
    **exergía del ciclo de fondo** (barras), procedimiento, control con TESPy
    (turbina de gas y lado agua–vapor), export, barridos (η y título) y
    teoría.
  - Gráficos de exergía y de la comparación con la paleta de referencia
    validada para daltonismo (también en /HRSG: el verde y el naranja de las
    barras de la 0.17.0 no se distinguían con protanopía).
- **Validación** (turbina de gas moderna, escape a 675 °C; alta 120 bar y
  565 °C, media 25 bar y baja 4 bar saturadas, recalentamiento a 565 °C,
  pinch 8 K, condensador 0,06 bar, η_T 0,90, η_B 0,80, sin desaireador):

  | Configuración | η_CC | η_CC con Baumann | x salida | Chimenea | Ẇ_TV / Ẋ_gases |
  |---|---|---|---|---|---|
  | 1 presión | 61,60 % | 61,32 % | 0,852 | 103 °C | 69,3 % |
  | 1 presión + RH | 61,11 % | 61,15 % | 0,948 | 150 °C | 67,6 % |
  | 2 presiones | 62,52 % | 61,90 % | 0,843 | 67 °C | 72,6 % |
  | 2 presiones + RH | 62,63 % | 62,56 % | 0,924 | 77 °C | 73,0 % |
  | 3 presiones | 62,78 % | 62,10 % | 0,837 | 67 °C | 73,5 % |
  | 3 presiones + RH | 62,94 % | 62,89 % | 0,932 | 79 °C | 74,1 % |

  - Una presión sin recalentar = el ciclo de la Fase 3.4 (con el Rankine de
    TESPy) a 1e-9, con y sin desaireador.
  - **TESPy** de todo el lado agua–vapor (banco paralelo con `Splitter` +
    `Merge` de gases): al 0,2 % en caudales, 0,02 % en Ẇ_TV y 0,1 K en la
    chimenea, en 11 configuraciones (con desaireador, Baumann, media y baja
    sobrecalentadas, alta saturada).
  - Recorrido del vapor a mano con PropsSI (1e-9), el 2×2 con numpy (1e-7) y
    los balances de energía, masa y exergía (cierran a ~1e-7 W).
- **Mensajes al alumno**: recalentamiento fuera de rango o a otra presión que
  la de media, recalentador que no calienta o más caliente que los gases,
  cruce en el banco paralelo, desaireador que calienta más que el
  economizador de baja o justo en una admisión, factor de Baumann, con el
  nombre de la parte.
- **Tests**: de 2253 a 2404 passed (10 skipped), sin warnings.
  - `tests/test_combined_multi.py` (101), `tests/test_page_ciclo_combinado_multi.py`
    (26, AppTest) y 24 nuevos en `tests/test_hrsg_multi.py`.
  - LaTeX: 4007 expresiones distintas validan con KaTeX estricto y entran en
    320 px como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: una presión (la 0.17.0), las
    cinco combinaciones nuevas y tres presiones con recalentamiento en SI,
    con el procedimiento, la teoría y el diagrama abiertos; sin ecuaciones con
    scroll ni desborde.
- **Dependencias**: ninguna nueva.

### Fase 3.7 — Turbina de gas: Brayton con interenfriamiento, recalentamiento y regenerador
- **Versión**: `0.19.0` (2026-10-07). Rama `claude/water-state-analyzer-f4fiev`
  («sí, mergeá y seguí con todo»: el plan se mandó y se implementó sin esperar,
  como en las fases anteriores).
- **Scope**:
  - **`core/cycles/gas_turbine.py`** (nuevo): generaliza la turbina de gas de la
    Fase 3.4 (que sigue siendo la del ciclo combinado) con las tres mejoras de
    Cengel §9-9 y §9-10: **compresión en etapas con interenfriamiento**
    (relaciones iguales, el mínimo trabajo del vademecum §6.3, o presiones a
    elección), **expansión en etapas con recalentamiento** (con metano, una
    segunda cámara que quema en los gases: combustión secuencial, con la
    composición por el F acumulado; con aire estándar, un intercambiador) y
    **regenerador** con la efectividad de Cengel (con combustión, T₅ se itera
    con la secante). Pérdidas de carga en cámaras, interenfriadores y
    regenerador; tamaño por caudal o por potencia neta. Estados con la
    numeración de Cengel (1–4, 1–6 con regenerador, en el sentido del flujo con
    etapas) y su descripción. `from_brayton` (la 3.4, idéntica a 1e-12),
    `gas_turbine_exergy` (combustible ≈ PCI, vademecum §16.13),
    `improvement_comparison` (simple, regenerador, interenfriamiento,
    recalentamiento y combinaciones), notas, nueve ejemplos, barridos (r_p,
    TIT, ε, T ambiente, presión intermedia y cantidad de etapas), líneas del
    T–s, export y `gas_turbine_tespy` (control).
  - **`core/cycles/gas_turbine_procedure.py`** (nuevo): presiones de cada
    etapa, compresores con la función s°, interenfriadores, cámaras (y el
    recalentamiento con Δh, f_j, F y λ) seguidas de su turbina, el regenerador
    después de la turbina (como el 9-7), el rendimiento (con Carnot si están
    las tres mejoras), las potencias y la exergía por componente con su balance.
  - **Página `/Brayton`** (nueva, «Brayton» en el menú, antes de HRSG): datos
    por bloques (turbina, interenfriamiento, recalentamiento, regenerador,
    tamaño), botón «Calcular», métricas (η, w_neto, r_bw, Ẇ, escape, λ), notas,
    T–s con todas las etapas, tablas de estados y de componentes, **«¿Cuánto
    ganás con cada mejora?»** (tabla y gráfico de puntos en dos paneles, η y
    w_neto), **exergía por componente** (barras), procedimiento, control con
    TESPy, export, barridos y teoría.
  - **Arreglo** (`core/ideal_gas.py`): bajo la T mínima de CoolProp (el agua
    bajo 0,01 °C) cada componente sigue como gas ideal con c_p constante. Con
    el ambiente bajo cero, la exergía del ciclo combinado de la 0.18.0 fallaba
    con un error crudo de CoolProp.
- **Validación** (aire estándar, tabla A-17):

  | Caso | App | Cengel |
  |---|---|---|
  | 9-5: Brayton ideal, r_p 8 | η 42,54 %, r_bw 0,402 | 42,6 %, 0,403 |
  | 9-6: real (η_C 0,80, η_T 0,85) | η 26,61 %, r_bw 0,592 | 26,6 %, 0,592 |
  | 9-7: regenerador ε = 0,80 | η 36,88 %, q_reg 220,4 kJ/kg | 36,9 %, 220,0 kJ/kg |
  | 9-8: 2 + 2 etapas | η 35,73 %, r_bw 0,304 | 35,8 %, 0,304 |
  | 9-8: con regenerador ideal | η 69,58 % | 69,6 % |

  - Con una etapa y sin regenerador = la turbina de la Fase 3.4 (combustión y
    aire estándar), a 1e-12.
  - **TESPy** en diez configuraciones (aire estándar y metano, etapas,
    regenerador, combustión secuencial, aire técnico): dentro de 0,05 puntos de
    η, 0,3 % en f y 2,5 K en el escape.
  - Balances de energía, masa y exergía: cierran a 1e-9.
  - Barridos: p_x óptima = √(p₁p₂); con regenerador, η máximo a r_p ≈ 4,5 y el
    corte en T₄ = T₂; con todo ideal, de 59,8 % (1 etapa) a 75,3 % (8 etapas)
    contra 76,9 % de Carnot (Ericsson).
- **Mensajes al alumno**: etapas fuera de rango, presiones intermedias que no
  crecen o no bajan, interenfriamiento que no enfría, recalentamiento que no
  calienta, efectividad fuera de (0, 1], regenerador que no sirve (T₄ ≤ T₂),
  λ < 1 en una cámara de recalentamiento (la nombra), TIT por debajo del aire
  que llega a la cámara.
- **Tests**: de 2404 a 2576 passed (10 skipped), sin warnings.
  - `tests/test_gas_turbine.py` (100), `tests/test_gas_turbine_procedure.py`
    (37), `tests/test_page_brayton.py` (31, AppTest), más la exergía bajo cero
    (`test_ideal_gas.py`, `test_combined_multi.py` y
    `test_page_ciclo_combinado_multi.py`).
  - LaTeX: 3299 expresiones distintas validan con KaTeX estricto y entran en
    316 px como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: los nueve casos (siete ejemplos,
    la secuencial en SI y el 9-8 b en Inglés), con el procedimiento, la teoría
    y TESPy abiertos; sin ecuaciones con scroll ni desborde.
- **Dependencias**: ninguna nueva.

### Fase 4 — Psicrometría: aire húmedo, procesos de acondicionamiento y torre
- **Versión**: `0.20.0` (2026-10-08). Rama `claude/water-state-analyzer-f4fiev`
  («sí, mergeá y seguí con todo»: el plan se mandó y se implementó sin esperar).
- **Scope**:
  - **`core/psychrometrics.py`** (nuevo): el modelo del vademecum §14 (gas
    ideal, c_p constantes) con p_vs de IAPWS (sobre líquido con CoolProp, la
    tabla A-4; sobre hielo con la sublimación de IAPWS 2011, la A-8). Estado
    con la presión y **13 pares de datos** (T, φ, T_bh, T_pr, ω y h), bulbo
    húmedo = saturación adiabática (de hielo bajo 0 °C), rocío o escarcha, μ,
    v, ρ, R, c_p, s y **exergía** ψ_tm + ψ_qu (con el ambiente que se elija),
    presión por altura (atmósfera estándar de ASHRAE), masas en un recinto,
    comparación con `HAPropsSI` (RP-1485), líneas y grilla de la carta, notas,
    ejemplos (Cengel 14-1 a 14-4, Buenos Aires y La Quiaca), barrido de la
    altura y export.
  - **`core/hvac.py`** (nuevo): tren de hasta 4 procesos (§14.12:
    calentamiento o enfriamiento sensible, calentamiento con humidificación,
    serpentín de enfriamiento y deshumidificación, humidificación adiabática y
    mezcla) numerado en el sentido del flujo, con el calor, el agua y la
    **exergía destruida** de cada proceso (la del agua, de Wepfer et al. 1979;
    la del calor, con la temperatura de la fuente o del serpentín); **torre de
    enfriamiento** (caudal de aire, reposición, rango, aproximación,
    efectividad, L/G, exergía); notas, ejemplos (Cengel 14-5 a 14-9, verano con
    recalentamiento, invierno en Bariloche, condensador de 100 MW), barridos y
    export.
  - **`core/psychrometrics_procedure.py`** (nuevo): el procedimiento del
    estado (presión por altura, los datos del par, ω, φ, μ, h en tres pasos, v,
    ρ, rocío, bulbo húmedo con su verificación, s y ψ), de cada proceso
    (balances de masa y energía y exergía) y de la torre.
  - **`ui/psychro_chart.py`** (nuevo): la **carta psicrométrica** en plotly a
    cualquier presión (φ, h, v y T_bh constantes, zona de confort de Cengel
    §14-6), con los estados y los procesos en la paleta validada. Una grilla
    invisible muestra el estado de cualquier punto al pasar el mouse y, al
    tocarlo, lo carga como dato.
  - **Página `/Psicrometria`** (nueva, después de Refrigeración): tres modos
    (estado, procesos y torre), con recalculo automático, notas, carta,
    tablas, comparación con CoolProp, exergía, procedimiento, teoría, export y
    barridos.
- **Validación** (modelo del vademecum; el libro lee la carta):

  | Ejemplo | App | Cengel |
  |---|---|---|
  | 14-1: 25 °C, 75 %, 100 kPa | p_a 97,62 kPa; ω 0,01515; m_a 85,56 kg | 97,62 kPa; 0,0152; 85,61 kg |
  | 14-2: 20 °C, 75 % | T_pr 15,44 °C | 15,4 °C |
  | 14-3: psicrómetro 25 y 15 °C | ω 0,006526; φ 33,2 %; h 41,75 kJ/kg | 0,00653; 33,2 %; 41,8 |
  | 14-4: 35 °C, 40 % | ω 0,01414; T_bh 23,9 °C; v 0,8927 | 0,0142; 24 °C; 0,893 |
  | 14-5: calentar y humidificar | Q̇ 668 kJ/min; ṁ_w 0,538 kg/min | 673; 0,539 |
  | 14-6: serpentín | Q̇ 511 kJ/min; ṁ_w 0,131 kg/min | 511; 0,131 |
  | 14-7: enfriador evaporativo | T₂ 70,3 °F | 70 °F |
  | 14-8: mezcla | ω₃ 0,01216; φ₃ 88,8 %; T₃ 18,9 °C | 0,0122; 89 %; 19,0 °C |
  | 14-9: torre | reposición 1,80 kg/s | 1,80 kg/s |

  - IAPWS 2011: 230 K → 8,94735 Pa; continuidad en el punto triple.
  - ASHRAE: 1000 m → 89,875 kPa; 3000 m → 70,108 kPa.
  - CoolProp (RP-1485): a 1 atm ω +0,4 % (factor de mejora), T_bh y T_pr
    < 0,02 K, v < 0,1 % (de −30 a 45 °C); a 7 bar, ω_s 2 %.
  - Ida y vuelta de los 13 pares en una grilla (de −30 a 150 °C, de 0,58 a
    7 bar): T a 1e-6 K y ω a 1e-7.
  - Exergía: X_dest (balance con ψ) = T₀·S_gen a 1e-10 en los procesos sin
    agua y a 3e-4 con agua (los redondeos 0,622 y 1,608); X_dest > 0 en todos
    los procesos y en la torre.
- **Mensajes al alumno**: φ > 100 % (niebla), T_bh o T_pr por encima de T,
  humedad negativa, vapor por encima de la presión total («el agua hierve a…»),
  pares que dicen lo mismo, enfriamiento sensible bajo el rocío, serpentín que
  no condensa o que escarcharía, fuente de calor más fría que el aire,
  humidificador que no humidifica, mezcla en la niebla, torre con el agua bajo
  el bulbo húmedo del aire o el aire más caliente que el agua; los errores de
  un tren nombran el proceso.
- **Tests**: de 2576 a 2921 passed (10 skipped), sin warnings.
  - `tests/test_psychrometrics.py` (181), `tests/test_hvac.py` (59),
    `tests/test_psychrometrics_procedure.py` (62) y
    `tests/test_page_psicrometria.py` (42, AppTest), más la página en
    `test_navigation.py`.
  - LaTeX: 3466 expresiones distintas (procedimiento y teoría) validan con
    KaTeX estricto y entran en 316 px como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: los tres modos en siete casos
    (con SI e Inglés), con el procedimiento y la teoría abiertos; sin
    ecuaciones con scroll ni desborde. Hover y toque de la carta: el punto
    tocado queda cargado como dato.
- **Dependencias**: ninguna nueva (CoolProp ya trae `HAPropsSI`).

### Fase 5 — Combustión: estequiometría, humos, llama adiabática y calor
- **Versión**: `0.21.0` (2026-10-08). Rama `claude/water-state-analyzer-f4fiev`
  («mergeá»: se mergeó el PR de la Fase 4 y la Fase 5 se planeó e implementó
  sin esperar, como en las fases anteriores).
- **Scope**:
  - **`data/nasa9_thermo.csv`** y **`scripts/extract_nasa9.py`** (nuevos): los
    polinomios NASA de 9 coeficientes de McBride, Zehe y Gordon (2002), la
    fuente de la tabla del vademecum §16.12, extraídos de `thermo.inp` de NASA
    CEA (Apache 2.0): 32 especies (gases de combustión, radicales,
    combustibles gaseosos y líquidos) en 60 tramos de 200 a 6000 K.
  - **`core/combustion/thermo.py`** (nuevo): c_p, h = h_f + Δh, s° (1 bar) y
    g° de cada especie; mezclas; h_fg del agua a 25 °C.
  - **`core/combustion/fuels.py`** (nuevo): mezclas gaseosas, líquidos puros
    (con propano y butano líquidos: el gas menos el h_fg de CoolProp) y
    análisis elemental con el PCS como dato (h_f desde el PCS; PCI = PCS −
    agua·h_fg); biblioteca de 14 combustibles.
  - **`core/combustion/stoichiometry.py`** (nuevo): aire técnico, seco u O₂
    con la humedad del ambiente; la cantidad de aire de seis formas (λ, e, %
    teórico, φ, AC u O₂ en humos secos); productos completos, con CO o con
    defecto de aire (Cengel 15-8 c, con el error de hollín); humos en base
    húmeda, seca y másica, M_g, GC, volumen normal, CO₂ máximo, rocío del agua y
    **ácido** (Verhoff y Banchero, 1974); Orsat (Cengel 15-4).
  - **`core/combustion/equilibrium.py`** (nuevo): equilibrio químico por
    minimización de la energía libre de Gibbs con potenciales de elementos,
    como NASA CEA (Gordon y McBride, 1994), a T y p o a T y V, y la llama
    adiabática HP o UV con 13 especies.
  - **`core/combustion/combustion.py`** (nuevo): la llama completa y con
    disociación (con la verificación de las K_p), la combustión a volumen
    constante, el calor con los humos a T_s (con condensación y el reparto del
    PCI), el rendimiento sobre PCI y PCS, el segundo principio (S_gen,
    X_dest = T₀·S_gen, X_comb ≈ PCI), el análisis de humos (O₂ y CO medidos u
    Orsat), notas, 13 + 3 ejemplos, barridos (λ, T del aire, T de los humos) y
    export.
  - **`core/combustion/combustion_procedure.py`** (nuevo): el procedimiento
    de los dos modos en los tres sistemas, con las unidades molares nuevas de
    `core/units_system.py` (J/mol, kJ/kmol, Btu/lbmol).
  - **`ui/combustion_charts.py`** (nuevo): composición de los humos, diagrama
    de combustión, cascada del PCI al calor útil y curvas de barrido.
  - **Página `/Combustion`** (nueva, después de Psicrometría): los dos modos,
    con recalculo automático, notas, tablas, gráficos, procedimiento, teoría
    (vademecum §16 con la tabla de §16.12, y §4.8), export y barridos.
- **Validación**:

  | Ejemplo | App | Cengel |
  |---|---|---|
  | 15-1: octano con 20 kmol de O₂ | AC 24,05 | 24,2 (con M_aire = 29) |
  | 15-2: etano, 20 % de exceso, 100 kPa | AC 19,19; T_pr 52,5 °C | 19,3; 52,3 °C |
  | 15-3: gas natural con aire a 20 °C y 80 % | T_pr 61,0 °C | 60,9 °C |
  | 15-4: Orsat del octano | 130,3 % de aire teórico; condensan 6,60 kmol | 131 %; 6,59 kmol |
  | 15-6: propano líquido, 10 % del C a CO | ṁ_aire 1,168 kg/min; Q̇ 411 kJ/min | 1,18 (M = 29); 413 |
  | 15-7: CH₄ + 3 O₂ en una bomba | Q 308 600 Btu/lbmol; p₂ 3,354 atm | 308 730; 3,35 atm |
  | 15-8: octano, 100 % y 400 % de aire | 2392 K y 962 K | 2395 K y 962 K |
  | 15-8 c: con 90 % de aire | 2283 K (*) | 2236 K |
  | 15-10: metano, 50 % de exceso | 1788 K | 1789 K |
  | 15-11: enfriado a 25 °C | Q 871 700 kJ/kmol; S_gen 2745; X_dest 818 400 | 871 400; 2746; 818 000 |

  (*) Con los productos y los datos del libro, a 2236 K los productos tienen
  4,249 MJ y hacen falta 4,367 MJ: el balance cierra en 2284 K. Los tests van
  contra el balance; la nota del ejemplo dice «revisá en tu edición».
  - Δh̄ contra las tablas A-18 a A-21 de Cengel (JANAF), 1000–2000 K: < 0,05 %
    (H₂O contra la A-23: hasta 0,5 % a 2000 K, NASA usa datos más nuevos); contra
    el gas ideal de CoolProp de `core/ideal_gas.py`: < 0,03 %.
  - PCS a 25 °C contra ISO 6976:2016 (CH₄, C₂H₆, C₃H₈, n-C₄H₁₀, H₂, CO): < 0,01 %.
    PCI del metano 50 027 kJ/kg (el de la turbina de gas); octano líquido
    44 422 y 47 889 kJ/kg (A-27: 44 430 y 47 890).
  - Llama de equilibrio contra Cantera 3.2 (fuera del proyecto, con los mismos
    polinomios y p° = 1 bar): metano con λ de 0,6 a 3, de 0,5 a 10 atm, aire a
    25 y 427 °C, a presión y a volumen constante, octano, hidrógeno y gas
    natural: 0,004 K como máximo; fracciones a 6·10⁻⁸.
  - Balances de elementos a 1e-12 y de energía en todos los ejemplos; el
    reparto del PCI es una identidad; X_dest = T₀·S_gen.
- **Mensajes al alumno**: λ fuera de rango, hollín con defecto de aire, O₂
  medido imposible (más que el del aire o menos que con λ = 1), combustible
  fuera de sus tablas o sin nada que se queme, análisis que no suma 100 % o PCI
  negativo, aire ambiente que condensaría antes del quemador, humos más
  calientes que la llama, calor entregado a una temperatura que daría
  S_gen < 0, llama completa que pasaría 6000 K (carbón con O₂ puro), Orsat
  incoherente o con λ < 1. Notas: defecto de aire y CO, disociación, NO de
  equilibrio, oxígeno puro, humedad del aire y del combustible, rocío ácido,
  condensación (η sobre el PCI > 100 %), aire precalentado, volumen
  constante y la exergía destruida.
- **Tests**: de 2921 a 3231 passed (10 skipped), sin warnings.
  - `tests/test_combustion_thermo.py` (43), `tests/test_combustion.py` (125),
    `tests/test_combustion_procedure.py` (73) y `tests/test_page_combustion.py`
    (43, AppTest), más la página en `test_navigation.py` y las unidades
    molares en `test_units_system.py`.
  - LaTeX: 14 745 expresiones distintas (procedimiento y teoría) validan con
    KaTeX estricto y entran en 321 px como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: diez casos en los dos modos (el
    15-1, la caldera de gas natural, la bomba del 15-7, el fueloil, el
    hidrógeno, el bagazo con el barrido de precalentamiento, el 15-10 en SI, la
    caldera de condensación en inglés, el Orsat del fueloil y el 15-4 en
    inglés), con el procedimiento y la teoría abiertos: sin errores, sin
    desborde y ninguna ecuación se pasa del borde de su expansor. Los barridos
    de λ y de la temperatura de los humos se revisaron en capturas.
- **Dependencias**: ninguna nueva (Cantera solo validó, fuera del proyecto).

### Fase 6 — Poder calorífico por correlaciones
- **Versión**: `0.22.0` (2026-10-08). Rama `claude/water-state-analyzer-f4fiev`
  («sí a todo, y continuá con el plan»: se mergeó el PR de la Fase 5, se
  aceptó seguir el PR nuevo y la Fase 6 se planeó e implementó sin esperar).
- **Scope**:
  - **`core/combustion/heating_value.py`** (nuevo): el PCS desde el análisis
    elemental (Dulong; Boie, 1953; Channiwala y Parikh, 2002) o inmediato
    (Parikh, Channiwala y Ghosal, 2005; Cordero et al., 2001), una función
    por correlación con la cita, el tipo de combustible y el rango de ajuste
    en el docstring, y el registro `CORRELATIONS` con los coeficientes
    publicados (MJ/kg por %, base seca). Bases tal cual, seca y seca y sin
    cenizas (ASTM D3180-25): `HeatingValueInputs.from_basis` acepta los datos
    en cualquiera, con el O o el CF por diferencia. PCI = PCS − h_fg·(8,94·H
    + W) y la humedad W* que lo anula. Avisos de rango y de tipo de
    combustible, notas, 18 ejemplos, barrido de la humedad y export.
  - **Datos de validación**: `data/ghugare2014_biomass.csv` (536 biomasas con
    el PCS medido, Ghugare et al., 2014, del paquete de R `modeldata`, MIT,
    con `data/LICENSE-modeldata.txt`) y `data/argonne_premium_coals.csv`
    (cinco carbones de Vorres, 1990, con los dos análisis y el PCS).
  - **`core/combustion/heating_value_procedure.py`** (nuevo): el cambio de
    base, cada correlación con la **tabla de aportes** de cada componente,
    el PCS en las tres bases, el PCI y los desvíos.
  - **`ui/heating_value_charts.py`** (nuevo): la comparación de las
    correlaciones (gráfico de puntos con la referencia), la paridad contra las
    536 biomasas y el error contra el O.
  - **Página `/Poder_Calorifico`** (nueva, después de Combustión): datos en
    cualquier base, elemental, inmediato o los dos, el H aparte para el PCI,
    el PCS medido (en cualquier base) o el exacto de una sustancia pura, la
    correlación principal; métricas, notas, comparación, el análisis en las
    tres bases, «¿Qué tan buenas son?», teoría, procedimiento, export y el
    barrido de la humedad (instantáneo: sin botón).
  - **`/Combustion`**: el PCS de un análisis elemental puede ser un dato o
    estimarse con Channiwala y Parikh, Boie o Dulong; `Fuel` suma
    `hhv_correlation` (`None` = dato, como en la 0.21.0) y el procedimiento y
    el export muestran la correlación.
- **Validación** (desvío del PCS estimado contra el medido o el exacto):
  - Biomasa (Ghugare et al., 2014, n = 536): Channiwala y Parikh AAE 4,94 % y
    ABE +0,12 %; Boie 5,23 % y 0,00 %; Dulong 11,46 % y −10,15 %. El sesgo de
    Dulong crece con el O: +3,7 % con O < 10 % y −18 % con O > 45 %.
  - Carbones de Argonne: las elementales aciertan a ±3 %; con el inmediato,
    Parikh da −10 a −23 % y Cordero −7 a −21 %, aunque estén dentro de sus
    rangos (una correlación con MV y CF no distingue la materia volátil del
    carbón, más rica en H, de la de la biomasa).
  - Sustancias puras (PCS exacto de las h_f de NASA): Boie y Channiwala y
    Parikh a ±2 % en metano, octano, etanol y metanol; Dulong +11 % en el
    metano (ignora la h_f) y +1,8 % en el H₂, donde las de ajuste dan −17 y
    −18 % (no extrapolan).
  - El fueloil (Channiwala y Parikh) y el carbón de Pensilvania (Dulong) de la
    biblioteca de la Fase 5 reproducen su PCS a 0,2 y 0,3 %.
- **Mensajes al alumno**: análisis que no suman 100 % en su base (dicen
  cuánto suman y qué tiene que sumar), el O o el CF por diferencia
  negativos, cenizas distintas entre los dos análisis, humedad o cenizas
  fuera de rango, PCS negativo. Notas: Dulong con O > 10 %, hidrocarburos y
  H₂, el inmediato en un carbón, fuera de rango o de tipo, sin H no hay PCI,
  humedad alta y W*, cenizas altas, ninguna correlación a menos de 5 %.
- **Tests**: de 3231 a 3398 passed (10 skipped), sin warnings.
  - `tests/test_heating_value.py` (56), `tests/test_heating_value_procedure.py`
    (72, incluido el paso del PCS estimado en /Combustion) y
    `tests/test_page_poder_calorifico.py` (34, AppTest), más 4 en
    `tests/test_page_combustion.py` y la página en `test_navigation.py`.
  - LaTeX: 2078 expresiones distintas (procedimiento, teoría y el paso nuevo
    de /Combustion) validan con KaTeX estricto y entran en 311 px como máximo
    en los tres sistemas (la tabla de aportes reemplazó a las sumas largas, que
    llegaban a 412 px).
  - Smoke test en Chromium a 1280 y 390 px: ocho ejemplos (el bagazo, el
    eucalipto, Pittsburgh con los dos análisis, Pocahontas con el inmediato,
    el metano en SI, el hidrógeno en inglés, el lodo en SI y el lignito en
    inglés) con la teoría, el procedimiento, los datos y el barrido abiertos:
    sin errores, sin desborde y ninguna ecuación se pasa de su expansor. Los
    gráficos se revisaron en capturas (la leyenda de la paridad y del error va
    debajo del eje: arriba pisaba el título a 390 px).
- **Dependencias**: ninguna nueva.
- **Red**: la política del entorno bloqueó academic.hep.com.cn, sites.bu.edu,
  en.wikipedia.org, pmc.ncbi.nlm.nih.gov, www2.et.byu.edu, zenodo.org,
  data.mendeley.com y pubs.usgs.gov; los coeficientes de Boie y Dulong salen
  de fuentes secundarias que coinciden, y los carbones de Argonne, de
  reproducciones del *Users Handbook* (coherentes entre sí).

### Fase 7 — Exergía física, química y por componente (diagrama de Grassmann)
- **Versión**: `0.23.0` (2026-10-08). Rama `claude/water-state-analyzer-f4fiev`
  («mergeá todo y avanzá»: se mergeó el PR de la Fase 6 y la Fase 7 se planeó
  e implementó sin esperar).
- **Scope**:
  - **`core/exergy/`** (paquete nuevo; reemplaza al placeholder `exergy.py`):
    - `physical.py`: `Ambient` (T₀, p₀), la exergía de flujo ψ y la de la masa
      φ de cualquier estado (los pares de Propiedades), con la parte térmica y
      la mecánica (Kotas, 1985), V²/2 y g·z, por kg, para una masa o un caudal;
      `ProcessExergy` (el trabajo reversible de 1 → 2), la exergía de un calor
      (factor de Carnot) y la de una fuente finita; notas, ejemplos (Cengel cap.
      8 y 10-8) y export.
    - `chemical.py` + `data/szargut_chemical_exergy.csv`: 42 sustancias en los
      dos modelos de la tabla A-26 de Moran y Shapiro (II: Szargut, Morris y
      Steward, 1988; I: Ahrendts, 1980; los valores digitales de TESPy 0.7.9,
      MIT), el método de Szargut (Δḡ_f de los polinomios NASA + la exergía de
      los elementos), `species_available`, φ = e/PCI de los combustibles,
      mezclas de gases (con el agua que condensa a 25 °C) y combustibles
      sólidos y líquidos por su análisis elemental (β de Szargut y Styrylska,
      1964, con la humedad y el azufre como Kotas).
    - `plant.py`: la exergía por componente (combustible, producto,
      destrucción y pérdida; Bejan, Tsatsaronis y Moran, 1996) de un Rankine,
      una refrigeración, una turbina de gas y un ciclo combinado, la exergía de
      cada corriente, `grassmann_rows`, notas y export.
    - `exergy_procedure.py`: el procedimiento de cada cálculo, con un paso por
      componente en la planta.
  - **`ui/exergy_charts.py`** (nuevo): el diagrama de Grassmann (una banda
    vertical: el Sankey no se leía a 390 px), ψ en el h–s, el factor de Carnot,
    la curva de la fuente finita, φ de los combustibles y ε por componente.
  - **Página `/Exergia`** (nueva, después de Ciclo combinado): exergía física
    (un fluido, un calor o un cuerpo), química (una sustancia, una mezcla o un
    combustible, en los dos ambientes) y de una planta componente por
    componente, con un ejemplo o con el último ciclo que se calculó en su
    página; teoría (vademecum §11 y §16.13), procedimiento y export.
  - Cambios chicos: `units_system` suma masa, energía y longitud;
    `refrigeration` hace públicas `default_reservoirs` y `check_reservoirs`;
    `HRSGExergy.gained_W` (lo que gana el agua en cada sección);
    `combined_multi.from_combined_with_pinch` (el diseño por chimenea de la
    3.4, como Cengel 10-9, con el pinch que resulta: mismo ciclo a 10⁻⁹).
- **Validación**:
  - Cengel 10-8: ψ = 1162,1 y 449,0 kJ/kg; 1110 kJ/kg destruidos en la
    caldera y 414 en el condensador. Cap. 8: tanque de aire comprimido 280,6 MJ
    (libro 281), compresor de R-134a 38,04 kJ/kg (38,0), viento 70,70 kW
    (70,7), hogar 2195 Btu/s, bloque de hierro 8191 kJ.
  - Química: el método de Szargut reproduce la tabla dentro de 0,15 % en los
    hidrocarburos (NO +1,2 %, por la h_f actualizada de NASA; agua líquida +5 %
    relativo, 0,05 kJ/mol). Las β de los líquidos contra sustancias puras:
    octano +0,6 %, etanol +1,2 %, metanol +2,7 %; el grafito da β = 1,0437
    contra 1,0426 de la tabla. Con las biomasas de Ghugare, e/PCS = 1,054 en
    promedio.
  - Plantas: todos los balances (Ẋ_F = Ẋ_P + Ẋ_D + Ẋ_L, por componente y de
    la planta) cierran a ≤ 7,5·10⁻¹⁰; la destrucción de la refrigeración es la
    de la Fase 3.2 y la de la turbina de gas con e_comb ≈ PCI, la de la 3.7
    (η_II = η térmico).
- **Mensajes al alumno**: ambiente fuera de −50 a 60 °C o de 0,4 a 10 bar,
  fuente de calor no más caliente que el vapor, sumidero más frío que el
  ambiente o más caliente que el condensador, fuentes de la refrigeración que
  invierten el calor, sustancias que el modelo I no trae («elegí» el otro),
  mezclas vacías o con fracciones negativas, análisis que no suman 100 %, o/c
  fuera de las formas de Szargut y Styrylska. Notas: el fluido más frío que
  el ambiente (también tiene exergía), ψ < 0 y la exergía del vacío, el
  estado muerto líquido, la cinética y la potencial como exergía pura, el
  calor bajo T₀, la mezcla que condensa, la humedad que sube e/PCI tal cual,
  la convención del condensador y lo que destruye cada componente.
- **Tests**: de 3398 a 3879 passed (10 skipped), sin warnings.
  - `tests/test_exergy_physical.py` (108), `tests/test_exergy_chemical.py`
    (67), `tests/test_exergy_plant.py` (73), `tests/test_exergy_procedure.py`
    (76), `tests/test_exergy_charts.py` (3) y `tests/test_page_exergia.py`
    (115, AppTest), más `from_combined_with_pinch` en
    `tests/test_combined_multi.py` y la página en `test_navigation.py`.
  - LaTeX: 6338 expresiones distintas del procedimiento y 505 de la teoría y
    del ciclo combinado de una presión validan con KaTeX estricto y entran en
    310 px como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: 13 casos (el fluido de 10-8, el
    viento en inglés, el calor en SI, el cuerpo, una sustancia en SI, los humos
    en inglés, el bagazo, y las plantas: Rankine 10-1 y 10-6, refrigeración 11-5
    en inglés, la combustión secuencial, el ciclo combinado de tres presiones y
    Cengel 10-9 por chimenea en SI) con la teoría, el procedimiento y las
    tablas abiertos: sin errores, sin desborde y ninguna ecuación se pasa de su
    expansor. Los gráficos se revisaron en capturas: a 390 px se cortaban los
    rótulos largos del Grassmann, que ahora se parten.
- **Dependencias**: ninguna nueva (TESPy 0.7.9 fue solo la fuente de los
  valores tabulados).
- **Red**: la política del entorno bloqueó doi.org (el DOI de Ahrendts, 1980,
  sale de RePEc) y los sitios académicos de las tablas; los valores digitales
  salen de la rueda de TESPy 0.7.9 en PyPI y se contrastaron con la tabla A-26.

### Fase 8.1 — Transferencia de calor: conducción, aletas y convección
- **Versión**: `0.24.0` (2026-10-09). Rama `claude/water-state-analyzer-f4fiev`
  («dale, armá el plan y seguí»: la Fase 8 se partió en dos entregas, como la
  3.1; el plan se mandó y se implementó sin esperar). La 8.2 trae la radiación
  y los intercambiadores.
- **Scope**:
  - **`core/heat_transfer/`** (paquete nuevo):
    - `conduction.py`: la red de resistencias de una pared plana, un cilindro o
      una esfera de una a cinco capas, con la resistencia de contacto y (en la
      pared) partes en paralelo con las caras isotérmicas; cada borde es un
      fluido (T∞ y h), una superficie a temperatura dada o un calor dado. Q̇,
      q″ en cada cara, la tabla de resistencias (con su % y su ΔT), las
      temperaturas de la red, U referido a cada cara, el perfil T(x) o T(r), el
      radio crítico y el barrido del radio exterior; notas, 8 ejemplos y export.
    - `fins.py`: aletas recta, de aguja y anular (funciones de Bessel
      escaladas) con las puntas de la tabla 3.4 de Incropera (convectiva,
      adiabática, a temperatura dada, infinita y con longitud corregida); m, M,
      Q̇, η, ε, la T de la punta, L∞ y Bi; arreglos de N aletas (η_o, el aumento
      y la efectividad del arreglo), el perfil, la comparación de las puntas y
      la curva η(m·L_c); notas, 6 ejemplos y export.
    - `convection.py`: 19 correlaciones, una función por correlación con la
      cita y el rango, y el registro `CORRELATIONS` con el LaTeX. Placa plana
      laminar, mixta y turbulenta desde el borde; cilindro (Churchill y
      Bernstein, Hilpert); esfera (Whitaker, Ranz y Marshall); tubo laminar
      desarrollado y Hausen, Dittus y Boelter, Gnielinski (f de Petukhov) y
      Sieder y Tate; natural en placas vertical (Churchill y Chu, McAdams) y
      horizontales (la cara de arriba y la de abajo), cilindro horizontal
      (Churchill y Chu, Morgan) y esfera (Churchill). Las propiedades salen de
      CoolProp para cualquier fluido del proyecto en una sola fase; el tubo
      itera la T media, con la pared a temperatura o flujo de calor constante
      (T de salida, ΔT_ml, T de pared a la salida, largos de entrada, f y Δp).
      La comparación de las correlaciones que se pueden aplicar, Nu(Re) o
      Nu(Ra), el perfil del tubo; notas, 17 ejemplos y export.
    - `procedure_common.py` y los procedimientos `conduction_procedure.py`,
      `fins_procedure.py` y `convection_procedure.py`.
  - **`ui/heat_transfer_charts.py`** (nuevo): el perfil de temperatura con las
    capas sombreadas y la película de cada fluido, el barrido del radio
    crítico, T(x) de la aleta con las otras puntas en gris, η(m·L_c), Nu(Re) o
    Nu(Ra) en escala logarítmica, la comparación de las correlaciones (gráfico
    de puntos) y T_m y T_s a lo largo del tubo.
  - **Página `/Transferencia_de_Calor`** (nueva, después de Exergía): tres
    modos (conducción, aletas y convección: forzada externa, en un tubo y
    natural), con la teoría (Çengel y Ghajar e Incropera: el vademecum todavía
    no tiene el capítulo y la página lo dice), el procedimiento y el export.
  - Cambios chicos: `units_system` suma 12 magnitudes (h, q″, resistencia
    térmica, área, longitudes chicas en mm o in, calor y calor por metro en W
    y no en kW como los ciclos, ṁ·c_p, 1/m, β, g y Δp); `core.fluids` hace
    pública `fluid_with_article` (la usan los mensajes de la convección).
- **Validación**:
  - Conducción, exacta contra Cengel y Ghajar: pared de ladrillo 630 W,
    ventana simple 266,2 W (cara interior a −2,2 °C), ventana doble 69,25 W,
    pared de ladrillos con revoque 261,9 W, caño de vapor aislado 120,8 W/m y
    alambre a 105,0 °C (r_cr = 12,5 mm).
  - Aletas: las varillas muy largas de Incropera 3.9 (8,3; 5,6 y 1,6 W); las
    cinco puntas contra `solve_bvp` (10⁻⁶) y la anular (10⁻⁵).
  - Convección, con las propiedades de los libros, exacta: Incropera 7.4
    (Hilpert 37,3; Churchill y Bernstein 40,6) y 9.2 (147); Cengel y Ghajar
    (124; 17,40; 69,4). Con las de CoolProp (el aire tiene una k 2,7 % mayor y
    un Pr 3 % menor que la tabla A-15) h queda 1,5 a 2 % arriba: caño de vapor
    con viento 1115 W/m (libro 1093), caño de agua caliente 449,8 W (443),
    agua calentada con resistencias Re = 10 758 y Nu = 69,5 con la pared a
    115,4 °C a la salida. El tubo cierra Q̇ = ṁ·c_p·ΔT.
- **Mensajes al alumno**: radio interior nulo (para un alambre, el de la
  aislación es el del alambre), contacto después de la última capa, calor dado
  en los dos bordes (sin temperatura de referencia), h nulo (usá
  «superficie»), fracciones del paralelo que no suman 1; aletas con la base a
  la temperatura del fluido o un arreglo que no entra en la base; un fluido
  que cambia de fase a la temperatura de película o dentro del tubo («subí» o
  «bajá» la presión, según hierva o condense), un flujo de calor que llevaría
  la salida bajo el cero absoluto, el agua entre 0 y 4 °C en convección
  natural (β ≤ 0) y los fluidos sin viscosidad en CoolProp (el R-1233zd(E) no
  está en la lista). Avisos: Bi > 0,1 (la aleta deja de ser unidimensional),
  correlaciones fuera de su rango (con 1 % de tolerancia) y ebullición o
  condensación en la pared. Notas: el radio crítico (solo con una capa
  aislante), la transición de la placa, la dispersión entre las correlaciones
  y las propiedades de CoolProp contra las del libro.
- **Tests**: de 3879 a 4348 passed (10 skipped), sin warnings.
  - `tests/test_heat_transfer_conduction.py` (32),
    `tests/test_heat_transfer_fins.py` (31),
    `tests/test_heat_transfer_convection.py` (96),
    `tests/test_heat_transfer_procedure.py` (178),
    `tests/test_heat_transfer_charts.py` (11) y
    `tests/test_page_transferencia.py` (43, AppTest), más la coherencia de las
    unidades del procedimiento en `tests/test_units_system.py` y la página en
    `test_navigation.py`.
  - LaTeX: 1865 expresiones distintas del procedimiento (los ejemplos y sus
    variantes) y 37 de la teoría validan con KaTeX estricto y entran en 308 y
    311 px como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: 20 casos (7 de las 8 redes de
    conducción, 4 aletas, 3 externas, 3 en un tubo y 3 naturales, repartidos
    en los tres sistemas) con la teoría y el procedimiento abiertos: sin
    errores, sin desborde y ninguna ecuación se pasa de su expansor. En las
    capturas, la curva Nu(Re) de la placa mixta bajaba de Re_cr y caía a
    Nu < 0 (la fórmula mixta fuera de su rango): ahora arranca en Re_cr. Y en
    el tanque frío (Q̇ < 0) el barrido del radio crítico mostraba un mínimo
    mientras el texto hablaba de un máximo de la pérdida: ahora se dibuja |Q̇|
    y el texto dice que el calor entra.
- **Dependencias**: ninguna nueva (las funciones de Bessel y `solve_bvp` son de
  SciPy, que ya estaba).
- **Red**: la política del entorno bloqueó Crossref, doi.org y los sitios
  académicos con las tablas de contenidos; las secciones de los libros y los
  DOI de las referencias se verificaron con búsquedas web (solo se citan los
  DOI confirmados).

### Fase 8.2 — Transferencia de calor: radiación e intercambiadores
- **Versión**: `0.25.0` (2026-10-09). Rama `claude/water-state-analyzer-f4fiev`
  («seguí mergeando y con el plan»: el PR de la 8.1 se mergeó, el plan de la 8.2
  se mandó y se implementó sin esperar). Dos páginas nuevas en vez de más modos
  en /Transferencia_de_Calor.
- **Scope**:
  - **`core/heat_transfer/radiation.py`** (nuevo): Planck, Wien y
    Stefan–Boltzmann con CODATA 2018; la fracción del cuerpo negro f(λT) (serie
    de Chang y Rhee) y su inversa; ε(λ) en bandas: la ε total y la α para un
    cuerpo negro a otra temperatura. Ocho factores de forma (rectángulos
    paralelos y perpendiculares y discos coaxiales; placas, placas con un borde
    común, recinto de tres lados, cilindros paralelos y una hilera de tubos en
    2D) con la reciprocidad, la regla de la suma y F contra la separación. Dos
    superficies (placas, cilindros y esferas concéntricos, objeto chico,
    general) con hasta tres pantallas: la red, las radiosidades y la T de cada
    pantalla. Recintos de N superficies grises por radiosidades (T dada, Q̇ dado
    o rerradiante) y cuatro configuraciones de tres superficies armadas desde la
    geometría (`ENCLOSURE_LAYOUTS`). La superficie con convección y radiación
    (h_rad, el sol, la T de equilibrio para un calor dado) y la termocupla.
    Notas, 28 ejemplos y export.
  - **`core/heat_transfer/exchangers.py`** (nuevo): ε(NTU, C_r) y NTU(ε, C_r)
    del doble tubo, casco y tubos (1 a n pasos de casco) y flujo cruzado (los
    dos sin mezclar con la serie exacta de Mason; uno mezclado), con C_r = 0;
    F = NTU_cc/NTU, P y R, la LMTD. Corrientes con c_p dado o de CoolProp
    (iterado a la T media, una sola fase) o que cambian de fase (h_fg de
    CoolProp o dado). Verificación, dimensionamiento (salida o Q̇ → A y largo) y
    ensayo con las cuatro temperaturas (U → Q̇ y caudales; un caudal → U). La
    exergía (Δs = C·ln(T₂/T₁), T₀·Ṡ_gen, η_ex del vademecum §11.10), la
    comparación de tipos, las curvas ε–NTU y F–P, el perfil del doble tubo y el
    U global de un tubo o una placa con el ensuciamiento de TEMA. Notas, 16
    ejemplos y export.
  - `radiation_procedure.py` y `exchangers_procedure.py`: los procedimientos en
    los tres sistemas.
  - **`ui/radiation_charts.py`** y **`ui/exchanger_charts.py`** (nuevos): Planck
    con la banda y la emisión real, los espectros normalizados con ε(λ), F
    contra la separación, el peso de cada resistencia de la red, el calor de
    cada superficie del recinto, q_conv y q_rad contra T_s; ε–NTU y F–P con la
    curva y el punto de los datos, el perfil del doble tubo, la comparación de
    los tipos y las resistencias del U.
  - **Páginas `/Radiacion` (☀️) e `/Intercambiadores` (🔄)**, en el menú
    después de Transferencia de calor, con la teoría, el procedimiento y el
    export; la home y /Transferencia_de_Calor las nombran.
  - Cambios chicos: `units_system` suma seis magnitudes (T absoluta en K o °R,
    λ en μm, λT, E_bλ por μm, R″_f y Ṡ).
- **Validación**:
  - Intercambiadores, contra Cengel y Ghajar: 11-2 (U_i = 399,3 y U_o = 315,3
    W/(m²·K)), 11-3 (ΔT_ml = 11,54 °C, 1,091 MW, 32,6 kg/s de agua), 11-4 y
    11-8 (A = 5,113 m², 108,5 m de tubo), 11-5 (F = 0,911, 1832 W), 11-6
    (F = 0,970 con la serie; la fórmula aproximada da 0,933), 11-7 (501,6 kW) y
    11-9 (ε = 0,462, 38,4 kW); Incropera 11.1 (65,9 m). ε(1, 1) de la serie del
    flujo cruzado = 0,4762 (Kays y London); F contra la fórmula de Bowman; ida y
    vuelta NTU → ε → NTU en todos los tipos; LMTD y ε-NTU dan la misma área.
  - Radiación: f(λT) contra `quad` (5·10⁻⁹); los factores de forma contra Monte
    Carlo, las cuerdas cruzadas y una caja cerrada; Cengel y Ghajar (bola negra
    23,23 kW/m² y 3846 W/(m²·μm); bandas 0,5206 y 12,09 kW/m²; placas 3626 y
    806 W/m² con una pantalla; ducto con la base a 543,4 K; termocupla 715,0 K;
    superficies al sol 306,4, 34,0, 574,7 y −234,3 W/m²) e Incropera (cavidad
    13.2, 1831 W; caño 1.2, 577 + 421 = 998 W/m; horno de pintura 36,98 kW/m y
    la pared aislada a 1102 K). Todos los recintos cierran ΣQ̇ = 0.
- **Mensajes al alumno**: un ε mayor que el máximo del tipo («probá con
  contracorriente o con más pasos de casco»), temperaturas que se cruzan, una
  corriente que hierve o condensa en el intercambiador («subí» o «bajá» la
  presión), un pseudo-puro que cambia de fase (deslizamiento), falta de h_fg o
  de c_p; bandas que no crecen, emisividades fuera de (0, 1], radios de las
  pantallas desordenados, F₁₂ imposible por reciprocidad, filas de F que no
  suman 1, un triángulo que no cierra, la abertura que no es negra, un recinto
  sin ninguna temperatura y uno que pediría temperaturas bajo el cero absoluto.
  Avisos: F < 0,75 y ε cerca del máximo. Notas: el C_mín, el NTU alto, la serie
  contra la aproximada, la exergía destruida y cuándo no se define η_ex.
- **Tests**: de 4348 a 4786 passed (10 skipped), sin warnings.
  - `tests/test_exchangers.py` (80), `tests/test_radiation.py` (56),
    `tests/test_exchangers_procedure.py` (67),
    `tests/test_radiation_procedure.py` (90),
    `tests/test_radiation_exchanger_charts.py` (12),
    `tests/test_page_radiacion.py` (39) y `tests/test_page_intercambiadores.py`
    (26, AppTest), más las unidades en `tests/test_units_system.py` y las dos
    páginas en `test_navigation.py`.
  - LaTeX: 3094 expresiones distintas (los procedimientos con los ejemplos y
    sus variantes, y la teoría) validan con KaTeX estricto y entran en 321 px
    como máximo en los tres sistemas.
  - Smoke test en Chromium a 1280 y 390 px: 25 casos (14 de radiación y 11 de
    intercambiadores, en los tres sistemas) con la teoría y el procedimiento
    abiertos, sin errores, sin desborde y sin ecuaciones que se pasen de su
    expansor. Las capturas encontraron cinco detalles que se corrigieron: los
    ejes logarítmicos rotulaban 0,2 como «2» (ahora van marcas 1-2-5), las
    barras del recinto decían «3,698e+04» y el rótulo de la negativa se pisaba
    con el nombre (ahora van adentro y sin exponente), la leyenda de las barras
    pisaba el título del eje (ahora va al pie), los nombres largos de los tipos
    de intercambiador aplastaban el eje de la comparación (ahora van nombres
    cortos; la tabla conserva los completos) y el rótulo del ε máximo quedaba
    encima de la curva de los datos (ahora va arriba de la línea).
- **Dependencias**: ninguna nueva (`gammainc` y `brentq` son de SciPy).
- **Red**: doi.org, Crossref y las editoriales siguen bloqueados; el DOI de
  CODATA 2018 se confirmó en NIST, los de Chang y Rhee y de Shah y Sekulić no, y
  se citan sin DOI.

---

## Pendientes / próximas fases

- **Detectado en Fase 1.7, sin resolver**:
  - ~~Ecuaciones más anchas que el celular en SI~~ (resuelto después de la
    0.16.0, con el OK del autor): en el Rankine, Propiedades e
    Isoentrópicos, una resta con números ×10ⁿ (o negativos) con un factor
    delante pasa el segundo número a otro renglón. Ahora entran en 324 px
    las 4309 expresiones distintas del Rankine (máx. 316 px en SI), las
    1097 de las páginas 1–4 y las 129 de Isoentrópicos; Técnico e Inglés del
    Rankine quedan idénticos.
  - Pseudo-puros con T y (s o v): dentro de la campana CoolProp los
    resuelve a una sola presión entre la de burbuja y la de rocío (es su
    modelo; el título no coincide con la palanca a T constante), y justo
    en el borde los clasifica como líquido comprimido / vapor
    sobrecalentado. Con p ya está resuelto; con T es un caso de borde raro.
- **Fase 2.3 (continuación)** — Matriz de normalización ISO 6976
  cuando se incorpore ISO 14912:2003 Formula (69).
- **HRSG (continuación)**: secciones intercaladas o en paralelo
  (economizadores partidos), quemadores suplementarios, pérdidas de carga y
  purga, diseño de las superficies (UA, NTU), condensación ácida.
- **Ciclo combinado (continuación)**: aire húmedo, gas natural con otros
  componentes en la página (el núcleo ya los acepta), generador y pérdidas
  mecánicas, cogeneración; con varias presiones, el diseño por chimenea,
  pérdidas de carga (HRSG y recalentador) y recirculación del precalentador
  por el punto de rocío.
- **Turbina de gas (continuación)**: refrigeración de álabes con aire del
  compresor, pérdidas mecánicas y del generador, la turbina con etapas
  (combustión secuencial) en el ciclo combinado, inyección de vapor o de
  agua, turbinas de propulsión (Cengel §9-11). Evaluar
  `tespy.tools.get_plotting_data` para los diagramas del Rankine (hoy con
  `core.diagrams.segments_overlays`).
- **Refrigeración (continuación)**: ciclo transcrítico de CO₂,
  intercambiador líquido–vapor, economizador cerrado (subenfriador),
  refrigeración por gas y por absorción.
- **Psicrometría (continuación)**: cargas del local y factor de calor
  sensible (recta de la sala, aire de impulsión), serpentín con ADP y factor
  de bypass, número de Merkel para dimensionar torres, niebla (mezclas
  sobresaturadas), aire húmedo en la turbina de gas y en el ciclo combinado.
- **Combustión (continuación)**: hollín (carbono sólido en el equilibrio),
  NOx con cinética (Zeldovich), inquemados sólidos en las cenizas, la llama
  completa más allá de 6000 K (hoy un error con O₂ puro y carbón), los
  polinomios NASA en la turbina de gas y el ciclo combinado (hoy CoolProp).
- **Poder calorífico (continuación)**: otras correlaciones (Mendeleev, IGT,
  Sheng y Azevedo; Goutal para carbones con el inmediato), el C, el H y el O
  desde el inmediato, el PCS a volumen constante de la bomba contra el de
  presión constante (ISO 18125:2017) y las bases seca al aire y libre de
  materia mineral (Parr, para el rango ASTM D388).
- **Exergía (continuación)**: exergoeconomía (el costo de la exergía en
  cada componente, SPECO), el ambiente a elección para la exergía química
  (corrección a T₀ ≠ 25 °C), la tabla de Szargut (2007), y la HRSG, la
  psicrometría y la combustión en el diagrama de Grassmann.
- **Transferencia de calor (continuación)**: conducción transitoria (capacidad
  concentrada, solución de un término), conducción bidimensional (factores de
  forma), generación interna, bancos de tubos, ebullición y condensación,
  aletas de perfil variable; radiación de gases (CO₂ y H₂O, gráficos de Hottel)
  e intensidad direccional; intercambiadores compactos (j de Colburn), la caída
  de presión del casco (Kern, Bell–Delaware), el U armado con las
  correlaciones de la 8.1 y la selección económica.

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
