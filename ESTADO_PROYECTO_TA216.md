# Estado del proyecto — TA216 Apps

> Bitácora viva de qué está cerrado, qué quedó deferido y qué viene.
> Se actualiza al cierre de cada fase como parte del checklist (junto
> con el bump de versión, el README y el `CITATION.cff`). Generada a
> partir del `git log` y la documentación interna del proyecto.
>
> **Versión actual**: `0.11.0` — Fase 3.1a cerrada (2026-10-05).

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

---

## Pendientes / próximas fases

- **Detectado en Fase 1.7, sin resolver**:
  - En sistema SI (J/kg con ×10ⁿ) algunas sustituciones siguen siendo más
    anchas que el celular (regla de la palanca, h₂ de Isoentrópicos: hasta
    475 px; en Rankine, la h de la turbina real: ≤ 347 px) y se deslizan;
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
- **Fase 3.1b** — Rankine regenerativo (calentadores abierto y
  cerrado), condensador con agua de enfriamiento, pérdidas de carga en
  caldera y condensador, ORC con otros fluidos (R134a, R1234yf…).
- **Fase 3.1 (continuación)** — Refrigeración por compresión de vapor;
  Brayton; ciclo combinado; exergía de los ciclos. Evaluar
  `tespy.tools.get_plotting_data` para los diagramas (hoy el ciclo se
  dibuja con `core.diagrams.cycle_overlays`).
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
