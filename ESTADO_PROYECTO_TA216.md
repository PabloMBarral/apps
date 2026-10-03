# Estado del proyecto — TA216 Apps

> Bitácora viva de qué está cerrado, qué quedó deferido y qué viene.
> Se actualiza al cierre de cada fase como parte del checklist (junto
> con el bump de versión, el README y el `CITATION.cff`). Generada a
> partir del `git log` y la documentación interna del proyecto.
>
> **Versión actual**: `0.9.0` — Fase 1.6 cerrada (2026-10-03).

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

---

## Pendientes / próximas fases

- **Fase 1.5b** — Conversión al sistema activo de los pasos didácticos
  LaTeX de Isoentrópicos (hoy hardcoded en Técnico para consistencia
  con Cengel). Implica reescribir los formatters de
  `core/isentropic.py` para emitir LaTeX por sistema. Se puede reusar
  `latex_number` / `latex_unit` / `pv_energy_factor` de
  `core/state_report.py`.
- **Detectado en Fase 1.6, sin resolver**:
  - Isoentrópicos sigue usando `state_from_pair` sin las validaciones de
    `fluid_state_from_pair` (rango de la ecuación de estado, mensajes en
    castellano); migrarla daría los mismos mensajes que Propiedades.
  - Las páginas 2, 3 y 4 todavía no tienen el expansor `📖 Fórmulas
    teóricas` (las constantes `VADEMECUM_*` ya están en `ui/branding.py`).
  - Navegación: Streamlit decide entre `st.navigation` y la carpeta
    `pages/` con una bandera global que se apaga recién cuando corre
    `streamlit_app.py`. Si el primer visitante después de un reinicio
    entra por un link directo (p. ej. `/Propiedades`), ve el menú
    automático ("streamlit app", "Interpolacion" sin tilde) hasta que
    alguien entre por la raíz. Se arregla renombrando `pages/` (p. ej. a
    `app_pages/`) y actualizando las rutas de `st.Page`; queda a decisión
    del autor porque toca la estructura documentada.
- **Fase 2.3 (continuación)** — Matriz de normalización ISO 6976
  cuando se incorpore ISO 14912:2003 Formula (69).
- **Fase 3.1** — Ciclos termodinámicos con TESPy: Rankine simple,
  recalentamiento, regeneración; refrigeración por compresión de
  vapor; Brayton; ciclo combinado. Integración nativa con
  `core.diagrams` vía `tespy.tools.get_plotting_data`.
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
  de cada `pages/N_*.py` afectada.
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
