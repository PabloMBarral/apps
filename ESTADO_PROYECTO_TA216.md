# Estado del proyecto — TA216 Apps

> Bitácora viva de qué está cerrado, qué quedó deferido y qué viene.
> Se actualiza al cierre de cada fase como parte del checklist (junto
> con el bump de versión, el README y el `CITATION.cff`). Generada a
> partir del `git log` y la documentación interna del proyecto.
>
> **Versión actual**: `0.8.0` — Fase 1.5a cerrada (2026-05-23).

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

---

## Pendientes / próximas fases

- **Fase 1.5b** — Conversión al sistema activo de los pasos didácticos
  LaTeX de Isoentrópicos (hoy hardcoded en Técnico para consistencia
  con Cengel). Implica reescribir los formatters de
  `core/isentropic.py` para emitir LaTeX por sistema.
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
