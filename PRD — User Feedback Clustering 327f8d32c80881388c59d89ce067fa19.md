# PRD — User Feedback Clustering

## Overview

User Feedback Clustering is a CLI tool that ingests raw user feedback from CSV exports (CRM interno + Jira), genera embeddings semánticos, agrupa el feedback en temas usando clustering, y produce un reporte en Markdown con los temas etiquetados, su frecuencia y acciones sugeridas. El objetivo es reducir el tiempo que un PM dedica a sintetizar feedback de horas a minutos.

---

## Problem statement

Los equipos de producto reciben feedback disperso en múltiples herramientas (CRM de soporte, Jira, etc.). Hoy ese proceso de síntesis es manual: alguien lee cientos de tickets, los agrupa mentalmente y genera un resumen. Ese proceso es lento, inconsistente entre personas y difícil de reproducir cada semana o sprint.

**Pain points concretos:**

- Leer y categorizar 200+ tickets lleva 3-5 horas por sprint
- La categorización varía según quién la hace
- No hay forma de detectar tendencias automáticamente entre períodos
- El feedback de Jira y el CRM vive en silos y no se cruza

---

## Goals

- Reducir el tiempo de síntesis de feedback a menos de 5 minutos
- Producir clusters consistentes y reproducibles
- Generar un output que un PM pueda compartir directamente con su equipo sin edición manual
- Soportar múltiples fuentes de CSV con configuración mínima

## Non-goals

- No es una herramienta de análisis de sentimiento (aunque puede extenderse)
- No conecta directamente con APIs de terceros en v1 (todo es CSV)
- No tiene UI gráfica en v1
- No almacena datos en una base de datos persistente

---

## Users

**Usuario primario:** Product Manager que recibe exports CSV de su CRM y Jira y quiere un resumen de temas cada sprint o cada semana.

**Usuario secundario:** Customer Success Manager que quiere entender patrones en tickets de soporte sin leer cada uno.

---

## User stories

> Como PM, quiero correr una CLI con un CSV de tickets y obtener un reporte de temas, para no tener que leer cada ticket manualmente.
> 

> Como PM, quiero que cada tema tenga ejemplos representativos del feedback real, para poder compartirlo con el equipo sin perder contexto.
> 

> Como PM, quiero poder cambiar qué columna del CSV contiene el texto a analizar, para usar la tool con exports de distintas herramientas.
> 

> Como PM, quiero combinar un export del CRM y uno de Jira en un solo análisis, para tener una visión unificada del feedback.
> 

> Como PM, quiero que el output sea un archivo Markdown, para poder pegarlo directamente en Notion o un doc compartido.
> 

---

## Functional requirements

### FR-01 · CSV ingestion

- La tool debe aceptar uno o más archivos CSV como input
- El usuario debe poder especificar qué columna(s) contienen el texto a analizar (via config o flag CLI)
- La tool debe concatenar el contenido de múltiples columnas de texto si se especifican (ej: `Summary` + `Description` de Jira)
- La tool debe manejar filas con valores nulos o vacíos sin romperse
- Soportar encodings UTF-8 y latin-1 (común en exports de herramientas legacy)

### FR-02 · Embeddings

- Generar un embedding semántico por fila usando OpenAI `text-embedding-3-small`
- Cachear embeddings localmente para no re-calcular en runs sucesivas sobre el mismo input
- El usuario debe poder ver el costo estimado antes de confirmar la generación (basado en token count)

### FR-03 · Clustering

- Implementar KMeans como algoritmo principal
- El número de clusters debe ser configurable (default: auto-detect via elbow method)
- Cada cluster debe tener: nombre generado por LLM, descripción de 1-2 oraciones, cantidad de items, y 3 ejemplos representativos
- Los clusters deben ordenarse por frecuencia (mayor a menor)

### FR-04 · Labeling con LLM

- Usar Claude API (`claude-sonnet-4-20250514`) para generar nombre y descripción de cada cluster
- El prompt debe incluir los ejemplos más representativos del cluster (los más cercanos al centroide)
- El label debe incluir: nombre del tema, descripción, y una sugerencia de acción concreta para el equipo de producto

### FR-05 · Output

- Generar un archivo `report.md` con el resumen completo
- El reporte debe incluir: fecha de generación, fuentes usadas, número total de items, tabla resumen de clusters, y detalle por cluster
- Opción de output a stdout para piping

### FR-06 · CLI interface

```bash
# Uso básico
feedback-cluster run --input tickets.csv --text-col description

# Múltiples fuentes
feedback-cluster run --input crm.csv --input jira.csv --config sources.yaml

# Especificar número de clusters
feedback-cluster run --input tickets.csv --clusters 8

# Output a archivo
feedback-cluster run --input tickets.csv --output report.md
```

---

## Non-functional requirements

- Tiempo de ejecución < 60 segundos para datasets de hasta 1.000 rows (excluyendo latencia de API)
- El reporte debe ser legible sin herramientas adicionales (Markdown plano)
- Errores deben mostrar mensajes claros y accionables (no stack traces crudos al usuario)
- Las API keys deben leerse de variables de entorno, nunca hardcodeadas
- Compatible con macOS y Linux

---

## Data model

### FeedbackItem

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `id` | str | Identificador original del ticket |
| `source` | str | Nombre del archivo CSV de origen |
| `text` | str | Texto concatenado de las columnas seleccionadas |
| `embedding` | list[float] | Vector de 1536 dimensiones (OpenAI small) |
| `cluster_id` | int | ID del cluster asignado |
| `distance_to_centroid` | float | Distancia euclidiana al centroide del cluster |

### Cluster

| Campo | Tipo | Descripción |
| --- | --- | --- |
| `id` | int | ID numérico del cluster |
| `label` | str | Nombre generado por Claude |
| `description` | str | Descripción de 1-2 oraciones |
| `suggested_action` | str | Acción sugerida para el equipo de producto |
| `size` | int | Cantidad de items en el cluster |
| `representative_examples` | list[str] | Top 3 ejemplos más cercanos al centroide |

---

## Pipeline architecture

```
[CSV files]
    ↓
[csv_loader.py]        → Normaliza columnas, concatena texto, filtra nulls
    ↓
[openai_embedder.py]   → Genera embeddings, cachea en .embeddings_cache.json
    ↓
[kmeans.py]            → Clustering, asigna cluster_id a cada item
    ↓
[claude_labeler.py]    → Genera label + descripción + acción por cluster
    ↓
[markdown_export.py]   → Produce report.md
```

---

## Report output format

```markdown
# Feedback Clustering Report
Generated: 2025-03-18 | Sources: crm_export.csv, jira_export.csv | Total items: 312

## Summary
| # | Theme | Items | % |
|---|---|---|---|
| 1 | Login & authentication issues | 87 | 27.9% |
| 2 | Slow loading / performance | 64 | 20.5% |
| 3 | Missing export functionality | 41 | 13.1% |
...

---

## 1 · Login & authentication issues
**Description:** Users report being unable to log in, session timeouts, and SSO configuration problems.
**Suggested action:** Prioritize auth reliability audit; check SSO edge cases in staging.

**Representative examples:**
- "Can't log in after password reset, keeps saying invalid credentials"
- "SSO with Google breaks when using company email alias"
- "Session expires after 10 minutes even with remember me checked"

---
```

---

## Out of scope for v1

- Integración directa con APIs (Intercom, Linear, Slack)
- Dashboard visual (Streamlit / web UI)
- Análisis de sentimiento por cluster
- Comparación entre períodos
- Fine-tuning del modelo de embeddings
- Soporte para idiomas distintos al español/inglés

---

## Success metrics

| Métrica | Target |
| --- | --- |
| Tiempo de ejecución (1.000 rows) | < 60s |
| Coherencia de clusters (silhouette score) | > 0.35 |
| Tiempo de síntesis para PM | < 5 min desde export |
| Claridad del reporte (test con 3 PMs) | Comprensible sin instrucciones |

---

## Open questions

- [ ]  ¿Cuántas columnas de texto tiene el export del CRM? ¿Hay campos de notas además de la descripción?
- [ ]  ¿Los exports de Jira incluyen comentarios o solo el issue principal?
- [ ]  ¿El feedback está en español, inglés, o ambos? Afecta la estrategia de embeddings.
- [ ]  ¿Qué tan grandes son los datasets típicos? (100 rows, 500, 2000+)

---

## Appendix · Sample source config

```yaml
# sources.yaml
sources:
  - name: crm
    file: crm_export.csv
    text_columns:
      - description
      - notes
    id_column: ticket_id
    encoding: utf-8

  - name: jira
    file: jira_export.csv
    text_columns:
      - Summary
      - Description
    id_column: Issue key
    encoding: utf-8
```