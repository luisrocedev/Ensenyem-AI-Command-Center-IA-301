# Rúbrica de Evaluación — Actividad 001 · Entrenamiento de IA Personalizada

> **Alumno:** Luis Jahir Rodríguez Cedeño · DNI: 53945291X · DAM2 2025/26  
> **Proyecto:** `001-Entrenamiento de IA personalizada / ollama_academic_trainer`  
> **Puerto:** `5101`

---

## Criterios de calificación (10/10)

| Bloque                              | Peso | Contenido exigido                                                   |
| ----------------------------------- | ---- | ------------------------------------------------------------------- |
| 1. Introducción y contextualización | 25 % | Concepto general + contexto de uso                                  |
| 2. Desarrollo detallado y preciso   | 25 % | Definiciones, terminología, proceso paso a paso, ejemplos de código |
| 3. Aplicación práctica              | 25 % | Ejemplo real ejecutable, errores comunes y prevención               |
| 4. Conclusión breve                 | 25 % | Resumen + enlace con otros contenidos de la unidad                  |

---

## 1. Introducción breve y contextualización (25 %)

### Qué se evalúa

- Explicar qué es un sistema de **entrenamiento de IA local** y por qué una empresa lo necesita.
- Contextualizar el caso de uso: **Ensenyem** necesita que su asistente WhatsApp responda con datos propios (no genéricos).

### Evidencia en la memoria

| Criterio                                | Cumple | Dónde se demuestra                                                           |
| --------------------------------------- | ------ | ---------------------------------------------------------------------------- |
| Concepto general explicado con claridad | ✅     | Sección 1 de la memoria: flujo ingesta → indexación → consulta → comparación |
| Contexto de uso identificado            | ✅     | _"responder dudas reales de empresa con información propia"_                 |
| Mención del enfoque técnico elegido     | ✅     | RAG local con Ollama + embeddings + SQLite                                   |
| Vocabulario profesional                 | ✅     | Corpus, chunking, embeddings, grounding                                      |

### Puntos clave que debe contener la respuesta

- **Corpus:** conjunto de documentos corporativos de Ensenyem para alimentar la IA.
- **Embeddings:** vectores numéricos que permiten recuperar contexto por similitud semántica.
- **Grounding:** respuesta condicionada por evidencia del corpus → reduce alucinaciones.
- **Flujo:** ingesta de fuentes → normalización → entrenamiento → consulta con evidencias → comparación baseline vs trained.

---

## 2. Desarrollo detallado y preciso (25 %)

### Qué se evalúa

- Definiciones técnicas correctas y completas.
- Terminología apropiada.
- Proceso paso a paso.
- Ejemplos reales de código funcional.

### Evidencia en la memoria

| Criterio                                               | Cumple | Dónde se demuestra                                                                                  |
| ------------------------------------------------------ | ------ | --------------------------------------------------------------------------------------------------- |
| Definiciones (corpus, chunking, embeddings, grounding) | ✅     | Sección 2 de la memoria                                                                             |
| Funcionamiento paso a paso (5 pasos)                   | ✅     | Carga fuentes → normaliza SQLite → entrena chunks/embeddings → recupera contexto → genera respuesta |
| Código real de `train_semantic_index()`                | ✅     | Snippet completo con DELETE + INSERT chunks + embedding                                             |
| Código real de `answer_question()`                     | ✅     | Snippet con retrieve_context + ollama_chat                                                          |
| Código compila y funciona                              | ✅     | `app.py` en puerto 5101 ejecutable                                                                  |

### Snippets clave que debe contener

```python
# Entrenamiento semántico — split + embed + persist
def train_semantic_index() -> dict:
    connection.execute("DELETE FROM corpus_chunks")
    connection.execute("DELETE FROM corpus_documents")
    for source in sources:
        doc_id = connection.execute("INSERT INTO corpus_documents ...").lastrowid
        chunks = split_chunks(source["content"])
        for index, chunk in enumerate(chunks):
            embedding = ollama_embedding(chunk)
            connection.execute("INSERT INTO corpus_chunks ...")
```

```python
# Respuesta con contexto entrenado
def answer_question(question, mode):
    context = retrieve_context(question)      # cosine_similarity sobre embeddings
    context_text = "\n\n".join([...])
    answer = ollama_chat(TRAINED_MODEL, [
        {"role": "system", "content": SYSTEM_TUNED},
        {"role": "user", "content": f"Contexto:\n{context_text}\n\nPregunta:\n{question}"},
    ])
```

```python
# Similitud coseno — núcleo de la recuperación
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2)
```

### Tablas SQLite relevantes

| Tabla                 | Propósito                                            |
| --------------------- | ---------------------------------------------------- |
| `knowledge_sources`   | Fuentes originales (document, pdf, youtube, website) |
| `corpus_documents`    | Documentos normalizados tras entrenamiento           |
| `corpus_chunks`       | Fragmentos con embeddings vectoriales                |
| `training_runs`       | Log de ejecuciones de entrenamiento                  |
| `benchmark_questions` | Preguntas de control con keywords esperadas          |
| `benchmark_results`   | Resultados baseline vs trained por run               |
| `interaction_logs`    | Histórico de preguntas/respuestas del usuario        |

---

## 3. Aplicación práctica (25 %)

### Qué se evalúa

- Demostración funcional del concepto.
- Ejemplo ejecutable (código real, no pseudocódigo).
- Errores comunes y cómo evitarlos.

### Evidencia en la memoria

| Criterio                      | Cumple | Dónde se demuestra                                           |
| ----------------------------- | ------ | ------------------------------------------------------------ |
| Instrucciones de ejecución    | ✅     | `cd ollama_academic_trainer && pip install && python app.py` |
| Flujo de uso en interfaz      | ✅     | Importar fuentes → entrenar → probar preguntas               |
| Errores comunes documentados  | ✅     | 3 errores con prevención explícita                           |
| Código comprobado y funcional | ✅     | Puerto 5101 operativo                                        |

### Errores comunes y prevención

| Error                                         | Prevención                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Entrenar sin fuentes cargadas                 | Validar que `knowledge_sources` tenga contenido antes de llamar a `/api/train/run`          |
| Embeddings fallan por modelo ausente          | Ejecutar `ollama pull nomic-embed-text` antes de entrenar                                   |
| Respuestas genéricas sin evidencia del corpus | Usar grounding: si `bestScore < 0.33` y `overlap < 0.12`, devolver _"No está en mi corpus"_ |

### Tipos de ingesta soportados

| Tipo             | Endpoint                          | Técnica                                       |
| ---------------- | --------------------------------- | --------------------------------------------- |
| Documento manual | `POST /api/sources/document`      | Texto pegado directamente                     |
| PDF corporativo  | `POST /api/sources/pdf`           | `pypdf.PdfReader` extrae texto                |
| YouTube          | `POST /api/sources/youtube`       | `youtube_transcript_api` extrae transcripción |
| Web completa     | `POST /api/sources/website/start` | Crawler BFS por dominio con `BeautifulSoup`   |

---

## 4. Conclusión breve (25 %)

### Qué se evalúa

- Resumen de los puntos clave.
- Conexión con otros contenidos de la unidad (actividades 002 y 003).

### Evidencia en la memoria

| Criterio                        | Cumple | Dónde se demuestra                                                                                       |
| ------------------------------- | ------ | -------------------------------------------------------------------------------------------------------- |
| Resumen de puntos clave         | ✅     | _"base técnica sólida: ingesta, entrenamiento semántico, recuperación contextual y validación objetiva"_ |
| Conexión con 002 y 003          | ✅     | _"enlaza directamente con la 002 (IA Generativa), donde el mismo corpus se usa para crear contenido"_    |
| Visión de continuidad del ciclo | ✅     | 001 → 002 → 003 como evolución técnica única                                                             |

### Puntos de resumen esperados

1. Se construye un **pipeline completo** de entrenamiento local: ingesta → chunking → embeddings → recuperación → respuesta.
2. El benchmark **baseline vs trained** cuantifica objetivamente la mejora.
3. El grounding con umbrales evita alucinaciones de forma programática.
4. La base de conocimiento enlaza directamente con la 002 (generación) y la 003 (agente).

---

## Criterios transversales de calidad

| Criterio                            | Estado |
| ----------------------------------- | ------ |
| Ortografía y gramática correctas    | ✅     |
| Organización en secciones/viñetas   | ✅     |
| Lenguaje técnico propio del alumno  | ✅     |
| Todo el código es válido y funciona | ✅     |
| No hay plagio                       | ✅     |

---

## Archivos de evidencia

| Archivo                                        | Propósito                                  |
| ---------------------------------------------- | ------------------------------------------ |
| `ollama_academic_trainer/app.py`               | Backend completo (Flask + SQLite + Ollama) |
| `ollama_academic_trainer/templates/index.html` | Interfaz web                               |
| `ollama_academic_trainer/static/app.js`        | Lógica frontend                            |
| `ollama_academic_trainer/static/styles.css`    | Estilos visuales                           |
| `ollama_academic_trainer/benchmark.json`       | Preguntas de control con keywords          |
| `Actividad_EntrenamientoIA_53945291X.md`       | Memoria de la actividad                    |
