# Rúbrica de Evaluación — Actividad 002 · IA Generativa

> **Alumno:** Luis Jahir Rodríguez Cedeño · DNI: 53945291X · DAM2 2025/26  
> **Proyecto:** `002-IA Generativa / ensenyem_generative_studio`  
> **Puerto:** `5102`

---

## Criterios de calificación (10/10)

| Bloque | Peso | Contenido exigido |
|--------|------|-------------------|
| 1. Introducción y contextualización | 25 % | Concepto general + contexto de uso |
| 2. Desarrollo detallado y preciso | 25 % | Definiciones, terminología, proceso paso a paso, ejemplos de código |
| 3. Aplicación práctica | 25 % | Ejemplo real ejecutable, errores comunes y prevención |
| 4. Conclusión breve | 25 % | Resumen + enlace con otros contenidos de la unidad |

---

## 1. Introducción breve y contextualización (25 %)

### Qué se evalúa

- Explicar qué es la **IA Generativa aplicada** a la creación de contenido empresarial.
- Contextualizar el caso de uso: **Ensenyem** necesita producir resúmenes, respuestas de WhatsApp, posts sociales y campañas de email usando su propio corpus.

### Evidencia en la memoria

| Criterio | Cumple | Dónde se demuestra |
|----------|--------|--------------------|
| Concepto general explicado con claridad | ✅ | Sección 1: generación condicionada por corpus |
| Contexto de uso identificado | ✅ | *"convertir el corpus entrenado en contenido útil: resúmenes, mensajes, posts"* |
| Diferenciación respecto a la actividad 001 | ✅ | 001 = entrenamiento y consulta; 002 = **creación de contenido estructurado** a partir de ese conocimiento |
| Vocabulario profesional | ✅ | Prompt engineering, RAG, generación condicionada, hybrid retrieval |

### Puntos clave que debe contener la respuesta

- **IA Generativa condicionada:** la IA no inventa libremente; genera a partir de evidencia del corpus entrenado.
- **Prompt Engineering estructurado:** cada tipo de tarea (resumen, WhatsApp, social, email) tiene su propio prompt template.
- **Hybrid Retrieval:** recuperación combinada vectorial (coseno 80 %) + léxica (keywords 20 %) para máxima relevancia.
- **Extensión natural de 001:** el mismo corpus entrenado se reutiliza para generar contenido real de negocio.

---

## 2. Desarrollo detallado y preciso (25 %)

### Qué se evalúa

- Definiciones técnicas correctas y completas.
- Terminología apropiada.
- Proceso paso a paso.
- Ejemplos reales de código funcional.

### Evidencia en la memoria

| Criterio | Cumple | Dónde se demuestra |
|----------|--------|--------------------|
| Definiciones (prompt engineering, generación condicionada, task types) | ✅ | Sección 2 de la memoria |
| Funcionamiento paso a paso (4 pasos) | ✅ | Seleccionar tarea → configurar → generar con contexto → revisar resultado |
| Código real de `build_generation_prompt()` | ✅ | Snippet con mapeo de TASKS → system instructions |
| Código real de `retrieve_context()` | ✅ | Snippet con puntuación híbrida cosine + keyword |
| Código compila y funciona | ✅ | `app.py` en puerto 5102 ejecutable |

### Snippets clave que debe contener

```python
# Tareas de generación definidas
TASKS = {
    "course_summary":  {"label": "Resumen de curso",    "icon": "📝", "system": "..."},
    "whatsapp_reply":  {"label": "Respuesta WhatsApp",  "icon": "💬", "system": "..."},
    "social_post":     {"label": "Post redes sociales", "icon": "📱", "system": "..."},
    "email_campaign":  {"label": "Campaña de email",    "icon": "📧", "system": "..."},
}
```

```python
# Construcción del prompt de generación
def build_generation_prompt(task_key, topic, context_chunks, extra_instructions=""):
    task = TASKS[task_key]
    context_block = "\n\n".join(
        [f"[Fragmento {i+1}]\n{c['text']}" for i, c in enumerate(context_chunks)]
    )
    system_msg = (
        f"{task['system']}\n\n"
        f"Usa EXCLUSIVAMENTE esta información de nuestro corpus:\n{context_block}\n\n"
        f"Instrucciones adicionales: {extra_instructions}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Genera contenido sobre: {topic}"},
    ]
```

```python
# Recuperación híbrida: vector + léxico
def retrieve_context(question, top_n=5):
    q_embedding = ollama_embedding(question)
    chunks = connection.execute("SELECT * FROM corpus_chunks").fetchall()
    scored = []
    for chunk in chunks:
        vec_score = cosine_similarity(q_embedding, json.loads(chunk["embedding"]))
        kw_score = keyword_score(question, chunk["chunk_text"])
        final = vec_score * 0.8 + kw_score * 0.2     # Hybrid weighting
        scored.append({...})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
```

### Tipos de generación — TASKS

| Clave | Label | Descripción |
|-------|-------|-------------|
| `course_summary` | Resumen de curso | Resumen académico estructurado del contenido formativo |
| `whatsapp_reply` | Respuesta WhatsApp | Mensaje breve y profesional para canal de mensajería |
| `social_post` | Post redes sociales | Copy orientado a engagement para Instagram/LinkedIn |
| `email_campaign` | Campaña de email | Email marketing con asunto, cuerpo y CTA |

### Tablas SQLite relevantes (añadidas sobre 001)

| Tabla | Propósito |
|-------|-----------|
| `generations` | Log de cada generación: task, topic, prompt, output, tokens, modelo, timestamp |
| `knowledge_sources` | Fuentes originales (heredadas de 001) |
| `corpus_documents` | Documentos normalizados |
| `corpus_chunks` | Fragmentos con embeddings |
| `training_runs` | Histórico de entrenamientos |
| `interaction_logs` | Preguntas y respuestas |

---

## 3. Aplicación práctica (25 %)

### Qué se evalúa

- Demostración funcional del concepto.
- Ejemplo ejecutable (código real, no pseudocódigo).
- Errores comunes y cómo evitarlos.

### Evidencia en la memoria

| Criterio | Cumple | Dónde se demuestra |
|----------|--------|--------------------|
| Instrucciones de ejecución | ✅ | `cd ensenyem_generative_studio && pip install && python app.py` |
| Flujo de generación completo | ✅ | Seleccionar task → escribir tema → configurar → generar |
| Errores comunes documentados | ✅ | 3 errores con prevención |
| Código comprobado y funcional | ✅ | Puerto 5102 operativo |

### Errores comunes y prevención

| Error | Prevención |
|-------|------------|
| Generar sin corpus entrenado | Entrenar primero: el panel muestra warning si `corpus_chunks` está vacío |
| Prompt demasiado largo (contexto excesivo) | Limitar `top_n` a 5 chunks; el max de 700 chars/chunk garantiza tamaño controlado |
| Contenido genérico sin datos propios | El sistema inyecta `"Usa EXCLUSIVAMENTE esta información de nuestro corpus"` en el prompt |

### Endpoints de generación

| Método | Ruta | Función |
|--------|------|---------|
| `POST` | `/api/generate` | Genera contenido con RAG condicionado |
| `GET` | `/api/generations` | Lista histórico de generaciones |
| `GET` | `/api/context/preview` | Previsualiza los chunks que se usarían |
| `GET` | `/api/train/status` | Estado actual de entrenamiento (chunks, sources) |

---

## 4. Conclusión breve (25 %)

### Qué se evalúa

- Resumen de los puntos clave.
- Conexión con otros contenidos de la unidad (actividades 001 y 003).

### Evidencia en la memoria

| Criterio | Cumple | Dónde se demuestra |
|----------|--------|--------------------|
| Resumen de puntos clave | ✅ | *"generación condicionada por corpus, structured prompts, 4 tipos de contenido"* |
| Conexión con 001 | ✅ | *"usa el mismo corpus preparado en la actividad 001"* |
| Conexión con 003 | ✅ | *"el mismo motor de generación se reutiliza para el agente autónomo"* |
| Visión de producto | ✅ | Command Center unificado que enlaza ingesta → generación → agente |

### Puntos de resumen esperados

1. Se construye un **sistema completo de generación condicionada** por datos propios de la empresa.
2. Cada tipo de contenido tiene su **prompt template especializado** que asegura el tono y formato adecuados.
3. La **recuperación híbrida** (coseno 80 % + léxica 20 %) maximiza la relevancia del contexto inyectado.
4. El corpus entrenado en 001 es reutilizado, y la generación alimenta al agente de la 003.

---

## Criterios transversales de calidad

| Criterio | Estado |
|----------|--------|
| Ortografía y gramática correctas | ✅ |
| Organización en secciones/viñetas | ✅ |
| Lenguaje técnico propio del alumno | ✅ |
| Todo el código es válido y funciona | ✅ |
| No hay plagio | ✅ |

---

## Archivos de evidencia

| Archivo | Propósito |
|---------|-----------|
| `ensenyem_generative_studio/app.py` | Backend unificado (Flask + SQLite + Ollama + Generación) |
| `ensenyem_generative_studio/templates/index.html` | Interfaz web Command Center |
| `ensenyem_generative_studio/static/app.js` | Lógica frontend |
| `ensenyem_generative_studio/static/styles.css` | Estilos visuales |
| `Actividad_IAGenerativa_53945291X.md` | Memoria de la actividad |
