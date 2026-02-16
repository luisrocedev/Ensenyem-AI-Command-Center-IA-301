# Actividad 001 · Entrenamiento de IA personalizada

**Alumno:** Luis Jahir Rodriguez Cedeño

# Actividad 001 · Entrenamiento de IA personalizada

**Alumno:** Luis Jahir Rodriguez Cedeño  
**DNI:** 53945291X  
**Curso:** DAM2 - Proyecto Intermodular II  
**Unidad:** 301-Actividades final de unidad - Segundo trimestre  
**Ruta:** `001-Entrenamiento de IA personalizada/ollama_academic_trainer`

## 1.- Introducción breve y contextualización (25%)

En esta actividad implemento un sistema de **entrenamiento de IA local** para Ensenyem, orientado a responder dudas reales de empresa con información propia y no con respuestas genéricas del modelo base.

El objetivo práctico es construir un flujo reproducible de:

1. ingesta de fuentes,
2. indexación semántica,
3. consulta con evidencias,
4. comparación entre comportamiento base y entrenado.

Este enfoque se utiliza cuando una organización necesita precisión sobre su propio catálogo, políticas y servicios.

## 2.- Desarrollo detallado y preciso (25%)

### Definiciones técnicas aplicadas

- **Corpus:** conjunto de documentos/fuentes de Ensenyem para alimentar la IA.
- **Chunking:** división del texto en bloques para indexación eficiente.
- **Embeddings:** vectores numéricos para recuperar contexto por similitud semántica.
- **Grounding:** respuesta condicionada por evidencia del corpus para reducir alucinaciones.

### Funcionamiento paso a paso

1. El usuario carga fuentes (`document`, `pdf`, `youtube`, `website`).
2. El backend normaliza y persiste datos en SQLite (`knowledge_sources`).
3. El entrenamiento crea `corpus_documents` y `corpus_chunks` con embeddings.
4. Las preguntas recuperan contexto (`retrieve_context`) y generan respuesta anclada.
5. Se registran trazas para análisis y mejora.

### Ejemplos reales de código

```python
def train_semantic_index() -> dict:
	with get_db() as connection:
		connection.execute("DELETE FROM corpus_chunks")
		connection.execute("DELETE FROM corpus_documents")

		total_chunks = 0
		for source in sources:
			doc_id = connection.execute(
				"INSERT INTO corpus_documents (filename, content, created_at) VALUES (?, ?, ?)",
				(source["title"], source["content"], now_iso()),
			).lastrowid

			chunks = split_chunks(source["content"])
			for index, chunk in enumerate(chunks):
				embedding = ollama_embedding(chunk)
				connection.execute(
					"""
					INSERT INTO corpus_chunks (document_id, chunk_index, content, embedding_json, created_at)
					VALUES (?, ?, ?, ?, ?)
					""",
					(doc_id, index, chunk, json.dumps(embedding), now_iso()),
				)
				total_chunks += 1
```

```python
def answer_question(question: str, mode: str) -> dict:
	context = retrieve_context(question)
	if not context:
		return {"answer": "No está en mi corpus de entrenamiento.", "context": []}

	context_text = "\n\n".join([f"[{item['filename']}|score={item['score']:.3f}]\n{item['content']}" for item in context])
	answer = ollama_chat(
		TRAINED_MODEL,
		[
			{"role": "system", "content": SYSTEM_TUNED},
			{"role": "user", "content": f"Contexto:\n{context_text}\n\nPregunta:\n{question}"},
		],
	)
	return {"answer": answer, "context": context}
```

## 3.- Aplicación práctica (25%)

### Aplicación en el prototipo

Se entrena el asistente de Ensenyem con documentación propia y luego se valida en consultas de negocio (servicios, formación, procesos internos).

### Cómo se usa en práctica

```bash
cd "001-Entrenamiento de IA personalizada/ollama_academic_trainer"
pip install -r requirements.txt
python app.py
```

Después, en la interfaz:

1. importo fuentes,
2. pulso entrenar,
3. pruebo preguntas en modo entrenado.

### Errores comunes y cómo evitarlos

- **Error:** entrenar sin fuentes.  
  **Prevención:** validar que `knowledge_sources` tenga contenido.
- **Error:** embeddings fallan por modelo ausente.  
  **Prevención:** ejecutar `ollama pull nomic-embed-text`.
- **Error:** respuestas genéricas sin evidencia.  
  **Prevención:** usar grounding y devolver "No está en mi corpus" cuando no haya contexto.

## 4.- Conclusión breve (25%)

La Actividad 001 deja una base técnica sólida: ingesta, entrenamiento semántico, recuperación contextual y validación objetiva. Esta base enlaza directamente con la 002 (IA Generativa), donde el mismo corpus se usa para crear contenido empresarial con trazabilidad.

## Evidencias de implementación (código real)

- `app.py` (`split_chunks`, `ollama_embedding`, `train_semantic_index`, `retrieve_context`, `answer_question`)
- `templates/index.html`
- `static/app.js`
- `static/styles.css`
- `benchmark.json`
