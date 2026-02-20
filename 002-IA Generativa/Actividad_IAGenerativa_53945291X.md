# Actividad 002 · IA Generativa

**Alumno:** Luis Jahir Rodriguez Cedeño  
**DNI:** 53945291X  
**Curso:** DAM2 - Proyecto Intermodular II  
**Unidad:** 301-Actividades final de unidad - Segundo trimestre

## 1.- Introducción breve y contextualización (25%)

La Actividad 002 aplica IA Generativa en un contexto empresarial real: crear mensajes y piezas comerciales de Ensenyem a partir del conocimiento entrenado en la actividad anterior.

No se trata de “preguntar al modelo y copiar la respuesta”, sino de un flujo controlado donde el contenido se genera con recuperación de contexto, reglas de seguridad y trazabilidad.

## 2.- Desarrollo detallado y preciso (25%)

### Definiciones y terminología técnica

- **RAG (Retrieval-Augmented Generation):** recuperar contexto relevante antes de generar.
- **Prompt estructurado:** prompt con objetivo, tono, audiencia y reglas de salida.
- **Persistencia de generaciones:** guardado de resultados y referencias para auditoría.

### Funcionamiento paso a paso

1. El usuario define `taskType`, `topic`, `tone`, `audience` y `length`.
2. El backend valida entrada y recupera contexto semántico (`retrieve_context`).
3. Se construye un prompt acotado (`build_generation_prompt`).
4. Ollama genera contenido con ese contexto.
5. Se guarda resultado en `generations` junto con referencias.

### Ejemplo real de código

```python
@app.post("/api/generate")
def api_generate():
    body = request.get_json(silent=True) or {}
    task_type = str(body.get("taskType", "")).strip()
    topic = str(body.get("topic", "")).strip()

    if task_type not in TASKS:
        return jsonify({"ok": False, "error": f"taskType inválido. Usa: {', '.join(TASKS.keys())}"}), 400
    if not topic:
        return jsonify({"ok": False, "error": "topic es obligatorio"}), 400

    context = retrieve_context(topic)
    if not context:
        return jsonify({"ok": False, "error": "No hay contexto relevante. Entrena primero en pestaña Entrenamiento."}), 400

    prompt = build_generation_prompt(task_type, topic, tone, audience, length, context)
    text = ollama_chat([
        {"role": "system", "content": SYSTEM_ENSENYEM},
        {"role": "user", "content": prompt},
    ], temperature=0.15, num_predict=420)
```

```python
def build_generation_prompt(task_type: str, topic: str, tone: str, audience: str, length: str, context: list[dict]) -> str:
    context_text = "\n\n".join([f"[{x['filename']}|score={x['score']:.3f}]\n{x['content']}" for x in context])
    return f"""
Objetivo: {TASKS.get(task_type, task_type)}
Tema: {topic}
Tono: {tone or 'Profesional'}
Audiencia: {audience or 'Interesados en formación'}
Longitud: {length or 'media'}

Reglas:
- Usa SOLO información del contexto.
- No inventes datos de procesos internos.
- Si falta dato, indica: 'Dato no disponible en el corpus de Ensenyem'.
""".strip()
```

## 3.- Aplicación práctica (25%)

### Aplicación en clase

En el panel “IA Generativa” se crean casos reales:

- respuesta WhatsApp a una empresa,
- email comercial de propuesta formativa,
- post de lanzamiento de programa.

### Ejecución práctica

```bash
cd "002-IA Generativa/ensenyem_generative_studio"
pip install -r requirements.txt
python app.py
```

### Errores comunes y prevención

- **Error:** generar sin entrenar índice.  
  **Prevención:** revisar `pendingRetrain` en `/api/train/status`.
- **Error:** contexto contaminado por ruido web.  
  **Prevención:** limpieza semántica del crawler y reentrenado.
- **Error:** prompts ambiguos.  
  **Prevención:** definir objetivo, audiencia y tono de forma explícita.

## 4.- Conclusión breve (25%)

La actividad demuestra que la IA Generativa aporta valor cuando está anclada al conocimiento propio de la empresa. Conecta directamente con la 001 (entrenamiento) y prepara la 003 (agente), manteniendo trazabilidad y calidad en un flujo único.

## Separación de entregas (001 / 002 / 003)

- `001-Entrenamiento de IA personalizada` → memoria propia.
- `002-IA Generativa` → esta memoria.
- `003-AgenteIA` → `003-AgenteIA/Actividad_AgenteIA_53945291X.md`.
