# Actividad 003 · Agente IA (integrada en Command Center Ensenyem)

**Alumno:** Luis Jahir Rodriguez Cedeño  
**DNI:** 53945291X  
**Curso:** DAM2 - Proyecto Intermodular II  
**Unidad:** 301-Actividades final de unidad - Segundo trimestre  
**Actividad:** 003-AgenteIA

## 1) Continuidad del método de trabajo (001 → 002 → 003)

Se mantiene exactamente el mismo método operativo por bloques:

1. `001-Entrenamiento de IA personalizada` → ingesta y entrenamiento semántico.
2. `002-IA Generativa` → creación de piezas de negocio con contexto entrenado.
3. `003-AgenteIA` → ejecución de agente con políticas, canales y trazas.

La implementación técnica del agente se hace sobre el mismo producto unificado (`ensenyem_generative_studio`) para conservar contexto y trazabilidad end-to-end.

## 2) Objetivo de la actividad 003

Diseñar e implementar un **panel operativo de Agente IA** para Ensenyem que permita:

- simular conversaciones por canal,
- aplicar políticas de actuación,
- responder con datos del corpus entrenado,
- registrar trazas auditables de cada ejecución.

## 3) Implementación realizada

# Actividad 003 · Agente IA

**Alumno:** Luis Jahir Rodriguez Cedeño  
**DNI:** 53945291X  
**Curso:** DAM2 - Proyecto Intermodular II  
**Unidad:** 301-Actividades final de unidad - Segundo trimestre  
**Actividad:** 003-AgenteIA

## 1.- Introducción breve y contextualización (25%)

Esta actividad implementa un **Agente IA operativo** para Ensenyem capaz de simular atención por distintos canales (webchat, WhatsApp, email y CRM).

El propósito es pasar de “generar textos” a “ejecutar decisiones guiadas” con reglas de negocio, contexto entrenado y trazabilidad.

## 2.- Desarrollo detallado y preciso (25%)

### Conceptos técnicos usados

- **Canales de operación:** mismo motor con salidas adaptadas a canal.
- **Políticas de actuación:** `informativo`, `comercial`, `soporte`.
- **Trazabilidad:** persistencia de runs (`agent_runs`) con contexto y respuesta.

### Flujo técnico paso a paso

1. El usuario configura canal, política, objetivo y mensaje del cliente.
2. El backend valida que el índice esté entrenado (`get_training_status`).
3. Recupera contexto útil (`retrieve_context`).
4. Construye prompt de agente (`build_agent_prompt`) con reglas explícitas.
5. Ejecuta generación, guarda run y devuelve resultado + evidencias.

### Ejemplos reales de código

```python
def run_agent(channel: str, objective: str, policy_mode: str, customer_message: str) -> dict:
    if channel not in AGENT_CHANNELS:
        raise RuntimeError(f"Canal inválido. Usa: {', '.join(sorted(AGENT_CHANNELS))}")
    if policy_mode not in AGENT_POLICIES:
        raise RuntimeError(f"Política inválida. Usa: {', '.join(AGENT_POLICIES.keys())}")

    status = get_training_status()
    if status.get("pendingRetrain"):
        raise RuntimeError("Hay fuentes nuevas sin entrenar. Entrena el índice semántico antes de ejecutar el agente.")

    retrieval_query = f"{objective}\n{customer_message}".strip()
    context = retrieve_context(retrieval_query, top_k=5)
    prompt = build_agent_prompt(channel, objective, policy_mode, customer_message, context)
    response = ollama_chat([
        {"role": "system", "content": SYSTEM_ENSENYEM},
        {"role": "user", "content": prompt},
    ], temperature=0.1, num_predict=420)
    return {"response": response, "context": context}
```

```python
@app.post("/api/agent/run")
def api_agent_run():
    body = request.get_json(silent=True) or {}
    channel = str(body.get("channel", "")).strip().lower()
    objective = str(body.get("objective", "")).strip()
    policy_mode = str(body.get("policyMode", "informativo")).strip().lower()
    customer_message = str(body.get("customerMessage", "")).strip()

    result = run_agent(channel, objective, policy_mode, customer_message)
    return jsonify({"ok": True, **result})
```

## 3.- Aplicación práctica (25%)

### Aplicación en el prototipo

En la pestaña Agente IA se opera con dos modos:

- **Ejecución manual:** lanzar una simulación concreta.
- **Workflow automático:** `Web → Entrenar → Agente` en un clic.

### Dónde está el código (aclaración importante)

El código de 003 **sí existe**, pero está integrado en el proyecto unificado:

- Backend: `002-IA Generativa/ensenyem_generative_studio/app.py`
  - `build_agent_prompt`, `run_agent`, `list_agent_runs`, `get_agent_overview`
  - endpoints `/api/agent/overview`, `/api/agent/run`, `/api/agent/runs`
- Frontend: `002-IA Generativa/ensenyem_generative_studio/static/app.js`
  - `runAgent`, `loadAgentRuns`, `runAutoWorkflow`, `renderAgentResponse`
- UI: `002-IA Generativa/ensenyem_generative_studio/templates/index.html`
  - formulario de agente, estado y trazas

### Errores comunes y prevención

- **Error:** ejecutar agente con índice desactualizado.  
  **Prevención:** comprobar `pendingRetrain` y reentrenar.
- **Error:** confundir texto bonito con exactitud.  
  **Prevención:** revisar sección de evidencias usadas.
- **Error:** no guardar trazas para revisión.  
  **Prevención:** usar `agent_runs` y consultar histórico.

## 4.- Conclusión breve (25%)

La Actividad 003 completa el ciclo 001→002→003: primero entreno conocimiento, luego genero contenido y finalmente ejecuto un agente con políticas y trazabilidad. Este diseño permite escalar a canales reales sin perder control técnico.
