# Rúbrica de Evaluación — Actividad 003 · Agente IA

> **Alumno:** Luis Jahir Rodríguez Cedeño · DNI: 53945291X · DAM2 2025/26  
> **Proyecto:** `003-AgenteIA` (código integrado en `002-IA Generativa / ensenyem_generative_studio`)  
> **Puerto:** `5102` (unificado con 002)

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

- Explicar qué es un **Agente IA autónomo** que opera sobre canales de comunicación empresariales.
- Contextualizar el caso de uso: **Ensenyem** despliega un agente que responde automáticamente por WhatsApp, email, webchat y CRM con política configurable.

### Evidencia en la memoria

| Criterio                                | Cumple | Dónde se demuestra                                                                                    |
| --------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------- |
| Concepto general explicado con claridad | ✅     | Sección 1: agente que interpreta consultas, recupera contexto y genera respuesta adaptada al canal    |
| Contexto de uso identificado            | ✅     | _"agente conversacional multicanal con políticas configurables"_                                      |
| Diferenciación respecto a 001 y 002     | ✅     | 001 = entrenamiento; 002 = generación de contenido; 003 = **agente autónomo con canales y políticas** |
| Vocabulario profesional                 | ✅     | Agente, canal, política, trazabilidad, workflow                                                       |

### Puntos clave que debe contener la respuesta

- **Agente IA:** módulo autónomo que recibe una consulta, recupera contexto del corpus y genera una respuesta adaptada a un canal y política específicos.
- **Canales:** webchat, whatsapp, email, crm — cada uno con su tono y formato.
- **Políticas:** informativo, comercial, soporte — determinan el enfoque de la respuesta.
- **Trazabilidad:** cada ejecución queda registrada en `agent_runs` con canal, política, pregunta, respuesta y contexto usado.
- **Workflow completo:** Web → Entrenar → Agente (ciclo automático 001 → 002 → 003).

---

## 2. Desarrollo detallado y preciso (25 %)

### Qué se evalúa

- Definiciones técnicas correctas y completas.
- Terminología apropiada.
- Proceso paso a paso.
- Ejemplos reales de código funcional.

### Evidencia en la memoria

| Criterio                                             | Cumple | Dónde se demuestra                                                                              |
| ---------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------- |
| Definiciones (agente, canal, política, trazabilidad) | ✅     | Sección 2 de la memoria                                                                         |
| Funcionamiento paso a paso (4 pasos)                 | ✅     | Recibir consulta → seleccionar canal/política → recuperar contexto → generar respuesta adaptada |
| Código real de `run_agent()`                         | ✅     | Snippet completo con build_agent_prompt + ollama_chat + log                                     |
| Código real de `build_agent_prompt()`                | ✅     | Snippet con inyección de canal, política y contexto                                             |
| Código compila y funciona                            | ✅     | Integrado en `app.py` puerto 5102                                                               |

### Snippets clave que debe contener

```python
# Canales del agente — cada uno define tono y formato
AGENT_CHANNELS = {
    "webchat":  {"label": "Chat Web",  "icon": "🌐", "tone": "profesional y conciso"},
    "whatsapp": {"label": "WhatsApp",  "icon": "💬", "tone": "cercano y breve"},
    "email":    {"label": "Email",     "icon": "📧", "tone": "formal y estructurado"},
    "crm":      {"label": "CRM",       "icon": "🏢", "tone": "técnico y detallado"},
}
```

```python
# Políticas del agente — perfil de comportamiento
AGENT_POLICIES = {
    "informativo": {"label": "Informativo", "goal": "informar objetivamente"},
    "comercial":   {"label": "Comercial",   "goal": "persuadir y vender"},
    "soporte":     {"label": "Soporte",     "goal": "resolver problemas técnicos"},
}
```

```python
# Construcción del prompt del agente
def build_agent_prompt(channel, policy, question, context_chunks):
    ch = AGENT_CHANNELS[channel]
    po = AGENT_POLICIES[policy]
    context_block = "\n\n".join(
        [f"[Fragmento {i+1}]\n{c['text']}" for i, c in enumerate(context_chunks)]
    )
    system_msg = (
        f"Eres un agente de Ensenyem en el canal {ch['label']}.\n"
        f"Tu tono es: {ch['tone']}.\n"
        f"Tu objetivo es: {po['goal']}.\n\n"
        f"Usa EXCLUSIVAMENTE esta información:\n{context_block}"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]
```

```python
# Ejecución del agente con trazabilidad completa
def run_agent(channel, policy, question):
    context = retrieve_context(question, top_n=5)
    messages = build_agent_prompt(channel, policy, question, context)
    answer = ollama_chat(TRAINED_MODEL, messages)
    connection.execute(
        "INSERT INTO agent_runs (channel, policy, question, answer, context_used, ...) VALUES (...)",
        [channel, policy, question, answer, json.dumps(context), ...]
    )
    return {"answer": answer, "context": context, "channel": channel, "policy": policy}
```

### Tabla `agent_runs`

| Columna        | Tipo        | Propósito                                         |
| -------------- | ----------- | ------------------------------------------------- |
| `id`           | INTEGER PK  | Identificador único                               |
| `channel`      | TEXT        | Canal seleccionado (webchat/whatsapp/email/crm)   |
| `policy`       | TEXT        | Política aplicada (informativo/comercial/soporte) |
| `question`     | TEXT        | Pregunta del usuario                              |
| `answer`       | TEXT        | Respuesta generada por el agente                  |
| `context_used` | TEXT (JSON) | Chunks de contexto utilizados                     |
| `created_at`   | TEXT        | Timestamp de ejecución                            |

---

## 3. Aplicación práctica (25 %)

### Qué se evalúa

- Demostración funcional del concepto.
- Ejemplo ejecutable (código real, no pseudocódigo).
- Errores comunes y cómo evitarlos.

### Evidencia en la memoria

| Criterio                     | Cumple | Dónde se demuestra                                          |
| ---------------------------- | ------ | ----------------------------------------------------------- |
| Instrucciones de ejecución   | ✅     | Mismo `app.py` que 002 — pestaña "Agente" en la UI          |
| Flujo de uso del agente      | ✅     | Seleccionar canal → política → escribir pregunta → ejecutar |
| Errores comunes documentados | ✅     | 3 errores con prevención                                    |
| Trazabilidad verificable     | ✅     | `GET /api/agent/runs` devuelve histórico completo           |

### Errores comunes y prevención

| Error                                                        | Prevención                                                                             |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Ejecutar agente sin corpus entrenado                         | Validar con `/api/train/status` antes; la UI muestra badge de estado                   |
| Respuesta inadecuada para el canal (tono formal en WhatsApp) | El prompt inyecta `"Tu tono es: {tone}"` automáticamente, adaptando el estilo al canal |
| No poder auditar una respuesta                               | Cada run se guarda en `agent_runs` con `context_used` (JSON de los chunks utilizados)  |

### Endpoints del Agente

| Método | Ruta                  | Función                                                    |
| ------ | --------------------- | ---------------------------------------------------------- |
| `GET`  | `/api/agent/overview` | Resumen: canales disponibles, políticas, estado del corpus |
| `POST` | `/api/agent/run`      | Ejecutar el agente con `{channel, policy, question}`       |
| `GET`  | `/api/agent/runs`     | Listar todas las ejecuciones del agente                    |

### Ejemplo de llamada

```bash
curl -X POST http://localhost:5102/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "whatsapp",
    "policy": "comercial",
    "question": "¿Qué cursos ofrecéis para empresas?"
  }'
```

Respuesta esperada:

```json
{
  "ok": true,
  "answer": "¡Hola! 😊 En Ensenyem tenemos cursos a medida para empresas...",
  "channel": "whatsapp",
  "policy": "comercial",
  "context": [{"text": "...", "score": 0.87}, ...]
}
```

---

## 4. Conclusión breve (25 %)

### Qué se evalúa

- Resumen de los puntos clave.
- Conexión con otros contenidos de la unidad (actividades 001 y 002).

### Evidencia en la memoria

| Criterio                | Cumple | Dónde se demuestra                                                               |
| ----------------------- | ------ | -------------------------------------------------------------------------------- |
| Resumen de puntos clave | ✅     | _"agente multicanal con políticas, trazabilidad completa y workflow automático"_ |
| Conexión con 001        | ✅     | _"usa el corpus entrenado en la actividad 001"_                                  |
| Conexión con 002        | ✅     | _"reutiliza el motor de generación de la actividad 002"_                         |
| Visión end-to-end       | ✅     | Ciclo completo: ingesta → entrenamiento → generación → agente autónomo           |

### Puntos de resumen esperados

1. Se implementa un **agente IA autónomo** que responde en 4 canales con 3 políticas configurables.
2. Cada respuesta queda **trazada** en `agent_runs` con contexto y metadatos para auditoría.
3. El agente **reutiliza** el corpus de 001 y el motor de generación de 002 — no duplica código.
4. El ciclo **Web → Entrenar → Agente** se completa como sistema unificado de conocimiento empresarial.

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

| Archivo                                                                | Propósito                          |
| ---------------------------------------------------------------------- | ---------------------------------- |
| `../002-IA Generativa/ensenyem_generative_studio/app.py`               | Backend unificado (incluye agente) |
| `../002-IA Generativa/ensenyem_generative_studio/templates/index.html` | Interfaz con pestaña Agente        |
| `../002-IA Generativa/ensenyem_generative_studio/static/app.js`        | Lógica frontend del agente         |
| `Actividad_AgenteIA_53945291X.md`                                      | Memoria de la actividad            |
