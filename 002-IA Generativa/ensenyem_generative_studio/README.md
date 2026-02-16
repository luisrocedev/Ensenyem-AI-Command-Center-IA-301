# Ensenyem AI Command Center · Actividad 002 (integrada)

Proyecto de **centro de mando único** para operar módulos IA por pestañas:

- **Entrenamiento IA (Actividad 001 integrada):** ingesta de fuentes, rastreo web, entrenamiento semántico y consulta entrenada.
- **IA Generativa (Actividad 002):** generación de piezas de negocio apoyadas en el contexto entrenado.
- **Agente IA (Actividad 003):** espacio preparado para evolución posterior.

## Objetivo

Centralizar en una sola app la operación del ciclo completo:

1. Cargar conocimiento de Ensenyem (documentos, PDF, YouTube, web).
2. Entrenar índice semántico local con Ollama.
3. Generar contenidos empresariales con trazabilidad de referencias.

## Funcionalidades

### Módulo Entrenamiento IA

- Alta de fuentes manuales.
- Importación de PDF y transcripciones de YouTube.
- Rastreo web automático (modo enterprise) con progreso visual en tiempo real.
- Entrenamiento de índice semántico local con embeddings.
- Preguntas entrenadas con evidencias.

### Módulo IA Generativa

# Ensenyem AI Command Center

Plataforma unificada para presentar tres mini proyectos integrados del módulo:

1. **001 Entrenamiento IA personalizada**
2. **002 IA Generativa**
3. **003 Agente IA**

La solución está implementada en una sola aplicación para conservar continuidad técnica y reutilizar el mismo contexto de conocimiento.

## Arquitectura funcional

### 001 · Entrenamiento IA

- Ingesta de fuentes: documento, PDF, YouTube y web.
- Rastreo web con progreso y cancelación.
- Entrenamiento semántico local con embeddings en Ollama.
- Validación de respuestas con evidencias del corpus.

### 002 · IA Generativa

- Generación de piezas de negocio (WhatsApp, email, social, ficha comercial).
- Prompting estructurado con reglas anti-alucinación.
- Context preview y trazabilidad de referencias.
- Historial persistente de generaciones.

### 003 · Agente IA

- Simulación operativa multicanal (`webchat`, `whatsapp`, `email`, `crm`).
- Políticas de actuación (`informativo`, `comercial`, `soporte`).
- Registro de ejecuciones en `agent_runs`.
- Workflow automático: **Web → Entrenar → Agente**.

## Stack técnico

- Backend: Flask
- Frontend: HTML + CSS + JavaScript (vanilla)
- IA local: Ollama (`/api/chat`, `/api/embeddings`)
- Base de datos: SQLite (`command_center.sqlite3`)

## Estructura relevante

- `app.py` → API, lógica de entrenamiento, generación y agente
- `templates/index.html` → UI por pestañas
- `static/app.js` → lógica cliente (workflow, acciones, render)
- `static/styles.css` → sistema visual y UX
- `docs/Actividad_IAGenerativa_53945291X.md` → memoria de la actividad 002

## Requisitos

- Python 3.11+
- Ollama activo en local

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
python app.py
```

Abrir en navegador: `http://127.0.0.1:5102`

## Variables de entorno opcionales

- `OLLAMA_BASE` (default: `http://localhost:11434`)
- `OLLAMA_GEN_MODEL` (default: `qwen2.5-coder:7b`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)

## Entregables académicos

- `../001-Entrenamiento de IA personalizada/Actividad_EntrenamientoIA_53945291X.md`
- `../002-IA Generativa/Actividad_IAGenerativa_53945291X.md`
- `../003-AgenteIA/Actividad_AgenteIA_53945291X.md`

## Autor

- Luis Jahir Rodriguez Cedeño
- DNI 53945291X
