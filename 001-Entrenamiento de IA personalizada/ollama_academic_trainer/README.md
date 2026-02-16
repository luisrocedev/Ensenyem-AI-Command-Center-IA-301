# Ensenyem AI Training Hub · Actividad 001

Espacio de entrenamiento local para el bot de Ensenyem (orientado a futura integración por WhatsApp), con comparación objetiva **antes/después**.

## Qué implementa

- **Ingesta de conocimiento Ensenyem:** PDF, documentos internos y enlaces YouTube con transcripción.
- **Antes (baseline):** respuesta con modelo base de Ollama sin contexto empresarial.
- **Después (trained):** respuesta con recuperación semántica (RAG local) sobre conocimiento Ensenyem.
- **Evaluación automática:** benchmark con preguntas de control y score por cobertura de palabras clave.
- **Persistencia completa (SQLite):** corpus, chunks vectoriales, ejecuciones de entrenamiento, benchmark y logs de interacción.

## Requisitos

- Python 3.13+
- Ollama ejecutándose en local (`http://localhost:11434`)
- Modelos sugeridos:
  - `ollama pull qwen2.5-coder:7b`
  - `ollama pull nomic-embed-text`
  - Para YouTube, el vídeo debe tener transcripción disponible.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Abrir en navegador: `http://127.0.0.1:5101`

## Flujo simple (interfaz)

1. **Paso 1: Cargar conocimiento**

- Documento manual (texto pegado)
- PDF corporativo
- URL de YouTube (se importa transcripción)
- Sitio web completo (rastreo por dominio con profundidad configurable)

2. **Paso 2: Entrenar bot Ensenyem**

- Botón `Entrenar bot Ensenyem`

3. **Paso 3: Simular WhatsApp**

- `Bot sin entrenar` vs `Bot Ensenyem entrenado`
- Ejecutar `Comparar antes/después` para benchmark cuantitativo

## Integración futura con WhatsApp

Este proyecto ya deja lista la parte de conocimiento + respuesta empresarial. Para conectarlo a WhatsApp, bastaría acoplar un webhook que reciba mensaje y llame a `/api/ask` en modo `trained`.

## Lectura de web corporativa

- Endpoint: `POST /api/sources/website`
- Parámetros: `url`, `title`, `maxPages` (1-60), `maxDepth` (0-4)
- Comportamiento: recorre enlaces del mismo dominio, limpia HTML y guarda el texto consolidado como fuente de entrenamiento.

## Técnica adicional de ajuste (Modelfile)

Para crear un modelo local con prompt de sistema personalizado:

```bash
python scripts/build_personal_model.py
ollama create tutor-dam-local -f models/Modelfile.tutor-dam
```

Luego lanzar app con:

```bash
export OLLAMA_TRAINED_MODEL=tutor-dam-local
python app.py
```

## Autor

- Luis Jahir Rodriguez Cedeño
- DNI: 53945291X
