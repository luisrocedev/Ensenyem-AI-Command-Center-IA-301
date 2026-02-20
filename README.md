<div align="center">

# 🧠 Ensenyem AI Command Center

### Plataforma integral de entrenamiento, generación y agente IA para empresas

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_AI-FF6B35?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.com)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Tres actividades conectadas en un ciclo completo: ingesta → entrenamiento → generación → agente autónomo.*

[Arquitectura](#arquitectura) · [Módulos](#módulos) · [Instalación](#instalación) · [API](#api-rest) · [Evaluación](#evaluación)

</div>

---

## 📋 Visión General

**Ensenyem AI Command Center** es un proyecto académico que implementa un **sistema completo de inteligencia artificial empresarial** desde cero, sin depender de APIs externas de pago. Utiliza [Ollama](https://ollama.com) para ejecutar modelos de lenguaje de forma 100 % local.

El sistema se compone de **tres módulos encadenados** que representan las tres fases de un pipeline de IA corporativo:

| Fase | Módulo | Puerto | Función |
|------|--------|--------|---------|
| 1️⃣ | **Entrenamiento IA** | `5101` | Ingesta de corpus, chunking, embeddings vectoriales, benchmark |
| 2️⃣ | **IA Generativa** | `5102` | Generación de contenido condicionado por RAG |
| 3️⃣ | **Agente IA** | `5102` | Agente autónomo multicanal con políticas configurables |

> Las fases 2 y 3 comparten backend en un **Command Center unificado** que integra entrenamiento, generación y agente en una sola interfaz.

---

## 🏗 Arquitectura

```
┌───────────────────────────────────────────────────────────────┐
│                   Ensenyem AI Command Center                  │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │  📥 Ingesta  │───▶│ 🧠 Training  │───▶│ 🔍 Retrieval │    │
│  │  doc/pdf/web │    │  chunks +    │    │  cosine 80%  │    │
│  │  /youtube    │    │  embeddings  │    │  lexical 20% │    │
│  └──────────────┘    └──────────────┘    └──────┬───────┘    │
│                                                  │            │
│                                    ┌─────────────┼──────┐    │
│                                    ▼             ▼      │    │
│                            ┌──────────┐  ┌───────────┐  │    │
│                            │ 📝 Gen.  │  │ 🤖 Agent  │  │    │
│                            │ content  │  │ multicanal │  │    │
│                            └──────────┘  └───────────┘  │    │
│                                                         │    │
│  ┌────────────────────────────────────────────────────┐ │    │
│  │                    SQLite DB                       │ │    │
│  │  sources · chunks · runs · generations · agent_runs│ │    │
│  └────────────────────────────────────────────────────┘ │    │
│                                                         │    │
│  ┌────────────────────────────────────────────────────┐ │    │
│  │              Ollama (Local AI)                     │ │    │
│  │  qwen2.5-coder:7b  ·  nomic-embed-text           │ │    │
│  └────────────────────────────────────────────────────┘ │    │
└───────────────────────────────────────────────────────────────┘
```

---

## 📦 Módulos

### 001 · Entrenamiento de IA Personalizada

> `001-Entrenamiento de IA personalizada/ollama_academic_trainer/`

Sistema completo de RAG (Retrieval-Augmented Generation) que transforma documentos corporativos en conocimiento consultable:

| Característica | Detalle |
|----------------|---------|
| **Ingesta multi-fuente** | Documentos manuales, PDF, YouTube transcripts, web crawling |
| **Chunking inteligente** | Fragmentos de 700 chars con overlap y separación por frases |
| **Embeddings vectoriales** | `nomic-embed-text` vía Ollama para representación semántica |
| **Recuperación híbrida** | Cosine similarity (80 %) + keyword overlap (20 %) |
| **Grounding** | Umbrales mínimos (`bestScore ≥ 0.33`, `overlap ≥ 0.12`) para evitar alucinaciones |
| **Benchmark** | Comparación objetiva baseline vs trained con preguntas de control |
| **Web Crawler** | BFS por dominio con filtrado de ruido y extracción de texto significativo |

### 002 · IA Generativa

> `002-IA Generativa/ensenyem_generative_studio/`

Motor de **generación de contenido condicionado** que utiliza el corpus entrenado para producir material útil:

| Tipo de generación | Descripción |
|--------------------|-------------|
| 📝 **Resumen de curso** | Resumen académico estructurado del contenido formativo |
| 💬 **Respuesta WhatsApp** | Mensaje breve y profesional para canal de mensajería |
| 📱 **Post redes sociales** | Copy orientado a engagement para Instagram / LinkedIn |
| 📧 **Campaña de email** | Email marketing con asunto, cuerpo y CTA |

Cada tipo tiene su **prompt template especializado** que asegura tono, formato y uso exclusivo de datos del corpus.

### 003 · Agente IA

> `003-AgenteIA/` (integrado en el Command Center de 002)

**Agente autónomo multicanal** que responde consultas adaptándose al canal y la política:

| Canales | Políticas |
|---------|-----------|
| 🌐 Web Chat | ℹ️ Informativo — informar objetivamente |
| 💬 WhatsApp | 💰 Comercial — persuadir y vender |
| 📧 Email | 🔧 Soporte — resolver problemas técnicos |
| 🏢 CRM | |

Cada ejecución queda registrada en `agent_runs` con contexto, canal, política y respuesta completa para **auditoría y trazabilidad**.

---

## 🚀 Instalación

### Requisitos previos

- **Python 3.11+**
- **Ollama** instalado y en ejecución (`ollama serve`)
- Modelos descargados:

```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

### Módulo 001 — Training Hub

```bash
cd "001-Entrenamiento de IA personalizada/ollama_academic_trainer"
pip install -r requirements.txt
python app.py                    # → http://localhost:5101
```

### Módulo 002 + 003 — Command Center (unificado)

```bash
cd "002-IA Generativa/ensenyem_generative_studio"
pip install -r requirements.txt
python app.py                    # → http://localhost:5102
```

---

## 🌐 API REST

### Módulo 001 — Training Hub (`:5101`)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/health` | Estado de salud del servicio |
| `GET` | `/api/stats` | Estadísticas (fuentes, chunks, runs) |
| `GET` | `/api/sources` | Listar fuentes del corpus |
| `POST` | `/api/sources/document` | Añadir documento manual |
| `POST` | `/api/sources/youtube` | Importar transcripción de YouTube |
| `POST` | `/api/sources/pdf` | Importar PDF corporativo |
| `POST` | `/api/sources/website/start` | Iniciar crawling de web |
| `DELETE` | `/api/sources/<id>` | Eliminar fuente |
| `POST` | `/api/train/run` | Ejecutar entrenamiento semántico |
| `POST` | `/api/ask` | Preguntar (baseline vs trained) |
| `POST` | `/api/evaluate` | Ejecutar benchmark |

### Módulo 002 + 003 — Command Center (`:5102`)

Incluye **todos los endpoints de 001** más:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/train/status` | Estado del entrenamiento |
| `GET` | `/api/context/preview` | Previsualizar chunks de contexto |
| `POST` | `/api/generate` | Generar contenido condicionado |
| `GET` | `/api/generations` | Histórico de generaciones |
| `GET` | `/api/agent/overview` | Resumen del agente (canales, políticas) |
| `POST` | `/api/agent/run` | Ejecutar el agente |
| `GET` | `/api/agent/runs` | Histórico de ejecuciones del agente |
| `POST` | `/api/sources/website/jobs/<id>/cancel` | Cancelar crawling activo |

---

## 🗄 Base de Datos

### Tablas principales

| Tabla | Módulo | Propósito |
|-------|--------|-----------|
| `knowledge_sources` | 001+ | Fuentes originales (document, pdf, youtube, website) |
| `corpus_documents` | 001+ | Documentos normalizados tras entrenamiento |
| `corpus_chunks` | 001+ | Fragmentos con embeddings vectoriales |
| `training_runs` | 001+ | Histórico de ejecuciones de entrenamiento |
| `benchmark_questions` | 001 | Preguntas de control con keywords esperadas |
| `benchmark_results` | 001 | Resultados comparativos baseline vs trained |
| `interaction_logs` | 001+ | Registro de preguntas/respuestas |
| `generations` | 002 | Log de generaciones (task, topic, prompt, output) |
| `agent_runs` | 003 | Log de ejecuciones del agente (canal, política, contexto) |

---

## 🔬 Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| **Backend** | Python 3.11 · Flask 3.x |
| **IA Local** | Ollama · qwen2.5-coder:7b · nomic-embed-text |
| **Base de datos** | SQLite 3 (ficheros locales) |
| **Web Crawling** | BeautifulSoup 4 · requests · threading |
| **PDF** | pypdf |
| **YouTube** | youtube-transcript-api |
| **Frontend** | HTML5 · CSS3 · JavaScript vanilla |

---

## 📊 Evaluación

Cada actividad incluye:

| Documento | Descripción |
|-----------|-------------|
| `Actividad_*_53945291X.md` | Memoria completa de la actividad (4 secciones × 25 %) |
| `Rubrica_Evaluacion_*.md` | Rúbrica de evaluación con criterios y evidencias |

### Estructura de evaluación

| Sección | Peso | Qué se evalúa |
|---------|------|----------------|
| Introducción y contextualización | 25 % | Concepto + contexto de uso |
| Desarrollo detallado | 25 % | Definiciones, código, proceso paso a paso |
| Aplicación práctica | 25 % | Ejemplo funcional, errores comunes |
| Conclusión | 25 % | Resumen + conexión con otras actividades |

---

## 📁 Estructura del repositorio

```
.
├── 001-Entrenamiento de IA personalizada/
│   ├── Actividad_EntrenamientoIA_53945291X.md
│   ├── Rubrica_Evaluacion_001.md
│   └── ollama_academic_trainer/
│       ├── app.py
│       ├── requirements.txt
│       ├── benchmark.json
│       ├── corpus/
│       ├── docs/
│       ├── scripts/
│       ├── static/
│       └── templates/
│
├── 002-IA Generativa/
│   ├── Actividad_IAGenerativa_53945291X.md
│   ├── Rubrica_Evaluacion_002.md
│   └── ensenyem_generative_studio/
│       ├── app.py
│       ├── requirements.txt
│       ├── docs/
│       ├── static/
│       └── templates/
│
├── 003-AgenteIA/
│   ├── Actividad_AgenteIA_53945291X.md
│   └── Rubrica_Evaluacion_003.md
│
└── README.md
```

---

## 👤 Autor

| | |
|---|---|
| **Alumno** | Luis Jahir Rodríguez Cedeño |
| **DNI** | 53945291X |
| **Ciclo** | DAM2 · 2025/26 |
| **Módulo** | IA-301 |
| **Centro** | IES de Teis |

---

<div align="center">

*Construido con ❤️ y modelos locales — sin APIs externas de pago.*

</div>
