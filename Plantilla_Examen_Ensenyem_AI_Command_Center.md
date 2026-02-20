# Ensenyem AI Command Center — Plantilla de Examen

**Alumno:** Luis Rodríguez Cedeño · **DNI:** 53945291X  
**Módulo:** IA-301 · **Curso:** DAM2 2025/26

---

## 1. Introducción

- **Qué es:** Plataforma integral de IA empresarial que unifica entrenamiento semántico (RAG), generación de contenido y agente autónomo multicanal, todo ejecutado de forma local con Ollama.
- **Contexto:** Módulo IA-301 — Tres actividades encadenadas que forman un pipeline completo de inteligencia artificial corporativa para la empresa ficticia **Ensenyem**.
- **Objetivos principales:**
  - Ingesta multi-fuente de conocimiento (documentos, PDF, YouTube, web crawling)
  - Entrenamiento semántico con chunking + embeddings vectoriales
  - Recuperación híbrida: coseno 80 % + léxica 20 %
  - Generación condicionada de contenido (4 tipos de tarea)
  - Agente IA autónomo con 4 canales y 3 políticas
  - Trazabilidad completa: cada operación queda registrada en SQLite
- **Tecnologías clave:**
  - Python 3.11, Flask 3.x, SQLite 3
  - Ollama local: `qwen2.5-coder:7b` (generación) + `nomic-embed-text` (embeddings)
  - BeautifulSoup 4 (crawling), pypdf (PDF), youtube-transcript-api (YouTube)
- **Arquitectura:** `app.py` (1361 líneas: ingesta, training, retrieval, generación, agente) → SQLite `command_center.sqlite3` (7 tablas) → Ollama API local → `templates/index.html` (SPA con pestañas)

---

## 2. Desarrollo de las partes

### 2.1 Esquema de Base de Datos — 7 tablas

- SQLite con 7 tablas que cubren el pipeline completo: fuentes → documentos → chunks → runs → logs → generaciones → agente
- Cada tabla tiene `created_at` para trazabilidad temporal

```python
def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                origin_ref TEXT DEFAULT '',
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corpus_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS corpus_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES corpus_documents(id)
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                chunks_indexed INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                status TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                context_chunks INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                topic TEXT NOT NULL,
                tone TEXT DEFAULT '',
                audience TEXT DEFAULT '',
                generated_text TEXT NOT NULL,
                refs_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                objective TEXT NOT NULL,
                policy_mode TEXT NOT NULL,
                customer_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                context_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
```

> **Explicación:** Las 7 tablas modelan el pipeline completo. `knowledge_sources` guarda las fuentes originales, `corpus_documents` y `corpus_chunks` almacenan los fragmentos indexados con embeddings, `training_runs` registra cada entrenamiento, `interaction_logs` traza preguntas/respuestas, `generations` guarda contenido generado y `agent_runs` almacena las ejecuciones del agente con contexto JSON.

### 2.2 Chunking inteligente — Fragmentación de texto

- `split_chunks()` parte el texto en fragmentos de máximo 700 caracteres
- Respeta fronteras de párrafo (`\n\n`) para mantener coherencia semántica
- Si un párrafo individual supera el límite, se corta secuencialmente

```python
def split_chunks(text: str, max_chars: int = 700) -> list[str]:
    paragraphs = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    if not paragraphs:
        stripped = text.strip()
        if not stripped:
            return []
        paragraphs = [stripped]

    chunks: list[str] = []
    current = ""

    def flush_long_paragraph(paragraph_text: str) -> None:
        start = 0
        total = len(paragraph_text)
        while start < total:
            end = min(start + max_chars, total)
            piece = paragraph_text[start:end].strip()
            if piece:
                chunks.append(piece)
            start = end

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            flush_long_paragraph(paragraph)
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks
```

> **Explicación:** El texto se divide por párrafos (`\n\n`). Si un párrafo cabe dentro del límite de 700 chars, se acumula con el anterior. Si no cabe, el acumulado se guarda como chunk y se empieza uno nuevo. Si un párrafo individual es más largo que 700 chars, se trocea secuencialmente con `flush_long_paragraph()`. Esto asegura que cada chunk mantenga coherencia de contenido.

### 2.3 Embeddings vectoriales — Ollama

- `ollama_embedding()` genera un vector numérico por cada fragmento de texto
- Compatible con múltiples versiones de la API de Ollama (fallback cascade)
- Modelo: `nomic-embed-text`

```python
def ollama_embedding(text: str) -> list[float]:
    last_error = None

    # Intento 1: /api/embeddings con "prompt"
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding:
            return embedding
    except Exception as exc:
        last_error = str(exc)

    # Intento 2: /api/embeddings con "input"
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if embedding and isinstance(embedding, list):
            return embedding
    except Exception as exc:
        last_error = f"{last_error} | {exc}"

    # Intento 3: /api/embed (API nueva)
    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        single = data.get("embedding")
        if single and isinstance(single, list):
            return single
        embeddings = data.get("embeddings") or []
        if embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
    except Exception as exc:
        last_error = f"{last_error} | {exc}"

    raise RuntimeError(f"No se pudo generar embeddings: {last_error}")
```

> **Explicación:** La función intenta 3 variantes de la API de Ollama para máxima compatibilidad: primero `/api/embeddings` con campo `prompt`, luego con `input`, y finalmente `/api/embed` (API más reciente). Este patrón de fallback cascade asegura que funcione con cualquier versión de Ollama instalada.

### 2.4 Entrenamiento semántico — Train Index

- `train_semantic_index()` procesa todas las fuentes: genera chunks + embeddings
- Limpia y reconstruye `corpus_documents` y `corpus_chunks` en cada entrenamiento
- Registra duración y estadísticas en `training_runs`

```python
def train_semantic_index() -> dict:
    ensure_ollama_embed_model_ready()

    with get_db() as conn:
        sources = conn.execute("SELECT title, content FROM knowledge_sources ORDER BY id").fetchall()
    if not sources:
        raise RuntimeError("No hay fuentes cargadas para entrenar")

    started = datetime.now(timezone.utc)
    with get_db() as conn:
        conn.execute("DELETE FROM corpus_chunks")
        conn.execute("DELETE FROM corpus_documents")

        chunks_total = 0
        for source in sources:
            doc_id = conn.execute(
                "INSERT INTO corpus_documents (filename, content, created_at) VALUES (?, ?, ?)",
                (source["title"], source["content"], now_iso()),
            ).lastrowid

            chunks = split_chunks(source["content"])
            for idx, chunk in enumerate(chunks):
                embedding = ollama_embedding(chunk)
                conn.execute(
                    "INSERT INTO corpus_chunks (document_id, chunk_index, content, embedding_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (doc_id, idx, chunk, json.dumps(embedding), now_iso()),
                )
                chunks_total += 1

        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        conn.execute(
            "INSERT INTO training_runs (run_type, model_name, chunks_indexed, duration_ms, status, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("semantic_index", EMBED_MODEL, chunks_total, duration_ms, "ok", f"docs={len(sources)}", now_iso()),
        )

    return {"documents": len(sources), "chunks": chunks_total, "durationMs": duration_ms}
```

> **Explicación:** Primero se verifican los prerrequisitos (modelo de embeddings disponible). Luego se vacían las tablas de chunks/documentos (rebuild limpio). Para cada fuente se crean chunks con `split_chunks()`, se genera el embedding vectorial de cada uno con `ollama_embedding()`, y se persisten en SQLite. Finalmente se registra el run con duración en milisegundos.

### 2.5 Similitud coseno + Overlap léxico — Recuperación Híbrida

- `cosine_similarity()` mide la similitud entre vectores de embeddings
- `lexical_overlap()` mide coincidencia de palabras clave (bag-of-words)
- `retrieve_context()` combina ambas: 80 % vector + 20 % léxico

```python
def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def lexical_overlap(query: str, text: str) -> float:
    q = tokenize(query)
    t = tokenize(text)
    if not q or not t:
        return 0.0
    return len(q.intersection(t)) / len(q)


def retrieve_context(topic: str, top_k: int = 4) -> list[dict]:
    query_embedding = ollama_embedding(topic)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT c.content, c.embedding_json, d.filename "
            "FROM corpus_chunks c JOIN corpus_documents d ON d.id = c.document_id"
        ).fetchall()

    scored: list[dict] = []
    for row in rows:
        emb = json.loads(row["embedding_json"])
        vector_score = cosine_similarity(query_embedding, emb)
        lexical = lexical_overlap(topic, row["content"])
        combined = vector_score * 0.8 + lexical * 0.2
        scored.append({
            "filename": row["filename"],
            "content": row["content"],
            "score": combined,
            "vectorScore": vector_score,
            "lexicalScore": lexical,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
```

> **Explicación:** La recuperación opera en dos niveles. **Nivel vectorial:** la pregunta se convierte en embedding y se compara con cada chunk por similitud coseno (producto escalar normalizado). **Nivel léxico:** se comparan las palabras del query vs el chunk (excluyendo stopwords). La puntuación final combina ambas: `combined = vector * 0.8 + lexical * 0.2`. Esto captura tanto semántica profunda como coincidencias exactas de términos.

### 2.6 Respuesta con contexto — RAG (answer_trained)

- Recupera contexto relevante → inyecta en el prompt → genera con Ollama
- Incluye reglas de grounding: si falta información, dice "No está en mi corpus"
- Cada interacción se registra en `interaction_logs`

```python
def answer_trained(question: str) -> dict:
    context = retrieve_context(question, top_k=5)
    if not context:
        return {"answer": "No está en mi corpus de entrenamiento.", "context": []}

    context_text = "\n\n".join(
        [f"[{item['filename']}|score={item['score']:.3f}]\n{item['content']}" for item in context]
    )
    prompt = (
        "REGLAS:\n"
        "- Responde solo con información explícita en el contexto.\n"
        "- Si falta información exacta, responde: 'No está en mi corpus de entrenamiento'.\n"
        "- Añade sección final 'Evidencias usadas'.\n\n"
        f"Contexto:\n{context_text}\n\nPregunta:\n{question}"
    )
    answer = ollama_chat([
        {"role": "system", "content": SYSTEM_ENSENYEM},
        {"role": "user", "content": prompt},
    ], temperature=0.05, num_predict=320)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO interaction_logs (mode, question, answer, context_chunks, created_at) VALUES (?, ?, ?, ?, ?)",
            ("trained", question, answer, len(context), now_iso()),
        )
    return {"answer": answer, "context": context}
```

> **Explicación:** El flujo RAG completo: 1) se recuperan los 5 chunks más relevantes, 2) se formatean con nombre del documento y puntuación, 3) se construye un prompt con reglas estrictas de grounding (no inventar), 4) se envía a Ollama con temperature muy baja (0.05) para máxima fidelidad, 5) se registra en `interaction_logs`. El LLM solo puede usar lo que hay en el contexto inyectado.

### 2.7 Web Crawler — Rastreo de dominio con BFS

- BFS (Breadth-First Search) por dominio con profundidad máxima
- Filtrado de ruido: elimina navegación, cookies, carrito, scripts
- Threading + cancelación: ejecuta en hilo de fondo con posibilidad de cancelar

```python
def crawl_website(base_url, max_pages=12, max_depth=2, progress_callback=None, should_stop=None):
    parsed_base = urlparse(candidate_url)
    base_domain = parsed_base.netloc.lower()
    queue = deque([(normalize_crawl_url(candidate_url), 0)])
    visited: set[str] = set()
    page_blocks: list[str] = []

    while queue and len(visited) < max_pages:
        if callable(should_stop) and should_stop():
            raise CrawlCancelled("Rastreo cancelado por el usuario")

        current_url, depth = queue.popleft()
        normalized = normalize_crawl_url(current_url)
        if normalized in visited:
            continue
        if is_blocked_crawl_path(urlparse(normalized).path):
            continue
        visited.add(normalized)

        response = requests.get(normalized, timeout=12)
        soup = BeautifulSoup(response.text, "html.parser")
        text = extract_meaningful_text_from_html(soup)
        if text:
            page_blocks.append(f"[URL] {normalized}\n[TITLE] {title}\n\n{text}")

        if depth < max_depth:
            for anchor in soup.find_all("a", href=True):
                next_url = normalize_crawl_url(urljoin(normalized, anchor["href"]))
                if urlparse(next_url).netloc.lower() == base_domain and next_url not in visited:
                    queue.append((next_url, depth + 1))

    return "\n\n\n".join(page_blocks), len(page_blocks)
```

> **Explicación:** El crawler usa una cola BFS (breadth-first search) con tupla `(url, profundidad)`. Para cada URL: normaliza, verifica que no esté visitada ni bloqueada (`/login`, `/checkout`), descarga HTML, extrae texto limpio (eliminando `<nav>`, `<footer>`, `<script>`, líneas de ruido tipo cookies), y extrae hipervínculos del mismo dominio. El `should_stop` callback permite cancelación vía endpoint REST. Máximo 80 páginas, profundidad 3, timeout 180s.

### 2.8 Generación de contenido — 4 tipos de tarea

- `TASKS` define 4 tipos: ficha de curso, WhatsApp, post social, email campaña
- `build_generation_prompt()` construye el prompt con contexto + reglas específicas
- El contenido generado se persiste en `generations`

```python
TASKS = {
    "course_summary": "Ficha comercial de curso",
    "whatsapp_reply": "Respuesta corta para WhatsApp",
    "social_post": "Post para redes sociales",
    "email_campaign": "Email de campaña",
}


def build_generation_prompt(task_type, topic, tone, audience, length, context):
    context_text = "\n\n".join(
        [f"[{x['filename']}|score={x['score']:.3f}]\n{x['content']}" for x in context]
    )
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
- Finaliza con sección 'Referencias usadas'.

Contexto:
{context_text}
""".strip()
```

> **Explicación:** Cada tipo de tarea tiene su etiqueta. El prompt se construye dinámicamente con: objetivo (de `TASKS`), tema libre del usuario, tono, audiencia, longitud, y los chunks de contexto recuperados por RAG. Las reglas obligan al modelo a usar solo datos del corpus y citar referencias, igual que en `answer_trained()` pero adaptado a generación de contenido.

### 2.9 Agente IA multicanal — Canales y Políticas

- `AGENT_CHANNELS`: webchat, whatsapp, email, crm
- `AGENT_POLICIES`: informativo, comercial, soporte — cada una con directrices propias
- `build_agent_prompt()` combina canal + política + contexto RAG en un prompt estructurado

```python
AGENT_CHANNELS = {"webchat", "whatsapp", "email", "crm"}
AGENT_POLICIES = {
    "informativo": "El agente informa con precisión y no cierra operaciones automáticamente.",
    "comercial": "El agente puede proponer siguientes pasos comerciales sin inventar datos.",
    "soporte": "El agente prioriza resolución práctica, escalando cuando falte información.",
}


def build_agent_prompt(channel, objective, policy_mode, customer_message, context):
    context_text = "\n\n".join(
        [f"[{x['filename']}|score={x['score']:.3f}]\n{x['content']}" for x in context]
    )
    policy_text = AGENT_POLICIES.get(policy_mode, AGENT_POLICIES["informativo"])
    channel_label = {
        "webchat": "Web Chat", "whatsapp": "WhatsApp", "email": "Email", "crm": "CRM",
    }.get(channel, channel)

    return f"""
Canal del agente: {channel_label}
Objetivo de negocio: {objective}
Política de actuación: {policy_mode} -> {policy_text}
Mensaje del cliente: {customer_message}

Reglas obligatorias:
- Responde solo con información explícita del contexto de Ensenyem.
- No inventes precios, fechas, promociones o condiciones no visibles en contexto.
- Si falta un dato clave, indica claramente que no está disponible en el corpus.
- Mantén tono profesional, claro y accionable.

Formato de salida:
1) Respuesta al cliente
2) Siguiente acción recomendada para el equipo de Ensenyem
3) Evidencias usadas (bullet points)

Contexto:
{context_text}
""".strip()
```

> **Explicación:** El prompt del agente es el más complejo: especifica canal (para adaptar tono: WhatsApp = cercano, email = formal), política (informativo = neutro, comercial = orientado a venta, soporte = resolución técnica), objetivo de negocio, y el mensaje real del cliente. Además, obliga un formato de salida triple: respuesta, siguiente acción y evidencias. Esto permite que el equipo de Ensenyem audite y actúe sobre cada interacción.

### 2.10 Ejecución del agente con trazabilidad

- `run_agent()` valida inputs → verifica entrenamiento → recupera contexto → genera → persiste
- Cada ejecución se guarda en `agent_runs` y `interaction_logs` paralelamente
- Incluye validaciones exhaustivas antes de ejecutar

```python
def run_agent(channel, objective, policy_mode, customer_message):
    if channel not in AGENT_CHANNELS:
        raise RuntimeError(f"Canal inválido. Usa: {', '.join(sorted(AGENT_CHANNELS))}")
    if policy_mode not in AGENT_POLICIES:
        raise RuntimeError(f"Política inválida. Usa: {', '.join(AGENT_POLICIES.keys())}")
    if not objective.strip():
        raise RuntimeError("El objetivo del agente es obligatorio")
    if len(customer_message.strip()) < 6:
        raise RuntimeError("El mensaje del cliente es demasiado corto")

    status = get_training_status()
    if status.get("pendingRetrain"):
        raise RuntimeError("Hay fuentes nuevas sin entrenar. Entrena primero.")
    if int(status.get("chunkCount", 0)) == 0:
        raise RuntimeError("No hay índice semántico disponible.")

    retrieval_query = f"{objective}\n{customer_message}".strip()
    context = retrieve_context(retrieval_query, top_k=5)

    prompt = build_agent_prompt(channel, objective, policy_mode, customer_message, context)
    response = ollama_chat(
        [{"role": "system", "content": SYSTEM_ENSENYEM}, {"role": "user", "content": prompt}],
        temperature=0.1, num_predict=420,
    )

    with get_db() as conn:
        run_id = conn.execute(
            "INSERT INTO agent_runs (channel, objective, policy_mode, customer_message, "
            "agent_response, context_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (channel, objective, policy_mode, customer_message, response,
             json.dumps(context, ensure_ascii=False), now_iso()),
        ).lastrowid
        conn.execute(
            "INSERT INTO interaction_logs (mode, question, answer, context_chunks, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("agent", customer_message, response, len(context), now_iso()),
        )

    return {"runId": run_id, "response": response, "context": context}
```

> **Explicación:** El flujo del agente tiene 5 fases: 1) validación estricta (canal, política, longitud), 2) verificación de estado de entrenamiento (no permite ejecutar sin índice), 3) recuperación de contexto combinando objetivo + mensaje del cliente, 4) generación con temperatura baja (0.1) para respuestas fiables, 5) persistencia dual en `agent_runs` (con contexto JSON completo) y `interaction_logs` (para estadísticas globales).

### 2.11 Ingesta de fuentes — Documento, PDF, YouTube, Web

- 4 tipos de ingesta con validación y normalización
- YouTube: soporta vídeo individual y perfil/canal completo
- PDF: extracción con `pypdf`

```python
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join([p for p in pages if p])
    if not text:
        raise RuntimeError("No se pudo extraer texto del PDF")
    return text


def fetch_youtube_transcript(url: str) -> str:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise RuntimeError("URL de YouTube no válida")
    transcript_items = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
    extracted_parts = []
    for item in transcript_items:
        text_piece = (item.get("text") or "").strip() if isinstance(item, dict) \
            else (getattr(item, "text", "") or "").strip()
        if text_piece:
            extracted_parts.append(text_piece)
    text = " ".join(extracted_parts).strip()
    if not text:
        raise RuntimeError("No se pudo extraer transcripción del vídeo")
    return text


def add_source(source_type, title, origin_ref, content):
    clean_title = title.strip()[:180]
    clean_content = content.strip()
    if not clean_title:
        raise RuntimeError("El título es obligatorio")
    if len(clean_content) < 40:
        raise RuntimeError("El contenido es demasiado corto (mínimo 40 caracteres)")
    with get_db() as conn:
        source_id = conn.execute(
            "INSERT INTO knowledge_sources (source_type, title, origin_ref, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_type, clean_title, origin_ref.strip(), clean_content, now_iso()),
        ).lastrowid
    return source_id
```

> **Explicación:** Cada tipo de fuente tiene su método de extracción. **PDF:** `pypdf.PdfReader` extrae texto página a página. **YouTube:** `youtube_transcript_api` descarga la transcripción en español o inglés. **Web:** el crawler BFS del punto 2.7. **Documento manual:** texto plano. Todas pasan por `add_source()` que valida título (≤180 chars) y contenido (≥40 chars) antes de persistir en `knowledge_sources`.

### 2.12 Tokenización y Stopwords — Filtrado léxico

- `tokenize()` extrae tokens alfabéticos ≥3 caracteres (con acentos y ñ)
- `STOPWORDS` filtra palabras funcionales del español que no aportan semántica

```python
STOPWORDS = {
    "para", "como", "donde", "cuando", "desde", "hasta", "sobre", "entre",
    "este", "esta", "estos", "estas", "ser", "estar", "tener", "puede",
    "pueden", "que", "con", "sin", "por", "una", "uno", "unos", "unas",
    "del", "las", "los", "el", "la", "de", "en", "y", "o", "a", "se",
    "su", "sus", "es", "son", "al",
}


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúñ0-9]{3,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS}
```

> **Explicación:** La regex `[a-záéíóúñ0-9]{3,}` captura palabras de al menos 3 caracteres incluyendo acentos y ñ del español. Se devuelve como `set` para operaciones de intersección eficientes. Las stopwords eliminan artículos, preposiciones y verbos auxiliares que aparecerían en casi cualquier texto y no aportan discriminación semántica.

---

## 3. Presentación del proyecto

- **Flujo demo:** Importar fuentes (web/pdf/youtube) → Entrenar índice semántico → Preguntar con RAG → Generar contenido (4 tipos) → Ejecutar agente (canal + política)
- **Puntos fuertes:**
  - Pipeline completo de IA empresarial en un solo backend (1361 líneas)
  - 100 % local con Ollama — sin APIs externas de pago
  - 4 fuentes de ingesta (documento, PDF, YouTube, web)
  - Recuperación híbrida (coseno 80 % + léxica 20 %)
  - 4 tipos de generación + agente con 4 canales × 3 políticas = 12 combinaciones
  - Trazabilidad total: 7 tablas SQLite, todo queda registrado
- **Demo paso a paso:**
  1. Abrir `http://localhost:5102` → pestaña Fuentes → importar URL web de Ensenyem
  2. Pestaña Entrenamiento → "Entrenar" → verificar chunks creados
  3. Pestaña Preguntas → hacer pregunta → ver respuesta con evidencias
  4. Pestaña Generación → seleccionar "Respuesta WhatsApp" → generar → ver resultado
  5. Pestaña Agente → canal "WhatsApp" + política "Comercial" → ejecutar → ver respuesta con acciones recomendadas
- **Arranque:**
  ```bash
  ollama serve                                           # Iniciar Ollama
  cd "002-IA Generativa/ensenyem_generative_studio"
  pip install -r requirements.txt
  python app.py                                          # → http://localhost:5102
  ```

---

## 4. Conclusión

- **Competencias:** RAG pipeline, embeddings vectoriales, chunking, prompt engineering, web crawling, agentes IA, API REST, SQLite
- **Concepto clave:** RAG = Retrieval-Augmented Generation → el modelo no inventa, genera a partir de evidencia recuperada del corpus
- **Hybrid Retrieval:** `combined = coseno * 0.8 + léxico * 0.2` captura semántica profunda + coincidencia exacta de términos
- **Tres actividades como fases de un pipeline:**
  - 001 → Ingesta + entrenamiento + benchmark (base de conocimiento)
  - 002 → Generación de contenido condicionado (producción)
  - 003 → Agente autónomo multicanal (distribución)
- **Valoración:** Sistema completo de IA empresarial que demuestra el ciclo entero: desde la ingesta de documentos hasta la respuesta automatizada por canales, con trazabilidad y grounding integrados
