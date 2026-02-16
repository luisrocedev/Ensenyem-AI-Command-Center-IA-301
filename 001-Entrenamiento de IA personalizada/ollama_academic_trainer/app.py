import json
import math
import os
import re
import sqlite3
import threading
import uuid
from collections import deque
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.parse import urldefrag, urljoin

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "trainer.sqlite3"
CORPUS_DIR = BASE_DIR / "corpus"
BENCHMARK_FILE = BASE_DIR / "benchmark.json"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
BASE_MODEL = os.environ.get("OLLAMA_BASE_MODEL", "qwen2.5-coder:7b")
TRAINED_MODEL = os.environ.get("OLLAMA_TRAINED_MODEL", BASE_MODEL)
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

SYSTEM_TUNED = (
    "Eres el bot oficial de Ensenyem para WhatsApp. "
    "Responde en español profesional, tono cercano, mensajes breves y útiles. "
    "Prioriza información del corpus de Ensenyem por encima de conocimiento general. "
    "Si no está en el corpus, indícalo explícitamente."
)

app = Flask(__name__)

AUTO_CRAWL_MAX_PAGES = 80
AUTO_CRAWL_MAX_DEPTH = 3
WEBSITE_JOBS: dict[str, dict] = {}
WEBSITE_JOBS_LOCK = threading.Lock()

GROUNDING_MIN_BEST_SCORE = 0.33
GROUNDING_MIN_OVERLAP = 0.12

SPANISH_STOPWORDS = {
    "para", "como", "donde", "cuando", "desde", "hasta", "sobre", "entre", "este", "esta", "estos",
    "estas", "ser", "estar", "tener", "puede", "pueden", "tambien", "solo", "cual", "cuales", "porque",
    "que", "con", "sin", "por", "una", "uno", "unos", "unas", "del", "las", "los", "el", "la", "de",
    "en", "y", "o", "a", "se", "su", "sus", "es", "son", "al", "lo", "le", "les",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with get_db() as connection:
        connection.executescript(
            """
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

            CREATE TABLE IF NOT EXISTS knowledge_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                origin_ref TEXT DEFAULT '',
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
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

            CREATE TABLE IF NOT EXISTS benchmark_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                expected_keywords_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                answer TEXT NOT NULL,
                score REAL NOT NULL,
                run_label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(question_id) REFERENCES benchmark_questions(id)
            );

            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                context_chunks INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """
        )


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return parsed.path.strip("/")
    if "youtube.com" in host:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            return (query.get("v") or [""])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[-1].split("/")[0]
    return ""


def fetch_youtube_transcript(url: str) -> str:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise RuntimeError("URL de YouTube no válida.")
    transcript_items = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
    text = " ".join([item.get("text", "").strip() for item in transcript_items if item.get("text")]).strip()
    if not text:
        raise RuntimeError("No se pudo extraer transcripción del vídeo.")
    return text


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    text = "\n\n".join([p for p in pages if p])
    if not text:
        raise RuntimeError("No se pudo extraer texto del PDF.")
    return text


def list_sources() -> list[dict]:
    with get_db() as connection:
        rows = connection.execute(
            """
            SELECT id, source_type, title, origin_ref, LENGTH(content) AS chars, created_at
            FROM knowledge_sources
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def add_source(source_type: str, title: str, origin_ref: str, content: str) -> int:
    clean_title = title.strip()[:180]
    clean_content = content.strip()
    if not clean_title:
        raise RuntimeError("El título es obligatorio.")
    if len(clean_content) < 40:
        raise RuntimeError("El contenido es demasiado corto para entrenamiento (mínimo 40 caracteres).")
    with get_db() as connection:
        source_id = connection.execute(
            """
            INSERT INTO knowledge_sources (source_type, title, origin_ref, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_type, clean_title, origin_ref.strip(), clean_content, now_iso()),
        ).lastrowid
    return source_id


def crawl_website(
    base_url: str,
    max_pages: int = 12,
    max_depth: int = 2,
    progress_callback=None,
) -> tuple[str, int]:
    candidate_url = base_url.strip()
    if candidate_url and not candidate_url.startswith(("http://", "https://")):
        candidate_url = f"https://{candidate_url}"

    parsed_base = urlparse(candidate_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise RuntimeError("URL web no válida. Debe incluir http(s)://")

    base_domain = parsed_base.netloc.lower()
    headers = {
        "User-Agent": "EnsenyemBotTrainer/1.0 (+local-learning)",
    }

    queue = deque([(candidate_url, 0)])
    visited: set[str] = set()
    page_blocks: list[str] = []

    while queue and len(visited) < max_pages:
        current_url, depth = queue.popleft()
        normalized = urldefrag(current_url).url
        if normalized in visited:
            continue
        visited.add(normalized)

        try:
            response = requests.get(normalized, timeout=12, headers=headers)
            response.raise_for_status()
        except Exception:
            if progress_callback:
                progress_callback(normalized, len(visited), len(page_blocks), "error")
            continue

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title else ""
        text = "\n".join([line.strip() for line in soup.get_text("\n").splitlines() if line.strip()])
        if text:
            safe_title = title or normalized
            page_blocks.append(f"[URL] {normalized}\n[TITLE] {safe_title}\n\n{text}")

        if progress_callback:
            progress_callback(normalized, len(visited), len(page_blocks), "ok")

        if depth >= max_depth:
            continue

        for anchor in soup.find_all("a", href=True):
            href = (anchor.get("href") or "").strip()
            if not href or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            next_url = urldefrag(urljoin(normalized, href)).url
            parsed_next = urlparse(next_url)
            if parsed_next.scheme not in {"http", "https"}:
                continue
            if parsed_next.netloc.lower() != base_domain:
                continue
            if next_url not in visited:
                queue.append((next_url, depth + 1))

    if not page_blocks:
        raise RuntimeError("No se pudo extraer contenido web del dominio indicado.")

    return "\n\n\n".join(page_blocks), len(page_blocks)


def update_website_job(job_id: str, **fields) -> None:
    with WEBSITE_JOBS_LOCK:
        current = WEBSITE_JOBS.get(job_id)
        if not current:
            return
        current.update(fields)
        current["updatedAt"] = now_iso()


def run_website_job(job_id: str, title: str, url: str) -> None:
    def on_progress(current_url: str, visited_count: int, indexed_count: int, state: str) -> None:
        update_website_job(
            job_id,
            status="running",
            message=f"Rastreando {current_url}",
            visitedCount=visited_count,
            indexedPages=indexed_count,
            lastUrl=current_url,
            lastState=state,
        )

    try:
        update_website_job(job_id, status="running", message="Iniciando rastreo de web...")
        website_text, pages_indexed = crawl_website(
            url,
            max_pages=AUTO_CRAWL_MAX_PAGES,
            max_depth=AUTO_CRAWL_MAX_DEPTH,
            progress_callback=on_progress,
        )
        source_id = add_source("website", title, url, website_text)
        update_website_job(
            job_id,
            status="done",
            message="Rastreo finalizado",
            sourceId=source_id,
            chars=len(website_text),
            indexedPages=pages_indexed,
            mode="auto-enterprise-crawl",
        )
    except Exception as exc:
        update_website_job(job_id, status="error", message=str(exc))


def seed_benchmark() -> None:
    if not BENCHMARK_FILE.exists():
        return
    with get_db() as connection:
        current = connection.execute("SELECT COUNT(*) AS n FROM benchmark_questions").fetchone()["n"]
        if current:
            return
        data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        connection.executemany(
            "INSERT INTO benchmark_questions (question, expected_keywords_json) VALUES (?, ?)",
            [(item["question"], json.dumps(item["expected_keywords"], ensure_ascii=False)) for item in data],
        )


def split_chunks(text: str, max_chars: int = 700) -> list[str]:
    paragraphs = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
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


def ollama_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError(f"Embedding vacío: {data}")
    return embedding


def ollama_chat(model: str, messages: list[dict], temperature: float = 0.2, num_predict: int = 300) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }

    response = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    if response.status_code == 404:
        conversation = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
        gen_payload = {
            "model": model,
            "prompt": conversation,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        gen_response = requests.post(f"{OLLAMA_BASE}/api/generate", json=gen_payload, timeout=120)
        if gen_response.status_code == 404:
            v1_payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": num_predict,
            }
            v1_response = requests.post(f"{OLLAMA_BASE}/v1/chat/completions", json=v1_payload, timeout=120)
            if v1_response.status_code == 404:
                raise RuntimeError(
                    "No se encontró una API compatible en OLLAMA_BASE. "
                    "Verifica que Ollama esté activo y accesible en http://localhost:11434."
                )
            v1_response.raise_for_status()
            v1_data = v1_response.json()
            choices = v1_data.get("choices") or []
            message = (choices[0].get("message") if choices else {}) or {}
            content = (message.get("content") or "").strip()
            if not content:
                raise RuntimeError(f"Respuesta vacía desde endpoint OpenAI-compatible: {v1_data}")
            return content

        gen_response.raise_for_status()
        gen_data = gen_response.json()
        content = (gen_data.get("response") or "").strip()
        if not content:
            raise RuntimeError(f"Respuesta vacía desde Ollama (/api/generate): {gen_data}")
        return content

    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError(f"Respuesta vacía desde Ollama: {data}")
    return content


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 4)


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúñ0-9]{3,}", text.lower())
    return {token for token in tokens if token not in SPANISH_STOPWORDS}


def overlap_ratio(reference_text: str, candidate_text: str) -> float:
    reference = tokenize(reference_text)
    candidate = tokenize(candidate_text)
    if not reference or not candidate:
        return 0.0
    return len(reference.intersection(candidate)) / max(1, len(reference))


def build_evidence_snippets(context: list[dict], max_items: int = 3, max_chars: int = 180) -> str:
    lines: list[str] = []
    for item in context[:max_items]:
        cleaned = " ".join(item["content"].split())[:max_chars]
        lines.append(f"- [{item['filename']}] {cleaned}...")
    return "\n".join(lines)


def train_semantic_index() -> dict:
    started = datetime.now(timezone.utc)
    with get_db() as connection:
        sources = connection.execute(
            "SELECT id, title, content FROM knowledge_sources ORDER BY id"
        ).fetchall()

    if not sources:
        files = sorted([p for p in CORPUS_DIR.glob("*.*") if p.suffix.lower() in {".md", ".txt"}])
        if not files:
            raise RuntimeError("No hay fuentes cargadas. Añade PDF/documento/YouTube o usa corpus/*.md")
        sources = [{"id": None, "title": file_path.name, "content": file_path.read_text(encoding="utf-8")} for file_path in files]

    with get_db() as connection:
        connection.execute("DELETE FROM corpus_chunks")
        connection.execute("DELETE FROM corpus_documents")

        total_chunks = 0
        for source in sources:
            content = source["content"]
            doc_id = connection.execute(
                "INSERT INTO corpus_documents (filename, content, created_at) VALUES (?, ?, ?)",
                (source["title"], content, now_iso()),
            ).lastrowid

            chunks = split_chunks(content)
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

        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        connection.execute(
            """
            INSERT INTO training_runs (run_type, model_name, chunks_indexed, duration_ms, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("semantic_index", EMBED_MODEL, total_chunks, duration_ms, "ok", f"docs={len(sources)}", now_iso()),
        )

    return {"documents": len(sources), "chunks": total_chunks, "durationMs": duration_ms}


def retrieve_context(question: str, top_k: int = 6) -> list[dict]:
    query_embedding = ollama_embedding(question)
    with get_db() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.content, d.filename, c.embedding_json
            FROM corpus_chunks c
            JOIN corpus_documents d ON d.id = c.document_id
            """
        ).fetchall()

    scored = []
    for row in rows:
        emb = json.loads(row["embedding_json"])
        score = cosine_similarity(query_embedding, emb)
        scored.append({
            "chunkId": row["id"],
            "filename": row["filename"],
            "content": row["content"],
            "score": score,
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def answer_question(question: str, mode: str) -> dict:
    if mode == "baseline":
        answer = ollama_chat(
            BASE_MODEL,
            [{"role": "user", "content": question}],
            temperature=0.35,
            num_predict=260,
        )
        context = []
    else:
        context = retrieve_context(question)
        if not context:
            answer = "No está en mi corpus de entrenamiento."
            with get_db() as connection:
                connection.execute(
                    "INSERT INTO interaction_logs (mode, question, answer, context_chunks, created_at) VALUES (?, ?, ?, ?, ?)",
                    (mode, question, answer, 0, now_iso()),
                )
            return {
                "answer": answer,
                "context": [],
                "grounding": {"bestScore": 0.0, "overlap": 0.0, "policy": "empty-context"},
            }

        best_score = float(context[0]["score"])
        context_text = "\n\n".join(
            [f"[{item['filename']}|score={item['score']:.3f}]\n{item['content']}" for item in context]
        )
        context_overlap = overlap_ratio(question, context_text)

        if best_score < GROUNDING_MIN_BEST_SCORE and context_overlap < GROUNDING_MIN_OVERLAP:
            answer = (
                "No está en mi corpus de entrenamiento.\n\n"
                "Evidencias más cercanas encontradas:\n"
                f"{build_evidence_snippets(context)}"
            )
            with get_db() as connection:
                connection.execute(
                    "INSERT INTO interaction_logs (mode, question, answer, context_chunks, created_at) VALUES (?, ?, ?, ?, ?)",
                    (mode, question, answer, len(context), now_iso()),
                )
            return {
                "answer": answer,
                "context": context,
                "grounding": {"bestScore": round(best_score, 4), "overlap": round(context_overlap, 4), "policy": "reject-low-support"},
            }

        prompt = (
            "Eres el bot de Ensenyem para WhatsApp.\n"
            "REGLAS OBLIGATORIAS:\n"
            "1) Responde EXCLUSIVAMENTE con hechos explícitos del contexto.\n"
            "2) No inventes pasos, formularios, teléfonos ni procesos.\n"
            "3) Si falta el dato exacto, responde: 'No está en mi corpus de entrenamiento'.\n"
            "4) Añade siempre una sección 'Evidencias' citando de qué documento sale la respuesta.\n\n"
            f"Contexto:\n{context_text}\n\nPregunta:\n{question}"
        )
        answer = ollama_chat(
            TRAINED_MODEL,
            [
                {"role": "system", "content": SYSTEM_TUNED},
                {"role": "user", "content": prompt},
            ],
            temperature=0.05,
            num_predict=300,
        )

        answer_overlap = overlap_ratio(answer, context_text)
        if "no está en mi corpus de entrenamiento" not in answer.lower() and answer_overlap < 0.10:
            answer = (
                "No está en mi corpus de entrenamiento.\n\n"
                "Evidencias más cercanas encontradas:\n"
                f"{build_evidence_snippets(context)}"
            )

        grounding = {
            "bestScore": round(best_score, 4),
            "overlap": round(context_overlap, 4),
            "answerOverlap": round(answer_overlap, 4),
            "policy": "strict-evidence",
        }

    with get_db() as connection:
        connection.execute(
            "INSERT INTO interaction_logs (mode, question, answer, context_chunks, created_at) VALUES (?, ?, ?, ?, ?)",
            (mode, question, answer, len(context), now_iso()),
        )

    return {"answer": answer, "context": context, "grounding": grounding if mode == "trained" else None}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/health")
def health():
    info = {"ok": True, "ollamaBase": OLLAMA_BASE, "baseModel": BASE_MODEL, "trainedModel": TRAINED_MODEL}
    try:
        version = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5).json()
        tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=8).json()
        info["ollamaVersion"] = version.get("version")
        info["modelsInstalled"] = [m.get("name") for m in tags.get("models", [])]
    except Exception as exc:
        info["ok"] = False
        info["error"] = str(exc)
    return jsonify(info)


@app.post("/api/train/run")
def api_train_run():
    try:
        result = train_semantic_index()
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/sources")
def api_sources_list():
    return jsonify({"ok": True, "sources": list_sources()})


@app.post("/api/sources/document")
def api_sources_document():
    body = request.get_json(silent=True) or {}
    title = str(body.get("title", "")).strip()
    content = str(body.get("content", "")).strip()
    try:
        source_id = add_source("document", title, "manual", content)
        return jsonify({"ok": True, "sourceId": source_id})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sources/youtube")
def api_sources_youtube():
    body = request.get_json(silent=True) or {}
    url = str(body.get("url", "")).strip()
    title = str(body.get("title", "")).strip() or "Video YouTube"
    if not url:
        return jsonify({"ok": False, "error": "url es obligatorio"}), 400
    try:
        transcript = fetch_youtube_transcript(url)
        source_id = add_source("youtube", title, url, transcript)
        return jsonify({"ok": True, "sourceId": source_id, "chars": len(transcript)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sources/website")
def api_sources_website():
    body = request.get_json(silent=True) or {}
    url = str(body.get("url", "")).strip()
    title = str(body.get("title", "")).strip() or "Web Ensenyem"

    if not url:
        return jsonify({"ok": False, "error": "url es obligatorio"}), 400

    try:
        website_text, pages_indexed = crawl_website(
            url,
            max_pages=AUTO_CRAWL_MAX_PAGES,
            max_depth=AUTO_CRAWL_MAX_DEPTH,
        )
        source_id = add_source("website", title, url, website_text)
        return jsonify(
            {
                "ok": True,
                "sourceId": source_id,
                "chars": len(website_text),
                "pagesIndexed": pages_indexed,
                "mode": "auto-enterprise-crawl",
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sources/website/start")
def api_sources_website_start():
    body = request.get_json(silent=True) or {}
    url = str(body.get("url", "")).strip()
    title = str(body.get("title", "")).strip() or "Web Ensenyem"
    if not url:
        return jsonify({"ok": False, "error": "url es obligatorio"}), 400

    job_id = uuid.uuid4().hex
    job_data = {
        "id": job_id,
        "status": "queued",
        "message": "Job en cola",
        "url": url,
        "title": title,
        "maxPages": AUTO_CRAWL_MAX_PAGES,
        "maxDepth": AUTO_CRAWL_MAX_DEPTH,
        "visitedCount": 0,
        "indexedPages": 0,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
    }
    with WEBSITE_JOBS_LOCK:
        WEBSITE_JOBS[job_id] = job_data

    thread = threading.Thread(target=run_website_job, args=(job_id, title, url), daemon=True)
    thread.start()

    return jsonify({"ok": True, "jobId": job_id, "status": "queued"})


@app.get("/api/sources/website/jobs/<job_id>")
def api_sources_website_job(job_id: str):
    with WEBSITE_JOBS_LOCK:
        job = WEBSITE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job no encontrado"}), 404
    return jsonify({"ok": True, "job": job})


@app.post("/api/sources/pdf")
def api_sources_pdf():
    file = request.files.get("file")
    title = (request.form.get("title") or "").strip()
    if not file:
        return jsonify({"ok": False, "error": "Debes subir un archivo PDF"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "El archivo debe ser .pdf"}), 400
    file_bytes = file.read()
    try:
        text = extract_pdf_text(file_bytes)
        source_title = title or file.filename
        source_id = add_source("pdf", source_title, file.filename, text)
        return jsonify({"ok": True, "sourceId": source_id, "chars": len(text)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.delete("/api/sources/<int:source_id>")
def api_sources_delete(source_id: int):
    with get_db() as connection:
        row = connection.execute("SELECT id FROM knowledge_sources WHERE id = ?", (source_id,)).fetchone()
        if not row:
            return jsonify({"ok": False, "error": "Fuente no encontrada"}), 404
        connection.execute("DELETE FROM knowledge_sources WHERE id = ?", (source_id,))
    return jsonify({"ok": True})


@app.post("/api/ask")
def api_ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    mode = str(body.get("mode", "baseline")).strip()
    if not question:
        return jsonify({"ok": False, "error": "question es obligatorio"}), 400
    if mode not in {"baseline", "trained"}:
        return jsonify({"ok": False, "error": "mode debe ser baseline o trained"}), 400
    try:
        result = answer_question(question, mode)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/evaluate")
def api_evaluate():
    run_label = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with get_db() as connection:
        questions = connection.execute("SELECT * FROM benchmark_questions ORDER BY id").fetchall()

    if not questions:
        return jsonify({"ok": False, "error": "No hay benchmark_questions cargadas."}), 400

    details = []
    summary = {"baseline": [], "trained": []}

    for row in questions:
        expected = json.loads(row["expected_keywords_json"])
        q = row["question"]

        base = answer_question(q, "baseline")
        trained = answer_question(q, "trained")

        score_base = keyword_score(base["answer"], expected)
        score_trained = keyword_score(trained["answer"], expected)

        with get_db() as connection:
            connection.execute(
                "INSERT INTO benchmark_results (question_id, mode, answer, score, run_label, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (row["id"], "baseline", base["answer"], score_base, run_label, now_iso()),
            )
            connection.execute(
                "INSERT INTO benchmark_results (question_id, mode, answer, score, run_label, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (row["id"], "trained", trained["answer"], score_trained, run_label, now_iso()),
            )

        summary["baseline"].append(score_base)
        summary["trained"].append(score_trained)
        details.append(
            {
                "question": q,
                "scoreBaseline": score_base,
                "scoreTrained": score_trained,
                "improvement": round(score_trained - score_base, 4),
            }
        )

    avg_baseline = round(sum(summary["baseline"]) / len(summary["baseline"]), 4)
    avg_trained = round(sum(summary["trained"]) / len(summary["trained"]), 4)

    with get_db() as connection:
        connection.execute(
            """
            INSERT INTO training_runs (run_type, model_name, chunks_indexed, duration_ms, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "benchmark",
                f"{BASE_MODEL} -> {TRAINED_MODEL}",
                len(details),
                0,
                "ok",
                f"baseline={avg_baseline};trained={avg_trained}",
                now_iso(),
            ),
        )

    return jsonify(
        {
            "ok": True,
            "runLabel": run_label,
            "averageBaseline": avg_baseline,
            "averageTrained": avg_trained,
            "delta": round(avg_trained - avg_baseline, 4),
            "details": details,
        }
    )


@app.get("/api/stats")
def api_stats():
    with get_db() as connection:
        totals = connection.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM corpus_documents) AS documents,
              (SELECT COUNT(*) FROM corpus_chunks) AS chunks,
                            (SELECT COUNT(*) FROM knowledge_sources) AS sources,
              (SELECT COUNT(*) FROM training_runs) AS training_runs,
              (SELECT COUNT(*) FROM benchmark_results) AS benchmark_results,
              (SELECT COUNT(*) FROM interaction_logs) AS interactions
            """
        ).fetchone()

        latest_eval = connection.execute(
            """
            SELECT run_label,
                   AVG(CASE WHEN mode='baseline' THEN score END) AS baseline,
                   AVG(CASE WHEN mode='trained' THEN score END) AS trained
            FROM benchmark_results
            GROUP BY run_label
            ORDER BY run_label DESC
            LIMIT 1
            """
        ).fetchone()

    payload = {"ok": True, "stats": dict(totals)}
    if latest_eval:
        payload["latestEvaluation"] = {
            "runLabel": latest_eval["run_label"],
            "baseline": round(float(latest_eval["baseline"] or 0), 4),
            "trained": round(float(latest_eval["trained"] or 0), 4),
            "delta": round(float((latest_eval["trained"] or 0) - (latest_eval["baseline"] or 0)), 4),
        }
    return jsonify(payload)


if __name__ == "__main__":
    init_db()
    seed_benchmark()
    app.run(debug=True, port=5101)
