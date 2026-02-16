import json
import math
import os
import re
import sqlite3
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "command_center.sqlite3"

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
GEN_MODEL = os.environ.get("OLLAMA_GEN_MODEL", "qwen2.5-coder:7b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

SYSTEM_ENSENYEM = (
    "Eres el asistente oficial de Ensenyem. Responde en español con precisión, "
    "sin inventar datos y apoyándote en el contexto entrenado."
)

TASKS = {
    "course_summary": "Ficha comercial de curso",
    "whatsapp_reply": "Respuesta corta para WhatsApp",
    "social_post": "Post para redes sociales",
    "email_campaign": "Email de campaña",
}

AGENT_CHANNELS = {"webchat", "whatsapp", "email", "crm"}
AGENT_POLICIES = {
    "informativo": "El agente informa con precisión y no cierra operaciones automáticamente.",
    "comercial": "El agente puede proponer siguientes pasos comerciales sin inventar datos.",
    "soporte": "El agente prioriza resolución práctica, escalando cuando falte información.",
}

STOPWORDS = {
    "para", "como", "donde", "cuando", "desde", "hasta", "sobre", "entre", "este", "esta", "estos", "estas",
    "ser", "estar", "tener", "puede", "pueden", "que", "con", "sin", "por", "una", "uno", "unos", "unas",
    "del", "las", "los", "el", "la", "de", "en", "y", "o", "a", "se", "su", "sus", "es", "son", "al",
}

AUTO_CRAWL_MAX_PAGES = 80
AUTO_CRAWL_MAX_DEPTH = 3
AUTO_CRAWL_MAX_SECONDS = 180
WEBSITE_JOBS: dict[str, dict] = {}
WEBSITE_JOBS_LOCK = threading.Lock()

app = Flask(__name__)


class CrawlCancelled(Exception):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(
            """
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
            """
        )


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-záéíóúñ0-9]{3,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


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


def is_embed_model_available(model_name: str, available_names: list[str]) -> bool:
    normalized = model_name.strip().lower()
    for candidate in available_names:
        cand = (candidate or "").strip().lower()
        if not cand:
            continue
        if cand == normalized:
            return True
        if cand.startswith(f"{normalized}:"):
            return True
        if normalized.startswith(f"{cand}:"):
            return True
    return False


def ensure_ollama_embed_model_ready() -> None:
    try:
        requests.get(f"{OLLAMA_BASE}/api/version", timeout=5).raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo conectar con Ollama. Verifica que esté activo y accesible en "
            f"{OLLAMA_BASE}. Detalle: {exc}"
        ) from exc

    try:
        tags_response = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=8)
        tags_response.raise_for_status()
        data = tags_response.json()
        models = data.get("models") or []
        available = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
        if not is_embed_model_available(EMBED_MODEL, available):
            raise RuntimeError(
                f"El modelo de embeddings '{EMBED_MODEL}' no está disponible en Ollama. "
                f"Ejecuta: ollama pull {EMBED_MODEL}"
            )
    except RuntimeError:
        raise
    except Exception:
        return


def ollama_embedding(text: str) -> list[float]:
    last_error = None

    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if embedding:
            return embedding
        last_error = f"Embedding vacío en /api/embeddings: {data}"
    except Exception as exc:
        last_error = str(exc)

    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": EMBED_MODEL, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if embedding and isinstance(embedding, list):
            return embedding
        last_error = f"Embedding vacío en /api/embeddings (input): {data}"
    except Exception as exc:
        last_error = f"{last_error} | retry /api/embeddings input: {exc}"

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
        last_error = f"Embedding vacío en /api/embed: {data}"
    except Exception as exc:
        last_error = f"{last_error} | fallback /api/embed: {exc}"

    raise RuntimeError(
        f"No se pudo generar embeddings con el modelo '{EMBED_MODEL}'. "
        f"Verifica Ollama y ejecuta: ollama pull {EMBED_MODEL}. Detalle: {last_error}"
    )


def normalize_crawl_url(url: str) -> str:
    parsed = urlparse(urldefrag(url).url)
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def is_blocked_crawl_path(path: str) -> bool:
    lowered = path.lower()
    blocked_fragments = [
        "/pago",
        "/checkout",
        "/cart",
        "/login",
        "/wp-admin",
        "/wp-login",
        "/account",
    ]
    return any(fragment in lowered for fragment in blocked_fragments)


NOISE_LINE_FRAGMENTS = [
    "gestionar consentimiento",
    "aceptar",
    "denegar",
    "guardar preferencias",
    "ver preferencias",
    "leer más sobre estos propósitos",
    "vendor_count",
    "abrir chat",
    "scroll al inicio",
    "carrito",
    "discount",
    "subtotal",
    "total installments",
    "añadir descuento",
    "mailchimp",
    "1&1",
    "1and1",
    "política de privacidad",
]


def clean_web_lines(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()

    for raw in lines:
        line = re.sub(r"\s+", " ", (raw or "").strip())
        if not line:
            continue

        lowered = line.lower()
        if any(fragment in lowered for fragment in NOISE_LINE_FRAGMENTS):
            continue

        if line.startswith("{") and line.endswith("}"):
            continue

        words = line.split()
        if len(words) < 4 and len(line) < 28:
            continue

        signature = lowered
        if signature in seen:
            continue
        seen.add(signature)
        cleaned.append(line)

    return cleaned


def extract_meaningful_text_from_html(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    for tag in soup.find_all(["nav", "footer", "aside", "form", "button"]):
        tag.decompose()

    candidate_selectors = ["main", "article", "section", "[role='main']"]
    candidate_lines: list[str] = []

    for selector in candidate_selectors:
        for node in soup.select(selector):
            candidate_lines.extend(node.get_text("\n").splitlines())

    if not candidate_lines:
        body = soup.body or soup
        candidate_lines = body.get_text("\n").splitlines()

    cleaned = clean_web_lines(candidate_lines)
    return "\n".join(cleaned)


def ollama_chat(messages: list[dict], temperature: float = 0.2, num_predict: int = 360) -> str:
    payload = {
        "model": GEN_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    response = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    if response.status_code == 404:
        conversation = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
        response = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": GEN_MODEL, "prompt": conversation, "stream": False, "options": {"temperature": temperature, "num_predict": num_predict}},
            timeout=120,
        )
    response.raise_for_status()
    data = response.json()
    if "message" in data:
        return (data.get("message", {}).get("content") or "").strip()
    return (data.get("response") or "").strip()


def list_sources() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
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
        raise RuntimeError("El título es obligatorio")
    if len(clean_content) < 40:
        raise RuntimeError("El contenido es demasiado corto (mínimo 40 caracteres)")
    with get_db() as conn:
        source_id = conn.execute(
            """
            INSERT INTO knowledge_sources (source_type, title, origin_ref, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_type, clean_title, origin_ref.strip(), clean_content, now_iso()),
        ).lastrowid
    return source_id


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


def is_youtube_profile_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtube.com" not in host:
        return False
    path = parsed.path.strip("/")
    if not path:
        return False
    return (
        path.startswith("@")
        or path.startswith("channel/")
        or path.startswith("user/")
        or path.startswith("c/")
    )


def extract_video_ids_from_profile(url: str, max_videos: int = 8) -> list[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "youtube.com" not in host:
        raise RuntimeError("La URL no pertenece a YouTube")

    base = url.rstrip("/")
    target_url = base if base.endswith("/videos") else f"{base}/videos"

    response = requests.get(
        target_url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (EnsenyemCommandCenter)"},
    )
    response.raise_for_status()
    html = response.text

    if "Before you continue to YouTube" in html or "consent.youtube.com" in html:
        raise RuntimeError(
            "YouTube está bloqueando la lectura automática del perfil/canal (pantalla de consentimiento). "
            "Usa URL de vídeo individual o importa varios vídeos uno a uno."
        )

    found: list[str] = []
    seen: set[str] = set()

    for match in re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html):
        if match not in seen:
            seen.add(match)
            found.append(match)
        if len(found) >= max_videos:
            break

    if len(found) < max_videos:
        for match in re.findall(r"watch\?v=([A-Za-z0-9_-]{11})", html):
            if match not in seen:
                seen.add(match)
                found.append(match)
            if len(found) >= max_videos:
                break

    return found[:max_videos]


def fetch_youtube_transcript(url: str) -> str:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise RuntimeError("URL de YouTube no válida")

    transcript_items = None
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        transcript_items = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
    else:
        api = YouTubeTranscriptApi()
        transcript_items = api.fetch(video_id, languages=["es", "en"])

    extracted_parts: list[str] = []
    for item in transcript_items:
        if isinstance(item, dict):
            text_piece = (item.get("text") or "").strip()
        else:
            text_piece = (getattr(item, "text", "") or "").strip()
        if text_piece:
            extracted_parts.append(text_piece)

    text = " ".join(extracted_parts).strip()
    if not text:
        raise RuntimeError("No se pudo extraer transcripción del vídeo")
    return text


def fetch_youtube_profile_transcripts(url: str, max_videos: int = 5) -> tuple[str, list[str]]:
    video_ids = extract_video_ids_from_profile(url, max_videos=max(1, min(max_videos, 12)))
    if not video_ids:
        raise RuntimeError("No se han detectado vídeos en el perfil/canal de YouTube")

    api = YouTubeTranscriptApi()
    blocks: list[str] = []
    used_ids: list[str] = []

    for video_id in video_ids:
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                transcript_items = YouTubeTranscriptApi.get_transcript(video_id, languages=["es", "en"])
            else:
                transcript_items = api.fetch(video_id, languages=["es", "en"])

            parts: list[str] = []
            for item in transcript_items:
                if isinstance(item, dict):
                    text_piece = (item.get("text") or "").strip()
                else:
                    text_piece = (getattr(item, "text", "") or "").strip()
                if text_piece:
                    parts.append(text_piece)

            text = " ".join(parts).strip()
            if text:
                used_ids.append(video_id)
                blocks.append(f"[VIDEO] https://www.youtube.com/watch?v={video_id}\n\n{text}")
        except Exception:
            continue

    if not blocks:
        raise RuntimeError("No se pudo extraer transcripción de los vídeos del perfil (pueden estar bloqueadas/desactivadas)")

    return "\n\n\n".join(blocks), used_ids


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join([p for p in pages if p])
    if not text:
        raise RuntimeError("No se pudo extraer texto del PDF")
    return text


def crawl_website(
    base_url: str,
    max_pages: int = 12,
    max_depth: int = 2,
    progress_callback=None,
    should_stop=None,
) -> tuple[str, int]:
    candidate_url = base_url.strip()
    if candidate_url and not candidate_url.startswith(("http://", "https://")):
        candidate_url = f"https://{candidate_url}"

    parsed_base = urlparse(candidate_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise RuntimeError("URL web no válida")

    base_domain = parsed_base.netloc.lower()
    queue = deque([(normalize_crawl_url(candidate_url), 0)])
    visited: set[str] = set()
    page_blocks: list[str] = []
    started_at = datetime.now(timezone.utc)

    while queue and len(visited) < max_pages:
        if (datetime.now(timezone.utc) - started_at).total_seconds() > AUTO_CRAWL_MAX_SECONDS:
            break
        if callable(should_stop) and should_stop():
            raise CrawlCancelled("Rastreo cancelado por el usuario")

        current_url, depth = queue.popleft()
        normalized = normalize_crawl_url(current_url)
        if normalized in visited:
            continue
        if is_blocked_crawl_path(urlparse(normalized).path):
            continue
        visited.add(normalized)

        try:
            response = requests.get(normalized, timeout=12, headers={"User-Agent": "EnsenyemCommandCenter/1.0"})
            response.raise_for_status()
        except Exception:
            if progress_callback:
                progress_callback(normalized, len(visited), len(page_blocks), "error")
            continue

        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else ""
        text = extract_meaningful_text_from_html(soup)
        if text:
            page_blocks.append(f"[URL] {normalized}\n[TITLE] {title or normalized}\n\n{text}")

        if progress_callback:
            progress_callback(normalized, len(visited), len(page_blocks), "ok")

        if depth >= max_depth:
            continue
        for anchor in soup.find_all("a", href=True):
            href = (anchor.get("href") or "").strip()
            if not href or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            next_raw = urldefrag(urljoin(normalized, href)).url
            parsed_raw = urlparse(next_raw)
            if parsed_raw.query:
                continue
            next_url = normalize_crawl_url(next_raw)
            parsed_next = urlparse(next_url)
            if parsed_next.scheme not in {"http", "https"}:
                continue
            if parsed_next.netloc.lower() != base_domain:
                continue
            if is_blocked_crawl_path(parsed_next.path):
                continue
            if next_url not in visited:
                queue.append((next_url, depth + 1))

    if not page_blocks:
        raise RuntimeError("No se pudo extraer contenido web del dominio indicado")
    return "\n\n\n".join(page_blocks), len(page_blocks)


def update_website_job(job_id: str, **fields) -> None:
    with WEBSITE_JOBS_LOCK:
        job = WEBSITE_JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updatedAt"] = now_iso()


def run_website_job(job_id: str, title: str, url: str) -> None:
    def should_stop() -> bool:
        with WEBSITE_JOBS_LOCK:
            job = WEBSITE_JOBS.get(job_id)
            if not job:
                return True
            return job.get("status") in {"cancelling", "cancelled"}

    def progress(url_value: str, visited_count: int, indexed_count: int, state: str) -> None:
        if should_stop():
            return
        update_website_job(
            job_id,
            status="running",
            message=f"Rastreando {url_value}",
            visitedCount=visited_count,
            indexedPages=indexed_count,
            lastUrl=url_value,
            lastState=state,
        )

    try:
        website_text, pages_indexed = crawl_website(
            url,
            max_pages=AUTO_CRAWL_MAX_PAGES,
            max_depth=AUTO_CRAWL_MAX_DEPTH,
            progress_callback=progress,
            should_stop=should_stop,
        )
        if should_stop():
            update_website_job(job_id, status="cancelled", message="Rastreo cancelado", mode="auto-enterprise-crawl")
            return
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
    except CrawlCancelled:
        update_website_job(job_id, status="cancelled", message="Rastreo cancelado", mode="auto-enterprise-crawl")
    except Exception as exc:
        update_website_job(job_id, status="error", message=str(exc))


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
                    """
                    INSERT INTO corpus_chunks (document_id, chunk_index, content, embedding_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doc_id, idx, chunk, json.dumps(embedding), now_iso()),
                )
                chunks_total += 1

        duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        conn.execute(
            """
            INSERT INTO training_runs (run_type, model_name, chunks_indexed, duration_ms, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("semantic_index", EMBED_MODEL, chunks_total, duration_ms, "ok", f"docs={len(sources)}", now_iso()),
        )

    return {"documents": len(sources), "chunks": chunks_total, "durationMs": duration_ms}


def get_training_status() -> dict:
    with get_db() as conn:
        latest_source = conn.execute(
            "SELECT id, title, created_at FROM knowledge_sources ORDER BY id DESC LIMIT 1"
        ).fetchone()
        latest_training = conn.execute(
            "SELECT id, created_at, chunks_indexed FROM training_runs WHERE status = 'ok' ORDER BY id DESC LIMIT 1"
        ).fetchone()

        source_count = conn.execute("SELECT COUNT(*) AS c FROM knowledge_sources").fetchone()["c"]
        training_count = conn.execute("SELECT COUNT(*) AS c FROM training_runs WHERE status = 'ok'").fetchone()["c"]
        chunk_count = conn.execute("SELECT COUNT(*) AS c FROM corpus_chunks").fetchone()["c"]

    latest_source_at = latest_source["created_at"] if latest_source else None
    latest_training_at = latest_training["created_at"] if latest_training else None

    pending_retrain = False
    if source_count == 0:
        pending_retrain = False
    elif not latest_training_at:
        pending_retrain = True
    elif latest_source_at and latest_source_at > latest_training_at:
        pending_retrain = True

    return {
        "ok": True,
        "sourceCount": source_count,
        "trainingCount": training_count,
        "chunkCount": chunk_count,
        "latestSource": dict(latest_source) if latest_source else None,
        "latestTraining": dict(latest_training) if latest_training else None,
        "pendingRetrain": pending_retrain,
    }


def retrieve_context(topic: str, top_k: int = 4) -> list[dict]:
    query_embedding = ollama_embedding(topic)
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT c.content, c.embedding_json, d.filename
            FROM corpus_chunks c
            JOIN corpus_documents d ON d.id = c.document_id
            """
        ).fetchall()

    scored: list[dict] = []
    for row in rows:
        emb = json.loads(row["embedding_json"])
        vector_score = cosine_similarity(query_embedding, emb)
        lexical = lexical_overlap(topic, row["content"])
        combined = vector_score * 0.8 + lexical * 0.2
        scored.append(
            {
                "filename": row["filename"],
                "content": row["content"],
                "score": combined,
                "vectorScore": vector_score,
                "lexicalScore": lexical,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


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
- Finaliza con sección 'Referencias usadas'.

Contexto:
{context_text}
""".strip()


def build_agent_prompt(channel: str, objective: str, policy_mode: str, customer_message: str, context: list[dict]) -> str:
    context_text = "\n\n".join([f"[{x['filename']}|score={x['score']:.3f}]\n{x['content']}" for x in context])
    policy_text = AGENT_POLICIES.get(policy_mode, AGENT_POLICIES["informativo"])
    channel_label = {
        "webchat": "Web Chat",
        "whatsapp": "WhatsApp",
        "email": "Email",
        "crm": "CRM",
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


def run_agent(channel: str, objective: str, policy_mode: str, customer_message: str) -> dict:
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
        raise RuntimeError("Hay fuentes nuevas sin entrenar. Entrena el índice semántico antes de ejecutar el agente.")
    if int(status.get("chunkCount", 0)) == 0:
        raise RuntimeError("No hay índice semántico disponible. Importa fuentes y entrena primero.")

    retrieval_query = f"{objective}\n{customer_message}".strip()
    context = retrieve_context(retrieval_query, top_k=5)
    if not context:
        raise RuntimeError("No se encontró contexto útil para ejecutar el agente")

    prompt = build_agent_prompt(channel, objective, policy_mode, customer_message, context)
    response = ollama_chat(
        [
            {"role": "system", "content": SYSTEM_ENSENYEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        num_predict=420,
    )

    with get_db() as conn:
        run_id = conn.execute(
            """
            INSERT INTO agent_runs (channel, objective, policy_mode, customer_message, agent_response, context_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                channel,
                objective.strip(),
                policy_mode,
                customer_message.strip(),
                response,
                json.dumps(context, ensure_ascii=False),
                now_iso(),
            ),
        ).lastrowid
        conn.execute(
            "INSERT INTO interaction_logs (mode, question, answer, context_chunks, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                "agent",
                customer_message.strip(),
                response,
                len(context),
                now_iso(),
            ),
        )

    return {"runId": run_id, "response": response, "context": context}


def list_agent_runs(limit: int = 20) -> list[dict]:
    safe_limit = max(1, min(limit, 100))
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, channel, objective, policy_mode, customer_message, agent_response, context_json, created_at
            FROM agent_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    items: list[dict] = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "channel": row["channel"],
                "objective": row["objective"],
                "policyMode": row["policy_mode"],
                "customerMessage": row["customer_message"],
                "agentResponse": row["agent_response"],
                "context": json.loads(row["context_json"]),
                "createdAt": row["created_at"],
            }
        )
    return items


def get_agent_overview() -> dict:
    status = get_training_status()
    with get_db() as conn:
        total_runs = conn.execute("SELECT COUNT(*) AS c FROM agent_runs").fetchone()["c"]
        last_run = conn.execute(
            "SELECT id, channel, objective, policy_mode, created_at FROM agent_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "ok": True,
        "channels": sorted(AGENT_CHANNELS),
        "policies": list(AGENT_POLICIES.keys()),
        "training": status,
        "totalRuns": int(total_runs),
        "lastRun": dict(last_run) if last_run else None,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/health")
def health():
    payload = {"ok": True, "model": GEN_MODEL, "embedModel": EMBED_MODEL}
    try:
        payload["ollamaVersion"] = requests.get(f"{OLLAMA_BASE}/api/version", timeout=5).json().get("version")
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = str(exc)

    with get_db() as conn:
        stats = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM knowledge_sources) AS sources,
              (SELECT COUNT(*) FROM corpus_documents) AS documents,
              (SELECT COUNT(*) FROM corpus_chunks) AS chunks,
              (SELECT COUNT(*) FROM training_runs) AS trainings,
              (SELECT COUNT(*) FROM generations) AS generations,
                            (SELECT COUNT(*) FROM interaction_logs) AS interactions,
                            (SELECT COUNT(*) FROM agent_runs) AS agentRuns
            """
        ).fetchone()
    payload["stats"] = dict(stats)
    return jsonify(payload)


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


@app.post("/api/sources/pdf")
def api_sources_pdf():
    file = request.files.get("file")
    title = (request.form.get("title") or "").strip()
    if not file:
        return jsonify({"ok": False, "error": "Debes subir un PDF"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "El archivo debe ser .pdf"}), 400
    try:
        text = extract_pdf_text(file.read())
        source_id = add_source("pdf", title or file.filename, file.filename, text)
        return jsonify({"ok": True, "sourceId": source_id, "chars": len(text)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/sources/youtube")
def api_sources_youtube():
    body = request.get_json(silent=True) or {}
    title = str(body.get("title", "")).strip() or "Video YouTube"
    url = str(body.get("url", "")).strip()
    max_videos = int(body.get("maxVideos", 5) or 5)
    if not url:
        return jsonify({"ok": False, "error": "url es obligatorio"}), 400
    try:
        if is_youtube_profile_url(url):
            text, used_ids = fetch_youtube_profile_transcripts(url, max_videos=max_videos)
            source_id = add_source("youtube_profile", title, url, text)
            return jsonify(
                {
                    "ok": True,
                    "sourceId": source_id,
                    "chars": len(text),
                    "mode": "profile",
                    "videosIndexed": len(used_ids),
                    "videoUrls": [f"https://www.youtube.com/watch?v={video_id}" for video_id in used_ids],
                }
            )

        text = fetch_youtube_transcript(url)
        source_id = add_source("youtube", title, url, text)
        return jsonify({"ok": True, "sourceId": source_id, "chars": len(text), "mode": "single-video"})
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
    job = {
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
        WEBSITE_JOBS[job_id] = job

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


@app.post("/api/sources/website/jobs/<job_id>/cancel")
def api_sources_website_cancel(job_id: str):
    with WEBSITE_JOBS_LOCK:
        job = WEBSITE_JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "Job no encontrado"}), 404

        current_status = job.get("status")
        if current_status in {"done", "error", "cancelled"}:
            return jsonify({"ok": True, "status": current_status, "message": "Job ya finalizado"})

        job["status"] = "cancelling"
        job["message"] = "Cancelación solicitada..."
        job["updatedAt"] = now_iso()

    return jsonify({"ok": True, "status": "cancelling"})


@app.delete("/api/sources/<int:source_id>")
def api_sources_delete(source_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM knowledge_sources WHERE id = ?", (source_id,)).fetchone()
        if not row:
            return jsonify({"ok": False, "error": "Fuente no encontrada"}), 404
        conn.execute("DELETE FROM knowledge_sources WHERE id = ?", (source_id,))
    return jsonify({"ok": True})


@app.post("/api/train/run")
def api_train_run():
    try:
        result = train_semantic_index()
        return jsonify({"ok": True, "result": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/train/status")
def api_train_status():
    try:
        return jsonify(get_training_status())
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/ask")
def api_ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return jsonify({"ok": False, "error": "question es obligatorio"}), 400
    try:
        result = answer_trained(question)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/context/preview")
def api_context_preview():
    topic = request.args.get("topic", "").strip()
    if not topic:
        return jsonify({"ok": False, "error": "topic es obligatorio"}), 400
    try:
        context = retrieve_context(topic)
        return jsonify({"ok": True, "context": context})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/generate")
def api_generate():
    body = request.get_json(silent=True) or {}
    task_type = str(body.get("taskType", "")).strip()
    topic = str(body.get("topic", "")).strip()
    tone = str(body.get("tone", "")).strip()
    audience = str(body.get("audience", "")).strip()
    length = str(body.get("length", "")).strip()

    if task_type not in TASKS:
        return jsonify({"ok": False, "error": f"taskType inválido. Usa: {', '.join(TASKS.keys())}"}), 400
    if not topic:
        return jsonify({"ok": False, "error": "topic es obligatorio"}), 400

    try:
        context = retrieve_context(topic)
        if not context:
            return jsonify({"ok": False, "error": "No hay contexto relevante. Entrena primero en pestaña Entrenamiento."}), 400
        prompt = build_generation_prompt(task_type, topic, tone, audience, length, context)
        text = ollama_chat([
            {"role": "system", "content": SYSTEM_ENSENYEM},
            {"role": "user", "content": prompt},
        ], temperature=0.15, num_predict=420)

        refs = sorted({item["filename"] for item in context})
        with get_db() as conn:
            generation_id = conn.execute(
                """
                INSERT INTO generations (task_type, topic, tone, audience, generated_text, refs_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (task_type, topic, tone, audience, text, json.dumps(refs, ensure_ascii=False), now_iso()),
            ).lastrowid
        return jsonify({"ok": True, "generationId": generation_id, "text": text, "refs": refs, "context": context})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/generations")
def api_generations():
    limit = int(request.args.get("limit", 20) or 20)
    limit = max(1, min(limit, 100))
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, task_type, topic, tone, audience, refs_json, created_at
            FROM generations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return jsonify(
        {
            "ok": True,
            "items": [
                {
                    "id": row["id"],
                    "taskType": row["task_type"],
                    "topic": row["topic"],
                    "tone": row["tone"],
                    "audience": row["audience"],
                    "refs": json.loads(row["refs_json"]),
                    "createdAt": row["created_at"],
                }
                for row in rows
            ],
        }
    )


@app.get("/api/agent/overview")
def api_agent_overview():
    try:
        return jsonify(get_agent_overview())
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/agent/run")
def api_agent_run():
    body = request.get_json(silent=True) or {}
    channel = str(body.get("channel", "")).strip().lower()
    objective = str(body.get("objective", "")).strip()
    policy_mode = str(body.get("policyMode", "informativo")).strip().lower()
    customer_message = str(body.get("customerMessage", "")).strip()

    try:
        result = run_agent(channel, objective, policy_mode, customer_message)
        return jsonify({"ok": True, **result})
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/agent/runs")
def api_agent_runs():
    limit = int(request.args.get("limit", 12) or 12)
    try:
        return jsonify({"ok": True, "items": list_agent_runs(limit=limit)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5102)
