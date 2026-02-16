const $ = (id) => document.getElementById(id);
let currentCrawlJobId = null;

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function stripMarkdownArtifacts(text) {
  return String(text || "")
    .replace(/\*\*/g, "")
    .replace(/^\s*\d+\)\s*/gm, "")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .trim();
}

function extractSection(text, startRegex, endRegex) {
  const start = text.search(startRegex);
  if (start === -1) {
    return "";
  }
  const sliced = text.slice(start);
  const withoutTitle = sliced.replace(startRegex, "").trim();
  if (!endRegex) {
    return withoutTitle.trim();
  }
  const end = withoutTitle.search(endRegex);
  if (end === -1) {
    return withoutTitle.trim();
  }
  return withoutTitle.slice(0, end).trim();
}

function parseAgentResponse(rawResponse) {
  const normalized = stripMarkdownArtifacts(rawResponse);
  const response = extractSection(
    normalized,
    /respuesta\s+al\s+cliente\s*:\s*/i,
    /siguiente\s+acción\s+recomendada\s+para\s+el\s+equipo\s+de\s+ensenyem\s*:/i
  );
  const nextAction = extractSection(
    normalized,
    /siguiente\s+acción\s+recomendada\s+para\s+el\s+equipo\s+de\s+ensenyem\s*:\s*/i,
    /evidencias\s+usadas\s*:/i
  );
  const evidencesText = extractSection(normalized, /evidencias\s+usadas\s*:\s*/i, null);
  const evidences = evidencesText
    .split(/\n+/)
    .map((line) => line.replace(/^•\s*/, "").trim())
    .filter(Boolean);

  if (!response && !nextAction && evidences.length === 0) {
    return { raw: normalized, response: "", nextAction: "", evidences: [] };
  }

  return {
    raw: normalized,
    response,
    nextAction,
    evidences,
  };
}

function renderAgentResponse(rawResponse) {
  const box = $("agent-response-output");
  if (!box) {
    return;
  }
  const parsed = parseAgentResponse(rawResponse);
  if (!parsed.response && !parsed.nextAction && parsed.evidences.length === 0) {
    box.innerHTML = `<div class="agent-rich-fallback">${escapeHtml(parsed.raw || "Sin respuesta")}</div>`;
    return;
  }

  const evidencesHtml = parsed.evidences.length
    ? parsed.evidences.map((evidence) => `<li>${escapeHtml(evidence)}</li>`).join("")
    : '<li>Sin evidencias explícitas en la respuesta.</li>';

  box.innerHTML = `
    <article class="agent-rich-card">
      <div class="agent-rich-section">
        <p class="agent-rich-kicker">Respuesta al cliente</p>
        <p class="agent-rich-text">${escapeHtml(parsed.response || "No disponible")}</p>
      </div>
      <div class="agent-rich-section">
        <p class="agent-rich-kicker">Siguiente acción recomendada</p>
        <p class="agent-rich-text">${escapeHtml(parsed.nextAction || "No disponible")}</p>
      </div>
      <div class="agent-rich-section">
        <p class="agent-rich-kicker">Evidencias usadas</p>
        <ul class="agent-evidence-list">${evidencesHtml}</ul>
      </div>
    </article>
  `;
}

function renderWorkflowOutput(type, title, lines = [], helpText = "") {
  const box = $("workflow-output");
  if (!box) {
    return;
  }
  const badgeClass = `workflow-badge is-${type}`;
  const linesHtml = lines.length
    ? `<ul class="workflow-list">${lines.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}</ul>`
    : "";
  const helpHtml = helpText ? `<p class="workflow-help">${escapeHtml(helpText)}</p>` : "";

  box.innerHTML = `
    <div class="workflow-card">
      <span class="${badgeClass}">${escapeHtml(title)}</span>
      ${linesHtml}
      ${helpHtml}
    </div>
  `;
}

function formatDateTime(isoDate) {
  if (!isoDate) {
    return "fecha desconocida";
  }
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) {
    return isoDate;
  }
  return date.toLocaleString("es-ES", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function sourceTypeLabel(sourceType) {
  const labels = {
    document: "Documento manual",
    pdf: "PDF",
    youtube: "Vídeo YouTube",
    youtube_profile: "Canal/Perfil YouTube",
    website: "Web rastreada",
  };
  return labels[sourceType] || sourceType || "Fuente";
}

function sourceNaturalDescription(source) {
  const title = source.title || "Sin título";
  const type = sourceTypeLabel(source.source_type);
  const chars = Number(source.chars || 0).toLocaleString("es-ES");
  const date = formatDateTime(source.created_at);
  return `${type}: “${title}” · ${chars} caracteres · añadida el ${date}`;
}

function sourceOriginLabel(source) {
  const origin = (source.origin_ref || "").trim();
  if (!origin || origin === "manual") {
    return "Origen: contenido introducido manualmente";
  }
  if (origin.startsWith("http://") || origin.startsWith("https://")) {
    return `Origen: ${origin}`;
  }
  return `Origen: ${origin}`;
}

function renderSources(sources) {
  const box = $("sources-output");
  if (!box) {
    return;
  }

  if (!Array.isArray(sources) || sources.length === 0) {
    box.innerHTML = '<div class="source-empty">No hay fuentes cargadas todavía.</div>';
    return;
  }

  box.innerHTML = sources
    .map(
      (source) => `
        <article class="source-card">
          <div class="source-main">
            <p class="source-title">${escapeHtml(source.title || "Sin título")}</p>
            <p class="source-description">${escapeHtml(sourceNaturalDescription(source))}</p>
            <p class="source-origin">${escapeHtml(sourceOriginLabel(source))}</p>
          </div>
          <div class="source-actions">
            <button type="button" class="danger-btn source-delete" data-source-id="${source.id}">Eliminar</button>
          </div>
        </article>
      `
    )
    .join("");
}

async function deleteSource(sourceId) {
  const numericId = Number(sourceId);
  if (!numericId) {
    return;
  }
  const accepted = window.confirm("¿Eliminar esta fuente? Esta acción no se puede deshacer.");
  if (!accepted) {
    return;
  }
  try {
    await api(`/api/sources/${numericId}`, { method: "DELETE" });
    $("train-output").textContent = `Fuente eliminada ✅ id=${numericId}.`;
    await refreshSources();
  } catch (error) {
    $("train-output").textContent = error.message;
  }
}

function setPipelineStatus(message, type = "info") {
  const box = $("pipeline-status");
  if (!box) {
    return;
  }
  box.textContent = message;
  box.classList.remove("is-info", "is-ok", "is-warn", "is-error");
  box.classList.add(`is-${type}`);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function activateTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

function setCrawlProgress(job) {
  const box = $("crawl-progress");
  const bar = $("crawl-bar");
  const status = $("crawl-status");
  const meta = $("crawl-meta");
  const message = $("crawl-message");
  const cancelBtn = $("btn-cancel-crawl");
  box.classList.remove("hidden");

  const target = Math.max(1, Number(job.maxPages || 1));
  const visited = Math.max(0, Number(job.visitedCount || 0));
  const pct = Math.min(100, Math.round((visited / target) * 100));
  bar.style.width = `${pct}%`;

  status.textContent = `${job.status || "running"}`.toUpperCase();
  meta.textContent = `visitadas: ${visited}/${target} · indexadas: ${job.indexedPages || 0}`;
  message.textContent = job.message || "Procesando...";

  box.classList.toggle("is-done", job.status === "done");
  box.classList.toggle("is-error", job.status === "error" || job.status === "cancelled");

  if (cancelBtn) {
    const canCancel = job.status === "queued" || job.status === "running" || job.status === "cancelling";
    cancelBtn.classList.toggle("hidden", !canCancel);
    cancelBtn.disabled = job.status === "cancelling";
  }
}

async function pollWebsiteJob(jobId) {
  const maxPolls = 240;
  for (let attempt = 0; attempt < maxPolls; attempt += 1) {
    const data = await api(`/api/sources/website/jobs/${jobId}`);
    setCrawlProgress(data.job);
    if (data.job.status === "done" || data.job.status === "error" || data.job.status === "cancelled") {
      if (data.job.status === "cancelled") {
        $("train-output").textContent = "Rastreo cancelado por el usuario.";
      }
      await refreshSources();
      currentCrawlJobId = null;
      return data.job;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  return null;
}

async function cancelCrawl() {
  if (!currentCrawlJobId) {
    return;
  }
  try {
    const data = await api(`/api/sources/website/jobs/${currentCrawlJobId}/cancel`, { method: "POST" });
    $("train-output").textContent = data.status === "cancelling"
      ? "Cancelación solicitada. Esperando parada del rastreo..."
      : "Rastreo finalizado o ya cancelado.";
  } catch (error) {
    $("train-output").textContent = error.message;
  }
}

async function checkHealth() {
  $("health-output").textContent = "Comprobando...";
  try {
    const data = await api("/api/health");
    $("health-output").textContent = pretty(data);
  } catch (error) {
    $("health-output").textContent = error.message;
  }
}

async function refreshSources() {
  const box = $("sources-output");
  if (box) {
    box.innerHTML = '<div class="source-empty">Cargando fuentes...</div>';
  }
  try {
    const data = await api("/api/sources");
    renderSources(data.sources);
    await refreshTrainStatus();
  } catch (error) {
    if (box) {
      box.innerHTML = `<div class="source-empty">${error.message}</div>`;
    }
  }
}

async function refreshTrainStatus() {
  try {
    const data = await api("/api/train/status");
    if (!data.sourceCount) {
      setPipelineStatus("Sin fuentes cargadas todavía. Importa contenido para empezar.", "info");
      return;
    }

    if (data.pendingRetrain) {
      setPipelineStatus(
        `Fuentes cargadas (${data.sourceCount}) pero el entrenamiento está desactualizado. Pulsa 'Entrenar índice semántico'.`,
        "warn"
      );
      return;
    }

    setPipelineStatus(
      `Entrenamiento al día ✅ · fuentes: ${data.sourceCount} · chunks indexados: ${data.chunkCount}`,
      "ok"
    );
  } catch (error) {
    setPipelineStatus(`No se pudo comprobar el estado del entrenamiento: ${error.message}`, "error");
  }
}

async function createDocument(event) {
  event.preventDefault();
  try {
    const payload = {
      title: $("doc-title").value.trim(),
      content: $("doc-content").value.trim(),
    };
    const data = await api("/api/sources/document", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("train-output").textContent = `Documento guardado ✅ sourceId=${data.sourceId}.\nSiguiente paso: pulsa 'Entrenar índice semántico' para que esta fuente se use en respuestas.`;
    event.target.reset();
    refreshSources();
  } catch (error) {
    $("train-output").textContent = error.message;
  }
}

async function uploadPdf(event) {
  event.preventDefault();
  const file = $("pdf-file").files[0];
  if (!file) {
    $("train-output").textContent = "Selecciona un PDF.";
    return;
  }
  const formData = new FormData();
  formData.append("title", $("pdf-title").value.trim());
  formData.append("file", file);
  try {
    const data = await api("/api/sources/pdf", {
      method: "POST",
      body: formData,
    });
    $("train-output").textContent = `PDF importado ✅ sourceId=${data.sourceId} · chars=${data.chars}.\nSiguiente paso: pulsa 'Entrenar índice semántico' para usar este contenido.`;
    event.target.reset();
    refreshSources();
  } catch (error) {
    $("train-output").textContent = error.message;
  }
}

async function importYoutube(event) {
  event.preventDefault();
  try {
    const payload = {
      title: $("yt-title").value.trim(),
      url: $("yt-url").value.trim(),
    };
    const data = await api("/api/sources/youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (data.mode === "profile") {
      $("train-output").textContent = `Perfil/canal YouTube importado ✅ sourceId=${data.sourceId} · chars=${data.chars} · vídeos indexados=${data.videosIndexed}.\nSiguiente paso: pulsa 'Entrenar índice semántico'.`;
    } else {
      $("train-output").textContent = `Vídeo YouTube importado ✅ sourceId=${data.sourceId} · chars=${data.chars}.\nSiguiente paso: pulsa 'Entrenar índice semántico' para usar esta transcripción.`;
    }
    event.target.reset();
    refreshSources();
  } catch (error) {
    if (error.message.includes("consentimiento") || error.message.includes("bloqueando")) {
      $("train-output").textContent = `YouTube bloquea la lectura automática de ese perfil/canal en este entorno.\nUsa URL de vídeo individual (watch?v=...) o importa varios vídeos uno a uno.\n\nDetalle técnico: ${error.message}`;
    } else {
      $("train-output").textContent = error.message;
    }
  }
}

async function crawlWebsite(event) {
  event.preventDefault();
  try {
    const payload = {
      title: $("web-title").value.trim(),
      url: $("web-url").value.trim(),
    };
    const data = await api("/api/sources/website/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("train-output").textContent = `Rastreo iniciado. jobId=${data.jobId}`;
    currentCrawlJobId = data.jobId;
    setCrawlProgress({
      status: "queued",
      message: "Job en cola",
      maxPages: 80,
      visitedCount: 0,
      indexedPages: 0,
    });
    pollWebsiteJob(data.jobId).catch((error) => {
      $("train-output").textContent = error.message;
    });
    refreshTrainStatus();
  } catch (error) {
    $("train-output").textContent = error.message;
  }
}

function applyQuickTemplate(event) {
  const button = event.currentTarget;
  $("task-type").value = button.dataset.task || "course_summary";
  $("topic").value = button.dataset.topic || "";
  $("tone").value = button.dataset.tone || "";
  $("audience").value = button.dataset.audience || "";
  $("length").value = button.dataset.length || "media";
}

async function runTraining() {
  $("train-output").textContent = "Entrenando índice semántico...";
  try {
    const data = await api("/api/train/run", { method: "POST" });
    $("train-output").textContent = `Entrenamiento completado ✅\n${pretty(data.result)}`;
    checkHealth();
    refreshTrainStatus();
  } catch (error) {
    if (error.message.includes("embeddings") || error.message.includes("ollama pull")) {
      $("train-output").textContent = `No se pudo entrenar porque falló la generación de embeddings en Ollama.\n\nAcción recomendada:\n1) Verifica que Ollama esté activo\n2) Ejecuta en terminal: ollama pull nomic-embed-text\n3) Reintenta "Entrenar índice semántico"\n\nDetalle técnico: ${error.message}`;
    } else {
      $("train-output").textContent = error.message;
    }
  }
}

async function trainIndex() {
  return api("/api/train/run", { method: "POST" });
}

async function askQuestion() {
  const question = $("question").value.trim();
  if (!question) {
    $("ask-output").textContent = "Escribe una pregunta.";
    return;
  }
  $("ask-output").textContent = "Consultando...";
  try {
    const data = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    $("ask-output").textContent = `${data.answer}\n\nContexto:\n${pretty(data.context)}`;
  } catch (error) {
    $("ask-output").textContent = error.message;
  }
}

function generationPayload() {
  return {
    taskType: $("task-type").value,
    topic: $("topic").value.trim(),
    tone: $("tone").value.trim(),
    audience: $("audience").value.trim(),
    length: $("length").value,
  };
}

async function previewContext() {
  const topic = $("topic").value.trim();
  if (!topic) {
    $("context-output").textContent = "Introduce un tema.";
    return;
  }
  $("context-output").textContent = "Buscando contexto...";
  try {
    const data = await api(`/api/context/preview?topic=${encodeURIComponent(topic)}`);
    $("context-output").textContent = pretty(data.context);
  } catch (error) {
    $("context-output").textContent = error.message;
  }
}

async function generateContent() {
  const payload = generationPayload();
  if (!payload.topic) {
    $("generation-output").textContent = "El tema es obligatorio.";
    return;
  }
  $("generation-output").textContent = "Generando contenido...";
  try {
    const data = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("generation-output").textContent = data.text;
    $("context-output").textContent = pretty(data.context);
    loadHistory();
  } catch (error) {
    $("generation-output").textContent = error.message;
  }
}

async function loadHistory() {
  $("history-output").textContent = "Cargando historial...";
  try {
    const data = await api("/api/generations?limit=25");
    $("history-output").textContent = pretty(data.items);
  } catch (error) {
    $("history-output").textContent = error.message;
  }
}

function renderAgentRuns(items) {
  const box = $("agent-runs-output");
  if (!box) {
    return;
  }

  if (!Array.isArray(items) || items.length === 0) {
    box.innerHTML = '<div class="source-empty">No hay ejecuciones del agente todavía.</div>';
    return;
  }

  box.innerHTML = items
    .map((item) => {
      const summary = `${item.channel.toUpperCase()} · ${item.policyMode} · ${formatDateTime(item.createdAt)}`;
      return `
        <article class="agent-run-card">
          <p class="agent-run-title">${escapeHtml(item.objective || "Sin objetivo")}</p>
          <p class="agent-run-meta">${escapeHtml(summary)}</p>
          <p class="agent-run-block"><strong>Cliente:</strong> ${escapeHtml(item.customerMessage || "")}</p>
          <p class="agent-run-block"><strong>Agente:</strong> ${escapeHtml((item.agentResponse || "").slice(0, 460))}${
            (item.agentResponse || "").length > 460 ? "…" : ""
          }</p>
        </article>
      `;
    })
    .join("");
}

async function refreshAgentOverview() {
  const box = $("agent-overview-output");
  if (box) {
    box.textContent = "Comprobando estado del agente...";
  }
  try {
    const data = await api("/api/agent/overview");
    const lines = [
      `Canales habilitados: ${data.channels.join(", ")}`,
      `Políticas: ${data.policies.join(", ")}`,
      `Ejecuciones registradas: ${data.totalRuns}`,
      `Entrenamiento pendiente: ${data.training.pendingRetrain ? "sí" : "no"}`,
      `Chunks disponibles: ${data.training.chunkCount}`,
    ];
    if (data.lastRun) {
      lines.push(`Última ejecución: #${data.lastRun.id} · ${data.lastRun.channel} · ${formatDateTime(data.lastRun.created_at)}`);
    }
    box.textContent = lines.join("\n");
  } catch (error) {
    if (box) {
      box.textContent = error.message;
    }
  }
}

function agentPayload() {
  return {
    channel: $("agent-channel").value,
    policyMode: $("agent-policy").value,
    objective: $("agent-objective").value.trim(),
    customerMessage: $("agent-customer-message").value.trim(),
  };
}

async function runAgent() {
  const payload = agentPayload();
  if (!payload.objective || !payload.customerMessage) {
    renderAgentResponse("Respuesta al cliente: Completa objetivo y mensaje del cliente.\nSiguiente acción recomendada para el equipo de Ensenyem: revisa los campos obligatorios.\nEvidencias usadas: formulario incompleto.");
    return;
  }

  $("agent-response-output").innerHTML = '<div class="agent-rich-fallback">Ejecutando agente...</div>';
  $("agent-context-output").textContent = "Buscando contexto...";

  try {
    const data = await api("/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderAgentResponse(data.response);
    $("agent-context-output").textContent = pretty(data.context);
    refreshAgentOverview();
    loadAgentRuns();
  } catch (error) {
    $("agent-response-output").innerHTML = `<div class="agent-rich-fallback">${escapeHtml(error.message)}</div>`;
    $("agent-context-output").textContent = "";
  }
}

async function runAgentWithPayload(payload) {
  return api("/api/agent/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function setWorkflowOutput(message) {
  renderWorkflowOutput("info", message, [], "");
}

async function runAutoWorkflow() {
  const button = $("btn-workflow-auto");
  if (button) {
    button.disabled = true;
  }

  try {
    const webTitle = ($("web-title").value || "").trim() || "Web Principal Ensenyem";
    const webUrl = ($("web-url").value || "").trim() || "https://ensenyem.es";

    renderWorkflowOutput("info", "Paso 1/3 · Iniciando rastreo web", ["Recopilando contenido actualizado de la web corporativa."], "Este paso refresca la base de conocimiento.");
    const start = await api("/api/sources/website/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: webTitle, url: webUrl }),
    });
    currentCrawlJobId = start.jobId;
    const job = await pollWebsiteJob(start.jobId);

    if (!job || job.status !== "done") {
      throw new Error(`El rastreo no finalizó correctamente (${job?.status || "timeout"}).`);
    }

    renderWorkflowOutput("info", "Paso 2/3 · Entrenando índice semántico", ["Transformando el contenido en embeddings para búsqueda semántica."], "Este paso permite que el agente encuentre información relevante.");
    const train = await trainIndex();

    renderWorkflowOutput("info", "Paso 3/3 · Ejecutando agente IA", ["Simulando respuesta al cliente con el contexto ya reentrenado."], "Este paso valida de forma práctica que el prototipo responde bien.");
    const payload = {
      channel: $("agent-channel").value,
      policyMode: $("agent-policy").value,
      objective: ($("agent-objective").value || "").trim() || "Atender consulta de empresa y proponer siguiente paso",
      customerMessage: ($("agent-customer-message").value || "").trim()
        || "Hola, somos una empresa interesada en formación BIM. ¿Qué servicios ofrece Ensenyem y cómo empezamos?",
    };

    const agent = await runAgentWithPayload(payload);
    renderAgentResponse(agent.response);
    $("agent-context-output").textContent = pretty(agent.context);

    renderWorkflowOutput(
      "ok",
      "Workflow completado ✅",
      [
        `Rastreo: ${job.indexedPages || 0} páginas`,
        `Entrenamiento: ${train.result?.chunks || 0} chunks`,
        `Ejecución agente: #${agent.runId}`,
      ],
      "¿Para qué sirve este punto? Para validar en un clic que la web está actualizada, el índice entrenado y el agente listo para atención real."
    );

    refreshSources();
    refreshTrainStatus();
    refreshAgentOverview();
    loadAgentRuns();
  } catch (error) {
    renderWorkflowOutput("error", "Workflow interrumpido ❌", [error.message], "Revisa el mensaje y vuelve a lanzar el flujo cuando se corrija el problema.");
  } finally {
    if (button) {
      button.disabled = false;
    }
  }
}

async function loadAgentRuns() {
  const box = $("agent-runs-output");
  if (box) {
    box.innerHTML = '<div class="source-empty">Cargando trazas del agente...</div>';
  }
  try {
    const data = await api("/api/agent/runs?limit=20");
    renderAgentRuns(data.items);
  } catch (error) {
    if (box) {
      box.innerHTML = `<div class="source-empty">${escapeHtml(error.message)}</div>`;
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  $("btn-health").addEventListener("click", checkHealth);
  $("btn-refresh-sources").addEventListener("click", refreshSources);
  $("btn-train").addEventListener("click", runTraining);
  $("btn-ask").addEventListener("click", askQuestion);

  $("form-document").addEventListener("submit", createDocument);
  $("form-pdf").addEventListener("submit", uploadPdf);
  $("form-youtube").addEventListener("submit", importYoutube);
  $("form-website").addEventListener("submit", crawlWebsite);
  $("btn-cancel-crawl").addEventListener("click", cancelCrawl);

  $("btn-preview").addEventListener("click", previewContext);
  $("btn-generate").addEventListener("click", generateContent);
  $("btn-history").addEventListener("click", loadHistory);

  $("btn-agent-overview").addEventListener("click", refreshAgentOverview);
  $("btn-agent-run").addEventListener("click", runAgent);
  $("btn-agent-runs").addEventListener("click", loadAgentRuns);
  $("btn-agent-runs-refresh").addEventListener("click", loadAgentRuns);
  $("btn-workflow-auto").addEventListener("click", runAutoWorkflow);

  document.querySelectorAll(".quick-template").forEach((button) => {
    button.addEventListener("click", applyQuickTemplate);
  });

  $("sources-output").addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    if (target.classList.contains("source-delete")) {
      deleteSource(target.dataset.sourceId);
    }
  });

  checkHealth();
  refreshSources();
  loadHistory();
  refreshTrainStatus();
  refreshAgentOverview();
  loadAgentRuns();
});
