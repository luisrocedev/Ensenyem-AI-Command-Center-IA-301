const statusEl = document.getElementById("status");
const answerEl = document.getElementById("answer");
const questionEl = document.getElementById("question");
const kpisEl = document.getElementById("kpis");
const sourcesListEl = document.getElementById("sourcesList");
const webProgressWrapEl = document.getElementById("webProgressWrap");
const webProgressStatusEl = document.getElementById("webProgressStatus");
const webProgressPercentEl = document.getElementById("webProgressPercent");
const webProgressBarEl = document.getElementById("webProgressBar");
const webProgressMetaEl = document.getElementById("webProgressMeta");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || "Error de API");
  }
  return payload;
}

async function apiForm(path, formData) {
  const response = await fetch(path, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || "Error de API");
  }
  return payload;
}

function setStatus(obj) {
  statusEl.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function updateWebProgress({ status, visitedCount = 0, indexedPages = 0, lastUrl = "", maxPages = 80 }) {
  webProgressWrapEl.classList.remove("hidden");
  const percentRaw = Math.round((Math.min(visitedCount, maxPages) / Math.max(1, maxPages)) * 100);
  const percent = status === "done" ? 100 : Math.min(percentRaw, 99);
  webProgressStatusEl.textContent = `Estado: ${status}`;
  webProgressPercentEl.textContent = `${percent}%`;
  webProgressBarEl.style.width = `${percent}%`;
  webProgressMetaEl.textContent = `Visitadas: ${visitedCount} · Indexadas: ${indexedPages}${lastUrl ? ` · URL: ${lastUrl}` : ""}`;
}

function resetWebProgress() {
  webProgressWrapEl.classList.add("hidden");
  webProgressStatusEl.textContent = "En cola";
  webProgressPercentEl.textContent = "0%";
  webProgressBarEl.style.width = "0%";
  webProgressMetaEl.textContent = "Esperando inicio...";
}

function renderKpis(stats, evalInfo = null) {
  const base = [
    ["Fuentes", stats.sources],
    ["Docs Indexados", stats.documents],
    ["Chunks", stats.chunks],
    ["Entrenamientos", stats.training_runs],
    ["Resultados benchmark", stats.benchmark_results],
    ["Interacciones", stats.interactions],
  ];

  if (evalInfo) {
    base.push(["Score Antes", evalInfo.baseline]);
    base.push(["Score Después", evalInfo.trained]);
    base.push(["Mejora", evalInfo.delta]);
  }

  kpisEl.innerHTML = base
    .map(([label, value]) => `<article class="kpi"><strong>${label}</strong><span>${value}</span></article>`)
    .join("");
}

async function loadStats() {
  const data = await api("/api/stats");
  renderKpis(data.stats, data.latestEvaluation || null);
  setStatus(data);
}

async function loadSources() {
  const data = await api("/api/sources");
  if (!data.sources.length) {
    sourcesListEl.innerHTML = "<p>No hay fuentes todavía. Añade PDF, documento o YouTube.</p>";
    return;
  }

  const rows = data.sources
    .map(
      (source) => `
      <tr>
        <td>${source.id}</td>
        <td>${source.source_type}</td>
        <td>${source.title}</td>
        <td>${source.chars}</td>
        <td><button data-del="${source.id}">Eliminar</button></td>
      </tr>
    `
    )
    .join("");

  sourcesListEl.innerHTML = `
    <table>
      <thead><tr><th>ID</th><th>Tipo</th><th>Título</th><th>Chars</th><th>Acción</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  sourcesListEl.querySelectorAll("button[data-del]").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.dataset.del;
      try {
        await api(`/api/sources/${id}`, { method: "DELETE" });
        await loadSources();
        await loadStats();
      } catch (error) {
        setStatus(`Error eliminando fuente: ${error.message}`);
      }
    });
  });
}

document.getElementById("btnHealth").addEventListener("click", async () => {
  try {
    const data = await api("/api/health");
    setStatus(data);
  } catch (error) {
    setStatus(`Error health: ${error.message}`);
  }
});

document.getElementById("btnTrain").addEventListener("click", async () => {
  try {
    setStatus("Entrenando índice semántico... esto puede tardar.");
    const data = await api("/api/train/run", { method: "POST", body: "{}" });
    setStatus(data);
    await loadStats();
  } catch (error) {
    setStatus(`Error entrenamiento: ${error.message}`);
  }
});

document.getElementById("btnEval").addEventListener("click", async () => {
  try {
    setStatus("Ejecutando benchmark antes/después...");
    const data = await api("/api/evaluate", { method: "POST", body: "{}" });
    setStatus(data);
    await loadStats();
  } catch (error) {
    setStatus(`Error benchmark: ${error.message}`);
  }
});

document.getElementById("btnStats").addEventListener("click", loadStats);

document.getElementById("btnAddDoc").addEventListener("click", async () => {
  const title = document.getElementById("docTitle").value.trim();
  const content = document.getElementById("docContent").value.trim();
  try {
    const data = await api("/api/sources/document", {
      method: "POST",
      body: JSON.stringify({ title, content }),
    });
    setStatus(data);
    document.getElementById("docContent").value = "";
    await loadSources();
    await loadStats();
  } catch (error) {
    setStatus(`Error documento: ${error.message}`);
  }
});

document.getElementById("btnAddYoutube").addEventListener("click", async () => {
  const title = document.getElementById("ytTitle").value.trim();
  const url = document.getElementById("ytUrl").value.trim();
  try {
    const data = await api("/api/sources/youtube", {
      method: "POST",
      body: JSON.stringify({ title, url }),
    });
    setStatus(data);
    await loadSources();
    await loadStats();
  } catch (error) {
    setStatus(`Error YouTube: ${error.message}`);
  }
});

document.getElementById("btnAddPdf").addEventListener("click", async () => {
  const title = document.getElementById("pdfTitle").value.trim();
  const fileInput = document.getElementById("pdfFile");
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Selecciona un PDF primero.");
    return;
  }
  try {
    const formData = new FormData();
    formData.append("title", title);
    formData.append("file", file);
    const data = await apiForm("/api/sources/pdf", formData);
    setStatus(data);
    fileInput.value = "";
    await loadSources();
    await loadStats();
  } catch (error) {
    setStatus(`Error PDF: ${error.message}`);
  }
});

document.getElementById("btnAddWebsite").addEventListener("click", async () => {
  const title = document.getElementById("webTitle").value.trim();
  const rawUrl = document.getElementById("webUrl").value.trim();
  const url = rawUrl && !rawUrl.startsWith("http://") && !rawUrl.startsWith("https://") ? `https://${rawUrl}` : rawUrl;
  const button = document.getElementById("btnAddWebsite");

  if (!url) {
    setStatus("Escribe una URL de web (por ejemplo: ensenyem.es o https://ensenyem.es)");
    return;
  }

  try {
    button.disabled = true;
    resetWebProgress();
    webProgressWrapEl.classList.remove("hidden");
    setStatus("Lanzando rastreo web...");
    const start = await api("/api/sources/website/start", {
      method: "POST",
      body: JSON.stringify({ title, url }),
    });

    const jobId = start.jobId;
    let done = false;
    while (!done) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const progress = await api(`/api/sources/website/jobs/${jobId}`);
      const job = progress.job;
      updateWebProgress({
        status: job.status,
        visitedCount: job.visitedCount,
        indexedPages: job.indexedPages,
        lastUrl: job.lastUrl,
        maxPages: job.maxPages || 80,
      });
      setStatus({
        jobId,
        status: job.status,
        message: job.message,
        visitedCount: job.visitedCount,
        indexedPages: job.indexedPages,
        lastUrl: job.lastUrl,
      });

      if (job.status === "done") {
        done = true;
        updateWebProgress({
          status: "done",
          visitedCount: job.visitedCount,
          indexedPages: job.indexedPages,
          lastUrl: job.lastUrl,
          maxPages: job.maxPages || 80,
        });
        setStatus({
          ok: true,
          status: "done",
          sourceId: job.sourceId,
          chars: job.chars,
          indexedPages: job.indexedPages,
          mode: job.mode,
        });
        await loadSources();
        await loadStats();
      }

      if (job.status === "error") {
        done = true;
        updateWebProgress({
          status: "error",
          visitedCount: job.visitedCount,
          indexedPages: job.indexedPages,
          lastUrl: job.lastUrl,
          maxPages: job.maxPages || 80,
        });
        throw new Error(job.message || "Error desconocido en rastreo web");
      }
    }
  } catch (error) {
    setStatus(`Error web: ${error.message}`);
  } finally {
    button.disabled = false;
  }
});

document.querySelectorAll(".btnAsk").forEach((button) => {
  button.addEventListener("click", async () => {
    const question = questionEl.value.trim();
    if (!question) {
      answerEl.textContent = "Escribe una pregunta primero.";
      return;
    }
    const mode = button.dataset.mode;
    try {
      answerEl.textContent = "Consultando...";
      const data = await api("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question, mode }),
      });
      answerEl.textContent = data.answer;
      setStatus({ mode, grounding: data.grounding || null, contextTop: data.context?.slice(0, 2) || [] });
      await loadStats();
    } catch (error) {
      answerEl.textContent = `Error: ${error.message}`;
    }
  });
});

Promise.all([loadStats(), loadSources()]).catch((error) => setStatus(`Error init: ${error.message}`));
resetWebProgress();
