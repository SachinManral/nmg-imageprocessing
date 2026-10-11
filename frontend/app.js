const API_BASE = window.location.origin.includes("http") ? window.location.origin : "http://localhost:8000";

let state = {
  fileA: null,
  fileB: null,
  presetA: "",
  presetB: "",
  queryFile: null,
  presetQuery: "",
  benchmarkRows: [],
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initDropzones();
  initCameraButtons();
  checkApiHealth();
  loadPresets();
  loadBenchmarkData();
  bindActions();
});

// Tab Navigation
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const target = btn.getAttribute("data-tab");
      document.getElementById(`tab-${target}`).classList.add("active");
    });
  });
}

// API Health Check
async function checkApiHealth() {
  const statusEl = document.getElementById("apiStatus");
  const labelEl = statusEl.querySelector(".status-label");
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      statusEl.className = "api-status connected";
      labelEl.textContent = "API Connected";
    } else {
      throw new Error();
    }
  } catch (err) {
    statusEl.className = "api-status disconnected";
    labelEl.textContent = "API Disconnected (port 8000)";
  }
}

// Load Presets
async function loadPresets() {
  try {
    const res = await fetch(`${API_BASE}/api/presets`);
    if (!res.ok) return;
    const data = await res.json();

    const selectA = document.getElementById("selectPresetA");
    const selectB = document.getElementById("selectPresetB");
    const selectQuery = document.getElementById("selectPresetQuery");

    selectA.innerHTML = '<option value="">-- [Auto-Match Against All References] --</option>';
    selectB.innerHTML = '<option value="">-- Choose Sample Query or Reference --</option>';
    selectQuery.innerHTML = '<option value="">-- Choose Target Query Image --</option>';

    // References group for Box A
    const grpA = document.createElement("optgroup");
    grpA.label = "Registered Inventory References";
    data.references.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r.id;
      opt.dataset.preview = r.preview_url;
      opt.textContent = `${r.name} (${r.filename})`;
      grpA.appendChild(opt);
    });
    selectA.appendChild(grpA);

    // Box B: Group into Genuine Matches, Lookalikes, and Identical Self-Matches
    const grpQueries = document.createElement("optgroup");
    grpQueries.label = "Sample Inspection Photos";


    data.queries.forEach((q) => {
      const opt = document.createElement("option");
      opt.value = q.id;
      opt.dataset.preview = q.preview_url;
      opt.textContent = `${q.name} (${q.filename})`;

      grpQueries.appendChild(opt);

      // Also for gallery search tab
      const optQ = document.createElement("option");
      optQ.value = q.id;
      optQ.dataset.preview = q.preview_url;
      optQ.textContent = `${q.name} (${q.filename})`;
      selectQuery.appendChild(optQ);
    });

    // Reference group in Box B for testing self / identical match
    const grpRefB = document.createElement("optgroup");
    grpRefB.label = "🔍 Test Identical / Self Match (Reference vs Itself)";
    data.references.forEach((r) => {
      const opt = document.createElement("option");
      opt.value = r.id;
      opt.dataset.preview = r.preview_url;
      opt.textContent = `[IDENTICAL] ${r.name} (${r.filename})`;
      grpRefB.appendChild(opt);
    });

    selectB.appendChild(grpQueries);
    selectB.appendChild(grpRefB);
  } catch (err) {
    console.warn("Presets could not be loaded:", err);
  }
}

// Helper to show preset image preview
function setPresetPreview(previewId, contentId, imgUrl) {
  const preview = document.getElementById(previewId);
  const content = document.getElementById(contentId);
  preview.src = imgUrl;
  preview.classList.add("active");
  content.classList.add("hidden");
}

// Setup Dropzones & File Uploads
function initDropzones() {
  setupSingleDropzone("dropzoneA", "fileInputA", "previewA", "dropContentA", (file) => {
    state.fileA = file;
    state.presetA = "";
    document.getElementById("selectPresetA").value = "";
  });

  setupSingleDropzone("dropzoneB", "fileInputB", "previewB", "dropContentB", (file) => {
    state.fileB = file;
    state.presetB = "";
    document.getElementById("selectPresetB").value = "";
  });

  setupSingleDropzone("dropzoneQuery", "fileInputQuery", "previewQuery", "dropContentQuery", (file) => {
    state.queryFile = file;
    state.presetQuery = "";
    document.getElementById("selectPresetQuery").value = "";
  });

  // Preset Selectors with immediate preview rendering
  document.getElementById("selectPresetA").addEventListener("change", (e) => {
    state.presetA = e.target.value;
    state.fileA = null;
    const selectedOpt = e.target.selectedOptions[0];
    if (selectedOpt && selectedOpt.dataset.preview) {
      setPresetPreview("previewA", "dropContentA", `${API_BASE}${selectedOpt.dataset.preview}`);
    } else {
      clearPreview("previewA", "dropContentA");
    }
  });

  document.getElementById("selectPresetB").addEventListener("change", (e) => {
    state.presetB = e.target.value;
    state.fileB = null;
    const selectedOpt = e.target.selectedOptions[0];
    if (selectedOpt && selectedOpt.dataset.preview) {
      setPresetPreview("previewB", "dropContentB", `${API_BASE}${selectedOpt.dataset.preview}`);
    } else {
      clearPreview("previewB", "dropContentB");
    }
  });

  document.getElementById("selectPresetQuery").addEventListener("change", (e) => {
    state.presetQuery = e.target.value;
    state.queryFile = null;
    const selectedOpt = e.target.selectedOptions[0];
    if (selectedOpt && selectedOpt.dataset.preview) {
      setPresetPreview("previewQuery", "dropContentQuery", `${API_BASE}${selectedOpt.dataset.preview}`);
    } else {
      clearPreview("previewQuery", "dropContentQuery");
    }
  });
}

// Mobile & Camera Trigger Buttons
function initCameraButtons() {
  const btnCamB = document.getElementById("btnCaptureCamera");
  const inputCamB = document.getElementById("cameraInputB");
  if (btnCamB && inputCamB) {
    btnCamB.addEventListener("click", () => inputCamB.click());
    inputCamB.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0], document.getElementById("previewB"), document.getElementById("dropContentB"), (file) => {
          state.fileB = file;
          state.presetB = "";
          document.getElementById("selectPresetB").value = "";
        });
      }
    });
  }

  const btnCamQ = document.getElementById("btnCaptureCameraQuery");
  const inputCamQ = document.getElementById("cameraInputQuery");
  if (btnCamQ && inputCamQ) {
    btnCamQ.addEventListener("click", () => inputCamQ.click());
    inputCamQ.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0], document.getElementById("previewQuery"), document.getElementById("dropContentQuery"), (file) => {
          state.queryFile = file;
          state.presetQuery = "";
          document.getElementById("selectPresetQuery").value = "";
        });
      }
    });
  }
}

function setupSingleDropzone(dropzoneId, inputId, previewId, contentId, onFile) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const content = document.getElementById(contentId);

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0], preview, content, onFile);
    }
  });

  input.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0], preview, content, onFile);
    }
  });
}

function handleFile(file, preview, content, onFile) {
  onFile(file);
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.add("active");
    content.classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

function clearPreview(previewId, contentId) {
  const preview = document.getElementById(previewId);
  const content = document.getElementById(contentId);
  preview.src = "";
  preview.classList.remove("active");
  content.classList.remove("hidden");
}

// Bind Action Buttons
function bindActions() {
  document.getElementById("btnVerify").addEventListener("click", handleVerify);
  document.getElementById("btnClear").addEventListener("click", handleClear);
  document.getElementById("btnSearchCatalog").addEventListener("click", handleCatalogSearch);

  // Filters
  document.getElementById("filterPairType").addEventListener("change", applyFilters);
  document.getElementById("filterBand").addEventListener("change", applyFilters);
}

// Clear Pair State
function handleClear() {
  state.fileA = null;
  state.fileB = null;
  state.presetA = "";
  state.presetB = "";

  document.getElementById("fileInputA").value = "";
  document.getElementById("fileInputB").value = "";
  document.getElementById("selectPresetA").value = "";
  document.getElementById("selectPresetB").value = "";

  clearPreview("previewA", "dropContentA");
  clearPreview("previewB", "dropContentB");
  document.getElementById("resultsPanel").classList.add("hidden");
}

// Verify Slab (Supports either 1 single photo or a pair)
async function handleVerify() {
  const hasA = state.fileA || state.presetA;
  const hasB = state.fileB || state.presetB;

  if (!hasA && !hasB) {
    alert("Please upload or capture at least one photo.");
    return;
  }

  const btn = document.getElementById("btnVerify");
  const btnText = document.getElementById("btnVerifyText");
  const spinner = document.getElementById("btnVerifySpinner");

  btn.disabled = true;
  btnText.textContent = hasA && hasB ? "Comparing Pair..." : "Scanning Reference Catalog...";
  spinner.classList.remove("hidden");

  const formData = new FormData();
  if (state.fileA) formData.append("file_a", state.fileA);
  else if (state.presetA) formData.append("preset_a", state.presetA);

  if (state.fileB) formData.append("file_b", state.fileB);
  else if (state.presetB) formData.append("preset_b", state.presetB);

  try {
    const res = await fetch(`${API_BASE}/api/match`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.error || "Failed to process images");
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert(`Verification Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btnText.textContent = "Verify Slab";
    spinner.classList.add("hidden");
  }
}

// Render Results
function renderResults(data) {
  const panel = document.getElementById("resultsPanel");
  panel.classList.remove("hidden");

  const banner = document.getElementById("verdictBanner");
  banner.className = `verdict-banner band-${data.band}`;

  const badge = document.getElementById("verdictBadge");
  badge.textContent = data.band;

  const title = document.getElementById("verdictTitle");
  const desc = document.getElementById("verdictDescription");
  const autoTag = document.getElementById("autoMatchBadge");
  const matchedName = document.getElementById("matchedRefName");

  if (data.is_auto_matched && data.matched_reference_name) {
    autoTag.classList.remove("hidden");
    matchedName.textContent = data.matched_reference_name;
  } else {
    autoTag.classList.add("hidden");
  }

  if (data.band === "SAME") {
    title.textContent = data.is_auto_matched ? "Verified Against Inventory" : "Verified Physical Slab Match";
    desc.textContent = "Robust planar homography confirmed across surface fissures (Identical physical stone).";
  } else if (data.band === "SIMILAR") {
    title.textContent = "Inconclusive Evidence";
    desc.textContent = "Partial feature alignment detected, but insufficient to prove same physical piece.";
  } else {
    title.textContent = data.is_auto_matched ? "No Inventory Match" : "Different Physical Piece";
    desc.textContent = "No consistent planar geometry was found between the visible surface features.";
  }

  document.getElementById("metricConfidence").textContent = `${data.confidence.toFixed(1)}%`;
  document.getElementById("metricInliers").textContent = Number(data.inliers).toLocaleString();
  document.getElementById("metricInlierRatio").textContent = `Ratio: ${(data.inlier_ratio * 100).toFixed(1)}%`;
  document.getElementById("metricGoodMatches").textContent = Number(data.good_matches).toLocaleString();
  document.getElementById("metricTotalMatches").textContent = `Total: ${Number(data.total_matches).toLocaleString()}`;
  document.getElementById("metricKeypoints").textContent = `${data.keypoints_a.toLocaleString()} / ${data.keypoints_b.toLocaleString()}`;
  document.getElementById("metricLatency").textContent = `${data.time_ms.toFixed(1)} ms`;

  const vizImg = document.getElementById("vizImage");
  if (data.visualization) {
    vizImg.src = data.visualization;
  }

  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Catalog Search (Tab 2)
async function handleCatalogSearch() {
  const hasQuery = state.queryFile || state.presetQuery;
  if (!hasQuery) {
    alert("Please upload or select a query image.");
    return;
  }

  const btn = document.getElementById("btnSearchCatalog");
  const btnText = document.getElementById("btnSearchText");
  const spinner = document.getElementById("btnSearchSpinner");

  btn.disabled = true;
  btnText.textContent = "Searching Catalog...";
  spinner.classList.remove("hidden");

  const formData = new FormData();
  if (state.queryFile) formData.append("query_file", state.queryFile);
  else formData.append("preset_query", state.presetQuery);

  try {
    const res = await fetch(`${API_BASE}/api/gallery-search`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error("Search failed");
    const data = await res.json();
    renderCatalogResults(data);
  } catch (err) {
    alert(`Catalog Search Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btnText.textContent = "Search All References";
    spinner.classList.add("hidden");
  }
}

function renderCatalogResults(data) {
  document.getElementById("catalogEmptyState").classList.add("hidden");
  const matchCard = document.getElementById("catalogMatchFound");
  const table = document.getElementById("catalogTable");
  const tbody = document.getElementById("catalogTableBody");

  if (data.top_match) {
    matchCard.classList.remove("hidden");
    document.getElementById("bestMatchName").textContent = data.top_match.reference_name;
    document.getElementById("bestMatchMeta").textContent =
      `Confidence: ${data.top_match.confidence.toFixed(1)}% | ${data.top_match.inliers} Inliers | ${data.top_match.band}`;
  } else {
    matchCard.classList.add("hidden");
  }

  tbody.innerHTML = "";
  data.candidates.forEach((c) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${c.reference_name}</strong></td>
      <td>${c.confidence.toFixed(1)}%</td>
      <td>${c.inliers}</td>
      <td>${(c.inlier_ratio * 100).toFixed(1)}%</td>
      <td><span class="badge badge-${c.band}">${c.band}</span></td>
    `;
    tbody.appendChild(tr);
  });

  table.classList.remove("hidden");
}

// Benchmark Audit (Tab 3)
async function loadBenchmarkData() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmark-results`);
    if (!res.ok) return;
    const data = await res.json();

    state.benchmarkRows = data.rows;

    document.getElementById("statSameAcc").textContent = `${data.summary.same_accuracy}%`;
    document.getElementById("statDiffAcc").textContent = `${data.summary.diff_accuracy}%`;
    document.getElementById("statLookalikeFA").textContent = data.summary.lookalike_false_acceptances;
    document.getElementById("statTotalPairs").textContent = data.summary.total_pairs;

    renderBenchmarkTable(data.rows);
  } catch (err) {
    console.warn("Could not load benchmark data:", err);
  }
}

function applyFilters() {
  const ptype = document.getElementById("filterPairType").value;
  const band = document.getElementById("filterBand").value;

  let filtered = state.benchmarkRows;

  if (ptype !== "all") {
    if (ptype === "same") {
      filtered = filtered.filter((r) => r.label === "same");
    } else {
      filtered = filtered.filter((r) => r.pair_type.toLowerCase() === ptype.toLowerCase());
    }
  }

  if (band !== "all") {
    filtered = filtered.filter((r) => r.band === band);
  }

  renderBenchmarkTable(filtered);
}

function renderBenchmarkTable(rows) {
  const tbody = document.getElementById("benchmarkTableBody");
  tbody.innerHTML = "";

  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-center">No matching records found.</td></tr>`;
    return;
  }

  rows.forEach((r) => {
    const nameA = r.image_a.split("/").pop().replace(".jpg", "");
    const nameB = r.image_b.split("/").pop().replace(".jpg", "");

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td title="${r.image_a}">${truncate(nameA, 28)}</td>
      <td title="${r.image_b}">${truncate(nameB, 28)}</td>
      <td><code>${r.pair_type}</code></td>
      <td>${r.keypoints_a} / ${r.keypoints_b}</td>
      <td><strong>${r.inliers}</strong></td>
      <td>${(parseFloat(r.inlier_ratio) * 100).toFixed(1)}%</td>
      <td>${parseFloat(r.confidence).toFixed(1)}%</td>
      <td><span class="badge badge-${r.band}">${r.band}</span></td>
      <td>${parseFloat(r.time_ms).toFixed(1)} ms</td>
    `;
    tbody.appendChild(tr);
  });
}

function truncate(str, max) {
  return str.length > max ? str.substring(0, max - 3) + "..." : str;
}
