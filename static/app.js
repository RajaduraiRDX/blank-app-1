const form = document.querySelector("#analysisForm");
const imageInput = document.querySelector("#imageInput");
const previewImage = document.querySelector("#previewImage");
const statusTitle = document.querySelector("#statusTitle");
const analysisMode = document.querySelector("#analysisMode");
const analysisEyebrow = document.querySelector("#analysisEyebrow");
const metricCards = document.querySelectorAll("#metricCards article");
const rows = document.querySelector("#measurementRows");
const headRow = document.querySelector("#measurementHead tr");
const tableTitle = document.querySelector("#tableTitle");
const unitBadge = document.querySelector("#unitBadge");
const chartCanvas = document.querySelector("#chartCanvas");
const exportButtons = document.querySelectorAll("[data-export]");
const themeToggle = document.querySelector("#themeToggle");
let currentRunId = null;

const surfaceViews = [
  "surface_roi",
  "illumination_corrected",
  "gradient_map",
  "laplacian_map",
  "lbp_map",
];
const morphologyViews = [
  "binary",
  "contour",
  "convex_hull",
  "bounding_feret",
  "major_minor_axes",
  "angular_corners",
];

syncModeUi();

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  previewImage.src = URL.createObjectURL(file);
  statusTitle.textContent = file.name;
});

analysisMode.addEventListener("change", syncModeUi);

themeToggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  themeToggle.textContent = document.body.classList.contains("dark") ? "Light mode" : "Dark mode";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(form);
  formData.set("multi_particle", document.querySelector("#multiParticle").checked ? "true" : "false");
  setBusy(true);
  statusTitle.textContent =
    analysisMode.value === "surface_texture"
      ? "Analyzing surface texture..."
      : "Analyzing aggregate morphology...";
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Analysis failed.");
    renderResult(result);
  } catch (error) {
    statusTitle.textContent = error.message;
  } finally {
    setBusy(false);
  }
});

exportButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!currentRunId) return;
    window.location.href = `/download/${currentRunId}/${button.dataset.export}`;
  });
});

function syncModeUi() {
  const isTexture = analysisMode.value === "surface_texture";
  document.body.classList.toggle("morphology", !isTexture);
  document.querySelectorAll(".morphology-control").forEach((control) => {
    control.style.display = isTexture ? "none" : "";
  });
  analysisEyebrow.textContent = isTexture
    ? "Square aggregate surface texture"
    : "Aggregate morphology and texture";
  statusTitle.textContent = isTexture
    ? "Ready for surface texture analysis"
    : "Ready for image analysis";
  tableTitle.textContent = isTexture ? "Surface Texture Measurements" : "Particle Measurements";
  setMetricLabels(isTexture);
}

function setBusy(isBusy) {
  form.querySelector("button[type='submit']").disabled = isBusy;
}

function renderResult(result) {
  currentRunId = result.run_id;
  const isTexture = result.analysis_mode === "surface_texture";
  statusTitle.textContent = isTexture
    ? "Surface texture analyzed"
    : `${result.particle_count} particle${result.particle_count === 1 ? "" : "s"} analyzed`;
  exportButtons.forEach((button) => (button.disabled = false));
  unitBadge.textContent = result.calibration_mm_per_px ? "SI units" : "px units";

  renderMetricCards(result, isTexture);
  renderImages(result.images, isTexture);
  renderTable(result.measurements, isTexture);
  drawHistogram(result.histograms, isTexture);
}

function setMetricLabels(isTexture) {
  const labels = isTexture
    ? ["Texture Index", "Roughness", "Entropy", "Contrast"]
    : ["Particles", "Mean area", "Angularity", "Texture"];
  metricCards.forEach((card, index) => {
    card.querySelector("span").textContent = labels[index];
    card.querySelector("strong").textContent = "0";
  });
}

function renderMetricCards(result, isTexture) {
  setMetricLabels(isTexture);
  const summary = result.summary;
  const values = isTexture
    ? [
        summary.texture_index?.mean,
        summary.roughness_index?.mean,
        summary.entropy?.mean,
        summary.intensity_contrast?.mean,
      ]
    : [
        result.particle_count,
        summary.area?.mean,
        summary.angularity_index?.mean,
        summary.texture_index?.mean,
      ];
  metricCards.forEach((card, index) => {
    card.querySelector("strong").textContent = values[index] ?? 0;
  });
}

function renderImages(images, isTexture) {
  document.querySelectorAll("[data-view]").forEach((img) => {
    img.removeAttribute("src");
  });
  const expected = isTexture ? surfaceViews : morphologyViews;
  Object.entries(images).forEach(([key, url]) => {
    const img = document.querySelector(`[data-view="${key}"]`);
    if (img) img.src = `${url}?v=${Date.now()}`;
  });
  expected.forEach((key) => {
    const img = document.querySelector(`[data-view="${key}"]`);
    if (img && images[key]) img.src = `${images[key]}?v=${Date.now()}`;
  });
}

function renderTable(measurements, isTexture) {
  const columns = isTexture
    ? [
        ["surface_area", "Area"],
        ["texture_index", "Texture"],
        ["micro_texture_index", "Micro texture"],
        ["roughness_index", "Roughness"],
        ["laplacian_energy", "Laplacian"],
        ["entropy", "Entropy"],
        ["intensity_contrast", "Contrast"],
        ["lbp_uniformity", "LBP uniformity"],
      ]
    : [
        ["id", "ID"],
        ["area", "Area"],
        ["perimeter", "Perimeter"],
        ["circularity", "Circularity"],
        ["feret_diameter", "Feret"],
        ["angularity_index", "Angularity"],
        ["texture_index", "Texture"],
      ];

  headRow.innerHTML = columns.map(([, label]) => `<th>${label}</th>`).join("");
  rows.innerHTML = measurements
    .map(
      (measurement) => `
      <tr>
        ${columns.map(([key]) => `<td>${measurement[key] ?? ""}</td>`).join("")}
      </tr>
    `
    )
    .join("");
}

function drawHistogram(histograms, isTexture) {
  const ctx = chartCanvas.getContext("2d");
  const width = chartCanvas.width;
  const height = chartCanvas.height;
  ctx.clearRect(0, 0, width, height);
  const datasets = isTexture
    ? [
        ["texture_index", "#0f8b8d", "Texture index"],
        ["roughness_index", "#c46b31", "Roughness index"],
        ["intensity_distribution", "#5967d8", "Gray intensity"],
      ]
    : [
        ["feret_diameter", "#0f8b8d", "Feret diameter"],
        ["angularity_index", "#c46b31", "Angularity index"],
        ["texture_index", "#5967d8", "Texture index"],
      ];
  const plotWidth = width / datasets.length;
  ctx.font = "14px system-ui";
  datasets.forEach(([key, color, xAxisLabel], panelIndex) => {
    const data = histograms[key] || { bins: [], counts: [] };
    const left = panelIndex * plotWidth + 44;
    const top = 30;
    const innerWidth = plotWidth - 72;
    const innerHeight = height - 110;
    const maxCount = Math.max(1, ...data.counts);
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--muted");
    ctx.fillText(key.replaceAll("_", " "), left, 20);
    data.counts.forEach((count, index) => {
      const barWidth = innerWidth / data.counts.length - 6;
      const barHeight = (count / maxCount) * innerHeight;
      const x = left + index * (innerWidth / data.counts.length);
      const y = top + innerHeight - barHeight;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, Math.max(4, barWidth), barHeight);
    });
    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--line");
    ctx.strokeRect(left, top, innerWidth, innerHeight);
    drawAxisLabels(ctx, left, top, innerWidth, innerHeight, xAxisLabel, "Frequency");
  });
}

function drawAxisLabels(ctx, left, top, innerWidth, innerHeight, xLabel, yLabel) {
  const ink = getComputedStyle(document.body).getPropertyValue("--ink");
  ctx.save();
  ctx.fillStyle = ink;
  ctx.font = "bold 13px system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(xLabel, left + innerWidth / 2, top + innerHeight + 6);
  ctx.textBaseline = "middle";
  ctx.translate(left - 6, top + innerHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}
