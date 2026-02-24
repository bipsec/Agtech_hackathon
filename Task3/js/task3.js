let t3SelectedSubplot = null;

function initTask3() {
  if (typeof WEED_DATA === "undefined") return;
  buildT3Table(WEED_DATA);
  wireT3Filters();
}

function buildT3Table(data) {
  const body = document.getElementById("t3Body");
  if (!body) return;
  body.innerHTML = "";

  data.subplots.forEach(s => {
    const cropColor = (typeof T2_CROP_COLORS !== "undefined" && T2_CROP_COLORS[s.crop])
      ? T2_CROP_COLORS[s.crop]
      : "#64748b";
    const tr = document.createElement("tr");
    tr.dataset.crop = s.crop;
    tr.dataset.treatment = s.treatment;
    tr.dataset.subplotId = s.subplot_id;
    tr.innerHTML = `
      <td><strong>${s.subplot_id}</strong></td>
      <td><span class="t2-crop-tag" style="background:${cropColor}">${s.crop}</span></td>
      <td><strong class="t3-count">${s.corn_count}</strong></td>
    `;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => selectT3Subplot(s));
    body.appendChild(tr);
  });
}

function selectT3Subplot(s) {
  t3SelectedSubplot = s;
  document.querySelectorAll("#t3Body tr").forEach(r => r.classList.remove("t2-row-selected"));
  const row = document.querySelector(`#t3Body tr[data-subplot-id="${s.subplot_id}"]`);
  if (row) row.classList.add("t2-row-selected");

  const title = document.getElementById("t3ViewerTitle");
  if (title) {
    title.innerHTML = `<strong>${s.subplot_id}</strong> — ${s.crop} <span style="color:#64748b;font-size:.72rem">(Corn: ${s.corn_count})</span>`;
  }

  showT3Overlay();
}

function showT3Overlay() {
  const body = document.getElementById("t3ViewerBody");
  if (!body) return;
  if (!t3SelectedSubplot || !t3SelectedSubplot.files || !t3SelectedSubplot.files.overlay) {
    body.innerHTML = '<div class="t2-viewer-empty">Overlay not available</div>';
    return;
  }

  const src = t3SelectedSubplot.files.overlay;
  body.innerHTML = `
    <div class="t2-img-wrap">
      <img class="t2-viewer-img" src="${src}" alt="Overlay"
           onerror="this.parentElement.innerHTML='<div class=\\'t2-viewer-empty\\'>Image not found.</div>'" />
    </div>
  `;
}

function wireT3Filters() {
  const cropSel = document.getElementById("t3CropFilter");
  const treatSel = document.getElementById("t3TreatFilter");
  if (!cropSel || !treatSel) return;

  function applyFilters() {
    const cropVal = cropSel.value;
    const treatVal = treatSel.value;
    document.querySelectorAll("#t3Body tr").forEach(tr => {
      const matchCrop = cropVal === "all" || tr.dataset.crop === cropVal;
      const matchTreat = treatVal === "all" || tr.dataset.treatment === treatVal;
      tr.classList.toggle("t2-row-hidden", !(matchCrop && matchTreat));
    });
  }

  cropSel.addEventListener("change", applyFilters);
  treatSel.addEventListener("change", applyFilters);
  applyFilters();
}
