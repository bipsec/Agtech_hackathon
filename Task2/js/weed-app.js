const T2_CROP_COLORS = {
  "Corn": "#f59e0b", "Wheat": "#10b981", "Soybeans": "#3b82f6",
  "Dry beans": "#ec4899", "Buckwheat": "#8b5cf6",
};

let t2SelectedSubplot = null;
let t2ActiveTab = "composite";

function initWeedApp() {
  const data = WEED_DATA;
  buildT2Table(data);
  wireT2Filters(data);
  wireT2Tabs();
  wireT2ModelSelect();
}

function buildT2Table(data) {
  const maxCorn = Math.max(...data.subplots.map(s => s.corn_count), 1);
  const body = document.getElementById("t2Body");
  body.innerHTML = "";

  data.subplots.forEach(s => {
    const color = T2_CROP_COLORS[s.crop] || "#475569";
    const barW = Math.round((s.corn_count / maxCorn) * 140);
    const tr = document.createElement("tr");
    tr.dataset.crop = s.crop;
    tr.dataset.treatment = s.treatment;
    tr.dataset.subplotId = s.subplot_id;
    tr.innerHTML = `
      <td><strong>${s.subplot_id}</strong></td>
      <td><span class="t2-crop-tag" style="background:${color}">${s.crop}</span></td>
      <td>${s.treatment}</td>
      <td>${s.row}</td>
      <td>${s.col}</td>
      <td>
        <div class="t2-bar-wrap">
          <div class="t2-bar t2-bar-corn" style="width:${barW}px"></div>
          <span class="t2-bar-label">${s.corn_count}</span>
        </div>
      </td>
    `;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => selectT2Subplot(s));
    body.appendChild(tr);
  });
}

function selectT2Subplot(s) {
  t2SelectedSubplot = s;

  document.querySelectorAll("#t2Body tr").forEach(r => r.classList.remove("t2-row-selected"));
  const row = document.querySelector(`#t2Body tr[data-subplot-id="${s.subplot_id}"]`);
  if (row) row.classList.add("t2-row-selected");

  const color = T2_CROP_COLORS[s.crop] || "#475569";
  document.getElementById("t2ViewerTitle").innerHTML =
    `<span class="t2-crop-tag" style="background:${color}">${s.crop}</span> ` +
    `<strong>${s.subplot_id}</strong> — ${s.treatment} ` +
    `<span style="color:#64748b;font-size:.72rem">(Corn: ${s.corn_count})</span>`;

  showT2Image(t2ActiveTab);
}

function showT2Image(panel) {
  t2ActiveTab = panel;
  const body = document.getElementById("t2ViewerBody");

  document.querySelectorAll(".t2-tab").forEach(b => b.classList.remove("active"));
  const activeBtn = document.querySelector(`.t2-tab[data-panel="${panel}"]`);
  if (activeBtn) activeBtn.classList.add("active");

  if (!t2SelectedSubplot || !t2SelectedSubplot.files) {
    body.innerHTML = '<div class="t2-viewer-empty">Click a subplot row to view detection results</div>';
    return;
  }

  const src = t2SelectedSubplot.files[panel];
  if (!src) {
    body.innerHTML = '<div class="t2-viewer-empty">Image not available</div>';
    return;
  }

  const labels = {
    composite: "Composite (2×2)",
    crop_img: "Original Crop",
    overlay: "Detection Overlay",
    mask: "Segmentation Mask",
  };

  body.innerHTML = `
    <div class="t2-img-wrap">
      <img class="t2-viewer-img" src="${src}" alt="${labels[panel]}"
           onerror="this.parentElement.innerHTML='<div class=\\'t2-viewer-empty\\'>Image not found.<br>Run <code>python batch_generate.py</code> in Task2/ first.</div>'" />
    </div>
  `;
}

function wireT2Tabs() {
  document.querySelectorAll(".t2-tab").forEach(btn => {
    btn.addEventListener("click", () => showT2Image(btn.dataset.panel));
  });
}


function wireT2Filters(data) {
  const cropSel = document.getElementById("t2CropFilter");
  const treatSel = document.getElementById("t2TreatFilter");

  function applyFilters() {
    const cropVal = cropSel.value;
    const treatVal = treatSel.value;
    document.querySelectorAll("#t2Body tr").forEach(tr => {
      const matchCrop = cropVal === "all" || tr.dataset.crop === cropVal;
      const matchTreat = treatVal === "all" || tr.dataset.treatment === treatVal;
      tr.classList.toggle("t2-row-hidden", !(matchCrop && matchTreat));
    });
  }

  cropSel.addEventListener("change", applyFilters);
  treatSel.addEventListener("change", applyFilters);
  applyFilters();
}

function wireT2ModelSelect() {
  const sel = document.getElementById("t2ModelSelect");
  const label = document.getElementById("t2ModelLabel");
  if (!sel || !label) return;

  function sync() {
    label.textContent = sel.value;
  }

  sel.addEventListener("change", sync);
  sync();
}
