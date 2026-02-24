const allCells = [];
let activeDetailCell = null;
let viewBox = { x: 0, y: 0, w: 5225, h: 5251 };

function fmtNum(n) {
  return n >= 1000 ? (n / 1000).toFixed(1) + "k" : n.toFixed(2);
}

function showDetailPanel(d) {
  const colors = CROP_COLORS[d.crop] || { bg: "#475569", fg: "#fff" };
  const body = document.getElementById("detailBody");

  const bboxStr = d.bbox
    ? `[${d.bbox.map(v => v.toFixed(1)).join(", ")}]`
    : "—";

  body.innerHTML = `
    <div class="dp-card">
      <div class="dp-card-header" style="background:${colors.bg}">
        <span class="dp-id">${d.subPlot}</span>
        <span class="dp-crop-name">${d.crop}</span>
      </div>
      <div class="dp-rows">
        <div class="dp-row">
          <span class="dp-row-label">Plot Number</span>
          <span class="dp-row-value">${d.plotNum}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">Subplot ID</span>
          <span class="dp-row-value">${d.subPlot}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">COCO Ann. #</span>
          <span class="dp-row-value">${d.id}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">Crop</span>
          <span class="dp-row-value" style="color:${colors.bg}">${d.crop}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">Treatment</span>
          <span class="dp-row-value">${d.treatment}</span>
        </div>
      </div>

      <div class="dp-section-title">Grid Position</div>
      <div class="dp-position">
        <div class="dp-pos-item">
          <span class="dp-pos-label">Col</span>
          <span class="dp-pos-val">${d.col}</span>
        </div>
        <div class="dp-pos-item">
          <span class="dp-pos-label">Row</span>
          <span class="dp-pos-val">${d.row}</span>
        </div>
        <div class="dp-pos-item">
          <span class="dp-pos-label">Sub</span>
          <span class="dp-pos-val">${d.sub}</span>
        </div>
      </div>

      <div class="dp-section-title">COCO Annotation</div>
      <div class="dp-rows dp-coco">
        <div class="dp-row">
          <span class="dp-row-label">BBox (px)</span>
          <span class="dp-row-value dp-mono">${bboxStr}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">Area (px&sup2;)</span>
          <span class="dp-row-value">${fmtNum(d.area_px2)}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">Seg. Vertices</span>
          <span class="dp-row-value">${d.pts.length}</span>
        </div>
        <div class="dp-row">
          <span class="dp-row-label">Image</span>
          <span class="dp-row-value">${CONFIG.IMG_W}&times;${CONFIG.IMG_H}</span>
        </div>
      </div>

      <div class="dp-dims">
        Subplot: <strong>${CONFIG.SUB_W_FT} &times; ${CONFIG.PLOT_H_FT} ft</strong>
        &nbsp;|&nbsp;
        Plot: <strong>${CONFIG.PLOT_W_FT} &times; ${CONFIG.PLOT_H_FT} ft</strong>
      </div>
    </div>
  `;
}

function setActiveDetailCell(g, d) {
  if (activeDetailCell) activeDetailCell.classList.remove("dp-active");
  activeDetailCell = g;
  g.classList.add("dp-active");
  showDetailPanel(d);
}

function clearDetailPanel() {
  if (activeDetailCell) {
    activeDetailCell.classList.remove("dp-active");
    activeDetailCell = null;
  }
  document.getElementById("detailBody").innerHTML =
    '<div class="dp-empty">Drag the drone over the field<br>and hover a revealed subplot</div>';
}

function computeViewBox() {
  const pad = CONFIG.VIEW_PAD;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  SUBPLOT_POLYGONS.forEach(d => {
    d.pts.forEach(([x, y]) => {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    });
  });
  viewBox = {
    x: minX - pad,
    y: minY - pad,
    w: (maxX - minX) + pad * 2,
    h: (maxY - minY) + pad * 2,
  };
}

function buildField() {
  const { IMG_W, IMG_H, SPEECH_TEXT, BG_IMAGE } = CONFIG;
  const svg = document.getElementById("fieldSvg");

  computeViewBox();
  svg.setAttribute("viewBox", `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);

  const NS = "http://www.w3.org/2000/svg";

  const defs = document.createElementNS(NS, "defs");
  const grad = document.createElementNS(NS, "linearGradient");
  grad.setAttribute("id", "t1-bubble-gradient");
  grad.setAttribute("x1", "0");
  grad.setAttribute("y1", "0");
  grad.setAttribute("x2", "1");
  grad.setAttribute("y2", "1");
  const stop1 = document.createElementNS(NS, "stop");
  stop1.setAttribute("offset", "0%");
  stop1.setAttribute("stop-color", "#fff7ed");
  const stop2 = document.createElementNS(NS, "stop");
  stop2.setAttribute("offset", "100%");
  stop2.setAttribute("stop-color", "#fed7aa");
  grad.appendChild(stop1);
  grad.appendChild(stop2);
  defs.appendChild(grad);
  svg.appendChild(defs);

  // Background image covers the full TIF coordinate space
  const bgImg = document.createElementNS(NS, "image");
  bgImg.setAttribute("href", BG_IMAGE);
  bgImg.setAttribute("x", 0);
  bgImg.setAttribute("y", 0);
  bgImg.setAttribute("width", IMG_W);
  bgImg.setAttribute("height", IMG_H);
  bgImg.setAttribute("preserveAspectRatio", "xMidYMid slice");
  bgImg.setAttribute("class", "field-bg");
  svg.appendChild(bgImg);

  const avgCellW = viewBox.w / 12;

  SUBPLOT_POLYGONS.forEach(d => {
    const colors = CROP_COLORS[d.crop] || { bg: "#475569", fg: "#fff" };

    const pointsStr = d.pts.map(([px, py]) => `${px},${py}`).join(" ");
    const cx = d.center[0];
    const cy = d.center[1];

    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "subplot-cell");
    g.dataset.id = d.id;

    const poly = document.createElementNS(NS, "polygon");
    poly.setAttribute("points", pointsStr);
    poly.setAttribute("class", "cell-poly");
    poly.style.fill = colors.bg;
    poly.style.stroke = colors.bg;
    g.appendChild(poly);

    const cover = document.createElementNS(NS, "polygon");
    cover.setAttribute("points", pointsStr);
    cover.setAttribute("class", "cell-cover");
    g.appendChild(cover);

    const fontSize = avgCellW * 0.35;

    const qmark = document.createElementNS(NS, "text");
    qmark.setAttribute("x", cx);
    qmark.setAttribute("y", cy + fontSize * 0.15);
    qmark.setAttribute("class", "q-mark");
    qmark.setAttribute("font-size", fontSize * 1.2);
    qmark.textContent = "?";
    g.appendChild(qmark);

    const bw = avgCellW * 3;
    const bh = avgCellW * 0.55;
    const bubble = document.createElementNS(NS, "g");
    bubble.setAttribute("class", "speech-bubble");
    bubble.setAttribute("transform", `translate(${cx}, ${cy - avgCellW * 0.75})`);

    const bubbleRect = document.createElementNS(NS, "rect");
    bubbleRect.setAttribute("x", -bw / 2);
    bubbleRect.setAttribute("y", -bh / 2);
    bubbleRect.setAttribute("width", bw);
    bubbleRect.setAttribute("height", bh);
    bubbleRect.setAttribute("rx", bh * 0.4);
    bubbleRect.setAttribute("class", "bubble-bg");
    bubble.appendChild(bubbleRect);

    const tail = document.createElementNS(NS, "polygon");
    const tw = avgCellW * 0.08;
    const th = avgCellW * 0.18;
    tail.setAttribute("points", `${-tw},${bh/2} ${tw},${bh/2} 0,${bh/2 + th}`);
    tail.setAttribute("class", "bubble-tail");
    bubble.appendChild(tail);

    const bubbleText = document.createElementNS(NS, "text");
    bubbleText.setAttribute("x", 0);
    bubbleText.setAttribute("y", 0);
    bubbleText.setAttribute("class", "bubble-text");
    bubbleText.setAttribute("font-size", bh * 0.45);
    bubbleText.textContent = SPEECH_TEXT;
    bubble.appendChild(bubbleText);

    g.appendChild(bubble);

    const idLabel = document.createElementNS(NS, "text");
    idLabel.setAttribute("x", cx);
    idLabel.setAttribute("y", cy - fontSize * 0.4);
    idLabel.setAttribute("class", "cell-id-label");
    idLabel.setAttribute("font-size", fontSize * 0.75);
    idLabel.style.fill = colors.fg;
    idLabel.textContent = d.subPlot;
    g.appendChild(idLabel);

    const cropLabel = document.createElementNS(NS, "text");
    cropLabel.setAttribute("x", cx);
    cropLabel.setAttribute("y", cy + fontSize * 0.6);
    cropLabel.setAttribute("class", "cell-crop-label");
    cropLabel.setAttribute("font-size", fontSize * 0.55);
    cropLabel.style.fill = colors.fg;
    cropLabel.textContent = d.crop;
    g.appendChild(cropLabel);

    g._data = d;
    g._poly = poly;
    allCells.push(g);

    g.addEventListener("mouseenter", e => {
      if (g.classList.contains("revealed")) {
        svg.appendChild(g);
        showTooltip(e, d);
        setActiveDetailCell(g, d);
      }
    });
    g.addEventListener("click", e => {
      if (g.classList.contains("revealed")) {
        svg.appendChild(g);
        showTooltip(e, d);
        setActiveDetailCell(g, d);
      }
    });
    g.addEventListener("mousemove", e => {
      if (g.classList.contains("revealed")) moveTooltip(e);
    });
    g.addEventListener("mouseleave", () => {
      hideTooltip();
    });

    svg.appendChild(g);
  });
}
