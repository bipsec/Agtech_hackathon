const droneDock = document.getElementById("droneDock");
const flyingDrone = document.getElementById("flyingDrone");
const droneRadius = document.getElementById("droneRadius");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");

let isDragging = false;

function updateProgress() {
  const total = allCells.length;
  const revealed = document.querySelectorAll(".subplot-cell.revealed").length;
  const pct = (revealed / total) * 100;
  progressFill.style.width = pct + "%";
  progressText.textContent = `${revealed} / ${total}`;
}

function revealNear(mx, my) {
  const radius = CONFIG.REVEAL_RADIUS;
  const svg = document.getElementById("fieldSvg");
  const ctm = svg.getScreenCTM();
  if (!ctm) return;

  let nearest = null;
  let nearestDist = Infinity;

  allCells.forEach(g => {
    if (g.classList.contains("revealed")) return;
    const d = g._data;
    const svgX = d.center[0];
    const svgY = d.center[1];
    const screenX = ctm.a * svgX + ctm.c * svgY + ctm.e;
    const screenY = ctm.b * svgX + ctm.d * svgY + ctm.f;
    const dist = Math.hypot(mx - screenX, my - screenY);
    if (dist < radius) {
      g.classList.add("revealed");
    }
  });

  allCells.forEach(g => {
    if (!g.classList.contains("revealed")) return;
    const d = g._data;
    const svgX = d.center[0];
    const svgY = d.center[1];
    const screenX = ctm.a * svgX + ctm.c * svgY + ctm.e;
    const screenY = ctm.b * svgX + ctm.d * svgY + ctm.f;
    const dist = Math.hypot(mx - screenX, my - screenY);
    if (dist < radius && dist < nearestDist) {
      nearest = { g, d };
      nearestDist = dist;
    }
  });

  if (nearest && typeof setActiveDetailCell === "function") {
    setActiveDetailCell(nearest.g, nearest.d);
  }
  updateProgress();
}

function moveDrone(x, y) {
  const radius = CONFIG.REVEAL_RADIUS;
  flyingDrone.style.left = (x - 32) + "px";
  flyingDrone.style.top = (y - 32) + "px";
  droneRadius.style.left = (x - radius) + "px";
  droneRadius.style.top = (y - radius) + "px";
}

function resetField() {
  allCells.forEach(g => g.classList.remove("revealed"));
  updateProgress();
  clearDetailPanel();
}

function initDrone() {
  const radius = CONFIG.REVEAL_RADIUS;

  droneDock.addEventListener("mousedown", e => {
    e.preventDefault();
    isDragging = true;
    flyingDrone.classList.add("active");
    droneRadius.classList.add("active");
    droneRadius.style.width = radius * 2 + "px";
    droneRadius.style.height = radius * 2 + "px";
    moveDrone(e.clientX, e.clientY);
    document.body.style.cursor = "none";
  });

  document.addEventListener("mousemove", e => {
    if (!isDragging) return;
    moveDrone(e.clientX, e.clientY);
    revealNear(e.clientX, e.clientY);
  });

  document.addEventListener("mouseup", () => {
    if (!isDragging) return;
    isDragging = false;
    flyingDrone.classList.remove("active");
    droneRadius.classList.remove("active");
    document.body.style.cursor = "";
  });

  document.getElementById("resetBtn").addEventListener("click", resetField);
  updateProgress();
}
