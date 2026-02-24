const tooltip = document.getElementById("tooltip");

function showTooltip(e, d) {
  tooltip.innerHTML = `
    <div><span class="tt-label">Subplot:</span> <span class="tt-value">${d.subPlot}</span></div>
    <div><span class="tt-label">Plot:</span> <span class="tt-value">${d.plotNum}</span></div>
    <div><span class="tt-label">Crop:</span> <span class="tt-value">${d.crop}</span></div>
    <div><span class="tt-label">Treatment:</span> <span class="tt-value">${d.treatment}</span></div>
    <div><span class="tt-label">Position:</span> <span class="tt-value">Col ${d.col}, Row ${d.row}, Sub ${d.sub}</span></div>
    <div><span class="tt-label">Area:</span> <span class="tt-value">${d.area_px2.toLocaleString()} px&sup2;</span></div>
  `;
  tooltip.style.display = "block";
  moveTooltip(e);
}

function moveTooltip(e) {
  let x = e.clientX + 14, y = e.clientY + 14;
  const r = tooltip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 10) x = e.clientX - r.width - 10;
  if (y + r.height > window.innerHeight - 10) y = e.clientY - r.height - 10;
  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";
}

function hideTooltip() {
  tooltip.style.display = "none";
}
