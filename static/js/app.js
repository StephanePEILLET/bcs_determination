const $ = id => document.getElementById(id);

const $dataset=$("sel-dataset"), $group=$("sel-group"), $image=$("sel-image");
const $backend=$("sel-backend"), $samMode=$("sel-sam-mode"), $samField=$("field-sam-mode");
const $backendUp=$("sel-backend-up"), $samModeUp=$("sel-sam-up"), $samFieldUp=$("field-sam-mode-up");
const $chkSeg=$("chk-seg"), $chkBox=$("chk-boxes"), $chkKpt=$("chk-keypoints");
const $btnRun=$("btn-run"), $btnPng=$("btn-export-png"), $btnJson=$("btn-export-json"), $btnSave=$("btn-save");
const $btnImport=$("btn-import-json"), $importFile=$("import-json-file");
const $spinner=$("spinner"), $error=$("error-msg"), $results=$("results");
const $srcImg=$("img-source");
const $ovrBadge=$("overlay-badge"), $gtName=$("gt-name"), $srcBreed=$("source-breed");
const $breedOvr=$("breed-name-ovr"), $confOvr=$("breed-conf-ovr"), $bcsBadgeOvr=$("bcs-badge-ovr");
const $speciesBadgeOvr=$("species-badge-ovr");
const $topk=$("obs-topk"), $segObs=$("obs-seg"), $poseObs=$("obs-pose"), $bcsObs=$("obs-bcs");
const $dropZone=$("drop-zone"), $fileInput=$("file-input"), $dzText=$("dz-text"), $dzPreview=$("dz-preview");
const $canvas=$("editor-canvas"), $canvasWrap=$("canvas-wrap");
const $commentInput=$("comment-input"), $commentAdd=$("comment-add"), $commentsList=$("comments-list"), $commentsPanel=$("comments-panel");
const $hintBar=$("hint-bar"), $histContent=$("history-content"), $toast=$("toast");
const $actionBar=$("action-bar");
const $btnPreload=$("btn-preload"), $btnPreloadStop=$("btn-preload-stop"), $preloadWrap=$("preload-progress-wrap"), $preloadFill=$("preload-bar-fill"), $preloadInfo=$("preload-info");
const ctx = $canvas.getContext("2d");

let datasetsCache = {};
let activeTab = "tab-dataset";
let uploadedFile = null;
let currentRunId = null;

const S = {
  sourceImg: null,
  segImg: null,
  imgW: 0, imgH: 0,
  boxes: [],
  keypoints: [],
  kptConfs: [],
  boxConfs: [],
  comments: [],
  commentNextId: 1,
  showSeg: true,
  showBoxes: true,
  showKpts: true,
  tool: "move",
  drag: null,
  imageName: "",
  imageSize: [],
  groundTruth: null,
  classification: null,
  segmentation: null,
  pose: null,
  maskCanvas: null,
  maskCtx: null,
  maskDirty: false,
  strokeCanvas: null,
  strokeCtx: null,
  brushSize: 30,
  cursorPos: null,
  dirty: false,
};

function markDirty() {
  S.dirty = true;
  if ($btnSave) $btnSave.classList.add("dirty");
}
function clearDirty() {
  S.dirty = false;
  S.maskDirty = false;
  if ($btnSave) $btnSave.classList.remove("dirty");
}
const BRUSH_RGB = "0,200,0";
const BRUSH_ALPHA = 0.4;
const $brushSize = $("brush-size");
const $brushSizeVal = $("brush-size-val");
const $brushSizeWrap = $("brush-size-wrap");

/* ── Tabs ────────────────────────────────────────────── */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === activeTab));
  });
});

/* ── Drop zone ───────────────────────────────────────── */
$dropZone.addEventListener("dragover", e => { e.preventDefault(); $dropZone.classList.add("dragover"); });
$dropZone.addEventListener("dragleave", () => $dropZone.classList.remove("dragover"));
$dropZone.addEventListener("drop", e => { e.preventDefault(); $dropZone.classList.remove("dragover"); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
$fileInput.addEventListener("change", () => { if ($fileInput.files.length) handleFile($fileInput.files[0]); });

function handleFile(f) {
  if (!f.type.startsWith("image/")) return;
  uploadedFile = f;
  $dzText.textContent = f.name;
  $dzPreview.src = URL.createObjectURL(f);
  $dzPreview.style.display = "block";
}

/* ── Helpers ─────────────────────────────────────────── */
function showSpinner(v) { $spinner.classList.toggle("active", v); $btnRun.disabled = v; }
function showError(msg) { $error.textContent = msg; $error.classList.add("active"); }
function hideError() { $error.classList.remove("active"); }
// Color band for a 1–9 BCS: blue (thin) → green (ideal) → red (overweight).
function bcsColor(score) {
  if (score < 4) return "#3b82f6";
  if (score <= 5) return "#10b981";
  return "#ef4444";
}

// Species badge (cascade stage 1). Accepts the top-level `species` dict and the
// classification dict (which may carry `species` after routing).
function renderSpecies(species, cls) {
  if (!$speciesBadgeOvr) return;
  const name = (species && species.species) || (cls && cls.species) || null;
  if (!name) { $speciesBadgeOvr.style.display = "none"; return; }
  const label = name === "dog" ? "Chien" : name === "cat" ? "Chat" : name;
  const conf = species && species.confidence != null
    ? ` ${species.confidence.toFixed(0)}%` : "";
  $speciesBadgeOvr.textContent = `${label}${conf}`;
  $speciesBadgeOvr.style.background = name === "dog" ? "#6366f1" : "#ec4899";
  $speciesBadgeOvr.style.display = "";
}

function renderBcs(bcs) {
  if (bcs && bcs.unavailable_for) {
    const sp = bcs.unavailable_for === "dog" ? "chien" : bcs.unavailable_for;
    if ($bcsObs) $bcsObs.innerHTML = `<div class="obs-kv"><span class="k">BCS</span><span class="v">— (indisponible pour ${sp} : modèle non encore entraîné)</span></div>`;
    if ($bcsBadgeOvr) $bcsBadgeOvr.style.display = "none";
    return;
  }
  if (!bcs || bcs.bcs == null) {
    if ($bcsObs) $bcsObs.innerHTML = '<div class="obs-kv"><span class="k">BCS</span><span class="v">— (modèle indisponible)</span></div>';
    if ($bcsBadgeOvr) $bcsBadgeOvr.style.display = "none";
    return;
  }
  const score = bcs.bcs, color = bcsColor(score), pct = Math.max(1, (score / 9) * 100);
  const masked = bcs.masked ? "silhouette masquée" : "image entière";
  const modelSp = bcs.model_species ? ` · modèle ${bcs.model_species === "cat" ? "chat" : bcs.model_species === "dog" ? "chien" : bcs.model_species}` : "";
  if ($bcsObs) {
    $bcsObs.innerHTML = `
      <div class="obs-kv">
        <span class="k">Score</span><span class="v" style="color:${color};font-weight:700;font-size:1.25em;">${score.toFixed(1)} / 9</span>
        <span class="k">État</span><span class="v" style="color:${color};font-weight:600;">${bcs.category || "—"}</span>
      </div>
      <div class="obs-bar-wrap" style="margin-top:8px;">
        <div class="obs-bar"><div class="obs-bar-fill" style="width:${pct}%;background:${color};"></div></div>
      </div>
      <div class="obs-kv" style="margin-top:8px;">
        <span class="k">Incertitude</span><span class="v">±${(bcs.std != null ? bcs.std : 0).toFixed(2)} (${bcs.num_folds || 0} folds)</span>
        <span class="k">Entrée</span><span class="v">${masked}${modelSp}</span>
      </div>`;
  }
  if ($bcsBadgeOvr) {
    $bcsBadgeOvr.textContent = `BCS ${score.toFixed(1)}/9`;
    $bcsBadgeOvr.style.background = color;
    $bcsBadgeOvr.style.display = "";
  }
}

function showResults(v) {
  $results.classList.toggle("active", v);
  $actionBar.style.display = v ? "" : "none";
  $btnPng.style.display = v ? "inline-block" : "none";
  $btnJson.style.display = v ? "inline-block" : "none";
  $btnImport.style.display = v ? "inline-block" : "none";
  $btnSave.style.display = v ? "inline-block" : "none";
  $hintBar.classList.toggle("active", v);
}
function getBackend() { return activeTab === "tab-upload" ? $backendUp.value : $backend.value; }

// Unified SAM prompt-input selector. Each backend has its own option list and
// its own remembered last-pick, so toggling sam2 ↔ sam3 doesn't lose state.
const SAM_MODES = {
  sam2: [
    { value: "prompted", label: "prompted (centre)" },
    { value: "automatic", label: "automatic (grille)" },
    { value: "pose_prompted", label: "pose_prompted" },
  ],
  sam3: [
    { value: "prompted", label: "prompted (centre)" },
    { value: "pose_prompted", label: "pose_prompted" },
    { value: "concept_prompted", label: "concept_prompted (race)" },
    { value: "pose_concept_prompted", label: "pose_concept_prompted" },
  ],
};
const SAM_MODE_DEFAULTS = { sam2: "pose_prompted", sam3: "pose_concept_prompted" };
const _samModeMemory = { sam2: SAM_MODE_DEFAULTS.sam2, sam3: SAM_MODE_DEFAULTS.sam3 };

function _activeSamSelect() { return activeTab === "tab-upload" ? $samModeUp : $samMode; }

function _populateSamSelect($select, backend) {
  $select.innerHTML = "";
  const opts = SAM_MODES[backend] || [];
  const target = _samModeMemory[backend] || SAM_MODE_DEFAULTS[backend];
  for (const o of opts) {
    const node = document.createElement("option");
    node.value = o.value;
    node.textContent = o.label;
    if (o.value === target) node.selected = true;
    $select.appendChild(node);
  }
}

function updateSamModeField() {
  // For each tab, hide the field on deeplab, repopulate on sam2/sam3.
  const pairs = [
    [$backend.value,   $samMode,   $samField],
    [$backendUp.value, $samModeUp, $samFieldUp],
  ];
  for (const [backend, $select, $field] of pairs) {
    if (backend === "sam2" || backend === "sam3") {
      _populateSamSelect($select, backend);
      $field.style.display = "";
    } else {
      $select.innerHTML = "";
      $field.style.display = "none";
    }
  }
}

function getSamMode() {
  const backend = getBackend();
  if (backend !== "sam2" && backend !== "sam3") return SAM_MODE_DEFAULTS.sam3;
  return _activeSamSelect().value || SAM_MODE_DEFAULTS[backend];
}

// The server still reads sam2_mode / sam3_mode separately. Route the unified
// selector's value to the correct field for the active backend; for the
// inactive one, send its remembered default so the payload is well-formed.
function getSam2() { return getBackend() === "sam2" ? getSamMode() : _samModeMemory.sam2; }
function getSam3() { return getBackend() === "sam3" ? getSamMode() : _samModeMemory.sam3; }

/* ── Tool mode ───────────────────────────────────────── */
function setCursorForTool() {
  if (S.tool === "brush" || S.tool === "eraser") $canvas.style.cursor = "none";
  else $canvas.style.cursor = "default";
}
document.querySelectorAll(".tool-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    S.tool = btn.dataset.tool;
    document.querySelectorAll(".tool-btn").forEach(b => b.classList.toggle("active", b === btn));
    const isPaint = S.tool === "brush" || S.tool === "eraser";
    $brushSizeWrap.style.display = isPaint ? "inline-flex" : "none";
    if (!isPaint) S.cursorPos = null;
    setCursorForTool();
    if (S.tool === "comment") {
      $commentsPanel.style.display = "";
      $commentInput.classList.add("tool-active");
      requestAnimationFrame(() => $commentInput.focus());
    } else {
      $commentsPanel.style.display = "none";
      $commentInput.classList.remove("tool-active");
    }
    render();
  });
});

$brushSize.addEventListener("input", () => {
  S.brushSize = parseInt($brushSize.value, 10) || 30;
  $brushSizeVal.textContent = S.brushSize;
  if (S.cursorPos) render();
});

/* ── Layer toggles ───────────────────────────────────── */
$chkSeg.addEventListener("change", () => { S.showSeg = $chkSeg.checked; render(); });
$chkBox.addEventListener("change", () => { S.showBoxes = $chkBox.checked; render(); });
$chkKpt.addEventListener("change", () => { S.showKpts = $chkKpt.checked; render(); });

/* ── Canvas sizing ───────────────────────────────────── */
function fitCanvas() {
  if (!S.imgW || !S.imgH) return;
  const maxW = $canvasWrap.clientWidth;
  const maxH = 520;
  const ratio = S.imgW / S.imgH;
  let cw, ch;
  if (maxW / maxH > ratio) { ch = maxH; cw = ch * ratio; }
  else { cw = maxW; ch = cw / ratio; }
  $canvas.style.width = cw + "px";
  $canvas.style.height = ch + "px";
}
window.addEventListener("resize", () => { fitCanvas(); });

function eventToImg(e) {
  const r = $canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * ($canvas.width / r.width),
    y: (e.clientY - r.top) * ($canvas.height / r.height),
  };
}
function imgToDisplay(ix, iy) {
  const r = $canvas.getBoundingClientRect();
  return { x: ix * (r.width / $canvas.width), y: iy * (r.height / $canvas.height) };
}

/* ── Rendering ───────────────────────────────────────── */
function render() {
  ctx.clearRect(0, 0, $canvas.width, $canvas.height);
  if (S.sourceImg) ctx.drawImage(S.sourceImg, 0, 0);
  if (S.showSeg) {
    if (S.maskCanvas) ctx.drawImage(S.maskCanvas, 0, 0);
    else if (S.segImg) ctx.drawImage(S.segImg, 0, 0);
    if (S.strokeCanvas) {
      ctx.save();
      ctx.globalAlpha = BRUSH_ALPHA;
      ctx.drawImage(S.strokeCanvas, 0, 0);
      ctx.restore();
    }
  }
  if (S.showBoxes) drawBoxes();
  if (S.showKpts) drawKeypoints();
  drawBrushCursor();
}

function drawBrushCursor() {
  if (!S.cursorPos) return;
  if (S.tool !== "brush" && S.tool !== "eraser") return;
  const r = S.brushSize / 2;
  ctx.save();
  ctx.beginPath();
  ctx.arc(S.cursorPos.x, S.cursorPos.y, r, 0, Math.PI * 2);
  ctx.lineWidth = Math.max(1, $canvas.width * 0.0015);
  ctx.strokeStyle = S.tool === "brush" ? "rgba(0,200,0,0.95)" : "rgba(255,80,80,0.95)";
  ctx.stroke();
  ctx.lineWidth = Math.max(1, $canvas.width * 0.0008);
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.stroke();
  ctx.restore();
}

function ensureStrokeBuffer() {
  if (!S.maskCanvas) return null;
  if (!S.strokeCanvas) {
    const sc = document.createElement("canvas");
    sc.width = S.maskCanvas.width;
    sc.height = S.maskCanvas.height;
    S.strokeCanvas = sc;
    S.strokeCtx = sc.getContext("2d");
  }
  return S.strokeCtx;
}

function commitStroke() {
  if (!S.strokeCanvas || !S.maskCtx) {
    S.strokeCanvas = null;
    S.strokeCtx = null;
    return;
  }
  const mctx = S.maskCtx;
  mctx.save();
  // Clear any prior mask under the stroke so painted pixels end up at exactly
  // BRUSH_ALPHA opacity instead of compounding with the existing overlay.
  mctx.globalCompositeOperation = "destination-out";
  mctx.drawImage(S.strokeCanvas, 0, 0);
  // Lay down the stroke at the canonical mask alpha.
  mctx.globalCompositeOperation = "source-over";
  mctx.globalAlpha = BRUSH_ALPHA;
  mctx.drawImage(S.strokeCanvas, 0, 0);
  mctx.restore();
  S.strokeCanvas = null;
  S.strokeCtx = null;
}

function paintAt(ix, iy) {
  if (!S.maskCtx) return;
  const r = S.brushSize / 2;
  if (S.tool === "brush") {
    const sctx = ensureStrokeBuffer();
    if (!sctx) return;
    sctx.save();
    sctx.globalCompositeOperation = "source-over";
    sctx.fillStyle = `rgba(${BRUSH_RGB},1)`;
    sctx.beginPath();
    sctx.arc(ix, iy, r, 0, Math.PI * 2);
    sctx.fill();
    sctx.restore();
  } else {
    const mctx = S.maskCtx;
    mctx.save();
    mctx.globalCompositeOperation = "destination-out";
    mctx.fillStyle = "rgba(0,0,0,1)";
    mctx.beginPath();
    mctx.arc(ix, iy, r, 0, Math.PI * 2);
    mctx.fill();
    mctx.restore();
  }
  S.maskDirty = true;
  markDirty();
}

function paintLine(x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0;
  const dist = Math.hypot(dx, dy);
  const step = Math.max(1, S.brushSize / 4);
  const n = Math.max(1, Math.ceil(dist / step));
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    paintAt(x0 + dx * t, y0 + dy * t);
  }
}

function scaledStroke() { return Math.max(2, $canvas.width * 0.004); }
function handleSize() { return Math.max(4, $canvas.width * 0.008); }
function kptRadius() { return Math.max(4, $canvas.width * 0.007); }
function hitR() { return Math.max(8, $canvas.width * 0.015); }
function fontSize() { return Math.max(11, $canvas.width * 0.02); }

function drawBoxes() {
  const lw = scaledStroke(), hs = handleSize(), fs = fontSize();
  for (let n = 0; n < S.boxes.length; n++) {
    const [x1, y1, x2, y2] = S.boxes[n];
    ctx.strokeStyle = "#32ff32";
    ctx.lineWidth = lw;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (S.boxConfs[n] !== undefined) {
      ctx.font = "bold " + fs + "px sans-serif";
      const label = (S.boxConfs[n] * 100).toFixed(0) + "%";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(x1, y1 - fs - 4, tw + 8, fs + 4);
      ctx.fillStyle = "#32ff32";
      ctx.fillText(label, x1 + 4, y1 - 4);
    }

    ctx.fillStyle = "#32ff32";
    const corners = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]];
    for (const [cx, cy] of corners) ctx.fillRect(cx - hs/2, cy - hs/2, hs, hs);
  }
}

function drawKeypoints() {
  const r = kptRadius(), lw = Math.max(1, $canvas.width * 0.002);
  for (let n = 0; n < S.keypoints.length; n++) {
    for (let k = 0; k < S.keypoints[n].length; k++) {
      if (S.kptConfs[n] && S.kptConfs[n][k] < 0.3) continue;
      const [kx, ky] = S.keypoints[n][k];
      ctx.beginPath();
      ctx.arc(kx, ky, r, 0, Math.PI * 2);
      ctx.fillStyle = "#ff00ff";
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = lw;
      ctx.stroke();
    }
  }
}

/* ── Hit testing ─────────────────────────────────────── */
function dist(ax, ay, bx, by) { return Math.hypot(ax - bx, ay - by); }

function findKptAt(ix, iy) {
  const hr = hitR();
  for (let n = 0; n < S.keypoints.length; n++)
    for (let k = 0; k < S.keypoints[n].length; k++) {
      if (S.kptConfs[n] && S.kptConfs[n][k] < 0.3) continue;
      if (dist(ix, iy, S.keypoints[n][k][0], S.keypoints[n][k][1]) < hr) return { det: n, kpt: k };
    }
  return null;
}

function findCornerAt(ix, iy) {
  const hr = hitR();
  for (let n = 0; n < S.boxes.length; n++) {
    const [x1, y1, x2, y2] = S.boxes[n];
    const cs = [[x1,y1,0],[x2,y1,1],[x2,y2,2],[x1,y2,3]];
    for (const [cx, cy, idx] of cs) if (dist(ix, iy, cx, cy) < hr) return { det: n, corner: idx };
  }
  return null;
}

function findBoxBodyAt(ix, iy) {
  for (let n = 0; n < S.boxes.length; n++) {
    const [x1, y1, x2, y2] = S.boxes[n];
    if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) return n;
  }
  return -1;
}

/* ── Canvas interaction ──────────────────────────────── */
$canvas.addEventListener("mousedown", onDown);
$canvas.addEventListener("mousemove", onMove);
$canvas.addEventListener("mouseup", onUp);
$canvas.addEventListener("mouseleave", onLeave);

$canvas.addEventListener("touchstart", e => { e.preventDefault(); onDown(touchToMouse(e)); }, { passive: false });
$canvas.addEventListener("touchmove", e => { e.preventDefault(); onMove(touchToMouse(e)); }, { passive: false });
$canvas.addEventListener("touchend", e => { onUp(); });

function touchToMouse(e) {
  const t = e.touches[0] || e.changedTouches[0];
  return { clientX: t.clientX, clientY: t.clientY };
}

function onDown(e) {
  if (!S.sourceImg) return;
  if (S.tool === "comment") return;
  const p = eventToImg(e);

  if (S.tool === "brush" || S.tool === "eraser") {
    paintAt(p.x, p.y);
    S.drag = { type: "paint", lx: p.x, ly: p.y };
    S.cursorPos = p;
    render();
    return;
  }

  const kpt = findKptAt(p.x, p.y);
  if (kpt) {
    S.drag = { type: "kpt", det: kpt.det, kpt: kpt.kpt };
    return;
  }
  const crn = findCornerAt(p.x, p.y);
  if (crn) {
    S.drag = {
      type: "corner", det: crn.det, corner: crn.corner,
      origBox: [...S.boxes[crn.det]],
      origKpts: S.keypoints[crn.det].map(k => [...k]),
    };
    return;
  }
  const bi = findBoxBodyAt(p.x, p.y);
  if (bi >= 0) {
    S.drag = {
      type: "box", det: bi, sx: p.x, sy: p.y,
      origBox: [...S.boxes[bi]],
      origKpts: S.keypoints[bi].map(k => [...k]),
    };
  }
}

function onMove(e) {
  if (!S.sourceImg) return;
  const p = eventToImg(e);

  if (S.tool === "brush" || S.tool === "eraser") {
    S.cursorPos = p;
    if (S.drag && S.drag.type === "paint") {
      paintLine(S.drag.lx, S.drag.ly, p.x, p.y);
      S.drag.lx = p.x; S.drag.ly = p.y;
    }
    render();
    return;
  }

  if (!S.drag) {
    const hit = findKptAt(p.x, p.y) || findCornerAt(p.x, p.y) || (findBoxBodyAt(p.x, p.y) >= 0);
    $canvas.style.cursor = hit ? "grab" : "default";
    return;
  }

  $canvas.style.cursor = "grabbing";

  if (S.drag.type === "kpt") {
    S.keypoints[S.drag.det][S.drag.kpt] = [p.x, p.y];
    markDirty();
  } else if (S.drag.type === "box") {
    const dx = p.x - S.drag.sx, dy = p.y - S.drag.sy;
    const ob = S.drag.origBox;
    S.boxes[S.drag.det] = [ob[0]+dx, ob[1]+dy, ob[2]+dx, ob[3]+dy];
    markDirty();
  } else if (S.drag.type === "corner") {
    const ob = S.drag.origBox;
    let nb = [...ob];
    if (S.drag.corner === 0) nb = [p.x, p.y, ob[2], ob[3]];
    else if (S.drag.corner === 1) nb = [ob[0], p.y, p.x, ob[3]];
    else if (S.drag.corner === 2) nb = [ob[0], ob[1], p.x, p.y];
    else nb = [p.x, ob[1], ob[2], p.y];
    nb = nb.map((v, i) => Math.max(0, Math.min(v, i % 2 === 0 ? S.imgW : S.imgH)));
    S.boxes[S.drag.det] = nb;
    markDirty();
  }
  render();
}

function onUp() {
  if (S.tool === "brush" && S.strokeCanvas) {
    commitStroke();
    render();
  }
  S.drag = null;
  setCursorForTool();
}

function onLeave() {
  if (S.tool === "brush" && S.strokeCanvas) commitStroke();
  S.drag = null;
  S.cursorPos = null;
  setCursorForTool();
  if (S.tool === "brush" || S.tool === "eraser") render();
}

/* ── Comments panel ──────────────────────────────────── */
function addCommentFromInput() {
  const txt = $commentInput.value.trim();
  if (!txt) return;
  S.comments.push({ id: S.commentNextId++, text: txt });
  $commentInput.value = "";
  markDirty();
  renderCommentsList();
}

function deleteComment(id) {
  const i = S.comments.findIndex(c => c.id === id);
  if (i < 0) return;
  S.comments.splice(i, 1);
  markDirty();
  renderCommentsList();
}

function renderCommentsList() {
  if (!$commentsList) return;
  if (!S.comments.length) {
    $commentsList.innerHTML = '<div class="comments-empty">Aucun commentaire pour cette inf&eacute;rence.</div>';
    return;
  }
  $commentsList.innerHTML = "";
  for (const c of S.comments) {
    const row = document.createElement("div");
    row.className = "comment-item";
    const txt = document.createElement("div");
    txt.className = "ci-text";
    txt.textContent = c.text;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ci-del";
    del.title = "Supprimer";
    del.textContent = "×";
    del.addEventListener("click", () => deleteComment(c.id));
    row.appendChild(txt);
    row.appendChild(del);
    $commentsList.appendChild(row);
  }
}

$commentInput.addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); addCommentFromInput(); }
});
$commentAdd.addEventListener("click", addCommentFromInput);

/* ── Datasets ────────────────────────────────────────── */
async function loadDatasets() {
  const res = await fetch("/api/datasets");
  datasetsCache = await res.json();
  $dataset.innerHTML = "";
  for (const [name, info] of Object.entries(datasetsCache)) {
    const o = document.createElement("option"); o.value = name;
    o.textContent = `${name}  (${info.total_images} images)`;
    $dataset.appendChild(o);
  }
  onDatasetChange();
}
function onDatasetChange() {
  const ds = datasetsCache[$dataset.value]; if (!ds) return;
  $group.innerHTML = "";
  for (const [g, c] of Object.entries(ds.groups)) { const o = document.createElement("option"); o.value = g; o.textContent = `${g}  (${c})`; $group.appendChild(o); }
  $group.disabled = false; onGroupChange();
}
async function onGroupChange() {
  const ds = $dataset.value, gr = $group.value;
  if (!ds || !gr) { $image.innerHTML = "<option>\u2014</option>"; $image.disabled = true; return; }
  const res = await fetch(`/api/images?dataset=${encodeURIComponent(ds)}&group=${encodeURIComponent(gr)}`);
  const files = await res.json();
  $image.innerHTML = "";
  for (const f of files) {
    const o = document.createElement("option");
    o.value = f.name;
    o.textContent = f.annotated ? "\u2705 " + f.name : f.name;
    if (f.annotated) o.classList.add("annotated");
    $image.appendChild(o);
  }
  $image.disabled = !files.length;
}

/* ── Run inference ───────────────────────────────────── */
async function runInference() {
  hideError(); showResults(false); showSpinner(true);
  try {
    if (activeTab === "tab-upload") { if (!uploadedFile) { showError("Veuillez s\u00e9lectionner une image."); return; } await runUpload(); }
    else { await runDataset(); }
  } catch (e) { showError("Erreur : " + e.message); }
  finally { showSpinner(false); }
}

async function runDataset() {
  const r = await fetch("/api/inference", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      dataset: $dataset.value, group: $group.value, filename: $image.value,
      seg_backend: getBackend(), sam2_mode: getSam2(), sam3_mode: getSam3(),
    }),
  });
  const d = await r.json();
  if (!r.ok) { showError(d.error || "Erreur serveur"); return; }
  displayResults(d);
  loadHistory();
}

async function runUpload() {
  const fd = new FormData();
  fd.append("file", uploadedFile);
  fd.append("seg_backend", getBackend());
  fd.append("sam2_mode", getSam2());
  fd.append("sam3_mode", getSam3());
  const r = await fetch("/api/inference/upload", { method: "POST", body: fd });
  const d = await r.json();
  if (!r.ok) { showError(d.error || "Erreur serveur"); return; }
  displayResults(d);
  loadHistory();
}

/* ── Display results ─────────────────────────────────── */
function displayResults(d) {
  currentRunId = d.run_id || null;
  clearDirty();
  $srcImg.src = "data:image/jpeg;base64," + d.source_b64;

  S.comments = d.user_comments
    ? d.user_comments.map(c => ({ id: c.id, text: c.text }))
    : [];
  S.commentNextId = S.comments.length ? Math.max(...S.comments.map(c => c.id || 0)) + 1 : 1;
  renderCommentsList();
  S.imageName = d.image_name || "";
  S.imageSize = d.image_size;
  S.groundTruth = d.ground_truth || null;
  S.classification = d.classification;
  S.segmentation = d.segmentation;
  S.pose = d.pose;
  S.bcs = d.bcs || null;

  const ann = d.pose_annotations || {};
  S.boxes = (ann.boxes || []).map(b => [...b]);
  S.keypoints = (ann.keypoints || []).map(det => det.map(k => [...k]));
  S.kptConfs = (ann.kpt_confs || []).map(det => [...det]);
  S.boxConfs = [...(ann.box_confs || [])];

  const imgW = d.image_size[0], imgH = d.image_size[1];
  S.imgW = imgW; S.imgH = imgH;
  $canvas.width = imgW; $canvas.height = imgH;

  $chkSeg.checked = true; $chkBox.checked = true; $chkKpt.checked = true;
  S.showSeg = true; S.showBoxes = true; S.showKpts = true;

  if (d.ground_truth) { $gtName.textContent = d.ground_truth; $srcBreed.style.display = ""; }
  else { $gtName.textContent = ""; $srcBreed.style.display = "none"; }

  const cls = d.classification;
  $breedOvr.textContent = cls.class_name || "\u2014";
  $confOvr.textContent = cls.confidence.toFixed(1) + " %";
  renderSpecies(d.species, cls);
  $ovrBadge.textContent = d.segmentation.backend;

  $topk.innerHTML = "";
  for (const e of cls.top_k) {
    const isTop = e.rank === 1;
    const row = document.createElement("div"); row.className = "obs-row";
    row.innerHTML = `
      <div class="obs-label">
        <span class="obs-rank ${isTop ? 'r1' : 'rn'}">${e.rank}</span>
        <span>${e.class_name || "\u2014"}</span>
      </div>
      <span class="obs-val">${e.confidence.toFixed(1)}%</span>
      <div class="obs-bar-wrap">
        <div class="obs-bar"><div class="obs-bar-fill" style="width:${Math.max(1, e.confidence)}%;background:${isTop ? '#3b82f6' : '#93c5fd'};"></div></div>
      </div>`;
    $topk.appendChild(row);
  }

  const seg = d.segmentation;
  const segColors = {0:"#10b981", 1:"#94a3b8", 2:"#f59e0b"};
  const segLabels = {0:"Foreground", 1:"Background", 2:"Border"};
  let sh = `<div class="obs-kv"><span class="k">Backend</span><span class="v">${seg.backend}</span></div>`;
  sh += '<div style="margin-top:8px;">';
  for (const s of seg.distribution) {
    const c = segColors[s.class] || "#94a3b8";
    sh += `<div class="obs-row">
      <div class="obs-label"><span style="color:${c};font-weight:600;">\u25CF</span> ${segLabels[s.class]||"cls"+s.class}</div>
      <span class="obs-val">${s.pct}%</span>
      <div class="obs-bar-wrap"><div class="obs-bar"><div class="obs-bar-fill" style="width:${Math.max(1, s.pct)}%;background:${c};"></div></div></div>
    </div>`;
  }
  sh += '</div>';
  $segObs.innerHTML = sh;

  const pose = d.pose;
  let ph = '<div class="obs-kv">';
  ph += `<span class="k">D\u00e9tections</span><span class="v">${pose.num_detections}</span>`;
  if (pose.best_conf !== null) ph += `<span class="k">Meilleure conf.</span><span class="v">${pose.best_conf.toFixed(2)}</span>`;
  ph += '</div>';
  $poseObs.innerHTML = ph;

  renderBcs(d.bcs);

  showResults(true);

  S.maskCanvas = null;
  S.maskCtx = null;
  S.maskDirty = false;
  S.strokeCanvas = null;
  S.strokeCtx = null;
  S.cursorPos = null;

  requestAnimationFrame(() => {
    fitCanvas();
    const sourceImg = new Image();
    sourceImg.onload = () => {
      S.sourceImg = sourceImg;
      const segImg = new Image();
      segImg.onload = () => {
        S.segImg = segImg;
        const mc = document.createElement("canvas");
        mc.width = S.imgW; mc.height = S.imgH;
        const mctx = mc.getContext("2d");
        const editedSrc = d.mask_b64 ? d.mask_b64 : null;
        if (editedSrc) {
          const editedImg = new Image();
          editedImg.onload = () => {
            mctx.drawImage(editedImg, 0, 0);
            S.maskCanvas = mc; S.maskCtx = mctx;
            render();
          };
          editedImg.src = "data:image/png;base64," + editedSrc;
        } else {
          mctx.drawImage(segImg, 0, 0);
          S.maskCanvas = mc; S.maskCtx = mctx;
        }
        render();
      };
      segImg.src = "data:image/png;base64," + d.seg_b64;
      render();
    };
    sourceImg.src = "data:image/jpeg;base64," + d.source_b64;
  });
}

/* ── Export PNG ───────────────────────────────────────── */
function exportPng() {
  if (!S.sourceImg) return;
  const tc = document.createElement("canvas");
  tc.width = S.imgW; tc.height = S.imgH;
  const tctx = tc.getContext("2d");
  tctx.drawImage(S.sourceImg, 0, 0);
  if (S.maskCanvas) tctx.drawImage(S.maskCanvas, 0, 0);
  else if (S.segImg) tctx.drawImage(S.segImg, 0, 0);

  const lw = Math.max(2, S.imgW * 0.004), hs = Math.max(4, S.imgW * 0.008);
  const fs = Math.max(11, S.imgW * 0.02), kr = Math.max(4, S.imgW * 0.007);

  if (S.showBoxes) {
    for (let n = 0; n < S.boxes.length; n++) {
      const [x1,y1,x2,y2] = S.boxes[n];
      tctx.strokeStyle = "#32ff32"; tctx.lineWidth = lw;
      tctx.strokeRect(x1, y1, x2-x1, y2-y1);
      tctx.fillStyle = "#32ff32";
      for (const [cx,cy] of [[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        tctx.fillRect(cx-hs/2, cy-hs/2, hs, hs);
    }
  }
  if (S.showKpts) {
    for (let n = 0; n < S.keypoints.length; n++)
      for (let k = 0; k < S.keypoints[n].length; k++) {
        if (S.kptConfs[n] && S.kptConfs[n][k] < 0.3) continue;
        const [kx,ky] = S.keypoints[n][k];
        tctx.beginPath(); tctx.arc(kx, ky, kr, 0, Math.PI*2);
        tctx.fillStyle = "#ff00ff"; tctx.fill();
      }
  }

  const a = document.createElement("a");
  a.href = tc.toDataURL("image/png");
  a.download = (S.imageName || "overlay") + "_annotated.png";
  a.click();
}

/* ── Export JSON ──────────────────────────────────────── */
function exportJson() {
  if (!S.sourceImg) { toast("Lancez d'abord une inférence", "err"); return; }
  const payload = {
    exported_at: new Date().toISOString(),
    run_id: currentRunId,
    image_name: S.imageName,
    image_size: S.imageSize,
    ground_truth: S.groundTruth,
    classification: S.classification,
    segmentation: S.segmentation,
    pose: S.pose,
    bcs: S.bcs,
    pose_annotations: {
      boxes: S.boxes,
      keypoints: S.keypoints,
      kpt_confs: S.kptConfs,
      box_confs: S.boxConfs,
    },
    comments: S.comments,
  };
  // Embed the (possibly edited) segmentation mask so the export round-trips
  // the user's brush corrections — not just keypoint/box/comment edits.
  if (S.maskCanvas) {
    payload.mask_b64 = S.maskCanvas.toDataURL("image/png").split(",", 2)[1];
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (S.imageName || "overlay") + "_annotations.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(a.href);
  const nb = S.boxes.length, nk = S.keypoints.reduce((s, d) => s + d.length, 0);
  const nc = S.comments.length, nm = S.maskCanvas ? 1 : 0;
  toast(`Export OK — ${nb} box, ${nk} kpts, ${nc} comm., ${nm ? "+ masque" : "pas de masque"}`);
}

/* ── Events ──────────────────────────────────────────── */
$dataset.addEventListener("change", onDatasetChange);
$group.addEventListener("change", onGroupChange);
$backend.addEventListener("change", updateSamModeField);
$backendUp.addEventListener("change", updateSamModeField);
// Remember each backend's last-picked mode so it survives a backend toggle.
$samMode.addEventListener("change", () => { _samModeMemory[$backend.value] = $samMode.value; });
$samModeUp.addEventListener("change", () => { _samModeMemory[$backendUp.value] = $samModeUp.value; });
$btnRun.addEventListener("click", runInference);
$btnPng.addEventListener("click", exportPng);
$btnJson.addEventListener("click", exportJson);
$btnSave.addEventListener("click", saveToDb);
$btnImport.addEventListener("click", importJson);

/* ── Toast ───────────────────────────────────────────── */
let _toastTimer = null;
function toast(msg, type) {
  $toast.textContent = msg;
  $toast.className = "toast " + (type || "ok") + " show";
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { $toast.classList.remove("show"); }, 2500);
}

/* ── Save to DB ──────────────────────────────────────── */
async function saveToDb() {
  if (!currentRunId) { toast("Aucun run a sauvegarder", "err"); return; }
  $btnSave.disabled = true;
  try {
    const payload = {
      boxes: S.boxes,
      keypoints: S.keypoints,
      kpt_confs: S.kptConfs,
      box_confs: S.boxConfs,
      comments: S.comments,
    };
    if (S.maskDirty && S.maskCanvas) {
      payload.mask_b64 = S.maskCanvas.toDataURL("image/png").split(",", 2)[1];
    }
    const r = await fetch(`/api/history/${currentRunId}/annotations`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (!r.ok) { toast(d.error || "Erreur", "err"); return; }
    clearDirty();
    const nc = S.comments.length;
    toast(`Sauvegarde OK (run #${currentRunId} — ${nc} commentaire${nc > 1 ? "s" : ""})`);
    loadHistory();
    onGroupChange();
  } catch (e) { toast("Erreur : " + e.message, "err"); }
  finally { $btnSave.disabled = false; }
}

/* ── Import JSON annotations ─────────────────────────── */
function importJson() {
  if (!S.sourceImg) { toast("Lancez d'abord une inférence", "err"); return; }
  $importFile.value = "";
  $importFile.click();
}

$importFile?.addEventListener("change", async () => {
  const file = $importFile.files && $importFile.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    applyImportedAnnotations(data);
    toast("Annotations importées");
  } catch (e) {
    toast("Fichier JSON invalide : " + e.message, "err");
  }
});

function applyImportedAnnotations(data) {
  const ann = data.pose_annotations || {};
  if (Array.isArray(ann.boxes)) S.boxes = ann.boxes.map(b => [...b]);
  if (Array.isArray(ann.keypoints)) S.keypoints = ann.keypoints.map(d => d.map(k => [...k]));
  if (Array.isArray(ann.kpt_confs)) S.kptConfs = ann.kpt_confs.map(d => [...d]);
  if (Array.isArray(ann.box_confs)) S.boxConfs = [...ann.box_confs];
  const cs = data.comments || data.user_comments;
  if (Array.isArray(cs)) {
    S.comments = cs.map(c => ({ id: c.id || 0, x: c.x || 0, y: c.y || 0, text: c.text }));
    S.commentNextId = S.comments.length ? Math.max(...S.comments.map(c => c.id || 0)) + 1 : 1;
    renderCommentsList();
  }
  if (data.mask_b64 && S.maskCanvas) {
    const img = new Image();
    img.onload = () => {
      const mctx = S.maskCtx;
      mctx.clearRect(0, 0, S.maskCanvas.width, S.maskCanvas.height);
      mctx.drawImage(img, 0, 0, S.maskCanvas.width, S.maskCanvas.height);
      S.maskDirty = true;
      render();
    };
    img.src = "data:image/png;base64," + data.mask_b64;
  }
  markDirty();
  render();
}

/* ── History ─────────────────────────────────────────── */
let _historyRuns = [];
let _historyTotal = 0;
let _historyPage = 1;
let _historyPageSize = 50;
let _historySort = "last_inferred_at";
let _historyOrder = "desc";
const _historySelected = new Set();

const HIST_COLUMNS = [
  { key: "id", label: "#" },
  { key: "last_inferred_at", label: "Date" },
  { key: "image_name", label: "Image" },
  { key: "predicted_species", label: "Esp&egrave;ce" },
  { key: "predicted_class", label: "Race pr&eacute;dite" },
  { key: "predicted_confidence", label: "Conf." },
  { key: "predicted_bcs", label: "BCS" },
  { key: "seg_backend", label: "Backend" },
  { key: "has_annotations", label: "Ann." },
];

function _toggleSort(col) {
  if (_historySort === col) {
    _historyOrder = _historyOrder === "asc" ? "desc" : "asc";
  } else {
    _historySort = col;
    _historyOrder = (col === "image_name" || col === "seg_backend" || col === "predicted_class" || col === "predicted_species") ? "asc" : "desc";
  }
  _historyPage = 1;
  loadHistory();
}

async function loadHistory() {
  try {
    const offset = (_historyPage - 1) * _historyPageSize;
    const url = `/api/history?limit=${_historyPageSize}&offset=${offset}`
      + `&sort=${encodeURIComponent(_historySort)}&order=${encodeURIComponent(_historyOrder)}`;
    const r = await fetch(url);
    const data = await r.json();
    _historyRuns = data.runs || [];
    _historyTotal = data.total || 0;
    const ids = new Set(_historyRuns.map(x => x.id));
    for (const id of [..._historySelected]) if (!ids.has(id)) _historySelected.delete(id);
    renderHistory();
  } catch (e) { $histContent.innerHTML = '<div class="history-empty">Erreur de chargement.</div>'; }
}

function renderHistory() {
  if (!_historyRuns.length && _historyPage === 1) {
    $histContent.innerHTML = '<div class="history-empty">Aucune inference enregistree.</div>';
    return;
  }
  const total = _historyTotal, sel = _historySelected.size;
  const shown = _historyRuns.length;
  const allSelected = sel === shown && shown > 0;
  const totalPages = Math.max(1, Math.ceil(total / _historyPageSize));
  let html = `
    <div class="history-toolbar">
      <span class="ht-info"><b>${total}</b> run${total>1?"s":""} · page <b>${_historyPage}</b>/${totalPages} · <b>${sel}</b> s&eacute;lectionn&eacute;${sel>1?"s":""}</span>
      <button class="ht-link" id="ht-select-all">Tout s&eacute;lectionner</button>
      <button class="ht-link" id="ht-select-none">Aucun</button>
      <span class="ht-spacer"></span>
      <button class="ht-bulk-del" id="ht-bulk-del" ${sel===0?"disabled":""}>&#128465;&ensp;Supprimer ${sel>0?`(${sel})`:""}</button>
    </div>
    <div class="history-table-wrap" id="history-table-wrap">
    <table class="history-table"><thead><tr>
      <th style="width:32px;"><input type="checkbox" class="select-all" id="ht-checkall" ${allSelected?"checked":""}></th>`;
  for (const col of HIST_COLUMNS) {
    const active = _historySort === col.key;
    const ind = active ? (_historyOrder === "asc" ? "&#9650;" : "&#9660;") : "&#9651;";
    html += `<th class="sortable${active?" active":""}" data-sort="${col.key}">${col.label}<span class="sort-ind">${ind}</span></th>`;
  }
  html += `<th>Actions</th>
    </tr></thead><tbody>`;
  for (const r of _historyRuns) {
    const dtRaw = r.last_inferred_at || r.created_at;
    const dt = dtRaw ? new Date(dtRaw).toLocaleString("fr-FR") : "—";
    const annBadge = r.has_annotations
      ? '<span class="badge-ann yes">&#10003;</span>'
      : '<span class="badge-ann no">&mdash;</span>';
    const isSel = _historySelected.has(r.id);
    html += `<tr class="${isSel?"selected":""}" data-id="${r.id}">
      <td><input type="checkbox" class="row-select" data-id="${r.id}" ${isSel?"checked":""}></td>
      <td>${r.id}</td>
      <td>${dt}</td>
      <td>${r.image_name || "—"}</td>
      <td>${r.predicted_species ? (r.predicted_species === "dog" ? "Chien" : r.predicted_species === "cat" ? "Chat" : r.predicted_species) : "—"}</td>
      <td>${r.predicted_class || "—"}</td>
      <td>${r.predicted_confidence != null ? r.predicted_confidence.toFixed(1) + "%" : "—"}</td>
      <td>${r.predicted_bcs != null ? '<b style="color:'+bcsColor(r.predicted_bcs)+';">'+r.predicted_bcs.toFixed(1)+'</b>' : "—"}</td>
      <td>${r.seg_backend}</td>
      <td>${annBadge}</td>
      <td>
        <button class="btn-sm" onclick="loadRun(${r.id})">Charger</button>
        <button class="btn-sm btn-del" onclick="deleteRun(${r.id})">Supprimer</button>
      </td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  html += '<div class="pagination">';
  html += `<button class="pg-btn" id="pg-first" ${_historyPage<=1?"disabled":""}>&laquo; D&eacute;but</button>`;
  html += `<button class="pg-btn" id="pg-prev" ${_historyPage<=1?"disabled":""}>&lsaquo; Pr&eacute;c.</button>`;

  const maxButtons = 5;
  let startPage = Math.max(1, _historyPage - Math.floor(maxButtons / 2));
  let endPage = Math.min(totalPages, startPage + maxButtons - 1);
  if (endPage - startPage + 1 < maxButtons) startPage = Math.max(1, endPage - maxButtons + 1);

  for (let p = startPage; p <= endPage; p++) {
    html += `<button class="pg-btn${p===_historyPage?" pg-active":""}" data-page="${p}">${p}</button>`;
  }

  html += `<button class="pg-btn" id="pg-next" ${_historyPage>=totalPages?"disabled":""}>Suiv. &rsaquo;</button>`;
  html += `<button class="pg-btn" id="pg-last" ${_historyPage>=totalPages?"disabled":""}>Fin &raquo;</button>`;
  html += `<span class="pg-info">${(_historyPage-1)*_historyPageSize+1}&ndash;${Math.min(_historyPage*_historyPageSize, total)} / ${total}</span>`;
  html += '</div>';

  $histContent.innerHTML = html;

  $("ht-select-all").addEventListener("click", () => {
    _historyRuns.forEach(r => _historySelected.add(r.id));
    renderHistory();
  });
  $("ht-select-none").addEventListener("click", () => {
    _historySelected.clear();
    renderHistory();
  });
  $("ht-checkall").addEventListener("change", e => {
    if (e.target.checked) _historyRuns.forEach(r => _historySelected.add(r.id));
    else _historySelected.clear();
    renderHistory();
  });
  $("ht-bulk-del").addEventListener("click", bulkDeleteSelected);
  $histContent.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => _toggleSort(th.dataset.sort));
  });
  $histContent.querySelectorAll("input.row-select").forEach(cb => {
    cb.addEventListener("change", e => {
      const id = parseInt(e.target.dataset.id, 10);
      if (e.target.checked) _historySelected.add(id);
      else _historySelected.delete(id);
      renderHistory();
    });
  });

  $("pg-first")?.addEventListener("click", () => { _historyPage = 1; loadHistory(); });
  $("pg-prev")?.addEventListener("click", () => { if (_historyPage > 1) { _historyPage--; loadHistory(); } });
  $("pg-next")?.addEventListener("click", () => { if (_historyPage < totalPages) { _historyPage++; loadHistory(); } });
  $("pg-last")?.addEventListener("click", () => { _historyPage = totalPages; loadHistory(); });
  $histContent.querySelectorAll(".pg-btn[data-page]").forEach(btn => {
    btn.addEventListener("click", () => { _historyPage = parseInt(btn.dataset.page, 10); loadHistory(); });
  });
}

async function bulkDeleteSelected() {
  const ids = [..._historySelected];
  if (!ids.length) return;
  if (!confirm(`Supprimer ${ids.length} run${ids.length>1?"s":""} de l'historique ?`)) return;
  try {
    const results = await Promise.allSettled(
      ids.map(id => fetch(`/api/history/${id}`, { method: "DELETE" }))
    );
    const ok = results.filter(r => r.status === "fulfilled" && r.value.ok).length;
    const ko = results.length - ok;
    _historySelected.clear();
    if (ko === 0) toast(`${ok} run${ok>1?"s":""} supprimé${ok>1?"s":""}`);
    else toast(`${ok} supprimé${ok>1?"s":""}, ${ko} échoué${ko>1?"s":""}`, "err");
    loadHistory();
  } catch (e) { toast("Erreur : " + e.message, "err"); }
}

async function loadRun(id) {
  try {
    const r = await fetch(`/api/history/${id}`);
    const d = await r.json();
    if (!r.ok) { toast(d.error || "Erreur", "err"); return; }
    displayResults(d);
    toast("Run #" + id + " charge");
  } catch (e) { toast("Erreur : " + e.message, "err"); }
}

async function deleteRun(id) {
  if (!confirm("Supprimer le run #" + id + " ?")) return;
  try {
    await fetch(`/api/history/${id}`, { method: "DELETE" });
    _historySelected.delete(id);
    toast("Run #" + id + " supprime");
    loadHistory();
  } catch (e) { toast("Erreur : " + e.message, "err"); }
}

/* ── Preload DB ──────────────────────────────────────── */
// Preload always targets the SAM 3 backend (the canonical dataset bake).
// The inference dropdown stays free for one-off interactive runs.
const PRELOAD_BACKEND = "sam3";
let _preloadPollTimer = null;
let _preloadComplete = false;

async function checkPreloadStatus() {
  try {
    const r = await fetch(`/api/preload/status?seg_backend=${PRELOAD_BACKEND}`);
    const d = await r.json();
    if (d.running) {
      _preloadComplete = false;
      $btnPreload.classList.remove("is-complete");
      $btnPreload.style.display = "none";
      $btnPreloadStop.style.display = "inline-block";
      $preloadWrap.classList.add("active");
      const prog = d.progress || {};
      const total = prog.total || d.remaining || 0;
      const done = prog.processed || 0;
      const pct = total > 0 ? Math.round(done / total * 100) : 0;
      $preloadFill.style.width = pct + "%";
      $preloadInfo.textContent = prog.message || `${done}/${total}`;
      if (!_preloadPollTimer) {
        _preloadPollTimer = setInterval(checkPreloadStatus, 1500);
      }
    } else {
      if (_preloadPollTimer) {
        clearInterval(_preloadPollTimer);
        _preloadPollTimer = null;
      }
      $preloadWrap.classList.remove("active");
      $preloadFill.style.width = "0%";
      $preloadInfo.textContent = "";
      $btnPreloadStop.style.display = "none";
      $btnPreload.style.display = "inline-block";
      _preloadComplete = !!d.complete;
      $btnPreload.classList.toggle("is-complete", _preloadComplete);
      $btnPreload.disabled = false;
      $btnPreload.title = _preloadComplete
        ? "Base SAM 3 d\u00e9j\u00e0 int\u00e9gralement charg\u00e9e"
        : "Pr\u00e9-charger les inf\u00e9rences SAM 3 pour toutes les images des datasets";
    }
  } catch (e) {
    $btnPreload.disabled = true;
  }
}

async function startPreload() {
  const sam3Mode = getSam3();
  try {
    const r = await fetch("/api/preload/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ seg_backend: PRELOAD_BACKEND, sam3_mode: sam3Mode }),
    });
    const d = await r.json();
    if (!r.ok) { toast(d.error || "Erreur", "err"); return; }
    $btnPreload.style.display = "none";
    $btnPreloadStop.style.display = "inline-block";
    $preloadWrap.classList.add("active");
    $preloadInfo.textContent = "D\u00e9marrage\u2026";
    _preloadPollTimer = setInterval(checkPreloadStatus, 1500);
    toast("Pr\u00e9-chargement SAM 3 d\u00e9marr\u00e9\u2026");
  } catch (e) { toast("Erreur : " + e.message, "err"); }
}

$btnPreload.addEventListener("click", () => {
  if (_preloadComplete) {
    toast("Base SAM 3 d\u00e9j\u00e0 int\u00e9gralement charg\u00e9e", "ok");
    return;
  }
  startPreload();
});

async function stopPreload() {
  try {
    const r = await fetch("/api/preload/stop", { method: "POST" });
    const d = await r.json();
    if (!r.ok) { toast(d.error || "Erreur", "err"); return; }
    $btnPreloadStop.disabled = true;
    $preloadInfo.textContent = "Arr\u00eat en cours\u2026";
    toast("Arr\u00eat du pr\u00e9-chargement\u2026");
  } catch (e) { toast("Erreur : " + e.message, "err"); }
}

$btnPreloadStop.addEventListener("click", stopPreload);

document.addEventListener("DOMContentLoaded", () => {
  updateSamModeField();
  loadDatasets();
  loadHistory();
  checkPreloadStatus();

  const $heightSlider = $("hist-height-slider");
  const $heightVal = $("hist-height-val");
  $heightSlider.addEventListener("input", () => {
    const v = parseInt($heightSlider.value, 10);
    $heightVal.textContent = v === 0 ? "Min" : v + "px";
    const wrap = document.getElementById("history-table-wrap");
    if (wrap) {
      if (v === 0) wrap.classList.add("collapsed");
      else { wrap.classList.remove("collapsed"); wrap.style.maxHeight = v + "px"; }
    }
  });
});
