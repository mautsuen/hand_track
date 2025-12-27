import { HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const startBtn = document.getElementById("startBtn");
const stopBtn  = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");

const video   = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx    = overlay.getContext("2d");

// ===== knobs =====
const CAMERA_W = 424;
const CAMERA_H = 240;

// run detection every N frames
const INFER_EVERY_N = 2;

// background CV update every N frames (mode B)
const BG_EVERY_N = 2;

// dim but still visible background (mode MP / mode A base)
const DIM_FILTER = "brightness(0.38) contrast(1.12) saturate(0.95)";

// skeleton main color/glow
const SKEL_COLOR = "rgba(0,255,255,0.92)";
const SKEL_GLOW  = "rgba(0,255,255,0.75)";

// random segment width range
const W_MIN = 6.5;
const W_MAX = 7.5;

// glow pass: draw a thick soft cyan tube around skeleton
const GLOW_ALPHA = 0.20;
const GLOW_BLUR = 22;
const GLOW_W_MULT = 2.3;

// shorten each segment (stable random + tiny smooth wobble)
const TRIM_BASE_MIN = 0.35;
const TRIM_BASE_MAX = 0.45;
const TRIM_WOBBLE   = 0.05;

// rounded endpoint blobs
const DOT_SCALE = 0.75;
const DOT_MIN_R = 0.6;

// IMPORTANT: removed palm web connections 5-9-13-17
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20]
];

// ===== Modes =====
const MODE_MP       = "mp";
const MODE_MP_CV_A  = "mp_cv_a"; // ROI x-ray texture (方案 A)
const MODE_MP_CV_B  = "mp_cv_b"; // medical background (方案 B)
let mode = MODE_MP;

// ===== Mode UI buttons =====
let modeBtnMP = null;
let modeBtnA  = null;
let modeBtnB  = null;

// ===== OpenCV (lazy) =====
let cvReadyPromise = null;
let cvReady = false;

// ===== MediaPipe =====
let handLandmarker = null;

// ===== main loop state =====
let stream = null;
let running = false;
let rafId = 0;
let frame = 0;
let lastRes = null;
let lastStatusTs = 0;

// ===== OpenCV caches =====
// (A) ROI texture cache (recomputed only on infer frames)
let xrayCache = []; // array of {x,y,w,h, canvas}
const xrayCanvasPool = []; // reuse canvases

// (B) processed background cache canvas
const bgCanvas = document.createElement("canvas");
const bgctx = bgCanvas.getContext("2d");

// offscreen for grabbing frame pixels
const frameCanvas = document.createElement("canvas");
const fctx = frameCanvas.getContext("2d", { willReadFrequently: true });

let bgDirty = true; // force update on resize / mode switch

// ---------- utils ----------
function setStatus(msg) {
  statusEl.textContent = msg;
  console.log("[status]", msg);
}

function resizeCanvasesToVideo() {
  const w = video.videoWidth || CAMERA_W;
  const h = video.videoHeight || CAMERA_H;

  overlay.width = w;
  overlay.height = h;

  frameCanvas.width = w;
  frameCanvas.height = h;

  bgCanvas.width = w;
  bgCanvas.height = h;

  bgDirty = true;
}

function hash01(n) {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

// (used for MP / A only)
function drawVignette() {
  const W = overlay.width, H = overlay.height;
  octx.save();
  const g = octx.createRadialGradient(
    W*0.5, H*0.5, Math.min(W,H)*0.15,
    W*0.5, H*0.5, Math.max(W,H)*0.75
  );
  g.addColorStop(0, "rgba(0,0,0,0.00)");
  g.addColorStop(1, "rgba(0,0,0,0.35)");
  octx.fillStyle = g;
  octx.fillRect(0, 0, W, H);
  octx.restore();
}

// MP / A background
function drawDimBackgroundFromVideo() {
  const W = overlay.width, H = overlay.height;

  octx.save();
  octx.filter = DIM_FILTER;
  octx.globalCompositeOperation = "source-over";
  octx.drawImage(video, 0, 0, W, H);
  octx.restore();

  // keep vignette for MP/A (x-ray vibe)
  drawVignette();
}

// compute shortened endpoints + random width (stable)
function segmentInfo(segIdx, ax, ay, bx, by, t) {
  const dx = bx - ax, dy = by - ay;
  const len = Math.hypot(dx, dy);
  if (len < 2) return null;

  const rTrim = hash01(segIdx + 17);
  const baseTrim = TRIM_BASE_MIN + (TRIM_BASE_MAX - TRIM_BASE_MIN) * rTrim;
  const wobble = TRIM_WOBBLE * Math.sin(t * 1.7 + segIdx * 2.3);
  const trimFrac = Math.max(0.02, baseTrim + wobble);

  const trim = (trimFrac * 0.5) * len;
  const ux = dx / len, uy = dy / len;

  const ax2 = ax + ux * trim;
  const ay2 = ay + uy * trim;
  const bx2 = bx - ux * trim;
  const by2 = by - uy * trim;

  const rW = hash01(segIdx + 999);
  const segW = W_MIN + (W_MAX - W_MIN) * rW;

  return { ax2, ay2, bx2, by2, segW };
}

function drawSkeletonGlowAndBones(results) {
  const hands = results?.landmarks ?? [];
  if (!hands.length) return;

  const W = overlay.width, H = overlay.height;
  const t = performance.now() * 0.001;

  octx.save();
  octx.lineCap = "round";
  octx.lineJoin = "round";

  for (const lm of hands) {
    // PASS 1: soft cyan glow tube (behind)
    octx.save();
    octx.globalCompositeOperation = "screen";
    octx.strokeStyle = `rgba(0,255,255,${GLOW_ALPHA})`;
    octx.shadowColor = "rgba(0,255,255,0.45)";
    octx.shadowBlur = GLOW_BLUR;

    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
      const [a, b] = HAND_CONNECTIONS[i];
      const pa = lm[a], pb = lm[b];

      const ax = pa.x * W, ay = pa.y * H;
      const bx = pb.x * W, by = pb.y * H;

      const info = segmentInfo(i, ax, ay, bx, by, t);
      if (!info) continue;

      octx.lineWidth = info.segW * GLOW_W_MULT;
      octx.beginPath();
      octx.moveTo(info.ax2, info.ay2);
      octx.lineTo(info.bx2, info.by2);
      octx.stroke();
    }
    octx.restore();

    // PASS 2: core skeleton with per-segment gradient
    octx.save();
    octx.globalCompositeOperation = "lighter";
    octx.shadowBlur = 0;
    octx.shadowColor = "transparent";

    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
      const [a, b] = HAND_CONNECTIONS[i];
      const pa = lm[a], pb = lm[b];

      const ax = pa.x * W, ay = pa.y * H;
      const bx = pb.x * W, by = pb.y * H;

      const info = segmentInfo(i, ax, ay, bx, by, t);
      if (!info) continue;

      const rB   = hash01(i + 4242);
      const rEnd = hash01(i + 4243);

      const aStart = 0.40 + 0.55 * rB;            // 0.40 .. 0.95
      const aEnd   = aStart * (0.08 + 0.22*rEnd); // end dimmer

      const grad = octx.createLinearGradient(info.ax2, info.ay2, info.bx2, info.by2);
      grad.addColorStop(0, `rgba(0,255,255,${aStart})`);
      grad.addColorStop(1, `rgba(0,255,255,${aEnd})`);
      octx.strokeStyle = grad;

      octx.lineWidth = info.segW;
      octx.beginPath();
      octx.moveTo(info.ax2, info.ay2);
      octx.lineTo(info.bx2, info.by2);
      octx.stroke();

      const dotR = Math.max(DOT_MIN_R, info.segW * DOT_SCALE);

      octx.fillStyle = `rgba(0,255,255,${aStart})`;
      octx.beginPath();
      octx.arc(info.ax2, info.ay2, dotR, 0, Math.PI * 2);
      octx.fill();

      octx.fillStyle = `rgba(0,255,255,${aEnd})`;
      octx.beginPath();
      octx.arc(info.bx2, info.by2, dotR, 0, Math.PI * 2);
      octx.fill();
    }

    octx.restore();
  }

  octx.restore();
}

// ======= Mode buttons UI =======
function initModeButtons() {
  const bar = document.createElement("div");
  bar.style.display = "flex";
  bar.style.gap = "8px";
  bar.style.flexWrap = "wrap";
  bar.style.alignItems = "center";
  bar.style.marginBottom = "8px";

  modeBtnMP = document.createElement("button");
  modeBtnMP.textContent = "MediaPipe";
  modeBtnMP.type = "button";

  modeBtnA = document.createElement("button");
  modeBtnA.textContent = "MP + OpenCV ROI X-ray";
  modeBtnA.type = "button";

  modeBtnB = document.createElement("button");
  modeBtnB.textContent = "MP + OpenCV Medical BG";
  modeBtnB.type = "button";

  for (const b of [modeBtnMP, modeBtnA, modeBtnB]) {
    b.style.padding = "8px 12px";
    b.style.borderRadius = "10px";
    b.style.border = "1px solid rgba(0,255,255,0.35)";
    b.style.background = "rgba(0,0,0,0.35)";
    b.style.color = "rgba(220,255,255,0.95)";
    b.style.cursor = "pointer";
  }

  modeBtnMP.onclick = () => setMode(MODE_MP);
  modeBtnA.onclick  = () => setMode(MODE_MP_CV_A);
  modeBtnB.onclick  = () => setMode(MODE_MP_CV_B);

  const parent = startBtn.parentElement || document.body;
  parent.insertBefore(bar, startBtn);
  bar.appendChild(modeBtnMP);
  bar.appendChild(modeBtnA);
  bar.appendChild(modeBtnB);

  updateModeButtons();
}

function updateModeButtons() {
  if (!modeBtnMP) return;

  const activeStyle = (btn) => {
    btn.style.background = "rgba(0,255,255,0.18)";
    btn.style.borderColor = "rgba(0,255,255,0.75)";
  };
  const idleStyle = (btn) => {
    btn.style.background = "rgba(0,0,0,0.35)";
    btn.style.borderColor = "rgba(0,255,255,0.35)";
  };

  idleStyle(modeBtnMP);
  idleStyle(modeBtnA);
  idleStyle(modeBtnB);

  if (mode === MODE_MP) activeStyle(modeBtnMP);
  if (mode === MODE_MP_CV_A) activeStyle(modeBtnA);
  if (mode === MODE_MP_CV_B) activeStyle(modeBtnB);
}

async function setMode(nextMode) {
  if (nextMode === mode) return;

  if (nextMode === MODE_MP_CV_A || nextMode === MODE_MP_CV_B) {
    setStatus("Switching mode: loading OpenCV if needed ...");
    await ensureOpenCV();
  }

  mode = nextMode;
  updateModeButtons();

  xrayCache = [];
  bgDirty = true;

  const name =
    (mode === MODE_MP) ? "MediaPipe" :
    (mode === MODE_MP_CV_A) ? "MP + OpenCV ROI X-ray" :
    "MP + OpenCV Medical BG";

  setStatus("Mode: " + name);
}

// ======= OpenCV loader (ONLY ./opencv/opencv.js) =======
function loadOpenCV() {
  return new Promise((resolve, reject) => {
    if (window.cv && typeof window.cv.Mat === "function") {
      cvReady = true;
      resolve();
      return;
    }

    window.Module = window.Module || {};
    window.Module.locateFile = (file) => "./opencv/" + file;

    const s = document.createElement("script");
    s.src = "./opencv/opencv.js";
    s.async = true;

    s.onload = () => {
      const start = performance.now();

      const check = () => {
        if (window.cv && typeof window.cv.Mat === "function") {
          cvReady = true;
          resolve();
          return;
        }
        if (performance.now() - start > 8000) {
          reject(new Error("OpenCV runtime init timeout"));
          return;
        }
        requestAnimationFrame(check);
      };

      if (window.cv) {
        window.cv.onRuntimeInitialized = () => {
          cvReady = true;
          resolve();
        };
      }
      check();
    };

    s.onerror = () => reject(new Error("Failed to load ./opencv/opencv.js"));
    document.head.appendChild(s);
  });
}

async function ensureOpenCV() {
  if (cvReady) return;
  if (!cvReadyPromise) {
    cvReadyPromise = (async () => {
      setStatus("OpenCV: loading ./opencv/opencv.js ...");
      await loadOpenCV();
      setStatus("OpenCV ready ✅");
    })();
  }
  return cvReadyPromise;
}

// ======= 方案 A: ROI X-ray texture (OpenCV) =======
function getHandBBoxPx(lm, W, H) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const p of lm) {
    const x = p.x * W;
    const y = p.y * H;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }

  const bw = Math.max(10, maxX - minX);
  const bh = Math.max(10, maxY - minY);
  const pad = Math.max(14, 0.22 * Math.max(bw, bh));

  minX -= pad; minY -= pad;
  maxX += pad; maxY += pad;

  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(W - 1, Math.ceil(maxX));
  maxY = Math.min(H - 1, Math.ceil(maxY));

  const w = Math.max(2, maxX - minX);
  const h = Math.max(2, maxY - minY);
  return { x: minX, y: minY, w, h };
}

function getCanvasFromPool(w, h) {
  for (const c of xrayCanvasPool) {
    if (c.width === w && c.height === h) return c;
  }
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  xrayCanvasPool.push(c);
  return c;
}

function buildXrayCacheFromResults(results) {
  if (!cvReady || !window.cv) return [];

  const hands = results?.landmarks ?? [];
  if (!hands.length) return [];

  const W = overlay.width, H = overlay.height;

  fctx.save();
  fctx.filter = "none";
  fctx.drawImage(video, 0, 0, W, H);
  fctx.restore();

  const cv = window.cv;
  const out = [];

  for (const lm of hands) {
    const bb = getHandBBoxPx(lm, W, H);
    if (bb.w < 10 || bb.h < 10) continue;

    const roi = fctx.getImageData(bb.x, bb.y, bb.w, bb.h);

    let src = null, gray = null, blur = null, edges = null, kernel = null;
    try {
      src = cv.matFromImageData(roi);
      gray = new cv.Mat();
      blur = new cv.Mat();
      edges = new cv.Mat();

      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

      cv.Canny(blur, edges, 55, 120);

      kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
      cv.dilate(edges, edges, kernel);

      cv.GaussianBlur(edges, edges, new cv.Size(7, 7), 0, 0, cv.BORDER_DEFAULT);

      const w = bb.w, h = bb.h;
      const rgba = new Uint8ClampedArray(w * h * 4);
      const ed = edges.data;

      const alphaScale = 0.85;
      const maxA = 210;

      for (let i = 0; i < w * h; i++) {
        const v = ed[i];
        let a = Math.round(v * alphaScale);
        if (a > maxA) a = maxA;
        if (a < 2) a = 0;

        const k = i * 4;
        rgba[k + 0] = 0;
        rgba[k + 1] = 255;
        rgba[k + 2] = 255;
        rgba[k + 3] = a;
      }

      const imgData = new ImageData(rgba, w, h);
      const c = getCanvasFromPool(w, h);
      const cctx = c.getContext("2d");
      cctx.clearRect(0, 0, w, h);
      cctx.putImageData(imgData, 0, 0);

      out.push({ x: bb.x, y: bb.y, w, h, canvas: c });
    } catch (e) {
      console.warn("OpenCV ROI error:", e);
    } finally {
      if (kernel) kernel.delete();
      if (edges) edges.delete();
      if (blur) blur.delete();
      if (gray) gray.delete();
      if (src) src.delete();
    }
  }

  return out;
}

function drawXrayCache() {
  if (!xrayCache.length) return;
  octx.save();
  octx.globalCompositeOperation = "screen";
  octx.shadowColor = "rgba(0,255,255,0.65)";
  octx.shadowBlur = 14;
  for (const item of xrayCache) {
    octx.drawImage(item.canvas, item.x, item.y);
  }
  octx.restore();
}

// ======= 方案 B: 背景降噪 + 灰藍醫療影像 (OpenCV) =======
// ✅ 已移除「放射狀暗角(vignette)」與「額外 cyan tint」避免中間青色圓形
function updateMedicalBackgroundCV() {
  if (!cvReady || !window.cv) return;

  const cv = window.cv;
  const W = overlay.width, H = overlay.height;

  fctx.save();
  fctx.filter = "none";
  fctx.drawImage(video, 0, 0, W, H);
  fctx.restore();

  let src = null, gray = null, den = null, color = null, adj = null;
  try {
    src = cv.imread(frameCanvas);
    gray = new cv.Mat();
    den = new cv.Mat();
    color = new cv.Mat();
    adj = new cv.Mat();

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // denoise but keep edges
    cv.bilateralFilter(gray, den, 7, 40, 40, cv.BORDER_DEFAULT);

    // medical-ish blue/gray
    cv.applyColorMap(den, color, cv.COLORMAP_BONE);

    // contrast/brightness tweak
    color.convertTo(adj, -1, 1.10, -12);

    // small blur for film feel
    cv.GaussianBlur(adj, adj, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);

    cv.imshow(bgCanvas, adj);
  } catch (e) {
    console.warn("Medical BG OpenCV error:", e);
  } finally {
    if (adj) adj.delete();
    if (color) color.delete();
    if (den) den.delete();
    if (gray) gray.delete();
    if (src) src.delete();
  }
}

function drawMedicalBackgroundFromCache() {
  const W = overlay.width, H = overlay.height;

  octx.save();
  octx.globalCompositeOperation = "source-over";
  octx.drawImage(bgCanvas, 0, 0, W, H);

  // very subtle uniform darkening (NO radial circle)
  octx.fillStyle = "rgba(0,0,0,0.10)";
  octx.fillRect(0, 0, W, H);

  octx.restore();
}

// ===== MediaPipe init (prefer GPU) =====
async function initMediaPipePreferGPU() {
  setStatus("Loading MediaPipe (prefer GPU) ...");

  const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
  const wasmFilesetSIMD = {
    wasmLoaderPath: `${WASM_BASE}/vision_wasm_internal.js`,
    wasmBinaryPath: `${WASM_BASE}/vision_wasm_internal.wasm`,
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(wasmFilesetSIMD, {
      baseOptions: {
        modelAssetPath: "./models/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.2,
      minHandPresenceConfidence: 0.2,
      minTrackingConfidence: 0.2,
    });
    setStatus("MediaPipe ready ✅ (GPU)");
    return;
  } catch (e) {
    console.warn("GPU init failed, fallback CPU:", e);
  }

  handLandmarker = await HandLandmarker.createFromOptions(wasmFilesetSIMD, {
    baseOptions: {
      modelAssetPath: "./models/hand_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.2,
    minHandPresenceConfidence: 0.2,
    minTrackingConfidence: 0.2,
  });

  setStatus("MediaPipe ready ✅ (CPU fallback)");
}

// ===== main loop =====
function loop() {
  if (!running) return;

  const doInfer = ((frame % INFER_EVERY_N) === 0);

  if (doInfer) {
    try {
      const ts = performance.now();
      lastRes = handLandmarker.detectForVideo(video, ts) || null;

      if (mode === MODE_MP_CV_A && cvReady) {
        xrayCache = buildXrayCacheFromResults(lastRes);
      }
    } catch (e) {
      console.warn("detectForVideo error:", e);
      lastRes = null;
      xrayCache = [];
    }
  }

  // draw background by mode
  if (mode === MODE_MP_CV_B && cvReady) {
    if (bgDirty || (frame % BG_EVERY_N) === 0) {
      updateMedicalBackgroundCV();
      bgDirty = false;
    }
    drawMedicalBackgroundFromCache(); // <-- NO vignette here
  } else {
    drawDimBackgroundFromVideo(); // MP/A background with vignette
  }

  // mode A overlay
  if (mode === MODE_MP_CV_A) {
    drawXrayCache();
  }

  // skeleton always on top
  drawSkeletonGlowAndBones(lastRes);

  // status
  const now = performance.now();
  if (now - lastStatusTs > 500) {
    lastStatusTs = now;
    const handsN = lastRes?.landmarks?.length ?? 0;
    const modeName =
      (mode === MODE_MP) ? "MP" :
      (mode === MODE_MP_CV_A) ? (cvReady ? "MP+CV(A)" : "MP+CV(A loading)") :
      (cvReady ? "MP+CV(B)" : "MP+CV(B loading)");
    setStatus(`Running. mode=${modeName} hands=${handsN}`);
  }

  frame++;
  rafId = requestAnimationFrame(loop);
}

async function start() {
  try {
    startBtn.disabled = true;
    stopBtn.disabled = false;

    setStatus("Requesting camera permission...");
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width:  { ideal: CAMERA_W },
        height: { ideal: CAMERA_H },
        frameRate: { ideal: 30, max: 30 }
      },
      audio: false
    });

    video.srcObject = stream;
    await new Promise((resolve) => (video.onloadedmetadata = resolve));
    await video.play();

    resizeCanvasesToVideo();
    window.addEventListener("resize", resizeCanvasesToVideo);

    setStatus(`Camera ready (${video.videoWidth}x${video.videoHeight}).`);

    await initMediaPipePreferGPU();

    running = true;
    frame = 0;
    xrayCache = [];
    bgDirty = true;

    setStatus("Running ✅");
    loop();
  } catch (e) {
    console.error(e);
    setStatus("ERROR: " + (e?.message || e));
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

function stop() {
  running = false;
  cancelAnimationFrame(rafId);

  octx.clearRect(0, 0, overlay.width, overlay.height);
  xrayCache = [];
  bgDirty = true;

  window.removeEventListener("resize", resizeCanvasesToVideo);

  if (video) {
    video.pause();
    video.srcObject = null;
  }
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }

  setStatus("Stopped.");
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

// init UI
initModeButtons();
console.log("app.js loaded OK");
setStatus("Idle");
