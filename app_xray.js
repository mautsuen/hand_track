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

// dim but still visible background
const DIM_FILTER = "brightness(0.38) contrast(1.12) saturate(0.95)";

// skeleton main color/glow
const SKEL_COLOR = "rgba(0,255,255,0.92)";
const SKEL_GLOW  = "rgba(0,255,255,0.75)";

// random segment width range
const W_MIN = 6.5;
const W_MAX = 7.5;

// glow pass: draw a thick soft cyan tube around skeleton
const GLOW_ALPHA = 0.28;
const GLOW_BLUR = 26;
const GLOW_W_MULT = 2.8;

// main pass glow
const MAIN_SHADOW_BLUR = 18;

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

// ---------- utils ----------
function setStatus(msg) {
  statusEl.textContent = msg;
  console.log("[status]", msg);
}

function strokeFadedSegment(ctx, x1, y1, x2, y2, width, a1, a2, steps = 8) {
  const dx = x2 - x1, dy = y2 - y1;

  for (let s = 0; s < steps; s++) {
    const t0 = s / steps;
    const t1 = (s + 1) / steps;

    const xa = x1 + dx * t0;
    const ya = y1 + dy * t0;
    const xb = x1 + dx * t1;
    const yb = y1 + dy * t1;

    // alpha at this sub-segment (linear fade)
    const tm = (t0 + t1) * 0.5;
    const a  = a1 + (a2 - a1) * tm;

    ctx.strokeStyle = `rgba(0,255,255,${a})`;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(xa, ya);
    ctx.lineTo(xb, yb);
    ctx.stroke();
  }
}

function cyanGrad(ctx, x1, y1, x2, y2, a1, a2) {
  const g = ctx.createLinearGradient(x1, y1, x2, y2);
  g.addColorStop(0, `rgba(0,255,255,${a1})`);
  g.addColorStop(1, `rgba(0,255,255,${a2})`);
  return g;
}

function resizeCanvasToVideo() {
  overlay.width  = video.videoWidth  || CAMERA_W;
  overlay.height = video.videoHeight || CAMERA_H;
}

// stable hash -> [0,1)
function hash01(n) {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function drawDimBackground() {
  const W = overlay.width, H = overlay.height;

  octx.save();
  octx.filter = DIM_FILTER;
  octx.globalCompositeOperation = "source-over";
  octx.drawImage(video, 0, 0, W, H);
  octx.restore();

  // subtle vignette for x-ray feel
  octx.save();
  const g = octx.createRadialGradient(W*0.5, H*0.5, Math.min(W,H)*0.15, W*0.5, H*0.5, Math.max(W,H)*0.75);
  g.addColorStop(0, "rgba(0,0,0,0.00)");
  g.addColorStop(1, "rgba(0,0,0,0.35)");
  octx.fillStyle = g;
  octx.fillRect(0,0,W,H);
  octx.restore();
}

// compute shortened endpoints + random width (stable)
function segmentInfo(segIdx, ax, ay, bx, by, t) {
  const dx = bx - ax, dy = by - ay;
  const len = Math.hypot(dx, dy);
  if (len < 2) return null;

  // trim fraction
  const rTrim = hash01(segIdx + 17);
  const baseTrim = TRIM_BASE_MIN + (TRIM_BASE_MAX - TRIM_BASE_MIN) * rTrim;
  const wobble = TRIM_WOBBLE * Math.sin(t * 1.7 + segIdx * 2.3);
  const trimFrac = Math.max(0.02, baseTrim + wobble);

  // trim both ends
  const trim = (trimFrac * 0.5) * len;
  const ux = dx / len, uy = dy / len;

  const ax2 = ax + ux * trim;
  const ay2 = ay + uy * trim;
  const bx2 = bx - ux * trim;
  const by2 = by - uy * trim;

  // stable random width
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
    // =========================
    // PASS 1: soft cyan glow tube (behind)
    // =========================
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

    // =========================
    // PASS 2: core skeleton with PER-SEGMENT GRADIENT (on top)
    // =========================
    octx.save();
    octx.globalCompositeOperation = "lighter";

    // IMPORTANT: no shadow blur here, otherwise gradient looks uniform
    octx.shadowBlur = 0;
    octx.shadowColor = "transparent";

    for (let i = 0; i < HAND_CONNECTIONS.length; i++) {
      const [a, b] = HAND_CONNECTIONS[i];
      const pa = lm[a], pb = lm[b];

      const ax = pa.x * W, ay = pa.y * H;
      const bx = pb.x * W, by = pb.y * H;

      const info = segmentInfo(i, ax, ay, bx, by, t);
      if (!info) continue;

      // ---- Randomize brightness per segment (stable) ----
      const rB   = hash01(i + 4242);   // stable 0..1
      const rEnd = hash01(i + 4243);

      // start alpha varies per segment
      const aStart = 0.40 + 0.55 * rB;                 // 0.40 .. 0.95

      // end alpha is smaller => visible fade
      const aEnd   = aStart * (0.2 + 0.1 * rEnd);    // ~2% .. 20% of start

      // ---- TRUE GRADIENT stroke (bright -> dim) ----
      const grad = octx.createLinearGradient(info.ax2, info.ay2, info.bx2, info.by2);
      grad.addColorStop(0, `rgba(0,255,255,${aStart})`);
      grad.addColorStop(1, `rgba(0,255,255,${aEnd})`);
      octx.strokeStyle = grad;

      octx.lineWidth = info.segW;
      octx.beginPath();
      octx.moveTo(info.ax2, info.ay2);
      octx.lineTo(info.bx2, info.by2);
      octx.stroke();

      // ---- endpoint blobs match brightness (start brighter than end) ----
      // ---- endpoint blobs match brightness (start brighter than end) ----
		const dotR = Math.max(DOT_MIN_R, info.segW * DOT_SCALE);

		// start blob (bright)
		octx.fillStyle = `rgba(0,255,255,${aStart})`;
		octx.beginPath();
		octx.arc(info.ax2, info.ay2, dotR, 0, Math.PI * 2);
		octx.fill();

		// end blob (dim)
		octx.fillStyle = `rgba(0,255,255,${aEnd})`;
		octx.beginPath();
		octx.arc(info.bx2, info.by2, dotR, 0, Math.PI * 2);
		octx.fill();

    }

    octx.restore();
  }

  octx.restore();
}

// ===== MediaPipe (no-SIMD) =====
let handLandmarker = null;

async function initMediaPipePreferGPU() {
  setStatus("Loading MediaPipe (prefer GPU) ...");

  const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";

  // 先用 SIMD（通常更快）；若你的目標包含舊手機/舊瀏覽器，再改成 nosimd
  const wasmFileset = {
    wasmLoaderPath: `${WASM_BASE}/vision_wasm_internal.js`,
    wasmBinaryPath: `${WASM_BASE}/vision_wasm_internal.wasm`,
  };

  try {
    handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
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
    console.warn("GPU init failed, falling back to CPU:", e);
  }

  // fallback CPU
  handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
    baseOptions: {
      modelAssetPath: "./models/hand_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.2,
    minHandPresenceConfidence: 0.2,
    minTrackingConfidence: 0.2,
  });

  setStatus("MediaPipe ready ✅ (CPU fallback)");
}


// ===== main loop =====
let stream = null;
let running = false;
let rafId = 0;
let frame = 0;
let lastRes = null;

let lastStatusTs = 0;

function loop() {
  if (!running) return;

  if ((frame % INFER_EVERY_N) === 0) {
    try {
      const ts = performance.now();
      lastRes = handLandmarker.detectForVideo(video, ts) || null;
    } catch (e) {
      console.warn("detectForVideo error:", e);
      lastRes = null;
    }
  }

  // draw dim background + skeleton glow/bones
  drawDimBackground();
  drawSkeletonGlowAndBones(lastRes);

  // status
  const now = performance.now();
  if (now - lastStatusTs > 500) {
    lastStatusTs = now;
    const handsN = lastRes?.landmarks?.length ?? 0;
    setStatus(`Running. hands=${handsN}`);
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

    resizeCanvasToVideo();
    window.addEventListener("resize", resizeCanvasToVideo);

    setStatus(`Camera ready (${video.videoWidth}x${video.videoHeight}).`);

    await initMediaPipePreferGPU();

    running = true;
    frame = 0;
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

  window.removeEventListener("resize", resizeCanvasToVideo);

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

console.log("app.js loaded OK");
setStatus("Idle");
