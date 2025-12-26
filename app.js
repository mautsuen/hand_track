import { HandLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton = null;
let webcamRunning = false;

const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode,
    numHands: 2,
  });

  demosSection.classList.remove("invisible");
};

createHandLandmarker();

/********************************************************************
// Demo 1: Click images to detect hands
********************************************************************/
const imageContainers = document.getElementsByClassName("detectOnClick");
for (let i = 0; i < imageContainers.length; i++) {
  imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
  if (!handLandmarker) {
    console.log("Wait for handLandmarker to load before clicking!");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await handLandmarker.setOptions({ runningMode: "IMAGE" });
  }

  const target = event.target;

  // remove old canvas overlays
  const allCanvas = target.parentNode.getElementsByClassName("canvas");
  for (let i = allCanvas.length - 1; i >= 0; i--) {
    allCanvas[i].parentNode.removeChild(allCanvas[i]);
  }

  const handLandmarkerResult = handLandmarker.detect(target);
  console.log(handLandmarkerResult?.handednesses?.[0]?.[0]);

  const canvas = document.createElement("canvas");
  canvas.className = "canvas";
  canvas.width = target.naturalWidth;
  canvas.height = target.naturalHeight;

  canvas.style.left = "0px";
  canvas.style.top = "0px";
  canvas.style.width = target.width + "px";
  canvas.style.height = target.height + "px";
  canvas.style.position = "absolute";
  canvas.style.pointerEvents = "none";

  target.parentNode.appendChild(canvas);

  const cxt = canvas.getContext("2d");

  // drawConnectors / drawLandmarks / HAND_CONNECTIONS are globals
  // provided by drawing_utils.js and hands.js that you included in HTML.
  for (const landmarks of handLandmarkerResult.landmarks) {
    drawConnectors(cxt, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
    drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
  }
}

/********************************************************************
// Demo 2: Webcam continuous hands landmarks detection
********************************************************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

function enableCam() {
  if (!handLandmarker) {
    console.log("Wait! handLandmarker not loaded yet.");
    return;
  }

  webcamRunning = !webcamRunning;
  enableWebcamButton.innerText = webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE PREDICTIONS";

  if (!webcamRunning) return;

  navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam, { once: true });
  });
}

let lastVideoTime = -1;
let results = undefined;

async function predictWebcam() {
  // set canvas size to match video
  canvasElement.style.width = video.videoWidth + "px";
  canvasElement.style.height = video.videoHeight + "px";
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  const startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (results?.landmarks) {
    for (const landmarks of results.landmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });
    }
  }

  canvasCtx.restore();

  if (webcamRunning) {
    window.requestAnimationFrame(predictWebcam);
  }
}
