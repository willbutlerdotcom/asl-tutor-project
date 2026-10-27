import { DrawingUtils, FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { normalize } from "./features";
import { loadModel, predict, type ModelSpec, type Prediction } from "./mlp";
import "./style.css";
import { onFrame } from "./tutor";
import { renderTutor, renderStats } from "./tutorView";

const video = document.querySelector<HTMLVideoElement>("#webcam")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const letterEl = document.querySelector<HTMLSpanElement>("#letter")!;
const confidenceEl = document.querySelector<HTMLSpanElement>("#confidence")!;
const ctx = canvas.getContext("2d")!;
const drawer = new DrawingUtils(ctx);

const BASE = import.meta.env.BASE_URL;
const WASM_PATH = `${BASE}wasm`;
const MODEL_PATH = `${BASE}hand_landmarker.task`;
const CLASSIFIER_PATH = `${BASE}model.json`;

// Below this, the letter is shown dimmed: the model isn't sure.
const SHOW_CONFIDENT = 0.5;

async function createLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  const options = (delegate: "GPU" | "CPU") => ({
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: "VIDEO" as const,
    numHands: 1,
  });
  try {
    return await HandLandmarker.createFromOptions(vision, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU", err);
    return await HandLandmarker.createFromOptions(vision, options("CPU"));
  }
}

async function startCamera(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  // Match the canvas's drawing buffer to the real video resolution.
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function draw(result: HandLandmarkerResult): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const hand of result.landmarks) {
    drawer.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00e0a0",
      lineWidth: 4,
    });
    drawer.drawLandmarks(hand, { color: "#ff3070", lineWidth: 1, radius: 4 });
  }
}

/** Raw landmarks -> normalize() (stage 1) -> scaler + MLP (stage 2). Same path as Python. */
function classify(result: HandLandmarkerResult, model: ModelSpec): Prediction | null {
  const hand = result.landmarks[0];
  const handed = result.handedness[0]?.[0];
  if (!hand || !handed) return null;
  // The image size must be the frame MediaPipe saw, not the on-screen size.
  const x = normalize(hand, handed.categoryName, video.videoWidth, video.videoHeight);
  return predict(model, x);
}

function showPrediction(p: Prediction | null): void {
  if (!p) {
    letterEl.textContent = "–";
    confidenceEl.textContent = "show a hand";
    letterEl.classList.add("unsure");
    return;
  }
  letterEl.textContent = p.label;
  confidenceEl.textContent = `${Math.round(p.confidence * 100)}%`;
  letterEl.classList.toggle("unsure", p.confidence < SHOW_CONFIDENT);
}

async function main(): Promise<void> {
  statusEl.textContent = "Loading models…";
  const [landmarker, model] = await Promise.all([createLandmarker(), loadModel(CLASSIFIER_PATH)]);
  statusEl.textContent = "Starting camera…";
  await startCamera();

  let lastVideoTime = -1;
  let frames = 0;
  let fps = 0;
  let fpsWindowStart = performance.now();

  const loop = () => {
    // Only run inference when the video has a new frame.
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      // Timestamps must strictly increase in VIDEO mode.
      const result = landmarker.detectForVideo(video, performance.now());
      draw(result);

      const prediction = classify(result, model);
      showPrediction(prediction);

      // Handy for poking at in DevTools: lastResult.landmarks[0], lastPrediction.probs
      (window as any).lastResult = result;
      (window as any).lastPrediction = prediction;

      frames++;
      const now = performance.now();
      if (now - fpsWindowStart >= 1000) {
        fps = (frames * 1000) / (now - fpsWindowStart);
        frames = 0;
        fpsWindowStart = now;
      }
      const hand = result.handedness[0]?.[0];
      statusEl.textContent = hand
        ? `${fps.toFixed(0)} fps · hand detected (${hand.categoryName}, ${hand.score.toFixed(2)})`
        : `${fps.toFixed(0)} fps · no hand`;
      onFrame(prediction?.label ?? null, prediction?.confidence ?? 0, now);
      renderTutor(now);
      renderStats();  
    }
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
});
