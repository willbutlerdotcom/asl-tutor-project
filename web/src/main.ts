import { DrawingUtils, FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import "./style.css";

const video = document.querySelector<HTMLVideoElement>("#webcam")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const ctx = canvas.getContext("2d")!;
const drawer = new DrawingUtils(ctx);

// Served from public/: the wasm runtime is copied from node_modules,
// the model is the same .task file the Python side uses.
const WASM_PATH = "/wasm";
const MODEL_PATH = "/hand_landmarker.task";

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

async function main(): Promise<void> {
  statusEl.textContent = "Loading model…";
  const landmarker = await createLandmarker();
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

      // Handy for poking at in DevTools: window.lastResult.landmarks[0]
      (window as any).lastResult = result;

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
    }
    requestAnimationFrame(loop);
  };
  loop();
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
});
