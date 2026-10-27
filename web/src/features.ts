/**
 * TypeScript port of features.normalize_batch (features.py), for ONE hand.
 *
 * It must produce the same 63 numbers as the Python version, in the same order
 * (x0, y0, z0, x1, y1, z1, ...), or the model sees inputs it was never trained on.
 * The parity test (tests/parity.test.ts) checks this on 20 real recorded frames.
 *
 *   1. Pixel-proportional units  -> undo the camera's aspect-ratio stretch
 *   2. Wrist to the origin       -> position invariance
 *   3. Divide by palm length     -> distance/scale invariance
 *   4. Mirror "Left" hands       -> handedness invariance
 *
 * Rotation is deliberately NOT normalized (K vs P, G vs Q, H vs U differ by orientation).
 */

export const WRIST = 0;
export const MIDDLE_MCP = 9;
export const N_LANDMARKS = 21;
export const N_FEATURES = N_LANDMARKS * 3;

/** Same shape as MediaPipe's NormalizedLandmark (extra fields are ignored). */
export interface Point3 {
  x: number;
  y: number;
  z: number;
}

/**
 * @param landmarks  21 image-normalized landmarks from MediaPipe (result.landmarks[0])
 * @param handedness "Left" or "Right" (result.handedness[0][0].categoryName)
 * @param imgW       width in pixels of the frame MediaPipe saw (video.videoWidth)
 * @param imgH       height in pixels of the frame MediaPipe saw (video.videoHeight)
 */
export function normalize(
  landmarks: readonly Point3[],
  handedness: string,
  imgW: number,
  imgH: number,
): Float32Array {
  if (landmarks.length !== N_LANDMARKS) {
    throw new Error(`Expected ${N_LANDMARKS} landmarks, got ${landmarks.length}`);
  }

  // 1. Pixel-proportional units. z uses the width, like features.py.
  const pts = landmarks.map((p) => [p.x * imgW, p.y * imgH, p.z * imgW]);

  // 2. Wrist at the origin. Copy the wrist first: the loop overwrites pts[WRIST].
  const [wx, wy, wz] = pts[WRIST];
  for (const p of pts) {
    p[0] -= wx;
    p[1] -= wy;
    p[2] -= wz;
  }

  // 3. Scale by palm size (wrist -> middle-finger knuckle). Same 1e-6 guard as Python.
  const [mx, my, mz] = pts[MIDDLE_MCP];
  const scale = Math.max(Math.hypot(mx, my, mz), 1e-6);

  // 4. Mirror left hands onto right hands.
  const flip = handedness === "Left" ? -1 : 1;

  // Float32Array matches Python's final .astype(np.float32).
  const out = new Float32Array(N_FEATURES);
  pts.forEach((p, i) => {
    out[3 * i] = (flip * p[0]) / scale;
    out[3 * i + 1] = p[1] / scale;
    out[3 * i + 2] = p[2] / scale;
  });
  return out;
}
