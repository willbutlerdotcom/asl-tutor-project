/**
 * MLP forward pass using only the numbers in model.json (written by train.py's export_model).
 * This is the TypeScript twin of predict_from_spec in train.py:
 *
 *   a = (x - scaler.mean) / scaler.scale          // stage 2: StandardScaler
 *   for each layer: a = activation(a · W + b)     // relu/tanh/... hidden, softmax last
 *   letter = labels[argmax(a)]
 */

export type Activation = "relu" | "tanh" | "logistic" | "identity" | "softmax";

export interface Layer {
  W: number[][]; // [inputs][outputs], exactly sklearn's coefs_ layout
  b: number[];
  activation: Activation;
}

export interface ModelSpec {
  format: string;
  labels: string[];
  n_features: number;
  scaler: { mean: number[]; scale: number[] };
  layers: Layer[];
}

export interface Prediction {
  label: string;
  confidence: number; // softmax probability of the winning letter, 0..1
  probs: number[]; // one per label, same order as spec.labels
}

export async function loadModel(url = "/model.json"): Promise<ModelSpec> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  const spec = (await res.json()) as ModelSpec;
  if (spec.format !== "asl-mlp-v1") throw new Error(`Unexpected model format: ${spec.format}`);
  return spec;
}

function activate(z: number[], kind: Activation): number[] {
  switch (kind) {
    case "relu":
      return z.map((v) => Math.max(v, 0));
    case "tanh":
      return z.map((v) => Math.tanh(v));
    case "logistic":
      return z.map((v) => 1 / (1 + Math.exp(-v)));
    case "identity":
      return z;
    case "softmax": {
      const max = Math.max(...z); // subtract the max so exp() can't overflow
      const e = z.map((v) => Math.exp(v - max));
      const sum = e.reduce((s, v) => s + v, 0);
      return e.map((v) => v / sum);
    }
    default:
      throw new Error(`Unknown activation: ${kind as string}`);
  }
}

export function predict(spec: ModelSpec, x: ArrayLike<number>): Prediction {
  if (x.length !== spec.n_features) {
    throw new Error(`Expected ${spec.n_features} features, got ${x.length}`);
  }
  const { mean, scale } = spec.scaler;
  let a = Array.from(x, (v, i) => (v - mean[i]) / scale[i]);

  for (const { W, b, activation } of spec.layers) {
    const z = b.slice(); // start from the bias, then add a · W
    for (let i = 0; i < a.length; i++) {
      const row = W[i];
      for (let j = 0; j < z.length; j++) z[j] += a[i] * row[j];
    }
    a = activate(z, activation);
  }

  let best = 0;
  for (let j = 1; j < a.length; j++) if (a[j] > a[best]) best = j;
  return { label: spec.labels[best], confidence: a[best], probs: a };
}
