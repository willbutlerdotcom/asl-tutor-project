/**
 * Parity test: the browser must reproduce Python exactly.
 *
 * parity_rows.json comes from `python export_parity.py`: 20 real recorded frames with
 * the features and predictions Python computed. Here the same raw landmarks go through
 * features.ts and mlp.ts. Two separate checks per row, so a failure tells you where:
 *   - features   -> a bug in the normalize() port
 *   - prediction -> a bug in the forward pass (or a stale model.json in public/)
 *
 * Run from web/:  npx vitest run
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { normalize, type Point3 } from "../src/features";
import { predict, type ModelSpec } from "../src/mlp";

interface ParityRow {
  file: string;
  true_label: string;
  handedness: string;
  img_w: number;
  img_h: number;
  landmarks: Point3[];
  py_features: number[];
  py_prediction: string;
  py_confidence: number;
}

const load = <T>(rel: string): T =>
  JSON.parse(readFileSync(new URL(rel, import.meta.url), "utf8")) as T;

const spec = load<ModelSpec>("../public/model.json");
const rows = load<ParityRow[]>("./parity_rows.json");

// Python works in float64 and casts to float32 at the end; TS does the same, so
// differences should be ~1e-7. 1e-4 leaves room without hiding real bugs.
const TOL = 1e-4;

describe("Python ↔ TypeScript parity", () => {
  it("has 20 rows", () => {
    expect(rows).toHaveLength(20);
  });

  rows.forEach((row, i) => {
    const name = `row ${String(i).padStart(2, "0")} (${row.true_label}, ${row.handedness}, ${row.file})`;

    it(`${name}: features match`, () => {
      const ts = normalize(row.landmarks, row.handedness, row.img_w, row.img_h);
      const maxDiff = Math.max(...row.py_features.map((v, k) => Math.abs(v - ts[k])));
      expect(maxDiff).toBeLessThan(TOL);
    });

    it(`${name}: prediction matches`, () => {
      const ts = normalize(row.landmarks, row.handedness, row.img_w, row.img_h);
      const p = predict(spec, ts);
      expect(p.label).toBe(row.py_prediction);
      expect(Math.abs(p.confidence - row.py_confidence)).toBeLessThan(TOL);
    });
  });
});
