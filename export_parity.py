"""Session 5 parity test, Python side.

Picks 20 recorded frames (one per letter, 20 different letters), runs them through the
real Python pipeline (features.py + model.json), and saves the raw landmarks together
with the Python features and predictions to web/tests/parity_rows.json.

The TypeScript test (web/tests/parity.test.ts) then feeds the same raw landmarks through
features.ts + mlp.ts and checks it gets the same answers.

Run from the project folder:  python export_parity.py
Re-run it whenever you retrain (model.json changes) or change features.py.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from features import N_LANDMARKS, RAW_COLS, features_from_df
from train import predict_from_spec

N_ROWS = 20
SEED = 0
MODEL_PATH = "model.json"
OUT_PATH = os.path.join("web", "tests", "parity_rows.json")


def probs_from_spec(spec, X):
    """Same forward pass as predict_from_spec, but returns the softmax probabilities."""
    act = {
        "relu": lambda a: np.maximum(a, 0),
        "tanh": np.tanh,
        "logistic": lambda a: 1 / (1 + np.exp(-a)),
        "identity": lambda a: a,
    }
    a = (X - np.asarray(spec["scaler"]["mean"])) / np.asarray(spec["scaler"]["scale"])
    for layer in spec["layers"]:
        a = a @ np.asarray(layer["W"]) + np.asarray(layer["b"])
        if layer["activation"] == "softmax":
            a = np.exp(a - a.max(axis=1, keepdims=True))
            a /= a.sum(axis=1, keepdims=True)
        else:
            a = act[layer["activation"]](a)
    return a


def main():
    paths = sorted(glob.glob("data/raw/*.csv"))
    if not paths:
        raise SystemExit("No CSVs in data/raw/. Run this from the project folder.")
    df = pd.concat([pd.read_csv(p).assign(_file=os.path.basename(p)) for p in paths], ignore_index=True)

    with open(MODEL_PATH) as f:
        spec = json.load(f)

    # One random frame per letter, then 20 of those letters: varied letters, files and lighting.
    rows = (
        df.groupby("label", group_keys=False)
        .sample(n=1, random_state=SEED)
        .sample(n=min(N_ROWS, df["label"].nunique()), random_state=SEED)
        .reset_index(drop=True)
    )

    X = features_from_df(rows)
    probs = probs_from_spec(spec, X)
    labels = np.asarray(spec["labels"])
    preds = labels[probs.argmax(axis=1)]

    # Sanity check: must agree with train.py's own forward pass.
    if not np.array_equal(preds, predict_from_spec(spec, X)):
        raise SystemExit("probs_from_spec disagrees with predict_from_spec; tell Claude.")

    out = []
    for i, r in rows.iterrows():
        raw = r[RAW_COLS].to_numpy(dtype=float).reshape(N_LANDMARKS, 3)
        out.append({
            "file": r["_file"],
            "true_label": str(r["label"]),
            "handedness": str(r["handedness"]),
            "img_w": float(r["img_w"]),
            "img_h": float(r["img_h"]),
            "landmarks": [{"x": x, "y": y, "z": z} for x, y, z in raw.tolist()],
            "py_features": X[i].astype(float).tolist(),
            "py_prediction": str(preds[i]),
            "py_confidence": float(probs[i].max()),
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)

    correct = sum(o["py_prediction"] == o["true_label"] for o in out)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(f"Python predictions: {correct}/{len(out)} match the recorded label (that's the model's")
    print("accuracy on these frames, not the parity test; the parity test is Python vs TypeScript).")
    for o in out:
        mark = " " if o["py_prediction"] == o["true_label"] else "x"
        print(f"  {mark} {o['true_label']} -> {o['py_prediction']}  ({o['py_confidence']:.2f})  {o['file']}")


if __name__ == "__main__":
    main()
