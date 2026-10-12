"""Train the real model: a small neural network (MLP) on normalized hand landmarks.

Usage:
    python train.py                  # auto: cross-signer if 2+ signers, else cross-session
    python train.py --split session  # hold out one signer+session at a time
    python train.py --split signer   # hold out one person at a time

What it does:
  1. Loads every CSV in data/raw/ and builds features with features.py (same as quick_check.py).
  2. Checks that each recording's handedness labels are consistent (see check_handedness).
  3. Evaluates with leave-one-group-out: every session (or signer) is held out once, a model is
     trained on everything else and tested on the held-out group. The MLP and the logistic
     regression baseline from quick_check.py run on exactly the same splits, so their numbers
     are directly comparable.
  4. Saves confusion_matrix.png from all held-out predictions pooled together.
  5. Trains one final MLP on ALL the data and exports it to model.json for the web app, then
     re-runs the exported numbers in plain numpy to prove they reproduce sklearn's predictions.

Why leave-one-group-out instead of quick_check.py's single held-out group: with only a few
sessions, one held-out group gives a noisy number. Holding out each one in turn and averaging
gives a steadier estimate and shows which recordings are hard.
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import N_LANDMARKS, features_from_df

HIDDEN_LAYERS = (128, 64)  # two hidden layers: 63 inputs -> 128 -> 64 -> 24 letters
SEED = 0                   # fixed so runs are repeatable
MODEL_PATH = "model.json"
CM_PATH = "confusion_matrix.png"


# ---------------------------------------------------------------- models

def make_mlp():
    """StandardScaler + MLP.

    The scaler shifts each of the 63 features to mean 0 and spread 1. Neural networks train
    much better on inputs of similar size; logistic regression cares less, which is why
    quick_check.py skips it. The scaler's numbers are exported with the model, because the
    web app must apply the exact same shift.
    """
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",     # max(0, x): lets the network learn curved decision boundaries
            alpha=1e-1,            # L2 penalty on weights: discourages memorizing one session
            max_iter=500,
            early_stopping=True,   # stop when a 10% validation slice stops improving
            n_iter_no_change=20,
            random_state=SEED,
        ),
    )


def make_baseline():
    """Identical to quick_check.py, so the comparison is apples to apples."""
    return LogisticRegression(max_iter=2000)


# ---------------------------------------------------------------- data

def load_data():
    files = sorted(glob.glob("data/raw/*.csv"))
    if not files:
        raise SystemExit("No data yet. Run capture.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(df)} samples from {len(files)} file(s)\n")
    print("Samples per signer/session:")
    print(df.groupby(["signer", "session"]).size().to_string(), "\n")
    return df


def check_handedness(df):
    """Warn when a recording mixes Left and Right labels.

    capture.py saves MediaPipe's per-frame handedness guess, and features.py mirrors every
    "Left" frame. If the guess flips on a few frames, those frames get mirrored while their
    neighbours don't, and the model sees a mirror-image letter under the right label.
    """
    print("Handedness per recording:")
    problems = False
    for (signer, session), g in df.groupby(["signer", "session"]):
        counts = g["handedness"].value_counts()
        line = ", ".join(f"{k} {v}" for k, v in counts.items())
        print(f"  {signer}_s{session}: {line}")
        if len(counts) > 1:
            minority = counts.index[-1]
            share = counts.iloc[-1] / counts.sum()
            if share > 0.01:
                problems = True
                letters = g.loc[g["handedness"] == minority, "label"].value_counts()
                top = ", ".join(f"{k} ({v})" for k, v in letters.head(6).items())
                print(f"    WARNING: {share:.1%} labelled {minority}, mostly in: {top}")
    if problems:
        print("  Mixed labels inside a recording usually mean MediaPipe's guess flipped on some frames.")
    print()


# ---------------------------------------------------------------- evaluation

def leave_one_group_out(X, y, groups, make_model):
    """Train on all groups but one, test on that one, repeat for every group."""
    per_group, y_true, y_pred = {}, [], []
    for g in sorted(set(groups)):
        test = groups == g
        model = make_model().fit(X[~test], y[~test])
        pred = model.predict(X[test])
        per_group[g] = accuracy_score(y[test], pred)
        y_true.append(y[test])
        y_pred.append(pred)
    return per_group, np.concatenate(y_true), np.concatenate(y_pred)


def random_split(X, y, make_model):
    """Fallback with only one group. Optimistic: neighbouring frames land in both halves."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    pred = make_model().fit(X_tr, y_tr).predict(X_te)
    return {"random 25%": accuracy_score(y_te, pred)}, y_te, pred


def top_mistakes(y_true, y_pred, labels, k=8):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    pairs = [(cm[i, j], labels[i], labels[j])
             for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]]
    return sorted(pairs, reverse=True)[:k]


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # write a file; no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"(Skipped {path}: run `pip install matplotlib` to get the image.)\n")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_totals, out=np.zeros_like(cm), where=row_totals > 0) * 100

    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if pct[i, j] >= 1:  # label every cell with at least 1% of that row
                color = "white" if pct[i, j] > 60 else ("black" if i == j else "crimson")
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=7, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path} (each row sums to 100%; red numbers are mistakes)\n")


# ---------------------------------------------------------------- export

def export_model(model, path):
    """Write everything the web app needs to reproduce predictions, as plain numbers.

    Prediction in the app is:
        x = normalize_batch(...)              # features.py, ported to TypeScript
        a = (x - scaler.mean) / scaler.scale
        for each layer: a = a @ W + b, then relu (hidden) or softmax (last)
        letter = labels[argmax(a)]
    """
    scaler = model.named_steps["standardscaler"]
    mlp = model.named_steps["mlpclassifier"]
    n_layers = len(mlp.coefs_)
    layers = []
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        layers.append({
            "W": np.round(W, 7).tolist(),   # shape [inputs][outputs]
            "b": np.round(b, 7).tolist(),
            "activation": "softmax" if i == n_layers - 1 else mlp.activation,
        })
    spec = {
        "format": "asl-mlp-v1",
        "labels": [str(c) for c in mlp.classes_],
        "n_features": int(scaler.mean_.shape[0]),
        "feature_order": [f"{a}{i}" for i in range(N_LANDMARKS) for a in ("x", "y", "z")],
        "normalization": "features.normalize_batch: pixel units, wrist origin, palm scale, mirror Left",
        "scaler": {"mean": np.round(scaler.mean_, 7).tolist(), "scale": np.round(scaler.scale_, 7).tolist()},
        "layers": layers,
    }
    with open(path, "w") as f:
        json.dump(spec, f)
    return spec


def predict_from_spec(spec, X):
    """Plain-numpy forward pass using only model.json: a blueprint for the TypeScript version."""
    a = (X - np.array(spec["scaler"]["mean"])) / np.array(spec["scaler"]["scale"])
    for layer in spec["layers"]:
        a = a @ np.array(layer["W"]) + np.array(layer["b"])
        if layer["activation"] == "relu":
            a = np.maximum(a, 0)
        # softmax doesn't change which output is largest, so argmax can skip it
    return np.array(spec["labels"])[a.argmax(axis=1)]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["auto", "session", "signer"], default="auto")
    args = ap.parse_args()

    df = load_data()
    check_handedness(df)

    X = features_from_df(df)
    y = df["label"].astype(str).to_numpy()
    labels = sorted(set(y))

    split = args.split
    if split == "auto":
        split = "signer" if df["signer"].nunique() >= 2 else "session"
    if split == "signer":
        groups = df["signer"].astype(str).to_numpy()
    else:
        groups = (df["signer"].astype(str) + "_s" + df["session"].astype(str)).to_numpy()

    if len(set(groups)) >= 2:
        print(f"Evaluation: cross-{split}, each of {len(set(groups))} groups held out in turn "
              "(this trains several models, give it a minute)\n")
        mlp_acc, y_true, y_pred = leave_one_group_out(X, y, groups, make_mlp)
        base_acc, _, _ = leave_one_group_out(X, y, groups, make_baseline)
    else:
        print(f"Only one {split} so far: using a random split (accuracy will look TOO good).\n")
        mlp_acc, y_true, y_pred = random_split(X, y, make_mlp)
        base_acc, _, _ = random_split(X, y, make_baseline)

    width = max(len(g) for g in mlp_acc)
    print(f"{'held out':<{width}}   MLP    baseline")
    for g in mlp_acc:
        print(f"{g:<{width}}  {mlp_acc[g]:5.1%}   {base_acc[g]:5.1%}")
    mlp_mean, base_mean = np.mean(list(mlp_acc.values())), np.mean(list(base_acc.values()))
    print(f"{'MEAN':<{width}}  {mlp_mean:5.1%}   {base_mean:5.1%}\n")

    print("MLP's most common mistakes (true -> predicted), all held-out groups pooled:")
    for count, t, p in top_mistakes(y_true, y_pred, labels):
        print(f"  {t} -> {p}: {count}")
    print()

    save_confusion_matrix(y_true, y_pred, labels,
                          f"MLP, cross-{split}: mean accuracy {mlp_mean:.1%}", CM_PATH)

    # The accuracy above came from models that never saw their test group. The model we ship
    # is trained on everything, so it should do at least as well on new people.
    final = make_mlp().fit(X, y)
    spec = export_model(final, MODEL_PATH)
    agree = np.mean(predict_from_spec(spec, X) == final.predict(X))
    print(f"Exported {MODEL_PATH} ({len(spec['labels'])} letters, layers "
          f"{spec['n_features']} -> {' -> '.join(str(len(l['b'])) for l in spec['layers'])})")
    print(f"Export check: plain-numpy predictions match sklearn on {agree:.2%} of samples")
    if agree < 0.999:
        raise SystemExit("Export check FAILED: model.json does not reproduce the trained model.")


if __name__ == "__main__":
    main()
