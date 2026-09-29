"""End-of-day sanity check: how much data do you have, and does a simple model learn from it?

Usage: python quick_check.py

Splits by session (train on all but one session, test on the held-out one). A random
row split would put near-identical consecutive frames in both train and test and
report an inflated accuracy.
"""
import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from features import features_from_df

files = sorted(glob.glob("data/raw/*.csv"))
if not files:
    raise SystemExit("No data yet. Run capture.py first.")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print(f"{len(df)} samples from {len(files)} file(s)\n")
print("Samples per signer/session:")
print(df.groupby(["signer", "session"]).size().to_string(), "\n")
per_letter = df["label"].value_counts().sort_index()
print("Samples per letter:")
print(per_letter.to_string(), "\n")
low = per_letter[per_letter < 0.5 * per_letter.median()]
if len(low):
    print(f"WARNING: under-collected letters: {', '.join(low.index)}\n")

X = features_from_df(df)
y = df["label"].to_numpy()
group = (df["signer"].astype(str) + "_s" + df["session"].astype(str)).to_numpy()

if len(set(group)) >= 2:
    held_out = sorted(set(group))[-1]
    train, test = group != held_out, group == held_out
    print(f"Train: everything except {held_out} | Test: {held_out}")
    X_tr, X_te, y_tr, y_te = X[train], X[test], y[train], y[test]
else:
    print("Only one session so far: using a random split (accuracy will look TOO good).")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print(f"Logistic regression baseline accuracy: {accuracy_score(y_te, pred):.1%}\n")

labels = sorted(set(y_te) | set(pred))
cm = confusion_matrix(y_te, pred, labels=labels)
np.fill_diagonal(cm, 0)
pairs = sorted(
    ((cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]),
    reverse=True,
)[:8]
if pairs:
    print("Most common mistakes (true -> predicted):")
    for count, t, p in pairs:
        print(f"  {t} -> {p}: {count}")
