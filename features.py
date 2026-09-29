"""Landmark normalization shared by training (Python) and, later, the web app (TypeScript port).

Raw MediaPipe landmarks depend on where the hand is in the frame, how close it is,
the camera's aspect ratio, and which hand is used. None of that says anything about
*which letter* is being signed, so we remove it:

  1. Convert to pixel-proportional units (x and y are normalized by different
     image dimensions, so raw values are distorted on a 16:9 camera).
  2. Translate so the wrist is the origin      -> position invariance
  3. Divide by wrist-to-middle-knuckle length   -> distance/scale invariance
  4. Mirror left hands onto right hands        -> handedness invariance

We deliberately do NOT normalize rotation: several ASL letters share a handshape
and differ only in orientation (e.g. K vs P, G vs Q, H vs U).
"""
import numpy as np

WRIST = 0
MIDDLE_MCP = 9
N_LANDMARKS = 21

RAW_COLS = [f"{a}{i}" for i in range(N_LANDMARKS) for a in ("x", "y", "z")]


def normalize_batch(landmarks, handedness, img_w, img_h):
    """landmarks: (n, 21, 3) or (n, 63) image-normalized coords from MediaPipe.
    handedness: array-like of "Left"/"Right" (length n, or a single string).
    img_w, img_h: image size in pixels (scalars or length-n arrays).
    Returns (n, 63) float32 feature matrix.
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(-1, N_LANDMARKS, 3).copy()
    n = lm.shape[0]
    w = np.broadcast_to(np.asarray(img_w, dtype=np.float64).reshape(-1, 1), (n, 1))
    h = np.broadcast_to(np.asarray(img_h, dtype=np.float64).reshape(-1, 1), (n, 1))

    # 1. pixel-proportional units (MediaPipe's z uses roughly the same scale as x)
    lm[:, :, 0] *= w
    lm[:, :, 1] *= h
    lm[:, :, 2] *= w

    # 2. wrist at the origin
    lm -= lm[:, WRIST:WRIST + 1, :]

    # 3. scale by palm size
    scale = np.linalg.norm(lm[:, MIDDLE_MCP, :], axis=1).reshape(-1, 1, 1)
    lm /= np.maximum(scale, 1e-6)

    # 4. mirror left hands
    hand = np.broadcast_to(np.asarray(handedness).reshape(-1), (n,))
    lm[hand == "Left", :, 0] *= -1

    return lm.reshape(n, N_LANDMARKS * 3).astype(np.float32)


def features_from_df(df):
    """Build the feature matrix from a DataFrame of rows written by capture.py."""
    return normalize_batch(
        df[RAW_COLS].to_numpy(),
        df["handedness"].to_numpy(),
        df["img_w"].to_numpy(),
        df["img_h"].to_numpy(),
    )
