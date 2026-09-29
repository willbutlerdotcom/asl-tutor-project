"""Day 1: collect hand-landmark samples for the 24 static ASL fingerspelling letters.

Usage:
    python capture.py --signer alex --session 1

Controls (click the video window first so it receives key presses):
    SPACE   record the current letter (3 s countdown, then --samples frames)
    n / b   next / previous letter
    u       undo the last recorded batch (use this whenever you mislabel)
    q       quit

Each run writes data/raw/<signer>_s<session>_<timestamp>.csv, re-saved after every batch.
While recording, move your hand a little: closer/farther, slight tilts, different
spots in the frame. Variety now = a model that generalizes later.

Frames go to MediaPipe UNFLIPPED (the browser app will do the same); the image is
mirrored only for display.
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # J and Z involve motion, handled later
MODEL_PATH = "models/hand_landmarker.task"
COUNTDOWN_S = 3.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]

HEADER = (
    ["signer", "session", "label", "t_ms", "handedness", "hand_score", "img_w", "img_h"]
    + [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
    + [f"w{a}{i}" for i in range(21) for a in ("x", "y", "z")]  # world coords (meters), kept for later experiments
)


def save(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def draw_hand(img, landmarks):
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(img, p, 4, (0, 0, 255), -1)


def put(img, text, y, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True, help="short ID for the person signing")
    ap.add_argument("--session", required=True, help="session number (new lighting/position = new session)")
    ap.add_argument("--samples", type=int, default=150, help="samples per batch")
    ap.add_argument("--every", type=int, default=2, help="keep every Nth frame (reduces near-duplicates)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    out = Path("data/raw") / f"{args.signer}_s{args.session}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera 1, or grant camera permission to your terminal.")

    rows, batch_starts = [], []
    counts = Counter()
    idx, state = 0, "idle"  # idle | countdown | recording
    countdown_end, recorded, frame_i = 0.0, 0, 0
    fps, fps_n, fps_t = 0.0, 0, time.monotonic()
    last_ts = -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            t_ms = int(time.monotonic() * 1000)
            if t_ms <= last_ts:  # detect_for_video needs strictly increasing timestamps
                t_ms = last_ts + 1
            last_ts = t_ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand:
                draw_hand(frame, hand)

            now = time.monotonic()
            if state == "countdown" and now >= countdown_end:
                state, recorded = "recording", 0
                batch_starts.append(len(rows))

            if state == "recording" and hand:
                frame_i += 1
                if frame_i % args.every == 0:
                    handed = result.handedness[0][0]
                    world = result.hand_world_landmarks[0]
                    row = [args.signer, args.session, LETTERS[idx], t_ms,
                           handed.category_name, round(handed.score, 4), w, h]
                    row += [round(v, 6) for p in hand for v in (p.x, p.y, p.z)]
                    row += [round(v, 6) for p in world for v in (p.x, p.y, p.z)]
                    rows.append(row)
                    recorded += 1
                    if recorded >= args.samples:
                        save(out, rows)
                        counts[LETTERS[idx]] += recorded
                        state = "idle"
                        idx = min(idx + 1, len(LETTERS) - 1)

            # FPS (updated twice a second)
            fps_n += 1
            if now - fps_t >= 0.5:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

            view = cv2.flip(frame, 1)  # mirror for display only
            put(view, f"Letter: {LETTERS[idx]}   ({counts[LETTERS[idx]]} saved)", 40, 1.1, (0, 255, 255))
            if state == "countdown":
                put(view, f"Get ready... {countdown_end - now:.1f}", 80, 0.9, (0, 200, 255))
            elif state == "recording":
                put(view, f"RECORDING {recorded}/{args.samples}  - move your hand a little", 80, 0.9, (0, 0, 255))
            else:
                put(view, "SPACE record | n/b next/prev | u undo | q quit", 80)
            if not hand:
                put(view, "No hand detected", 120, 0.8, (0, 0, 255))
            done = sum(1 for L in LETTERS if counts[L] > 0)
            put(view, f"{fps:.0f} FPS   letters done: {done}/{len(LETTERS)}   total rows: {len(rows)}", h - 20, 0.6)
            cv2.imshow("capture", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if state != "idle":
                continue
            if key == ord(" "):
                state, countdown_end = "countdown", time.monotonic() + COUNTDOWN_S
            elif key == ord("n"):
                idx = min(idx + 1, len(LETTERS) - 1)
            elif key == ord("b"):
                idx = max(idx - 1, 0)
            elif key == ord("u") and batch_starts:
                start = batch_starts.pop()
                removed = rows[start:]
                del rows[start:]
                label = removed[0][2]
                counts[label] -= len(removed)
                idx = LETTERS.index(label)
                save(out, rows)
                print(f"Undid {len(removed)} samples of '{label}'")

    cap.release()
    cv2.destroyAllWindows()
    if rows:
        save(out, rows)
        print(f"Saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
