"""Run with: pytest -q
These tests prove the normalization does what the docstring claims."""
import numpy as np

from features import normalize_batch

W, H = 1280, 720
rng = np.random.default_rng(0)


def random_hand():
    # plausible hand: points clustered around the middle of the frame
    pts = rng.uniform(0.35, 0.65, size=(21, 3))
    pts[:, 2] = rng.uniform(-0.1, 0.1, size=21)
    return pts


def test_translation_invariance():
    hand = random_hand()
    moved = hand.copy()
    moved[:, 0] += 0.2
    moved[:, 1] -= 0.15
    a = normalize_batch(hand, "Right", W, H)
    b = normalize_batch(moved, "Right", W, H)
    assert np.allclose(a, b, atol=1e-5)


def test_scale_invariance():
    hand = random_hand()
    wrist = hand[0].copy()
    bigger = wrist + (hand - wrist) * 1.8  # hand moved closer to the camera
    a = normalize_batch(hand, "Right", W, H)
    b = normalize_batch(bigger, "Right", W, H)
    assert np.allclose(a, b, atol=1e-5)


def test_left_hand_mirrors_onto_right():
    hand = random_hand()
    mirrored = hand.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    a = normalize_batch(hand, "Right", W, H)
    b = normalize_batch(mirrored, "Left", W, H)
    assert np.allclose(a, b, atol=1e-5)


def test_wrist_is_origin_and_palm_is_unit_length():
    f = normalize_batch(random_hand(), "Right", W, H).reshape(21, 3)
    assert np.allclose(f[0], 0)
    assert np.isclose(np.linalg.norm(f[9]), 1.0, atol=1e-5)


def test_batch_matches_single():
    hands = np.stack([random_hand() for _ in range(5)])
    handed = np.array(["Right", "Left", "Right", "Left", "Right"])
    batch = normalize_batch(hands, handed, W, H)
    for i in range(5):
        assert np.allclose(batch[i], normalize_batch(hands[i], handed[i], W, H)[0])
