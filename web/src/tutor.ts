export type Phase = "waiting" | "holding" | "success";

export const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split(""); // 24, no J/Z
const HOLD_FRAMES = 10;
const MIN_CONF = 0.8;
const COOLDOWN_MS = 1000;
const STORAGE_KEY = "asl-tutor-stats-v1";

let target = pick();
let hold = 0;
let cooldownUntil = 0;
let streak = 0;
let score = 0;
let version = 0;
let lastCorrect = "";   // the letter shown during the success flash

type LetterStats = { correct: number; attempts: number };
const stats: Record<string, LetterStats> = {};

function pick(): string {
  return LETTERS[Math.floor(Math.random() * LETTERS.length)];
}

export function onFrame(letter: string | null, conf: number, now: number) {
  if (now < cooldownUntil) return;
  if (letter === target && conf >= MIN_CONF) hold++;
  else hold = 0;
  if (hold >= HOLD_FRAMES) {
    recordResult(target, true);
    lastCorrect = target;
    cooldownUntil = now + COOLDOWN_MS;
    next();
  }
}

export function skip() {
  recordResult(target, false);
  cooldownUntil = 0;  // no success flash on a skip
  next();
}

function next() {
  hold = 0;
  let n;
  do { n = pick(); } while (n === target);
  target = n;
}

export function getState(now: number) {
  if (now < cooldownUntil) {
    return { phase: "success" as Phase, letter: lastCorrect, progress: 1 };
  }
  return {
    phase: (hold > 0 ? "holding" : "waiting") as Phase,
    letter: target,
    progress: hold / HOLD_FRAMES,
  };
}

function clearStats() {
  for (const l of LETTERS) stats[l] = { correct: 0, attempts: 0 };
  score = 0;
  streak = 0;
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ stats, score }));
  } catch {
    // Storage blocked or full: the stats just won't survive a reload.
  }
}

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    for (const l of LETTERS) {
      const s = data?.stats?.[l];
      if (s && Number.isFinite(s.correct) && Number.isFinite(s.attempts)) {
        stats[l] = { correct: s.correct, attempts: s.attempts };
      }
    }
    if (Number.isFinite(data?.score)) score = data.score;
  } catch {
    // Unreadable or blocked storage: start fresh.
  }
}

// Reset and overwrite any previous stats upon starting the program
clearStats();  // fill in all 24 letters at zero
load();

function recordResult(letter: string, correct: boolean) {
  stats[letter].attempts++;
  if (correct) {
    stats[letter].correct++;
    score++;
    streak++;
  } else {
    streak = 0;
  }
  version++;
  save();
}

export function resetStats() {
  clearStats();
  version++;
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
}

export function getStats() {
  return { stats, score, streak, version };
}
