import { getState, getStats, resetStats, LETTERS, skip } from "./tutor";

const tutorEl = document.getElementById("tutor")!;
const promptEl = document.getElementById("prompt")!;
const targetEl = document.getElementById("target")!;
const fillEl = document.getElementById("bar-fill")!;

const PROMPTS = {
  waiting: "Sign this letter",
  holding: "Hold it…",
  success: "Correct!",
};

export function renderTutor(now: number) {
  const s = getState(now);
  tutorEl.dataset.phase = s.phase;
  promptEl.textContent = PROMPTS[s.phase];
  targetEl.textContent = s.phase === "success" ? `✓ ${s.letter}` : s.letter;
  fillEl.style.transform = `scaleX(${s.progress})`;
}

const gridEl = document.getElementById("grid")!;
const scoreEl = document.getElementById("score")!;
const streakEl = document.getElementById("streak")!;

// Build the 24 cells once and keep references to them.
const cells: Record<string, { cell: HTMLDivElement; acc: HTMLSpanElement }> = {};
for (const l of LETTERS) {
  const cell = document.createElement("div");
  cell.className = "cell";
  const letter = document.createElement("span");
  letter.className = "cell-letter";
  letter.textContent = l;
  const acc = document.createElement("span");
  cell.append(letter, acc);
  gridEl.append(cell);
  cells[l] = { cell, acc };
}

document.getElementById("reset")!.addEventListener("click", () => {
  if (confirm("Reset all stats?")) resetStats();
});
document.getElementById("skip")!.addEventListener("click", () => {
  skip();
});

let lastVersion = -1;

export function renderStats() {
  const { stats, score, streak, version } = getStats();
  if (version === lastVersion) return;  // nothing changed since last draw
  lastVersion = version;

  scoreEl.textContent = String(score);
  streakEl.textContent = String(streak);

  for (const l of LETTERS) {
    const { correct, attempts } = stats[l];
    const { cell, acc } = cells[l];
    cell.title = `${correct} / ${attempts}`;
    if (attempts === 0) {
      acc.textContent = "–";
      cell.style.backgroundColor = "";  // back to the CSS default grey
    } else {
      const a = correct / attempts;
      acc.textContent = `${Math.round(a * 100)}%`;
      // Hue 0 is red, 120 is green; accuracy slides between them.
      cell.style.backgroundColor = `hsl(${a * 120}, 70%, 85%)`;
    }
  }
}
