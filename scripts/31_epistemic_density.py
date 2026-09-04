#!/usr/bin/env python
"""Measure epistemic-token density in generated traces.

Separates two questions that accuracy alone conflates:
  1. Did SFT actually instill epistemic verbalization? (this script)
  2. Did that translate into accuracy?                  (the eval)

Kim et al.'s nine epistemic tokens (paper section 4.2), plus the bypass phrases they observed
when those tokens are suppressed (Appendix H.1), so a model that routes around the vocabulary
is not scored as epistemic-free.

Usage:
  .venv-infer/bin/python scripts/31_epistemic_density.py outputs/base/.../x.jsonl outputs/limo_ep15/.../y.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Paper section 4.2, with reported co-occurrence frequency in LRM traces.
EPISTEMIC_TOKENS = ["wait", "hmm", "perhaps", "maybe", "actually",
                    "alternatively", "seems", "might", "check"]

# Appendix H.1 — substitutions observed when the nine tokens are logit-banned.
BYPASS_PHRASES = ["hold on", "let me verify", "let me re-?examine", "on closer look",
                  "i realize", "another way to see", "it'?s possible that",
                  "that'?s not quite right", "let me reconsider", "double-?check"]

TOK_RE = {t: re.compile(rf"\b{t}\b", re.I) for t in EPISTEMIC_TOKENS}
BYP_RE = [re.compile(p, re.I) for p in BYPASS_PHRASES]
WORD_RE = re.compile(r"\S+")


def analyse(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    per_tok = {t: 0 for t in EPISTEMIC_TOKENS}
    bypass = 0
    words = 0
    n_resp = 0
    per_resp_epistemic = []

    for r in rows:
        for resp in r["generated_responses"]:
            n_resp += 1
            words += len(WORD_RE.findall(resp))
            c = 0
            for t, rx in TOK_RE.items():
                k = len(rx.findall(resp))
                per_tok[t] += k
                c += k
            bypass += sum(len(rx.findall(resp)) for rx in BYP_RE)
            per_resp_epistemic.append(c)

    total = sum(per_tok.values())
    return {
        "path": path,
        "n_responses": n_resp,
        "words": words,
        "per_tok": per_tok,
        "total_epistemic": total,
        "bypass": bypass,
        "per_response": total / max(n_resp, 1),
        "per_1k_words": 1000 * total / max(words, 1),
        "mean_words": words / max(n_resp, 1),
        "responses_with_any": sum(1 for c in per_resp_epistemic if c > 0),
    }


def main() -> int:
    paths = [Path(p) if Path(p).is_absolute() else REPO / p for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2

    results = [analyse(p) for p in paths]

    print(f"{'metric':<28}" + "".join(f"{p.parts[-5] if len(p.parts)>5 else p.stem:>18}" for p in paths))
    print("-" * (28 + 18 * len(paths)))

    def row(label, fn, fmt="{:.2f}"):
        print(f"{label:<28}" + "".join(f"{fmt.format(fn(r)):>18}" for r in results))

    row("responses", lambda r: r["n_responses"], "{:.0f}")
    row("mean words/response", lambda r: r["mean_words"], "{:.0f}")
    row("epistemic tokens total", lambda r: r["total_epistemic"], "{:.0f}")
    row("epistemic per response", lambda r: r["per_response"])
    row("epistemic per 1k words", lambda r: r["per_1k_words"])
    row("responses w/ >=1 (of n)", lambda r: r["responses_with_any"], "{:.0f}")
    row("bypass phrases", lambda r: r["bypass"], "{:.0f}")
    print()
    print(f"{'per-token counts':<28}" + "".join(f"{p.parts[-5] if len(p.parts)>5 else p.stem:>18}" for p in paths))
    print("-" * (28 + 18 * len(paths)))
    for t in EPISTEMIC_TOKENS:
        print(f"  {t:<26}" + "".join(f"{r['per_tok'][t]:>18}" for r in results))

    print("\nReference: LIMO training data averages ~77 'wait' per response (paper Figure 6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
