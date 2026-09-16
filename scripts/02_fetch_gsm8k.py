#!/usr/bin/env python
"""Fetch GSM8K and convert it to the eval harness's format.

Why this exists: `third_party/kim_eval/data/grade_school_math/` is NOT GSM8K despite the name --
it is 210 rows of Chinese elementary maths. GSM8K has to be added.

Design notes (results/gsm8k_proposal.md):
  * EVALUATION ONLY. The training set stays fixed at LIMO's 800 problems; "same problems, only the
    traces differ" is the whole controlled-comparison design.
  * A fixed 500-problem subsample by default, seed pinned, so it carries the same weight as
    MATH500 and costs the same (~3 h per condition vs ~8 h for the full 1,319).
  * GSM8K gold answers sit after '####' and carry thousands separators ("1,000"). Those must be
    stripped or the grader compares "1,000" against the model's "1000" and scores it wrong.

Usage:
    scripts/02_fetch_gsm8k.py                 # 500-problem subsample, seed 0
    scripts/02_fetch_gsm8k.py --n 0           # the full 1,319
"""
import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "third_party" / "kim_eval" / "data" / "gsm8k"
PROMPT = REPO / "third_party" / "kim_eval" / "prompts" / "qwen-instruct" / "gsm8k.py"


def gold(answer_field: str) -> str | None:
    """GSM8K stores the worked solution then '#### <final answer>'."""
    if "####" not in answer_field:
        return None
    a = answer_field.split("####")[-1].strip()
    a = a.replace(",", "").replace("$", "").rstrip(".")
    return a or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="subsample size; 0 = keep all 1,319")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"loaded {len(ds)} GSM8K test problems")

    rows = []
    for ex in ds:
        g = gold(ex["answer"])
        if g is None:
            continue
        rows.append({"problem": ex["question"].strip(), "answer": g})
    print(f"  with an extractable gold answer: {len(rows)}")

    if a.n and a.n < len(rows):
        random.Random(a.seed).shuffle(rows)
        rows = rows[:a.n]
        print(f"  subsampled to {len(rows)} (seed {a.seed})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "test.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_DIR / 'test.jsonl'}")

    # Byte-identical to math.py. Any deviation breaks comparability with the other four
    # benchmarks AND the train/eval format match every fine-tuned condition depends on.
    math_py = (PROMPT.parent / "math.py").read_text(encoding="utf-8")
    PROMPT.write_text(math_py, encoding="utf-8")
    print(f"wrote {PROMPT} (copied verbatim from math.py)")

    nums = sum(bool(re.fullmatch(r"-?\d+(\.\d+)?", r["answer"])) for r in rows)
    print(f"\nsanity: {nums}/{len(rows)} gold answers are plain numbers")
    print("examples:", [r["answer"] for r in rows[:8]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
