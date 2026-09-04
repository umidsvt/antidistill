#!/usr/bin/env python
"""Task 0.2 — validate our grader against Kim et al.'s own generations.

Re-grades the reference AIME24 outputs shipped in `reference/` with OUR installed
`utils/grader.py` + `utils/parser.py` and asserts we recover the published numbers:

    Qwen2.5-7B base : 4/30 = 13.33%
    Qwen2.5-7B LIMO : 8/30 = 26.67%

Any per-problem disagreement with the stored `is_correct` field is a grader or
sympy-version problem, not a modelling problem, and must be resolved here —
before a single GPU-hour is spent.

Run:  .venv-infer/bin/python scripts/00_validate_grader.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "kim_eval"))
warnings.filterwarnings("ignore")

from utils.grader import check_is_correct  # noqa: E402
from utils.parser import extract_answer  # noqa: E402

# (label, path relative to repo root, expected correct count, total)
CASES = [
    (
        "Qwen2.5-7B base",
        "reference/kim_example_eval_outputs/base/Qwen2.5-7B/aime/test_qwen-instruct_t0.0_k1_s0_e30.jsonl",
        4,
        30,
    ),
    (
        "Qwen2.5-7B LIMO",
        "reference/kim_example_eval_outputs/limo/Qwen2.5-7B/aime/test_qwen-instruct_t0.0_k1_s0_e30.jsonl",
        8,
        30,
    ),
]

# eval.py calls `extract_answer(response, args.data_name)`. The second positional
# parameter is actually `use_last_number`, so passing "aime" just means True.
# We replicate the call exactly rather than "fixing" it, so our grading matches theirs.
DATA_NAME = "aime"


def regrade(path: Path) -> tuple[int, int, list[dict]]:
    """Re-grade one reference file. Returns (n_correct, n_total, disagreements)."""
    n_correct = 0
    n_total = 0
    disagreements = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_total += 1

        gold = row["gold_answer"]
        ours = [
            check_is_correct(extract_answer(resp, DATA_NAME), gold)
            for resp in row["generated_responses"]
        ]
        ours_any = any(ours)
        if ours_any:
            n_correct += 1

        theirs_any = bool(row["is_correct"])
        if ours_any != theirs_any:
            disagreements.append(
                {
                    "id": row.get("id"),
                    "gold": gold,
                    "their_verdict": theirs_any,
                    "our_verdict": ours_any,
                    "their_extracted": row.get("generated_answers"),
                    "our_extracted": [
                        extract_answer(r, DATA_NAME) for r in row["generated_responses"]
                    ],
                }
            )

    return n_correct, n_total, disagreements


def main() -> int:
    failed = False
    report: list[str] = ["# Task 0.2 — grader validation\n"]

    for label, rel, exp_correct, exp_total in CASES:
        path = REPO / rel
        if not path.exists():
            print(f"MISSING  {rel}")
            return 2

        n_correct, n_total, disagreements = regrade(path)
        pct = 100.0 * n_correct / n_total
        ok = (n_correct, n_total) == (exp_correct, exp_total) and not disagreements
        status = "PASS" if ok else "FAIL"
        failed |= not ok

        line = (
            f"{status}  {label:18s} {n_correct}/{n_total} = {pct:5.2f}%  "
            f"(expected {exp_correct}/{exp_total} = {100.0*exp_correct/exp_total:.2f}%)"
        )
        print(line)
        report.append(f"- `{status}` **{label}** — {n_correct}/{n_total} = {pct:.2f}% "
                      f"(expected {exp_correct}/{exp_total})")

        for d in disagreements:
            msg = (
                f"    problem id={d['id']} gold={d['gold']!r} "
                f"theirs={d['their_verdict']} ours={d['our_verdict']} "
                f"their_extracted={d['their_extracted']!r} our_extracted={d['our_extracted']!r}"
            )
            print(msg)
            report.append(f"  - disagreement: {msg.strip()}")

    out = REPO / "results" / "task0.2_grader_validation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO)}")

    if failed:
        print("\nGRADER VALIDATION FAILED — resolve before training anything.")
        return 1
    print("\nGrader reproduces Kim et al.'s numbers exactly. Safe to proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
