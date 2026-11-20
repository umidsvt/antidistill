#!/usr/bin/env python
"""Phase 0, S2 training files (results/m8_audit_design.md §24, decided with the user 2026-11-19 20:55): the final line
of each training target is built from the teacher's own answer when the teacher finished.

The setup appends a forced answer to every solution ("**Final Answer**\n\[\boxed{" + 32 greedy tokens). For the 32B
teacher on MATH that forced answer contradicts the teacher's own correct answer in 8.1% of cases ("D", "ANSWER", a
dropped sign). Rule:
- the teacher finished (stopped by itself) and wrote a boxed answer after "</think>": the final line becomes
  "**Final Answer**\n\[\boxed{<its answer>}\]" + the teacher's EOS string, the same format as before; its answer is
  every box on the last line after "</think>" that has one (several answers are joined as "\boxed{a}, \boxed{b}");
- otherwise (cut off at the limit, or no boxed answer of its own) the forced line is kept: it is the only answer.
No reference answer is used. The original file is kept as train_s<seed>_forced.json, and the forced line of each row
as "af_final_forced"; "final_from" says which rule applied. The new file replaces train_s<seed>.json atomically.

    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_own_final.py \
        --dir outputs/phase0/teacher32b_math
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EOS = "<｜end▁of▁sentence｜>"


def own_answer(trace: str) -> list[str] | None:
    """The boxed answers on the last line after "</think>" that has one. Several answers ("the solutions are \\boxed{81}
    and \\boxed{256}") are all kept, as the forced line writes them ("\\boxed{81}, \\boxed{256}")."""
    from clean_sweep.eval.correctness import _find_boxed_spans
    if "</think>" not in trace:
        return None
    post = trace.split("</think>")[-1]
    spans = _find_boxed_spans(post)   # over the whole text, so a box that spans lines (a matrix) is found
    if not spans:
        return None
    line_start = post.rfind("\n", 0, spans[-1][0]) + 1   # the line where the last box starts
    return [content for start, _, content in spans if start >= line_start]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", default="123,124,125")
    a = ap.parse_args()
    d = Path(a.dir)
    report = {}
    for seed in a.seeds.split(","):
        src, keep = d / f"train_s{seed}.json", d / f"train_s{seed}_forced.json"
        if not keep.exists():
            keep.write_text(src.read_text())   # the original, kept before anything is replaced
        rows = json.loads(keep.read_text())
        out, n_own, n_changed = [], 0, 0
        for r in rows:
            own = own_answer(r["trace"]) if r["finished"] else None
            forced = r["af_final_answer_only"]
            if own is not None:
                boxes = ", ".join(f"\\boxed{{{c}}}" for c in own)
                final = f"**Final Answer**\n\\[{boxes}\\]{EOS}"
                n_own += 1
                n_changed += final != forced
                out.append({**r, "af_final_answer_only": final, "af_final_forced": forced, "final_from": "own"})
            else:
                out.append({**r, "af_final_forced": forced, "final_from": "forced"})
        tmp = d / f".train_s{seed}.json.tmp"
        tmp.write_text(json.dumps(out, indent=1))
        os.replace(tmp, src)
        report[f"train_s{seed}"] = {"n": len(out), "final_from_own": n_own, "final_from_forced": len(out) - n_own,
                                    "final_line_changed": n_changed}
    (d / "own_final.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
