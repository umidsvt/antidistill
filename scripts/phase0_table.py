#!/usr/bin/env python
"""Phase 0 results (results/m8_audit_design.md §24) from the stored per-problem files: the instruction-tuned student
untrained (I0) and trained on the teacher's solutions (U_inst), seeds 123-125, and the gap G = U_inst - I0 with a
seed-paired 95% interval (t, 2 degrees of freedom).

  setup 2   S1: the field's teacher (R1-Distill-Qwen-7B, 1,024 tokens), students scored at 1,024 tokens
            S2: R1-Distill-Qwen-32B at 8,192 tokens, students scored at 8,192 tokens
  puzzles   the step-2 test puzzles (§18), scored on the last answer (kk_eval) and on the first answer

    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/phase0_table.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
P0 = REPO / "outputs/phase0"
SEEDS = (123, 124, 125)
T975 = {1: 12.706, 2: 4.303}


def ci(values: list[float]) -> tuple[float, float, float]:
    m = statistics.mean(values)
    if len(values) < 2:
        return m, float("nan"), float("nan")
    h = T975[len(values) - 1] * statistics.stdev(values) / len(values) ** 0.5
    return m, m - h, m + h


def setup2_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    rows = d["rows"]
    tok = sorted(r["n_tokens"] for r in rows)
    return {"acc": 100 * sum(r["correct"] for r in rows) / len(rows),
            "finished": 100 * sum(r["finished"] for r in rows) / len(rows), "median_tokens": tok[len(tok) // 2]}


def first_answer(text: str) -> str:
    part = text.split("</think>", 1)[1] if "</think>" in text else text
    c = re.search(r"conclusion\s*:", part, re.I)
    return part[: c.end() + 600] if c else part


def kk_file(path: Path, test: dict) -> dict | None:
    from kk_pilot import parse_labels
    if not path.exists():
        return None
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 1000 and {r["id"] for r in rows} == set(test)
    first = sum(parse_labels(first_answer(r["text"]), test[r["id"]]["names"], False) == list(test[r["id"]]["solution"])
                for r in rows)
    return {"last": 100 * sum(r["correct"] for r in rows) / len(rows), "first": 100 * first / len(rows),
            "finished": 100 * sum(r["finish_reason"] == "stop" for r in rows) / len(rows),
            "mean_tokens": statistics.mean(r["tokens"] for r in rows)}


def main() -> int:
    out = {"setup2": {}, "puzzles": {}}
    for ds in ("gsm8k", "math"):
        for name, budget in (("S1", 1024), ("S2", 8192)):
            per = {}
            for s in SEEDS:
                per[s] = {"I0": setup2_file(P0 / f"I0/{ds}_s{s}_{budget}.json"),
                          "U_inst": setup2_file(P0 / f"{name}/{ds}_s{s}/test_{budget}.json")}
            gaps = [per[s]["U_inst"]["acc"] - per[s]["I0"]["acc"] for s in SEEDS if per[s]["I0"] and per[s]["U_inst"]]
            m, lo, hi = ci(gaps) if gaps else (float("nan"),) * 3
            out["setup2"][f"{ds}_{name}"] = {"budget": budget, "seeds": per, "G": gaps, "G_mean": m, "G_ci": [lo, hi]}
            print(f"{ds:5s} {name} @{budget}: " + "  ".join(
                f"s{s} I0 {per[s]['I0']['acc']:.2f} U {per[s]['U_inst']['acc']:.2f} (fin {per[s]['U_inst']['finished']:.1f}, "
                f"med {per[s]['U_inst']['median_tokens']})" for s in SEEDS if per[s]["I0"] and per[s]["U_inst"])
                  + f"  | G {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    test = {json.loads(line)["id"]: json.loads(line) for line in open(REPO / "data/kk/step2_test.jsonl")}
    st = REPO / "outputs/kk_step2/students"
    for label, pattern in (("I0", "untrained_inst_s{s}/test/results.jsonl"), ("U_inst", "Uinst_s{s}/test/results.jsonl")):
        out["puzzles"][label] = {s: kk_file(st / pattern.format(s=s), test) for s in SEEDS}
        print(f"puzzles {label}: " + "  ".join(
            f"s{s} " + (f"last {r['last']:.1f} first {r['first']:.1f} fin {r['finished']:.1f} tok {r['mean_tokens']:.0f}"
                        if r else "missing") for s, r in out["puzzles"][label].items()))
    (REPO / "outputs/phase0/table.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
