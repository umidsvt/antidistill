#!/usr/bin/env python
"""Recompute a hindsight sidecar's answer agreement, and report the audit.

The sidecar's `answer_matches_gold` is written during generation. A run made before
answers_agree() landed used a last-boxed-value comparison, which overcounts
disagreement ~2x on multi-part answers. This recomputes the field in place and prints
the breakdown that belongs in the write-up.

Three outcomes, and they mean different things:
  agree=True   the rewrite reaches the source trace's answer
  agree=False  a GENUINE disagreement -- flag it, keep the trace (policy (a))
  agree=None   unjudgeable: one side has no extractable \\boxed{}. This includes
               SOURCE-side defects, e.g. LIMO traces that trail off without boxing.

Usage:
    scripts/44_audit_answers.py data/defended/limo_hindsight_chat.json \
        --source data/raw/limo_v2.json [--write]
"""
import argparse, collections, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--source", default="data/raw/limo_v2.json")
    ap.add_argument("--write", action="store_true",
                    help="Update answer_matches_gold in the sidecar (default: report only).")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from antidistill.defenses.hindsight import answers_agree, load_grader
    math_equal, _ = load_grader()

    src = json.loads(Path(a.source).read_text(encoding="utf-8"))
    ds = json.loads(Path(a.dataset).read_text(encoding="utf-8"))
    meta_path = Path(a.dataset).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    changed = 0
    disagree, unjudgeable = [], []
    for m, rec in zip(meta, ds):
        i = m["index"]
        agree = answers_agree(src[i]["output"], rec["output"] or "", math_equal)
        if agree is False:
            disagree.append(i)
        elif agree is None:
            unjudgeable.append(i)
        if m.get("answer_matches_gold") != agree:
            changed += 1
        m["answer_matches_gold"] = agree

    n = len(meta)
    verd = collections.Counter(m["verdict"] for m in meta)
    print(f"{a.dataset}: {n} traces\n")
    print("judge verdicts:")
    for k, v in verd.most_common():
        print(f"  {k:>12s} {v:>5d}  ({100*v/n:.1f}%)")
    agree_n = n - len(disagree) - len(unjudgeable)
    print(f"\nanswer agreement with the source trace:")
    print(f"  agree        {agree_n:>5d}  ({100*agree_n/n:.1f}%)")
    print(f"  DISAGREE     {len(disagree):>5d}  ({100*len(disagree)/n:.1f}%)  {disagree[:15]}")
    print(f"  unjudgeable  {len(unjudgeable):>5d}  ({100*len(unjudgeable)/n:.1f}%)  "
          f"(no \\boxed on one side, often the SOURCE)")

    # The case the LLM judge exists for: right answer, reasoning rejected.
    rw = [m["index"] for m in meta
          if m["answer_matches_gold"] is True and m["verdict"] == "exhausted"]
    print(f"\nright answer but judge rejected the REASONING: {len(rw)}  {rw[:15]}")
    empty = [m["index"] for m in meta if m["verdict"] == "no_answer"]
    print(f"empty output (must be regenerated, not shippable): {len(empty)}  {empty}")

    if a.write:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsidecar updated ({changed} fields changed): {meta_path}")
    else:
        print(f"\n(dry run; {changed} fields would change -- pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
