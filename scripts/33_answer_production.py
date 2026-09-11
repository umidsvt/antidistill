#!/usr/bin/env python
"""Answer-production and termination audit — the `accuracy | finished` split of CLAUDE.md §7.1.

Raw pass@1 silently conflates several different behaviours, and this arm turned up a failure
mode §7.1 does not describe. Three quantities, all different:

  pass@1       correct / total                    (what eval_table.md reports)
  answered     a parseable answer was extracted
  terminated   generation stopped on its own rather than hitting the token cap

§7.1 assumes `answered` and `terminated` move together: truncation cuts the trailing
<|im_end|> off the longest traces, the model never learns to stop, and so it produces no
answer. That is one failure mode. **We measured a second one**: a model that answers
correctly and *then* fails to emit EOS, degenerating into a repetition loop until the cap.
`hindsight_lora16k` answers 97.8% of problems but terminates on only 10.3% of them —
reading `answered` alone would call that the healthiest model in the arm when it is the least
usable. The repeated text is not in the training data (zero occurrences of it in either
`data/raw/limo_v2.json` or `data/defended/limo_hindsight_ds32b.json`); it is the base model's
pretraining distribution surfacing once EOS fails.

`terminated` therefore has to be recovered by re-tokenizing, because the harness does not
store vLLM's finish_reason. That makes this script slower than a pure-JSON pass, which is why
the tokenizer is loaded lazily and can be skipped with --no-termination.

Usage:
    .venv-infer/bin/python scripts/33_answer_production.py
    .venv-infer/bin/python scripts/33_answer_production.py --no-termination   # fast, pass@1 only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def is_answered(row: dict) -> bool:
    """True if a parseable answer was extracted.

    NOTE: this is NOT the same as terminating. A model can emit \\boxed{} correctly and then
    fail to produce EOS, degenerating into a repetition loop until it hits the token cap.
    Report both — `answered` alone reads as healthy behaviour when it is not.
    """
    answers = row.get("generated_answers") or []
    if not answers:
        return False
    a = answers[0]
    return a is not None and str(a).strip() not in ("", "None")


def response_text(row: dict) -> str:
    responses = row.get("generated_responses") or []
    return responses[0] if responses else ""


def terminated_flags(texts: list[str], tokenizer, cap: int) -> list[bool]:
    """True where generation stopped on its own rather than being cut off at the token cap.

    Recovered by re-tokenizing: a response at (or within rounding distance of) `cap` tokens
    ran out of budget. `cap` must match the --max_tokens the eval ran with (32768 in
    scripts/10_eval.sh).
    """
    lens = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    return [L < cap - 2 for L in lens]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(REPO / "outputs"))
    ap.add_argument("--out", default=str(REPO / "results" / "answer_production.md"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--cap", type=int, default=32768, help="must match eval --max_tokens")
    ap.add_argument("--no-termination", action="store_true",
                    help="skip the re-tokenization pass (much faster, drops `terminated`)")
    args = ap.parse_args()

    outputs = Path(args.outputs)
    if not outputs.is_dir():
        print(f"no outputs directory at {outputs} — run the evals first")
        return 1

    tok = None
    if not args.no_termination:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    records = []
    for f in sorted(outputs.rglob("*.jsonl")):
        rel = f.relative_to(outputs)
        if len(rel.parts) < 3:
            continue
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        n = len(rows)
        answered = [is_answered(r) for r in rows]
        correct = [bool(r.get("is_correct")) for r in rows]
        term = terminated_flags([response_text(r) for r in rows], tok, args.cap) if tok else None
        n_ans, n_cor = sum(answered), sum(correct)
        records.append({
            "condition": rel.parts[0],
            "benchmark": rel.parts[-2],
            "n": n,
            "correct": n_cor,
            "answered": n_ans,
            "terminated": sum(term) if term else None,
            "acc_given_answered": (100.0 * sum(c and a for c, a in zip(correct, answered)) / n_ans)
                                  if n_ans else None,
        })

    if not records:
        print(f"no .jsonl outputs found under {outputs}")
        return 1

    lines = [
        "# Answer production and termination (CLAUDE.md §7.1)\n",
        "Recomputed from stored per-problem verdicts.\n",
        "- `answered` — a parseable answer was extracted.",
        "- `terminated` — generation stopped on its own instead of running to the "
        f"{args.cap:,}-token cap.",
        "",
        "**These come apart.** A model can answer correctly and then fail to emit EOS,",
        "looping until the cap. Judge usability on `terminated`, not on `answered`.\n",
        "| condition | benchmark | pass@1 | answered | terminated | acc \\| answered |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(records, key=lambda r: (r["benchmark"], r["condition"])):
        acc = f"{r['acc_given_answered']:.1f}%" if r["acc_given_answered"] is not None else "—"
        t = (f"{r['terminated']}/{r['n']} = {100.0*r['terminated']/r['n']:.1f}%"
             if r["terminated"] is not None else "—")
        lines.append(
            f"| {r['condition']} | {r['benchmark']} | "
            f"{r['correct']}/{r['n']} = {100.0*r['correct']/r['n']:.2f}% | "
            f"{r['answered']}/{r['n']} = {100.0*r['answered']/r['n']:.1f}% | {t} | {acc} |"
        )

    # Pooled over all four benchmarks — the 600-problem view CLAUDE.md §0 insists on, since
    # AIME24 alone is 30 problems at 3.33 pp granularity.
    pooled: dict[str, dict[str, int]] = {}
    for r in records:
        p = pooled.setdefault(r["condition"], {"n": 0, "correct": 0, "answered": 0, "terminated": 0})
        p["n"] += r["n"]
        p["correct"] += r["correct"]
        p["answered"] += r["answered"]
        if r["terminated"] is not None:
            p["terminated"] += r["terminated"]

    lines += [
        "\n## Pooled across benchmarks\n",
        "| condition | benchmarks | pass@1 | answered | terminated |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cond, p in sorted(pooled.items()):
        nb = sum(1 for r in records if r["condition"] == cond)
        t = f"{p['terminated']}/{p['n']} = {100.0*p['terminated']/p['n']:.1f}%" if tok else "—"
        lines.append(
            f"| {cond} | {nb} | {p['correct']}/{p['n']} = {100.0*p['correct']/p['n']:.2f}% | "
            f"{p['answered']}/{p['n']} = {100.0*p['answered']/p['n']:.1f}% | {t} |"
        )

    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
