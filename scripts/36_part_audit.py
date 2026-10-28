#!/usr/bin/env python
"""Audit the PART-defended dataset against its pre-registered gates, and compute the verdict once
the PART student has been trained and evaluated. Every threshold and the decision rule are fixed
in results/m4_part_design.md (§6 gates, §9 rule) -- this script only applies them.

    .venv-infer/bin/python scripts/36_part_audit.py              # gates on data/defended/limo_part.*
    .venv-infer/bin/python scripts/36_part_audit.py --pilot      # gates on the 20-trace pilot
    .venv-infer/bin/python scripts/36_part_audit.py --verdict    # after scripts/22_part_run.sh

Exit code 1 if any gate fails (gates mode). A failing gate means fix and regenerate the affected
segments -- not train on the dataset.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "kim_eval"))

from antidistill.attacks.strip_epistemic import held_out_density, marker_density  # noqa: E402
from antidistill.constants import EPISTEMIC_TOKENS                                  # noqa: E402

# ---------------------------------------------------------------- §6 gates (do not edit post hoc)
GATE_FALLBACK_MAX = 0.02          # share of segments with a fallback in either pass
GATE_NUMBER_RECALL_MIN = 0.95     # mean distinct-number recall per trace
GATE_LEXICAL_MIN = 0.80           # sentence-level match at 0.7 (paper: 0.91 GPT-4o, 0.88 1.5B)
GATE_SELF_TALK_DROP_MIN = 0.70    # self-talk density reduction vs LIMO (paper: -86%)

# ---------------------------------------------------------------- §9 rule (do not edit post hoc)
LIMO_POOLED, HINDSIGHT_V2_POOLED, BAND = 62.8, 49.5, 2.0
BENCH = ["math", "amc", "aime", "aime25"]

# Self-talk density (design doc §6, revised 2026-09-30 before any PART output existed): SINGULAR
# first person plus interjections per 1k words, counted outside mathematics (so "Let \( I \) be
# the incenter" is not first person). "I" is case-sensitive and must be followed by a lower-case
# word or a contraction; lower-case "i" is the imaginary unit or an index.
# Plural first person ("we have", "let us denote") is NOT counted: it is ordinary declarative
# mathematical prose -- hindsight v2's confident traces score 22 per 1k words on it. It is
# reported separately, because the PART prompt does ask the rewriter to avoid "we" and "let's".
_MATH = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]*?\$", re.S)
_FIRST_I = re.compile(r"(?<!\blet )(?<!\bLet )\bI(?:'m|'ll|'ve|'d)?\s+(?=[a-z])|\bI'(?:m|ll|ve|d)\b")
_SELF_TALK = re.compile(r"\b(?:me|my|myself|mine|hmm+|wait|oh|okay|ok|alright|um|uh)\b", re.I)
_PLURAL = re.compile(r"\b(?:we|us|our|ours|ourselves|let's|let’s)\b", re.I)
_KIM = re.compile(r"\b(?:" + "|".join(EPISTEMIC_TOKENS) + r")\b", re.I)


def per_1kw(n: int, text: str) -> float:
    return 1000 * n / max(len(text.split()), 1)


def self_talk_density(text: str) -> float:
    t = _MATH.sub(" ", text or "")
    return per_1kw(len(_FIRST_I.findall(t)) + len(_SELF_TALK.findall(t)), t)


def plural_density(text: str) -> float:
    t = _MATH.sub(" ", text or "")
    return per_1kw(len(_PLURAL.findall(t)), t)


def kim_density(text: str) -> float:
    return per_1kw(len(_KIM.findall(text or "")), text or "")


# The qwen template as LLaMA-Factory renders it with LIMO's system prompt; same as
# scripts/32_trace_budget.py, so truncation figures are comparable with every other condition.
SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPT = ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n"
          "<|im_start|>assistant\n")


def describe(name: str, rows: list[dict], tok, cutoff: int = 16384) -> dict:
    outs = [r["output"] or "" for r in rows]
    n_out = [len(x) for x in tok(outs, add_special_tokens=False)["input_ids"]]
    n_prompt = [len(x) for x in tok([PROMPT.format(system=SYSTEM, instruction=r["instruction"])
                                     for r in rows], add_special_tokens=False)["input_ids"]]
    kept, trunc, trunc_tok = 0, 0, 0
    for p, o in zip(n_prompt, n_out):
        if p + o + 1 <= cutoff:                     # +1: the trailing <|im_end|>
            kept += o + 1
        else:                                        # supervised.py drops the stop token
            k = max(0, cutoff - p)
            kept, trunc, trunc_tok = kept + k, trunc + 1, trunc_tok + k
    mean = lambda f: sum(f(x) for x in outs) / len(outs)   # noqa: E731
    return {"name": name, "n": len(rows), "tokens": sum(n_out) / len(rows),
            "self_talk": mean(self_talk_density), "plural": mean(plural_density),
            "kim9": mean(kim_density),
            "s1_markers": mean(marker_density), "held_out": mean(held_out_density),
            "truncated": trunc / len(rows), "no_stop": trunc_tok / max(kept, 1)}


def gates(pilot: bool, stem: str | None = None) -> int:
    from transformers import AutoTokenizer
    stem = Path(stem) if stem else REPO / "data" / "defended" / ("limo_part_pilot" if pilot else "limo_part")
    meta = json.loads(Path(f"{stem}.meta.json").read_text(encoding="utf-8"))
    rows_m, idx = meta["traces"], meta["metadata"]["indices"]
    part = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
    step1 = json.loads(Path(f"{stem}_step1.json").read_text(encoding="utf-8"))
    limo_all = json.loads((REPO / "data/raw/limo_v2.json").read_text(encoding="utf-8"))
    limo = [limo_all[i] for i in idx]
    hind_path = REPO / "data/defended/limo_hindsight_chat.json"
    hind = [json.loads(hind_path.read_text(encoding="utf-8"))[i] for i in idx] if hind_path.exists() else []
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    print(f"# PART audit — {'PILOT, ' if pilot else ''}{len(rows_m)} traces "
          f"(rewriter {meta['metadata']['rewriter_model']}, segments of "
          f"{meta['metadata']['segment_tokens']} tokens, {meta['metadata']['similarity_impl']})\n")

    seg = sum(r["segments"] for r in rows_m)
    fb = sum(r["segments_with_fallback"] for r in rows_m)
    bad_answer = [r["index"] for r in rows_m if r["answer_agrees"] is False]
    recall = [1 - r["numbers_missing"] / r["numbers_total"] for r in rows_m if r["numbers_total"]]
    lex = [sum(r["lexical_sentence"][k] for r in rows_m) for k in (0, 1)]
    lex_step = [sum(r["lexical_step"][k] for r in rows_m) for k in (0, 1)]
    d = {x["name"]: x for x in (describe("LIMO", limo, tok), describe("PART", part, tok),
                                describe("PART step 1 only", step1, tok))
         + ((describe("hindsight v2", hind, tok),) if hind else ())}
    drop = 1 - d["PART"]["self_talk"] / d["LIMO"]["self_talk"]
    complete = len(rows_m) == (20 if pilot else 800) and all(r["segments"] >= 1 for r in rows_m)

    checks = [
        ("completeness", f"{len(rows_m)} traces, {seg} segments", complete),
        ("fallbacks <= 2% of segments", f"{fb}/{seg} = {100 * fb / seg:.2f}%",
         fb / seg <= GATE_FALLBACK_MAX),
        ("final answer agrees with LIMO", f"{len(bad_answer)} disagree {bad_answer[:8]}", not bad_answer),
        ("number recall >= 0.95", f"mean {sum(recall) / len(recall):.4f}",
         sum(recall) / len(recall) >= GATE_NUMBER_RECALL_MIN),
        ("lexical match @0.7 >= 0.80", f"sentence {lex[0] / lex[1]:.3f} (step {lex_step[0] / lex_step[1]:.3f})",
         lex[0] / lex[1] >= GATE_LEXICAL_MIN),
        ("self-talk density down >= 70%",
         f"{d['LIMO']['self_talk']:.2f} -> {d['PART']['self_talk']:.2f} /1kw ({100 * drop:.0f}%)",
         drop >= GATE_SELF_TALK_DROP_MIN),
    ]
    print("## Gates (design §6)\n\n| gate | measured | |\n| --- | --- | --- |")
    for name, val, ok in checks:
        print(f"| {name} | {val} | {'PASS' if ok else '**FAIL**'} |")

    ro = sum(r["segments_reordered"] for r in rows_m)
    print("\n## Reported (no threshold)\n")
    print(f"- segments reordered by pass 2: {ro}/{seg} = {100 * ro / seg:.1f}%; "
          f"sentences moved {sum(r['sentences_moved'] for r in rows_m) / seg:.2f}/segment; "
          f"sub-conclusions {sum(r['subs'] for r in rows_m) / seg:.2f}/segment, of which hoisted "
          f"{sum(r['subs_hoisted'] for r in rows_m) / max(sum(r['subs'] for r in rows_m), 1):.0%}")
    flags = collections.Counter(f for r in rows_m for f in r["flags"])
    print(f"- output clean-ups: {dict(flags) or 'none'}")
    cache = Path(f"{stem}.segments.jsonl")
    if cache.exists():
        why = collections.Counter()
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                why.update(f"p1 {w.split(':')[0]}" for w in r["p1_fail"])
                why.update(f"p2 {w.split(':')[0]}" for w in r["p2_fail"])
        print(f"- rejected attempts by reason: {dict(why.most_common()) or 'none'}")

    print("\n## The datasets side by side (same problems)\n")
    print("| dataset | tokens/trace | self-talk /1kw | we/us/let's /1kw | Kim et al. nine /1kw "
          "| S1 markers /1kw | held-out /1kw | truncated @16384 | no-stop tokens |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for x in d.values():
        print(f"| {x['name']} | {x['tokens']:,.0f} | {x['self_talk']:.2f} | {x['plural']:.2f} | {x['kim9']:.2f} | "
              f"{x['s1_markers']:.2f} | {x['held_out']:.2f} | {100 * x['truncated']:.1f}% | "
              f"{100 * x['no_stop']:.1f}% |")
    print(f"\nLength retained: {d['PART']['tokens'] / d['LIMO']['tokens']:.1%} of LIMO's tokens. "
          "Held-out reconsideration is mostly first person, so its drop under PART is by "
          "construction (design §11).")

    failed = [c[0] for c in checks if not c[2]]
    print(f"\n{'ALL GATES PASS' if not failed else 'GATES FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


# ----------------------------------------------------------------------------------- verdict

def score(cond: str) -> dict:
    """Per-benchmark counts from the stored per-problem verdicts (as scripts/35_strip_s1_analysis)."""
    from utils.grader import math_equal
    num = re.compile(r"-?\d[\d,]*\.?\d*")
    out = {}
    for b in BENCH + ["gsm8k"]:
        c = cond if b != "gsm8k" else cond.replace("_ep15", "") + "_gsm8k"
        f = glob.glob(f"{REPO}/outputs/{c}/**/{b}/*_t0.0_k1_*.jsonl", recursive=True)
        if not f:
            continue
        rows = [json.loads(line) for line in open(f[0])]
        ok = sum(bool(r["answers_correctness"][0]) for r in rows)
        fin = sum(1 for r in rows if (r["generated_answers"][0] or "").strip())
        rec = 0
        for r in rows:                               # GSM8K last-number fallback (deviations §8)
            if (r["generated_answers"][0] or "").strip():
                continue
            m = num.findall((r["generated_responses"][0] or "").replace(",", ""))
            if m:
                try:
                    rec += bool(math_equal(m[-1].rstrip("."), str(r["gold_answer"])))
                except Exception:
                    pass
        out[b] = {"n": len(rows), "ok": ok, "fb": ok + rec, "fin": fin}
    return out


def verdict() -> int:
    conds = [("base", "base"), ("LIMO (undefended)", "limo_ep15"),
             ("hindsight v2", "hindsight_v2_ep15"), ("**PART**", "part_ep15"),
             ("PART step 1 only", "part_step1_ep15")]
    print("| condition | MATH500 | AMC23 | AIME24 | AIME25 | **pooled** | MATH500 acc\\|fin | GSM8K* |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    pooled = {}
    for label, c in conds:
        s = score(c)
        if not all(b in s for b in BENCH):
            print(f"| {label} | not evaluated |"); continue
        ok, n = sum(s[b]["ok"] for b in BENCH), sum(s[b]["n"] for b in BENCH)
        pooled[c] = 100 * ok / n
        m = s["math"]
        g = f"{100 * s['gsm8k']['fb'] / s['gsm8k']['n']:.1f}%" if "gsm8k" in s else "—"
        cells = " | ".join(f"{100 * s[b]['ok'] / s[b]['n']:.1f}%" for b in BENCH)
        print(f"| {label} | {cells} | **{pooled[c]:.1f}%** | {100 * m['ok'] / max(m['fin'], 1):.1f}% "
              f"({m['fin']}/{m['n']}) | {g} |")
    print("\n\\* GSM8K fallback-graded (results/deviations.md §8).")
    for ref, val in (("limo_ep15", LIMO_POOLED), ("hindsight_v2_ep15", HINDSIGHT_V2_POOLED)):
        if ref in pooled and abs(pooled[ref] - val) > 0.05:
            print(f"WARNING: {ref} recomputes to {pooled[ref]:.1f}%, rule was fixed at {val}%")
    if "part_ep15" not in pooled:
        print("\nPART not evaluated yet.")
        return 0
    p = pooled["part_ep15"]
    if abs(p - LIMO_POOLED) <= BAND:
        v = "PART DOES NOT DEFEND under this recipe (within 2.0 pp of LIMO)"
    elif abs(p - HINDSIGHT_V2_POOLED) <= BAND:
        v = "PART defends AS STRONGLY AS the behavioural defense (within 2.0 pp of hindsight v2)"
    else:
        share = (LIMO_POOLED - p) / (LIMO_POOLED - HINDSIGHT_V2_POOLED)
        v = f"PARTIAL — PART removes {share:.0%} of LIMO's lead over hindsight v2"
    print(f"\nPre-registered verdict (design §9): PART = {p:.1f}% pooled -> {v}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--stem", default=None, help="dataset path without .json (default: data/defended/limo_part[_pilot])")
    a = ap.parse_args()
    raise SystemExit(verdict() if a.verdict else gates(a.pilot, a.stem))
