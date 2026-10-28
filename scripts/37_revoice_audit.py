#!/usr/bin/env python
"""Checks and verdict for the re-voice attack on PART (results/m6e_revoice_design.md §5, §6).

    .venv-infer/bin/python scripts/37_revoice_audit.py              # §5 checks, full dataset
    .venv-infer/bin/python scripts/37_revoice_audit.py --pilot      # §5 checks, 20-problem pilot
    .venv-infer/bin/python scripts/37_revoice_audit.py --verdict    # §6 rule, after evaluation

Measures are imported from scripts/36_part_audit.py so every number is comparable with PART's audit.
Exit code 1 if a check fails (checks mode).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("part_audit", REPO / "scripts" / "36_part_audit.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

# §5 and §6 -- fixed before any result; do not edit after seeing data
FALLBACK_MAX, NUMBER_RECALL_MIN, SELF_TALK_SHARE_MIN = 0.02, 0.95, 0.50
PART_POOLED, LIMO_POOLED, BAND = 50.5, 62.8, 2.0

# Per attack: data stem, the restoration check (measure, label), and the condition name evaluated.
# style_part's check is on the uncertainty words themselves (the S1 marker list: wait, hmm, maybe,
# perhaps, actually, alternatively, ...) because re-voicing restored first person but not those
# (m6f §5, fixed before its pilot).
ATTACKS = {
    "revoice_part": {"label": "re-voiced PART", "metric": "self_talk", "check": "self-talk restored >= 50% of LIMO"},
    "style_part": {"label": "Style on PART", "metric": "s1_markers", "check": "uncertainty words restored >= 50% of LIMO"},
}


def checks(pilot: bool, waive_self_talk: bool = False, attack: str = "revoice_part",
           waive_answer: frozenset = frozenset()) -> int:
    from transformers import AutoTokenizer
    cfg = ATTACKS[attack]
    stem = REPO / "data" / "curated" / (f"{attack}_pilot" if pilot else attack)
    meta = json.loads(Path(f"{stem}.meta.json").read_text(encoding="utf-8"))
    rows_m, idx = meta["traces"], meta["metadata"]["indices"]
    rv = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
    load = lambda p: json.loads((REPO / p).read_text(encoding="utf-8"))           # noqa: E731
    limo = [load("data/raw/limo_v2.json")[i] for i in idx]
    part = [load("data/defended/limo_part.json")[i] for i in idx]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    d = {x["name"]: x for x in (pa.describe("LIMO", limo, tok), pa.describe("PART (input)", part, tok),
                                pa.describe(cfg["label"], rv, tok))}

    seg = sum(r["segments"] for r in rows_m)
    fb = sum(r["segments_with_fallback"] for r in rows_m)
    bad = [r["index"] for r in rows_m if r["answer_agrees"] is False]
    recall = [1 - r["numbers_missing"] / r["numbers_total"] for r in rows_m if r["numbers_total"]]
    key, lab = cfg["metric"], cfg["label"]
    share = d[lab][key] / d["LIMO"][key]
    gates = [
        ("completeness", f"{len(rows_m)} problems, {seg} pieces", len(rows_m) == (20 if pilot else 800)),
        ("pieces kept as PART text <= 2%", f"{fb}/{seg} = {100 * fb / seg:.2f}%", fb / seg <= FALLBACK_MAX),
        ("final answers agree with PART", f"{len(bad)} disagree {bad[:8]}", not bad),
        ("numbers kept >= 0.95", f"mean {sum(recall) / len(recall):.4f}", sum(recall) / len(recall) >= NUMBER_RECALL_MIN),
        (cfg["check"],
         f"{d[lab][key]:.2f} vs LIMO {d['LIMO'][key]:.2f} /1kw ({100 * share:.0f}%)",
         share >= SELF_TALK_SHARE_MIN),
    ]
    print(f"# {lab} audit — {'PILOT, ' if pilot else ''}{len(rows_m)} problems\n")
    print("| check | measured | |\n| --- | --- | --- |")
    # A waiver never changes a threshold: the check is still computed and printed, it only stops blocking.
    waivers = {}
    if waive_self_talk:
        waivers[cfg["check"]] = "decision 2026-10-06, m6e §5"
    if bad and set(bad) <= waive_answer:          # only the named problems; any other disagreement still blocks
        waivers["final answers agree with PART"] = (
            "decision 2026-10-13, m6f: the final-answer section is identical to PART's on every problem; "
            "the checker misread an unclosed box in the source text")
    for n, v, ok in gates:
        mark = "PASS" if ok else (f"WAIVED ({waivers[n]})" if n in waivers else "**FAIL**")
        print(f"| {n} | {v} | {mark} |")
    print("\n| dataset | tokens/problem | self-talk /1kw | uncertainty words (S1 list) /1kw | Kim et al. nine /1kw | over 16k limit |")
    print("| --- | --- | --- | --- | --- | --- |")
    for x in d.values():
        print(f"| {x['name']} | {x['tokens']:,.0f} | {x['self_talk']:.2f} | {x['s1_markers']:.2f} | {x['kim9']:.2f} | {100 * x['truncated']:.1f}% |")
    failed = [g[0] for g in gates if not g[2] and g[0] not in waivers]
    used = [g[0] for g in gates if not g[2] and g[0] in waivers]
    ok_line = "ALL CHECKS PASS" if not used else "NO BLOCKING FAILURE; WAIVED: " + ", ".join(used)
    print(f"\n{ok_line if not failed else 'CHECKS FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


def verdict(attack: str = "revoice_part") -> int:
    cfg = ATTACKS[attack]
    cond = f"{attack}_ep15"
    pooled = {}
    print("| condition | MATH500 | AMC23 | AIME24 | AIME25 | **pooled** | MATH500 acc\\|fin | GSM8K* |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for label, c in [("LIMO", "limo_ep15"), ("hindsight v2", "hindsight_v2_ep15"), ("PART", "part_ep15"),
                     ("**" + cfg["label"] + "**", cond)]:
        s = pa.score(c)
        if not all(b in s for b in pa.BENCH):
            print(f"| {label} | not evaluated |"); continue
        ok, n = sum(s[b]["ok"] for b in pa.BENCH), sum(s[b]["n"] for b in pa.BENCH)
        pooled[c] = 100 * ok / n
        m = s["math"]
        g = f"{100 * s['gsm8k']['fb'] / s['gsm8k']['n']:.1f}%" if "gsm8k" in s else "—"
        cells = " | ".join(f"{100 * s[b]['ok'] / s[b]['n']:.1f}%" for b in pa.BENCH)
        print(f"| {label} | {cells} | **{pooled[c]:.1f}%** | {100 * m['ok'] / max(m['fin'], 1):.1f}% ({m['fin']}/{m['n']}) | {g} |")
    if cond not in pooled:
        print("\nRe-voiced PART not evaluated yet."); return 0
    r = pooled[cond]
    if abs(r - PART_POOLED) <= BAND:
        v = "the attack DOES NOT recover what PART removed"
    elif r >= LIMO_POOLED - BAND:
        v = "the attack FULLY DEFEATS PART"
    else:
        v = f"PARTIAL: recovers {(r - PART_POOLED) / (LIMO_POOLED - PART_POOLED):.0%} of the PART-to-LIMO gap"
    print(f"\nPre-registered verdict ({'m6e' if attack == 'revoice_part' else 'm6f'} §6): {cfg['label']} = {r:.1f}% -> {v}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--waive-self-talk", action="store_true",
                    help="report the self-talk check but do not fail on it (decided 2026-10-06 after three "
                         "pilots plateaued at ~48-50% of LIMO; see m6e §5)")
    ap.add_argument("--waive-answer", default="",
                    help="comma-separated problem indices whose final-answer flag does not block (decided "
                         "2026-10-13 for problem 355 of style_part; see m6f, 'Full data')")
    ap.add_argument("--attack", choices=sorted(ATTACKS), default="revoice_part")
    a = ap.parse_args()
    wa = frozenset(int(x) for x in a.waive_answer.split(",") if x.strip())
    raise SystemExit(verdict(a.attack) if a.verdict else checks(a.pilot, a.waive_self_talk, a.attack, wa))
