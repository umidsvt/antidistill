#!/usr/bin/env python
"""Compute the S1 marker-stripping verdict once `strip_s1` is trained and evaluated.

Run this after scripts/21_strip_s1_run.sh. Everything it needs is fixed in advance so the result
cannot be read to fit a story (results/decoupling_design.md §C "What S1 CAN test instead"):

  QUESTION  Are Kim et al.'s nine epistemic marker tokens CAUSAL, or a PROXY for the reasoning
            that produces them?

  S1 vs its exact counterfactual `recon7b_solo`: identical config, identical 800 problems, same
  traces with only the nine marker words deleted (97.5% of length kept, 298/298 correct answers
  kept, held-out reconsideration language unchanged 4.01 -> 4.03).

  DECISION RULE (pre-registered 2026-09-17), on the pooled 600-problem hard suite, boxed-only:
    S1 within 2.0 pp of solo (69.5%)      -> markers are a PROXY
    S1 within 2.0 pp of defended (49.5%)  -> markers are CAUSAL
    otherwise                             -> PARTIAL; report where it falls on the solo-defended span

  2.0 pp = 12 problems of 600, roughly the solo-vs-search gap we have been treating as noise.

It is NOT the density/length decoupling. S1 leaves reconsideration content in place (held-out
4.03 vs the defense's 0.07), so it cannot say whether epistemic CONTENT matters, only whether the
nine SURFACE TOKENS do.
"""
import glob, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "kim_eval"))

BENCH = ["math", "amc", "aime", "aime25"]
CONDS = [("base", "base"), ("defended v2", "hindsight_v2_ep15"), ("LIMO", "limo_ep15"),
         ("7B solo  [counterfactual]", "recon7b_solo_ep15"), ("S1 marker-stripped", "strip_s1_ep15")]
SOLO, DEFENDED, BAND = 69.5, 49.5, 2.0
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def score(cond):
    from utils.grader import math_equal
    out = {}
    for b in BENCH + ["gsm8k"]:
        pat = f"{REPO}/outputs/{cond if b != 'gsm8k' else cond.replace('_ep15','') + '_gsm8k'}/**/{b}/*_t0.0_k1_*.jsonl"
        f = glob.glob(pat, recursive=True)
        if not f:
            continue
        rows = [json.loads(l) for l in open(f[0])]
        ok = sum(bool(r["answers_correctness"][0]) for r in rows)
        fin = sum(1 for r in rows if (r["generated_answers"][0] or "").strip())
        rec = 0
        for r in rows:
            if (r["generated_answers"][0] or "").strip():
                continue
            m = NUM.findall((r["generated_responses"][0] or "").replace(",", ""))
            if m:
                try: rec += bool(math_equal(m[-1].rstrip("."), str(r["gold_answer"])))
                except Exception: pass
        out[b] = dict(n=len(rows), ok=ok, fb=ok + rec, fin=fin)
    return out


def main() -> int:
    from antidistill.attacks.strip_epistemic import marker_density, held_out_density
    print("=== 1. the training pools (should reproduce the recorded pre-training numbers) ===")
    for lab, p in [("7B solo", "data/curated/recon7b_solo.json"),
                   ("S1", "data/curated/strip_s1_recon7b_solo.json")]:
        t = [r["output"] for r in json.load(open(REPO / p))]
        print(f"  {lab:8s} markers {sum(map(marker_density,t))/len(t):6.2f}   "
              f"held-out {sum(map(held_out_density,t))/len(t):5.2f}")
    print("  recorded: solo 26.06 / 4.01   S1 0.04 / 4.03")

    print("\n=== 2. results ===")
    hdr = f"  {'condition':28s}" + "".join(f"{b:>9s}" for b in BENCH) + f"{'POOLED':>9s}{'+fallbk':>9s}{'GSM8K*':>9s}"
    print(hdr)
    res = {}
    for lab, c in CONDS:
        s = score(c)
        if not all(b in s for b in BENCH):
            print(f"  {lab:28s}  (not evaluated yet)")
            continue
        n = sum(s[b]["n"] for b in BENCH)
        pooled = 100 * sum(s[b]["ok"] for b in BENCH) / n
        pfb = 100 * sum(s[b]["fb"] for b in BENCH) / n
        g = f"{100*s['gsm8k']['fb']/s['gsm8k']['n']:8.1f}%" if "gsm8k" in s else "      --"
        res[c] = pooled
        print(f"  {lab:28s}" + "".join(f"{100*s[b]['ok']/s[b]['n']:8.1f}%" for b in BENCH)
              + f"{pooled:8.1f}%{pfb:8.1f}%{g}")
    print("  * GSM8K is fallback-graded, TP=1 (results/gsm8k_results.md)")

    print("\n=== 3. verdict (pre-registered rule) ===")
    if "strip_s1_ep15" not in res:
        print("  strip_s1 not evaluated yet — run scripts/21_strip_s1_run.sh first.")
        return 1
    s1 = res["strip_s1_ep15"]
    span = SOLO - DEFENDED
    print(f"  S1 = {s1:.1f}%   solo = {SOLO}%   defended = {DEFENDED}%   "
          f"position on defended->solo span: {100*(s1-DEFENDED)/span:.0f}%")
    if abs(s1 - SOLO) <= BAND:
        print("  => PROXY. Removing the nine markers leaves the student as good as solo.")
        print("     Kim et al.'s density metric measures a correlate; our r=0.940 is not mechanistic evidence.")
    elif abs(s1 - DEFENDED) <= BAND:
        print("  => CAUSAL. Removing only the nine surface tokens reproduces the defense.")
    else:
        print("  => PARTIAL. Report the span position above; neither pure account fits.")
    print("\n  Reminder: this tests the marker LEXICON, not epistemic CONTENT (held-out stays 4.03).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
