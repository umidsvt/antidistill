#!/usr/bin/env python
"""Re-voice attack on PART -- rewrite PART's declarative statements back into thinking aloud.

Design, gates and the pre-registered rule: results/m6e_revoice_design.md.

Why this attack: the step-1-only ablation (results/REPORT.md Part 8) showed that PART's defensive
effect comes entirely from its step 1, which rewrites first-person self-talk as plain statements;
reordering adds nothing. Step 1 keeps the checks, corrections and alternative methods -- only their
wording changes ("Wait, is the number of people still six? Let me check." -> "The total number of
people remains six."). So the attacker reverses step 1 piece by piece.

Threat model: the attacker has the PART-defended solutions and an ordinary instruction-tuned model
(no reasoning model). It never sees the original LIMO text.

The engine is PART's (antidistill.defenses.part): the same ~1,024-token pieces at paragraph breaks,
the same retry/fallback logic, cache and multi-server client. Only the prompt and the acceptance
checks differ: re-voicing ADDS words, so the length bounds are wider upward, and every number and
most content words of the input must survive.

Usage:
    python -m antidistill.attacks.revoice --input data/defended/limo_part.json \
        --output data/curated/revoice_part.json --base-url http://127.0.0.1:8003/v1 \
        --model Qwen/Qwen2.5-32B-Instruct [--pilot 20]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from antidistill.defenses.part import (
    PART, _LIST_LINE, MAX_NEW_LIST_LINES, MIN_RATIO_CHARS, MISSING_NUMBERS_FRAC, _write_json,
    clean_pass1, content_recall, missing_numbers, pilot_indices)

# Written to mirror PART's prompt 1 (Ding et al. 2025, Appendix B) in reverse. PART's prompt says
# "Avoid using first-person expressions such as 'I', 'me', 'we', or 'let's'"; this one asks for them
# throughout.
#
# Revised 2026-10-05 after the pilot, before any training (m6e §3): the first version asked to convert
# only the checks and corrections and to keep plain calculation as it was. The rewriter then re-voiced
# lightly -- self-talk 8.8 per 1k words against LIMO's 31.8 (28%), failing the 50% check -- because
# PART's step 1 removed first person from the WHOLE text, not just from the checks.
REVOICE_PROMPT_V1 = (
    "Rewrite the given text, which is a part of a complete reasoning process. Convert the parts "
    "that state a check, a correction, a doubt, a comparison of options or a change of approach "
    "into first-person thinking aloud, the way a person working the problem out for the first "
    "time would say it, for example \"Wait, let me check that...\", \"Hmm, maybe...\", "
    "\"Actually, ...\", \"Let me try another way.\" Keep every calculation, number and result "
    "exactly as in the original text, and keep the parts that are plain calculation as they are.\n"
    "Do not add new steps or new information. Do not include any introductory phrases.\n"
    "Text:"
)
REVOICE_PROMPT_V2 = (
    "Rewrite the given text, which is a part of a complete reasoning process, as a first-person "
    "account of someone working the problem out for the first time and thinking aloud. Use "
    "first-person expressions throughout, such as 'I', 'me', 'let me' and 'let's'. Where the text "
    "checks a result, corrects itself, doubts a step, compares options or changes approach, say it "
    "the way a person thinking aloud would, for example \"Wait, ...\", \"Hmm, ...\", \"Let me check "
    "that...\", \"Actually, ...\", \"Maybe ...\". Keep every calculation, number and result exactly "
    "as in the original text.\n"
    "Do not add new steps or new information. Do not include any introductory phrases.\n"
    "Text:"
)
# Revised again 2026-10-05, before any training (m6e §3): V2 reached exactly half of LIMO's self-talk
# (49.8%) but condensed or dropped numbers in 3.7% of pieces. V3 asks for continuous thinking aloud
# and states that only the wording may change.
REVOICE_PROMPT = (
    "Rewrite the given text, which is a part of a complete reasoning process, as continuous "
    "first-person thinking aloud by someone working the problem out for the first time. Change only "
    "how things are said, never what is said: keep every sentence's content, every calculation, "
    "every number and every result, so the rewrite is at least as long as the original. Use "
    "first-person expressions throughout, for example \"So I have...\", \"Let me compute...\", "
    "\"Okay, so...\", \"I think...\". Where the text checks a result, corrects itself, doubts a "
    "step, compares options or changes approach, say it the way a person thinking aloud would, for "
    "example \"Wait, ...\", \"Hmm, ...\", \"Let me check that...\", \"Actually, ...\", "
    "\"Maybe ...\".\n"
    "Do not add new steps or new information. Do not include any introductory phrases.\n"
    "Text:"
)

RATIO = (0.90, 1.60)          # re-voicing adds words; it must not summarise or balloon
CONTENT_RECALL_MIN = 0.70     # most content words of the input must survive


def check_revoice(src: str, out: str, finish: str | None, ratio: tuple = RATIO,
                  content_min: float = CONTENT_RECALL_MIN) -> str | None:
    """None if `out` is an acceptable rewrite of `src`, otherwise the reason it is not. `ratio` and
    `content_min` are per attack (Revoice.ratio / Revoice.content_min)."""
    if finish == "length":
        return "truncated"
    if not out.strip():
        return "empty"
    if len(src) >= MIN_RATIO_CHARS:
        r = len(out) / len(src)
        if r < ratio[0]:
            return f"too_short:{r:.2f}"
        if r > ratio[1]:
            return f"too_long:{r:.2f}"
    miss, tot = missing_numbers(src, out)
    if miss > max(1, math.ceil(MISSING_NUMBERS_FRAC * tot)):
        return f"numbers_missing:{miss}/{tot}"
    rec = content_recall(src, out)
    if rec is not None and rec < content_min:
        return f"content_lost:{rec:.2f}"
    if len(_LIST_LINE.findall(out)) - len(_LIST_LINE.findall(src)) > MAX_NEW_LIST_LINES:
        return "list_introduced"
    return None


class Revoice(PART):
    """One pass per piece with REVOICE_PROMPT. Reuses PART's run(), cache and assembly; the
    re-voiced text is stored as both "p1" and "p2", so PART's assembly yields it as the output."""

    name = "revoice"
    # Per-attack settings; subclasses (e.g. antidistill.attacks.style_part) override these.
    prompt = REVOICE_PROMPT
    ratio = RATIO
    content_min = CONTENT_RECALL_MIN

    def rewrite_segment(self, text: str) -> dict:
        rec = {"p1": text, "p2": text, "subs": [], "p1_fallback": False, "p2_fallback": False,
               "p1_fail": [], "p2_fail": [], "flags": []}
        if not text.strip():
            return rec
        n_tok = self.count([text])[0]
        budget = min(self.max_new_tokens, int(2.0 * n_tok) + 256)
        prompt = self.prompt + "\n" + text
        for attempt in range(self.max_retries + 1):
            temp = 0.0 if attempt == 0 else self.retry_temperature
            try:
                raw, finish = self._client.chat(prompt, temp, budget)
            except Exception as e:
                rec["p1_fail"].append(f"api:{type(e).__name__}")
                continue
            cand, flags = clean_pass1(raw)
            why = check_revoice(text, cand, finish, self.ratio, self.content_min)
            if why is None:
                rec["p1"] = rec["p2"] = cand
                rec["flags"] += [f"rv_{f}" for f in flags]
                return rec
            rec["p1_fail"].append(why)
            if why == "truncated":
                budget = min(self.max_new_tokens, budget * 2)
        rec["p1_fallback"] = rec["p2_fallback"] = True      # keep PART's text for this piece
        return rec

    def metadata(self) -> dict:
        return {**super().metadata(), "attack": self.name, "prompt": self.prompt,
                "length_ratio_bounds": self.ratio, "content_recall_min": self.content_min}


def main(cls=None, default_output: str = "data/curated/revoice_part.json", doc: str = __doc__) -> int:
    cls = cls or Revoice
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/defended/limo_part.json")
    ap.add_argument("--output", default=default_output)
    ap.add_argument("--base-url", action="append", default=None)
    ap.add_argument("--server-concurrency", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--segment-tokens", type=int, default=1024)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=128)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--pilot", type=int, default=0)
    sel.add_argument("--indices", default="")
    a = ap.parse_args()

    traces = json.loads(Path(a.input).read_text(encoding="utf-8"))
    indices = (pilot_indices(traces, a.pilot) if a.pilot else
               [int(x) for x in a.indices.split(",")] if a.indices else list(range(len(traces))))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    rv = cls(base_urls=a.base_url, model=a.model, segment_tokens=a.segment_tokens,
                 max_retries=a.max_retries, concurrency=a.concurrency,
                 cache_path=f"{stem}.segments.jsonl",
                 server_caps=[int(x) for x in a.server_concurrency.split(",")] if a.server_concurrency else None)
    res = rv.run(traces, indices)
    m = res["meta"]
    _write_json(out, res["defended"])
    _write_json(Path(f"{stem}.meta.json"),
                {"metadata": {**rv.metadata(), "input": a.input, "indices": res["indices"],
                              "pilot": a.pilot, "written": time.strftime("%Y-%m-%d %H:%M:%S")},
                 "traces": m})
    seg = sum(r["segments"] for r in m)
    fb = sum(r["segments_with_fallback"] for r in m)
    bad = [r["index"] for r in m if r["answer_agrees"] is False]
    print(f"\nDONE: {len(m)} traces, {seg} pieces -> {out}")
    print(f"  pieces kept as PART text (all retries failed): {fb} ({100 * fb / max(seg, 1):.2f}%; gate <= 2%)")
    print(f"  final answer disagrees with the input: {len(bad)} {bad[:10]}")
    print("  audit: scripts/37_revoice_audit.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
