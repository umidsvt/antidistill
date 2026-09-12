#!/usr/bin/env python
"""M6b — regenerate epistemic verbalization that a defense stripped out.

Design and predictions: results/m6b_reconstruction_design.md

The defender serves confident, doubt-free traces. Nothing epistemic survives inside them to be
recovered by parsing -- measured, the defended pool carries 0.024 epistemic tokens per 1k words
against LIMO's 35.571, and even the teacher's PRIVATE scratchpad carries only 0.732, because the
rewrite prompt orders it to "State everything with full confidence". The doubt was never
generated. So the attacker must re-generate it.

Three modes, and the third is the control that makes the other two interpretable:

  style   (A1) Rewrite the defended trace as a first-person exploration with false starts.
               The doubt is FABRICATED -- the rewriter never searched. Pure test of "is
               epistemic verbalization just a linguistic habit?".

  search  (A2) Hand the model the problem AND the defended solution as a hint, let it reason
               naturally with NO confidence instruction, and keep its own reasoning_content
               plus its answer. The doubt is REAL (its actual search) but steered to the
               defended answer. This is precisely what the v2 hindsight pipeline throws away.

  solo    (A3) THE CONTROL. Attacker's model solves from scratch, no defended data at all.
               If A2 does not beat A3, the attacker never needed the defended traces and the
               defense is irrelevant rather than broken. Do not skip this.

Requires the teacher served with --reasoning-parser deepseek_r1 (scripts/40_serve_teacher.sh),
which splits .reasoning_content (scratchpad) from .content (answer).

Usage:
    python -m antidistill.attacks.epistemic_reconstruction --mode search \
        --defended data/defended/limo_hindsight_chat.json \
        --output data/curated/limo_recon_search.json \
        --base-url http://127.0.0.1:8011/v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from antidistill.defenses.hindsight import answers_agree, load_grader  # noqa: E402

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

# A1: the rewriter is TOLD to add doubt it never actually experienced. That is the point --
# it isolates "does doubt-shaped text suffice" from "does searched doubt matter".
STYLE_PROMPT = """You are given a QUESTION and a polished SOLUTION.

QUESTION:
{question}

SOLUTION:
{solution}

Rewrite the solution as a first-person account of working the problem out for the first time.
Show the reasoning as it would actually unfold: consider an approach before committing to it,
note where a step could go wrong, double back when something does not check out, and verify
results that are easy to get wrong. Keep every mathematical step and the same final answer.
Put your final answer within \\boxed{{}}."""

# A2: no confidence instruction, and the defended solution is a HINT rather than a script.
# We keep the model's own reasoning_content, so the doubt is genuine search, not performance.
SEARCH_PROMPT = """Solve this problem.

QUESTION:
{question}

A reference solution is available if you need it; you do not have to follow its route:
{solution}

Work the problem out yourself, step by step, showing all key equations and intermediate algebra.
Put your final answer within \\boxed{{}}."""

# A3: no defended data whatsoever.
SOLO_PROMPT = """Solve this problem, step by step, showing all key equations and intermediate
algebra.

QUESTION:
{question}

Put your final answer within \\boxed{{}}."""

PROMPTS = {"style": STYLE_PROMPT, "search": SEARCH_PROMPT, "solo": SOLO_PROMPT}

# Distinct-12-gram floor for salvaging an unterminated reasoning trace. Healthy prose sits
# near 1.0; calibrate against the completed pools before changing this.
MIN_REPETITION_RATIO = 0.60


# Manual chat template with an ALREADY-CLOSED think block. Prefilling this leaves the model no
# scratchpad, so instructed doubt has to appear in the visible output.
#
# Why not the chat endpoint: DeepSeek's template applies content.split('</think>')[-1] to any
# assistant message, which deletes the prefill. The completions endpoint takes the string as-is.
NOTHINK_TEMPLATE = "<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n</think>\n\n"


def strip_think(text: str) -> str:
    """Remove any </think> the model emits anyway.

    That delimiter is the artifact 499/500 v1-trained students learned to reproduce
    (results/hindsight_versions.md); no generated dataset in this project may carry it.
    """
    return text.replace("</think>", "").replace("<think>", "").strip()


def repetition_ratio(text: str, n: int = 12) -> float:
    """Fraction of DISTINCT n-grams. Near 1.0 is healthy prose; a degenerate loop tends
    toward 0 because it emits the same window over and over."""
    w = text.split()
    if len(w) < n * 4:
        return 1.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(grams)) / len(grams)


def build_output(mode: str, reasoning: str | None, content: str | None) -> str | None:
    """Assemble the SFT target from what the server returned.

    In `search`/`solo` the epistemic content lives in reasoning_content -- that IS the attack,
    so it is joined to the answer. A trailing </think> is never emitted: the delimiter is what
    499/500 v1 students learned to reproduce (results/hindsight_versions.md), so the two halves
    are joined with plain whitespace instead.
    """
    content = strip_think(content or "")
    reasoning = strip_think(reasoning or "")
    if mode == "style":
        # Generated WITHOUT a scratchpad (see NOTHINK_TEMPLATE), so everything is in `content`.
        return content or None
    if content:
        return f"{reasoning}\n\n{content}" if reasoning else content
    # No answer: the model ran out of budget before closing its reasoning block. The
    # reasoning itself is NOT wasted -- it is the epistemic search trace this attack exists
    # to harvest, and LIMO targets are themselves first-person reasoning of exactly this
    # shape. Discarding it cost 50/800 problems on the first pass.
    #
    # Salvage it only if it is not a degenerate repetition loop, which is R1's documented
    # failure mode here. A looping trace would teach the student to loop.
    if not reasoning:
        return None
    if repetition_ratio(reasoning) >= MIN_REPETITION_RATIO:
        return reasoning
    # The whole trace loops, but a loop has an ONSET: everything before it is genuine
    # reasoning the model actually did. Keep the longest clean prefix rather than discard
    # 76k-134k characters of real search. Binary-search the cut point.
    return clean_prefix(reasoning)


def clean_prefix(text: str, min_words: int = 400, win: int = 400, step: int = 100) -> str | None:
    """Longest prefix of `text` taken up to the ONSET of a repetition loop.

    A degenerate trace is genuine for a while and then sticks, so truncating at the onset
    keeps the real search and drops the stuck tail.

    Detection must be LOCAL, not a global average over the prefix: a prefix can absorb
    several loop iterations and still average above MIN_REPETITION_RATIO when the clean
    part is long enough. Measured on a real 12,948-word looping trace, a global binary
    search returned a "clean" prefix that still ended mid-loop. So this slides a window
    and cuts at the first window that degenerates.

    Returns None when the clean part is shorter than min_words -- better no example than a
    stub.
    """
    w = text.split()
    if len(w) < min_words:
        return None
    cut = len(w)
    for i in range(0, len(w) - win, step):
        if repetition_ratio(" ".join(w[i:i + win]), n=8) < MIN_REPETITION_RATIO:
            cut = i                      # loop starts inside this window
            break
    if cut < min_words:
        return None
    out = " ".join(w[:cut])
    b = max(out.rfind(". "), out.rfind(".\n"))
    out = (out[:b + 1] if b > len(out) * 0.5 else out).strip()
    # Local windows cannot see LONG-RANGE repetition: a trace that repeats a passage every
    # ~1000 words passes every 400-word window while scoring 0.29 globally (observed).
    # So validate the cut prefix globally and refuse rather than ship a degenerate trace.
    return out if repetition_ratio(out) >= MIN_REPETITION_RATIO else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["style", "search", "solo"], required=True)
    ap.add_argument("--defended", default="data/defended/limo_hindsight_chat.json",
                    help="Defended pool. Supplies the problems, and for style/search the "
                         "solution. Ignored as a solution source in --mode solo.")
    ap.add_argument("--source", default="data/raw/limo_v2.json",
                    help="Undefended pool, used ONLY to score answer agreement for the audit.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="DeepSeek recommend 0.5-0.7 for R1-Distill; below that it is prone to "
                         "endless repetition.")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N problems (pilot runs).")
    ap.add_argument("--request-timeout", type=int, default=3600)
    ap.add_argument("--api-retries", type=int, default=4)
    a = ap.parse_args()

    math_equal, _ = load_grader()
    defended = json.loads(Path(a.defended).read_text(encoding="utf-8"))
    source = json.loads(Path(a.source).read_text(encoding="utf-8"))
    if a.limit:
        defended, source = defended[:a.limit], source[:a.limit]

    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=a.request_timeout, max_retries=0)

    def call(prompt: str):
        """Returns (reasoning, content).

        `style` suppresses the scratchpad, because R1 obeys "show your doubt" in
        reasoning_content and still emits a clean, confident .content -- measured 44.89
        epistemic tokens/1k words in the scratchpad against 0.00 in the answer. Harvesting
        .content there would yield a pool indistinguishable from the defended one (0.193 vs
        0.024 in a 20-problem pilot), i.e. no attack at all. With no scratchpad available the
        instructed doubt lands in the visible output (26.48/1k words).
        """
        last = None
        for attempt in range(a.api_retries + 1):
            try:
                if a.mode == "style":
                    r = client.completions.create(
                        model=a.model, prompt=NOTHINK_TEMPLATE.format(prompt=prompt),
                        temperature=a.temperature, max_tokens=a.max_new_tokens)
                    return None, r.choices[0].text
                m = client.chat.completions.create(
                    model=a.model, messages=[{"role": "user", "content": prompt}],
                    temperature=a.temperature, max_tokens=a.max_new_tokens,
                ).choices[0].message
                return getattr(m, "reasoning_content", None), m.content
            except Exception as e:
                last = e
                time.sleep(min(60, 5 * 2 ** attempt))
        raise last

    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")
    records, meta = [], []
    n = len(defended)
    start = time.time()

    for i in range(0, n, a.batch_size):
        chunk = list(range(i, min(i + a.batch_size, n)))
        prompts = [PROMPTS[a.mode].format(question=defended[j]["instruction"],
                                          solution=defended[j]["output"]) for j in chunk]
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            results = list(ex.map(call, prompts))

        for j, (reasoning, content) in zip(chunk, results):
            text = build_output(a.mode, reasoning, content)
            rec = dict(defended[j]); rec["output"] = text or ""
            records.append(rec)
            meta.append({
                "index": j,
                "mode": a.mode,
                "empty": not text,
                "salvaged_unterminated": bool(text) and not (content or "").strip()
                                          and a.mode != "style",
                "reasoning_chars": len(reasoning or ""),
                "answer_chars": len(content or ""),
                "output_chars": len(text or ""),
                # Audited against the UNDEFENDED trace, so `search` cannot be scored correct
                # merely for copying the hint it was given.
                "answer_matches_source": answers_agree(source[j]["output"], text or "", math_equal),
            })

        out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        el = time.time() - start
        done = len(records)
        print(f"[{done}/{n}] empty {sum(m['empty'] for m in meta)} | "
              f"elapsed {time.strftime('%H:%M:%S', time.gmtime(el))} | "
              f"ETA {time.strftime('%H:%M:%S', time.gmtime(el/done*(n-done)))}", flush=True)

    agree = sum(m["answer_matches_source"] is True for m in meta)
    print(f"\nDONE [{a.mode}]. {len(records)} traces -> {out_path}")
    print(f"  empty                     : {sum(m['empty'] for m in meta)}")
    print(f"  answer matches UNDEFENDED : {agree}/{len(meta)}")
    print(f"  mean scratchpad chars     : {sum(m['reasoning_chars'] for m in meta)//max(len(meta),1)}")
    print(f"  sidecar                   : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
