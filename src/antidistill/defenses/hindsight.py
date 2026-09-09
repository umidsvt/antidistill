#!/usr/bin/env python
"""Hindsight defense — rewrite reasoning traces as confident, doubt-free derivations.

This is the "no epistemic verbalization" condition of the proposal's Section 2.2 table, and a
faithful reimplementation of Kim et al.'s `make_hint_dataset.py`. The rewrite prompt, the forced
`"Okay, so I"` prefix, the GOOD/BAD validation rubric and the resume-on-restart behaviour are
preserved verbatim, because they define the condition.

What we add on top (and why):
  * CLI arguments instead of module-level constants.
  * A **retry budget**. Upstream allows 20 regenerations per item with no global cap; on a 32k-token
    model that is an unbounded tail. Items that exhaust their budget fall back to their last
    attempt, which may be an *incorrect* trace — a real contamination risk for the condition — so
    we cap it and record exactly which items were affected.
  * A metadata sidecar (`<output>.meta.json`) with per-item retry count, validator verdict and
    output length, so the dataset can be audited rather than trusted.

The rewriter is model-independent: it transforms the LIMO traces themselves, so one run of this
serves every student model in the table.

Usage:
    python -m antidistill.defenses.hindsight \
        --input data/raw/limo_v2.json \
        --output data/defended/limo_hindsight_ds32b.json \
        --base-url http://localhost:8001/v1 \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

# --- verbatim from Kim et al. `make_hint_dataset.py` -------------------------------------------

REWRITE_PROMPT = """You are given a QUESTION and its SOLUTION.

QUESTION:
{question}

SOLUTION:
{solution}

Using only the actual solving approach from the solution above — excluding any wandering or trial-and-error — as a reference, re-derive the result independently from scratch, step by step. Include all key equations and intermediate algebra. Do not express any uncertainty — never say "I think," "probably," or "it seems." State everything with full confidence.

Put your final answer within \\boxed{{}}.

Okay, so I
"""

FORCED_PREFIX = "Okay, so I"


def load_grader():
    """The deterministic math grader from the eval harness, loaded lazily.

    CORRECTNESS (antidistill) -- Kim et al. validate each rewrite with an LLM judge whose
    prompt says it is "Given a math problem, a proposed solution, and the ground-truth
    answer", but their .format() call passes only question and response: no ground truth is
    ever supplied (make_hint_dataset.py). The judge therefore has to re-solve every
    competition problem from scratch, which costs thousands of reasoning tokens and still
    often fails to reach a verdict.

    The gold answer is already available -- it is the \boxed{} in the LIMO trace being
    rewritten -- so the check can be exact and free. Measured on a 20-problem sample:
    grader validated 20/20; the LLM judge validated 14/20 at 3072 tokens each with 9
    retries, and disagreed with the grader on none of them.
    """
    import sys
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[3] / "third_party" / "kim_eval"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.grader import math_equal
    from utils.parser import extract_answer
    return math_equal, extract_answer


_BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def answers_agree(source_trace: str, rewrite: str, math_equal) -> bool | None:
    """Does the rewrite reach the same final answer as the trace it was derived from?

    Comparing only the LAST \boxed{} of each (the obvious implementation) overcounts
    disagreement roughly 2x, because the two write multi-part answers differently:
    LIMO emits \boxed{1, 2} while the rewrite emits "\boxed{1} and \boxed{2}", so a
    last-box check reads gold "1, 2" against pred "2" and calls it a mismatch.

    This collects every boxed value on each side, splits comma/semicolon-separated
    lists, and asks whether the rewrite covers the source's answer set.

    Returns None when either side has no extractable \boxed{} at all -- which happens
    on the SOURCE too: some LIMO traces trail off without boxing a final answer (e.g.
    "...so I'll proceed with the boxed answers as above"). Those are unjudgeable rather
    than wrong, and must not be counted as rewrite errors.
    """
    g, p = _BOX_RE.findall(source_trace or ""), _BOX_RE.findall(rewrite or "")
    if not g or not p:
        return None
    def parts(v):
        return [x.strip() for x in re.split(r"[,;]", v) if x.strip()] or [v]
    gset = parts(g[-1])
    pall = [x for b in p for x in parts(b)] + p
    def eq(a, b):
        try:
            return bool(math_equal(a, b))
        except Exception:
            return False
    return all(any(eq(x, y) for y in pall) for x in gset)


def harvest_answer(completion: str) -> str | None:
    """Extract the assistant's ANSWER from a reasoning-model completion.

    DeepSeek-R1 defines this itself, in its own chat template
    (tokenizer_config.json), for replaying an assistant turn:

        {% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}

    Everything before the final </think> is scratchpad that the model vendor
    discards. The template's generation prompt ends with '<|Assistant|><think>\n',
    i.e. it OPENS the block, which is why a completion contains a closing </think>
    with no opener.

    DEVIATION (antidistill) -- Kim et al.'s make_hint_dataset.py stores the raw
    completion (`new_item["output"] = teacher_text`), so every trace kept BOTH the
    scratchpad and the answer: two complete solutions, each with its own \boxed,
    plus a stray </think>. Measured consequence: 499/500 MATH500 responses from the
    hindsight-trained student emit </think>, against 0/500 for LIMO and 0/500 for
    base. Pass --keep-scratchpad to reproduce their behaviour exactly.

    Returns None if the block never closed, which means generation was truncated
    mid-scratchpad and no answer exists -- caller should retry rather than keep it.
    """
    if "</think>" not in completion:
        return None
    return completion.split("</think>")[-1].strip()

EVAL_PROMPT_TEMPLATE = """You are an automatic solution evaluator.
Given a math problem, a proposed solution, and the ground-truth answer,
decide whether the proposed solution is correct.
Rules:
- If the solution is logically sound and the final conclusion is correct, output exactly: The solution is GOOD
- If the answer is incorrect, the solution contains logical gaps or errors, or the solver attempted to use a Python program, output exactly: The solution is BAD
- Do not provide any explanation.
- Do not output anything else.
Problem:
{question}
Proposed Solution:
{response}
Decision:
""".strip()

# -----------------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw/limo_v2.json")
    p.add_argument("--output", default="data/defended/limo_hindsight_ds32b.json")
    p.add_argument("--api", choices=["chat", "completions"], default="chat",
                   help="chat (default, correct): use the model's chat template, which opens "
                        "<think> for it. completions: Kim et al.'s raw-text call with the "
                        "'Okay, so I' prefix hack -- kept only to reproduce their dataset.")
    p.add_argument("--keep-scratchpad", action="store_true",
                   help="Store the raw completion (scratchpad + answer), reproducing Kim et "
                        "al.'s make_hint_dataset.py exactly. Default is to harvest only the "
                        "answer, per DeepSeek-R1's own chat template. See harvest_answer().")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    # 100 (upstream) means 100 concurrent 32k-token generations in one HTTP request: each call
    # takes >40 min, exceeds the client timeout, and checkpoints only every ~100 min. 32 is close
    # to the server's observed concurrency (~28 running) so the GPUs stay busy while writing to
    # disk ~4x more often.
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--request-timeout", type=float, default=3600.0,
                   help="Seconds. The openai client defaults to 600s, which a batch of long "
                        "generations blows through — that killed the first two attempts.")
    p.add_argument("--api-retries", type=int, default=4,
                   help="Retries for transient API failures (timeout, connection reset). An "
                        "8-hour job must not die on one bad response.")
    p.add_argument("--max-new-tokens", type=int, default=32784)
    p.add_argument("--validator", choices=["llm", "grader"], default="llm",
                   help="llm (default): the model judges the SOLUTION, not just the answer -- a "
                        "rewrite can reach the right result by faulty reasoning, and it is the "
                        "reasoning we distil. grader: exact boxed-answer match only; cheap, but "
                        "blind to unsound derivations. The grader verdict is recorded in the "
                        "sidecar either way, so 'right answer, wrong reasoning' is measurable.")
    p.add_argument("--eval-max-tokens", type=int, default=8192,
                   help="Token budget for the validator call. Must leave room for the "
                        "reasoning model to think AND emit its verdict; too small and "
                        ".content is empty, which reads as a failed validation.")
    p.add_argument("--model-max-len", type=int, default=65536,
                   help="Must match the server's --max-model-len. vLLM requires "
                        "input_tokens + max_tokens <= this, so max_tokens is reduced per batch "
                        "when the prompt is long. The rewrite prompt embeds the full LIMO "
                        "solution, which reaches ~33k tokens for the longest traces.")
    # DeepSeek recommend 0.5-0.7 (0.6) for R1-Distill; below that the model is prone to
    # endless repetition. Kim et al. use 0.4, which is likely why 40/800 of our traces ran
    # to the 32k cap without ever closing their <think> block.
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-retries", type=int, default=8,
                   help="Per-item regeneration cap. Upstream uses 20 with no global limit; "
                        "8 bounds the tail. Items that exhaust it are recorded in the sidecar.")
    p.add_argument("--retry-budget", type=int, default=1200,
                   help="Global cap on total regenerations across the whole run.")
    # Sharding lets N generators run against N independent teacher replicas. On this host that
    # matters: two TP=4 servers (GPUs 0-3 and 4-7) keep every tensor-parallel all-reduce inside a
    # NUMA node, whereas a single TP=8 server would send all 128 per-token collectives across the
    # socket boundary. Shards are contiguous index ranges so each resumes independently.
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def fmt(sec: float) -> str:
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    meta_path = out_path.with_suffix(".meta.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY",
                    timeout=args.request_timeout, max_retries=0)

    def _call(fn, **kw):
        """One API call, retried on transient failures. A multi-hour job must not die on one."""
        last = None
        for attempt in range(args.api_retries + 1):
            try:
                return fn(**kw)
            except Exception as e:                      # timeout, conn reset, 5xx
                last = e
                wait = min(60, 5 * 2 ** attempt)
                print(f"  API error ({type(e).__name__}) attempt {attempt+1}/"
                      f"{args.api_retries+1}; retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise last

    def complete(prompt, **kw):
        """Generate one continuation per prompt; returns a list[str] either way.

        CORRECTNESS (antidistill) -- DeepSeek-R1-Distill is a chat/reasoning model and
        must be called through its chat template. The template's generation prompt ends
        with '<|Assistant|><think>\n', i.e. it OPENS the reasoning block; the model then
        closes it with </think> and writes its answer.

        Kim et al. instead call the raw completions endpoint, so the prompt carries no
        chat markers, no BOS and no <think> opener, and they substitute the literal
        prefix "Okay, so I" to force reasoning voice. That is off-distribution use of an
        instruction-tuned model and is why every trace ends up with an unmatched
        </think>. --api completions reproduces it for comparison.

        With the server started by scripts/40_serve_teacher.sh (--reasoning-parser
        deepseek_r1) the chat path returns the ANSWER in .content and the scratchpad in
        .reasoning_content, so no string splitting is needed at all. If the parser is
        absent the answer still arrives inside .content and harvest_answer() recovers it.
        """
        if args.api == "completions":
            r = _call(client.completions.create, prompt=prompt, **kw)
            return [FORCED_PREFIX + c.text for c in r.choices]

        # The chat endpoint takes one conversation per call, so the batching that the
        # completions endpoint gave us for free has to be done with concurrent requests;
        # vLLM batches them server-side. Serialising here would cost ~10x wall-clock.
        def one(text):
            m = _call(client.chat.completions.create,
                      messages=[{"role": "user", "content": text}], **kw).choices[0].message
            # With --reasoning-parser deepseek_r1 the scratchpad is split off into
            # .reasoning_content and .content holds the answer alone. Without it,
            # .content holds both and harvest_answer() splits it downstream.
            return m.content
        with ThreadPoolExecutor(max_workers=min(len(prompt), args.batch_size)) as ex:
            return list(ex.map(one, prompt))

    math_equal, extract_answer = load_grader()

    # Needed to size max_tokens per batch (see --model-max-len).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def budget(prompts: list[str]) -> int:
        """Largest max_tokens that fits for every prompt in this batch."""
        longest = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
        return max(0, min(args.max_new_tokens, args.model_max_len - longest - 64))

    all_data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    n_all = len(all_data)
    # Contiguous slice for this shard; `offset` maps back to original indices in the sidecar.
    per = math.ceil(n_all / args.num_shards)
    offset = args.shard * per
    data = all_data[offset:offset + per]
    print(f"Loaded {n_all} records from {args.input}")
    if args.num_shards > 1:
        print(f"Shard {args.shard}/{args.num_shards}: indices [{offset}, {offset+len(data)}) "
              f"= {len(data)} records")

    # Resume from a partial run.
    if out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
        print(f"Resuming: {len(done)} already written")
    else:
        done, meta = [], []

    pending = list(range(len(done), len(data)))
    if not pending:
        print("Nothing to do.")
        return 0
    print(f"Remaining: {len(pending)}")

    start = time.time()
    spent_retries = 0
    n_batches = math.ceil(len(pending) / args.batch_size)

    for bi in range(n_batches):
        idxs = pending[bi * args.batch_size:(bi + 1) * args.batch_size]
        items = [data[i] for i in idxs]
        prompts = [REWRITE_PROMPT.format(question=it["instruction"], solution=it["output"]).strip()
                   for it in items]

        best: list[str | None] = [None] * len(idxs)
        verdict: list[str] = ["unvalidated"] * len(idxs)
        retries = [0] * len(idxs)
        active = list(range(len(idxs)))

        while active:
            cur = [prompts[j] for j in active]
            gen_budget = budget(cur)
            if gen_budget < 512:
                # Prompt so long that no useful generation fits. Record rather than crash.
                print(f"  WARNING: batch {bi} budget only {gen_budget} tokens; skipping shard batch",
                      flush=True)
                for j in active:
                    best[j] = best[j] or ""
                    verdict[j] = "prompt_too_long"
                active = []
                break
            raw = complete(cur, model=args.model, temperature=args.temperature,
                           max_tokens=gen_budget)
            # Chat + reasoning-parser already returns the answer; harvest_answer is a
            # no-op unless </think> is still embedded (parser absent, or --api completions).
            texts = raw if args.keep_scratchpad else [
                t if (t is not None and "</think>" not in t) else harvest_answer(t or "")
                for t in raw]

            # The validator is the same reasoning model, so it spends tokens thinking
            # before it answers. With a 50-token budget the whole budget goes to
            # reasoning_content and .content comes back EMPTY -- every item then reads as
            # not-GOOD and burns its full retry allowance. It needs room to finish.
            # Only pay for the LLM judge on items the grader cannot decide.
            need_llm = [k for k, (j, t) in enumerate(zip(active, texts))
                        if args.validator == "llm"
                        or not t or not extract_answer(items[j]["output"])]
            evs = [""] * len(active)
            if need_llm:
                got = complete(
                    [EVAL_PROMPT_TEMPLATE.format(question=items[active[k]]["instruction"],
                                                 response=texts[k] or "")
                     for k in need_llm],
                    model=args.model, temperature=0.0, max_tokens=args.eval_max_tokens)
                for k, g in zip(need_llm, got):
                    evs[k] = g or ""

            still: list[int] = []
            for j, text, verd in zip(active, texts, evs):
                best[j] = text  # always keep the latest attempt
                # Exact check first: compare the rewrite's boxed answer with the gold
                # answer of the LIMO trace it was derived from. Falls through to the LLM
                # judge only when no gold answer can be extracted (72/800 traces).
                if args.validator != "llm" and text:
                    gold = extract_answer(items[j]["output"])
                    if gold:
                        pred = extract_answer(text)
                        try:
                            match = bool(pred) and math_equal(pred, gold)
                        except Exception:
                            match = False
                        if match:
                            verdict[j] = "good"
                            continue
                        if retries[j] + 1 >= args.max_retries or spent_retries >= args.retry_budget:
                            verdict[j] = "wrong_answer"  # kept, but does NOT reach gold
                        else:
                            retries[j] += 1
                            spent_retries += 1
                            still.append(j)
                        continue
                if text is None:
                    # </think> never closed: truncated mid-scratchpad, no answer exists.
                    verdict[j] = "no_answer"
                    if retries[j] + 1 < args.max_retries and spent_retries < args.retry_budget:
                        retries[j] += 1
                        spent_retries += 1
                        still.append(j)
                elif not (verd or "").strip():
                    # Validator produced no verdict (ran out of budget while reasoning).
                    # The trace itself may be fine; keep it and flag rather than retry.
                    verdict[j] = "unvalidated"
                elif "GOOD" in verd:
                    verdict[j] = "good"
                elif retries[j] + 1 >= args.max_retries or spent_retries >= args.retry_budget:
                    verdict[j] = "exhausted"  # fell back to last attempt; may be WRONG
                else:
                    retries[j] += 1
                    spent_retries += 1
                    still.append(j)
            active = still

        for j, (gi, it) in enumerate(zip(idxs, items)):
            rec = dict(it)
            rec["output"] = best[j]
            done.append(rec)
            answer_ok = answers_agree(it["output"], best[j] or "", math_equal)
            # Recorded, never used to gate: an item with answer_ok=True and verdict!="good"
            # is a right-answer/wrong-reasoning case, which is precisely what the LLM judge
            # exists to catch and what a boxed-answer check cannot see.
            meta.append({"index": offset + gi, "retries": retries[j], "verdict": verdict[j],
                         "answer_matches_gold": answer_ok,
                         "output_chars": len(best[j] or "")})

        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        el = time.time() - start
        n_done = len(done) - (len(data) - len(pending))
        rate = n_done / el if el else 0
        eta = (len(pending) - n_done) / rate if rate else float("inf")
        n_exh = sum(1 for m in meta if m["verdict"] == "exhausted")
        print(f"[batch {bi+1}/{n_batches}] {len(done)}/{len(data)} | "
              f"retries {spent_retries}/{args.retry_budget} | exhausted {n_exh} | "
              f"elapsed {fmt(el)} | ETA {fmt(eta)}", flush=True)

    n_exh = sum(1 for m in meta if m["verdict"] == "exhausted")
    print(f"\nDONE. {len(done)} traces -> {out_path}")
    print(f"  validated GOOD : {sum(1 for m in meta if m['verdict']=='good')}")
    print(f"  exhausted      : {n_exh}  <-- fell back to last attempt; may be incorrect")
    print(f"  total retries  : {spent_retries}")
    print(f"  sidecar        : {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
