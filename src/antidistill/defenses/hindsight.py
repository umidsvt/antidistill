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
import time
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
    p.add_argument("--model-max-len", type=int, default=65536,
                   help="Must match the server's --max-model-len. vLLM requires "
                        "input_tokens + max_tokens <= this, so max_tokens is reduced per batch "
                        "when the prompt is long. The rewrite prompt embeds the full LIMO "
                        "solution, which reaches ~33k tokens for the longest traces.")
    p.add_argument("--temperature", type=float, default=0.4)
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

    def complete(**kw):
        """One API call, retried on transient failures. A multi-hour job must not die on one."""
        last = None
        for attempt in range(args.api_retries + 1):
            try:
                return client.completions.create(**kw)
            except Exception as e:                      # timeout, conn reset, 5xx
                last = e
                wait = min(60, 5 * 2 ** attempt)
                print(f"  API error ({type(e).__name__}) attempt {attempt+1}/"
                      f"{args.api_retries+1}; retrying in {wait}s", flush=True)
                time.sleep(wait)
        raise last

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
            resp = complete(
                model=args.model,
                prompt=cur,
                temperature=args.temperature,
                max_tokens=gen_budget,
            )
            texts = [FORCED_PREFIX + c.text for c in resp.choices]

            ev = complete(
                model=args.model,
                prompt=[EVAL_PROMPT_TEMPLATE.format(question=items[j]["instruction"], response=t)
                        for j, t in zip(active, texts)],
                temperature=0.0,
                max_tokens=50,
            )
            evs = [c.text for c in ev.choices]

            still: list[int] = []
            for j, text, verd in zip(active, texts, evs):
                best[j] = text  # always keep the latest attempt
                if "GOOD" in verd:
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
            meta.append({"index": offset + gi, "retries": retries[j], "verdict": verdict[j],
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
