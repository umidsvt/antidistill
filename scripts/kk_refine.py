#!/usr/bin/env python
"""Phase 2, I9 (results/m8_audit_design.md §33): answer-space iterative refinement with an answers-only student.

The student must have been trained on infill views (kk_views.py views: the teacher's labels of some people given, the
full answer as target), so it can label one person given the others. At test time:
  1. solve: the plain puzzle, greedy, gives a first answer A (unreadable: every person a knight);
  2. sweeps: for each person i in turn, the infill question with A's labels of everyone but i (kk_task_sets.mask_quiz);
     the reply's label for i replaces A[i] (unreadable reply: A[i] kept). Repeated --sweeps times.
Accuracy (exact, against the reference) is reported after the solve step and after every sweep, with the generated
tokens per puzzle. --truth-check adds an oracle diagnostic (never part of the method): with every other person given
their reference label, how often is each person labeled right, and on how many puzzles is the reference answer a
fixed point of the sweep (every person right)?

    CUDA_VISIBLE_DEVICES=3 .venv-dg/bin/python scripts/kk_refine.py --adapter outputs/kk_phase2/students/I6_views_s123/adapter \
        --out outputs/kk_phase2/refine/I6_views_s123 --sweeps 3 --truth-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_pilot import parse_labels  # noqa: E402
from kk_sft import DATE, STUDENT_TOKENIZER, user_turn  # noqa: E402
from kk_task_sets import mask_quiz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--puzzles", default="data/kk/step2_test.jsonl")
    ap.add_argument("--student", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--schedule", default="", help="block size per sweep, e.g. 1,1,1,2,3 (overrides --sweeps); a sweep "
                    "with block size b > 1 asks N random blocks of b people (seeded), each block's labels replaced jointly")
    ap.add_argument("--truth-check", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(line) for line in open(a.puzzles)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.student, tokenizer=STUDENT_TOKENIZER, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, seed=a.seed, enable_lora=True, max_lora_rank=128, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens, seed=a.seed, stop_token_ids=[eot])
    lora = LoRARequest("student", 1, a.adapter)

    def ask(quizzes):
        prompts = [tok.apply_chat_template([user_turn(q)], tokenize=False, add_generation_prompt=True,
                                           date_string=DATE) for q in quizzes]
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs], [len(o.outputs[0].token_ids) for o in outs]

    t0 = time.time()
    texts, ntok = ask([r["quiz"] for r in rows])
    first = [parse_labels(t, r["names"], reasoning=False) for t, r in zip(texts, rows)]
    cur = [list(x) if x is not None else [True] * len(r["names"]) for x, r in zip(first, rows)]
    tokens = list(ntok)
    traj = [[list(c)] for c in cur]

    def acc():
        return round(sum(c == list(r["solution"]) for c, r in zip(cur, rows)) / len(rows), 4)

    summary = {"adapter": a.adapter, "puzzles": len(rows), "first readable": round(sum(x is not None for x in first) / len(rows), 4),
               "accuracy by step": {"solve": acc()}, "changes by sweep": {}, "unreadable by sweep": {}}
    import random
    rng = random.Random(a.seed)
    schedule = [int(x) for x in a.schedule.split(",")] if a.schedule else [1] * a.sweeps
    summary["schedule"] = schedule
    for s, b in enumerate(schedule, 1):
        changes = unread = 0
        nmax = max(len(r["names"]) for r in rows)
        # the blocks of this sweep, per puzzle: b = 1, each person in order; b > 1, N random blocks of b people
        blocks = [[[i] for i in range(len(r["names"]))] if b == 1 else
                  [sorted(rng.sample(range(len(r["names"])), b)) for _ in range(len(r["names"]))] for r in rows]
        for j in range(nmax):
            idx = [k for k in range(len(rows)) if j < len(blocks[k])]
            replies, ntok = ask([mask_quiz(rows[k], cur[k], blocks[k][j]) for k in idx])
            for k, t, nt in zip(idx, replies, ntok):
                tokens[k] += nt
                lab = parse_labels(t, rows[k]["names"], reasoning=False)
                if lab is None:
                    unread += 1
                    continue
                for i in blocks[k][j]:
                    if lab[i] != cur[k][i]:
                        cur[k][i] = lab[i]
                        changes += 1
        for k in range(len(rows)):
            traj[k].append(list(cur[k]))
        summary["accuracy by step"][f"sweep {s} (block {b})"] = acc()
        summary["changes by sweep"][f"sweep {s}"] = changes
        summary["unreadable by sweep"][f"sweep {s}"] = unread
        print(json.dumps({"sweep": s, "block": b, "accuracy": acc(), "changes": changes, "unreadable": unread}), flush=True)
    summary["mean generated tokens per puzzle"] = round(sum(tokens) / len(tokens), 1)
    if a.truth_check:   # oracle diagnostic only
        right_people = people = fixed = 0
        per = [[None] * len(r["names"]) for r in rows]
        for i in range(max(len(r["names"]) for r in rows)):
            idx = [k for k, r in enumerate(rows) if i < len(r["names"])]
            replies, _ = ask([mask_quiz(rows[k], list(rows[k]["solution"]), [i]) for k in idx])
            for k, t in zip(idx, replies):
                lab = parse_labels(t, rows[k]["names"], reasoning=False)
                per[k][i] = lab is not None and lab[i] == rows[k]["solution"][i]
        for p in per:
            people += len(p); right_people += sum(p); fixed += all(p)
        summary["truth check: person right given the others' reference labels"] = round(right_people / people, 4)
        summary["truth check: reference answer is a fixed point"] = round(fixed / len(rows), 4)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    with (out / "trajectories.jsonl").open("w") as f:
        for r, tr in zip(rows, traj):
            f.write(json.dumps({"id": r["id"], "solution": r["solution"], "steps": tr}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    try:  # vLLM can hang at interpreter exit (CLAUDE.md rule 6)
        import psutil
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
