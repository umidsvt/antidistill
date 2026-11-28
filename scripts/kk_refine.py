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
    ap.add_argument("--chains", type=int, default=1, help="m8 §33 test 3: K refinement chains per puzzle; chain 0 starts "
                    "from the greedy answer, the others from answers sampled at --init-temp; the final answer is the most "
                    "common chain end among the chains left unchanged by the last sweep (all chains if none)")
    ap.add_argument("--init-temp", type=float, default=1.0)
    ap.add_argument("--tta", type=int, default=1, help="m8 §33 test 4: ask every question (solve and each one-person "
                    "step) in T versions, the original and T-1 with every name replaced (kk_views.rename, names from the "
                    "training puzzles, fixed per puzzle); each person's label is the majority over the versions (ties: "
                    "the original's)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(line) for line in open(a.puzzles)]
    if a.limit:
        rows = rows[:a.limit]
    base, K = rows, a.chains
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.student, tokenizer=a.student if "Qwen" in a.student else STUDENT_TOKENIZER, dtype="bfloat16",
              max_model_len=4096,
              gpu_memory_utilization=0.85, seed=a.seed, enable_lora=True, max_lora_rank=128, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    eot = tok.convert_tokens_to_ids("<|im_end|>" if "Qwen" in a.student else "<|eot_id|>")  # m8 §39: Qwen student
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens, seed=a.seed, stop_token_ids=[eot])
    lora = LoRARequest("student", 1, a.adapter)

    def prompts_of(quizzes):
        return [tok.apply_chat_template([user_turn(q)], tokenize=False, add_generation_prompt=True,
                                        date_string=DATE) for q in quizzes]

    def ask(quizzes):
        outs = llm.generate(prompts_of(quizzes), sp, lora_request=lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs], [len(o.outputs[0].token_ids) for o in outs]

    import random as _r
    from kk_views import rename
    pool_names = sorted({x for line in open("data/kk/step2_train.jsonl") for x in json.loads(line)["names"]})
    trng = _r.Random(a.seed + 7)
    versions = {r["id"]: [r] + [rename(r, trng.sample(pool_names, len(r["names"]))) for _ in range(a.tta - 1)]
                for r in rows}

    def ask_vote(items):
        """items: (puzzle row, labels given or None, hidden people or None). Returns (labels per item or None,
        generated tokens per item): with --tta 1 the plain reply; otherwise the per-person majority over versions."""
        qs, owners = [], []
        for n, (r, given, hidden) in enumerate(items):
            for v in versions[r["id"]]:
                qs.append(v["quiz"] if hidden is None else mask_quiz(v, given, hidden))
                owners.append((n, v["names"]))
        texts_, ntok_ = ask(qs)
        labs = [[] for _ in items]
        toks = [0] * len(items)
        for (n, names), t, nt in zip(owners, texts_, ntok_):
            labs[n].append(parse_labels(t, names, reasoning=False))
            toks[n] += nt
        out_ = []
        for n, ls in enumerate(labs):
            ok = [x for x in ls if x is not None]
            if not ok:
                out_.append(None); continue
            m = len(ok[0])
            vote_ = []
            for i in range(m):
                yes = sum(x[i] for x in ok)
                vote_.append(ok[0][i] if 2 * yes == len(ok) else 2 * yes > len(ok))
            out_.append(vote_ if ls[0] is not None or len(ok) > 0 else None)
        return out_, toks

    t0 = time.time()
    if a.tta > 1:
        assert K == 1, "--tta with --chains is not implemented"
        first, ntok = ask_vote([(r, None, None) for r in base])
        texts = None
    else:
        texts, ntok = ask([r["quiz"] for r in base])
    if K > 1:   # the other chains start from sampled answers; rows becomes K copies of each puzzle (chain = k % K)
        sp_s = SamplingParams(n=K - 1, temperature=a.init_temp, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed,
                              stop_token_ids=[eot])
        outs = llm.generate(prompts_of([r["quiz"] for r in base]), sp_s, lora_request=lora, use_tqdm=False)
        tx, nx = [], []
        for t, n, o in zip(texts, ntok, outs):
            tx += [t] + [c.text for c in o.outputs]
            nx += [n] + [len(c.token_ids) for c in o.outputs]
        rows, texts, ntok = [r for r in base for _ in range(K)], tx, nx
    if texts is not None:
        first = [parse_labels(t, r["names"], reasoning=False) for t, r in zip(texts, rows)]
    cur = [list(x) if x is not None else [True] * len(r["names"]) for x, r in zip(first, rows)]
    tokens = list(ntok)
    traj = [[list(c)] for c in cur]

    def acc():   # chain 0 (the greedy start)
        return round(sum(cur[k] == list(rows[k]["solution"]) for k in range(0, len(rows), K)) / len(base), 4)

    changed = [True] * len(rows)

    def vote():
        from collections import Counter
        right = anyr = 0
        for b in range(len(base)):
            ks = range(b * K, (b + 1) * K)
            pool = [tuple(cur[k]) for k in ks if not changed[k]] or [tuple(cur[k]) for k in ks]
            right += list(Counter(pool).most_common(1)[0][0]) == list(base[b]["solution"])
            anyr += any(cur[k] == list(base[b]["solution"]) for k in ks)
        return round(right / len(base), 4), round(anyr / len(base), 4)

    summary = {"adapter": a.adapter, "tta": a.tta, "puzzles": len(rows), "first readable": round(sum(x is not None for x in first) / len(rows), 4),
               "accuracy by step": {"solve": acc()}, "changes by sweep": {}, "unreadable by sweep": {}, "chains": K}
    if K > 1:
        summary["voted accuracy / any chain right, by step"] = {"solve": vote()}
    import random
    rng = random.Random(a.seed)
    schedule = [int(x) for x in a.schedule.split(",")] if a.schedule else [1] * a.sweeps
    summary["schedule"] = schedule
    for s, b in enumerate(schedule, 1):
        changes = unread = 0
        changed = [False] * len(rows)
        nmax = max(len(r["names"]) for r in rows)
        # the blocks of this sweep, per puzzle: b = 1, each person in order; b > 1, N random blocks of b people
        blocks = [[[i] for i in range(len(r["names"]))] if b == 1 else
                  [sorted(rng.sample(range(len(r["names"])), b)) for _ in range(len(r["names"]))] for r in rows]
        for j in range(nmax):
            idx = [k for k in range(len(rows)) if j < len(blocks[k])]
            if a.tta > 1:
                labs_, ntok = ask_vote([(rows[k], cur[k], blocks[k][j]) for k in idx])
            else:
                replies, ntok = ask([mask_quiz(rows[k], cur[k], blocks[k][j]) for k in idx])
                labs_ = [parse_labels(t, rows[k]["names"], reasoning=False) for k, t in zip(idx, replies)]
            for k, lab, nt in zip(idx, labs_, ntok):
                tokens[k] += nt
                if lab is None:
                    unread += 1
                    continue
                for i in blocks[k][j]:
                    if lab[i] != cur[k][i]:
                        cur[k][i] = lab[i]
                        changes += 1
                        changed[k] = True
        for k in range(len(rows)):
            traj[k].append(list(cur[k]))
        summary["accuracy by step"][f"sweep {s} (block {b})"] = acc()
        if K > 1:
            summary["voted accuracy / any chain right, by step"][f"sweep {s}"] = vote()
        summary["changes by sweep"][f"sweep {s}"] = changes
        summary["unreadable by sweep"][f"sweep {s}"] = unread
        print(json.dumps({"sweep": s, "block": b, "accuracy": acc(), "voted/any": vote() if K > 1 else None,
                          "changes": changes, "unreadable": unread}), flush=True)
    summary["mean generated tokens per puzzle"] = round(sum(tokens) / len(base), 1)
    rows_x, rows = rows, base   # the oracle check runs once per puzzle
    if a.truth_check:   # oracle diagnostic only
        right_people = people = fixed = 0
        per = [[None] * len(r["names"]) for r in rows]
        for i in range(max(len(r["names"]) for r in rows)):
            idx = [k for k, r in enumerate(rows) if i < len(r["names"])]
            if a.tta > 1:
                labs_, _ = ask_vote([(rows[k], list(rows[k]["solution"]), [i]) for k in idx])
            else:
                replies, _ = ask([mask_quiz(rows[k], list(rows[k]["solution"]), [i]) for k in idx])
                labs_ = [parse_labels(t, rows[k]["names"], reasoning=False) for k, t in zip(idx, replies)]
            for k, lab in zip(idx, labs_):
                per[k][i] = lab is not None and lab[i] == rows[k]["solution"][i]
        for p in per:
            people += len(p); right_people += sum(p); fixed += all(p)
        summary["truth check: person right given the others' reference labels"] = round(right_people / people, 4)
        summary["truth check: reference answer is a fixed point"] = round(fixed / len(rows), 4)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    with (out / "trajectories.jsonl").open("w") as f:
        for k, (r, tr) in enumerate(zip(rows_x, traj)):
            f.write(json.dumps({"id": r["id"], "chain": k % K, "solution": r["solution"], "steps": tr}) + "\n")
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
