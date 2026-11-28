#!/usr/bin/env python
"""I12, answer-pruned stepwise search (results/m8_audit_design.md §38): build reasoning traces in the student's own words
whose every step agrees with the teacher's answer, without ever showing the answer to the student.

The student solves one person at a time. Each step ends with a claim "Therefore, NAME is a knight/knave." At every
step K candidate steps are sampled; a candidate is kept only if its claim (a) names a person not yet determined,
(b) agrees with the teacher's label for that person, and (c) is the student's own majority view: among the
candidates claiming that person, the share claiming the teacher's label is at least --min-share, with at least
--min-votes of them (so a 50/50 guess does not pass). Among kept persons the one with the most agreeing candidates is
taken, and its first agreeing candidate becomes the step. A step that keeps nothing is retried once; a second miss
ends the search for that puzzle (reported, not trained on). A finished trace is the accepted steps followed by the
CONCLUSION lines; the training example pairs it with the plain puzzle prompt (kk_sft.user_turn), so the student is
trained and scored without the step instruction, as U is.

    CUDA_VISIBLE_DEVICES=7 .venv-dg/bin/python scripts/kk_aps.py --out outputs/kk_phase2/aps/llama_r1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_pilot import INSTRUCTION  # noqa: E402
from kk_sft import DATE, STUDENT_TOKENIZER  # noqa: E402

METHOD = ("Solve it one person at a time. In each step, explain how the statements, together with the identities "
          "already found, determine one more person, and end the step with the sentence \"Therefore, NAME is a "
          "knight.\" or \"Therefore, NAME is a knave.\" Start each step with \"Step k:\". When every person is "
          "determined, give the CONCLUSION.")


def conclusion(names, labels):
    return "CONCLUSION:\n" + "".join(f"({i}) {n} is a {'knight' if k else 'knave'}\n"
                                     for i, (n, k) in enumerate(zip(names, labels), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--student", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="", help="a trained student (later rounds)")
    ap.add_argument("--pool", default="data/kk/step2_train.jsonl")
    ap.add_argument("--teacher-gen", default="outputs/kk_step2/gen")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min-share", type=float, default=0.7)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--max-step-tokens", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", action="store_true", help="debugging: write every candidate of every round")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    qwen = "Qwen" in a.student
    T = load_tries(Path(a.teacher_gen), "T")
    pool = [r for r in map(json.loads, open(a.pool)) if T[r["id"]][0]["labels"] is not None]
    if a.limit:
        pool = pool[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.student, tokenizer=a.student if qwen else STUDENT_TOKENIZER, dtype="bfloat16",
              max_model_len=6144, gpu_memory_utilization=0.85, seed=a.seed, enable_lora=bool(a.adapter),
              max_lora_rank=128, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    lora = LoRARequest("student", 1, a.adapter) if a.adapter else None
    kw = {} if qwen else {"date_string": DATE}
    heads = {r["id"]: tok.apply_chat_template(
        [{"role": "user", "content": f"{INSTRUCTION}\n### Question: {r['quiz']}\n\n{METHOD}"}], tokenize=False,
        add_generation_prompt=True, **kw) for r in pool}
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=1.0, max_tokens=a.max_step_tokens, seed=a.seed,
                        stop=["\nStep ", "CONCLUSION"])
    st = {r["id"]: {"steps": [], "known": {}, "misses": 0, "done": False, "failed": False} for r in pool}
    teacher = {r["id"]: T[r["id"]][0]["labels"] for r in pool}
    pats = {r["id"]: re.compile(r"Therefore,?\s+(" + "|".join(map(re.escape, r["names"])) + r")\s+is\s+a\s+(knight|knave)",
                                re.I) for r in pool}
    t0 = time.time()
    stats = Counter()
    for rnd in range(40):
        active = [r for r in pool if not st[r["id"]]["done"] and not st[r["id"]]["failed"]]
        if not active:
            break
        prompts = [heads[r["id"]] + "".join(st[r["id"]]["steps"]) + f"Step {len(st[r['id']]['steps']) + 1}:"
                   for r in active]
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        for r, o in zip(active, outs):
            s, names, tl = st[r["id"]], r["names"], teacher[r["id"]]
            claims = []   # (person index, label, candidate text cut after its claim)
            for c in o.outputs:
                ms = list(pats[r["id"]].finditer(c.text))
                if not ms:
                    continue
                m = ms[-1]
                who = next(i for i, n in enumerate(names) if n.lower() == m.group(1).lower())
                claims.append((who, m.group(2).lower() == "knight", c.text[:m.end()] + "."))
            stats["candidates"] += len(o.outputs); stats["with a claim"] += len(claims)
            if a.dump:
                with (out / "dump.jsonl").open("a") as fd:
                    fd.write(json.dumps({"round": rnd, "id": r["id"], "known": s["known"], "teacher": tl,
                                         "claims": [(names[w], l) for w, l, _ in claims],
                                         "texts": [c.text for c in o.outputs]}) + "\n")
            best = None
            for who in {w for w, _, _ in claims if w not in s["known"]}:
                about = [x for x in claims if x[0] == who]
                agree = [x for x in about if x[1] == tl[who]]
                if len(agree) >= a.min_votes and len(agree) / len(about) >= a.min_share:
                    if best is None or len(agree) > best[0]:
                        best = (len(agree), who, agree[0][2])
            if best is None:
                s["misses"] += 1
                stats["missed steps"] += 1
                if s["misses"] >= 2:
                    s["failed"] = True
                continue
            _, who, text = best
            s["misses"] = 0
            s["known"][who] = tl[who]
            s["steps"].append(f"Step {len(s['steps']) + 1}:{text.rstrip()}\n")
            stats["accepted steps"] += 1
            if len(s["known"]) == len(names):
                s["done"] = True
        print(json.dumps({"round": rnd, "active": len(active), "done": sum(v["done"] for v in st.values()),
                          "failed": sum(v["failed"] for v in st.values()), "minutes": round((time.time() - t0) / 60, 1)}),
              flush=True)
    rows, done = [], 0
    with (out / "traces.jsonl").open("w") as f:
        for r in pool:
            s = st[r["id"]]
            text = "".join(s["steps"]) + "\n" + conclusion(r["names"], teacher[r["id"]]) if s["done"] else None
            done += s["done"]
            f.write(json.dumps({"id": r["id"], "done": s["done"], "steps": len(s["steps"]), "text": text}) + "\n")
            if s["done"]:
                rows.append({"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"],
                             "text": text, "correct": teacher[r["id"]] == list(r["solution"])})
    (out / "sft.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary = {"student": a.student, "adapter": a.adapter or None, "k": a.k, "temperature": a.temperature,
               "min_share": a.min_share, "min_votes": a.min_votes, "puzzles": len(pool), "traces": done,
               "coverage": round(done / len(pool), 4), "traces with the right answer": sum(x["correct"] for x in rows),
               "stats": dict(stats), "minutes": round((time.time() - t0) / 60, 1)}
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
