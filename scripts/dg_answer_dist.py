#!/usr/bin/env python
"""Answer distributions of the attacker's own models (results/m8_audit_design.md §16).

For every training problem of a setup-2 pool (GSM8K or MATH, one seed), sample K tries from one model the attacker
owns and store every try: its text, its final answer, whether that answer matches the teacher's released answer, and
whether it is correct. Nothing is trained here. The stored tries give (a) the measure of §16, the surprisal of the
teacher's answer under the attacker's own models, and (b) the solution pools for later answer-filtered students.

Levels (m8 §9.3; in setup 2 the level-D model is the teacher itself, so there is no level D):
  A  base Llama-3.2-3B (the student), few-shot raw-text prompt (GSM8K: Wei et al.'s 8 examples, as dg_star.py;
     MATH: the 4 examples of Lewkowycz et al. 2022), no answer forcing
  B  Llama-3.2-3B-Instruct, the setup's chat prompt, answer forcing as in the setup's evaluation
  C  Qwen2.5-7B-Instruct, same as B

Phase gen (vLLM) writes raw.jsonl; phase grade (CPU, the setup's checker) writes tries.jsonl and summary.json.

    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_answer_dist.py --phase gen \
        --level C --config configs/setup2/gsm8k_U_s123.yaml \
        --teacher-train outputs/setup2/gsm8k_U_s123_<stamp>/teacher/train_standard.json
    ... --phase grade (same arguments)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dg_star import ANSWER_FORCE, fewshot_prompt, fewshot_solution, hard_exit  # noqa: E402

MODELS = {"A": "unsloth/Llama-3.2-3B", "B": "unsloth/Llama-3.2-3B-Instruct", "C": "Qwen/Qwen2.5-7B-Instruct"}

# Level A on MATH: the 4-example MATH prompt of Lewkowycz et al. (2022, Minerva), written by people (the examples'
# solutions are MATH reference solutions). Typed from the published prompt; whitespace may differ slightly.
MINERVA = [
    ("Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
     "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x "
     "\\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, "
     "the domain of the expression is $\\boxed{[2,5)}$.", "[2,5)"),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
     "24"),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must "
     "Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If "
     "he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of "
     "weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad "
     "n&=480/30=\\boxed{16}\n\\end{align*}", "16"),
    ("If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ "
     "where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know "
     "that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$", "-\\frac{2}{3}"),
]
FINAL_RE = re.compile(r"Final Answer:\s*The final answer is\s*\$?(.+?)\$?\s*\.\s*I hope", re.S)


def minerva_prompt(problem: str, hint: str | None = None) -> str:
    """The 4-shot MATH prompt; with a hint (STaR's rationalization, m8 §19), every problem shows its final answer."""
    def q(text, ans):
        return text + (f"\n(The final answer is ${ans}$.)" if hint is not None else "")
    shots = "".join(f"Problem:\n{q(p, a)}\n\nSolution:\n{s}\nFinal Answer: The final answer is ${a}$. I hope it is "
                    f"correct.\n\n" for p, s, a in MINERVA)
    return shots + f"Problem:\n{q(problem.strip(), hint)}\n\nSolution:\n"


def minerva_solution(text: str) -> str | None:
    """Cut a few-shot MATH completion before its 'Final Answer' line and make it end in a boxed answer; None if it
    states no answer (no answer forcing in this mode)."""
    body = text.split("Final Answer:")[0].strip()
    if "\\boxed{" in body:
        return body
    m = FINAL_RE.search(text)
    if m and m.group(1).strip():
        return f"{body}\nThe final answer is $\\boxed{{{m.group(1).strip()}}}$."
    return None


def teacher_ref(r: dict) -> str:
    """The teacher's released answer as a boxed reference for the setup's checker (m8 §16 deviation, 2026-11-09).
    The stored extracted_answer is normalized, which deletes commas: '(0, 0, 3)' becomes '(0 0 3)' and '-3, 5' becomes
    '-3 5', so multi-part MATH answers never matched. This takes the teacher's own boxed text instead, from the same
    source the setup used: its raw solution if that had an answer, else the answer-forced ending."""
    from clean_sweep.eval.correctness import _find_boxed_spans, normalize_answer
    for text, has in ((r.get("trace") or "", r.get("raw_extracted_answer") not in (None, "None", "")),
                      (r.get("af_final_answer_only") or "", r.get("af_extracted_answer") not in (None, "None", ""))):
        if has:
            spans = [c for _, _, c in _find_boxed_spans(text.strip()) if normalize_answer(c).upper() != "ANSWER"]
            return "\\boxed{" + (spans[-1].strip() if spans else r.get("extracted_answer") or "") + "}"
    return ""


def load_pool(a):
    """The seed's training pool, each problem with its true solution and the teacher's released answer."""
    from clean_sweep.config import FullConfig
    from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
    cfg = FullConfig.from_yaml(a.config)
    splits = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                                 holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)
    fmt = format_prompt_gsm8k if cfg.data.dataset_name == "gsm8k" else format_prompt_math
    rows = {r["example_id"]: r for r in json.loads(Path(a.teacher_train).read_text())}
    pool = []
    for ex in splits["train"]:
        r = rows.get(ex["example_id"])
        if r is None or r.get("prompt", "").strip() != ex["problem"].strip():
            raise SystemExit("train problems do not match the teacher file: wrong seed or file")
        pool.append({"example_id": ex["example_id"], "problem": ex["problem"], "solution": ex["solution"],
                     "teacher_answer": r.get("extracted_answer") or "", "teacher_ref": teacher_ref(r),
                     "teacher_correct": str(r.get("correct")) == "True"})
    if a.limit:  # pilot and smoke tests only
        pool = pool[:a.limit]
    out = Path(a.out or f"outputs/setup2_answers/{cfg.data.dataset_name}_{a.level}_s{cfg.run.seed}")
    out.mkdir(parents=True, exist_ok=True)
    return cfg, fmt, pool, out


def phase_gen(a) -> int:
    from vllm import LLM, SamplingParams
    cfg, fmt, pool, out = load_pool(a)
    ds, fewshot = cfg.data.dataset_name, a.level == "A"
    max_len = 4096 if fewshot else cfg.generation.max_prompt_tokens + cfg.generation.max_new_tokens + 128
    tok_name = cfg.model.student_tokenizer if fewshot else MODELS[a.level]  # base model: the setup's student tokenizer
    llm = LLM(model=MODELS[a.level], tokenizer=tok_name, dtype="bfloat16", max_model_len=max_len,
              gpu_memory_utilization=a.gpu_mem, seed=cfg.run.seed)
    tok = llm.get_tokenizer()
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=cfg.generation.max_new_tokens,
                        seed=cfg.run.seed, stop=(["\nQ:"] if ds == "gsm8k" else ["\nProblem:"]) if fewshot else None)
    t0 = time.time()
    if fewshot:
        prompts = [fewshot_prompt(x["problem"]) if ds == "gsm8k" else minerva_prompt(x["problem"]) for x in pool]
    else:
        prompts = [tok.apply_chat_template(fmt(x["problem"]), tokenize=False, add_generation_prompt=True) for x in pool]
    outs = llm.generate(prompts, sp)
    tries = []  # (problem index, text, generated tokens, answer forced)
    need_force = []
    for i, o in enumerate(outs):
        for c in o.outputs:
            if fewshot:
                sol = fewshot_solution(c.text) if ds == "gsm8k" else minerva_solution(c.text)
                tries.append([i, sol if sol is not None else c.text, len(c.token_ids), False, sol is not None])
            else:
                tries.append([i, c.text, len(c.token_ids), False, True])
                if "\\boxed{" not in c.text:
                    need_force.append((len(tries) - 1, list(o.prompt_token_ids) + list(c.token_ids)))
    if need_force:  # answer forcing, as in dg_star.py and the setup's evaluation (built from token ids)
        force_ids = tok.encode(ANSWER_FORCE, add_special_tokens=False)
        room = max_len - cfg.generation.answer_force_max_new_tokens - len(force_ids)
        fo = llm.generate([{"prompt_token_ids": ids[:room] + force_ids} for _, ids in need_force],
                          SamplingParams(n=1, temperature=0.0, max_tokens=cfg.generation.answer_force_max_new_tokens))
        for (j, _), o in zip(need_force, fo):
            tries[j][1] = tries[j][1] + ANSWER_FORCE + o.outputs[0].text
            tries[j][3] = True
    by_problem = [[] for _ in pool]
    for i, text, n_tok, forced, answered in tries:
        by_problem[i].append({"text": text, "tokens": n_tok, "forced": forced, "answered": answered})
    with (out / "raw.jsonl").open("w") as f:
        for x, ts in zip(pool, by_problem):
            f.write(json.dumps({**{k: x[k] for k in ("example_id", "teacher_answer", "teacher_ref", "teacher_correct",
                                                     "solution")},
                                "tries": ts}) + "\n")
    meta = {"level": a.level, "model": MODELS[a.level], "dataset": ds, "seed": cfg.run.seed, "problems": len(pool),
            "k": a.k, "temperature": a.temperature, "top_p": a.top_p, "max_new_tokens": cfg.generation.max_new_tokens,
            "fewshot": fewshot, "forced": len(need_force), "minutes_gen": round((time.time() - t0) / 60, 1),
            "mean_tokens": round(sum(t[2] for t in tries) / max(len(tries), 1), 1)}
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    return 0


def _grade(args):
    from clean_sweep.eval.correctness import extract_answer, is_correct_gsm8k
    text, ref, solution = args
    return extract_answer(text), bool(ref) and is_correct_gsm8k(text, ref), is_correct_gsm8k(text, solution)


def phase_grade(a) -> int:
    import multiprocessing as mp
    _, _, pool, out = load_pool(a)  # the teacher's references are rebuilt from the teacher file, not from raw.jsonl
    refs = {x["example_id"]: x["teacher_ref"] for x in pool}
    rows = [json.loads(line) for line in (out / "raw.jsonl").open()]
    for r in rows:
        r["teacher_ref"] = refs[r["example_id"]]
    jobs = [(t["text"] if t["answered"] else "", r["teacher_ref"], r["solution"]) for r in rows for t in r["tries"]]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(_grade, jobs, chunksize=64)
    j, hist = 0, Counter()
    with (out / "tries.jsonl").open("w") as f:
        for r in rows:
            for t in r["tries"]:
                t["answer"], t["match_teacher"], t["correct"] = res[j]
                j += 1
            m = sum(t["match_teacher"] for t in r["tries"])
            hist[m] += 1
            f.write(json.dumps({**{k: r[k] for k in ("example_id", "teacher_answer", "teacher_ref", "teacher_correct")},
                                "m_teacher": m, "m_correct": sum(t["correct"] for t in r["tries"]), "tries": r["tries"]}) + "\n")
    k = len(rows[0]["tries"]) if rows else 0
    n = len(rows)
    summary = {"problems": n, "k": k, "minutes_grade": round((time.time() - t0) / 60, 1),
               "tries_matching_teacher": round(sum(m * c for m, c in hist.items()) / max(n * k, 1), 4),
               "tries_correct": round(sum(t[2] for t in res) / max(len(res), 1), 4),
               "problems_never_matching": hist[0], "problems_always_matching": hist[k],
               "mean_surprisal_bits_reachable": round(sum(-math.log2(m / k) * c for m, c in hist.items() if m) /
                                                      max(n - hist[0], 1), 3),
               "m_teacher_histogram": {str(m): hist[m] for m in range(k + 1)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "m_teacher_histogram"}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "grade"], required=True)
    ap.add_argument("--level", choices=list(MODELS), required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-train", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.6)  # the setup's own sampling (configs/setup2/*.yaml)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N training problems (pilot and smoke tests)")
    args = ap.parse_args()
    if args.phase == "gen":
        hard_exit(phase_gen(args))
    raise SystemExit(phase_grade(args))
