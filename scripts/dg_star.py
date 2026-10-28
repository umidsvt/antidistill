#!/usr/bin/env python
"""Setup 2, level A by self-training (results/m8_audit_design.md §9.3): one round of STaR from the UNTRAINED student,
using only the teacher's released final answers.

Phase "gen" (vLLM): for every training problem, sample K attempts from the untrained student in the setup's own
prompt format (answer forcing as in its evaluation) and keep the first that reaches the teacher's answer. For
problems never solved, give the teacher's answer as a hint ("rationalization", as in STaR) and keep a worked solution
that ends at it; the training text keeps the original prompt, without the hint. Every problem is kept (no problem
is dropped); a problem unsolved even with the hint after K_SALVAGE samples is recorded and reported.
Phase "train": fine-tune the untrained student on the result exactly as the distilled students are (LoRA r128,
3 epochs) and score it with the setup's own evaluation.
--fewshot (approved 2026-10-27): the untrained base model does not follow the setup's chat prompt, so its attempts
were mostly unrelated text; this mode prompts it with 8 hand-written worked examples instead (raw text, no answer
forcing). Training and scoring are unchanged: setup's prompt, no examples.

    PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_star.py --phase gen \
        --config configs/setup2/gsm8k_U_s123.yaml --teacher-train outputs/setup2/gsm8k_U_s123_<stamp>/teacher/train_standard.json
    ... --phase train (same arguments)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ANSWER_FORCE = "\n\n**Final Answer**\n\\[\\boxed{"

# --fewshot (m8 §9.3, approved 2026-10-27): the 8 hand-written GSM8K chain-of-thought examples of Wei et al. (2022),
# used as a raw-text prompt for the untrained base model; only the generation prompt changes.
FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there "
     "will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been "
     "21 - 15 = 6.", "6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.", "5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they "
     "had 74 - 35 = 39.", "39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason "
     "give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.", "8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.", "9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to "
     "thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were "
     "added. 9 + 20 is 29.", "29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls "
     "did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had "
     "35 - 2 = 33 golf balls.", "33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. "
     "23 - 15 is 8.", "8"),
]
ANSWER_RE = re.compile(r"The answer is\s*([^\n]*)")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def fewshot_prompt(problem: str, hint: str | None = None) -> str:
    """Raw-text 8-shot prompt; with a hint, every question (the examples' too) carries its answer."""
    def q(text, ans):
        return text + (f" (The answer is {ans}.)" if hint else "")
    shots = "".join(f"Q: {q(text, ans)}\nA: {steps} The answer is {ans}.\n\n" for text, steps, ans in FEWSHOT)
    return shots + f"Q: {q(problem.strip(), hint)}\nA:"


def fewshot_solution(text: str) -> str | None:
    """Cut a few-shot completion at its first stated answer and end it in a boxed answer; None if no number is
    stated (no answer forcing in this mode)."""
    m = ANSWER_RE.search(text)
    nums = NUMBER_RE.findall(m.group(1)) if m else []
    if not nums:
        return None
    return f"{text[:m.start()].strip()} The answer is \\boxed{{{nums[0].replace(',', '')}}}.".strip()


def load_common(a):
    from clean_sweep.config import FullConfig
    from clean_sweep.data import format_prompt_gsm8k, format_prompt_math, load_dataset_splits
    from clean_sweep.utils import set_seed
    cfg = FullConfig.from_yaml(a.config)
    set_seed(cfg.run.seed)
    splits = load_dataset_splits(cfg.data.dataset_name, seed=cfg.run.seed, train_size=cfg.data.train_size,
                                 holdout_size=cfg.data.holdout_size, test_size=cfg.data.test_size)
    fmt = format_prompt_gsm8k if cfg.data.dataset_name == "gsm8k" else format_prompt_math
    rows = {r["example_id"]: r for r in json.loads(Path(a.teacher_train).read_text())}
    train = []
    for ex in splits["train"]:
        r = rows.get(ex["example_id"])
        if r is None or r.get("prompt", "").strip() != ex["problem"].strip():
            raise SystemExit("train problems do not match the teacher file: wrong seed or file")
        train.append({"example_id": ex["example_id"], "problem": ex["problem"], "teacher_answer": r.get("extracted_answer") or ""})
    if a.limit:  # smoke tests only
        train = train[:a.limit]
    tag = "fewshot_" if a.fewshot else ""
    out = Path(a.out or Path(cfg.run.output_dir) / f"star_{tag}{cfg.data.dataset_name}_s{cfg.run.seed}")
    out.mkdir(parents=True, exist_ok=True)
    return cfg, splits, fmt, train, out


def phase_gen(a) -> int:
    from vllm import LLM, SamplingParams
    from clean_sweep.eval.correctness import is_correct_gsm8k
    cfg, _, fmt, train, out = load_common(a)
    # the 8-shot prompt is about 1,000 tokens longer than the setup's own prompt
    max_len = 4096 if a.fewshot else cfg.generation.max_prompt_tokens + cfg.generation.max_new_tokens + 128
    llm = LLM(model=cfg.model.student, tokenizer=cfg.model.student_tokenizer, dtype="bfloat16",
              max_model_len=max_len, gpu_memory_utilization=a.gpu_mem, seed=cfg.run.seed)
    tok = llm.get_tokenizer()
    force_ids = tok.encode(ANSWER_FORCE, add_special_tokens=False)
    room = max_len - cfg.generation.answer_force_max_new_tokens - len(force_ids)

    def render(problem, hint=None):
        msgs = fmt(problem)
        if hint:
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + f"(The correct final answer is {hint}. "
                        "Show the steps that lead to it.)\n"}
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def attempt(items, hint: bool, n: int, name: str):
        """items: list of train dicts; returns {example_id: solution text or None}. Each round is saved, so a
        restarted job resumes after the last finished round."""
        cache = out / f"round_{name}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        if a.fewshot:
            prompts = [fewshot_prompt(x["problem"], x["teacher_answer"] if hint else None) for x in items]
            sp = SamplingParams(n=n, temperature=a.temperature, top_p=0.95, max_tokens=cfg.generation.max_new_tokens,
                                seed=cfg.run.seed, stop=["\nQ:"])
            outs = llm.generate(prompts, sp)
            texts = [(i, fewshot_solution(c.text)) for i, o in enumerate(outs) for c in o.outputs]
            texts = [(i, t) for i, t in texts if t]
        else:
            prompts = [render(x["problem"], x["teacher_answer"] if hint else None) for x in items]
            sp = SamplingParams(n=n, temperature=a.temperature, top_p=0.95, max_tokens=cfg.generation.max_new_tokens,
                                seed=cfg.run.seed)
            outs = llm.generate(prompts, sp)
            texts, need_force = [], []
            for i, o in enumerate(outs):
                for c in o.outputs:
                    texts.append((i, c.text))
                    if "\\boxed{" not in c.text:
                        need_force.append((len(texts) - 1, list(o.prompt_token_ids) + list(c.token_ids)))
            # answer forcing, as in the setup's evaluation: append the final-answer cue and let the student finish it.
            # Built from the generated token ids: re-encoding the decoded text can be much longer (2,476 tokens
            # against a 1,664 limit) and crashed every gen run on 2026-10-23.
            if need_force:
                fp = [{"prompt_token_ids": ids[:room] + force_ids} for _, ids in need_force]
                fo = llm.generate(fp, SamplingParams(n=1, temperature=0.0, max_tokens=cfg.generation.answer_force_max_new_tokens))
                for (j, _), o in zip(need_force, fo):
                    i, t = texts[j]
                    texts[j] = (i, t + ANSWER_FORCE + o.outputs[0].text)
        found = {x["example_id"]: None for x in items}
        for i, t in texts:
            x = items[i]
            if found[x["example_id"]] is None and x["teacher_answer"] and is_correct_gsm8k(t, f"\\boxed{{{x['teacher_answer']}}}"):
                found[x["example_id"]] = t
        cache.write_text(json.dumps(found))
        return found

    t0 = time.time()
    solved = attempt(train, hint=False, n=a.k, name="own")
    n_round1 = sum(v is not None for v in solved.values())
    left = [x for x in train if solved[x["example_id"]] is None]
    hinted = attempt(left, hint=True, n=a.k_hint, name="hint") if left else {}
    n_hint = sum(v is not None for v in hinted.values())
    left2 = [x for x in left if hinted.get(x["example_id"]) is None]
    salvaged = attempt(left2, hint=True, n=a.k_salvage, name="salvage") if left2 else {}
    n_salv = sum(v is not None for v in salvaged.values())
    traces, unsolved = [], []
    for x in train:
        t = solved.get(x["example_id"]) or hinted.get(x["example_id"]) or salvaged.get(x["example_id"])
        if t is None:
            unsolved.append(x["example_id"]); continue
        src = "own" if solved.get(x["example_id"]) else "hint"
        traces.append({"example_id": x["example_id"], "problem": x["problem"], "trace": t, "source": src})
    summary = {"train_problems": len(train), "solved_own": n_round1, "solved_with_hint": n_hint, "salvaged_with_hint": n_salv,
               "unsolved": len(unsolved), "k": a.k, "k_hint": a.k_hint, "k_salvage": a.k_salvage,
               "temperature": a.temperature, "fewshot": a.fewshot, "minutes": round((time.time() - t0) / 60, 1)}
    (out / "traces.json").write_text(json.dumps(traces, indent=1))
    (out / "gen_summary.json").write_text(json.dumps({**summary, "unsolved_ids": unsolved}, indent=1))
    print(json.dumps(summary))
    return 0


def phase_train(a) -> int:
    import torch
    from clean_sweep.generation import generate_teacher_traces
    from clean_sweep.train import run_distill
    from clean_sweep.utils import set_seed
    cfg, splits, fmt, _, out = load_common(a)
    traces = json.loads((out / "traces.json").read_text())
    device = torch.device("cuda")
    set_seed(cfg.run.seed)
    _, model, tok = run_distill(cfg=cfg, train_traces=traces, holdout_traces=[], output_dir=out / "student",
                                device=device, mode="naive", beta_s=1.0)
    set_seed(cfg.run.seed)
    compact, _ = generate_teacher_traces(cfg=cfg, dataset=splits["test"], format_prompt=fmt, method_name="standard",
                                         device=device, model=model, tokenizer=tok)
    acc = sum(1 for r in compact if r["correct"]) / max(len(compact), 1)
    (out / "test_star.json").write_text(json.dumps(compact, indent=1))
    gen = json.loads((out / "gen_summary.json").read_text())
    res = {"student": cfg.model.student, "seed": cfg.run.seed, "dataset": cfg.data.dataset_name, "n_test": len(compact),
           "accuracy": acc, "train_examples": len(traces), **{k: v for k, v in gen.items() if k != "unsolved_ids"}}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"STaR (one round) from the untrained student, {cfg.data.dataset_name} test (seed {cfg.run.seed}): {acc:.4f} "
          f"(n={len(compact)}; trained on {len(traces)} solutions, {gen['solved_own']} its own) -> {out}")
    return 0


def hard_exit(rc: int) -> None:
    """Exit without waiting on vLLM, which can hang at interpreter exit; stop this process's own vLLM engine
    processes first, or they live on as orphans holding the GPU (2026-10-27, 41 GB on GPU 5)."""
    sys.stdout.flush(); sys.stderr.flush()
    try:
        import psutil
        for child in psutil.Process().children(recursive=True):
            child.kill()
    except Exception:
        pass
    os._exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gen", "train"], required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-train", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k-hint", type=int, default=8)
    ap.add_argument("--k-salvage", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="first N training problems (smoke test)")
    ap.add_argument("--fewshot", action="store_true", help="8-shot raw-text generation prompt (m8 §9.3, 2026-10-27)")
    args = ap.parse_args()
    if args.phase == "gen":
        # vLLM can hang at interpreter exit after everything is written (few-shot smoke test, 2026-10-27)
        hard_exit(phase_gen(args))
    raise SystemExit(phase_train(args))
