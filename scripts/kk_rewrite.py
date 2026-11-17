#!/usr/bin/env python
"""Phase 1, first defense: trace rewriting on the puzzle testbed (results/m8_audit_design.md §21 and its amendments).

Ma et al. 2026's optimized rewriting instruction (their Appendix A.2), applied to each of the teacher's step-2 training
solutions. Answers are preserved by construction: any CONCLUSION the rewriter writes is cut off and the teacher's own
CONCLUSION section is appended verbatim. Output: a training set for scripts/kk_sft.py (`text` = <think> rewrite
</think> + the teacher's CONCLUSION, the shape of U's targets).

Rewriters:
- `gpt-oss-120b` (§21 second amendment, the run in use): Ma et al.'s rewriter. Run with `.venv-infer` (vLLM 0.11).
  Its analysis channel is discarded and the final channel kept.
- `teacher` (R1-Distill-Qwen-32B; the first two smoke tests): `--think 1` lets it think first, `--think 0` prefills an
  empty thinking block. Run with `.venv-dg`.
A try whose reasoning never ends within the budget is retried with the next two seeds.

    .venv-infer/bin/python scripts/kk_rewrite.py --pool data/kk/step2_train.jsonl --gen outputs/kk_step2/gen \
        --out outputs/kk_step2/rewrite/part1.jsonl --tp 2 --part 1 --parts 2 [--limit 16]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kk_build import load_tries  # noqa: E402
from kk_pilot import parse_labels  # noqa: E402

TEACHER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
REWRITERS = {"gpt-oss-120b": "openai/gpt-oss-120b", "teacher": TEACHER}
# Ma et al. 2026 (arXiv 2602.15143v2), Appendix A.2, "Optimized instruction for rewriting", verbatim.
INSTRUCTION = ("Your task is to process a provided chain of logical deductions and reformulate it utilizing an esoteric, "
               "formal, and densely technical lexicon, thereby obfuscating its clarity for a non-specialist audience. "
               "Refrain from any extraneous discourse.")
CONC = re.compile(r"[*_`#\s]*conclusion[*_`]*\s*:", re.I)
FINAL = "<|channel|>final<|message|>"          # gpt-oss (harmony format): start of the final channel
HARMONY_TOKEN = re.compile(r"<\|[a-z_]+\|>")


def teacher_conclusion(text: str, n_people: int) -> str | None:
    """The teacher's own CONCLUSION section (heading and its numbered lines 1..N), verbatim; None if it has none."""
    part = text.rsplit("</think>", 1)[1] if "</think>" in text else None
    if part is None:
        return None
    m = list(CONC.finditer(part))
    if not m:
        return None
    tail = part[m[-1].start():].strip()
    lines, seen = [], set()
    for line in tail.split("\n"):
        lines.append(line)
        k = re.match(r"\s*[*_]*\((\d+)\)", line)
        if k:
            seen.add(int(k.group(1)))
        if len(seen) >= n_people:
            break
    return "\n".join(lines).strip() if len(seen) >= n_people else None


def split_output(text: str, rewriter: str, think: int) -> tuple[bool, str, str]:
    """(reasoning ended, the rewriter's reasoning, the rewrite) of one generation."""
    if rewriter == "gpt-oss-120b":
        if FINAL not in text:
            return False, text, HARMONY_TOKEN.sub(" ", text).strip()
        head, rewrite = text.rsplit(FINAL, 1)
        return True, head, HARMONY_TOKEN.sub("", rewrite).strip()
    if not think:
        return True, "", text
    if "</think>" not in text:
        return False, text, text
    head, rewrite = text.rsplit("</think>", 1)
    return True, head, rewrite


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gen", required=True, help="step-2 writers' folder (the teacher's first try per puzzle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rewriter", choices=sorted(REWRITERS), default="gpt-oss-120b")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--think", type=int, default=1, help="teacher only. 1: thinks first; 0: empty thinking block")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles of this part (smoke tests only)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    oss = a.rewriter == "gpt-oss-120b"
    pool = [json.loads(line) for line in open(a.pool)]
    pool = pool[a.part - 1::a.parts]
    if a.limit:
        pool = pool[:a.limit]
    T = load_tries(Path(a.gen), "T")
    # DEVIATION (antidistill, m8 §21): prefix caching off for gpt-oss. On L40S with vLLM 0.11 every prompt that reuses
    # the cached shared prefix (system message + instruction) decodes NaN logits ("!!!..."): 15 of 16 in the smoke
    # test; 0 of 16 with prefix caching off.
    extra = {"enable_prefix_caching": False} if oss else {}
    llm = LLM(model=REWRITERS[a.rewriter], dtype="auto" if oss else "bfloat16", tensor_parallel_size=a.tp,
              max_model_len=8192 + a.max_tokens + 512, gpu_memory_utilization=a.gpu_mem, seed=a.seed, **extra)
    tok = llm.get_tokenizer()
    ttok = AutoTokenizer.from_pretrained(TEACHER)  # lengths in the teacher's tokens, the units of orig_tokens

    def n_tok(s: str) -> int:
        return len(ttok(s, add_special_tokens=False).input_ids)
    prompts = []
    for r in pool:
        src = T[r["id"]][0]["text"].replace("</think>", "\n")   # the teacher's solution as plain text
        p = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + "\n\n" + src}], tokenize=False,
                                    add_generation_prompt=True)
        if not oss:
            if a.think:
                p = p if p.rstrip().endswith("<think>") else p + "<think>\n"
            else:
                p = p + "\n</think>\n\n" if p.rstrip().endswith("<think>") else p + "<think>\n\n</think>\n\n"
        prompts.append(p)
    t0 = time.time()

    def sp(seed):
        if oss:  # Ma et al.: temperature 0.6; vLLM defaults otherwise. Special tokens kept to find the final channel.
            return SamplingParams(n=1, temperature=0.6, max_tokens=a.max_tokens, seed=seed, skip_special_tokens=False)
        return SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=seed)

    def ended(c) -> bool:
        return split_output(c.text, a.rewriter, a.think)[0]
    outs = [o.outputs[0] for o in llm.generate(prompts, sp(a.seed))]
    tries = [1] * len(outs)
    for extra in (1, 2) if (oss or a.think) else ():   # reasoning that never ended: next seeds, those solutions only
        todo = [i for i, c in enumerate(outs) if not ended(c)]
        if not todo:
            break
        for i, o in zip(todo, llm.generate([prompts[i] for i in todo], sp(a.seed + extra))):
            outs[i], tries[i] = o.outputs[0], tries[i] + 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, capped, ratios, no_conc, never_ended, think_toks = [], 0, [], 0, 0, []
    own = {"wrote": 0, "agrees": 0, "teacher_wrong": 0, "agrees_teacher_wrong": 0, "true_teacher_wrong": 0}
    for r, c, n_try in zip(pool, outs, tries):
        t = T[r["id"]][0]
        closed, reasoning, rewrite = split_output(c.text, a.rewriter, a.think)
        never_ended += not closed
        if closed and reasoning:
            think_toks.append(len(tok(reasoning, add_special_tokens=False).input_ids))
        m = CONC.search(rewrite)
        own_labels = None
        if m:
            own_labels = parse_labels(rewrite[m.start():], r["names"], False)
            rewrite = rewrite[:m.start()]
        labels = t["labels"]
        if own_labels is not None and labels is not None:   # the rewriter's own conclusion, against the teacher's
            own["wrote"] += 1
            own["agrees"] += own_labels == labels
            if not t["correct"]:
                own["teacher_wrong"] += 1
                own["agrees_teacher_wrong"] += own_labels == labels
                own["true_teacher_wrong"] += own_labels == list(r["solution"])
        conc = teacher_conclusion(t["text"], r["n_people"])
        if conc is None and labels is not None:  # readable answer in another layout: the same answer, standard layout
            conc = "CONCLUSION:\n" + "\n".join(f"({i + 1}) {n} is a {'knight' if k else 'knave'}"
                                                for i, (n, k) in enumerate(zip(r["names"], labels)))
        no_conc += conc is None
        text = "<think>\n" + rewrite.strip() + "\n</think>\n\n" + (conc or "")
        capped += c.finish_reason == "length"
        n_rw = n_tok(rewrite.strip())
        ratios.append(n_rw / max(t["tokens"], 1))
        rows.append({"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"], "text": text,
                     "correct": t["correct"], "rewrite_tokens": n_rw, "generated_tokens": len(c.token_ids),
                     "orig_tokens": t["tokens"], "finish_reason": c.finish_reason, "tries": n_try,
                     "reasoning_ended": closed, "own_conclusion_agrees": None if own_labels is None or labels is None
                     else own_labels == labels, "rewriter": a.rewriter, "raw": c.text})
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary = {"part": a.part, "parts": a.parts, "rewriter": a.rewriter, "think": a.think if not oss else None,
               "rewritten": len(rows), "capped": capped, "never_ended": never_ended,
               "retried": sum(t > 1 for t in tries),
               "mean_reasoning_tokens": round(sum(think_toks) / max(len(think_toks), 1)),
               "mean_rewrite_tokens": round(sum(x["rewrite_tokens"] for x in rows) / max(len(rows), 1)),
               "teacher_without_conclusion": no_conc, "mean_length_ratio": round(sum(ratios) / max(len(ratios), 1), 3),
               "own_conclusion": own, "minutes": round((time.time() - t0) / 60, 1)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
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
