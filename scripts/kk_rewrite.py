#!/usr/bin/env python
"""Phase 1, first defense: trace rewriting on the puzzle testbed (results/m8_audit_design.md §21).

Ma et al. 2026's optimized rewriting instruction (their Appendix A.2), applied by the teacher itself
(R1-Distill-Qwen-32B) to each of its step-2 training solutions. The rewriter writes directly (an empty thinking block is
prefilled). Answers are preserved by construction: any CONCLUSION the rewriter writes is cut off and the teacher's own
CONCLUSION section is appended verbatim. Output: a training set for scripts/kk_sft.py (`text` = <think> rewrite </think>
+ the teacher's CONCLUSION, the shape of U's targets).

    .venv-dg/bin/python scripts/kk_rewrite.py --pool data/kk/step2_train.jsonl --gen outputs/kk_step2/gen \
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

REWRITER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# Ma et al. 2026 (arXiv 2602.15143v2), Appendix A.2, "Optimized instruction for rewriting", verbatim.
INSTRUCTION = ("Your task is to process a provided chain of logical deductions and reformulate it utilizing an esoteric, "
               "formal, and densely technical lexicon, thereby obfuscating its clarity for a non-specialist audience. "
               "Refrain from any extraneous discourse.")
CONC = re.compile(r"[*_`#\s]*conclusion[*_`]*\s*:", re.I)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gen", required=True, help="step-2 writers' folder (the teacher's first try per puzzle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--part", type=int, default=1)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0, help="first N puzzles of this part (smoke tests only)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    pool = [json.loads(line) for line in open(a.pool)]
    pool = pool[a.part - 1::a.parts]
    if a.limit:
        pool = pool[:a.limit]
    T = load_tries(Path(a.gen), "T")
    llm = LLM(model=REWRITER, dtype="bfloat16", tensor_parallel_size=a.tp, max_model_len=8192 + a.max_tokens + 512,
              gpu_memory_utilization=a.gpu_mem, seed=a.seed)
    tok = llm.get_tokenizer()
    prompts, sources = [], []
    for r in pool:
        src = T[r["id"]][0]["text"].replace("</think>", "\n")   # the teacher's solution as plain text
        sources.append(src)
        p = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + "\n\n" + src}], tokenize=False,
                                    add_generation_prompt=True)
        p = p + "\n</think>\n\n" if p.rstrip().endswith("<think>") else p + "<think>\n\n</think>\n\n"
        prompts.append(p)
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, capped, ratios, no_conc = [], 0, [], 0
    for r, o, src in zip(pool, outs, sources):
        c = o.outputs[0]
        rewrite = c.text
        m = CONC.search(rewrite)
        if m:
            rewrite = rewrite[:m.start()]
        conc = teacher_conclusion(T[r["id"]][0]["text"], r["n_people"])
        labels = T[r["id"]][0]["labels"]
        if conc is None and labels is not None:  # readable answer in another layout: the same answer, standard layout
            conc = "CONCLUSION:\n" + "\n".join(f"({i + 1}) {n} is a {'knight' if k else 'knave'}"
                                                for i, (n, k) in enumerate(zip(r["names"], labels)))
        no_conc += conc is None
        text = "<think>\n" + rewrite.strip() + "\n</think>\n\n" + (conc or "")
        capped += c.finish_reason == "length"
        ratios.append(len(c.token_ids) / max(T[r["id"]][0]["tokens"], 1))
        rows.append({"id": r["id"], "n_people": r["n_people"], "quiz": r["quiz"], "names": r["names"], "text": text,
                     "correct": T[r["id"]][0]["correct"], "rewrite_tokens": len(c.token_ids),
                     "orig_tokens": T[r["id"]][0]["tokens"], "finish_reason": c.finish_reason})
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    summary = {"part": a.part, "parts": a.parts, "rewritten": len(rows), "capped": capped,
               "teacher_without_conclusion": no_conc, "mean_length_ratio": round(sum(ratios) / max(len(ratios), 1), 3),
               "minutes": round((time.time() - t0) / 60, 1)}
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
