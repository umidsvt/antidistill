#!/usr/bin/env python
"""Phase 1 on the puzzle testbed: ADS and PoE (results/m8_audit_design.md §23), with the Distillation Game's own code.

  grad  The proxy student's (Qwen2.5-3B) holdout gradient for ADS: the mean gradient of its completion-only loss on
        the teacher's first tries for 64 + 64 pilot puzzles at N = 7, 8 (their compute_student_holdout_grad, with
        the puzzle prompt instead of the GSM8K one).
  gen   The teacher (R1-Distill-Qwen-32B, Hugging Face, split over the visible GPUs) writes one try per puzzle under
        standard sampling, ADS (their AntidistillationLogitsProcessor, identity penalty, eps 0.01) or PoE (their
        ProductOfExpertsLogitsProcessor). The teacher fills all visible GPUs but the last, where the proxies sit. Output: tries.jsonl in the
        format of scripts/kk_pilot.py, and summary.json. --timing writes no labels and no accuracy (§23 pilot).

    CUDA_VISIBLE_DEVICES=0 .venv-dg/bin/python scripts/kk_defend.py grad --out outputs/kk_defense/proxy_grad.pt
    CUDA_VISIBLE_DEVICES=0,1,2,3 .venv-dg/bin/python scripts/kk_defend.py gen --method ads --strength 0.1 \
        --puzzles data/kk/calib.jsonl --grad outputs/kk_defense/proxy_grad.pt --out outputs/kk_defense/calib/ads_0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "third_party/ext/distillation-game/src"))
from kk_build import as_target  # noqa: E402
from kk_pilot import INSTRUCTION, MODELS, parse_labels  # noqa: E402

TEACHER, _, MAX_NEW = MODELS["T"]
PROXY = "Qwen/Qwen2.5-3B"   # the Distillation Game configs' proxy student
EPS = 0.01                  # their configs
MAX_LEN = 8704              # the students' maximum length (§18)


def user_turn(quiz: str) -> list[dict]:
    return [{"role": "user", "content": INSTRUCTION + "\n### Question: " + quiz}]


def dg_cfg():
    from clean_sweep.config import FullConfig, GenerationConfig, ModelConfig
    return FullConfig(model=ModelConfig(teacher=TEACHER, proxy_student=PROXY, student="unsloth/Llama-3.2-3B"),
                      generation=GenerationConfig(eps=EPS))


def causal_lm_loss(model, batch, chunk: int = 1024):
    """The loss Hugging Face computes for a causal LM (shifted labels, cross-entropy averaged over the batch's labelled
    tokens), with the vocabulary projection done in checkpointed chunks: the same value and gradient, without
    materializing batch x length x vocabulary logits (2 x 8,704 x 152k in fp32 did not fit; §23)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask")).last_hidden_state
    h, y = hidden[:, :-1], batch["labels"][:, 1:]
    n_tok = (y != -100).sum()
    head = model.get_output_embeddings()

    def part(hc, yc):
        return F.cross_entropy(head(hc).float().reshape(-1, head.out_features), yc.reshape(-1), ignore_index=-100,
                               reduction="sum")
    total = sum(checkpoint(part, h[:, i:i + chunk], y[:, i:i + chunk], use_reentrant=False)
                for i in range(0, h.shape[1], chunk))
    return total / n_tok


def cmd_grad(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from clean_sweep.train.collator import DataCollatorForCompletionOnlyLM
    from clean_sweep.train.distill import _response_template_for_model, _to_input_ids
    torch.manual_seed(a.seed)
    puzzles = [json.loads(line) for line in open(a.puzzles)]
    tries = {r["id"]: r["tries"] for r in map(json.loads, open(a.teacher_tries))}
    held = [r for r in puzzles if r["n_people"] == 7][:64] + [r for r in puzzles if r["n_people"] == 8][:64]
    tok = AutoTokenizer.from_pretrained(PROXY)
    model = AutoModelForCausalLM.from_pretrained(PROXY, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").to("cuda")
    model.gradient_checkpointing_enable()   # memory only; the gradient is unchanged
    model.config.use_cache = False
    collator = DataCollatorForCompletionOnlyLM(response_template=_response_template_for_model(PROXY, tok),
                                               tokenizer=tok, mlm=False)
    # their compute_student_holdout_grad, with the puzzle prompt (DEVIATION (antidistill): theirs is GSM8K's)
    model.train()
    grads = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n, t0 = 0, time.time()
    for start in range(0, len(held), a.batch):
        batch_rows = held[start:start + a.batch]
        inputs = []
        for r in batch_rows:
            msgs = user_turn(r["quiz"]) + [{"role": "assistant", "content": as_target(tries[r["id"]][0]["text"], "T")}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=False, truncation=True, max_length=MAX_LEN)
            inputs.append({"input_ids": _to_input_ids(ids)})
        batch = {k: v.to("cuda") for k, v in collator(inputs).items()}
        model.zero_grad()
        (causal_lm_loss(model, batch) * len(batch_rows)).backward()
        n += len(batch_rows)
        for name, p in trainable:
            if p.grad is not None:
                grads[name].add_(p.grad)
        model.zero_grad()
    grads = {k: (v / max(n, 1)).cpu() for k, v in grads.items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(grads, a.out)
    norm = sum(float(g.float().pow(2).sum()) for g in grads.values()) ** 0.5
    meta = {"proxy": PROXY, "held_out": len(held), "ids": [r["id"] for r in held], "grad_norm": norm,
            "minutes": round((time.time() - t0) / 60, 1), "max_length": MAX_LEN}
    Path(a.out + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "ids"}))


class OnDevice:
    """Runs a Distillation Game logits processor on the proxies' GPU and returns its scores to the teacher's."""

    def __init__(self, inner, device):
        self.inner, self.device = inner, device

    def __call__(self, input_ids, scores):
        return self.inner(input_ids.to(self.device), scores.to(self.device)).to(scores.device)


def cmd_gen(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    from clean_sweep.generation.core import build_proxy_plus_minus, load_proxy_for_poe
    from clean_sweep.generation.methods import AntidistillationLogitsProcessor, ProductOfExpertsLogitsProcessor
    torch.manual_seed(a.seed)
    rows = [json.loads(line) for line in open(a.puzzles)]
    if a.limit:
        rows = rows[:a.limit]
    if a.length_from:   # batch puzzles of similar length together (speed only): the teacher's step-2 try lengths
        lengths = {r["id"]: r["tries"][0]["tokens"] for f in sorted(Path(a.length_from).glob("T*/tries.jsonl"))
                   for r in map(json.loads, open(f))}
        rows.sort(key=lambda r: -lengths.get(r["id"], MAX_NEW))
    n_gpu = torch.cuda.device_count()
    pdev = torch.device(f"cuda:{n_gpu - 1}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto",
        max_memory={i: f"{a.teacher_gib}GiB" for i in range(n_gpu - 1)})   # the last GPU is the proxies'
    assert all(isinstance(d, int) for d in teacher.hf_device_map.values()), f"offloaded: {teacher.hf_device_map}"
    teacher.eval()
    first = teacher.get_input_embeddings().weight.device
    cfg = dg_cfg()
    wrappers, getter = [], None
    if a.method == "ads":
        grad = torch.load(a.grad)
        plus, minus = build_proxy_plus_minus(cfg, tok, pdev, grad)
        del grad
        wrappers = [plus, minus]
        getter = lambda i, m: (plus(i, m)[:, -1], minus(i, m)[:, -1])  # noqa: E731  (as in their core.py)
    elif a.method == "poe":
        poe = load_proxy_for_poe(cfg, tok, pdev)
        wrappers = [poe]
        getter = lambda i, m: poe(i, m)[:, -1]  # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, steps, gen_tokens = time.time(), 0, 0
    eos = tok.eos_token_id
    with (out / "tries.jsonl").open("w") as f:
        for start in range(0, len(rows), a.batch):
            br = rows[start:start + a.batch]
            enc = tok([tok.apply_chat_template(user_turn(r["quiz"]), tokenize=False, add_generation_prompt=True)
                       for r in br], return_tensors="pt", padding=True, add_special_tokens=False)
            ids, mask = enc["input_ids"].to(first), enc["attention_mask"].to(first)
            for w in wrappers:
                w.reset_cache()
            proc = None
            if a.method == "ads" and a.strength:
                proc = LogitsProcessorList([OnDevice(AntidistillationLogitsProcessor(
                    lam=a.strength, eps=EPS, get_logits_plus_minus=getter, penalty_transform="identity",
                    attention_mask=mask.to(pdev), temperature=a.temperature), pdev)])
            elif a.method == "poe" and a.strength:
                proc = LogitsProcessorList([OnDevice(ProductOfExpertsLogitsProcessor(
                    gamma=a.strength, get_proxy_logits=getter, attention_mask=mask.to(pdev)), pdev)])
            with torch.inference_mode():
                o = teacher.generate(ids, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=a.temperature, top_p=a.top_p, logits_processor=proc,
                                     renormalize_logits=proc is not None, use_cache=True,
                                     pad_token_id=tok.pad_token_id, eos_token_id=eos, repetition_penalty=1.0)
            new = o[:, ids.shape[1]:]
            steps += new.shape[1]
            for r, g in zip(br, new.tolist()):
                stop = eos in g
                g = g[:g.index(eos)] if stop else g
                gen_tokens += len(g)
                text = tok.decode(g, skip_special_tokens=True)
                labels = None if a.timing else parse_labels(text, r["names"], True)
                f.write(json.dumps({"id": r["id"], "n_people": r["n_people"], "tries": [{
                    "text": text, "tokens": len(g), "finish_reason": "stop" if stop else "length", "labels": labels,
                    "correct": None if a.timing else labels == list(r["solution"])}]}) + "\n")
            f.flush()
            print(f"batch {start // a.batch + 1}: {new.shape[1]} steps, {time.time() - t0:.0f} s", flush=True)
    sec = time.time() - t0
    summary = {"method": a.method, "strength": a.strength, "puzzles": len(rows), "batch": a.batch, "gpus": n_gpu,
               "minutes": round(sec / 60, 1), "steps": steps, "sec_per_step": round(sec / max(steps, 1), 3),
               "generated_tokens": gen_tokens, "tokens_per_sec": round(gen_tokens / sec, 1),
               "minutes_per_100": round(sec / 60 / len(rows) * 100, 1),
               "peak_gib": [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(n_gpu)]}
    if not a.timing:
        tries = [json.loads(line)["tries"][0] for line in open(out / "tries.jsonl")]
        by = defaultdict(list)
        for r, t in zip(rows, tries):
            by[r["n_people"]].append(t)
        summary.update({"accuracy": round(sum(t["correct"] for t in tries) / len(tries), 4),
                        "finished": round(sum(t["finish_reason"] == "stop" for t in tries) / len(tries), 4),
                        "by_n": {n: round(sum(t["correct"] for t in v) / len(v), 4) for n, v in sorted(by.items())}})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grad")
    g.add_argument("--puzzles", default=str(REPO / "data/kk/pilot_test.jsonl"))
    g.add_argument("--teacher-tries", default=str(REPO / "outputs/kk_pilot/T/tries.jsonl"))
    g.add_argument("--out", required=True)
    g.add_argument("--batch", type=int, default=2)   # their holdout_grad_batch_size
    g.add_argument("--seed", type=int, default=123)
    s = sub.add_parser("gen")
    s.add_argument("--method", choices=["standard", "ads", "poe"], required=True)
    s.add_argument("--strength", type=float, default=0.0, help="ADS lambda or PoE gamma")
    s.add_argument("--puzzles", required=True)
    s.add_argument("--grad", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--teacher-gib", type=int, default=22, help="teacher weights per GPU, all but the last (GiB)")
    s.add_argument("--temperature", type=float, default=0.6)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--length-from", default=None, help="gen folder with the teacher's step-2 tries, to sort by length")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--timing", action="store_true", help="§23 timing pilot: no labels, no accuracy")
    s.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    (cmd_grad if a.cmd == "grad" else cmd_gen)(a)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
