# M6b — the reconstruction prompts, and how generation works

Companion to `results/m6b_reconstruction_design.md` (why these three conditions exist).
Source of truth: `src/antidistill/attacks/epistemic_reconstruction.py`.

---

## 1. The problem the prompts have to solve

The defender serves confident, doubt-free traces. **Nothing epistemic survives inside them to be
parsed back out.** Measured over the 800-trace pools:

| pool | epistemic tokens / 1k words |
| --- | --- |
| LIMO — a real search trace | **35.571** |
| teacher's *private* scratchpad, v1 | 0.732 |
| teacher's answer, v1 | 0.033 |
| v2 defended trace, what the attacker receives | 0.024 |

The scratchpad matters most here. Kim et al.'s rewrite prompt contains:

> *"Do not express any uncertainty — never say 'I think,' 'probably,' or 'it seems.' State
> everything with full confidence."*

That instruction suppresses doubt **during generation**, so the model never produces it, even in
reasoning the user was never meant to see. The doubt is not hidden; it does not exist. So the
attack cannot be an extraction — it has to re-generate.

---

## 2. Prompt A1 — `style`: fabricate the doubt

```text
You are given a QUESTION and a polished SOLUTION.

QUESTION:
{question}

SOLUTION:
{solution}

Rewrite the solution as a first-person account of working the problem out for the first time.
Show the reasoning as it would actually unfold: consider an approach before committing to it,
note where a step could go wrong, double back when something does not check out, and verify
results that are easy to get wrong. Keep every mathematical step and the same final answer.
Put your final answer within \boxed{}.
```

**What it does.** Hands the model the finished answer and asks it to *perform* the search that
would have produced it.

**Why it is built this way.** It is the deliberate inverse of the defense: where the defense says
"state everything with full confidence", this says "show where it could go wrong". Same trace,
opposite instruction.

**The catch, which is the point.** The rewriter already knows the answer. It never hit a dead end,
so any "double back" it writes is invented — reconstructed doubt, not experienced doubt. If a
student trained on this matches one trained on real LIMO traces, then epistemic verbalization is a
*linguistic habit* and placement does not matter (H1). If it does not, placement carries real
information (H2).

**Output taken:** `.content` only. The model's own scratchpad is discarded here, because the
fabricated doubt we want to study is in the visible rewrite.

---

## 3. Prompt A2 — `search`: harvest doubt the model actually had

```text
Solve this problem.

QUESTION:
{question}

A reference solution is available if you need it; you do not have to follow its route:
{solution}

Work the problem out yourself, step by step, showing all key equations and intermediate algebra.
Put your final answer within \boxed{}.
```

**Three deliberate choices:**

1. **No confidence instruction.** This is the single word-level difference from the defense. The
   model reasons in its natural voice, so the doubt is genuine.
2. **The defended solution is offered, not imposed** — *"available if you need it; you do not have
   to follow its route"*. Phrased as a script, the model would transcribe it and skip the search,
   which is exactly what we need it not to do. Phrased as a hint, it explores and is steered
   toward the right answer.
3. **"Solve this problem"**, not "rewrite this solution". The task is framed as first-contact
   problem solving, which is the state that produces real doubt.

**Output taken:** `reasoning_content` **joined to** `.content`. The scratchpad *is* the attack —
it is the model's genuine search, complete with the branches it abandoned.

> This is precisely what our own v2 defended pipeline **throws away**. With
> `--reasoning-parser deepseek_r1` the server splits the scratchpad into `reasoning_content` and
> the answer into `content`; the defense keeps `content`. The attack keeps both. Same server, same
> model, opposite halves.

---

## 4. Prompt A3 — `solo`: the control

```text
Solve this problem, step by step, showing all key equations and intermediate
algebra.

QUESTION:
{question}

Put your final answer within \boxed{}.
```

**No defended data at all.** Identical to A2 with the reference solution removed, so the
difference between them is exactly "did the defended trace help".

**Why this is not optional.** An attacker who owns a model able to synthesise epistemic traces
invites an obvious question: why not distil from that model directly and ignore the defense?

- **A2 > A3** → the defended traces contribute something the attacker's model cannot supply (the
  correct approach and answer). The attack genuinely combines two sources, and the defense is
  *broken*.
- **A2 ≈ A3** → the attacker never needed the defended data. The defense is *irrelevant*, not
  broken — a different and much weaker claim.

Without A3 the other two numbers cannot be interpreted.

---

## 5. How generation actually runs

**Server.** `scripts/40_serve_teacher.sh` starts vLLM with `--reasoning-parser deepseek_r1`, which
splits each reply into `reasoning_content` (scratchpad) and `content` (answer).

**Call.** Chat endpoint, one user message, no system prompt, `temperature 0.6` (DeepSeek's
recommended 0.5–0.7 for R1-Distill; below that it is prone to endless repetition — 0.4 is what
produced 40 non-terminating traces in the v1 defended run). Requests are issued concurrently
through a thread pool because the chat endpoint takes one conversation per call; serialising costs
roughly 10x wall-clock.

**Assembling the training target** (`build_output`):

| mode | target |
| --- | --- |
| `style` | `content` |
| `search` | `reasoning_content` + `"\n\n"` + `content` |
| `solo` | `reasoning_content` + `"\n\n"` + `content` |

The two halves are joined with **plain whitespace, never a `</think>` delimiter**. That tag is the
artifact 499/500 v1-trained students learned to reproduce (`results/hindsight_versions.md`);
reintroducing it would rebuild the exact defect we spent the week removing.

Empty `content` means the model never closed its reasoning block, so no answer exists — recorded
as `empty` rather than shipped.

**Audit.** Every trace is scored with `answers_agree()` against the **undefended LIMO trace**, not
against the defended one it was shown. Otherwise `search` could score "correct" merely for copying
the hint it was handed.

---

## 6. What gets measured

Accuracy alone cannot separate H1 from H2, so the fidelity measures matter as much as pass@1:

- **Epistemic density**, against LIMO's 35.571 and the defended pool's 0.024.
- **Doubt placement.** Real LIMO doubt is spread evenly — **34% / 34% / 32%** across the thirds of
  a trace (192,783 occurrences, median position 0.48). Fabricated doubt is expected to clump at
  openings or in a closing "let me verify". A `style` pool that matches LIMO's profile is evidence
  for H1; one that clumps is evidence for H2.
- **Token budget and no-stop-token share** (`scripts/32_trace_budget.py`) for every pool *before*
  attributing any accuracy difference to epistemic content — M6a showed trace length drives
  termination through `cutoff_len` truncation, and reconstructed traces will be far longer than
  the 486-token defended ones.

---

## 7. Running it

```bash
GPUS=4,5,6,7 PORT=8011 scripts/40_serve_teacher.sh      # needs --reasoning-parser deepseek_r1

for m in style search solo; do
  .venv-infer/bin/python src/antidistill/attacks/epistemic_reconstruction.py \
      --mode $m \
      --defended data/defended/limo_hindsight_chat.json \
      --output data/curated/limo_recon_$m.json \
      --base-url http://127.0.0.1:8011/v1
done
```

Add `--limit 20` for a pilot. Each full pass is ~1–2 h on 4 GPUs; there is no validation retry
loop, so it is cheaper than the defended generation run.
