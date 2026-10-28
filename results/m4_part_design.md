# M4 — PART: implementation plan

**Status: implemented 2026-09-30 (§7a); pilot passed every gate 2026-10-02 (§7b); full 800-problem
dataset generated and passed every gate 2026-10-03 (§7c); word-for-word copies from step 2 removed
the same day (§7c, decision). Trained and evaluated 2026-10-04: **50.5% pooled, as strong as hindsight v2 (49.5%)** — results and analysis in `results/REPORT.md` Part 8.** The second defense in the study, and
the first from the proposal's *structural* family (§3.2). Hindsight (behavioral, §3.3) is the first;
PoE (token-level, §3.1) is being implemented separately by a collaborator.

---

## 0. Which paper — and a citation error in the proposal

**arXiv:2510.11545v1** (2025-10-13, preprint under review):
*"Information-Preserving Reformulation of Reasoning Traces for Antidistillation"* —
**Jiayu Ding, Lei Cui, Li Dong, Nanning Zheng, Furu Wei** (IAIR, Xi'an Jiaotong University;
Microsoft Research).

**It is the intended paper.** The proposal's §3.2 describes PART as removing self-talk sentences and
reordering the trace so that sub-conclusions appear before their derivations — exactly this paper's
two steps — and the paper expands its acronym as "information-**P**reserving **A**ntidistillation
**R**eformulation of reasoning **T**races", which is the title the proposal gives it.

**The proposal's author attribution is wrong.** It cites "Chen et al., 2025" in three places
(lines 17, 104 and 169). It should be **Ding et al., 2025**. The likely source of the confusion: the
first reference PART cites is [CSW+25], Xinghao **Chen** et al., *"Unveiling the key factors for
distilling chain-of-thought reasoning"* (Findings of ACL 2025). The proposal file itself is left
untouched here; the correction is recorded in this document and in `results/REPORT.md`.

**There is no code.** The paper says "We will release the code and data". As of 2026-09-30 nothing
has been released: arXiv has only v1, the Hugging Face Papers page lists no repository, and GitHub
search for the title, the acronym and the arXiv id returns nothing. Everything below is implemented
from the paper's text.

---

## 1. What PART does, according to the paper

PART rewrites the reasoning trace `r` of an output `y = (r, a)` and **keeps the final answer `a`
unchanged** (§2.1). Two steps:

1. **Remove self-talk behaviours (token level, §2.2).** The squared gradient norm of the token loss
   is `Σ p_i² + 1 − 2 p_y`, so tokens the student predicts poorly dominate the update. Self-talk
   tokens ("Hmm", "Wait", "Let's") stay at low probability throughout training (their Fig. 2) — they
   steer the gradient while carrying little information. Rewrite them in declarative form.
2. **Reorder sub-conclusions (structure level, §2.2).** Humans read conclusion-before-process
   without difficulty (lemma before proof). A transformer cannot produce a conclusion before the
   computation that yields it (bounded computation per token), so a student imitating
   conclusion-first traces learns to assert before deriving.

**How:** GPT-4o, prompted with the two prompts of their Appendix B (reproduced verbatim in §3), applied
**segment by segment**: "each original reasoning trace is divided into multiple segments, which are
reformulated individually and then concatenated back into a complete trace" (§2.3). A compact
**Qwen2.5-1.5B-Instruct** reformulator fine-tuned on GPT-4o's (original, rewritten) pairs from
Bespoke-Stratos-17k serves the same job at ~4% overhead and was used for the OpenThoughts-114k run.

**Information preservation (§3):** lexical — RapidFuzz `partial_ratio_alignment`, match ratio of
original sentences/steps found in the rewrite (**91% at threshold 0.7** for GPT-4o, 88% for the 1.5B
model); semantic — Qwen3-Embedding-4B, mean cosine 0.950; a 31-participant human study.

**Their recipe:** LLaMA-Factory SFT, `cutoff_len 16384`, batch 128, lr 3e-5 (1e-4 for 1.5B),
3 epochs, cosine, warmup 0.1. Eval at T=0.6, top-p 0.95, 32,768 max tokens, avg@4 (avg@16 on AIME24).

**Results closest to our setting** (7B students, Bespoke-Stratos-17k, original → PART):

| student | MATH500 | AIME24 |
| --- | --- | --- |
| Qwen2.5-Math-7B | 88.95 → 80.00 (**−8.95**) | 32.71 → 22.08 (**−10.63**) |
| Qwen2.5-7B-Instruct | 83.05 → 70.85 (**−12.20**) | 21.04 → 12.29 (**−8.75**) |
| Qwen2.5-Math-7B, OpenThoughts-114k (1.5B reformulator) | 90.40 → 78.50 (**−11.90**) | 46.67 → 28.54 (**−18.13**) |

**What the paper does not report:** an ablation separating the two steps, the segment length, the
sampling temperature of the rewriter, and any validation or failure handling for the rewrites.

---

## 2. How PART enters this study

**Source traces: the same 800 LIMO-v2 traces that hindsight v2 rewrote.** LIMO is then the shared
undefended reference for both defenses, and PART is one more cell of the existing design: same 800
problems, same config, only the traces differ. Output dataset: `limo_part`.

| contrast | what it measures |
| --- | --- |
| PART vs **LIMO** (62.8% pooled) | PART's passive defensive effect |
| PART vs **hindsight v2** (49.5%) | structural vs behavioural defense |
| PART vs **S1** (−7.0 pp from 7B `solo`) | LLM-converted self-talk vs deleting nine marker words (different source pool — indicative only) |

**Why PART is a useful second defense for this project specifically.** Hindsight v2 traces average
484 tokens against LIMO's 11,450, so every hindsight contrast is confounded with length and with
truncation at `cutoff_len` (hindsight never truncates; LIMO loses the stop token on 32% of traces).
PART is designed to keep the information, and therefore the length: **it is the first defended
dataset in the study whose traces are as long as the undefended ones**, so LIMO vs PART carries
roughly the same termination tax on both sides. That is measured, not assumed (§6).

---

## 3. The prompts — verbatim from the paper's Appendix B

**Pass 1, removing self-talk behaviours:**

```
Rewrite the given text, which is a part of a complete reasoning process. Convert only the parts expressed in a self-talk style into a declarative format. Avoid using first-person expressions such as 'I', 'me', 'we', or 'let's'. Do not alter any parts that are not self-talk; keep them exactly as in the original text.
Do not add any extra information. Do not include any introductory phrases.
Text:
```

**Pass 2, reordering the sub-conclusions:**

```
You will process the given text in two steps. The given text is a part of a complete reasoning process.
Step 1: Extract and list the most important sub-conclusions in the given reasoning process. Keep the number of sub-conclusions small and focused.
Wrap the sub-conclusions in the tags <SUB> and </SUB> for easy extraction.
Step 2: Move the sentences corresponding to these sub-conclusions to appear *before* their respective reasoning processes. Keep the sub-conclusions unnumbered and naturally integrated into the context. Do not modify any other parts of the original text.
Wrap the entire transformed text in the tags <REWRITTEN> and </REWRITTEN> for easy extraction.
Text:
```

Pass 2 receives pass 1's output for the same segment (the paper's order: remove, then reorder).
The segment follows `Text:` on a new line; the prompt is the user turn, with no system prompt.

---

## 4. Design decisions — where the paper is silent, or where we must differ

| item | paper | ours | why |
| --- | --- | --- | --- |
| **rewriter** | GPT-4o (+ fine-tuned Qwen2.5-1.5B-Instruct) | **Qwen2.5-32B-Instruct**, bf16, vLLM TP=4 on GPUs 4–7 | GPT-4o is closed. The paper's own 1.5B reformulator reproduces the effect (88% lexical match, comparable degradation), so a 32B instruct model is ample. Open and ungated, no quantization. Llama-3.3-70B-Instruct in the shared cache was considered and rejected: its snapshot has weights only, no config or tokenizer, and the repo is gated. Any OpenAI-compatible endpoint is a drop-in via `--base-url/--model` — including GPT-4o, if a key is available |
| **segment size** | not stated | consecutive paragraphs (blank-line separated) packed greedily up to **1,024 Qwen tokens**, short tails merged; **10,746 segments** | LIMO paragraphs are short (median 51 chars, max 3,711), so no paragraph is ever cut. ~1k tokens keeps each request well inside what an instruct model rewrites faithfully, and a sub-conclusion almost always sits in the same segment as its derivation. Splitting is exact: re-joining the segments with their original separators reproduces the trace byte for byte |
| **sampling** | not stated | greedy first attempt; T=0.7 on retries | faithful rewriting is a low-entropy task; retries need diversity |
| **final answer** | kept unchanged | everything from the **last** `**Final Answer**` marker on is copied verbatim (782/800 traces; 86 traces contain the marker more than once). The 18 without one are rewritten whole, and their final `\boxed{}` is checked against the original | the paper's `a`, located in LIMO's format |
| **validation** | none described | per-segment guards, up to 3 retries, then fallback (§5) | ~20k LLM calls will produce some failures; they must not silently corrupt mathematics or drop content |
| **information preservation** | lexical, semantic, human | lexical (the paper's RapidFuzz function) + distinct-number recall + final-answer agreement | embedding model not available locally; a human study is out of scope |
| **training** | lr 3e-5, batch 128, 3 epochs | **our LIMO recipe** (lr 5e-6, global batch 8, 15 epochs, 1,500 steps); config byte-identical to hindsight v2's apart from dataset and `output_dir` | every condition in this study holds the recipe fixed; only the traces differ |
| **student** | Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Qwen2.5-7B-Instruct | Qwen2.5-7B (base) | this study's student |
| **evaluation** | T=0.6, avg@4/16 | greedy pass@1 on the 600-problem hard suite, plus GSM8K fallback-graded | comparability with the other 13 conditions |

---

## 5. Implementation and integration with the codebase

| file | change | purpose |
| --- | --- | --- |
| `src/antidistill/defenses/base.py` | **new** | the `Defense` protocol from `IMPLEMENTATION_PLAN.md` §6: `name`, `apply(traces) -> traces`, `metadata()` |
| `src/antidistill/defenses/part.py` | **new** | PART behind `Defense`; CLI `python -m antidistill.defenses.part` |
| `tests/test_part.py` | **new** | offline checks on the real LIMO data: exact segment round-trip on all 800, answer-section split, tag extraction, guards. Runs with or without pytest |
| `scripts/40_serve_teacher.sh` | env overrides | `SERVED_NAME`, `REASONING_PARSER` (`none` disables it — the `deepseek_r1` parser would file a non-reasoning model's entire output under `reasoning_content`), `MAX_MODEL_LEN`. Defaults unchanged, so existing invocations behave identically |
| `scripts/45_generate_part.sh` | **new** | serve the rewriter on GPUs 4–7, run the generator (`PILOT=1` for §7), stop the server |
| `scripts/36_part_audit.py` | **new** | the pre-training gates (§6) and, after evaluation, the verdict (§9) |
| `configs/train/qwen2.5-7b_part.yaml`, `..._part_step1.yaml` | **new** | byte-identical to `qwen2.5-7b_hindsight_v2.yaml` apart from dataset and `output_dir` |
| `data/dataset_info.json` | +2 entries | `limo_part`, `limo_part_step1` |
| `scripts/22_part_run.sh` | **new** | train on GPUs 4–7 with the pruner, then evaluate (hard suite TP=4; GSM8K TP=1 on GPU 7), as `21_strip_s1_run.sh` |
| `.venv-infer` | + `rapidfuzz` | the paper's similarity function. If unavailable, `part.py` falls back to a `difflib` implementation of the same partial-ratio alignment and records which one it used |

**Inside `part.py`:**

- `split_answer(output) -> (reasoning, answer_section)`; `segment(reasoning, max_tokens) -> [(text, separator)]`.
- Two passes per segment, **concurrently across segments** (thread pool; vLLM batches server-side).
- **Guards, each a retry trigger** (pass 1 compared with the original segment; pass 2 with pass 1):
  output non-empty and not truncated (`finish_reason != "length"`); pass 2 has a `<REWRITTEN>` block;
  length ratio within [0.35, 1.30] (pass 1) or [0.75, 1.35] (pass 2), for inputs of 300+ characters;
  every distinct number of the input present in the output, allowing `max(1, 10%)` missing; at least
  50% (pass 1) or 70% (pass 2) of the input's distinct content words kept — this one applies at any
  length, so a short segment cannot be swapped for unrelated text; and no more than two list lines
  introduced (prose restructured into a numbered list keeps every number and word, yet is not the
  rewrite the prompt asks for). A leading preamble ("Here is the rewritten text:") is stripped and
  counted. **The guards accept an unchanged copy of all 10,746 LIMO segments**, so they cannot reject
  a faithful rewrite merely for resembling the original.
- **No stub segments:** greedy packing would leave short tails such as "Therefore, the answer should
  be 64." on their own, too small to validate. A final segment under a quarter of the target is
  merged into the one before it.
- **Fallback after 3 retries:** pass 1 → the original segment; pass 2 → the pass-1 text. Flagged per
  segment, so no problem is ever dropped (all 800 stay in the dataset) and every fallback is visible.
- **Resume cache:** one JSONL line per finished segment, keyed by (trace index, segment index, hash of
  the segment text). A restart redoes only what is missing.
- **Outputs:** `data/defended/limo_part.json` (the training set), `limo_part_step1.json` (pass 1 only —
  the ablation, for free), `limo_part.meta.json` (per-trace audit), `limo_part.segments.jsonl`
  (every intermediate text, including the `<SUB>` lists).

**Not changed:** `hindsight.py`. Moving it behind `Defense` is the other half of M4 and is left to a
separate change, so a working generator is not edited mid-study. Condition names for evaluation are
`part_ep15` and `part_gsm8k` (`30_collect_results.py` keys on the first directory under `outputs/`).

---

## 6. Validation gates — fixed before any training

Computed by `scripts/36_part_audit.py` over all 800 traces. **A failing gate means fix and regenerate
the affected segments, not train.**

| gate | threshold | basis |
| --- | --- | --- |
| completeness | 800/800 traces, every segment processed | no dropped problems |
| fallbacks | ≤ **2%** of segments (either pass) | the hindsight `exhausted` rule |
| final answer | agrees with LIMO's on every trace where LIMO boxes one | PART keeps `a` unchanged |
| numbers preserved | mean distinct-number recall ≥ **0.95** | the mathematics must survive |
| lexical match at 0.7 | ≥ **0.80** of original sentences found (paper: 0.91 GPT-4o, 0.88 1.5B) | the paper's own preservation measure |
| self-talk removed | **self-talk density** down ≥ **70%** vs LIMO (paper's self-talk keyword frequency: 2.9% → 0.4%, −86%) | step 1 actually happened |

**Self-talk density** = **singular** first person (`I` — case-sensitive and followed by a word or
contraction, so neither the imaginary unit nor "Let \( I \) be the incenter" counts — `me`, `my`,
`myself`, `mine`) plus interjections (`hmm`, `wait`, `oh`, `okay`, `ok`, `alright`, `um`, `uh`) per
1k words, counted outside mathematics. That is the voice the pass-1 prompt targets and what the
paper's Figure 1 deletes. The paper does not publish its keyword list, so this is ours.

*Revised 2026-09-30, before any PART output existed.* The first definition also counted plural first
person (`we`, `us`, `our`, `let's`, `let us`). Exercising the audit on the existing datasets showed
that plural first person is ordinary declarative mathematical prose: **hindsight v2's confident
traces score 22.5 per 1k words on it, against LIMO's 7.3** (the 20 pilot problems). It is reported
separately (the PART prompt does ask the rewriter to avoid "we" and "let's") but is not part of the
gate. On the revised measure, the same 20 problems score 29.4 in LIMO and 0.00 in hindsight v2.
**Kim et al.'s nine epistemic tokens are deliberately not the gate**: three of them (`check`,
`might`, `seems`) also occur in declarative prose, and a declarative rewrite keeps them by design
(the paper's own Appendix E example keeps "Check option E").

**Reported, no threshold:** Kim et al.'s nine-token density (`scripts/31_epistemic_density.py`) and
the S1 stripper's nine-marker density, for comparison with every other condition; share of segments
whose pass-2 sentence order differs from pass 1 (step 2 actually happened — the paper gives no
figure to compare with); mean sub-conclusions per segment; length retained in Qwen tokens;
truncation and no-stop-token share at `cutoff_len 16384` (`scripts/32_trace_budget.py`); held-out
reconsideration density.

---

## 7. Pilot — before the full run

20 traces (10 stratified across length deciles + 10 random, fixed seed; **269 segments**). Measure
every §6 quantity, output tokens/s and the projected full-run time; read five segments through all
three stages by hand. Proceed only if the gates hold. Command: `PILOT=1 scripts/45_generate_part.sh`,
then `scripts/36_part_audit.py --pilot`.

### 7a. Already verified without the rewriter (2026-09-30)

- **Offline tests** (`tests/test_part.py`, 13/13): exact segment round-trip and answer split on all
  800 traces; output parsing; every guard; the full pipeline with a scripted fake rewriter, including
  resume from the cache (a second run makes zero calls) and fallback accounting.
- **Full segmentation:** 800 traces → **10,746 segments** (median 1,001 tokens, max 1,268).
- **Against a real vLLM server**, using a 1B instruct model as a stand-in rewriter on two traces:
  serving with the reasoning parser disabled, the chat path, finish reasons and throughput logging
  all work. The 1B model is far too weak for the job — **8 of 13 segments fell back**, rejected for
  summarising, truncation, missing tags or lost numbers — which is the guards doing their job. It
  also exposed the list-restructuring failure, now a guard (§5).
- **The audit** runs end to end, and correctly **fails** the self-talk gate for the fake rewriter,
  which only deletes "Hmm, " and "Wait, " (−41%).

### 7b. Pilot result (2026-10-02) — ALL GATES PASS

Qwen2.5-32B-Instruct, bf16, TP=4 on GPUs 4–7, 128 concurrent requests. 20 traces, 269 segments,
**12.2 min**.

| gate | pilot | threshold |
| --- | --- | --- |
| fallbacks | 1/269 = **0.37%** | ≤ 2% |
| final answer agrees with LIMO | **20/20** | all |
| number recall | **0.994** | ≥ 0.95 |
| lexical match @0.7 | **0.880** sentence, 0.899 step (paper: 0.91 GPT-4o, 0.88 1.5B) | ≥ 0.80 |
| self-talk density | 29.35 → **1.20** /1kw (**−96%**; paper −86%) | ≥ 70% down |

| same 20 problems | tokens/trace | self-talk | we/us/let's | Kim et al. nine | truncated @16384 | no-stop tokens |
| --- | --- | --- | --- | --- | --- | --- |
| LIMO | 12,962 | 29.35 | 7.28 | 37.78 | 30.0% | 41.1% |
| **PART** | **12,126 (93.5%)** | 1.20 | 0.93 | 14.73 | **20.0%** | **28.8%** |
| PART step 1 only | 11,730 | 1.24 | 0.95 | 15.52 | 15.0% | 22.0% |
| hindsight v2 | 478 | 0.00 | 22.46 | 0.27 | 0.0% | 0.0% |

**Read by hand, it does what the paper describes.** Pass 1 turns "Hmm, but I feel like maybe I made a
mistake in the problem constraints. Wait the problem says…" into "The problem states…": the doubt
goes, the check that followed it stays, now declarative. Pass 2 moves "Therefore, all cases are
valid, so the total is22." from mid-segment to the top. (Missing spaces such as "is22" are LIMO's
own, not the rewriter's — `results/deviations.md`, upstream data defects.)

**Three things to carry forward:**

1. **Kim et al.'s nine tokens fall only 61%** (37.8 → 14.7) while self-talk falls 96%. Expected
   (§6): `check`, `might`, `seems`, `alternatively` survive a declarative rewrite. PART removes the
   *voice* of doubt far more completely than the *vocabulary* — the opposite of S1.
2. **The termination tax shifts in PART's favour.** PART traces are 6.5% shorter, so traces near
   the cutoff drop below it: truncated 30% → 20%, no-stop token share 41% → 29% on these 20.
   Fewer non-terminating targets would make the PART student *better* at finishing, biasing the
   result towards "PART does not defend". Measured precisely on the 800 before training;
   `accuracy | finished` separates the two at evaluation.
3. **Reordering is real but light**: 46% of segments had at least one sentence moved verbatim
   (1.17 per segment). Only 1 of 269 came back character-identical, but how many of the other
   changes are reworded hoists rather than edits cannot be measured from the pilot's records
   (next point). The model marks sub-conclusions *inside* the rewritten text (264/269
   segments) rather than listing them first; the parser stripped those tags correctly but did not
   count them, so the pilot's sub-conclusion figures (0.11/segment) are undercounts. Fixed for the
   full run — metadata only; the training text is unaffected.

**Throughput:** ~670 output tokens/s, ~22 segments/min, communication-bound (128 requests running,
KV cache 53% used, ~185 W per GPU). **Full run at this setting: ~8 h for 10,746 segments.**


### 7c. Full dataset (2026-10-03) — ALL GATES PASS

Generated 2026-10-02 20:32 → 2026-10-03 01:48 on GPUs 2–7 (TP=4 + TP=2 replicas), resumed once after the
harness killed the first launch at 840 segments; the cache made the restart lossless.

| gate | 800 problems | threshold |
| --- | --- | --- |
| fallbacks | 109/10,746 = **1.01%** | ≤ 2% |
| final answer agrees with LIMO | **800/800** | all |
| number recall | **0.993** | ≥ 0.95 |
| lexical match @0.7 | **0.880** sentence, 0.899 step | ≥ 0.80 |
| self-talk density | 30.39 → **1.73** /1kw (**−94%**) | ≥ 70% down |

| all 800 problems | tokens/trace | self-talk | we/us/let's | Kim et al. nine | truncated @16384 | no-stop tokens |
| --- | --- | --- | --- | --- | --- | --- |
| LIMO | 12,996 | 30.39 | 7.30 | 39.01 | 32.0% | 45.4% |
| **PART** | **12,162 (93.6%)** | 1.73 | 0.85 | 15.64 | **27.1%** | **40.3%** |
| PART step 1 only | 11,810 | 1.84 | 0.88 | 16.45 | 24.0% | 36.3% |
| hindsight v2 | 485 | 0.17 | 20.76 | 0.57 | 0.0% | 0.0% |

**The truncation shift is smaller than the pilot suggested** (32.0% → 27.1% truncated, against 30% → 20%
on the 20 pilot problems), so the termination advantage PART gets over LIMO is modest.

**Open issue — step 2 copies more than it moves.** The paper's prompt says to *move* each sub-conclusion
ahead of its derivation. Of 19,159 sub-conclusions in the 10,620 segments that have them:

| what step 2 did | share |
| --- | --- |
| moved, as the paper intends | 9.7% |
| copied to the front, original sentence kept | 24.1% |
| new summary sentence at the front, original kept | 32.9% |
| new summary at the front, original removed (a reworded move) | 33.3% |

**64% of segments state a result twice** — once before its derivation, once after. That weakens the
structural half of PART relative to the paper's description (whether GPT-4o behaves differently is
unknown; the paper reports no such statistic). Options, undecided: (a) train on it as is and report the
deviation; (b) delete the later copy where it is verbatim (24% of cases; the reworded 33% are harder);
(c) rerun step 2 with a stricter instruction (a further deviation from the paper's prompt). Step 1 output
(`limo_part_step1.json`) is unaffected by this choice.

**Decision (2026-10-03): option (b), word-for-word copies only.** Implemented as
`dedup_verbatim_copies()` in `part.py` (`--dedup-verbatim`), applied when the dataset is assembled
from the cached rewrites, so no model calls are involved:

- a sub-conclusion counts as copied only if pass 2 holds it more often than pass 1 did (LIMO's own
  repeated sentences stay); the first occurrence, the hoisted one, is kept;
- a later occurrence is removed only if it is a whole sentence, including a short lead-in the hoisted
  copy dropped ("So x can't be an integer." when the copy is "x can't be an integer.") — at most four
  plain words, no digits or math;
- reworded summaries (33%) are not touched.

Result: **4,074 copies removed** in 740 of 800 problems (1.1% of the text); **733 left**, almost all
pieces of multi-line formulas ending in a comma, where deletion could change the meaning. Every gate
still passes (fallbacks 1.01%, answers 800/800, numbers 0.993, lexical 0.880, self-talk −94%). Length
is now 92.6% of LIMO's; 25.6% of traces exceed the 16k cutoff (LIMO 32.0%).

The pre-dedup dataset is kept as `data/defended/limo_part_nodedup.json` (+ `.meta.json`), so both
versions can be compared or trained.

---

## 8. Full run — GPUs 4–7 only (0–1 reserved; 2–3 left free)

| step | estimate |
| --- | --- |
| rewriter download (one time, ~65 GB, to `/home/umid/AntiDistillation/models/`, root disk — not the 96%-full vault) | minutes to an hour |
| generation: 10,746 segments × 2 passes, ≈ 20 M output tokens | **set by the pilot** (rough: 2–4 h) |
| training (`scripts/22_part_run.sh`), LIMO-sized token budget on 4 GPUs | ~8 h |
| evaluation: hard suite TP=4, then GSM8K TP=1 | ~3 h + ~7 h |

Training and evaluation run serially, never concurrently (CLAUDE.md §7.3).

---

## 9. Pre-registered decision rule

Pooled 600-problem hard suite, greedy, boxed-only — the basis of every verdict in this study:

| PART result | verdict |
| --- | --- |
| within **2.0 pp** of LIMO (**62.8%**) | PART **does not defend** under this recipe |
| within **2.0 pp** of hindsight v2 (**49.5%**) | PART defends **as strongly as the behavioural defense**, while keeping traces long and readable |
| otherwise | **partial** — report the share of LIMO's lead over hindsight v2 that PART removes, `(62.8 − PART) / (62.8 − 49.5)` |

2.0 pp is 12 problems of 600, the band used for S1. `accuracy | finished` and GSM8K are reported
alongside. For orientation only, not a gate: the paper's 7B deltas (−9 to −12 pp on MATH500) would
put PART at roughly 57–60% on MATH500 against LIMO's 69.0%.

---

## 10. Optional ablation — step 1 alone

`limo_part_step1` (self-talk removed, nothing reordered) is produced at no extra generation cost. One
more training run separates the token-level from the structural contribution, which the paper never
does. Worth running only if PART shows an effect worth attributing.

**Run 2026-10-04/05: 46.0% pooled** (PART 50.5%). Step 1 alone carries the whole effect; reordering adds
nothing to it. Details and the stopping problem: `results/REPORT.md` Part 8.

---

## 11. Risks to the reading

- **Rewriter fidelity.** A 32B open model instead of GPT-4o may paraphrase more than instructed
  (caught by the lexical gate) or alter mathematics (caught by the number gate).
- **Silent non-reordering.** The model may return pass 2's input unchanged; the reorder share
  measures this directly.
- **Length change and truncation.** If PART shortens traces, fewer lose their stop token and the
  termination tax shifts in PART's favour; `32_trace_budget.py` quantifies it before training.
- **The held-out reconsideration lexicon is not independent here.** It is mostly first-person
  ("let me double-check", "I made a mistake"), and PART removes first person by instruction, so its
  drop under PART is by construction. It was an independent instrument for S1; for PART it is not.

---

## 12. Out of scope, for later

Adaptive attacks against PART (A1 *style* and A2 *search* fed PART traces — `solo` is
defense-independent and already measured); the compact 1.5B reformulator; embedding similarity;
refactoring hindsight behind `Defense`.

---

## 13. Notes for the paper's appendix

Points a reader needs to judge our PART numbers; to go into the paper's appendix as written here.

1. **Attribution.** PART is Ding, Cui, Dong, Zheng, Wei (2025), arXiv:2510.11545. The project
   proposal cited it as "Chen et al."; that was an error.
2. **No released code or data.** Implemented from the paper's text: its two prompts verbatim
   (Appendix B), applied to ~1,024-token pieces of each LIMO solution, greedy decoding, final answer
   untouched. Segment length, sampling, and validation are our choices (§4–§5).
3. **Rewriter.** Qwen2.5-32B-Instruct in place of the paper's GPT-4o.
4. **Validation.** Automatic checks with up to 3 retries per pass; 1.01% of pieces kept their
   original text after all retries failed. Quality matched the paper's own measure (88% of sentences
   found at threshold 0.7; paper: 88–91%).
5. **Step 2 copies more than it moves.** The paper's prompt says to move each sub-conclusion ahead of
   its derivation. Our rewriter moved 9.7% as asked; 24.1% were copied word for word to the front
   with the original kept; 32.9% were added as a reworded summary with the original kept; 33.3% were
   added reworded with the original removed. The paper reports no such statistic, so we cannot say
   whether GPT-4o behaves differently; its single full example (Appendix D, from its 1.5B rewriter)
   shows a clean move plus one reworded restatement.
6. **Post-processing.** We removed the word-for-word copies (4,074 removed; 733 left because they
   are pieces of longer formulas), keeping the hoisted copy. Reworded summaries were left as
   written. Both the processed and the unprocessed datasets are kept.
7. **Training recipe.** Ours (LIMO's: lr 5e-6, batch 8, 15 epochs, 1,500 steps), not the paper's
   (lr 3e-5, batch 128, 3 epochs), so PART is comparable with every other condition in this study.

