# M6b — Epistemic reconstruction attack (design)

**Status: generation complete 2026-09-10; training pending.** Design below was fixed before any
result was seen. What actually happened during generation — including two defects the pilot
caught and how the trace pools were repaired to a full 800 — is in section 7.

The three prompts, verbatim, with the reasoning behind each wording choice and the mechanics
of generation: **`results/m6b_prompts.md`**.

The `Score()` function of proposal section 4.1 flags a trace pool as epistemic-stripped. What
should the attacker do about it? Section 4.3's answer was *supplementation* — mix in real
epistemic traces from elsewhere (M6a). This is the harder case: **the attacker has no external
epistemic traces and must reconstruct them from the defended data itself.**

---

## 1. The question this actually answers

Kim et al. claim epistemic verbalization is what makes a trace teachable. This attack forces the
claim to be precise, because two readings predict opposite outcomes:

- **H1 — verbalization is BEHAVIOUR.** The student learns a linguistic habit: emit "wait", pause,
  reconsider. Any plausible-sounding doubt installs the habit. **Reconstruction works.**
- **H2 — verbalization is INFORMATION.** The doubt marks *which specific steps* are error-prone,
  discovered by actually searching. Doubt invented after the fact sits in the wrong places and
  teaches the student to hesitate where nothing is hard. **Reconstruction fails.**

M2 gives H1 some support already: 800 examples took `wait` from 0 to 6,013 occurrences, so the
habit installs readily. Whether the *placement* carries the value is untested, and this is the
experiment that separates them. That is a stronger contribution than one more attack row.

---

## 2. What the defense actually destroys

Measured over the 800-trace pools (`scripts/31_epistemic_density.py` conventions):

| pool | epistemic tokens / 1k words |
| --- | --- |
| LIMO — a real search trace | **35.571** |
| v1 teacher's private scratchpad | 0.732 |
| v1 teacher's answer | 0.033 |
| v2 defended trace (served to the attacker) | 0.024 |

**The scratchpad is 49x below LIMO.** The rewrite prompt's instruction — *"Do not express any
uncertainty ... State everything with full confidence"* — suppresses doubt during generation, so
it is never produced, not merely hidden. Nothing epistemic survives inside the defended artifact
to be recovered by parsing. It has to be re-generated.

Second measurement, needed for section 5: in real LIMO traces doubt is spread evenly —
**34% / 34% / 32%** across the first, middle and last third (192,783 occurrences, median position
0.48). Real doubt is woven through the search, not appended as a final check.

---

## 3. Three attacks and the control that makes them interpretable

All produce 800 traces on the same problems, trained with identical hyperparameters.

**A1 — style re-injection.** Ask an LLM to rewrite the defended trace as a first-person
exploration with false starts and self-correction. Cheapest, and the purest test of H1: the doubt
is *fabricated*, since the rewriter never searched. **Implementation changed after the pilot** —
it runs with the model's scratchpad suppressed, because harvesting the visible answer alone
produced no doubt at all; see §7.1.

**A2 — guided re-derivation.** Give the model the problem *and* the defended solution as a hint,
then let it reason naturally with **no confidence instruction**, and keep its genuine
`reasoning_content` plus its answer. **Changed after the first pass** — when the model never closes
its reasoning block the `reasoning` is now salvaged as the trace rather than discarded; see §7.2. The doubt here is *real* — it is the model's own search —
but steered toward the defended answer. Note this is exactly what our own v2 pipeline discards:
`--reasoning-parser deepseek_r1` splits the scratchpad off and we keep only `.content`.

**A3 — attacker-only baseline. THE CONTROL.** The attacker's model solves the problems from
scratch with no defended data at all.

> **Why A3 decides whether any of this is publishable.** If the attacker owns a model good enough
> to synthesise epistemic traces, the obvious question is why they do not simply distil from that
> model and ignore the defended data. A2 beating A3 is what shows the defended traces still
> contribute something the attacker's own model cannot supply (the correct approach and answer),
> i.e. that the attack genuinely combines two sources. **A2 ~= A3 would mean the defense is
> irrelevant rather than broken** — the attacker never needed it. Without A3 the result is
> uninterpretable, and this is the comparison a reviewer will reach for first.

Conditions to train: `recon_style` (A1), `recon_search` (A2), `attacker_only` (A3), against the
existing `base`, `limo`, `hindsight_v2`.

---

## 4. Predictions, recorded before running

| comparison | H1 predicts | H2 predicts |
| --- | --- | --- |
| A1 vs hindsight | large gain | little or none |
| A2 vs A1 | similar | A2 clearly better (real search vs invented doubt) |
| A2 vs LIMO |near parity | A2 short of LIMO |
| A2 vs A3 | — | A2 > A3 if defended answers add value |

Writing these down first; the informative outcome is whichever way it falls, and A1-vs-A2 is the
cleanest single discriminator because both carry doubt and only one carries *searched* doubt.

---

## 5. Reconstruction fidelity — beyond pass@1

Accuracy alone will not distinguish H1 from H2. Two placement measures, both cheap:

**Positional profile.** Compare the distribution of doubt positions against LIMO's 34/34/32
baseline. Fabricated doubt is expected to clump (openers, or a closing "let me verify").

**Placement-vs-difficulty correlation.** Sample the *student* n times per problem and locate the
step where its solutions diverge from one another — an empirical marker of where the problem is
genuinely hard. Then measure whether a trace's doubt coincides with those steps. Real traces
should correlate; fabricated ones should not. This operationalises H2 directly and, as far as we
can tell, is not something Kim et al. measure.

---

## 7. What generation actually did (2026-09-10)

### 7.1 The pilot caught A1 harvesting the wrong half

A 20-problem pilot before the full run showed `style` producing **0.193** epistemic tokens/1k
words with doubt in **1/20** traces — indistinguishable from the defended pool's 0.024. Diagnosing
one call explained it:

| | chars | epistemic /1k words |
| --- | --- | --- |
| `reasoning_content` | 13,749 | **44.89** |
| `.content` | 824 | **0.00** |

The model *obeys* "show where it could go wrong" — but R1 always emits messy thinking then a clean
answer, so the doubt lands in the scratchpad and `.content` returns confident regardless of the
prompt. **That is structurally the same behaviour the defense exploits.**

Fix: give A1 **no scratchpad**. A manual chat template with a pre-closed `<think></think>` block,
sent through the completions endpoint (the chat template applies `content.split('</think>')[-1]`
to assistant messages and would delete the prefill). The instructed doubt then has nowhere to go
but the visible output: **0.00 -> 26.48** on the diagnostic problem, 13.9 across the full pool.

Had the pilot been skipped, A1 would have been a null result caused by a harvesting bug rather
than by anything about epistemic verbalization.

### 7.2 Discarding unterminated reasoning cost 50 problems

`build_output` originally returned `None` whenever `.content` was empty — throwing away the
`reasoning_content` with it. But an unterminated trace still contains 48k-134k characters of
genuine search, which **is the thing this attack harvests**. Salvaging it recovers the problem.

### 7.3 Degenerate loops, and why a guard was needed

Salvaging blindly would ship repetition loops. Calibrated on real data:

| pool | min | p1 | median |
| --- | --- | --- | --- |
| LIMO (real traces) | **0.879** | 0.959 | 0.995 |
| style | 0.154 | 0.327 | 0.995 |
| search | 0.650 | 0.838 | 0.991 |
| solo | 0.281 | 0.725 | 0.980 |

Score = fraction of distinct 12-word windows. **Real LIMO traces never fall below 0.879**; our
worst generated trace scores **0.154** and is one sentence repeated for 71,111 characters. Such
traces also *inflate* epistemic density (52.7 and 60.6 per 1k words, because "Wait...Wait...") — a
reason not to trust that metric without the repetition check. There were few enough that pool
means barely moved (style 13.911 -> 13.525 excluding them).

`MIN_REPETITION_RATIO = 0.60` sits above every true loop (<0.2) and below LIMO's floor. **This is
a filter of our own invention**, not inherited from LIMO or Kim et al.; recorded in
`results/deviations.md`.

### 7.4 Loop-onset truncation

A degenerate trace is usually genuine for a while and then sticks, so the clean prefix is
recoverable. Two implementations were needed:

- **Global binary search (wrong).** A prefix can absorb several loop iterations and still average
  above threshold. On trace 258 it returned a "clean" prefix that still ended mid-loop.
- **Local sliding window (used).** Cut at the first window that degenerates. Trace 258:
  12,948w @ 0.154 -> 1,798w @ 0.965.
- **Plus a global re-check.** Local windows cannot see *long-range* repetition: trace 71 repeated
  a passage every ~1000 words, passing every local window while scoring 0.291 overall. The cut
  prefix is therefore validated globally and refused if it still fails.

### 7.5 Final trace accounting

| | style | search | solo |
| --- | --- | --- | --- |
| first pass | 800 | 783 | 765 |
| after salvaging unterminated reasoning | 800 | **800** | 786 |
| degenerate (<0.70) found | 18 | 1 | 5 |
| ... recovered by loop-onset truncation | 12 | 1 | 4 |
| ... unsalvageable, regenerated | 6 | 0 | 2 |

An earlier plan to drop the 50 problems all modes could not cover was **rejected**: it would have
biased the set (the dropped problems were ~11% harder by LIMO trace length) and broken the
800-problem coverage every other condition has.

---

## 6. Cost and confounds

Generation ~1-2 h per condition on 4 GPUs (A2 is cheaper than the hindsight run: no validation
retries needed if the defended answer is taken as ground truth). Training ~6 h per condition,
evaluation ~2 h.

**Problem coverage — all three pools are complete at 800/800 (2026-09-10).**

The teacher fails to produce a usable trace on some problems: it never closes its `</think>`
block within the token budget (repetition loops, R1's documented failure mode), so no answer
exists. Empty targets cannot be shipped — they teach the student to emit nothing.

An earlier version of this plan dropped those problems, leaving the 750 all three modes could
cover. **That was rejected**, for two reasons:

- **It biased the set.** The 50 uncovered problems are **~11% harder** (median LIMO trace 13,272
  tokens vs 11,934 for the rest) — expected, since harder problems make the teacher reason longer
  and longer reasoning is what loops. Evaluating on the remainder would have made the attack look
  better than it is.
- **It broke coverage parity** with the 800-problem base / LIMO / hindsight conditions, which is
  the property every comparison in this project rests on.

They were recovered instead, in three stages (see 7.2-7.5): salvaging unterminated reasoning,
cutting degenerate traces at the loop onset, and regenerating what neither fixed.

| | style | search | solo |
| --- | --- | --- | --- |
| first pass | 800 | 783 | 765 |
| + salvaged unterminated reasoning | 800 | **800** | 786 |
| + loop-onset truncation | 800 | 800 | 790 |
| + regeneration | **800** | **800** | **800** |

The failures were themselves a result about the control: **an unaided attacker does not merely get
wrong answers on the hardest problems, it sometimes gets no answer at all.** Of the `solo` traces
recovered from that state, essentially none reach LIMO's answer — the problems where it loops are
the problems it cannot solve.

**Confounds to control up front:**

- **Trace length.** M6a showed length drives termination via `cutoff_len` truncation, and
  reconstructed traces will be far longer than the 486-token defended ones. Report the token
  budget and no-stop-token share for every pool (`scripts/32_trace_budget.py`) *before* attributing
  any difference to epistemic content.
- **Teacher identity.** Using DeepSeek-R1-32B as both defender's teacher and attacker's model is
  circular. It is the right *first* experiment because it holds capability fixed and isolates the
  prompt, but the realistic version uses a weaker attacker model, and the write-up must say which
  was run.
- **The defended answer as hint.** A2 leaks the answer into the prompt, so its traces are
  correct-by-construction. That flatters A2 against A3 on accuracy; the `accuracy | finished`
  split and A3 are what keep it honest.
