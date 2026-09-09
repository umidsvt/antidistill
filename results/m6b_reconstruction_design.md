# M6b — Epistemic reconstruction attack (design)

**Status: designed 2026-09-08, not yet run.** Blocked only on GPUs (hindsight_v2 training).

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
is *fabricated*, since the rewriter never searched.

**A2 — guided re-derivation.** Give the model the problem *and* the defended solution as a hint,
then let it reason naturally with **no confidence instruction**, and keep its genuine
`reasoning_content` plus its answer. The doubt here is *real* — it is the model's own search —
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

## 6. Cost and confounds

Generation ~1-2 h per condition on 4 GPUs (A2 is cheaper than the hindsight run: no validation
retries needed if the defended answer is taken as ground truth). Training ~6 h per condition,
evaluation ~2 h.

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
