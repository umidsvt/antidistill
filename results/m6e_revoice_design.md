# M6e — Re-voice attack on PART: design

**Status: complete 2026-10-07. Verdict: re-voicing does not recover what PART removed (52.5% vs PART 50.5%).** Results and interpretation: `results/REPORT.md` Part 9. Results will be added to `results/REPORT.md`.

---

## 1. Why this attack

PART (Ding et al. 2025) defends as strongly as hindsight in our setup: 50.5% on the 600 problems,
against LIMO's 62.8% and hindsight v2's 49.5% (`results/REPORT.md` Part 8). The step-1-only ablation
showed **where that effect comes from: step 1 alone** (46.0%). Step 1 rewrites first-person self-talk
as plain statements; reordering (step 2) adds nothing.

Step 1 does not delete the reasoning. The checks, corrections and alternative methods are still there;
only their wording changes. On problem 450:

> LIMO: *"Wait, hold on, no. If a person leaves, we're reducing the number of people in the room as
> well, right? Or does the problem say that someone leaves and someone else enters? Let me check."*
>
> PART: *"The total number of people remains the same, which is six, because one left and one entered."*

So an attacker can try to **reverse step 1**: rewrite PART's statements back into thinking aloud.

## 2. Threat model

The attacker has the **800 PART-defended solutions** (with their final answers) and an **ordinary
instruction-tuned model with no reasoning mode**. It never sees the original LIMO text.

This is the setting the meeting's question 1 asks about: an attacker *without* an open reasoning model.
An attacker with one would just run Solo (69.5%) and ignore the defense.

The model is **Qwen2.5-32B-Instruct**, the same model we used as PART's rewriter. That is a convenience,
not a requirement: any comparable instruction model would do. It is noted as a caveat (§8).

## 3. The prompt

Written to mirror PART's prompt 1 in reverse:

```
Rewrite the given text, which is a part of a complete reasoning process. Convert the parts that state
a check, a correction, a doubt, a comparison of options or a change of approach into first-person
thinking aloud, the way a person working the problem out for the first time would say it, for example
"Wait, let me check that...", "Hmm, maybe...", "Actually, ...", "Let me try another way." Keep every
calculation, number and result exactly as in the original text, and keep the parts that are plain
calculation as they are.
Do not add new steps or new information. Do not include any introductory phrases.
Text:
{one piece of the PART solution}
```

### Revision after the pilot (2026-10-05, before any training)

The first prompt (above) failed the self-talk check of §5 on the 20-problem pilot: self-talk rose from
1.6 to only **8.8 per 1k words**, 28% of LIMO's 31.8, against the 50% required. Every other check passed
(no failed pieces, all answers kept, numbers 1.00). It asked to convert only the checks and corrections,
but PART's step 1 removed first person from the **whole** text ("Let me compute 25 × 6" → "25 × 6 is
computed"), so the faithful reverse is to restore it throughout. The prompt actually used:

```
Rewrite the given text, which is a part of a complete reasoning process, as a first-person account of
someone working the problem out for the first time and thinking aloud. Use first-person expressions
throughout, such as 'I', 'me', 'let me' and 'let's'. Where the text checks a result, corrects itself,
doubts a step, compares options or changes approach, say it the way a person thinking aloud would, for
example "Wait, ...", "Hmm, ...", "Let me check that...", "Actually, ...", "Maybe ...". Keep every
calculation, number and result exactly as in the original text.
Do not add new steps or new information. Do not include any introductory phrases.
Text:
{one piece of the PART solution}
```

**Second pilot (V2 prompt above):** self-talk reached 15.81 per 1k words, **49.8%** of LIMO's, a hair under
the 50% line, and 9 of 244 pieces (3.7%) fell back to PART's text, over the 2% limit. The rejections were
the checks working: the rewriter sometimes condensed a piece (to 0.6–0.9 of its length) or dropped
numbers (up to 6 of 27). **Third version, the one used** (with up to 5 attempts per piece instead of 3):

```
Rewrite the given text, which is a part of a complete reasoning process, as continuous first-person
thinking aloud by someone working the problem out for the first time. Change only how things are said,
never what is said: keep every sentence's content, every calculation, every number and every result, so
the rewrite is at least as long as the original. Use first-person expressions throughout, for example
"So I have...", "Let me compute...", "Okay, so...", "I think...". Where the text checks a result, corrects
itself, doubts a step, compares options or changes approach, say it the way a person thinking aloud would,
for example "Wait, ...", "Hmm, ...", "Let me check that...", "Actually, ...", "Maybe ...".
Do not add new steps or new information. Do not include any introductory phrases.
Text:
{one piece of the PART solution}
```

**Decision (2026-10-06): proceed with V3.** Its third pilot passed every check except self-talk:
15.26 per 1k words, **48%** of LIMO's (V2 had reached 49.8%). V2 and V3 landing in the same place suggests
this rewriter tops out at about half of LIMO's thinking-aloud, and further prompt changes tuned on the same
20 problems would start fitting the check rather than testing the attack. The 50% threshold was our own
choice, not from the paper, so the run proceeds with **that one check waived and reported**
(`37_revoice_audit.py --waive-self-talk`); all other checks still block training. The experiment therefore
tests whether restoring **about half** of the thinking-aloud recovers what PART removed; any recovery is a
lower bound on what a stronger rewriter could do.

The checks of §5 and the rule of §6 are otherwise unchanged throughout. All prompt versions are kept in
`revoice.py` (`REVOICE_PROMPT_V1`, `REVOICE_PROMPT_V2`, `REVOICE_PROMPT`).

## 4. Method

`src/antidistill/attacks/revoice.py`, built on PART's engine:

- the same ~1,024-token pieces at paragraph breaks; the final-answer section is never sent and is
  copied unchanged;
- greedy first attempt, up to 3 retries at temperature 0.7;
- **a rewrite is rejected if** it is cut off or empty, shorter than 0.9× or longer than 1.6× the input
  (for inputs of 300+ characters), loses more than max(1, 10%) of the input's distinct numbers, keeps
  under 70% of its content words, or turns sentences into a list;
- if all attempts fail, the piece keeps PART's text and is counted.

## 5. Checks before training (fixed now)

| check | threshold | why |
| --- | --- | --- |
| completeness | 800/800 problems | no dropped problems |
| pieces that kept PART's text | ≤ 2% | as for PART |
| final answers | agree with PART's on every problem | the answer section is copied |
| numbers kept vs PART | mean ≥ 0.95 | the mathematics must survive |
| **self-talk restored** | **≥ 50% of LIMO's self-talk density** (LIMO 30.4 per 1k words, so ≥ 15.2) | the attack actually did its job |

Self-talk density is the same measure as in the PART audit (singular first person plus interjections,
outside mathematics). Also reported: Kim et al.'s nine words, length, and the share of solutions over the
16k training limit.

If the self-talk check fails, the prompt is revised and the pilot re-run **before** any training.

## 6. Pre-registered decision rule

Pooled 600 problems, greedy, standard grading (the basis of every verdict in this study):

| re-voiced student | verdict |
| --- | --- |
| within **2.0 pp** of PART (**50.5%**) | re-voicing **does not** recover what PART removed |
| at or above LIMO − 2.0 pp (**60.8%**) | re-voicing **fully defeats** PART |
| otherwise | **partial**: share of the PART→LIMO gap recovered, `(R − 50.5) / (62.8 − 50.5)` |

Reported alongside, not part of the verdict: accuracy among finished MATH500 answers, unfinished answers
and loops, the score counting answers given before a loop, and GSM8K. These matter here because the
PART students' biggest problem is failing to stop.

## 7. What each outcome would mean

- **Full recovery** would mean PART removes only the wording, and the wording can be put back by anyone
  with an ordinary instruction model: PART is broken by a cheap attack, much as the Style attack beat
  hindsight.
- **No recovery** would mean the invented thinking-aloud does not replace the real one. Note that Style,
  which also invents uncertainty, did recover most of hindsight's loss, so this outcome would be informative.
- **Partial** is the most likely: our S1 experiment showed the marker words carry part, not all, of the effect.

## 8. Caveats

- **Same model as the defender's rewriter.** Re-voicing with the model that did the rewriting could be
  easier than with an unrelated model. A second attacker model would test this if the result is strong.
- **The invented thinking aloud is not the original.** It reflects where an instruction model guesses
  uncertainty belongs, not where LIMO's author actually doubted.
- **Stopping.** Both PART students fail to stop after answering far more often than LIMO's; the
  re-voiced student may inherit that, so the secondary measures in §6 are reported for every run.

## 9. Cost and schedule (GPUs 4–7)

| step | estimate |
| --- | --- |
| pilot, 20 problems | ~15 min |
| generation, 800 problems (~10,700 pieces, one pass) | ~4 h on one 4-GPU server |
| training | ~8 h |
| evaluation: 600 problems, then GSM8K | ~3 h + ~10 h |

## 10. Files

| file | purpose |
| --- | --- |
| `src/antidistill/attacks/revoice.py` | the attack (PART's engine, new prompt and checks) |
| `scripts/45_generate_part.sh` | generation wrapper (`GEN_MODULE`, `INPUT`, `OUTPUT`) |
| `scripts/37_revoice_audit.py` | the §5 checks and the §6 verdict |
| `configs/train/qwen2.5-7b_revoice_part.yaml` | identical to PART's config except dataset and output folder |
| `scripts/23_revoice_run.sh` | generation → checks → training → evaluation, in one detached job |
