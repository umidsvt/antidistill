# M6f — Style attack on PART: design

**Status: complete (2026-10-16): PARTIAL — recovers 68% of the PART-to-LIMO gap (58.8%); GSM8K 84.2%.** Results will be
added to `results/REPORT.md` Part 9.

---

## 1. Why this attack

The re-voice attack (`results/m6e_revoice_design.md`, REPORT Part 9) failed: 52.5% against PART's 50.5%.
It restored PART's first-person voice (self-talk 1.8 → 14.3 per 1k words) but **not its uncertainty**:
the uncertainty words stayed where PART left them (S1 list: 11.0 per 1k words, LIMO 35.6), and the student
it produced does not use them.

The reason is visible in the data. PART's step 1 does not just change the voice; it **collapses each
checking episode into its conclusion**:

> LIMO: *"Wait, hold on, no. If a person leaves, we're reducing the number of people in the room as
> well, right? Or does the problem say that someone leaves and someone else enters? Let me check."*
>
> PART: *"The total number of people remains the same, which is six, because one left and one entered."*

A rewrite that only changes the voice has no doubt to restore. This attack instead asks the rewriter to
**put checking back**: consider an approach before committing, note where a step could go wrong, double
back, verify. That is our Style prompt, which recovered most of hindsight's loss (64.2% with the 32B
attacker, against the defense's 49.5%), applied to PART's full-length solutions piece by piece.

## 2. Threat model

As for re-voice: the attacker has PART's **800 defended solutions** and an **ordinary instruction model
with no reasoning mode** (Qwen2.5-32B-Instruct); it never sees LIMO's text.

## 3. The prompt

The Style prompt (`results/m6b_prompts.md`, A1), adapted from a whole solution to one piece of it:

```
Rewrite the given text, which is a part of a complete reasoning process, as a first-person account of
working the problem out for the first time. Show the reasoning as it would actually unfold: consider an
approach before committing to it, note where a step could go wrong, double back when something does not
check out, and verify results that are easy to get wrong, for example "Wait, ...", "Hmm, ...", "Let me
check that...", "Maybe ...", "Actually, ...". Keep every calculation, number and result from the original
text.
Do not include any introductory phrases.
Text:
{one piece of the PART solution}
```

Unlike re-voice, it may add content: checks and corrections that were not in the text. That is the point.

### Revision 1 (2026-10-12, after the first pilot, before any training)

The prompt above failed two checks on the 20-problem pilot: **8.6% of pieces** had every attempt rejected
(limit 2%), and uncertainty words reached only **13.1 per 1k words, 36% of LIMO's** (PART: 11.9). The
instruction model condensed instead of expanding: rejected attempts came back at about 0.72× the input,
losing content. Revision 1 says to keep everything and names where to add doubt:

```
Rewrite the given text, which is a part of a complete reasoning process, as a first-person account of
working the problem out for the first time. Keep all of the original content: every sentence's content,
every calculation, every number and every result, so the rewrite is longer than the original. Add what a
person working it out would say along the way, several times in the text: before a step, briefly wonder
whether the approach is right ("Hmm, maybe I should ...", "Let me think..."); after a result that is easy
to get wrong, check it ("Wait, let me check that...", "Actually, ..."); where the text settles a question,
show the doubt that came first ("Wait, is that right?"). Do not repeat the same check twice.
Do not include any introductory phrases.
Text:
{one piece of the PART solution}
```

Pilot result with revision 1: pieces with every attempt rejected **1.23%** (passes), numbers kept 0.992,
answers agree on all 20, but uncertainty words **17.16 per 1k words, 47% of LIMO's 36.48** on the same 20
problems (PART: 11.88). The check needs 50%, so it fails by a small margin.

### Revision 2 (2026-10-12, after the second pilot, before any training) — the last one allowed

Word by word, the shortfall was in "wait" (3.4 per 1k words, LIMO 10.9), "perhaps" (1.7, LIMO 4.8) and
"maybe" (3.5, LIMO 6.4). The rewrite also dropped part of PART's own "alternatively" (5.5 → 3.8), and only
57% of paragraphs longer than 25 words received any added doubt. Revision 2 asks for doubt in every
paragraph and for the original's "alternatively"/"perhaps" passages to be kept:

```
Rewrite the given text, which is a part of a complete reasoning process, as a first-person account of
working the problem out for the first time. Keep all of the original content: every sentence's content,
every calculation, every number and every result, so the rewrite is longer than the original. Where the
original considers another method or another possibility ("Alternatively, ...", "Perhaps ..."), keep that
wording. In every paragraph, add what a person working it out would say along the way: before a step,
wonder whether the approach is right ("Hmm, maybe I should ...", "Perhaps ..."); after a result, stop and
question it ("Wait, is that right? Let me check...", "Wait, actually ..."); where the text settles a
question, show the doubt that came first ("Wait, maybe ..."). Do not repeat the same check twice.
Do not include any introductory phrases.
Text:
{one piece of the PART solution}
```

The prompt now names the words the check counts. That is what an attacker would do (the words are
published in Kim et al.), but it means passing the check shows the words are present, not that the
checking they introduce is sound. The trained student's accuracy is the test of that.

No revision remains. If this pilot fails a check, the run stops and the decision comes back to us.

**Pilot result with revision 2 (2026-10-12 23:06): all checks pass.**

| check | measured | |
| --- | --- | --- |
| completeness | 20 problems, 244 pieces | pass |
| pieces that kept PART's text | 3/244 = 1.23% | pass |
| final answers agree with PART | 20/20 | pass |
| numbers kept | mean 0.994 | pass |
| uncertainty words restored | 23.94 per 1k words, **66% of LIMO's 36.48** | pass |

| dataset (20 pilot problems) | tokens/problem | self-talk /1kw | uncertainty words (S1 list) /1kw | Kim et al. nine /1kw | over 16k limit |
| --- | --- | --- | --- | --- | --- |
| LIMO | 12,544 | 31.75 | 36.48 | 40.35 | 30% |
| PART (input) | 11,561 | 1.64 | 11.88 | 16.98 | 25% |
| Style on PART | 13,199 | 23.83 | 23.94 | 32.83 | 40% |

No accepted piece repeats itself heavily (0 of 241 with fewer than 80% distinct 8-word sequences). One
thing to watch: the rewrite is 14% longer than PART, so more solutions pass the 16k training limit (40% of
the pilot against LIMO's 30%), and those lose their end-of-answer token. If the student stops less often
than PART's, this is the first place to look. The revision-1 pilot files are kept as
`data/curated/style_part_pilot_rev1.*`.

The full run was started on 2026-10-12 23:38 on GPUs 4–7 (`logs/style_part_run.log`).

### Full data (2026-10-13 04:17): four checks pass, one fails on a single problem — training not started

| check | measured | |
| --- | --- | --- |
| completeness | 800 problems, 10,014 pieces | pass |
| pieces that kept PART's text | 83/10,014 = 0.83% | pass |
| final answers agree with PART | **1 of 800 flagged (problem 355)** | **fail** |
| numbers kept | mean 0.995 | pass |
| uncertainty words restored | 23.23 per 1k words, 65% of LIMO's 35.57 | pass |

| dataset (800 problems) | tokens/problem | self-talk /1kw | uncertainty words (S1 list) /1kw | Kim et al. nine /1kw | over 16k limit |
| --- | --- | --- | --- | --- | --- |
| LIMO | 12,996 | 30.39 | 35.57 | 39.01 | 32.0% |
| PART (input) | 12,040 | 1.75 | 10.99 | 15.71 | 25.6% |
| Style on PART | 13,887 | 24.70 | 23.23 | 32.42 | 36.8% |

**What the flag on problem 355 is.** The final-answer section is the same as PART's, character for
character, on all 800 problems including 355 (checked directly with `split_answer`). In problem 355 LIMO's
own text ends with an unclosed box, `**Final Answer**\n\boxed{6000`, so the checker cannot read the final
answer and falls back to the last complete box, which sits in the middle of the solution. There LIMO and
PART have *"...higher revenue by approximately 2,666.67 units. **Final Answer** ... greater by
\boxed{6666.67} units"* (the box contradicts the sentence before it), and the rewrite changed the boxed
number to 2,666.67. So the rewrite altered one number inside the reasoning of one problem; the final
answer did not change.

The runner stopped before training, as designed. No prompt or threshold was changed.

**Decision (2026-10-13): train on the data as it is.** The flag on problem 355 is waived by name
(`--waive-answer 355` in `scripts/37_revoice_audit.py`); a flag on any other problem would still block.
The check is still computed and printed in every audit. For the paper's appendix: one of 800 training
solutions has one number inside its reasoning changed by the rewrite; no final answer differs from PART's.

**Order on GPUs 4–7** (`scripts/25_style_part_after_solo.sh`): training starts when the OlympiadBench
queue (`results/m7_hard_eval_design.md`) has finished the four students its rule needs; then the pooled-600
evaluation and the verdict; then the remaining OlympiadBench students; GSM8K for this student last.

## 4. Method

`src/antidistill/attacks/style_part.py`, the re-voice engine with this prompt. Same ~1,024-token pieces,
final answer copied unchanged, greedy first attempt and up to 5 attempts. A rewrite is rejected if it is
cut off or empty, **shorter than 0.9× or longer than 2.2×** the input (wider than re-voice's 1.6× because
added checks lengthen the text), loses more than max(1, 10%) of the input's numbers, keeps under 70% of its
content words, or turns sentences into a list. A piece whose attempts all fail keeps PART's text.

## 5. Checks before training (fixed now)

| check | threshold |
| --- | --- |
| completeness | 800/800 problems |
| pieces that kept PART's text | ≤ 2% |
| final answers | agree with PART's on every problem |
| numbers kept vs PART | mean ≥ 0.95 |
| **uncertainty words restored** | **≥ 50% of LIMO's density on the S1 marker list** (LIMO 35.6 per 1k words, so ≥ 17.8; PART has 11.0) |

The restoration check is on the uncertainty words themselves, not on first person, because first person
alone was shown not to matter (re-voice). Self-talk and Kim et al.'s nine words are reported too.

**Lesson from re-voice applied in advance:** at most **two** prompt revisions on the pilot. If the check
still fails after that, the run stops and the decision comes back to us, rather than tuning further on the
same 20 problems.

## 6. Pre-registered decision rule

Pooled 600 problems, greedy, standard grading, the same rule as re-voice:

| Style-on-PART student | verdict |
| --- | --- |
| within **2.0 pp** of PART (**50.5%**) | does not recover what PART removed |
| at or above **60.8%** (LIMO − 2.0) | fully defeats PART |
| otherwise | partial: `(S − 50.5) / (62.8 − 50.5)` of the gap recovered |

Reported alongside: accuracy among finished MATH500 answers, unfinished answers and loops, the score
counting answers given before a loop, GSM8K, and the uncertainty words the trained student uses.

## 7. What each outcome would mean

- **Recovery** means what PART removes can be re-invented: doubt and checking written by a model that
  never doubted are enough, as with Style against hindsight. PART would then be broken by an attacker
  with only an instruction model.
- **No recovery**, after re-voice also failed, would make PART the first defense in this study that an
  attacker without a reasoning model cannot undo — a result worth reporting on its own.

## 8. Caveats

- Same model as PART's rewriter (see re-voice §8).
- Invented checks may sit in the wrong places. Style's invented doubt against hindsight was placed like
  LIMO's (32/39/29% across thirds vs 34/34/32%); the same placement measure will be reported here.
- The stopping problem of the PART students may carry over; the secondary measures in §6 track it.

## 9. Schedule

Needs at least 2 GPUs for the rewriter. On 2026-10-09 only GPU 4 was free (0–3 and 5 khizar, 6–7 ktran).

| step | estimate (4 GPUs) |
| --- | --- |
| pilot, 20 problems | ~15 min |
| generation, 800 problems | ~5 h (longer outputs than re-voice) |
| training | ~8 h |
| evaluation | ~3 h + ~10 h |

Command: `ATTACK=style_part scripts/23_revoice_run.sh` (pilot first:
`PILOT=1 GEN_MODULE=antidistill.attacks.style_part INPUT=data/defended/limo_part.json
OUTPUT=data/curated/style_part.json scripts/45_generate_part.sh`, then
`scripts/37_revoice_audit.py --attack style_part --pilot`).

## Result (2026-10-14)

| student | MATH500 | AMC23 | AIME24 | AIME25 | **pooled (600)** | MATH500 acc \| finished |
| --- | --- | --- | --- | --- | --- | --- |
| LIMO | 69.0% | 55.0% | 20.0% | 13.3% | **62.8%** | 78.2% (441/500) |
| **Style on PART** | 65.4% | 42.5% | 16.7% | 13.3% | **58.8%** | 73.0% (448/500) |
| re-voiced PART | 58.2% | 40.0% | 13.3% | 13.3% | **52.5%** | 67.4% (432/500) |
| PART | 55.2% | 50.0% | 6.7% | 16.7% | **50.5%** | 67.2% (411/500) |

**Verdict by §6: partial — (58.8 − 50.5) / (62.8 − 50.5) = 68% of the gap recovered.** The student writes
uncertainty words at 22.3 per 1k words in its MATH500 answers (PART's student 7.1, re-voice 4.6, LIMO
27.6). Training took 8:12 on GPUs 4–7 (2026-10-13 18:35 to 2026-10-14 02:48). **GSM8K (TP=1, last-number
fallback): 84.2%** (boxed only 75.0%; finished 445/500), against PART 76.2% and LIMO 85.2%: 89% of the gap. Not yet done: the placement of invented doubt (§8).
