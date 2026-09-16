# M6c — Does the reconstruction attack need a strong attacker?

**Status: complete (2026-09-15).** Three training runs, 1,800 problem-evaluations.
Follows `results/m6b_reconstruction.md`, which ran the same three attacks with
DeepSeek-R1-Distill-Qwen-**32B** as the attacker's model.

**Question.** M6b concluded the hindsight defense is *irrelevant* rather than broken, because the
`solo` control — an attacker using no defended data — beat `search`. The obvious objection: the
attacker was **the same model as the defender's teacher**, so of course it needed no help. This
repeats everything with **DeepSeek-R1-Distill-Qwen-7B**, an attacker **no larger than the student
it is training**. Same family, same reasoning parser, byte-identical prompts — capability is the
only variable.

---

## 1. Headline — the weaker attacker produced the BETTER students

Greedy pass@1, pooled over 600 problems. All conditions: 800 problems, identical hyperparameters,
1,500 steps.

| condition | attacker | epi /1kw | MATH500 | AMC23 | AIME24 | AIME25 | **pooled** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base (untrained) | — | — | 55.0% | 40.0% | 20.0% | 6.7% | **49.8%** |
| **defended v2** | — | 0.02 | 56.8% | 27.5% | 3.3% | 3.3% | **49.5%** |
| LIMO (undefended ceiling) | — | 35.57 | 69.0% | 55.0% | 20.0% | 13.3% | **62.8%** |
| A1 style | 32B | 13.71 | 70.8% | 65.0% | 13.3% | 3.3% | **64.2%** |
| A2 search | 32B | 16.49 | 71.2% | 47.5% | 16.7% | 20.0% | **64.3%** |
| A3 solo | 32B | 24.18 | 74.2% | 57.5% | 16.7% | 20.0% | **67.5%** |
| **A1 style** | **7B** | 4.66 | 68.4% | 47.5% | 10.0% | 3.3% | **60.8%** |
| **A2 search** | **7B** | 18.82 | **76.2%** | 57.5% | **6.7%** | 13.3% | **68.3%** |
| **A3 solo** | **7B** | 25.99 | **77.0%** | 55.0% | 16.7% | 16.7% | **69.5%** |

**The 7B's advantage is concentrated in MATH500** — +5.0 pp (search) and +2.8 pp (solo) on the only
benchmark with the resolution to show it, while AMC23 and the AIME sets move within one or two
problems either way. The single column running against the 7B is **AIME24 for `search` (6.7% vs
16.7%)**, a 3-problem swing with normal termination (20/30 finished), so noise-dominated rather
than a hard-problem deficit.

**The 7B attacker beat the 32B on both real-doubt conditions** — `search` 68.3% vs 64.3%, `solo`
69.5% vs 67.5%. `7B solo` is the best result in the project, above the undefended LIMO ceiling by
6.7 pp.

This was the opposite of the prediction. The experiment was designed expecting degradation that
would expose the defended data's value.

---

## 2. The control replicates: the defense is irrelevant at both capability levels

| attacker | `search` | `solo` | winner |
| --- | --- | --- | --- |
| 32B | 64.3% | **67.5%** | solo by 3.2 pp |
| **7B** | 68.3% | **69.5%** | solo by 1.2 pp |

The pre-registered criterion (design §3): `search` > `solo` means the attack exploits the defended
data; `search` ≈ `solo` means the attacker never needed it.

**`solo` ≥ `search` at both levels.** At 7B the margin is 1.2 pp — 7 problems in 600, inside noise
— so the honest statement there is `solo` ≈ `search`, which satisfies the criterion either way.

This holds even though the correctness gap between the two pools **widened** at 7B:

| | 32B | 7B |
| --- | --- | --- |
| `search` traces reaching LIMO's answer | 89% | 81% |
| `solo` traces reaching LIMO's answer | 49% | **37%** |

An attacker whose unaided traces are wrong **63% of the time** still gained nothing from being
handed correct defended solutions. That is the strongest form of the M6b finding.

---

## 3. Epistemic density tracks the outcome — but is CONFOUNDED with length

Across all six reconstruction runs:

| run | attacker | epi /1k words | mean trained tokens | pooled |
| --- | --- | --- | --- | --- |
| A1 style | 7B | 4.66 | 1,658 | 60.8% |
| A1 style | 32B | 13.71 | 3,578 | 64.2% |
| A2 search | 32B | 16.49 | 5,526 | 64.3% |
| A2 search | 7B | 18.82 | 7,294 | 68.3% |
| A3 solo | 32B | 24.18 | 9,664 | 67.5% |
| A3 solo | 7B | 25.99 | 10,452 | 69.5% |

| correlation with pooled accuracy | r |
| --- | --- |
| epistemic density | **+0.940** |
| trained tokens | **+0.938** |
| **epistemic density vs trained tokens** | **+0.978** |

**These two cannot be separated in this data.** Density and length are almost perfectly collinear
(r = 0.978) and predict the outcome equally well. **No claim that epistemic content *causes* the
improvement is supported here** — it is equally consistent with "longer training traces produce
better students", which is a far less interesting hypothesis.

Disentangling them needs a condition that breaks the collinearity: e.g. long traces with doubt
stripped, or short traces with doubt concentrated. That experiment does not exist yet and is the
single most valuable follow-up from this milestone.

**LIMO is the one point off the line** — highest density of all (35.57) but only 62.8%, below runs
with half its density. Its 32.0% truncation is the obvious explanation, but see §4.

---

## 4. CORRECTION — the M6a termination mechanism does not generalise

`results/m6_supplementation.md` §2 attributed the non-monotonic mixture result to a specific
mechanism: **trained tokens carrying no stop signal damage the student's ability to terminate.**
Within that mixture series it held cleanly (mix10: 33.0% no-stop tokens, 455/500 finished;
hindsight: 11.3%, 499/500).

**It does not hold across these six runs.** Correlation of no-stop token share with MATH500
finished count is **r = +0.637 — the wrong sign entirely:**

| run | no-stop tokens | MATH500 finished |
| --- | --- | --- |
| 32B style | 5.1% | 467/500 |
| 32B search | 6.6% | 462/500 |
| 7B style | 9.8% | 473/500 |
| 7B search | 21.7% | 475/500 |
| 32B solo | 22.1% | 465/500 |
| **7B solo** | **36.2%** | **477/500** |

`7B solo` has **1.6x** the no-stop token share of `32B solo` and yet terminates *better*
(477 vs 465). LIMO, with the highest share (45.4%), does have the worst termination (441/500) — so
the mechanism is not simply wrong, but it is **not the general law M6a's write-up implied**. Across
these runs, no-stop token share is largely a proxy for trace length, and length helps.

**What this changes:** M6a's non-monotonicity finding stands as an observation, but its stated
*mechanism* is now only demonstrated within the mixture series, not as a general property. M6b §4,
which used the same mechanism to attribute the attacks' margin over LIMO to termination, inherits
the same qualification.

---

## 5. The attack genuinely out-reasons the undefended ceiling

MATH500, conditioned on producing an answer:

| condition | finished | **accuracy \| finished** |
| --- | --- | --- |
| base | 415/500 | 66.3% |
| defended v2 | 498/500 | 57.0% |
| LIMO | 441/500 | 78.2% |
| 32B solo | 465/500 | 79.8% |
| **7B search** | 475/500 | **80.2%** |
| **7B solo** | **477/500** | **80.7%** |

`7B solo` beats LIMO on **both** factors — it finishes more often *and* is more often right when it
does. So unlike the 32B runs (where the entire margin was termination, M6b §4), this one is not
explainable as a termination artifact. A student trained on traces that reach the right answer 37%
of the time reasons better than one trained on the undefended reference data.

---

## 6. `style` degrades sharply — fabrication is capability-dependent

| | 32B | 7B |
| --- | --- | --- |
| epistemic /1k words | 13.71 | **4.66** |
| traces containing any doubt | 794/800 | **430/800** |
| pooled | 64.2% | **60.8%** |

The 7B follows the *first-person* half of the A1 instruction but not the *show-where-it-could-go-wrong*
half, writing past-tense reportage: *"First, I calculated... Next, I aimed to find..."*. 7B `style`
is the only attack that falls **below** the undefended LIMO ceiling.

**The prompt was deliberately left identical** across attackers. Tuning it per-model would have
made capability and prompt vary together and confounded the comparison. The consequence is that
**7B `style` does not test H1** — it tests whether the 7B *can* fabricate doubt on command. The H1
conclusion (fabricated doubt works as well as searched doubt) rests on the 32B pair only, where
`style` and `search` were separated by 0.1 pp at comparable density.

That fabrication is capability-dependent while *harvesting* real doubt is not (7B `search` and
`solo` both improved) is itself a finding: the cheapest attack variant is the one that needs the
strongest attacker.

---

## 7. Caveats

| | |
| --- | --- |
| **Density/length collinearity (§3)** is the dominant limitation. Nothing here separates "epistemic content helps" from "long traces help". |
| **n = 6 runs.** r = 0.940 on six points is suggestive, not conclusive, and two of the six are the degraded `style` conditions. |
| **AIME is noise-dominated.** 7B `search` scores 6.7% on AIME24 vs the 32B's 16.7%, while beating it by 5 pp on MATH500 — a 3-problem swing at 3.33 pp per problem. Quote the pooled 600. |
| **Still only two attacker models**, both from the same R1-distill family. Nothing tests a different lineage. |
| **The `cutoff_len` control remains outstanding** and is now more important, since §4 shows our understanding of the truncation mechanism is incomplete. |

---

## 8. Reproduction

```bash
# 7B attacker: one GPU per mode, all three in parallel (~5 h total)
for i in 0 1 2; do MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B GPUS=$i PORT=$((8020+i)) \
    HF_HUB_OFFLINE=1 scripts/40_serve_teacher.sh; done
# style:8020 search:8021 solo:8022
GPUS=0,1,2,3 scripts/19_train_recon7b.sh
GPUS=0,1,2,3 scripts/18b_recon7b_eval.sh search solo style
```

Generation ~5 h (vs ~14 h for the 32B); training 7.29 / 7.82 / 6.34 h; evaluation ~9 h.
A 7B serves at TP=1 on one 46 GB card — no collectives, and it sidesteps the host's broken P2P
entirely.
