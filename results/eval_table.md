# Eval results (recomputed from stored per-problem verdicts)

| condition | benchmark | temp | k | pass@k | avg@k | mean resp tokens |
| --- | --- | --- | --- | --- | --- | --- |
| base | aime | 0.0 | 1 | 6/30 = 20.00% | — | — |
| base | aime | 0.7 | 16 | 8/30 = 26.67% | 7.08% | 1791 |
| hindsight_ep10 | aime | 0.0 | 1 | 1/30 = 3.33% | — | — |
| hindsight_ep15 | aime | 0.0 | 1 | 2/30 = 6.67% | — | — |
| hindsight_ep5 | aime | 0.0 | 1 | 1/30 = 3.33% | — | — |
| limo_ep10 | aime | 0.0 | 1 | 6/30 = 20.00% | — | — |
| limo_ep15 | aime | 0.0 | 1 | 6/30 = 20.00% | — | — |
| limo_ep5 | aime | 0.0 | 1 | 4/30 = 13.33% | — | — |
| base | aime25 | 0.0 | 1 | 2/30 = 6.67% | — | — |
| hindsight_ep15 | aime25 | 0.0 | 1 | 1/30 = 3.33% | — | — |
| limo_ep15 | aime25 | 0.0 | 1 | 4/30 = 13.33% | — | — |
| base | amc | 0.0 | 1 | 16/40 = 40.00% | — | — |
| hindsight_ep15 | amc | 0.0 | 1 | 15/40 = 37.50% | — | — |
| limo_ep15 | amc | 0.0 | 1 | 22/40 = 55.00% | — | — |
| base | math | 0.0 | 1 | 275/500 = 55.00% | — | — |
| hindsight_ep15 | math | 0.0 | 1 | 321/500 = 64.20% | — | — |
| limo_ep15 | math | 0.0 | 1 | 345/500 = 69.00% | — | — |

## Reference — Kim et al. / proposal, Qwen2.5-7B, AIME24 greedy pass@1

| base | LIMO | hindsight |
| --- | --- | --- |
| 13.3% (4/30) | 26.7% (8/30) | 3.3% (1/30) |

> Absolute levels are not directly comparable: greedy decoding differs across GPU
> architectures (see `results/m1_base.md`). Judge on ordering and on avg@k, and use
> our own base row as the baseline.
