# Eval results (recomputed from stored per-problem verdicts)

| condition | benchmark | temp | k | pass@k | avg@k | mean resp tokens |
| --- | --- | --- | --- | --- | --- | --- |
| base | aime | 0.0 | 1 | 6/30 = 20.00% | — | — |
| base | aime | 0.7 | 16 | 8/30 = 26.67% | 7.08% | 1791 |
| hindsight_ep10 | aime | 0.0 | 1 | 1/30 = 3.33% | — | — |
| hindsight_ep15 | aime | 0.0 | 1 | 2/30 = 6.67% | — | — |
| hindsight_ep5 | aime | 0.0 | 1 | 1/30 = 3.33% | — | — |
| hindsight_v2_ep15 | aime | 0.0 | 1 | 1/30 = 3.33% | — | — |
| limo_ep10 | aime | 0.0 | 1 | 6/30 = 20.00% | — | — |
| limo_ep15 | aime | 0.0 | 1 | 6/30 = 20.00% | — | — |
| limo_ep5 | aime | 0.0 | 1 | 4/30 = 13.33% | — | — |
| mixed10_ep15 | aime | 0.0 | 1 | 3/30 = 10.00% | — | — |
| mixed25_ep15 | aime | 0.0 | 1 | 5/30 = 16.67% | — | — |
| mixed50_ep15 | aime | 0.0 | 1 | 5/30 = 16.67% | — | — |
| recon_search_ep15 | aime | 0.0 | 1 | 5/30 = 16.67% | — | — |
| recon_solo_ep15 | aime | 0.0 | 1 | 5/30 = 16.67% | — | — |
| recon_style_ep15 | aime | 0.0 | 1 | 4/30 = 13.33% | — | — |
| base | aime25 | 0.0 | 1 | 2/30 = 6.67% | — | — |
| hindsight_ep15 | aime25 | 0.0 | 1 | 1/30 = 3.33% | — | — |
| hindsight_v2_ep15 | aime25 | 0.0 | 1 | 1/30 = 3.33% | — | — |
| limo_ep15 | aime25 | 0.0 | 1 | 4/30 = 13.33% | — | — |
| mixed10_ep15 | aime25 | 0.0 | 1 | 2/30 = 6.67% | — | — |
| mixed25_ep15 | aime25 | 0.0 | 1 | 4/30 = 13.33% | — | — |
| mixed50_ep15 | aime25 | 0.0 | 1 | 2/30 = 6.67% | — | — |
| recon_search_ep15 | aime25 | 0.0 | 1 | 6/30 = 20.00% | — | — |
| recon_solo_ep15 | aime25 | 0.0 | 1 | 6/30 = 20.00% | — | — |
| recon_style_ep15 | aime25 | 0.0 | 1 | 1/30 = 3.33% | — | — |
| base | amc | 0.0 | 1 | 16/40 = 40.00% | — | — |
| hindsight_ep15 | amc | 0.0 | 1 | 15/40 = 37.50% | — | — |
| hindsight_v2_ep15 | amc | 0.0 | 1 | 11/40 = 27.50% | — | — |
| limo_ep15 | amc | 0.0 | 1 | 22/40 = 55.00% | — | — |
| mixed10_ep15 | amc | 0.0 | 1 | 15/40 = 37.50% | — | — |
| mixed25_ep15 | amc | 0.0 | 1 | 19/40 = 47.50% | — | — |
| mixed50_ep15 | amc | 0.0 | 1 | 20/40 = 50.00% | — | — |
| recon_search_ep15 | amc | 0.0 | 1 | 19/40 = 47.50% | — | — |
| recon_solo_ep15 | amc | 0.0 | 1 | 23/40 = 57.50% | — | — |
| recon_style_ep15 | amc | 0.0 | 1 | 26/40 = 65.00% | — | — |
| base | math | 0.0 | 1 | 275/500 = 55.00% | — | — |
| hindsight_ep15 | math | 0.0 | 1 | 321/500 = 64.20% | — | — |
| hindsight_v2_ep15 | math | 0.0 | 1 | 284/500 = 56.80% | — | — |
| limo_ep15 | math | 0.0 | 1 | 345/500 = 69.00% | — | — |
| mixed10_ep15 | math | 0.0 | 1 | 293/500 = 58.60% | — | — |
| mixed25_ep15 | math | 0.0 | 1 | 332/500 = 66.40% | — | — |
| mixed50_ep15 | math | 0.0 | 1 | 356/500 = 71.20% | — | — |
| recon_search_ep15 | math | 0.0 | 1 | 356/500 = 71.20% | — | — |
| recon_solo_ep15 | math | 0.0 | 1 | 371/500 = 74.20% | — | — |
| recon_style_ep15 | math | 0.0 | 1 | 354/500 = 70.80% | — | — |

## Reference — Kim et al. / proposal, Qwen2.5-7B, AIME24 greedy pass@1

| base | LIMO | hindsight |
| --- | --- | --- |
| 13.3% (4/30) | 26.7% (8/30) | 3.3% (1/30) |

> Absolute levels are not directly comparable: greedy decoding differs across GPU
> architectures (see `results/m1_base.md`). Judge on ordering and on avg@k, and use
> our own base row as the baseline.
