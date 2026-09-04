# Task 0.3 — LIMO-v2 trace pool statistics

- dataset: `GAIR/LIMO-v2` split `train`
- rows: **800** (expected 800)
- tokenizer: `Qwen/Qwen2.5-7B`
- training `cutoff_len`: **16384**

## Sequence length (chat-templated prompt + response), tokens

| stat | prompt+response | response only |
| --- | --- | --- |
| min | 1207 | 1166 |
| p25 | 7411 | 7349 |
| median | 12212 | 12091 |
| mean | 13103 | 12995 |
| p75 | 17834 | 17683 |
| p95 | 25025 | 24910 |
| max | 32900 | 32768 |

## Truncation at cutoff_len=16384

**256/800 = 32.00%** of traces exceed the cutoff and will be truncated during SFT.

> This is inherited from the LIMO default config, which Kim et al. state they used unchanged. We keep it for the replication; the number is recorded here so any later change to `cutoff_len` is a documented deviation rather than a silent one.
