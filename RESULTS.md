# Voxtral Realtime simplification: A100 throughput and WER rerun

Four paired rounds; each comparison uses the same physical A100-SXM4-80GB. Original BF16 weights, existing decode graphs, request lists and runtime are held fixed. Throughput is audio seconds processed per wall second through the Realtime WebSocket server; these are complete serving results, not isolated kernel timings.

- **Native:** current PR model code with audio embedding graphs disabled.
- **Published:** current PR `4233576700a4`, with audio embedding graphs enabled.
- **Simplified:** Andy’s `e3a0be385e2a`, plus the tested normalization regression and docs link; audio embedding graphs enabled.

18,432 measured throughput requests; 6,510 quality/streaming requests. Warmups excluded. Gains are geometric means of four paired ratios; round ranges are recorded in `SUMMARY.json`.

| Concurrency | Short 1–3 s | Long 15–30 s | Mixed |
| --- | --- | --- | --- |
| 1 | +4.8% → +4.7% | +4.0% → +4.3% | +3.3% → +4.0% |
| 8 | +3.1% → +3.7% | +4.6% → +4.9% | +3.9% → +5.1% |
| 32 | +22.0% → +18.1% | +13.7% → +13.5% | +12.6% → +12.2% |
| 64 | +38.8% → +42.0% | +43.8% → +43.8% | +39.2% → +37.9% |

Cells show **published → simplified** throughput gains over native. Refactor-only ratios and per-round ranges are in the JSON; small changes should not be called improvements without considering those ranges.

| WER pass | Native | Published | Simplified |
| --- | --- | --- | --- |
| quality_earnings511 | 12.721% (1058/8317) | 12.673% (1054/8317) | 12.601% (1048/8317) |
| quality_earnings511:C32:R1 | 12.709% (1057/8317) | 12.601% (1048/8317) | 12.685% (1055/8317) |
| quality_earnings511:C32:R2 | 12.685% (1055/8317) | 12.673% (1054/8317) | 12.733% (1059/8317) |
| quality_earnings511:C1:R0 | 12.625% (1050/8317) | 12.625% (1050/8317) | 12.625% (1050/8317) |

Each WER pass uses the same 511 Earnings22 clips and normalization. The three C32 passes repeat the same clips; they are a batching-variation control, not 1,533 independent recordings. The C1 pass is reported separately. Text need not match bitwise across serving runs.

Long 60/180/300 s and ragged-packet controls have no aligned WER references. Their results establish complete streaming and delta/output consistency, not long-form accuracy. Actual graph hits, zero fallback for the captured schedule, exact request identities and token counts, source hashes, GPU UUIDs and output-delta consistency are audited before this report is produced.

Memory snapshots and individual latency results are retained in `SUMMARY.json` and the raw serving JSON.

A separate short-C32 control used two paired rounds in AB/BA order, 1,024 requests per cell (4,096 total), on another A100 host. Simplified versus published throughput changed +0.16% overall; the two paired results were -0.39%, +0.71%. This was added to investigate one short-cell outlier; it does not replace any primary result above.
