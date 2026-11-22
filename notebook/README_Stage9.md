# Stage 9 — Fast Adaptive Context & Encoding Overhead Ablation

## Completion and reproducibility

Completed on 22 November 2026 across Stage9-Session01 and Stage9-Session02. Source commit `62f09e3105296d62c39e8d95cf8b8775e63f2677`; original codec source unchanged. GPT-2 FP32, SDPA, Tesla T4 GPU 0, PyTorch 2.10.0+cu128, Transformers 5.0.0, NumPy 2.0.2, max batch tokens 1024, TF32 disabled, no adapter. Six 100,000-byte Stage 8 samples (Wikipedia, English literature, Python, CSV, Vietnamese, mixed); five independent source corpora, mixed derived. This is ablation on reused data, not fresh independent validation. Target contexts 64 and 128.

## Methods

| Method | Candidate contexts | Lambda count | Proposals |
|---|---|---:|---:|
| Fixed | Target context, with tail handling | — | 0 |
| BC-DP Full | 32,64,128,256,512 | 122 | 488 |
| Reduced Search | 32,64,128,256,512 | 32 | 128 |
| Reduced Levels | 32,64,128 | 122 | 488 |
| Fast BC-DP | 32,64,128 | 32 | 128 |

Frozen Stage 7 objectives (NLL and rank proxy), fitted quadratic and measured latency models; dense lambda grid zero plus 121 logarithmically spaced points from 0.01 to 10000; reduced grid every fourth dense entry plus final entry. Tree DP generates candidates; BZip2 rank payload plus layout flag size selects the smallest feasible candidate under the predicted fixed decode budget, breaking size ties by predicted latency. Search is heuristic over generated candidates, not a guarantee of global constrained optimality. A restricted level set can generate candidates absent from the full-grid search.

## Bottleneck and fresh inference

Stage 8 encoding shares: GPT-2 analysis 76.3507%, search 7.7032%, codec 15.9461%. Eliminating search alone would yield mean theoretical speedup 1.0836x. Fresh analysis: two level sets, six samples, three repetitions, 36/36 rank and NLL consistency checks passed against Stage 8. Mean reduction in median analysis time 39.7133%; mean speedup 1.6594x. Peak allocated GPU memory unchanged (889.2363 MiB). Cached search ablation 48/48; Fast search speedup 4.2443x (search only).

## Real encoding results

60 configurations, fresh analysis/search for each adaptive run; one measured encode per configuration. Means are unweighted across 12 dataset/context pairs. Positive size penalty indicates larger output relative to Full.

| Method | Mean size penalty % | Max penalty % | Mean encode saving % | Mean speedup | Slowdown vs Fixed |
|---|---:|---:|---:|---:|---:|
| Full | 0 | 0 | 0 | 1.0000 | 6.5203 |
| Reduced Levels | -0.0229 | 0.1195 | 31.6143 | 1.4624 | 4.4589 |
| Reduced Search | 0.8171 | 1.7966 | 4.8198 | 1.0508 | 6.2067 |
| Fast | 0.4051 | 1.6729 | 36.5581 | 1.5765 | 4.1357 |

Independent-corpora-only averages (10 pairs): Reduced Levels size penalty 0.0135%, encode saving 31.7026%; Fast size penalty 0.3598%, encode saving 36.5751%. All 48 fresh adaptive layouts matched cached screening. Reduced Levels is the conservative compression-preserving variant; Fast trades a small average size increase for further encoding savings. The original 50% encoding reduction target was not reached.

## Lossless audit

60/60 verified: 36 matched Stage 8 artifacts by original and compressed SHA-256, 15 newly decoded distinct artifacts, 9 aliases of those new artifacts. This is not 60 independent fresh decode measurements. All 15 new full decodes and all 24 controlled measurements were token-exact, byte-exact and SHA-256 exact. Unique decode time 1.5907 hours; controlled decode time 1.3900 hours.

## Controlled decoding, target 64

Three repetitions per method/sample, rotated execution order (24 measurements). Each dataset's comparisons are within its own session; Wikipedia Session01, Python Session02. Reduced Search was not separately timed in the controlled panel.

| Dataset | Method | Bytes | Mean decode (s) | SD (s) | CV % | Encode (s) |
|---|---|---:|---:|---:|---:|---:|
| Wikipedia | Fixed | 30262 | 193.4654 | 1.5958 | 0.8248 | 3.3230 |
| Wikipedia | Full | 30340 | 185.5115 | 0.6253 | 0.3371 | 22.2729 |
| Wikipedia | Reduced Levels | 30340 | 185.0709 | 0.1465 | 0.0791 | 15.1358 |
| Wikipedia | Fast | 30697 | 169.6608 | 0.0733 | 0.0432 | 13.6179 |
| Python | Fixed | 29802 | 238.7796 | 0.7597 | 0.3182 | 4.9422 |
| Python | Full | 29504 | 237.3931 | 0.2690 | 0.1133 | 31.2043 |
| Python | Reduced Levels | 29504 | 237.4959 | 0.8124 | 0.3421 | 21.2998 |
| Python | Fast | 29931 | 220.6055 | 0.3081 | 0.1397 | 19.7864 |

Fast mean paired decode saving across the two samples: 9.9553% versus Fixed and 7.8076% versus Full. Positive versus both baselines in all six paired repetitions. Size penalties versus Full: 1.1767% and 1.4473%; estimated single-decode total-time savings versus Fixed: 6.8651% and 1.3662%. Python encode and decode measurements came from different sessions, so summed total times are estimates, not synchronized end-to-end measurements. Adaptive decoding allocated 767.9121 MiB versus Fixed 639.7500 MiB (+20.0332%). Full and Reduced Levels produced identical artifacts on both controlled samples; observed tiny decode differences are timing variability, not evidence of a different decode algorithm.

## Evidence archive and next work

`stage9_final_evidence.zip`: results, 12 PNG and 12 PDF plots, original source snapshots, both notebook histories, restored benchmark functions and frozen references. `stage9_final_artifacts.zip`: 60 compressed artifacts, 15 unique decoded outputs, six source samples and token arrays, fresh analysis caches, selected layouts and decode queue. Keep both Stage 8 archives for inherited verification. Key files: `stage9_final_encode_summary.csv`, `stage9_controlled_summary.csv`, `stage9_controlled_paired_runs.csv`, `stage9_controlled_pair_comparison.csv`, `stage9_scope_aggregate.csv`, `stage9_metadata.json`, `artifact_audit.json`.

Limitations: one end-to-end encoding observation per configuration; repeated decoding only two samples at target 64; no statistical significance claim from three repetitions; no generalization claim from reused corpus samples; no blanket superiority over conventional codecs. Next priorities: independent larger samples, comparable traditional/neural baselines, adapter accounting and ablation, and repeated timing of the final chosen policies. Preserve negative results and report actual artifact sizes including container metadata.
