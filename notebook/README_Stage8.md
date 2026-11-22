# Stage 8 — Multi-dataset Benchmark

## Status

Completed: 24/24 encodes and full decodes, all token-exact, byte-exact and SHA-256 verified. Source commit: `62f09e3105296d62c39e8d95cf8b8775e63f2677`. Session: Stage8-Session01. Total measured decoding time: 2.6372 hours.

## Experimental design

GPT-2, FP32, SDPA, Tesla T4 (GPU 0), PyTorch 2.10.0+cu128, Transformers 5.0.0, NumPy 2.0.2; TF32 disabled; no adapter. Context candidates: 32, 64, 128, 256, 512; effective batch sizes: 8, 8, 8, 4, 2. Fixed and BC-DP comparisons target contexts 64 and 128. Stage 7 rank table and latency models were frozen prior to testing. The search evaluates 488 proposals per dataset (two objectives × two latency models × 122 lambda values) and selects the smallest estimated variable payload within the predicted fixed-context decode budget. Real compressed files include container overhead. Adaptive encoding time includes analysis and search, charged in full to each configuration.

## Datasets

Each sample contains 100,000 UTF-8 bytes. Mixed is derived from the other five and is not independent. All six tokenizer round trips passed.

| Dataset | Provenance | GPT-2 tokens | Empirical token entropy (bits/token) |
|---|---|---:|---:|
| enwik8_new | enwik8 offset 5,000,000 | 31,776 | 8.9489 |
| english | GITenberg Pride and Prejudice, 1342-0.txt | 27,648 | 8.3727 |
| python_code | Local Python standard-library source | 45,350 | 5.0200 |
| csv_data | seaborn-data diamonds.csv | 62,858 | 5.2752 |
| vietnamese | UD Vietnamese-VTB r2.15, dev split # text fields | 66,557 | 6.3045 |
| mixed | Concatenated portions of the five samples | 48,741 | 7.8854 |

Exact source metadata and hashes are in `results/dataset_manifest.json`. Check upstream data licensing before redistributing samples publicly.

## Measured paired results

Positive percentages indicate reductions relative to fixed context. Single measurement per configuration.

| Dataset | Target | Fixed bytes | BC bytes | Size saving % | Decode saving % | Total time saving % |
|---|---:|---:|---:|---:|---:|---:|
| enwik8_new | 64 | 30262 | 30340 | -0.2577 | 4.9757 | -4.0277 |
| enwik8_new | 128 | 27671 | 27608 | 0.2277 | 2.6156 | -2.1691 |
| english | 64 | 31324 | 31316 | 0.0255 | 2.7749 | -6.0353 |
| english | 128 | 29911 | 29936 | -0.0836 | 3.4627 | -1.2324 |
| python_code | 64 | 29802 | 29504 | 0.9999 | 1.5032 | -7.3373 |
| python_code | 128 | 25934 | 25878 | 0.2159 | 1.9735 | -2.6936 |
| csv_data | 64 | 45419 | 45336 | 0.1827 | 0.2224 | -9.1749 |
| csv_data | 128 | 35724 | 35724 | 0.0000 | -0.4480 | -5.4659 |
| vietnamese | 64 | 58253 | 58232 | 0.0360 | 0.4573 | -8.8496 |
| vietnamese | 128 | 54707 | 54726 | -0.0347 | 0.9435 | -4.0073 |
| mixed | 64 | 42175 | 41904 | 0.6426 | 2.8181 | -6.6979 |
| mixed | 128 | 37450 | 37584 | -0.3578 | 5.6510 | 0.5599 |

## Aggregate results

Across all 12 pairs, mean size reduction 0.1330%, mean observed decode reduction 2.2458%, mean encode slowdown 6.3075x, latency prediction MAPE 5.4343%. Seven pairs reduced size and decode time simultaneously; eleven reduced decode time; only one improved end-to-end time. Among independent corpora, six of ten pairs improved both objectives. Mean reductions for independent corpora: BC-64 size 0.1973%, decode 1.9867%; BC-128 size 0.0651%, decode 1.7095%.

## Interpretation and limitations

- Python BC-64 produced the largest size reduction (~1%). Mixed BC-128 produced the largest observed decode reduction (5.6510%), but increased compressed size.
- CSV BC-128 selected an identical layout and identical artifact to Fixed-128. Its 0.448% slower measurement is evidence of timing variability, not a causal algorithmic difference.
- BC-64 increased decode peak memory by approximately 20.03% on five samples; CSV showed no increase. BC-128 showed no increase on these samples.
- Small timing differences and break-even estimates are exploratory: one measurement per configuration, no variance estimate. Repeat paired timings with counterbalanced order before claiming statistically reliable speed improvements. Avoid interpreting mixed as independent validation.
- The method does not generally improve end-to-end time for a single decode; expensive encoder-side search is the main limitation. No claim of general superiority to conventional compressors is supported by this stage alone.

## Evidence and reproducibility

Keep `stage8_final_evidence.zip` (results, plots, notebook history, frozen reference and source snapshots) separate from `stage8_final_artifacts.zip` (compressed files, decoded files, token and analysis caches, selected layouts, source samples). Preserve exact commit and hashes. In a fresh Kaggle session, clone and check out the specified commit, restore the archives and use the same software/model configuration. Do not overwrite original source files.

Key result files: `stage8_final_summary.csv`, `stage8_pair_comparison.csv`, `stage8_aggregate_summary.csv`, `stage8_metadata.json`, `dataset_manifest.json`, `dataset_tokenization.csv`, `selected_candidates.csv`. Figures 01–10 are supplied in both PNG and PDF: compression, decode, total-time savings, encoding overhead, decode memory, joint trade-off, latency prediction, dataset Pareto plots, dataset characteristics, and context distribution.

## Next research priorities

Repeated timing controls, faster search/encoding ablation, larger independent samples, and fair comparisons with conventional compressors and neural baselines. Results are suitable as preliminary multi-corpus evidence, not a guarantee of journal acceptance.
