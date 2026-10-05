# Validation on real public data

Both final standalone notebooks executed every code cell in order on **5 October 2026**, using real attached-format public data. The local runner overrides only the input/output folders. Synthetic fixtures are used only in regression tests, never for reported results.

| Pipeline | Data processed | Complete notebook wall time, local CPU |
|---|---|---|
| Delhi EBM classification | 393,440 raw records; 307,441 complete examples; 215,196 training rows; 46,117 final-test rows | 71.64 seconds |
| Three-commodity GMM clustering | 14,437,080 raw rows scanned once; 1,829,255 retained unique records; 1,750 eligible profiles | 70.09 seconds |

The sum of these wall times is approximately **2 minutes 22 seconds**. The two runs overlapped locally; this sum is not a benchmark of sequential execution. Files and dependencies were already present. Timings include imports, charts and report/ZIP exports, but exclude data download, session startup and package installation. Allow roughly **2–5 minutes per notebook on Kaggle**, with attached data; actual CPU/storage load varies. T4 selection does not accelerate these CPU models.

An initial local EBM attempt encountered a joblib worker/psutil process-discovery failure. The final EBM uses `n_jobs=1` with a single bag, avoiding unnecessary child-process creation. The final complete run passed with that configuration.

## Classification findings

| Model | Final-test F1 | Sensitivity | Specificity |
|---|---|---|---|
| EBM: main model | 0.9635 | 0.9607 | 0.9809 |
| Logistic Regression: demand/calendar | 0.9658 | 0.9677 | 0.9793 |
| Logistic Regression: weather enhanced | 0.9648 | 0.9688 | 0.9775 |
| Persistence | 0.9409 | 0.8931 | 0.9970 |

EBM improves F1 over persistence but **does not improve F1 over either LR baseline** on this test period. Its justification is readable nonlinear effects and prespecified interactions, not a claim of universally superior performance. It was specified as the main model before test evaluation.

When current demand is below the peak threshold, EBM F1 is **0.6897**, sensitivity **0.6801** and specificity **0.9821**. This is the more demanding advance-warning subset; overall scores partly reflect persistent already-high demand. Three historical expanding-window EBM checks passed, with F1 approximately 0.9495, 0.9331 and 0.9655.

The data contain 21,569 missing five-minute demand slots; these are not interpolated. Alert cutoffs use chronological validation only; peak definitions and feature fitting use training data only. One-hour split embargo assertions passed. Balanced scores are not demonstrated calibrated probabilities. Learned effects and temperature scenarios are associations/sensitivities, not causal evidence. Demand units, an operational peak limit and historical weather availability still need confirmation for deployment.

## Clustering findings

All selected main GMMs converged, satisfied the minimum hard-group size and used full covariance. Each commodity has separately fitted clipping/scaling/model parameters.

| Commodity | Retained daily records | Eligible profiles | Components | Hard-label silhouette | Mean converged subsampling ARI | Membership below 0.65 |
|---|---|---|---|---|---|---|
| Onion | 622,885 | 606 | 6 | 0.0707 | 0.6524 | 8.75% |
| Potato | 621,863 | 576 | 4 | 0.1141 | 0.6875 | 7.29% |
| Tomato | 584,507 | 568 | 4 | 0.1382 | 0.5561 | 10.74% |

These are **overlapping descriptive behavior groups**, not strongly separated or economically verified categories. BIC selects probabilistic fits; the relatively low hard-label silhouettes and moderate ARI should be presented honestly. GMM membership confidence measures fitted overlap, not future price risk.

Outlier treatment matters. ARI relative to the main 1% clipping fit:

| Commodity | No clipping | 2% clipping | Add log mean price |
|---|---|---|---|
| Onion | 0.3874 | 0.6890 | 0.4745 |
| Potato | 0.4315 | 0.6741 | 0.5349 |
| Tomato | 0.7548 | 0.3805 | 0.6189 |

Unclipped onion/potato solutions create hard groups with only two profiles and do not satisfy the main selection constraint; they remain explicit diagnostic alternatives. Group descriptions use medians, and extreme profiles are exported for review. There were 3,725 known inconsistent-price rows removed across the three commodities. No commodity was silently skipped.

Classification accuracy/specificity cannot be computed meaningfully for unlabeled clustering. Commodity group IDs and BIC are not cross-commodity rankings. Calendar coverage includes non-trading days; quantities, transport costs and farmer-level selling prices are not modeled.

## Verification

- Eight regression tests passed: specificity; future-feature leakage; gap-crossing target exclusion; price consistency/consecutive-day returns; commodity-preserving aggregation; GMM posterior normalization/BIC selection; single notebook section display/no repository runtime dependency; embedded-source synchronization/compilation.
- Both notebooks passed nbformat schema validation and complete real-data execution.
- Classification split embargo assertions passed; all three historical backtests executed.
- Each notebook has sections 1–5 exactly once in its Markdown cells. Report-writing helpers export the full report without displaying it again; the inference cell displays only the inference text. No implementation-link section or platform fallback remains.
- Generated HTML chart paths resolve, including nested commodity folders.
- Representative EBM learned effects and GMM selection/projection charts were visually checked.

Selected current-run CSVs, PNGs and reports under `validation/` replace the earlier LR/onion-only examples. Dataset files and model binaries are not committed. Kaggle Run All regenerates complete outputs from its attached datasets.
