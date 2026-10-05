# Validation on real public data

The standalone three-commodity K-means notebook executed every code cell in order on **5 October 2026**, using the full real 2022–2024 files. Classification uses the unchanged EBM model source validated earlier the same day. The local notebook runner overrides only input/output folders. Synthetic fixtures are used only in regression tests, never for reported results.

| Pipeline | Data processed | Complete notebook wall time, local CPU |
|---|---|---|
| Delhi EBM classification | 393,440 raw records; 307,441 complete examples; 215,196 training rows; 46,117 final-test rows | 71.64 seconds |
| Three-commodity K-means clustering | 14,437,080 raw rows scanned once; 1,829,255 retained unique records; 1,750 eligible profiles | 58.69 seconds |

These are local CPU observations with files/dependencies already available, not Kaggle benchmarks. Times include imports, charts and report/ZIP exports but exclude session startup and package installation. Allow roughly **2–5 minutes per notebook on Kaggle** with attached inputs; storage/CPU load varies. T4 selection does not accelerate these CPU implementations.

The EBM uses `n_jobs=1` with one bag to avoid unnecessary child-process creation. Its model, preprocessing and evaluation are unchanged by the clustering update.

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

Each commodity has separately fitted clipping/scaling/K-means parameters. Cluster count is selected from 2–6 by highest silhouette among fits meeting minimum group size and iteration checks. Supporting metrics are Davies–Bouldin (lower is better), Calinski–Harabasz (higher is better) and inertia/elbow diagnostics.

| Commodity | Retained daily records | Eligible profiles | Groups | Silhouette | Davies–Bouldin | Mean subsampling ARI |
|---|---|---|---|---|---|---|
| Onion | 622,885 | 606 | 2 | 0.4408 | 1.1361 | 0.8930 |
| Potato | 621,863 | 576 | 2 | 0.3834 | 1.2769 | 0.9883 |
| Tomato | 584,507 | 568 | 2 | 0.3678 | 1.2195 | 0.9421 |

Groups are descriptive price-behavior categories, not forecasts or verified economic labels. Per-profile silhouette flags weak separation without representing a probability. Median behavior summaries, representatives, reporting/price flags and geometric diagnostics remain available for interpretation.

Outlier treatment matters. ARI relative to the main 1% clipping fit:

| Commodity | No clipping | 2% clipping | Add log mean price |
|---|---|---|---|
| Onion | 0.7083 | 0.8535 | 0.9243 |
| Potato | -0.0459 | 0.9852 | 0.8523 |
| Tomato | 0.8422 | 0.9926 | 0.8451 |

Low ARI reveals sensitivity, not a reason to hide the alternative result. Sensitivity fits retain the selected group count and report their sizes and iteration checks even when an alternative creates tiny groups. Extreme profiles are exported for review. There were 3,725 known inconsistent-price rows removed across the three commodities. No requested commodity was silently skipped.

Classification accuracy/specificity cannot be interpreted for unlabeled clustering. Group IDs are local to each commodity. Calendar coverage includes non-trading days; quantities, transport costs and farmer-level selling prices are not modeled.

## Preserved corrections

- Onion, potato and tomato are filtered in one projected/batched scan, then modeled independently. Commodity remains in aggregation keys, preventing mixed-price profiles.
- Positive/consistent price checks, duplicates and daily medians; shared 2022–2024 window, annual/monthly coverage, minimum consecutive-day returns; stale prices, extreme changes and dominant-variety switches remain checked.
- Standardization, documented 1% clipping, no-clipping/2% sensitivity, absolute-price exclusion and price-level sensitivity remain intact.
- Multiple initializations, seed checks and 80% subsampling ARI; median summaries, representative histories, PCA and heatmaps; drop/coverage what-if tables and stage runtimes remain intact.
- Weak-separation diagnostics use per-profile silhouette, not model probabilities. Calculations are capped above 5,000 profiles to avoid unbounded pairwise work; aggregate silhouette remains sampled for larger datasets.
- Both notebooks remain self-contained for Kaggle Add Input → Run All. Numbered sections 1–5 display once; full reports are exported without redisplay. No repository runtime dependency or implementation-link section exists.

## Verification

- Eight regression tests passed: specificity; future-feature leakage; gap-crossing target exclusion; price consistency/consecutive-day returns; commodity-preserving aggregation; K-means selection/minimum-size constraints; notebook section counts and runtime independence; embedded-source synchronization/compilation.
- Both notebooks passed nbformat schema validation; the complete updated clustering notebook passed on all real annual files.
- The unchanged classification source retains its earlier real-data execution, split embargo checks and three historical backtests.
- Generated HTML chart paths resolve, including all nested commodity folders. ZIP and saved model bundles were checked.
- Representative cluster-selection and projection charts were visually inspected.

Selected current-run tables, PNGs and reports are stored under `validation/`. Dataset files and saved model binaries are not committed. Kaggle Run All regenerates complete outputs from attached inputs.
