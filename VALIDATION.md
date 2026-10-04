# Validation on real public data

Both standalone notebooks executed all code cells in order on 4 October 2026. No synthetic records were used for reported model results. The command-line notebook runner only overrides input/output locations for the local environment.

| Pipeline | Data processed | Complete notebook wall time on local CPU |
|---|---|---|
| Delhi classification | 393,440 raw records; 307,441 complete examples; 46,117 final-test observations | 16.39 seconds |
| Mandi clustering | 14,437,080 raw rows across 2022–2024 Parquet files; 622,885 valid unique onion rows; 606 eligible profiles | 22.22 seconds |

The total measured computation/execution was approximately 39 seconds. Files were already downloaded. These timings include imports, charts, model artifacts and report/ZIP packaging, but exclude dataset download and notebook-session startup. The runs overlapped locally; these are observations, not hardware-normalized benchmarks. Kaggle CPU, storage, software and workload differ. Allow roughly 1–3 minutes per notebook with data already attached; a T4 does not accelerate these CPU models.

## Classification findings

Weather-enhanced Logistic Regression: test F1 0.9648, sensitivity 0.9688, specificity 0.9775. Persistence: F1 0.9409. On observations currently below the peak threshold, enhanced-model F1 is 0.7030 and sensitivity is 0.7197.

Demand/calendar-only Logistic Regression achieved slightly better test F1 (0.9658). Weather improved validation F1 slightly but did not improve final-test F1. The result should not be described as a demonstrated benefit of adding weather. The enhanced model was specified before inspecting the test results. Alert cutoffs use validation data only.

There are 21,569 missing five-minute demand slots. They are not interpolated. The classifier is a historical experiment with a percentile-based peak proxy; operational threshold, data availability and source units need confirmation. High overall scores partly reflect persistence of already-high demand, which is why advance-warning-only results are also reported.

## Clustering findings

K-means chose two behavior groups: 157 profiles with greater daily volatility/drop frequency and 449 with lower values. Silhouette is 0.4379. Mean ARI under 80% subsampling is 0.8874. Adding log mean price yields ARI 0.9380 relative to behavior-only grouping.

Important sensitivity: removing profile winsorization produces ARI 0.0123 relative to the main grouping. Extreme profiles dominate that alternative solution; its high silhouette is not evidence of useful market groups. The notebook exports extreme profiles, unclipped cluster sizes and this sensitivity result. The main groups should be presented as dependent on the documented outlier treatment, not universally robust economic categories.

Classification specificity and accuracy do not apply to this unlabeled clustering task. Its metrics describe geometric separation and reproducibility, not confirmed economic ground truth.

## Verification

- Five regression tests passed: specificity calculation, future-feature leakage, gap-crossing target exclusion, inconsistent mandi price removal/consecutive-day returns, and notebook-source synchronization/compilation.
- Both notebooks passed nbformat schema validation.
- The complete notebook pipelines passed on the real Delhi CSV and real 2022, 2023 and 2024 mandi Parquet files.
- Classification split embargo assertions passed.
- Representative classification and clustering charts were visually inspected.

The CSVs, charts and generated assignment reports under `validation/` are example outputs from these runs. Data/model binaries are not committed. Run All regenerates your own outputs using the attached inputs and your CONFIG.
