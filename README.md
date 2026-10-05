# Big Data Essentials: two Indian applications

This repository stores the project sources and two **standalone Kaggle notebooks**. The notebooks are the final deliverables: each embeds every helper and model definition, reads attached datasets and executes top to bottom with **Run All**. No repository access, clone, companion scripts or automatic dataset download is needed during execution.

| Application | Main model | Notebook | Attach this Kaggle input |
|---|---|---|---|
| Delhi next-hour peak-demand warning | Explainable Boosting Machine (EBM) | [Classification notebook](notebooks/01_delhi_peak_classification.ipynb) | [Delhi five-minute electricity demand](https://www.kaggle.com/datasets/yug201/delhi-5-minute-electricity-demand-for-forecasting) |
| Onion, potato and tomato market behavior | Separate Gaussian Mixture Models (GMM) | [Clustering notebook](notebooks/02_mandi_price_clustering.ipynb) | [Historical daily commodity prices](https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india) |

## Run on Kaggle

1. Import the desired `.ipynb` into Kaggle.
2. Use **Add Input** to attach the dataset listed above. Mandi needs the historical 2022–2024 files, not a single-day snapshot.
3. Enable Internet if a missing dependency must be installed. Classification installs `interpret-core` and `holidays` if needed; clustering checks `pyarrow`. Data are read from attached inputs only.
4. Click **Run All**. Defaults process the complete eligible Delhi history or all three selected commodities.
5. Download results from Kaggle Output; save a notebook version for submission.

T4 sessions work, but these implementations run on **CPU**. GPU selection does not accelerate EBM or scikit-learn GMM. With several compatible demand files attached, use `input_path` to choose one. Mandi prefers equivalent Parquet files over CSV and skips named annual files outside 2022–2024.

## Assignment outputs

Each notebook displays these numbered sections **once**: problem statement; model and justification; coding; results; inference and what-if analysis. The complete Markdown/HTML report is exported without redisplaying those headings. There is no implementation-link section or URL configuration.

Results include quality/volume tables, comparisons, charts, appropriate metrics, actual inferred findings, sensitivity checks, configuration, timings and saved models. A ZIP packages the outputs. The mandi HTML report includes charts and summaries from all three commodity folders. Keep extracted folders together to preserve relative chart paths.

## Classification design

- EBM learns nonlinear additive effects with two explicit interactions: current demand × recent change, and temperature × humidity. It uses 64 main-effect bins, 16 interaction bins, 800 maximum rounds and one bag for bounded computation. Internal random validation/early stopping are disabled; external chronological validation chooses the alert cutoff.
- Balanced training weights; current demand, lags, trends, cyclic time, Indian holidays and available weather features. Missing demand is not interpolated; incomplete targets are excluded. Missing weather is handled natively by EBM. LR/tree baselines use train-fitted imputation/scaling where relevant.
- Chronological 70/15/15 splits with one-hour embargoes, a training-only peak percentile and validation-only alert cutoffs. Logistic Regression, a shallow Decision Tree, persistence and yesterday/last-week baselines remain comparisons.
- F1, sensitivity, specificity, precision, ROC-AUC, average precision, confusion counts, day-block intervals, quarterly results, historical EBM backtests and advance-warning-only evaluation.
- Learned effect plots, term importance, example predictions and false alarms/missed peaks per 1,000 five-minute checks. Alert-cutoff, peak-definition and temperature scenarios show descriptive sensitivity.

The peak percentile is an experimental proxy, not an operator capacity limit. Balanced scores are not demonstrated calibrated probabilities. Weather availability at prediction time and source demand units need confirmation. Correlated lags can share importance; effect plots are associations, not causes. Overall scores partly reflect already-high demand, so advance-warning results matter. EBM is established, not a newly invented algorithm.

## Clustering design

- Scan millions of historical records **once**, retain Onion/Potato/Tomato, then construct and model each commodity independently. Aggregation keys include commodity, preventing mixed prices even when market/variety/grade names match.
- Remove known price inconsistencies and duplicates; aggregate daily median prices. Require shared-window coverage, represented months/years and enough consecutive-day returns. Flag stale reporting, extreme returns and dominant-variety switches.
- Fit standardized behavior features: price CV, consecutive-day return volatility, sharp-drop frequency, monthly variation and relative spread if available. Exclude absolute price level; document 1% tail clipping and compare no clipping/2% clipping.
- Choose 2–6 GMM components with diagonal or full covariance by lowest BIC among converged fits satisfying minimum hard-group sizes. Compare K-means at the same group count.
- Export every component-membership probability and maximum confidence; flag memberships below 0.65. Confidence is fitted overlap, not future loss probability. Report median group behavior to reduce distortion from extremes.
- Show BIC/AIC, hard-assignment silhouette, initialization/subsampling ARI, price-level sensitivity, representative histories, PCA displays and confidence histograms. Drop, confidence and coverage what-if tables are descriptive and do not refit the groups.

GMM fitting uses hundreds of aggregated profiles per commodity, while preprocessing handles millions of raw rows. Report both volumes honestly. Group IDs and BIC are local to each commodity. No true group labels exist, so classification accuracy/specificity do not apply. Monthly variation is not proof of seasonality. Calendar coverage includes non-trading days; logistics, traded quantities and farmer-level prices are absent. Established GMM methods provide the enhancement, not algorithmic novelty.

## Validation and maintenance

[VALIDATION.md](VALIDATION.md) records real-data results, measured runtimes and limitations. Allow roughly **2–5 minutes per notebook** on Kaggle with attached data, excluding setup; see the measured CPU times in the validation report. Example reports/CSVs/charts are under `validation/`; datasets and model binaries are not committed. Runtime tables exclude final ZIP packaging; the validation runner measures full notebook wall time including imports and exports.

The scripts below are for maintaining this storage repository, not prerequisites for Kaggle execution:

```bash
pip install -r requirements.txt
python tools/build_notebooks.py
python -m unittest discover -s tests -v
python tools/run_notebook.py notebooks/01_delhi_peak_classification.ipynb --input-root /path/to/delhi --output-dir /path/to/electricity_results
python tools/run_notebook.py notebooks/02_mandi_price_clustering.ipynb --input-root /path/to/mandi --output-dir /path/to/mandi_results
```

## Method references

- [EBM documentation](https://interpret.ml/docs/ebm.html)
- [EBM classifier API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html)
- [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html)
- [GMM API](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

The mandi collection is a community mirror of government AGMARKNET records. Notebook results are historical analyses, not deployed operational decisions.
