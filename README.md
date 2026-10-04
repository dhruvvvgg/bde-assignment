# Big Data Essentials: two practical Indian applications

Two standalone Python notebooks, each designed to execute from top to bottom with **Run All**. All model code is embedded in each notebook. The files in `src/` are the maintainable source used to generate the notebooks; downloading them is not necessary to run a notebook.

| Problem | Notebook | Open in Colab | Kaggle input |
|---|---|---|---|
| Delhi next-hour peak-demand classification | [01_delhi_peak_classification.ipynb](notebooks/01_delhi_peak_classification.ipynb) | [Run classification](https://colab.research.google.com/github/dhruvvvgg/bde-assignment/blob/main/notebooks/01_delhi_peak_classification.ipynb) | [Delhi five-minute demand with weather](https://www.kaggle.com/datasets/yug201/delhi-5-minute-electricity-demand-for-forecasting) |
| Indian mandi price-behavior clustering | [02_mandi_price_clustering.ipynb](notebooks/02_mandi_price_clustering.ipynb) | [Run clustering](https://colab.research.google.com/github/dhruvvvgg/bde-assignment/blob/main/notebooks/02_mandi_price_clustering.ipynb) | [Historical daily commodity prices, 2001–2026](https://www.kaggle.com/datasets/khandelwalmanas/daily-commodity-prices-india) |

## Kaggle: Run All

1. Import one notebook into Kaggle.
2. Choose **Add Input** and attach the corresponding dataset above.
3. Keep the default configuration. The mandi notebook selects Onion and a shared 2022–2024 window. It prefers Parquet over equivalent CSV files and skips named year files outside the window.
4. Click **Run All**. No manual cell edits or companion script are needed with the specified inputs.
5. Save a Kaggle notebook version. Put that actual URL into `CONFIG['implementation_url']`, then run again before submission. The default Colab links are valid implementation links to these GitHub notebooks, not invented Kaggle URLs.
6. Download the generated results ZIP from the Output pane.

T4 sessions are compatible, but these scikit-learn models run on **CPU**. Selecting GPU will not accelerate them. Notebook Internet is only required when an optional dependency is missing. For an entirely offline run, dependencies must already be installed and data attached.

Colab downloads public datasets automatically through kagglehub when `/kaggle/input` is absent. The full historical mandi download is about 1.51 GB compressed; downloading all its files can take much longer than the model computation. Kaggle attached data avoids this download step. To use predownloaded files, set `input_root` before running.

## Outputs mapped to the assignment rubric

The screenshot assigns six half-mark items per problem, plus presentation marks. Each notebook explicitly includes:

| Required item | Implementation |
|---|---|
| Problem statement | India-specific decision, prediction/grouping unit, and scope |
| Model and justification | Why the simple model fits, preprocessing assumptions, comparisons and limitations |
| Coding | Embedded implementation and a single pipeline invocation |
| Results | Data-quality tables, model/cluster comparisons, metrics and saved charts |
| Inference | Actual computed findings, what-if analysis and limitations |
| Implementation URL | Colab link; configurable actual Kaggle link |

Both create `assignment_report.md`, an HTML report, CSV tables, PNG charts, environment/config metadata, saved model bundles, and a ZIP. HTML charts use relative image paths, so keep the unzipped bundle together. Runtime is reported by stage; the pipeline's total excludes final report/ZIP packaging. The validation runner additionally measures complete notebook wall time.

### Classification enhancements

- Current demand, short lags/trends, yesterday and last-week lags; cyclical time features, weekends and Indian holidays.
- Current temperature, humidity, dew point, wind speed, pressure and a temperature–humidity interaction where available. Weather timestamp availability must be checked before deployment.
- Demand/calendar Logistic Regression, weather-enhanced Logistic Regression and a shallow Decision Tree.
- Persistence, same-hour-yesterday and same-hour-last-week baselines.
- Chronological 70/15/15 splits, one-hour target embargoes, training-only preprocessing and peak threshold, validation-only alert cutoffs.
- Demand gaps are not interpolated. Unusual demand is flagged for review using training IQR fences and sharp jumps; genuine peaks are retained.
- Accuracy, balanced accuracy, precision, sensitivity, **specificity**, F1, ROC-AUC, average precision and confusion counts.
- Advance-warning-only evaluation when current demand is below the threshold; quarterly test evaluation and three historical rolling backtests.
- Day-block approximate confidence intervals, coefficient interpretation, alert-cutoff tradeoffs, alternative peak definitions and conditional temperature scenarios.

The class-weighted model produces alert scores, not demonstrated calibrated probabilities. The percentile peak definition is an experimental proxy, not an electricity-system capacity limit. Historical weather and current-bin readings are assumed to have arrived by prediction time. Demand units remain the source's native units pending confirmation.

### Clustering enhancements

- Projected CSV chunks and Parquet batches; one commodity filtered before retaining rows.
- Common observation window, reporting coverage, represented months/year checks and minimum consecutive-day returns.
- Positive-price and known minimum ≤ modal ≤ maximum consistency checks; duplicate removal and daily median aggregation.
- Profiles keyed by state/district/market/variety/grade. Stale reporting, extreme changes and annual dominant-variety switches are flagged.
- Price CV, consecutive-day return volatility, sharp-drop frequency, monthly variation proxy and relative price spread when sufficiently available.
- Absolute price excluded from the main model; price-level ablation provided.
- Standardization, documented profile winsorization and a no-winsorization sensitivity check with extreme profiles listed for review.
- k comparison using silhouette, Davies–Bouldin and Calinski–Harabasz; minimum cluster-size constraint, seed checks and 80% subsampling ARI.
- Cluster summaries, representatives, normalized price histories, PCA display, heatmap and coverage/drop-definition what-if tables.

These are descriptive clusters, not forecasts or profit recommendations. No ground-truth clusters exist, so accuracy/specificity/confusion matrices are inappropriate. Calendar coverage includes non-trading days. Monthly variation is not proof of seasonality. ARI checks hold preprocessing fixed and do not establish economic validity. Logistics, traded quantities and farmer-level selling prices are not modeled.

## Validation and runtime

Both notebook pipelines were executed on the actual public datasets. See [VALIDATION.md](VALIDATION.md) for measured times, real results and limitations. These are local CPU measurements, not T4 benchmarks. Plan approximately **1–3 minutes per notebook on Kaggle with inputs attached**, excluding session startup and any dependency installation; actual performance varies.

Five focused regression tests cover specificity, future-feature leakage, incomplete target windows, price consistency/consecutive-day returns, and notebook/source synchronization. Assertions inside the real classification run verify split embargoes. Both notebooks also pass nbformat schema validation.

```bash
pip install -r requirements.txt
python tools/build_notebooks.py
python -m unittest discover -s tests -v
python tools/run_notebook.py notebooks/01_delhi_peak_classification.ipynb --input-root /path/to/delhi --output-dir /path/to/electricity_results
python tools/run_notebook.py notebooks/02_mandi_price_clustering.ipynb --input-root /path/to/mandi --output-dir /path/to/mandi_results
```

## References

- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
- [Indian onion-market quality-check reference](https://github.com/MrVinamra/Mandi-Price-Cointegration-VECM)
- [Delhi forecasting baseline reference](https://github.com/pyaf/load_forecasting)

The pipeline implementation is original to this assignment; the references inform methods and quality checks. The Kaggle mandi collection is a community mirror of government AGMARKNET data. Dataset files are not committed.
