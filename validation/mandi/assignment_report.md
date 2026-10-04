# Clustering: Indian mandi price-behavior groups

## 1. Problem statement
Group Onion market/variety/grade profiles by price variability and sharp-drop behavior over the shared window 2022-01-01 to 2024-12-31. Farmer producer organizations could use these descriptive groups to prioritize market monitoring. This does not identify the most profitable market or predict future prices.

## 2. Model and justification
K-means is fast on a small numeric profile table and produces interpretable centers. Chunked CSV/Parquet processing handles the much larger input history. Standardized behavior features prevent units from dominating distance. The main model excludes absolute price; an ablation checks its influence. Missing spread is median-imputed only when at least 80% of profiles have it; otherwise that feature is omitted. Profile features are winsorized at the 1st/99th percentiles, with bounds exported and a no-winsorization sensitivity check. This limits outlier influence but can conceal extremes. Silhouette and minimum cluster-size checks guide k; stability checks test reproducibility.

## 3. Coding
Only one commodity and a common date window are retained. CSV files are read in chunks and Parquet in batches; equivalent CSV/Parquet filenames are deduplicated. Positive prices and known minimum ≤ modal ≤ maximum are required. Exact duplicates are removed; multiple same-day prices are summarized by medians and conflicts reported. Profiles retain separate varieties and grades. Consecutive-calendar-day returns avoid confusing multi-day changes with daily changes. Coverage, year representation and valid-return eligibility are enforced. Stale prices, large changes and annual dominant-variety switches are flagged, not automatically declared wrong.

## 4. Results
Raw rows scanned: 14,437,080; retained unique raw rows: 622,885; eligible profiles clustered: 606. Selected k: 2; sampled silhouette: 0.438. Mean subsampling ARI: 0.887. Minimum cluster-size constraint relaxed: False. Consult exported selection, cluster summaries, representative histories, stability and quality tables. Classification accuracy, specificity and confusion matrices do not apply because no true cluster labels exist.

## 5. Inference and what-if analysis
Cluster 0 contains 157 profiles: mean relative price variability 0.556, daily return volatility 0.347, and sharp-drop frequency 13.7%.
Cluster 1 contains 449 profiles: mean relative price variability 0.413, daily return volatility 0.088, and sharp-drop frequency 5.6%.
Higher volatility/drop-frequency groups warrant closer monitoring; stable histories do not guarantee future stability. Subsampling ARI near 1 indicates reproducible assignments under fixed preprocessing; low values suggest weak groups. Removing profile winsorization changes assignments with ARI 0.012; values far below 1 indicate strong sensitivity to extreme-profile treatment. Inspect extreme_profiles_for_review and the smallest unclipped cluster; a high silhouette from an isolated outlier is not evidence of useful groups. Adding price level changes assignments with ARI 0.938; this tests whether price level dominates grouping. What-if tables show event prevalence at 5%, 10% and 15% daily-drop definitions and the loss of eligible profiles under stricter coverage. These tables do not refit clusters. Monthly variation is a proxy, not proof of seasonality. Calendar coverage includes non-trading days. Missing markets, reporting biases, variety recoding and genuine extreme movements can affect interpretation. Transport costs, traded quantities and farmer-level selling prices are not modeled.

## 6. URL of implementation
https://colab.research.google.com/github/dhruvvvgg/bde-assignment/blob/main/notebooks/02_mandi_price_clustering.ipynb
This Colab URL opens the GitHub notebook. For a Kaggle submission, replace implementation_url with the saved Kaggle notebook URL and run again.