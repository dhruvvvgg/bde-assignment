from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import pyarrow.parquet as pq


MARKET_KEYS = ['state', 'district', 'market', 'variety', 'grade']
MANDI_ALIASES = {'state': ['state', 'state name'], 'district': ['district', 'district name'], 'market': ['market', 'market name', 'mandi'], 'commodity': ['commodity', 'commodity name'], 'date': ['arrival date', 'price date', 'date', 'reported date'], 'price': ['modal price', 'modalprice'], 'minimum': ['min price', 'minimum price'], 'maximum': ['max price', 'maximum price'], 'variety': ['variety'], 'grade': ['grade']}


def stream_mandi(paths, cfg, out):
    cache = out / 'filtered_batches'
    cache.mkdir(parents=True, exist_ok=True)
    for p in cache.glob('part_*.parquet'):
        p.unlink()
    manifest, parts = [], []
    start, end = pd.Timestamp(cfg['start_date']), pd.Timestamp(cfg['end_date'])
    for path in paths:
        stats = {'file': str(path), 'bytes': path.stat().st_size, 'raw_rows_scanned': 0, 'commodity_rows': 0, 'window_rows': 0, 'invalid_rows': 0, 'inconsistent_price_rows': 0, 'exact_duplicate_rows_in_batches': 0, 'valid_rows': 0}
        columns = pq.ParquetFile(path).schema.names if path.suffix.lower() == '.parquet' else pd.read_csv(path, nrows=0).columns
        mapping = {name: find_column(columns, aliases, name not in ['variety', 'grade', 'minimum', 'maximum']) for name, aliases in MANDI_ALIASES.items()}
        usecols = [c for c in mapping.values() if c]
        if path.suffix.lower() == '.parquet':
            chunks = (batch.to_pandas() for batch in pq.ParquetFile(path).iter_batches(batch_size=cfg['chunksize'], columns=usecols))
        else:
            chunks = pd.read_csv(path, usecols=usecols, chunksize=cfg['chunksize'], dtype=str)
        for chunk in chunks:
            stats['raw_rows_scanned'] += len(chunk)
            selected = chunk.loc[chunk[mapping['commodity']].astype(str).str.strip().str.casefold().eq(cfg['commodity'].casefold())]
            stats['commodity_rows'] += len(selected)
            if selected.empty:
                continue
            part = pd.DataFrame({k: selected[v] for k, v in mapping.items() if v})
            part['date'] = pd.to_datetime(part.date, format='mixed', dayfirst=True, errors='coerce').dt.normalize()
            part = part.loc[part.date.between(start, end)].copy()
            stats['window_rows'] += len(part)
            if part.empty:
                continue
            for key in MARKET_KEYS:
                if key not in part:
                    part[key] = 'unspecified'
                part[key] = part[key].astype('string').str.strip().str.casefold()
                if key in ['variety', 'grade']:
                    part[key] = part[key].fillna('unspecified').replace('', 'unspecified')
            part['price'] = number(part.price)
            bad = part[['state', 'district', 'market', 'price']].isna().any(axis=1) | part.price.le(0)
            for key in ['state', 'district', 'market']:
                bad |= part[key].eq('').fillna(True)
            stats['invalid_rows'] += int(bad.sum())
            part = part.loc[~bad].copy()
            part['relative_spread'] = np.nan
            if mapping['minimum'] and mapping['maximum']:
                low, high = number(part.minimum), number(part.maximum)
                known = low.notna() & high.notna()
                inconsistent = known & ((low <= 0) | (high < low) | (part.price < low) | (part.price > high))
                stats['inconsistent_price_rows'] += int(inconsistent.sum())
                part.loc[known & ~inconsistent, 'relative_spread'] = (high[known & ~inconsistent] - low[known & ~inconsistent]) / part.loc[known & ~inconsistent, 'price']
                part = part.loc[~inconsistent]
            part = part[MARKET_KEYS + ['date', 'price', 'relative_spread']]
            before = len(part)
            part = part.drop_duplicates()
            stats['exact_duplicate_rows_in_batches'] += before - len(part)
            stats['valid_rows'] += len(part)
            if len(part):
                target = cache / f'part_{len(parts):06d}.parquet'
                part.to_parquet(target, index=False)
                parts.append(target)
        manifest.append(stats)
        print(f'{path.name}: scanned {stats["raw_rows_scanned"]:,}; retained {stats["valid_rows"]:,}')
    if not parts:
        raise ValueError('No valid commodity records in the common window. Check commodity spelling and attach the historical dataset.')
    data = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates()
    grouped = data.groupby(MARKET_KEYS + ['date'], observed=True)
    daily = grouped.agg(price=('price', 'median'), relative_spread=('relative_spread', 'median'), daily_reports=('price', 'size'), distinct_reported_prices=('price', 'nunique')).reset_index().sort_values(MARKET_KEYS + ['date'])
    counts = data.groupby(['state', 'district', 'market', data.date.dt.year, 'variety'], observed=True).size().rename('rows').reset_index()
    dominant = counts.sort_values('rows', ascending=False).drop_duplicates(['state', 'district', 'market', 'date'])
    dominance = dominant.groupby(['state', 'district', 'market']).agg(annual_dominant_varieties=('variety', 'nunique')).reset_index()
    dominance['dominant_variety_switch'] = dominance.annual_dominant_varieties.gt(1)
    daily = daily.merge(dominance, on=['state', 'district', 'market'], how='left')
    for p in parts:
        p.unlink()
    cache.rmdir()
    return daily, pd.DataFrame(manifest), {'cross_batch_duplicate_rows_removed': duplicates, 'retained_unique_raw_rows': len(data), 'unique_daily_profiles': len(daily), 'days_with_conflicting_reported_prices': int(daily.distinct_reported_prices.gt(1).sum())}


def make_profiles(daily, cfg):
    start, end = pd.Timestamp(cfg['start_date']), pd.Timestamp(cfg['end_date'])
    calendar_days = (end - start).days + 1
    years = list(range(start.year, end.year + 1))
    eligible, excluded = [], []
    for identity, group in daily.groupby(MARKET_KEYS, sort=False, observed=True):
        group = group.sort_values('date')
        price = group.price
        gaps = group.date.diff().dt.days
        returns = price.pct_change(fill_method=None).where(gaps.eq(1))
        annual = group.groupby(group.date.dt.year).size().reindex(years, fill_value=0)
        monthly = group.groupby(group.date.dt.to_period('M')).price.mean()
        same = price.eq(price.shift()) & gaps.eq(1)
        run_ids = (~same).cumsum()
        run_positions = group.groupby(run_ids).cumcount() + 1
        stale = run_positions.ge(3)
        row = dict(zip(MARKET_KEYS, identity))
        row.update({'observations': len(group), 'calendar_coverage': len(group) / calendar_days, 'months_observed': len(monthly), 'min_year_observations': annual.min(), 'valid_daily_returns': int(returns.notna().sum()), 'stale_price_fraction': float(stale.mean()), 'large_daily_change_fraction': float(returns.abs().gt(1).sum() / max(returns.notna().sum(), 1)), 'dominant_variety_switch': bool(group.dominant_variety_switch.any()), 'mean_price': price.mean(), 'log_mean_price': np.log1p(price.mean()), 'price_cv': price.std() / price.mean(), 'return_volatility': returns.std(), 'sharp_drop_fraction': returns.dropna().le(-cfg['drop_fraction']).mean(), 'monthly_variation_cv': monthly.std(ddof=0) / monthly.mean(), 'median_relative_spread': group.relative_spread.median() if group.relative_spread.notna().any() else np.nan})
        reasons = []
        if len(group) < cfg['min_observations']:
            reasons.append('few observations')
        if row['calendar_coverage'] < cfg['min_calendar_coverage']:
            reasons.append('low calendar coverage')
        if len(monthly) < cfg['min_months']:
            reasons.append('few represented months')
        if annual.min() < cfg['min_year_observations']:
            reasons.append('uneven year coverage')
        if row['valid_daily_returns'] < cfg['min_daily_returns']:
            reasons.append('few consecutive-day changes')
        row['eligibility_reason'] = '; '.join(reasons) if reasons else 'eligible'
        (excluded if reasons else eligible).append(row)
    profiles = pd.DataFrame(eligible)
    if len(profiles) < 10:
        raise ValueError(f'Only {len(profiles)} eligible profiles. Add the full 2022–2024 history or review coverage settings; do not silently relax eligibility.')
    return profiles, pd.DataFrame(excluded)


def safe_silhouette(x, labels, limit=2000):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return np.nan
    rng = np.random.default_rng(42)
    ids = np.arange(len(labels))
    if len(ids) > limit:
        selected = []
        for cluster in np.unique(labels):
            group = ids[labels == cluster]
            selected.extend(rng.choice(group, min(2, len(group)), replace=False))
        remaining = np.setdiff1d(ids, selected)
        selected.extend(rng.choice(remaining, min(limit - len(selected), len(remaining)), replace=False))
        ids = np.asarray(selected)
    return float(silhouette_score(x[ids], np.asarray(labels)[ids]))


def cluster_profiles(profiles, cfg):
    features = ['price_cv', 'return_volatility', 'sharp_drop_fraction', 'monthly_variation_cv']
    spread_missing = profiles.median_relative_spread.isna().mean()
    if spread_missing <= .20:
        features.append('median_relative_spread')
    values = profiles[features].replace([np.inf, -np.inf], np.nan)
    bounds = pd.DataFrame({'feature': features, 'lower_1pct': values.quantile(.01).values, 'upper_99pct': values.quantile(.99).values})
    clipped = values.clip(values.quantile(.01), values.quantile(.99), axis=1)
    preprocessing = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    x = preprocessing.fit_transform(clipped)
    scores, models = [], {}
    unique = len(np.unique(x, axis=0))
    min_size = max(3, int(np.ceil(len(profiles) * cfg['min_cluster_fraction'])))
    with threadpool_limits(limits=4):
        for k in range(2, min(cfg['max_clusters'], len(x) - 1, unique) + 1):
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(x)
            counts = np.bincount(labels, minlength=k)
            sil = safe_silhouette(x, labels)
            if not np.isfinite(sil):
                continue
            models[k] = model
            scores.append({'k': k, 'silhouette_sampled': sil, 'davies_bouldin': davies_bouldin_score(x, labels), 'calinski_harabasz': calinski_harabasz_score(x, labels), 'inertia': model.inertia_, 'smallest_cluster': counts.min(), 'meets_minimum_cluster_size': counts.min() >= min_size})
    selection = pd.DataFrame(scores)
    if selection.empty:
        raise ValueError('No meaningful cluster solution: profiles may have insufficient variation.')
    feasible = selection.loc[selection.meets_minimum_cluster_size]
    fallback = feasible.empty
    winner = (selection if fallback else feasible).sort_values(['silhouette_sampled', 'k'], ascending=[False, True]).iloc[0]
    model = models[int(winner.k)]
    return model, preprocessing, x, features, bounds, selection, fallback, float(spread_missing)


def run_mandi(cfg):
    start = time.perf_counter()
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    with timed('stream_filter_and_daily_aggregation', timings):
        paths = discover(cfg, 'mandi')
        daily, manifest, audit = stream_mandi(paths, cfg, out)
    manifest = table(manifest, 'file_manifest', out)
    daily.to_parquet(out / 'clean_daily_prices.parquet', index=False)
    with timed('profile_features', timings):
        profiles, excluded = make_profiles(daily, cfg)
    excluded.to_csv(out / 'excluded_profiles.csv', index=False)
    with timed('cluster_selection', timings):
        model, preprocessing, x, features, bounds, selection, fallback, missing = cluster_profiles(profiles, cfg)
    profiles['cluster'] = model.labels_
    profiles.to_csv(out / 'market_profiles.csv', index=False)
    selection = table(selection, 'cluster_selection', out)
    table(bounds, 'profile_winsorization_bounds', out)
    summary = profiles.groupby('cluster').agg(profile_count=('market', 'size'), mean_price=('mean_price', 'mean'), mean_price_cv=('price_cv', 'mean'), mean_daily_return_volatility=('return_volatility', 'mean'), mean_sharp_drop_fraction=('sharp_drop_fraction', 'mean'), mean_monthly_variation=('monthly_variation_cv', 'mean'), mean_calendar_coverage=('calendar_coverage', 'mean'), mean_stale_price_fraction=('stale_price_fraction', 'mean'), variety_switch_profile_fraction=('dominant_variety_switch', 'mean')).reset_index()
    summary = table(summary, 'cluster_summary', out)
    stability = []
    with timed('stability_and_ablation', timings), threadpool_limits(limits=4):
        for seed in [7, 21, 84, 123, 2026]:
            labels = KMeans(n_clusters=model.n_clusters, n_init=10, random_state=seed).fit_predict(x)
            stability.append({'check': 'different initialization', 'seed': seed, 'adjusted_rand_index': adjusted_rand_score(model.labels_, labels)})
        rng = np.random.default_rng(42)
        for repeat in range(cfg['stability_repeats']):
            ids = rng.choice(len(x), max(model.n_clusters + 1, int(.8 * len(x))), replace=False)
            candidate = KMeans(n_clusters=model.n_clusters, n_init=10, random_state=repeat).fit(x[ids])
            stability.append({'check': '80% profile subsample; fixed preprocessing', 'seed': repeat, 'adjusted_rand_index': adjusted_rand_score(model.labels_, candidate.predict(x))})
        stability = table(pd.DataFrame(stability), 'cluster_stability', out)
        price_x = np.column_stack([x, StandardScaler().fit_transform(profiles[['log_mean_price']])])
        price_model = KMeans(n_clusters=model.n_clusters, n_init=10, random_state=42).fit(price_x)
        ablation = table(pd.DataFrame([{'feature_set': 'behavior only: main model', 'k': model.n_clusters, 'silhouette': safe_silhouette(x, model.labels_), 'ARI_vs_main': 1.0}, {'feature_set': 'behavior plus log mean price', 'k': model.n_clusters, 'silhouette': safe_silhouette(price_x, price_model.labels_), 'ARI_vs_main': adjusted_rand_score(model.labels_, price_model.labels_)}]), 'price_feature_ablation', out)
        no_clip = preprocessing.transform(profiles[features])
        unclipped = KMeans(n_clusters=model.n_clusters, n_init=10, random_state=42).fit(no_clip)
        clip_ari = adjusted_rand_score(model.labels_, unclipped.labels_)
        clip_comparison = table(pd.DataFrame([{'check': 'without 1st/99th percentile profile winsorization', 'ARI_vs_main': clip_ari, 'silhouette': safe_silhouette(no_clip, unclipped.labels_), 'smallest_cluster': int(np.bincount(unclipped.labels_).min())}]), 'winsorization_sensitivity', out)
        extremes = profiles.loc[profiles.return_volatility.nlargest(10).index, MARKET_KEYS + ['mean_price', 'return_volatility', 'large_daily_change_fraction', 'cluster']]
        table(extremes, 'extreme_profiles_for_review', out)
    nearest = []
    for cluster in range(model.n_clusters):
        ids = np.flatnonzero(model.labels_ == cluster)
        order = ids[np.argsort(np.linalg.norm(x[ids] - model.cluster_centers_[cluster], axis=1))[:3]]
        nearest.append(profiles.iloc[order])
    representatives = table(pd.concat(nearest, ignore_index=True)[MARKET_KEYS + ['cluster', 'price_cv', 'return_volatility', 'sharp_drop_fraction', 'calendar_coverage']], 'representative_markets', out)
    changed = daily.copy()
    changed['daily_return'] = changed.groupby(MARKET_KEYS, observed=True).price.pct_change(fill_method=None)
    changed.loc[changed.groupby(MARKET_KEYS, observed=True).date.diff().dt.days.ne(1), 'daily_return'] = np.nan
    whatif = []
    eligible_daily = changed.merge(profiles[MARKET_KEYS], on=MARKET_KEYS, how='inner')
    for drop in [.05, .10, .15]:
        whatif.append({'sharp_drop_definition': f'{drop:.0%} or larger consecutive-day decline', 'observed_event_fraction': eligible_daily.daily_return.dropna().le(-drop).mean(), 'valid_daily_changes': eligible_daily.daily_return.notna().sum(), 'clusters_refitted': False})
    whatif = table(pd.DataFrame(whatif), 'what_if_price_drop_definition', out)
    coverage_rows = []
    for coverage in [.20, .25, .35, .50]:
        coverage_rows.append({'minimum_calendar_coverage': coverage, 'currently_eligible_profiles_retained': int(profiles.calendar_coverage.ge(coverage).sum()), 'interpretation': 'sensitivity within current eligible set; stricter criteria only, no reclustering'})
    coverage = table(pd.DataFrame(coverage_rows), 'what_if_reporting_coverage', out)
    quality = {**audit, 'raw_rows_scanned': int(manifest.raw_rows_scanned.sum()), 'raw_bytes_scanned': int(manifest.bytes.sum()), 'invalid_rows_removed': int(manifest.invalid_rows.sum()), 'inconsistent_price_rows_removed': int(manifest.inconsistent_price_rows.sum()), 'eligible_profiles': len(profiles), 'excluded_profiles': len(excluded), 'spread_missing_profile_fraction': missing, 'spread_used': 'median_relative_spread' in features, 'chosen_k': model.n_clusters, 'minimum_cluster_size_constraint_relaxed': fallback, 'common_window_start': cfg['start_date'], 'common_window_end': cfg['end_date'], 'profile_unit': 'state/district/market/variety/grade', 'features': features}
    write_json(out / 'metrics.json', quality)
    quality_table = table(pd.DataFrame([{'check': k, 'value': str(v)} for k, v in quality.items()]), 'data_quality', out)
    with timed('charts', timings):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        selection.plot(x='k', y='silhouette_sampled', marker='o', ax=axes[0])
        selection.plot(x='k', y='inertia', marker='o', ax=axes[1])
        figure('cluster_selection', out)
        projection = PCA(n_components=2, random_state=42)
        points = projection.fit_transform(x)
        plt.figure(figsize=(9, 5))
        for c in range(model.n_clusters):
            mask = model.labels_ == c
            plt.scatter(points[mask, 0], points[mask, 1], label=f'Cluster {c}', alpha=.7)
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.title(f'Behavior profiles; PCA explains {projection.explained_variance_ratio_.sum():.1%} (display only)')
        plt.legend()
        figure('cluster_projection', out)
        plt.figure(figsize=(9, 4))
        plt.imshow(model.cluster_centers_, aspect='auto', cmap='coolwarm')
        plt.colorbar(label='Standard deviations after preprocessing')
        plt.xticks(range(len(features)), features, rotation=20)
        plt.yticks(range(model.n_clusters), [f'Cluster {c}' for c in range(model.n_clusters)])
        figure('cluster_feature_heatmap', out)
        summary.plot.bar(x='cluster', y='profile_count', figsize=(8, 4), legend=False)
        figure('cluster_sizes', out)
        plt.figure(figsize=(11, 5))
        for representative in pd.concat(nearest).groupby('cluster').head(1).to_dict('records'):
            mask = np.ones(len(daily), dtype=bool)
            for key in MARKET_KEYS:
                mask &= daily[key].eq(representative[key]).to_numpy()
            curve = daily.loc[mask].set_index('date').price.resample('MS').mean()
            (curve / curve.mean()).plot(label=f'Cluster {representative["cluster"]}: {representative["market"]}')
        plt.ylabel('Monthly mean / profile mean'); plt.legend()
        figure('representative_price_histories', out)
        stability.boxplot(column='adjusted_rand_index', by='check', figsize=(11, 4), rot=15)
        plt.suptitle(''); plt.ylabel('Adjusted Rand Index')
        figure('cluster_stability', out)
    joblib.dump({'model': model, 'preprocessing': preprocessing, 'features': features, 'winsorization_bounds': bounds, 'config': cfg}, out / 'clustering.joblib')
    interpretation = []
    for row in summary.to_dict('records'):
        interpretation.append(f'Cluster {int(row["cluster"])} contains {int(row["profile_count"])} profiles: mean relative price variability {row["mean_price_cv"]:.3f}, daily return volatility {row["mean_daily_return_volatility"]:.3f}, and sharp-drop frequency {row["mean_sharp_drop_fraction"]:.1%}.')
    stability_mean = stability.loc[stability.check.str.startswith('80%'), 'adjusted_rand_index'].mean()
    sections = ['# Clustering: Indian mandi price-behavior groups',
                f'## 1. Problem statement\nGroup {cfg["commodity"]} market/variety/grade profiles by price variability and sharp-drop behavior over the shared window {cfg["start_date"]} to {cfg["end_date"]}. Farmer producer organizations could use these descriptive groups to prioritize market monitoring. This does not identify the most profitable market or predict future prices.',
                '## 2. Model and justification\nK-means is fast on a small numeric profile table and produces interpretable centers. Chunked CSV/Parquet processing handles the much larger input history. Standardized behavior features prevent units from dominating distance. The main model excludes absolute price; an ablation checks its influence. Missing spread is median-imputed only when at least 80% of profiles have it; otherwise that feature is omitted. Profile features are winsorized at the 1st/99th percentiles, with bounds exported and a no-winsorization sensitivity check. This limits outlier influence but can conceal extremes. Silhouette and minimum cluster-size checks guide k; stability checks test reproducibility.',
                '## 3. Coding\nOnly one commodity and a common date window are retained. CSV files are read in chunks and Parquet in batches; equivalent CSV/Parquet filenames are deduplicated. Positive prices and known minimum ≤ modal ≤ maximum are required. Exact duplicates are removed; multiple same-day prices are summarized by medians and conflicts reported. Profiles retain separate varieties and grades. Consecutive-calendar-day returns avoid confusing multi-day changes with daily changes. Coverage, year representation and valid-return eligibility are enforced. Stale prices, large changes and annual dominant-variety switches are flagged, not automatically declared wrong.',
                f'## 4. Results\nRaw rows scanned: {quality["raw_rows_scanned"]:,}; retained unique raw rows: {quality["retained_unique_raw_rows"]:,}; eligible profiles clustered: {len(profiles):,}. Selected k: {model.n_clusters}; sampled silhouette: {safe_silhouette(x, model.labels_):.3f}. Mean subsampling ARI: {stability_mean:.3f}. Minimum cluster-size constraint relaxed: {fallback}. Consult exported selection, cluster summaries, representative histories, stability and quality tables. Classification accuracy, specificity and confusion matrices do not apply because no true cluster labels exist.',
                '## 5. Inference and what-if analysis\n' + '\n'.join(interpretation) + f'\nHigher volatility/drop-frequency groups warrant closer monitoring; stable histories do not guarantee future stability. Subsampling ARI near 1 indicates reproducible assignments under fixed preprocessing; low values suggest weak groups. Removing profile winsorization changes assignments with ARI {clip_ari:.3f}; values far below 1 indicate strong sensitivity to extreme-profile treatment. Inspect extreme_profiles_for_review and the smallest unclipped cluster; a high silhouette from an isolated outlier is not evidence of useful groups. Adding price level changes assignments with ARI {ablation.iloc[1].ARI_vs_main:.3f}; this tests whether price level dominates grouping. What-if tables show event prevalence at 5%, 10% and 15% daily-drop definitions and the loss of eligible profiles under stricter coverage. These tables do not refit clusters. Monthly variation is a proxy, not proof of seasonality. Calendar coverage includes non-trading days. Missing markets, reporting biases, variety recoding and genuine extreme movements can affect interpretation. Transport costs, traded quantities and farmer-level selling prices are not modeled.',
                f'## 6. URL of implementation\n{cfg["implementation_url"]}\nThis Colab URL opens the GitHub notebook. For a Kaggle submission, replace implementation_url with the saved Kaggle notebook URL and run again.']
    finish_report(out, cfg, sections, {'Cluster summary': summary, 'Selection': selection, 'Representatives': representatives, 'Stability': stability, 'Data quality': quality_table}, timings, start)
    return {'quality': quality, 'profiles': profiles, 'daily': daily, 'stability': stability, 'selection': selection, 'timings': timings}
