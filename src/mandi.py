from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import pyarrow.parquet as pq
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings


MARKET_KEYS = ['state', 'district', 'market', 'variety', 'grade']
MANDI_ALIASES = {'state': ['state', 'state name'], 'district': ['district', 'district name'], 'market': ['market', 'market name', 'mandi'], 'commodity': ['commodity', 'commodity name'], 'date': ['arrival date', 'price date', 'date', 'reported date'], 'price': ['modal price', 'modalprice'], 'minimum': ['min price', 'minimum price'], 'maximum': ['max price', 'maximum price'], 'variety': ['variety'], 'grade': ['grade']}


def stream_mandi(paths, cfg, out):
    cache = out / 'filtered_batches'
    cache.mkdir(parents=True, exist_ok=True)
    for p in cache.glob('part_*.parquet'):
        p.unlink()
    manifest, parts = [], []
    start, end = pd.Timestamp(cfg['start_date']), pd.Timestamp(cfg['end_date'])
    wanted = {c.casefold(): c for c in cfg.get('commodities', [cfg.get('commodity', 'Onion')])}
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
            selected = chunk.loc[chunk[mapping['commodity']].astype(str).str.strip().str.casefold().isin(wanted)]
            stats['commodity_rows'] += len(selected)
            if selected.empty:
                continue
            part = pd.DataFrame({k: selected[v] for k, v in mapping.items() if v})
            part['commodity'] = part.commodity.astype(str).str.strip().str.casefold().map(wanted)
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
            part = part[['commodity'] + MARKET_KEYS + ['date', 'price', 'relative_spread']]
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
    grouped = data.groupby(['commodity'] + MARKET_KEYS + ['date'], observed=True)
    daily = grouped.agg(price=('price', 'median'), relative_spread=('relative_spread', 'median'), daily_reports=('price', 'size'), distinct_reported_prices=('price', 'nunique')).reset_index().sort_values(['commodity'] + MARKET_KEYS + ['date'])
    counts = data.groupby(['commodity', 'state', 'district', 'market', data.date.dt.year, 'variety'], observed=True).size().rename('rows').reset_index()
    dominant = counts.sort_values('rows', ascending=False).drop_duplicates(['commodity', 'state', 'district', 'market', 'date'])
    dominance = dominant.groupby(['commodity', 'state', 'district', 'market']).agg(annual_dominant_varieties=('variety', 'nunique')).reset_index()
    dominance['dominant_variety_switch'] = dominance.annual_dominant_varieties.gt(1)
    daily = daily.merge(dominance, on=['commodity', 'state', 'district', 'market'], how='left')
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


def fit_gmm(x, k, covariance, seed=42):
    model = GaussianMixture(n_components=k, covariance_type=covariance, n_init=3, max_iter=300, reg_covar=1e-4, random_state=seed)
    with warnings.catch_warnings(record=True), threadpool_limits(limits=4):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(x)
    return model


def cluster_profiles(profiles, cfg):
    features = ['price_cv', 'return_volatility', 'sharp_drop_fraction', 'monthly_variation_cv']
    spread_missing = profiles.median_relative_spread.isna().mean()
    if spread_missing <= .20:
        features.append('median_relative_spread')
    values = profiles[features].replace([np.inf, -np.inf], np.nan)
    q = cfg.get('clip_quantile', .01)
    lower, upper = values.quantile(q), values.quantile(1 - q)
    bounds = pd.DataFrame({'feature': features, 'lower': lower.values, 'upper': upper.values})
    clipped = values.clip(lower, upper, axis=1)
    clipping_flags = ((values.lt(lower, axis=1) | values.gt(upper, axis=1)).any(axis=1)).to_numpy()
    preprocessing = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    x = preprocessing.fit_transform(clipped)
    scores, models = [], {}
    unique = len(np.unique(x, axis=0))
    minimum = max(3, int(np.ceil(len(profiles) * cfg['min_cluster_fraction'])))
    for k in range(2, min(cfg['max_clusters'], len(x) - 1, unique) + 1):
        for covariance in ['diag', 'full']:
            model = fit_gmm(x, k, covariance)
            labels = model.predict(x)
            counts = np.bincount(labels, minlength=k)
            sil = safe_silhouette(x, labels)
            models[(k, covariance)] = model
            scores.append({'components': k, 'covariance_type': covariance, 'BIC': model.bic(x), 'AIC': model.aic(x), 'silhouette_hard_assignments': sil, 'smallest_hard_cluster': counts.min(), 'converged': model.converged_, 'meets_minimum_cluster_size': counts.min() >= minimum})
    selection = pd.DataFrame(scores)
    feasible = selection.loc[selection.converged & selection.meets_minimum_cluster_size & selection.silhouette_hard_assignments.notna()]
    if feasible.empty:
        raise ValueError('No converged GMM meets the minimum group size. Review profiles and selection diagnostics; constraints are not silently relaxed.')
    winner = feasible.sort_values(['BIC', 'components', 'covariance_type']).iloc[0]
    model = models[(int(winner.components), winner.covariance_type)]
    return model, preprocessing, x, features, bounds, selection, clipping_flags, float(spread_missing)


def analyze_commodity(daily, cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    with timed('profile_features', timings):
        profiles, excluded = make_profiles(daily, cfg)
    excluded.to_csv(out / 'excluded_profiles.csv', index=False)
    with timed('GMM_selection', timings):
        model, preprocessing, x, features, bounds, selection, flags, missing = cluster_profiles(profiles, cfg)
    membership = model.predict_proba(x)
    labels = membership.argmax(axis=1)
    profiles['commodity'] = cfg['commodity']
    profiles['cluster'] = labels
    profiles['membership_confidence'] = membership.max(axis=1)
    profiles['uncertain_membership'] = profiles.membership_confidence.lt(cfg['membership_confidence_threshold'])
    profiles['profile_was_clipped'] = flags
    for c in range(model.n_components):
        profiles[f'group_{c}_membership'] = membership[:, c]
    order = profiles.groupby('cluster').return_volatility.median().sort_values().index.tolist()
    descriptions = {c: f'Volatility group {rank + 1}/{len(order)} (low to high)' for rank, c in enumerate(order)}
    profiles['group_description'] = profiles.cluster.map(descriptions)
    profiles.to_csv(out / 'market_profiles.csv', index=False)
    selection = table(selection, 'GMM_component_selection', out)
    table(bounds, 'profile_clipping_bounds', out)
    summary = profiles.groupby('cluster').agg(profile_count=('market', 'size'), mean_price=('mean_price', 'mean'), median_price_cv=('price_cv', 'median'), mean_daily_return_volatility=('return_volatility', 'mean'), median_daily_return_volatility=('return_volatility', 'median'), median_sharp_drop_fraction=('sharp_drop_fraction', 'median'), median_monthly_variation=('monthly_variation_cv', 'median'), mean_calendar_coverage=('calendar_coverage', 'mean'), uncertain_profile_fraction=('uncertain_membership', 'mean'), clipped_profile_fraction=('profile_was_clipped', 'mean'), mean_membership_confidence=('membership_confidence', 'mean')).reset_index()
    summary.insert(0, 'commodity', cfg['commodity'])
    summary['group_description'] = summary.cluster.map(descriptions)
    summary = table(summary, 'group_summary', out)
    with timed('stability_baseline_and_sensitivity', timings), threadpool_limits(limits=4):
        baseline = KMeans(n_clusters=model.n_components, n_init=10, random_state=42).fit(x)
        comparison = table(pd.DataFrame([{'model': 'GMM: main', 'groups': model.n_components, 'silhouette_hard_assignments': safe_silhouette(x, labels), 'ARI_vs_GMM': 1.0}, {'model': 'K-means: baseline at same group count', 'groups': model.n_components, 'silhouette_hard_assignments': safe_silhouette(x, baseline.labels_), 'ARI_vs_GMM': adjusted_rand_score(labels, baseline.labels_)}]), 'model_comparison', out)
        stability_rows = []
        for seed in [7, 21, 84, 123, 2026]:
            candidate = fit_gmm(x, model.n_components, model.covariance_type, seed)
            stability_rows.append({'check': 'different initialization', 'seed': seed, 'converged': candidate.converged_, 'ARI_vs_main': adjusted_rand_score(labels, candidate.predict(x))})
        rng = np.random.default_rng(42)
        for repeat in range(cfg['stability_repeats']):
            ids = rng.choice(len(x), max(model.n_components + 1, int(.8 * len(x))), replace=False)
            candidate = fit_gmm(x[ids], model.n_components, model.covariance_type, repeat)
            stability_rows.append({'check': '80% profile subsample; fixed preprocessing', 'seed': repeat, 'converged': candidate.converged_, 'ARI_vs_main': adjusted_rand_score(labels, candidate.predict(x))})
        stability = table(pd.DataFrame(stability_rows), 'group_stability', out)
        sensitivities = []
        for q in [0, .01, .02]:
            values = profiles[features].copy()
            if q:
                values = values.clip(values.quantile(q), values.quantile(1-q), axis=1)
            prep = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
            changed_x = prep.fit_transform(values)
            candidate = fit_gmm(changed_x, model.n_components, model.covariance_type)
            changed_labels = candidate.predict(changed_x)
            sensitivities.append({'clip_quantile_each_tail': q, 'converged': candidate.converged_, 'ARI_vs_main': adjusted_rand_score(labels, changed_labels), 'smallest_hard_cluster': np.bincount(changed_labels, minlength=model.n_components).min(), 'silhouette': safe_silhouette(changed_x, changed_labels)})
        sensitivity = table(pd.DataFrame(sensitivities), 'clipping_sensitivity', out)
        price_x = np.column_stack([x, StandardScaler().fit_transform(profiles[['log_mean_price']])])
        price_model = fit_gmm(price_x, model.n_components, model.covariance_type)
        price_ari = adjusted_rand_score(labels, price_model.predict(price_x))
        table(pd.DataFrame([{'check': 'include log mean price', 'converged': price_model.converged_, 'ARI_vs_main': price_ari}]), 'price_level_sensitivity', out)
    nearest = []
    for c in range(model.n_components):
        ids = np.flatnonzero(labels == c)
        ranked = ids[np.argsort(np.linalg.norm(x[ids] - model.means_[c], axis=1))[:3]]
        nearest.append(profiles.iloc[ranked])
    representatives = table(pd.concat(nearest, ignore_index=True)[['commodity'] + MARKET_KEYS + ['cluster', 'group_description', 'membership_confidence', 'uncertain_membership', 'price_cv', 'return_volatility', 'sharp_drop_fraction']], 'representative_markets', out)
    uncertain = profiles.sort_values('membership_confidence').head(10)
    table(uncertain[['commodity'] + MARKET_KEYS + ['cluster', 'membership_confidence', 'uncertain_membership']], 'least_confident_assignments', out)
    table(profiles.loc[profiles.return_volatility.nlargest(10).index, MARKET_KEYS + ['return_volatility', 'large_daily_change_fraction', 'profile_was_clipped']], 'extreme_profiles_for_review', out)
    changed = daily.copy()
    changed['daily_return'] = changed.groupby(MARKET_KEYS, observed=True).price.pct_change(fill_method=None)
    changed.loc[changed.groupby(MARKET_KEYS, observed=True).date.diff().dt.days.ne(1), 'daily_return'] = np.nan
    eligible = changed.merge(profiles[MARKET_KEYS], on=MARKET_KEYS, how='inner')
    whatif = table(pd.DataFrame([{'drop_definition': f'{drop:.0%} consecutive-day decline', 'event_fraction': eligible.daily_return.dropna().le(-drop).mean(), 'valid_daily_changes': eligible.daily_return.notna().sum(), 'groups_refitted': False} for drop in [.05, .10, .15]]), 'what_if_daily_drops', out)
    table(pd.DataFrame([{'minimum_confidence': cutoff, 'profiles_below_cutoff': int(profiles.membership_confidence.lt(cutoff).sum()), 'interpretation': 'ambiguity of fitted membership, not future price risk'} for cutoff in [.50, .65, .80, .90]]), 'what_if_membership_confidence', out)
    table(pd.DataFrame([{'minimum_calendar_coverage': cutoff, 'currently_eligible_profiles_retained': int(profiles.calendar_coverage.ge(cutoff).sum()), 'groups_refitted': False} for cutoff in [.25, .35, .50]]), 'what_if_reporting_coverage', out)
    with timed('charts', timings):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for covariance, part in selection.groupby('covariance_type'):
            axes[0].plot(part.components, part.BIC, marker='o', label=covariance)
            axes[1].plot(part.components, part.silhouette_hard_assignments, marker='o', label=covariance)
        axes[0].set(xlabel='GMM components', ylabel='BIC (lower is better)', title=cfg['commodity'])
        axes[1].set(xlabel='GMM components', ylabel='Hard-assignment silhouette')
        axes[0].legend(); axes[1].legend()
        figure('GMM_selection', out)
        projection = PCA(n_components=2, random_state=42)
        points = projection.fit_transform(x)
        plt.figure(figsize=(9, 5))
        for c in range(model.n_components):
            mask = labels == c
            plt.scatter(points[mask, 0], points[mask, 1], label=f'Group {c}', alpha=.65)
        uncertain_ids = profiles.uncertain_membership.to_numpy()
        plt.scatter(points[uncertain_ids, 0], points[uncertain_ids, 1], facecolors='none', edgecolors='black', label='Uncertain membership')
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
        plt.title(f'{cfg["commodity"]}: PCA display explains {projection.explained_variance_ratio_.sum():.1%}')
        figure('group_projection', out)
        plt.figure(figsize=(9, 4))
        plt.imshow(model.means_, aspect='auto', cmap='coolwarm'); plt.colorbar(label='Standardized component mean')
        plt.xticks(range(len(features)), features, rotation=20)
        plt.yticks(range(model.n_components), [f'Group {c}' for c in range(model.n_components)])
        plt.title(cfg['commodity'])
        figure('GMM_feature_heatmap', out)
        summary.plot.bar(x='cluster', y='profile_count', legend=False, figsize=(8, 4), title=cfg['commodity'])
        figure('group_sizes', out)
        plt.figure(figsize=(11, 5))
        for representative in pd.concat(nearest).groupby('cluster').head(1).to_dict('records'):
            mask = np.ones(len(daily), dtype=bool)
            for key in MARKET_KEYS:
                mask &= daily[key].eq(representative[key]).to_numpy()
            curve = daily.loc[mask].set_index('date').price.resample('MS').mean()
            (curve / curve.mean()).plot(label=f'Group {representative["cluster"]}: {representative["market"]}')
        plt.ylabel('Monthly mean / profile mean'); plt.legend(); plt.title(cfg['commodity'])
        figure('representative_price_histories', out)
        profiles.membership_confidence.plot.hist(bins=20, figsize=(8, 4), title=cfg['commodity'] + ': GMM membership confidence')
        plt.xlabel('Maximum fitted component probability')
        figure('membership_confidence', out)
    joblib.dump({'model': model, 'preprocessing': preprocessing, 'features': features, 'clipping_bounds': bounds, 'group_descriptions': descriptions, 'config': cfg}, out / 'GMM.joblib')
    selected = selection.loc[selection.components.eq(model.n_components) & selection.covariance_type.eq(model.covariance_type)].iloc[0]
    bootstrap_mean = stability.loc[stability.check.str.startswith('80%') & stability.converged, 'ARI_vs_main'].mean()
    metrics = {'commodity': cfg['commodity'], 'retained_daily_records': len(daily), 'eligible_profiles': len(profiles), 'excluded_profiles': len(excluded), 'groups': model.n_components, 'covariance_type': model.covariance_type, 'BIC': selected.BIC, 'silhouette_hard_assignments': safe_silhouette(x, labels), 'mean_subsampling_ARI': bootstrap_mean, 'uncertain_profile_fraction': profiles.uncertain_membership.mean(), 'clipped_profile_fraction': flags.mean(), 'spread_missing_fraction': missing, 'price_feature_ARI': price_ari}
    write_json(out / 'metrics.json', metrics)
    table(pd.DataFrame([{'stage': stage, 'seconds': seconds} for stage, seconds in timings.items()]), 'runtime', out)
    descriptions_text = []
    for row in summary.to_dict('records'):
        descriptions_text.append(f'Group {int(row["cluster"])} contains {int(row["profile_count"])} profiles; median daily volatility {row["median_daily_return_volatility"]:.3f}, median sharp-drop frequency {row["median_sharp_drop_fraction"]:.1%}, uncertain membership {row["uncertain_profile_fraction"]:.1%}.')
    unclipped_ari = sensitivity.loc[sensitivity.clip_quantile_each_tail.eq(0), 'ARI_vs_main'].iloc[0]
    two_pct_ari = sensitivity.loc[sensitivity.clip_quantile_each_tail.eq(.02), 'ARI_vs_main'].iloc[0]
    inference = f'**{cfg["commodity"]}:** selected {model.n_components} Gaussian components ({model.covariance_type} covariance), silhouette {metrics["silhouette_hard_assignments"]:.3f}, mean converged subsampling ARI {bootstrap_mean:.3f}. ' + ' '.join(descriptions_text) + f' Assignment ARI without clipping is {unclipped_ari:.3f}; with 2% clipping it is {two_pct_ari:.3f}. Low values reveal dependence on extreme-value treatment. Adding price level changes assignments with ARI {price_ari:.3f}. Component membership describes overlap in observed behavior, not a probability of future loss.'
    return {'metrics': metrics, 'profiles': profiles, 'summary': summary, 'selection': selection, 'stability': stability, 'sensitivity': sensitivity, 'model_comparison': comparison, 'inference': inference}


def run_mandi(cfg):
    start = time.perf_counter()
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    requested = cfg.get('commodities', ['Onion', 'Potato', 'Tomato'])
    if len(requested) != len(set(c.casefold() for c in requested)):
        raise ValueError('Commodity names must be unique.')
    with timed('shared_scan_filter_and_aggregation', timings):
        paths = discover(cfg, 'mandi')
        daily, manifest, audit = stream_mandi(paths, cfg, out)
    manifest = table(manifest, 'file_manifest', out)
    daily.to_parquet(out / 'clean_daily_prices.parquet', index=False)
    analyses = {}
    with timed('three_commodity_analysis', timings):
        for commodity in requested:
            display(Markdown('### ' + commodity + ' results'))
            subset = daily.loc[daily.commodity.eq(commodity)].copy()
            if subset.empty:
                raise ValueError(f'No records for {commodity}; this run cannot silently omit a requested commodity.')
            commodity_cfg = {**cfg, 'commodity': commodity}
            analyses[commodity] = analyze_commodity(subset, commodity_cfg, out / commodity.casefold())
    overview = table(pd.DataFrame([result['metrics'] for result in analyses.values()]), 'commodity_overview', out)
    overall_profiles = pd.concat([result['profiles'] for result in analyses.values()], ignore_index=True)
    overall_profiles.to_csv(out / 'all_commodity_profiles.csv', index=False)
    quality = {**audit, 'raw_rows_scanned_once': int(manifest.raw_rows_scanned.sum()), 'raw_bytes_scanned_once': int(manifest.bytes.sum()), 'inconsistent_price_rows_removed': int(manifest.inconsistent_price_rows.sum()), 'commodities': requested, 'eligible_profiles_total': len(overall_profiles), 'common_window_start': cfg['start_date'], 'common_window_end': cfg['end_date']}
    write_json(out / 'metrics.json', quality)
    quality_table = table(pd.DataFrame([{'check': key, 'value': str(value)} for key, value in quality.items()]), 'data_quality', out)
    inference = '\n\n'.join(result['inference'] for result in analyses.values()) + '\n\nEach commodity was preprocessed and modeled separately, so group numbers and BIC values are not comparable across commodities. Higher median volatility/drop-frequency groups may warrant closer monitoring. Monthly variation is a proxy, not proof of seasonality; reports can reflect missing markets, variety recoding and extreme movements. Confidence cutoffs change which memberships are flagged as ambiguous, without refitting. Drop/coverage what-if tables are descriptive, not forecasts. Median summaries resist extreme-value inflation. Covariance selection uses BIC with convergence and minimum hard-group-size checks; silhouette is supplementary. ARI stability holds preprocessing fixed and does not establish economic validity. Transport costs, quantities and farmer-level selling prices are absent.'
    sections = ['# Clustering: Indian mandi price-behavior groups',
                f'## 1. Problem statement\nGroup Onion, Potato and Tomato market/variety/grade profiles by observed price variability over {cfg["start_date"]} to {cfg["end_date"]}. Farmer producer organizations could prioritize markets for monitoring. Each commodity is modeled separately, preventing commodity identity from driving the groups.',
                '## 2. Model and justification\nGaussian Mixture Models allow overlapping behavior groups and provide fitted component-membership probabilities. Standardized behavior features use price CV, consecutive-day return volatility, sharp-drop frequency, monthly variation and relative spread when available. GMM component count and diagonal/full covariance are selected by BIC among converged solutions meeting minimum hard-group size. K-means remains a baseline at the same group count. This is an established method, not a novel algorithm. Classification accuracy and specificity are inapplicable without true cluster labels.',
                '## 3. Coding\nAll code is embedded in the Kaggle notebook. Year files are scanned once in CSV chunks or Parquet batches and all three commodities are filtered together. Subsequent profiles, clipping, scaling and GMM fitting are separate by commodity. Known price inconsistencies and duplicates are removed; daily medians resolve multiple reports. Common-window coverage and consecutive-calendar-day returns are enforced. Stale reporting, extreme returns and variety switches are flagged. GMM confidence describes fitted overlap, not future price risk.',
                f'## 4. Results\nRaw records scanned once: {quality["raw_rows_scanned_once"]:,}; retained unique commodity records: {audit["retained_unique_raw_rows"]:,}; eligible profiles: {len(overall_profiles):,}. Per-commodity tables/charts report GMM selection, K-means comparison, median behavior, membership confidence, stability, clipping/price sensitivity and representative histories.',
                '## 5. Inference and what-if analysis\n' + inference]
    report_tables = {'Commodity overview': overview, 'Data quality': quality_table}
    for commodity, result in analyses.items():
        report_tables[commodity + ' group summary'] = result['summary']
        report_tables[commodity + ' model selection'] = result['selection']
        report_tables[commodity + ' sensitivity'] = result['sensitivity']
    finish_report(out, cfg, sections, report_tables, timings, start)
    return {'quality': quality, 'overview': overview, 'analyses': analyses, 'timings': timings, 'inference': inference}
