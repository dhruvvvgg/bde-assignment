from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
import holidays


def binary_metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {'rows': len(y), 'peak_fraction': float(y.mean()) if len(y) else np.nan,
            'accuracy': accuracy_score(y, pred) if len(y) else np.nan,
            'precision': precision_score(y, pred, zero_division=0), 'recall_sensitivity': sensitivity,
            'specificity': specificity, 'f1': f1_score(y, pred, zero_division=0),
            'balanced_accuracy': (sensitivity + specificity) / 2,
            'roc_auc': roc_auc_score(y, score) if score is not None and len(np.unique(y)) == 2 else np.nan,
            'average_precision': average_precision_score(y, score) if score is not None and len(np.unique(y)) == 2 else np.nan,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}


def fit_cutoff(y, score):
    choices = np.linspace(.05, .95, 91)
    scores = [f1_score(y, score >= p, zero_division=0) for p in choices]
    return float(choices[np.argmax(scores)])


def electricity_frame(raw, cfg):
    dt = find_column(raw.columns, ['datetime', 'timestamp', 'date time'])
    power = find_column(raw.columns, ['power demand', 'Delhi demand', 'demand', 'load'])
    timestamps = pd.to_datetime(raw[dt], format='mixed', errors='coerce')
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    data = pd.DataFrame({'timestamp': timestamps, 'demand': number(raw[power])})
    weather = []
    for name, aliases in {'temperature': ['temp', 'temperature'], 'humidity': ['rhum', 'humidity', 'relative humidity'], 'dew_point': ['dwpt', 'dew point'], 'wind_speed': ['wspd', 'wind speed'], 'pressure': ['pres', 'pressure']}.items():
        source = find_column(raw.columns, aliases, False)
        if source:
            data[name] = number(raw[source])
            weather.append(name)
    if not {'temperature', 'humidity'}.issubset(weather):
        raise ValueError('The enhanced notebook expects the Delhi dataset with temperature and humidity columns.')
    invalid = data['timestamp'].isna() | data['demand'].isna() | data['demand'].le(0)
    quality = {'raw_rows': len(raw), 'invalid_demand_or_timestamp': int(invalid.sum())}
    data = data.loc[~invalid]
    quality['duplicate_timestamp_rows'] = int(data['timestamp'].duplicated().sum())
    quality['off_grid_timestamp_rows'] = int((data['timestamp'].dt.second.ne(0) | data['timestamp'].dt.minute.mod(5).ne(0)).sum())
    if quality['off_grid_timestamp_rows']:
        raise ValueError('Off-grid readings require explicit timestamp alignment; silent binning could change prediction timing.')
    grid = data.groupby('timestamp').mean(numeric_only=True).sort_index().resample('5min').asfreq()
    demand = grid['demand']
    quality.update({'grid_intervals': len(grid), 'missing_demand_intervals': int(demand.isna().sum()), 'start': str(grid.index.min()), 'end': str(grid.index.max())})
    frame = pd.DataFrame({'demand_now': demand})
    for minutes in [5, 15, 30, 60, 1440, 10080]:
        frame[f'lag_{minutes}min'] = demand.shift(minutes // 5)
    for minutes in [30, 60, 180]:
        win = demand.rolling(minutes // 5, min_periods=minutes // 5)
        frame[f'mean_{minutes}min'] = win.mean()
        frame[f'std_{minutes}min'] = win.std()
    frame['change_30min'] = demand - demand.shift(6)
    hour = frame.index.hour + frame.index.minute / 60
    for name, values, period in [('hour', hour, 24), ('weekday', frame.index.dayofweek, 7), ('month', frame.index.month, 12)]:
        frame[name + '_sin'] = np.sin(2 * np.pi * values / period)
        frame[name + '_cos'] = np.cos(2 * np.pi * values / period)
    frame['weekend'] = (frame.index.dayofweek >= 5).astype(int)
    calendar = holidays.country_holidays('IN', years=sorted(set(frame.index.year)))
    frame['india_holiday'] = [int(d.date() in calendar) for d in frame.index]
    base_features = list(frame.columns)
    for name in weather:
        frame[name] = grid[name]
    frame['temperature_x_humidity'] = frame['temperature'] * frame['humidity'] / 100
    weather_features = weather + ['temperature_x_humidity']
    future = pd.concat([demand.shift(-i) for i in range(1, 13)], axis=1)
    frame['future_hour_max'] = future.max(axis=1).where(future.notna().all(axis=1))
    frame['yesterday_hour_max'] = frame['future_hour_max'].shift(288)
    frame['last_week_hour_max'] = frame['future_hour_max'].shift(2016)
    keep = base_features + ['future_hour_max', 'yesterday_hour_max', 'last_week_hour_max']
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=keep)
    quality['usable_examples'] = len(frame)
    quality['removed_incomplete_examples'] = len(grid) - len(frame)
    if len(frame) < 5000:
        raise ValueError('Fewer than 5,000 complete examples. Attach the full Delhi history.')
    return grid, frame, base_features, weather_features, quality


def block_bootstrap(test, predictions, repeats=100):
    days = test.index.normalize().unique()
    if len(days) < 8:
        return pd.DataFrame()
    rng = np.random.default_rng(42)
    y = test['target'].to_numpy()
    p = np.asarray(predictions)
    groups = [np.flatnonzero(test.index.normalize() == day) for day in days]
    samples = []
    for _ in range(repeats):
        indices = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        tn, fp, fn, tp = confusion_matrix(y[indices], p[indices], labels=[0, 1]).ravel()
        samples.append([2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan, tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp) if tn + fp else np.nan])
    vals = np.asarray(samples)
    return pd.DataFrame([{'metric': name, 'lower_95': np.nanquantile(vals[:, i], .025), 'upper_95': np.nanquantile(vals[:, i], .975), 'resampling_unit': 'calendar day'} for i, name in enumerate(['f1', 'recall_sensitivity', 'specificity'])])


def rolling_backtests(frame, base, weather, cfg):
    rows = []
    last = frame.index.max().normalize()
    for months_back in [12, 6, 0]:
        end = last - pd.DateOffset(months=months_back)
        eval_start = end - pd.Timedelta(days=90)
        tune_start = eval_start - pd.Timedelta(days=30)
        fit = frame.loc[frame.index + pd.Timedelta(hours=1) < tune_start].copy()
        tune = frame.loc[(frame.index >= tune_start) & (frame.index + pd.Timedelta(hours=1) < eval_start)].copy()
        check = frame.loc[(frame.index >= eval_start) & (frame.index + pd.Timedelta(hours=1) <= end)].copy()
        if min(len(fit), len(tune), len(check)) < 500:
            continue
        peak = fit.demand_now.quantile(cfg['peak_quantile'])
        for part in [fit, tune, check]:
            part['target'] = part.future_hour_max.ge(peak).astype(int)
        if fit.target.nunique() < 2 or tune.target.nunique() < 2:
            rows.append({'evaluation_start': eval_start, 'evaluation_end': end, 'status': 'skipped: one-class fit/tuning period'})
            continue
        columns = base + weather
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42))
        with threadpool_limits(limits=4):
            model.fit(fit[columns], fit.target)
            cutoff = fit_cutoff(tune.target, model.predict_proba(tune[columns])[:, 1])
            score = model.predict_proba(check[columns])[:, 1]
        rows.append({'evaluation_start': eval_start, 'evaluation_end': end, 'status': 'evaluated', 'threshold_native_units': peak, 'probability_cutoff': cutoff, 'training_rows': len(fit), **binary_metrics(check.target, score >= cutoff, score)})
    return pd.DataFrame(rows)


def run_electricity(cfg):
    start = time.perf_counter()
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    with timed('read_and_features', timings):
        path = discover(cfg, 'electricity')[0]
        raw = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        grid, frame, base, weather, quality = electricity_frame(raw, cfg)
        quality['input_file'] = str(path)
    quality_table = table(pd.DataFrame([{'check': k, 'value': v} for k, v in quality.items()]), 'data_quality', out)
    i, j = int(len(frame) * .70), int(len(frame) * .85)
    vstart, tstart = frame.index[i], frame.index[j]
    train = frame.loc[frame.index + pd.Timedelta(hours=1) < vstart].copy()
    valid = frame.loc[(frame.index >= vstart) & (frame.index + pd.Timedelta(hours=1) < tstart)].copy()
    test = frame.loc[frame.index >= tstart].copy()
    threshold = float(train['demand_now'].quantile(cfg['peak_quantile']))
    for part in [train, valid, test]:
        part['target'] = (part['future_hour_max'] >= threshold).astype(int)
    if train['target'].nunique() < 2 or valid['target'].nunique() < 2:
        raise ValueError('Training or validation has only one class. Adjust peak_quantile or provide longer history.')
    assert train.index.max() + pd.Timedelta(hours=1) < valid.index.min()
    assert valid.index.max() + pd.Timedelta(hours=1) < test.index.min()
    splits = table(pd.DataFrame([{'split': name, 'rows': len(p), 'start': p.index.min(), 'end': p.index.max(), 'peak_fraction': p.target.mean()} for name, p in [('train', train), ('validation', valid), ('test', test)]]), 'split_summary', out)
    features = base + weather
    q1, q3 = train.demand_now.quantile([.25, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flags = grid[['demand']].copy()
    flags['outside_training_IQR_fences'] = flags.demand.lt(low) | flags.demand.gt(high)
    flags['absolute_change_5min_fraction'] = flags.demand.pct_change(fill_method=None).abs()
    unusual = flags.loc[flags.outside_training_IQR_fences | flags.absolute_change_5min_fraction.gt(.30)]
    unusual.to_csv(out / 'unusual_demand_for_review.csv', index_label='timestamp')
    table(pd.DataFrame([{'training_IQR_lower': low, 'training_IQR_upper': high, 'flagged_grid_rows': len(unusual), 'action': 'flag only; retain genuine peaks'}]), 'demand_outlier_audit', out)
    variants = {'Logistic Regression: demand/calendar': (base, make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42))),
                'Logistic Regression: weather enhanced': (features, make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42))),
                'Shallow Decision Tree': (features, make_pipeline(SimpleImputer(strategy='median'), DecisionTreeClassifier(max_depth=5, min_samples_leaf=max(50, len(train) // 1000), class_weight='balanced', random_state=42)))}
    fitted, rows, predictions, validation_rows = {}, [], {}, []
    with timed('train_and_evaluate_models', timings), threadpool_limits(limits=4):
        for name, (cols, model) in variants.items():
            tick = time.perf_counter()
            model.fit(train[cols], train.target)
            pv = model.predict_proba(valid[cols])[:, 1]
            cutoff = fit_cutoff(valid.target, pv)
            pt = model.predict_proba(test[cols])[:, 1]
            pred = (pt >= cutoff).astype(int)
            fitted[name] = (model, cols, cutoff)
            predictions[name] = (pred, pt)
            validation_rows.append({'model': name, 'probability_cutoff': cutoff, **binary_metrics(valid.target, pv >= cutoff, pv)})
            rows.append({'model': name, 'fit_and_score_seconds': time.perf_counter() - tick, **binary_metrics(test.target, pred, pt)})
    for name, source in [('Persistence', 'demand_now'), ('Same hour yesterday', 'yesterday_hour_max'), ('Same hour last week', 'last_week_hour_max')]:
        pred = (test[source] >= threshold).astype(int).to_numpy()
        predictions[name] = (pred, test[source].to_numpy())
        rows.append({'model': name, 'fit_and_score_seconds': 0, **binary_metrics(test.target, pred, test[source])})
    comparison = table(pd.DataFrame(rows), 'model_comparison_test', out)
    validation_results = table(pd.DataFrame(validation_rows), 'model_comparison_validation', out)
    main = 'Logistic Regression: weather enhanced'
    model, cols, cutoff = fitted[main]
    pred, prob = predictions[main]
    advance = test['demand_now'].lt(threshold).to_numpy()
    advance_table = table(pd.DataFrame([{'model': name, **binary_metrics(test.target.to_numpy()[advance], p[advance], s[advance])} for name, (p, s) in predictions.items()]), 'advance_warning_only', out)
    period_rows = []
    for period, ids in test.groupby(test.index.to_period('Q')).groups.items():
        mask = test.index.isin(ids)
        for name, (p, s) in predictions.items():
            period_rows.append({'period': str(period), 'model': name, **binary_metrics(test.target.to_numpy()[mask], p[mask], s[mask])})
    periods = table(pd.DataFrame(period_rows), 'quarterly_test_performance', out)
    with timed('historical_rolling_backtests', timings):
        backtests = table(rolling_backtests(train, base, weather, cfg) if cfg.get('rolling_backtests', True) else pd.DataFrame(), 'historical_rolling_backtests', out)
    thresholds = table(pd.DataFrame([{'probability_cutoff': c, **binary_metrics(test.target, prob >= c, prob)} for c in [.2, .35, .5, cutoff, .8]]), 'what_if_alert_cutoffs', out)
    uncertainty = table(block_bootstrap(test, pred, cfg['bootstrap_repeats']), 'day_block_confidence_intervals', out)
    coefficients = table(pd.DataFrame({'feature': cols, 'standardized_coefficient': model[-1].coef_[0]}).sort_values('standardized_coefficient'), 'coefficients', out)
    policy_rows = []
    for q in [.85, .90, .95]:
        policy = train.demand_now.quantile(q)
        policy_rows.append({'train_quantile': q, 'demand_threshold_native_units': policy, 'test_future_peak_fraction': test.future_hour_max.ge(policy).mean(), 'model_retrained': False})
    policy_table = table(pd.DataFrame(policy_rows), 'what_if_peak_definition', out)
    scenarios = []
    anchor = test.loc[test.demand_now.lt(threshold)].tail(1)[cols].copy()
    if not anchor.empty:
        for delta in [-3, 0, 3]:
            x = anchor.copy()
            x['temperature'] += delta
            if 'dew_point' in x:
                x['dew_point'] += delta
            x['temperature_x_humidity'] = x.temperature * x.humidity / 100
            scenarios.append({'temperature_change_C': delta, 'model_alert_score': model.predict_proba(x)[:, 1][0], 'interpretation': 'conditional model sensitivity; not a causal effect or calibrated probability'})
    scenario_table = table(pd.DataFrame(scenarios), 'what_if_temperature', out)
    export = test[['demand_now', 'future_hour_max', 'target']].copy()
    export['alert_score'], export['prediction'] = prob, pred
    export.to_csv(out / 'test_predictions.csv', index_label='timestamp')
    with timed('charts', timings):
        grid.demand.resample('D').max().plot(figsize=(11, 4), title='Delhi daily maximum demand; native source units')
        plt.axhline(threshold, color='red', linestyle='--', label='Training-derived peak threshold')
        plt.legend()
        figure('demand_history', out)
        ConfusionMatrixDisplay.from_predictions(test.target, pred, display_labels=['No peak', 'Peak'], colorbar=False)
        figure('confusion_matrix', out)
        if test.target.nunique() == 2:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            RocCurveDisplay.from_predictions(test.target, prob, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(test.target, prob, ax=axes[1])
            axes[1].axhline(test.target.mean(), linestyle='--', color='grey', label='Peak prevalence')
            axes[1].legend()
            figure('roc_and_precision_recall', out)
        comparison.set_index('model')[['f1', 'recall_sensitivity', 'specificity']].plot.bar(figsize=(11, 4), ylim=(0, 1), rot=20)
        figure('model_comparison', out)
        coefficients.set_index('feature').standardized_coefficient.plot.barh(figsize=(9, 8), title='Standardized coefficients: associations, not causes')
        figure('coefficients', out)
        thresholds.plot(x='probability_cutoff', y=['precision', 'recall_sensitivity', 'specificity'], marker='o', figsize=(9, 4))
        figure('alert_cutoff_tradeoff', out)
    joblib.dump({'model': model, 'feature_columns': cols, 'alert_cutoff': cutoff, 'peak_threshold': threshold, 'config': cfg}, out / 'classifier.joblib')
    full = comparison.loc[comparison.model.eq(main)].iloc[0]
    persistence = comparison.loc[comparison.model.eq('Persistence')].iloc[0]
    no_weather = comparison.loc[comparison.model.eq('Logistic Regression: demand/calendar')].iloc[0]
    advance_main = advance_table.loc[advance_table.model.eq(main)].iloc[0]
    sections = ['# Classification: next-hour peak-demand warning in Delhi',
                f'## 1. Problem statement\nAt each observed five-minute timestamp, predict whether any demand reading in the following hour will exceed a threshold fixed from the training-period {cfg["peak_quantile"]:.0%} quantile. The application is an early warning for grid planning; the percentile is an experimental proxy, not an operator capacity limit.',
                '## 2. Model and justification\nLogistic Regression provides a fast, interpretable baseline for numeric demand, weather and calendar features. Standardization and regularization limit scale effects; balanced class weights address unequal class frequencies. A shallow Decision Tree tests nonlinear interactions without deep learning. Validation chooses alert cutoffs; the test period is untouched during fitting and selection. Class-weighted outputs are alert scores, not established calibrated probabilities.',
                f'## 3. Coding\nThe notebook contains the full pipeline. Current and historical features only; one-hour embargo at split boundaries. Missing demand is not interpolated, and incomplete feature/target windows are excluded. Weather imputation and scaling are learned only from training data. Training-derived IQR fences and sharp five-minute jumps flag unusual demand for review; genuine peaks are retained. Input: {path.name}.',
                f'## 4. Results\nRaw records: {len(raw):,}; usable examples: {len(frame):,}; missing demand slots: {quality["missing_demand_intervals"]:,}. Peak threshold: {threshold:.2f} native source units. Enhanced Logistic Regression test F1: {full.f1:.3f}; sensitivity: {full.recall_sensitivity:.3f}; specificity: {full.specificity:.3f}. Tables, confusion matrix, ROC/PR curves, quarterly results and day-block uncertainty are exported. Specificity = TN/(TN+FP), sensitivity = TP/(TP+FN).',
                f'## 5. Inference and what-if analysis\nEnhanced-model F1 minus persistence F1: {full.f1 - persistence.f1:+.3f}. Weather-model F1 minus demand/calendar model F1: {full.f1 - no_weather.f1:+.3f}; a negative difference means the added weather features did not help on this holdout. On timestamps currently below the peak threshold, sensitivity is {advance_main.recall_sensitivity:.3f}. This subset measures advance warning. Lower alert cutoffs trade false alarms against missed peaks; higher demand thresholds change event prevalence, and the policy table does not retrain models. Temperature scenarios show conditional score sensitivity, not causal effects. Three historical rolling backtests use expanding training, a separate 30-day tuning period and 90-day evaluation windows within the training era; these do not choose the final model. Quarterly results expose seasonal weakness; one chronological holdout and approximate day-block intervals do not prove deployment reliability. Demand units and weather timestamp availability must be confirmed before operational use.',
                f'## 6. URL of implementation\n{cfg["implementation_url"]}\nThis Colab URL opens the GitHub notebook. For a Kaggle submission, replace implementation_url with the saved Kaggle notebook URL and run again.']
    finish_report(out, cfg, sections, {'Model comparison': comparison, 'Advance warning': advance_table, 'Quarterly performance': periods, 'Historical backtests': backtests, 'Data quality': quality_table}, timings, start)
    return {'quality': quality, 'comparison': comparison, 'advance': advance_table, 'timings': timings, 'threshold': threshold, 'splits': (train, valid, test), 'model': model}
