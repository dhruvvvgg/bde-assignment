import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
namespace = {'__name__': 'pipeline_test'}
for filename in ['common.py', 'electricity.py', 'mandi.py']:
    exec(compile((ROOT / 'src' / filename).read_text(), filename, 'exec'), namespace)
np, pd = namespace['np'], namespace['pd']


class PipelineChecks(unittest.TestCase):
    def test_specificity_uses_actual_negatives(self):
        metrics = namespace['binary_metrics']([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])
        self.assertAlmostEqual(metrics['specificity'], 2 / 3)
        self.assertAlmostEqual(metrics['recall_sensitivity'], 1 / 2)
        self.assertEqual((metrics['TN'], metrics['FP'], metrics['FN'], metrics['TP']), (2, 1, 1, 1))

    def test_future_perturbation_does_not_change_predictor(self):
        dates = pd.date_range('2022-01-01', periods=16000, freq='5min')
        raw = pd.DataFrame({'datetime': dates, 'Power demand': 3000 + 200 * np.sin(np.arange(len(dates)) / 40), 'temp': 25, 'rhum': 60})
        _, original, base, weather, _ = namespace['electricity_frame'](raw, {})
        cutoff = dates[11000]
        changed = raw.copy()
        changed.loc[changed.datetime > cutoff, 'Power demand'] += 2000
        changed.loc[changed.datetime > cutoff, 'temp'] += 10
        _, modified, _, _, _ = namespace['electricity_frame'](changed, {})
        pd.testing.assert_series_equal(original.loc[cutoff, base + weather], modified.loc[cutoff, base + weather])
        self.assertNotEqual(original.loc[cutoff, 'future_hour_max'], modified.loc[cutoff, 'future_hour_max'])

    def test_gap_crossing_target_is_excluded(self):
        dates = pd.date_range('2022-01-01', periods=16000, freq='5min')
        raw = pd.DataFrame({'datetime': dates, 'Power demand': 3000, 'temp': 25, 'rhum': 60}).drop(index=11001)
        _, frame, _, _, quality = namespace['electricity_frame'](raw, {})
        self.assertNotIn(dates[11000], frame.index)
        self.assertEqual(quality['missing_demand_intervals'], 1)

    def test_multiday_change_excluded_and_bad_price_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-05', '2022-01-06'])
            rows = []
            for market in range(12):
                for date, price in zip(dates, [100, 100, 200, 210]):
                    rows.append({'State': 'A', 'District': 'D', 'Market': f'M{market}', 'Commodity': 'Onion', 'Arrival_Date': date, 'Modal_Price': price, 'Min_Price': price - 10, 'Max_Price': price + 10, 'Variety': 'V', 'Grade': 'FAQ'})
            rows.append({**rows[0], 'Modal_Price': 999})
            path = root / '2022.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            cfg = {'start_date': '2022-01-01', 'end_date': '2022-01-06', 'commodity': 'Onion', 'chunksize': 10, 'min_observations': 4, 'min_calendar_coverage': .1, 'min_months': 1, 'min_year_observations': 1, 'min_daily_returns': 2, 'drop_fraction': .1}
            daily, manifest, _ = namespace['stream_mandi']([path], cfg, root / 'out')
            profiles, _ = namespace['make_profiles'](daily, cfg)
            self.assertEqual(manifest.inconsistent_price_rows.sum(), 1)
            self.assertTrue(profiles.valid_daily_returns.eq(2).all())
            self.assertTrue(profiles.sharp_drop_fraction.eq(0).all())
            self.assertTrue(np.allclose(profiles.return_volatility, np.std([0, .05], ddof=1)))

    def test_three_commodities_keep_separate_daily_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{'State': 'A', 'District': 'D', 'Market': 'M', 'Commodity': commodity, 'Arrival_Date': '2022-01-01', 'Modal_Price': price, 'Min_Price': price - 10, 'Max_Price': price + 10, 'Variety': 'V', 'Grade': 'FAQ'} for commodity, price in [('Onion', 100), ('Potato', 200), ('Tomato', 300)]]
            path = root / '2022.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            cfg = {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'commodities': ['Onion', 'Potato', 'Tomato'], 'chunksize': 2}
            daily, manifest, audit = namespace['stream_mandi']([path], cfg, root / 'out')
            self.assertEqual(daily.set_index('commodity').price.to_dict(), {'Onion': 100, 'Potato': 200, 'Tomato': 300})
            self.assertEqual(manifest.raw_rows_scanned.sum(), 3)
            self.assertEqual(audit['retained_unique_raw_rows'], 3)

    def test_kmeans_selection_respects_size_and_separation(self):
        rng = np.random.default_rng(42)
        features = ['price_cv', 'return_volatility', 'sharp_drop_fraction', 'monthly_variation_cv', 'median_relative_spread']
        x = np.vstack([rng.normal(.3, .04, (60, 5)), rng.normal(.8, .06, (60, 5))])
        profiles = pd.DataFrame(x, columns=features)
        cfg = {'max_clusters': 3, 'min_cluster_fraction': .02, 'clip_quantile': .01}
        model, _, transformed, _, _, selection, _, _ = namespace['cluster_profiles'](profiles, cfg)
        feasible = selection.loc[~selection.iteration_limit_reached & selection.meets_minimum_cluster_size & selection.silhouette.notna()]
        chosen = selection.loc[selection.clusters.eq(model.n_clusters)].iloc[0]
        self.assertAlmostEqual(chosen.silhouette, feasible.silhouette.max())
        self.assertTrue(chosen.meets_minimum_cluster_size)
        self.assertEqual(model.n_clusters, 2)
        self.assertTrue(np.array_equal(model.predict(transformed), model.labels_))

    def test_notebook_sections_once_and_no_runtime_repository_access(self):
        for path in (ROOT / 'notebooks').glob('*.ipynb'):
            doc = json.loads(path.read_text())
            markdown = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'markdown')
            code = '\n'.join(''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code')
            for point in range(1, 6):
                self.assertEqual(markdown.count(f'## {point}. '), 1)
            self.assertNotIn('## 6.', markdown)
            for forbidden in ['co' + 'lab', 'implementation' + '_url', 'kagglehub', 'git clone']:
                self.assertNotIn(forbidden, path.read_text().lower())
            self.assertNotIn("display(Markdown((Path(CONFIG['output_dir'])", code)
            self.assertEqual(code.count("display(Markdown(RESULTS['inference']))"), 1)

    def test_notebooks_embed_current_sources_and_compile(self):
        for task, name in [('electricity', '01_delhi_peak_classification.ipynb'), ('mandi', '02_mandi_price_clustering.ipynb')]:
            doc = json.loads((ROOT / 'notebooks' / name).read_text())
            sources = [''.join(c['source']) for c in doc['cells'] if c['cell_type'] == 'code']
            self.assertIn((ROOT / 'src' / f'{task}.py').read_text(), sources)
            self.assertIn((ROOT / 'src/common.py').read_text(), sources)
            for source in sources:
                compile(source, name, 'exec')


if __name__ == '__main__':
    unittest.main()
