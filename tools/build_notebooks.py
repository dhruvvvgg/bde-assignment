import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)}


for task, filename, title, slug in [
    ('electricity', '01_delhi_peak_classification.ipynb', 'Delhi next-hour peak-demand classification with EBM', 'yug201/delhi-5-minute-electricity-demand-for-forecasting'),
    ('mandi', '02_mandi_price_clustering.ipynb', 'Onion, potato and tomato mandi behavior groups with K-means', 'khandelwalmanas/daily-commodity-prices-india'),
]:
    cfg = {'input_root': '/kaggle/input', 'input_path': None, 'output_dir': f'/kaggle/working/{task}_results'}
    if task == 'electricity':
        cfg.update({'peak_quantile': .90, 'bootstrap_repeats': 100, 'rolling_backtests': True, 'ebm_rounds': 800})
        problem = 'At each five-minute observation, predict whether Delhi electricity demand will exceed a training-derived peak threshold anywhere in the next hour. Advance warnings can help grid operators prepare extra capacity. This is a historical experiment, not a deployed grid alarm.'
        justification = 'The main model is an **Explainable Boosting Machine (EBM)**: boosted shallow trees learn nonlinear feature effects while preserving readable effect plots. Two prespecified interactions represent demand with recent change and temperature with humidity. Bounded boosting rounds and histogram bins keep computation manageable. Compare it with Logistic Regression, a shallow Decision Tree and persistence/yesterday/last-week baselines. Chronological train/validation/test splits and one-hour embargoes protect future target windows. Balanced training weights address imbalance; scores are not proven calibrated probabilities. Evaluate F1, sensitivity, specificity, ROC-AUC, average precision and advance-warning performance. EBM is an established algorithm; the contribution is its practical application and evaluation.'
        method_reference = '- EBM: https://interpret.ml/docs/ebm.html\n- EBM API: https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html\n- Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html'
    else:
        cfg.update({'commodities': ['Onion', 'Potato', 'Tomato'], 'start_date': '2022-01-01', 'end_date': '2024-12-31', 'chunksize': 200000, 'min_observations': 180, 'min_calendar_coverage': .25, 'min_months': 18, 'min_year_observations': 60, 'min_daily_returns': 60, 'drop_fraction': .10, 'max_clusters': 6, 'min_cluster_fraction': .02, 'stability_repeats': 10, 'clip_quantile': .01})
        problem = 'Discover price-behavior groups for Indian onion, potato and tomato markets. Scan the historical data once, then fit a separate model for each commodity. A profile represents a state/district/market/variety/grade combination over the same 2022–2024 window. Farmer producer organizations can use the descriptions to prioritize market monitoring; the groups do not forecast prices or profits.'
        justification = 'The main model is **K-means**, a fast and readable method for grouping standardized numeric market behavior. Scan the large history once, then fit each commodity separately. Choose 2–6 groups by highest silhouette among solutions satisfying minimum group-size and iteration checks. Davies–Bouldin, Calinski–Harabasz and inertia support interpretation. Use documented profile clipping, exclude absolute price from the main fit and test initialization, subsampling, clipping and price-level sensitivity. Per-profile silhouette flags weak separation; it is not a probability. Median summaries reduce extreme-value distortion. Specificity/accuracy require ground-truth labels and do not apply. K-means is established; the enhancements are data quality, three-commodity scope and robust evaluation.'
        method_reference = '- K-means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html\n- Silhouette: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html\n- Adjusted Rand Index: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html'
        cfg['output_dir'] = '/kaggle/working/mandi_kmeans_results'
    setup = '''import importlib.util
import subprocess
import sys
from pathlib import Path

needed = [('holidays', 'holidays>=0.50,<1'), ('interpret', 'interpret-core>=0.7,<0.8')] if TASK == 'electricity' else [('pyarrow', 'pyarrow>=15')]
for module, package in needed:
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
if CONFIG['input_path'] is None and not Path(CONFIG['input_root']).exists():
    raise FileNotFoundError('Attach the dataset with Kaggle Add Input before clicking Run All.')
print('Device: CPU. A T4 session is compatible; EBM and K-means here run on CPU.')
print('Attached input:', CONFIG['input_root'])
'''
    cells = [
        markdown(f'# {title}\n\nBig Data Essentials. This notebook embeds all its executable code and is the final Kaggle deliverable.\n\n**Setup:** use Kaggle Add Input to attach [{slug}](https://www.kaggle.com/datasets/{slug}), then click **Run All**. For mandi, attach the historical collection containing 2022, 2023 and 2024, not a one-day snapshot. No companion files or repository access are required. Enable Internet if the setup cell needs to install a missing dependency; data are read only from attached inputs. A T4 session is compatible, but these models execute on CPU.\n\nAll displayed findings are computed from the attached data. Each numbered assignment section appears once; the full report is exported for download, not redisplayed.'),
        markdown('## 1. Problem statement\n\n' + problem),
        markdown('## 2. Model and justification\n\n' + justification),
        markdown('## Configuration\n\nDefaults work with the listed Kaggle inputs. Set `input_path` only to select a particular compatible file. The mandi defaults process all three commodities. Raw-row counts and aggregated profile counts are reported separately: millions of rows are processed, but the clustering fit uses the smaller profile table. Runtime is measured by stage.'),
        code(f'TASK = {task!r}\nCONFIG = {cfg!r}\n'),
        markdown('## 3. Coding and implementation\n\nInstall missing dependencies, define the embedded pipeline and execute it below. Missing or incompatible input raises an explicit error; synthetic data are never substituted.'),
        code(setup),
        code((ROOT / 'src/common.py').read_text()),
        code((ROOT / f'src/{task}.py').read_text()),
        markdown('## 4. Results\n\nRun preprocessing, comparisons, metrics, charts, what-if checks and exports. Displayed tables are also saved as CSVs. Every commodity has its own model and output directory; group IDs are local to that commodity.' if task == 'mandi' else '## 4. Results\n\nRun preprocessing, model comparisons, specificity and other classification metrics, charts, what-if checks and exports. Displayed tables are also saved as CSVs.'),
        code(f'RESULTS = run_{task}(CONFIG)'),
        markdown('## 5. Inference and what-if analysis\n\nThe text below summarizes the actual results. What-if checks describe model sensitivity, not causal effects. Read the coverage, outlier and evaluation limitations alongside the findings.'),
        code("display(Markdown(RESULTS['inference']))"),
        markdown('## Download results\n\nDownload the generated ZIP from Kaggle Output. It contains CSVs, charts, saved models, configuration, timings and Markdown/HTML reports. Keep the extracted folder together so HTML chart paths resolve.'),
        code("from IPython.display import FileLink\ndisplay(FileLink(str(Path(CONFIG['output_dir']).parent / (Path(CONFIG['output_dir']).name + '.zip'))))"),
        markdown('## References\n\n- Dataset: https://www.kaggle.com/datasets/' + slug + '\n' + method_reference),
    ]
    for i, cell in enumerate(cells):
        cell['id'] = f'{task}-{i:02d}'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}, 'kaggle': {'accelerator': 'none', 'isInternetEnabled': True}}, 'nbformat': 4, 'nbformat_minor': 5}
    (ROOT / 'notebooks' / filename).write_text(json.dumps(notebook, indent=1) + '\n')
    print(filename, len(cells), 'cells')
