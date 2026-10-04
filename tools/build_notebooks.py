import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)}


for task, filename, title, slug in [
    ('electricity', '01_delhi_peak_classification.ipynb', 'Delhi next-hour peak-demand classification', 'yug201/delhi-5-minute-electricity-demand-for-forecasting'),
    ('mandi', '02_mandi_price_clustering.ipynb', 'Indian mandi price-behavior clustering', 'khandelwalmanas/daily-commodity-prices-india'),
]:
    common = (ROOT / 'src/common.py').read_text()
    specific = (ROOT / f'src/{task}.py').read_text()
    cfg = {'input_root': '/kaggle/input', 'input_path': None, 'output_dir': f'/kaggle/working/{task}_results', 'implementation_url': f'https://colab.research.google.com/github/dhruvvvgg/bde-assignment/blob/main/notebooks/{filename}'}
    if task == 'electricity':
        cfg.update({'peak_quantile': .90, 'bootstrap_repeats': 100, 'rolling_backtests': True})
        problem = 'Predict whether demand will exceed a training-derived peak threshold at any five-minute observation in the next hour. Practical application: advance warnings for Delhi grid planning.'
        justification = 'Logistic Regression is a fast, interpretable main model. Compare demand/calendar-only and weather-enhanced versions against a shallow Decision Tree, persistence, yesterday and last-week baselines. Chronological splits and one-hour embargoes avoid overlapping target windows. Report sensitivity, specificity, precision, F1, ROC-AUC and average precision, including advance-warning-only observations.'
    else:
        cfg.update({'commodity': 'Onion', 'start_date': '2022-01-01', 'end_date': '2024-12-31', 'chunksize': 200000, 'min_observations': 180, 'min_calendar_coverage': .25, 'min_months': 18, 'min_year_observations': 60, 'min_daily_returns': 60, 'drop_fraction': .10, 'max_clusters': 6, 'min_cluster_fraction': .02, 'stability_repeats': 20})
        problem = 'Group Indian onion mandi/variety/grade profiles by observed price stability, daily volatility and sharp-drop behavior. Practical application: prioritize markets for monitoring by farmer producer organizations.'
        justification = 'K-means is fast on standardized numeric market profiles. Most work is filtering the larger historical dataset. The main model excludes price level, checks cluster sizes, compares several k values and tests stability under seeds and subsampling. Silhouette, Davies–Bouldin, Calinski–Harabasz and ARI apply; specificity and classification accuracy do not apply to unlabeled clusters.'
    config_source = 'CONFIG = ' + repr(cfg) + '\n'
    setup = '''import importlib.util
import subprocess
import sys
from pathlib import Path

needed = ['holidays'] if TASK == 'electricity' else ['pyarrow']
for package in needed:
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
if not Path(CONFIG['input_root']).exists() and CONFIG['input_path'] is None:
    if importlib.util.find_spec('kagglehub') is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kagglehub'])
    import kagglehub
    CONFIG['input_root'] = kagglehub.dataset_download(DATASET_SLUG)
if not Path('/kaggle/working').exists() and CONFIG['output_dir'].startswith('/kaggle/working'):
    CONFIG['output_dir'] = str(Path.cwd() / (TASK + '_results'))
print('Device: CPU. A T4 is compatible but these models do not use GPU acceleration.')
print('Input:', CONFIG['input_root'])
'''
    cells = [markdown(f'# {title}\n\nBig Data Essentials assignment. This standalone notebook contains all executable code; no GitHub clone or companion Python file is needed.\n\n**Kaggle:** choose Add Input and attach [{slug}](https://www.kaggle.com/datasets/{slug}), then click Run All. The mandi notebook uses the historical 2022–2024 files, not a single-day snapshot. Kaggle Internet is only needed if a dependency is missing. The models run on CPU, including in T4 sessions.\n\n**Colab:** open the implementation URL below and Run All. If `/kaggle/input` is absent, public data is downloaded through kagglehub; the mandi download is approximately 1.51 GB and can take substantially longer than computation.\n\nNo results are fabricated. All result and interpretation sections are generated from the input during execution.'),
             markdown('## 1. Problem statement\n\n' + problem),
             markdown('## 2. Model and justification\n\n' + justification),
             markdown('## Configuration\n\nDefaults work with the specified Kaggle inputs. Set `input_path` only when several compatible datasets are attached. Change `implementation_url` to your saved Kaggle notebook URL before the final submission. Timing is measured by stage and exported.'),
             code(f"TASK = {task!r}\nDATASET_SLUG = {slug!r}\n" + config_source),
             markdown('## 3. Coding and implementation\n\nPrepare dependencies, define helpers and execute the complete pipeline. Missing input causes an explicit setup error; the pipeline never substitutes synthetic data.'),
             code(setup), code(common), code(specific),
             markdown('## 4. Results\n\nThis block runs preprocessing, model comparisons, metrics, charts, stability/sensitivity checks and report exports. Displayed tables are also saved as CSV files.'),
             code(f"RESULTS = run_{task}(CONFIG)"),
             markdown('## 5. Inference and what-if analysis\n\nThe generated report below interprets the actual computed results. What-if scenarios are descriptive model sensitivities, not causal claims. Read the limitations alongside the findings.'),
             code("display(Markdown((Path(CONFIG['output_dir']) / 'assignment_report.md').read_text()))"),
             markdown('## 6. URL of implementation\n\nThe default link opens this notebook in Colab from GitHub. Save a Kaggle version and place its actual URL in CONFIG if required. Export the generated ZIP from the Kaggle Output pane for the document and presentation.'),
             code("display(Markdown('**Implementation:** [' + CONFIG['implementation_url'] + '](' + CONFIG['implementation_url'] + ')'))\nfrom IPython.display import FileLink\ndisplay(FileLink(str(Path(CONFIG['output_dir']).parent / (Path(CONFIG['output_dir']).name + '.zip'))))"),
             markdown('## Presentation guide\n\n1. Explain the practical problem and model choice.\n2. Demonstrate the configuration and Run All pipeline.\n3. Show data-quality and dataset-volume tables.\n4. Explain model/cluster comparison charts and the appropriate metrics.\n5. Discuss the generated inference, what-if checks and limitations.\n6. Include the actual notebook implementation URL.\n\n## References\n\n- Dataset: https://www.kaggle.com/datasets/' + slug + '\n- Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html\n- K-means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html\n- Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html\n- Adjusted Rand Index: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html\n- Mandi quality-check reference: https://github.com/MrVinamra/Mandi-Price-Cointegration-VECM\n- Delhi forecasting reference: https://github.com/pyaf/load_forecasting')]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}, 'kaggle': {'accelerator': 'none', 'isInternetEnabled': True}}, 'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'{task}-{i:02d}'
    (ROOT / 'notebooks' / filename).write_text(json.dumps(notebook, indent=1) + '\n')
    print(filename, len(cells), 'cells')
