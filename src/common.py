import json
import re
import time
import platform
import html
from pathlib import Path
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits
import sklearn


def norm(value):
    return re.sub(r'[^a-z0-9]', '', str(value).casefold())


def find_column(columns, aliases, required=True):
    lookup = {norm(c): c for c in columns}
    found = next((lookup[norm(a)] for a in aliases if norm(a) in lookup), None)
    if found is None and required:
        raise ValueError(f'Missing column: {aliases}. Available columns: {list(columns)}')
    return found


def number(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '', regex=False), errors='coerce').replace([np.inf, -np.inf], np.nan)


def table(df, name, out):
    df.to_csv(out / f'{name}.csv', index=False)
    display(df)
    return df


def figure(name, out):
    plt.tight_layout()
    plt.savefig(out / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def write_json(path, data):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (pd.Timestamp, Path)):
            return str(x)
        return x
    path.write_text(json.dumps(clean(data), indent=2), encoding='utf-8')


@contextmanager
def timed(name, timings):
    start = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - start, 3)


def discover(cfg, task):
    root = Path(cfg['input_root'])
    explicit = cfg.get('input_path')
    paths = [Path(explicit)] if explicit else sorted(root.rglob('*'))
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in ('.csv', '.parquet')]
    aliases = ['power demand', 'Delhi demand', 'demand', 'load'] if task == 'electricity' else ['modal price', 'modalprice']
    accepted = []
    for p in paths:
        try:
            if p.suffix.lower() == '.csv':
                cols = pd.read_csv(p, nrows=0).columns
            else:
                import pyarrow.parquet as pq
                cols = pq.ParquetFile(p).schema.names
            if find_column(cols, aliases, False):
                accepted.append(p)
        except (OSError, ValueError) as exc:
            print(f'Skipped unreadable file {p.name}: {exc}')
    if not accepted:
        slug = 'yug201/delhi-5-minute-electricity-demand-for-forecasting' if task == 'electricity' else 'khandelwalmanas/daily-commodity-prices-india'
        raise FileNotFoundError(f'Attach https://www.kaggle.com/datasets/{slug} with Add Input, or set input_path/input_root. No compatible files found under {root}.')
    if task == 'electricity' and len(accepted) != 1:
        raise ValueError('Multiple demand files found. Set input_path to the intended Delhi CSV.')
    if task == 'mandi':
        stems = {p.stem for p in accepted if p.suffix.lower() == '.parquet'}
        accepted = [p for p in accepted if p.suffix.lower() != '.csv' or p.stem not in stems]
        if not explicit:
            lower, upper = pd.Timestamp(cfg['start_date']).year, pd.Timestamp(cfg['end_date']).year
            accepted = [p for p in accepted if not re.fullmatch(r'20\d{2}', p.stem) or lower <= int(p.stem) <= upper]
    if not accepted:
        raise FileNotFoundError('No files overlap the configured date window.')
    return accepted


def finish_report(out, cfg, sections, tables, timings, total_start):
    timings['total_seconds'] = round(time.perf_counter() - total_start, 3)
    table(pd.DataFrame([{'stage': k, 'seconds': v} for k, v in timings.items()]), 'runtime', out)
    write_json(out / 'run_config.json', cfg)
    write_json(out / 'environment.json', {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'device': 'CPU', 'timings': timings})
    body = '\n\n'.join(sections)
    (out / 'assignment_report.md').write_text(body, encoding='utf-8')
    markup = '<html><head><meta charset="utf-8"><title>BDE assignment results</title></head><body>'
    markup += '<pre style="white-space:pre-wrap">' + html.escape(body) + '</pre>'
    for name, df in tables.items():
        markup += '<h2>' + html.escape(name) + '</h2>' + df.to_html(index=False)
    for p in sorted(out.glob('*.png')):
        markup += f'<h2>{html.escape(p.stem)}</h2><img style="max-width:100%" src="{p.name}">'
    (out / 'assignment_report.html').write_text(markup + '</body></html>', encoding='utf-8')
    display(Markdown(body))
    import shutil
    archive = shutil.make_archive(str(out.parent / out.name), 'zip', out)
    print(f'Output folder: {out}\nReport bundle: {archive}\nImplementation: {cfg["implementation_url"]}')
