import argparse
import json
import os
import time
from pathlib import Path


def run(path, input_root, output_dir):
    start = time.perf_counter()
    notebook = json.loads(Path(path).read_text())
    namespace = {'__name__': '__notebook__'}
    os.environ.setdefault('MPLBACKEND', 'Agg')
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        print(f'Executing cell {i + 1}', flush=True)
        exec(compile(text, f'{path}:cell{i + 1}', 'exec'), namespace)
        if 'CONFIG = ' in text:
            namespace['CONFIG']['input_root'] = str(Path(input_root).resolve())
            namespace['CONFIG']['output_dir'] = str(Path(output_dir).resolve())
    namespace['RESULTS']['notebook_wall_seconds'] = time.perf_counter() - start
    print('Notebook wall seconds:', namespace['RESULTS']['notebook_wall_seconds'], flush=True)
    return namespace['RESULTS']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook')
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run(args.notebook, args.input_root, args.output_dir)
