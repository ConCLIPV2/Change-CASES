"""Resolve external source data; keep generated cases separate from source code."""
from pathlib import Path
import os

REPO=Path(__file__).resolve().parent

def workspace_root():
    root=os.environ.get('CASE_WORKSPACE')
    if not root:raise RuntimeError('Set CASE_WORKSPACE or use run.py --workspace PATH')
    return Path(root).expanduser().resolve()

def output_dir(name):
    path=Path(os.environ.get('CASE_OUTPUT_ROOT',str(REPO/'outputs'))).expanduser().resolve()/name
    path.mkdir(parents=True,exist_ok=True)
    return path

def relocate(value):
    if isinstance(value,str):
        normalized=value.replace('\\','/')
        prefix='D:/ConceptCLIP/'
        if normalized.lower().startswith(prefix.lower()):
            return str(workspace_root()/normalized[len(prefix):])
        return value
    if isinstance(value,list):return [relocate(x) for x in value]
    if isinstance(value,dict):return {relocate(k):relocate(v) for k,v in value.items()}
    return value
