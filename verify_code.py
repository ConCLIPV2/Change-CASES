"""Validate transformation source and exact case-edit instructions, without images."""
from pathlib import Path
import ast, hashlib, json

ROOT=Path(__file__).resolve().parent
receipts=json.loads((ROOT/'source_receipts.json').read_text())['source_sha256']
counts={}
for folder,expected in [('luna_manual_batch_2026-10-17',32),('breast_manual_batch_2026-10-17',31)]:
    base=ROOT/'workflows'/folder
    requests=json.loads((base/'request.json').read_text())
    plans=sum((json.loads((base/'plans'/f'group_{g}.json').read_text()) for g in 'abc'),[])
    key=lambda r:(r['case_id'],r['attribute'])
    assert len(plans)==len(requests)==len({key(p) for p in plans})==expected
    assert {key(p) for p in plans}=={key(r) for r in requests}
    for plan in plans:
        patches=plan['target_patches']
        assert len({(r,c) for r,c,w in patches})==len(patches)
        assert all(isinstance(r,int) and isinstance(c,int) and 0<=r<27 and 0<=c<27 and 0<w<=1 for r,c,w in patches)
        assert max(w for r,c,w in patches)==1
    for p in base.rglob('*.json'):
        rel=f'{folder}/{p.relative_to(base).as_posix()}'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==receipts[rel],rel
    builder=ast.parse((base/'build_batch.py').read_text())
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ['write_gallery','write_index','bundle'] for n in ast.walk(builder))
    counts[folder]=expected
for p in ROOT.rglob('*.py'):
    if '.git' not in p.parts:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
assert not any(ROOT.rglob('*.html'))
assert not any(p.suffix.lower() in {'.png','.npz','.jpg','.zip'} for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts)
print(json.dumps({'status':'passed','requested_map_counts':counts,'exact_plans_and_source_syntax':'passed','website_and_case_assets_included':False}))
