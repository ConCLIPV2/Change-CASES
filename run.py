"""Run the original case-changing algorithms against an external data workspace."""
import argparse, os, subprocess, sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path,help='Original ConceptCLIP source-data workspace')
    parser.add_argument('--dataset',choices=['luna','breast','both'],default='both')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--prepare-review-inputs',action='store_true',help='Also regenerate the image-review sheets from external sources')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    workspace=args.workspace.resolve();output=args.output.resolve()
    if not workspace.is_dir():parser.error('The source workspace does not exist')
    if output==workspace or workspace in output.parents:
        parser.error('Choose an output directory outside the source workspace')
    env=os.environ.copy();env['CASE_WORKSPACE']=str(workspace);env['CASE_OUTPUT_ROOT']=str(output)
    names={'luna':'luna_manual_batch_2026-10-17','breast':'breast_manual_batch_2026-10-17'}
    for dataset in (list(names) if args.dataset=='both' else [args.dataset]):
        script=root/'workflows'/names[dataset]/'build_batch.py'
        if args.prepare_review_inputs:
            subprocess.run([sys.executable,str(script.with_name('prepare_review_inputs.py'))],env=env,check=True)
        subprocess.run([sys.executable,str(script)],env=env,check=True)

if __name__=='__main__':main()
