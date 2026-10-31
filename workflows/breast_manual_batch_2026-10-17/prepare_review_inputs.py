from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from case_paths import workspace_root, output_dir, relocate
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLAN_ROOT=Path(__file__).resolve().parent
OUT=output_dir('breast_manual_batch_2026-10-17')
GALLERY=workspace_root()/'shared_research/full_dataset_four_attribute_review_2026-10-12'
NUMERIC=workspace_root()/'LUNA16 Attributes Localization masks TEST/Numeric'
REQUEST={244:['halo'],164:['halo','margin'],27:['halo'],37:['halo'],158:['halo','margin'],101:['halo','calcifications'],13:['calcifications'],149:['calcifications'],125:['halo'],79:['calcifications','margin'],113:['shape','halo','calcifications'],172:['halo','calcifications'],249:['halo','margin'],73:['halo','calcifications'],103:['calcifications'],219:['halo','calcifications'],94:['margin','halo','calcifications'],143:['shape','halo']}

def main():
    (OUT/'review_inputs').mkdir(parents=True,exist_ok=True)
    (OUT/'plans').mkdir(exist_ok=True)
    d=relocate(json.loads((GALLERY/'ranked_manifest.json').read_text(encoding='utf-8-sig')))
    lookup={int(c['case_id']):c for c in d['cases']}
    rankings=list(map(int,d['max_selection']['ranking']))
    rows=[];request=[]
    for case,attributes in REQUEST.items():
        c=lookup[case]; cid=f'BrEaST-Case{case:03d}'
        paths={k:str(GALLERY/v) for k,v in c['assets'].items()}
        image=np.asarray(Image.open(paths['input']).convert('RGB'))
        mask=np.asarray(Image.open(paths['reference_crop']).convert('L'))>0
        y,x=np.where(mask)
        bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
        zoom=[max(0,bbox[0]-35),max(0,bbox[1]-35),min(378,bbox[2]+35),min(378,bbox[3]+35)]
        for attr in attributes:
            entry={'case_id':cid,'numeric_id':case,'attribute':attr,'variant':'p01','source':paths['original'],'input':paths['input'],'reference':paths['input_reference'],'reference_mask':paths['reference_crop'],'source_image':c['source_image'],'source_attributes':c['source_attributes'],'split':c['split'],'export_split':c['export_split'],'sample_id':c['sample_id'],'source_rank':rankings.index(case)+1,'zoom_box_xyxy':zoom,'lesion_bbox_xyxy':bbox,'definition':d['definitions'][attr]}
            for key,model in [('base','ConceptCLIP'),('moe29','MOE29')]:
                entry[key+'_overlay']=str(GALLERY/c['attributes'][attr]['maps'][model]['overlay'])
                entry[key+'_numeric']=str(NUMERIC/f'GALLERY_{key}_NUMERIC_ONLY_20261016/{key}/BrEaST/{case:03d}/{attr}_p01_heatmap.npz')
            entry['review_image']=str(OUT/'review_inputs'/f'{cid}_{attr}.png')
            fig,axes=plt.subplots(2,4,figsize=(17,9.6),dpi=130)
            fig.suptitle(f'{cid} | {attr.upper()} | clean ultrasound and both original p01 maps',fontsize=18,y=.99)
            arrays=[np.asarray(Image.open(paths['original']).convert('RGB')),image,np.asarray(Image.open(entry['base_overlay']).convert('RGB')),np.asarray(Image.open(entry['moe29_overlay']).convert('RGB'))]
            for ax,a,title in zip(axes[0],arrays,['Original full-field ultrasound','Unchanged model input','ConceptCLIP p01','MOE29 p01']):
                ax.imshow(a,interpolation='nearest');ax.set_title(title,fontsize=12);ax.axis('off')
            arrays=[image,np.asarray(Image.open(paths['input_reference']).convert('RGB')),arrays[2],arrays[3]]
            for i,(ax,a,title) in enumerate(zip(axes[1],arrays,['Clean zoom + patch coordinates','Lesion extent only, not attribute GT','ConceptCLIP target zoom','MOE29 target zoom'])):
                ax.imshow(a,interpolation='nearest');ax.set_xlim(zoom[0]-.5,zoom[2]-.5);ax.set_ylim(zoom[3]-.5,zoom[1]-.5);ax.set_title(title,fontsize=11)
                if i==0:
                    for v in range(0,379,14):ax.axhline(v-.5,color='yellow',lw=.4,alpha=.55);ax.axvline(v-.5,color='yellow',lw=.4,alpha=.55)
                    ax.set_xticks(range(((zoom[0]+27)//28)*28,zoom[2],28));ax.set_yticks(range(((zoom[1]+27)//28)*28,zoom[3],28));ax.tick_params(labelsize=8)
                else:ax.axis('off')
            labels=c['source_attributes']
            fig.text(.5,.057,f"Source labels: Shape={labels.get('Shape')} | Margin={labels.get('Margin')} | Halo={labels.get('Halo')} | Calcifications={labels.get('Calcifications')}",ha='center',fontsize=9)
            fig.text(.5,.032,'Grid row=floor(y/14), column=floor(x/14), zero-based. Labels do not locate attribute pixels or confirm a visible focus.',ha='center',fontsize=10)
            fig.text(.5,.012,d['definitions'][attr],ha='center',fontsize=9)
            fig.subplots_adjust(left=.025,right=.985,bottom=.085,top=.94,hspace=.17,wspace=.15)
            fig.savefig(entry['review_image']);plt.close(fig)
            rows.append(entry);request.append({'case_id':cid,'attribute':attr,'variant':'p01'})
            print(cid,attr,flush=True)
    (OUT/'review_inputs/manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    (OUT/'definitions.json').write_text(json.dumps({'definitions':d['definitions'],'source_protocol':str(GALLERY/'PROTOCOL.md'),'sources':['https://www.ippt.pan.pl/repository/open/o8782.pdf','https://www.uclahealth.org/departments/radiology/education/breast-imaging-teaching-resources/birads/ultrasound-masses']},indent=2),encoding='utf-8')
    print(f'Prepared {len(rows)} pairs in {len(REQUEST)} cases',flush=True)

if __name__=='__main__':main()
