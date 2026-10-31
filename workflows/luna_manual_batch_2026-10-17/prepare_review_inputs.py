from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from case_paths import workspace_root, output_dir, relocate
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=workspace_root()
PLAN_ROOT=Path(__file__).resolve().parent
OUT=output_dir('luna_manual_batch_2026-10-17')
GALLERY=ROOT/'shared_research/luna16_train_attribute_gallery_2026-10-15'
REQUEST={283:['margin','texture'],545:['margin','texture'],1018:['margin','texture'],300:['texture'],336:['margin','texture'],390:['texture'],391:['spiculation','texture'],472:['margin','texture'],633:['margin'],891:['margin','texture'],972:['margin','texture'],1051:['margin'],1092:['margin','texture'],1134:['margin'],15:['margin','texture'],136:['margin'],155:['margin','texture'],194:['margin','texture'],257:['margin'],927:['texture']}

def main():
    (OUT/'review_inputs').mkdir(exist_ok=True)
    (OUT/'plans').mkdir(exist_ok=True)
    requested=[{'case_id':f'LUNA16_{c}','attribute':a,'variant':'p01'} for c,attrs in REQUEST.items() for a in attrs]
    (OUT/'request.json').write_text(json.dumps(requested,indent=2),encoding='utf-8')
    records=[]
    for cid,attrs in REQUEST.items():
        case=f'LUNA16_{cid}'
        prepared=json.loads((GALLERY/f'prepared_cases/{case}.json').read_text())
        zoom=prepared['zoom_box_xyxy']
        source=GALLERY/prepared['assets']['source'];inp=GALLERY/prepared['assets']['input'];ref=GALLERY/prepared['assets']['reference']
        for attr in attrs:
            base=GALLERY/f'assets/{case}/ConceptCLIP_{attr}_p01_overlay.png'
            moe=GALLERY/f'assets/{case}/MOE29_{attr}_p01_overlay.png'
            paths=[source,inp,base,moe,inp,ref,base,moe]
            titles=['Original source CT','Unchanged model input','ConceptCLIP p01','MOE29 p01','Clean target zoom + patch coordinates','Reference extent only, not attribute GT','ConceptCLIP target zoom','MOE29 target zoom']
            fig,axes=plt.subplots(2,4,figsize=(17,9.4),dpi=120)
            fig.suptitle(f'{case} | {attr.upper()} | clean CT, reference and both p01 maps',fontsize=17,y=.99)
            for index,(ax,path,title) in enumerate(zip(axes.flat,paths,titles)):
                arr=np.asarray(Image.open(path).convert('RGB'));ax.imshow(arr,interpolation='nearest')
                if index>=4:
                    x0,y0,x1,y1=zoom;ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
                ax.set_title(title,fontsize=11,pad=10)
                if index==4:
                    ax.set_xticks(np.arange((zoom[0]//14+1)*14,zoom[2],14));ax.set_yticks(np.arange((zoom[1]//14+1)*14,zoom[3],14));ax.grid(alpha=.35,color='yellow');ax.tick_params(labelsize=9)
                else:ax.axis('off')
            fig.text(.5,.018,'Patch row=floor(y/14), column=floor(x/14); zero-based. Grid is 27x27; images use 384x384 model-input coordinates.',ha='center',fontsize=11)
            fig.subplots_adjust(top=.92,bottom=.07,left=.025,right=.99,hspace=.18,wspace=.08)
            filename=OUT/f'review_inputs/{case}_{attr}.png'
            fig.savefig(filename);plt.close(fig)
            records.append({'case_id':case,'attribute':attr,'review_image':str(filename),'source':str(source),'input':str(inp),'reference':str(ref),'base_overlay':str(base),'moe29_overlay':str(moe),'zoom_box_xyxy':zoom})
    (OUT/'review_inputs/manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'Prepared {len(records)} sheets for {len(REQUEST)} cases',flush=True)

if __name__=='__main__':main()
