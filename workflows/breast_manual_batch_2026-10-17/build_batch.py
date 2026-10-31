from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from case_paths import workspace_root, output_dir, relocate
"""Explicit manual numerical illustrations; read-only access to model/source data."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import shutil
import zipfile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLAN_ROOT=Path(__file__).resolve().parent
OUT=output_dir('breast_manual_batch_2026-10-17')
NAMES={'base':'ConceptCLIP','moe29':'MOE29'}
RETAIN=np.float32(.75)

def read(path):return relocate(json.loads(Path(path).read_text(encoding='utf-8-sig')))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def expand(grid):
    result=np.zeros((384,384),np.float32)
    result[:378,:378]=np.repeat(np.repeat(grid,14,0),14,1)
    return result
def overlay(image,heat):
    alpha=np.float32(.7)*heat[...,None]
    return np.clip(image.astype(np.float32)*(1-alpha)+np.array([220,40,40],np.float32)*alpha,0,255).astype(np.uint8)
def dilate(mask):
    padded=np.pad(mask,1)
    return np.logical_or.reduce([padded[r:r+27,c:c+27] for r in range(3) for c in range(3)])

def distractors(image,mask,target,seed):
    rng=np.random.default_rng(seed)
    means=image[:378,:378].mean(axis=2).reshape(27,14,27,14).mean(axis=(1,3))
    lesion=mask[:378,:378].reshape(27,14,27,14).mean(axis=(1,3))>.03
    excluded=dilate(lesion)|dilate(target>0)
    eligible=(means>15)&(means<235)&~excluded
    eligible[:2,:]=False;eligible[-2:,:]=False;eligible[:,:2]=False;eligible[:,-2:]=False
    choices=np.argwhere(eligible)
    assert len(choices)>=25,('Insufficient visible remote tissue',len(choices))
    grid=np.zeros((27,27),np.float32);centers=[]
    for distance in [4,3,2]:
        for index in rng.permutation(len(choices)):
            r,c=map(int,choices[index])
            if any((r-a)**2+(c-b)**2<distance**2 for a,b in centers):continue
            centers.append((r,c));peak=float(rng.uniform(.48,.60));grid[r,c]=peak
            for dr,dc,factor in [(-1,0,.67),(0,-1,.81),(1,0,.77),(0,1,.62),(1,1,.44),(-1,-1,.38)]:
                rr,cc=r+dr,c+dc
                if eligible[rr,cc] and rng.random()>.18:grid[rr,cc]=max(grid[rr,cc],peak*factor*float(rng.uniform(.8,1.08)))
            if len(centers)==5:break
        if len(centers)==5:break
    assert len(centers)==5
    available=np.argwhere(eligible&(grid==0));count=min(14,len(available))
    for index in rng.choice(len(available),size=count,replace=False):
        r,c=available[index];grid[r,c]=float(rng.uniform(.12,.30))
    assert not grid[excluded].any()
    return grid,{'seed':seed,'cluster_centers_row_column':centers,'cluster_count':5,'sparse_patch_count':count,'eligible_patch_count':len(choices),'excluded_reference_lesion_and_target_neighborhood':True,'max_weight':float(grid.max())}

def draw(ax,array,title,zoom=None):
    ax.imshow(array,interpolation='nearest');ax.set_title(title,fontsize=11,pad=8);ax.axis('off')
    if zoom:
        x0,y0,x1,y1=zoom;ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)

def comparison(path,plan,inp,image,sources,rendered):
    fig,axes=plt.subplots(2,4,figsize=(14,8.5),dpi=140)
    fig.suptitle(f'{plan["case_id"]} | {plan["attribute"].title()}',fontsize=21,y=.986)
    status=plan['cue_assessability'].replace('_',' ')
    fig.text(.5,.942,f'MANUAL ILLUSTRATION from {NAMES[plan["source_model"]]} p01 | deliberately dirty | cue: {status}',ha='center',fontsize=11,color='#812e2e')
    columns=[(image,'Clean model input'),(sources['base']['overlay'],'Original ConceptCLIP p01'),(sources['moe29']['overlay'],'Original MOE29 p01'),(rendered,'MANUAL derivative')]
    for row in range(2):
        for col,(array,title) in enumerate(columns):draw(axes[row,col],array,title if row==0 else 'Target zoom',inp['zoom_box_xyxy'] if row else None)
    source_label=inp['source_attributes'].get({'calcifications':'Calcifications'}.get(plan['attribute'],plan['attribute'].title()))
    fig.text(.5,.056,f'Source {plan["attribute"]} label: {source_label} | emphasis is manual inspection guidance, not confirmed feature presence',ha='center',fontsize=10,color='#805333')
    fig.text(.5,.033,'75% of chosen model activation + uneven manual emphasis + additional remote distraction clusters',ha='center',fontsize=11)
    fig.text(.5,.012,'27 x 27 patches | red (220,40,40), alpha = 0.7 x weight | no smoothing | synthetic weights, not model predictions',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.02,right=.98,top=.89,bottom=.10,hspace=.12,wspace=.05)
    fig.savefig(path);plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'#f7f7f5'})
    request=read(PLAN_ROOT/'request.json');inputs={(x['case_id'],x['attribute']):x for x in read(PLAN_ROOT/'review_inputs/manifest.json')}
    plan_list=sum((read(PLAN_ROOT/'plans'/f'group_{g}.json') for g in 'abc'),[])
    plans={(p['case_id'],p['attribute']):p for p in plan_list}
    assert len(plans)==len(plan_list)==len(request)==31
    assert set(plans)==set(inputs)=={(r['case_id'],r['attribute']) for r in request}
    (OUT/'maps').mkdir(exist_ok=True);before={};records=[]
    for item in request:
        cid,attr=item['case_id'],item['attribute'];plan=plans[cid,attr];inp=inputs[cid,attr]
        image=np.asarray(Image.open(inp['input']).convert('RGB'));mask=np.asarray(Image.open(inp['reference_mask']).convert('L'))>0
        assert image.shape==(384,384,3) and mask.shape==(384,384)
        for key in ['source','input','reference','reference_mask','base_overlay','moe29_overlay','base_numeric','moe29_numeric']:
            before[inp[key]]=sha(inp[key])
        case_dir=OUT/'case_inputs'/cid;case_dir.mkdir(parents=True,exist_ok=True)
        for key in ['source','input','reference']:shutil.copyfile(inp[key],case_dir/f'{key}.png')
        sources={}
        for model in NAMES:
            with np.load(inp[model+'_numeric'],allow_pickle=False) as z:
                raw=z['native_gradcam_raw'];heat=z['model_input_heatmap_normalized'];meta=json.loads(str(z['metadata_json']))
            assert raw.shape==(27,27) and np.isfinite(raw).all()
            assert int(meta['gallery_id'])==inp['numeric_id'] and meta['sample_id']==inp['sample_id']
            assert meta['model_key']==model and meta['prompt_index']==0 and meta['split']==inp['export_split']
            assert meta['original_image_filename']==f'case{inp["numeric_id"]:03d}.png'
            span=float(raw.max())-float(raw.min())
            grid=np.clip((raw-float(raw.min()))/span,0,1).astype(np.float32) if span>0 else np.zeros((27,27),np.float32)
            assert np.array_equal(expand(grid),heat)
            rendered=overlay(image,heat)
            assert np.array_equal(rendered,np.asarray(Image.open(inp[model+'_overlay']).convert('RGB')))
            sources[model]={'grid':grid,'overlay':rendered,'metadata':meta}
        model=plan['source_model'];assert model in NAMES and plan['variant']=='p01'
        assert plan['cue_assessability'] in ['visible_candidate','uncertain','unresolved']
        target=np.zeros((27,27),np.float32);coords=set()
        for r,c,w in plan['target_patches']:
            assert isinstance(r,int) and isinstance(c,int) and 0<=r<27 and 0<=c<27 and 0<w<=1 and (r,c) not in coords
            target[r,c]=w;coords.add((r,c))
        assert target.max()==1
        source=sources[model]['grid'];retained=RETAIN*source
        seed=int(hashlib.sha256(f'{cid}:{attr}:dirty-v1'.encode()).hexdigest()[:8],16)
        dirt,dirt_info=distractors(image,mask,target,seed)
        final=np.maximum(np.maximum(retained,target),dirt);heat=expand(final);rendered=overlay(image,heat)
        stem=f'{cid}_{attr}_MANUAL_FROM_{NAMES[model].upper()}';prefix=OUT/'maps'/stem
        info={'case_id':cid,'numeric_id':inp['numeric_id'],'dataset':'BrEaST','split':inp['split'],'export_split':inp['export_split'],'sample_id':inp['sample_id'],'attribute':attr,'created_date':'2026-10-17',
          'artifact_type':'manual_model_derived_synthetic_illustration','is_original_model_output':False,'is_ground_truth':False,'clinically_validated':False,'quality_score':None,
          'cue_assessability':plan['cue_assessability'],'source_attributes':inp['source_attributes'],'intended_quality':'Deliberately dirty and sub-ideal; not an independently measured quality grade',
          'definition':inp['definition'],'definition_sources':read(PLAN_ROOT/'definitions.json')['sources'],'source_rank':inp['source_rank'],
          'starting_model':NAMES[model],'starting_variant':'p01','prompt_index':0,'selection_reason':plan['reason'],'intended_focus':plan['focus'],'case_specific_caveat':plan['caveat'],'reviewed_files':plan['inspected_files'],
          'starting_numeric_path':inp[model+'_numeric'],'starting_numeric_sha256':before[inp[model+'_numeric']],
          'compared_sources':{m:{'path':inp[m+'_numeric'],'sha256':before[inp[m+'_numeric']]} for m in NAMES},
          'formula':'final_grid = maximum(0.75 * selected_model_normalized_grid, manual_target_floor, extra_distractor_grid); no further normalization',
          'source_retention_scale':float(RETAIN),'distractor_construction':dirt_info,'target_patches_row_column_weight':plan['target_patches'],
          'coordinate_system':'384x384 unchanged model input; zero-based 27x27 grid with14x14 footprints;6 trailing rows and columns uncovered',
          'source_image_path':inp['source'],'source_image_sha256':before[inp['source']],'model_input_path':inp['input'],'model_input_sha256':before[inp['input']],
          'source_to_model_geometry':sources[model]['metadata']['mask_geometry'],'zoom_box_xyxy':inp['zoom_box_xyxy'],
          'overlay_rgb':[220,40,40],'overlay_alpha_scale':.7,'smoothing':False,'render_formula':'clip(input_float32*(1-.7*heat)+[220,40,40]*(.7*heat),0,255).astype(uint8)',
          'reference_mask_used_for_target_weights':False,'reference_mask_used_to_exclude_lesion_from_extra_distractions':True,
          'original_files_changed':False,'inserted_into_original_model_outputs':False,
          'limitations':['Ultrasound screenshot interpretation; no confirmed pixel labels for individual attributes','Some requested cues are source-negative or unresolved; their weights show an exploratory inspection ROI, not evidence of feature presence','Coarse patches, uneven coverage, spill and competing activation are intentional']}
        np.savez_compressed(str(prefix)+'.npz',native_manual_weights=final,source_model_normalized_grid=source,retained_model_grid=retained,manual_target_floor=target,extra_distractor_grid=dirt,manual_added_weights=final-retained,model_input_heatmap_normalized=heat,metadata_json=json.dumps(info))
        np.savetxt(str(prefix)+'_grid.csv',final,delimiter=',',fmt='%.9g')
        Image.fromarray(rendered).save(str(prefix)+'_overlay.png');Image.fromarray((heat*255).astype(np.uint8)).save(str(prefix)+'_heatmap.png');Image.fromarray((final*255).astype(np.uint8)).save(str(prefix)+'_native27.png')
        Path(str(prefix)+'_metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        comparison(str(prefix)+'_comparison.png',plan,inp,image,sources,rendered)
        with np.load(str(prefix)+'.npz',allow_pickle=False) as z:
            exact=np.array_equal(final,z['native_manual_weights']) and np.array_equal(heat,z['model_input_heatmap_normalized'])
        checks={'both_original_overlays_reconstructed_pixel_exact':True,'native_shape':list(final.shape),'expanded_shape':list(heat.shape),'finite_unit_range':bool(np.isfinite(heat).all() and heat.min()>=0 and heat.max()<=1),'saved_npz_exact':bool(exact),'saved_csv_float32_exact':bool(np.array_equal(final,np.loadtxt(str(prefix)+'_grid.csv',delimiter=',').astype(np.float32))),
          'saved_overlay_pixel_exact':bool(np.array_equal(rendered,np.asarray(Image.open(str(prefix)+'_overlay.png')))),'saved_heatmap_pixel_exact':bool(np.array_equal((heat*255).astype(np.uint8),np.asarray(Image.open(str(prefix)+'_heatmap.png')))),
          'all_nonzero_original_locations_retained':bool(np.all(final[source>0]>0)),'zero_heat_pixels_unchanged':bool(np.array_equal(rendered[heat==0],image[heat==0])),'uncovered_border_zero':bool(not heat[378:,:].any() and not heat[:,378:].any()),
          'target_max':float(final[target>0].max()),'outside_target_max':float(final[target==0].max()),'target_stronger_than_every_off_target_patch':bool(final[target>0].max()>final[target==0].max()),'extra_distractor_does_not_overlap_target':bool(not dirt[target>0].any()),'manual_target_patch_count':int(np.count_nonzero(target)),'extra_distractor_patch_count':int(np.count_nonzero(dirt))}
        assert all(v for v in checks.values() if isinstance(v,bool)),checks
        records.append({**plan,'stem':stem,'source_rank':inp['source_rank'],'numeric_id':inp['numeric_id'],'split':inp['split'],'validation':checks})
        print(f'{len(records):02}/31 {cid} {attr}: {NAMES[model]} / {plan["cue_assessability"]}',flush=True)
    assert all(sha(path)==digest for path,digest in before.items())
    validation={'status':'passed','map_count':31,'case_count':18,'attribute_counts':dict(Counter(r['attribute'] for r in records)),'source_model_counts':dict(Counter(NAMES[r['source_model']] for r in records)),'cue_assessability_counts':dict(Counter(r['cue_assessability'] for r in records)),'original_source_files_unchanged':True,'hashed_original_file_count':len(before),'checks':{r['case_id']+'/'+r['attribute']:r['validation'] for r in records}}
    for name,value in [('construction.json',records),('validation.json',validation),('source_hashes.json',before)]:
        (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

    examples=[('BrEaST-Case158','halo'),('BrEaST-Case164','margin'),('BrEaST-Case113','shape'),('BrEaST-Case094','calcifications')]
    fig,axes=plt.subplots(2,4,figsize=(13,7.7),dpi=160)
    fig.suptitle('BrEaST | deliberately dirty manual heatmaps',fontsize=20,y=.985)
    fig.text(.5,.94,'Four examples from the 31-map batch. Original ultrasound pixels remain unchanged.',ha='center',fontsize=11)
    for col,key in enumerate(examples):
        rec=next(r for r in records if (r['case_id'],r['attribute'])==key);inp=inputs[key]
        im=np.asarray(Image.open(OUT/'maps'/f'{rec["stem"]}_overlay.png'))
        draw(axes[0,col],im,f'{key[0]} | {key[1].title()}');draw(axes[1,col],im,'Target zoom',inp['zoom_box_xyxy'])
    fig.text(.5,.035,'MANUAL illustrations | target peak 1.00, off-target patches at most 0.75 | 27 x 27 grid | alpha = 0.7 x weight',ha='center',fontsize=10,color='#812e2e')
    fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.08,hspace=.13,wspace=.05);fig.savefig(OUT/'batch_preview.png');plt.close(fig)
    print(json.dumps({k:v for k,v in validation.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
