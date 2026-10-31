"""Reproducible synthetic edits of saved numerical maps; original files read only.

The new maps are manual illustrations derived from ConceptCLIP, not model outputs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(r'D:\ConceptCLIP')
OUT=Path(__file__).resolve().parent
GALLERY=ROOT/'shared_research/luna16_train_attribute_gallery_2026-10-15'
INPUT=GALLERY/'assets/LUNA16_283/input.png'
SOURCE=GALLERY/'assets/LUNA16_283/source.png'
DEFS=ROOT/'shared_research/luna16_user_study_review_2026-10-12/definitions.json'
NUMERIC=ROOT/'LUNA16 Attributes Localization masks TEST/Numeric'
RETAIN=np.float32(.60)
ZOOM=(145,140,241,236)
TARGETS={
    'margin':[(11,12,.16),(12,11,.30),(12,12,.86),(12,13,.50),
              (13,11,.60),(13,12,1.0),(14,11,.18),(14,12,.55),
              (14,13,.22),(15,11,.32)],
    'texture':[(12,12,.20),(12,13,.35),(13,12,.55),(13,13,1.0),
               (13,14,.68),(14,12,.43),(14,13,.90),(14,14,.72),
               (15,13,.18),(15,14,.16)],
}
REASONS={
    'margin':'ConceptCLIP p01 has localized left and lower edge contact; MOE29 p01 is diffuse and largely spares the target. Manual emphasis follows the selected softer upper-left/left transition, with incomplete coverage and neighboring-tissue spill.',
    'texture':'ConceptCLIP p01 has visible central/lower nodule-interior contact; MOE29 p01 leaves the target interior essentially unmarked. Manual emphasis targets visible solid-appearing interior with uneven coverage and edge spill.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(grid):
    h=np.zeros((384,384),np.float32)
    h[:378,:378]=np.repeat(np.repeat(grid,14,axis=0),14,axis=1)
    return h


def overlay(image,h):
    alpha=h[...,None]*np.float32(.7)
    return np.clip(image.astype(np.float32)*(1-alpha)+np.array([220,40,40],np.float32)*alpha,0,255).astype(np.uint8)


def draw(ax,array,title,zoom=False,gray=False):
    ax.imshow(array,interpolation='nearest',**({'cmap':'gray','vmin':0,'vmax':1} if gray else {}))
    if zoom:
        x0,y0,x1,y1=ZOOM
        ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_title(title,fontsize=12,pad=10);ax.axis('off')


def main():
    before={str(p):sha(p) for p in [INPUT,SOURCE,DEFS]}
    image=np.asarray(Image.open(INPUT).convert('RGB'))
    original=np.asarray(Image.open(SOURCE).convert('RGB'))
    assert image.shape==(384,384,3) and original.shape==(256,256,3)
    definitions={d['id']:d for d in json.loads(DEFS.read_text())['attributes']}
    results={};checks={};manifest={}
    for attr in ['margin','texture']:
        sources={}
        for model in ['base','moe29']:
            file=NUMERIC/f'GALLERY_{model}_NUMERIC_ONLY_20261016/{model}/LUNA16/LUNA16_283/{attr}_p01_heatmap.npz'
            before[str(file)]=sha(file)
            with np.load(file,allow_pickle=False) as z:
                raw=z['native_gradcam_raw'];heat=z['model_input_heatmap_normalized'];meta=json.loads(str(z['metadata_json']))
            norm=np.clip((raw-float(raw.min()))/(float(raw.max())-float(raw.min())),0,1)
            assert meta['sample_id']=='LUNA16_283' and meta['attribute']==attr and meta['model']==model
            assert np.array_equal(expand(norm),heat)
            name='ConceptCLIP' if model=='base' else 'MOE29'
            saved=GALLERY/f'assets/LUNA16_283/{name}_{attr}_p01_overlay.png'
            before[str(saved)]=sha(saved)
            assert np.array_equal(overlay(image,heat),np.asarray(Image.open(saved).convert('RGB')))
            sources[model]={'grid':norm,'heat':heat,'overlay':overlay(image,heat),'file':str(file),'sha256':before[str(file)],'metadata':meta}
        base=sources['base']['grid']
        retained=base*RETAIN
        target=np.zeros_like(base)
        for r,c,w in TARGETS[attr]:target[r,c]=w
        combined=np.maximum(retained,target)
        added=combined-retained
        heat=expand(combined);rendered=overlay(image,heat)
        info={
            'case_id':'LUNA16_283','dataset':'LUNA16','split':'train','attribute':attr,'date':'2026-10-17',
            'artifact_type':'manual_model_derived_synthetic_illustration','is_model_output':False,
            'is_ground_truth':False,'quality_score':None,'clinically_validated':False,
            'intended_quality':'Deliberately imperfect with competing off-target activation; no measured or validated quality grade.',
            'definition':definitions[attr],'definition_source':str(DEFS),
            'definition_reference':'https://pylidc.github.io/annotation.html',
            'starting_model':'ConceptCLIP','starting_variant':'p01','selection_reason':REASONS[attr],
            'selection_scope':'Visual choice for this illustration, not a validated superiority claim.',
            'starting_numeric_path':sources['base']['file'],'starting_numeric_sha256':sources['base']['sha256'],
            'alternative_inspected_numeric_path':sources['moe29']['file'],'alternative_numeric_sha256':sources['moe29']['sha256'],
            'formula':'final_grid = maximum(0.60 * ConceptCLIP_p01_normalized_grid, manual_target_floor); no further normalization.',
            'source_retention_scale':float(RETAIN),'random_noise_added':False,
            'distractor_origin':'Existing ConceptCLIP activation at 60 percent amplitude, including original off-target areas; not additional attribute evidence.',
            'target_floor_patches':[{'row':r,'column':c,'weight':w} for r,c,w in TARGETS[attr]],
            'coordinate_system':'384x384 unchanged model-input image; zero-based rows/columns; 27x27 grid;14x14 footprints;6 trailing rows/columns uncovered.',
            'source_image_path':str(SOURCE),'source_image_sha256':before[str(SOURCE)],'source_image_hw':[256,256],
            'model_input_path':str(INPUT),'model_input_sha256':before[str(INPUT)],'model_input_hw':[384,384],
            'overlay_rgb':[220,40,40],'overlay_alpha_scale':.7,'smoothing':False,
            'render_formula':'clip(input_float32*(1-.7*heat)+[220,40,40]*(.7*heat),0,255).astype(uint8)',
            'limitations':['Partial cue coverage','Coarse patches include neighboring structures','Remote competing activity retained','PNG interpretation only; not raw DICOM/HU analysis'],
            'reference_mask_used_for_weights':False,'original_files_changed':False,'inserted_into_user_study':False,
        }
        stem=f'LUNA16_283_{attr}_MANUAL_FROM_CONCEPTCLIP'
        np.savez_compressed(OUT/f'{stem}.npz',native_manual_weights=combined,
            source_model_normalized_grid=base,retained_model_grid=retained,manual_target_floor=target,
            manual_added_weights=added,model_input_heatmap_normalized=heat,metadata_json=json.dumps(info))
        plt.imsave(OUT/f'{stem}_overlay.png',rendered)
        gray=np.repeat((heat*255).astype(np.uint8)[...,None],3,axis=2)
        plt.imsave(OUT/f'{stem}_heatmap.png',gray)
        off=target==0
        with np.load(OUT/f'{stem}.npz',allow_pickle=False) as z:
            reload_exact=np.array_equal(z['native_manual_weights'],combined) and np.array_equal(z['model_input_heatmap_normalized'],heat)
        saved_overlay=np.asarray(Image.open(OUT/f'{stem}_overlay.png').convert('RGB'))
        checks[attr]={
            'source_arrays_reproduce_original_overlays':True,'native_shape':list(combined.shape),'pixel_shape':list(heat.shape),
            'finite':bool(np.isfinite(heat).all()),'range':[float(heat.min()),float(heat.max())],
            'nonzero_source_patches':int(np.count_nonzero(base)),'nonzero_final_patches':int(np.count_nonzero(combined)),
            'manual_changed_patches':int(np.count_nonzero(added)),
            'main_target_max':float(combined[~off].max()),'outside_manual_support_max':float(combined[off].max()),
            'outside_manual_support_exactly_scaled_source':bool(np.array_equal(combined[off],retained[off])),
            'all_original_nonzero_locations_retained':bool(np.all(combined[base>0]>0)),
            'saved_arrays_exact':bool(reload_exact),'saved_overlay_exact':bool(np.array_equal(saved_overlay,rendered)),
            'zero_heat_pixels_unchanged':bool(np.array_equal(saved_overlay[heat==0],image[heat==0])),
            'uncovered_border_zero':bool(not heat[378:,:].any() and not heat[:,378:].any()),
        }
        results[attr]={'sources':sources,'heat':heat,'overlay':rendered,'retained':retained,'added':added}
        manifest[attr]=info
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'#f7f7f5'})
    fig,ax=plt.subplots(2,3,figsize=(12,9),dpi=160)
    fig.suptitle('LUNA16_283 | margin and texture',fontsize=19,y=.987)
    fig.text(.5,.94,'MANUAL EXAMPLES FROM CONCEPTCLIP p01  |  deliberately imperfect  |  not original model predictions',ha='center',fontsize=10,color='#8b2525')
    for r,zoom in enumerate([False,True]):
        draw(ax[r,0],image,'Unchanged CT at model-input resolution' if r==0 else 'Clean target zoom',zoom)
        draw(ax[r,1],results['margin']['overlay'],'Margin: softer boundary + distractions' if r==0 else 'Margin: incomplete edge coverage',zoom)
        draw(ax[r,2],results['texture']['overlay'],'Texture: interior + distractions' if r==0 else 'Texture: uneven interior coverage',zoom)
    fig.text(.5,.045,'Existing background activation retained at 60%; selected target patches strengthened by hand.',ha='center',fontsize=11)
    fig.text(.5,.02,'27 x 27 grid  |  14 x 14 patch footprints  |  red RGB (220,40,40), alpha = 0.7 x weight  |  no smoothing',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(left=.02,right=.98,top=.895,bottom=.09,hspace=.15,wspace=.06)
    fig.savefig(OUT/'LUNA16_283_MARGIN_TEXTURE_preview.png');plt.close(fig)
    fig,ax=plt.subplots(2,4,figsize=(14,8.1),dpi=150)
    fig.suptitle('LUNA16_283 | original maps and manual derivatives',fontsize=18,y=.987)
    fig.text(.5,.94,'Target zooms, same image coordinates and overlay formula. Only the last column is manually modified.',ha='center',fontsize=10,color='#8b2525')
    for row,attr in enumerate(['margin','texture']):
        for col,(arr,label) in enumerate([(image,'Clean CT'),(results[attr]['sources']['base']['overlay'],'ConceptCLIP p01'),(results[attr]['sources']['moe29']['overlay'],'MOE29 p01'),(results[attr]['overlay'],'MANUAL derivative')]):
            draw(ax[row,col],arr,f'{attr.title()} | {label}',True)
    fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.035,hspace=.14,wspace=.05)
    fig.savefig(OUT/'LUNA16_283_original_vs_manual.png');plt.close(fig)
    fig,ax=plt.subplots(2,3,figsize=(12,8.8),dpi=150)
    fig.suptitle('Numerical construction | source, manual addition, result',fontsize=18,y=.987)
    fig.text(.5,.94,'Display weights in [0,1]; white=1, black=0. The additions are explicitly synthetic.',ha='center',fontsize=11)
    for row,attr in enumerate(['margin','texture']):
        for col,(arr,label) in enumerate([(expand(results[attr]['retained']),'60% of ConceptCLIP'),(expand(results[attr]['added']),'manual addition'),(results[attr]['heat'],'final map')]):
            draw(ax[row,col],arr,f'{attr.title()} | {label}',gray=True)
    fig.subplots_adjust(left=.02,right=.98,top=.89,bottom=.035,hspace=.13,wspace=.06)
    fig.savefig(OUT/'LUNA16_283_map_components.png');plt.close(fig)
    checks['original_inputs_unchanged']=all(sha(Path(p))==h for p,h in before.items())
    (OUT/'construction.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
