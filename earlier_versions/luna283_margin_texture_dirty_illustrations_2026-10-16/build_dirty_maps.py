"""Reproducible, deliberately dirty manual scientific heatmaps for LUNA16_283.

All original inputs and earlier illustrations are read only. Manual and nuisance
weights are separately saved. No inference, model editing, or study insertion.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(r"D:\ConceptCLIP")
OUT=Path(__file__).resolve().parent
GALLERY=ROOT/"shared_research/luna16_train_attribute_gallery_2026-10-15"
INPUT=GALLERY/"assets/LUNA16_283/input.png"
ORIGINAL=GALLERY/"assets/LUNA16_283/source.png"
DEFINITIONS=ROOT/"shared_research/luna16_user_study_review_2026-10-12/definitions.json"
PREVIOUS=ROOT/"shared_research/luna283_margin_illustration_2026-10-16/LUNA16_283_margin_MANUAL_ILLUSTRATION.npz"
ZOOM=(145,140,241,236)

# Zero-based (row,column,weight), manually chosen after inspecting source and input CT.
TARGETS={
    "margin":[(11,12,.24),(12,11,.34),(12,12,1.0),(12,13,.55),
              (13,11,.47),(13,12,.66),(14,11,.20),(14,12,.32),
              (11,13,.17),(12,14,.24),(14,13,.20),(15,12,.16)],
    "texture":[(12,12,.28),(12,13,.52),(13,12,.64),(13,13,1.0),
               (13,14,.69),(14,12,.48),(14,13,.89),(14,14,.74),
               (15,12,.17),(15,13,.29),(15,14,.24)],
}
# Deliberately irrelevant clusters over remote lung, vessels and chest-wall tissue.
# These are imperfections, not additional attribute evidence.
CLUSTERS={
    "margin":[(14,5,.42),(20,10,.34),(22,5,.30),(8,17,.38),(17,14,.29)],
    "texture":[(15,7,.39),(21,11,.43),(23,6,.28),(9,16,.34),(18,4,.29)],
}
SEEDS={"margin":28361,"texture":28362}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expand(grid):
    pixel=np.zeros((384,384),np.float32)
    pixel[:378,:378]=np.repeat(np.repeat(grid,14,axis=0),14,axis=1)
    return pixel


def blend(image,heat):
    alpha=heat[...,None]*np.float32(.7)
    return np.clip(image.astype(np.float32)*(1-alpha)+np.array([220,40,40],np.float32)*alpha,0,255).astype(np.uint8)


def make_grid(attribute,image):
    target=np.zeros((27,27),np.float32)
    dirt=np.zeros_like(target)
    for r,c,w in TARGETS[attribute]:
        target[r,c]=w
    rng=np.random.default_rng(SEEDS[attribute])
    candidates=[]
    for r in range(6,25):
        for c in range(1,25):
            # Exclude the target neighborhood from random remote distractions.
            if 10<=r<=16 and 10<=c<=16:
                continue
            # Restrict scatter to visible anatomy; intensity is used only for this
            # display constraint, not as an estimator of attribute importance.
            if image[r*14:(r+1)*14,c*14:(c+1)*14].mean()>8:
                candidates.append((r,c))
    chosen=rng.choice(len(candidates),size=48,replace=False)
    for idx in chosen:
        r,c=candidates[idx]
        dirt[r,c]=rng.uniform(.055,.19)
    for r,c,w in CLUSTERS[attribute]:
        for dr,dc,scale in [(0,0,1),(-1,0,.50),(0,1,.62),(1,0,.38),(0,-1,.47)]:
            rr,cc=r+dr,c+dc
            if not (10<=rr<=16 and 10<=cc<=16):
                dirt[rr,cc]=max(float(dirt[rr,cc]),w*scale)
    grid=np.maximum(target,dirt)
    return grid,target,dirt


def draw(ax,data,title,zoom=False,is_heat=False):
    ax.imshow(data,interpolation="nearest",**({"cmap":"gray","vmin":0,"vmax":1} if is_heat else {}))
    if zoom:
        x0,y0,x1,y1=ZOOM
        ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_title(title,fontsize=12,pad=9)
    ax.axis("off")


def main():
    hashes={str(p):sha(p) for p in [INPUT,ORIGINAL,DEFINITIONS,PREVIOUS]}
    image=np.array(Image.open(INPUT).convert("RGB"))
    original=np.array(Image.open(ORIGINAL).convert("RGB"))
    assert image.shape==(384,384,3) and original.shape==(256,256,3)
    definitions={v["id"]:v for v in json.loads(DEFINITIONS.read_text())["attributes"]}
    metadata={}
    previews={}
    checks={}
    for attr in ["margin","texture"]:
        grid,target,dirt=make_grid(attr,image)
        heat=expand(grid)
        rendered=blend(image,heat)
        info={
            "case_id":"LUNA16_283","dataset":"LUNA16","split":"train","attribute":attr,
            "version":"dirty_manual_v2" if attr=="margin" else "dirty_manual_v1",
            "artifact_type":"synthetic_manual_illustration","is_model_output":False,
            "is_ground_truth":False,"clinically_validated":False,"quality_score":None,
            "intended_quality":"Deliberately sub-ideal and dirty, with attribute-oriented main emphasis and intentional distractors; not a validated rubric score.",
            "attribute_definition":definitions[attr],
            "primary_focus":"Selected softer upper-left nodule-lung transition" if attr=="margin" else "Visible dense nodule interior / solid-appearing component",
            "limitations":["Incomplete cue coverage", "Coarse patches span several structures", "Manual adjacent-tissue spill", "Random low-level remote activation", "Five deliberately irrelevant clusters"],
            "source_ct":str(ORIGINAL),"source_ct_sha256":hashes[str(ORIGINAL)],
            "model_input_ct":str(INPUT),"model_input_ct_sha256":hashes[str(INPUT)],
            "source_hw":[256,256],"model_input_hw":[384,384],"native_grid_hw":[27,27],
            "patch_hw":[14,14],"uncovered_trailing_rows_and_columns":6,
            "coordinate_system":"Original uncropped field of view at existing 384x384 model-input resolution; zero-based patch rows and columns",
            "weights":"Directly assigned [0,1] display emphasis, not probabilities or measured model saliency",
            "combination":"maximum(manual_target_weights,manual_distractor_weights); no renormalization",
            "random_seed":SEEDS[attr],"random_patch_count_before_cluster_overwrite":48,
            "random_weight_range":[.055,.19],"random_placement_exclusion":"rows10..16 AND columns10..16, inclusive",
            "random_placement_intensity_filter":"mean patch RGB >8, used only to restrict distractions to visible anatomy",
            "target_patches":[{"row":r,"column":c,"weight":w} for r,c,w in TARGETS[attr]],
            "distractor_cluster_centers":[{"row":r,"column":c,"weight":w} for r,c,w in CLUSTERS[attr]],
            "overlay_color":[220,40,40],"overlay_alpha_scale":.7,
            "rendering":"float32 blend input*(1-.7*heat)+[220,40,40]*(.7*heat), clip0..255 then truncate uint8; no smoothing",
            "original_inputs_changed":False,"inserted_into_study":False,
        }
        stem=f"LUNA16_283_{attr}_DIRTY_MANUAL"
        np.savez_compressed(OUT/f"{stem}.npz",native_manual_weights=grid,
            manual_target_weights=target,manual_distractor_weights=dirt,
            model_input_heatmap_normalized=heat,metadata_json=json.dumps(info))
        plt.imsave(OUT/f"{stem}_overlay.png",rendered)
        # Use the exact uint8(255*h) convention of the model's grayscale exports.
        gray=np.repeat((heat*255).astype(np.uint8)[...,None],3,axis=2)
        plt.imsave(OUT/f"{stem}_heatmap.png",gray)
        with np.load(OUT/f"{stem}.npz",allow_pickle=False) as z:
            exact=bool(np.array_equal(z["native_manual_weights"],grid) and np.array_equal(z["model_input_heatmap_normalized"],heat))
        saved=np.array(Image.open(OUT/f"{stem}_overlay.png").convert("RGB"))
        checks[attr]={"finite":bool(np.isfinite(heat).all()),"range":[float(heat.min()),float(heat.max())],
            "grid_shape":list(grid.shape),"pixel_shape":list(heat.shape),
            "main_patches":int(np.count_nonzero(target)),"distractor_patches":int(np.count_nonzero(dirt)),
            "total_nonzero_patches":int(np.count_nonzero(grid)),
            "strongest_target":float(target.max()),"strongest_distractor":float(dirt.max()),
            "border_zero":bool(not heat[378:,:].any() and not heat[:,378:].any()),
            "saved_numeric_exact":exact,"saved_overlay_exact":bool(np.array_equal(saved,rendered)),
            "zero_weight_pixels_equal_input":bool(np.array_equal(saved[heat==0],image[heat==0]))}
        metadata[attr]=info
        previews[attr]=(heat,rendered,target,dirt)
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"#f7f7f5"})
    fig,axes=plt.subplots(2,3,figsize=(12,9),dpi=160)
    fig.suptitle("LUNA16_283 | dirtier manual examples",fontsize=18,y=.987)
    fig.text(.5,.941,"SYNTHETIC ILLUSTRATIONS  /  intentional spill and off-target activation  /  not model outputs",ha="center",fontsize=10,color="#8b2525")
    for row,zoom in enumerate([False,True]):
        draw(axes[row,0],image,"Original CT at model-input resolution" if row==0 else "Target zoom: unchanged CT",zoom)
        draw(axes[row,1],previews["margin"][1],"Margin: indistinct-edge emphasis" if row==0 else "Margin: incomplete edge + spill",zoom)
        draw(axes[row,2],previews["texture"][1],"Texture: interior emphasis" if row==0 else "Texture: interior + spill",zoom)
    fig.text(.5,.043,"Main cue stays strongest; weaker remote patches are deliberately distracting, not extra attribute evidence.",ha="center",fontsize=10.5)
    fig.text(.5,.019,"27 x 27 grid  |  14 x 14 input-pixel patches  |  RGB (220,40,40), alpha = 0.7 x weight  |  no smoothing",ha="center",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.02,right=.98,bottom=.085,top=.90,wspace=.055,hspace=.14)
    fig.savefig(OUT/"LUNA16_283_MARGIN_TEXTURE_DIRTY_preview.png");plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8.8),dpi=150)
    fig.suptitle("How the dirty maps were constructed",fontsize=18,y=.985)
    fig.text(.5,.94,"Hand-set emphasis and added distractions are separately available in the numeric files.",ha="center",fontsize=11)
    for row,attr in enumerate(["margin","texture"]):
        for col,(arr,label) in enumerate([(expand(previews[attr][2]),"main emphasis"),(expand(previews[attr][3]),"intentional distractions"),(previews[attr][0],"combined map")]):
            draw(axes[row,col],arr,f"{attr.title()}: {label}",is_heat=True)
    fig.text(.5,.025,"Display weights only. White = 1; black = 0. combined = maximum(main, distractions).",ha="center",fontsize=11)
    fig.subplots_adjust(left=.025,right=.975,top=.89,bottom=.07,hspace=.12,wspace=.055)
    fig.savefig(OUT/"LUNA16_283_dirty_map_components.png");plt.close(fig)
    checks["source_files_and_previous_illustration_unchanged"]=all(sha(Path(p))==v for p,v in hashes.items())
    (OUT/"construction.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (OUT/"validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
