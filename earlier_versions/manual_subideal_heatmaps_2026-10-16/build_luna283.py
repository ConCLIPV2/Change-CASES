"""Plot disclosed manually constructed margin/texture heatmaps for LUNA16_283.

Uses scientific plotting from numeric arrays. No inference or input-file edits.
The approved prior margin grid is reused exactly; texture is manually specified.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(r"D:\ConceptCLIP")
OUT=Path(__file__).resolve().parent/"LUNA16_283"
GALLERY=ROOT/"shared_research/luna16_train_attribute_gallery_2026-10-15"
INPUT=GALLERY/"assets/LUNA16_283/input.png"
SOURCE=GALLERY/"assets/LUNA16_283/source.png"
REFERENCE=GALLERY/"provenance/public_train/masks/LUNA16_283/union384.png"
DEFINITIONS=ROOT/"shared_research/luna16_user_study_review_2026-10-12/definitions.json"
PRIOR=ROOT/"shared_research/luna283_margin_illustration_2026-10-16/LUNA16_283_margin_MANUAL_ILLUSTRATION.npz"
ZOOM=(145,140,241,236)
TEXTURE_PATCHES=[
    (12,12,.22,"Weak superior-left coverage; part of the upper extension remains poorly covered"),
    (12,13,.38,"Partial upper-body support"),
    (13,12,.50,"Visible left intranodular body with coarse edge mixing"),
    (13,13,.78,"Strong central intranodular opacity support"),
    (13,14,.30,"Weaker upper-right coverage with adjacent-tissue spill"),
    (14,12,.32,"Incomplete lower-left interior support"),
    (14,13,1.00,"Strongest weight on the visible lower-central nodule body"),
    (14,14,.68,"Visible lower-right body, with edge spill from coarse patch geometry"),
    (15,13,.12,"Deliberate weak inferior spill outside the nodule body"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(grid):
    heat=np.zeros((384,384),dtype=np.float32)
    heat[:378,:378]=np.repeat(np.repeat(grid,14,axis=0),14,axis=1)
    return heat


def render(image,heat):
    alpha=np.float32(.7)*heat[...,None]
    return np.clip(image.astype(np.float32)*(1-alpha)+np.array([220,40,40],dtype=np.float32)*alpha,0,255).astype(np.uint8)


def panel(ax,data,title,zoom=False,gray=False):
    ax.imshow(data,interpolation="nearest",**({"cmap":"gray","vmin":0,"vmax":1} if gray else {}))
    if zoom:
        x0,y0,x1,y1=ZOOM
        ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_title(title,fontsize=12,pad=10);ax.axis("off")


def figure_layout(fig,headline,footer):
    fig.suptitle(headline,fontsize=17,y=.988)
    fig.text(.5,.941,"MANUAL EXAMPLES  |  hand-set weights, not model outputs or attribute ground truth",ha="center",fontsize=10,color="#8b2525")
    fig.text(.5,.043,footer,ha="center",fontsize=10)
    fig.text(.5,.020,"27 x 27 grid | 14 x 14 pixel footprints | red RGB (220,40,40) | alpha = 0.7 x weight | no smoothing",ha="center",fontsize=9,color="#555555")
    fig.subplots_adjust(left=.022,right=.978,bottom=.085,top=.895,hspace=.14,wspace=.055)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[INPUT,SOURCE,REFERENCE,DEFINITIONS,PRIOR]
    hashes={str(p):sha(p) for p in inputs}
    image=np.asarray(Image.open(INPUT).convert("RGB"))
    source=np.asarray(Image.open(SOURCE).convert("RGB"))
    assert image.shape==(384,384,3)
    definitions={d["id"]:d for d in json.loads(DEFINITIONS.read_text())["attributes"]}
    with np.load(PRIOR,allow_pickle=False) as z:
        margin=z["native_manual_weights"].copy()
        margin_pixel=z["model_input_heatmap_normalized"].copy()
        prior_meta=json.loads(str(z["metadata_json"]))
    texture=np.zeros((27,27),dtype=np.float32)
    for row,col,weight,_ in TEXTURE_PATCHES:
        texture[row,col]=weight
    maps={"margin":margin,"texture":texture}
    descriptions={
        "margin":{"focus":"Selected softer upper-left nodule-to-lung transition",
                  "limits":"Incomplete boundary coverage and adjacent-tissue spill",
                  "patches":prior_meta["patches"]},
        "texture":{"focus":"Visible intranodular opacity, especially the central and lower body",
                   "limits":"Uneven interior coverage, weak upper-extension support, and minor boundary spill",
                   "patches":[{"row":r,"column":c,"weight":w,"reason":why} for r,c,w,why in TEXTURE_PATCHES]},
    }
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"#f7f7f5","axes.facecolor":"#f7f7f5"})
    overlays={};validation={};provenance={}
    for attribute,grid in maps.items():
        heat=expand(grid);overlays[attribute]=render(image,heat)
        meta={"case_id":"LUNA16_283","dataset":"LUNA16","split":"train","attribute":attribute,
              "artifact_type":"manually_constructed_synthetic_illustration","is_model_output":False,
              "is_ground_truth":False,"clinically_validated":False,
              "definition":definitions[attribute],"definition_source":str(DEFINITIONS),
              "original_source_image":str(SOURCE),"original_source_shape":list(source.shape),
              "overlay_coordinate_image":str(INPUT),"overlay_coordinate_image_shape":list(image.shape),
              "source_image_sha256":hashes[str(SOURCE)],"model_input_image_sha256":hashes[str(INPUT)],
              "reference_use":"Visual orientation only; the lesion contour is not an attribute-specific mask.",
              "reference_path":str(REFERENCE),"grid_hw":[27,27],"model_input_hw":[384,384],
              "patch_hw":[14,14],"trailing_uncovered_rows_columns":6,
              "overlay_rgb":[220,40,40],"overlay_alpha_scale":.7,
              "numeric_semantics":"Directly assigned relative weights, not confidence, probability, measured saliency, or model activations.",
              "quality_target":"Deliberately sub-ideal illustrative guidance; not a calibrated or assigned rubric score.",
              **descriptions[attribute]}
        if attribute=="margin":
            meta["approved_previous_map"]=str(PRIOR)
            meta["approved_previous_map_sha256"]=hashes[str(PRIOR)]
            meta["approved_previous_numeric_arrays_reused_exactly"]=bool(np.array_equal(heat,margin_pixel))
        prefix=f"LUNA16_283_{attribute}_MANUAL"
        np.savez_compressed(OUT/f"{prefix}.npz",native_manual_weights=grid,
                            model_input_heatmap_normalized=heat,metadata_json=json.dumps(meta))
        plt.imsave(OUT/f"{prefix}_overlay.png",overlays[attribute])
        # uint8 conversion matches the existing gray heatmap export exactly.
        plt.imsave(OUT/f"{prefix}_grayscale.png",np.repeat((heat*255).astype(np.uint8)[...,None],3,axis=2))
        (OUT/f"{prefix}_construction.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
        fig,axes=plt.subplots(2,3,figsize=(12,8.8),dpi=160)
        for row,zoom in enumerate([False,True]):
            panel(axes[row,0],image,"Unchanged CT input" if not zoom else "CT target zoom",zoom)
            panel(axes[row,1],heat,f"{attribute.title()}: numeric heatmap" if not zoom else "Heatmap target zoom",zoom,True)
            panel(axes[row,2],overlays[attribute],f"Manual {attribute} overlay" if not zoom else "Overlay target zoom",zoom)
        figure_layout(fig,f"LUNA16_283 | manual {attribute} example",descriptions[attribute]["focus"]+".")
        fig.savefig(OUT/f"{prefix}_preview.png");plt.close(fig)
        with np.load(OUT/f"{prefix}.npz",allow_pickle=False) as z:
            reload_exact=np.array_equal(z["native_manual_weights"],grid) and np.array_equal(z["model_input_heatmap_normalized"],heat)
        saved_overlay=np.asarray(Image.open(OUT/f"{prefix}_overlay.png").convert("RGB"))
        saved_gray=np.asarray(Image.open(OUT/f"{prefix}_grayscale.png").convert("RGB"))
        validation[attribute]={"native_shape":list(grid.shape),"pixel_shape":list(heat.shape),
            "finite_and_in_range":bool(np.isfinite(heat).all() and heat.min()>=0 and heat.max()<=1),
            "active_patches":int(np.count_nonzero(grid)),"arrays_reload_exact":bool(reload_exact),
            "saved_overlay_exact":bool(np.array_equal(saved_overlay,overlays[attribute])),
            "saved_grayscale_exact":bool(np.array_equal(saved_gray,np.repeat((heat*255).astype(np.uint8)[...,None],3,axis=2))),
            "uncovered_border_zero":bool(not heat[378:,:].any() and not heat[:,378:].any())}
        provenance[attribute]=meta
    validation["approved_margin_arrays_unchanged"]=bool(np.array_equal(expand(margin),margin_pixel))
    # Combined figure: both attributes on the identical CT coordinate system.
    fig,axes=plt.subplots(2,3,figsize=(12,8.8),dpi=160)
    for row,zoom in enumerate([False,True]):
        panel(axes[row,0],image,"Unchanged CT input" if not zoom else "CT target zoom",zoom)
        panel(axes[row,1],overlays["margin"],"Margin: softer upper-left transition" if not zoom else "Margin: target zoom",zoom)
        panel(axes[row,2],overlays["texture"],"Texture: visible nodule interior" if not zoom else "Texture: target zoom",zoom)
    figure_layout(fig,"LUNA16_283 | manual sub-ideal margin and texture","Margin retains partial coverage and spill. Texture retains uneven interior coverage and minor spill.")
    fig.savefig(OUT/"LUNA16_283_margin_texture_MANUAL_comparison.png");plt.close(fig)
    # Exact patch-placement explanation for both manual maps.
    fig,axes=plt.subplots(1,2,figsize=(11.5,6),dpi=160)
    for ax,attribute in zip(axes,["margin","texture"]):
        ax.imshow(overlays[attribute],interpolation="nearest")
        for patch in descriptions[attribute]["patches"]:
            r,c,w=patch["row"],patch["column"],patch["weight"]
            ax.add_patch(Rectangle((14*c-.5,14*r-.5),14,14,fill=False,edgecolor="white",linewidth=.65))
            ax.text(14*c+6.5,14*r+6.5,f"{w:.2f}",ha="center",va="center",color="white",fontsize=8,
                    bbox={"facecolor":"black","edgecolor":"none","alpha":.45,"pad":1})
        ax.set(xlim=(145,241),ylim=(236,140),xlabel="Model-input x",ylabel="Model-input y",title=attribute.title()+": hand-set patch weights")
    fig.suptitle("LUNA16_283 | numeric construction",fontsize=17,y=.99)
    fig.text(.5,.025,"All unmarked patches = 0. Weights are design choices, not model confidence or measured clinical evidence.",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.11,1,.94]);fig.savefig(OUT/"LUNA16_283_margin_texture_MANUAL_weights.png");plt.close(fig)
    validation["all_original_inputs_unchanged"]=all(sha(Path(p))==v for p,v in hashes.items())
    (OUT/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    (OUT/"source_hashes.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print(json.dumps(validation,indent=2))


if __name__=="__main__":
    main()
