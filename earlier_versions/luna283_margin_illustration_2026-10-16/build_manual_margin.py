"""Create a disclosed synthetic numeric heatmap and scientific comparison plots.

Original CT, reference masks, and model outputs are read-only inputs.
This creates manually chosen illustration weights, NOT Grad-CAM or ground truth.
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
OUT=Path(__file__).resolve().parent
GALLERY=ROOT/"shared_research/luna16_train_attribute_gallery_2026-10-15"
SOURCE=GALLERY/"assets/LUNA16_283/input.png"
MASK=GALLERY/"provenance/public_train/masks/LUNA16_283/union384.png"
ZOOM=(145,140,241,236)

# Zero-based patch row/column. Each patch occupies [14*c,14*(c+1)) x [14*r,14*(r+1)).
# These are deliberately chosen design parameters, not measurements of saliency.
PATCHES=[
    (11,12,0.18,"Weak superior spill; deliberate imperfection"),
    (12,11,0.28,"Upper-left perilesional transition; includes adjacent tissue"),
    (12,12,1.00,"Main emphasis: soft upper-left edge"),
    (12,13,0.55,"Partial upper edge; incomplete coverage"),
    (13,11,0.42,"Left transition with deliberate neighboring-tissue spill"),
    (13,12,0.62,"Left boundary; coarse patch also covers some lesion interior"),
    (14,11,0.12,"Weak lower-left spill"),
    (14,12,0.25,"Limited lower-left edge support"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay(image,heat):
    alpha=heat[...,None]*np.float32(0.7)
    return np.clip(image.astype(np.float32)*(1-alpha)+np.array([220,40,40],np.float32)*alpha,0,255).astype(np.uint8)


def panel(ax,array,title,zoom=False,heat=False):
    ax.imshow(array,interpolation="nearest",**({"cmap":"gray","vmin":0,"vmax":1} if heat else {}))
    if zoom:
        x0,y0,x1,y1=ZOOM
        ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_title(title,fontsize=12,pad=9)
    ax.axis("off")


def main():
    before={str(p):sha(p) for p in [SOURCE,MASK]}
    image=np.asarray(Image.open(SOURCE).convert("RGB"))
    assert image.shape==(384,384,3)
    grid=np.zeros((27,27),dtype=np.float32)
    for row,col,value,_ in PATCHES:
        grid[row,col]=value
    heat=np.zeros((384,384),dtype=np.float32)
    heat[:378,:378]=np.repeat(np.repeat(grid,14,axis=0),14,axis=1)
    rendered=overlay(image,heat)
    metadata={
        "case_id":"LUNA16_283","dataset":"LUNA16","split":"train","attribute":"margin",
        "artifact_type":"manually_constructed_synthetic_illustration",
        "is_model_output":False,"is_ground_truth":False,"clinically_validated":False,
        "source_image":str(SOURCE),"source_sha256":before[str(SOURCE)],
        "reference_mask":str(MASK),"reference_sha256":before[str(MASK)],
        "reference_use":"Visual orientation only. Lesion union is not an attribute mask; it was not converted to a margin target.",
        "definition":"Positive margin prompt means indistinct/poorly defined nodule-to-lung boundary, contrasted with a well-defined margin.",
        "local_definition":str(ROOT/"shared_research/luna16_user_study_review_2026-10-12/definitions.json"),
        "external_scale_reference":"https://pylidc.github.io/annotation.html#pylidc.Annotation.margin",
        "associated_reader_overall_margin_ratings":[3,3,4,4],
        "reader_rating_scope":"Overall nodule ratings, medium to near sharp; no claim that the entire boundary is indistinct or that readers annotated this local transition.",
        "prompt_reference":"p01; The lung nodule on this chest CT has an indistinct margin.",
        "construction":"Manual patch weights emphasizing the visually softer upper-left transition; all other weights zero.",
        "deliberate_limitations":["Incomplete boundary coverage", "Coarse 14x14 patches span both the edge and interior/background", "Small adjacent-tissue spill"],
        "intended_quality":"Useful with limitations, an informal one-step-below-ideal design target; no validated or assigned quality score.",
        "grid_hw":[27,27],"model_input_hw":[384,384],"patch_hw":[14,14],
        "uncovered_bottom_and_right_pixels":6,"coordinate_system":"384x384 model input, zero-based row and column; y increases downwards",
        "normalization":"Directly assigned float32 weights in [0,1]; not raw model activations, probabilities, or measured importance.",
        "overlay_color":[220,40,40],"overlay_alpha_scale":0.7,
        "patches":[{"row":r,"column":c,"weight":w,"reason":why} for r,c,w,why in PATCHES],
        "source_edits":False,
    }
    np.savez_compressed(OUT/"LUNA16_283_margin_MANUAL_ILLUSTRATION.npz",
        native_manual_weights=grid,model_input_heatmap_normalized=heat,metadata_json=json.dumps(metadata))
    plt.imsave(OUT/"LUNA16_283_margin_MANUAL_overlay.png",rendered)
    plt.imsave(OUT/"LUNA16_283_margin_MANUAL_grayscale.png",heat,cmap="gray",vmin=0,vmax=1)
    (OUT/"construction.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    # A full-field view and a nearest-neighbor zoom preserve the source coordinates.
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"#f7f7f5","axes.facecolor":"#f7f7f5"})
    fig,axes=plt.subplots(2,3,figsize=(12,8.8),dpi=160)
    fig.suptitle("LUNA16_283 | constructed margin example",fontsize=17,y=.988)
    fig.text(.5,.940,"MANUAL ILLUSTRATION  /  not a model output or an attribute ground truth",ha="center",fontsize=10,color="#8b2525")
    for row,zoom in enumerate([False,True]):
        panel(axes[row,0],image,"Unchanged CT" if not zoom else "Target zoom: unchanged CT",zoom)
        panel(axes[row,1],heat,"Hand-set numeric heatmap" if not zoom else "Same map: target zoom",zoom,True)
        panel(axes[row,2],rendered,"Red overlay: same display formula" if not zoom else "Sub-ideal example: target zoom",zoom)
    x0,y0,x1,y1=ZOOM
    axes[0,0].add_patch(Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor="#54d8e6",linewidth=1))
    fig.text(.5,.044,"Focus: the softer upper-left edge. Deliberate limits: missed boundary segments + adjacent-tissue spill.",ha="center",fontsize=11)
    fig.text(.5,.020,"27 x 27 grid; 14 x 14 pixel footprints; red RGB (220, 40, 40); alpha = 0.7 x weight. No smoothing.",ha="center",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.022,right=.978,bottom=.085,top=.90,hspace=.13,wspace=.055)
    fig.savefig(OUT/"LUNA16_283_margin_MANUAL_preview.png",dpi=160);plt.close(fig)
    # Compare unchanged existing p01 model outputs against the synthetic example.
    originals=[]
    original_overlay_matches=[]
    for model in ["base","moe29"]:
        file=ROOT/f"LUNA16 Attributes Localization masks TEST/Numeric/GALLERY_{model}_NUMERIC_ONLY_20261016/{model}/LUNA16/LUNA16_283/margin_p01_heatmap.npz"
        with np.load(file,allow_pickle=False) as data:
            originals.append(overlay(image,data["model_input_heatmap_normalized"]))
        name="ConceptCLIP" if model=="base" else "MOE29"
        original_overlay_matches.append(bool(np.array_equal(originals[-1],np.asarray(Image.open(GALLERY/f"assets/LUNA16_283/{name}_margin_p01_overlay.png").convert("RGB")))))
        metadata[f"comparison_{model}_numeric_file"]=str(file)
    fig,axes=plt.subplots(2,4,figsize=(15,8.5),dpi=150)
    fig.suptitle("LUNA16_283 | margin, p01 | existing models and a constructed example",fontsize=17,y=.987)
    fig.text(.5,.940,"The last column is hand-authored. It is not a rerun, correction, or validated replacement for either model.",ha="center",fontsize=10,color="#8b2525")
    for row,zoom in enumerate([False,True]):
        for col,(array,title) in enumerate(zip([image,*originals,rendered],["Unchanged CT","ConceptCLIP: original p01","MOE29: original p01","MANUAL: sub-ideal example"])):
            panel(axes[row,col],array,title if row==0 else "Target zoom",zoom)
    fig.text(.5,.025,"Same CT, grid geometry, red color and alpha rule in every overlay. The manual example narrows attention to the upper-left transition.",ha="center",fontsize=11)
    fig.subplots_adjust(left=.015,right=.985,bottom=.065,top=.89,hspace=.12,wspace=.045)
    fig.savefig(OUT/"LUNA16_283_margin_comparison_p01.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,7),dpi=160)
    ax.imshow(rendered,interpolation="nearest")
    for r,c,w,_ in PATCHES:
        ax.add_patch(Rectangle((14*c-.5,14*r-.5),14,14,fill=False,edgecolor="white",linewidth=.8,alpha=.8))
        ax.text(14*c+6.5,14*r+6.5,f"{w:.2f}",ha="center",va="center",color="white",fontsize=10,
                bbox={"facecolor":"black","edgecolor":"none","alpha":.4,"pad":1})
    ax.set(xlim=(145,241),ylim=(236,140),xlabel="Model-input x (pixels)",ylabel="Model-input y (pixels)",
           title="Manual patch weights | all unmarked patches = 0")
    fig.text(.5,.018,"Illustration weights, not model scores. Each white box is one 14 x 14 input patch.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(OUT/"LUNA16_283_margin_construction.png");plt.close(fig)
    checks={"native_grid_shape":list(grid.shape),"pixel_shape":list(heat.shape),
            "all_finite":bool(np.isfinite(heat).all()),"range":[float(heat.min()),float(heat.max())],
            "active_patches":int(np.count_nonzero(grid)),"active_pixels":int(np.count_nonzero(heat)),
            "comparison_original_model_overlays_exact":all(original_overlay_matches),
            "bottom_right_uncovered_border_zero":bool(not heat[378:,:].any() and not heat[:,378:].any()),
            "source_ct_and_reference_unchanged":all(sha(Path(p))==v for p,v in before.items())}
    with np.load(OUT/"LUNA16_283_margin_MANUAL_ILLUSTRATION.npz",allow_pickle=False) as data:
        checks["saved_arrays_reload_exact"]=bool(np.array_equal(data["native_manual_weights"],grid) and np.array_equal(data["model_input_heatmap_normalized"],heat))
    (OUT/"validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    (OUT/"construction.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
