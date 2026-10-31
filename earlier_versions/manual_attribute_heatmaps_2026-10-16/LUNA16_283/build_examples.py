"""Numeric scientific illustrations for LUNA16_283; original inputs are read-only."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(r"D:\ConceptCLIP")
OUT=Path(__file__).resolve().parent
PREVIOUS=ROOT/"shared_research/luna283_margin_illustration_2026-10-16"
ASSETS=ROOT/"shared_research/luna16_train_attribute_gallery_2026-10-15/assets/LUNA16_283"
INPUT=ASSETS/"input.png"
SOURCE=ASSETS/"source.png"
DEFINITION=ROOT/"shared_research/luna16_user_study_review_2026-10-12/definitions.json"
TEXTURE_PATCHES=[
    (12,13,.28,"Weak upper interior; apex incompletely covered"),
    (13,12,.52,"Left interior and coarse edge overlap"),
    (13,13,1.00,"Main visible solid-appearing interior"),
    (13,14,.44,"Partial right interior/edge coverage"),
    (14,12,.40,"Lower-left interior"),
    (14,13,.86,"Lower central interior"),
    (14,14,.68,"Lower-right solid-appearing body"),
    (14,15,.16,"Deliberate weak right-edge spill"),
    (15,13,.12,"Deliberate weak inferior spill"),
]
ZOOM=(145,140,241,236)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def render(image,heat):
    alpha=np.asarray(heat,np.float32)[...,None]*np.float32(.7)
    return np.clip(image.astype(np.float32)*(1-alpha)+np.array([220,40,40],np.float32)*alpha,0,255).astype(np.uint8)

def panel(ax,array,title,zoom=False,gray=False):
    ax.imshow(array,interpolation="nearest",**({"cmap":"gray","vmin":0,"vmax":1} if gray else {}))
    if zoom:
        x0,y0,x1,y1=ZOOM;ax.set_xlim(x0-.5,x1-.5);ax.set_ylim(y1-.5,y0-.5)
    ax.set_title(title,fontsize=12,pad=9);ax.axis("off")

def main():
    input_hashes={str(p):sha(p) for p in [INPUT,SOURCE,DEFINITION]}
    image=np.asarray(Image.open(INPUT).convert("RGB"))
    assert image.shape==(384,384,3)
    # Preserve the accepted margin example exactly, including its numeric archive.
    for suffix in ["ILLUSTRATION.npz","overlay.png","grayscale.png"]:
        name=f"LUNA16_283_margin_MANUAL_{suffix}"
        shutil.copy2(PREVIOUS/name,OUT/name)
        assert sha(PREVIOUS/name)==sha(OUT/name)
    with np.load(OUT/"LUNA16_283_margin_MANUAL_ILLUSTRATION.npz",allow_pickle=False) as z:
        margin=z["model_input_heatmap_normalized"];margin_grid=z["native_manual_weights"]
    grid=np.zeros((27,27),np.float32)
    for row,col,w,_ in TEXTURE_PATCHES:grid[row,col]=w
    heat=np.zeros((384,384),np.float32)
    heat[:378,:378]=np.repeat(np.repeat(grid,14,axis=0),14,axis=1)
    definition=next(a for a in json.loads(DEFINITION.read_text())["attributes"] if a["id"]=="texture")
    meta={"case_id":"LUNA16_283","dataset":"LUNA16","split":"train","attribute":"texture",
        "artifact_type":"manually_constructed_synthetic_illustration","is_model_output":False,
        "is_ground_truth":False,"clinically_validated":False,
        "definition":definition,"definition_source":str(DEFINITION),
        "source_ct_inspected":str(SOURCE),"source_ct_sha256":sha(SOURCE),
        "unchanged_model_input":str(INPUT),"model_input_sha256":sha(INPUT),
        "coordinate_system":"384x384 model input; zero-based row/column, y down",
        "native_grid_hw":[27,27],"patch_hw":[14,14],"uncovered_bottom_right_pixels":6,
        "weights":"Directly designed float32 emphasis in [0,1]; not model scores or probabilities",
        "intended_quality":"Sub-ideal discussion example; no assigned quality rating",
        "chosen_cue":"Visible solid-appearing nodule interior, with stronger central/lower body emphasis",
        "deliberate_imperfections":["Incomplete upper-apex coverage","Uneven interior emphasis","Weak nearby right/inferior spill"],
        "overlay_color":[220,40,40],"alpha_scale":.7,"smoothing":False,
        "patches":[{"row":r,"column":c,"weight":w,"reason":s} for r,c,w,s in TEXTURE_PATCHES],
        "render_formula":"uint8(clip(input_rgb*(1-0.7*h) + [220,40,40]*(0.7*h),0,255)) using float32",
        "limitations":"Radiographic appearance in saved CT PNG; does not distinguish solid from part-solid or establish calcium/diagnosis."}
    np.savez_compressed(OUT/"LUNA16_283_texture_MANUAL_ILLUSTRATION.npz",
        native_manual_weights=grid,model_input_heatmap_normalized=heat,metadata_json=json.dumps(meta))
    (OUT/"texture_construction.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    margin_overlay=render(image,margin);texture_overlay=render(image,heat)
    plt.imsave(OUT/"LUNA16_283_texture_MANUAL_overlay.png",texture_overlay)
    plt.imsave(OUT/"LUNA16_283_texture_MANUAL_grayscale.png",heat,cmap="gray",vmin=0,vmax=1)
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"#f7f7f5","axes.facecolor":"#f7f7f5"})
    fig,axes=plt.subplots(2,3,figsize=(12,8.8),dpi=160)
    fig.suptitle("LUNA16_283 | manual margin and texture examples",fontsize=17,y=.988)
    fig.text(.5,.940,"MANUAL ILLUSTRATIONS  /  deliberately imperfect  /  not model predictions",ha="center",fontsize=10,color="#8b2525")
    for row,zoom in enumerate([False,True]):
        for col,(a,title) in enumerate([(image,"Unchanged CT input"),(margin_overlay,"Margin: softer upper-left transition"),(texture_overlay,"Texture: solid-appearing interior")]):
            panel(axes[row,col],a,title if row==0 else "Target zoom",zoom)
    fig.text(.5,.044,"Margin example retained unchanged. Texture adds uneven interior emphasis and modest nearby spill.",ha="center",fontsize=11)
    fig.text(.5,.020,"27 x 27 grid; 14 x 14 pixel footprints; red RGB (220,40,40); alpha = 0.7 x weight; no smoothing.",ha="center",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.022,right=.978,bottom=.085,top=.90,hspace=.13,wspace=.055)
    fig.savefig(OUT/"LUNA16_283_margin_texture_MANUAL_preview.png");plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8.8),dpi=150)
    fig.suptitle("LUNA16_283 | manual texture example",fontsize=17,y=.988)
    fig.text(.5,.940,"Constructed weights for discussion; not a model output or an attribute annotation",ha="center",fontsize=10,color="#8b2525")
    for row,zoom in enumerate([False,True]):
        panel(axes[row,0],image,"Unchanged CT" if not zoom else "Target zoom",zoom)
        panel(axes[row,1],heat,"Manual numeric heatmap" if not zoom else "Same heatmap: target zoom",zoom,True)
        panel(axes[row,2],texture_overlay,"Manual texture overlay" if not zoom else "Same overlay: target zoom",zoom)
    fig.text(.5,.03,"Visible nodule interior receives emphasis; upper coverage is incomplete and weak lower/right spill remains.",ha="center",fontsize=11)
    fig.subplots_adjust(left=.022,right=.978,bottom=.075,top=.90,hspace=.13,wspace=.055)
    fig.savefig(OUT/"LUNA16_283_texture_MANUAL_preview.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,6),dpi=150)
    for ax,weights,ov,title in [(axes[0],margin_grid,margin_overlay,"Margin: accepted manual weights"),(axes[1],grid,texture_overlay,"Texture: new manual weights")]:
        ax.imshow(ov,interpolation="nearest")
        for r,c in zip(*np.nonzero(weights)):
            ax.add_patch(Rectangle((14*c-.5,14*r-.5),14,14,fill=False,edgecolor="white",linewidth=.7))
            ax.text(14*c+6.5,14*r+6.5,f"{weights[r,c]:.2f}",color="white",ha="center",va="center",fontsize=9,bbox={"facecolor":"black","edgecolor":"none","alpha":.45,"pad":1})
        x0,y0,x1,y1=ZOOM;ax.set(xlim=(x0-.5,x1-.5),ylim=(y1-.5,y0-.5),title=title,xlabel="Model-input x",ylabel="Model-input y")
    fig.text(.5,.025,"All unmarked patches = 0. These are manually chosen emphasis weights, not model scores.",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(OUT/"LUNA16_283_manual_patch_weights.png");plt.close(fig)
    checks={"source_files_unchanged":all(sha(Path(p))==h for p,h in input_hashes.items()),
        "margin_archive_byte_identical_to_accepted_example":sha(OUT/"LUNA16_283_margin_MANUAL_ILLUSTRATION.npz")==sha(PREVIOUS/"LUNA16_283_margin_MANUAL_ILLUSTRATION.npz"),
        "margin_overlay_pixels_exact":bool(np.array_equal(np.asarray(Image.open(OUT/"LUNA16_283_margin_MANUAL_overlay.png").convert("RGB")),margin_overlay)),
        "texture_overlay_saved_pixels_exact":bool(np.array_equal(np.asarray(Image.open(OUT/"LUNA16_283_texture_MANUAL_overlay.png").convert("RGB")),texture_overlay)),
        "texture_finite_in_range":bool(np.isfinite(heat).all() and heat.min()>=0 and heat.max()<=1),
        "texture_native_shape":list(grid.shape),"texture_pixel_shape":list(heat.shape),
        "texture_nonzero_patches":int(np.count_nonzero(grid)),
        "texture_border_zero":bool(not heat[378:,:].any() and not heat[:,378:].any())}
    with np.load(OUT/"LUNA16_283_texture_MANUAL_ILLUSTRATION.npz",allow_pickle=False) as z:
        checks["texture_archive_round_trip_exact"]=bool(np.array_equal(z["native_manual_weights"],grid) and np.array_equal(z["model_input_heatmap_normalized"],heat))
    assert all(v for v in checks.values())
    (OUT/"validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    (OUT/"README.txt").write_text("LUNA16_283: manual sub-ideal Margin and Texture examples\n\nMargin is copied byte-for-byte from the previously accepted example.\nTexture was manually constructed after inspecting the original 256x256 source CT and its unchanged 384x384 model input.\n\nMargin emphasizes the softer upper-left transition; texture emphasizes the visible solid-appearing interior.\nTexture deliberately has uneven coverage, a weak upper apex, and modest right/inferior spill.\nWeights are illustrative choices, not model scores, measured quality, or attribute ground truth.\n\nBoth use a 27x27 float32 grid, 14x14 model-input patch footprints, six trailing zero rows/columns, and RGB(220,40,40) with alpha=0.7*weight.\nStandalone overlays are on the unchanged 384x384 CT model input so the patch coordinates match your existing numeric maps.\n\nThe NPZs contain native_manual_weights, model_input_heatmap_normalized, metadata_json. Read with allow_pickle=False.\nThe raw CT, original maps, reference masks, study files, and earlier accepted example were not modified.\n\nReproduce with: python build_examples.py\nLocal definitions: "+str(DEFINITION)+"\n",encoding="utf-8")
    print(json.dumps(checks,indent=2))

if __name__=="__main__":main()
