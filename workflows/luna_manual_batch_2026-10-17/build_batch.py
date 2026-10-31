from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from case_paths import workspace_root, output_dir, relocate
"""Reproducible manual heatmap illustrations. All source model files are read-only.

Numerical edits follow reviewed, case-specific patch plans. They are explicitly
manual derivatives and are never written into the original model/study folders.
"""
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

ROOT=workspace_root()
PLAN_ROOT=Path(__file__).resolve().parent
OUT=output_dir('luna_manual_batch_2026-10-17')
NUMERIC = ROOT / 'LUNA16 Attributes Localization masks TEST/Numeric'
DEFS = ROOT / 'shared_research/luna16_user_study_review_2026-10-12/definitions.json'
RETAIN = np.float32(.75)
NAMES = {'base': 'ConceptCLIP', 'moe29': 'MOE29'}


def read_json(path):
    return relocate(json.loads(path.read_text(encoding='utf-8-sig')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expand(grid):
    heat = np.zeros((384, 384), np.float32)
    heat[:378, :378] = np.repeat(np.repeat(grid, 14, axis=0), 14, axis=1)
    return heat


def overlay(image, heat):
    alpha = np.float32(.7) * heat[..., None]
    return np.clip(image.astype(np.float32) * (1-alpha) +
                   np.array([220, 40, 40], np.float32) * alpha, 0, 255).astype(np.uint8)


def distractors(image, zoom, seed):
    """Structured nuisance patches, separate from the manually interpreted cue.

    Choose five separated clusters and fourteen sparse patches outside the clean
    target zoom. An intensity filter avoids completely black/white surroundings;
    it is not segmentation or evidence that these areas express the attribute.
    """
    rng = np.random.default_rng(seed)
    centers = np.arange(27) * 14 + 6.5
    yy, xx = np.meshgrid(centers, centers, indexing='ij')
    means = image[:378, :378].mean(axis=2).reshape(27, 14, 27, 14).mean(axis=(1, 3))
    x0, y0, x1, y1 = zoom
    # Keep a full additional patch between distractors and the displayed target.
    excluded = (xx >= x0-14) & (xx <= x1+14) & (yy >= y0-14) & (yy <= y1+14)
    eligible = (means > 28) & (means < 235) & ~excluded
    eligible[:2, :] = False; eligible[-2:, :] = False
    eligible[:, :2] = False; eligible[:, -2:] = False
    choices = np.argwhere(eligible)
    assert len(choices) >= 20
    grid = np.zeros((27, 27), np.float32)
    selected = []
    for index in rng.permutation(len(choices)):
        r, c = map(int, choices[index])
        if any((r-a)**2 + (c-b)**2 < 25 for a, b in selected):
            continue
        selected.append((r, c))
        peak = float(rng.uniform(.48, .60))
        grid[r, c] = peak
        for dr, dc, factor in [(-1, 0, .67), (0, -1, .81), (1, 0, .77),
                               (0, 1, .62), (1, 1, .44), (-1, -1, .38)]:
            rr, cc = r+dr, c+dc
            if eligible[rr, cc] and rng.random() > .18:
                grid[rr, cc] = max(grid[rr, cc], peak*factor*float(rng.uniform(.8, 1.08)))
        if len(selected) == 5:
            break
    scatter = np.argwhere(eligible & (grid == 0))
    for index in rng.choice(len(scatter), size=min(14, len(scatter)), replace=False):
        r, c = scatter[index]
        grid[r, c] = float(rng.uniform(.12, .30))
    assert len(selected) == 5 and not grid[excluded].any()
    return grid, {'seed': seed, 'cluster_centers_row_column': selected,
                  'cluster_count': len(selected), 'sparse_patch_count': 14,
                  'nonzero_patches': int(np.count_nonzero(grid)),
                  'actual_max': float(grid.max()),
                  'target_zoom_exclusion_padding_pixels': 14}


def draw(ax, image, title, zoom=None):
    ax.imshow(image, interpolation='nearest')
    if zoom:
        x0, y0, x1, y1 = zoom
        ax.set_xlim(x0-.5, x1-.5); ax.set_ylim(y1-.5, y0-.5)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')


def render_comparison(path, case, attr, image, sources, rendered, zoom, model):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.4), dpi=140)
    fig.suptitle(f'{case.replace("_", "-")} | {attr.title()}', fontsize=21, y=.985)
    fig.text(.5, .94, f'MANUAL ILLUSTRATION from {NAMES[model]} p01 | deliberately dirty | strongest patches at the selected cue',
             ha='center', fontsize=11, color='#812e2e')
    columns = [(image, 'Clean model input'), (sources['base']['overlay'], 'Original ConceptCLIP p01'),
               (sources['moe29']['overlay'], 'Original MOE29 p01'), (rendered, 'MANUAL derivative')]
    for row in range(2):
        for col, (array, title) in enumerate(columns):
            draw(axes[row, col], array, title if row == 0 else 'Target zoom', zoom if row else None)
    fig.text(.5, .047, '75% of chosen model activation retained + uneven manual cue emphasis + extra remote distractions',
             ha='center', fontsize=11)
    fig.text(.5, .021, '27 x 27 patches | red RGB (220,40,40) | alpha = 0.7 x weight | no smoothing | synthetic weights, not model predictions',
             ha='center', fontsize=10, color='#555555')
    fig.subplots_adjust(left=.02, right=.98, top=.885, bottom=.085, hspace=.12, wspace=.05)
    fig.savefig(path); plt.close(fig)




def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': '#f7f7f5'})
    request = read_json(PLAN_ROOT/'request.json')
    inputs = {(r['case_id'], r['attribute']): r for r in read_json(PLAN_ROOT/'review_inputs/manifest.json')}
    plan_list = sum((read_json(PLAN_ROOT/'plans'/f'group_{g}.json') for g in 'abc'), [])
    plans = {(p['case_id'], p['attribute']): p for p in plan_list}
    requested = {(r['case_id'], r['attribute']) for r in request}
    assert len(plan_list) == len(plans) == len(request) == len(requested) == 32
    assert set(plans) == set(inputs) == requested
    definitions = {d['id']: d for d in read_json(DEFS)['attributes']}
    (OUT/'maps').mkdir(exist_ok=True)
    before = {str(DEFS): sha(DEFS)}
    records = []
    for item in request:
        case, attr = item['case_id'], item['attribute']
        plan = plans[case, attr]; inp = inputs[case, attr]; model = plan['source_model']
        assert model in NAMES and plan['variant'] == 'p01'
        image = np.asarray(Image.open(inp['input']).convert('RGB'))
        assert image.shape == (384, 384, 3)
        for key in ['source', 'input', 'reference', 'base_overlay', 'moe29_overlay']:
            path = Path(inp[key]); before[str(path)] = sha(path)
        copied = OUT/'case_inputs'/case; copied.mkdir(parents=True, exist_ok=True)
        for key in ['source', 'input', 'reference']:
            shutil.copyfile(inp[key], copied/f'{key}.png')
        sources = {}
        for source_model in NAMES:
            path = NUMERIC/f'GALLERY_{source_model}_NUMERIC_ONLY_20261016/{source_model}/LUNA16/{case}/{attr}_p01_heatmap.npz'
            before[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                raw = z['native_gradcam_raw']; heat = z['model_input_heatmap_normalized']
                meta = json.loads(str(z['metadata_json']))
            assert meta['sample_id'] == case and meta['attribute'] == attr and meta['model'] == source_model
            assert meta['split'] == 'train'
            assert meta['variant'] == 1 and meta['prompt_index'] == definitions[attr]['global_prompt_index']
            assert raw.shape == (27, 27) and np.isfinite(raw).all()
            span = float(raw.max())-float(raw.min())
            grid = np.clip((raw-float(raw.min()))/span, 0, 1).astype(np.float32) if span > 0 else np.zeros_like(raw, dtype=np.float32)
            assert np.array_equal(expand(grid), heat)
            rendered = overlay(image, heat)
            assert np.array_equal(rendered, np.asarray(Image.open(inp[f'{source_model}_overlay']).convert('RGB')))
            sources[source_model] = {'path': str(path), 'sha256': before[str(path)], 'grid': grid, 'overlay': rendered}
        source_grid = sources[model]['grid']; retained = RETAIN*source_grid
        target = np.zeros((27, 27), np.float32)
        coordinates = set()
        for r, c, weight in plan['target_patches']:
            assert isinstance(r, int) and isinstance(c, int) and 0 <= r < 27 and 0 <= c < 27
            assert 0 < weight <= 1 and (r, c) not in coordinates
            coordinates.add((r, c)); target[r, c] = weight
        assert target.max() == 1
        seed = int(hashlib.sha256(f'{case}:{attr}:dirty-v1'.encode()).hexdigest()[:8], 16)
        dirt, dirt_info = distractors(image, inp['zoom_box_xyxy'], seed)
        assert not dirt[target > 0].any()
        combined = np.maximum(np.maximum(retained, target), dirt)
        manual_added = combined-retained
        heat = expand(combined); rendered = overlay(image, heat)
        stem = f'{case}_{attr}_MANUAL_FROM_{NAMES[model].upper()}'
        prefix = OUT/'maps'/stem
        info = {
            'case_id': case, 'dataset': 'LUNA16', 'split': 'train', 'attribute': attr,
            'created_date': '2026-10-17', 'artifact_type': 'manual_model_derived_synthetic_illustration',
            'is_original_model_output': False, 'is_ground_truth': False, 'clinically_validated': False,
            'intended_quality': 'Deliberately very dirty and sub-ideal; no measured or validated quality grade.',
            'definition': definitions[attr], 'definition_source': str(DEFS),
            'definition_reference': 'https://pylidc.github.io/annotation.html',
            'starting_model': NAMES[model], 'starting_variant': 'p01',
            'selection_reason': plan['reason'], 'intended_focus': plan['focus'], 'case_specific_caveat': plan['caveat'],
            'selection_scope': 'Visual starting-point choice for an illustration; not a measured superiority claim.',
            'reviewed_files': plan['inspected_files'],
            'starting_numeric_path': sources[model]['path'], 'starting_numeric_sha256': sources[model]['sha256'],
            'compared_sources': {m: {'path': v['path'], 'sha256': v['sha256']} for m, v in sources.items()},
            'formula': 'final_grid = maximum(0.75 * selected_model_p01_normalized_grid, manual_target_floor, extra_distractor_grid); no further normalization',
            'source_retention_scale': float(RETAIN), 'distractor_construction': dirt_info,
            'distractor_interpretation': 'Explicit synthetic nuisance activation outside the target zoom; not attribute evidence.',
            'target_patches_row_column_weight': plan['target_patches'],
            'coordinate_system': '384x384 unchanged model input; zero-based rows and columns;27x27 grid;14x14 footprints;6 trailing rows and columns uncovered',
            'source_image_path': inp['source'], 'source_image_sha256': before[inp['source']],
            'model_input_path': inp['input'], 'model_input_sha256': before[inp['input']],
            'zoom_box_xyxy': inp['zoom_box_xyxy'], 'overlay_rgb': [220, 40, 40],
            'overlay_alpha_scale': .7, 'smoothing': False,
            'render_formula': 'clip(input_float32*(1-.7*heat)+[220,40,40]*(.7*heat),0,255).astype(uint8)',
            'reference_mask_used_for_weights': False, 'original_files_changed': False, 'inserted_into_user_study': False,
            'limitations': ['Single exported CT slice rather than raw DICOM/HU analysis',
                           'Manual approximate cue locations with intentionally incomplete coverage and spill',
                           'No attribute-specific spatial ground truth or independently validated quality grade']}
        np.savez_compressed(str(prefix)+'.npz', native_manual_weights=combined,
                            source_model_normalized_grid=source_grid, retained_model_grid=retained,
                            manual_target_floor=target, extra_distractor_grid=dirt,
                            manual_added_weights=manual_added, model_input_heatmap_normalized=heat,
                            metadata_json=json.dumps(info))
        # Nine significant digits preserve float32 values when parsed back.
        np.savetxt(str(prefix)+'_grid.csv', combined, delimiter=',', fmt='%.9g')
        Image.fromarray(rendered).save(str(prefix)+'_overlay.png')
        Image.fromarray((heat*255).astype(np.uint8), mode='L').save(str(prefix)+'_heatmap.png')
        Image.fromarray((combined*255).astype(np.uint8), mode='L').save(str(prefix)+'_native27.png')
        Path(str(prefix)+'_metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        render_comparison(str(prefix)+'_comparison.png', case, attr, image, sources, rendered, inp['zoom_box_xyxy'], model)
        with np.load(str(prefix)+'.npz', allow_pickle=False) as z:
            reload_ok = np.array_equal(z['native_manual_weights'], combined) and np.array_equal(z['model_input_heatmap_normalized'], heat)
            formula_ok = np.array_equal(z['native_manual_weights'], np.maximum(np.maximum(z['retained_model_grid'], z['manual_target_floor']), z['extra_distractor_grid']))
        csv_ok = np.array_equal(np.loadtxt(str(prefix)+'_grid.csv', delimiter=',').astype(np.float32), combined)
        saved_overlay = np.asarray(Image.open(str(prefix)+'_overlay.png').convert('RGB'))
        saved_heat = np.asarray(Image.open(str(prefix)+'_heatmap.png'))
        checks = {
            'both_original_overlays_reconstructed_pixel_exact': True,
            'native_shape': list(combined.shape), 'expanded_shape': list(heat.shape), 'dtype': str(combined.dtype),
            'finite_and_unit_range': bool(np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1),
            'saved_npz_exact': bool(reload_ok), 'saved_csv_float32_exact': bool(csv_ok),
            'saved_overlay_pixel_exact': bool(np.array_equal(saved_overlay, rendered)),
            'saved_heatmap_pixel_exact': bool(np.array_equal(saved_heat, (heat*255).astype(np.uint8))),
            'formula_exact': bool(formula_ok),
            'uncovered_border_zero': bool(not heat[378:, :].any() and not heat[:, 378:].any()),
            'zero_heat_pixels_unchanged': bool(np.array_equal(saved_overlay[heat == 0], image[heat == 0])),
            'all_nonzero_source_locations_retained': bool(np.all(combined[source_grid > 0] > 0)),
            'no_extra_distractor_over_manual_target': bool(not dirt[target > 0].any()),
            'target_max': float(combined[target > 0].max()), 'outside_target_max': float(combined[target == 0].max()),
            'target_stronger_than_all_off_target_patches': bool(combined[target > 0].max() > combined[target == 0].max()),
            'manual_target_patch_count': int(np.count_nonzero(target)),
            'extra_distractor_patch_count': int(np.count_nonzero(dirt)),
            'patches_changed_from_retained_source': int(np.count_nonzero(manual_added)),
            'source_nonzero_patches': int(np.count_nonzero(source_grid)), 'final_nonzero_patches': int(np.count_nonzero(combined))}
        assert all(v for v in checks.values() if isinstance(v, bool)), (case, attr, checks)
        records.append({**plan, 'stem': stem, 'validation': checks})
        print(f'{len(records):02}/32 {case} {attr}: {NAMES[model]}, target=1.0, outside={checks["outside_target_max"]:.3f}', flush=True)
    unchanged = all(sha(Path(path)) == digest for path, digest in before.items())
    assert unchanged
    validation = {'status': 'passed', 'map_count': len(records),
                  'case_count': len({r['case_id'] for r in records}),
                  'attribute_counts': dict(Counter(r['attribute'] for r in records)),
                  'starting_model_counts': dict(Counter(NAMES[r['source_model']] for r in records)),
                  'all_requested_pairs_exactly_once': True, 'original_source_files_unchanged': unchanged,
                  'hashed_original_file_count': len(before),
                  'checks': {f'{r["case_id"]}/{r["attribute"]}': r['validation'] for r in records}}
    (OUT/'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    (OUT/'construction.json').write_text(json.dumps(records, indent=2), encoding='utf-8')

    # A compact representative figure for delivery; all 32 have larger comparisons.
    representatives = [('LUNA16_283', 'margin'), ('LUNA16_545', 'texture'),
                       ('LUNA16_391', 'spiculation'), ('LUNA16_391', 'texture')]
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.7), dpi=160)
    fig.suptitle('LUNA16 | deliberately dirty manual heatmaps', fontsize=20, y=.985)
    fig.text(.5, .938, 'Examples from the 32-map batch; all source CT images and original model maps are preserved.', ha='center', fontsize=10)
    for col, key in enumerate(representatives):
        rec = next(r for r in records if (r['case_id'], r['attribute']) == key)
        inp = inputs[key]; rendered = np.asarray(Image.open(OUT/'maps'/f'{rec["stem"]}_overlay.png'))
        draw(axes[0, col], rendered, f'{key[0].replace("_", "-")} | {key[1].title()}')
        draw(axes[1, col], rendered, 'Target zoom', inp['zoom_box_xyxy'])
    fig.text(.5, .04, 'MANUAL derivatives | target peak 1.00, competing patches at most 0.75 | 27 x 27 grid | alpha = 0.7 x weight', ha='center', fontsize=10, color='#812e2e')
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.08, hspace=.13, wspace=.05)
    fig.savefig(OUT/'batch_preview.png'); plt.close(fig)
    print(json.dumps({k: v for k, v in validation.items() if k != 'checks'}, indent=2))




if __name__ == '__main__':
    main()
