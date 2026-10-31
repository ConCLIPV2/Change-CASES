# Change-CASES

**The code used to change the heat maps and create the deliberately sub-ideal LUNA and BrEaST cases.** This repository contains the transformation process and its exact manual instructions. It contains no gallery website, case images, model-input arrays, generated heat maps, or case bundles.

## Code

- [`workflows/luna_manual_batch_2026-10-17/build_batch.py`](workflows/luna_manual_batch_2026-10-17/build_batch.py): final LUNA transformation, 32 requested maps across 20 cases.
- [`workflows/breast_manual_batch_2026-10-17/build_batch.py`](workflows/breast_manual_batch_2026-10-17/build_batch.py): final BrEaST transformation, 31 requested maps across 18 cases.
- Each workflow's `plans/group_a.json`, `group_b.json`, and `group_c.json`: exact hand-selected patch coordinates, weights, starting-model choices, reasons and caveats. These are necessary inputs to the transformation.
- Each workflow's `request.json` and `review_inputs/manifest.json`: requested case/attribute mappings and external source-file locations. These are text instructions and metadata, not case assets.
- Each workflow's `prepare_review_inputs.py`: code that assembled the source/model comparison sheets used to choose the manual patches.
- [`earlier_versions/`](earlier_versions/): all five earlier case-283 construction scripts, preserved verbatim with their original workspace assumptions.
- [`source_receipts.json`](source_receipts.json): source-code/plan SHA-256 receipts and the adaptations made for this export.

The final builders retain the original transformation and rendering calculations. Website creation and gallery ZIP generation were removed; external data and output locations are configurable.

## What the process does

1. Read the original saved native Grad-CAM grids for both models and the unchanged model-input image. Verify their identities and reconstruct the original overlays.
2. Use the starting model specified in the case-specific manual plan, with p01 fixed throughout.
3. Normalize its 27 × 27 grid using the original min/max rule. Construct a target grid from the saved `[row, column, weight]` entries.
4. Generate deterministic distracting activation: five remote clusters and weaker scattered patches. LUNA excludes the target zoom neighborhood; BrEaST excludes the lesion-reference area and target neighborhood.
5. Apply the elementwise rule:

   ```text
   final = maximum(0.75 * original_normalized_grid, manual_target_floor, distractions)
   ```

   No further normalization is applied. Target peak 1.00 and off-target maximum 0.75 are imposed by the construction. Original nonzero activity is retained.
6. Expand each native cell into a 14 × 14 pixel block on the 384 × 384 model input. The last six rows and columns remain uncolored. No smoothing is used.
7. Render the red overlay with RGB `(220,40,40)` and `alpha = 0.7 * weight`. Save numerical arrays, CSVs, overlays, comparisons, construction details and consistency checks to the chosen output directory.

The patch positions, strengths and starting-model selections were assistant visual judgments encoded as explicit plans. The code executes those choices; it does not automatically establish clinically correct attribute locations. No model retraining or new inference is involved. Some requested cues were uncertain, source-negative or unresolved, as recorded in the plans.

## Run the final transformations

Requires Python 3.10+ and the **external original ConceptCLIP data workspace**. The data are deliberately not included here. That workspace supplies:

- `LUNA16 Attributes Localization masks TEST/Numeric/`: original p01 numerical exports for the requested pairs, both models.
- `shared_research/luna16_train_attribute_gallery_2026-10-15/`: original LUNA images, overlays and references.
- `shared_research/full_dataset_four_attribute_review_2026-10-12/`: original BrEaST images, overlays and references.
- `shared_research/luna16_user_study_review_2026-10-12/definitions.json`: LUNA attribute definitions.

```sh
python -m pip install -r requirements.txt
python run.py --workspace "D:/ConceptCLIP" --dataset both --output "D:/Change-CASES-output"
```

Use `--dataset luna` or `--dataset breast` to run one workflow. Outputs must be outside the source-data workspace. Saved historical `D:/ConceptCLIP` file references are resolved against `--workspace` in memory; source data remain read-only.

Add `--prepare-review-inputs` to regenerate the source/model comparison sheets as well. Final transformations use the committed request, input manifest and manual plans so that the saved editing decisions remain explicit and repeatable.

The earlier scripts are historical stages, not the final formula: some use fully synthetic targets and noise, while the preceding model-guided case-283 version uses `maximum(0.60*S,T)` without extra distractions. Their source code retains these distinctions.

## Verification

```sh
python verify_code.py
```

This checks source syntax, exact requested-pair/plan coverage, patch coordinates and weights, and agreement with the saved source-plan hashes. `verification_report.json` also records a full local execution of both final transformations against the external source workspace. File/rendering consistency does not establish clinical correctness or measured improvement.
