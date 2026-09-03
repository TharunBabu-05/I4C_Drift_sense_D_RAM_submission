# Phase-2 Master Generator Audit Report

## 1. Master Generator File

| Item | Value |
|------|-------|
| **Primary Round-1 master generator** | `phase2_generator_60/master_generator_v2.py` |
| **Round-1 training dataset generator** | `dataset/generate_dataset.py` |
| **Existing Phase-2 200-pair generator** | `phase2_generator_60/generate_phase2_60_dataset.py` |
| **Degradation engine** | `phase2_generator_60/degradation_engine.py` |
| **Noise engine** | `phase2_generator_60/noise_engine.py` |

---

## 2. The 60 Generator Scripts

All 60 scripts live in: `phase2_generator_60/clean_60_scripts/`
`generate_folder_001.py` through `generate_folder_060.py` — **Total: 60 scripts. None will be modified.**

---

## 3. How the Master Script Calls the 60 Generators

The master generator uses `discover_render_functions()`:

- Scans for `generate_folder_NNN.py` files using importlib.util
- For each script, looks for functions starting with `render_*` or `generate_*`
- Returns a registry list of dicts: `{ folder_num, script_name, render_fn, generate_fn, module }`

**Call priority**:
1. `render_fn(h=h, w=w, gt_x=int(gt_x), gt_y=int(gt_y), seed=seed)`
2. Falls back to `render_fn(h=h, w=w, seed=seed)`
3. Falls back to `render_fn()`
4. If no render_fn: `generate_fn(output_dir=pair_dir, seed=render_seed)`

---

## 4. Input / Output of Each Generator

| Property | Value |
|----------|-------|
| **Input** | h, w (typically 1000x1000), gt_x, gt_y (target center), seed |
| **Output** | np.ndarray shape (h, w), dtype uint8, grayscale |
| **Fixed Scale** | 10x in Round-1 |
| **Special cases** | Generators 001, 002 may use generate_* and write to disk |

---

## 5. Current Round-1 Dataset Structure

```
<output_dir>/
  dataset_manifest.json
  pair_001/
    reference.png    <- clean 100x100 crop upscaled to 1000x1000
    target.png       <- same as reference.png
    search.png       <- degraded 1000x1000 search
    groundtruth.json <- {center_x, center_y, scale_factor=10.0, ...}
```

---

## 6. Siamese Training Dataset Structure (DriftSenseSiameseDataset)

The DataLoader (dataset/dataset.py) expects:
```
<data_dir>/
  dataset_manifest.json    <- {"pairs": [{pair_id, split, center_x, center_y}]}
  train/
    pair_0001/
      reference.png
      search.png
  val/
    ...
  test/
    ...
```

Returns triplet: `{reference, positive, negatives}` at Level 1 (InfoNCE with 30 negatives)

---

## 7. Phase-2 vs. Round-1 Key Differences

| Property | Round-1 | Phase-2 |
|----------|---------|---------|
| Scale | Fixed 10x | Uniform [8x, 12x] |
| Rotation | ~0 degrees | Uniform [-5, +5] degrees |
| Absent pairs | None | ~20% |
| Reference | 100x100 crop upscaled | 100x100 crop (raw) |
| Degradation | 2 random models | 4 explicit severity levels |
| Dataset split | Not in master | 80/10/10 train/val/test |
| Manifest format | JSON dict | JSON dict + CSV manifests |

---

## 8. Files Modified

- **NEW**: `phase2_generator_60/master_generator_phase2.py` (Phase-2 master generator)
- **NOT MODIFIED**: Any of the 60 `generate_folder_NNN.py` scripts
