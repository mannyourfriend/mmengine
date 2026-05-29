# Neuron Mask2Former — Project Handoff Guide

This repo is a fork of [MMDetection](https://github.com/open-mmlab/mmdetection) (v3.3.0)
customized to train and run a **Mask2Former** instance-segmentation model on fluorescence
microscopy images of neurons. The goal is to segment individual neuron structures
(somas, neurites, clusters) so they can be measured automatically.

This guide covers **only the custom parts** of the project. For anything generic
(installation, MMDetection internals, other model families), see the upstream
[MMDetection docs](https://mmdetection.readthedocs.io/).

> **Status:** The model produces decent results but did not fully reach the target
> accuracy. This document is meant to let the next person pick it up quickly.

---

## 1. The 30-second mental model

```
                 configs/a_custom_mask2former/*.py        <- training recipes (model + data + schedule)
                              |
   tools/train.py  ───────────┘  reads a config, writes checkpoints to work_dirs/
                              |
                              v
   work_dirs/<config_name>_run1/   <- checkpoints (.pth), logs, a copy of the config
                              |
              ┌───────────────┴────────────────┐
              v                                  v
   tools/test.py                       test_models_*.py  (custom inference + rotation-fusion)
   (standard COCO mAP eval)            (the "real" inference pipeline + scoring vs hand annotations)
```

The dataset glue that lets MMDetection read our COCO-format annotations lives in
`mmdet/datasets/coco_custom_rle.py` (class `COCOCustomDataset`).

---

## 2. The custom files (what to actually look at)

| File | Role |
|------|------|
| `mmdet/datasets/coco_custom_rle.py` | **The live dataset class** (`COCOCustomDataset`). Reads COCO JSON, converts polygon segmentations to RLE masks. Registered in `mmdet/datasets/__init__.py`. |
| `configs/a_custom_mask2former/*.py` | Training configs. One per experiment (see §5). |
| `tools/train.py` | Stock MMDet trainer **+ a custom `--repeat N` flag** (runs training N times into `_run1`, `_run2`, … work dirs). |
| `tools/test.py` | Stock MMDet evaluator (COCO bbox + segm mAP). Unmodified. |
| `rotation_utils.py` | Shared helpers: rotate/de-rotate images and masks, mask→polygon, polygon IoU, evenly-spaced rotation angles. |
| `test_models_custom.py` | **The main custom inference pipeline.** Runs the model at several rotations, de-rotates the masks back, fuses overlapping detections by IoU, drops edge-touching blobs, and renders overlay images. Writes a `run_meta_*.json`. |
| `test_models_hand_annos.py` | Same rotation-fusion idea, but **scores predictions against hand-drawn ground-truth annotations** (COCO eval). Takes `--config` / `--checkpoint`. |
| `test_models_hand_annos_auto.py` | Batch driver: sweeps every model in `work_dirs/`, runs `test_models_hand_annos.py` on each, and collects the numbers into an Excel sheet. |
| `test_models.py` | **Older precursor** to `test_models_custom.py` (single hardcoded image, no rotation fusion). Kept for reference; prefer `test_models_custom.py`. |
| `mmdet/visualization/local_visualizer.py` | **A core MMDet file that was modified in place** — see §6 for the warning. |

---

## 3. Data format the model expects

Each experiment lives in its own `data_root` folder (an absolute path baked into the
config — see §4). The dataset class expects a standard **COCO instance-segmentation**
layout:

```
<data_root>/
├── train/                 # all images (train + val + test point here in current configs)
├── coco_train.json        # COCO annotations for training split
├── coco_val.json          # COCO annotations for validation split
└── coco_test.json         # COCO annotations for test split
```

- Annotations are COCO `segmentation` fields — either polygon lists **or** RLE.
  `coco_custom_rle.py` normalizes both to merged RLE masks.
- The COCO `categories` in the JSON **must** match the class names declared in
  `COCOCustomDataset.METAINFO`:
  `['NeuriteSoma', 'OutOfBound', 'Soma', 'Cluster']`.
  The config's `num_things_classes` selects how many of those a given experiment uses.
- Some experiments also point at a separate hand-annotated validation set
  (e.g. `ValReal_YodaCrunch.json` + a `valReal/` image folder) via the commented-out
  `monitor_dataloader` in the configs.

---

## 4. ⚠️ Hardcoded paths — read this before running anything

Absolute Windows paths are baked into both the **configs** and the **inference scripts**.
These are the #1 thing to update for a new machine or dataset. They were intentionally
left in place (not parameterized) to avoid changing working code during handoff.

**In every config** (`configs/a_custom_mask2former/*.py`, ~line 151):
```python
data_root = r"C:\Users\five\Desktop\Manny\05a11_20neuron/"
```

**In the inference scripts** (`test_models*.py`), near the top:
```python
CONFIG_FILE     = r"...\work_dirs\custom_mask2former_20neuron\custom_mask2former_20neuron.py"
CHECKPOINT_FILE = r"...\best_coco_segm_mAP_50_epoch_18.pth"
img_path        = r"C:\Users\five\Desktop\Manny\40xImgs\temp\..."   # input image(s)
```
Some scripts also reference data on the `S:\Phys\...` network share.

➡️ **To run on new data/machine:** grep for `C:\Users\five` and `S:\Phys` across the repo
and update those paths. (A future cleanup could lift these into CLI args or env vars.)

---

## 5. The config variants

All configs in `configs/a_custom_mask2former/` are near-identical Mask2Former recipes
(ResNet-50 backbone, 512×512 input, AdamW, 20 epochs, saving the best
`coco/segm_mAP_50` checkpoint). They differ only in **which dataset** and **how many
classes** they train on:

| Config | Classes | `data_root` dataset |
|--------|:-------:|---------------------|
| `custom_mask2former.py` | 4 | `05a11_2neuron` (original base recipe) |
| `custom_mask2former_2neuron.py` | 3 | `05a11_2neuron` |
| `custom_mask2former_10neuron.py` | 3 | `05a11_10neuron` |
| `custom_mask2former_20neuron.py` | 3 | `05a11_20neuron` ← used by `test_models_custom.py` |
| `custom_mask2former_10neuron_noSoma.py` | 2 | `05a11_10neuron_noSoma` |
| `custom_mask2former_10neuron_cluster_noSoma.py` | 3 | `05a11_10neuron_cluster_noSoma` |
| `custom_mask2former_10neuron_clusters_crowdSoma.py` | 3 | `05a11_10neuron_clusters_crowdSoma` |
| `custom_mask2former_10neuron_crowdClusterAndSoma.py` | 3 | `05a11_10neuron_crowdClusterAndSoma` |
| `custom_mask2former_10neuron_crowdClusterAndSoma_expand2.py` | 3 | `05a11_subset_expand2` |
| `custom_mask2former_somaAndCluster.py` | 3 | `05a11_subset` |

The "Nneuron" prefix refers to how many manually-annotated neurons were in that training
set; "noSoma" / "crowdSoma" / "cluster" describe how soma and cluster objects were
labeled (e.g. crowd/ignore regions). Trained outputs land in `work_dirs/<config_name>_run*/`.

> **Tip for the next person:** if you only keep one as the canonical example, use
> `custom_mask2former_20neuron.py` — it's the one the inference scripts are wired to.

---

## 6. ⚠️ The modified core file: `local_visualizer.py`

`mmdet/visualization/local_visualizer.py` is a **core MMDetection file that was edited
in place** (~877 lines removed, ~358 added vs. upstream) to customize how masks, labels,
and overlays are drawn for this project. The custom inference scripts import
`DetLocalVisualizer` from it.

**Why this matters:** if anyone reinstalls or upgrades `mmdet`
(`pip install -U mmdet`, or `pip install -e .` over a fresh clone), these edits will be
**silently overwritten** and the visualizations will change or break.

**Do not** `pip install` over this checkout without first backing up this file. To see
exactly what was changed:
```bash
git log --follow -p mmdet/visualization/local_visualizer.py
```
(A more robust long-term fix would be to extract the customizations into a separate
subclass file outside the `mmdet` package, but that was deliberately left as-is here.)

---

## 7. How to run things

> Assumes the MMDetection environment is already installed (PyTorch + CUDA + mmcv +
> mmengine + this repo in editable mode). See upstream install docs if starting fresh.

### Train
```bash
python tools/train.py configs/a_custom_mask2former/custom_mask2former_20neuron.py
# repeat the full run 3× into _run1/_run2/_run3 (custom flag):
python tools/train.py configs/a_custom_mask2former/custom_mask2former_20neuron.py --repeat 3
# optional: mixed precision
python tools/train.py <config> --amp
```
Checkpoints, logs, and a frozen copy of the config are written to
`work_dirs/<config_name>_run1/`.

### Standard COCO evaluation (mAP)
```bash
python tools/test.py \
    work_dirs/custom_mask2former_20neuron_run1/custom_mask2former_20neuron.py \
    work_dirs/custom_mask2former_20neuron_run1/best_coco_segm_mAP_50_epoch_18.pth
```

### Custom inference with rotation-fusion (produces overlay images)
1. Edit the `CONFIG_FILE`, `CHECKPOINT_FILE`, and `img_path` constants at the top of
   `test_models_custom.py`.
2. Run:
   ```bash
   python test_models_custom.py
   ```
   Overlays + a `run_meta_*.json` (which records the rotation/IoU/threshold settings)
   are written next to the input images in a timestamped `results_*` folder.

   Key knobs at the top of the file: `ROTATION_ANGLES_COUNT` (how many rotations to
   ensemble), `MASK_SCORE_THR` (confidence cutoff), `FUSE_IOU_THR_*` (how aggressively
   to merge overlapping detections).

### Score a model against hand annotations
```bash
python test_models_hand_annos.py --config <config.py> --checkpoint <checkpoint.pth>
```

### Score every model in work_dirs/ → Excel
```bash
python test_models_hand_annos_auto.py    # edit the work_dirs / output paths inside first
```

---

## 8. Known rough edges / suggestions for the next person

- **Hardcoded paths everywhere** (§4) — the biggest blocker to reuse. Lifting `data_root`
  and the script paths into CLI args / a small `.env` would make this portable.
- **In-place edit of `local_visualizer.py`** (§6) — fragile against reinstalls; consider
  subclassing instead.
- **Config sprawl** — 10 configs differing mostly by `data_root`. They could share a
  common `_base_` and override just the data path + class count.
- **`max_iters` in the configs is a leftover** from an iteration-based schedule; training
  actually runs on `EpochBasedTrainLoop` (20 epochs), so the `param_scheduler` milestones
  in iterations don't fire. Worth cleaning up if you revisit the LR schedule.
- **`coco_custom.py` was removed** during handoff — it was an unused duplicate of
  `coco_custom_rle.py` (same class name, but stored raw polygons instead of RLE and was
  never imported). The RLE version is the only one that was ever wired in.
```
