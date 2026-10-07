# LiDAR place recognition: NDT-Transformer vs BEVPlace++

Final project, Deep Learning for 3D Computer Vision (Hebrew University).
A controlled comparison of NDT-Transformer and BEVPlace++ under one evaluation protocol.

## What is mine and what is official

| Part | Source |
|---|---|
| Network, lazy quadruplet loss, tuple sampling, hard-negative scheme (NDT-Transformer) | official repo, `third_party/NDT-Transformer` @ `51f90df` |
| BEVPlace++ network (REIN), official checkpoint | official repo, `third_party/BEVPlace2` @ `2d0636d` |
| BEVPlace++ BEV images (vectorised, pixel-identical to the official script) | `src/lpr/bev.py`, check: `tests/test_bev_matches_official.py` |
| BEVPlace++ training set, hard mining, loss, NetVLAD init (reproduced from `main.py`, `kitti_dataset.py`) | `src/lpr/bevplace.py` |
| NDT cell generation G (not in the official repo) | `src/lpr/ndt_g.py` |
| Cropping and ground removal | `src/lpr/preprocess.py` |
| KITTI download, tuples, training loop, evaluation | `scripts/`, `src/lpr/` |

The official repositories are not copied into this repo; `scripts/setup_third_party.sh`
clones them at fixed commits.

## Stage 1: NDT-Transformer on KITTI sequence 00 (sanity check)

Run `notebooks/01_ndt_kitti00.ipynb` on Colab with a GPU runtime. It:

1. downloads only the 4541 scans of sequence 00 (range requests into the 80 GB zip) and the poses;
2. crops to 80 m x 80 m, removes the ground (Patchwork++) and runs G on every scan;
3. stores the NDT cells on Google Drive (about 0.9 GB), so this is done once;
4. builds training tuples on frames 0-2999 (positive < 5 m, negative > 7 m, as in BEVPlace++);
5. trains with the paper's hyper-parameters and resumes automatically after a disconnect;
6. evaluates with the BEVPlace++ protocol: database = frames 0-2999, queries = frames 3200-4540,
   correct if within 5 m; Recall@1 and Recall@1%.

The per-epoch evaluation during training uses the query frames and is for monitoring only.
The reported number is the last epoch, not the best one.

## Experiments: NDT variants and training runs

Everything is driven from cell 1 of the notebooks (`01_ndt_kitti00.ipynb` for NDT-Transformer,
`02_bevplace_kitti.ipynb` for BEVPlace++): the preprocessing parameters (`NDT_PARAMS` / `BEV_PARAMS`) and
`TRAIN_PARAMS`, each holding only what differs from the defaults in `src/lpr/experiments.py`.
Results are kept on Google Drive, one folder per method:

```
MyDrive/lpr/ndt/<variant>/              config.json, cells_<seq>.tar, stats_<seq>.csv, preview_<seq>.png
MyDrive/lpr/bev/<variant>/              config.json, imgs_<seq>.tar, stats_<seq>.csv
MyDrive/lpr/runs/ndt/<run>/             config.json, ndt_config.json, metrics.csv, epoch_XX.pth, latest.pth
MyDrive/lpr/runs/bev/<run>/             same, with bev_config.json
MyDrive/lpr/runs/bev/official_<variant>/  the authors' checkpoint evaluated on our BEV images
MyDrive/lpr/runs/<method>/<run>/eval/<eval type>/<checkpoint>/   one evaluation (config.json, summary.csv, ...)
MyDrive/lpr/runs_summary.csv            one row per run, column "method" (scripts/summarize_runs.py)
MyDrive/lpr/evals_summary.csv           every evaluation, one row per sequence / angle / ...
```
Older Drive folders are moved into this layout by `scripts/migrate_layout.py` (run by both notebooks).

**Running a notebook.** Set cell 1 and use *Runtime -> Run all*. `MODE` says what the run is for:
`"train"` (data, training of a new run or continuation of an unfinished one, then the evaluations),
`"eval"` (only the evaluations of an existing run, checkpoint `EVAL_CKPT`, nothing trained or built) or
`"smoke"` (a short timing / GPU-memory test). Cells that a mode does not need print `skipped`.

**Evaluations.** `EVALS` in cell 1 lists evaluation types from `EVAL_TYPES` in `src/lpr/evals.py`;
`scripts/evaluate.py` runs one type on one checkpoint and stores it in `<run>/eval/<type>/<checkpoint>/`.
A complete evaluation with the same parameters and checkpoint is not repeated. Types so far:
`seqs_heading` (recall on 00, 02, 05, 06, 08 and Recall@1 per heading group, both methods) and `tnet`
(NDT-Transformer: scans rotated about the vertical axis; how far the T-Net is from the identity, whether it
follows the rotation, and how far the descriptor moves; `scripts/tnet_check.py`) and `compass` (both methods,
no retraining: every scan is rotated about z by its heading from the ground-truth pose, so map and queries
share one orientation, and each query gets a compass error drawn from N(0, sigma^2) for sigma in 0, 2, 5,
10, 20, 45 degrees; NDT cells are rotated exactly, BEV images are rotated as images). A new type is an entry in
`EVAL_TYPES` plus a function in `scripts/evaluate.py`.

Names are `<date>_baseline_<hash>` for the default parameters; any other variant or run is
`<date>_<tag>_<hash>` with the tag set in cell 1, or `<date>_<hash>` when the tag is empty. The same parameters always map to the same folder:
an NDT variant is never built twice, an unfinished run resumes after a disconnect, and a finished run is
not trained again (change the seed to repeat it). `train_ndt_kitti.py` refuses to write into a run folder
that holds different parameters.

## All sequences on Azure

The raw scans of 00, 02, 05, 06 and 08, the NDT cells and the BEV images are built on an Azure VM and kept in
an Azure Blob container; Colab copies them into the variant's Drive folder.
A variant is independent of the sequence, so every sequence lives in the same variant folder.

```
blob container kitti/
  raw/kitti_seq<seq>_raw.tar        raw scans + poses + calib (shared by both methods)
  ndt/<variant>/cells_<seq>.tar     bev/<variant>/imgs_<seq>.tar
  <method>/<variant>/stats_<seq>.csv, config_azure.json
```

**What was set up** (Azure for Students; the subscription only allows some regions, and in most of them
only some VM families):
1. Storage account `dovlpr2026` (Poland Central, Standard, LRS, account key access enabled),
   private container `kitti`. Container -> Shared access tokens: permissions Read, Add, Create, Write, List
   (no Delete) -> copy the **Blob SAS URL**. It works like a password: keep it in Colab Secrets as
   `AZURE_SAS`, never in a notebook or in git.
2. VM `lpr-vm` (Austria East, `Standard_DS3_v2`: 4 vCPU, 14 GB, about 0.46 $/h), Ubuntu Server 24.04,
   SSH key, OS disk 128 GiB Standard SSD. Poland Central only offered burstable B-series sizes and
   Spain Central was refused by the region policy at deployment time.
   **Auto-shutdown is not available in Austria East**: stop the VM in the portal (status
   *Stopped (deallocated)*) whenever a build ends. A budget alert (Cost Management -> Budgets) mails at 50% / 100%.

**On the VM** (`ssh -i ~/.ssh/lpr-vm_key.pem azureuser@<public IP>`):
```
git clone https://github.com/dov85/lidar-place-recognition.git && cd lidar-place-recognition   # once
git pull
tmux new -s build                       # the build keeps running if SSH disconnects
METHOD=bev VARIANT=<variant name from notebook cell 3> bash scripts/azure_build.sh "<Blob SAS URL>"
```
`METHOD` is `ndt` (default; cells of 02-08) or `bev` (images of 00-08). For a variant with non-default
parameters also set `ARGS` (the notebook prints the full command in cell 3). Raw scans already on the VM
disk or in the container are reused, never downloaded from KITTI again. `tmux attach -t build` to look again,
`tail ~/work/build.log` for the log.

**In Colab:** notebook 01 section 9 (NDT) or notebook 02 section 4 (BEV) copy the variant to Drive, then evaluate
every sequence and run the heading analysis (`scripts/heading_analysis.py`: how far the query headings are
from the map frames at the same place, and Recall@1 for same / crossing / opposite headings).

## BEVPlace++ (`notebooks/02_bevplace_kitti.ipynb`)

1. BEV images on Azure (`METHOD=bev`), with the official parameters: crop |x|, |y|, |z| < 40 m, 0.4 m voxel
   down-sampling, 201 x 201 pixels of 0.4 m, points per pixel capped at 10, no ground removal.
   `src/lpr/bev.py` gives the same PNG as the official `gen_bev_images.py`, pixel for pixel (notebook cell 4b).
2. Reproduction check: the authors' checkpoint (`third_party/BEVPlace2/runs/Aug08_10-17-29/model_best.pth.tar`,
   epoch 5, chosen by them on NCLT) evaluated on our images with our protocol.
3. Training with the official recipe: frames 0-2999 of KITTI 00, positives < 5 m, negatives > 7 m, 10 hard
   negatives (descriptor cache refreshed every epoch), batch 4, triplet margin 0.3, Adam 1e-4, 40 epochs,
   random rotation of every image, NetVLAD initialised by k-means. Differences from `main.py`: no NCLT
   validation and no best-checkpoint selection on test sequences (the last epoch is reported, as for
   NDT-Transformer); checkpoints every 150 iterations so a Colab disconnect costs at most a few minutes;
   if 4 x 12 images do not fit in GPU memory, one tuple per forward pass with gradient accumulation
   (`split_forward` in `metrics.csv`; same loss, BatchNorm statistics over 12 images instead of 48).
4. Evaluation on 00, 02, 05, 06, 08 and the heading analysis, in the same format as NDT-Transformer;
   section 10 of the notebook puts both methods side by side.

## Design decisions in G (see `src/lpr/ndt_g.py`)

From the paper: k = 2000 cells, 0.8 m, binary search on the resolution, removal of redundant
cells. From the authors (D. Adolfsson, personal communication, Sept. 2026): the search targets
0-10% more cells than required. Assumptions of this implementation:

- r = 0.8 m is the radius used to compute each cell from the original points;
- a cell needs at least 5 points;
- redundancy = largest Bhattacharyya coefficient with one of the 8 nearest cells, removed greedily;
- coordinates are divided by 40 m (means in [-1, 1]); the official loader multiplies covariances by 1000.

Also note: in the official code `convariance_transform = False`, so the T-Net transform is applied
to the means only, not propagated to the covariances as described in the paper. The training
parameter `cov_transform=True` turns the official switch on (Σ' = TᵀΣT, the same transform as the means);
`eval_ndt_kitti.py` takes the setting from the run's `config.json`.
`rot_aug=True` rotates every training cloud (query, each positive and negative) about z by its own random
angle in [0, 360): means R m, covariances R S R^T.

## Tests without KITTI

`python tests/test_bev_matches_official.py` compares `src/lpr/bev.py` with the official BEV script on synthetic
scans (`--scans a.bin b.bin` for real ones; needs open3d).

`tests/make_fake_kitti.py` builds a small synthetic sequence in the same zip layout and
`tests/range_server.py` serves it with HTTP range support, so the whole pipeline can be run offline:

```
python tests/make_fake_kitti.py /tmp/fake 240
python tests/range_server.py /tmp/fake 8765 &
KITTI_BASE_URL=http://127.0.0.1:8765/ python scripts/download_kitti.py --out /tmp/kitti --seqs 00
```
