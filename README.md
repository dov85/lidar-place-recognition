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
| Own NDT network: NDT-Transformer re-implemented, with REIN-like rotations and either head; trained with the BEVPlace++ recipe | `src/lpr/models/`, `src/lpr/myndt.py`, check: `tests/test_models.py` |

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

Point-rotated training copies for NDT-Transformer (`rot_points=N`): `ROT_COPIES=N` builds N copies of every
training frame of `ROT_SEQS` (default 00, frames before the split point) whose raw points are rotated about z
by a random angle before crop, ground removal and cells; `CELL_SEQS=""` builds only these. Notebook 01
section 4 prints the exact command and copies `cells_00_rot<N>.tar` to Drive.

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
5. Ablation `arch="plain"` (training parameter): the same network with a single rotation (0 deg) in REIN's
   rotation-equivariant module, i.e. one pass of the same ResNet34 layers, the same (identity) warps and
   NetVLAD, so no built-in rotation invariance; with `rot_aug=True` or `False`. Same parameters and
   checkpoint format; the evaluation scripts read `arch` from the run's `config.json`.

## Own NDT network with the BEVPlace++ recipe (`notebooks/04_myndt_kitti.ipynb`, method `myndt`)

To compare the representations (NDT cells vs BEV image) with everything else equal:

- **Network** (`src/lpr/models/ndt_net.py`, own code): the layers of NDT-Transformer — optional T-Net
  (`stn`), conv1 12→256, Transformer encoder (3 layers, 4 heads), conv2 512→`feat_dim` — and a head from
  `src/lpr/models/heads.py`: `vlad` (the NetVLAD of BEVPlace++, 64 × `feat_dim` = 8192 with `feat_dim=128`,
  k-means initialisation, cell features L2-normalised before it, as BEVPlace++ does) or `loupe` (NetVLADLoupe of
  NDT-Transformer, 256-d, with an added final L2 normalisation so the triplet margin means the same).
  `rotations=R` adds REIN-like rotation equivariance: the cells are rotated about z by 0, 360/R, …
  (means R m, covariances R S Rᵀ), every copy goes through the same encoder, and each cell keeps the maximum
  over the copies. Cells keep their order, so no warp back is needed, and a scan rotated by a multiple of
  360/R gives exactly the same descriptor. `NDTNet(stn=True, rotations=1, head="loupe", feat_dim=1024)` is the
  official network: `tests/test_models.py` loads official weights into it and gets the same descriptors, and
  checks the NetVLAD head against the official BEVPlace++ one.
- **Recipe**: the training code of the BEV runs, unchanged (`src/lpr/bevplace.py`; a BEV sample is loaded by
  `TrainingDataset.load`, which `src/lpr/myndt.py` overrides to load NDT cells): positives < 5 m, negatives > 7 m
  among frames 0–2999, 10 negatives, batch 4, triplet loss (margin 0.3, hardest negative), hard mining from a
  descriptor cache of all training frames rebuilt every epoch, Adam 1e-4, 40 epochs, a random 0–360° rotation of
  every training sample (`rot_aug`), optionally the point-rotated copies (`rot_points`). Mixed precision
  (`amp=True`) for GPU memory; if a batch does not fit, the tuples go one by one with accumulated gradients
  (as for BEV).
  `hard_neg_after_epoch=N` (also for BEV runs; 0 = the official recipe): random negatives in epochs 1..N and
  hard mining from epoch N+1, as NDT-Transformer does. A network trained from scratch can collapse when it is
  pushed away from the hardest negatives from the first step (BEVPlace++ starts from ImageNet weights).
  `metrics.csv` row 0 is the recall right after the NetVLAD initialisation, before any training
  (`--no_eval_at_start` to skip; `--eval_at_start` for BEV runs).
- Parameters: `MYNDT_TRAIN_DEFAULTS` in `src/lpr/experiments.py`; runs in `runs/myndt/<run>/` with the same
  bookkeeping as the other runs; `eval_ndt_kitti.py` builds the network from the run's `config.json`
  (evaluations `seqs_heading` and `compass`).

## Sparsity: 32 and 16 laser beams (`src/lpr/beams.py`, evaluation type `sparsity`)

A cheaper LiDAR is simulated on the same KITTI scans: the variant parameter `beams` (64 / 32 / 16, for NDT
and BEV variants) keeps every 2nd or 4th laser ring, before crop, ground removal, cells or image. KITTI files
have no ring index; it is recovered from the order of the points (a scan is stored ring after ring; a new ring
starts where the azimuth wraps around), with the elevation angle against the nominal HDL-64E angles as a
fallback, and the rings are ranked by elevation. `scripts/check_beams.py` checks this on real scans (rings
found per scan, share of points kept, a picture with 64 / 32 / 16 beams). `beams=64` is left out of the
variant hash, so existing variants keep their names.

The sparse variants of a 64-beam variant are named by `experiments.sparse_variant`
(`<date of the base variant>_beams<N>_<hash>`) and built on Azure:
`METHOD=ndt VARIANT=2026-10-04_beams16_ec17f5 ARGS="--beams 16" CELL_SEQS="00 02 05 06 08" bash scripts/azure_build.sh "<SAS URL>"`
(BEV: `METHOD=bev VARIANT=2026-10-05_beams16_50ef74 ARGS="--beams 16"`; 32 beams: `_beams32_c9807e` / `_beams32_9d369f`).

All runs at once: `notebooks/05_sparsity_all_runs.ipynb` (every finished run and the official BEVPlace++
checkpoint, then one table). For a single run, `EVALS = [..., "sparsity"]` in notebooks 01 / 02 / 04 copies them next to the run's variant
(`scripts/fetch_sparse.py`) and evaluates the run's checkpoint without retraining, in two modes:
`both` (map and queries from the sparse sensor) and `query` (map from 64 beams, queries sparse;
`--query_dir` of the evaluation scripts). Results: `<run>/eval/sparsity/<checkpoint>/summary.csv`
(R@1 per sequence, beams and mode, with the heading groups) and `sparsity.png`.

Density augmentation in training (`beam_aug`, every method): e.g. `TRAIN_PARAMS = dict(beam_aug=[64, 32, 16])` draws
every training sample from the run's variant or one of its 32 / 16-beam variants at random (the same scan with
fewer rings; the notebooks copy the training sequence of those variants with `fetch_sparse.py`). It targets the
"query" mode, where runs trained with random rotations fail: `scripts/density_check.py` (notebook 05, section 7)
shows that their descriptor of a 16-beam scan is about as far from the same scan with 64 beams as from other places.

## Rotation curve (evaluation type `rotation`)

The map stays as recorded and every query is rotated about z by its own angle drawn from U[-alpha, alpha]
(seeded per sequence and alpha), alpha in {0, 5, 15, 45, 90, 180} degrees; Recall per sequence and alpha, with
the heading groups. Same mechanism as `compass` (`--yaw_db` / `--yaw_q` of the evaluation scripts: NDT cells
m -> R m, S -> R S R^T; BEV images rotated about their centre). No retraining. All runs at once:
`notebooks/05_sparsity_all_runs.ipynb` with `EVAL_TYPE = "rotation"` (and `ONLY_RUNS` to pick runs); one run:
`EVALS = [..., "rotation"]` in notebooks 01 / 02 / 04. Results: `<run>/eval/rotation/<checkpoint>/summary.csv`
(column `alpha_deg`) and `rotation.png`.

The 08 split of the BEVPlace++ paper (Table II: database 0-2999) differs from the official code (database
0-999, queries from 1200), which this project follows. `eval_bev_kitti.py --split 3000` (and
`eval_ndt_kitti.py --split`) evaluates with another split point; notebook 05, section 8 (`SPLIT08_CHECK`)
runs the official checkpoint with both.

## Ground removal and NDT cells (`scripts/ground_coverage.py`, notebook 06)

For 300 random scans per sequence, the cells are built twice from the same cropped scan, with Patchwork++ ground
removal (the default) and with the ground kept; Patchwork++ labels every point as ground / non-ground for both.
Per scan: coverage (% within 0.8 m of a cell mean) of all points, of the non-ground and of the ground points, % of
cells on the ground, voxel size found by G. On the Azure VM for all sequences:
`SEQS="02 05 06 08" N_SCANS=300 bash scripts/azure_ground_check.sh "<SAS URL>"`; notebook 06 copies the
results from the blob container (`ground_check/<tag>/`) and runs what is missing in Colab. The training run
without ground removal is notebook 04 with `NDT_TAG = "ground_none"`, `NDT_PARAMS = dict(ground="none")`
(variant `2026-10-26_ground_none_6aebc9`, built on Azure with `ARGS="--ground none"`).

## Resources and the method figure (notebook 07)

`scripts/benchmark.py`: on N raw scans, the time per scan of every preprocessing step on the CPU (crop + ground
removal, G with and without the ground, BEV image) and the stored size per scan; for every network (random
weights), the time per descriptor (batch 1 and batched), the peak GPU memory of inference and of one training
step with the recipe's batch (AMP), the time of a training step and an epoch / 20-epoch estimate on that GPU
(steps per epoch x step time + the descriptor caches for hard negatives; no data loading), the retrieval time in a
map of 3000 descriptors, the total time of one localisation (preprocessing + descriptor + retrieval), and the exact
number of parameters; all in `comparison.csv`. `scripts/figure_method.py`: one scan as
points, NDT cells and BEV image. Results: `MyDrive/lpr/benchmark/` (`benchmark.json`, `networks.csv`,
`preprocessing_per_scan.csv`, `fig_method.png`).

## Checks of the NDT cells (`src/lpr/ndt_check.py`, notebook 03)

Built into every extraction (`extract_ndt.py`, in Colab and on Azure): for every scan, how well the cells cover
the points left after crop and ground removal (`cov_0.8`: % of the points within 0.8 m of a cell mean;
`dense_cov`: % of the 1 m squares with >= 5 points that have a cell; coverage by range; extent of the cells),
as extra columns of `stats_<seq>.csv` and a summary in `config.json`; and pictures of 10 random scans per
sequence (the same scans for every variant): before ground removal, after ground removal, cells
(`<variant>/previews_<seq>/`).

Variants built before that (`2026-10-04_baseline_717e81`): `notebooks/03_ndt_checks.ipynb` uploads the cells
built in Colab (00) to the blob container and prints the command for the VM,
`VARIANT=<name> bash scripts/azure_coverage.sh "<SAS URL>"` (`scripts/ndt_coverage.py` on every scan of every
sequence), then copies the results to `<variant>/coverage/` and shows tables, histograms and the pictures.
Patchwork++ adapts its thresholds from scan to scan, so these recomputed points can differ slightly (< 1%)
from the ones the cells were built from.

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
`rot_points=N` instead takes every training cloud from one of N copies of its scan whose raw points were
rotated before the cells were built (`extract_ndt.py --rot_copies N`), so the cells themselves differ
(axis-aligned crop and voxel grid), as when the vehicle really turns. The angle of copy c of scan i is fixed
by (seed, sequence, i, c) and stored in `stats_<seq>_rot<N>.csv`. Hard-negative mining and evaluation use the
unrotated cells, as with `rot_aug`. Both can be combined.

## Reproducing the report

Every number comes from files on Drive written by the notebooks (`runs_summary.csv`, `evals_summary.csv`:
`scripts/summarize_runs.py`, run by every notebook). The runs, with the cell-1 settings of their notebook
(everything else at the defaults of `src/lpr/experiments.py`; the run folder name ends with the hash of the
parameters, so the same settings find the same run):

| Run | Notebook | `NDT_PARAMS` / `BEV_PARAMS` | `TRAIN_PARAMS` |
|---|---|---|---|
| `2026-10-04_baseline_4ed638` | 01 | `{}` | `{}` |
| `2026-10-06_cov_transform_5c90e2` | 01 | `{}` | `dict(cov_transform=True)` |
| `2026-10-07_f60015` | 01 | `{}` | `dict(cov_transform=True, rot_aug=True)` |
| `2026-10-12_e39427` | 01 | `{}` | `dict(cov_transform=True, rot_points=8)` |
| `2026-10-25_dbc4f4` | 01 | `{}` | `dict(cov_transform=True, rot_aug=True, beam_aug=[64, 32, 16])` |
| `2026-10-08_without_Rein_b14094` | 02 | `{}` | `dict(arch="plain", rot_aug=False, epochs=20)` |
| `2026-10-08_without_Rein_with_rot_216fa5` | 02 | `{}` | `dict(arch="plain", epochs=20)` |
| `2026-10-25_without_Rein_26962e` | 02 | `{}` | `dict(arch="plain", epochs=20, beam_aug=[64, 32, 16])` |
| `2026-10-05_baseline_4ac7d1` | 02 | `{}` | the defaults of its commit (20 epochs; stopped after 18) |
| `official_2026-10-05_baseline_7711ab` | 02 | `{}` | the authors' checkpoint, evaluated only |
| `2026-10-19_283ebc` | 04 | `{}` | `dict(epochs=20, lr=1e-5, stn=True, rotations=1, head="loupe", feat_dim=1024)` |
| `2026-10-19_6f8bf1` | 04 | `{}` | `dict(epochs=20, lr=1e-5, stn=True, rotations=1)` |
| `2026-10-20_065c48` | 04 | `{}` | `dict(epochs=20, lr=1e-5, head="loupe", feat_dim=1028)` |
| `2026-10-20_3fe7a5` | 04 | `{}` | `dict(epochs=20, lr=1e-5, rot_aug=False, head="loupe", feat_dim=1028)` |
| `2026-10-25_45cb1e` | 04 | `{}` | as 3fe7a5, `beam_aug=[64, 32, 16]` |
| `2026-11-08_2dd507` | 04 | `{}` | as 3fe7a5, `seed=1` (repeat) |
| `2026-11-08_fec6c9` | 04 | `dict(ground="none")`, `NDT_TAG="ground_none"` | as 3fe7a5, `seed=1` (repeat of eeb744) |
| `2026-10-26_eeb744` | 04 | `dict(ground="none")`, `NDT_TAG="ground_none"` | as 3fe7a5 |

(Every run's `config.json` holds its full parameters and code commit; older runs used the defaults of
their commit. `feat_dim=1028` is a typo for 1024 kept so the runs can be found again.) Tables and figures of the report:

| Report | Evaluation / script | Where |
|---|---|---|
| Recall per sequence, heading groups, main table | `seqs_heading` (notebooks 01 / 02 / 04, `EVALS`) | `evals_summary.csv`, `runs_summary.csv` |
| Compass table and figure | `compass` | same; `scripts/report_figures.py` -> `fig_compass.png` |
| Rotation curve | `rotation` (notebook 05, `EVAL_TYPE = "rotation"`) | same; `fig_rotation.png` |
| T-Net check | `tnet` (notebook 01) | same; `fig_tnet.png` |
| Sparsity tables and figure, beam_aug | `sparsity` (notebook 05) | `sparsity_all_runs.csv`; `fig_sparsity.png` |
| Density check | notebook 05, section 7 | `density_check.csv` |
| Official checkpoint with the paper's 08 split | notebook 05, section 8 | `runs/bev/official_*/eval/split08/` |
| Ground removal | notebook 06 (+ `scripts/azure_ground_check.sh`) | `ground_check/<tag>/summary_all.csv`, previews |
| Resources, method figure | notebook 07 | `benchmark/` |
| NDT cell coverage | built into `extract_ndt.py` (notebook 03 for older variants) | `ndt/<variant>/stats_<seq>.csv` |

`python scripts/report_figures.py --drive /content/drive/MyDrive/lpr --out_dir /content/drive/MyDrive/lpr/report_figures`
draws the result figures from `evals_summary.csv`.

## Tests without KITTI

`python tests/test_models.py` checks the own network against the official code (same descriptors from the same
weights), the NetVLAD head against BEVPlace++, and the exact 45° invariance with 8 rotations (CPU, a few minutes).

`python tests/test_bev_matches_official.py` compares `src/lpr/bev.py` with the official BEV script on synthetic
scans (`--scans a.bin b.bin` for real ones; needs open3d).

`tests/make_fake_kitti.py` builds a small synthetic sequence in the same zip layout and
`tests/range_server.py` serves it with HTTP range support, so the whole pipeline can be run offline:

```
python tests/make_fake_kitti.py /tmp/fake 240
python tests/range_server.py /tmp/fake 8765 &
KITTI_BASE_URL=http://127.0.0.1:8765/ python scripts/download_kitti.py --out /tmp/kitti --seqs 00
```
