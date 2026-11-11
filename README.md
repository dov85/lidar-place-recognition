# LiDAR place recognition: NDT-Transformer vs BEVPlace++

Code of the final project of Deep Learning for 3D Computer Vision (Hebrew University): a controlled comparison of
NDT-Transformer and BEVPlace++ on KITTI, along two axes (rotation / compass, and scan sparsity), with an own
re-implementation of NDT-Transformer that can be trained with the BEVPlace++ recipe.

## Structure

```
notebooks/            the three Colab notebooks that run everything (settings in cell 1, then Run all)
  1_data.ipynb          NDT cells / BEV images: commands for the Azure VM, blob container -> Drive,
                        optional: build in Colab, BEV pixel check, ground check
  2_train.ipynb         training of one run: METHOD = "ndt" | "bev" | "myndt"
  3_evaluate.ipynb      evaluations of all runs, tables, density check, 08 split check, resources and figures
src/lpr/              the library (imported by the scripts)
  kitti.py              scans, poses, the place-recognition split of every sequence
  preprocess.py         crop (80 x 80 m square) and ground removal (Patchwork++)
  ndt_g.py              G: the 2000 NDT cells of a scan (not in the official repository)
  ndt_check.py          coverage checks of the cells, run by every extraction
  bev.py                BEV density images (vectorised, pixel-identical to the official script)
  beams.py              fewer laser beams: ring recovery and thinning 64 -> 32 / 16
  rotation.py           rotations about z of NDT cells and BEV images (augmentation, compass, rotation curve)
  bevplace.py           the BEVPlace++ training recipe (training set, hard mining, loss, NetVLAD init)
  myndt.py              the data of the own network inside that recipe
  models/ndt_net.py     own NDT network: T-Net, Transformer, REIN-like rotations, either head
  models/heads.py       NetVLAD (BEVPlace++) and NetVLADLoupe (NDT-Transformer)
  evaluation.py         one protocol for every method: descriptors in, Recall@1 / Recall@1% out
  evals.py              evaluation types and where their results live
  experiments.py        parameters, defaults and names of variants and runs (hash of the parameters)
  colab.py              helpers of the notebooks (run a script, copy a variant to the machine)
scripts/
  setup_third_party.sh  clones the official repositories at fixed commits into third_party/
  download_kitti.py     only the needed parts of KITTI Odometry (range requests into the zip)
  azure_build.sh        builds the cells / images of several sequences on an Azure VM -> blob container
  azure_ground_check.sh ground check on the Azure VM
  fetch_from_azure.py   blob container -> Drive;  fetch_sparse.py: the 32 / 16-beam variants
  extract_ndt.py        cells of every scan of a sequence;  extract_bev.py: BEV images
  make_kitti_tuples.py  training tuples of NDT-Transformer (official pickle format)
  train_ndt_kitti.py    official NDT-Transformer, official recipe
  train_bev_kitti.py    official BEVPlace++ (or arch="plain"), BEVPlace++ recipe
  train_myndt_kitti.py  own NDT network, BEVPlace++ recipe
  evaluate.py           one evaluation type on one checkpoint (calls the scripts below)
  eval_ndt_kitti.py, eval_bev_kitti.py   descriptors and recall of one sequence
  heading_analysis.py   query heading vs map heading, recall per heading group
  tnet_check.py         what the T-Net learned (scans rotated about z)
  density_check.py      16-beam vs 64-beam descriptors of the same scans
  ground_coverage.py    NDT cells with and without ground removal
  summarize_runs.py     runs_summary.csv and evals_summary.csv (the source of the report's tables)
  benchmark.py          time, memory and parameters of the pipelines and networks
  figure_method.py, figure_networks.py   Figures 1 and 2 of the report
  check_beams.py        check of the ring recovery on real scans
tests/
  test_models.py        own network = official network (same weights, same descriptors), NetVLAD head, 45 deg invariance
  test_bev_matches_official.py   src/lpr/bev.py = the official BEV script, pixel by pixel
requirements.txt
```

## What is mine and what is official

The official repositories are not copied here; `scripts/setup_third_party.sh` clones them into `third_party/`
(NDT-Transformer @ `51f90df`, BEVPlace2 @ `2d0636d`). Used from them, unchanged: the NDT-Transformer network,
its lazy quadruplet loss, tuple sampling and hard-negative scheme (`train_ndt_kitti.py` imports them); the
BEVPlace++ network (REIN) and the authors' checkpoint (`train_bev_kitti.py`, `eval_bev_kitti.py` import them).
Reproduced from the official code: the BEVPlace++ training recipe in `src/lpr/bevplace.py` (training set, hard
mining, loss, NetVLAD k-means initialisation; from `main.py` and `kitti_dataset.py`) and the BEV image in
`src/lpr/bev.py` (same output as `gen_bev_images.py`, checked by `tests/test_bev_matches_official.py`).
Everything else is mine: NDT cell generation G, cropping and ground removal, beam thinning, the own network and
its heads (`tests/test_models.py` checks them against the official code), the training loops, the evaluation
protocol and all evaluation types, the Azure / Drive data pipeline and the notebooks.

## Setup

Colab with a GPU runtime. The notebooks clone the code from `REPO_URL` (a private repository needs a GitHub token
in Colab Secrets as `GITHUB_TOKEN`), or copy it from `CODE_ON_DRIVE` (a copy of this folder on Google Drive), run `scripts/setup_third_party.sh` and `pip install -r requirements.txt`
(torch / torchvision are preinstalled on Colab). Everything that is kept lives on Google Drive in `MyDrive/lpr/`:

```
lpr/ndt/<variant>/        config.json, cells_<seq>.tar, stats_<seq>.csv      (NDT cells, one tar per sequence)
lpr/bev/<variant>/        config.json, imgs_<seq>.tar, stats_<seq>.csv       (BEV images)
lpr/runs/<method>/<run>/  config.json, metrics.csv, epoch_XX.pth             (method: ndt, bev, myndt)
lpr/runs/<method>/<run>/eval/<type>/<checkpoint>/   one evaluation (config.json, summary.csv, figures)
lpr/runs_summary.csv, lpr/evals_summary.csv         all runs and all evaluations
```

Variant and run names end with the hash of their parameters, so the same settings always find the same folder:
data is built once, an interrupted run continues, a finished run is not trained again, and a complete evaluation is
not repeated. Every `config.json` holds the full parameters and the code commit.

Data: the cells and images of KITTI 00, 02, 05, 06, 08 are built on an Azure VM (4 cores) and kept in an Azure Blob
container; the container SAS URL (read + list for Colab) goes into Colab Secrets as `AZURE_SAS`. Notebook 1 prints
the exact commands for the VM. Without Azure, notebook 1 can build NDT cells in Colab (`BUILD_HERE`, about 0.5 s
per scan).

## Reproducing the results

1. **Data** (notebook 1, once per variant): `METHOD="ndt"` and `METHOD="bev"` with the default parameters, with
   `BEAMS=[32, 16]` (sparsity). For NDT-REIN-ground also `METHOD="ndt"`, `VARIANT_TAG="ground_none"`,
   `VARIANT_PARAMS=dict(ground="none")`.
2. **Training** (notebook 2, one run per row; `VARIANT_PARAMS` empty unless stated):

| Run (report) | `METHOD` | `TRAIN_PARAMS` |
|---|---|---|
| NDT-orig | `ndt` | `dict(cov_transform=True)` |
| NDT-rot | `ndt` | `dict(cov_transform=True, rot_aug=True)` |
| NDT-BEVrecipe | `myndt` | `dict(epochs=20, lr=1e-5, stn=True, rotations=1, head="loupe", feat_dim=1024)` |
| NDT-BEVhead | `myndt` | `dict(epochs=20, lr=1e-5, stn=True, rotations=1)` |
| NDT-REIN | `myndt` | `dict(epochs=20, lr=1e-5, rot_aug=False, head="loupe", feat_dim=1028)` |
| NDT-REIN-rot | `myndt` | `dict(epochs=20, lr=1e-5, head="loupe", feat_dim=1028)` |
| NDT-REIN-ground | `myndt` | as NDT-REIN, with `VARIANT_TAG="ground_none"`, `VARIANT_PARAMS=dict(ground="none")` |
| BEV-plain | `bev` | `dict(epochs=20, arch="plain", rot_aug=False)` |
| BEV-plain-rot | `bev` | `dict(epochs=20, arch="plain")` |
| BEV-REIN-ours | `bev` | `dict(epochs=20)` (the reported run stopped after epoch 18: `EVAL_CKPT = "epoch_18.pth"`) |
| BEV-REIN-authors | – | not trained: the authors' checkpoint, evaluated by notebook 3 (`OFFICIAL_BEV = True`) |

   (`feat_dim=1028` is a typo for 1024 that the runs were made with; it is kept so the runs are found again.)

3. **Evaluation** (notebook 3, all runs at once): the default `EVAL_TYPES` give the main table, the heading groups,
   the compass and rotation curves, the sparsity tables and the T-Net check; `DENSITY_CHECK=True` the density check,
   `SPLIT08_CHECK=True` the 08 split check, `BENCHMARK=True` the resource tables and Figures 1-2.
   Section 5 prints the tables; all numbers are in `MyDrive/lpr/evals_summary.csv`.
4. **Ground check** (notebook 1, `GROUND_CHECK=True`): the coverage table of the ground-removal section.

## Tests

`python tests/test_models.py` (CPU, a few minutes) and `python tests/test_bev_matches_official.py` (needs open3d;
synthetic scans, or `--scans a.bin b.bin` for real ones), after `bash scripts/setup_third_party.sh`.
