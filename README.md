# LiDAR place recognition: NDT-Transformer vs BEVPlace++

Final project, Deep Learning for 3D Computer Vision (Hebrew University).
A controlled comparison of NDT-Transformer and BEVPlace++ under one evaluation protocol.

## What is mine and what is official

| Part | Source |
|---|---|
| Network, lazy quadruplet loss, tuple sampling, hard-negative scheme (NDT-Transformer) | official repo, `third_party/NDT-Transformer` @ `51f90df` |
| BEVPlace++ network, BEV images, training | official repo, `third_party/BEVPlace2` @ `2d0636d` |
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

## Design decisions in G (see `src/lpr/ndt_g.py`)

From the paper: k = 2000 cells, 0.8 m, binary search on the resolution, removal of redundant
cells. From the authors (D. Adolfsson, personal communication, Sept. 2026): the search targets
0-10% more cells than required. Assumptions of this implementation:

- r = 0.8 m is the radius used to compute each cell from the original points;
- a cell needs at least 5 points;
- redundancy = largest Bhattacharyya coefficient with one of the 8 nearest cells, removed greedily;
- coordinates are divided by 40 m (means in [-1, 1]); the official loader multiplies covariances by 1000.

Also note: in the official code `convariance_transform = False`, so the T-Net transform is applied
to the means only, not propagated to the covariances as described in the paper.

## Tests without KITTI

`tests/make_fake_kitti.py` builds a small synthetic sequence in the same zip layout and
`tests/range_server.py` serves it with HTTP range support, so the whole pipeline can be run offline:

```
python tests/make_fake_kitti.py /tmp/fake 240
python tests/range_server.py /tmp/fake 8765 &
KITTI_BASE_URL=http://127.0.0.1:8765/ python scripts/download_kitti.py --out /tmp/kitti --seqs 00
```
