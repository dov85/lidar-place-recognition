"""One evaluation protocol for every method: descriptors in, recalls out.

Recall@1: fraction of queries whose top-1 database match is within the
threshold. Recall@1%: at least one of the top ceil(1% of the database) matches
is within the threshold. Queries without any database frame within the
threshold are not counted (as in the BEVPlace++ evaluation code).
"""
import math

import numpy as np


def recall(db_desc, q_desc, db_xyz, q_xyz, threshold=5.0, top_ns=(1, 5, 10)):
    db_desc = np.asarray(db_desc, dtype=np.float32)
    q_desc = np.asarray(q_desc, dtype=np.float32)
    one_pct = max(int(math.ceil(len(db_desc) / 100.0)), 1)
    k = max(max(top_ns), one_pct)

    # descriptor distances (squared L2), processed in chunks to bound memory
    order = np.empty((len(q_desc), k), dtype=np.int64)
    db_sq = (db_desc ** 2).sum(1)
    for s in range(0, len(q_desc), 1024):
        q = q_desc[s:s + 1024]
        d = (q ** 2).sum(1)[:, None] + db_sq[None] - 2 * q @ db_desc.T
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        order[s:s + 1024] = part[rows, np.argsort(d[rows, part], axis=1)]

    geo = np.linalg.norm(q_xyz[:, None, :] - db_xyz[None, :, :], axis=2)
    has_pos = (geo < threshold).any(1)
    hit = np.take_along_axis(geo, order, axis=1) < threshold
    hit = hit[has_pos]
    out = {f"R@{n}": float(hit[:, :n].any(1).mean() * 100) for n in top_ns}
    out["R@1%"] = float(hit[:, :one_pct].any(1).mean() * 100)
    out["top1pct_k"] = one_pct
    out["n_queries"] = int(len(q_desc))
    out["n_queries_with_positive"] = int(has_pos.sum())
    out["top1_indices"] = order[:, 0]
    return out
