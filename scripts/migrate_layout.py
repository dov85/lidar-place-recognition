"""Moves older results into the current Drive layout (safe to run every time).

Step 1, runs and heading analyses under a method folder:

Before                                   After
  <drive>/runs/<run>/                ->  <drive>/runs/<method>/<run>/
  <drive>/analysis/heading/<run>/    ->  <drive>/analysis/heading/<method>/<run>/
The method of a run comes from its config.json ("bev_variant" -> bev, otherwise ndt);
a heading folder takes the method of the run with the same name.

Step 2, evaluations under <run>/eval/<type>/<checkpoint>/ (src/lpr/evals.py):
  <run>/eval_XX.json, eval_XX_matches.npz             -> <run>/eval/seqs_heading/<ckpt>/
  <drive>/analysis/heading/<method>/<run>/*           -> the same folder
  other old evaluation files (eval_00_epoch_20.pth*)  -> <run>/eval/seqs_heading/<ckpt>/legacy/
<ckpt> is read from the "ckpt" field of the evaluation files. A config.json and a
summary.csv are written there, so the notebooks see these evaluations as done.
Only what is still at the old place is moved.

  python scripts/migrate_layout.py --drive /content/drive/MyDrive/lpr          (plan)
  python scripts/migrate_layout.py --drive /content/drive/MyDrive/lpr --apply  (move)
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import evals  # noqa: E402

METHODS = ("ndt", "bev")


def run_method(run_dir):
    try:
        cfg = json.load(open(os.path.join(run_dir, "config.json")))
    except (OSError, ValueError):
        return None
    return cfg.get("method") or ("bev" if "bev_variant" in cfg else "ndt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    plan, methods = [], {}
    runs = os.path.join(a.drive, "runs")
    for name in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        src = os.path.join(runs, name)
        if name in METHODS or not os.path.isdir(src):
            continue
        m = run_method(src)
        if m is None:
            print("skipped (no config.json):", src)
            continue
        methods[name] = m
        plan.append((src, os.path.join(runs, m, name)))
    heading = os.path.join(a.drive, "analysis", "heading")
    for name in sorted(os.listdir(heading)) if os.path.isdir(heading) else []:
        src = os.path.join(heading, name)
        if name in METHODS or not os.path.isdir(src):
            continue
        m = methods.get(name) or ("bev" if os.path.isdir(os.path.join(runs, "bev", name)) else "ndt")
        plan.append((src, os.path.join(heading, m, name)))

    if plan:
        for src, dst in plan:
            print(("exists, kept " if os.path.exists(dst) else "move ") + src + "\n  -> " + dst)
        if a.apply:
            for src, dst in plan:
                if not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
    moved = migrate_evals(a.drive, a.apply)
    if not plan and not moved:
        print("nothing to migrate")
    elif not a.apply:
        print("\n(dry run; add --apply to move)")
    else:
        print("done")


def migrate_evals(drive, apply):
    """Step 2 (see the module docstring). Returns True if anything was (or would be) moved."""
    any_moved = False
    for method in METHODS:
        for run_dir in sorted(glob_dirs(os.path.join(drive, "runs", method))):
            files = sorted(os.listdir(run_dir))
            cur = [f for f in files if re.fullmatch(r"eval_\d\d(\.json|_matches\.npz)", f)]
            legacy = [f for f in files if f.startswith("eval_") and f not in cur
                      and (f.endswith(".json") or f.endswith(".npz"))]
            head_src = os.path.join(drive, "analysis", "heading", method, os.path.basename(run_dir))
            if not cur and not legacy and not os.path.isdir(head_src):
                continue
            stems = collections.Counter(
                evals.ckpt_name(json.load(open(os.path.join(run_dir, f))).get("ckpt", "unknown"))
                for f in cur if f.endswith(".json"))
            stem = stems.most_common(1)[0][0] if stems else "unknown"
            dst = os.path.join(run_dir, "eval", "seqs_heading", stem)
            any_moved = True
            print(f"{run_dir}\n  {len(cur)} evaluation files, {len(legacy)} older ones"
                  + (", heading analysis" if os.path.isdir(head_src) else "") + f"\n  -> {dst}")
            if not apply:
                continue
            os.makedirs(dst, exist_ok=True)
            for f in cur:
                if not os.path.exists(os.path.join(dst, f)):
                    shutil.move(os.path.join(run_dir, f), os.path.join(dst, f))
            if legacy:
                os.makedirs(os.path.join(dst, "legacy"), exist_ok=True)
                for f in legacy:
                    shutil.move(os.path.join(run_dir, f), os.path.join(dst, "legacy", f))
            if os.path.isdir(head_src):
                for f in os.listdir(head_src):
                    if not os.path.exists(os.path.join(dst, f)):
                        shutil.move(os.path.join(head_src, f), os.path.join(dst, f))
                if not os.listdir(head_src):
                    os.rmdir(head_src)
            seqs = [s for s in evals.ALL_SEQS if os.path.exists(os.path.join(dst, f"eval_{s}.json"))]
            if seqs and not os.path.exists(os.path.join(dst, "config.json")):
                ck = json.load(open(os.path.join(dst, f"eval_{seqs[0]}.json"))).get("ckpt", stem)
                complete = seqs == evals.ALL_SEQS and os.path.exists(os.path.join(dst, "heading_summary.csv"))
                evals.write_config(dst, "seqs_heading", {"seqs": seqs}, ck, run_dir, complete)
                cfg = json.load(open(os.path.join(dst, "config.json")))
                cfg["migrated"] = True
                json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
                evals.write_seqs_summary(dst, seqs)
    if apply:  # drop the old analysis folders once they are empty
        for d in (os.path.join(drive, "analysis", "heading", m) for m in METHODS):
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        for d in (os.path.join(drive, "analysis", "heading"), os.path.join(drive, "analysis")):
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
    return any_moved


def glob_dirs(d):
    return [os.path.join(d, n) for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))] \
        if os.path.isdir(d) else []


if __name__ == "__main__":
    main()
