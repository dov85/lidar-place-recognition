"""One-off: put runs and heading analyses under a method folder (ndt / bev).

Before                                   After
  <drive>/runs/<run>/                ->  <drive>/runs/<method>/<run>/
  <drive>/analysis/heading/<run>/    ->  <drive>/analysis/heading/<method>/<run>/
The method of a run comes from its config.json ("bev_variant" -> bev, otherwise ndt);
a heading folder takes the method of the run with the same name. Variants already live
in <drive>/ndt/ and <drive>/bev/. Safe to run again: it only moves what is still at the
old place.

  python scripts/migrate_layout.py --drive /content/drive/MyDrive/lpr          (plan)
  python scripts/migrate_layout.py --drive /content/drive/MyDrive/lpr --apply  (move)
"""
import argparse
import json
import os
import shutil

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

    if not plan:
        print("nothing to migrate")
        return
    for src, dst in plan:
        print(("exists, kept " if os.path.exists(dst) else "move ") + src + "\n  -> " + dst)
    if not a.apply:
        print("\n(dry run; add --apply to move)")
        return
    for src, dst in plan:
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
    print("done")


if __name__ == "__main__":
    main()
