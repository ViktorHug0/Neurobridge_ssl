#!/usr/bin/env python
"""Per-meta-subject accuracy and the spread across the 18 meta-subjects.

Each meta-subject is 4 EEG1 subjects averaged on the same stimuli; one shared EEGProject model per
shuffle scores all 9 of its meta-subjects. 200-way retrieval (NOT comparable to the 150-way
single-subject intra bars).

Averaging depth and the count of images no group member saw are printed alongside the accuracy:
if the spread across meta-subjects tracks those instead of group membership, it is an artefact of
which validation images the members happened to see, not meta-subject variability.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

DATA = Path("/nasbrain/p20fores/NICE-EEG/Data/Things-EEG1/Meta_subjects")


def main(out_dir):
    rows = {}  # (shuffle, meta) -> top1
    for path in Path(out_dir, "per_meta").glob("*/result.csv"):
        m = re.search(r"-shuffle(\d+)_meta(\d+)$", path.parent.name)
        if m:
            r = list(csv.DictReader(open(path)))[-1]
            rows[(int(m.group(1)), int(m.group(2)))] = float(r["top1 acc"])

    if not rows:
        print("no per-meta results yet")
        return

    allv = []
    for sh in sorted({s for s, _ in rows}):
        man = {}
        gj = DATA / f"shuffle{sh}" / "groups.json"
        if gj.is_file():
            man = json.load(open(gj))["groups"]
        print(f"\nshuffle {sh}")
        print(f"{'meta':>6} {'top1':>7} {'depth':>6} {'unseen':>7}   members")
        for (s, g), acc in sorted(rows.items()):
            if s != sh:
                continue
            info = man.get(f"sub-{g:02d}", {})
            print(f"{g:>6} {acc:7.2f} {info.get('test_mean_depth', float('nan')):6.2f} "
                  f"{info.get('test_images_unseen_by_group', -1):7d}   {info.get('members', '')}")
            allv.append(acc)

    print(f"\n{len(allv)} meta-subjects: mean {statistics.fmean(allv):.2f}  "
          f"sd {statistics.stdev(allv):.2f}  min {min(allv):.2f}  max {max(allv):.2f}  "
          f"range {max(allv) - min(allv):.2f} pp\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/things_eeg1/meta-subjects")
