"""Per-subject best-epoch selection for a pooled intra-subject run (intra_pooled.sh).

Mirrors a single-subject run's rule: each subject's best epoch is its LOWEST test loss (ties -> higher
top-1), and its reported "best top1 acc" is the top-1 at that epoch. Writes pooled_select.csv next to
per_subject_test.csv and prints the per-subject values and their mean.

    pooled_select.py <run_dir>
"""

import csv
import os
import sys

run_dir = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(run_dir, "per_subject_test.csv"))))
best = {}
for r in rows:
    key = (float(r["test_loss"]), -float(r["top1"]))
    if r["subject"] not in best or key < best[r["subject"]][0]:
        best[r["subject"]] = (key, r)
out = [best[s][1] for s in sorted(best, key=int)]
with open(os.path.join(run_dir, "pooled_select.csv"), "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["subject", "epoch", "test_loss", "top1", "top5"])
    writer.writeheader()
    for r in out:
        writer.writerow({k: r[k] for k in writer.fieldnames})
for r in out:
    print(f"sub-{int(r['subject']):02d} epoch {r['epoch']:>2} loss {r['test_loss']} top1 {r['top1']} top5 {r['top5']}")
print(f"mean best top1 {sum(float(r['top1']) for r in out) / len(out):.2f} over {len(out)} subjects")
