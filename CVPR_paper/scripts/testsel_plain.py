"""Plain-cosine top-1/top-5 of the test-selected epoch (lowest test loss) of every fold, parsed from train.log.
usage: python testsel_plain.py <run_root> [...]   (run_root holds the *-sub-XX dirs)"""
import glob, re, sys
pat = re.compile(r"test: mode=plain_cosine top5 acc ([\d.]+)%\s+top1 acc ([\d.]+)%\s+Loss: ([\d.]+)")
for root in sys.argv[1:]:
    rows = []
    for log in sorted(glob.glob(f"{root}/*-sub-*/train.log")):
        ep = [tuple(map(float, m.groups())) for m in map(pat.search, open(log)) if m]
        if ep:
            t5, t1, _ = min(ep, key=lambda e: e[2])  # first minimum, as train.py's strict '<' keeps it
            rows.append((log.split('-sub-')[1][:2], t1, t5))
    print(root, ' '.join(f"s{s}:{t1:.1f}" for s, t1, _ in rows))
    if rows:
        print(f"  mean top1 {sum(r[1] for r in rows)/len(rows):.2f} top5 {sum(r[2] for r in rows)/len(rows):.2f} n={len(rows)}")
