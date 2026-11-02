"""Stage a Neurobridge eeg_data_dir onto node-local disk so training reads its memmaps locally.

module/dataset.py opens <eeg_data_dir>/.cache/neurobridge_eeg/<key>.npy (header only when a
<key>_fp16.npy sibling exists: un-averaged runs read that fp16 memmap) and resolves
sub-XX/{train,test}.npy even when a cache hit means they are never read. So the staged tree
copies each fp16 memmap and each small cache without one (e.g. repetition-averaged test sets),
and symlinks everything else back to the NAS. If a cache is not built on the NAS yet, or the copy
would leave less than 20% of the (shared) local disk free, nothing is copied and the run reads the
NAS memmaps directly.

    stage_eeg_cache.py <eeg_data_dir> <local_dir> --subjects 1 2 ... --selected_channels ... \
        --time_window 0 120 [--reserve_gb 4]
Prints the directory to pass as --eeg_data_dir: <local_dir> if everything fits with
--reserve_gb to spare, else <eeg_data_dir> unchanged (run straight off the NAS).
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module.dataset import _eeg_cache_key  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("eeg_data_dir")
p.add_argument("local_dir")
p.add_argument("--subjects", type=int, nargs="+", required=True, help="training subjects")
p.add_argument("--test_subjects", type=int, nargs="*", default=None, help="default: --subjects")
p.add_argument("--selected_channels", nargs="*", default=[])
p.add_argument("--time_window", type=int, nargs=2, default=[0, 250])
p.add_argument("--reserve_gb", type=float, default=4.0)
args = p.parse_args()

def warm_page_cache(paths, total):
    """Read the NAS memmaps once, sequentially, so training's random single-trial reads hit RAM (page cache)
    instead of the NAS: batch-1024 conv models otherwise run ~10 s/step off the NAS. Only if they fit in half
    the job's memory limit (page cache counts against it)."""
    limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")  # machine RAM
    if os.environ.get("SLURM_MEM_PER_NODE"):  # the job's --mem, in MB
        limit = min(limit, int(os.environ["SLURM_MEM_PER_NODE"]) << 20)
    # ponytail: 0.65, not 0.5 -- the 18.1 GiB *_e1_things set (+THINGS-EEG1) runs at full speed warmed on a 30 GiB job
    # (trainer RSS 2-4 GiB, 2026-10-13) and at 1/5 speed unwarmed; lower it again if a bigger model OOMs
    if total > 0.65 * limit:
        print(f"stage_eeg_cache: {total / 2**30:.1f} GiB > 65% of the {limit / 2**30:.0f} GiB memory limit; "
              "not warming the page cache", file=sys.stderr)
        return
    for path in paths:
        with open(path, "rb") as handle:
            while handle.read(64 << 20):
                pass
    print(f"stage_eeg_cache: warmed {len(paths)} NAS memmaps ({total / 2**30:.1f} GiB) into the page cache",
          file=sys.stderr)


src_cache = os.path.join(args.eeg_data_dir, ".cache", "neurobridge_eeg")
dst_cache = os.path.join(args.local_dir, ".cache", "neurobridge_eeg")
copies, links, missing = [], [], []  # (src, dst)
test_subjects = args.subjects if args.test_subjects is None else args.test_subjects
for sub in sorted(set(args.subjects) | set(test_subjects)):
    name = f"sub-{sub:02d}"
    for f in os.listdir(os.path.join(args.eeg_data_dir, name)):
        links.append((os.path.join(args.eeg_data_dir, name, f), os.path.join(args.local_dir, name, f)))
    # single-trial train (training), averaged test (train.py eval), single-trial test (sealed scorer)
    for train, average in ((True, False), (False, True), (False, False)):
        if (sub in args.subjects) != train and not (train is False and sub in test_subjects):
            continue  # training reads train caches of training subjects, scoring test caches of test subjects
        key = _eeg_cache_key(subject_id=sub, train=train, average=average,
                             selected_channels=args.selected_channels, time_window=args.time_window)
        base, fp16 = os.path.join(src_cache, f"{key}.npy"), os.path.join(src_cache, f"{key}_fp16.npy")
        if os.path.isfile(fp16):
            copies.append((fp16, os.path.join(dst_cache, os.path.basename(fp16))))
            links.append((base, os.path.join(dst_cache, os.path.basename(base))))
        elif os.path.isfile(base):
            copies.append((base, os.path.join(dst_cache, os.path.basename(base))))
        else:
            missing.append((sub, train, average))

if missing:
    # the loader would build these into the local copy: ~1 GB each on a shared /tmp, and lost with the copy
    print(f"stage_eeg_cache: {len(missing)} caches not built on the NAS yet {missing[:3]}...; using the NAS directly",
          file=sys.stderr)
    print(args.eeg_data_dir)
    sys.exit(0)

copies = [(s, d) for s, d in copies if not os.path.isfile(d) or os.path.getsize(d) != os.path.getsize(s)]
need = sum(os.path.getsize(s) for s, _ in copies)  # only what is not staged yet (a job reuses its copy)
os.makedirs(dst_cache, exist_ok=True)
usage = shutil.disk_usage(args.local_dir)
free, reserve = usage.free, max(args.reserve_gb * 2**30, 0.2 * usage.total)  # /tmp is shared with other users
if need + reserve > free:
    print(f"stage_eeg_cache: need {need / 2**30:.1f} GiB + {reserve / 2**30:.0f} reserve, "
          f"{free / 2**30:.1f} free; using the NAS directly", file=sys.stderr)
    warm_page_cache([s for s, _ in copies], need)
    print(args.eeg_data_dir)
    sys.exit(0)
for s, d in links:
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if not os.path.lexists(d):
        os.symlink(s, d)
for s, d in copies:
    shutil.copyfile(s, d)
shutil.copyfile(os.path.join(args.eeg_data_dir, "info.json"), os.path.join(args.local_dir, "info.json"))
print(f"stage_eeg_cache: copied {len(copies)} files ({need / 2**30:.1f} GiB), linked {len(links)}",
      file=sys.stderr)
print(args.local_dir)
