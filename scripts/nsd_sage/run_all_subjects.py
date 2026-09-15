"""Reproduce MindEye2's retrieval table over all four evaluated subjects.

Table 1 averages subjects 1, 2, 5 and 7, so a single subject cannot confirm or refute the
published numbers -- subject 1 alone scores well above the mean in both settings. This
downloads what each subject needs, extracts clipvoxels and evaluates retrieval, then prints
the four-subject average next to the published values.

Published (Table 1, avg over subj 1,2,5,7):
    MindEye2 (1 hour)      image 79.0   brain 57.4
    MindEye2 (40 sessions) image 98.8   brain 98.3

Usage:  python run_all_subjects.py            # all four subjects, both settings
        python run_all_subjects.py --subjects 2 5 7 --settings 1sess
"""
import argparse
import re
import subprocess
import sys

HERE = "/nasbrain/p20fores/Neurobridge_SSL/scripts/nsd_sage"
PY = "/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python"
DATA = "/nasbrain/p20fores/mindeye_data"
PUBLISHED = {"1sess": (79.0, 57.4), "40sess": (98.8, 98.3)}


def fetch(subj, setting):
    from huggingface_hub import hf_hub_download
    ss = f"{subj:02d}"
    for f in [f"betas_all_subj{ss}_fp32_renorm.hdf5",
              f"wds/subj{ss}/new_test/0.tar",
              f"train_logs/final_subj{ss}_pretrained_{setting}_24bs/last.pth"]:
        hf_hub_download("pscotti/mindeyev2", f, repo_type="dataset", local_dir=DATA)
        print(f"  have {f}", flush=True)


def run(subj, setting):
    model = f"final_subj{subj:02d}_pretrained_{setting}_24bs"
    subprocess.run([PY, f"{HERE}/extract_clipvoxels.py", "--subj", str(subj),
                    "--model_name", model], check=True, capture_output=True)
    out = subprocess.run([PY, f"{HERE}/reproduce_retrieval.py", "--subj", str(subj),
                          "--model_name", model], check=True, capture_output=True, text=True).stdout
    # "300-way  image retrieval (fwd) 94.2%   brain retrieval (bwd) 78.3%   chance 0.3%"
    m300 = re.search(r"300-way\s+image retrieval \(fwd\) ([\d.]+)%\s+brain retrieval \(bwd\) ([\d.]+)%", out)
    mfull = re.search(r"\d+-way image retrieval \(fwd\) ([\d.]+)%\s+brain retrieval \(bwd\) ([\d.]+)%", out)
    if not m300 or not mfull:
        raise RuntimeError(f"could not parse retrieval output:\n{out}")
    return tuple(float(x) for x in m300.groups()), tuple(float(x) for x in mfull.groups())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--settings", nargs="+", default=["1sess", "40sess"])
    args = ap.parse_args()

    results = {}
    for setting in args.settings:
        for subj in args.subjects:
            print(f"=== subj{subj:02d} {setting} ===", flush=True)
            try:
                fetch(subj, setting)
                results[(setting, subj)] = run(subj, setting)
                (i3, b3), (i1k, b1k) = results[(setting, subj)]
                print(f"  300-way {i3:5.1f} / {b3:5.1f}    1000-way {i1k:5.1f} / {b1k:5.1f}", flush=True)
            except Exception as e:
                print(f"  FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)

    print("\n================ summary (300-way, the published protocol) ================")
    for setting in args.settings:
        rows = [(s, results[(setting, s)][0]) for s in args.subjects if (setting, s) in results]
        if not rows:
            continue
        print(f"\n{setting}:")
        for s, (i, b) in rows:
            print(f"  subj{s:02d}   image {i:5.1f}   brain {b:5.1f}")
        mi = sum(i for _, (i, _) in rows) / len(rows)
        mb = sum(b for _, (_, b) in rows) / len(rows)
        pi, pb = PUBLISHED[setting]
        print(f"  MEAN     image {mi:5.1f}   brain {mb:5.1f}   (n={len(rows)})")
        print(f"  PAPER    image {pi:5.1f}   brain {pb:5.1f}")
        print(f"  DELTA    image {mi-pi:+5.1f}   brain {mb-pb:+5.1f}")


if __name__ == "__main__":
    sys.exit(main())
