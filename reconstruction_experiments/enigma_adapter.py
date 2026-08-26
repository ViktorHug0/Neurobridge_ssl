"""Make our THINGS-EEG2 data readable by the ENIGMA reference implementation.

The EEG itself already matches: /nasbrain/p20fores/NICE-EEG/.../sub-XX/preprocessed_eeg_{training,
test}.npy carry `preprocessed_eeg_data` (16540,4,63,250) / (200,80,63,250), `ch_names` and `times`,
and our 63 channel names equal ENIGMA's config list. Only two things are missing:

  1. experiment_metadata.parquet per subject -- one row per STIMULUS, in the same order as the EEG
     array's first axis, with partition / image_path / category_name / category_num / subject.
  2. ViT-H-14_features_{train,test}.pt -- {"image": {image_path: vec}, "text": {category: vec}}.
     We have the image side as .npy already; the text side needs open_clip.

The ordering is the whole ballgame: THINGS-EEG2's preprocessed arrays follow sorted concept dirs and
sorted files within them, which is the same convention caption_bridge.embeddings() relies on. Row i
of the EEG must be image i of that walk, so build both from one list and assert the counts.

  .venv/bin/python enigma_adapter.py --split both
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

ROOT = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"
EEG = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz"
FEAT = "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/ViT-H-14_final"
OUT = "/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/enigma"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
# averaging one subject reads ~4GB; only build what we actually train on
SUBJECTS_TODO = SUBJECTS


def stimulus_walk(split):
    """(image_path, category_name, category_num) per stimulus, in preprocessed-EEG row order."""
    d = os.path.join(ROOT, f"{split}_images")
    rows = []
    for num, concept in enumerate(sorted(os.listdir(d))):
        for f in sorted(glob.glob(os.path.join(d, concept, "*"))):
            # concept dirs are "00001_aircraft_carrier"; the readable half is the text-feature key
            rows.append((f, concept.split("_", 1)[1], num))
    return rows


def build_eeg(split):
    """ENIGMA's select_channels wants (trials, channels, time); ours keeps a repetition axis.

    The paper averages repetitions for both splits ("4 image repetitions per training sample, and 80
    per inference sample... we average together these multiple trials"), which collapses
    (N, reps, 63, 250) -> (N, 63, 250) and makes one metadata row per stimulus correct. average_eeg()
    then no-ops on the test split, since every stimulus is already a single row.
    """
    out_paths = []
    for s in SUBJECTS_TODO:
        src = os.path.join(EEG, s, f"preprocessed_eeg_{split}.npy")
        d = np.load(src, allow_pickle=True)
        d = d if isinstance(d, dict) else d.item()
        x = d["preprocessed_eeg_data"]
        assert x.ndim == 4, f"{src}: expected (N, reps, ch, t), got {x.shape}"
        avg = x.mean(axis=1)
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
        p = os.path.join(OUT, s, f"preprocessed_eeg_{split}_avg.npy")
        np.save(p, {"preprocessed_eeg_data": avg, "ch_names": d["ch_names"], "times": d["times"]})
        out_paths.append(p)
    print(f"  {split}: averaged {x.shape} -> {avg.shape} for {len(SUBJECTS_TODO)} subject(s)")
    return out_paths


def build_metadata(splits):
    """One parquet per subject holding BOTH partitions -- get_metadata() filters it by `partition`
    and resets the index, so only the row order *within* each partition has to match the EEG."""
    per_split = {}
    frames = []
    for split, partition in splits:
        rows = stimulus_walk(split)
        d = np.load(os.path.join(EEG, "sub-01", f"preprocessed_eeg_{split}.npy"), allow_pickle=True)
        d = d if isinstance(d, dict) else d.item()  # same guard as ENIGMA's dataset.py
        n_eeg = d["preprocessed_eeg_data"].shape[0]
        assert len(rows) == n_eeg, f"{split}: {len(rows)} images walked but {n_eeg} EEG rows"
        df = pd.DataFrame(rows, columns=["image_path", "category_name", "category_num"])
        df["partition"] = partition
        frames.append(df)
        per_split[split] = rows
        print(f"  {split}: {len(rows)} stimuli match {n_eeg} EEG rows")

    both = pd.concat(frames, ignore_index=True)
    for i, s in enumerate(SUBJECTS, start=1):
        df = both.copy()
        df["subject"] = i
        df["dropped"] = False
        os.makedirs(os.path.join(OUT, s), exist_ok=True)
        df.to_parquet(os.path.join(OUT, s, "experiment_metadata.parquet"), engine="pyarrow")
    print(f"wrote experiment_metadata.parquet x {len(SUBJECTS)} subjects -> {OUT}/<sub>/")
    return per_split


def build_features(split, rows, device):
    import open_clip
    img = np.load(os.path.join(FEAT, f"image_{'train' if split == 'training' else 'test'}.npy"))
    img = torch.from_numpy(img.reshape(-1, img.shape[-1])).float()
    assert len(img) == len(rows), f"{len(img)} image features vs {len(rows)} stimuli"

    cats = sorted({c for _, c, _ in rows})
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
    tok = open_clip.get_tokenizer("ViT-H-14")
    with torch.no_grad():
        # ENIGMA keys text features by bare category name; prompt matches CLIP's usual template
        t = model.encode_text(tok([f"a photo of a {c.replace('_', ' ')}" for c in cats]).to(device))
    feats = {"image": {r[0]: img[i] for i, r in enumerate(rows)},
             "text": {c: t[i].float().cpu() for i, c in enumerate(cats)}}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"ViT-H-14_features_{'train' if split == 'training' else 'test'}.pt")
    torch.save(feats, p)
    print(f"{split}: {len(feats['image'])} image + {len(feats['text'])} text features -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["training", "test", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="subjects to average EEG for (default: all)")
    a = ap.parse_args()
    global SUBJECTS_TODO
    SUBJECTS_TODO = a.subjects or SUBJECTS
    todo = [("training", "stim_train"), ("test", "stim_test")] if a.split == "both" \
        else [(a.split, "stim_train" if a.split == "training" else "stim_test")]
    for split, _ in todo:
        build_eeg(split)
    per_split = build_metadata(todo)
    for split, _ in todo:
        build_features(split, per_split[split], a.device)


if __name__ == "__main__":
    main()
