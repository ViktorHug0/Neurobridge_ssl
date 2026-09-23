"""Import the challenge's cached DINOv2 target embeddings into this repo's layout.

The NeuralBench start-kit already computed `facebook/dinov2-giant` embeddings for every
THINGS-EEG-2 image with the official Track-1 settings (layers=0.6667, token_aggregation=mean,
imsize=518).  Its exca cache is a JSONL index whose `#key` is the absolute image path and whose
value is an offset into a flat float32 memmap, so we can re-key those exact vectors into
`image_train.npy` / `image_test.npy` instead of re-running the backbone.

Doing it this way removes an entire class of error: the target space is bit-identical to the one
the Codabench grader ranks against, so any gap we measure is attributable to the EEG side.

ponytail: no re-extraction path here; `extract_feature.py --model_type dinov2` remains the
fallback if the cache is ever unavailable, and `verify_targets.py` cross-checks the two.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

CACHE_DIR = (
    "/nasbrain/p20fores/Neurips_challenge/cache/"
    "neuralset.extractors.image.HuggingFaceImage._get_data,v6/"
    "imsize=518,layers=0.67,name=HuggingFaceImage,model_name=facebook-dinov2-giant-707b728a"
)
IMAGE_SET_DIR = "/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set"
DIM = 1536


def load_cache_index(cache_dir: str) -> dict[tuple[str, str, str], tuple[str, int]]:
    """Map (split_dir, concept_dir, filename) -> (memmap path, byte offset)."""
    index: dict[tuple[str, str, str], tuple[str, int]] = {}
    for entry in sorted(os.listdir(cache_dir)):
        if not entry.endswith("-info.jsonl"):
            continue
        with open(os.path.join(cache_dir, entry)) as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("shape") != [DIM] or record.get("dtype") != "float32":
                    raise ValueError(f"unexpected cache record: {record}")
                parts = record["#key"].split("/")
                key = (parts[-3], parts[-2], parts[-1])
                index[key] = (os.path.join(cache_dir, record["filename"]), record["offset"])
    return index


def list_images(image_dir: str) -> list[tuple[str, str]]:
    """Enumerate (concept_dir, filename) in extract_feature.py's order."""
    out = []
    for concept in sorted(os.listdir(image_dir)):
        for filename in sorted(os.listdir(os.path.join(image_dir, concept))):
            out.append((concept, filename))
    return out


def build_split(split: str, images_per_object: int, index, image_set_dir: str) -> np.ndarray:
    image_dir = os.path.join(image_set_dir, split)
    images = list_images(image_dir)
    memmaps: dict[str, np.memmap] = {}
    features = np.empty((len(images), DIM), dtype=np.float32)
    missing = []
    for i, (concept, filename) in enumerate(images):
        hit = index.get((split, concept, filename))
        if hit is None:
            missing.append(f"{split}/{concept}/{filename}")
            continue
        path, offset = hit
        if path not in memmaps:
            memmaps[path] = np.memmap(path, dtype=np.float32, mode="r")
        features[i] = memmaps[path][offset // 4 : offset // 4 + DIM]
    if missing:
        raise KeyError(f"{len(missing)} images absent from the cache, e.g. {missing[:3]}")
    if len(images) % images_per_object:
        raise ValueError(f"{len(images)} images is not a multiple of {images_per_object}")
    return features.reshape(-1, images_per_object, DIM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--image_set_dir", default=IMAGE_SET_DIR)
    parser.add_argument(
        "--output_dir",
        default="./data/things_eeg/image_feature/dinov2-giant_track1",
    )
    parser.add_argument("--num_images_per_object", type=int, default=10)
    args = parser.parse_args()

    index = load_cache_index(args.cache_dir)
    print(f"cache entries: {len(index)}")

    os.makedirs(args.output_dir, exist_ok=True)
    train = build_split("training_images", args.num_images_per_object, index, args.image_set_dir)
    print(f"train: {train.shape}")
    np.save(os.path.join(args.output_dir, "image_train.npy"), train)

    test = build_split("test_images", 1, index, args.image_set_dir)
    print(f"test: {test.shape}")
    np.save(os.path.join(args.output_dir, "image_test.npy"), test)

    # The grader's gallery is the set of unique test targets; if these collide the task is
    # ill-posed and every downstream number is wrong.
    unique = np.unique(test.reshape(-1, DIM), axis=0)
    print(f"unique test targets: {len(unique)} (expected {test.shape[0]})")
    assert len(unique) == test.shape[0], "duplicate test target embeddings"


if __name__ == "__main__":
    main()
