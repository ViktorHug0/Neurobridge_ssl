"""Extract several InternViT intermediate layers in a single forward pass.

`extract_feature.py` hooks one block per run, so N layers cost N full passes over
the image set with a 6B backbone loaded each time.  This hooks every requested
block at once and writes one output directory per layer, matching the layout and
pooling of `--model_type internvit --feature_source intermediate
--intermediate_pool mean`:

    <output-template>/image_train.npy   (1654, 10, D) float16
    <output-template>/image_test.npy    (200, 1, D)   float16

Example:
    python extract_internvit_layers.py --layers 35 39 44 \
      --output-template ./data/things_eeg/image_feature/InternViT-6B_layer{layer}_mean_8bit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from extract_feature import _DEFAULT_IMAGE_SET_DIR


def list_images(image_dir: str) -> list[str]:
    """Same ordering as extract_feature.extract_image_features."""
    paths = []
    for image_class in sorted(os.listdir(image_dir)):
        class_path = os.path.join(image_dir, image_class)
        for image_file in sorted(os.listdir(class_path)):
            paths.append(os.path.join(class_path, image_file))
    return paths


def resolve_blocks(model):
    if hasattr(model, "vision_model") and hasattr(model.vision_model.encoder, "layers"):
        return model.vision_model.encoder.layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return model.encoder.layers
    raise ValueError("backbone exposes neither encoder.layers nor vision_model.encoder.layers")


def extract(
    image_dir: str,
    images_per_object: int,
    processor,
    model,
    device: str,
    layers: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    blocks = resolve_blocks(model)
    for layer in layers:
        if layer < 0 or layer >= len(blocks):
            raise ValueError(f"layer {layer} outside [0, {len(blocks) - 1}]")

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module, _input, output):
            captured[layer] = output[0] if isinstance(output, tuple) else output

        return hook

    handles = [blocks[layer].register_forward_hook(make_hook(layer)) for layer in layers]
    paths = list_images(image_dir)
    collected: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    try:
        for start in tqdm(range(0, len(paths), batch_size), desc=os.path.basename(image_dir)):
            batch = [
                Image.open(p).convert("RGB").resize((224, 224))
                for p in paths[start : start + batch_size]
            ]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            pixel_values = inputs.pixel_values.to(
                model.dtype if hasattr(model, "dtype") else torch.float16
            )
            captured.clear()
            with torch.no_grad():
                model(pixel_values)
            if len(captured) != len(layers):
                raise RuntimeError(f"hooks captured {sorted(captured)}, expected {layers}")
            for layer in layers:
                hidden = captured[layer]
                # mean over patch tokens, dropping CLS -- matches intermediate_pool=mean
                pooled = hidden[:, 1:, :].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
                collected[layer].append(pooled.detach().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    out = {}
    for layer in layers:
        features = np.concatenate(collected[layer], axis=0)
        assert features.shape[0] % images_per_object == 0, features.shape
        out[layer] = features.reshape(-1, images_per_object, features.shape[-1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output-template", required=True, help="path containing {layer}")
    parser.add_argument("--backbone", default="OpenGVLab/InternViT-6B-448px-V1-5")
    parser.add_argument("--quantization", default="8bit", choices=["none", "8bit", "4bit"])
    parser.add_argument("--image-set-dir", default=_DEFAULT_IMAGE_SET_DIR)
    parser.add_argument("--num-images-per-object", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if "{layer}" not in args.output_template:
        raise SystemExit("--output-template must contain {layer}")
    targets = {layer: Path(args.output_template.format(layer=layer)) for layer in args.layers}
    for layer, path in targets.items():
        if path.exists() and any(path.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty {path} (layer {layer})")

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if args.quantization == "8bit":
        load_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
    elif args.quantization == "4bit":
        load_kwargs.update({"load_in_4bit": True, "device_map": "auto"})
    model = AutoModel.from_pretrained(args.backbone, **load_kwargs)
    if args.quantization == "none":
        model = model.to(args.device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(args.backbone, trust_remote_code=True)
    print(f"hooking layers {args.layers} of {len(resolve_blocks(model))} blocks", flush=True)

    for split, subdir, per_object, name in (
        ("train", "training_images", args.num_images_per_object, "image_train.npy"),
        ("test", "test_images", 1, "image_test.npy"),
    ):
        features = extract(
            os.path.join(args.image_set_dir, subdir),
            per_object,
            processor,
            model,
            args.device,
            args.layers,
            args.batch_size,
        )
        for layer, array in features.items():
            targets[layer].mkdir(parents=True, exist_ok=True)
            np.save(targets[layer] / name, array)
            print(f"layer {layer} {split}: {array.shape} {array.dtype} -> {targets[layer] / name}", flush=True)


if __name__ == "__main__":
    main()
