import os
import random
from PIL import Image
import argparse
import sys

import cv2
import numpy as np
from torchvision import transforms
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModel
import torch
from tqdm import tqdm


# --- image augmentations for --aug_type (formerly module/image_augmentation.py) ---
class ColorJitter:
    def __init__(self, s=0.5, p=0.8):
        self.transform = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.p = p

    def __call__(self, img:Image)-> Image:
        transform = transforms.RandomApply([self.transform], p=self.p)
        img = transform(img)
        return img


class RandomCrop:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.transform = transforms.RandomCrop(size=self.size)

    def __call__(self, img:Image)-> Image:
        img = img.resize((336, 336), Image.BILINEAR)
        img = self.transform(img)
        return img


# Reduce resolution without changing the size
class LowResolution:
    def __init__(self, scale=0.5):
        """
        scale: The factor by which to reduce the resolution. For example, 0.5 means reducing the resolution to half of the original.
        """
        self.scale = scale

    def __call__(self, img:Image) -> Image:
        w, h = img.size
        new_size = (int(w * self.scale), int(h * self.scale))
        img = img.resize(new_size, Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


class Mosaic:
    def __init__(self, mosaic_level=16):
        """
        mosaic_level: Controls the level of mosaic (block size). The larger the value, the coarser the mosaic.
        Typical values: 8, 16, 32, 64...
        """
        self.mosaic_level = mosaic_level

    def __call__(self, img:Image) -> Image:
        img = np.array(img)
        h, w, _ = img.shape

        # Resize the image to a smaller size and then back to the original size
        small = cv2.resize(img, (w // self.mosaic_level, h // self.mosaic_level), interpolation=cv2.INTER_LINEAR)
        mosaic_img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return Image.fromarray(mosaic_img)


class GrayScale:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = transforms.RandomGrayscale(p=self.p)

    def __call__(self, img)-> Image:
        img = self.transform(img)
        return img


class GaussianBlur:
    def __init__(self, blur_kernel_size, fluctuation_range=0):
        self.blur_kernel_size = blur_kernel_size
        self.fluctuation_range = fluctuation_range

    def __call__(self, img):
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        if self.fluctuation_range > 0:
            # Randomly fluctuate the blur kernel size
            blur_kernel_size = random.randint(max(1, self.blur_kernel_size - self.fluctuation_range), 
                                              self.blur_kernel_size + self.fluctuation_range)
            # Ensure the kernel size is within a valid range
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
        else:
            blur_kernel_size = self.blur_kernel_size
        
        img_blur = cv2.GaussianBlur(img_np, (blur_kernel_size, blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)
    
    
class GaussianNoise:
    def __init__(self, mean=0.0, std=10.0, fluctuation_range=0):
        """
        Adds a Gaussian noise augmentation class.

        Args:
            mean (float): The mean of the noise.
            std (float): The standard deviation (magnitude) of the noise.
        """
        self.mean = mean
        self.std = std
        self.fluctuation_range = fluctuation_range

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)
        
        if self.fluctuation_range > 0:
            # Randomly fluctuate the standard deviation
            # Ensure the standard deviation is within a valid range
            std = random.randint(max(1, self.std - self.fluctuation_range), self.std + self.fluctuation_range)
        else:
            std = self.std

        # Generate Gaussian noise with the same shape as the image
        noise = np.random.normal(self.mean, std, img_np.shape).astype(np.float32)

        # Add the noise to the image and clip the values to ensure they remain valid pixel values
        img_noisy = img_np + noise
        img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(img_noisy)


# THINGS image layout: <this dir>/training_images/, test_images/
_DEFAULT_IMAGE_SET_DIR = "/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set"

'''
Example usage for InternVIT
source .venv/bin/activate
python scripts/data/extract_feature.py \
  --model_type internvit \
  --backbone OpenGVLab/InternViT-6B-448px-V1-5 \
  --quantization 8bit \
  --feature_source intermediate \
  --intermediate_layer 25 \
  --intermediate_pool mean \
  --output_dir ./data/things_eeg/image_feature/InternViT-6B_layer25_mean_8bit 
'''


def extract_clip(image:str, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    feature = image_features.detach().cpu().numpy()
    # shape: (1, D)
    return feature


def extract_open_clip(image, processor, model, augmentation, device):
    if augmentation is not None:
        image = augmentation(image)
    
    # Use fp16 on CUDA to match the model's precision
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    image = processor(image).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        image_features = model.encode_image(image)
    feature = image_features.detach().cpu().numpy()  # shape: (1, D)
    return feature


def extract_open_clip_intermediate(image, processor, model, augmentation, device, layer_idx: int, pool_type: str):
    if augmentation is not None:
        image = augmentation(image)
    
    # Use fp16 on CUDA to match the model's precision
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    image = processor(image).unsqueeze(0).to(device=device, dtype=dtype)

    visual = model.visual
    convolutional_stage = False
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
    elif hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        blocks = visual.trunk.blocks
    elif all(hasattr(visual, f"layer{index}") for index in range(1, 5)):
        # OpenCLIP's ModifiedResNet exposes the four residual stages directly.
        # The shallow-alignment paper's RN50/RN101 layer 3 is therefore index 2.
        blocks = [getattr(visual, f"layer{index}") for index in range(1, 5)]
        convolutional_stage = True
    else:
        raise ValueError(
            "Selected open_clip backbone does not expose transformer blocks or "
            "four residual stages."
        )

    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError(f"Invalid intermediate layer index {layer_idx}. Valid range: [0, {len(blocks)-1}]")

    layer_output = {}

    def hook_fn(module, _input, output):
        layer_output["hidden"] = output[0] if isinstance(output, tuple) else output

    hook = blocks[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            # Use visual directly instead of encode_image to avoid device mismatch with text tower
            _ = model.visual(image)
    finally:
        hook.remove()

    if "hidden" not in layer_output:
        raise RuntimeError("Forward hook did not capture intermediate output.")

    hidden = layer_output["hidden"]
    if convolutional_stage:
        if pool_type != "mean":
            raise ValueError("Intermediate ResNet stages require --intermediate_pool mean")
        if hidden.ndim != 4:
            raise RuntimeError(f"Expected BCHW ResNet stage output, got {tuple(hidden.shape)}")
        pooled = hidden.mean(dim=(-2, -1))
    elif pool_type == "cls":
        pooled = hidden[:, 0, :]
    elif pool_type == "mean":
        pooled = hidden[:, 1:, :].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

    return pooled.detach().cpu().numpy()


def extract_internvit_intermediate(
    image,
    processor,
    model,
    augmentation,
    device,
    layer_idx: int,
    pool_type: str,
):
    if augmentation is not None:
        image = augmentation(image)

    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values.to(model.dtype) if hasattr(model, "dtype") else inputs.pixel_values.to(torch.float16)

    if hasattr(model, "vision_model") and hasattr(model.vision_model.encoder, "layers"):
        resblocks = model.vision_model.encoder.layers
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        resblocks = model.encoder.layers
    else:
        raise ValueError("Selected InternViT backbone does not expose encoder.layers or vision_model.encoder.layers.")

    if layer_idx < 0 or layer_idx >= len(resblocks):
        raise ValueError(f"Invalid intermediate layer index {layer_idx}. Valid range: [0, {len(resblocks)-1}]")

    layer_output = {}

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            layer_output["hidden"] = output[0]
        else:
            layer_output["hidden"] = output

    hook = resblocks[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(pixel_values)
    finally:
        hook.remove()

    if "hidden" not in layer_output:
        raise RuntimeError("Forward hook did not capture intermediate output.")

    hidden = layer_output["hidden"]

    if pool_type == "cls":
        pooled = hidden[:, 0, :]
    elif pool_type == "mean":
        pooled = hidden[:, 1:, :].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

    feature = pooled.detach().cpu().numpy()
    return feature


def extract_dinov2(image:str, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_dim]
    feature = cls_embedding.detach().cpu().numpy()
    return feature


def extract_dinov2_intermediate(
    image,
    processor,
    model,
    augmentation,
    device,
    layer_idx: int,
    pool_type: str,
):
    if augmentation is not None:
        image = augmentation(image)

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("DINOv2 model did not return hidden states.")

    num_layers = len(hidden_states) - 1
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid intermediate layer index {layer_idx}. Valid range: [0, {num_layers-1}]")

    # hidden_states[0] is the patch embedding output; block outputs start at index 1.
    hidden = hidden_states[layer_idx + 1]

    if pool_type == "cls":
        pooled = hidden[:, 0, :]
    elif pool_type == "mean":
        pooled = hidden[:, 1:, :].mean(dim=1) if hidden.shape[1] > 1 else hidden.mean(dim=1)
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

    return pooled.detach().cpu().numpy()


def extract_image_features(
    image_dir,
    num_images_per_object,
    processor,
    model,
    model_type,
    augmentation,
    device,
    feature_source="final",
    intermediate_layer=11,
    intermediate_pool="cls",
):
    image_classes = sorted(os.listdir(image_dir))
    image_list = []
    for image_class in image_classes:
        image_class_path = os.path.join(image_dir, image_class)
        image_files = sorted(os.listdir(image_class_path))
        for image_file in image_files:
            image_file_path = os.path.join(image_class_path, image_file)
            image_list.append(image_file_path)

    all_features = []
    for image in tqdm(image_list):
        image = Image.open(image).convert('RGB')
        image = image.resize((224, 224))
        if model_type == 'clip':
            feature = extract_clip(image, processor, model, device)
        elif model_type == 'open_clip':
            if feature_source == "intermediate":
                feature = extract_open_clip_intermediate(
                    image,
                    processor,
                    model,
                    augmentation,
                    device,
                    layer_idx=intermediate_layer,
                    pool_type=intermediate_pool,
                )
            else:
                feature = extract_open_clip(image, processor, model, augmentation, device)
        elif model_type == 'dinov2':
            if feature_source == "intermediate":
                feature = extract_dinov2_intermediate(
                    image,
                    processor,
                    model,
                    augmentation,
                    device,
                    layer_idx=intermediate_layer,
                    pool_type=intermediate_pool,
                )
            else:
                feature = extract_dinov2(image, processor, model, device)
        elif model_type == 'internvit':
            if feature_source == "intermediate":
                feature = extract_internvit_intermediate(
                    image,
                    processor,
                    model,
                    augmentation,
                    device,
                    layer_idx=intermediate_layer,
                    pool_type=intermediate_pool,
                )
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)
                pixel_values = inputs.pixel_values.to(model.dtype) if hasattr(model, "dtype") else inputs.pixel_values.to(torch.float16)
                with torch.no_grad():
                    outputs = model(pixel_values)
                if hasattr(outputs, "last_hidden_state"):
                    feature = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
                elif isinstance(outputs, tuple):
                    feature = outputs[0][:, 0].detach().cpu().numpy()
                else:
                    feature = outputs[:, 0].detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        all_features.append(feature)

    all_features = np.concatenate(all_features, axis=0)
    print(all_features.shape)

    assert all_features.shape[0] % num_images_per_object == 0

    reshaped_features = all_features.reshape(-1, num_images_per_object, all_features.shape[-1])
    return reshaped_features


def extract_text_features(image_dir, processor, model, device):
    pass


def preprocess(image:Image.Image, augmentation=None):
    transform_resize = transforms.Resize((224, 224))
    image = transform_resize(image)
    if augmentation is not None:
        image = augmentation(image)
    # Convert the image to a tensor and normalize
    transform_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    image = transform_norm(image)
    return image


# ViT-B-16: laion2b_s34b_b88k
# ViT-B-32: laion2b_s34b_b79k
# ViT-L-14: laion2b_s32b_b82k
# ViT-H-14: laion2b_s32b_b79k
# ViT-g-14: laion2b_s34b_b88k
# ViT-bigG-14: laion2b_s39b_b160k
# RN50: openai
# RN101: openai
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help="training device")
    parser.add_argument("--model_type", type=str, choices=["clip", "open_clip", "dinov2", "internvit"], default="open_clip")
    parser.add_argument("--backbone", type=str, default="RN50")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--image_set_dir", type=str, default=_DEFAULT_IMAGE_SET_DIR)
    parser.add_argument("--output_dir", type=str, default="./data/things_eeg/image_feature/RN50")
    parser.add_argument("--aug_type", type=str, default="None", choices=["GaussianBlur", "GaussianNoise", "Mosaic", "RandomCrop", "LowResolution", "ColorJitter", "GrayScale", "None"])
    parser.add_argument("--num_images_per_object", type=int, default=10)
    parser.add_argument("--feature_source", type=str, choices=["final", "intermediate"], default="final")
    parser.add_argument(
        "--intermediate_layer",
        type=int,
        default=11,
        help=(
            "0-indexed ViT transformer block output (after patch embed), or "
            "0-indexed residual stage for OpenCLIP RN50/RN101. E.g. ViT-B/16 "
            "has 12 blocks (0–11), while paper ResNet layer 3 is index 2."
        ),
    )
    parser.add_argument("--intermediate_pool", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="none", help="Quantization for InternViT models")
    args = parser.parse_args()

    print('Input arguments:')
    for key, val in vars(args).items():
        print(f'{key:22} {val}')

    # Check if output directory exists
    if os.path.exists(args.output_dir):
       print(f"Output directory {args.output_dir} already exists, skipping feature extraction...")
       sys.exit(0)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.feature_source == "intermediate" and args.model_type not in {"dinov2", "internvit", "open_clip"}:
        raise ValueError("--feature_source intermediate is only supported for --model_type dinov2, internvit, or open_clip")

    if args.model_type == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.model_type == "open_clip":
        # Load on CPU first to allow pruning text tower before GPU move
        model, _, processor = open_clip.create_model_and_transforms(
            args.backbone, 
            pretrained=args.pretrained, 
            precision='fp16', 
            device='cpu'
        )
        if device.type == 'cuda':
            # Move ONLY the vision tower to GPU (saves ~2-3GB for bigG)
            model.visual = model.visual.to(device)
            # Clear CPU memory and CUDA fragmentation
            import gc
            torch.cuda.empty_cache()
            gc.collect()
        else:
            model = model.to(device)
    elif args.model_type == "dinov2":
        # small, base, large, giant
        dinov2_backbone = args.backbone if args.backbone != "RN50" else "facebook/dinov2-giant"
        processor = AutoImageProcessor.from_pretrained(dinov2_backbone)
        model = AutoModel.from_pretrained(dinov2_backbone).to(device)
    elif args.model_type == "internvit":
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        if args.quantization == "8bit":
            load_kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        elif args.quantization == "4bit":
            load_kwargs.update({"load_in_4bit": True, "device_map": "auto"})

        model = AutoModel.from_pretrained(args.backbone, **load_kwargs)
        if args.quantization == "none":
            model = model.to(device)

        processor = AutoImageProcessor.from_pretrained(args.backbone, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.aug_type == "GaussianBlur":
        augmentation = GaussianBlur(blur_kernel_size=15, fluctuation_range=0)
    elif args.aug_type == "GaussianNoise":
        augmentation = GaussianNoise(mean=0.0, std=25.0, fluctuation_range=0)
    elif args.aug_type == "Mosaic":
        augmentation = Mosaic(mosaic_level=5)
    elif args.aug_type == "RandomCrop":
        augmentation = RandomCrop(size=(224, 224))
    elif args.aug_type == "LowResolution":
        augmentation = LowResolution(scale=0.1)
    elif args.aug_type == "ColorJitter":
        augmentation = ColorJitter(s=0.5, p=1.0)
    elif args.aug_type == "GrayScale":
        augmentation = GrayScale(p=1.0)
    else:
        augmentation = None

    os.makedirs(args.output_dir, exist_ok=True)

    _feat_kw = dict(
        feature_source=args.feature_source,
        intermediate_layer=args.intermediate_layer,
        intermediate_pool=args.intermediate_pool,
    )

    if augmentation is None:
        train_image_features = extract_image_features(os.path.join(args.image_set_dir, "training_images"), args.num_images_per_object, processor, model, args.model_type, augmentation, device, **_feat_kw)
        print(f"Train image feature shape: {train_image_features.shape}")
        np.save(os.path.join(args.output_dir, "image_train.npy"), train_image_features)
    else:
        train_image_features_list = []
        for i in range(args.repeat_times):
            train_image_features = extract_image_features(os.path.join(args.image_set_dir, "training_images"), args.num_images_per_object, processor, model, args.model_type, augmentation, device, **_feat_kw)
            train_image_features_list.append(train_image_features)
        train_image_features = np.stack(train_image_features_list, axis=0)
        print(f"Train image feature shape: {train_image_features.shape}")
        np.save(os.path.join(args.output_dir, "train.npy"), train_image_features)

    if augmentation is None:
        test_image_features = extract_image_features(os.path.join(args.image_set_dir, "test_images"), 1, processor, model, args.model_type, augmentation, device, **_feat_kw)
        print(f"Test image feature shape: {test_image_features.shape}")
        np.save(os.path.join(args.output_dir, "image_test.npy"), test_image_features)
    else:
        test_image_features_list = []
        for i in range(args.repeat_times):
            test_image_features = extract_image_features(os.path.join(args.image_set_dir, "test_images"), 1, processor, model, args.model_type, augmentation, device, **_feat_kw)
            test_image_features_list.append(test_image_features)
        test_image_features = np.stack(test_image_features_list, axis=0)
        print(f"Test image feature shape: {test_image_features.shape}")
        np.save(os.path.join(args.output_dir, "test.npy"), test_image_features)
