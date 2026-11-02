"""ATM-style OpenCLIP ViT-H-14 (laion2b_s32b_b79k) features for the SATTC reproduction (E3.1).

SATTC's loader (Retrieval/eegdatasets_leaveone.py) reads {features_path}/ViT-H-14_features_{train,test}.pt with
'img_features' (16540 or 200 rows, sorted class folders then sorted files) and 'text_features' (one row per class,
"This picture is <name>"). Both are L2-normalised, and images go through open_clip's TRAIN transform, as in ATM's
eegdatasets_leaveone.py (Li et al. 2024), whose features SATTC inherits.
usage: python sattc_vith_features.py <image_set_dir> <out_dir>
"""
import os, sys
import open_clip, torch
import torch.nn.functional as F
from PIL import Image

image_set, out_dir = sys.argv[1], sys.argv[2]
torch.manual_seed(0)  # the train transform has a random crop
model, preprocess_train, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k',
                                                                   precision='fp32', device='cuda')
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model.eval()
for split, sub in (('test', 'test_images'), ('train', 'training_images')):
    path = os.path.join(out_dir, f'ViT-H-14_features_{split}.pt')
    if os.path.exists(path):
        continue
    root = os.path.join(image_set, sub)
    folders = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    texts = [f"This picture is {d[d.index('_') + 1:]}" for d in folders]
    images = [os.path.join(root, d, f) for d in folders
              for f in sorted(x for x in os.listdir(os.path.join(root, d)) if x.lower().endswith(('.png', '.jpg', '.jpeg')))]
    with torch.no_grad():
        text = F.normalize(model.encode_text(tokenizer(texts).cuda()), dim=-1).cpu()
        img = torch.cat([F.normalize(model.encode_image(torch.stack(
            [preprocess_train(Image.open(p).convert('RGB')) for p in images[i:i + 20]]).cuda()), dim=-1).cpu()
            for i in range(0, len(images), 20)])
    assert len(img) == (16540 if split == 'train' else 200), len(img)
    torch.save({'text_features': text, 'img_features': img}, path)
    print(split, tuple(text.shape), tuple(img.shape), flush=True)
