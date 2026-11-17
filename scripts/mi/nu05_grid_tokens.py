"""T1 (round 11) auxiliary targets that keep the image layout: DINOv2-giant hidden state 26 (the grader's layer) of the 16,540
training images, patch tokens averaged on a G x G grid instead of over the whole image
-> data/things_eeg/image_feature/dinov2-giant_track1_grid<G>_z/image_train.npy (1654, 10, G*G*1536), z-scored per dimension, float16.
Check: the mean over all tokens reproduces the grader targets (dinov2-giant_track1), cosine printed.
    nu05_grid_tokens.py"""
import os, numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import to_tensor

R = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature'; D = '/nasbrain/ProCOM-EEG/NeuroBridge/NeuroBridge-main/data/things_eeg/image_set/training_images'
files = [f'{D}/{c}/{f}' for c in sorted(os.listdir(D)) for f in sorted(os.listdir(f'{D}/{c}'))]; assert len(files) == 16540
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', do_rescale=False); model = AutoModel.from_pretrained('facebook/dinov2-giant', torch_dtype=torch.float16).cuda().eval()
ref = np.load(f'{R}/dinov2-giant_track1/image_train.npy', mmap_mode='r').reshape(16540, 1536); out = {2: [], 3: []}; cos = []
for b in range(0, 16540, 32):
    x = proc(images=[to_tensor(Image.open(f).convert('RGB')) for f in files[b:b + 32]], return_tensors='pt')['pixel_values'].cuda().half()
    with torch.no_grad():
        h = model(pixel_values=x, output_hidden_states=True).hidden_states[26].float()
    cos += torch.cosine_similarity(h.mean(1).cpu(), torch.tensor(np.asarray(ref[b:b + 32]))).tolist()
    t = h[:, 1:]; s = int(t.shape[1] ** .5); t = t.transpose(1, 2).reshape(len(t), 1536, s, s)
    for g in out:
        out[g].append(torch.nn.functional.adaptive_avg_pool2d(t, g).flatten(1).cpu())
    if b % 3200 == 0:
        print(b, 'min cosine to the grader target so far %.4f' % min(cos), flush=True)
print('cosine to the grader targets: mean %.4f, min %.4f' % (np.mean(cos), min(cos)), flush=True); assert np.mean(cos) > 0.99
for g, v in out.items():
    v = torch.cat(v); v = (v - v.mean(0)) / v.std(0); os.makedirs(f'{R}/dinov2-giant_track1_grid{g}_z', exist_ok=True)
    np.save(f'{R}/dinov2-giant_track1_grid{g}_z/image_train.npy', v.reshape(1654, 10, -1).numpy().astype(np.float16)); print('saved grid', g, tuple(v.shape), flush=True)
