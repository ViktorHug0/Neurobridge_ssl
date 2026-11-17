"""Y01 grader-style targets (DINOv2-giant hidden state 26, token mean, HF processor: as nq01 / logs/things_distractors.py) for images that
nobody in any EEG set saw: 40,000 COCO train2017 photographs (scenes) and the 50,000 ImageNet validation images (objects). The sealed
gallery holds ~40k images and at most 9.4k of them can be THINGS images (E00), so most of it is outside THINGS. Uses:
(1) an out-of-THINGS gallery for the clean proxy, (2) an "open bank" for the loss (train.py --open_bank).
-> data/things_eeg/image_feature/dinov2-giant_track1/open_{coco,imagenet}.npy (float16) + open_*_names.txt
    ny01_open_images.py"""
import glob, os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import to_tensor

OUT = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1'
SETS = {'coco': sorted(glob.glob('/nasbrain/datasets/coco_full/images/train2017/*.jpg'))[::2][:40000],  # every other file: spread over the set
        'imagenet': sorted(glob.glob('/nasbrain/datasets/imagenet/images/val/*/*.JPEG'))}
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', do_rescale=False)
model = AutoModel.from_pretrained('facebook/dinov2-giant', torch_dtype=torch.float32).cuda().eval()


class DS(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return proc(images=[to_tensor(Image.open(self.paths[i]).convert('RGB'))], return_tensors='pt')['pixel_values'][0]


for name, paths in SETS.items():
    if os.path.exists(f'{OUT}/open_{name}.npy'):
        continue
    out = []
    for x in torch.utils.data.DataLoader(DS(paths), batch_size=32, num_workers=7):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):  # distractors / bank entries: half precision is enough (cos > .9999)
            out.append(model(pixel_values=x.cuda(), output_hidden_states=True).hidden_states[26].float().mean(1).cpu())
        if len(out) % 100 == 0:
            print(name, len(out) * 32, flush=True)
    np.save(f'{OUT}/open_{name}.npy', torch.cat(out).numpy().astype(np.float16)); open(f'{OUT}/open_{name}_names.txt', 'w').write('\n'.join(paths))
    print('saved', name, len(paths), flush=True)
