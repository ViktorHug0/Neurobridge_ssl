"""AUG_LOG 2026-10-30 route 1 (a) warm start: the InternViT-L28 REVE-d6 member with its 3200-d head composed with a bridge
W (3200 x 1536) into the grader layer, head' = W^T head (exact: head'(x) = head(x) @ W), as --init_checkpoint of
reve_d6_wsIV28_things.      warmstart_ckpt.py <fold> <W.npy>  ->  cv4/warmstart/reve_d6_tIV28_f<k>.pth"""
import glob, os, sys
import numpy as np, torch

R = 'results/alljoined/track1/cv4'
k, wpath = int(sys.argv[1]), sys.argv[2]
ck = torch.load(sorted(glob.glob(f'{R}/reve_d6_tIV28_things/fold{k}/*/2026*/checkpoint_sealed_best.pth'))[0], map_location='cpu', weights_only=False)
W = torch.as_tensor(np.load(wpath)).float()
sd = ck['model_state_dict']
x = torch.randn(4, sd['head.weight'].shape[1])
ref = (x @ sd['head.weight'].T + sd['head.bias']) @ W
sd['head.weight'], sd['head.bias'] = W.T @ sd['head.weight'], sd['head.bias'] @ W
assert torch.allclose(x @ sd['head.weight'].T + sd['head.bias'], ref, atol=1e-3 * ref.abs().max()), 'composition'
os.makedirs(f'{R}/warmstart', exist_ok=True)
out = f'{R}/warmstart/reve_d6_tIV28_f{k}.pth'
torch.save({'model_state_dict': sd, 'eeg_projector_state_dict': {}, 'img_projector_state_dict': {}, 'text_projector_state_dict': {}}, out + '.tmp')
os.replace(out + '.tmp', out)
print('saved', out, tuple(sd['head.weight'].shape), 'from', wpath)
