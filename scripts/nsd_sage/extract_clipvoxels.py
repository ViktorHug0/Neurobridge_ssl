"""Retrieval-submodule outputs for one held-out NSD subject, without the reconstruction stack.

MindEye2 ships no retrieval-only path: recon_inference.ipynb computes `clip_voxels` but in the
same loop it runs the diffusion prior, SDXL unCLIP and a GIT caption model, none of which the
retrieval metric touches. The paper is explicit that "for retrieval inference, only the retrieval
submodule's outputs are necessary", so this reproduces exactly the `clip_voxels` half of that
loop -- same ridge, same backbone, same averaging over the 3 repetitions of each test image.

Only `ridge` and `backbone` are built, and the checkpoint is filtered to those two prefixes.
That drops ~7 GB of prior/decoder weights we would otherwise load to ignore, and lets us set
blurry_recon=False: the blurry branch feeds the low-level reconstruction, never `clip_voxels`
(which comes off `clip_proj`), so disabling it cannot change the retrieval output.

Usage:
  python extract_clipvoxels.py --subj 1 --model_name final_subj01_pretrained_1sess_24bs
Writes clipvoxels_subj{N}_{model}.pt: dict(clipvoxels[1000,256,1664], image_idx[1000]).
"""
import argparse
import os
import sys
import types

import h5py
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

MINDEYE_SRC = "/nasbrain/p20fores/MindEyeV2/src"


def _shim_missing_imports():
    """Satisfy models.py's module-scope imports without the reconstruction dependencies.

    Three things are imported for code the retrieval path never reaches:
      * `clip` -- used by a separate CLIP wrapper class, not by BrainNetwork;
      * `diffusers.models.vae.Decoder` -- only instantiated when blurry_recon=True, and
        diffusers moved vae to models.autoencoders.vae after the version MindEye2 targets;
      * `generative_models.sgm.util.append_dims`, imported by utils.py at line 253 to serve
        unclip_recon. Importing it for real pulls in sgm's __init__ -> DiffusionEngine ->
        pytorch_lightning, so stub the three package levels instead. append_dims is
        reproduced faithfully in case some path does reach it.
    """
    sys.modules.setdefault("clip", types.ModuleType("clip"))
    try:
        import diffusers.models.vae  # noqa: F401
    except ModuleNotFoundError:
        from diffusers.models.autoencoders import vae as _vae
        sys.modules["diffusers.models.vae"] = _vae

    # models.py line 223+ imports ~17 names from dalle2_pytorch to define the diffusion-prior
    # classes (BrainDiffusionPrior, PriorNetwork, ...). BrainNetwork is fully defined by line
    # 119, but Python still executes the rest of the module. Installing dalle2_pytorch drags in
    # its own dependency cascade, so serve those names from a stub that mints a fresh class per
    # attribute -- enough for `from x import y` and for `class BrainDiffusionPrior(DiffusionPrior)`
    # to subclass. Any real use would fail loudly rather than silently misbehave.
    class _AutoStub(types.ModuleType):
        def __getattr__(self, name):
            obj = type(name, (), {})
            setattr(self, name, obj)
            return obj

    for mod in ("dalle2_pytorch", "dalle2_pytorch.dalle2_pytorch", "dalle2_pytorch.train_configs"):
        sys.modules.setdefault(mod, _AutoStub(mod))

    if "generative_models.sgm.util" not in sys.modules:
        def append_dims(x, target_dims):
            dims_to_append = target_dims - x.ndim
            if dims_to_append < 0:
                raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
            return x[(...,) + (None,) * dims_to_append]

        util = types.ModuleType("generative_models.sgm.util")
        util.append_dims = append_dims
        sgm = types.ModuleType("generative_models.sgm")
        sgm.util = util
        gm = types.ModuleType("generative_models")
        gm.sgm = sgm
        sys.modules.update({"generative_models": gm, "generative_models.sgm": sgm,
                            "generative_models.sgm.util": util})


def load_test_trials(data_path, subj, num_test):
    """(voxels[trials, V], image_idx[trials]) for the held-out subject's test session.

    behav[:,0,0] is the 73k COCO index, behav[:,0,5] the row of the betas matrix -- the same
    two columns recon_inference.ipynb reads.
    """
    with h5py.File(f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        betas = torch.from_numpy(f["betas"][:])

    url = f"{data_path}/wds/subj0{subj}/new_test/0.tar"
    ds = (wds.WebDataset(url, resampled=False, nodesplitter=lambda urls: urls)
            .decode("torch")
            .rename(behav="behav.npy", past_behav="past_behav.npy",
                    future_behav="future_behav.npy", olds_behav="olds_behav.npy")
            .to_tuple("behav", "past_behav", "future_behav", "olds_behav"))
    dl = torch.utils.data.DataLoader(ds, batch_size=num_test, shuffle=False, drop_last=True)

    image_idx, voxels = [], []
    for behav, *_ in dl:
        voxels.append(betas[behav[:, 0, 5].long()])
        image_idx.append(behav[:, 0, 0].numpy())
    return torch.cat(voxels), np.concatenate(image_idx).astype(int)


def build_model(num_voxels, hidden_dim, clip_emb_dim, clip_seq_dim, ckpt_path, device):
    sys.path.insert(0, MINDEYE_SRC)
    _shim_missing_imports()
    from models import BrainNetwork

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    # recon_inference.ipynb defaults to 2048, but the released checkpoints are not all trained
    # at that width (the subj01 1-session model is 4096). Read it off the ridge instead of
    # trusting a default: ridge.linears.0.weight is [hidden_dim, num_voxels].
    ckpt_hidden = state["ridge.linears.0.weight"].shape[0]
    if hidden_dim is not None and hidden_dim != ckpt_hidden:
        print(f"overriding --hidden_dim {hidden_dim} with {ckpt_hidden} from the checkpoint")
    hidden_dim = ckpt_hidden

    class RidgeRegression(torch.nn.Module):
        # verbatim from recon_inference.ipynb so the checkpoint keys line up
        def __init__(self, input_sizes, out_features):
            super().__init__()
            self.out_features = out_features
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(s, out_features) for s in input_sizes])

        def forward(self, x, subj_idx):
            return self.linears[subj_idx](x[:, 0]).unsqueeze(1)

    model = torch.nn.Module()
    model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)
    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1,
                                  clip_size=clip_emb_dim,
                                  out_dim=clip_emb_dim * clip_seq_dim,
                                  blurry_recon=False)

    wanted = {k: v for k, v in state.items() if k.startswith(("ridge.", "backbone."))}
    missing, unexpected = model.load_state_dict(wanted, strict=False)
    # Every retrieval weight must be filled: `missing` has to be empty. The only tolerated
    # leftovers are the blurry-reconstruction branch, which the checkpoint carries but
    # blurry_recon=False does not build.
    blurry = ("backbone.blin", "backbone.bnorm", "backbone.bupsampler", "backbone.b_maps")
    stray = [k for k in unexpected if not k.startswith(blurry)]
    assert not missing, f"unfilled retrieval weights: {list(missing)[:5]}"
    assert not stray, f"unused non-blurry checkpoint keys: {stray[:5]}"
    print(f"loaded {len(wanted) - len(unexpected)} retrieval tensors "
          f"(skipped {len(unexpected)} blurry + {len(state) - len(wanted)} prior/decoder)")
    # backbone_linear alone is 425984 x hidden_dim: 6.98 GB in fp32 at hidden_dim=4096, and
    # under autocast every call allocates a fresh fp16 copy on top, which overruns a 10 GB
    # card. Storing the weights in fp16 halves the resident model and removes the per-call
    # copy. The published run casts these same weights to fp16 inside autocast anyway, so the
    # matmul sees identically rounded inputs.
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--model_name", default="final_subj01_pretrained_1sess_24bs")
    ap.add_argument("--data_path", default="/nasbrain/p20fores/mindeye_data")
    ap.add_argument("--out_dir", default="/nasbrain/p20fores/mindeye_data/clipvoxels")
    ap.add_argument("--hidden_dim", type=int, default=None, help="inferred from the checkpoint when omitted")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    clip_emb_dim, clip_seq_dim = 1664, 256
    num_test = 2188 if args.subj in (4, 8) else 2371 if args.subj in (3, 6) else 3000

    voxels, image_idx = load_test_trials(args.data_path, args.subj, num_test)
    uniq = np.unique(image_idx)
    print(f"subj0{args.subj}: {len(voxels)} trials, {voxels.shape[-1]} voxels, {len(uniq)} images")

    model = build_model(voxels.shape[-1], args.hidden_dim, clip_emb_dim, clip_seq_dim,
                        f"{args.data_path}/train_logs/{args.model_name}/last.pth", args.device)

    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for img in tqdm(uniq, desc="images"):
            locs = np.where(image_idx == img)[0]
            # NSD shows each test image 3 times; pad the rare short case exactly as
            # recon_inference does so the /3 average stays comparable.
            if len(locs) == 1:
                locs = locs.repeat(3)
            elif len(locs) == 2:
                locs = locs.repeat(2)[:3]
            assert len(locs) == 3
            voxel = voxels[None, locs].to(args.device)

            clip_voxels = None
            for rep in range(3):
                _, cv, _ = model.backbone(model.ridge(voxel[:, [rep]], 0))
                clip_voxels = cv if clip_voxels is None else clip_voxels + cv
            # fp32 on the way out: the weights are fp16 but the retrieval maths downstream is
            # fp32, and a half-precision file silently mismatches the fp32 gallery.
            out.append((clip_voxels / 3).float().cpu())

    clipvoxels = torch.vstack(out)
    assert clipvoxels.shape == (len(uniq), clip_seq_dim, clip_emb_dim), clipvoxels.shape
    os.makedirs(args.out_dir, exist_ok=True)
    path = f"{args.out_dir}/clipvoxels_subj{args.subj}_{args.model_name}.pt"
    torch.save({"clipvoxels": clipvoxels, "image_idx": uniq}, path)
    print(f"wrote {path}  {tuple(clipvoxels.shape)}")


if __name__ == "__main__":
    main()
