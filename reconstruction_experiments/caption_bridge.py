"""EEG/image embedding -> caption, through a bridge into BLIP-2's frozen OPT decoder.

BLIP-2 only ever prepends `language_projection(query_output)` to the text embeddings
(modeling_blip_2.py:1732). So the bridge predicts `query_output` (32x768) and decoding needs
nothing but `language_projection` + `language_model`; the vision tower and Q-Former are used in
--stage cache only, once.

  --stage cache     BLIP-2 vision+Q-Former over THINGS images -> qformer_<split>.npy (N,32,768)
  --stage train     fit  emb(512|3200) -> 32x768   [MSE on tokens + CE through the frozen OPT]
  --stage generate  bridge on image OR EEG embeddings -> jsonl for caption_eval.py --hyp_captions

Image order is sorted(concepts) x sorted(files), i.e. extract_feature.py's and make_captions.py's,
so row i of image_train.npy, qformer_training.npy and training.qwen.jsonl are the same image.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

BLIP2 = "Salesforce/blip2-opt-2.7b"
ROOT = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"
FEAT = "data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
CAP = "data/things_eeg/captions"


def image_paths(split, per_concept):
    d = os.path.join(ROOT, f"{split}_images")
    return [os.path.join(d, c, f)
            for c in sorted(os.listdir(d))
            for f in sorted(os.listdir(os.path.join(d, c)))[:per_concept]]


def load_blip2(hf_cache, with_vision):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    proc = Blip2Processor.from_pretrained(BLIP2, cache_dir=hf_cache)
    m = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2, torch_dtype=torch.float16, cache_dir=hf_cache).to("cuda").eval()
    m.requires_grad_(False)
    if not with_vision:  # ~4GB of the 8GB is the ViT-g we no longer need
        del m.vision_model, m.qformer
        torch.cuda.empty_cache()
    return proc, m


def stage_cache(args):
    """query_output for every image -- the bridge's regression target."""
    proc, m = load_blip2(args.hf_cache, with_vision=True)
    paths = image_paths(args.split, args.per_concept)
    out = np.zeros((len(paths), m.config.num_query_tokens, m.config.qformer_config.hidden_size),
                   dtype=np.float16)
    for i in range(0, len(paths), args.batch_size):
        pil = [Image.open(p).convert("RGB") for p in paths[i: i + args.batch_size]]
        px = proc(images=pil, return_tensors="pt").pixel_values.to("cuda", torch.float16)
        with torch.no_grad():
            ie = m.vision_model(px, return_dict=True).last_hidden_state
            qo = m.qformer(query_embeds=m.query_tokens.expand(len(pil), -1, -1),
                           encoder_hidden_states=ie,
                           encoder_attention_mask=torch.ones(ie.shape[:-1], dtype=torch.long,
                                                             device="cuda"),
                           return_dict=True).last_hidden_state
        out[i: i + len(pil)] = qo.cpu().numpy()
        if i % (50 * args.batch_size) == 0:
            print(f"[{i}/{len(paths)}]", flush=True)
    path = os.path.join(CAP, f"qformer_{args.split}.npy")
    np.save(path, out)
    print(f"wrote {path} {out.shape}")


def embeddings(split, args):
    """(N, D) bridge inputs, in image order. `proj` = the subject's shared 512-d space."""
    if args.pred_npy:
        # E1: fit on the encoder's own predictions so the bridge sees the real error geometry.
        # Gaussian noise only matches the error's magnitude; ENIGMA's is structured, and its
        # rows here are in stimulus-walk order, the same order qformer_<split>.npy uses.
        x = torch.from_numpy(np.load(args.pred_npy)).float()
        return nn.functional.normalize(x, dim=-1)
    feat = args.feat_dir if args.input == "vith" else FEAT
    x = np.load(os.path.join(feat, f"image_{'train' if split == 'training' else 'test'}.npy"))
    x = torch.from_numpy(x.reshape(-1, x.shape[-1])).float()
    if args.input == "vith":
        # ENIGMA regresses raw ViT-H but overshoots the norm (33.8 vs 22.3) with almost no
        # variance, so scale is uninformative here: drop it on both sides.
        return nn.functional.normalize(x, dim=-1)
    if args.input == "raw":
        return x
    from module.projector import ProjectorLinear
    ck = torch.load(os.path.join(args.checkpoint_dir, "checkpoint_test_best.pth"),
                    map_location="cpu", weights_only=False)
    head = ProjectorLinear(x.shape[-1], args.feature_dim)
    head.load_state_dict(ck["img_projector_state_dict"])
    with torch.no_grad():
        x = head(x)
    return nn.functional.normalize(x, dim=-1)  # img_l2norm=True at training time


def load_captions(split):
    """Flat list in image order, matching embeddings()/qformer_<split>.npy."""
    with open(os.path.join(CAP, f"{split}.qwen.jsonl")) as f:
        by_concept = {d["concept"]: d["captions"] for d in map(json.loads, filter(str.strip, f))}
    return [c for k in sorted(by_concept) for c in by_concept[k]]


def stage_train(args):
    y = torch.from_numpy(np.load(os.path.join(CAP, f"qformer_{args.split}.npy"))).float()
    x = embeddings(args.split, args)
    assert len(x) == len(y), f"{len(x)} embeddings vs {len(y)} targets"
    n_tok, d_tok = y.shape[1], y.shape[2]
    # MSE is on raw Q-Former values, so it is unreadable without the constant-predictor baseline:
    # mse above this number means the bridge is doing worse than predicting the mean (R^2 < 0)
    print(f"target per-element variance {y.flatten(1).var(0, unbiased=False).mean():.4f}"
          f"  <- MSE of the best constant predictor", flush=True)

    bridge = nn.Linear(x.shape[1], n_tok * d_tok).cuda()
    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=1e-2)
    # the CE term backprops through a frozen fp16 OPT into an fp32 bridge -> gradients underflow
    scaler = torch.amp.GradScaler("cuda", enabled=args.ce_weight > 0)

    caps, proc, m = None, None, None
    if args.ce_weight > 0:
        caps = load_captions(args.split)
        assert len(caps) == len(x), f"{len(caps)} captions vs {len(x)} images"
        proc, m = load_blip2(args.hf_cache, with_vision=False)

    for epoch in range(args.epochs):
        perm = torch.randperm(len(x))
        tot = totce = 0.0
        for i in range(0, len(x), args.batch_size):
            idx = perm[i: i + args.batch_size]
            xb = x[idx].cuda()
            if args.noise > 0:  # fresh draw per batch, so the bridge sees the corruption, not a fixed offset
                xb = nn.functional.normalize(xb + args.noise * torch.randn_like(xb) / xb.shape[1] ** 0.5, dim=-1)
            pred = bridge(xb).view(len(idx), n_tok, d_tok)
            loss = nn.functional.mse_loss(pred, y[idx].cuda())
            ce = torch.zeros((), device="cuda")
            if args.ce_weight > 0:
                ce = caption_ce(m, proc, pred, [caps[j] for j in idx.tolist()])
                totce += ce.item() * len(idx)
            opt.zero_grad()
            scaler.scale(loss + args.ce_weight * ce).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(idx)
        print(f"epoch {epoch:3d}  mse {tot / len(x):.4f}  ce {totce / len(x):.4f}", flush=True)

    torch.save({"state_dict": bridge.state_dict(), "in_dim": x.shape[1],
                "n_tok": n_tok, "d_tok": d_tok}, args.bridge)
    print(f"wrote {args.bridge}")


def caption_ce(m, proc, pred, caps):
    """Cross-entropy of the reference caption under the frozen OPT, conditioned on pred tokens."""
    dev = pred.device
    li = m.language_projection(pred.half())
    tok = proc.tokenizer(caps, return_tensors="pt", padding=True, truncation=True,
                         max_length=48).to(dev)
    te = m.language_model.get_input_embeddings()(tok.input_ids)
    embeds = torch.cat([li, te], dim=1)
    mask = torch.cat([torch.ones(li.shape[:-1], dtype=torch.long, device=dev),
                      tok.attention_mask], dim=1)
    labels = torch.cat([torch.full(li.shape[:-1], -100, dtype=torch.long, device=dev),
                        tok.input_ids.masked_fill(tok.attention_mask == 0, -100)], dim=1)
    return m.language_model(inputs_embeds=embeds, attention_mask=mask, labels=labels).loss.float()


def stage_generate(args):
    torch.manual_seed(args.seed)
    ck = torch.load(args.bridge, map_location="cpu", weights_only=False)
    bridge = nn.Linear(ck["in_dim"], ck["n_tok"] * ck["d_tok"]).cuda()
    bridge.load_state_dict(ck["state_dict"])
    bridge.eval()

    if args.enigma_dir:
        concepts_n = len(os.listdir(os.path.join(ROOT, f"{args.split}_images")))
        x = torch.stack([torch.load(os.path.join(args.enigma_dir, str(i), "predicted_embeds.pt"),
                                    weights_only=False) for i in range(concepts_n)])
        x = nn.functional.normalize(x.float(), dim=-1)
    elif args.npz:  # step 4: EEG. data_average=True -> one row per test concept, in concept order
        d = np.load(args.npz)
        x = nn.functional.normalize(torch.from_numpy(d["eeg"]).float(), dim=-1)
        assert (d["object"] == np.arange(len(x))).all(), "npz rows are not in concept order"
    else:         # step 3: the same images the bridge was fit on
        x = embeddings(args.split, args)

    proc, m = load_blip2(args.hf_cache, with_vision=False)
    bos = torch.tensor([[m.config.text_config.bos_token_id]], device="cuda")
    concepts = sorted(os.listdir(os.path.join(ROOT, f"{args.split}_images")))
    outs = []
    for i in range(0, len(x), args.batch_size):
        b = x[i: i + args.batch_size].cuda()
        with torch.no_grad():
            li = m.language_projection(bridge(b).view(len(b), ck["n_tok"], ck["d_tok"]).half())
            # mirror BLIP-2.generate: query tokens then a BOS embedding
            be = m.language_model.get_input_embeddings()(bos.expand(len(b), -1))
            ids = m.language_model.generate(
                inputs_embeds=torch.cat([li, be], dim=1),
                attention_mask=torch.ones(li.shape[0], li.shape[1] + 1, dtype=torch.long,
                                          device="cuda"),
                max_new_tokens=40,
                # beam search on a fuzzy prefix walks into repetition loops; nucleus sampling
                # cannot, because it never commits to the argmax continuation
                **({"do_sample": True, "top_p": args.top_p, "temperature": args.temperature}
                   if args.do_sample else {"num_beams": args.num_beams}),
                # a weak prefix sends OPT into "the the the ..."; both arms degenerated without
                # these, the image ceiling included, so it was a decoding artifact not a result
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                min_new_tokens=args.min_new_tokens)
        outs += [c.strip() for c in proc.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    assert len(outs) == len(concepts), f"{len(outs)} captions vs {len(concepts)} concepts"
    with open(args.out, "w") as f:
        for c, cap in zip(concepts, outs):
            f.write(json.dumps({"concept": c, "captions": [cap]}) + "\n")
    print(f"wrote {args.out}")
    for c, cap in list(zip(concepts, outs))[:10]:
        print(f"  {c:28s} {cap}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["cache", "train", "generate"], required=True)
    p.add_argument("--split", choices=["test", "training"], default="training")
    p.add_argument("--per_concept", type=int, default=10, help="cache only; 1 for the test split")
    p.add_argument("--input", choices=["proj", "raw", "vith"], default="proj",
                   help="proj = the subject's 512-d shared space; raw = 3200-d InternViT")
    p.add_argument("--checkpoint_dir", default=None, help="required for --input proj")
    p.add_argument("--feature_dim", type=int, default=512)
    p.add_argument("--npz", default=None, help="generate: EEG dump from evaluate.py --dump_npz")
    p.add_argument("--feat_dir", default=None, help="--input vith: raw ViT-H feature dir")
    p.add_argument("--noise", type=float, default=0.0,
                   help="train: sigma of Gaussian noise added to unit-norm inputs. ENIGMA's "
                        "predicted embedding has cos 0.36 with the real one, and cos ~ "
                        "1/sqrt(1+sigma^2), so sigma 2.59 matches the corruption at test time")
    p.add_argument("--pred_npy", default=None,
                   help="train: use these predicted embeddings as the bridge input instead of real "
                        "image features. Cross-split cosine differs (0.531 train vs 0.360 test), so "
                        "pair it with --noise to bring the training cosine down to the test one")
    p.add_argument("--enigma_dir", default=None,
                   help="generate: caption ENIGMA's own predicted ViT-H embeddings")
    p.add_argument("--bridge", default="data/things_eeg/captions/bridge.pth")
    p.add_argument("--out", default="data/things_eeg/captions/test.bridge.jsonl")
    p.add_argument("--ce_weight", type=float, default=1.0, help="0 = MSE only (no captions needed)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=3)
    p.add_argument("--do_sample", action="store_true", help="nucleus sampling instead of beams")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="sampling is stochastic; pin it")
    p.add_argument("--repetition_penalty", type=float, default=1.5)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--min_new_tokens", type=int, default=5)
    p.add_argument("--hf_cache", default=os.environ.get("HF_HOME"))
    args = p.parse_args()
    assert args.input != "proj" or args.checkpoint_dir, "--input proj needs --checkpoint_dir"
    assert args.input != "vith" or args.feat_dir, "--input vith needs --feat_dir"
    {"cache": stage_cache, "train": stage_train, "generate": stage_generate}[args.stage](args)


if __name__ == "__main__":
    main()
