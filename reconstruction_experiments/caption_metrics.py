"""Caption metrics for the EEG->text arms, each reported against its own permutation null.

Colour accuracy is deliberately gone. It was circular -- the reference prompt instructed the VLM to
name a colour, so we were scoring colour decoding against colours we had asked for -- and it was
pure recall with no precision, so hedging was free: an arm emitting 1.9 colour words per caption
instead of 1.0 raised its own permutation null from 18.4 to 36.9, which is most of what looked like
a 29.94 -> 51.98 improvement.

  ident       200-way top-1/top-5 of the CAPTION against the 200 real test IMAGES in OpenCLIP
              ViT-H/14 joint space. Reference-free, so no reference prompt can bias it, and
              structurally immune to hedging: a generic caption retrieves nothing. Chance 0.5%.
              This is the identification protocol Horikawa's mind captioning reports, and the
              self-retrieval criterion of Luo et al. (ECCV 2018).
  concept     CHAIR-style. `hit` = the caption names the target THINGS concept. `wup` = best
              Wu-Palmer similarity between any noun in the caption and the target, so "kangaroo"
              for antelope scores as the near miss it is rather than as a flat error.
  CIDEr       tf-idf n-gram consensus over the references (pycocoevalcap).
  BERTScore   contextual-embedding F1, max over references.

Every score is paired with a permutation null: the arm's own captions dealt to the wrong concepts.
That holds the emitted distribution exactly fixed and destroys only the pairing, so hedging cannot
help it. Report the margin over the null, never the raw number.

  .venv/bin/python reconstruction_experiments/caption_metrics.py --arms test.gencap_pred_n108.jsonl ...
"""
import argparse
import functools
import json
import os
import re

import numpy as np
import torch

CAP = "data/things_eeg/captions"
ROOT = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"
VITH = "data/things_eeg/image_feature/ViT-H-14_final"


def load(path):
    with open(path) as f:
        return {json.loads(l)["concept"]: json.loads(l)["captions"] for l in f if l.strip()}


def words(s):
    return re.findall(r"[a-z]+", s.lower())


# ---- identification --------------------------------------------------------
def load_clip(device):
    """ViT-H is ~4GB; load it once for every arm, not once per arm."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    img = np.load(os.path.join(VITH, "image_test.npy"))
    img = img.reshape(-1, img.shape[-1]).astype(np.float32)
    return (model.to(device).eval(), open_clip.get_tokenizer("ViT-H-14"),
            img / np.linalg.norm(img, axis=1, keepdims=True))


def ident_matrix(hyps, clip, device):
    """(n_caption, n_image) cosine in OpenCLIP ViT-H/14 space -- the space the features are in."""
    model, tok, img = clip
    with torch.no_grad():
        t = model.encode_text(tok(hyps).to(device)).float()
    return torch.nn.functional.normalize(t, dim=-1).cpu().numpy() @ img.T


# ---- concept-level CHAIR ---------------------------------------------------
@functools.lru_cache(maxsize=None)
def _syn(word):
    from nltk.corpus import wordnet as wn
    return tuple(wn.synsets(word, pos=wn.NOUN)[:3])


@functools.lru_cache(maxsize=None)
def _wup(word, concept):
    best = 0.0
    for a in _syn(word):
        for b in _syn(concept):
            best = max(best, a.wup_similarity(b) or 0.0)
    return best


def concept_matrices(hyps, concepts):
    """(n_caption, n_concept) exact-hit and Wu-Palmer matrices, so permutations are lookups."""
    import nltk
    nltk.data.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".nltk_data"))
    heads = [c.split("_", 1)[1].replace("_", " ") for c in concepts]
    hit = np.zeros((len(hyps), len(heads)))
    wup = np.zeros_like(hit)
    hw = [set(words(h)) for h in hyps]
    for j, name in enumerate(heads):
        tgt, last = name.replace(" ", "_"), name.split()[-1]
        for i, ws in enumerate(hw):
            hit[i, j] = float(last in ws or name in hyps[i].lower())
            wup[i, j] = max((_wup(w, tgt) or _wup(w, last) for w in ws), default=0.0)
    return hit, wup


# ---- CIDEr / BERTScore -----------------------------------------------------
def cider(hyps, refs_list):
    from pycocoevalcap.cider.cider import Cider
    g = {i: [h] for i, h in enumerate(hyps)}
    r = {i: list(v) for i, v in enumerate(refs_list)}
    return 100 * Cider().compute_score(r, g)[0]


def bertscore(hyps, refs_list, scorer):
    """scorer is a preloaded BERTScorer -- bert_score.score() reloads RoBERTa on every call."""
    flat_h, flat_r, owner = [], [], []
    for i, (h, rs) in enumerate(zip(hyps, refs_list)):
        for r in rs:
            flat_h.append(h); flat_r.append(r); owner.append(i)
    f1 = scorer.score(flat_h, flat_r)[2].numpy()
    owner = np.array(owner)
    return 100 * float(np.mean([f1[owner == i].max() for i in range(len(hyps))]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--refs", default=f"{CAP}/test.qwen_neutral.jsonl")
    p.add_argument("--n", type=int, default=1000, help="permutations for ident/concept/CIDEr")
    p.add_argument("--n_bert", type=int, default=50, help="BERTScore permutations; each one rescores")
    p.add_argument("--n_cider", type=int, default=200,
                   help="CIDEr permutations. It rebuilds its document frequency on every call "
                        "(~0.4s at 5 refs), so it dominates the run; 200 draws still resolve "
                        "p<0.005, well past the 0.05 we report")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ref_d = load(a.refs)
    concepts = sorted(ref_d)
    refs_list = [ref_d[c] for c in concepts]
    print(f"{len(concepts)} concepts, {np.mean([len(r) for r in refs_list]):.1f} refs each, "
          f"{a.n} permutations ({a.n_bert} for BERTScore)\n")
    print(f"{'arm':26s} {'top1':>12s} {'top5':>12s} {'concept':>12s} {'wup':>12s} "
          f"{'CIDEr':>12s} {'BERTScore':>12s}")
    rng = np.random.default_rng(a.seed)
    perms = [rng.permutation(len(concepts)) for _ in range(a.n)]
    from bert_score import BERTScorer
    clip = load_clip(a.device)
    scorer = BERTScorer(lang="en", device=a.device, batch_size=64, rescale_with_baseline=False)

    for arm in a.arms:
        path = arm if os.path.sep in arm else os.path.join(CAP, arm)
        hyp_d = load(path)
        hyps = [hyp_d[c][0] for c in concepts]
        n = len(concepts)

        sim = ident_matrix(hyps, clip, a.device)
        order = np.argsort(-sim, axis=1)
        rank = np.array([int(np.where(order[i] == i)[0][0]) for i in range(n)])
        hitm, wupm = concept_matrices(hyps, concepts)
        eye = np.arange(n)

        obs = dict(top1=100 * (rank == 0).mean(), top5=100 * (rank < 5).mean(),
                   concept=100 * hitm[eye, eye].mean(), wup=100 * wupm[eye, eye].mean(),
                   CIDEr=cider(hyps, refs_list),
                   BERTScore=bertscore(hyps, refs_list, scorer))
        null = {k: [] for k in obs}
        for pm in perms:
            r = np.array([int(np.where(order[pm[i]] == i)[0][0]) for i in range(n)])
            null["top1"].append(100 * (r == 0).mean())
            null["top5"].append(100 * (r < 5).mean())
            null["concept"].append(100 * hitm[pm, eye].mean())
            null["wup"].append(100 * wupm[pm, eye].mean())
            if len(null["CIDEr"]) < a.n_cider:
                null["CIDEr"].append(cider([hyps[i] for i in pm], refs_list))
        for pm in perms[:a.n_bert]:
            null["BERTScore"].append(bertscore([hyps[i] for i in pm], refs_list, scorer))

        cells = []
        for k in ("top1", "top5", "concept", "wup", "CIDEr", "BERTScore"):
            nul = np.array(null[k])
            pv = (nul >= obs[k]).mean()
            cells.append(f"{obs[k]:6.2f}/{nul.mean():5.2f}{'*' if pv < 0.05 else ' '}")
        print(f"{os.path.basename(path)[5:-6]:26s} " + " ".join(f"{c:>12s}" for c in cells))
    print("\nobserved/null, * = p<0.05 against the permutation null")


if __name__ == "__main__":
    main()
