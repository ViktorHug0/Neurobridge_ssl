"""Cross-subject EEG->caption by retrieval-copy, plus the controls the literature omits.

Retrieve the nearest test image for each EEG trial, emit that image's caption. No decoder, no
training. This is the number a generative decoder has to beat -- RealMind-style L2Cap is only
worth building if it clears this.

  .venv/bin/python evaluate.py --checkpoint_dir <ckpt> --output_dir /tmp/x --output_name y \
      --eval_mode plain_cosine --test_subject_id 1 --dump_npz feats/sub1.npz
  .venv/bin/python caption_eval.py --npz feats/sub1.npz

Controls (--arms): retrieval | random | prior | shuffle. `prior` emits one fixed generic caption
for every trial and is the floor -- report your score minus that, not your score.
"""
import argparse
import json
import os

import numpy as np

CONCEPT_DIR = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set/test_images"
PRIOR_CAPTION = "a close up of an object on a white background"


def _load_captions(path):
    """{concept: [caption, ...]} from either a .json dict or the resumable .jsonl."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            return {d["concept"]: d["captions"] for d in map(json.loads, filter(str.strip, f))}
    return json.load(open(path))


COLOURS = set("red orange yellow green blue purple violet pink brown black white grey gray gold "
              "golden silver beige tan cream turquoise teal maroon navy amber ivory "
              "colorful colourful".split())


def _words(s):
    import re
    return re.findall(r"[a-z]+", s.lower())


def _template_words(captions, frac=0.15):
    """Words shared by >frac of the caption pool -- the scaffolding every arm gets for free.

    Colours are exempt: they clear the frequency bar (88% of references name one) but they are
    the attribute we are trying to decode, so stripping them would hide the signal.
    """
    from collections import Counter
    counts = Counter(w for c in captions for w in set(_words(c)))
    return {w for w, k in counts.items() if k > frac * len(captions)} - COLOURS


def _colour_acc(hyps, refs):
    """Of the references that name a colour, how often does the hypothesis name one of them."""
    pairs = [(set(_words(h)) & COLOURS, set(_words(r)) & COLOURS) for h, r in zip(hyps, refs)]
    scored = [bool(h & r) for h, r in pairs if r]
    return 100 * float(np.mean(scored)) if scored else float("nan")


def _scores(hyps, refs, template):
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    # effective_order: captions are short, so drop missing higher-order n-grams instead of zeroing
    out = {f"BLEU-{n}": BLEU(max_ngram_order=n, effective_order=True).corpus_score(hyps, [refs]).score
           for n in (1, 4)}
    # cwBLEU: same metric with the shared scaffolding stripped, so chance sits near 0 instead of ~35
    strip = lambda s: " ".join(w for w in _words(s) if w not in template) or "."
    out["cwBLEU"] = BLEU(max_ngram_order=1, effective_order=True).corpus_score(
        [strip(h) for h in hyps], [[strip(r) for r in refs]]).score
    out["colour"] = _colour_acc(hyps, refs)
    rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    pairs = [rs.score(r, h) for h, r in zip(hyps, refs)]
    out["ROUGE-1"] = 100 * float(np.mean([p["rouge1"].fmeasure for p in pairs]))
    out["ROUGE-L"] = 100 * float(np.mean([p["rougeL"].fmeasure for p in pairs]))
    try:
        import nltk
        # nltk's own downloader is blocked here; wordnet was fetched by hand into .nltk_data
        nltk.data.path.insert(0, os.path.join(os.path.dirname(__file__), ".nltk_data"))
        from nltk.translate.meteor_score import meteor_score
        out["METEOR"] = 100 * float(np.mean(
            [meteor_score([r.split()], h.split()) for h, r in zip(hyps, refs)]))
    except Exception:
        out["METEOR"] = float("nan")  # needs .nltk_data/corpora/wordnet
    return out


def _predict(eeg, image, arm, rng):
    """Return pool index chosen for each trial."""
    n = len(eeg)
    if arm == "random":
        return rng.integers(0, n, size=n)
    if arm == "shuffle":
        eeg = eeg[rng.permutation(n)]
    sim = eeg @ (image / np.linalg.norm(image, axis=1, keepdims=True)).T
    return np.argmax(sim, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", nargs="+", default=[], help="feature dumps (one per subject); omit for --hyp_captions only")
    p.add_argument("--captions", default="data/things_eeg/captions/test.qwen.jsonl", help="reference captions")
    p.add_argument("--hyp_captions", default=None,
                   help="captions of the REAL images (e.g. test.blip2.jsonl) -> adds the 'ceiling' arm")
    p.add_argument("--arms", nargs="+", default=["retrieval", "prior", "random", "shuffle"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    captions = _load_captions(args.captions)
    concepts = sorted(os.listdir(CONCEPT_DIR))
    assert set(concepts) <= set(captions), "reference captions do not cover the test concepts"
    cap = [captions[c][0] for c in concepts]
    template = _template_words(cap)

    rng = np.random.default_rng(args.seed)
    rows = {}
    if args.hyp_captions:
        hyp = _load_captions(args.hyp_captions)
        # name the row after the file: this flag serves the ceiling AND the bridge arms
        name = os.path.basename(args.hyp_captions).replace("test.", "").replace(".jsonl", "")
        rows[name] = {"top1": float("nan"),
                      **_scores([hyp[c][0] for c in concepts], cap, template)}
    for arm in args.arms if args.npz else []:
        hyps, refs, hits = [], [], []
        for path in args.npz:
            d = np.load(path)
            obj = d["object"]
            pred = np.arange(len(obj)) if arm == "prior" else _predict(d["eeg"], d["image"], arm, rng)
            hyps += [PRIOR_CAPTION if arm == "prior" else cap[obj[j]] for j in pred]
            refs += [cap[o] for o in obj]
            hits += list(obj[pred] == obj) if arm != "prior" else [0] * len(obj)
        rows[arm] = {"top1": 100 * float(np.mean(hits)), **_scores(hyps, refs, template)}

    keys = list(next(iter(rows.values())))
    print(f"{'arm':<10}" + "".join(f"{k:>9}" for k in keys))
    for arm, r in rows.items():
        print(f"{arm:<10}" + "".join(f"{r[k]:>9.2f}" for k in keys))
    if "retrieval" in rows and "prior" in rows:
        print("\nabove-floor (retrieval - prior):")
        print(f"{'':<10}" + "".join(f"{rows['retrieval'][k] - rows['prior'][k]:>9.2f}" for k in keys))


if __name__ == "__main__":
    main()
