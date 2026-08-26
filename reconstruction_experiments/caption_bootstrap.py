"""Two null models for a caption arm's cwBLEU, because they answer different questions.

--mode floor    : is the arm worth more than one fixed generic caption? (the practical bar)
--mode permute  : does the concept<->caption PAIRING carry information, holding the emitted
                  caption distribution exactly fixed? (controls bank, style and length at once)
--vs <other>    : paired bootstrap between two arms over the same resampled concepts.

Note the template pools differ by design. `floor`/`permute` build it from refs+hyps, so their
cwBLEU is arm-dependent and NOT comparable across arms -- valid only against their own null, which
uses the same template. `--vs` builds it from the references alone, matching caption_eval.py, so
the two arms are scored identically. That is why a `--vs` cwBLEU can differ from a `permute` one
for the same file.

An arm can pass `permute` and fail `floor`: specific-but-wrong captions score badly when
mispaired, while a safe generic caption always scores middling. Report both.

cwBLEU is corpus-level, so a per-concept score cannot just be averaged -- each resample has to
recompute the corpus statistic over the resampled concepts.

  .venv/bin/python caption_bootstrap.py --hyp data/things_eeg/captions/test.enigma_blip2.jsonl
"""
import argparse

import numpy as np
from sacrebleu.metrics import BLEU

from caption_eval import PRIOR_CAPTION, _load_captions, _template_words, _words


def cwbleu(hyps, refs, template):
    strip = lambda s: " ".join(w for w in _words(s) if w not in template) or "."
    return BLEU(max_ngram_order=1, effective_order=True).corpus_score(
        [strip(h) for h in hyps], [[strip(r) for r in refs]]).score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hyp", required=True)
    p.add_argument("--refs", default="data/things_eeg/captions/test.qwen.jsonl")
    p.add_argument("--mode", default="floor", choices=["floor", "permute"])
    p.add_argument("--vs", default=None, help="second arm; paired bootstrap, refs-only template")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    ref_d, hyp_d = _load_captions(a.refs), _load_captions(a.hyp)
    concepts = sorted(set(ref_d) & set(hyp_d))
    refs = [ref_d[c][0] for c in concepts]
    hyps = [hyp_d[c][0] for c in concepts]
    base = [PRIOR_CAPTION] * len(concepts)
    template = _template_words(refs + hyps)

    rng = np.random.default_rng(a.seed)
    if a.vs:
        other_d = _load_captions(a.vs)
        cs = sorted(set(ref_d) & set(hyp_d) & set(other_d))
        r = [ref_d[c][0] for c in cs]
        h1, h2 = [hyp_d[c][0] for c in cs], [other_d[c][0] for c in cs]
        tmpl = _template_words(r)  # refs only, so both arms face the same metric
        idx = rng.integers(0, len(cs), size=(a.n, len(cs)))
        d = np.array([cwbleu([h1[i] for i in row], [r[i] for i in row], tmpl)
                      - cwbleu([h2[i] for i in row], [r[i] for i in row], tmpl) for row in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{len(cs)} concepts, {a.n} resamples, refs-only template")
        print(f"A {cwbleu(h1, r, tmpl):.2f}  B {cwbleu(h2, r, tmpl):.2f}")
        print(f"delta {cwbleu(h1, r, tmpl) - cwbleu(h2, r, tmpl):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"  p={2 * min((d <= 0).mean(), (d >= 0).mean()):.4f}")
        return
    if a.mode == "permute":
        obs = cwbleu(hyps, refs, template)
        null = np.array([cwbleu([hyps[i] for i in rng.permutation(len(concepts))], refs, template)
                         for _ in range(a.n)])
        lo, hi = np.percentile(null, [2.5, 97.5])
        print(f"{len(concepts)} concepts, {a.n} permutations")
        print(f"cwBLEU {obs:.2f}  null mean {null.mean():.2f}  null 95% [{lo:.2f}, {hi:.2f}]"
              f"  p={(null >= obs).mean():.4f}")
        return

    obs = cwbleu(hyps, refs, template) - cwbleu(base, refs, template)
    idx = rng.integers(0, len(concepts), size=(a.n, len(concepts)))
    deltas = np.array([cwbleu([hyps[i] for i in row], [refs[i] for i in row], template)
                       - cwbleu([base[i] for i in row], [refs[i] for i in row], template)
                       for row in idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p: how often the resampled delta lands on the other side of zero
    pv = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    print(f"{len(concepts)} concepts, {a.n} resamples")
    print(f"cwBLEU hyp {cwbleu(hyps, refs, template):.2f}  prior {cwbleu(base, refs, template):.2f}")
    print(f"delta {obs:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p={pv:.4f}")


def demo():
    """Identical arms must give delta 0; a strictly better arm must give delta > 0."""
    refs = ["a red apple on a table", "a blue car on a road"] * 20
    tmpl = _template_words(refs)
    assert abs(cwbleu(refs, refs, tmpl) - 100.0) < 1e-6, "self-BLEU must be 100"
    worse = ["a close up of an object"] * len(refs)
    assert cwbleu(refs, refs, tmpl) > cwbleu(worse, refs, tmpl), "identical must beat constant"
    print("demo ok")


if __name__ == "__main__":
    import sys
    demo() if "--demo" in sys.argv else main()
