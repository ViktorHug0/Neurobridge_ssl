"""Shape/label plumbing for caption_bridge.py, on CPU with a stub decoder.

The CE term concatenates 32 query tokens with the tokenized caption and masks the query positions
with -100. Getting that off by one silently trains the bridge to predict the wrong token, so check
it here instead of discovering it after the hour-long Q-Former cache stage.

  .venv/bin/python test_caption_bridge.py
"""
import torch
import torch.nn as nn

import caption_bridge as cb

D_TOK, N_TOK, D_LM = 768, 32, 64


class _StubLM(nn.Module):
    """Stands in for BLIP-2's frozen OPT: records what it was handed, returns a real loss."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, D_LM)
        self.head = nn.Linear(D_LM, vocab)
        self.seen = {}

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds, attention_mask, labels):
        self.seen = {"embeds": inputs_embeds.shape, "mask": attention_mask.shape,
                     "labels": labels.clone()}
        logits = self.head(inputs_embeds)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, self.vocab), labels[:, 1:].reshape(-1), ignore_index=-100)
        return type("Out", (), {"loss": loss})()


class _StubBlip2(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.language_projection = nn.Linear(D_TOK, D_LM)
        self.language_model = _StubLM(vocab)


def main():
    import os

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cb.BLIP2, cache_dir=os.environ.get("HF_HOME"))
    proc = type("P", (), {"tokenizer": tok})()
    m = _StubBlip2(max(len(tok), int(tok.vocab_size) + 10)).float()

    caps = ["a red bus with two stories", "a blue bowl of cashews on a table"]
    pred = torch.randn(len(caps), N_TOK, D_TOK, requires_grad=True)
    # caption_ce casts to .half(); stub runs in fp32, so patch the cast away for the test
    orig_half = torch.Tensor.half
    torch.Tensor.half = lambda self: self
    try:
        loss = cb.caption_ce(m, proc, pred, caps)
    finally:
        torch.Tensor.half = orig_half

    seen, n_text = m.language_model.seen, m.language_model.seen["labels"].shape[1] - N_TOK
    assert seen["embeds"] == (len(caps), N_TOK + n_text, D_LM), seen["embeds"]
    assert seen["mask"] == seen["labels"].shape, (seen["mask"], seen["labels"].shape)

    labels = seen["labels"]
    assert (labels[:, :N_TOK] == -100).all(), "query positions must be ignored by the loss"
    assert (labels[:, N_TOK:] != -100).any(), "no caption token is being supervised"
    # the shorter caption is right-padded, and padding must not be supervised
    pad = tok(caps, return_tensors="pt", padding=True).attention_mask == 0
    assert (labels[:, N_TOK:][pad] == -100).all(), "padding leaked into the labels"
    # first real target sits at N_TOK+1: HF shifts, so it is predicted from the last query token
    assert labels[0, N_TOK] == tok(caps[0]).input_ids[0], "caption does not start at the query edge"

    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0, "no gradient reaches the bridge"

    # generate() prepends exactly one BOS embedding, so the mask is one longer than the tokens
    assert seen["mask"][1] == N_TOK + n_text
    print(f"loss={loss.item():.3f}  embeds={tuple(seen['embeds'])}  "
          f"supervised={int((labels != -100).sum())}/{labels.numel()} positions")
    print("OK")


if __name__ == "__main__":
    main()
