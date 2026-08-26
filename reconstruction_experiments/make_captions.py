"""Caption THINGS images -> data/things_eeg/captions/<split>.<model>.jsonl (one line per concept).

Two roles, two models, deliberately different so the references are not the generator's own output:
  --model qwen   attribute-rich ground-truth captions ("a blue bowl of cashews")
  --model blip2  the ceiling run: what a real captioner produces from the real image

Caption order within a concept matches sorted(os.listdir(...)), i.e. the same order
extract_feature.py uses, so captions[concept][j] pairs with image_train[concept_idx, j].

Resumable: appends per concept and skips concepts already in the jsonl (low qos gets requeued).

  .venv/bin/python make_captions.py --model qwen  --split test
  .venv/bin/python make_captions.py --model qwen  --split training --images_per_concept 10
"""
import argparse
import json
import os

import torch
from PIL import Image

ROOT = "/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set"
# No class label in the prompt -- MindAlign injects "<label>" into theirs, which makes the text a
# class-name embedding in disguise. We want the caption to depend only on the pixels.
#
# --neutral drops every attribute instruction. The original prompt below asked for the colour, so
# 88% of references named one; measuring "colour accuracy" against references our own prompt had
# instructed to mention colours was circular, and it was also what pushed colour words over
# cwBLEU's 15% template-frequency bar and forced the exemption that made the metric gameable.
# The neutral prompt is the analogue of the free-form human COCO captions the fMRI captioning
# literature scores against.
QWEN_PROMPT = ("Describe this image in one short sentence. Name the main object, its colour, "
               "and what it is on or in front of. Do not start with 'a photo of'.")
QWEN_PROMPT_NEUTRAL = "Describe this image in one sentence."


def load_qwen(cache):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    name = "Qwen/Qwen2-VL-7B-Instruct"
    # cap visual tokens: THINGS images are small and single-object, 1024 is plenty and keeps
    # the 7B in fp16 comfortably inside a 3090's 24GB
    proc = AutoProcessor.from_pretrained(name, cache_dir=cache,
                                         min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    proc.tokenizer.padding_side = "left"  # batched generation
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.float16, cache_dir=cache).to("cuda").eval()

    def run(pils, prompt=QWEN_PROMPT, n_samples=1):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        batch = proc(text=[text] * len(pils), images=pils, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            # n_samples>1 mimics COCO's five independent human references, which is what CIDEr's
            # tf-idf consensus weighting is defined over -- with one reference it is high variance
            # top_k MUST be overridden: Qwen2-VL ships generation_config top_k=1 with
            # temperature 0.01, so setting only temperature/top_p still decodes greedily and every
            # "sample" comes back identical (caught after a full 200-concept run).
            kw = dict(do_sample=True, top_k=50, top_p=0.9, temperature=1.0,
                      num_return_sequences=n_samples) if n_samples > 1 else dict(do_sample=False)
            out = model.generate(**batch, max_new_tokens=40, **kw)
        out = out[:, batch["input_ids"].shape[1]:]  # strip the prompt
        flat = [c.strip() for c in proc.batch_decode(out, skip_special_tokens=True)]
        return [flat[i * n_samples:(i + 1) * n_samples] for i in range(len(pils))] \
            if n_samples > 1 else flat

    return run


def load_blip2(cache):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    name = "Salesforce/blip2-opt-2.7b"
    proc = Blip2Processor.from_pretrained(name, cache_dir=cache)
    model = Blip2ForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.float16, cache_dir=cache).to("cuda").eval()

    def run(pils):
        batch = proc(images=pils, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=40, num_beams=3)
        return [c.strip() for c in proc.batch_decode(out, skip_special_tokens=True)]

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["qwen", "blip2"], required=True)
    p.add_argument("--split", choices=["test", "training"], required=True)
    p.add_argument("--images_per_concept", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output_dir", default="data/things_eeg/captions")
    p.add_argument("--hf_cache", default=os.environ.get("HF_HOME"))
    p.add_argument("--flat_dir", default=None,
                   help="caption recon_*.png from one directory instead of the concept tree; "
                        "file order is taken to match sorted <split> concept order")
    p.add_argument("--out_name", default=None, help="override the output jsonl basename")
    p.add_argument("--neutral", action="store_true",
                   help="qwen: prompt with no attribute instruction (see QWEN_PROMPT_NEUTRAL)")
    p.add_argument("--n_samples", type=int, default=1,
                   help="qwen: independent sampled references per image; 5 matches COCO, which is "
                        "what CIDEr's consensus weighting expects")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir,
                        args.out_name or f"{args.split}.{args.model}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(line)["concept"] for line in f if line.strip()}
        print(f"resuming: {len(done)} concepts already captioned", flush=True)

    image_dir = os.path.join(ROOT, f"{args.split}_images")
    all_concepts = sorted(os.listdir(image_dir))
    flat = None
    if args.flat_dir:
        import glob as _glob
        # ENIGMA writes JPEG, our own reconstruct.py writes PNG; only one is ever present
        flat = sorted(_glob.glob(os.path.join(args.flat_dir, "recon_*.png"))
                      + _glob.glob(os.path.join(args.flat_dir, "recon_*.jpg")))
        assert len(flat) == len(all_concepts), f"{len(flat)} images vs {len(all_concepts)} concepts"
    concepts = [c for c in all_concepts if c not in done]
    if not concepts:
        print("nothing to do")
        return

    run = {"qwen": load_qwen, "blip2": load_blip2}[args.model](args.hf_cache)
    with open(path, "a") as f:
        for i, concept in enumerate(concepts):
            if flat:
                pils = [Image.open(flat[all_concepts.index(concept)]).convert("RGB")]
            else:
                files = sorted(os.listdir(os.path.join(image_dir, concept)))[: args.images_per_concept]
                pils = [Image.open(os.path.join(image_dir, concept, x)).convert("RGB") for x in files]
            kw = {} if args.model == "blip2" else {
                "prompt": QWEN_PROMPT_NEUTRAL if args.neutral else QWEN_PROMPT,
                "n_samples": args.n_samples}
            caps = []
            for j in range(0, len(pils), args.batch_size):
                out = run(pils[j: j + args.batch_size], **kw)
                # n_samples>1 returns a list per image; one image per test concept, so flatten
                caps += [c for o in out for c in o] if args.n_samples > 1 else out
            f.write(json.dumps({"concept": concept, "captions": caps}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(concepts)}] {concept}: {caps[0]}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
