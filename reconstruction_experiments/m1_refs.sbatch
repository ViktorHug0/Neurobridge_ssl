#!/bin/bash
#SBATCH --job-name=m1_refs
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/m1_refs_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/m1_refs_%j.err

# Regenerate the test references with a NEUTRAL prompt and five samples each.
#
# Why neutral: the old prompt said "Name the main object, its colour, and what it is on or in front
# of", so 88% of references named a colour. Scoring colour decoding against colours our own prompt
# demanded was circular, and it is also what pushed colour words past cwBLEU's 15% template bar and
# forced the exemption that let a tuned constant caption score 16.94. Removing the instruction
# removes both problems at the source.
#
# Why five: CIDEr's tf-idf weighting is a *consensus* over references and is high variance with one.
# COCO, which the fMRI captioning literature scores against, has five human captions per image.
# Qwen2-VL 7B fp16 needs ~16GB, so this wants a 3090 rather than a 3080.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python

$PY reconstruction_experiments/make_captions.py --model qwen --split test \
    --images_per_concept 1 --neutral --n_samples 5 \
    --out_name test.qwen_neutral.jsonl

$PY -c "
import json
d=[json.loads(l) for l in open('data/things_eeg/captions/test.qwen_neutral.jsonl') if l.strip()]
print(f'{len(d)} concepts, {sum(len(x[\"captions\"]) for x in d)/len(d):.1f} refs each')
import re
COL=set('red orange yellow green blue purple pink brown black white grey gray'.split())
f=sum(bool(set(re.findall(r'[a-z]+',c.lower()))&COL) for x in d for c in x['captions'])/sum(len(x['captions']) for x in d)
print(f'fraction naming a colour: {100*f:.1f}%  (old prompt: 88%)')
for x in d[:5]: print(' ', x['concept'], '|', x['captions'][0])
"
