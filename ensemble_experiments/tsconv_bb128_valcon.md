# TSConv backbone128 / alignment128 ValCon reference

Requested before launching F1–F6. Inspection of 6,155 configuration files under
`results/things_eeg` found no TSConv run combining backbone128 with concept-based
validation selection. Historical TSConv layer ValCon sweeps use backbone1024 and
alignment512; the compact IV33 solo uses backbone1024 and alignment128. The older
mechanism study includes width128 paired models but has a different subject and
concept-validation protocol.

Train five LOSO folds, outer subjects 1–5, nine source subjects per fold. Preserve
the compact IV33 solo recipe, changing the backbone output width 1024 → 128:
TSConv_parameterizable, 40 temporal/spatial/projection filters, temporal kernel30,
pool51/stride5, 250 samples, 63 electrodes. TS position resolution stays35.
The EEG projector is linear128→128; the image projector is linear3200→128.
Image target InternViT layer33, pairwise mixup alpha0.5, seed3300, FP32,
batch1024 (effective1017), AdamW lr3e-4/wd1e-4, no scheduler, max100 epochs,
patience20. Select the minimum source-concept validation loss using the same
165 held concepts and seed20260822. Outer test is evaluated after selection.

The wrapper substitutes the model constructor into the existing tested compact
trainer in its own process. Historical files are not modified. The trainer saves
epoch-resumable optimizer/RNG state and the selected checkpoint. A source hash
manifest guards every launch/resume; each worker/run has a filesystem lock.

Before launch, `--prepare` runs the established full-batch FP32 train/inference
benchmark with the new constructor. Worker0 on513 runs subjects1/3/5; worker1 on
an additional3080 runs2/4. `summary.json` reports the new TSConv accuracy and its
equal row-z ensemble with the archived ATM reference, checking candidate IDs.
Do not launch the sharing grid until this reference has been evaluated.

Outputs: `results/things_eeg/tsconv_bb128_valcon_20260911/`.

```bash
source .venv/bin/activate
python -m ensemble_experiments.tsconv_bb128_valcon --prepare
sbatch ensemble_experiments/tsconv_bb128_valcon.sbatch
tmux new-session -d -s ts128-valcon-513 'bash /nasbrain/p20fores/Neurobridge_SSL/ensemble_experiments/tsconv_bb128_valcon_local.sh'
```
