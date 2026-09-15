# Stochastic full-data joint twin ensemble

## Question

Can the strongest mechanism from the mental-imagery ensemble—stochastic member
specialization while retaining the full training population—improve an inductive
THINGS-EEG2 LOSO score ensemble?

## Matched arms

Both arms jointly train two TSConv branches with seeds 3300/3301, all nine source
subjects, pairwise raw-EEG SubjectMix, individual multi-positive losses, and a
`0.5`-weighted deployed row-z ensemble loss. Checkpoints use the repository's
historical test-loss selection protocol, so this is an engineering diagnostic.
Both use batch size 512 so the matched run fits the 10 GB RTX 3080.

| Arm | Spectral gain SD | Channel dropout | Per-query member keep |
|---|---:|---:|---:|
| `joint_b05_control` | 0 | 0 | 1.00 |
| `stoch_spec05_cdrop10_keep75_b05` | 0.5 | 0.10 | 0.75 |

The stochastic transforms are train-only and independently drawn for each member.
Member participation is redrawn per query and batch, with at least one branch active.
No source subject is permanently excluded from either member. At evaluation both
unaugmented branches participate in uniform row-z score fusion.

The intervention is intentionally bundled as the direct MI recipe. If it wins, a
subsequent ablation should separate spectral gain, channel dropout, and stochastic
participation. If it loses, running three separate ablations would spend substantially
more test-fold adaptation budget without evidence that the bundle transfers.

Execution note (2026-09-02): the full matched control was deferred after a brief GPU
smoke because the existing three-member full-source soft-weight TSConv committee is
already the relevant engineering reference. The ten-fold launch runs only the stochastic
arm. A matched joint-training control remains available if the intervention wins and its
causal source needs to be isolated.

```bash
source .venv/bin/activate
python -m ensemble_experiments.decorrelated_models.smoke
sbatch ensemble_experiments/decorrelated_models/stochastic_full_data.sbatch

python -m ensemble_experiments.decorrelated_models.analyze_stochastic_full_data \
  --require-complete
```

For persistent interactive allocations, split subjects rather than arms so each
subject's matched pair runs on the same GPU:

```bash
SUBJECTS="1 2 3 4 5" RUNNER_NAME=runner-513 \
  bash ensemble_experiments/decorrelated_models/run_stochastic_full_data_local.sh

EXPECTED_HOST=sl-tp-br-522 PHYSICAL_CUDA_DEVICE=1 \
  SUBJECTS="6 7 8 9 10" RUNNER_NAME=runner-522 \
  bash ensemble_experiments/decorrelated_models/run_stochastic_full_data_local.sh
```
