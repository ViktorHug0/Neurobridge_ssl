# NeuroBridge
This repo is about decoding images from EEG, mainly on THINGS-EEG-2 in the inter-subject setting: train on several people, then test on a held-out one.

At a high level, the pipeline is pretty simple. EEG gets preprocessed, images are represented with frozen visual features, and an EEG encoder is trained to land in the same embedding space so EEG trials can retrieve the right image. The main training recipe is a contrastive loss with grouped multi-positive batches, plus a few optional variants like subject mixup and train-time SAW.

At test time, the interesting part is the adaptation side. Besides plain cosine retrieval, the repo also has the SAW / CSLS / Sinkhorn / soft-Procrustes tools that reshape the held-out subject's query geometry and give a much stronger transductive evaluation setup.

If you just want the main pieces, start here:

- `train.py`
- `evaluate.py`
- `module/dataset.py`
- `module/loss.py`
- `module/sampler.py`
- `module/util.py`
- `module/eeg_encoder/model.py`
- `scripts/things_eeg/inter-subjects.sh`

There are still some older utilities around for augmentation and feature extraction, but they are not the main path anymore.

To run things, use the local virtual environment:

```bash
source .venv/bin/activate
```

Main THINGS-EEG sweeps:

- `scripts/things_eeg/inter-subjects.sh`
- `scripts/things_eeg/inter-subject-mixup.sh`
- `scripts/things_eeg/projector_size_sweep.sh`
- `scripts/things_eeg/multipos_loss_sweep.sh`

Main adaptation scripts:

- `scripts/things_eeg/progressive_sattc_candidate_sweep.py`
- `scripts/things_eeg/session_split_transfer_experiment.py`
- `scripts/things_eeg/transfer_calibration_experiment.py`


