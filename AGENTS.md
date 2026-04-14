Guideline for THINGS-EEG-2 Image Decoding Project:

- The pipeline decodes images using the THINGS-EEG-2 dataset.
- Extract image embeddings from a pre-trained vision transformer (Intern-ViT), using features from an intermediate layer (to balance perceptual details and semantic content).
- Train an EEG encoder (TSConv, a basic CNN) to produce EEG embeddings.
- Use InfoNCE loss to match EEG embeddings to the corresponding image embeddings.
- The main evaluation task is 200-way matching (identify the correct image among 200 candidates).
- The primary goal is to optimize performance in the inter-subject setting: train on 9 subjects and test on a held-out 10th subject.
- Please note:
    - Data augmentation is not used anymore — do not reference it.
    - The most critical files are: train.py, dataset.py, sampler.py, loss.py, model.py, and scripts/things_eeg/inter-subject*.py.
- Always make minimal edits when modifying code.
- Always run source .venv/bin/activate before executing python scripts. They won't work otherwise.
