# Three fused TSConv/ATM designs

User-approved comparison: G1/G2/G3, each on one RTX3080, outer subjects1–5,
one seed3300, pairwise mixup, 128D alignment. Each fold trains on the other
nine subjects from1–10. No distillation, learned fusion, diversity regularizer,
transductive adaptation, or externally selected test checkpoint.

## Architectures

All temporal stems retain40 filters, physical-time kernel30, pool51/stride5,
producing40×63×35 electrode-resolved features. Two private alignment/image heads
target raw IV33 and IV28 features. Training minimizes mean branch contrastive loss.

- **G1, static + dynamic pooling:** compute the temporal stem once. The static
  branch uses native TS spatial convolution/readout. The dynamic branch maps
  each electrode's40×35 descriptor to64D with private LayerNorm, adds a learned
  electrode identity embedding, and uses four learned queries to pool over all63
  electrodes. Four64D pooled vectors feed a256→128 readout. These are four queries,
  not four replicated backbones. No subject-ID token in this new pooling module.
- **G2, shared readout + private corrections:** reproduce F4's shared stem,
  private BN/spatial tails,250D post-stem electrode attention, identity adapters,
  and private LayerNorm. Replace each of the two readout dense matrices by a
  shared dense matrix plus branch-private rank16 correction. Corrections are
  initialized to zero output, making initial evaluation exactly equal to F4.
  Shared dense weights process two inputs; weight sharing does not imply one
  dense computation. Native F4 subject-token policy is retained.
- **G3, attention side-path:** one TS backbone computes its stem, BN and ELU;
  a64D, four-head electrode self-attention block supplies a residual correction
  before spatial collapse. Its FFN width is128. Learned electrode identities,
  private LayerNorm and an initial tanh(0.1) residual gain preserve a direct
  convolutional bypass. One spatial tail/readout feeds two private heads.
  Setting residual gain to zero exactly recovers the TS backbone in evaluation.
  There is no subject-ID token in this new side-path.

G1/G3 are intentionally lightweight new mechanisms, not exact ATM replicas.
G2 is the controlled rank-correction idea and retains F4's larger attention.

## Training and selection

FP32, AdamW3e-4/wd1e-4, grouped nominal batch1024/effective1017 (113 images ×9
subjects), pairwise alpha0.5. Pure conv/pool and attention blocks use activation
recomputation where needed; BatchNorm updates are not recomputed. Seed3300,
training-dropout seed13300, isolated epoch-specific mixup RNG.

**Maximum100 epochs; patience20.** Checkpoint selection uses strictly increasing
fixed equal-row-z fused validation top1. Ties keep the earliest checkpoint and
consume patience; validation loss does not break ties. One coherent shared-model
checkpoint serves both heads. This differs from the old F1–F6 mean-loss selector,
so comparisons to that grid are not architecture-only causal contrasts.

The source-concept split is checked against existing TS128 reference files:
165 held concepts,1650 unique held images,14850 EEG queries (9source subjects).
Sort (object,image) keys, permute with dedicated NumPy seed3300, partition into
eight200-query panels and one50-query panel. Pad the final candidate panel with
the first150 keys as distractors only. All panels contain200 distinct images and
are identical across architectures/folds. Every held image is queried once for
each source subject, with **no averaging EEG across subjects**. Candidate vectors
come only from held image features and their appropriate image projector.

Only165 concepts are held, so these panels cannot have200 distinct concepts;
they contain multiple images from some concepts. Final outer testing is the usual
200-image/200-concept retrieval. Neither outer EEG nor labels are loaded until
after source-validation checkpoint selection has finished. Report this protocol
distinction rather than calling validation panels identical to the outer task.

## Benchmarks and provenance

Measured on513 during preflight; four training steps (median final three), full
effective1017 contrastive batch. EEG-only inference uses200 trials, excludes image
projection and ranking, and is distinct from end-to-end retrieval latency.

| Arm | Total parameters | EEG-side parameters | Train step | EEG inference/200 | Peak allocated GPU |
|---|---:|---:|---:|---:|---:|
| G1 | 1,291,194 | 471,736 | 0.138s | 6.08ms | 6.81GiB |
| G2 | 2,403,423 | 1,583,965 | 0.242s | 11.10ms | 6.80GiB |
| G3 | 1,373,427 | 553,969 | 0.158s | 7.36ms | 6.09GiB |

Training-step figures exclude loading/mixup and validation; full-epoch times are
recorded by production workers. Image heads contain819456 parameters total.
Preflight validates structures, genuine single stem execution, alias persistence,
initial G2/F4 equivalence, private correction isolation, G3 bypass, historical
pairwise RNG equivalence, panel identity mapping, full-batch memory, and bitwise
interrupted/resumed model/optimizer/RNG/selection for all three architectures.

Output: `results/things_eeg/fusion_three_20260915`. Each worker loads code once
and runs five folds in the same process; it does not depend on live `train.py`.
Startup/resume requires the preflight source hashes; in-memory code is unchanged
by unrelated edits while a queue runs. Sources are recorded in every manifest.
Training helpers are local, with pairwise behavior tested against the historical
implementation. No old experiment source or output is modified.

## Launch / recovery

`run.sh G1` runs in local tmux; `remote.sbatch` allocates one3080 each forG2/G3,
excluding513, with29GiB RAM perjob. GPU and arm locks prevent duplicate workers.
Every epoch atomically saves model, optimizer, RNG, best snapshot, history and
patience state in `last.pth`, plus `best.pth`. Restart the same worker to replay
an interrupted partial epoch, resume incomplete folds and skip completed folds.
Per-arm `status-G*.json` reports failures; `summary.json` aggregates completed
production folds only. Per-fold `panels.json` records the full validation protocol;
`test_scores.npz` retains both score matrices and raw normalized embeddings.

Launch September15: local tmux `fusion-three-G1-513`; remote Slurm array53581,
tasks2 and3 forG2/G3. All three production queues run subjects1→5 sequentially.
