# Tiny TSConv + ATM: a controlled sharing study

Status: design only, 2026-09-09. Ten architecture configurations; no training
launched by this plan. Existing Nash and IV28-only experiments remain untouched.

## 1. Questions and scope

1. Which independent mechanisms preserve the pair's complementary errors?
2. Does sharing hurt individual strength, complementarity, or both?
3. Is the limitation parameter independence, normalization state, optimization,
   checkpoint coupling, or the need for genuinely different computations?
4. Which sharing saves measured GPU time rather than only parameter storage?

This is a branching intervention study, not an assumed monotonic curve in percent
parameters shared. Parameter location matters; performance may improve with some
sharing. Train from scratch first. Do not combine trained unlike weights, introduce
distillation, or change the optimizer to Nash in the primary grid.

Primary targets: TS branch -> raw InternViT33; ATM branch -> raw InternViT28.
Keep independent linear image projectors and EEG alignment heads, both 128D, in
all ten configurations. Fix uniform row-z score fusion. Do not change to SLERP
while changing sharing: that would confound optimization/architecture with fusion.

## 2. Existing tiny models and important compatibility issues

Source: run_tiny_compute_ensemble_fold.sh, run_tiny40_reference_fold.sh,
module/eeg_encoder/model.py and module/eeg_encoder/atm/atm.py.

| Native encoder | TinyTSConv | TinyATM |
|---|---|---|
| Convolution temporal/spatial/projection widths | 10 / 10 / 10 | 12 / 12 / 12 |
| Kernel, pool, stride | 25, 51, 5 | 25, 51, 5 |
| Input to convolution | EEG time axis, length250 | learned latent axis, length112 |
| Pooled positions | 36 | 8 |
| Dense backbone output | 256 | 128 |
| Attention | none | d_model112, d_ff112, 4 heads, 1 layer |
| Registered encoder parameters | 165,440 | 201,908 |

These registered counts were checked by CPU construction in this turn. ATM includes
56,500 parameters in subject_wise_linear modules that its forward does not use,
plus other conditional/unused embedding parameters. Report registered, trainable,
and forward-reachable counts separately; removing dead parameters is not compression
of a useful mechanism. A single backward's nonzero/allocated gradient count is not
the global reachable count (known versus unknown subject-token paths differ).

Image projector 3200->128 has 409,728 parameters PER branch, exceeding either tiny
encoder. Always report EEG-only and image-side counts separately. Frozen image
features do not make the learned image projectors free at training time.

The archived tiny pair reports 29.75% TS / 29.50% ATM / 34.60% fused, but both
members target IV28 and use test-selected checkpoints. It is historical context,
not the clean baseline for the proposed IV33/IV28 source-validation experiment.
Do not expect the new tiny reference to reach the full pair's 40.20%.

Legacy ATM indexing must be explicit: DataEmbedding prepends a subject token, and
iTransformer.forward retains the FIRST channels_num outputs. Thus the native
63-row convolution input includes that token and omits the last electrode output.
C0 reproduces this behavior; C1 explicitly retrieves the 63 electrode outputs
(indices1:64 before the legacy crop) so tying spatial filters has consistent row
identities. This is an architectural bridge change, NOT an effect of sharing.
Retain native known/unknown token policy in the attention input, audit both cases,
and do not silently repair archived scores. Unknown IDs must not change behavior
merely because evaluation chunk size changes.

## 3. Common compatible interface (C1--C7)

C1 keeps attention BEFORE the ATM convolution and no attention in the TS branch.
It makes module shapes compatible without tying any parameters:

- TS convolution widths become12. Both dense backbone outputs become256;
  alignment remains128. Retaining256 rather than shrinking both to128 keeps
  the bridge's total registered EEG-side parameter budget close to the native pair.
- Both have temporal convolution1->12, kernel(1,25), pool(1,51), stride(1,5).
- Both have spatial convolution12->12, kernel(63,1), and projection12->12, 1x1.
- After the convolution/projection, adaptive-average-pool the position dimension
  to8. ATM already has8 positions; TS reduces36->8. Flatten in the SAME
  position-major, then feature-channel order in both paths.
- Both use Linear96->256, GELU/Linear256->256 residual with dropout. LayerNorm
  follows; EEG alignment projector256->128 remains private.
- Convolution BN affine weights and running state are private, except C7.
- ATM's explicit electrode-output extraction is applied here and in C2--C7.

Name the shareable groups:

- T: temporal convolution weight/bias only; pooling and activation are fixed.
- S: spatial convolution AND 1x1 projection weight/bias.
- R: the two dense weight/bias matrices in the readout; not LayerNorm.
- N: convolution BatchNorm affine/running state and readout LayerNorm affine.

C1 is a bridge control, not a supposedly equivalent native pair. Changes in width,
position pooling, flatten order, and token indexing are bundled here. C0->C1
measures their aggregate effect and must NEVER be called a sharing penalty.

Analytic registered EEG-side count including EEG projectors: native416,756 versus
bridge430,388 (+3.27%), excluding image projectors and scalar loss parameters.
This must be verified by instantiated implementation before launch; it does not
guarantee matched FLOPs, useful capacity, branch balance, or accuracy. Both image
projectors remain unchanged. Allocating256 to both readouts avoids an aggressive
unintended tiny-TS compression merely to match the native ATM output width.

Temporal weight tying across C1 branches also ties filters on different axes
(raw time versus learned latent coordinates). A loss from C2 diagnoses that
constraint, not proof that two physical-time feature extractors cannot share.

## 4. Exactly ten configurations

| ID | Configuration | Intervention | Primary comparator / inference |
|---|---|---|---|
| C0 | Native independent tiny pair | Original native architectures, no parameter sharing | Clean reference under the new targets/protocol; coupled versus independently selected checkpoints are two readouts of this one condition |
| C1 | Compatible, fully independent pair | Common interface above, all copies untied | C0: quantify bridge cost BEFORE making sharing claims |
| C2 | Shared temporal filters | C1 + tie T only | C1: can temporal/latent-axis filter coefficients be common? |
| C3 | Shared electrode-pooling block | C1 + tie S only | C1: does electrode aggregation need branch-specific weights? |
| C4 | Shared dense readout | C1 + tie R only | C1: can later EEG mappings be shared while early feature extraction remains private? |
| C5 | Shared entire convolutional block | C1 + tie T and S | C2/C3/C1: do temporal and spatial independence interact? |
| C6 | Shared convolution and dense weights | C1 + tie T, S, R; N and alignment heads remain private | C5/C4: is an ATM-specific attention front plus branch-specific normalization/head sufficient? |
| C7 | Shared normalization too | C6 + tie N | C6: is distribution-specific normalization what protects the two routes? |
| C8 | Compute-sharing topology, untied control | Move ATM-like attention to AFTER two independent raw-EEG temporal stems; details below; all tails remain private | C1: topology-change cost, not a sharing effect; C9's exact untied counterpart |
| C9 | Compute-once temporal stem | C8 with one shared temporal stem evaluated ONCE; branch-specific attention/bypass and tails | C8: causal effect of sharing early activations and computation, not just reusing weights |

Contrast graph:

    C0 --bridge--> C1 --T--> C2 --S--> C5 --R--> C6 --N--> C7
                    |                 ^         ^
                    +--S--> C3 --T----+         |
                    +--R--> C4 ----------------+
                    |
                    +--new ordering--> C8 --shared activation/stem--> C9

All C1--C7 weight ties mean one actual Parameter object/optimizer state, not two
equal initial copies. No EMA or periodic copying. The graph still executes the
shared module twice on different inputs: this mostly reduces parameter/optimizer
storage, NOT arithmetic. C9 alone guarantees a removed duplicate stem forward.

For each fold, report the temporal/spatial interaction contrast
A(C5)-A(C2)-A(C3)+A(C1), in percentage points, alongside each direct difference.
A negative value means the combined tie loses more accuracy than the sum of the
individual ties on this metric scale. Similarly compare
A(C6)-A(C5)-A(C4)+A(C1) for the additional dense tie. These are empirical
interactions, not proof that mechanisms are additive or causally independent.

### C7 normalization semantics

Do not call one BatchNorm object sequentially and let branch execution order
decide running statistics. Evaluate both pre-normalization activations at each
BN boundary, compute branch moments, then use equally weighted pooled moments
for both paths and one running-state update. For means ma,mb and population
variances va,vb, pooled variance is (va+vb)/2 + (ma-mb)^2/4. Fix the running
moment convention and epsilon/momentum in the implementation manifest. Equal
branch weighting avoids the TS36-versus-ATM8 length imbalance dominating moments.
This condition intentionally tests replacing private normalization by shared
normalization, including its statistical behavior; it is not pure affine tying.
LayerNorm has shared affine weights but still computes statistics per sample.

### C8/C9 exact topology and matching

Raw EEG -> temporal conv1->12, kernel25 -> pool51/stride5 -> BN -> ELU gives
[B,12,63,36]. C8 has independent Ta and Tb copies; C9 evaluates a single T.

- TS route: pass the stem activation directly to its spatial/readout tail.
- ATM-like route: preserve63 electrode tokens. Flatten each electrode's12x36
  descriptor (432D), LayerNorm432 -> Linear432->112 -> one Transformer encoder
  layer (4 heads, d_ff112, dropout.25) -> Linear112->432 -> reshape. Add this as
  a residual to the stem activation, scaled by tanh(gamma), initialized gamma=.1.
- No extra subject token in this new residual block; this is NOT the native ATM
  front. That topology difference is identical in C8 and C9.
- Both private tails use C1's spatial/projection/pool8/readout256 interface,
  followed by private alignment128 projectors.
- No dropout before the shared branch point; downstream dropout is branch-private.
- C8's two stem copies start with identical weights/state but independent objects.
  C9 starts with those same tensors. Verify exact initial eval outputs and losses.

This ordering resembles the earlier electrode-attention direction, but now has
an exact untied counterpart. The purpose is to identify the COST OF SHARING that
computation, not claim that attention-after-stem is a new or proven architecture.
C8's descriptor projections can add cost; include them in the budget. C9 is a
single branched model, not necessarily a fully single-path or faster-than-ATM model.

## 5. Matching and training rules

- One replicate initially: masterseed3300, deterministic named submodule/branch
  RNG streams; native ATM may use historical seed4300 as a fixed branch offset,
  not a second independent experimental replicate.
- Common sampler, concept split, batch identity, and exact pairwise-mixed EEG
  tensor across conditions and both branches. Mixup alpha.5, probability1.
- For every potentially tied C1--C7 block, both independent C1 copies start with
  the exact template values used by the tied variant. They are free to diverge
  thereafter. This avoids confounding tying with a different initialization.
- Initialize private modules identically across matched comparisons. Separate
  branch dropout RNG streams; module construction must not perturb data RNG.
- Mean of two grouped multi-positive losses. Keep private-head loss scaling
  identical in every condition; standard AdamW3e-4, wd1e-4, no scheduler.
- Match the existing fixed-temperature/softplus behavior: initial .07,
  temperature not learnable. Image embeddings normalized after projection;
  retain existing EEG training normalization convention, cosine at evaluation.
- FP32 initially. No BF16/MPS speed claims without on-hardware equivalence and
  throughput tests. CPU/data/evaluation overhead can dominate these tiny models.
- Primary sharing intervention uses the natural gradient of the mean loss. It
  does not import Nash weighting or per-layer norm rescaling into the grid.

## 6. Checkpoint coupling is a measured cost

C0 and C1 preserve branch-best weights chosen by each branch's validation loss,
plus the joint checkpoint chosen by mean validation loss. Evaluate both readouts
from the same observed training trajectory; no new runs are needed.

The PRIMARY C1--C9 comparison always uses ONE common epoch, selected by mean
validation loss. Never assemble incompatible best-epoch halves of a shared model.
Report C0 independent-checkpoint versus C0 common-checkpoint difference separately
from C0->C1 bridge difference and C1->shared differences. These costs need not
add linearly across interventions. Private optimizers/checkpoints do not imply
separate mini-batches or different data pipelines in the controlled baseline.

## 7. Fast screen without choosing architectures on outer test subjects

### Phase A: implementation and bridge gate

Before fitting, CPU/GPU tests verify tensor interfaces, true parameter aliasing,
optimizer deduplication, untouched private parameters, both branches contributing
to tied gradients, correct summed-gradient equality, one stem call in C9, and
all state_dict/resume aliases. Check that copied untied and tied models have the
same initial outputs. Audit electrode ordering and native token handling explicitly.

Run C0 and C1 on the first screening fold before filling the grid. If C1 destroys
the native pair's ensemble advantage, stop and revise the bridge; do not attribute
that failure to sharing or continue a large uninformative sweep. Operational
warning threshold: >2pp drop in fused inner-validation accuracy OR loss of more
than half of the positive gain over the best individual branch. This threshold
is a feasibility flag, not a statistical equivalence criterion. C8/C9 require
the analogous topology-gate check before interpreting compute sharing.

### Phase B: three inner-subject screening folds, all ten configs

Predeclare outer slots1,2,3, with inner held-out source subjects2,3,4 respectively.
For each slot, EXCLUDE the outer subject entirely and train on the other eight
non-inner subjects. Hold out the original165 concepts using seed20260822.
The inner subject contributes NO training EEG. Use its held-out-concept EEG for
checkpoint selection and screening diagnostics. This tests an unseen subject,
unlike source-concept validation on the same people used for training.

Use fixed seed3300 200-candidate galleries of unique validation images; partition
the1650 images, pad the last50-query-image panel with150 earlier candidates, and
evaluate every query exactly once. All conditions use identical galleries. Train
only the non-held concepts. Eight subjects/image, effective batch1024 (128 images
x8); final LOSO returns to nine/image and1017. Log actual step counts.

Forty epochs per condition, no validation-driven early elimination of particular
sharing arms. Save best snapshots within the first20 epochs and within40 to
measure ranking stability without separate short runs. All choices are made from
inner-validation metrics and measured resource use. No outer EEG/scores are
loaded in screening. Source validation remains reused for selection, so its best
scores are optimistic development estimates, not final generalization estimates.

Total grid:10 architectures x3 screening folds =30 paired-model jobs, not30
independent single encoders. The initial bridge runs are INCLUDED in that count
if the bridge is unchanged. Revisions require a new version/manifest, not silent
replacement. Do not sweep extra seeds, targets, fusion weights, or optimizers here.

### Phase C: confirmation after the screen

Retain C0, C1, the strongest supported parameter-sharing candidate, and the
compute-sharing candidate C9 only if useful; include C8 whenever C9 is confirmed.
Thus typically3--5 architectures,10 LOSO folds each, masterseed3300. Freeze this
set from INNER validation only. Train nine subjects, original held165 concepts,
max100 epochs, patience20, original mean-head source-concept checkpoint selection.
Evaluate each outer test fold once after selection. Reuse nothing from inner runs
as a warm-start. Report all selected configurations, not just the outer-test winner.

The outer subjects are historically studied, so call this held-out within this
experiment, not a pristine never-before-seen benchmark. One-seed paired LOSO
differences are exploratory; overlapping training sets preclude treating ten
folds as ten fully independent experimental replications. Replicate a boundary
finding with more seeds only after permission to expand beyond the current seed.

## 8. Measurements and interpretation

At each selected validation checkpoint, and at final outer evaluation:

1. Both branch top1/top5, fixed-fusion top1/top5; gain above average AND best branch.
2. Correctness table: both correct, TS only, ATM only, neither; prediction agreement.
3. Oracle top1 = either branch correct (diagnostic ceiling, never deployed).
4. Fusion rescues when both solos fail and fusion losses when at least one succeeds.
5. Correct-versus-hardest-distractor margins and distractor-complementarity bonus
   under exactly the same row-z convention. Preserve full score matrices and keys.
6. Task-gradient cosine, norms, and directional loss changes at candidate shared
   blocks. For untied controls compare homologous gradients in their common
   initialization coordinate system, not as proof of representational equivalence.
   Sample diagnostics at fixed intervals; separate their overhead from main timing.
7. Activation geometry (centered similarity/linear alignment on source data), plus
   BN branch moments; do not fit representation alignment on outer test data.
8. Unique EEG parameters, image-side parameters, reachable versus unused parameters,
   forward/backward operations, peak GPU and host RAM, warmed step time, full epoch
   time, selected/stopping epochs, total GPU-hours and EEG-only inference200 time.

Do not report only "percentage shared": tie locations, unique storage, number of
module invocations and measured wall time are separate axes. Keep branches on the
same GPU for timing, cache identical data, and compare end-to-end training at the
same epoch budget as well as early-stopped total cost.

Interpretation map:

| Observation | Supported interpretation / limitation |
|---|---|
| Both solos fall after one sharing intervention | Capacity/optimization or mismatched feature coordinates may be damaged; not automatically loss of diversity |
| Solos stable but oracle and fused gain fall | Useful error complementarity is reduced |
| Oracle stable but fusion falls | Margin/calibration or score-combination failure; inspect rankings before adding capacity |
| C7 worse than C6 | Branch-specific normalization protects performance under this topology |
| C6 retains performance but runtime barely changes | Parameter independence is unnecessary there; compute independence is still unresolved |
| C9 retains C8 performance and lowers step time | Actual evidence for shareable computation in the redesigned topology |
| C8 already weak | Moving attention, not stem sharing, may be responsible; C9 cannot answer the native sharing question |
| C0 common-epoch already loses substantially | Checkpoint coupling explains part of the apparent joint-model gap |

Screening priorities are the accuracy/compute trade-off and interpretable contrasts,
not the highest of ten inner-validation numbers. A provisional engineering target
is within1pp of the relevant untied baseline with >=15% measured training-step
savings; report uncertainty and full deltas rather than declaring equivalence.
Do not require tiny models themselves to reach40%; ask how much of THEIR pair's
advantage survives, then test a confirmed mechanism at the full model scale.

## 9. Deliverables and budget before any launch

Create isolated experiment code and manifests; do not modify code hashed by the
currently running Nash/IV28 jobs. Each manifest records module graph, tied tensor
names, normalization state ownership, initialization seeds/hashes, data splits,
checkpoint rules, and per-condition hardware measurements. Save resumable model,
optimizer, BN and RNG state every epoch. Source changes must invalidate resumes.

Deliver a10-row accuracy/resource table, the contrast graph above annotated with
paired accuracy deltas, and a correctness-overlap/margin decomposition. These are
more informative than a single accuracy-versus-parameter line plot.

No defensible exact runtime before benchmarking paired forward/backward and
validation. Historical tiny solos take roughly20 seconds/epoch on3080, but adding
two losses, sharing or data changes is not linear. Budget by
30 x40 x measured_pair_epoch_seconds for screening, plus startup/validation/storage.
For example,30--40 seconds per full paired epoch implies10--13.3 GPU-hours for
the30-job screen; this is a scenario, not a measured ETA. Confirmation is separate.
Use existing authorized resources only after a launch request and availability check.
