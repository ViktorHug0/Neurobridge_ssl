# Why score ensembling is the strongest inductive path so far

## Scope and bottom line

This is a read-only audit of the repository as it stood on 2026-08-23. No model was
trained, no checkpoint was re-evaluated, and no new experimental result was generated.
All new calculations reconstruct cosine matrices from score dumps already present under
`results/things_eeg/synthetic_subjects/ensemble_screen/dumps/`. The calculations are
reproducible with:

```bash
source .venv/bin/activate
python ensemble_experiments/analysis/ensemble_evidence_audit.py
```

The central conclusion is:

> Ensembling works here because independently trained EEG-to-vision decoders are noisy
> measurements of a shared latent stimulus signal, while different EEG inductive biases
> and different frozen visual geometries make their *wrong-candidate residuals* less
> correlated. Score averaging preserves shared evidence for the true image and cancels
> candidate-specific errors. The useful diversity is not seed noise and is not model
> count by itself; it is diversity in both the EEG mapping and the visual target kernel.

This explanation is supported by four independent findings:

1. Seed-only pairs have mean score correlation `0.956` and gain only `1.58` points over
   the better member. Pairs that change both EEG encoder and image target have correlation
   `0.814` and gain `5.17` points on average.
2. The original diverse triple reaches `44.60%` versus `37.20%` for a three-seed
   committee of similarly strong models. The shrunk-width controls already in the repo
   independently show `42.25%` for cross-target diversity versus `34.65%` for seeds.
3. The current simple four-member score committee reaches `48.20%`; removing any one
   member costs `2.55` to `3.45` points even though its weakest member is only `29.80%`
   alone. This is a committee effect, not a disguised single-model improvement.
4. Learned routing, candidate rescoring, temporal TTA, same-target seed averaging, and
   single-model multi-target fusion all fail to reproduce the gain. They either cannot
   identify the correct expert or collapse the independent evidence that averaging needs.

The important qualification is equally strong: there is not yet an unbiased `48%`
inductive result. The members and checkpoints behind the highest figures were developed
against these ten test folds, and most checkpoints are selected using held-out test loss.
The best all-fold complex rule is `48.65%`, but its nested leave-one-fold-out estimate is
only `44.35%`. The new simple `48.20%` rule is much more stable within its declared pool,
but the pool and Squeezeformer arm were still developed using these subjects. The result
is compelling mechanism evidence and a strong engineering result, not a clean estimate
for an unseen eleventh subject.

## Protocol taxonomy: numbers that must not be conflated

All rows below are inductive, plain-cosine 200-way retrieval: no SAW, CSLS, Sinkhorn,
Procrustes, or other set-level test adaptation. They differ in checkpoint and member/rule
selection.

| Result | Top-1 | What was selected where | Interpretation |
|---|---:|---|---|
| Best established solo, `ge100` | 37.05 | checkpoint by held-out test loss | engineering baseline |
| Same-recipe seed committee (`p3300/1/2`) | 37.20 | test-selected checkpoints | correlated-clone control |
| Diverse triple (`atm_iv + tsconv_eva + tsconv_vith`) | 44.60 | test-selected checkpoints; stable in the original nested scan | first robust diversity result |
| Reference quartet | 45.35 | fixed quartet, raw score mean | historical reference, not nested-LOFO |
| Frozen five, row-z + signed power 1.25 | 47.00 | member/transform choice on subjects 1/3/5; 46.64 on seven untouched subjects | strongest predeclared holdout check in this track |
| Complex five-member rule | 48.65 | selected and tuned on all ten folds | optimistic post-hoc diagnostic; nested-LOFO 44.35 |
| New Squeezeformer four, row-z mean | **48.20** | best fixed k=4 in a declared 10-arm pool; same set wins all ten nested-LOFO selections | strongest simple rule in the audited focused pool, but pool/model development is not external |
| Concept-validation pool, best raw k=4 | 43.10 | checkpoints selected on held-out training concepts; members selected post hoc | 42.35 nested-LOFO member selection |
| Matched test-selected controls, best raw k=4 | 46.20 | same training trajectories, checkpoints selected by test loss | 45.55 nested-LOFO member selection |

The complex-rule figures come directly from
`results/things_eeg/ensemble50_testselected/rule_final_ledger16.json`. The frozen rule and
its selection split are recorded in
`ensemble_experiments/test_selected/frozen_fusion_baseline.json`. The concept-validation runners save
both a validation-selected checkpoint and a test-loss-selected control from the same
trajectory; see `ensemble_experiments/validation/run_concept_val.sh` and the checkpoint logic
in `train.py`.

### Checkpoint-selection effect

Across ten matched InternViT-depth arms (five ATM, five TSConv), changing only the saved
checkpoint from source-concept-validation selection to held-out-test-loss selection adds
`2.235` points on average. Individual arm deltas range from `+0.25` to `+4.05` points.
Because both checkpoints come from the same run, this is not explained by a different
training set, seed, optimizer, or epoch budget.

For fixed member sets, the effect remains visible:

| Fixed committee | Concept-val checkpoint | Matched test-selected checkpoint | Difference |
|---|---:|---:|---:|
| ATM-28 + TSConv-28, signed-power mean | 39.80 | 42.45 | +2.65 |
| ATM-33 + TSConv-28, signed-power mean | 41.35 | 43.05 | +1.70 |
| Four fixed depth/encoder members | 43.40 | 45.75 | +2.35 |

Thus ensembling itself is real—the honest ATM-33 + TSConv-28 pair beats the strongest
concept-validation anchor by `+7.95` points on average, on all ten subjects (fold-level
95% t interval `[+5.07, +10.83]`, Wilcoxon `p=0.00195`)—but test selection raises the
absolute level. Neither the interval nor p-value should be read as confirmatory after
post-hoc member choice; it quantifies the consistency of the observed conditional effect.

## Mechanism

### A multi-view error model

For query `q`, candidate `c`, and expert `m`, a useful abstraction is

\[
s_m(q,c) = a_m(q) + b_m(q)\,[g(q,c) + \epsilon_m(q,c)].
\]

`g` is stimulus evidence that is stable across representations. `a_m` and `b_m` are
query/model-specific location and scale. `epsilon_m` contains EEG estimation error and
the expert's visual-target-specific candidate bias. Per-query row standardization removes
`a_m` and `b_m`; averaging then retains `g` while reducing `epsilon` only to the extent
that the residuals are not shared.

This is also a log-linear opinion pool interpretation. If each standardized similarity
row is a monotone proxy for log likelihood, adding rows approximately multiplies expert
evidence. The true image often receives moderate support from several experts, whereas a
wrong candidate that is spuriously high in one visual space is less likely to be high in
all of them.

The visual target is not merely a label encoding. Each frozen backbone/layer defines a
candidate kernel

\[
K_m(c,c') = v_m(c)^\top v_m(c'),
\]

and therefore a different neighborhood structure over the 200 images. The EEG encoder
defines a second source of error through the learned mapping `EEG -> v_m`. Changing the
visual target changes which distractors are close; changing the EEG architecture changes
which neural features and nuisance variations are retained. Changing both axes is most
effective because the two sources of correlated error are attacked simultaneously.

### Controlled diversity evidence

The table aggregates pairs from comparable stored sweeps. Score correlation is Pearson
correlation between per-query-standardized 200-way matrices over all ten subjects;
correctness correlation is between the two 2,000-element correct/incorrect vectors.
Pair gain uses raw cosine averaging and is measured against the better solo member.

| Pair family | n | Score corr. | Correctness corr. | Mean pair gain |
|---|---:|---:|---:|---:|
| Same encoder + target, seed only | 3 | **0.956** | **0.653** | **+1.58** |
| Same InternViT-28 target, encoder differs | 10 | 0.869 | 0.402 | +1.43 |
| Same ATM encoder, target differs | 6 | 0.893 | 0.470 | +2.70 |
| Same TSConv encoder, target differs | 15 | 0.899 | 0.458 | +2.76 |
| **Encoder and target both differ** | 23 | **0.814** | **0.339** | **+5.17** |
| Same ATM + InternViT family, depth differs | 10 | 0.915 | 0.542 | +2.60 |
| Same TSConv + InternViT family, depth differs | 10 | 0.918 | 0.497 | +3.20 |

The architecture-only mean includes weak EEGNet/EEGProject/Conformer members and is
therefore not evidence that architecture never helps. Its useful cases are instructive:
ATM + TSConv on the same InternViT target reaches `40.20%` (`+4.10` over the better
member, score correlation `0.851`). The newer Squeezeformer is another successful case.
The correct conclusion is that architecture diversity must retain sufficient solo signal;
arbitrary architectural novelty is not useful by itself.

Score correlation contains shared true signal as well as error, so it is not a direct
estimate of residual covariance. It is used here as a relative diagnostic. Correctness
correlation, oracle coverage, unique-correct counts, and removal marginals independently
lead to the same ordering.

## Anatomy of the current simple 48.20% committee

The committee is a row-standardized arithmetic mean of:

| Member | Solo dump top-1 | Complementary axis |
|---|---:|---|
| `atm_vith` | 29.80 | ATM + ViT-H target; weak but unusually different |
| `atm_iv_group_e75` | 35.55 | ATM + InternViT-28 + group SubjectMix |
| `iv33_group_e75` | 35.65 | TSConv-parameterizable + InternViT-33 |
| `sqf28` | 36.65 | TSConv/Squeezeformer temporal model + InternViT-28 |

Its fold scores are:

```text
59.5 57.0 36.5 40.0 47.0 42.0 48.0 34.5 52.5 65.0
mean = 48.20
```

The standalone Squeezeformer summary rounds to `36.70`; the exported score dumps used
for fusion reproduce `36.65` because subject 8 is `25.0` in standalone `evaluate.py`
rather than the `25.5` logged during training. Ensemble claims in this audit consistently
use the dumped matrices, not a mixture of training and standalone metrics.

Within the declared ten-arm focused pool, this exact four-member set is selected on all
ten leave-one-subject-out nine-fold subsets, so the pool-conditioned nested estimate is
also `48.20%`. This stability is much stronger than the global complex-rule search, but it
does not undo prior reuse of the folds to develop the pool and architecture.

### Every member is doing work

| Removed member | Three-member accuracy | Drop from 48.20 |
|---|---:|---:|
| `atm_vith` | 45.30 | 2.90 |
| `atm_iv_group_e75` | 45.65 | 2.55 |
| `iv33_group_e75` | 44.75 | 3.45 |
| `sqf28` | 44.95 | 3.25 |

The full committee beats `ge100` by `+11.15` points, improves all ten subjects, and has a
fold-level 95% t interval of `[+7.36, +14.94]` with Wilcoxon `p=0.00195`. Again, this is a
conditional consistency calculation, not a post-selection confirmatory test.

The weakest solo member, `atm_vith`, is not dead weight. It has 104 trials that no other
committee member gets right and its two most useful pairings have the lowest score
correlations in the committee: `0.774` with `iv33_group_e75` and `0.804` with `sqf28`.

### What averaging fixes

The individual-member oracle is `62.65%`, leaving a `14.45`-point gap between realizable
coverage and fixed fusion. Conditional behavior shows classic evidence aggregation:

| Number of individually correct members | Trials | Ensemble accuracy |
|---:|---:|---:|
| 0 | 747 | 5.0% |
| 1 | 429 | 37.8% |
| 2 | 351 | 83.2% |
| 3 | 270 | 100.0% |
| 4 | 203 | 100.0% |

The ensemble also recovers 37 trials on which no expert places the truth at rank one,
showing that it combines near-miss rank evidence rather than merely majority-voting the
individual winners.

For comparison, the three-seed committee has only `46.80%` oracle coverage, and its score
correlation is `0.956`. It reaches `37.20%`, just `+2.15` over seed 3300; the paired
fold-level interval is `[-0.03, +4.33]`. The diverse four has more useful disagreement and
is much better at turning it into a common ranking.

## Why learned routing has not worked

The repo contains a progressively stronger set of negative controls:

| Method | Matched uniform | Learned result | Oracle |
|---|---:|---:|---:|
| Source-only R0 router | 45.45 | 44.50 | 66.65 |
| One pooled all-ten router | 47.20 | 46.80 | 67.05 |
| Pooled linear candidate residual | 47.20 | 43.45 | 67.05 |
| Pooled MLP candidate residual | 47.20 | 47.30 | 67.05 |

The failures are informative. The oracle says different experts are right, but a router
must know *which* expert is right before seeing the label. Query-local confidence is only
moderately informative: over the pooled-router members, AUC for correctness is `0.675`
for top-1 margin and `0.699` for top-1-versus-top-5 gap. Choosing the maximum-confidence
expert gives only `42.20–42.45%`, below raw uniform fusion at `45.65%`. For the new four,
the same rules give `42.15–42.30%`, below raw uniform `46.95%` and row-z uniform `48.20%`.

There are three reasons:

1. **Confident errors are representation-specific.** A large within-model gap means the
   model is internally decisive, not that its visual neighborhood is correct for this EEG
   query.
2. **Cross-expert confidence is not calibrated.** Different projectors, dimensions,
   target kernels, and temperatures make “high confidence” incomparable even after simple
   row normalization.
3. **The effective sample size is small.** There are 2,000 queries, not 400,000 independent
   query-candidate examples. Candidate rows and subjects are highly dependent, while the
   top-1 decision is discontinuous. A flexible scorer can improve its surrogate loss or a
   few subjects and still exchange correct decisions for wrong ones elsewhere.

The pooled router's subject pattern confirms the last point: it improves subjects 6–8 but
loses more on 1, 2, 4, and 10. Subject identity is correctly excluded, yet the score-row
features do not contain a subject-invariant expert-reliability signal.

## When ensembling works

1. **Experts share the task but not the dominant error.** The useful region in these dumps
   is roughly score correlation `<=0.85` and correctness correlation around `0.25–0.40`,
   provided both models retain meaningful signal. These are diagnostics, not universal
   thresholds.
2. **Diversity crosses causal axes.** Different target kernels change distractors;
   different EEG architectures/training recipes change estimation error. Changing both is
   substantially better than changing only seed, layer, or architecture.
3. **Scores are combined before argmax.** This preserves rank-two/rank-three evidence and
   can recover items no individual model wins. Winner voting throws this away.
4. **Scale is controlled.** Raw averaging works when member score scales happen to match;
   row-z or a mild signed power is safer for heterogeneous experts. Stronger transform and
   weight searches overfit the ten folds.
5. **The committee stays small and every member has a positive marginal.** The current
   useful range is 3–5. Unique-correct coverage is necessary but not sufficient; the score
   geometry must let fixed fusion capture it.
6. **The candidate set and query ordering are shared.** Score fusion is natural in this
   fixed 200-way retrieval setting because every expert evaluates the same hypotheses.

## When it does not

1. **Seed clones:** correlation `0.956`; three seeds add only `2.15` points over one seed.
2. **Weak novelty without compatible signal:** broad orthogonal-architecture sweeps often
   lowered solo quality more than they lowered correlation. Several architecture-only
   pairs are worse than their best member.
3. **Adding members for oracle coverage alone:** adding seven new arms raised union
   coverage from `66.50%` to `75.30%`, yet fixed fusion stayed around `47%` and routing
   failed. Coverage that cannot be recognized from scores is not deployable accuracy.
4. **Collapsing targets inside one model:** the five-backbone target arm scored `32.67%`
   and entered no best gate committee. Auxiliary multi-target arms improved solo accuracy
   but did not reproduce the independent-expert gain. One bottleneck/projector forces a
   compromise representation and removes separate likelihoods.
5. **Test-time temporal averaging:** shifts `[-1,0,1]` reduced ATM fold 1 from `48.0` to
   `47.5` and reduced tested ensemble scores by `2–2.5` points; ViT-H declined on every
   gate fold. The ERP is time-locked enough that shifts blur signal rather than average
   nuisance.
6. **Small-fold promotion:** the ViT-H group arm improved the 1/3/5 gate to `48.83%`, but
   the five-fold-selected rule reached only `42.90%` on subjects 6–10 (`45.95%` overall).
   Subject heterogeneity makes three- or five-fold gates unreliable for small gains.
7. **Complex fusion search:** `48.65%` all-fold versus `44.35%` nested-LOFO is direct
   evidence that transform/weight/member degrees of freedom fit fold idiosyncrasies.
8. **Test-selected checkpoints:** matched controls inflate arms by `2.235` points on
   average and fixed committees by roughly `1.7–2.7` points in the tested comparisons.

## Recommended path from here

### 1. Freeze the scientific claim before optimizing again

The ten folds have been used repeatedly for architecture, target, member, transform, and
router decisions. No resampling calculation can make them fresh again. Maintain two
explicit leaderboards:

- **engineering/test-selected:** focused-pool simple result `48.20%`; post-hoc diagnostic
  ceiling `48.65%`;
- **source-selected:** concept-validation checkpoint results, with member/rule selection
  labeled post hoc unless it was frozen independently.

Do not present either `48.20` or `48.65` as an unbiased held-out-subject estimate. A clean
final claim requires a predeclared rule on fresh subjects/dataset or a protocol whose
outer tests are not revisited. Within the current repository, the strongest defensible
claim is the *mechanistic* one: diversity gives a large, fold-consistent conditional gain.

### 2. Search for experts, not nominally different models

Future candidates should be screened on a Pareto criterion, not solo top-1:

1. source-validation solo accuracy high enough to contribute;
2. low standardized score and correctness correlation with the frozen committee;
3. positive fixed-fusion leave-one-out marginal;
4. unique-correct trials that fixed score averaging actually recovers;
5. stability across subjects rather than a large mean driven by subjects 1/2/10.

The best-supported design direction is to cross one useful EEG inductive bias
(ATM, TSConv-parameterizable, Squeezeformer) with a genuinely different visual target
(InternViT depth, ViT-H, EVA, BigG), rather than adding seeds or many close InternViT
depths. The new four is a concrete template, not a confirmed final set.

### 3. Preserve separate evidence paths

If compute or deployment cost motivates a unified model, share only an early EEG stem.
Keep target-specific encoder tails/projectors and separate score heads through inference,
then fuse their calibrated rows. The failed concatenated/multi-target arms argue against
forcing all targets through one alignment vector. A diversity regularizer should act on
wrong-candidate margins or residual score rows, not merely parameter distance.

### 4. Use a deliberately boring fusion rule

Predeclare per-query row-z followed by uniform averaging. Signed power near 1 can be kept
only if selected entirely on source validation. Do not spend more test-fold degrees of
freedom on per-member temperatures, truncations, confidence exponents, candidate biases,
or subject-conditioned branches: the existing searches show at most a few tenths in-sample
and materially worse nested transfer.

### 5. Revisit routing only with new reliability information

Score-row statistics alone are insufficient. A future gate needs variables tied to EEG
measurement quality rather than model decisiveness: repetition-level variance, ERP SNR,
artifact/channel quality, or uncertainty across independently averaged repetitions. Such
features are not present in the current averaged score dumps, so no existing artifact can
validate this idea. Any router must be fitted out-of-fold and compared with the exact same
uniform pool.

### 6. Make stopping rules marginal and protocol-aware

For each predeclared candidate, report:

- ten subject scores and arithmetic mean;
- source-selected checkpoint protocol;
- solo, uniform committee, and oracle on the same dumps;
- score/correctness correlation and double-fault rate;
- unique correct trials and drop-one-member delta;
- paired fold differences with uncertainty;
- the complete history of member/rule selection.

Reject a member when its fixed-rule drop-one marginal is non-positive, even if it raises
oracle coverage. Require gains larger than the observed protocol noise: a sub-point gate
gain is not persuasive when checkpoint selection alone moves results by about two points
and a 1/3/5 gate previously failed to transfer by `6.1` points.

## Claims supported by the evidence

The repository supports the following claims:

- Score ensembling is the most successful **inductive engineering direction** so far,
  raising a `37.05%` solo baseline to `45–48%` without transductive test adaptation.
- The gain is driven primarily by complementary visual-target and EEG-encoder error
  geometry, not random seeds or raw parameter count.
- A weak solo model can be valuable when its errors are complementary; solo ranking is an
  invalid ensemble selection rule.
- Uniform calibrated score fusion is currently more reliable than learned query routing.
- The mechanism survives source-concept-validation checkpoint selection, although the
  absolute score drops materially.

The repository does **not** yet support these stronger claims:

- an unbiased `48%+` result on a never-used held-out subject;
- a successful learned router that approaches the member oracle;
- superiority of a single multi-target model over separate experts;
- equivalence between this inductive ensemble track and the separate `~71%` transductive
  SATTC family.

That distinction is the most important practical conclusion. The ensemble direction is
real and theoretically coherent. The next bottleneck is no longer finding another clever
fusion rule; it is producing strong experts whose errors are independently useful, then
validating one frozen committee without consuming the evaluation folds during design.
