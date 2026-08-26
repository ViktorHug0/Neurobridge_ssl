# Ensemble-to-50 experiment ledger

## Current objective

Raise the fixed four-model ensemble from **45.35%** to **at least 50.00% mean
top-1** over all ten THINGS-EEG2 cross-subject LOSO folds, using a handful of
members (target range: **2--5**) and at most **four project GPUs at once**.

The new result must use the same base-model evaluation setting as the 45.35%
reference:

1. For held-out subject `h`, train each member on the other nine subjects.
2. Evaluate subject `h` after every epoch and retain the epoch with the lowest
   held-out test loss (`--select_best_on test`; top-1 breaks an exact loss tie).
3. Use `plain_cosine` 200-way retrieval. No SAW, CSLS, Sinkhorn, Procrustes, or
   other transductive adaptation is part of this target.
4. Independently L2-normalize each member's EEG and image embeddings, form its
   `200 x 200` cosine matrix, then fuse score matrices.
5. Router/member development may use labeled data pooled over all ten folds.
   At inference, the frozen router receives only the current query's candidate
   score rows, not subject identity.
6. The final member set, score transform, routing function, and all fusion
   hyperparameters must be **one fixed global rule applied unchanged to all ten
   subjects**. Subject-specific selection or a different rule per subject is
   forbidden.
7. Every newly launched training arm uses **75 epochs**. Runs already executing
   under their original 100-epoch contract on 2026-08-20 are retained as
   historical evidence, but distinct `*_e75` names prevent the two budgets from
   being mixed into one claimed arm.

This is deliberately **not** the stricter eight-train/one-validation/one-test
Stage-A/Stage-B protocol previously described in this directory. Those runs are
retained as diagnostics, but they are not the protocol for the current target.

## Reference result and correction

The 45.35% reference is the fixed quartet:

| member | EEG encoder | image target | solo mean |
|---|---|---|---:|
| `atm_iv` | ATM | InternViT-6B layer 28 | 35.20 |
| `ge100` | TSConv-parameterizable, group SubjectMix, 100 epochs | InternViT-6B layer 28 | 37.05 |
| `tsconv_eva` | TSConv, pairwise SubjectMix | EVA-02 layer 35 | 33.95 |
| `tsconv_vith` | TSConv, pairwise SubjectMix | ViT-H/14 layer 10 | 32.65 |

Uniform score-average top-1 by fold is:

```text
54.0  53.0  37.0  38.5  46.5  37.0  43.5  38.0  47.0  59.0
mean = 45.35
```

Important bookkeeping correction: **45.35% is a fixed-quartet score, not the
output of exhaustive nested-LOFO member selection.** The actual nested-LOFO
four-member estimate in `ensemble_experiments/legacy/scratch_claude/ens45_results.json` is 43.95%. The best
fixed quartet selected on all ten folds is 45.85%, but that uses a different
post-hoc member set. The current work targets an unambiguous >=50.00% ten-fold
score and will state how its member set and fusion rule were chosen.

## Evidence guiding the search

- Image-target diversity is the only lever that has produced a large repeatable
  gain: `atm_iv + tsconv_eva + tsconv_vith` reaches 44.60% with three members.
- Same-target seed ensembles and alternative fusion rules saturate early.
- A focused label-free fusion sweep on the reference quartet moved 45.35% only
  to 45.45%. A subsequent audit over the frozen eight-member focused pool found
  a more useful parameter-free rule: standardize each member's 200 candidate
  scores independently for every query. A small query-local transform family,
  selected on subjects 1/3/5, chose signed power 1.25 followed by equal averaging
  of `atm_iv + ge100 + iv33 + tsconv_bigg + atm_vith`. That frozen rule scores
  **46.64% on the seven untouched subjects and 47.00% over all ten**. Nested
  leave-one-fold-out member/transform selection reaches 46.50%. This removes
  1.65 points of the original gap without using another fold's queries or the
  scored fold's labels, but stronger members must still provide 3.00 points.
- Broad EEG-architecture replacement did not lower score correlation enough to
  improve the ensemble. Architecture diversity is still useful when it
  strengthens a proven member, which motivates the ATM group-training arm.
- Deterministic per-query temporal shift averaging was added as an optional,
  non-transductive evaluator check. The default shift-0 path reproduced the
  original ATM fold-1 dump exactly, but averaging shifts `[-1, 0, 1]` reduced
  that arm from 48.0% to 47.5% and reduced its tested ensemble scores by
  2--2.5 points. The same shifts reduced group ViT-H on every gate fold
  (43.0/20.5/32.5 to 42.0/19.0/32.0) and did not enter any best fixed
  size-3--5 ensemble. Temporal TTA is therefore rejected.
- Two additional label-free routing families were audited on the complete
  ten-fold frozen score pool while the new models trained. Visual-cluster
  reliability estimated from the other nine subjects reached only 47.20% in a
  diagnostic global hyperparameter ranking; selecting its hyperparameters on
  folds 1/3/5 did not improve the seven untouched folds. Per-query model-subset
  selection, top-gap routing, and agreement/centrality weighting likewise did
  not exceed the frozen 47.00% rule. These routes are rejected rather than
  being tuned further on the ten target folds.
- The historical `group_e100` recipe is the strongest solo recipe. New target
  transfers preserve that recipe except for the globally reduced 75-epoch
  budget and therefore use `*_group_e75` arm names.

## Experiment plan

As of 2026-08-21, **all ten subjects are one pooled development set**. Subjects
1--5 are no longer a gate, sanity population, validation population, or
preferred subset. New arms run all ten folds at 75 epochs. A run may be stopped
only for a clear implementation failure or genuinely catastrophic performance;
a slightly weak fold or solo mean is not sufficient because a weak member may
still add useful error diversity.

Every report uses the selected checkpoint and records
solo accuracy, fixed size-2--5 ensemble contribution, unique correct trials,
oracle coverage, correctness/error correlation, double-fault rate, and score
correlation over all ten subjects. Member selection, router fitting, and scoring
may use the pooled all-ten development data. The hard constraint is that the
resulting member pool, fitted router, feature normalization, and scoring code
are identical for every subject. Subject identity cannot be a router input, a
split key, a source of weights, or a switch between scoring rules.

### Gate 1 -- group training on complementary targets (training complete)

Both arms exactly copy the `ge100` recipe: TSConv-parameterizable, backbone 1024,
alignment dimension 512, 100 epochs, batch 1024, grouped multi-positive batches,
nine-source raw-EEG group SubjectMix, alpha 0.5, seed 3300.

| arm | target | purpose |
|---|---|---|
| `eva35_group_e100` | EVA-02 layer 35 | strengthen the EVA member without losing cross-backbone diversity |
| `vith10_group_e100` | ViT-H/14 layer 10 | strengthen the ViT-H member that is essential to the best diverse triple |

- GPU job: `48849` (two array tasks, throttle 2)
- Final minimum-test-loss solo top-1 on folds 1/3/5:

  | arm | fold 1 | fold 3 | fold 5 |
  |---|---:|---:|---:|
  | EVA group | 36.0 | 21.5 | 32.0 |
  | ViT-H group | 43.0 | 20.5 | 32.5 |

- The checkpoints and results completed, but both Slurm tasks ended with exit
  127 in the post-training aggregation command. Consequently, dependent CPU
  job `48851` was never released. This does not invalidate the six completed
  training runs.
- All three ViT-H folds have now been dumped; the missing fold-5 export was
  recovered on CPU job `48881`. In the complete focused pool, the best fixed
  gate ensemble containing group ViT-H scores **48.83%** on folds 1/3/5
  (58.0, 38.0, 50.5), versus 47.83% for the frozen pre-gate rule. This confirms
  promotion. Adding it directly to the old quartet scores 46.50% versus 45.83%
  for that quartet on the same folds. EVA's fold-5 export (`48883`) reproduced
  its selected 32.0% solo result; the complete combined audit still excludes
  EVA from every best size-2--5 gate ensemble, confirming rejection. The exact
  decision is frozen in `ensemble_experiments/test_selected/gate1_promotion.json`.
- The promoted historical ViT-H arm completed all ten subjects at
  **43.0/44.0/20.5/32.0/32.5/30.0/30.0/24.0/26.0/42.5%** (33.45% solo mean).
  Job `48864_1` exited 127 only after writing all results; export job `48869`
  successfully produced all ten score dumps. The five-fold-selected global
  five-member rule containing it scores 48.20% on subjects 1--5 but only
  **45.95% over all ten**. The best fixed all-ten diagnostic remains the frozen
  47.00% rule and excludes this arm. The gate gain therefore did not transfer
  to subjects 6--10, an important negative result against promoting from too
  few folds.

### Gate 2 -- architecture strengthening and joint visual target (partly complete)

| arm | configuration | hypothesis |
|---|---|---|
| `atm_iv_group_e100` | ATM, bb/fd 128, InternViT-28, group SubjectMix, 100 epochs | retain ATM's useful error diversity while improving its training recipe |
| `multibackbone5_group_e100` | TSConv-parameterizable, concatenated InternViT/EVA/ViT-H/BigG/DINO target, group SubjectMix, 100 epochs | let one linear image projector learn a joint target from all five complementary visual spaces |

- GPU job: `48853`, two tasks with throttle 2. ATM--InternViT task 0 is still
  running on fold 5. Its final selected scores so far are **48.0% on fold 1**
  (epoch 89, loss 1.816) and **22.5% on fold 3** (epoch 92, loss 2.852). Fold 3
  is six points below the old ATM member, but the CPU export (`48892`) revealed
  complementary errors: in the combined two-fold pool, adding the new ATM arm
  raises the best Gate-1 ensemble from 48.0% to **49.5%** (59.5/39.5). Fold 1's
  ATM result is a 9.5-point solo gain over the old member. The completed
  fold-1 CPU score export (`48880`) also shows a real ensemble contribution on this
  fold: replacing `atm_iv` in the frozen five raises signed-power fusion from
  57.5% to 58.5%; raw averaging of the replacement set scores 59.5%. This is
  promising gate evidence, not yet a Gate-2 promotion decision.
- Historical ATM fold 5 finished at **36.5%** (minimum loss 2.118, epoch 99).
  The complete fixed three-fold audit on subjects 1/3/5 reaches **49.50%** with
  one five-member rule containing both historical ATM and ViT-H group arms
  (61.0/35.5/52.0). This is useful diversity evidence, but it is not an all-ten
  result and the clean 75-epoch ATM reproduction is now running on subjects
  1--5.
- The five-backbone task 1 has completed all three gate folds. Its selected
  solo scores are **42.5%, 23.0%, and 32.5%** on folds 1/3/5 (mean 32.67%).
  All three score dumps are complete. In the combined Gate-1/Gate-2 audit, this
  arm does not improve the 48.83% fixed Gate-1 ensemble and is absent from every
  best fixed size-2--5 combination. It is therefore rejected; its seven
  remaining folds will not be trained.
- Original postprocessors `48854` and `48870` were superseded after their
  dependency chain failed. CPU scorer `48888` was then prepared, but historical
  ATM's wrapper exited 127 after writing every result, making its `afterok`
  dependency unsatisfiable; it too was cancelled. CPU jobs `48900` and `48901`
  recovered ATM fold 5 and ViT-H folds 2/4 directly without using a GPU.
- The old automatic selector `48889` was cancelled when the protocol changed
  to five-fold sanity checks, 75 epochs, conservative completion, and an
  expanded GPU allowance. It launched no successor.

### Sanity checks and full-fold evaluation

After each gate:

1. Export the selected checkpoint's EEG/image embeddings with `evaluate.py` in
   `plain_cosine` mode.
2. Reproduce each solo score from the dump.
3. Measure `old quartet + candidate` at `k=5`, every one-for-one `k=4`
   replacement, and focused exhaustive fixed combinations for `k=2..5`.
4. Inspect per-fold gains, unique correct trials, oracle coverage, score/error
   correlation, and double-fault rate rather than judging only solo mean.
5. Stop only broken or catastrophic arms under the conservative rule above;
   slightly worse but potentially diverse arms complete all ten folds.
6. Complete a surviving new arm on subjects `6, 7, 8, 9, 10` with the exact
   same 75-epoch configuration used on subjects 1--5.
7. Choose one global 2--5-member set and one fusion rule, then apply both
   unchanged to all ten subjects. Per-subject member or rule selection is not
   permitted.

The historical 100-epoch ViT-H Gate-1 arm and historical ATM--InternViT gate are
complete and remain separate from the new 75-epoch arm family.

The prepared runners are under `ensemble_experiments/test_selected/`:

- `run_group_target_testselected.sh`
- `group_target_gate.sbatch`
- `second_gate.sbatch`
- `group_target_full.sbatch`
- `dump_testselected_arm.py`
- `dump_single_cpu.sbatch`
- `score_target_gate.py`
- `postprocess_group_target_gate.sbatch`
- `postprocess_second_gate.sbatch`
- `priority75_gate.sbatch`
- `priority75_full.sbatch`

- `overnight75_auxiliary.sbatch`

### Completed 50-fold batch

Jobs `48896`, `48897`, `48902`, and `48912`--`48919` completed successfully.
All seven new arms now have ten selected checkpoints and ten plain-cosine score
dumps:

| arm | all-ten solo mean |
|---|---:|
| `iv_vith_dino_aux025_group_e75` | **36.40** |
| `iv25_31_aux025_group_e75` | 35.85 |
| `iv33_group_e75` | 35.65 |
| `atm_iv_group_e75` | 35.55 |
| `atm_vith_group_e75` | 32.80 |
| `bigg27_group_e75` | 32.65 |
| `dino16_group_e75` | 30.90 |

No new arm exceeds `ge100` at 37.05% solo. The best fixed rule found after
looking at all ten target folds is a four-member 47.40% diagnostic, only 0.40
above the frozen 47.00% rule and not an honest new claim. Selecting a rule on
subjects 1--5 reached 49.00% there but only 42.90% on subjects 6--10 (45.95%
overall). This is retained as historical evidence that the old 1--5 gate did
not transfer; the current protocol removes that split and develops one rule on
all ten subjects together.

The important positive result is diversity. The union of individually correct
predictions rises from **66.50% oracle coverage** for the frozen five to
**75.30%** after adding the seven new arms. Fixed averaging cannot identify the
right member per query, motivating a learned router rather than more variants
of the same target-transfer recipe.

## Next direction -- one learned pooled all-ten router

### What “globally fixed” means

The following choices are frozen once in
`ensemble_experiments/test_selected/pooled_router_config.json` and cannot change by subject:

1. member pool and order;
2. row-standardized signed-power-1.25 score transform;
3. router features and neural architecture;
4. optimizer, regularization, epoch count, seed, and uniform-weight floor;
5. the fitting and evaluation algorithm.

All 2,000 labeled queries from subjects 1--10 are pooled to fit **one** router.
That produces one feature normalization and one byte-identical checkpoint,
which is then applied to every subject. Subject ID is retained only after
inference to print the ten accuracy rows; it is never provided to the router.
There are no fold-specific fits, per-subject weights, per-subject thresholds,
or per-subject member subsets.

This is deliberately a pooled, test-selected development protocol: its score
is not an unbiased estimate of router generalization to a new eleventh subject.
That limitation is explicit in the output report. It nevertheless matches the
current objective, whose only routing restriction is one subject-independent
rule over the available ten subjects.

### Frozen pooled five-member routing pool

An exhaustive all-ten scan of available size-five pools gave the following
useful Pareto points before fitting the new pooled router:

| pool summary | uniform top-1 | individual-member oracle |
|---|---:|---:|
| highest oracle | 46.10% | 67.40% |
| selected balance | **47.20%** | **67.05%** |
| historical frozen five | 47.00% | 66.50% |

The selected balance pool is frozen as:

| member | role |
|---|---|
| `atm_iv` | strongest established architecture-diverse anchor |
| `iv33` | established complementary InternViT-depth member |
| `atm_vith_group_e75` | largest unique-correct contribution among the new arms |
| `bigg27_group_e75` | complementary OpenCLIP BigG target |
| `iv_vith_dino_aux025_group_e75` | strongest solo new auxiliary target arm |

The total pool is five members, so every routed prediction remains within the
agreed 2--5-member limit. The pool is never pruned or changed per subject.

### Router inputs and architecture

For each query and member, the router sees only label-free statistics of that
member's 200 candidate scores: maximum, top-1/top-2 margin, top-5 and top-10
means, normalized entropy, row norm, top-1 vote support, similarity to the
member consensus, and two consensus-at-winner scores. Every statistic is
query-local: no other query from the target subject is used.

A shared `10 -> 16 -> 1` MLP scores each member, plus one learned bias per fixed
member. Softmax produces query-dependent weights. The final weights retain a
50% uniform component to reduce the catastrophic confident routing observed in
R0. The objective is aligned more directly with top-1: a smooth margin against
the strongest wrong candidate, plus 0.25 retrieval cross-entropy and a fixed
0.02 KL penalty toward uniform weights. Training is always 300 epochs with
AdamW, seed 3300, and no subject-based validation, early stopping, or
hyperparameter selection.

### Current all-ten protocol

The existing test-selected score dumps provide 200 queries x 200 candidates for
each of the five members and ten subjects. They are transformed identically,
concatenated into one 2,000-query fit, and passed to one router. The fitted state
is evaluated once, then grouped by subject only for reporting. The router is
compared with uniform signed-power fusion of the exact same pool and with the
individual-member oracle. Success remains all-ten arithmetic mean top-1
>=50.00%.

The earlier source-only R0 has finished and is now a historical diagnostic. Its
old pool scored 45.45% uniform, 44.50% routed, and 66.65% oracle. It improved
NLL but reduced top-1 by 0.95 point, showing that cross-entropy alone learned
overconfident weights without solving member selection. The proposed strict R1
inner-refit array is retired under the new pooled protocol and remains
unsubmitted.

### Pooled router result and next submitted comparison

- `pooled_router_config.json`: immutable all-ten pool, transform, loss,
  optimizer, and subject-independence contract.
- `learned_pooled_router.py`: loads the 50 existing test-selected dumps, pools
  all ten subjects, fits one state dict, and reports matched uniform/router/
  oracle accuracy overall and per subject. Subject identity is never an input.
- `pooled_router_fit.sbatch`: CPU launch wrapper. Job `49005` completed in 15
  seconds: **46.80% router**, 47.20% matched uniform, and 67.05% oracle. The
  router helped subjects 6--8 but lost more on subjects 1, 2, 4, and 10, so it
  is rejected.
- `router_config.json`, `learned_global_router.py`, `router_fit.sbatch`, and the
  R0/R1 export utilities remain unchanged solely to reproduce the historical
  source-only result. They are not the current protocol.

The next fixed global test moves from one weight vector per query to one shared
candidate-scoring function. It receives each candidate's five transformed
member scores, five within-row ranks, ten score interactions, consensus
statistics, and vote support. It receives neither subject ID nor candidate ID,
and its single fitted checkpoint is applied to all ten subjects. Both variants
start exactly at uniform fusion through a zero-initialized residual:

| task | scorer | purpose |
|---|---|---|
| `49006_0` | global linear residual | controlled test of score/rank calibration and pairwise agreement |
| `49006_1` | global `25 -> 32 -> 32 -> 1` MLP residual | learn nonlinear agreement patterns needed to approach the member oracle |

Array `49006` completed on 2026-08-21. The linear scorer fell to **43.45%** and
is rejected. The MLP reached **47.30%**, only +0.10 above matched uniform. Both
used the same five-member pool, loss, seed, 400 epochs, and all ten pooled
subjects; only the predeclared scorer capacity differed. Code and frozen
settings are in `learned_pooled_candidate_scorer.py`,
`pooled_candidate_config.json`, and `pooled_candidate_scorer.sbatch`.

Routing has therefore not closed the gap despite using pooled labels. The next
model-side batch uses two full all-ten, 75-epoch arms with group SubjectMix:

| task | arm | rationale |
|---|---|---|
| `49010_0` | `eva35_group_e75` | strengthen the proven complementary EVA target with the best group recipe |
| `49010_1` | `iv_eva_aux025_group_e75` | retain a strong InternViT primary while regularizing its EEG representation toward EVA |

Array `49010` is running with throttle two on the two authorized GPUs. Each
task trains subjects 1--10 in the same loop, exports all ten selected-checkpoint
plain-cosine dumps, and performs no early five-subject gate. The exact commands
are frozen in `next_all10_targets.sbatch`.

## Checklist

### Protocol and baseline

- [x] Trace the exact 45.35% evaluation protocol from configs and scoring code.
- [x] Reproduce 45.35% and its ten per-fold values from stored score dumps.
- [x] Correct the false statement that 45.35% was the nested-LOFO result.
- [x] Establish the no-transductive-adaptation constraint.
- [x] Update the resource limit from two, then three, to four project GPUs.
- [x] Freeze 75 epochs for every newly launched arm; keep historical e100 and
  new e75 result roots distinct.
- [x] Historical: expand the then-current sanity checks to subjects 1--5.
- [x] Forbid subject-specific member selection, weights, transforms, or fusion
  rules; the final rule must be global and identical on all ten subjects.
- [x] Adopt conservative early stopping that preserves weak-but-diverse arms.
- [x] Verify that fusion-only variants cannot plausibly close the gap (45.45%).
- [x] Validate query-local calibration: 46.50% nested LOFO; frozen gate rule 47.00%.

### Historical gate and training ledger

- [x] Confirm no reusable test-selected group-EVA/group-ViT-H ten-fold runs exist.
- [x] Launch Gate 1 on folds 1/3/5 (`48849`).
- [x] Finish Gate-1 fold 1: EVA 36.0%, ViT-H 43.0%.
- [x] Finish Gate-1 folds 3 and 5.
- [x] Export and score Gate-1 folds 1 and 3.
- [x] Re-export group ViT-H fold 5 on CPU and score the complete three-fold gate (`48881`).
- [x] Promote group ViT-H at 48.83% on the gate; queue its seven missing folds (`48864`).
- [x] Export group EVA fold 5 and confirm rejection from the complete gate (`48883`).
- [x] Queue Gate 2 under the two-GPU dependency (`48853`).
- [x] Start both Gate-2 arms on the two available GPUs.
- [x] Finish Gate-2 fold 1: ATM--InternViT 48.0%; five-backbone 42.5%.
- [x] Export ATM--InternViT fold 1 and verify its solo and ensemble scores (`48880`).
- [x] Finish and export five-backbone folds 1/3/5; reject the arm (32.67% solo
  mean and no gain over the 48.83% Gate-1 ensemble).
- [x] Finish and export ATM--InternViT fold 3: 22.5% solo, but a positive
  two-fold ensemble marginal (`48892`).
- [x] Finish ATM--InternViT fold 5 and recover its selected checkpoint dump.
- [x] Replace failed postprocessors with direct CPU exports; cancel the
  dependency-never-satisfied scorer `48888`.
- [x] Export all historical gate arms and score the combined fixed pool.
- [x] Cancel obsolete automatic promotion selector `48889` before it launches
  any successor.
- [x] Start the promoted ViT-H arm's seven remaining folds (`48864_1`).
- [x] Add a configurable epoch budget to the common runner.
- [x] Add five-fold `priority75_gate.sbatch` and matching subjects-6--10
  `priority75_full.sbatch` runners.
- [x] Start ATM--ViT-H e75 on the third GPU (`48896_0`).
- [x] Queue ATM--InternViT e75 behind the historical ATM gate (`48897_1`).
- [x] Queue BigG e75 behind the historical ViT-H export (`48898_2`).
- [x] Finish historical ATM--InternViT fold 5: 36.5%; recover its dump on CPU
  (`48900`).
- [x] Recover historical ViT-H folds 2/4 on CPU (`48901`) and score its complete
  subjects-1--5 fixed ensemble at 48.20%.
- [x] Release BigG e75 onto the fourth GPU (`48898_2`).
- [x] Queue InternViT-33 e75 behind the historical ViT-H export (`48902_3`).
- [x] Finish and export historical ViT-H on all ten subjects; record the
  negative all-ten transfer (45.95% for its five-fold-selected rule).
- [x] Finish and score BigG e75 on subjects 1--5.
- [x] Start DINO e75 on subjects 1--5 (`48912_4`).
- [x] Queue exactly 50 additional 75-epoch folds across four dependency lanes
  (`48913`--`48919`, including the five DINO sanity folds in `48912`).
- [x] Finish and score ATM--ViT-H e75 on all ten subjects.
- [x] Finish and score ATM--InternViT e75 on all ten subjects.
- [x] Finish and score InternViT-33 and DINO e75 on all ten subjects.
- [x] Finish the two auxiliary-target arms on all ten subjects.
- [x] Audit all completed arms: frozen honest rule remains 47.00%; best
  all-fold post-hoc diagnostic is 47.40%.

### Promotion and completion


- [x] Diagnose candidates by ensemble contribution and unique correct coverage,
  not solo score alone.
- [x] Complete every non-catastrophic e75 candidate on subjects 6--10.
- [x] Export all ten selected-checkpoint score matrices for every completed arm.
- [x] Retire the subjects-1--5 gate; use subjects 1--10 as one pooled
  development set.
- [x] Fix one global five-member pool and one subject-independent router
  contract that are applied unchanged to all ten subjects.
- [ ] Report all ten per-fold top-1 values and their arithmetic mean.
- [ ] Verify mean top-1 is **>=50.00%** in the exact reference setting.
- [ ] Record final configs, run roots, member list, and reproduction command here.

### Learned-router track

- [x] Complete historical R0 exports and manifest audit (`48952`).
- [x] Complete the historical per-outer source-only fit (`48953`): 44.50%
  router versus 45.45% matched uniform and 66.65% oracle.
- [x] Diagnose the R0 failure as objective mismatch: NLL improved while top-1
  fell by 0.95 point.
- [x] Retire the strict R1 inner-refit plan under the new pooled-all-ten rule;
  leave its 150-task array unsubmitted.
- [x] Search size-five pools globally over all ten subjects and freeze the
  47.20%-uniform / 67.05%-oracle Pareto pool.
- [x] Freeze one all-ten router config with no subject input, subject-specific
  parameters, member subsets, thresholds, or scoring branches.
- [x] Replace pure cross-entropy with a fixed hard-negative top-1 margin loss,
  retain cross-entropy/KL regularization, and raise the uniform floor to 50%.
- [x] Add the single pooled fitter and prepared CPU job wrapper.
- [x] Run the pooled fit exactly once with the frozen config (`49005`).
- [x] Confirm the report names one checkpoint shared by all ten subjects and
  contains all ten per-subject rows.
- [x] Record the pooled router failure: 46.80%, or -0.40 point versus matched
  uniform despite the top-1-aligned loss.
- [x] Freeze and submit the two global candidate-level scorers on two GPUs
  (`49006_0` linear and `49006_1` MLP).
- [x] Report both candidate scorers: 43.45% linear and 47.30% MLP; neither
  provides a meaningful route toward 50%.
- [x] Submit pure EVA-group and InternViT+EVA-auxiliary training for all ten
  subjects at 75 epochs (`49010_0`, `49010_1`; two-GPU throttle).
- [ ] Finish and export all ten folds for both new target arms.
- [ ] Re-run the all-ten size-2--5 pool audit including both new arms.
- [ ] Reach >=50.00%, or record the new experts' gain and failure mode.

## Completion criterion

This effort is complete only when stored all-fold checkpoints and score dumps
reproduce a 2--5-member, plain-cosine ensemble with arithmetic mean top-1
**>=50.00%** using the same global member/fusion rule on every fold. A strong
single fold, a sanity-check mean, a maximum-over-epoch accuracy, a
subject-specific rule, or a transductive score does not satisfy the objective.
