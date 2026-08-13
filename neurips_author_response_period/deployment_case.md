# Steelmanning the deployment case for SAGE-TTA

Working notes for the 28770 author response. Everything below is anchored to results already on
disk (see agent memory `neurips-2026-rebuttal`), not to hypotheticals.

---

## 0. The crux: SAGE-TTA is a *calibration* procedure, not a *decoding* procedure

This is the single reframe that wins the argument, and the paper currently does not make it.

Both critical reviewers (Rekw Q2, 6S7i Q2/Q5) assume the query set used to **fit** the alignment and
the query being **decoded** are the same object. In the 77% number they are — that is what
"transductive" means. But the *output* of SAGE-TTA is not a set of labels. It is a triple

    (mu, Sigma^{-1/2}, R)   —   a mean, a whitening matrix, a d x d rotation

and none of those depend on which stimulus is being decoded. They are properties of **the subject +
headset + session**, not of the query batch. So they can be estimated once, frozen, and applied to
any future trial, one at a time, against any candidate set.

You have already measured exactly this (Exp A, `run_calibrate_then_deploy.py`, 10 subjects):

| 200-way, disjoint calibration/deployment blocks (fit 40 reps / eval 40 reps) | top-1 |
|---|---|
| no adaptation, per-trial | 31.7 |
| **frozen map, per-trial cosine, zero batch statistics** | **49.9** |
| frozen map + CSLS over the deployment batch | 54.6 |
| fresh full transductive re-fit on the deployment block | 54.4 |

The last two lines are the killer detail: **54.6 ≈ 54.4**. A map fit on a *disjoint* block does as
well as re-fitting on the block being decoded. That is a direct empirical refutation of "the method
is merely exploiting the closed-set test structure" — if it were exploiting the test batch, transfer
of a frozen map would collapse. It does not.

### 0.1 Compare at matched query SNR, or the comparison is meaningless

**Do not put 49.9 and 68.3 in the same column.** Those two numbers use different queries: 68.3 is on
80-repetition averaged queries (the full THINGS-EEG-2 test set), 49.9 is on 40-repetition averages,
because a disjoint calibration/deployment split of an 80-rep test set caps each block at 40. Reading
them as an 18-point "deployment penalty" is wrong, and it understates the method.

At **matched 40-rep SNR**, everything sits within a few points:

| 200-way, 40-rep averaged queries | top-1 | batch access at decode time? |
|---|---|---|
| no adaptation, per-trial | 31.7 | none |
| **frozen map, strictly per-trial cosine** | **49.9** | **none** |
| frozen map + CSLS over the deployment batch | 54.6 | scoring only |
| full transductive re-fit on the block being decoded | 54.4 | fit **and** scoring |

Two independent experiments agree on this, which is worth stating in the rebuttal. The repetition
sweep already in `answers.tex` (Rekw Q3) reports 32.4 zero-shot / 54.9 full TTA at 40 test
repetitions; Exp A independently reports 31.7 / 54.4. Same numbers, different code path.

So the correct decomposition of 49.9 → 68.3 is:

- **≈ +14 points is query SNR** (40 → 80 repetitions), not adaptation regime. It is the same axis
  as the Rekw-Q3 table, and it applies to the baseline too (31.7 → 35.9).
- **≈ +4.5 points is CSLS batch scoring**, recoverable in principle (see §5.1).
- **≈ 0 points is the transductive fit itself.** 54.6 vs 54.4 — the map transfers at no measurable
  loss.

That last line is the claim to make loudly: *what the deployable configuration gives up is a
batch-level hubness correction, not the geometric alignment.* The alignment — the actual
contribution of the paper — costs nothing to freeze.

So the honest table has an explicit SNR column:

| Setting | Fit set | Decode unit | Batch at decode | Reps/query | top-1 (200-way) |
|---|---|---|---|---|---|
| SAGE-zero-shot | — | 1 query | none | 80 | 35.9 |
| SAGE-zero-shot | — | 1 query | none | 40 | 31.7 |
| **SAGE-calibrated (deployable)** | past block, unlabeled | **1 query** | **none** | 40 | **49.9** |
| SAGE-calibrated + batch CSLS | past block, unlabeled | batch | scoring | 40 | 54.6 |
| SAGE-TTA (transductive) | the decoded batch itself | batch | fit + scoring | 40 | 54.4 |
| SAGE-TTA (transductive) | the decoded batch itself | batch | fit + scoring | 80 | 68.3 |

Put that in the paper. It costs half a column and answers 6S7i Q1, Q2 and Q5 at once.

Note the ITR table in §1 is unaffected — its before/after pairs are all within Exp A at fit40/eval40,
so they are already matched.


---

## 1. Ranked deployment scenarios

Ordered by how well the evidence you already have supports them.

### S1. Label-free enrollment for an assistive menu interface — **strongest, fully measured**

A new user (ALS, locked-in, stroke, spinal injury) receives a BCI menu/speller. Standard clinical
workflow already begins with an enrollment block. Present each of the K menu icons in randomised
order; record EEG; fit (mu, Sigma, R) label-free; freeze; deploy.

Why the assumptions are *free* here, not tolerated:
- **Bijection is imposed by the protocol, not discovered in data.** One pass over the menu is
  bijective by construction. The reviewer's objection ("obtaining paired data from a new subject
  is inconsistent with an unseen subject") dissolves: you are not obtaining *labels*, you are
  choosing the *presentation order*, which the system controls anyway.
- **The candidate set is the menu.** Assistive interfaces are finite by design. A closed candidate
  bank is a description of how the UI works, not a modelling concession.
- Cost: your appendix says the whole 10-subject TTA stage runs "in a few seconds on CPU". Per user
  this is milliseconds and a few hundred KB of parameters. It runs on the device.

Numbers you have (frozen map, strictly per-trial, no batch stats):
200-way 31.7 → 49.9 · 50-way 55.0 → 71.5 · 10-way 78.7 → 95.3.

Converted to Wolpaw ITR (bits/selection, at matched trial budget — I computed these from your own
accuracies, please re-derive before using):

| Menu size | before | after | gain |
|---|---|---|---|
| 10-way | 1.90 | 2.90 | **+53%** |
| 50-way | 2.12 | 3.18 | **+50%** |
| 200-way | 1.53 | 2.82 | **+85%** |

ITR is the currency the BCI community actually cares about, and a +50% ITR from a label-free
enrollment block that costs minutes is a serious deployment result. State it in those terms.

### S2. Silent, continuous recalibration against session drift — **the genuinely unique argument**

This is where label-freeness is *load-bearing*, and S1 is where it is not (see §4, honest concession).

EEG is non-stationary: cap displacement, impedance drift, gel drying, fatigue, time of day, and above
all re-donning between sessions. Session-to-session non-stationarity, and the daily supervised
recalibration block it forces on the user, is the textbook barrier to BCI deployment and the most
common usability complaint in the clinical literature.

In a menu-driven interface the system *always* knows the candidate set (it drew the screen) and
*never* knows the user's intent. That is precisely SAGE-TTA's input signature. So: keep a rolling
buffer of the last W selections, periodically re-fit against the on-screen bank, swap the frozen map.
The user is never interrupted, never cued, never asked to confirm anything. **Zero-cost
recalibration.** No supervised alternative can do this, because there are no labels to be had.

This is the scenario to lead with when arguing *significance*, because supervised alignment is not a
competitor here — it is inapplicable.

### S3. Users who cannot be instructed — clinical core population

Supervised enrollment requires the user to attend a cued item reliably *and* requires you to trust
they did. In minimally-conscious, late-stage locked-in, paediatric or severely fatigued users, both
assumptions fail — and that population is the primary clinical target for visual BCIs, not a niche.
Label-free set-level alignment needs only that you know the *set* of things on screen. Free-viewing
or passive-viewing calibration becomes possible.

### S4. Consumer / dry-electrode wearables — every donning is a new subject

Your AllJoined-1.6M results are on a 32-channel consumer headset. Each time a consumer puts on a dry
headset, the spatial mixing changes — that is close to a per-session orthogonal-ish distortion, which
is exactly what R models. You cannot ask a consumer to run a supervised calibration block every
morning; nobody would use the product. But you *can* run a silent re-fit over their first ~40
interactions with a known UI. The right analogy is AirPods spatial-audio ear scan / hearing-aid
fitting / eye-tracker 9-point calibration: an enrollment step nobody objects to because it is
invisible or takes seconds.

### S5. Offline neuroscience analysis — transduction is simply the correct setting

Not every use is real-time. When you have recorded a subject viewing a known stimulus set and want the
best single-trial decode for downstream analysis (RSA, single-trial representational dynamics, dataset
QC, detecting bad sessions via alignment residual), the entire test set *is* genuinely available and
refusing to use it leaves accuracy on the table for no reason. Transduction is not a euphemism for
leakage; it is a well-posed learning setting (Vapnik: don't solve a harder intermediate problem than
the one you have). The 68.3 number is a legitimate, useful number *for this use*. Say so, and scope it
to this use.

### S6. Context-restricted candidate banks / two-stage rerank — the paper's own idea, underdeveloped

Already in your discussion §Limitations, and it is better than you make it sound. Real interfaces are
finite-state machines over menus: spreadsheet → {copy, paste, sort, …}; editor → {brush, erase, zoom};
smart-home → {lights, TV, thermostat, call nurse}. The task context *defines* a small candidate set.
And for open vocabulary, an inductive first pass narrows to a shortlist and calibration runs on the
shortlist — which dissolves vxam's "would probably not scale beyond N=200".

**Caveat, and you must handle it:** your own open-gallery run
(`openset_streaming_20260723-200138`) shows the shortlist has to be tight. With 200 real + D
distractors, TTA gain dies at D=200 and inverts by D=800. Root cause is understood
(`sinkhorn_normalize` forces every column to sum to 1, pushing mass onto distractors) and the fix is
the free/unbalanced marginal you already validated in the partial-coverage run. Do not claim S6
without the rho<1 fix, or a reviewer who reads carefully will find the hole.

---

## 2. Legitimacy: SAGE-TTA is not exotic, it is the retrieval-space version of BCI standard practice

Worth one or two sentences in the rebuttal, because it reframes the whole objection. The BCI
transfer-learning literature's *standard* baselines are unlabeled, batch-computed target-domain
statistics:

- **Riemannian re-centering** (Zanini et al., IEEE TBME 2018): compute the Riemannian mean covariance
  of the new session's *unlabeled* trials, re-center to it.
- **Euclidean Alignment** (He & Wu, IEEE TBME 2020): same idea, Euclidean reference matrix from
  unlabeled target trials.

Both require batch access to unlabeled target data. Both are accepted, deployed, and uncontroversial.
Your ZCA whitening step *is* a re-centering; the novelty is extending it to a cross-modal rotation
against a known candidate bank. Framing it this way turns "unrealistically strong assumption" into
"the assumption the neighbouring field has been making for a decade", which is a much better place to
be. The paper already cites `barachant2010riemannian` and `samek2013transferring`, so this is in scope.

Secondary supports: batch/episodic TTA is an established subfield (TENT, SHOT, T3A); and every
deployed biometric-input system on earth has a per-user enrollment step (Face ID, voice enrollment,
fingerprint, eye-tracker calibration, IMU figure-8). Demanding a BCI work with *zero* per-user
calibration is a standard applied to no other sensor-to-intent system.

---

## 3. Where the many-to-one / non-bijective structure is the *realistic* one

You asked this in `my_answers.md`. Clean taxonomy, both halves of which you have measured:

| Phase | Structure | Right marginal | Evidence |
|---|---|---|---|
| Enrollment block | bijective by construction (one pass over menu) | balanced Sinkhorn (rho=1) | Exp A |
| Usage stream | many-to-one (user picks "copy" 50×, "sort" once) + unselected items | **unbalanced / free (rho<1)** | Exp B, partial-coverage |

Exp B: balanced R-to-1 beats matched-SNR single-block at every R (R=2 62.9 vs 55.2; R=4 50.4 vs 39.2;
R=8 33.4 vs 25.5) — redundancy in the stream *helps*, it is not a violation.
Partial coverage: free-marginal rho=0 beats balanced across the whole partial regime and crosses the
plain baseline at U≈75/200 coverage vs U≈100 for balanced — **relaxing the marginal widens the
deployable range**. One knob spans imposed-bijection → sparse-partial.

This is the direct, decisive answer to 6S7i Q5 ("show it is not merely exploiting closed-set
structure"). You have it. Foreground it.

Also note: your existing Fig. 5a (full-set regime, N queries vs 200 candidates) is *already*
non-bijective for N<200 and already in the paper. Relabel it as partial-candidate evidence — that
reframing costs nothing.

---

## 4. What to concede, explicitly and first

6S7i says the score *drops* if you keep equating transductive calibration with new-subject
generalization. So the strongest version of this argument **leads with the concession**:

1. The 77% (and 68.3) figures are transductive. They are not deployment numbers and should not be
   read as per-trial new-subject decoding. Fix the abstract line "bringing neural decoding to a
   regime where concrete deployment becomes conceivable" — it is doing the exact thing the reviewer
   threatened to penalise. Attach deployment language to the calibrated numbers, not to 77.
   But concede the *regime*, not the *magnitude*: per §0.1, at matched query SNR the deployable
   configuration is within a few points of the transductive one, and the map itself transfers at no
   loss. The concession is "batch access buys you a hubness correction and nothing else", which is
   much narrower — and much more defensible — than "the transductive number does not transfer".
2. In the fixed-menu enrollment scenario (S1), labels are actually *available* for free — you
   presented the stimuli. So the label-free property is not the selling point there; supervised
   Procrustes is an upper bound, not a competitor. Say this yourself before a reviewer does. The
   label-free property is load-bearing in S2/S3 only.
3. Hard floors you have measured, state them:
   - calibration needs ≈20–40 repetitions per item (5 reps is actively *harmful*: 19.5 < 31.7);
   - candidate coverage below ≈40% of the bank makes all TTA variants worse than no adaptation;
   - open galleries with many distractors break balanced Sinkhorn outright;
   - single raw-trial deployment is weak in absolute terms (6.5 vs 4.5 at 200-way) — deployment needs
     trial averaging, which is exactly what P300/SSVEP spellers already do (5–15 flash sequences per
     selection), so this is protocol-consistent, but say it rather than hiding it.
4. Nothing here helps the true cold-start, open-vocabulary, first-trial-of-session case. That is
   SAGE-zero-shot's job, and 35.9 vs prior 21.8 is a real inductive result that stands on its own.

Conceding 1–4 explicitly *is* what buys you the right to assert S1–S6. This reviewer rewards
delimitation more than he rewards numbers.

---

## 5. Highest-value remaining work, ranked by rebuttal impact per hour

### 5.1 Ranked

1. **Val-select the inductive rotation** (`run_inductive_rotation_sweep.py`, 62.15 top-1 per-trial
   pure cosine, no CSLS/Sinkhorn, 80-rep queries). Memory flags alpha was picked on test = oracle.
   If a val-selected re-run holds anywhere near 62, that is a *genuine per-trial deployment number at
   the same SNR as the headline 68.3* — i.e. the whole deployment argument collapses to a ~6-point
   gap on a matched comparison. Far more persuasive than anything else here. Highest value by a
   distance.
2. **Per-trial CSLS with a frozen candidate-side term** — see §5.2. Small change, plausibly recovers
   most of the 49.9 → 54.6 gap with strictly no batch access.
3. **Re-run the open-gallery arm with rho<1** (free/unbalanced marginal). Converts a known failure
   into "we identified the cause and fixed it", which reads as strength, not weakness. Cheap: CPU.
4. **The §0.1 SNR-matched table + ITR conversion.** Zero compute, pure writing, answers three
   questions at once.
5. Nothing else. The partial-coverage, many-to-one and FM experiments are done.

### 5.2 Per-trial CSLS is available and you are not using it

`csls_scores` (`module/util.py:89`) has two correction terms, and only one of them is batch-dependent:

```python
rx = np.partition(similarities, kth=n_c - k_eff, axis=1)[:, -k_eff:].mean(axis=1, keepdims=True)  # axis=1: over CANDIDATES
ry = np.partition(similarities, kth=n_q - k_eff, axis=0)[-k_eff:, :].mean(axis=0, keepdims=True)  # axis=0: over QUERIES
```

- `rx` is per-query, taken over the **fixed candidate gallery**. At deployment the gallery is known
  and static, so `rx` is computable for a single trial in isolation. No batch needed. Ever.
- `ry` is per-candidate, taken over the query set. This is the *only* thing that requires a batch —
  and it is just a K-vector, one scalar per gallery item, that describes how hubby each candidate is
  for this subject and this gallery. That is a calibration-time property, exactly like `mu`, `Sigma`
  and `R`.

So: estimate `ry` on the calibration block alongside the map, freeze it, ship it as part of the
subject profile. Decoding stays strictly one trial at a time. If this recovers the gap, the
deployable configuration matches the transductive one at matched SNR and the "closed-set exploitation"
objection has nothing left to stand on.

Cheap to test — it is a scoring-side change on cached embeddings, CPU seconds, reusing the Exp A
fit40/eval40 split. Target to beat: 49.9 (per-trial, no CSLS) toward 54.6 (batch CSLS).
Worth checking whether a frozen `ry` from the calibration block is stable enough to help at all — if
the hubness structure is query-distribution-dependent rather than gallery-dependent, it will not
transfer, and that is itself a publishable negative detail.

---

## 6. Placement, given your character budget

Memory says per-review usage is Rekw ≈8.5k / 6S7i ≈9.1k / vxam ≈6.2k against a 10k limit. So:

- **6S7i has ~900 chars.** He is the reviewer who cares most about this, and there is no room. You
  will have to cut his Q4 or Q1 answer to make space for the three-setting table in §0. Do it — the
  table is worth more than anything currently in that response.
- **vxam has ~3.8k spare** and his weakness #4 is precisely "limits its significance for downstream
  BCI deployment", even though he asked no question about it. He is your accept vote. Spend the spare
  budget there on S1 + S2 + the ITR numbers; reinforcing an accept is cheaper than converting a reject.
- **Rekw Q2** gets the one-line version: bijection is imposed by protocol, plus a pointer.
