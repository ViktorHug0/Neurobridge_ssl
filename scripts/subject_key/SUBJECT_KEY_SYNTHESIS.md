# Subject keys in cross-subject EEG-to-image decoding: a synthesis

*Status: working synthesis, 19 November 2026. Sources: `scripts/mi/MI_LOG.md` (mechanistic experiments, 10 November)
and `scripts/subject_key/KEY_LOG.md` (follow-up, 17 to 19 November). Every number below is in one of the two logs.*

## Summary

We study why EEG-to-image decoders trained on several people lose about half of their accuracy on a person they have
never seen. Opening the models gave one finding that organises everything else in this document.

- **The key mechanism.** Write a trial as `x = m_s + c`, where `m_s` is the subject's mean evoked response over all
  images (it carries no image information) and `c` is what remains. Every encoder we tested decodes `c`
  *conditionally on* `m_s`. We call `m_s` the subject's **key**. Give a trained subject another subject's key and
  accuracy falls by 10 to 55 points depending on the architecture, although the image-specific signal is untouched.
- **Misgeneralisation to a new subject.** A new subject arrives with a key the model has never seen. The model still
  produces an informative embedding, but the cloud of embeddings is displaced and turned relative to the image
  embeddings. A large part of the displacement is one constant vector per subject, and that vector is a function of
  the key.
- **Four practical results on ATM**, in increasing order of what they need from the new subject:

| Method | Needs from the new subject | Top-1, one head | Top-1, three heads |
|---|---|---|---|
| Plain model | the query | 32.2 | 35.5 |
| Key-swap training | the query | no reliable gain (three folds) | – |
| In-trial simulated re-centring | the query, in a periodic stimulus stream | 45.5 ᵃ | 48.5 ᵃ |
| Real re-centring | about 20 to 50 unlabelled trials | 48.6 | 52.6 |

200-way retrieval, leave-one-subject-out, mean over 10 held-out subjects and 3 seeds, except ᵃ (one seed). The best
published number in the first regime is 35.3.

**Epistemic status.** We are confident in the key dependence itself (nine architectures, ten folds, large effects) and
in the real re-centring and three-head gains (three seeds, consistent across subjects). The in-trial simulated
re-centring is consistent across the ten subjects, but its three-head number rests on one seed. The localisation
claims come from one seed and from checkpoints selected on the test subject, which is acceptable for a diagnostic and
would need repeating for a paper. The key-swap training result is a three-fold screen. Everything is on one dataset
with ten subjects, and all gains concern queries averaged over many repetitions.

## Setting

**Data.** THINGS-EEG2: ten subjects, 63 electrodes, images shown in a rapid serial stream at 5 Hz (100 ms image, 100 ms
blank). An epoch is the first second after an image onset, sampled at 250 Hz, so it also contains the responses to the
four images that follow. Training uses 16,540 images with 4 repetitions each; the test set has 200 images of unseen
concepts with 80 repetitions each.

**Task.** An EEG encoder is trained with a contrastive loss to map an epoch to the embedding of the viewed image in a
frozen vision model. At test time a query is the average of the 80 repetitions of one test image, and it is ranked
against the 200 test images by cosine similarity. We report top-1 accuracy.

**Protocol.** Leave-one-subject-out: nine subjects for training, the tenth for testing. The checkpoint is selected on
held-out training concepts, never on the test subject, unless we say otherwise.

**Models.** Most of the work is on ATM, an encoder that applies self-attention across electrode tokens, then a
temporal-spatial convolution, then a projection head. We also use TSConv (a purely convolutional encoder) and, for
generality checks, EEGProject, EEGNet, EEGConformer, two token transformers and the pretrained LaBraM.

**Three information regimes.** The amount of data taken from the new subject matters for any claim, so we fix a
vocabulary:

1. *Pure inductive*: the model sees nothing of the new subject except the query being decoded.
2. *Pure inductive, paradigm-dependent*: same, but the method uses a known property of the stimulation protocol.
3. *Test data required*: the method uses a set of unlabelled trials of the new subject. No labels and no knowledge of
   the candidate images are used in any regime of this document.

## Mechanistic investigation

### The mean-ERP swap

The decisive experiment is simple. For each subject we compute `m_s`, the average of all its epochs, a 63 × 250
array. Because it averages over thousands of different images, it contains no information about any particular
image. We then recombine: the content `c = x − m_s` of one subject is added to the mean of another subject and
decoded by the frozen model.

For a *seen* subject, accuracy falls from 67 % to about 51 % when its mean is replaced by that of another training
subject (TSConv). Removing the mean altogether is worse. The image-specific signal is identical in all three
conditions. The model therefore uses an image-independent pattern to decide how to read the image-dependent signal,
which is why we call `m_s` a key.

For the *held-out* subject, the same swap changes nothing: its own key is worth no more than a random training key.
The average of the nine training keys, which we write `K`, is slightly better than its own.

### Localisation by activation patching

To find where the key is used, we run a seen subject under a wrong key and restore, at one stage of the network, the
mean activation that the right key would have produced.

- **TSConv.** Restoring the mean activation recovers the full accuracy at every stage up to the first linear layer of
  the projection head, and recovers nothing after the next block. The key is thus carried additively through the
  convolutional stages and is used in a single place: the residual block `h + Linear(GELU(h))` of the head. Our
  reading is that the key sets the operating point of this nonlinearity, and thereby selects which linear readout is
  applied to the content. A linearisation around the key reproduces most of the accuracy but not all of it.
- **ATM.** The key effect is larger (a wrong key removes about half of the seen-subject accuracy) and has three sites
  of similar size: the electrode attention map, the feed-forward layer of the transformer block, and the same residual
  head block as in TSConv.

### Generality across architectures

The swap experiment gives the same sign in all nine architectures. Two regularities stand out. The convolutional
encoders depend least on the key and are the best on a new subject without any correction. The encoders with the
highest accuracy on seen subjects (ATM, the token transformers, LaBraM) depend most on the key. Decoding conditionally
on the subject appears to be how these models reach high within-cohort accuracy, and it is the part that transfers
worst.

### ATM's attention as a key-derived spatial filter

In ATM the attention map over electrodes barely depends on the image: maps for two images of the same subject
correlate at .96, and at .98 with the map computed from the key alone. One fixed map per subject, computed from the
key and reused for every image, gives the same accuracy as per-image attention. The attention layer thus behaves as a
small hypernetwork that turns a subject's key into a 63 × 63 electrode mixing.

This lets us separate two uses of the key. If the attention map is computed from the new subject's *own* key while
the downstream readout is fed the *average training key* `K`, accuracy rises from 32 to 42 % without any training or
label. The opposite assignment hurts. The spatial use of the key generalises to a new person; the readout use does
not. We call this the targeted key substitution. It needs the subject's mean response, hence unlabelled trials.

### Physical nature of the key

The key is the steady-state response to the 5 Hz image stream. Its deviation from `K` is almost entirely periodic
with a 200 ms period (94 % of its energy), whereas the image-specific content of an epoch is not periodic at that lag.
In an 80-repetition average the two components have about the same energy. This observation has no role in the
mechanism itself, but it is what makes result 3 possible.

### An account of the failure on a new subject

We can now state what goes wrong, with the evidence for each step.

1. **The readout is indexed by the key.** During training the model sees nine keys and learns a readout for each
   region of key space. A new subject's key falls outside these nine. For ATM, a seen subject decoded under a wrong
   key lands within a few points of a new subject, before and after re-centring. As far as we can measure, a new subject
   and a mis-keyed seen subject are in the same situation for this model. TSConv keeps an additional deficit of
   about 17 points that does not go through the key.
2. **The output is informative but misplaced.** We compared the embeddings of a new subject with those of a seen one.
   The shape of the cloud is nearly the same (CKA with the image embeddings .73 against .75). What differs is
   its position and orientation. One constant vector per subject accounts for a third of the squared distance between
   EEG and image embeddings. After removing it, a single orthogonal map fitted with labels brings the new subject
   within about 12 points of a seen subject. The rotation is an oracle and we do not use it as a method.
3. **The misplacement is a function of the key.** Take the new subject's key, add to it image-specific contents
   borrowed from the *training* subjects, and embed these synthetic trials. Their mean embedding reproduces the new
   subject's true mean embedding (cosine .98) and corrects the decoding equally well. The offset is therefore
   determined by the key and by nothing else specific to the subject.
4. **The output is the right place to correct it.** Replacing the new subject's mean activation by the training mean
   works better the later it is done, monotonically from the input to the output. Substituting the key at the input
   also changes what every later nonlinearity computes, whereas the network run on the subject's own key produces a
   usable geometry that only needs moving. For a new subject, 70 % of the output change caused by a key swap is one
   constant vector, which explains why key substitution and output re-centring overlap and do not add up.
5. **There is unused headroom in the key.** If we train only the 63 × 250 input key of a frozen ATM with labels of
   the new subject, accuracy reaches about 58 % and is still rising. This is supervised adaptation and serves only to
   show that the choice of key carries a large part of the gap.

All of these effects are effects of the low-noise regime. With single-repetition queries, trial noise dominates and
none of the corrections helps. They grow with the number of repetitions averaged in the query.

## Results

### 1. Three-head ATM

**Idea.** A standard encoder is aligned to one vision model. Here one ATM trunk feeds three linear heads, each
aligned to a different image embedding space (two layers of InternViT-6B and one layer of OpenCLIP bigG). A query is
scored in each space against that space's own image embeddings, and the three cosine matrices are averaged. The cost
is one encoder forward pass and three small matrices.

**Two ways to obtain it.** The heads can be distilled: three single-space ATMs act as teachers, and each head is a
closed-form ridge regression from the frozen trunk's features to one teacher's EEG embeddings. Or the trunk and the
three heads can be trained together from scratch with three contrastive losses. The two versions are equal within the
noise of our folds in every scoring mode, so the teachers are unnecessary and the jointly trained model is the
simpler recipe.

**Size and nature of the gain.** Three heads add 3 to 4 points in every scoring mode: plain, simulated re-centring and real
re-centring (table above). Under real re-centring the gain is significant across subjects for each of the three
seeds, and it recovers roughly half of what an ensemble of six separate encoders adds.

**Interpretation.** The extra heads do not improve the main head, so the trunk is not better. Heads on different
image spaces fail on different queries although they read the same EEG features: the hit/miss patterns of two heads
correlate at about .5 across spaces, against .73 for two seeds of the same space. Much of what an ensemble
contributes is a set of different target spaces, and these can be read from one trunk.

**Limits.** A head helps only if its space already aligns well as a single-space model; heads on CLIP-H or DINOv2 lower the average.
Nonlinear heads and a second seed of the same space add nothing. This lever is independent of the key mechanism and
stacks with every key-related correction we tried.

### 2. Key-swap training of ATM

**Motivation.** If the readout fails because it is indexed by the key, one can try to prevent the indexing during
training. This would give a pure inductive method that does not rely on the stimulation paradigm.

**Method.** For each training trial we subtract the subject's key and add the key of a randomly chosen training
subject. In ATM the attention map is still computed from the raw trial, since that use of the key generalises, and
only the readout sees the swapped trial. At test time the model receives the raw trial of the new subject.

**Result.** On three folds the gain is under 2 points for both ATM and TSConv, with mixed signs across folds, inside
the epoch-to-epoch noise. The swap-trained ATM still gains about 7 points from the simulated re-centring of result 3,
almost as much as a standard ATM. Training under swapped keys did not make the readout independent of the key.

**Related training arms.** Two neighbouring ideas give the same picture.

- *Key removed in training and at test.* This helps the most key-dependent simple encoder (EEGProject, about +8) and
  TSConv modestly. It requires an estimate of the new subject's key at test time, so it does not belong to the first
  regime.
- *Decomposition built into ATM.* The readout is trained on trials whose periodic component has been removed (see
  result 3). It reaches about 40 % on the three screening folds, below the 43.5 % obtained by applying the same
  decomposition at test time to a standard ATM.

**Interpretation (speculative).** A model that is allowed to use the key seems to learn a cleaner shared code, by
factoring subject effects out through the key; a test-time correction then selects that code. A model denied the key
has to absorb subject variability into the content pathway. A second possible reason for the null result is that
nine keys are too few to learn invariance to an unseen tenth. We have not tested either explanation.

**Consequence.** The best paradigm-independent pure inductive number remains the plain three-head model.

### 3. In-trial simulated re-centring

This is the main pure inductive result. It combines two observations from the mechanistic section.

**Step 1: a key estimate from the query alone.** The key is periodic at 200 ms and the image content is not. Averaging
the five 200 ms segments of a one-second query, and tiling the result, isolates the periodic part of the trial. We
use it as the deviation of the subject's key from `K`. This comb filter has no parameter and uses no other data of
the subject. It explains between half and two thirds of the true key deviation. A Wiener filter fitted on the
training subjects recovers the same period without being told and decodes no better.

A first use of this estimate is the targeted substitution of the previous section, now within one trial: the
attention map from the raw query, the readout fed the query with its estimated key replaced by `K`. This already
gives about 40 % with one head.

**Step 2: the output centre simulated from the key.** Real re-centring (result 4) subtracts the subject's mean
embedding, which requires a set of the subject's trials. The embedding of the key alone is a poor substitute, because
the mean of the embeddings differs from the embedding of the mean. The account above says the centre is a function of
the key, so we simulate it: we add the estimated key to a small number of key-free contents taken from the training
subjects, embed these synthetic trials, and average. The query is then scored as

`e(x) − o(k(x)) + anchor`,

where `o` is the simulated centre and the anchor is the mean image embedding of the training images. Two details
matter in practice. The borrowed contents must have the noise level of the query, so we use averages over the nine
training subjects. And pairing each synthetic trial with the same content under the key `K` removes most of the
sampling noise, so that 16 synthetic trials per query are enough.

**Result.** With a frozen standard ATM, accuracy goes from 33 to 45.5 % with one head and from 36 to 48.5 % with three
heads (one seed, ten folds). All ten held-out subjects improve. The same procedure helps the other encoders we tried
and retains between 40 and 75 % of what real re-centring gives them.

**Remaining gap.** With the true key in place of the in-trial estimate, the simulated centre matches the real one. The
3 to 5 points that separate this method from real re-centring are therefore due to the quality of the key estimate.

**Limits.**

- *Paradigm.* Only step 1 depends on the stimulation protocol, and it depends on it entirely. Estimators that cannot
  use the periodicity (short temporal lags, spatial statistics, projection on the training keys) all decode below the
  plain model, because they remove as much content as key. In a paradigm that shows one image per trial, a single
  trial does not reveal its key by any means we found.
- *Noise.* The method needs averaged queries: it still helps at 16 repetitions and hurts at 4.
- *Cost.* Sixteen additional forward passes per query.
- *Evidence.* The three-head figure is from one seed; multi-seed confirmation is pending.

### 4. Real re-centring

**Method.** Shift the new subject's normalised embeddings by one vector, so that their mean coincides with the mean
image embedding of the training images:

`e ← e − mean(e over the subject's unlabelled trials) + anchor`.

No label, no candidate image and no training step is involved. Subtracting the mean without adding the anchor is much
weaker.

**Result.** One ATM goes from 32 to 48.6 %, and to 52.6 % with three heads (three seeds, ten folds). Every encoder
gains, and the gain follows the key dependence measured earlier: about 5 points for convolutional encoders and 13 to
16 for attention and token encoders.

**What the mean may be estimated from.** The mean is a property of the subject and not of the queries being scored.
Estimated on other images it gives the same accuracy. About 20 unlabelled trials give most of the gain and 50 give all
of it. With a shrinkage toward "no shift", the correction helps from the third trial on and never hurts. A fully
causal version, where each query is corrected with the queries seen before it, loses about 1.5 points including the
cold start.

**Conditions of validity.**

- The trials used for the mean must sample images broadly. A mean taken over a semantically narrow set contains that
  set's content, and removing it is worse than no correction.
- They must have the same number of repetitions as the queries, since the output centre moves with the noise level.

**What the correction contains.** Seen subjects also gain from re-centring on new images (about 7 points for ATM).
Part of the offset is thus a mismatch between training trials (4 repetitions) and test queries (80 repetitions), and
the rest is specific to the new subject. Roughly half of the 15-point gain on a new subject belongs to each source.
Any claim about "subject calibration" should state this.

**Relation to the other corrections.** Re-centring and key substitution correct the same quantity and do not add up;
re-centring alone is the better of the two. Matching second moments adds about one point on ATM and nothing
elsewhere. Training the model with per-subject centring gave no benefit.

## Negative results

For completeness, these directions were tested and closed: a linear electrode re-mixing of the content derived from
the keys; fitting the key without labels by agreement between repetitions; a fixed output shift computed from training
data only; refitting the heads under substituted keys; nonlinear heads; heads on weakly aligned image spaces;
single-head distillation of an ensemble; and in-trial key estimates that do not use the stimulus period.

## Open questions

1. **A paradigm-independent inductive method.** Key-swap training was the only candidate and it did not work on three
   folds. A meta-learned readout or a larger pool of training keys (more subjects) are untested.
2. **The orientation of the embedding cloud.** An oracle rotation is worth more than 20 points after re-centring.
   We have no label-free way to obtain it within the regimes considered here.
3. **Few repetitions.** Every correction vanishes on noisy queries. We do not know whether the key mechanism matters
   in the single-trial regime or is simply masked there.
4. **Generality.** One dataset, ten subjects, one paradigm. The key is a steady-state response here; in a slow
   paradigm it would be an ordinary mean ERP, and the mechanism may differ in size.
5. **Confirmation.** The three-head simulated re-centring needs the two other seeds, and the localisation
   experiments need validation-selected checkpoints.

## Pointers

- Mechanistic experiments: `scripts/mi/MI_LOG.md`, entries E01 to E17 (code `scripts/mi/`).
- Follow-up: `scripts/subject_key/KEY_LOG.md`. Real re-centring K09 to K19; three heads K20, K21, K24; centre and
  orientation K32; in-trial key K36 to K44; simulated re-centring K46 to K51; training arms at the end of the log.
- Scripts: `k10_calib.py` (real re-centring), `k21_heads.py` and `k24_multispace.py` (three heads),
  `k51_centre_any.py` (simulated re-centring on any model), `train.py --subject_key swap | swap_split` (key-swap
  training), `train.py --aux_targets` (jointly trained heads).
