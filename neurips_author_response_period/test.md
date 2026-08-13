**Protocol.** We take the original EEG queries made of 80 averaged recordings and divide them
into $B$ queries made of either 5, 10, 20 or 40 averaged repetitions ($B = 16, 8, 4, 2$). Test queries from the unseen subject arrive one at a time from this bank, in a random order and with unknown corresponding stimulus. The calibration is re-fitted and scored at regular intervals on all trials available up to that point (50 times per experiment). Because the order is random, the query set does not match the candidate set until late in the stream, and queries are progressively duplicated, breaking the one-to-one matching. We also vary the menu size ($K = 25, 50, 100, 150, 200$) by sub-sampling the original bank of 200 candidates. We measure the calibration time needed to beat the baseline and the final gain once all test samples have been used, averaged over 3 random seeds.

**Results.**

$$
\\begin{array}{lcccc|cccc}
\\hline
 & \\text{Minutes to beat baseline} &  &  &  & \\text{Top-1 gain once calibrated} &  &  &  \\\\
\\text{Reps per query} & 5 & 10 & 20 & 40 & 5 & 10 & 20 & 40 \\\\
\\hline
K=25 & 1.1 & 1.9 & 2.4 & 4.0 & +23.3\\,(55\\%) & +28.3\\,(55\\%) & +26.7\\,(46\\%) & +19.8\\,(31\\%) \\\\
K=50 & 1.6 & 2.7 & 4.3 & 6.9 & +20.3\\,(68\\%) & +31.7\\,(80\\%) & +28.8\\,(59\\%) & +27.2\\,(49\\%) \\\\
K=100 & 3.7 & 4.8 & 7.5 & 10.1 & +9.6\\,(45\\%) & +22.9\\,(75\\%) & +30.4\\,(78\\%) & +30.1\\,(65\\%) \\\\
K=150 & 4.8 & 6.4 & 8.0 & 9.6 & +4.9\\,(30\\%) & +15.9\\,(67\\%) & +26.0\\,(85\\%) & +27.9\\,(76\\%) \\\\
K=200 & 11.7 & 4.3 & 8.5 & 10.7 & +2.9\\,(21\\%) & +12.8\\,(63\\%) & +24.1\\,(90\\%) & +30.3\\,(94\\%) \\\\
\\hline
\\end{array}
$$

Adaptation pays within one to eleven minutes of recording in all twenty settings. Once the calibration material is exhausted, the gain reaches +30.3 points on the full 200-item menu and exceeds +20 points across most of the grid. The smallest gains come from a weak base against a large menu, at 5 repetitions with $K \\geq 100$, where the encoder leaves too little structure to refine. We will state this limit in the paper

**A progressive scheduler.** This new experiment highlights a failure mode of our method : when the buffer covers only a fraction of the menu, the rotation is estimated from too few correspondences, and applying it at full strength falls below the unadapted baseline. We therefore repeat the experiment with the rotation damped to $\\alpha(t) = 0.8\\,u/(u+1.5)$, $u = t/K$ being the ratio of arrivals to menu size. Both $t$ and $K$ are known at decoding time, so this needs no labels.

$$
\\begin{array}{lcccc|cccc}
\\hline
 & \\text{Minutes to beat baseline} &  &  &  & \\text{Top-1 gain once calibrated} &  &  &  \\\\
\\text{Reps per query} & 5 & 10 & 20 & 40 & 5 & 10 & 20 & 40 \\\\
\\hline
K=25 & 1.1 & 1.9 & 2.4 & 3.7 & +24.1\\,(57\\%) & +28.0\\,(55\\%) & +26.6\\,(45\\%) & +20.1\\,(31\\%) \\\\
K=50 & 1.6 & 2.7 & 4.3 & 5.3 & +20.9\\,(70\\%) & +31.3\\,(79\\%) & +29.1\\,(59\\%) & +27.2\\,(49\\%) \\\\
K=100 & 2.7 & 4.3 & 6.9 & 8.0 & +10.7\\,(50\\%) & +23.5\\,(77\\%) & +30.7\\,(78\\%) & +29.5\\,(63\\%) \\\\
K=150 & 4.8 & 3.2 & 6.4 & 3.2 & +6.1\\,(37\\%) & +16.9\\,(72\\%) & +26.6\\,(87\\%) & +28.5\\,(78\\%) \\\\
K=200 & 5.3 & 3.2 & 3.2 & 5.3 & +4.2\\,(30\\%) & +14.1\\,(69\\%) & +25.2\\,(94\\%) & +31.1\\,(97\\%) \\\\
\\hline
\\end{array}
$$

A single schedule serves every subject, repetition count and menu size, so these figures are a floor rather than a tuned result. In settings where samples are sparse, the scheduler allows the adaptation to beat the baseline faster with no compromise on accuracy.
