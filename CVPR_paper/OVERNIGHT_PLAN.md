# Overnight runs — deferred from the first draft (user, 2026-10-21 15:05)

**QUEUED 2026-10-21 19:05** (appended behind the encoder study and the SAMGA "Standard" folds; starts about 21:00). Pre-append checks passed: 4 GPU workers (514, 515, 517, 502), polling CPU workers in job 63835, only atm128 seed-3302 fold 1 present (finished). To cancel a line, delete it from the live queue file. The exact queue lines are prepared in `CVPR_paper/queue/overnight/{gpu,cpu}.txt`. To start, append them to the live queues. Each command takes the queue lock, then appends:

```bash
cd /nasbrain/p20fores/Neurobridge_SSL
flock CVPR_paper/queue/gpu_queue.txt.lock bash -c 'grep -v "^#" CVPR_paper/queue/overnight/gpu.txt >> CVPR_paper/queue/gpu_queue.txt'
flock CVPR_paper/queue/cpu_queue.txt.lock bash -c 'grep -v "^#" CVPR_paper/queue/overnight/cpu.txt >> CVPR_paper/queue/cpu_queue.txt'
```

## What runs, in priority order (GPU queue order)

| # | Runs | GPU work | Then (CPU, automatic) | Feeds |
|---|---|---|---|---|
| P1 | ATM-128 seed 3302 (100 epochs), folds 2–10 (fold 1 done) | 9 folds, ~45 min each | `final_analyses.sh atm128 3302` + 50-fit streams (SAGE and SCORE) | 3-seed main table, R1–R4, stream figure |
| P2 | SCORE (original) seeds 3301 and 3302, all folds | 20 folds, ~50 min | battery (`FINAL_CKPT=checkpoint_last.pth`) + 50-fit SCORE stream, per seed | SCORE (original) column, 3 seeds |
| P3 | SAMGA intra-subject (`intra.sh` verbatim, seed 2025) | 10 small runs, ~10–15 min | battery `INTRA=1` (SCORE deployment + SAGE on SAMGA intra) | intra SCORE (original) column |
| P4 | Squeezeformer folds 2,4–7,9,10 and rmix folds 2–10, seed 3300 (user 20:40) | 16 folds, ~30 min | `encoder_battery.sh` per arm, then the 11-encoder `error_decomposition.py` pass | encoder-gain analysis (E8.4) |
| — | 3-seed table | — | `seed_table.py --arms atm128 score_orig --seeds 3300 3301 3302` → `final/main_table_3seeds.md` | main table |

Rough total GPU work: about 2,000 GPU-minutes, i.e. about 8–9 h on 4 GPUs. If the night is short, P1 finishes first, so the 3-seed ATM-128 numbers land even if P2 or P3 do not.

**2026-10-21 19:50 HOLD on P2 (SCORE (original) seeds 3301/3302):** moved to the END of the GPU queue. Our SCORE
reproduction is ~5 pts below the paper (zero-shot 24.5 vs 29.33 +- 0.57); the SAMGA "Standard" anchor (queued just
before) decides whether the gap is in the base pipeline or in our SCORE-loss reimplementation. If the latter, the
seeds must be rerun with the fixed loss: delete these 20 lines from the live queue before they start.

**2026-10-22 01:50 P2 PARKED** (lines in `CVPR_paper/queue/overnight/parked_score_seeds.txt`; their CPU waiters removed;
3-seed table now ATM-128 only). Replaced by a 10-fold SCORE component ablation (`scoremp`, `scorerec`, seed 3300, 20
folds), at the end of the GPU queue. See EXPERIMENT_LOG E3.2c.

## Before appending: checks

- **Workers.**
  - `squeue -u $USER` should show 4 GPU workers in total (node cap, user rule).
  - At least one batch job must run POLLING CPU workers (`cpu_batch.sh`, as in job 63835). Batch jobs whose own workers exit on an empty queue end the job and kill every step inside it. This happened twice on 2026-10-21.
- **Partial dirs.** Partial seed-3302 fold dirs, i.e. without `result.csv`, were deleted 2026-10-21 15:20. A partial dir next to a finished rerun would make `find_checkpoint_dir` see two dirs for the same subject. Re-check before appending: `ls results/things_eeg/cvpr/loso/atm128/seed3302/*/result.csv`.
- **Checkpoint for SCORE (original).** SCORE (original) must be evaluated on `checkpoint_last.pth`, since SCORE reports the final epoch. This is set by `FINAL_CKPT` on the battery line and by the 4th argument of `stream50.sh`. Never set it through `SAGE_CHECKPOINT_NAME`: `cpu_worker.sh` exports a legacy default for that variable.
- **Thread count.** Streams pin 2 BLAS threads, because the SAGE fit on small buffers depends on the thread count.

## Not in this plan (decided)

- TSConv seed 3302: ditched by the user (2026-10-21 15:25). Seed repetition only for the main methods, SAGE on ATM-128 vs SCORE. TSConv stays at 2 seeds.

- ATM-64: ditched by the user. Its R1 SAGE result was 77.65 vs 82.30 for ATM-128.
- LaBraM full fine-tuning (E1.6): no working LOSO recipe.
- SAMGA "Standard" (SCORE Table 2 reproduction anchor): it runs during the day at seed 3300 only. Extra seeds are not needed.
- E1.6 grid encoders: seed 3300 screens only (folds 1/3/8), queued during the day.

## Log

Record every launch and result in `CVPR_paper/EXPERIMENT_LOG.md`.

## 2026-10-22 11:15 — STATUS: ALL OVERNIGHT GPU WORK DONE

P1 (ATM-128 seed 3302 + battery + streams + 3-seed table), P3 (SAMGA intra + battery), P4 (sqf, rmix + batteries +
11-stack diagnostics) and the SCORE component ablation are all finished. GPU queue empty; 0 real failures (the 19 in
`failed.txt` are deliberate kills). P2 stays parked (see EXPERIMENT_LOG 11:15 E3.2c). Batch jobs 63791/63794 exited
with the empty queue; 63835 (502, polling) and the interactive 63981 remain.
