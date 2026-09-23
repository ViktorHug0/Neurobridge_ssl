#!/bin/bash
# Launch the four Track-1 sweep cells, one per GPU.
#
#   launch_sweep.sh --dummy    2 epochs per config, scoring on 2 subjects (end-to-end check)
#   launch_sweep.sh            the real overnight sweep, 50 epochs
#
# Cell 250hz_avg runs here on sl-tp-br-517 inside a tmux session, as requested, so it survives a
# dropped connection. The other three go to Slurm.
#
# 250hz_noavg goes to an RTX 3090, not a 3080: un-averaged 250 Hz training data is 38.8 GB and a
# 3080 node has 31 GB. The other three cells fit on a 3080 (9.7 / 4.7 / 18.6 GB).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO=$(pwd)
DUMMY="${1:-}"
TAG=$([[ "$DUMMY" == "--dummy" ]] && echo dummy || echo sweep)
LOGS="${REPO}/results/things_eeg/neurips_track1/${TAG}/logs"
mkdir -p "$LOGS"

submit() {  # submit <cell-preproc> <cell-avg> <gres> <mem>
  local preproc=$1 avg=$2 gres=$3 mem=$4 name="${TAG}_${1}_${2}"
  sbatch --job-name="$name" \
         --partition=Brain_GPU --account=brain --qos=low \
         --gres="$gres" --cpus-per-task=8 --mem="$mem" --time=20:00:00 --requeue \
         --output="${LOGS}/${name}_%j.out" --error="${LOGS}/${name}_%j.err" \
         --wrap="cd ${REPO} && neurips_challenge/sweep_cell.sh ${preproc} ${avg} ${DUMMY}"
}

# Three cells to Slurm.
submit 250hz noavg gpu:rtx3090:1 56G
submit 120hz avg   gpu:rtx3080:1 24G
submit 120hz noavg gpu:rtx3080:1 28G

# One cell here, in tmux, on the GPU this session already holds.
SESSION="track1_${TAG}"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd ${REPO} && neurips_challenge/sweep_cell.sh 250hz avg ${DUMMY} 2>&1 | tee ${LOGS}/${TAG}_250hz_avg_local.log"

echo
echo "tmux session : $SESSION   (tmux attach -t $SESSION)"
echo "local log    : ${LOGS}/${TAG}_250hz_avg_local.log"
echo "slurm logs   : ${LOGS}/"
squeue -u "$USER" -o '%.10i %.24j %.9P %.8T %.10M %R' | tail -n +1
