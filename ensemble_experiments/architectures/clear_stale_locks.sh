#!/bin/bash
# A fold's lock dir is claimed atomically by one worker and only removed when the
# fold finishes. A cancelled/preempted job leaves its lock behind and blocks the
# retry, so this clears locks that are genuinely abandoned.
#
# Stale = no result.csv AND the run dir has had no write for 10 minutes. The
# second test matters: a running fold also has no result.csv yet, and clearing
# its lock would let a second worker start the same fold in parallel.
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
for d in results/things_eeg/ortho_arch/*/seed3300/.lock-*; do
  [ -d "$d" ] || continue
  tag=${d##*.lock-}; base=${d%/.lock-*}
  compgen -G "$base/*-sub-$tag/result.csv" >/dev/null && continue      # finished
  live=$(find "$base"/*-sub-"$tag" -newermt '-10 minutes' 2>/dev/null | head -1)
  [ -n "$live" ] && { echo "[busy] $d"; continue; }                    # still running
  rmdir "$d" && echo "[cleared] $d"
done
