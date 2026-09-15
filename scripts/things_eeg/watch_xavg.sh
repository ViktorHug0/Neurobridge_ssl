#!/bin/bash
# Live monitor for the cross-subject-average sweep. Refreshes every 5s.
# Usage: bash scripts/things_eeg/watch_xavg.sh [session_dir]
SESS="${1:-$(ls -dt /nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/xavg512_beta*/ 2>/dev/null | head -1)}"
SESS="${SESS%/}/seed3300"
while true; do
    clear
    echo "=== xavg sweep: $SESS ==="
    date
    echo
    echo "--- per-subject best top1 so far ---"
    SUMM="$SESS/inter_subject_summary.csv"
    [ -f "$SUMM" ] && column -t -s, "$SUMM" || echo "(no summary yet)"
    echo
    LATEST="$(ls -dt "$SESS"/*sub-*/ 2>/dev/null | head -1)"
    if [ -n "$LATEST" ]; then
        echo "--- currently training: $(basename "$LATEST") ---"
        grep -aE 'Epoch \[|best test loss' "$LATEST/train.log" 2>/dev/null | tail -4
    fi
    sleep 5
done
