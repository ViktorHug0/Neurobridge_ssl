#!/bin/bash
# Per-arm fold table for the orthogonal-architecture sweep.
cd /nasbrain/p20fores/Neurobridge_SSL
for d in results/things_eeg/ortho_arch/*/seed3300; do
  arm=$(basename "$(dirname "$d")")
  vals=$(for f in $(ls -d "$d"/*-sub-*/result.csv 2>/dev/null | sort); do
           tail -1 "$f" | awk -F, '{print $6}'; done)
  n=$(echo "$vals" | grep -c . || true)
  [ "$n" -eq 0 ] && { printf '%-9s  0 folds\n' "$arm"; continue; }
  mean=$(echo "$vals" | awk '{s+=$1} END {printf "%.2f", s/NR}')
  printf '%-9s %2d folds  mean=%-6s  %s\n' "$arm" "$n" "$mean" "$(echo "$vals" | tr '\n' ' ')"
done
