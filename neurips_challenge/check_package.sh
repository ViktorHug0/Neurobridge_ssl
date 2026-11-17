#!/bin/bash
# Contract check of a Track-1 submission zip through the official ingestion + scoring programs. The competition's
# check_submission.sh is BCI-only and the image warm-up data (THINGS-EEG2) is not prepared here, so this runs a copy of
# tracks/image_decoding whose Simulated set serves the sealed shape: 32 ch x 120 samples under the Alljoined (Emotiv)
# names in a shuffled order, 1536-D targets, 600 test windows (9 x 64 + 24). The score is chance; the point is that
# load_model(meta) + predict run clean on CUDA.
#   usage: check_package.sh <submission.zip>
set -euo pipefail
zip=$(realpath "$1")
C=/nasbrain/p20fores/mental_imagery/2026-competition
INFO=/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json
work=$(mktemp -d -p /nasbrain/p20fores/Neurobridge_SSL/logs); trap 'rm -rf "$work"' EXIT
cp -rL $C/tracks/image_decoding "$work/track"; mkdir "$work/phase"
sed 's/^  - Image\[.*/  - Simulated/' $C/codabench/phases/warmup/image_decoding/config.yaml > "$work/phase/config.yaml"
python3 - "$work/track/datasets/simulated.py" "$INFO" <<'EOF'
import json, random, sys
p = sys.argv[1]; s = open(p).read()
ch = json.load(open(sys.argv[2]))['ch_names']; random.Random(0).shuffle(ch)
i = s.index('STANDARD_1020 = ['); j = s.index(']', i) + 1
s = s[:i] + 'STANDARD_1020 = ' + repr(ch) + s[j:]
for a, b in (('"n_chans, n_times": [(8, 120)]', '"n_chans, n_times": [(32, 120)]'), ('"n_outputs": [32]', '"n_outputs": [1536]'),
             ('self._make_windows(rng, 120, templates', 'self._make_windows(rng, 600, templates')):
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(p, 'w').write(s)
EOF
unzip -q "$zip" -d "$work/sub"
cd $C; export HOME=/nasbrain/p20fores; source env.sh
python codabench/ingestion_program/ingestion.py --submission-dir "$work/sub" --output-dir "$work/out" \
  --input-data "$work/phase" --benchmark-dir "$work/track" 2>&1 | grep -v -i warn
python codabench/scoring_program/scoring.py --prediction-dir "$work/out" --output-dir "$work/score" >/dev/null
cat "$work"/score/*; echo
