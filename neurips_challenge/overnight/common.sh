# Sourced by the overnight chains. Do not edit while a chain is running (bash reads scripts
# incrementally; an edit shifts the byte offset under a live run).
cd /nasbrain/p20fores/Neurobridge_SSL
R=neurips_challenge/run_recipe.sh
T2="--learning_rate 1e-3 --tsconv_pool_kernel 25 --tsconv_pool_stride 2"
ATMFIX="--learning_rate 1e-3 --atm_no_subject_token"
SINGLE="--num_epochs 100 --early_stop_patience 10"
# Bootstrap/averaged epochs hold 4x fewer rows: 4x epochs and patience for matched steps.
SHORT="--num_epochs 400 --early_stop_patience 40"
BOOT="--bootstrap_repetition_average --bootstrap_repetition_count 2"
step() { echo "=== $(date '+%F %T') START $*"; "$@" || echo "=== $(date '+%F %T') FAILED $*"; }
# Slurm jobs can start with an expired Kerberos key for /homes (sec=krb5); env's execvp then
# fails with EKEYEXPIRED on the first PATH entry under /homes. Keep the chains off /homes.
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/tmp/${USER}-home && mkdir -p "$HOME"
# True once a recipe tag has a grader score; the rescue chains only fill in what is missing.
scored() { ls results/things_eeg/neurips_track1/recipe/"$1"/*/*/track1_score.json >/dev/null 2>&1; }
