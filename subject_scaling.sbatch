#!/bin/bash
#SBATCH --job-name=subjscale
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-mee-br-501
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/subject_scaling_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/subject_scaling_%j.err

# Subject-count scaling curve, two test subjects per GPU.
# run_subject_scaling.sh skips finished runs, so a requeue resumes rather than repeats.
cd /nasbrain/p20fores/Neurobridge_SSL
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date)"

DEVICE=cuda:0 TEST_SUBJECTS="8 5" ./run_subject_scaling.sh &
P0=$!
DEVICE=cuda:1 TEST_SUBJECTS="9 10" ./run_subject_scaling.sh &
P1=$!
wait $P0 $P1

source .venv/bin/activate
python scripts/things_eeg/subject_scaling_report.py
echo "DONE $(date)"
