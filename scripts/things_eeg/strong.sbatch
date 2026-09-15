#!/bin/bash
#SBATCH --job-name=strongsims
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --nodelist=sl-tp-br-513
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/goal25/strongsims_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/goal25/strongsims_%j.err
cd /nasbrain/p20fores/Neurobridge_SSL
.venv/bin/python cache_strong.py --device cuda:0
