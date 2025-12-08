#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2               
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

set -euo pipefail

# Load modules if needed
module load python/3.11           # adjust to cluster

cd $HOME/atari-world-model

# Activate venv
source venv/bin/activate

 python -u train.py \
    -o pepg \
    -n 8 \
    -t 8 \
    --num_episode 16 \
    --eval_steps 5 \
    --retrain 1 \
    --cap_time 0
