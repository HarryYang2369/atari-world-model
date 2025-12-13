#!/bin/bash
#SBATCH --job-name=rnn_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err

set -euo pipefail

# Load modules if needed
module load python/3.11         

cd $HOME/atari-world-model

# Activate venv
source venv/bin/activate



python rnn_train.py