#!/bin/bash
SBATCH --account=bepg-delta-gpu --partition=gpu --nodes=1 --ntasks-per-node=4 --gpus-per-node=1 --constraint=gpuA100x8 --time=24:00:00

cd ~/trl
git checkout test-tldr
git pull --rebase origin test-tldr
conda activate trl
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=8
export LOCAL_WORLD_SIZE=8
srun python3 ~/trl/trl/trainer/magrpo_trainer.py