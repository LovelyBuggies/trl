#!/bin/bash
#SBATCH --job-name=trl_magrpo
#SBATCH --account=YOUR_ACCOUNT_HERE  # Replace with your allocation account
#SBATCH --partition=gpu              # GPU partition
#SBATCH --nodes=1                    # Request one node
#SBATCH --ntasks-per-node=8          # 8 tasks
#SBATCH --gpus-per-node=8            # Request 8 GPUs
#SBATCH --constraint=gpu200          # Specify GPU 200 devices
#SBATCH --time=24:00:00              # Set your time limit (HH:MM:SS)
#SBATCH --output=trl_magrpo_%j.out   # Output file with job ID
#SBATCH --error=trl_magrpo_%j.err    # Error file with job ID

cd ~/trl
git checkout test-tldr
git pull --rebase origin test-tldr
conda activate trl
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=8
export LOCAL_WORLD_SIZE=8
srun python3 ~/trl/trl/trainer/magrpo_trainer.py