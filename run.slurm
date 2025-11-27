#!/bin/bash
#SBATCH --job-name=shrink_lr0.00002_e20_b32
#SBATCH --partition=gpu-a100
#SBATCH --output=shrink_lr0.00002_e20_b32.%j.out
#SBATCH --error=shrink_lr0.00002_e20_b32.%j.err
#SBATCH --account=CCR24017
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_50,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=mbf1102@rit.edu

set -e
cd $SLURM_SUBMIT_DIR

export TOKENIZERS_PARALLELISM=false

srun -N1 -n1 --exclusive bash -c "source /scratch/10746/maxfroh/ling581/envs/ling581/bin/activate && python /scratch/10746/maxfroh/ling581/ling581_final/trainer.py --num_epochs=20 --learning_rate=2e-5 --batch_size=32 --shrink" &

wait

