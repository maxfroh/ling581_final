#!/bin/bash
#SBATCH --job-name=lr0.01_e10
#SBATCH --partition=gpu-a100
#SBATCH --output=lr0.01_e10.%j.out
#SBATCH --error=lr0.01_e10.%j.err
#SBATCH --account=CCR24017
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL,TIME_LIMIT_50,TIME_LIMIT_90,TIME_LIMIT
#SBATCH --mail-user=mbf1102@rit.edu

set -e
cd $SLURM_SUBMIT_DIR

export TOKENIZERS_PARALLELISM=false

srun -N1 -n1 --exclusive bash -c "source /scratch/10746/maxfroh/ling581/envs/ling581/bin/activate && python /scratch/10746/maxfroh/ling581/ling581_final/trainer.py --num_epochs=10 --learning_rate=0.01 --batch_size=32" &

wait

