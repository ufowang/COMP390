#!/bin/bash
#SBATCH --job-name=QMIX_Multi_Step_Hare
#SBATCH --output=result_%x_%j.output
#SBATCH --error=error_%x_%j.output
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-user=sglwan19@liverpool.ac.uk
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"

module load libs/cuda/11.3

conda activate pymarl

/users/sglwan19/QMIX/src/QMIX_Multi_Step_Hare.sh 0