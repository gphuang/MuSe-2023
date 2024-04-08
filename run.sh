#!/bin/bash
#SBATCH --time=05:59:59
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/muse_%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda

source activate pytorch-env

