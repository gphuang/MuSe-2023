#!/bin/bash
#SBATCH --time=00:59:59
#SBATCH --mem=100G
#SBATCH --output=logs/%A.out
#SBATCH --job-name=muse
#SBATCH -n 1

module load miniconda
module load cuda/11.8 

source activate pytorch-env

python data_preprocesser.py --feat_extractor mfcc
python data_preprocesser.py --feat_extractor melspec
python data_preprocesser.py --feat_extractor egemaps