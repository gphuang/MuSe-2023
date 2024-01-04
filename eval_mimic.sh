#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c1_eval_%A.out
#SBATCH --job-name=muse_c1
#SBATCH -n 1

# audio - from results/logs_muse
python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-16-46_[egemaps]_[256_2_False_64]_[0.001_256] \
        --feature egemaps --eval_seed 101 --predict

python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-16-53_[deepspectrum]_[256_4_False_64]_[0.0005_256] \
        --feature deepspectrum --eval_seed 103 --predict

python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-17-31_[w2v-msp]_[128_2_False_64]_[0.001_256] \
        --feature w2v-msp --eval_seed 104 --predict

# video
python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-17-57_[faus]_[256_4_True_64]_[0.0005_256] \
        --feature faus --eval_seed 104 --predict

python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-18-15_[vit]_[256_4_True_64]_[0.001_256] \
        --feature vit --eval_seed 102 --predict

# text
python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-18-06_[electra]_[128_1_True_64]_[0.005_256] \
        --feature electra --eval_seed 103 --predict