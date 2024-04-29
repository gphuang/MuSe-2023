#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_eval_%A.out
#SBATCH --job-name=muse_c3
#SBATCH -n 1

source activate data2vec

# best audio - from results/logs_muse
python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-17-31_[w2v-msp]_[128_2_False_64]_[0.001_256] \
        --feature w2v-msp --eval_seed 104 --predict

# best video
python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-17-57_[faus]_[256_4_True_64]_[0.0005_256] \
        --feature faus --eval_seed 104 --predict

# best text
python3 main.py --task mimic \
        --eval_model RNN_2023-12-27-18-06_[electra]_[128_1_True_64]_[0.005_256] \
        --feature electra --eval_seed 103 --predict

# fusion
python3 late_fusion.py --task mimic \
                        --model_ids RNN_2023-12-27-17-31_[w2v-msp]_[128_2_False_64]_[0.001_256] RNN_2023-12-27-17-57_[faus]_[256_4_True_64]_[0.0005_256] RNN_2023-12-27-18-06_[electra]_[128_1_True_64]_[0.005_256] \
                        --seeds 104 104 103
