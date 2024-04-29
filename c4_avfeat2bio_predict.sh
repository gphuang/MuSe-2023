#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c4_eval_%A.out
#SBATCH --job-name=muse_c4
#SBATCH -n 1

module load miniconda

source activate muse

#### BPM_normalized 

# egemaps
python3 main.py --task personalisation --eval_model RNN_2023-12-21-14-55_[egemaps]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature egemaps  --normalize --eval_seed 105 --predict --cache  --use_gpu # 
# ds
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-02_[ds]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature ds  --eval_seed 104 --predict --use_gpu

# w2v-msp
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-10_[w2v-msp]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature w2v-msp  --eval_seed 105 --predict --use_gpu

# faus
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-20_[faus]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature faus  --eval_seed 103 --predict --use_gpu
                            
# vit
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-27_[vit]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature vit  --normalize --eval_seed 105 --predict --use_gpu
                            
                            
# facenet
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-34_[facenet]_[BPM_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature  facenet --eval_seed 104 --predict --use_gpu
                            
                             
#### ECG_normalized 

# egemaps
python3 main.py --task personalisation --eval_model RNN_2023-12-21-14-57_[egemaps]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature egemaps  --normalize --eval_seed 105 --predict --use_gpu
                            
                            
# ds
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-06_[ds]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature ds  --eval_seed 101 --predict --use_gpu
                            
                            
# w2v-msp
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-14_[w2v-msp]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature w2v-msp  --eval_seed 101 --predict --use_gpu
                            
                            
# faus
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-23_[faus]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature faus  --eval_seed 104 --predict --use_gpu \
                            
                            
# vit
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-30_[vit]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature vit  --normalize --eval_seed 101 --predict --use_gpu
                            
                            
# facenet
python3 main.py --task personalisation --eval_model RNN_2023-12-21-15-38_[facenet]_[ECG_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature facenet  --eval_seed 105 --predict --use_gpu
                            
                            
#### resp_normalized

# egemaps
python3 main.py --task personalisation --eval_model RNN_2023-12-22-15-57_[egemaps]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature egemaps  --normalize --eval_seed 102 --predict --use_gpu
                            
                            
# ds
python3 main.py --task personalisation --eval_model RNN_2023-12-22-16-01_[ds]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature ds  --eval_seed 101 --predict --use_gpu
                            
                            
# w2v-msp
python3 main.py --task personalisation --eval_model RNN_2023-12-22-16-08_[w2v-msp]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature 2v-msp  --eval_seed 103 --predict --use_gpu
                            
                            
# faus
python3 main.py --task personalisation --eval_model RNN_2023-12-22-16-13_[faus]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature faus  --eval_seed 101 --predict --use_gpu
                            
                            
# vit
python3 main.py --task personalisation --eval_model RNN_2023-12-22-16-17_[vit]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature vit  --normalize --eval_seed 105 --predict --use_gpu
                            
                            
# facenet
python3 main.py --task personalisation --eval_model RNN_2023-12-22-16-22_[facenet]_[resp_normalized]_[256_4_False_64]_[0.002_256] \
                        --feature facenet  --eval_seed 102 --predict --use_gpu