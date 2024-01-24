#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/c3_eval_%A.out
#SBATCH --job-name=muse_c3
#SBATCH -n 1

module load miniconda

source activate muse

### AROUSAL

# egemaps
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-06_[egemaps]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --feature egemaps --predict --eval_seed  101 --use_gpu  # --cache 

# ds
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-11_[ds]_[physio-arousal]_[32_2_False_64]_[0.005_256] \
        --feature ds --predict --eval_seed  102 --use_gpu

# w2v
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-24_[w2v-msp]_[physio-arousal]_[32_4_True_64]_[0.005_256] \
        --feature w2v-msp --predict --eval_seed 104 --use_gpu  

# faus
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-37_[faus]_[physio-arousal]_[128_4_True_64]_[0.005_256] \
        --feature faus --predict --eval_seed 101 --use_gpu  

# vit 
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-40_[vit]_[physio-arousal]_[256_4_True_64]_[0.005_256] --normalize \
        --feature vit --predict --eval_seed 105 --use_gpu  
 
# facenet 
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-46_[facenet]_[physio-arousal]_[256_1_False_64]_[0.001_256] \
        --feature facenet --predict --eval_seed 101 --use_gpu  
 
# BPM
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-03-12-36_[BPM]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --feature BPM --normalize --predict --eval_seed 105 --use_gpu
 
# ECG
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-03-12-42_[ECG]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --feature ECG --normalize --predict --eval_seed 104 --use_gpu  
 
# resp
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-24-15-28_[resp]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --feature resp --normalize --predict --eval_seed 103 --use_gpu  
 
# biosignals
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-15-02_[biosignals]_[physio-arousal]_[256_4_False_64]_[0.002_256] \
        --feature biosignals --normalize --predict --eval_seed 104 --use_gpu  
 

### VALENCE 

### VALENCE 

# egemaps
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-50_[egemaps]_[valence]_[256_4_False_64]_[0.002_256] --normalize \
        --feature egemaps --predict --eval_seed 101 --use_gpu  

# ds
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-09-55_[ds]_[valence]_[64_2_False_64]_[0.001_256] \
        --feature ds --predict --eval_seed 102 --use_gpu  

# w2v
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-10-01_[w2v-msp]_[valence]_[128_4_True_64]_[0.005_256] \
        --feature w2v-msp --predict --eval_seed 104 --use_gpu 


# faus
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-10-08_[faus]_[valence]_[128_4_True_64]_[0.005_256] \
        --feature faus --predict --eval_seed 101 --use_gpu  


# vit 
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-10-13_[vit]_[valence]_[128_4_True_64]_[0.001_256] --normalize \
        --feature vit --predict --eval_seed 103 --use_gpu 


# facenet 
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-10-19_[facenet]_[valence]_[128_2_False_64]_[0.005_256] \
        --feature facenet --predict --eval_seed 102 --use_gpu  
 

# BPM 
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-03-12-39_[BPM]_[valence]_[256_4_False_64]_[0.002_256] \
        --feature BPM --normalize --predict --eval_seed 103 --use_gpu  

# ECG
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-03-12-45_[ECG]_[valence]_[256_4_False_64]_[0.002_256] \
        --feature ECG --normalize --predict --eval_seed 105 --use_gpu  

# resp
python3 main.py --task personalisation \
        --eval_model RNN_2024-01-24-15-31_[resp]_[valence]_[256_4_False_64]_[0.002_256] \
        --feature resp --normalize --predict --eval_seed 103 --use_gpu  

# biosignals
python3 main.py --task personalisation \
        --eval_model RNN_2023-12-21-15-05_[biosignals]_[valence]_[256_4_False_64]_[0.002_256] \
        --feature biosignals --normalize --predict --eval_seed 103 --use_gpu  


