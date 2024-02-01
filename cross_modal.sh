## Setup
# conda create -n muse python=3.10
# conda activate muse
# conda install pip
# conda clean --all
# pip cache purge
# pip install -r requirements.txt --no-cache-dir

## 0. Data exploration
# - feat-feat, feat-label correlation analysis?
# - baseline paper description of stats and results

## 1. Baselines
# av-feats to emotion_class (approval/disappoitment/uncertainty) - c1_mimic.sh
# av-feats to humor_class (presence or absence of humor) - c2_humor.sh (c2)
# av-feats & biosignals (BPM, ECG, resp) to emotion_values (arousal/valence) - [personalisation, affect]_[1st, 2nd]_step_best.sh (c3)
# av-feats to biosignals - c4_avfeat2bio_[1st, 2nd]_step.sh (c4)
# biosignals to biosignals - c5_bio2bio_[1st, 2nd]_step.sh (c5)

## 2. Derive biosignals (default RNN setup)
# - python data_preprocessing.py && copy [ECG, BPM, resp]_normalized to label(_segments) folder
# - av-feats to biosignals (bpm, ecg, resp) (c4)
# - normalized BPM ECG resp as features and targets (c5)
# - biosignal as wav
# - smoothing? 
# - how biosignals & emotion_values are 'personalized'?
# - view loss vs. feat, acc. vs. feat
# - view training history? loss vs. feat, acc. vs. feat
# - c3.5 hyperparameter grid search for bpm ecg resp for arousal & valence
# AROUSAL
# 1k random sampling hyper para search
# BPM 27833890 # 0.01 lr 27977261
# ECG 27837924
# resp 27837934
# VALENCE
# 1k random sampling hyper para search
# BPM 27837980
# ECG 27837981
# resp 27837982



## 3. Evaluate '--predict' CodaLab
# - rebuild muse conda env
# eval_c1_mimic.sh # done   
# eval_c2_humor.sh # done
# eval_c3_affect.sh # done 
# eval_c4_inversion.sh # done

## 4. Fusion/Cross-modal/Multi-modal
# av-feats, derived emotion_values, derived biosignals to emotion_class (c1)
# - used trained models from c3 to derive emotion_values and biosigals
# - prep & preprocess (?) derived features as input for c1
# - train & eval
python late_fusion.py --task mimic \
                    --model_ids []\
                    --seeds []

## 5. compare systems
# ? predict on new data c1~c4
# ? get class-wise performance for mimic & affect
# error analysis, are systems complementary (go to fusion)
python compare_systems.py /teamwork/t40511/muse_2023/exp/MuSe-2023-Aalto-models/results/prediction_muse/humor/RNN_2023-06-13-12-03_[bert-mono-en]_[128_4_False_64]_[0.001_256]/105/ \
                          /teamwork/t40511/muse_2023/exp/MuSe-2023-baselines/results/prediction_muse/humor/RNN_2023-06-02-16-39_[bert-multilingual]_[128_4_False_64]_[0.001_256]/101/

## 5. Deep learning with biosigs, avfeats, affects
# derive biosignals with '--predict' in main.py from avfeats for other tasks e.g. humor, mimic
# derive affects (approval/disappoitment/uncertainty, humor, arousal/valence) with '--predict' in main.py 
# not essensial (lstm vs. GRU)
# hyper parameter (grid) search
# convolutional layer CRNN
# attention 27942819 AttnRNN

## 6. Results 
# analyse_results.py
# affect does not use text-based features
task='personalisation'
for out in physio-arousal valence 
do 
for var in egemaps ds w2v-msp faus vit facenet biosignals resp BPM ECG 
do
ls ./results/log_muse/$task/RNN_*$var*$out*.txt
less ./results/log_muse/$task/*$var*$out*.txt | grep 'Best' # Best Mean Pearsons
done
done