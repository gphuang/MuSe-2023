## 0. Data exploration
# - feat-feat, feat-label correlation analysis?
# - baseline paper description of stats and results
# -- c1 facenet has error logs/c1_facenet_26707042.out
# -- rerun c3 1st step bpm ecg to affect (kinit)
# -- rerun c3 1st step avfeat to [bpm, ecg, resp]_normalized

## 1. Baselines
# av-feats to emotion_class (approval/disappoitment/uncertainty) - mimic_best.sh (c1)
# av-feats to humor_class (presence or absence of humor) - humor_best.sh (c2)*
# av-feats to emotion_values (arousal/valence) - personalisation_[1st, 2nd]_step_best.sh  (c3)
# biosignals (bpm, ecg, resp) to emotion_values (arousal/valence) - c3_biosignal_to_arousal_valence_[1st, 2nd]_step.sh (c3+)
# av-feats to biosignals - c3_avfeat_to_biosignal_[1st, 2nd]_step .sh (c3+)

## 2. Evaluate
# ? best a, v feature and model combo - eval_best_[mimic, humor, personalisation].sh
# ? Test Mean Pearson: nan c1
# - eval_mimic.sh 
# - eval_best_mimic.sh
# - feat2bio, bio2affects went through 2nd step in 'personalisation'
# - view training history? loss vs. feat, acc. vs. feat
# - optimize lr? rnn?

## 3. Derive biosignals (default RNN setup)
# av-feats to biosignals (bpm, ecg, resp) (c3+)
# - normalize BPM ECG resp as target
# - smoothing? 
# - biosignals & emotion_values are 'personalized'?
# - rnn mapping
# - python data_preprocessing.py && copy [ECG, BPM, resp]_normalized to label(_segments) folder
# - view loss vs. feat, acc. vs. feat

## Cross modal
# av-feats, derived emotion_values, derived biosignals to emotion_class (c1)
# - used trained models from c3 to derive emotion_values and biosigals
# - prep & preprocess (?) derived features as input for c1
# - train & eval
# - python late_fusion.py ?

## Deep learning with biosigs, avfeats, affects
# hyper parameter search
# bi-directional rnn
# convolutional recurrent network
# lstm, attention

## evaluation? compare systems
# error variations
python compare_systems.py /teamwork/t40511/muse_2023/exp/MuSe-2023-Aalto-models/results/prediction_muse/humor/RNN_2023-06-13-12-03_[bert-mono-en]_[128_4_False_64]_[0.001_256]/105/ \
                          /teamwork/t40511/muse_2023/exp/MuSe-2023-baselines/results/prediction_muse/humor/RNN_2023-06-02-16-39_[bert-multilingual]_[128_4_False_64]_[0.001_256]/101/

### validation in results/log_muse

#### mimic 
for var in egemaps deepspectrum w2v-msp faus vit facenet electra
do 
ls ./mimic/RNN_2023-12-27*$var*.txt
less ./mimic/RNN_2023-12-27*$var*.txt | grep 'Best Mean Pearsons '
done 

#### humor
for var in egemaps ds w2v-msp faus vit facenet bert-multilingual
do 
ls ./humor/RNN_2023-12-22*$var*.txt 
less ./humor/RNN_2023-12-22*$var*.txt | grep 'Best AUC-Score'
done 

#### affect
for output in physio-arousal valence # 
do 
for var in egemaps ds w2v-msp faus vit facenet BPM ECG resp biosignals
do 
ls ./personalisation/RNN*$var*$output*.txt 
less ./personalisation/RNN*$var*$output*.txt | grep 'Best CCC'
done 
done  

#### avfeat to biosig
for output in BPM_normalized ECG_normalized resp_normalized
do 
for var in egemaps ds w2v-msp faus vit facenet 
do 
ls ./personalisation/RNN*$var*$output*.txt 
less ./personalisation/RNN*$var*$output*.txt | grep 'Best CCC'
done 
done 

### evaluation error
logs/c1_eval_26795079.out
logs/c2_eval_26795155.out