## 0. Data exploration
# - feat-feat, feat-label correlation analysis?
# - baseline paper description of stats and results
# -- c1 facenet has error logs/c1_facenet_26707042.out
# -- rerun c3 1st step bpm ecg to affect (kinit)
# -- rerun c3 1st step avfeat to [bpm, ecg, resp]_normalized

## 1. Baselines
# av-feats to emotion_class (approval/disappoitment/uncertainty) - mimic_best.sh (c1)
# av-feats to humor_class (presence or absence of humor) - humor_best.sh (c2)*
# av-feats & biosignals (BPM, ECG, resp) to emotion_values (arousal/valence) - personalisation_[1st, 2nd]_step_best.sh (c3)
# av-feats to biosignals - c4_avfeat_to_biosignal_[1st, 2nd]_step.sh (c4)
# biosignals to biosignals - c5_bio2bio_[1st, 2nd]_step.sh (c5)

## 2. Derive biosignals (default RNN setup)
# - python data_preprocessing.py && copy [ECG, BPM, resp]_normalized to label(_segments) folder
# - av-feats to biosignals (bpm, ecg, resp) (c4)
# - normalized BPM ECG resp as features and targets (c5)
# - smoothing? 
# - how biosignals & emotion_values are 'personalized'?
# - view loss vs. feat, acc. vs. feat
# - view training history? loss vs. feat, acc. vs. feat

## 3. Fusion/Cross-modal/Multi-modal
# av-feats, derived emotion_values, derived biosignals to emotion_class (c1)
# - used trained models from c3 to derive emotion_values and biosigals
# - prep & preprocess (?) derived features as input for c1
# - train & eval
python late_fusion.py --task mimic \
                    --model_ids []\
                    --seeds []

## 4. Evaluate
# running c1 eval 26916860, 'should not give test scores'
# running c2 eval 26865398 label csv has 'nan' values
# running c3 2nd step 26852336 done
# running c4 2nd step 26852331 done
# running c5 2nd step 26993779 ?what/where is the output of personalisation 2nd step?
# test-set performance
# - eval_[mimic, humor, personalisation].sh 
# - eval_best_[mimic, humor, personalisation].sh
# - ? how to get class-wise performance for mimic & affect
# compare systems
# error analysis, are systems complementary (go to fusion)
python compare_systems.py /teamwork/t40511/muse_2023/exp/MuSe-2023-Aalto-models/results/prediction_muse/humor/RNN_2023-06-13-12-03_[bert-mono-en]_[128_4_False_64]_[0.001_256]/105/ \
                          /teamwork/t40511/muse_2023/exp/MuSe-2023-baselines/results/prediction_muse/humor/RNN_2023-06-02-16-39_[bert-multilingual]_[128_4_False_64]_[0.001_256]/101/

## 5. Deep learning with biosigs, avfeats, affects
# derive biosignals with '--predict' in main.py from avfeats for other tasks e.g. humor, mimic
# derive affects (approval/disappoitment/uncertainty, humor, arousal/valence) with '--predict' in main.py 
# hyper parameter search
# bi-directional rnn
# convolutional recurrent network
# lstm, attention


## 6. shell 
# results/log_muse
# - analyse_results.py
# mimic 
for var in egemaps deepspectrum w2v-msp faus vit facenet electra
do 
ls ./mimic/RNN_*$var*.txt
less ./mimic/RNN_2023-12-27*$var*.txt | grep 'Best' # Best Mean Pearsons
done 