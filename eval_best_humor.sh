# best audio
python3 main.py --task humor \
                --eval_model RNN_2023-12-21-09-15_[w2v-msp]_[128_2_False_64]_[0.005_256] \
                --feature w2v-msp --eval_seed 102 --predict

# best text
python3 main.py --task humor \
                --eval_model RNN_2023-12-21-09-21_[bert-multilingual]_[128_4_False_64]_[0.001_256] \
                --feature bert-multilingual --eval_seed 104 --predict

# best video
python3 main.py --task humor \
                --eval_model RNN_2023-12-21-09-31_[vit]_[64_2_False_64]_[0.0001_256] \
                --feature vit --normalize --eval_seed 105 --predict
