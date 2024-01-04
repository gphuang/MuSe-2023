# best audio
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-30_[w2v-msp]_[128_2_False_64]_[0.005_256] \
                --feature w2v-msp --eval_seed 105 --predict

# best video
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-43_[vit]_[64_2_False_64]_[0.0001_256] \
                --feature vit --normalize --eval_seed 105 --predict

# best text
python3 main.py --task humor \
                --eval_model RNN_2023-12-22-15-35_[bert-multilingual]_[128_4_False_64]_[0.001_256] \
                --feature bert-multilingual --eval_seed 105 --predict