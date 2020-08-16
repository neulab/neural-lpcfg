#!/bin/bash
# test.sbatch
#
#SBATCH -J ptb-lexicalized-pcfg-10-20-len40-seed2345
#SBATCH -c 4
#SBATCH -o log/ptb-lexicalized-pcfg-10-20-len40-seed2345
#SBATCH --gres gpu:1
#SBATCH --mem 32G
#SBATCH --time=200:00:00

python train.py --nt_states 10 --t_states 20 --state_dim 300 --seperate_nt_emb_for_emission --head_first --tie_word_emb --save_path dump_model/ptb-lexicalized-pcfg-10-20-len40-seed2345.pt --evaluate_dep --print_every 100 --max_length 20 --final_max_length 40 --num_epochs 30 --print_every 100 --opt_level O2 --pretrained_word_emb /home/hzhu2/pretrained-model/glove.840B.300d.pkl --t_emb_init /home/hzhu2/pretrained-model/glove-kmeans-20-centroids.txt  --log_dir runs/ptb-lexicalized-pcfg-10-20-len40-seed2345 --seed 2345
