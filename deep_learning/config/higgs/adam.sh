#!/bin/bash

mkdir -p results/adam/higgs/$1
python run_experiments.py --opt adam --data higgs --trials 1 --epochs 50 --act $3 --wd 0.0005 --bs 128 --lr-decay 0.1 --lr-decay-epoch 10 20 --ngpu 4 --init-seed 1234 --batch-seed $2 --dir $1
