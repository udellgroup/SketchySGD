#!/bin/bash

mkdir -p results/kfac/miniboone/$1
python run_experiments.py --opt kfac --data miniboone --trials 1 --epochs 30 --act $3 --wd 0.0005 --bs 128 --lr-decay 0.1 --lr-decay-epoch 10 20 --ngpu 1 --init-seed 1234 --batch-seed $2 --dir $1
