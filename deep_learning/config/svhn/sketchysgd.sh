#!/bin/bash

mkdir -p results/sketchysgd/svhn/$1
python run_experiments.py --opt sketchysgd --data svhn --trials 1 --epochs 200 --act $3 --wd 0.0005 --bs 128 --lr-decay 0.1 --lr-decay-epoch 80 120 160 --lr-rho-eq --lr-rho-prop --ngpu 4 --init-seed 1234 --batch-seed $2 --dir $1