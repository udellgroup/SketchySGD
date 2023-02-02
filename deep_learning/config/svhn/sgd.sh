#!/bin/bash

mkdir -p results/sgd/svhn/$1
python run_experiments.py --opt sgd --data svhn --trials 1 --epochs 200 --act $3 --wd 0.0005 --bs 128 --lr-decay 0.1 --lr-decay-epoch 80 120 160 --ngpu 4 --init-seed 1234 --batch-seed $2 --dir $1