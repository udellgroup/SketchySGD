#!/bin/bash

mkdir -p results/kfac/cifar10/$1
python run_experiments.py --opt kfac --data cifar10 --trials 1 --epochs 80 --act $3 --wd 0.0005 --bs 128 --lr-decay 0.1 --lr-decay-epoch 30 50 70 --ngpu 1 --init-seed 1234 --batch-seed $2 --dir $1
