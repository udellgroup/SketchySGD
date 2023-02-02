#!/bin/bash

datasets=(rcv1 yearmsd real-sim news20 e2006 w8a)

for i in "${datasets[@]}"
do
        python sketchysgd_ablation.py --param update_freq --param_list 0.5 1 2 5 40 --data $i --data_folder ./data --epochs 20 --r_seed 123 234 345 --np_seed 246 468 690
        python sketchysgd_ablation.py --param rank --param_list 1 2 5 10 20 50 --data $i --data_folder ./data --epochs 20 --r_seed 123 234 345 --np_seed 246 468 690
done