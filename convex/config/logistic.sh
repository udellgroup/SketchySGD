#!/bin/bash

opts=(sgd svrg slbfgs lkatyusha sketchysgd)
datasets=(rcv1 real-sim news20)

for i in "${datasets[@]}"
do
    for j in "${opts[@]}"
    do
        python general_experiments.py --opt $j --data $i --data_folder ./data --epochs 20 --search_sz 10 --r_seed 123 234 345 --np_seed 246 468 690
    done
done