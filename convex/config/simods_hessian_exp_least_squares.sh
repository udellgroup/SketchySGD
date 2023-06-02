#!/bin/bash

datasets=(yolanda e2006 yearpredictionmsd)
problem_type=least_squares
n_epochs=1
k=500
mu=0.01
destination=./simods_hessian_results

for name in "${datasets[@]}"
do
    python hessian_spectrums.py --data $name --problem $problem_type --epochs $n_epochs --eig_num $k --mu $mu --dest $destination
done