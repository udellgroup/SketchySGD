#!/bin/bash

datasets=(ijcnn1 real-sim susy)
problem_type=logistic
n_epochs=40
k=500
mu=0.01
destination=./simods_hessian_results

for name in "${datasets[@]}"
do
    python hessian_spectrums.py --data $name --problem $problem_type --epochs $n_epochs --eig_num $k --mu $mu --dest $destination
done