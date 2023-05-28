#!/bin/bash

datasets=(e2006 yearpredictionmsd yolanda)
n_runs_vec=(10 10 10)
opts_sketchy=(sketchysgd)
precond=(nystrom)

freq_list=(0.5 1 2 5 100000) # Use 100000 to indicate no updates to the preconditioner after first iteration
rank_list=(1 2 5 10 20 50)

problem_type=least_squares
r_seed=1234
np_seed=2468
n_epochs=40
mu=0.01

destination=./simods_sensitivity_results

for i in "${!datasets[@]}"
do
    name=${datasets[$i]}
    n_runs=${n_runs_vec[$i]}
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            python sensitivity_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --freq_list "${freq_list[@]}" --rank_list "${rank_list[@]}" --epochs $n_epochs --mu $mu --n_runs $n_runs --dest $destination
        done
    done
done