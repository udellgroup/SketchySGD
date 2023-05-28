#!/bin/bash

datasets=(e2006 yearpredictionmsd yolanda)
opts=(sgd svrg saga lkatyusha slbfgs)
opts_sketchy=(sketchysgd)
precond=(nystrom ssn)

problem_type=least_squares
n_epochs=20
mu=0.01
n_runs=10

destination=./simods_performance_results

for name in "${datasets[@]}"
do
    for opt in "${opts[@]}"
    do
        if [ $opt == "sgd" ]
        then
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --n_runs $n_runs --dest $destination
        elif [ $opt == "saga" ]
        then
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --n_runs $n_runs --dest $destination
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --auto_lr --n_runs $n_runs --dest $destination
        elif [ $opt == "svrg" ] || [ $opt == "lkatyusha" ]
        then
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu --n_runs $n_runs --dest $destination
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu --auto_lr --n_runs $n_runs --dest $destination
        else
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu  --n_runs $n_runs --dest $destination
        fi
    done
    for opt in "${opts_sketchy[@]}"
    do
        for precond_type in "${precond[@]}"
        do
            if [ $opt == "sketchysgd" ]
            then
                python performance_experiments.py --data $name --problem $problem_type --opt $opt --precond $precond_type --epochs $((n_epochs * 2)) --mu $mu --n_runs $n_runs --dest $destination
            fi
        done
    done
done