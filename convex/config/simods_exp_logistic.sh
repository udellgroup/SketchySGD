#!/bin/bash

datasets=(ijcnn1 real-sim susy)
n_runs_vec=(10 10 3)
opts=(sgd svrg saga lkatyusha slbfgs rsn nsketch lbfgs)
opts_sketchy=(sketchysgd)
precond=(nystrom ssn)

problem_type=logistic
n_epochs=20
mu=0.01

destination=./simods_performance_results

for i in "${!datasets[@]}"
do
    name=${datasets[$i]}
    n_runs=${n_runs_vec[$i]}
    for opt in "${opts[@]}"
    do
        if [ $opt == "sgd" ] || [ $opt == "rsn" ] || [ $opt == "nsketch" ]
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
        elif [ $opt == "lbfgs" ]
        then
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $((n_epochs * 2)) --mu $mu --n_runs 1 --dest $destination
        else
            python performance_experiments.py --data $name --problem $problem_type --opt $opt --epochs $n_epochs --mu $mu --n_runs $n_runs --dest $destination
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