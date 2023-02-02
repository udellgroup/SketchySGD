#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for sketchysgd
# $4 = folder name for sgd
# $5 = folder name for adam
# $6 = folder name for adamw
./config/cifar10/sketchysgd.sh run_$3_$1 $2 $1
./config/cifar10/sgd.sh run_$4_$1 $2 $1
./config/cifar10/adam.sh run_$5_$1 $2 $1
./config/cifar10/adamw.sh run_$6_$1 $2 $1
