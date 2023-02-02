#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for adahessian
./config/cifar10/adahessian.sh run_$3_$1 $2 $1
./config/cifar10/shampoo.sh run_$4_$1 $2 $1
./config/cifar10/kfac.sh run_$5_$1 $2 $1
./config/cifar10/seng.sh run_$6_$1 $2 $1
