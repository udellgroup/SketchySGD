#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for adahessian
# $4 = folder name for shampoo
# $5 = folder name for kfac
# $6 = folder name for seng
./config/miniboone/adahessian.sh run_$3_$1 $2 $1
./config/miniboone/shampoo.sh run_$4_$1 $2 $1
./config/miniboone/kfac.sh run_$5_$1 $2 $1
./config/miniboone/seng.sh run_$6_$1 $2 $1
