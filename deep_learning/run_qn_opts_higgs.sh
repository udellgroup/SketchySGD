#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for adahessian
# $4 = folder name for shampoo
# $5 = folder name for kfac
# $6 = folder name for seng
./config/higgs/kfac.sh run_$3_$1 $2 $1
./config/higgs/seng.sh run_$4_$1 $2 $1
