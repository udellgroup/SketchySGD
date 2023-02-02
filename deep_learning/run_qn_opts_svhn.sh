#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for adahessian
./config/svhn/adahessian.sh run_$3_$1 $2 $1
