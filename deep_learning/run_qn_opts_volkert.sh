#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for adahessian
# $4 = folder name for shampoo
# $5 = folder name for kfac
./config/volkert/shampoo.sh run_$4_$1 $2 $1
./config/volkert/kfac.sh run_$5_$1 $2 $1

# ./run_qn_opts_volkert.sh relu 1111 1 1 1
