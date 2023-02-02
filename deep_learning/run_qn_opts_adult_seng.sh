#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for seng
./config/volkert/seng.sh run_$3_$1 $2 $1

# ./run_qn_opts_adult.sh relu 1111 1
