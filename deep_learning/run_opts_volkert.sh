#!/bin/bash

# $1 = activation
# $2 = batch seed
# $3 = folder name for sketchysgd
# $4 = folder name for sgd
# $5 = folder name for adam
./config/volkert/sketchysgd.sh run_$3_$1 $2 $1
./config/volkert/sgd.sh run_$4_$1 $2 $1
./config/volkert/adam.sh run_$5_$1 $2 $1