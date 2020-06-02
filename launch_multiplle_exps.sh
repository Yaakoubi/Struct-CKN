#!/usr/bin/env bash

for regParSDCA in 0.01 0.001 0.0001 0.00001
do
 for scaler in 0 3
 do
  for npass in 1 10
  do
   for zero_prob in 0 0.0001
   do
    sbatch --gres=gpu --mem=40gb script_supervised_SVM.sh $regParSDCA $scaler $npass $zero_prob
    sleep 5
   done
  done
 done
done
