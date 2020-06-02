#!/usr/bin/env bash
echo Running on $HOSTNAME
echo with old_init_W
echo with fit
echo with clf-fit-init = True each epoch
echo n_filters 200
benchmark=0
predictor=sdca
numberEpochsUnsupCKN=100
numberEpochsFW=10
numberEpochsCKN=100
regParBCFW=1.0
##regParSDCA=
gpu=0
##scaler=3
lr=0.1
#npass= 1 10
size_patch=5
## zero_prob=0 or 0.0001
non_uniformity=0.8
sampling_scheme=gap
line_search=golden
init_previous_step_size=0
use_warm_start=1
echo benchmark $benchmark
echo predictor $predictor
echo numberEpochsUnsupCKN $numberEpochsUnsupCKN
echo numberEpochsFW $numberEpochsFW
echo numberEpochsCKN $numberEpochsCKN
echo regParBCFW $regParBCFW
echo regParSDCA $1
echo gpu $gpu
echo scaler $2
echo lr $lr
echo npass $3
echo size_patch $size_patch
echo zero_prob $4
echo non_uniformity $non_uniformity
echo sampling_scheme $sampling_scheme
echo line_search $line_search
echo init_previous_step_size $init_previous_step_size
echo use_warm_start $use_warm_start
python3.6 supervised_OCR.py --numberEpochsFW $numberEpochsFW --numberEpochsCKN $numberEpochsCKN --regParBCFW $regParBCFW --gpu $gpu --scaler $2 --lr $lr --npass $3 --size-patch $size_patch --zero-prob $4 --non-uniformity $non_uniformity --sampling-scheme $sampling_scheme --line-search $line_search --init-previous-step-size $init_previous_step_size --use-warm-start $use_warm_start --predictor $predictor --numberEpochsUnsupCKN $numberEpochsUnsupCKN --benchmark $benchmark --regParSDCA $1