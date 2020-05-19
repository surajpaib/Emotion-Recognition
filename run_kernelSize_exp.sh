#!/bin/sh

## Baseline has 3,3 -- i.e. kernel size 3 for block 1 and 3 for block 2


# 5,3
echo "> Training 53kernel model"
python train.py --model_config "
" --wandb 0 --epochs 100

# 5,5
echo "> Training 55kernel model"
python train.py --model_config "./config/kernelSize_exp/55kernel.json" --wandb 0 --epochs 100

# 7,3
echo "> Training 73kernel model"
python train.py --model_config "./config/kernelSize_exp/73kernel.json" --wandb 0 --epochs 100

# 7,5
echo "> Training 75kernel model"
python train.py --model_config "./config/kernelSize_exp/75kernel.json" --wandb 0 --epochs 100
