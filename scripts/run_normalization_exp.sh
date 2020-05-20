#!/bin/sh

## Baseline has 3,3 -- i.e. kernel size 3 for block 1 and 3 for block 2


echo "> Training Baseline"
python train.py --model_config "./config/Baseline.json" --wandb 0 --epochs 100


echo "> Training Batch Norm model"
python train.py --model_config "./config/normalization_exp/BatchNorm.json" --wandb 0 --epochs 100


echo "> Training Instance model"
python train.py --model_config "./config/normalization_exp/InstanceNorm.json" --wandb 0 --epochs 100