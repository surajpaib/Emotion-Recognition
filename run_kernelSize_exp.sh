#!/bin/sh

## Baseline has 3,3 -- i.e. kernel size 3 for block 1 and 3 for block 2


# 5,3
python train.py --model_config "./config/kernelSize_exp/53kernel.json" --wandb 1

# 5,5
python train.py --model_config "./config/kernelSize_exp/55kernel.json" --wandb 1

# 7,3
python train.py --model_config "./config/kernelSize_exp/73kernel.json" --wandb 1

# 7,5
python train.py --model_config "./config/kernelSize_exp/75kernel.json" --wandb 1
