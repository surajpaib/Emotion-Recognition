#!/bin/sh




# 5 blocks
echo "> Training 5blocks model"
python train.py --model_config "./config/nLayers_exp/5blocks.json" --wandb 0 --epochs 100

# 6 blocks
echo "> Training 6blocks model"
python train.py --model_config "./config/nLayers_exp/6blocks.json" --wandb 0 --epochs 100

# 7 blocks
echo "> Training 7blocks model"
python train.py --model_config "./config/nLayers_exp/7blocks.json" --wandb 0 --epochs 100
