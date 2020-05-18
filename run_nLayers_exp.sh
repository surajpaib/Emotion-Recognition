#!/bin/sh

# Baseline (4 blocks)
baseline_config_path="./config/Baseline.json"
echo "> Training baseline model"
python train.py --model_config $baseline_config_path --wandb 0 --epochs 1

# 5 blocks
echo "> Training 5blocks model"
python train.py --model_config "./config/nLayers_exp/5blocks.json" --wandb 0 --epochs 1

# 6 blocks
echo "> Training 6blocks model"
python train.py --model_config "./config/nLayers_exp/6blocks.json" --wandb 0 --epochs 1

# 7 blocks
echo "> Training 7blocks model"
python train.py --model_config "./config/nLayers_exp/7blocks.json" --wandb 0 --epochs 1
