#!/bin/sh

# Baseline (4 blocks)
baseline_config_path="./config/Baseline.json"
python train.py --model_config $baseline_config_path --wandb 1

# 5 blocks
python train.py --model_config "./config/nLayers_exp/5blocks.json"--wandb 1

# 6 blocks
python train.py --model_config "./config/nLayers_exp/6blocks.json" --wandb 1

# 7 blocks
python train.py --model_config "./config/nLayers_exp/7blocks.json" --wandb 1

# 8 blocks
python train.py --model_config "./config/nLayers_exp/8blocks.json" --wandb 1

# 9 blocks
python train.py --model_config "./config/nLayers_exp/9blocks.json" --wandb 1