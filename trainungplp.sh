#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=expungplp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_ungplp.yaml
model=model/CyCTRablaungplp.py
mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp trainungplp.sh trainungplp.py ${config} ${model} ${exp_dir}

python3 -u trainungplp.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
