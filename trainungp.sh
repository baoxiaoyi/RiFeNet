#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=expungp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_ungp.yaml
model=model/CyCTRablaungp.py
mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp trainungp.sh trainungp.py ${config} ${model} ${exp_dir}

python3 -u trainungp.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
