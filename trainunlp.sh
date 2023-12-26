#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=expunlp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_unlp.yaml
model=model/CyCTRablaunlp.py
mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp trainunlp.sh trainunlp.py ${config} ${model} ${exp_dir}

python3 -u trainunlp.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
