#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=rifenet/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh trainunlabel.py ${config} ${exp_dir}
CUDA_VISIBLE_DEVICES=0,1 python3 -u  train.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
