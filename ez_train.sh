#!/bin/bash

# example usage: ./ez_train.sh 0.10 inf out_dir

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
set -e

eps=$1 # eps for attack
lp=$2 # lp norm for attack, either of: ['2', 'inf']
ckpt_dir=$3 # out dir to save model in

./imagenet_resnet.py --gpu 0,1,2,3,4,5,6,7 --checkpoint-dir $ckpt_dir --image-size 224 --eps $eps --lp $lp

