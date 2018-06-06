#!/bin/bash

set -e

trap "exit 1" TERM
export TOP_PID=$$

if [ -d data/ ]; then
    rm -rf data/
fi

mkdir data
cd data

function dl_it() {
    wget https://github.com/MadryLab/robust-features-code/releases/download/v0.2/$1
    unzip $1
}

dl_it train_224_nat_slim.zip
dl_it train_224_robust_eps_0.005_lp_inf_slim.zip
dl_it train_224_robust_eps_1.0_lp_2_slim.zip

mv train_224_robust_eps_1.0_slim train_224_robust_eps_1.0_lp_2_slim
mv train_224_robust_eps_0_lp_2_slim train_224_nat_slim

cd ..

pip install -r ./requirements.txt --user


