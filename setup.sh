#!/bin/bash

if compgen -G "data/*/" > /dev/null; then
    rm -R -- data/*/
fi

set -e

trap "exit 1" TERM
export TOP_PID=$$

mkdir -p data
cd data

function unzip_it() {
    if [ -f $1 ]; then
        unzip $1
    else
        echo 'Download` train_224_nat_slim.zip`, `train_224_robust_eps_0.005_lp_inf_slim.zip`, and `train_224_robust_eps_1.0_lp_2_slim.zip` from https://github.com/MadryLab/robust-resnet-release/releases/tag/v0.1 and into data/, then restart the script'
        kill -s TERM $TOP_PID
    fi
}

unzip_it train_224_nat_slim.zip
unzip_it train_224_robust_eps_0.005_lp_inf_slim.zip
unzip_it train_224_robust_eps_1.0_lp_2_slim.zip

mv train_224_robust_eps_1.0_slim train_224_robust_eps_1.0_lp_2_slim
mv train_224_robust_eps_0_lp_2_slim train_224_nat_slim

cd ..

pip install -r ./requirements.txt --user


