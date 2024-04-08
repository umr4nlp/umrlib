#!/bin/bash

# SPRING + LeakDistill

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output sapienza_home config checkpoint beamsize venv [EXTRA]" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
sapienza_home=$(readlink -m $3)

config=$4
checkpoint=$5
beamsize=$6
venv=$7

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $sapienza_home

echo -e "\nRunning Sapienza AMR Parser"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Config: $config"
echo -e " Checkpoint: $checkpoint"
echo -e " Beam Size: $beamsize\n"

# run inference
python bin/predict_amrs.py \
    --config $config \
    --datasets $input \
    --gold-path data/tmp/gold.txt \
    --pred-path $output  \
    --beamsize $beamsize \
    --checkpoint $checkpoint \
    --device cuda \
    ${@:8}

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
