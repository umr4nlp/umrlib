#!/bin/bash

# SPRING + LeakDistill

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 9 ] && echo "e.g. $0 input output sapienza_home config checkpoint beamsize venv cuda cvd [EXTRA]" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
sapienza_home=$(readlink -m $3)

config=$4
checkpoint=$5
beamsize=$6

venv=$7
cuda=$8
cvd=$9

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
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
    ${@:10}

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
