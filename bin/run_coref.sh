#!/bin/bash

# caw-coref or wl-coref

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 8 ] && echo -e "\n[!] e.g. \`$0\` input output coref_home model checkpoint venv cuda cvd" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
coref_home=$(readlink -m $3)
model=$4
checkpoint=$5
venv=$6
cuda=$7
cvd=$8

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $coref_home

echo -e "\nRunning coref"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model: $model"
echo -e " Checkpoint: $checkpoint\n"

# main call
python predict.py $model $input $output --weights $checkpoint

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
