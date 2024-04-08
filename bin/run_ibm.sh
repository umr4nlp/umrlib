#!/bin/bash

# IBM transition parser

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output ibm_home checkpoint batch_size beamsize venv [EXTRA]" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
ibm_home=$(readlink -m $3)

checkpoint=$4
batch_size=$5
beamsize=$6
venv=$7

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $ibm_home

echo -e "\nRunning IBM AMR Transition Parser"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model: $model"
echo -e " Checkpoint(s): $checkpoint"
echo -e " Batch Size: $batch_size"
echo -e " Beam Size: $beamsize\n"

python src/transition_amr_parser/parse.py \
 -i $input \
 -o $output \
 -c $checkpoint \
 --beam $beamsize \
 --batch-size $batch_size \
 --jamr --no-isi \
 ${@:8}

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
