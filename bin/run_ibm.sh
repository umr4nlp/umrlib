#!/bin/bash

# IBM transition parser

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 10 ] && echo "e.g. $0 input output ibm_home torch_home checkpoint batch_size beamsize venv cuda cvd [EXTRA]" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
ibm_home=$(readlink -m $3)
torch_home=$(readlink -m $4)

checkpoint=$5
batch_size=$6
beamsize=$7

venv=$8
cuda=$9
cvd=${10}

echo -e "Torch HUB: $torch_home"
export TORCH_HOME=$torch_home

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
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
echo -e " Batch Size: $batch_size"
echo -e " Beam Size: $beamsize\n"

python src/transition_amr_parser/parse.py \
 -i $input \
 -o $output \
 -c $checkpoint \
 --beam $beamsize \
 --batch-size $batch_size \
 --jamr --no-isi \
 ${@:11}

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
