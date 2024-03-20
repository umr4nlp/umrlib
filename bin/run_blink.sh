#!/bin/bash

# BLINK

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output blink_home blink_models venv cuda cvd" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
blink_home=$(readlink -m $3)
blink_models=$(readlink -m $4)
venv=$5
cuda=$6
cvd=$7

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $blink_home

echo -e "\nRunning BLINK"
echo -e " Input: $input"
echo -e " Output: $output\n"

# BLINK should be present locally
export PYTHONPATH=BLINK

python bin/blinkify.py \
    --datasets $input \
    --out $output \
    --device cuda \
    --blink-models-dir ${blink_models}/

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
