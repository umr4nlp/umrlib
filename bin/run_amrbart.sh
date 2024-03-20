#!/bin/bash

# AMRBART

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output amrbart_home model venv cuda cvd" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
amrbart_home=$(readlink -m $3)
model=$4
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

# this is AMRBART root, should move further into `fine-tune`
# but must copy data to `examples` before that
cd $amrbart_home

cp $input $amrbart_home/examples

cd fine-tune

echo -e "\nRunning AMRBART"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model: $model\n"

bash inference-amr.sh "$model"

# copy output
cp outputs/Infer-examples-AMRBART-large-AMRParing-bsz16-lr-1e-5-UnifiedInp/val_outputs/test_generated_predictions_0.txt $output

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
