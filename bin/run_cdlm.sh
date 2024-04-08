#!/bin/bash

# CDLM

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 5 ] && echo "e.g. $0 output cdlm_home config cdlm_name venv" && exit 1

output=$(readlink -m $1)
cdlm_home=$(readlink -m $2)
config=$3
cdlm_name=$4 # should be fixed, as `cdlm`
venv=$5

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $cdlm_home

echo -e "\nRunning CDLM"
echo -e " Config: $config"
echo -e " Data Prefix: $cdlm_name"
echo -e " Output: $output\n"

python predict_long.py --config $config

# copy the output to user-requested location
thresholds="0.5 0.55 0.6 0.65 0.7"
for threshold in $thresholds
do
cp models/span_scorers_longformer_reg_method3_full_span_entities/checkpoint_8/${cdlm_name}_events_average_${threshold}_corpus_level.conll $output/cdlm.${threshold}.conll
done

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."

