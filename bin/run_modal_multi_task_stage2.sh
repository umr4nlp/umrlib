#!/bin/bash

# Modal Dependency Parsing STAGE 2

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output modal_home max_seq_length venv cuda cvd" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
modal_home=$(readlink -m $3)
max_seq_length=$4
venv=$5
cuda=$6
cvd=$7

# constant
model=bert-base-cased
best_on_dev_output_dir=output_modal/multi_task/dev_output

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $modal_home

echo -e "\Modal Multi-Task Stage 2"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model (fixed): $model"
echo -e " Max Seq. Length: $max_seq_length\n"

python parse.py --evaluate_pred_edges \
  --model $model \
  --classifier multi_task \
  --data_type modal \
  --dev_file $input  \
  --test_file $input \
  --gold_test_file $input \
  --max_seq_length $max_seq_length \
  --output_dir $best_on_dev_output_dir \
  --best_on_dev_output_dir $best_on_dev_output_dir

cp "${modal_home}/output_modal/multi_task/dev_output/bert-base-cased_multitask_test_stage2_auto_nodes.txt" ${output}

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
