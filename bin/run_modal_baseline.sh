#!/bin/bash

# Modal Baseline

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 5 ] && echo "e.g. $0 input stage1_output stage2_output modal_home max_seq_length venv" && exit 1

input=$(readlink -m $1)
stage1_output=$(readlink -m $2)
stage2_output=$(readlink -m $3)
modal_home=$(readlink -m $4)
max_seq_length=$5
venv=$6

# constant
model=bert-base-cased
best_on_dev_output_dir=output_modal/multi_task/dev_output

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $modal_home

echo -e "\Modal Baseline"
echo -e " Input: $input"
echo -e " Stage 1 Output: $stage1_output"
echo -e " Stage 2 Output: $stage2_output"
echo -e " Model (fixed): $model"
echo -e " Max Seq. Length: $max_seq_length\n"

echo -e "\nModal Baseline Stage 1"
python parse.py \
  --model $model \
  --classifier multi_task \
  --data_type modal \
  --dev_file $input \
  --test_file  $input \
  --gold_test_file  $input \
  --max_seq_length $max_seq_length \
  --output_dir $best_on_dev_output_dir \
  --best_on_dev_output_dir $best_on_dev_output_dir

echo -e "\nModal Baseline Stage 2"
python parse.py --evaluate_pred_edges \
  --model $model \
  --classifier multi_task \
  --data_type modal \
  --dev_file $best_on_dev_output_dir/bert-base-cased_multitask_test_stage1.txt \
  --test_file $best_on_dev_output_dir/bert-base-cased_multitask_test_stage1.txt \
  --gold_test_file $best_on_dev_output_dir/bert-base-cased_multitask_test_stage1.txt \
  --max_seq_length $max_seq_length \
  --output_dir $best_on_dev_output_dir \
  --best_on_dev_output_dir $best_on_dev_output_dir

cp "${modal_home}/output_modal/multi_task/dev_output/bert-base-cased_multitask_test_stage1.txt" $stage1_output
cp "${modal_home}/output_modal/multi_task/dev_output/bert-base-cased_multitask_test_stage2_auto_nodes.txt" $stage2_output

echo -e "Stage 1 Output: $stage1_output"
echo -e "Stage 2 Output: $stage2_output"

# done
cd -
echo -e "\nDone."
