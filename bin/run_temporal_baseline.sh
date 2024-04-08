#!/bin/bash

# temporal baseline STAGE 1

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 12 ] && echo "e.g. $0 input output temporal_home model_dir model_name clf_model pipeline data_type max_seq_length batch_size seed venv" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
temporal_home=$(readlink -m $3)

model_dir=$(readlink -m $4)
model_name=$5
clf_model=$6
pipeline=$7
data_type=$8

max_seq_length=$9
batch_size=${10}
seed=${11}
venv=${12}

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $temporal_home

echo -e "\nRunning Temporal Baseline"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model: $clf_model"
echo -e " Model Dir: $model_dir"
echo -e " Model Name: $model_name"
echo -e " Pipeline: $pipeline"
echo -e " Data Type: $data_type"
echo -e " Max Seq. Length: $max_seq_length"
echo -e " Eval Batch Size: $batch_size"
echo -e " Seed: $seed\n"

python parse.py \
  --model $clf_model \
  --input_plain $input \
  --data_type $data_type \
  --language eng \
  --max_seq_length $max_seq_length \
  --encoding_method overlap \
  --parse_stage1  \
  --classifier pipeline_$pipeline \
  --extract_event \
  --output_dir $model_dir \
  --outmodel_name $model_name \
  --eval_batch_size $batch_size \
  --seed $seed \

cp "${model_dir}/${clf_model}_${pipeline}.txt" $output

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
