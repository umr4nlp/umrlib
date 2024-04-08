#!/bin/bash

# mdp_prompt

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 12 ] && echo "e.g. $0 input stage1_output stage2_output mdp_prompt_home model_dir model_name clf_model model_type max_seq_length batch_size seed venv" && exit 1

input=$(readlink -m $1)
stage1_output=$(readlink -m $2)
stage2_output=$(readlink -m $3)
mdp_prompt_home=$(readlink -m $4)

model_dir=$(readlink -m $5)
model_name=$6
clf_model=$7
model_type=$8

max_seq_length=$9
batch_size=${10}
seed=${11}
venv=${12}

# constant
num_labels=5

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $mdp_prompt_home

echo -e "\nMDP Prompt"
echo -e " Input: $input"
echo -e " Stage1 Output: $stage1_output"
if [ ${model_type} == "end2end" ]
then
  echo -e " Stage2 Output: $stage2_output"
fi
echo -e " Model: $clf_model"
echo -e " Model Dir: $model_dir"
echo -e " Model Name: $model_name"
echo -e " Model Type: $model_type"
echo -e " Max Seq. Length: $max_seq_length"
echo -e " Eval Batch Size: $batch_size"
echo -e " Num. Labels (fixed): $num_labels"
echo -e " Seed: $seed\n"

echo -e "\nMDP Prompt Stage 1"
python parse.py \
  --model $clf_model \
  --language eng \
  --data_type modal \
  --classifier $model_type \
  --input_plain $input \
  --max_seq_length $max_seq_length \
  --encoding_method overlap \
  --num_labels $num_labels \
  --parse_stage1  \
  --output_dir $model_dir \
  --outmodel_name $model_name \
  --eval_batch_size $batch_size \
  --seed $seed \
  --extract_conc \
  --extract_event

cp $model_dir/$(basename $clf_model)_stage1.txt $stage1_output

if [ ${model_type} == "end2end" ]
then
  echo -e "\nMDP Prompt Stage 2"
  python parse.py \
   --model $clf_model \
   --language eng \
   --data_type modal \
   --classifier $model_type \
   --input_file $model_dir/$(basename $clf_model)_stage1.txt  \
   --max_seq_length $max_seq_length \
   --encoding_method overlap \
   --num_labels $num_labels \
   --parse_stage2 \
   --output_dir $model_dir \
   --outmodel_name $model_name \
   --eval_batch_size $batch_size \
   --seed $seed

  cp $model_dir/$(basename $clf_model)_stage2.txt $stage2_output
fi

echo -e "Stage 1 Output: $stage1_output"
if [ ${model_type} == "end2end" ]
then
  echo -e " Stage2 Output: $stage2_output"
fi

# done
cd -
echo -e "\nDone."
