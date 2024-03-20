#!/bin/bash

# mdp_prompt end2end STAGE 1 + 2

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 11 ] && echo "e.g. $0 input output mdp_home max_seq_length batch_size seed model_dir model_name venv cuda cvd" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
mdp_home=$(readlink -m $3)

max_seq_length=$4
batch_size=$5
seed=$6

model_dir=$(readlink -m $7)
model_name=$8

venv=$9
cuda=${10}
cvd=${11}

# constant
model=bert-large-cased
num_labels=5

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $mdp_home

echo -e "\nMDP-Prompt End2End"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model (fixed): $model"
echo -e " Num. Labels (fixed): $num_labels"
echo -e " Max Seq. Length: $max_seq_length"
echo -e " Eval Batch Size: $batch_size"
echo -e " Seed: $seed"
echo -e " Model Dir: $model_dir"
echo -e " Modle Name: $model_name\n"

echo -e "\nMDP-Prompt End2End Stage 1"
python parse.py \
 --model $model \
  --language eng \
  --data_type modal \
  --classifier end2end \
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

echo -e "\nMDP-Prompt End2End Stage 2"
python parse.py \
 --model $model \
 --language eng \
 --data_type modal \
 --classifier end2end \
 --input_file ./outputs/end2end/bert-large-cased_preds_stage1.txt  \
 --max_seq_length $max_seq_length \
 --encoding_method overlap \
 --num_labels $num_labels \
 --parse_stage2 \
 --output_dir $model_dir \
 --outmodel_name $model_name \
 --eval_batch_size $batch_size \
 --seed $seed

# copy the output to user-requested location
# if `output` is a file not dir, stage1 will be overwritten
cp ./outputs/end2end/bert-large-cased_preds_stage1.txt $output
cp ./outputs/end2end/bert-large-cased_preds_auto_nodes.txt $output

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
