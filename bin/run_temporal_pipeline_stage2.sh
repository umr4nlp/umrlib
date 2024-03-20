#!/bin/bash

# temporal baseline STAGE 2

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output temporal_home max_seq_length venv cuda cvd model_dir model_name" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)
temporal_home=$(readlink -m $3)

max_seq_length=$4

venv=$5
cuda=$6
cvd=$7

model_dir=$(readlink -m $8)
model_name=$9

# constant
model=xlm-roberta-base

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $temporal_home

echo -e "\nRunning Temporal Baseline"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model (fixed): $model"
echo -e " Max Seq. Length: $max_seq_length"
echo -e " Seed: $seed"
echo -e " Model Dir: $model_dir"
echo -e " Model Name: $model_name\n"

python parse.py \
  --model $model \
  --input_file $input \
  --data_type temporal_time \
  --language eng \
  --max_seq_length $max_seq_length \
  --encoding_method overlap \
  --parse_stage2_gold  \
  --classifier pipeline_stage2 \
  --output_dir $model_dir \
  --outmodel_name $model_name

cp "${model_dir}/xlm-roberta-base_stage2.txt" $output

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
