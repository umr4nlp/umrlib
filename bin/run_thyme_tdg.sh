#!/bin/bash

# thyme_tdg STAGE 2

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 7 ] && echo "e.g. $0 input output thyme_tdg_home model_dir venv cuda cvd" && exit 1

input=$(readlink -m $1)
output=$(readlink -m $2)

thyme_tdg_home=$(readlink -m $3)
model_dir=$(readlink -m $4)

venv=$5
cuda=$6
cvd=$7

# constant
output_file=thyme_tdg.stage2.txt

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $thyme_tdg_home

echo -e "\nRunning Thyme-TDG"
echo -e " Input: $input"
echo -e " Output: $output"
echo -e " Model Dir: $model_dir\n"

python parsers/run_parser.py \
  --input_file $input \
  --model_dir $model_dir \
  --output_dir $model_dir  \
  --output_file $output_file

cp "${model_dir}/${output_file}" $output

# done
cd -
echo -e "\nDone."
