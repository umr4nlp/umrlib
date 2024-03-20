#!/bin/bash

# MBSE

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 4 ] && echo "e.g. $0 output mbse_home venv cuda cvd INPUTS1 INPUT2 ..." && exit 1

output=$(readlink -m $1)
mbse_home=$(readlink -m $2)
venv=$3
inputs=${@:4}

# activate venv
echo -e "\nPyenv VENV: $venv"
. $PYENV_ROOT/versions/$venv/bin/activate
python -V

cd $mbse_home

echo -e "\nRunning MBSE"
for input in "$inputs"
do
    echo -e " Input: $input"
done
echo -e " Output: $output\n"

python scripts/mbse.py \
  --in-amrs $inputs \
  --out-amr $output

echo -e "Output: $output"

# done
cd -
echo -e "\nDone."
