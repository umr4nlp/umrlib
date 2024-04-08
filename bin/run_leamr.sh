#!/bin/bash

# LEAMR

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 2 ] && echo "e.g. $0 input aligner_home" && exit 1

input=$(readlink -m $1)
aligner_home=$(readlink -m $2)

# venv is unnecessary as `run_aligners.sh` takes care of it itself
if [[ "$VIRTUAL_ENV" != "" ]]
then
  source deactivate
fi

echo -e "\nCUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvcc --version

# aligner
cd $aligner_home

echo -e "\nRunning LEAMR Aligner"
echo -e " Input: $input\n"

./scripts/run_aligners.sh $input

# done
cd -
echo -e "\nDone."
