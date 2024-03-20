#!/bin/bash

# LEAMR

set -o errexit
set -o pipefail

echo -e "\nInside \`$0\`"
[ "$#" -lt 4 ] && echo "e.g. $0 input aligner_home cuda cvd" && exit 1

input=$(readlink -m $1)
aligner_home=$(readlink -m $2)
cuda=$3
cvd=$4

# venv is unnecessary as `run_aligners.sh` takes care of it itself
if [[ "$VIRTUAL_ENV" != "" ]]
then
  source deactivate
fi

echo -e "\n$cvd\n"
export $cvd

# set correct cuda version
source $HOME/scripts/enable_$cuda
nvcc --version

# aligner
cd $aligner_home

echo -e "\nRunning LEAMR Aligner"
echo -e " Input: $input\n"

./scripts/run_aligners.sh $input

# done
cd -
echo -e "\nDone."
