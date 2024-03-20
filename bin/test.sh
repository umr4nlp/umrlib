#!/bin/bash

set -o errexit
set -o pipefail

#echo ${@:5}
for input in ${@:1}
do
  cat -e $input
#cp models/span_scorers_longformer_reg_method3_full_span_entities/checkpoint_8/${cdlm_name}_events_average_${threshold}_corpus_level.conll $output/${threshold}.cdlm.conll
#echo $input
done
