# Models
* CUDA 11.7

## A] AMR Parsing
### 1. [LeakDistill](https://github.com/SapienzaNLP/LeakDistill)
* default setup requires `transformers==4.21.2`
* also supports [SPRING](https://github.com/SapienzaNLP/spring)
* contact the authors for LeakDistill checkpoints
  * each checkpoint should be stored in `checkpoints` sub-dir
* cmd (from model root):
```shell
$ python bin/predict_amrs.py \
  --config ./configs/config_leak_distill.yaml  \
  --datasets ../../DATA/umr-v1.0-en/prep/amrs.txt \
  --gold-path ./data/tmp/gold.txt \
  --pred-path ../../EXP/umr-v1.0-en/tmp/LeakDistill_best-smatch_checkpoint_12_0.8534.pt.amr.txt \
  --beamsize 10 \
  --checkpoint ./checkpoints/best-smatch_checkpoint_12_0.8534.pt \
  --device cuda
```
* [paper](https://arxiv.org/abs/2306.13467)

### 2. IBM [transition-amr-parser](https://github.com/IBM/transition-amr-parser)
* default setup requires CUDA-11.7 for `torch-scatter`
* we always run an ensemble of 3 different seeds
  *  see [here](https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py) for MBSE
* cmd (from model root):
```shell
$ python src/transition_amr_parser/parse.py \
  -i ../../DATA/umr-v1.0-en/prep/toks.txt  \
  -o ../../EXP/umr-v1.0-en/tmp/ibm_amr2joint_ontowiki2_g2g_seeds42:43:44.amr.txt \
  -c {SEED42_CKPT}:{SEED43_CKPT}:{SEED44_CKPT} \
  --beam 10 \
  --batch-size 32 \
  --jamr --no-isi 
```
* refer to [README](https://github.com/IBM/transition-amr-parser/blob/master/README.md) for the full list of papers

### 3. [AMRBART](https://github.com/goodbai-nlp/AMRBART)
* we use a forked version whose `inference-amr.sh` always overwrites without asking for confirmation
  * [https://github.com/goodbai-nlp/AMRBART](https://github.com/goodbai-nlp/AMRBART)
* our modified [version](https://github.com/BERT-Brandeis/AMRBART)
  * always overwrites without asking for confirmation
* cmd (from `fine-tune` subdir):
```shell
$ bash inference-amr.sh "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2"
```
* [paper](https://aclanthology.org/2022.acl-long.415/)

### 4. [BLINK](https://github.com/facebookresearch/BLINK)
* we use `blinkify.py` inside `LeakDistill`
  * see their [instructions](https://github.com/SapienzaNLP/LeakDistill?tab=readme-ov-file#evaluation) for installation
* cmd (from model root):
```shell
$ PYTHONPATH=BLINK python bin/blinkify.py \
  --datasets ../../EXP/umr-v1.0-en/tmp/LeakDistill_best-smatch_checkpoint_12_0.8534.pt.amr.txt \
  --out ../../EXP/umr-v1.0-en/tmp/LeakDistill_best-smatch_checkpoint_12_0.8534.pt.blink.amr.txt \
  --device cuda \
  --blink-models-dir BLINK/models/
```
* [paper](https://arxiv.org/abs/1911.03814)

### 5. [LEAMR](https://github.com/ablodge/leamr)
LeakDistill or Spring needs a separate alinger module if used alone
* requires [JAMR Aligner](https://github.com/jflanigan/jamr)
  * Java, scala and sbt can be easily installed with [SDKMAN!](https://sdkman.io/)
    * Java: openjdk "1.8.0_392" (Temurin)
    * Scala: 2.11.12
    * SBT: 1.0.2
  * these versions must be reflected in some of the scripts
  * references:
    1. https://github.com/jflanigan/jamr/issues/44
    2. https://github.com/DreamerDeo/JAMR
* requires [ISI Aligner](https://github.com/melanietosik/string-to-amr-alignment)
  * mgiza++
* requires [neuralcoref](https://github.com/huggingface/neuralcoref)
  * must be built from source
* cmd (from model root):
```shell
$ ./scripts/run_aligners.sh ../../EXP/umr-v1.0-en/tmp/LeakDistill_best-smatch_checkpoint_12_0.8534.pt.blink.amr.txt
```
* [paper](https://aclanthology.org/2021.acl-long.257/)

## 2] Modal Dependency Parsing

MDP consists of 2 stages:
1. Stage1 identifies Events + Conceivers
2. Stage2 generates modal edges along with the labels

### 1. modal baseline
* as far as we know, this model is not available to the public
  * contact the author for details
* Stage 1 + 2 cmd (from model root):
```shell
$ ./run_multi_task_modal.sh ../../DATA/umr-v1.0-en/prep/docs.txt
```
* [paper](https://aclanthology.org/2021.acl-long.122/)

### 2. [mdp_prompt](https://github.com/Jryao/mdp_prompt)
* must be trained on [MDG corpus](https://github.com/Jryao/modal_dependency/tree/main/data)
  * contact the author for the version with the sentences
* `end2end` suppots both stages
* our modified [version](https://github.com/BERT-Brandeis/mdp_prompt)
  * original code is missing a function `generate_e_conc_from_bio_tag` and `tokenize_doc_no_overlap`
    * `generate_e_conc_from_bio_tag` has been added by contacting the author
    * `tokenize_doc_no_overlap` is unused and any references to it are removed
* Stage 1 cmd (from `src` sub-dir):
```shell
$ python parse.py \
  --model bert-large-cased \
  --language eng \
  --data_type modal \
  --classifier end2end \
  --input_plain ../../DATA/umr-v1.0-en/prep/docs.txt \
  --max_seq_length 384 \
  --encoding_method overlap \
  --num_labels 5 \
  --parse_stage1  \
  --output_dir ./outputs/end2end/ \
  --outmodel_name gen_end2end \
  --eval_batch_size 6 \
  --seed 42 \
  --extract_conc \
  --extract_event
```
* Stage 2 cmd (from `src` sub-dir):
```shell
$ python parse.py \
  --model bert-large-cased \
  --language eng \
  --data_type modal \
  --classifier end2end \
  --input_file ./outputs/end2end/bert-large-cased_preds_stage1.txt  \
  --max_seq_length 384 \
  --encoding_method overlap \
  --num_labels 5 \
  --parse_stage2 \
  --output_dir ./outputs/end2end/ \
  --outmodel_name gen_end2end \
  --eval_batch_size 6 \
  --seed 42
```
* [paper](https://aclanthology.org/2022.naacl-main.211/)

## 3] Temporal Dependency Parsing

TDP consists of 2 stages:
1. Stage1 identifies Events + Time Expressions
2. Stage2 generates temporal edges along with the labels

### 1. temporal baseline
* only supports Stage 1
* as far as we know, this model is not available to the public
  * contact the author for details
* Stage 1 cmd (from model root):
```shell
$ python parse.py \
  --model xlm-roberta-base \
  --input_plain ../../DATA/umr-v1.0-en/prep/docs.txt \
  --data_type temporal_time \
  --language eng \
  --max_seq_length 384 \
  --encoding_method overlap \
  --parse_stage1  \
  --classifier pipeline_stage1 \
  --extract_event \
  --output_dir ./outputs/pipeline_stage1 \
  --outmodel_name temporal_stage1
```
* [paper](https://aclanthology.org/2020.emnlp-main.432/) 
  * the paper references [this ranking model](https://github.com/yuchenz/tdp_ranking) which is different from this baseline

### 2. [thyme_tdg](https://github.com/Jryao/thyme_tdg/tree/master)
* requires transformers later than `4.21.0` (tested with `4.36.1`)
* only supports Stage 2
* general-domain version must be trained on [TDG corpus](https://github.com/Jryao/temporal_dependency_graphs_crowdsourcing/tree/master/tdg_data)
* our modified [version](https://github.com/BERT-Brandeis/thyme_tdg)
  * can process edge annotations without any parent information
    * ex) `2_16_16	Timex` 
  * original code's general-domain parser only iterates over a single test example
* Stage 2 cmd (from model root):
```shell
$ python parsers/run_parser.py \
  --input_file ../temporal/outputs/pipeline_stage1/xlm-roberta-base_stage1.txt  \
  --model_dir ./outputs/general_128/  \
  --output_dir ./outputs/general_128/  \
  --output_file thyme_tdg.stage2.txt
```
* [paper](https://aclanthology.org/2023.clinicalnlp-1.25/)

#### NOTE
* `temporal_baseline` accepts data format which is different from what `thyme_tdg` accepts
  * use [this script](../scripts/convert_temporal2thyme.py) to convert the Stage 1 output to Stage 2 input

## 4] Coreference

### 1. [CDLM](https://github.com/aviclu/CDLM/tree/main/cross_encoder) for event-coref
* default setup requires `transformers==3.0.0`
* trained on [ECB+ corpus](http://www.newsreader-project.eu/results/data/the-ecb-corpus/)
  * then populate `span_repr_path`, `span_scorer_path`, `model_path` accordingly
* relevant fields (+more) to be modified in `configs/config_pairwise_long_reg_span.json`, along with default values used in our pipeline
  * `gpu_num` : [0,1,2,3]
  * `split` : cdlm 
  * `mention_type` : events (cdlm does support entity linking)
  * `batch_size` : 128
  * `data_folder` : ../../EXP/umr-v1.0-en/tmp
* cmd (from model root):
```shell
$ python predict_long.py --config configs/config_pairwise_long_reg_span.json
```
* [paper](https://aclanthology.org/2021.findings-emnlp.225/)

### 2. [caw-coref](https://github.com/kareldo/wl-coref) (and [wl-coref](https://github.com/vdobrovolskii/wl-coref))
* save [pretrained model](https://www.dropbox.com/scl/fi/yhtf9h9sml91qs8sazdx6/roberta_-e20_2023.09.08_16.14-_release.pt?rlkey=kf60obnpqjyelsg7019g92kv5&dl=0) as `roberta.pt` at model root
  * or elsewhere, then spedcify after `--weights` flag
* cmd (from model root):
```shell
$ python predict.py roberta {INPUT.jsonlines} {OUTPUT.jsonlines} --weights roberta.pt   
```
* [paper](https://arxiv.org/abs/2310.06165)

