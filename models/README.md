# Models

## A] AMR Parsing
### 1. [LeakDistill](https://github.com/SapienzaNLP/LeakDistill)
* default setup requires `transformers==4.21.2`
* also supports [SPRING](https://github.com/SapienzaNLP/spring)
* contact the authors for LeakDistill checkpoints
* [paper](https://arxiv.org/abs/2306.13467)

### 2. IBM [transition-amr-parser](https://github.com/IBM/transition-amr-parser)
* default setup requires CUDA-11.7 for `torch-scatter`
* we always run an ensemble of 3 different seeds
  *  see [here](https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py) for MBSE
* refer to [README](https://github.com/IBM/transition-amr-parser/blob/master/README.md) for the full list of papers

### 3. [AMRBART](https://github.com/goodbai-nlp/AMRBART)
* we use a forked version whose `inference-amr.sh` always overwrites without asking for confirmation
  * [https://github.com/goodbai-nlp/AMRBART](https://github.com/goodbai-nlp/AMRBART)
* [paper](https://aclanthology.org/2022.acl-long.415/)
* our modified [version](https://github.com/BERT-Brandeis/AMRBART)
  * always overwrites without asking for confirmation

### 4. [BLINK](https://github.com/facebookresearch/BLINK)
* we use `blinkify.py` inside `LeakDistill`
  * see their [instructions](https://github.com/SapienzaNLP/LeakDistill?tab=readme-ov-file#evaluation) for installation
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
* [paper](https://aclanthology.org/2021.acl-long.257/)

## 2] Modal Dependency Parsing

MDP consists of 2 stages:
1. Stage1 identifies Events + Conceivers
2. Stage2 generates modal edges along with the labels

### 1. modal baseline
* as far as we know, this model is not available to the public
  * contact the author for details
* [paper](https://aclanthology.org/2021.acl-long.122/)

### 2. [mdp_prompt](https://github.com/Jryao/mdp_prompt)
* must be trained on [MDG corpus](https://github.com/Jryao/modal_dependency/tree/main/data)
  * contact the author for the version with the sentences
* `end2end` suppots both stages
* [paper](https://aclanthology.org/2022.naacl-main.211/)
* our modified [version](https://github.com/BERT-Brandeis/mdp_prompt)
  * original code is missing a function `generate_e_conc_from_bio_tag` and `tokenize_doc_no_overlap`
    * `generate_e_conc_from_bio_tag` has been added by contacting the author
    * `tokenize_doc_no_overlap` is unused and any references to it are removed

## 3] Temporal Dependency Parsing

TDP consists of 2 stages:
1. Stage1 identifies Events + Time Expressions
2. Stage2 generates temporal edges along with the labels

### 1. temporal baseline
* only supports Stage 1
* as far as we know, this model is not available to the public
  * contact the author for details
* [paper](https://aclanthology.org/2020.emnlp-main.432/) 
  * the paper references [this ranking model](https://github.com/yuchenz/tdp_ranking) which is different from this baseline

### 2. [thyme_tdg](https://github.com/Jryao/thyme_tdg/tree/master)
* requires transformers later than `4.21.0` (tested with `4.36.1`)
* only supports Stage 2
* general-domain version must be trained on [TDG corpus](https://github.com/Jryao/temporal_dependency_graphs_crowdsourcing/tree/master/tdg_data)
* [paper](https://aclanthology.org/2023.clinicalnlp-1.25/)
* our modified [version](https://github.com/BERT-Brandeis/thyme_tdg)
  * can process edge annotations without any parent information
    * ex) `2_16_16	Timex` 
  * original code's general-domain parser only iterates over a single test example

#### NOTE
* `temporal_baseline` accepts data format which is different from what `thyme_tdg` accepts
  * use [this script](../scripts/convert_temporal2thyme.py) to convert the Stage 1 output to Stage 2 input

## 4] Coreference

### 1. [CDLM](https://github.com/aviclu/CDLM/tree/main/cross_encoder) for event-coref
* default setup requires `transformers==3.0.0`
* [paper](https://aclanthology.org/2021.findings-emnlp.225/)

### 2. [caw-coref](https://github.com/kareldo/wl-coref)
* based on [wl-coref](https://github.com/vdobrovolskii/wl-coref) ([paper](https://aclanthology.org/2021.emnlp-main.605/)) which is also supported in this pipeline
* [paper](https://arxiv.org/abs/2310.06165)

