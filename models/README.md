# Models
Here we provide instructions/tips for interacting with the models in our pipeline

## AMR Parsing
### [LeakDistill](https://github.com/SapienzaNLP/LeakDistill)
* default setup requires `transformers==4.21.0`
* also supports [SPRING](https://github.com/SapienzaNLP/spring)
* contact the authors for LeakDistill checkpoints

#### [BLINK]((https://github.com/facebookresearch/BLINK))
* we use `blinkify.py` inside `LeakDistill`
  * see their [instructions](https://github.com/SapienzaNLP/LeakDistill?tab=readme-ov-file#evaluation) for installation

### IBM [transition-amr-parser](https://github.com/IBM/transition-amr-parser)
* default setup requires CUDA-11.7 for `torch-scatter`
* we always run an ensemble of 3 different seeds
  *  see [here](https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py) for MBSE

#### [LEAMR](https://github.com/ablodge/leamr)
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

## Modal Dependency Parsing

MDP consists of 2 stages:
1. Stage1 identifies Events + Conceivers
2. Stage2 generates modal edges along with the labels

### [mdp_prompt](https://github.com/Jryao/mdp_prompt)
* must be trained on [MDG corpus](https://github.com/Jryao/modal_dependency/tree/main/data)
  * contact the author for the version with the sentences
* `end2end` suppots both stages

### modal baseline
* as far as we know, this model is not available to the public
  * contact the author for details

## Temporal Dependency Parsing

TDP consists of 2 stages:
1. Stage1 identifies Events + Time Expressions
2. Stage2 generates temporal edges along with the labels

### [thyme_tdg](https://github.com/Jryao/thyme_tdg/tree/master)
* requires transformers later than `4.21.0`
* only supports Stage 2
* general-domain version must be trained on [TDG corpus](https://github.com/Jryao/temporal_dependency_graphs_crowdsourcing/tree/master/tdg_data)

### temporal baseline
* only supports Stage 1
* as far as we know, this model is not available to the public
  * contact the author for details

#### NOTE
* `temporal_baseline` accepts data format which is different from what `thyme_tdg` accepts
  * use [this script](../scripts/convert_temporal2thyme.py) to convert the Stage 1 output to Stage 2 input

## Coreference

### [CDLM](https://github.com/aviclu/CDLM/tree/main/cross_encoder) for event-coref
* default setup requires `transformers==3.0.0`

### [caw-coref](https://github.com/kareldo/wl-coref)
* based on [wl-coref](https://github.com/vdobrovolskii/wl-coref) which is also supported in this pipeline

