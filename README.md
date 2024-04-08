# UMR Parsing Platform
by Jayeol Chun

## Setup
* all venvs configured via [pyenv](https://github.com/pyenv/pyenv)


* [webui](requirements/requirements_webui.txt) for WebUI
* [cdlm](requirements/requirements_cdlm.txt) for CDLM
  * see [official requirements](https://github.com/aviclu/CDLM/blob/main/cross_encoder/requirements.txt)
* [thyme](requirements/requirements_thyme.txt) for thyme_tdg
* [models](requirements/requirements_models.txt) for every other model
* see [Aligners](https://github.com/chunjy92/Aligners) for setting up LEAMR

Python 3.10 for WebUI; 3.8 for the rest

### System Configuration
All experiments performed under the following configuration:
* Ubuntu 20.04
* RTX 3090
* CUDA 11.7

## UMR v1.0 Corpus
* [https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5198](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5198)

## Preprocessing
[script](./scripts/preprocess_umr.py) consists of 2 stages:
1. Cleanup to fix obvious annotation mistakes + ensure labeling consistency
   * 0002 and 0004 stored in `pruned` subdir for future evaluation
2. Prepare model inputs (except CDLM, which requires MDP to first identify events; see [here](scripts/prepare_cdlm_inputs.py))

* can be used to preprocess arbitrary text input if it follows UMR-style annotations

## Evaluate
### UMR
[this script](./scripts/evaluate_umr.py) computes per-document AnCast++ scores + Macro F1 across all documents
* [AnCast++](https://github.com/sxndqc/ancast)
### AMR (or UMR Sentence Graph)
* [SMATCH](./scripts/compute_smatch.py)
* [AnCast](./scripts/evaluate_umr.py), part of AnCast++, provides Sentence Graph Evaluation

## WebUI
```shell
$ python launch_webui.py 
```



