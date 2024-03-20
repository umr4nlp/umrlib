# UMR Parsing Platform
by Jayeol Chun

## Setup
The platform has limited requirements but preparing inputs for and/or running the pipeline models may require additional venv and/or dependencies.
* Consult each model's platform for details
* see [here](models/README.md) for details

### Dev Environment
* Python 3.10
* (WebUI only) PyTorch 2.0.1
* (WebUI only) Gradio 4.21.0

## UMR v1.0
* [https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5198](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5198)

## Preprocessing
[script](./scripts/preprocess_umr_en_v1.0.py) consists of 2 stages:
1. Cleanup to fix obvious annotation mistakes + ensure labeling consistency
   * 0002 and 0004 stored in `pruned` subdir for future evaluation
2. Prepare model inputs (except CDLM, which requires MDP or TDP to first identify events; see [here](scripts/prepare_cdlm_inputs.py))

## RUN
To set up each of the pipeline models, see [here](models/README.md)
* shell scripts in `bin` are meant for WebUI use

## Evaluate
[script](./scripts/evaluate_umr.py) computes per-document scores + Macro F1 across all documents
* [UMR Evaluation Metric](https://github.com/sxndqc/UMR-Inference) by Haibo Xun and Nianwen Xue

## WebUI
```shell
$ python launch_webui.py 

options:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  listen port
  --share               whether to have a shareable link
  --debug               whether to log at DEBUG level
```
* default port number: 7860
* assumes `pyenv` as virtualenv manager



