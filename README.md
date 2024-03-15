# UMR Parsing Platform
by Jayeol Chun

## Setup
The platform has limited requirements but preparing inputs for and/or running the pipeline models may require additional venv and/or dependencies.
* Consult each model's platform for details
* This [README](models/README.md) may be a good starting point

### Dev Environment
* Python 3.10
* (WebUI only) PyTorch 2.0.1
* (WebUI only) Gradio 4.21.0

## UMR v1.0
Download from [this link](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5198)

## Preprocessing
1. We first apply [cleanup](scripts/cleanup_umr_en_v1.0.py) to fix obvious annotation mistakes + ensure labeling consistency
2. (optional) [Prune](scripts/prune_overlaps.py) any UMR data whose sentences are part of [AMR R3](https://catalog.ldc.upenn.edu/LDC2020T02)
3. (optional) [Split](scripts/split_long_documents.py) `english_umr-0004.txt` which has >140 sentences, making it too long for some pipeline models
4. [Prepare inputs](scripts/prepare_inputs.py) for various models down the pipeline (except CDLM, which requires MDP or TDP to first identify events; see [here](scripts/prepare_cdlm_inputs.py))

## RUN
To set up each of the pipeline models, see this [README](models/README.md)
* shell files in `bin` are meant to be used by WebUI

## Evaluate
* [UMR Evaluation Metric](https://github.com/sxndqc/UMR-Inference) by Haibo Xun and Nianwen Xue

## WebUI
```shell
$ python scripts/launch_webui.py 

options:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  listen port
  --share               whether to have a shareable link
  --debug               whether to log at DEBUG level
```
* default port number: 7860




