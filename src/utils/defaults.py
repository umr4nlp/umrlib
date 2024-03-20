#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/5/24 5:06 AM
"""default paths"""
import os
import platform
import socket

from utils import consts as C

### system
PLATFORM = platform.system()
HOSTNAME = socket.gethostname()

### CORE
HOME = os.path.expanduser('~')
PROJECT = os.getcwd()
if C.SCRIPTS in os.path.basename(PROJECT):
  PROJECT = os.path.abspath(os.path.join(PROJECT, os.pardir))

if PLATFORM == 'Darwin':
  if 'MacStudio' in HOSTNAME:
    _BMRP_HOME = "/Volumes/SamsungSSD/Research/B-MRP"
  else:
    _BMRP_HOME = os.path.join(HOME, "Documents/B-MRP")
  DATA = os.path.join(_BMRP_HOME, "DATA")
  UMR_EVAL = os.path.join(_BMRP_HOME, "UMR-Inference")
  ALIGNERS = C.NOT_FOUND
else:
  if 'omics' in HOSTNAME:
    DATA = os.path.join(HOME, "misc/lan/DATA")
    ALIGNERS = os.path.join(HOME, "misc/lan/Aligners")
    UMR_EVAL = os.path.join(HOME, "misc/lan/UMR-Inference")
  else:
    _BMRP_HOME = os.path.join(HOME, "Documents/lan/B-MRP")
    DATA = os.path.join(_BMRP_HOME, "DATA")
    ALIGNERS = os.path.join(_BMRP_HOME, "Aligners")
    UMR_EVAL = os.path.join(_BMRP_HOME, "UMR-Inference")

DATA = DATA if os.path.exists(DATA) else C.NOT_FOUND
ALIGNERS = ALIGNERS if os.path.exists(ALIGNERS) else C.NOT_FOUND # technically LEAMR
UMR_EVAL = UMR_EVAL if os.path.exists(UMR_EVAL) else C.NOT_FOUND
TORCH_HUB = os.path.join(HOME, '.cache/torch')

# corpora
UMR_EN = os.path.join(DATA, 'umr-v1.0/english')
UMR_EN = UMR_EN if os.path.exists(UMR_EN) else C.NOT_FOUND
AMR_R3 = os.path.join(DATA, 'AMR_R3_LDC2020T02')
AMR_R3 = AMR_R3 if os.path.exists(AMR_R3) else C.NOT_FOUND
MDG = os.path.join(DATA, 'modal_dependency')
MDG = MDG if os.path.exists(MDG) else C.NOT_FOUND
TDG = os.path.join(DATA, 'temporal_dependency_graphs_crowdsourcing')
TDG = TDG if os.path.exists(TDG) else C.NOT_FOUND

# local dirs
PROJ_DATA = os.path.join(PROJECT, C.DATA)
PROJ_EXP = os.path.join(PROJECT, C.EXP)
PROJ_MODELS = os.path.join(PROJECT, C.MODELS)

# model home
SAPIENZA = os.path.join(PROJ_MODELS, C.LEAK_DISTILL)
SAPIENZA = SAPIENZA if os.path.exists(SAPIENZA) else C.NOT_FOUND
IBM = os.path.join(PROJ_MODELS, C.IBM_PARSER)
IBM = IBM if os.path.exists(IBM) else C.NOT_FOUND
AMRBART = os.path.join(PROJ_MODELS, C.AMRBART)
AMRBART = AMRBART if os.path.exists(AMRBART) else C.NOT_FOUND
MDP_BASELINE = os.path.join(PROJ_MODELS, C.MODAL)
MDP_BASELINE = MDP_BASELINE if os.path.exists(MDP_BASELINE) else C.NOT_FOUND
MDP_PROMPT = os.path.join(PROJ_MODELS, C.MDP_PROMPT, 'src')
MDP_PROMPT = MDP_PROMPT if os.path.exists(MDP_PROMPT) else C.NOT_FOUND
TDP_BASELINE = os.path.join(PROJ_MODELS, C.TDP)
TDP_BASELINE = TDP_BASELINE if os.path.exists(TDP_BASELINE) else C.NOT_FOUND
THYME_TDG = os.path.join(PROJ_MODELS, C.THYME_TDG)
THYME_TDG = THYME_TDG if os.path.exists(THYME_TDG) else C.NOT_FOUND
CDLM = os.path.join(PROJ_MODELS, C.CDLM)
CDLM = CDLM if os.path.exists(CDLM) else C.NOT_FOUND
WL_COREF = os.path.join(PROJ_MODELS, C.WL_COREF)
WL_COREF = WL_COREF if os.path.exists(WL_COREF) else C.NOT_FOUND
CAW_COREF = os.path.join(PROJ_MODELS, C.CAW_COREF)
CAW_COREF = CAW_COREF if os.path.exists(CAW_COREF) else C.NOT_FOUND

# model inputs
TOKS_TXT = f'{C.TOKS}.{C.TXT}' # input to ibm transition amr parser
AMRS_TXT = f'{C.AMRS}.{C.TXT}' # input to sapienza
AMRS_JSONL = f'{C.AMRS}.{C.JSONL}' # input to AMRBART
DOCS_TXT = f'{C.DOCS}.{C.TXT}' # input to MDP + TDP stage 1
DOCS_TXT_TEMP = f'{C.DOCS}_%s.{C.TXT}'  # placeholder for split size
COREF_JSONLINES = f'{C.COREF}.{C.JSONLINES}' # input to (and output of) coref

# mappings
DOC2SNT_MAPPING_JSON = f'{C.DOC2SNT_MAPPING}.{C.JSON}'
PRUNED_MAPPING_JSON = f"{C.PRUNED_MAPPING}.{C.JSON}"
SPLIT_MAPPING_JSON = f"{C.SPLIT_MAPPING}.{C.JSON}"

# fixed model outputs
UDPIPE_JSON = f'{C.UDPIPE}.{C.JSON}' # output
CDLM_JSON = f'{C.CDLM}.{C.JSON}' # doc input
CDLM_EVENTS_JSON = f'{C.CDLM}_events.{C.JSON}' # event input

# logs
PRP_LOG = f'{C.PRP}.{C.LOG}' # scripts/preprocess_umr_en_v1.0.py
SAPIENZA_LOG = f'{C.SAPIENZA}.{C.LOG}' # bin/run_sapienza.sh
IBM_LOG = f'{C.IBM}.{C.LOG}' # bin/run_ibm.sh
AMRBART_LOG = f'{C.AMRBART}.{C.LOG}' # bin/run_amrbart.sh
BLINK_LOG = f'{C.BLINK}.{C.LOG}' # bin/run_blink.sh
LEAMR_LOG = f'{C.LEAMR}.{C.LOG}' # bin/run_leamr.sh
MBSE_LOG = f'{C.MBSE}.{C.LOG}' # bin/run_mbse.sh
MDP_BASELINE_LOG = f'{C.MODAL_MULTI_TASK}.{C.LOG}' # bin/run_modal_multi_task*.sh
MDP_PROMPT_LOG = f'{C.MDP_PROMPT}.{C.LOG}' # bin/run_mdp_prompt*.sh
MERGE_LOG = f'{C.MERGE}.{C.LOG}' # scripts/merge_stage1s.py
PREP_THYME_TDG_LOG = f'{C.PREP}_{C.THYME_TDG}.{C.LOG}' # scripts/prepare_thyme_tdg_inputs.py
TDP_BASELINE_LOG = f'{C.TDP}.{C.LOG}' # bin/run_temporal_pipeline*.sh
TDP_THYME_LOG = f'{C.THYME_TDG}.{C.LOG}' # bin/run_thyme_tdg.sh
PREP_CDLM_LOG = f'{C.PREP}_{C.CDLM}.{C.LOG}' # scripts/prepare_cdlm_inputs.py
CDLM_LOG = f'{C.CDLM}.{C.LOG}' # scripts/prepare_cdlm_inputs.py
COREF_LOG = f'{C.COREF}.{C.LOG}' # bin/run_coref.sh
UDPIPE_LOG = f'{C.UDPIPE}.{C.LOG}' # scripts/run_udpipe.py
CONV_LOG = f'{C.CONV}.{C.LOG}' # scripts/convert_amr2umr.py
EVAL_LOG = f'{C.EVAL}.{C.LOG}' # scripts/evaluate_umr.py
