#! /usr/bin/subprocess_utils.PYTHON3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/16/24 5:02 PM
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import sys
sys.path.append('src')
import argparse
import logging
import multiprocessing as mp
import time
from datetime import datetime, timezone
from typing import List, Optional, Union

import gradio as gr
import torch

from __about__ import __title__, __version__, __uri__
from data import analysis, amr_utils
from utils import consts as C, defaults as D, io_utils, misc_utils, subprocess_utils

argparser = argparse.ArgumentParser(f'{__title__} v{__version__} WebUI Argparser')
argparser.add_argument('-p', '--port', type=int, default=7860, help='listen port')
argparser.add_argument('--share', action='store_true', help='whether to have a shareable link')
argparser.add_argument('--debug', action='store_true', help='whether to log at DEBUG level')
args, _ = argparser.parse_known_args()

logger = logging.getLogger(__name__)


# unmutable globals
PROJ_DATA, PROJ_EXP  = D.PROJ_DATA, D.PROJ_EXP
os.makedirs(PROJ_DATA, exist_ok=True)
os.makedirs(PROJ_EXP, exist_ok=True)

# conditions for updating choices
CONDITIONS = {
  # `x`: dir
  C.CORPUS: lambda x: ( len([xx for xx in x if xx.endswith(C.TXT)]) > 0 and \
                        all( xx.startswith('english_umr') for xx in x if xx.endswith(C.TXT)) ),
  C.TMP: lambda x: x == C.TMP,
  # `x`: input file
  D.TOKS_TXT: lambda x: x == D.TOKS_TXT,
  D.AMRS_TXT: lambda x: x == D.AMRS_TXT,
  D.AMRS_JSONL: lambda x: x == D.AMRS_JSONL,
  D.COREF_JSONLINES: lambda x: x == D.COREF_JSONLINES,
  D.DOCS_TXT: lambda x: C.DOCS in x and x.endswith(C.TXT),
  # `x`: intermdiate files that also feature in the conversion
  C.DOC2SNT_MAPPING: lambda x: x == D.DOC2SNT_MAPPING_JSON,
  C.SPLIT_MAPPING: lambda x: x == D.SPLIT_MAPPING_JSON,
  C.UDPIPE: lambda x: x == D.UDPIPE_JSON,
  # `x`: output file
  C.AMRS: lambda x: C.AMR in x,
  C.MDP: lambda x: C.MODAL in x or C.MDP_PROMPT in x,
  C.MDP_STAGE1: lambda x: (C.MODAL in x or C.MDP_PROMPT in x or C.MERGED in x) and C.STAGE1 in x,
  C.MDP_STAGE2: lambda x: (C.MODAL in x or C.MDP_PROMPT in x)  and C.STAGE2 in x,
  C.TDP: lambda x: C.TEMPORAL in x or C.THYME_TDG in x,
  C.TDP_STAGE1: lambda x: (C.TEMPORAL in x or C.THYME_TDG in x or C.MERGED in x) and C.STAGE1 in x,
  C.TDP_STAGE2: lambda x: (C.TEMPORAL in x or C.THYME_TDG in x) and C.STAGE2 in x,
  C.STAGE1: lambda x: C.STAGE1 in x,
  C.CDLM: lambda x: x.endswith(C.CDLM_CONLL),
  C.COREF: lambda x: C.COREF in x and x.endswith(C.JSONLINES),
  # by extension
  C.JSON: lambda x: x.endswith(C.JSON),
}

# general refresh behavior
def refresh(root, keys: Optional[Union[str, List]] = None, with_empty_option=False):
  if keys:
    if isinstance(keys, list):
      conditions = {k: CONDITIONS[k] for k in keys}
    else:
      conditions = {keys: CONDITIONS[keys]}
  else:
    conditions = CONDITIONS

  data = {k: set() for k in conditions}
  for subroot, dirs, files in os.walk(root):
    if C.CORPUS in conditions and conditions[C.CORPUS](files):
      data[C.CORPUS].add(subroot)

    for subdir in dirs:
      if C.TMP in conditions and conditions[C.TMP](subdir):
        data[C.TMP].add(os.path.join(subroot, subdir))

    for file in files:
      for k, condition in conditions.items():
        if condition(file):
          data[k].add(os.path.join(subroot, file))

  # allow for empty option
  if with_empty_option:
    out = {k: [""] + list(v) for k,v in data.items()}
  else:
    out = {k: list(v) for k, v in data.items()}
  return out

# event fns
def refresh_exp(keys: Union[str, List[str]]):
  if isinstance(keys, str):
    keys = [keys]
  choices = refresh(root=PROJ_EXP, keys=keys)
  if len(keys) == 1:
    return {'choices': choices[keys[0]], '__type__': 'update'}
  else:
    return [{'choices': choices[k], '__type__': 'update'} for k in keys]

def refresh_data(keys: Union[str, List[str]]):
  if isinstance(keys, str):
    keys = [keys]
  choices = refresh(root=PROJ_DATA, keys=keys)
  if len(keys) == 1:
    return {'choices': choices[keys[0]], '__type__': 'update'}
  else:
    return [{'choices': choices[k], '__type__': 'update'} for k in keys]

INIT_DATA_CHOICES = refresh(root=PROJ_DATA, with_empty_option=True)
INIT_EXP_CHOICES = refresh(root=PROJ_EXP, with_empty_option=True)

### venvs
PYENV_VENVS = []
PYENV_VENV_ROOT = os.path.join(os.getenv('PYENV_ROOT'), 'versions')
for x in sorted(os.listdir(PYENV_VENV_ROOT)):
  if not x.startswith('2') and not x.startswith('3'):
    PYENV_VENVS.append(x)

CUDAS = ['CPU']
for x in sorted(os.listdir('/usr/local')):
  if x.startswith('cuda-'):
    CUDAS.append(x)

GPUS, CVDS = ['CPU'], ['CPU']
for i in range(torch.cuda.device_count()):
  CVDS.append(f'CUDA_VISIBLE_DEVICES={i}')
  GPUS.append(torch.cuda.get_device_name(i))

def set_parsing_subroot_fn(parsing_subroot_value: str):
  subroot_fdir = os.path.join(PROJ_EXP, parsing_subroot_value)
  os.makedirs(subroot_fdir, exist_ok=True)
  tmp_fdir = os.path.join(subroot_fdir, C.TMP)
  os.makedirs(tmp_fdir, exist_ok=True)
  logger.info("Current Parsing Subroot: %s" % subroot_fdir)
  return (subroot_fdir,)*2 + (tmp_fdir,)*16

def define_long_running_event(button, fn=None, inputs=None, outputs=None,
                              fn2=None, inputs2=None, outputs2=None):
  # button.click(
  #   fn=lambda: {'interactive': False, '__type__': 'update'},
  #   outputs=button,
  # ).then(
  #   fn=fn, inputs=inputs, outputs=outputs
  # ).then(
  #   fn=fn2, inputs=inputs2, outputs=outputs2
  # ).then(
  #   fn=lambda: {'interactive': True, '__type__': 'update'},
  #   outputs=button,
  # )
  button.click(fn=fn, inputs=inputs, outputs=outputs)

def run_preprocessing(input_fpath, output_fpath, aggression, snt2tok, partition_strategy):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname = io_utils.get_dirname(output_fpath, mkdir=True)
  log_fpath = os.path.join(dirname, D.PRP_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.PYTHON,
    "scripts/preprocess_umr_en_v1.0.py",
    "-i",
    input_fpath,
    '-o',
    output_fpath,
    '--aggression',
    str(aggression),
    '--partition-strategy',
    partition_strategy,
    '--snt-to-tok',
    snt2tok,
  ]
  cmd = " ".join(cmd)
  logger.info("Corpus Cleanup CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_sapienza_amr_parser(input_fpath, output_fpath, model_home, model, checkpoint,
                            tokenize, beam, venv, cuda, cvd):
  # maybe `output_fpath` is a dir?
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg
  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.AMRS_TXT)

  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(
      output_fpath, f"{model}_{checkpoint}.{C.AMR_TXT}")
  log_fpath = os.path.join(dirname, D.SAPIENZA_LOG)
  logger.info("Logging at `%s`", log_fpath)

  assert os.path.exists(model_home)

  checkpoints_dir = os.path.join(model_home, 'checkpoints')
  if os.path.exists(checkpoints_dir):
    # noinspection PyTypeChecker
    checkpoint = os.path.join(checkpoints_dir, checkpoint)
  assert os.path.exists(checkpoint)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_sapienza.sh',
    input_fpath,
    output_fpath,
    model_home,
    f'configs/config_{"spring" if model == C.SPRING else "leak_distill"}.yaml',
    checkpoint,
    str(beam),
    venv,
    cuda,
    cvd,
  ]
  if not tokenize:
    cmd.append("--snt-to-tok")
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("%s CMD: %s", model, cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_ibm_amr_parser(input_fpath, output_fpath, model_home, torch_hub, model,
                       seeds, tokenize, batch_size, beam, venv, cuda, cvd):
  # maybe `output_fpath` is a dir?
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg
  if len(seeds) == 0:
    msg = "At least one model seed is required"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.TOKS_TXT)

  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    # output_fpath = os.path.join(output_fpath, D.IBM_TXT)
    seeds_str = ':'.join([str(seed) for seed in seeds])
    output_fpath = io_utils.get_unique_fpath(
      output_fpath, f"{C.IBM}_{model}_seeds{seeds_str}.{C.AMR_TXT}")
  log_fpath = os.path.join(dirname, D.IBM_LOG)
  logger.info("Logging at `%s`", log_fpath)

  # get checkpoints
  checkpoint_home = os.path.join(torch_hub, C.DATA)
  if C.DATA not in os.listdir(torch_hub):
    checkpoint_home = os.path.join(torch_hub, os.pardir, C.DATA)
  checkpoint_home = os.path.join(
    os.path.abspath(checkpoint_home), model, C.MODELS, f'{model}-structured-bart-large')

  checkpoints = []
  for seed in seeds:
    checkpoints.append(
      os.path.join(checkpoint_home, f'seed{seed}', 'checkpoint_wiki.smatch_top5-avg.pt'))

  cmd = [
    subprocess_utils.BASH,
    './bin/run_ibm.sh',
    input_fpath,
    output_fpath,
    model_home,
    torch_hub,
    ':'.join(checkpoints),
    str(batch_size),
    str(beam),
    venv,
    cuda,
    cvd
  ]
  if tokenize:
    cmd.append('--tokenize')
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("IBM CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_amrbart(input_fpath, output_fpath, model_home, model, venv, cuda, cvd):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.AMRS_JSONL)

  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_fpath, f"{model}.{C.AMR_TXT}")

  if os.path.basename(model_home) == 'fine-tune':
    model_home = os.path.dirname(model_home)

  log_fpath = os.path.join(dirname, D.AMRBART_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_amrbart.sh',
    os.path.abspath(input_fpath),
    os.path.abspath(output_fpath),
    os.path.abspath(model_home),
    f'xfbai/{model}',
    venv,
    cuda,
    cvd
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("AMRBART CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_blink(input_fpath, output_fpath, model_home, models, venv, cuda, cvd):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  input_fname = os.path.basename(input_fpath)

  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    if input_fname.endswith(C.AMR_TXT):
      output_fname = input_fname.replace(C.AMR_TXT, f'blink.{C.AMR_TXT}')
    else:
      output_fname = input_fname.replace(C.TXT, f'blink.{C.AMR_TXT}')
    output_fpath = io_utils.get_unique_fpath(output_fpath, output_fname)
  log_fpath = os.path.join(dirname, D.BLINK_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_blink.sh',
    input_fpath,
    output_fpath,
    model_home,
    models,
    venv,
    cuda,
    cvd
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("BLINK CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_leamr(input_fpath, model_home, cuda, cvd):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # there is no output; should log at input dir
  dirname = os.path.dirname(input_fpath)
  log_fpath = os.path.join(dirname, D.LEAMR_LOG)
  logger.info("Logging at `%s`", log_fpath)

  assert os.path.exists(model_home)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_leamr.sh',
    input_fpath,
    model_home,
    cuda,
    cvd
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("LeAMR CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_mbse(input_fpath_list, output_fpath, model_home, venv):
  if len(input_fpath_list) == 0:
    msg = "At least one input AMR file is required"
    logger.warning(msg)
    gr.Warning(msg)
    return msg
  for input_fpath in input_fpath_list:
    if not os.path.isfile(input_fpath):
      msg = f"either `{input_fpath}` does not exist, or is a dir"
      logger.warning(msg)
      gr.Warning(msg)
      return msg
    else:
      # ensure ends in newline
      text = io_utils.load_txt(input_fpath)
      if not text.endswith('\n\n'):
        if text.endswith('\n'):
          text = f'{text}\n'
        else:
          text = f'{text}\n\n'
      io_utils.save_txt(text, input_fpath)

  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    canonical_input_fname = io_utils.get_canonical_fname(input_fpath_list[0])
    output_fpath = io_utils.get_unique_fpath(
      output_fpath, f'{canonical_input_fname}.{len(input_fpath_list)}way_{C.MBSE}.{C.AMR_TXT}')
  log_fpath = os.path.abspath(os.path.join(dirname, D.MBSE_LOG))
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_mbse.sh',
    output_fpath,
    model_home,
    venv,
  ] + input_fpath_list
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("MBSE CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_modal_multi_task(input_fpath, output_fpath, model_home, max_seq_length,
                         venv, cuda, cvd, pipeline):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.DOCS_TXT)

  # output may be a dir or file
  modal_output_fpath = output_fpath
  dirname, is_dir = io_utils.get_dirname(
    modal_output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    ext = C.STAGE1_TXT if pipeline == C.STAGE1 else C.STAGE2_TXT
    output_fpath = io_utils.get_unique_fpath(output_fpath, f'{C.MODAL}.{ext}')
  log_fpath = os.path.join(dirname, D.MDP_BASELINE_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    f'bin/run_{C.MODAL_MULTI_TASK}_{pipeline}.sh',
    os.path.abspath(input_fpath),
    os.path.abspath(output_fpath),
    os.path.abspath(model_home),
    str(max_seq_length),
    venv,
    cuda,
    cvd
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("%s CMD: %s", C.MODAL_MULTI_TASK, cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_mdp_prompt_end2end(input_fpath, output_fpath, model_home, max_seq_length, model_dir,
                           model_name, batch_size, seed, venv, cuda, cvd, pipeline):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.DOCS_TXT)

  # output may be a dir or file
  mdp_output_fpath = output_fpath
  dirname, is_dir = io_utils.get_dirname(
    mdp_output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    ext = C.STAGE1_TXT if pipeline == C.STAGE1 else C.STAGE2_TXT
    output_fpath = io_utils.get_unique_fpath(
      output_fpath, f'{C.MDP_PROMPT}.{ext}')
  log_fpath = os.path.join(dirname, D.MDP_PROMPT_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    f'bin/run_{C.MDP_PROMPT_END2END}_{pipeline}.sh',
    os.path.abspath(input_fpath),
    os.path.abspath(output_fpath),
    os.path.abspath(model_home),
    str(max_seq_length),
    str(batch_size),
    str(seed),
    os.path.abspath(os.path.join(model_home, model_dir)),
    model_name,
    venv,
    cuda,
    cvd
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("%s CMD: %s", C.MDP_PROMPT_END2END, cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_mdp_stage1(input_fpath, output_fpath, model, model_home, max_len, model_dir,
                   model_name, batch_size, seed, venv, cuda, cvd,):
  if model == C.MDP_PROMPT_END2END:
    for x in run_mdp_prompt_end2end(
            input_fpath, output_fpath, model_home, max_len, model_dir, model_name,
            batch_size, seed, venv, cuda, cvd, pipeline=C.STAGE1):
      yield x
  else:
    for x in run_modal_multi_task(input_fpath, output_fpath, model_home, max_len,
                                  venv, cuda, cvd, pipeline=C.STAGE1):
      yield x

def run_mdp_stage2(input_fpath, output_fpath, model, model_home, max_len, model_dir,
                   model_name, batch_size, seed, venv, cuda, cvd,):
  if model == C.MDP_PROMPT_END2END:
    for x in run_mdp_prompt_end2end(
            input_fpath, output_fpath, model_home, max_len, model_dir, model_name,
            batch_size, seed, venv, cuda, cvd, pipeline=C.STAGE2):
      yield x
  else:
    for x in run_modal_multi_task(input_fpath, output_fpath, model_home, max_len,
                                  venv, cuda, cvd, pipeline=C.STAGE2):
      yield x

def run_merge_stage1s(input_fpath_list, output_fpath, include_conceiver=False,
                      include_timex=False, for_thyme_tdg=False):
  for input_fpath in input_fpath_list:
    if not os.path.exists(input_fpath):
      msg = f"Input file `{input_fpath}` does not exist"
      logger.warning(msg)
      gr.Warning(msg)
      return msg

  # make paths absolute
  input_fpath_list = [os.path.abspath(input_fpath) for input_fpath in input_fpath_list]

  output_dir, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_dir, f'{C.MERGED}.{C.STAGE1_TXT}')
  log_fpath = os.path.join(output_dir, D.MERGE_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.PYTHON,
    'scripts/merge_stage1s.py',
    '--stage1s',
    " ".join(input_fpath_list),
    '--output',
    output_fpath
  ]
  if include_conceiver:
    cmd.append('--include_conceiver')
  if include_timex:
    cmd.append('--include_timex')
  if for_thyme_tdg:
    cmd.append('--for_thyme_tdg')
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("Stage1 Merging CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_temporal_pipeline(input_fpath, output_fpath, model_home, max_len, venv,
                          cuda, cvd, model_dir=None, model_name=None, pipeline=None):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  temporal_output_fpath = output_fpath
  dirname, is_dir = io_utils.get_dirname(temporal_output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    ext = C.STAGE1_TXT if pipeline == C.STAGE1 else C.STAGE2_TXT
    output_fpath = io_utils.get_unique_fpath(output_fpath, f'{C.TEMPORAL}.{ext}')
  log_fpath = os.path.join(dirname, D.TDP_BASELINE_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    f'./bin/run_{C.TEMPORAL_PIPELINE}_{pipeline}.sh',
    input_fpath,
    output_fpath,
    model_home,
    str(max_len),
    venv,
    cuda,
    cvd
  ]
  if model_dir and model_name:
    cmd += [os.path.abspath(os.path.join(model_home, model_dir)), model_name]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("%s CMD: %s", C.TEMPORAL_PIPELINE, cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_tdp_stage1(input_fpath, output_fpath, model_home, max_len, venv, cuda, cvd,):
  # thyme_tdg doesn't support stage 1
  for x in run_temporal_pipeline(
          input_fpath, output_fpath, model_home, max_len, venv, cuda, cvd, pipeline=C.STAGE1):
    yield x

def run_thyme_tdg(input_fpath, output_fpath, model_home, model_dir, venv, cuda, cvd):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  thyme_output_fpath = output_fpath
  dirname, is_dir = io_utils.get_dirname(thyme_output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_fpath, f'{C.THYME_TDG}.{C.STAGE2_TXT}')
  log_fpath = os.path.join(dirname, D.TDP_BASELINE_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    f'./bin/run_thyme_tdg.sh',
    input_fpath,
    output_fpath,
    model_home,
    os.path.abspath(os.path.join(model_home, model_dir)),
    venv,
    cuda,
    cvd
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("%s CMD: %s", C.TEMPORAL_PIPELINE, cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_tdp_stage2(input_fpath, output_fpath, model_home,  model_dir, venv, cuda, cvd,):
  # UPDATE: only use thyme_tdg for stage 2

  # if model == C.TEMPORAL_PIPELINE:
  #   for x in run_temporal_pipeline(input_fpath, output_fpath, model_home, max_len,venv, cuda, cvd,
  #                                  model_dir=model_dir, model_name=model_name, pipeline=C.STAGE2):
  #     yield x
  # else:

  for x in run_thyme_tdg(input_fpath, output_fpath, model_home, model_dir, venv, cuda, cvd):
    yield x

def prepare_cdlm_inputs(input_fpath, output_fpath):
  dirname = io_utils.get_dirname(output_fpath, mkdir=True)
  log_fpath = os.path.join(dirname, D.PREP_CDLM_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = f"{subprocess_utils.PYTHON} scripts/prepare_cdlm_inputs.py -i {input_fpath} -o {output_fpath}"
  logger.info("Prepare CDLM Inputs CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_cdlm(input_fpath, output_fpath, model_home, name,  batch_size, gpu_ids, venv, cuda):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isfile(input_fpath):
    input_fpath = os.path.dirname(input_fpath)

  # output should be dir
  os.makedirs(output_fpath, exist_ok=True)
  log_fpath = os.path.join(output_fpath, D.CDLM_LOG)
  logger.info("Logging at `%s`", log_fpath)

  # update config
  config_fpath = os.path.join(model_home, 'configs/config_pairwise_long_reg_span.json')
  config = io_utils.load_json(config_fpath)

  # update
  config['gpu_num'] = gpu_ids if isinstance(gpu_ids, list) else [gpu_ids]
  config['split'] = name
  config['mention_type'] = 'events'
  config['batch_size'] = batch_size
  config['data_folder'] = input_fpath

  # save updated config
  config_fpath = os.path.join(model_home, 'configs/config_pairwise_long_reg_span_tmp.json')
  io_utils.save_json(config, config_fpath)

  # now run with new config
  cmd = [
    subprocess_utils.BASH,
    './bin/run_cdlm.sh',
    os.path.abspath(output_fpath),
    model_home,
    config_fpath,
    name,
    venv,
    cuda
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("CDLM CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_coref(input_fpath, output_fpath, model, model_home, checkpoint, venv, cuda, cvd, ):
  # maybe `output_fpath` is a dir?
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.COREF_JSONLINES)

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(
      output_fpath, f'{model}_{D.COREF_JSONLINES}')
  log_fpath = os.path.abspath(os.path.join(dirname, D.COREF_LOG))
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_coref.sh',
    input_fpath,
    output_fpath,
    model_home,
    checkpoint,
    f'{checkpoint}.pt',
    venv,
    cuda,
    cvd,
  ]
  cmd = f'{" ".join(cmd)} > {log_fpath} 2>&1'
  logger.info("%s CMD: %s", model, cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_udpipe(input_fpath, output_fpath):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg
  assert os.path.isfile(input_fpath)

  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_fpath, D.UDPIPE_JSON)
  log_fpath = os.path.abspath(os.path.join(dirname, D.UDPIPE_LOG))
  logger.info("Logging at `%s`", log_fpath)

  cmd = f"{subprocess_utils.PYTHON} scripts/run_udpipe.py -i {input_fpath} -o {output_fpath}"
  logger.info("UDPipe CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_conversion(
        input_fpath, output_fpath, amr_alignment,  split_mapping,
        ud_input, doc2snt_mapping,
        mdp, tdp, cdlm, coref
):
  # `input_fpath`: AMR
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  assert os.path.isdir(output_fpath)
  log_fpath = os.path.abspath(os.path.join(output_fpath, D.CONV_LOG))
  logger.info("Logging at `%s`", log_fpath)

  cmd = [
    subprocess_utils.PYTHON,
    'scripts/convert_amr2umr.py',
    f'--input {input_fpath}',
    f'--output {output_fpath}',
    f'--aligner {amr_alignment.lower()}'
  ]
  if split_mapping:
    cmd.append(f'--split_mapping {split_mapping}')
  if doc2snt_mapping:
    cmd.append(f'--doc2snt_mapping {doc2snt_mapping}')
  if mdp:
    cmd.append(f'--modal {mdp}')
  if tdp:
    cmd.append(f'--temporal {tdp}')
  if cdlm:
    cmd.append(f'--cdlm {cdlm}')
  if coref:
    cmd.append(f'--coref {coref}')
  if ud_input:
    cmd.append(f'--ud_conllu {ud_input}')
  cmd = " ".join(cmd)

  logger.info("AMR2UMR Conversion CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

def run_eval(pred_fpath, gold_fpath, eval_home):
  if not os.path.exists(pred_fpath):
    msg = f"Input file `{pred_fpath}` does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # for now, only dirs
  assert os.path.isdir(pred_fpath) # should contain predictions which end with .txt
  assert os.path.isdir(gold_fpath) # gold dir with same filenames as predictions
  assert os.path.exists(eval_home)

  log_fpath = os.path.join(pred_fpath, D.EVAL_LOG)
  logger.info("Logging at `%s`", log_fpath)

  cmd = f"{subprocess_utils.PYTHON} scripts/evaluate_umr.py --pred {pred_fpath} --gold {gold_fpath} --eval_home {eval_home}"
  logger.info("UMR Eval CMD: %s", cmd)
  for x in subprocess_utils.run_cmd(cmd, log_fpath):
    yield x

################################################################################
### freq. used helpers
def build_input_outputs(
        input_label=None, input_info=None, input_value=None,
        output_label=None, output_info=None, output_value=None,
        home_value=None, input_as_dropdown=False, output_as_dropdown=False,
        input_choices=None, output_choices=None, as_row=False):
  with gr.Row() if as_row else gr.Column():
    init_home = home_value is not None
    if init_home:
      home = gr.Textbox(
        label="Home",
        value=home_value,
        interactive=True
      )
    if input_as_dropdown:
      if input_choices is not None and input_value is None:
        input_value = ""
      input = gr.Dropdown(
        label="Input (`input`)" if input_label is None else input_label,
        value=input_value,
        info=input_info,
        choices=input_choices,
        interactive=True,
        allow_custom_value=True,
      )
    else:
      input = gr.Textbox(
        label="Input (`input`)" if input_label is None else input_label,
        value=input_value,
        info=input_info,
        interactive=True,
      )
    if output_as_dropdown:
      if output_choices is not None and output_value is None:
        output_value = ""
      output = gr.Dropdown(
        label="Output (`output`)" if output_label is None else output_label,
        value=output_value,
        info=output_info,
        choices=output_choices,
        interactive=True,
        allow_custom_value=True,
      )
    else:
      output = gr.Textbox(
        label="Output (`output`)" if output_label is None else output_label,
        value=output_value,
        info=output_info,
        interactive=True
      )
    if init_home:
      return home, input, output
    return input, output

def build_envs(no_venv=False, no_cvd=False,
               default_venv="umr-py3.8-torch-1.13.1-cu11.7",
               default_cuda="cuda-11.7",
               default_cvd='CUDA_VISIBLE_DEVICES=0',
               as_column=False):
  pyenvs = PYENV_VENVS + ["N/A"]
  with gr.Column() if as_column else gr.Row():
    venv = None
    if not no_venv:
      venv = gr.Dropdown(
        label="venv",
        info="pyenv virtualenv",
        choices=pyenvs,
        value=default_venv if default_venv in pyenvs else pyenvs[0],
        interactive=True,
      )
    cuda = gr.Dropdown(
      label="CUDA Version",
      info="no-cuda if CPU",
      choices=CUDAS,
      value=default_cuda if default_cuda in CUDAS else CUDAS[0],
      interactive=True,
    )
    if no_cvd:
      assert not no_venv
      return venv, cuda
    cvd = gr.Dropdown(
      label="CUDA_VISIBLE_DEVICES",
      info="ignored if no-cuda",
      choices=CVDS,
      value=default_cvd if default_cvd in CVDS else CVDS[0],
      interactive=True
    )
    if no_venv:
      assert not no_cvd
      return cuda, cvd
    return venv, cuda, cvd

def build_run_button_info(run_button_label="RUN", info_label="Output Info", max_lines=5, as_row=False,
                          with_refresh_button=False, refresh_button_label="Refresh",):
  with gr.Row() if as_row else gr.Column():
    if with_refresh_button:
      with gr.Row():
        run_button = gr.Button(run_button_label, variant="primary")
        refresh_button = gr.Button(refresh_button_label, variant="secondary")
    else:
      run_button = gr.Button(run_button_label, variant="primary")
    info = gr.Textbox(label=info_label, lines=max_lines)
    if with_refresh_button:
      return run_button, refresh_button, info
    return run_button, info

################################ App Interface #################################
with gr.Blocks() as app:
  ### Global Header
  gr.Markdown(f"# {__title__} v{__version__} WebUI")
  gr.Markdown(f"* Project Home: {__uri__}")
  gr.Markdown(f"Launched on %s %s" % (datetime.now().strftime("%Y-%m-%d (%a) %I:%M:%S %p"), datetime.now(timezone.utc).astimezone().tzinfo))

  ### System info
  with gr.Group():
    gr.Markdown('&nbsp;&nbsp;&nbsp;System Information')
    with gr.Row():
      hostname = gr.Textbox(label="Hostname", value=D.HOSTNAME, visible=True, )
      platform = gr.Textbox(label="Platform", value=D.PLATFORM, visible=True, )
      cpus_count = gr.Textbox(label="#CPUs", value=mp.cpu_count(), visible=True, )
      _gpus = GPUS[1:] # drop cpu
      gpus_info = gr.Textbox(label="GPUs", value=f'{_gpus[0]} (x{len(_gpus)})' if len(_gpus) > 0 else "", visible=True, )

  ### Global Params
  with gr.Group():
    gr.Markdown('&nbsp;&nbsp;&nbsp;Global Parameters')
    with gr.Row():
      global_data_root = gr.Textbox(
        label="DATA Root",
        info="preprocessed data and model inputs (model inputs stored in `prep` subdir)",
        value=PROJ_DATA,
        interactive=False
      )
      global_exp_root = gr.Textbox(
        label="EXPERIMENT Root",
        info="model outputs experiment results (intermediate results cached in `tmp` subdir)",
        value=PROJ_EXP,
        interactive=False
      )

  global_flush_button = gr.Button("Flush Events", variant="secondary")
  global_flush_button.click(
    fn=lambda: {'interactive': False, '__type__': 'update'},
    outputs=global_flush_button,
  ).then(fn=lambda : time.sleep(0.33)).then(
    fn=lambda: {'interactive': True, '__type__': 'update'},
    outputs=global_flush_button,
  )

  ### MAIN
  with gr.Tabs():
    with gr.Tab("README"):
      with open(os.path.join('README.md')) as f:
        gr.Markdown(f.read())

    with gr.Tab("Preprocessing"):
      gr.Markdown("## UMR Corpus Cleanup & Data Preprocessing")
      prp_input, prp_output = build_input_outputs(
        input_value=D.UMR_EN, output_value=os.path.join(D.PROJ_DATA,'umr-v1.0-en'))

      with gr.Row():
        prp_aggression = gr.Radio(
          choices=[1, 2],
          label="Aggresion Setting (`aggr`)",
          info="1 for minimal fixes; 2 for labeling consistency",
          value=2,
          interactive=True
        )
        prp_snt2tok = gr.Radio(
          label="Snt-to-Tok",
          info="Select `None` to split on white-space; `IBM` corresponds to `jamr-like` tokenizer",
          value='None',
          choices=['None', 'Spring', 'IBM'],
          interactive=True
        )
        prp_strategy = gr.Radio(
          label="Partition Strategy",
          info="How to distribute sentences when partitioning",
          choices=['Greedy', 'Even'],
          value="Greedy",
          interactive=True
        )

      prp_run_button, prp_info = build_run_button_info()
      prp_split_json = gr.JSON(label="Split Mapping")

      ### PRP RUN EVENT
      define_long_running_event(
        button=prp_run_button,
        fn=run_preprocessing,
        inputs=[prp_input, prp_output, prp_aggression, prp_snt2tok, prp_strategy],
        outputs=prp_info,
        fn2=lambda x: io_utils.load_json(os.path.join(x, C.PREP), D.SPLIT_MAPPING_JSON),
        inputs2=prp_output,
        outputs2=prp_split_json
      )

    with gr.Tab("Parsing"):
      gr.Markdown("## UMR Parsing Pipeline")
      gr.Markdown("* set the name of sub-folder within EXP Root containing preprocessed data to populate fields automatically")
      with gr.Row():
        parsing_subroot = gr.Textbox(
          label="Subroot Name",
          info="if empty, processed data will be stored under EXP Root",
          value="umr-v1.0-en", # placeholder
          interactive=True,
        )
        parsing_set_subroot_button = gr.Button("Set Subroot", variant="primary")
        parsing_reset_subroot_button = gr.Button("Reset", variant="secondary")
      parsing_subroot_info = gr.Textbox(label="Current Subroot", max_lines=1, interactive=False)

      with gr.Tabs():
        with gr.Tab("AMR"):
          gr.Markdown("### AMR Parsing")

          with gr.Tabs():
            with gr.Tab("Sapienza"):
              gr.Markdown('#### LeakDistill + SPRING')
              gr.Markdown("* Project Homepage: [https://github.com/SapienzaNLP/LeakDistill](https://github.com/SapienzaNLP/LeakDistill)")
              sapienza_home, sapienza_input, sapienza_output = build_input_outputs(
                output_info="if empty, will be stored under Subroot",
                home_value=D.SAPIENZA,
                input_as_dropdown=True,
                input_choices=INIT_DATA_CHOICES[D.AMRS_TXT]
              )

              with gr.Group():
                with gr.Row():
                  sapienza_model = gr.Radio(
                    label="Model",
                    choices=[C.LEAK_DISTILL, C.SPRING],
                    value=C.LEAK_DISTILL,
                    interactive=True,
                  )
                  ld_choices = ['best-smatch_checkpoint_12_0.8534.pt', 'best-smatch_checkpoint_20_0.8564.pt']
                  spring_choices = ['AMR2.parsing.pt', 'AMR3.parsing.pt']
                  sapienza_checkpoint = gr.Radio(
                    label="Checkpoint",
                    choices=ld_choices,
                    value='best-smatch_checkpoint_12_0.8534.pt',
                    interactive=True
                  )
                  sapienza_model.select( # locally bound event
                    fn=lambda x: {'choices': spring_choices if x == C.SPRING else ld_choices, '__type__': 'update'},
                    inputs=sapienza_model,
                    outputs=sapienza_checkpoint
                  )
                with gr.Row():
                  sapienza_tokenize = gr.Radio(
                    label="Tokenize",
                    choices=[True, False],
                    value=True,
                    interactive=True
                  )
                  sapienza_beam = gr.Number(
                    label="Beam Size",
                    minimum=1,
                    maximum=20,
                    value=10,
                    interactive=True,
                  )

              sapienza_venv, sapienza_cuda, sapienza_cvd = build_envs()
              sapienza_run_button, sapienza_refresh_button, sapienza_info = build_run_button_info(with_refresh_button=True)

              ### LeakDistill + SPRING RUN EVENT
              define_long_running_event(
                button=sapienza_run_button,
                fn=run_sapienza_amr_parser,
                inputs=[
                  sapienza_input, sapienza_output, sapienza_home, sapienza_model, sapienza_checkpoint,
                  sapienza_tokenize, sapienza_beam, sapienza_venv, sapienza_cuda, sapienza_cvd
                ],
                outputs=sapienza_info
              )

              ### LeakDistill + SPRING REFRESH EVENT
              sapienza_refresh_button.click(
                fn=refresh_data,
                inputs=gr.Textbox(value=D.AMRS_TXT, visible=False),
                outputs=sapienza_input,
              )

            with gr.Tab("IBM"):
              gr.Markdown('#### IBM transition-amr-parser')
              gr.Markdown("* Project Homepage: [https://github.com/IBM/transition-amr-parser/tree/master](https://github.com/IBM/transition-amr-parser/tree/master)")
              ibm_torch_hub = gr.Textbox(
                label="Torch HUB",
                info="if empty, default location",
                value=D.TORCH_HUB if D.TORCH_HUB is not None else torch.hub.get_dir(),
                interactive=True
              )
              ibm_home, ibm_input, ibm_output = build_input_outputs(
                home_value=D.IBM, input_as_dropdown=True, input_choices=INIT_DATA_CHOICES[D.TOKS_TXT])

              with gr.Group():
                with gr.Row():
                  ibm_model = gr.Radio(
                    label="Model",
                    choices=['amr2joint_ontowiki2_g2g', 'amr3joint_ontowiki2_g2g'],
                    value='amr3joint_ontowiki2_g2g',
                    interactive=True,
                  )
                  ibm_seeds = gr.Checkboxgroup(
                    label="Model Seeds",
                    info="Select multiple for Ensemble",
                    choices=[42, 43, 44],
                    value=[42,43,44],
                    interactive=True
                  )
                with gr.Row():
                  ibm_tokenize = gr.Radio(
                    label="Tokenize",
                    choices=[True, False],
                    value=False,
                    interactive=True
                  )
                  ibm_batch_size = gr.Number(
                    label="Batch Size",
                    minimum=1,
                    value=32,
                    interactive=True
                  )
                  ibm_beam_size = gr.Number(
                    label="Beam Size",
                    minimum=1,
                    maximum=10,
                    value=10,
                    interactive=True
                  )

              ibm_venv, ibm_cuda, ibm_cvd = build_envs()
              ibm_run_button, ibm_refresh_button, ibm_info = build_run_button_info(with_refresh_button=True)

              ### IBM RUN EVENT
              define_long_running_event(
                button=ibm_run_button,
                fn=run_ibm_amr_parser,
                inputs=[
                  ibm_input, ibm_output, ibm_home, ibm_torch_hub, ibm_model, ibm_seeds,
                  ibm_tokenize, ibm_batch_size, ibm_beam_size, ibm_venv, ibm_cuda, ibm_cvd],
                outputs=ibm_info,
              )

              ### IBM REFRESH EVENT
              ibm_refresh_button.click(
                fn=refresh_data,
                inputs=gr.Textbox(value=D.TOKS_TXT, visible=False),
                outputs=ibm_input,
              )

            with gr.Tab("AMRBART"):
              gr.Markdown('#### AMRBART')
              gr.Markdown("* Project Homepage: [https://github.com/goodbai-nlp/AMRBART](https://github.com/goodbai-nlp/AMRBART)")
              amrbart_home, amrbart_input, amrbart_output = build_input_outputs(
                home_value=D.AMRBART, input_as_dropdown=True, input_choices=INIT_DATA_CHOICES[D.AMRS_JSONL])

              amrbart_model = gr.Radio(
                label="Model",
                choices=['AMRBART-large-finetuned-AMR2.0-AMRParsing-v2', 'AMRBART-large-finetuned-AMR3.0-AMRParsing-v2'],
                value='AMRBART-large-finetuned-AMR3.0-AMRParsing-v2',
                interactive=True,
              )

              amrbart_venv, amrbart_cuda, amrbart_cvd = build_envs()
              amrbart_run_button, amrbart_refresh_button, amrbart_info = build_run_button_info(with_refresh_button=True)

              ### AMRBART RUN EVENT
              define_long_running_event(
                amrbart_run_button,
                fn=run_amrbart,
                inputs=[
                  amrbart_input, amrbart_output, amrbart_home, amrbart_model, amrbart_venv, amrbart_cuda, amrbart_cvd
                ],
                outputs=amrbart_info
              )

              ### AMRBART REFRESH EVENT
              amrbart_refresh_button.click(
                fn=refresh_data,
                inputs=gr.Textbox(value=D.AMRS_JSONL, visible=False),
                outputs=amrbart_input,
              )

            with gr.Tab("MBSE"):
              gr.Markdown("#### Maximum Bayes Smatch Ensemble Distillation")
              gr.Markdown("* Script: [https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py](https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py)")
              gr.Markdown("##### Meta (e.g., Alignment) always inherits from the first (`canonical`) of the inputs!")
              mbse_home = gr.Textbox(
                label="Home",
                value=D.IBM,
                interactive=True
              )
              with gr.Group():
                mbse_input = gr.Dropdown(
                  label="Input `input`",
                  info="Select all",
                  choices=INIT_EXP_CHOICES[C.AMRS],
                  allow_custom_value=True,
                  multiselect=True
                )
                mbse_canonical = gr.Textbox(label="Canonical Input (first of the input AMRs)")
                mbse_input.change(
                  fn=lambda xs: xs[0] if len(xs) > 0 else "",
                  inputs=mbse_input,
                  outputs=mbse_canonical
                )
              mbse_output = gr.Textbox(label="Output `output`", interactive=True)

              mbse_venv = gr.Dropdown(
                label="venv",
                info="pyenv virtualenv",
                choices=PYENV_VENVS + ["N/A"],
                value="umr-py3.8-torch-1.13.1-cu11.7" if "umr-py3.8-torch-1.13.1-cu11.7" in PYENV_VENVS else PYENV_VENVS[0],
                interactive=True,
              )
              mbse_run_button, mbse_refresh_button, mbse_info = build_run_button_info(with_refresh_button=True)

              ### MBSE RUN EVENT
              define_long_running_event(
                mbse_run_button,
                fn=run_mbse,
                inputs=[mbse_input, mbse_output, mbse_home, mbse_venv],
                outputs=mbse_info
              )

              ### MBSE REFRESH EVENT
              mbse_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Textbox(value=C.AMRS, visible=False),
                outputs=mbse_input,
              )

            with gr.Tab("BLINK"):
              gr.Markdown("#### BLINK Enitity Linger")
              gr.Markdown("* Project Homepage: [https://github.com/facebookresearch/BLINK](https://github.com/facebookresearch/BLINK)")
              gr.Markdown("Here we prefer LeakDistill's `bin/blinkify.py` script")
              blink_home, blink_input, blink_output = build_input_outputs(
                home_value=D.SAPIENZA, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.AMRS])

              # noinspection PyTypeChecker
              blink_models = gr.Textbox(
                label="BLINK models directory",
                info="location of model weights and caches",
                value=os.path.join(D.SAPIENZA, C.BLINK, C.MODELS),
                interactive=True
              )
              blink_venv, blink_cuda, blink_cvd = build_envs()
              blink_run_button, blink_refresh_button, blink_info = build_run_button_info(with_refresh_button=True)

              ### BLINK RUN EVENT
              define_long_running_event(
                blink_run_button,
                fn=run_blink,
                inputs=[blink_input, blink_output, blink_home, blink_models, blink_venv, blink_cuda, blink_cvd],
                outputs=blink_info
              )

              ### BLINK REFRESH EVENT
              blink_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Textbox(value=C.AMRS, visible=False),
                outputs=blink_input,
              )

            with gr.Tab("LEAMR"):
              gr.Markdown("#### LEAMR Aligner")
              gr.Markdown("* Project Homepage: [https://github.com/ablodge/leamr](https://github.com/ablodge/leamr)")
              gr.Markdown("requires JAMR and ISI")
              with gr.Column():
                leamr_home = gr.Textbox(
                  label="Home",
                  value=D.ALIGNERS,
                  interactive=True
                )
                leamr_input = gr.Dropdown(
                  label="Input (`input`)",
                  value="",
                  choices=INIT_EXP_CHOICES[C.AMRS],
                  allow_custom_value=True,
                  interactive=True,
                )

              leamr_cuda, leamr_cvd = build_envs(no_venv=True)
              leamr_run_button, leamr_refresh_button, leamr_info = build_run_button_info(with_refresh_button=True)

              ### LEAMR RUN EVENT
              define_long_running_event(
                leamr_run_button,
                fn=run_leamr,
                inputs=[leamr_input, leamr_home, leamr_cuda, leamr_cvd],
                outputs=leamr_info
              )

              ### LEAMR REFRESH EVENT
              leamr_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Textbox(value=C.AMRS, visible=False),
                outputs=leamr_input,
              )

        with gr.Tab("MDP"):
          gr.Markdown("### Modal Dependency Parsing")
          gr.Markdown("Project Homepages:")
          gr.Markdown("1. Baseline: [https://github.com/Jryao/modal_dependency](https://github.com/Jryao/modal_dependency)")
          gr.Markdown("2. mdp_prompt: [https://github.com/Jryao/mdp_prompt](https://github.com/Jryao/mdp_prompt)")

          with gr.Tabs():
            with gr.Tab("STAGE1") as mdp_stage1_tab:
              mdp_stage1_home, mdp_stage1_input, mdp_stage1_output = build_input_outputs(
                home_value=D.MDP_BASELINE, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[D.DOCS_TXT])

              with gr.Group():
                with gr.Row():
                  mdp_stage1_model = gr.Radio(
                    label="MDP Stage1 Model",
                    choices=[C.MODAL_MULTI_TASK, C.MDP_PROMPT_END2END],
                    value=C.MODAL_MULTI_TASK,
                    interactive=True
                  )
                  mdp_stage1_max_len = gr.Number(label="Max Seq. Length", value=128, interactive=True,)
                  mdp_stage1_model.select(  # locally bound event
                    fn=lambda x: (D.MDP_PROMPT, 384) if x == C.MDP_PROMPT_END2END else (D.MDP_BASELINE, 128),
                    inputs=mdp_stage1_model,
                    outputs=[mdp_stage1_home, mdp_stage1_max_len]
                  )
                with gr.Row():
                  mdp_stage1_model_dir = gr.Textbox(
                    label="MDP-Prompt Model Dir",
                    info="only for mdp-prompt model",
                    value="outputs/end2end",
                    interactive=False,
                    visible=True,
                  )
                  mdp_stage1_model_name = gr.Textbox(
                    label='MDP-Prompt Model Name',
                    info="only for mdp-prompt model",
                    value="gen_end2end",
                    interactive=False,
                    visible=True
                  )
                with gr.Row():
                  mdp_stage1_batch_size = gr.Number(label="Batch Size", info="only for mdp-prompt model", value=8, )
                  mdp_stage1_seed = gr.Number(label="Model Seed", info="only for mdp-prompt model", value=42)

            with gr.Tab("Merge Stage1s") as mdp_merge_tab:
              mdp_merge_input = gr.Dropdown(
                label="Stage1 Inputs `stage1s`",
                info="Select all",
                choices=INIT_EXP_CHOICES[C.MDP_STAGE1],
                allow_custom_value=True,
                multiselect=True
              )
              mdp_merge_output = gr.Textbox(label="Output `output`", interactive=True)

              with gr.Row():
                mdp_merge_conc = gr.Radio(
                  label="Include Conceiver",
                  info="whether the merged Stage1 should include Conceivers",
                  choices=[True, False],
                  value=True,
                  interactive=True,
                )
                mdp_merge_timex = gr.Radio(
                  label="Include Timex",
                  info="whether the merged Stage1 should include Timex",
                  choices=[True, False],
                  value=False,
                  interactive=False,
                )
                mdp_merge_thyme_tdg = gr.Radio(
                  label="For Thyme-TDG",
                  info="whether the merged Stage1 is to be consumed by `thyme_tdg`",
                  choices=[True, False],
                  value=False,
                  interactive=False,
                )

              mdp_merge_run_button, mdp_merge_info = build_run_button_info("Merge")

              ### MDP Stage1 Merging RUN Event
              define_long_running_event(
                mdp_merge_run_button,
                fn=run_merge_stage1s,
                inputs=[mdp_merge_input, mdp_merge_output, mdp_merge_conc, mdp_merge_timex, mdp_merge_thyme_tdg],
                outputs=mdp_merge_info
              )

            with gr.Tab("STAGE2") as mdp_stage2_tab:
              mdp_stage2_home, mdp_stage2_input, mdp_stage2_output = build_input_outputs(
                home_value=D.MDP_BASELINE, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.MDP_STAGE1])
              with gr.Group():
                with gr.Row():
                  mdp_stage2_model = gr.Radio(
                    label="MDP Stage2 Model",
                    choices=[C.MODAL_MULTI_TASK, C.MDP_PROMPT_END2END],
                    value=C.MODAL_MULTI_TASK,
                    interactive=True
                  )
                  mdp_stage2_max_len = gr.Number(label="Max Seq. Length", value=128, interactive=True, )
                  mdp_stage2_model.select(  # locally bound event
                    fn=lambda x: (D.MDP_PROMPT, 384) if x == C.MDP_PROMPT_END2END else (D.MDP_BASELINE, 128),
                    inputs=mdp_stage2_model,
                    outputs=[mdp_stage2_home, mdp_stage2_max_len]
                  )
                with gr.Row():
                  mdp_stage2_model_dir = gr.Textbox(
                    label="MDP-Prompt Model Dir",
                    info="only for mdp-prompt model",
                    value="outputs/end2end",
                    interactive=False,
                    visible=True,
                  )
                  mdp_stage2_model_name = gr.Textbox(
                    label='MDP-Prompt Model Name',
                    info="only for mdp-prompt model",
                    value="gen_end2end",
                    interactive=False,
                    visible=True
                  )
                with gr.Row():
                  mdp_stage2_batch_size = gr.Number(label="Batch Size", info="only for mdp-prompt model", value=8, )
                  mdp_stage2_seed = gr.Number(label="Model Seed", info="only for mdp-prompt model", value=42)

          mdp_venv, mdp_cuda, mdp_cvd = build_envs()
          mdp_refresh_button = gr.Button("Refresh", variant="secondary")
          with gr.Row():
            mdp_stage1_run_button, mdp_stage1_info = build_run_button_info("RUN MDP Stage 1")
            mdp_stage2_run_button, mdp_stage2_info = build_run_button_info("RUN MDP Stage 2")

          ### MDP Tab-select EVENT
          mdp_stage1_tab.select(
            fn=lambda : ({'interactive': True, '__type__': 'update'}, {'interactive': False, '__type__': 'update'}),
            outputs=[mdp_stage1_run_button, mdp_stage2_run_button]
          )
          mdp_merge_tab.select(
            fn=lambda : ({'interactive': False, '__type__': 'update'}, {'interactive': False, '__type__': 'update'}),
            outputs=[mdp_stage1_run_button, mdp_stage2_run_button]
          )
          mdp_stage2_tab.select(
            fn=lambda : ({'interactive': False, '__type__': 'update'}, {'interactive': True, '__type__': 'update'}),
            outputs=[mdp_stage1_run_button, mdp_stage2_run_button]
          )

          ### MDP RUN EVENT
          define_long_running_event(
            mdp_stage1_run_button,
            fn=run_mdp_stage1,
            inputs=[
              mdp_stage1_input, mdp_stage1_output, mdp_stage1_model, mdp_stage1_home, mdp_stage1_max_len, mdp_stage1_model_dir,
              mdp_stage1_model_name, mdp_stage1_batch_size, mdp_stage1_seed, mdp_venv, mdp_cuda, mdp_cvd
            ],
            outputs=mdp_stage1_info
          )
          define_long_running_event(
            mdp_stage2_run_button,
            fn=run_mdp_stage2,
            inputs=[
              mdp_stage2_input, mdp_stage2_output, mdp_stage2_model, mdp_stage2_home, mdp_stage2_max_len, mdp_stage2_model_dir,
              mdp_stage2_model_name, mdp_stage2_batch_size, mdp_stage2_seed, mdp_venv, mdp_cuda, mdp_cvd
            ],
            outputs=mdp_stage2_info
          )

          ### MDP REFRESH EVENT
          mdp_refresh_button.click(
            fn=refresh_data,
            inputs=gr.Textbox(value=D.DOCS_TXT, visible=False),
            outputs=mdp_stage1_input,
          ).then(
            fn=refresh_exp,
            inputs=gr.Textbox(value=C.MDP_STAGE1, visible=False),
            outputs=mdp_stage2_input,
          ).then(
            fn=refresh_exp,
            inputs=gr.Textbox(value=C.MDP_STAGE1, visible=False),
            outputs=mdp_merge_input,
          )

        with gr.Tab("TDP"):
          gr.Markdown("### Temporal Dependency Parsing")
          gr.Markdown("Project Homepages:")
          gr.Markdown("1. Baseline: N/A")
          gr.Markdown("2. thyme_tdg: [https://github.com/Jryao/thyme_tdg/tree/master)")

          with gr.Tabs():
            with gr.Tab("STAGE1") as tdp_stage1_tab:
              gr.Markdown("* Default output dir (local): `./outputs/pipeline_stage1`")
              gr.Markdown("* Default Model name: `temporal_stage1` -> `temporal_stage1_pytorch_model.bin`")
              tdp_stage1_home, tdp_stage1_input, tdp_stage1_output = build_input_outputs(
                home_value=D.TDP_BASELINE, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[D.DOCS_TXT])
              tdp_stage1_max_len = gr.Number(label="Max Seq. Length", value=384, interactive=True,)

            with gr.Tab("Merge Stage1s") as tdp_merge_tab:
              tdp_merge_input = gr.Dropdown(
                label="Stage1 Inputs `stage1s`",
                info="Select all",
                choices=INIT_EXP_CHOICES[C.STAGE1],
                allow_custom_value=True,
                multiselect=True
              )
              tdp_merge_output = gr.Textbox(label="Output `output`", interactive=True)

              with gr.Row():
                tdp_merge_conc = gr.Radio(
                  label="Include Conceiver",
                  info="whether the merged Stage1 should include Conceivers",
                  choices=[True, False],
                  value=False,
                  interactive=True,
                )
                tdp_merge_timex = gr.Radio(
                  label="Include Timex",
                  info="whether the merged Stage1 should include Timex",
                  choices=[True, False],
                  value=True,
                  interactive=False,
                )
                tdp_merge_thyme_tdg = gr.Radio(
                  label="For Thyme-TDG",
                  info="whether the merged Stage1 is to be consumed by `thyme_tdg`",
                  choices=[True, False],
                  value=True,
                  interactive=False,
                )

              tdp_merge_run_button, tdp_merge_info = build_run_button_info("Merge")

              ### MDP Stage1 Merging RUN Event
              define_long_running_event(
                tdp_merge_run_button,
                fn=run_merge_stage1s,
                inputs=[tdp_merge_input, tdp_merge_output, tdp_merge_conc, tdp_merge_timex, tdp_merge_thyme_tdg],
                outputs=tdp_merge_info
              )

            with gr.Tab("STAGE2") as tdp_stage2_tab:
              gr.Markdown("")
              tdp_stage2_home, tdp_stage2_input, tdp_stage2_output = build_input_outputs(
                home_value=D.THYME_TDG, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.TDP_STAGE1])
              tdp_stage2_model_dir = gr.Radio(
                label="Model Dir",
                value="outputs/general_128",
                choices=["outputs/general_128", "outputs/general_384"],
                interactive=True,
              )

          tdp_venv, tdp_cuda, tdp_cvd = build_envs()
          tdp_refresh_button = gr.Button("Refresh", variant="secondary")
          with gr.Row():
            tdp_stage1_run_button, tdp_stage1_info = build_run_button_info("RUN TDP Stage 1")
            tdp_stage2_run_button, tdp_stage2_info = build_run_button_info("RUN TDP Stage 2")

          ### TDP Tab-select EVENT
          tdp_stage1_tab.select(
            fn=lambda : (
              {'interactive': True, '__type__': 'update'},
              {'interactive': False, '__type__': 'update'},
              "umr-py3.8-torch-1.13.1-cu11.7"
            ),
            outputs=[tdp_stage1_run_button, tdp_stage2_run_button, tdp_venv]
          )
          tdp_merge_tab.select(
            fn=lambda : ({'interactive': False, '__type__': 'update'}, {'interactive': False, '__type__': 'update'}),
            outputs=[tdp_stage1_run_button, tdp_stage2_run_button]
          )
          tdp_stage2_tab.select(
            fn=lambda : (
              {'interactive': False, '__type__': 'update'},
              {'interactive': True, '__type__': 'update'},
              "umr-py3.8-torch-1.13.1-cu11.7-latest"
            ),
            outputs=[tdp_stage1_run_button, tdp_stage2_run_button, tdp_venv]
          )

          ### TDP RUN EVENT
          define_long_running_event(
            tdp_stage1_run_button,
            fn=run_tdp_stage1,
            inputs=[tdp_stage1_input, tdp_stage1_output, tdp_stage1_home, tdp_stage1_max_len, tdp_venv, tdp_cuda, tdp_cvd],
            outputs=tdp_stage1_info,
          )
          define_long_running_event(
            tdp_stage2_run_button,
            fn=run_tdp_stage2,
            inputs=[tdp_stage2_input, tdp_stage2_output, tdp_stage2_home, tdp_stage2_model_dir, tdp_venv, tdp_cuda, tdp_cvd],
            outputs=tdp_stage2_info,
          )

          ### TDP REFRESH EVENT
          tdp_refresh_button.click(
            fn=refresh_data,
            inputs=gr.Textbox(value=D.DOCS_TXT, visible=False),
            outputs=tdp_stage1_input,
          ).then(
            fn=refresh_exp,
            inputs=gr.Textbox(value=C.TDP_STAGE1, visible=False),
            outputs=tdp_stage2_input,
          ).then(
            fn=refresh_exp,
            inputs=gr.Textbox(value=C.STAGE1, visible=False),
            outputs=tdp_merge_input,
          )

        with gr.Tab("Coref"):
          gr.Markdown("### Coreference")

          with gr.Tabs():
            with gr.Tab("CDLM"):
              gr.Markdown("Cross-Document Event Coreference")
              gr.Markdown("* Project Homepage: [https://github.com/aviclu/CDLM/tree/main/cross_encoder](https://github.com/aviclu/CDLM/tree/main/cross_encoder)")
              gr.Markdown("[!] requires event detection first, then inputs must be prepared")
              with gr.Group():
                gr.Markdown("&nbsp;&nbsp;&nbsp;1) Prepare CDLM inputs from Event Detection model output")
                cdlm_prep_input, cdlm_prep_output = build_input_outputs(
                  input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.MDP_STAGE1])

              cdlm_prep_button, cdlm_prep_refresh_button, cdlm_prep_info = build_run_button_info(
                run_button_label="Prepare CDLM Inputs", info_label="CDLM Prep Info", with_refresh_button=True)

              ### CDLM PREP RUN EVENT
              define_long_running_event(
                cdlm_prep_button,
                fn=prepare_cdlm_inputs,
                inputs=[cdlm_prep_input, cdlm_prep_output],
                outputs=cdlm_prep_info
              )

              ### CDLM PREP REFRESH EVENT
              cdlm_prep_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Textbox(value=C.MDP_STAGE1, visible=False),
                outputs=cdlm_prep_input,
              )

              with gr.Group():
                gr.Markdown("&nbsp;&nbsp;&nbsp;2) Clustering")
                cdlm_home, cdlm_input, cdlm_output = build_input_outputs(
                  home_value=D.CDLM, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.TMP])

              with gr.Row():
                cdlm_name = gr.Textbox(
                  label="Data Prefix (FIXED)",
                  value="cdlm",
                  interactive=False,
                )
                cdlm_batch_size = gr.Number(
                  label="Batch Size",
                  value=64,
                  minimum=1,
                  interactive=True
                )
                cdlm_gpus = gr.Checkboxgroup(
                  label="GPU ids",
                  choices=[i for i in range(len(GPUS)-1)],
                  interactive=True
                )
              cdlm_venv, cdlm_cuda = build_envs(no_cvd=True, default_venv='cdlm')
              cdlm_run_button, cdlm_refresh_button, cdlm_info = build_run_button_info(with_refresh_button=True)

              ### CDLM RUN EVENT
              define_long_running_event(
                cdlm_run_button,
                fn=run_cdlm,
                inputs=[cdlm_prep_input, cdlm_output, cdlm_home, cdlm_name, cdlm_batch_size,
                        cdlm_gpus, cdlm_venv, cdlm_cuda],
                outputs=cdlm_info
              )

              ### CDLM REFRESH EVENT
              cdlm_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Textbox(value=C.TMP, visible=False),
                outputs=cdlm_input,
              )

            with gr.Tab("coref"):
              gr.Markdown("Conjunction-Aware Word-Level Coreference Resolution")
              gr.Markdown("* Project Homepage: [https://github.com/KarelDO/wl-coref](https://github.com/KarelDO/wl-coref)")
              gr.Markdown("* Based on [wl-coref](https://github.com/vdobrovolskii/wl-coref)")
              coref_model = gr.Radio(
                label="Model",
                choices=[C.CAW_COREF, C.WL_COREF],
                value=C.CAW_COREF,
                interactive=True
              )
              coref_home, coref_input, coref_output = build_input_outputs(
                home_value=D.CAW_COREF, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[D.COREF_JSONLINES])
              coref_checkpoint = gr.Textbox(
                label="Pretrained Checkpoint Name",
                value="roberta",
                interactive=True,
              )
              coref_model.select(
                fn=lambda x: D.CAW_COREF if x == C.CAW_COREF else D.WL_COREF,
                inputs=coref_model,
                outputs=[coref_home]
              )
              coref_venv, coref_cuda, coref_cvd = build_envs()
              coref_run_button, coref_refresh_button, coref_info = build_run_button_info(with_refresh_button=True)

              ### COREF RUN EVENT
              define_long_running_event(
                coref_run_button,
                fn=run_coref,
                inputs=[coref_input, coref_output, coref_model, coref_home, coref_checkpoint, coref_venv, coref_cuda, coref_cvd],
                outputs=coref_info,
              )

              ### COREF REFRESH EVENT
              coref_refresh_button.click(
                fn=refresh_data,
                inputs=gr.Textbox(value=D.COREF_JSONLINES, visible=False),
                outputs=coref_input,
              )

        with gr.Tab("Conversion"):
          gr.Markdown("### AMR2UMR Conversion")
          gr.Markdown("* [!] AMR parses are required; Alignment is required in order to integrate Document-level inputs")

          with gr.Tabs():
            with gr.Tab("UDPipe"):
              gr.Markdown("#### UD Parsing")
              gr.Markdown("* UDPipe2 Homepage: [https://lindat.mff.cuni.cz/services/udpipe/](https://lindat.mff.cuni.cz/services/udpipe/)")
              conv_ud_input, conv_ud_output = build_input_outputs(
                input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[D.TOKS_TXT])
              conv_ud_run_button, conv_ud_refresh_button, conv_ud_info = build_run_button_info(with_refresh_button=True)

              ### UDPIPE RUN EVENT
              define_long_running_event(
                conv_ud_run_button,
                fn=run_udpipe,
                inputs=[conv_ud_input, conv_ud_output],
                outputs=conv_ud_info,
              )

              ### UDPIPE REFRESH EVENT
              conv_ud_refresh_button.click(
                fn=refresh_data,
                inputs=gr.Textbox(value=D.TOKS_TXT, visible=False),
                outputs=conv_ud_input,
              )

            with gr.Tab("Document-level Inputs"):
              gr.Markdown("#### Document-Level Information")
              gr.Markdown("* Coref, MDP & TDP outputs")
              gr.Markdown("if any of these is empty, the resulting UMR structure will also be missing the same component(s)")

              conv_doc_refresh_button = gr.Button('REFRESH', variant='secondary')

              conv_mdp = gr.Dropdown(
                label="MDP",
                value="",
                choices=INIT_EXP_CHOICES[C.MDP_STAGE2],
                interactive=True,
                allow_custom_value=True,
              )
              conv_tdp = gr.Dropdown(
                label="TDP",
                value="",
                choices=INIT_EXP_CHOICES[C.TDP_STAGE2],
                interactive=True,
                allow_custom_value=True,
              )
              conv_cdlm = gr.Dropdown(
                label="CDLM",
                value="",
                choices=INIT_EXP_CHOICES[C.CDLM],
                interactive=True,
                allow_custom_value=True,
              )
              conv_coref = gr.Dropdown(
                label="Coref",
                value="",
                choices=INIT_EXP_CHOICES[C.COREF],
                interactive=True,
                allow_custom_value=True,
              )

              ### CONVERSION DOC-DATA REFRESH EVENT
              conv_doc_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Dropdown(value=[C.MDP_STAGE2, C.TDP_STAGE2, C.CDLM, C.COREF], allow_custom_value=True, visible=False),
                outputs=[conv_mdp, conv_tdp, conv_cdlm, conv_coref]
              )

            with gr.Tab("CONVERSION"):
              gr.Markdown('#### AMR2UMR Conversion')
              gr.Markdown('* all fields MUST be populated to run')
              conv_input, conv_output = build_input_outputs(
                input_label="AMR Input (`input`)", input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.AMRS])
              conv_alignment = gr.Radio(
                label="AMR Alignment",
                info="REQUIRED if integrating document-level information",
                choices=[C.LEAMR, C.IBM_PARSER],
                value=C.LEAMR,
                interactive=True
              )

              with gr.Group():
                conv_ud = gr.Dropdown(
                  label="UDPipe Input",
                  info="UD CoNLLU JSON file",
                  value="",
                  choices=INIT_EXP_CHOICES[C.UDPIPE],
                  interactive=True,
                  allow_custom_value=True
                )
                conv_doc2snt_mapping = gr.Dropdown(
                  label="doc2snt Mapping",
                  info="Mapping from Data ID to int index in sent enumeration",
                  value="",
                  choices=INIT_EXP_CHOICES[C.DOC2SNT_MAPPING],
                  allow_custom_value=True,
                  interactive=True
                )
                conv_split_mapping = gr.Dropdown(
                  label="Split Mapping",
                  info="REQUIRED in order to merge fragments if split into fragments during preprocessing ; otherwise optional",
                  value="",
                  choices=INIT_EXP_CHOICES[C.SPLIT_MAPPING],
                  allow_custom_value=True,
                  interactive=True
                )

              conv_run_button, conv_refresh_button, conv_info = build_run_button_info(with_refresh_button=True)

              ### CONVERSION RUN EVENT
              define_long_running_event(
                conv_run_button,
                fn=run_conversion,
                inputs=[conv_input, conv_output, conv_alignment, conv_split_mapping, conv_ud, conv_doc2snt_mapping,
                        conv_mdp, conv_tdp, conv_cdlm, conv_coref],
                outputs=conv_info
              )

              ### CONVERSION REFRESH EVENT
              conv_refresh_button.click(
                fn=refresh_exp,
                inputs=gr.Dropdown(value=[C.AMRS, C.UDPIPE], allow_custom_value=True, visible=False),
                outputs=[conv_input, conv_ud]
              ).then(
                fn=refresh_data,
                inputs=gr.Dropdown(value=[C.DOC2SNT_MAPPING, C.SPLIT_MAPPING], allow_custom_value=True, visible=False),
                outputs=[conv_doc2snt_mapping, conv_split_mapping]
              )

    with gr.Tab("Evaluation"):
      gr.Markdown("## UMR Parsing Evaluation")
      gr.Markdown("* Project Homepage: [https://github.com/sxndqc/UMR-Inference](https://github.com/sxndqc/UMR-Inference)")
      eval_home, eval_pred, eval_gold = build_input_outputs(
        input_label="Predictions", input_info="Prediction UMR dir containing files with same names in Gold",
        output_label="Gold", output_info="Gold UMR dir containing files with same names in Predictions",
        output_value=D.UMR_EN, home_value=D.UMR_EVAL, input_as_dropdown=True, output_as_dropdown=True,
        input_choices=INIT_EXP_CHOICES[C.CORPUS], output_choices=[D.UMR_EN] + INIT_DATA_CHOICES[C.CORPUS][1:]
      )
      eval_button, eval_refresh_button, eval_info = build_run_button_info(with_refresh_button=True)

      ### EVAL RUN EVENT
      define_long_running_event(
        eval_button,
        fn=run_eval,
        inputs=[eval_pred, eval_gold, eval_home],
        outputs=eval_info
      )

      ### EVAL REFRESH EVENT
      eval_refresh_button.click(
        fn=refresh_exp,
        inputs=gr.Textbox(value=C.CORPUS, visible=False),
        outputs=eval_pred
      ).then(
        fn=lambda : {'choices': [D.UMR_EN] + refresh_data(C.CORPUS)['choices'][1:], '__type__': 'update'},
        outputs=eval_gold
      )

    with gr.Tab("Analysis"):
      gr.Markdown("## Data Analysis & Statistics")
      with gr.Tabs():
        with gr.Tab("UMR"):
          # noinspection PyTypeChecker
          umr_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            value="",
            choices=INIT_DATA_CHOICES[C.CORPUS] + INIT_EXP_CHOICES[C.CORPUS][1:], # remove empty choice from exp choices
            allow_custom_value=True,
            interactive=True
          )

          with gr.Row():
            umr_analysis_load_button = gr.Button("Load", variant='primary')
            umr_analysis_refresh_button = gr.Button("Refresh", variant='secondary')
            umr_analysis_refresh_button.click(
              fn=lambda : {'choices': refresh_data(C.CORPUS)['choices'] + refresh_exp(C.CORPUS)['choices'][1:], '__type__': 'update'},
              outputs=umr_analysis_input
            )
          with gr.Row():
            umr_analysis_search_var = gr.Textbox(
              label="Variable",
              info="this will erase previous inspection results",
              interactive=True,
              visible=False,
            )
            umr_analysis_search_button = gr.Button("Search", variant="primary", visible=False)
          with gr.Row():
            umr_analysis = gr.Highlightedtext(
              label="Inspection",
              combine_adjacent=True,
              show_legend=True,
            )
            umr_stats = gr.Json(label="Summary Statistics")
          umr_analysis_load_button.click(
            fn=analysis.inspect_umr,
            inputs=umr_analysis_input,
            outputs=[umr_analysis, umr_stats]
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'}, {'visible': True, '__type__': 'update'}),
            outputs=[umr_analysis_search_var, umr_analysis_search_button]
          )
          umr_analysis_search_button.click(
            fn=analysis.search_var_umr,
            inputs=[umr_analysis_input, umr_analysis_search_var],
            outputs=umr_analysis,
          )

        with gr.Tab("AMR"):
          gr.Markdown("* ALIGNMENT ONLY, as of March 2024")
          # noinspection PyTypeChecker
          amr_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            info="for LEAMR, use the output parse/BLINK name, before extensions added by LEAMR",
            value="",
            choices=INIT_EXP_CHOICES[C.AMRS],
            allow_custom_value=True,
            interactive=True
          )
          amr_analysis_doc2snt_mapping = gr.Dropdown(
            label="Doc2Snt Mapping",
            info="if specified, IBM-style alignment will be assumed (otherwise, LEAMR)",
            value="",
            choices=INIT_EXP_CHOICES[C.DOC2SNT_MAPPING],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            amr_analysis_load_button = gr.Button("Load", variant='primary')
            amr_analysis_refresh_button = gr.Button("Refresh", variant='secondary')
            amr_analysis_refresh_button.click(
              fn=refresh_exp,
              inputs=gr.Textbox(value=C.AMRS, visible=False),
              outputs=amr_analysis_input
            ).then(
              fn=refresh_data,
              inputs=gr.Textbox(value=C.DOC2SNT_MAPPING, visible=False),
              outputs=amr_analysis_doc2snt_mapping
            )

          with gr.Row():
            amr_search_id = gr.Textbox(
              label="AMR ID",
              info="# ::id {THIS}",
              interactive=True,
              visible=False,
            )
            amr_search_target = gr.Textbox(
              label="Token(s) or AMR Variable",
              info="(1) Span of tokens (from :tok) or (2) AMR Variable",
              interactive=True,
              visible=False,
            )
          amr_search_button = gr.Button("Search", variant='primary', visible=False)

          with gr.Row():
            amr_analysis = gr.Highlightedtext(
              label="Inspection",
              combine_adjacent=True,
              show_legend=True,
            )
            amr_stats = gr.Json(label="Summary Statistics")

          amr_analysis_load_button.click(
            fn=analysis.load_amr,
            inputs=[amr_analysis_input, amr_analysis_doc2snt_mapping],
            outputs=[amr_analysis, amr_stats]
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'},) * 3,
            outputs=[amr_search_id, amr_search_target, amr_search_button]
          )
          amr_search_button.click(
            fn=analysis.search_amr,
            inputs=[amr_analysis_input, amr_search_id, amr_search_target, amr_analysis_doc2snt_mapping],
            outputs=amr_analysis
          )

        with gr.Tab("MDG"):
          # noinspection PyTypeChecker
          mdg_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            value="",
            choices=INIT_EXP_CHOICES[C.MDP],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            mdg_analysis_load_button = gr.Button("LOAD", variant='primary')
            mdg_analysis_refresh_button = gr.Button("Refresh", variant="secondary")
            mdg_analysis_refresh_button.click(
              fn=refresh_exp,
              inputs=gr.Textbox(value=C.MDP, visible=False),
              outputs=mdg_analysis_input
            )
          with gr.Row():
            mdg_doc_id = gr.Number(
              label="Document ID",
              info="<doc id={THIS}>",
              value=1,
              interactive=True,
              visible=False,
            )
            mdg_anno = gr.Textbox(
              label="Annotation",
              info="{}_{}_{} in the first (or third) column",
              value="-3_-3_-3",
              interactive=True,
              visible=False,
            )
          mdg_search_button = gr.Button("Search", variant='primary', visible=False)
          with gr.Row():
            mdg_analysis = gr.Highlightedtext(
              label="Inspection",
              combine_adjacent=True,
              show_legend=True,
            )
            mdg_stats = gr.Json(label="Summary Statistics")
          mdg_analysis_load_button.click(
            fn=analysis.load_mtdg,
            inputs=mdg_analysis_input,
            outputs=[mdg_analysis, mdg_stats]
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'},) * 3,
            outputs=[mdg_doc_id, mdg_anno, mdg_search_button]
          )
          mdg_search_button.click(
            fn=analysis.search_anno_mtdg,
            inputs=[mdg_analysis_input, mdg_doc_id, mdg_anno],
            outputs=mdg_analysis
          )

        with gr.Tab("TDG"):
          # noinspection PyTypeChecker
          tdg_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            value="",
            choices=INIT_EXP_CHOICES[C.TDP],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            tdg_analysis_load_button = gr.Button("LOAD", variant='primary')
            tdg_analysis_refresh_button = gr.Button("Refresh", variant="secondary")
            tdg_analysis_refresh_button.click(
              fn=refresh_exp,
              inputs=gr.Textbox(value=C.TDP, visible=False),
              outputs=tdg_analysis_input
            )
          with gr.Row():
            tdg_doc_id = gr.Number(
              label="Document ID",
              info="<doc id={THIS}>",
              value=1,
              interactive=True,
              visible=False,
            )
            tdg_anno = gr.Textbox(
              label="Annotation",
              info="{}_{}_{} in the first (or third) column",
              value="-3_-3_-3",
              interactive=True,
              visible=False,
            )
          tdg_search_button = gr.Button("Search", variant='primary', visible=False)
          with gr.Row():
            tdg_analysis = gr.Highlightedtext(
              label="Inspection",
              combine_adjacent=True,
              show_legend=True,
            )
            tdg_stats = gr.Json(label="Summary Statistics")
          tdg_analysis_load_button.click(
            fn=analysis.load_mtdg,
            inputs=tdg_analysis_input,
            outputs=[tdg_analysis, tdg_stats]
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'},) * 3,
            outputs=[tdg_doc_id, tdg_anno, tdg_search_button]
          )
          tdg_search_button.click(
            fn=analysis.search_anno_mtdg,
            inputs=[tdg_analysis_input, tdg_doc_id, tdg_anno],
            outputs=tdg_analysis
          )

        with gr.Tab("CDLM"):
          cdlm_analysis_input = gr.Dropdown(
            label="Input",
            value="",
            choices=INIT_EXP_CHOICES[C.CDLM],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            cdlm_analysis_load_button = gr.Button("LOAD", variant="primary")
            cdlm_analysis_refresh_button = gr.Button("Refresh", variant="secondary")
            cdlm_analysis_refresh_button.click(
              fn=refresh_exp,
              inputs=gr.Textbox(value=C.CDLM, visible=False),
              outputs=cdlm_analysis_input
            )
          with gr.Row():
            cdlm_analysis_cluster_id = gr.Number(
              label="Cluster ID",
              minimum=0,
              interactive=True,
              visible=False,
            )
            cdlm_analysis_search_button = gr.Button("Search", variant="primary", visible=False)
          with gr.Row():
            cdlm_analysis = gr.Highlightedtext(
              label="Inspection",
              combine_adjacent=True,
              show_legend=True,
            )
            cdlm_stats = gr.Json(label="Summary Statistics")
          cdlm_analysis_load_button.click(
            fn=analysis.load_cdlm,
            inputs=cdlm_analysis_input,
            outputs=[cdlm_analysis, cdlm_stats]
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'}, {'visible': True, '__type__': 'update'}),
            outputs=[cdlm_analysis_cluster_id, cdlm_analysis_search_button]
          )
          cdlm_analysis_search_button.click(
            fn=analysis.search_cluster_cdlm,
            inputs=[cdlm_analysis_input, cdlm_analysis_cluster_id],
            outputs=cdlm_analysis
          )

        with gr.Tab("coref"):
          gr.Markdown("* NOT IMPLEMENTED")
          with gr.Row():
            coref_analysis_input = gr.Textbox(label="Input (`input`)", interactive=True)
            coref_load_button = gr.Button("Load", variant='secondary')
          with gr.Row():
            coref_analysis_doc_id = gr.Textbox(
              label="Document ID",
              interactive=True,
              visible=False,
            )
            coref_analysis_cluster_id = gr.Number(
              label="Cluster ID",
              minimum=0,
              interactive=True,
              visible=False,
            )
            coref_analysis_search_button = gr.Button("Search", variant="primary", visible=False)
          with gr.Row():
            coref_analysis = gr.Json(label="Inspection",)
            coref_stats = gr.Json(label="Summary Statistics")
          # coref_load_button.click(
          #   fn=analysis.load_coref,
          #   inputs=coref_analysis_input,
          #   outputs=[coref_analysis, coref_stats]
          # ).then(
          #   fn=lambda: ({'visible': True, '__type__': 'update'},)*3,
          #   outputs=[coref_analysis_doc_id, coref_analysis_cluster_id, coref_analysis_search_button]
          # )
          # coref_analysis_search_button.click(
          #   fn=analysis.search_cluster_coref,
          #   inputs=[coref_analysis_input, coref_analysis_doc_id, coref_analysis_cluster_id],
          #   outputs=coref_analysis
          # )

        with gr.Tab("JSON"):
          json_analysis_input = gr.Dropdown(
            label="Input",
            info="does not accept .jsonlines",
            value="",
            choices=INIT_DATA_CHOICES[C.JSON] + INIT_EXP_CHOICES[C.JSON][1:],
            allow_custom_value=True,
            interactive=True,
          )

          with gr.Row():
            json_analysis_run_button = gr.Button("RUN", variant="primary")
            json_analysis_refresh_button = gr.Button("Refresh", variant='secondary')
            json_analysis_refresh_button.click(
              fn=lambda : {'choices': refresh_data(C.JSON)['choices'] + refresh_exp(C.JSON)['choices'][1:], '__type__': 'update'},
              outputs=json_analysis_input
            )
          json_analysis_info = gr.JSON(label='Info')
          json_analysis_run_button.click(
            fn=lambda x: io_utils.load_json(x),
            inputs=json_analysis_input,
            outputs=json_analysis_info
          )

    # with gr.Tab("Misc."):
    #   gr.Markdown("## Utilities")
    #
    #   with gr.Tabs():
    #     with gr.Tab("File & Dir"):
    #       gr.Markdown("* Move or copy dirs or files; collapse and reopen to see applied changes")
    #
    #       rename_source, rename_target = build_input_outputs(input_label="From", output_label="To", as_row=True)
    #       with gr.Row():
    #         rename_fs = gr.FileExplorer(
    #           root_dir=D.PROJECT,
    #           ignore_glob=f"**/.*",
    #           interactive=True,
    #         )
    #         app.load(
    #           lambda: gr.FileExplorer(
    #             root_dir=D.PROJECT,
    #             ignore_glob=f"**/.*",
    #             interactive=True,
    #           ), None, rename_fs, every=5
    #         )
    #         rename_fs.change(
    #           fn=lambda x, y, z: ({'value': x[0] if x and len(x) > 0 else y, "__type__": "update"},
    #                               {'value': x[0] if x and len(z) == 0 and len(x) > 0 else z, "__type__": "update"}),
    #           inputs=[rename_fs, rename_source, rename_target],
    #           outputs=[rename_source, rename_target]
    #         )
    #         rename_button, copy_button, rename_info = build_run_button_info(
    #           run_button_label="Move", with_refresh_button=True, refresh_button_label="Copy")
    #         rename_button.click(
    #           fn=io_utils.move,
    #           inputs=[rename_source, rename_target],
    #           outputs=rename_info
    #         )
    #         copy_button.click(
    #           fn=io_utils.copy,
    #           inputs=[rename_source, rename_target],
    #           outputs=rename_info
    #         )

  # global event
  parsing_set_subroot_button.click(
    fn=lambda : ({'interactive': False, '__type__': 'update'},)*2,
    outputs=[parsing_subroot, parsing_set_subroot_button]
  ).then(
    fn=set_parsing_subroot_fn,
    inputs=parsing_subroot,
    outputs=[
      parsing_subroot_info,
      conv_output,
      eval_pred,
      sapienza_output,
      ibm_output,
      amrbart_output,
      blink_output,
      mbse_output,
      mdp_stage1_output,
      mdp_merge_output,
      mdp_stage2_output,
      tdp_stage1_output,
      tdp_merge_output,
      tdp_stage2_output,
      cdlm_prep_output,
      cdlm_output,
      coref_output,
      conv_ud_output,
    ]
  )
  parsing_reset_subroot_button.click( # technically local
    fn=lambda : ({'interactive': True, '__type__': 'update'},)*2,
    outputs=[parsing_subroot, parsing_set_subroot_button]
  )

################################################################################
if __name__ == '__main__':
  misc_utils.init_logging(args.debug, suppress_penman=True, suppress_httpx=True)
  app.launch(
    server_name="0.0.0.0",
    inbrowser=True,
    server_port=args.port,
    share=args.share,
    quiet=True,
    debug=args.debug
  )
