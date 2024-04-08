#! /usr/bin/python3
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
from itertools import groupby
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
CUR_EXP = PROJ_EXP

# conditions for updating choices
CONDITIONS = {
  # `x`: dir
  C.CORPUS: lambda x: ( len([xx for xx in x if xx.endswith(C.TXT)]) > 0 and
                        any( xx.startswith('english_umr') for xx in x if xx.endswith(C.TXT)) ),
  C.TMP: lambda x: x == C.TMP,
  # `x`: input file
  D.SNTS_TXT: lambda x: x.endswith(D.SNTS_TXT),
  D.TOKS_TXT: lambda x: x.endswith(D.TOKS_TXT),
  D.AMR_TXT: lambda x: x.endswith(D.AMR_TXT) ,
  D.AMRS_TXT: lambda x: x.endswith(D.AMRS_TXT),
  D.AMRS_JSONL: lambda x: x.endswith(D.AMRS_JSONL),
  D.COREF_JSONLINES: lambda x: x.endswith(D.COREF_JSONLINES),
  D.DOCS_TXT: lambda x: C.DOCS in x and x.endswith(C.TXT),
  # `x`: intermdiate files produced during prp that are also used in conversion
  C.DOC2SNT_MAPPING: lambda x: x == D.DOC2SNT_MAPPING_JSON,
  C.SPLIT_MAPPING: lambda x: x == D.SPLIT_MAPPING_JSON,
  C.UDPIPE: lambda x: x == D.UDPIPE_JSON,
  # `x`: output file
  C.AMRS: lambda x: C.AMR in x and x.endswith(C.TXT),
  C.MDP: lambda x: C.MODAL in x or C.MDP_PROMPT in x,
  C.MDP_STAGE1: lambda x: (C.MODAL in x or C.MDP_PROMPT in x) and x.endswith(C.STAGE1_TXT),
  C.MDP_STAGE2: lambda x: (C.MODAL in x or C.MDP_PROMPT in x) and x.endswith(C.STAGE2_TXT),
  C.TDP: lambda x: C.TEMPORAL in x or C.THYME_TDG in x,
  C.TDP_STAGE1: lambda x: (C.TEMPORAL in x or C.THYME_TDG in x) and x.endswith(C.STAGE1_TXT),
  C.TDP_MERGED_STAGE1: lambda x: (C.TEMPORAL in x or C.THYME_TDG in x or C.MERGED in x) and x.endswith(C.STAGE1_TXT),
  C.TDP_STAGE2: lambda x: (C.TEMPORAL in x or C.THYME_TDG in x) and x.endswith(C.STAGE2_TXT),
  C.STAGE1: lambda x: C.STAGE1 in x,
  C.CDLM: lambda x: C.CDLM in x and x.endswith(C.CONLL),
  C.COREF: lambda x: C.COREF in x and x.endswith(C.JSONLINES),
  # by extension
  C.JSON: lambda x: x.endswith(C.JSON),
}

# general refresh behavior
def refresh(root, keys: Optional[Union[str, List]] = None, get_abs_path=False,
            sort_options=False, with_empty_option=False):
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

  if get_abs_path:
    data = {k: [os.path.abspath(vv) for vv in v] for k,v in data.items()}

  out = dict()
  for k,v in data.items():
    v = sorted(v) if sort_options else list(v)
    if with_empty_option:
      v = [""] + v
    out[k] = v
  return out

# event fns
def get_default_value_from_choices(choices):
  if not choices:
    return ""
  size = len(choices)
  if size == 0:
    return ""
  elif size == 1:
    return choices[0]
  else:
    value = choices[0]
    if value == "":
      value = choices[1]
    return value

def refresh_choices(root, keys, values=None, get_abs_path=False, with_empty_option=True):
  if isinstance(keys, str):
    keys = [keys]
  choices = refresh(
    root=root, keys=keys, get_abs_path=get_abs_path, with_empty_option=with_empty_option)
  if len(keys) == 1:
    choices = sorted(choices[keys[0]])
    out = {'choices': choices, '__type__': 'update',
           'value': values if values else get_default_value_from_choices(choices)}
    if values == C.NONE:
      out.pop('value')
  else:
    out = []
    for i, k in enumerate(keys):
      k_choices = sorted(choices[k])
      k_out = {'choices': k_choices, '__type__': 'update',
               'value': values[i] if values else get_default_value_from_choices(k_choices)}
      if k_out['value'] == C.NONE:
        k_out.pop('value')
      out.append(k_out)
  return out

print("Collecting init choices..")
INIT_DATA_CHOICES = refresh(PROJ_DATA, sort_options=True, with_empty_option=True)
INIT_EXP_CHOICES = refresh(PROJ_EXP, sort_options=True, with_empty_option=True)

### pyenv venvs
VENVS = ['']
# first check pyenv
print("Collecting venvs..")
for x in sorted(os.listdir(os.path.join(os.getenv('PYENV_ROOT'), 'versions'))):
  if not x.startswith('2') and not x.startswith('3'):
    VENVS.append(x)

# assumed that CUDA lives in `usr/local`
CUDAS = ['']
print("Collecting CUDAs..")
for x in sorted(os.listdir('/usr/local')):
  if x.startswith('cuda-'):
    CUDAS.append(x)

GPUS, CVDS = [''], ['']
print("Collecting GPU info..")
for i in range(torch.cuda.device_count()):
  CVDS.append(f'CUDA_VISIBLE_DEVICES={i}')
  GPUS.append(torch.cuda.get_device_name(i))

def set_parsing_subroot_fn(parsing_subroot_value: str):
  subroot_fdir = os.path.join(PROJ_EXP, parsing_subroot_value)
  os.makedirs(subroot_fdir, exist_ok=True)
  tmp_fdir = os.path.join(subroot_fdir, C.TMP)
  os.makedirs(tmp_fdir, exist_ok=True)
  logger.info("Current Parsing Subroot: %s" % subroot_fdir)
  return (subroot_fdir,)*3 + (tmp_fdir,)*14

def define_long_running_event(button, fn=None, inputs=None, outputs=None,
                              concurrency_limit: Union[str, None] = 'default',
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
  button.click(
    fn=fn, inputs=inputs, outputs=outputs, concurrency_limit=concurrency_limit,
  ).then(
    fn=fn2, inputs=inputs2, outputs=outputs2, concurrency_limit=None
  )

def run_preprocessing(input_fpath, output_fpath, aggression, snt2tok, max_per_docs, partition_strategy):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname = io_utils.get_dirname(output_fpath, mkdir=True)
  log_fpath = os.path.join(dirname, D.PRP_LOG)

  cmd = [
    subprocess_utils.PYTHON,
    "scripts/preprocess_umr.py",
    "-i",
    input_fpath,
    '-o',
    output_fpath,
    '--aggression',
    str(aggression),
    '--max-per-doc',
    " ".join([str(x) for x in max_per_docs]),
    '--partition-strategy',
    partition_strategy,
    '--snt-to-tok',
    snt2tok,
  ]
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
    yield x

def run_sapienza_amr_parser(input_fpath, output_fpath, model_home, model, checkpoint,
                            tokenize, beam, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  # maybe `output_fpath` is a dir?
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if model == C.SPRING:
    model_name = "spring"
    model_fname = f'{model_name}_{checkpoint.split(".")[0]}.{C.AMR_TXT}'
  else:
    model_name = "leak_distill"
    model_fname = f'{model_name}_{"ckpt12" if "_12_" in checkpoint else "ckpt20"}.{C.AMR_TXT}'

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(dirname, model_fname)
  log_fpath = os.path.join(dirname, D.SAPIENZA_LOG)

  assert os.path.exists(model_home)
  if not os.path.exists(checkpoint):
    checkpoint = os.path.join(model_home, 'checkpoints', checkpoint)
  assert os.path.exists(checkpoint)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_sapienza.sh',
    input_fpath,
    output_fpath,
    model_home,
    f'configs/config_{model_name}.yaml',
    checkpoint,
    str(beam),
    venv,
  ]
  if not tokenize:
    cmd.append("--snt-to-tok")
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_ibm_amr_parser(input_fpath, output_fpath, model_home, torch_hub, model,
                       seeds, tokenize, batch_size, beam, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home, torch_hub = io_utils.get_abs_paths(
    input_fpath, output_fpath, model_home, torch_hub)

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

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    seeds_str = ':'.join([str(seed) for seed in seeds])
    output_fpath = io_utils.get_unique_fpath(
      dirname, f"{C.IBM}_{model[:4]}_seeds{seeds_str}.{C.AMR_TXT}")
  log_fpath = os.path.join(dirname, D.IBM_LOG)
  logger.info("Logging at `%s`", log_fpath)

  # get checkpoints
  checkpoints = []
  checkpoint_home = os.path.join(torch_hub, C.DATA)
  if C.DATA not in os.listdir(torch_hub):
    checkpoint_home = os.path.join(torch_hub, os.pardir, C.DATA)
  # noinspection PyTypeChecker
  checkpoint_home = os.path.join(
    checkpoint_home, model, C.MODELS, f'{model}-structured-bart-large')
  for seed in seeds:
    # noinspection PyTypeChecker
    checkpoints.append(
      os.path.join(checkpoint_home, f'seed{seed}', 'checkpoint_wiki.smatch_top5-avg.pt'))

  assert os.path.exists(model_home)

  # noinspection PyTypeChecker
  cmd = [
    subprocess_utils.BASH,
    './bin/run_ibm.sh',
    input_fpath,
    output_fpath,
    model_home,
    ':'.join(checkpoints),
    str(batch_size),
    str(beam),
    venv,
  ]
  if tokenize:
    cmd.append('--tokenize')
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_amrbart(input_fpath, output_fpath, model_home, model, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    model_fname = f"{C.AMRBART}_{model.split('-')[-3]}.{C.AMR_TXT}"
    output_fpath = io_utils.get_unique_fpath(dirname, model_fname)
  log_fpath = os.path.join(dirname, D.AMRBART_LOG)

  if os.path.basename(model_home) == 'fine-tune':
    model_home = os.path.dirname(model_home)
  assert os.path.exists(model_home)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_amrbart.sh',
    input_fpath,
    output_fpath,
    model_home,
    f'xfbai/{model}',
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_amr_postprocessing(input_fpath, output_fpath, doc2snt_mapping=None, snts=None):
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname = io_utils.get_dirname(output_fpath, mkdir=True)
  log_fpath = os.path.join(dirname, D.AMR_POST_LOG)

  cmd = [
    subprocess_utils.PYTHON,
    "scripts/amr_postprocessing.py",
    "-i",
    input_fpath,
    '-o',
    output_fpath
  ]
  if doc2snt_mapping:
    cmd += ['--doc2snt', doc2snt_mapping]
  if snts:
    cmd += ['--snts', snts]
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
    yield x

def run_blink(input_fpath, output_fpath, model_home, models, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # blink inherits the input fname
  input_fname = os.path.basename(input_fpath)

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    if input_fname.endswith(C.AMR_TXT):
      output_fname = input_fname.replace(f'.{C.AMR_TXT}', f'_{C.BLINK}.{C.AMR_TXT}')
    else:
      output_fname = input_fname.replace(f'.{C.TXT}', f'_{C.BLINK}.{C.AMR_TXT}')
    output_fpath = io_utils.get_unique_fpath(dirname, output_fname)
  log_fpath = os.path.join(dirname, D.BLINK_LOG)

  assert os.path.exists(model_home)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_blink.sh',
    input_fpath,
    output_fpath,
    model_home,
    models,
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_leamr(input_fpath, model_home, cuda, cvd):
  # get absolute paths
  input_fpath, model_home = io_utils.get_abs_paths(input_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # there is no output; should log at input dir
  dirname = os.path.dirname(input_fpath)
  log_fpath = os.path.join(dirname, D.LEAMR_LOG)

  assert os.path.exists(model_home)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_leamr.sh',
    input_fpath,
    model_home,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

# utility
def all_equal(iterable):
  g = groupby(iterable)
  return next(g, True) and not next(g, False)

def run_mbse(input_fpath_list, output_fpath, model_home, venv):
  # get absolute paths
  output_fpath, model_home = io_utils.get_abs_paths(output_fpath, model_home)

  if len(input_fpath_list) == 0:
    msg = "At least one input AMR file is required"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  amr_ids = []
  for i, input_fpath in enumerate(input_fpath_list):
    cur_amr_ids = []

    input_fpath = os.path.abspath(input_fpath)
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
    input_fpath_list[i] = input_fpath

    # for sanity check
    metas, _ = amr_utils.load_amr_file(input_fpath)
    for meta in metas:
      cur_amr_id = meta[C.ID]
      cur_amr_ids.append(cur_amr_id)
      assert C.SNT in meta
      assert C.TOK in meta
    amr_ids.append(cur_amr_ids)

  # sanity check
  for cur_amr_ids in zip(*amr_ids):
    assert all_equal(cur_amr_ids), f"ids not matching: `{', `'.join(cur_amr_ids)}`"

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    canonical_input_fname = io_utils.get_canonical_fname(input_fpath_list[0])
    output_fpath = io_utils.get_unique_fpath(
      dirname, f'{canonical_input_fname}.{len(input_fpath_list)}way_{C.MBSE}.{C.AMR_TXT}')
  log_fpath = os.path.join(dirname, D.MBSE_LOG)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_mbse.sh',
    output_fpath,
    model_home,
    venv,
  ] + input_fpath_list
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath, pipe_to_log=True):
    yield x

def run_modal_baseline(input_fpath, output_fpath, model_home, max_seq_length, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # output may be a dir or file
  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  assert is_dir
  canonical_fname = io_utils.get_canonical_fname(input_fpath)
  stage1_fpath = io_utils.get_unique_fpath(
      dirname, f'{canonical_fname}.{C.MODAL}.{C.STAGE1_TXT}')
  stage2_fpath = io_utils.get_unique_fpath(
      dirname, f'{canonical_fname}.{C.MODAL}.{C.STAGE2_TXT}')
  log_fpath = os.path.join(dirname, D.MDP_BASELINE_LOG)

  cmd = [
    subprocess_utils.BASH,
    f'bin/run_modal_baseline.sh',
    input_fpath,
    stage1_fpath,
    stage2_fpath,
    model_home,
    str(max_seq_length),
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_mdp_prompt(input_fpath, output_fpath, model_home, model_dir, model_name, clf_model,
                   model_type, max_seq_length, batch_size, seed, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # output may be a dir or file
  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  assert is_dir
  canonical_fname = io_utils.get_canonical_fname(input_fpath, depth=-1)
  stage1_output_fpath = io_utils.get_unique_fpath(
      dirname, f'{canonical_fname}.{C.MDP_PROMPT}:{model_name}.{C.STAGE1_TXT}')
  stage2_output_fpath = io_utils.get_unique_fpath(
      dirname, f'{canonical_fname}.{C.MDP_PROMPT}:{model_name}.{C.STAGE2_TXT}')
  log_fpath = os.path.join(dirname, D.MDP_PROMPT_LOG)

  if 'xnli-anli' in clf_model:
    clf_model = f'vicgalle/{clf_model}'

  cmd = [
    subprocess_utils.BASH,
    f'bin/run_mdp_prompt.sh',
    input_fpath,
    stage1_output_fpath,
    stage2_output_fpath,
    model_home,
    os.path.join(model_home, model_dir),
    model_name,
    clf_model,
    model_type,
    str(max_seq_length),
    str(batch_size),
    str(seed),
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_temporal_baseline(input_fpath, output_fpath, model_home, model_dir, model_name, clf_model,
                          pipeline, max_seq_length, batch_size, data_type, seed, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  assert is_dir

  pipeline = pipeline.lower()
  if pipeline == C.STAGE1:
    canonical_fname = io_utils.get_canonical_fname(input_fpath, depth=2)
    output_fpath = io_utils.get_unique_fpath(
      dirname, f'{canonical_fname}.{C.TEMPORAL}:{model_name}.{C.STAGE1_TXT}')
  else:
    canonical_fname = io_utils.get_canonical_fname(input_fpath, depth=4)
    output_fpath = io_utils.get_unique_fpath(
        dirname, f'{canonical_fname}.{C.MDP_PROMPT}:{model_name}.{C.STAGE2_TXT}')
  log_fpath = os.path.join(dirname, D.TEMPORAL_BASELINE_LOG)

  cmd = [
    subprocess_utils.BASH,
    f'./bin/run_temporal_baseline.sh',
    input_fpath,
    output_fpath,
    model_home,
    os.path.join(model_home, model_dir),
    model_name,
    clf_model,
    pipeline,
    data_type, # unused if stage 1
    str(max_seq_length),
    str(batch_size),
    str(seed),
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_merge_stage1s(input_fpath_list, output_fpath, include_conceiver=False,
                      include_timex=False, for_thyme_tdg=False):
  # get absolute paths
  output_fpath = os.path.abspath(output_fpath)
  for i, input_fpath in enumerate(input_fpath_list):
    input_fpath = os.path.abspath(input_fpath)
    if not os.path.exists(input_fpath):
      msg = f"Input file `{input_fpath}` does not exist"
      logger.warning(msg)
      gr.Warning(msg)
      return msg
    input_fpath_list[i] = input_fpath

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    canonicals = [io_utils.get_canonical_fname(x, depth=2) for x in input_fpath_list]
    output_fpath = io_utils.get_unique_fpath(
      dirname, f'{":".join(canonicals)}.{C.MERGED}.{C.STAGE1_TXT}')
  log_fpath = os.path.join(dirname, D.MERGE_STAGE1_LOG)

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
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath, pipe_to_log=True):
    yield x

def prepare_thyme_tdg_inputs(input_fpath, output_fpath):
  pass

def run_thyme_tdg(input_fpath, output_fpath, model_home, model_dir, venv, cuda, cvd):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  assert is_dir
  canonical_fname = io_utils.get_canonical_fname(input_fpath)
  model_name = os.path.basename(model_dir)
  if model_name.startswith('checkpoint'):
    model_name = os.path.basename(os.path.dirname(model_dir))
  output_fpath = io_utils.get_unique_fpath(
    dirname, f'{canonical_fname}.{C.THYME_TDG}:{model_name}.{C.STAGE2_TXT}')
  log_fpath = os.path.join(dirname, D.TEMPORAL_BASELINE_LOG)

  cmd = [
    subprocess_utils.BASH,
    f'./bin/run_thyme_tdg.sh',
    input_fpath,
    output_fpath,
    model_home,
    os.path.join(model_home, model_dir),
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def prepare_cdlm_inputs(input_fpath, output_fpath):
  # get absolute paths
  input_fpath, output_fpath = io_utils.get_abs_paths(input_fpath, output_fpath)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  dirname = io_utils.get_dirname(output_fpath, mkdir=True)
  log_fpath = os.path.join(dirname, D.PREP_CDLM_LOG)

  cmd = f"{subprocess_utils.PYTHON} scripts/prepare_cdlm_inputs.py -i {input_fpath} -o {output_fpath}"
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
    yield x

def run_cdlm(input_fpath, output_fpath, model_home, name,  batch_size, gpu_ids, venv, cuda):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # update config
  config = io_utils.load_json(
    os.path.join(model_home, 'configs/config_pairwise_long_reg_span.json'))

  # update
  config['gpu_num'] = gpu_ids if isinstance(gpu_ids, list) else [gpu_ids]
  config['split'] = name
  config['mention_type'] = 'events'
  config['batch_size'] = batch_size
  config['data_folder'] = input_fpath

  # save updated config
  config_fpath = os.path.join(model_home, 'configs/config_pairwise_long_reg_span_tmp.json')
  io_utils.save_json(config, config_fpath)

  # output should be dir
  os.makedirs(output_fpath, exist_ok=True)
  log_fpath = os.path.join(output_fpath, D.CDLM_LOG)

  # now run with new config
  cmd = [
    subprocess_utils.BASH,
    './bin/run_cdlm.sh',
    output_fpath,
    model_home,
    config_fpath,
    name,
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath, cuda=cuda, pipe_to_log=True):
    yield x

def run_coref(input_fpath, output_fpath, model, model_home, checkpoint, venv, cuda, cvd, ):
  # get absolute paths
  input_fpath, output_fpath, model_home = \
    io_utils.get_abs_paths(input_fpath, output_fpath, model_home)

  # maybe `output_fpath` is a dir?
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  if os.path.isdir(input_fpath):
    input_fpath = os.path.join(input_fpath, D.COREF_JSONLINES)

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(
      dirname, f'{model}.{C.JSONLINES}')
  log_fpath = os.path.join(dirname, D.COREF_LOG)

  cmd = [
    subprocess_utils.BASH,
    './bin/run_coref.sh',
    input_fpath,
    output_fpath,
    model_home,
    checkpoint,
    f'{checkpoint}.pt',
    venv,
  ]
  for x in subprocess_utils.run_cmd_webui(
          cmd, log_fpath, cuda=cuda, cvd=cvd, pipe_to_log=True):
    yield x

def run_udpipe(input_fpath, output_fpath):
  # get absolute paths
  input_fpath, output_fpath = io_utils.get_abs_paths(input_fpath, output_fpath)

  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg
  assert os.path.isfile(input_fpath)

  dirname, is_dir = io_utils.get_dirname(
    output_fpath, mkdir=True, get_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(dirname, D.UDPIPE_JSON)
  log_fpath = os.path.join(dirname, D.UDPIPE_LOG)

  cmd = f"{subprocess_utils.PYTHON} scripts/run_udpipe.py -i {input_fpath} -o {output_fpath}"
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
    yield x

def run_conversion(input_fpath, output_fpath, amr_alignment, split_mapping,
                   ud_input, mdp, tdp, cdlm, coref):
  # `input_fpath`: AMR
  if not os.path.exists(input_fpath):
    msg = "Input file does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  assert os.path.isdir(output_fpath)
  log_fpath = os.path.join(output_fpath, D.CONV_LOG)

  cmd = [
    subprocess_utils.PYTHON,
    'scripts/convert_amr2umr.py',
    f'--input {input_fpath}',
    f'--output {output_fpath}',
    f'--aligner {amr_alignment.lower()}',
    f'--udpipe {ud_input}'
  ]
  if mdp:
    cmd.append(f'--modal {mdp}')
  if tdp:
    cmd.append(f'--temporal {tdp}')
  if cdlm:
    cmd.append(f'--cdlm {cdlm}')
  if coref:
    cmd.append(f'--coref {coref}')
  if split_mapping:
    cmd.append(f'--split_mapping {split_mapping}')
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
    yield x

def run_ancast_umr(pred_fpath, gold_fpath, ancast_home):
  # get absolute paths
  pred_fpath, gold_fpath, ancast_home = io_utils.get_abs_paths(pred_fpath, gold_fpath, ancast_home)

  if not os.path.exists(pred_fpath):
    msg = f"Input file `{pred_fpath}` does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # accepts only dirs
  assert os.path.isdir(pred_fpath) # should contain predictions which end with .txt
  assert os.path.isdir(gold_fpath) # gold dir with same filenames as predictions
  assert os.path.exists(ancast_home)
  log_fpath = os.path.join(pred_fpath, D.ANCAST_UMR_LOG)

  cmd = [
    subprocess_utils.PYTHON,
    'scripts/evaluate_umr.py',
    "--pred",
    pred_fpath,
    "--gold",
    gold_fpath,
    "--ancast_home",
    ancast_home
  ]
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
    yield x

def run_ancast_amr(pred_fpath, gold_fpath, ancast_home):
  # get absolute paths
  pred_fpath, gold_fpath, ancast_home = io_utils.get_abs_paths(pred_fpath, gold_fpath, ancast_home)

  if not os.path.exists(pred_fpath):
    msg = f"Input file `{pred_fpath}` does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  # accepts only dirs
  assert os.path.isfile(gold_fpath)
  assert os.path.exists(ancast_home)

  pred_dirname = os.path.abspath(io_utils.get_dirname(pred_fpath))
  pred_canonical_fname = io_utils.get_canonical_fname(pred_fpath)
  # overwrite csvs
  csv_fpath = os.path.join(pred_dirname, f'{pred_canonical_fname}.csv')
  log_fpath = os.path.join(pred_dirname, D.ANCAST_AMR_LOG)

  cmd = [
    subprocess_utils.PYTHON,
    'src/run.py',
    pred_fpath,
    gold_fpath,
    '--output',
    csv_fpath,
    '--format',
    'amr'
  ]
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath, cwd=ancast_home, pipe_to_log=True):
    yield x

def run_smatch(pred_fpath, gold_fpath):
  # get absolute paths
  pred_fpath, gold_fpath = io_utils.get_abs_paths(pred_fpath, gold_fpath)

  if not os.path.exists(pred_fpath):
    msg = f"Input file `{pred_fpath}` does not exist"
    logger.warning(msg)
    gr.Warning(msg)
    return msg

  assert os.path.isfile(pred_fpath)
  assert os.path.isfile(gold_fpath)

  dirname = io_utils.get_dirname(pred_fpath)
  log_fpath = os.path.join(dirname, D.SMATCH_LOG)

  cmd = [
    subprocess_utils.PYTHON,
    'scripts/compute_smatch.py',
    "--pred",
    pred_fpath,
    "--gold",
    gold_fpath,
  ]
  for x in subprocess_utils.run_cmd_webui(cmd, log_fpath):
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
      # noinspection PyShadowingBuiltins
      input = gr.Dropdown(
        label="Input (`input`)" if input_label is None else input_label,
        value=get_default_value_from_choices(input_choices),
        info=input_info,
        choices=input_choices,
        interactive=True,
        allow_custom_value=True,
      )
    else:
      # noinspection PyShadowingBuiltins
      input = gr.Textbox(
        label="Input (`input`)" if input_label is None else input_label,
        value=input_value,
        info=input_info,
        interactive=True,
      )
    if output_as_dropdown:
      output = gr.Dropdown(
        label="Output (`output`)" if output_label is None else output_label,
        value=get_default_value_from_choices(output_choices),
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
  pyenvs = VENVS + ["N/A"]
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
      info="no-cuda if not set",
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

def build_run_button_info(
        run_button_label="RUN", run_button_interactive=True,
        info_label="Output Info", max_lines=5, as_row=False,
        with_refresh_button=False, refresh_button_label="Refresh",):
  with gr.Row() if as_row else gr.Column():
    if with_refresh_button:
      with gr.Row():
        run_button = gr.Button(run_button_label, variant="primary", interactive=run_button_interactive)
        refresh_button = gr.Button(refresh_button_label, variant="secondary")
    else:
      run_button = gr.Button(run_button_label, variant="primary", interactive=run_button_interactive)
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
    concurrency_limit=None,
  ).then(fn=lambda : time.sleep(0.33)).then(
    fn=lambda: {'interactive': True, '__type__': 'update'},
    outputs=global_flush_button,
    concurrency_limit=None,
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
          choices=[0, 1, 2],
          label="Aggresion Setting (`aggr`)",
          info="0 for no clean-up, 1 for minimal fixes; 2 for labeling consistency",
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
        prp_max_per_docs = gr.Checkboxgroup(
          label="Max. per Doc.",
          info="Maximum number of sents per Document",
          choices=[0,30,80],
          value=[30,80],
          interactive=True,
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
        inputs=[prp_input, prp_output, prp_aggression, prp_snt2tok, prp_max_per_docs, prp_strategy],
        outputs=prp_info,
        concurrency_limit=None,
        fn2=lambda x: io_utils.load_json(os.path.join(x, C.PREP), D.SPLIT_MAPPING_JSON),
        inputs2=prp_output,
        outputs2=prp_split_json
      )

    with gr.Tab("Parsing"):
      gr.Markdown("## UMR Parsing Pipeline")
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
                    outputs=sapienza_checkpoint,
                    concurrency_limit=None,
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
              sapienza_run_button, sapienza_refresh_button, sapienza_info = \
                build_run_button_info(with_refresh_button=True)

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
                fn=lambda value: refresh_choices(PROJ_DATA, D.AMRS_TXT, value),
                inputs=sapienza_input,
                outputs=sapienza_input,
                concurrency_limit=None,
              )

            with gr.Tab("IBM"):
              gr.Markdown('#### IBM transition-amr-parser')
              gr.Markdown("* Project Homepage: [https://github.com/IBM/transition-amr-parser/tree/master](https://github.com/IBM/transition-amr-parser/tree/master)")
              ibm_torch_hub = gr.Textbox(
                label="Torch HUB",
                info="default checkpoints location",
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
                    value=True,
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
                fn=lambda value: refresh_choices(PROJ_DATA, D.TOKS_TXT, value),
                inputs=ibm_input,
                outputs=ibm_input,
                concurrency_limit=None,
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
              amrbart_run_button, amrbart_refresh_button, amrbart_info = \
                build_run_button_info(with_refresh_button=True)

              ### AMRBART RUN EVENT
              define_long_running_event(
                amrbart_run_button,
                fn=run_amrbart,
                inputs=[
                  amrbart_input, amrbart_output, amrbart_home, amrbart_model,
                  amrbart_venv, amrbart_cuda, amrbart_cvd
                ],
                outputs=amrbart_info
              )

              ### AMRBART REFRESH EVENT
              amrbart_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_DATA, D.AMRS_JSONL, value),
                inputs=amrbart_input,
                outputs=amrbart_input,
                concurrency_limit=None,
              )

            with gr.Tab("Postprocessing"):
              gr.Markdown('#### AMR Postprocessing')
              gr.Markdown('* checks for `id`, `snt` and `tok` fields; these are added if absent')

              post_input, post_output = build_input_outputs(
                input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[D.AMR_TXT])
              post_doc2snt = gr.Dropdown(
                label="doc2snt Mapping",
                info="Mapping from Data ID to int index in sent enumeration",
                value=get_default_value_from_choices(INIT_DATA_CHOICES[C.DOC2SNT_MAPPING]),
                choices=INIT_DATA_CHOICES[C.DOC2SNT_MAPPING],
                allow_custom_value=True,
                interactive=True
              )
              post_snts = gr.Dropdown(
                label="Sentences",
                info="Original Sentences to overwrite and/or update ::snt in AMR parse",
                value="",
                choices=INIT_DATA_CHOICES[D.SNTS_TXT],
                allow_custom_value=True,
                interactive=True
              )
              def post_update(x):
                if not x:
                  return ""
                dirname = io_utils.get_dirname(x)
                canonical, ext = io_utils.get_canonical_fname(x, get_ext=True)
                return os.path.join(dirname, f'{canonical}.post.{ext}')
              post_input.select(
                fn=post_update,
                inputs=post_input,
                outputs=post_output,
                concurrency_limit=None,
              )
              if len(post_input.value) > 0:
                post_output.value = post_update(post_input.value)
              post_run_button, post_refresh_button, post_info = \
                build_run_button_info(with_refresh_button=True)

              ### AMR-POST RUN EVENT
              define_long_running_event(
                post_run_button,
                fn=run_amr_postprocessing,
                inputs=[post_input, post_output, post_doc2snt, post_snts],
                outputs=post_info,
                concurrency_limit=None,
              )

              ### AMR-POST REFRESH EVENT
              post_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, D.AMR_TXT, value),
                inputs=post_input,
                outputs=post_input,
                concurrency_limit=None,
              ).then(
                fn=lambda *values: refresh_choices(PROJ_DATA, [C.DOC2SNT_MAPPING, D.SNTS_TXT], values),
                inputs=[post_doc2snt, post_snts],
                outputs=[post_doc2snt, post_snts],
                concurrency_limit=None,
              )

            with gr.Tab("MBSE"):
              gr.Markdown("#### Maximum Bayes Smatch Ensemble Distillation")
              gr.Markdown("* Script: [https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py](https://github.com/IBM/transition-amr-parser/blob/master/scripts/mbse.py)")
              mbse_home = gr.Textbox(
                label="Home",
                value=D.IBM,
                interactive=True
              )
              mbse_input = gr.Dropdown(
                label="Input `input`",
                info="Select all",
                choices=INIT_EXP_CHOICES[D.AMR_TXT],
                allow_custom_value=True,
                multiselect=True
              )
              mbse_output = gr.Textbox(label="Output `output`", interactive=True)
              mbse_venv = gr.Dropdown(
                label="venv",
                info="pyenv virtualenv",
                choices=VENVS + ["N/A"],
                value="umr-py3.8-torch-1.13.1-cu11.7" if "umr-py3.8-torch-1.13.1-cu11.7" in VENVS else VENVS[0],
                interactive=True,
              )
              mbse_run_button, mbse_refresh_button, mbse_info = \
                build_run_button_info(with_refresh_button=True)

              ### MBSE RUN EVENT
              define_long_running_event(
                mbse_run_button,
                fn=run_mbse,
                inputs=[mbse_input, mbse_output, mbse_home, mbse_venv],
                outputs=mbse_info
              )

              ### MBSE REFRESH EVENT
              mbse_refresh_button.click(
                fn=lambda : refresh_choices(PROJ_EXP, D.AMR_TXT, values=C.NONE),
                outputs=mbse_input,
                concurrency_limit=None,
              )

            with gr.Tab("BLINK"):
              gr.Markdown("#### BLINK Enitity Linger")
              gr.Markdown("* Project Homepage: [https://github.com/facebookresearch/BLINK](https://github.com/facebookresearch/BLINK)")
              gr.Markdown("Here we prefer LeakDistill's `bin/blinkify.py` script")
              blink_home, blink_input, blink_output = build_input_outputs(
                home_value=D.SAPIENZA, input_as_dropdown=True,
                input_choices=INIT_EXP_CHOICES[D.AMR_TXT]
              )
              # noinspection PyTypeChecker
              blink_models = gr.Textbox(
                label="BLINK models directory",
                info="location of model weights and caches",
                value=os.path.join(D.SAPIENZA, C.BLINK, C.MODELS),
                interactive=True
              )
              blink_venv, blink_cuda, blink_cvd = build_envs()
              blink_run_button, blink_refresh_button, blink_info = \
                build_run_button_info(with_refresh_button=True)

              ### BLINK RUN EVENT
              define_long_running_event(
                blink_run_button,
                fn=run_blink,
                inputs=[blink_input, blink_output, blink_home, blink_models,
                        blink_venv, blink_cuda, blink_cvd],
                outputs=blink_info
              )

              ### BLINK REFRESH EVENT
              blink_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, C.AMRS, value),
                inputs=blink_input,
                outputs=blink_input,
                concurrency_limit=None,
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
                  value=get_default_value_from_choices(INIT_EXP_CHOICES[D.AMR_TXT]),
                  choices=INIT_EXP_CHOICES[D.AMR_TXT],
                  allow_custom_value=True,
                  interactive=True,
                )
              leamr_cuda, leamr_cvd = build_envs(no_venv=True)
              leamr_run_button, leamr_refresh_button, leamr_info = \
                build_run_button_info(with_refresh_button=True)

              ### LEAMR RUN EVENT
              define_long_running_event(
                leamr_run_button,
                fn=run_leamr,
                inputs=[leamr_input, leamr_home, leamr_cuda, leamr_cvd],
                outputs=leamr_info
              )

              ### LEAMR REFRESH EVENT
              leamr_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, C.AMRS, value),
                inputs=leamr_input,
                outputs=leamr_input,
                concurrency_limit=None,
              )

        with gr.Tab("MTDP"):
          gr.Markdown("### Modal & Temporal Dependency Parsing")

          with gr.Tabs():
            with gr.Tab("MDP"):
              gr.Markdown("#### Modal Dependency Parsing")

              mdp_input, mdp_output = build_input_outputs(
                input_as_dropdown=True, input_choices=INIT_DATA_CHOICES[D.DOCS_TXT])
              mdp_refresh_button = gr.Button("REFRESH", variant="secondary")
              mdp_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_DATA, D.DOCS_TXT, value),
                inputs=mdp_input,
                outputs=mdp_input,
                concurrency_limit=None,
              )

              with gr.Tabs():
                with gr.Tab("mdp_prompt"):
                  gr.Markdown('##### MDP Prompt')
                  gr.Markdown("* Project Homepage: [https://github.com/Jryao/mdp_prompt](https://github.com/Jryao/mdp_prompt)")
                  mdp_prompt_home = gr.Textbox(label="Home", value=D.MDP_PROMPT, interactive=True)

                  mdp_output_dir_choices = []
                  for subroot, _, files in os.walk(D.MDP_PROMPT):
                    for file in files:
                      if file.endswith("_pytorch_model.bin"):
                        subroot_rel_fpath = os.path.relpath(subroot, D.MDP_PROMPT)
                        mdp_output_dir_choices.append(subroot_rel_fpath)
                        break
                  mdp_output_dir = get_default_value_from_choices(mdp_output_dir_choices)

                  with gr.Group():
                    with gr.Row():
                      # noinspection PyTypeChecker
                      mdp_prompt_model_dir = gr.Dropdown(
                        label="Model / Output Dir",
                        choices=mdp_output_dir_choices,
                        info="Relative Path from model sub-root",
                        value=mdp_output_dir,
                        interactive=True,
                      )
                      mdp_prompt_model_name = gr.Textbox(
                        label='Model Name',
                        info="dirname corresponds to model name",
                        value=os.path.basename(mdp_output_dir),
                        interactive=True,
                      )
                      mdp_prompt_clf_model = gr.Radio(
                        label="Classifier",
                        info="core PLM",
                        choices=["bert-large-cased", "xlm-roberta-large", "xlm-roberta-large-xnli-anli"],
                        value="bert-large-cased" if any(x.startswith("bert-large-cased") for x in os.listdir(os.path.join(D.MDP_PROMPT, mdp_output_dir))) else "xlm-roberta-large",
                        interactive=True
                      )
                      mdp_prompt_model_type = gr.Radio(
                        label="Model Type",
                        info="MDP classifier type",
                        choices=["end2end", "pipeline_stage1"],
                        value="pipeline_stage1" if "pipeline_stage1" in mdp_output_dir else "end2end",
                        interactive=True,
                      )
                      mdp_prompt_model_dir.select(
                        fn=lambda x: (
                          os.path.basename(x),
                          "bert-large-cased" if any(xx.startswith("bert-large-cased") for xx in os.listdir(os.path.join(D.MDP_PROMPT, x))) else "xlm-roberta-large",
                          "pipeline_stage1" if "pipeline_stage1" in x else "end2end"
                        ),
                        inputs=mdp_prompt_model_dir,
                        outputs=[mdp_prompt_model_name, mdp_prompt_clf_model, mdp_prompt_model_type],
                        concurrency_limit=None,
                      )

                    with gr.Row():
                      mdp_prompt_max_len = gr.Radio(
                        label="Max Seq. Length",
                        value=128,
                        choices=[128,384],
                        interactive=True
                      )
                      mdp_prompt_batch_size = gr.Number(label="Batch Size", value=8, )
                      mdp_prompt_seed = gr.Number(label="Model Seed", value=42)

                  mdp_prompt_venv, mdp_prompt_cuda, mdp_prompt_cvd = build_envs()
                  mdp_prompt_run_button, mdp_prompt_info = build_run_button_info("RUN")

                  ### mdp_prompt RUN EVENT
                  define_long_running_event(
                    mdp_prompt_run_button,
                    fn=run_mdp_prompt,
                    inputs=[mdp_input, mdp_output, mdp_prompt_home, mdp_prompt_model_dir,
                            mdp_prompt_model_name, mdp_prompt_clf_model, mdp_prompt_model_type,
                            mdp_prompt_max_len, mdp_prompt_batch_size, mdp_prompt_seed,
                            mdp_prompt_venv, mdp_prompt_cuda, mdp_prompt_cvd],
                    outputs=mdp_prompt_info
                  )

                with gr.Tab("Baseline"):
                  gr.Markdown('##### Modal Baseline')
                  gr.Markdown("* Project Homepage: N/A")

                  modal_baseline_home = gr.Textbox(label="Home",value=D.MODAL_BASELINE, interactive=True)
                  modal_baseline_max_len = gr.Number(label="Max Seq. Length (FIXED)", value=128, interactive=False, )
                  modal_baseline_venv, modal_baseline_cuda, modal_baseline_cvd = build_envs()
                  modal_baseline_run_button, modal_baseline_info = build_run_button_info()

                  ### modal baseline RUN EVENT
                  define_long_running_event(
                    modal_baseline_run_button,
                    fn=run_modal_baseline,
                    inputs=[mdp_input, mdp_output, modal_baseline_home, modal_baseline_max_len,
                            modal_baseline_venv, modal_baseline_cuda, modal_baseline_cvd],
                    outputs=modal_baseline_info
                  )

            with gr.Tab("TDP"):
              gr.Markdown("#### Temporal Dependency Parsing")
              gr.Markdown("* run TDP Stage 1; merge with MDP Stage1 Events; run TDP Stage 2; run thyme-tdg")

              with gr.Tabs():
                with gr.Tab("baseline"):
                  gr.Markdown('##### Temporal Baseline')
                  gr.Markdown("* Project Homepage: N/A")
                  temporal_home, temporal_input, temporal_output = build_input_outputs(
                    home_value=D.TEMPORAL_BASELINE,  input_as_dropdown=True, input_choices=INIT_DATA_CHOICES[D.DOCS_TXT])

                  temporal_output_dir_choices = []
                  for subroot, _, files in os.walk(D.TEMPORAL_BASELINE):
                    for file in files:
                      if file.endswith("_pytorch_model.bin"):
                        subroot_rel_fpath = os.path.relpath(subroot, D.TEMPORAL_BASELINE)
                        temporal_output_dir_choices.append(subroot_rel_fpath)
                        break
                  temporal_output_dir = get_default_value_from_choices(temporal_output_dir_choices)

                  with gr.Group():
                    with gr.Row():
                      # noinspection PyTypeChecker
                      temporal_model_dir = gr.Dropdown(
                        label="Model / Output Dir",
                        choices=temporal_output_dir_choices,
                        info="Relative Path from model sub-root",
                        value=temporal_output_dir,
                        interactive=True,
                      )
                      temporal_model_name = gr.Textbox(
                        label='Model Name',
                        info="dirname corresponds to model name",
                        value=os.path.basename(temporal_output_dir),
                        interactive=True,
                      )
                      temporal_clf_model = gr.Radio(
                        label="Classifier",
                        info="core PLM",
                        choices=["xlm-roberta-base", "xlm-roberta-large"],
                        value="xlm-roberta-base" if any(x.startswith("xlm-roberta-base") for x in os.listdir(os.path.join(D.TEMPORAL_BASELINE, temporal_output_dir))) else "xlm-roberta-large",
                        interactive=True
                      )
                      temporal_model_dir.select(
                        fn=lambda x: (
                          os.path.basename(x),
                          "xlm-roberta-base" if any(xx.startswith("xlm-roberta-base") for xx in os.listdir(os.path.join(D.TEMPORAL_BASELINE, x))) else "xlm-roberta-large",
                        ),
                        inputs=temporal_model_dir,
                        outputs=[temporal_model_name, temporal_clf_model],
                        concurrency_limit=None,
                      )
                      temporal_pipeline = gr.Radio(
                        label="Pipeline",
                        info="Classifier type",
                        choices=["Stage1", "Stage2"],
                        value="Stage1",
                        interactive=True,
                      )

                    with gr.Row():
                      temporal_max_len = gr.Radio(
                        label="Max Seq. Length",
                        value=384,
                        choices=[128,384],
                        interactive=True
                      )
                      temporal_batch_size = gr.Number(label="Batch Size", value=16)
                      temporal_seed = gr.Number(label="Model Seed", value=42)
                      temporal_data_type = gr.Radio(
                        label="Data Type",
                        choices=['temporal_time', 'temporal_event'],
                        value="temporal_time",
                        interactive=False,
                        visible=False,
                      )
                      temporal_pipeline.select(
                        fn=lambda x: {'interactive': True, 'visible': True, '__type__': 'update'} if x == 'Stage2' \
                                    else {'interactive': False, 'visible': False, '__type__': 'update'},
                        inputs=temporal_pipeline,
                        outputs=temporal_data_type
                      )

                  temporal_venv, temporal_cuda, temporal_cvd = build_envs()
                  temporal_run_button, temporal_refresh_button, temporal_info = \
                    build_run_button_info(with_refresh_button=True)

                  ### temporal baseline RUN EVENT
                  define_long_running_event(
                    temporal_run_button,
                    fn=run_temporal_baseline,
                    inputs=[temporal_input, temporal_output, temporal_home, temporal_model_dir, temporal_model_name,
                            temporal_clf_model, temporal_pipeline, temporal_max_len, temporal_batch_size,
                            temporal_data_type, temporal_seed, temporal_venv, temporal_cuda, temporal_cvd],
                    outputs=temporal_info
                  )

                  ### temporal baseline REFRESH EVENT
                  temporal_refresh_button.click(
                    fn=lambda value: refresh_choices(PROJ_DATA, D.DOCS_TXT, value),
                    inputs=temporal_input,
                    outputs=temporal_input,
                    concurrency_limit=None,
                  )

                with gr.Tab("thyme-tdg"):
                  gr.Markdown('##### Thyme TDG')
                  gr.Markdown("* Project Homepage: [https://github.com/Jryao/thyme_tdg/tree/master](https://github.com/Jryao/thyme_tdg/tree/master)")

                  thyme_home, thyme_input, thyme_output = build_input_outputs(
                    home_value=D.THYME_TDG, input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.TDP_MERGED_STAGE1])

                  thyme_output_dir_choices = []
                  for subroot, _, files in os.walk(D.THYME_TDG):
                    subroot_basename = os.path.basename(subroot)
                    for file in files:
                      if 'model.safetensors' in file:
                        subroot_rel_fpath = os.path.relpath(subroot, D.THYME_TDG)
                        thyme_output_dir_choices.append(subroot_rel_fpath)
                        break
                  thyme_output_dir = get_default_value_from_choices(thyme_output_dir_choices)

                  # noinspection PyTypeChecker
                  thyme_model_dir = gr.Dropdown(
                    label="Model Dir",
                    value=thyme_output_dir,
                    choices=thyme_output_dir_choices,
                    allow_custom_value=True,
                    interactive=True,
                  )

                  thyme_venv, thyme_cuda, thyme_cvd = build_envs(
                    default_venv='umr-py3.8-torch-1.13.1-cu11.7-latest')
                  thyme_run_button, thyme_refresh_button, thyme_info = \
                    build_run_button_info(with_refresh_button=True)

                  ### thyme_tdg RUN EVENT
                  define_long_running_event(
                    thyme_run_button,
                    fn=run_thyme_tdg,
                    inputs=[thyme_input, thyme_output, thyme_home, thyme_model_dir,
                            thyme_venv, thyme_cuda, thyme_cvd],
                    outputs=thyme_info
                  )

                  ### thyme_tdg REFRESH EVENT
                  thyme_refresh_button.click(
                    fn=lambda value: refresh_choices(PROJ_EXP, C.TDP_MERGED_STAGE1, value),
                    inputs=thyme_input,
                    outputs=thyme_input,
                    concurrency_limit=None,
                  )

            with gr.Tab("Merge Stage1s"):
              gr.Markdown("#### Stage 1 Merging")

              stage1_merge_input = gr.Dropdown(
                label="Stage1 Inputs `stage1s`",
                info="Select all",
                choices=INIT_EXP_CHOICES[C.STAGE1],
                allow_custom_value=True,
                multiselect=True
              )
              stage1_merge_output = gr.Textbox(label="Output `output`", interactive=True)

              with gr.Row():
                stage1_merge_conc = gr.Radio(
                  label="Include Conceiver",
                  info="whether the merged Stage1 should include Conceivers",
                  choices=[True, False],
                  value=False,
                  interactive=True,
                )
                stage1_merge_timex = gr.Radio(
                  label="Include Timex",
                  info="whether the merged Stage1 should include Timex",
                  choices=[True, False],
                  value=True,
                  interactive=True,
                )
                stage1_merge_thyme_tdg = gr.Radio(
                  label="For Thyme-TDG",
                  info="whether the merged Stage1 is to be consumed by `thyme_tdg`",
                  choices=[True, False],
                  value=True,
                  interactive=True,
                )
              stage1_merge_run_button, stage1_merge_refresh_button, stage1_merge_info = \
                build_run_button_info("Merge", with_refresh_button=True)

              ### Stage 1 Merging RUN Event
              define_long_running_event(
                stage1_merge_run_button,
                fn=run_merge_stage1s,
                inputs=[stage1_merge_input, stage1_merge_output, stage1_merge_conc,
                        stage1_merge_timex, stage1_merge_thyme_tdg],
                outputs=stage1_merge_info,
                concurrency_limit=None,
              )

              ### Stage 1 Merging REFRESH Event
              stage1_merge_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, C.STAGE1, value),
                inputs=stage1_merge_input,
                outputs=stage1_merge_input,
                concurrency_limit=None,
              )

        with gr.Tab("Coref"):
          gr.Markdown("### Coreference")

          with gr.Tabs():
            with gr.Tab("CDLM"):
              gr.Markdown("Cross-Document Event Coreference")
              gr.Markdown("* Project Homepage: [https://github.com/aviclu/CDLM/tree/main/cross_encoder](https://github.com/aviclu/CDLM/tree/main/cross_encoder)")
              gr.Markdown("[!] requires event detection first --> MDP Stage 1, then proceed with CDLM inputs prep")
              with gr.Group():
                gr.Markdown("&nbsp;&nbsp;&nbsp;1) Prepare CDLM inputs")
                cdlm_prep_input, cdlm_prep_output = build_input_outputs(
                  input_as_dropdown=True, input_choices=INIT_EXP_CHOICES[C.MDP_STAGE1])

              cdlm_prep_button, cdlm_prep_refresh_button, cdlm_prep_info = build_run_button_info(
                run_button_label="Prepare CDLM Inputs", info_label="CDLM Prep Info", with_refresh_button=True)

              ### CDLM PREP RUN EVENT
              define_long_running_event(
                cdlm_prep_button,
                fn=prepare_cdlm_inputs,
                inputs=[cdlm_prep_input, cdlm_prep_output],
                outputs=cdlm_prep_info,
                concurrency_limit=None,
              )

              ### CDLM PREP REFRESH EVENT
              cdlm_prep_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, C.MDP_STAGE1, value),
                inputs=cdlm_prep_input,
                outputs=cdlm_prep_input,
                concurrency_limit=None,
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
                  value=128,
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
                inputs=[cdlm_input, cdlm_output, cdlm_home, cdlm_name,
                        cdlm_batch_size, cdlm_gpus, cdlm_venv, cdlm_cuda],
                outputs=cdlm_info
              )

              ### CDLM REFRESH EVENT
              cdlm_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, C.TMP, value),
                inputs=cdlm_input,
                outputs=cdlm_input,
                concurrency_limit=None,
              )

            with gr.Tab("coref"):
              gr.Markdown("Coreference Resolution")
              gr.Markdown("* caw-coref Homepage: [https://github.com/KarelDO/wl-coref](https://github.com/KarelDO/wl-coref)")
              gr.Markdown("* wl-coref Homepage: [https://github.com/vdobrovolskii/wl-coref](https://github.com/vdobrovolskii/wl-coref)")
              coref_model = gr.Radio(
                label="Model",
                choices=[C.CAW_COREF, C.WL_COREF],
                value=C.CAW_COREF,
                interactive=True
              )
              coref_home, coref_input, coref_output = build_input_outputs(
                home_value=D.CAW_COREF, input_as_dropdown=True,
                input_choices=INIT_EXP_CHOICES[D.COREF_JSONLINES]
              )
              coref_checkpoint = gr.Textbox(
                label="Pretrained Checkpoint Name",
                value="roberta",
                interactive=True,
              )
              coref_model.select(
                fn=lambda x: D.CAW_COREF if x == C.CAW_COREF else D.WL_COREF,
                inputs=coref_model,
                outputs=[coref_home],
                concurrency_limit=None,
              )
              coref_venv, coref_cuda, coref_cvd = build_envs()
              coref_run_button, coref_refresh_button, coref_info = \
                build_run_button_info(with_refresh_button=True)

              ### COREF RUN EVENT
              define_long_running_event(
                coref_run_button,
                fn=run_coref,
                inputs=[coref_input, coref_output, coref_model, coref_home,
                        coref_checkpoint, coref_venv, coref_cuda, coref_cvd],
                outputs=coref_info,
              )

              ### COREF REFRESH EVENT
              coref_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_DATA, D.COREF_JSONLINES, value),
                inputs=coref_input,
                outputs=coref_input,
                concurrency_limit=None,
              )

        with gr.Tab("UDPipe"):
          gr.Markdown("#### UD Parsing")
          gr.Markdown(
            "* UDPipe2 Homepage: [https://lindat.mff.cuni.cz/services/udpipe/](https://lindat.mff.cuni.cz/services/udpipe/)")
          conv_ud_input, conv_ud_output = build_input_outputs(
            input_as_dropdown=True, input_choices=INIT_DATA_CHOICES[D.TOKS_TXT])
          conv_ud_run_button, conv_ud_refresh_button, conv_ud_info = \
            build_run_button_info(with_refresh_button=True)

          ### UDPIPE RUN EVENT
          define_long_running_event(
            conv_ud_run_button,
            fn=run_udpipe,
            inputs=[conv_ud_input, conv_ud_output],
            outputs=conv_ud_info,
            concurrency_limit=None,
          )

          ### UDPIPE REFRESH EVENT
          conv_ud_refresh_button.click(
            fn=lambda value: refresh_choices(PROJ_DATA, D.TOKS_TXT, value),
            inputs=conv_ud_input,
            outputs=conv_ud_input,
            concurrency_limit=None,
          )

        with gr.Tab("Conversion"):
          gr.Markdown("### AMR2UMR Conversion")
          gr.Markdown("* based on [Mapping AMR to UMR](https://aclanthology.org/2023.tlt-1.8) and [umr-guidelines](https://github.com/umr4nlp/umr-guidelines/blob/master/guidelines.md)")

          conv_input, conv_output = build_input_outputs(
            input_label="AMR Input (`input`)", input_as_dropdown=True,
            input_choices=INIT_EXP_CHOICES[D.AMR_TXT]
          )

          with gr.Group():
            gr.Markdown("&nbsp;&nbsp;&nbsp;**REQUIRED** for Conversion")
            conv_alignment = gr.Radio(
              label="AMR Aligner",
              choices=[C.LEAMR, C.IBM],
              value=C.LEAMR,
              interactive=True
            )
            conv_ud = gr.Dropdown(
              label="UD Dep. Parse (UDPipe v2)",
              info="UD CoNLL JSON file",
              value=get_default_value_from_choices(INIT_EXP_CHOICES[C.UDPIPE]),
              choices=INIT_EXP_CHOICES[C.UDPIPE],
              interactive=True,
              allow_custom_value=True
            )

          with gr.Group():
            gr.Markdown("&nbsp;&nbsp;&nbsp;(optional) Document-level Predictions")
            conv_mdp = gr.Dropdown(
              label="Modal Dependency Parsing (MDP)",
              value=get_default_value_from_choices(INIT_EXP_CHOICES[C.MDP_STAGE2]),
              choices=INIT_EXP_CHOICES[C.MDP_STAGE2],
              interactive=True,
              allow_custom_value=True,
            )
            conv_tdp = gr.Dropdown(
              label="Temporal Dependency Parsing (TDP)",
              value=get_default_value_from_choices(INIT_EXP_CHOICES[C.TDP_STAGE2]),
              choices=INIT_EXP_CHOICES[C.TDP_STAGE2],
              interactive=True,
              allow_custom_value=True,
            )
            conv_cdlm = gr.Dropdown(
              label="Cross-Sentence Event Coref. (CDLM)",
              value=get_default_value_from_choices(INIT_EXP_CHOICES[C.CDLM]),
              choices=INIT_EXP_CHOICES[C.CDLM],
              interactive=True,
              allow_custom_value=True,
            )
            conv_coref = gr.Dropdown(
              label="Cross-Sentence Entity Coref. (Coref)",
              value=get_default_value_from_choices(INIT_EXP_CHOICES[C.COREF]),
              choices=INIT_EXP_CHOICES[C.COREF],
              interactive=True,
              allow_custom_value=True,
            )
            conv_split_mapping = gr.Dropdown(
              label="Split Mapping",
              info="REQUIRED in order to merge fragments if split into fragments during preprocessing ; otherwise optional",
              value=get_default_value_from_choices(INIT_DATA_CHOICES[C.SPLIT_MAPPING]),
              choices=INIT_DATA_CHOICES[C.SPLIT_MAPPING],
              allow_custom_value=True,
              interactive=True
            )

          conv_run_button, conv_refresh_button, conv_info = build_run_button_info(with_refresh_button=True)

          ### CONVERSION RUN EVENT
          define_long_running_event(
            conv_run_button,
            fn=run_conversion,
            inputs=[conv_input, conv_output, conv_alignment, conv_split_mapping,
                    conv_ud, conv_mdp, conv_tdp, conv_cdlm, conv_coref],
            outputs=conv_info,
            concurrency_limit=None,
          )

          ### CONVERSION REFRESH EVENT
          conv_exp_refresh_keys = [D.AMR_TXT, C.UDPIPE, C.MDP_STAGE2, C.TDP_STAGE2, C.CDLM, C.COREF]
          conv_refresh_button.click(
            fn=lambda *values: refresh_choices(PROJ_EXP, conv_exp_refresh_keys, values),
            inputs=[conv_input, conv_ud, conv_mdp, conv_tdp, conv_cdlm, conv_coref],
            outputs=[conv_input, conv_ud, conv_mdp, conv_tdp, conv_cdlm, conv_coref],
            concurrency_limit=None,
          ).then(
            fn=lambda value: refresh_choices(PROJ_DATA, C.SPLIT_MAPPING, value),
            inputs=conv_split_mapping,
            outputs=conv_split_mapping,
            concurrency_limit=None,
          )

    with gr.Tab("Evaluation"):
      gr.Markdown('## Parsing Performance Evaluation')

      with gr.Tabs():
        with gr.Tab('AnCast'):
          gr.Markdown("### UMR Parsing Evaluation")
          gr.Markdown("* Project Homepage: [https://github.com/sxndqc/ancast](https://github.com/sxndqc/ancast)")

          ancast_home = gr.Textbox(
            label="Home",
            value=D.ANCAST,
            interactive=True
          )

          with gr.Tabs():
            with gr.Tab("AnCast++"):
              gr.Markdown("#### Full UMR Evaluation")

              ancast_umr_pred, ancast_umr_gold = build_input_outputs(
                input_label="Predictions", input_info="Prediction UMR dir containing files with same names in Gold",
                output_label="Gold", output_info="Gold UMR dir containing files with same names in Predictions",
                output_value=D.UMR_EN, input_as_dropdown=True, output_as_dropdown=True,
                input_choices=INIT_EXP_CHOICES[C.CORPUS], output_choices=[D.UMR_EN] + INIT_DATA_CHOICES[C.CORPUS][1:]
              )
              ancast_umr_run_button, ancast_umr_refresh_button, ancast_umr_info = build_run_button_info(with_refresh_button=True)

              ### AnCast UMR RUN EVENT
              define_long_running_event(
                ancast_umr_run_button,
                fn=run_ancast_umr,
                inputs=[ancast_umr_pred, ancast_umr_gold, ancast_home],
                outputs=ancast_umr_info,
                concurrency_limit=None,
              )

              ### AnCast UMR REFRESH EVENT
              ancast_umr_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, C.CORPUS, value),
                inputs=ancast_umr_pred,
                outputs=ancast_umr_pred,
                concurrency_limit=None,
              ).then(
                fn=lambda value: refresh_choices(PROJ_DATA, C.CORPUS, value),
                inputs=ancast_umr_gold,
                outputs=ancast_umr_gold,
                concurrency_limit=None,
              )

            with gr.Tab("AnCast"):
              gr.Markdown("#### Sentence UMR (or AMR) Evaluation")

              ancast_amr_pred, ancast_amr_gold = build_input_outputs(
                input_label="Predictions", input_info="Prediction UMR Snt Graphs",
                output_label="Gold", output_info="Gold UMR Snt Graphs",
                input_as_dropdown=True, output_as_dropdown=True,
                input_choices=INIT_EXP_CHOICES[D.AMR_TXT], output_choices=INIT_DATA_CHOICES[D.AMRS_TXT]
              )
              ancast_amr_run_button, ancast_amr_refresh_button, ancast_amr_info = build_run_button_info(with_refresh_button=True)

              ### AnCast UMR RUN EVENT
              define_long_running_event(
                ancast_amr_run_button,
                fn=run_ancast_amr,
                inputs=[ancast_amr_pred, ancast_amr_gold, ancast_home],
                outputs=ancast_amr_info,
                concurrency_limit=None,
              )

              ### AnCast UMR REFRESH EVENT
              ancast_amr_refresh_button.click(
                fn=lambda value: refresh_choices(PROJ_EXP, D.AMR_TXT, value),
                inputs=ancast_amr_pred,
                outputs=ancast_amr_pred,
                concurrency_limit=None,
              ).then(
                fn=lambda value: refresh_choices(PROJ_DATA, D.AMRS_TXT, value),
                inputs=ancast_amr_gold,
                outputs=ancast_amr_gold,
                concurrency_limit=None,
              )

        with gr.Tab("Smatch"):
          gr.Markdown("### AMR Parsing Evaluation")
          gr.Markdown("* Project Homepage: [https://github.com/snowblink14/smatch](https://github.com/snowblink14/smatch)")
          gr.Markdown("* Enhanced version based on [https://github.com/mdtux89/amr-evaluation](https://github.com/mdtux89/amr-evaluation) and  [https://github.com/bjascob/amrlib/tree/master/amrlib/evaluate](https://github.com/bjascob/amrlib/tree/master/amrlib/evaluate)")

          smatch_pred, smatch_gold = build_input_outputs(
            input_label="Predictions", input_info="Prediction AMR", output_label="Gold", output_info="Gold AMR",
            input_as_dropdown=True, output_as_dropdown=True,
            input_choices=INIT_EXP_CHOICES[D.AMR_TXT], output_choices=INIT_DATA_CHOICES[D.AMRS_TXT]
          )
          smatch_run_button, smatch_refresh_button, smatch_info = build_run_button_info(with_refresh_button=True)

          ### EVAL RUN EVENT
          define_long_running_event(
            smatch_run_button,
            fn=run_smatch,
            inputs=[smatch_pred, smatch_gold],
            outputs=smatch_info,
            concurrency_limit=None,
          )

          ### EVAL REFRESH EVENT
          smatch_refresh_button.click(
            fn=lambda value: refresh_choices(PROJ_EXP, D.AMR_TXT, value),
            inputs=smatch_pred,
            outputs=smatch_pred,
            concurrency_limit=None,
          ).then(
            fn=lambda value: refresh_choices(PROJ_DATA, D.AMRS_TXT, value),
            inputs=smatch_gold,
            outputs=smatch_gold,
            concurrency_limit=None,
          )

    with gr.Tab("Analysis"):
      gr.Markdown("## Data Analysis & Statistics")
      with gr.Tabs():
        with gr.Tab("UMR"):
          umr_analysis_choices = INIT_DATA_CHOICES[C.CORPUS] + INIT_EXP_CHOICES[C.CORPUS][1:]
          # noinspection PyTypeChecker
          umr_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            value=get_default_value_from_choices(umr_analysis_choices),
            choices=umr_analysis_choices, # remove empty choice from exp choices
            allow_custom_value=True,
            interactive=True
          )

          with gr.Row():
            umr_analysis_load_button = gr.Button("Load", variant='primary')
            umr_analysis_refresh_button = gr.Button("Refresh", variant='secondary')
            umr_analysis_refresh_button.click(
              fn=lambda : {'choices': refresh_choices(PROJ_DATA, C.CORPUS)['choices'] + refresh_choices(PROJ_EXP, C.CORPUS)['choices'][1:], '__type__': 'update'},
              outputs=umr_analysis_input,
              concurrency_limit=None,
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
            outputs=[umr_analysis, umr_stats],
            concurrency_limit=None,
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'}, {'visible': True, '__type__': 'update'}),
            outputs=[umr_analysis_search_var, umr_analysis_search_button],
            concurrency_limit=None,
          )
          umr_analysis_search_button.click(
            fn=analysis.search_var_umr,
            inputs=[umr_analysis_input, umr_analysis_search_var],
            outputs=umr_analysis,
            concurrency_limit=None,
          )

        with gr.Tab("AMR"):
          gr.Markdown("* ALIGNMENT ONLY, as of March 2024")
          # noinspection PyTypeChecker
          amr_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            info="for LEAMR, use the output parse/BLINK name, before extensions added by LEAMR",
            value=get_default_value_from_choices(INIT_EXP_CHOICES[D.AMR_TXT]),
            choices=INIT_EXP_CHOICES[D.AMR_TXT],
            allow_custom_value=True,
            interactive=True
          )
          amr_analysis_doc2snt_mapping = gr.Dropdown(
            label="Doc2Snt Mapping",
            info="if specified, IBM-style alignment will be assumed (otherwise, LEAMR)",
            value=get_default_value_from_choices(INIT_EXP_CHOICES[C.DOC2SNT_MAPPING]),
            choices=INIT_EXP_CHOICES[C.DOC2SNT_MAPPING],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            amr_analysis_load_button = gr.Button("Load", variant='primary')
            amr_analysis_refresh_button = gr.Button("Refresh", variant='secondary')
            amr_analysis_refresh_button.click(
              fn=lambda value: refresh_choices(PROJ_EXP, D.AMR_TXT, value),
              inputs=amr_analysis_input,
              outputs=amr_analysis_input,
              concurrency_limit=None,
            ).then(
              fn=lambda value: refresh_choices(PROJ_DATA, C.DOC2SNT_MAPPING, value),
              inputs=amr_analysis_doc2snt_mapping,
              outputs=amr_analysis_doc2snt_mapping,
              concurrency_limit=None,
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
            outputs=[amr_analysis, amr_stats],
            concurrency_limit=None,
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'},) * 3,
            outputs=[amr_search_id, amr_search_target, amr_search_button],
            concurrency_limit=None,
          )
          amr_search_button.click(
            fn=analysis.search_amr,
            inputs=[amr_analysis_input, amr_search_id, amr_search_target, amr_analysis_doc2snt_mapping],
            outputs=amr_analysis,
            concurrency_limit=None,
          )

        with gr.Tab("MDG"):
          # noinspection PyTypeChecker
          mdg_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            value=get_default_value_from_choices(INIT_EXP_CHOICES[C.MDP]),
            choices=INIT_EXP_CHOICES[C.MDP],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            mdg_analysis_load_button = gr.Button("LOAD", variant='primary')
            mdg_analysis_refresh_button = gr.Button("Refresh", variant="secondary")
            mdg_analysis_refresh_button.click(
              fn=lambda value: refresh_choices(PROJ_EXP, C.MDP, value),
              inputs=mdg_analysis_input,
              outputs=mdg_analysis_input,
              concurrency_limit=None,
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
            outputs=[mdg_analysis, mdg_stats],
            concurrency_limit=None,
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'},) * 3,
            outputs=[mdg_doc_id, mdg_anno, mdg_search_button],
            concurrency_limit=None,
          )
          mdg_search_button.click(
            fn=analysis.search_anno_mtdg,
            inputs=[mdg_analysis_input, mdg_doc_id, mdg_anno],
            outputs=mdg_analysis,
            concurrency_limit=None,
          )

        with gr.Tab("TDG"):
          # noinspection PyTypeChecker
          tdg_analysis_input = gr.Dropdown(
            label="Input (`input`)",
            value=get_default_value_from_choices(INIT_EXP_CHOICES[C.TDP]),
            choices=INIT_EXP_CHOICES[C.TDP],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            tdg_analysis_load_button = gr.Button("LOAD", variant='primary')
            tdg_analysis_refresh_button = gr.Button("Refresh", variant="secondary")
            tdg_analysis_refresh_button.click(
              fn=lambda value: refresh_choices(PROJ_EXP, C.TDP, value),
              inputs=tdg_analysis_input,
              outputs=tdg_analysis_input,
              concurrency_limit=None,
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
            outputs=[tdg_analysis, tdg_stats],
            concurrency_limit=None,
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'},) * 3,
            outputs=[tdg_doc_id, tdg_anno, tdg_search_button],
            concurrency_limit=None,
          )
          tdg_search_button.click(
            fn=analysis.search_anno_mtdg,
            inputs=[tdg_analysis_input, tdg_doc_id, tdg_anno],
            outputs=tdg_analysis,
            concurrency_limit=None,
          )

        with gr.Tab("CDLM"):
          cdlm_analysis_input = gr.Dropdown(
            label="Input",
            value=get_default_value_from_choices(INIT_EXP_CHOICES[C.CDLM]),
            choices=INIT_EXP_CHOICES[C.CDLM],
            allow_custom_value=True,
            interactive=True
          )
          with gr.Row():
            cdlm_analysis_load_button = gr.Button("LOAD", variant="primary")
            cdlm_analysis_refresh_button = gr.Button("Refresh", variant="secondary")
            cdlm_analysis_refresh_button.click(
              fn=lambda value: refresh_choices(PROJ_EXP, C.CDLM, value),
              inputs=cdlm_analysis_input,
              outputs=cdlm_analysis_input,
              concurrency_limit=None,
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
            outputs=[cdlm_analysis, cdlm_stats],
            concurrency_limit=None,
          ).then(
            fn=lambda: ({'visible': True, '__type__': 'update'}, {'visible': True, '__type__': 'update'}),
            outputs=[cdlm_analysis_cluster_id, cdlm_analysis_search_button],
            concurrency_limit=None,
          )
          cdlm_analysis_search_button.click(
            fn=analysis.search_cluster_cdlm,
            inputs=[cdlm_analysis_input, cdlm_analysis_cluster_id],
            outputs=cdlm_analysis,
            concurrency_limit=None,
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
          json_analysis_choices = INIT_DATA_CHOICES[C.JSON] + INIT_EXP_CHOICES[C.JSON][1:]
          json_analysis_input = gr.Dropdown(
            label="Input",
            info="does not accept .jsonlines",
            value=get_default_value_from_choices(json_analysis_choices),
            choices=json_analysis_choices,
            allow_custom_value=True,
            interactive=True,
          )

          with gr.Row():
            json_analysis_run_button = gr.Button("RUN", variant="primary")
            json_analysis_refresh_button = gr.Button("Refresh", variant='secondary')
            json_analysis_refresh_button.click(
              fn=lambda : {'choices': refresh_choices(PROJ_DATA, C.JSON)['choices'] + refresh_choices(PROJ_EXP, C.JSON)['choices'][1:], '__type__': 'update'},
              outputs=json_analysis_input,
              concurrency_limit=None,
            )
          json_analysis_info = gr.JSON(label='Info')
          json_analysis_run_button.click(
            fn=lambda x: io_utils.load_json(x),
            inputs=json_analysis_input,
            outputs=json_analysis_info,
            concurrency_limit=None,
          )

  # global event
  parsing_set_subroot_button.click(
    fn=lambda : ({'interactive': False, '__type__': 'update'},)*2,
    outputs=[parsing_subroot, parsing_set_subroot_button],
    concurrency_limit=None,
  ).then(
    fn=set_parsing_subroot_fn,
    inputs=parsing_subroot,
    outputs=[
      parsing_subroot_info,
      conv_output,
      ancast_umr_pred,
      sapienza_output,
      ibm_output,
      amrbart_output,
      blink_output,
      mbse_output,
      mdp_output,
      temporal_output,
      stage1_merge_output,
      thyme_output,
      cdlm_prep_output,
      cdlm_output,
      coref_output,
      conv_ud_output,
    ],
    concurrency_limit=None,
  )
  parsing_reset_subroot_button.click( # technically local
    fn=lambda : ({'interactive': True, '__type__': 'update'},)*2,
    outputs=[parsing_subroot, parsing_set_subroot_button],
    concurrency_limit=None,
  )

################################################################################
if __name__ == '__main__':
  misc_utils.init_logging(args.debug, suppress_penman=True, suppress_httpx=True)
  app.queue(max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    server_port=args.port,
    share=args.share,
    quiet=True,
    debug=args.debug
  )
