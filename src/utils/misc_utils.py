#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/5/24 5:17 PM
import argparse
import logging
import os
import random
import time
from datetime import datetime

import numpy as np

from utils.config import Config

logger = logging.getLogger(__name__)


def init_argparser(add_args_fn=None):
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-i', '--input', nargs="?", help='input file or dir')
  argparser.add_argument('-o', '--output', help='output file or dir')
  argparser.add_argument('--seed', type=int, default=42, help='random seed')
  argparser.add_argument('--overwrite', action='store_true', help='overwrite existing files')
  argparser.add_argument('--debug', action='store_true', help='whether to log debug messages')
  if add_args_fn is not None:
    add_args_fn(argparser)
  return argparser

def display_config(config):
  logger.info("***** CONFIG *****")
  for k, v in vars(config).items():
    logger.info(" %s: %s", k, v)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)

def add_log2file(log_fpath, overwrite=False):
  if overwrite and os.path.exists(log_fpath):
    os.remove(log_fpath)
  sample_handler = logging.root.handlers[0]
  log2file = logging.FileHandler(log_fpath)
  log2file.setLevel(sample_handler.level)
  log2file.setFormatter(sample_handler.formatter)
  logging.root.addHandler(log2file)
  logger.info("Logging at %s", log_fpath)

def init_logging(debug=False, log_fpath=None, suppress_penman=False, suppress_httpx=False, overwrite=False):
  for handler in logging.root.handlers:
    logging.root.removeHandler(handler)
  logging.root.handlers = []

  level = logging.DEBUG if debug else logging.INFO
  logging.root.setLevel(level)
  formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]%(message)s")

  # init console logging
  log2console = logging.StreamHandler()
  log2console.setLevel(level)
  log2console.setFormatter(formatter)
  logging.root.addHandler(log2console)

  if log_fpath is not None: # init file logging
    add_log2file(log_fpath, overwrite=overwrite)

  if suppress_penman:
    logging.getLogger('penman').disabled = True
    logging.getLogger('penman.layout').disabled = True
    logging.getLogger('penman._lexer').disabled = True

  if suppress_httpx:
    logging.getLogger('httpx').disabled = True

def script_setup(add_args_fn=None, log_fpath=None, skip_display=True, suppress_penman_logging=True, suppress_httpx_logging=True):
  argparser = init_argparser(add_args_fn)
  args, others = argparser.parse_known_args()

  config = Config(vars(args))

  init_logging(
    config.debug,
    log_fpath=log_fpath,
    suppress_penman=suppress_penman_logging,
    suppress_httpx=suppress_httpx_logging,
    overwrite=config.overwrite
  )

  set_seed(config.seed)
  if not skip_display:
    display_config(config)

  return config

def timer(func):
  def inner(*args, **kwargs):
    begin = time.time()
    print("`%s` started on %s" % (func.__name__, datetime.now().strftime('%c')))
    func(*args, **kwargs)
    print("`%s` finished on %s" % (func.__name__, datetime.now().strftime('%c')), end=' ')

    # count total duration
    exec_time = time.time() - begin
    if exec_time > 60:
      et_m, et_s = int(exec_time / 60), int(exec_time % 60)
      print("| Execution Time: %dm %ds" % (et_m, et_s))
    else:
      print("| Execution Time: %.2fs" % exec_time)
  return inner
