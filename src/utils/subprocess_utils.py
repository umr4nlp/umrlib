#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/17/24 11:28 PM
import logging
import os
import time
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)


# executables
BASH = 'bash'
PYTHON = sys.executable

# envs
PATH = 'PATH'
LD_LIB_PATH = 'LD_LIBRARY_PATH'
CVD = 'CUDA_VISIBLE_DEVICES'
CUDA = 'cuda'

def if_done(done, p):
  while 1:
    if p.poll() is None:
      time.sleep(0.5)
    else:
      break
  done[0] = True

def prepare_env(cuda=None, cvd=None, pysrc=False):
  env = os.environ.copy()
  if pysrc:
    env['PYTHONPATH'] = 'src'

  if cuda:
    if isinstance(cuda, str):
      try:
        float(cuda)
        cuda = f'{CUDA}-{cuda}'
      except ValueError:
        pass
    elif isinstance(cuda, float):
      cuda = f'{CUDA}-{cuda}'

    cuda = cuda.lower()
    assert cuda.startswith(CUDA)
    env[PATH] = f'/usr/local/{cuda}/bin:{os.getenv(PATH)}'
    env[LD_LIB_PATH] = f'/usr/local/{cuda}/lib64:{os.getenv(LD_LIB_PATH)}'

  if cvd:
    if isinstance(cvd, int):
      cvd = str(cvd)
    if cvd.startswith(CVD):
      cvd = cvd.split('=')[-1]
    env[CVD] = cvd

  return env

def run_cmd_webui(cmd, log_fpath, cwd=None, cuda=None, cvd=None, pipe_to_log=False):
  logger.info("Logging at `%s`", log_fpath)
  with open(log_fpath, 'w'): # create empty log file
    pass

  if isinstance(cmd, list):
    pysrc = 'python' in cmd[0]
    cmd = " ".join(cmd)
  else:
    pysrc = 'python' in cmd.split()[0]
  assert isinstance(cmd, str)

  if pipe_to_log:
    cmd = f"{cmd} > {log_fpath} 2>&1"
  logger.info("CMD: %s", cmd)

  env = prepare_env(cuda=cuda, cvd=cvd, pysrc=pysrc)

  done = [False]
  p = subprocess.Popen(cmd, cwd=cwd, shell=True, env=env)
  threading.Thread(target=if_done, args=(done, p)).start()

  while True:
    with open(log_fpath, "r") as f:
      yield f.read()
    time.sleep(1)
    if done[0]:
      break

  with open(log_fpath, "r") as f:
    log = f.read()
  yield log

def run_cmd(cmd, env=None, cwd=None):
  exec_cmd = cmd.split()[0]
  if 'python' in exec_cmd:
    if env is None:
      env = {'PYTHONPATH': "src"}
    elif 'PYTHONPATH' not in env:
      env['PYTHONPATH'] = 'src'

  out = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
  return out
