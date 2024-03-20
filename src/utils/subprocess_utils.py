#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/17/24 11:28 PM
import logging
import time
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)


# executables
BASH = 'bash'
PYTHON = sys.executable

def if_done(done, p):
  while 1:
    if p.poll() is None:
      time.sleep(0.5)
    else:
      break
  done[0] = True

def run_cmd(cmd, log_fpath, env=None, cwd=None):
  exec_cmd = cmd.split()[0]
  if 'python' in exec_cmd:
    if env is None:
      env = {'PYTHONPATH': "src"}
    elif 'PYTHONPATH' not in env:
      env['PYTHONPATH'] = 'src'

  with open(log_fpath, 'w'): # create empty log file
    pass

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

def run_cmd_wait(cmd, env=None, cwd=None):
  exec_cmd = cmd.split()[0]
  if 'python' in exec_cmd:
    if env is None:
      env = {'PYTHONPATH': "src"}
    elif 'PYTHONPATH' not in env:
      env['PYTHONPATH'] = 'src'

  out = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
  return out
