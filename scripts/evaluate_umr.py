#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/21/24 1:24 AM
import logging
import os
import subprocess
import sys

import numpy as np
from tqdm.contrib import tzip

from utils import consts as C, defaults as D
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


SENT = "Sent"
MODALITY = "Modality"
TEMPORAL = "Temporal"
COREF = "Coref"
COMPREHENSIVE = "Comprehensive"
MACRO_F1 = "Macro F1"
SCORES = 'Scores'

def run_eval(pred_fpath_list, gold_fpath_list, output_dir, eval_home):
  eval_home = os.path.abspath(eval_home)
  assert os.path.exists(eval_home)

  results = {
    SENT: [],
    MODALITY: [],
    TEMPORAL: [],
    COREF: [],
    COMPREHENSIVE: []
  }

  ### LET GOLD TAKE PRECEDENCE
  # for pred_fpath, gold_fpath in tzip(pred_fpath_list, gold_fpath_list):
  for gold_fpath, pred_fpath in tzip(gold_fpath_list, pred_fpath_list):
    fname = os.path.basename(pred_fpath)
    logger.info("### Evaluating `%s` ###", fname)

    cmd = [
      sys.executable,
      "src/run.py",
      os.path.abspath(pred_fpath),
      os.path.abspath(gold_fpath),
      '-o',
      os.path.join(os.path.abspath(output_dir), fname.replace('.txt', '.csv')),
      '--format',
      'umr',
    ]
    logger.info("CMD: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=eval_home)

    warnings = []
    score_lines = []

    # collect scores
    for line in res.stdout.split('\n'):
      line = line.strip()
      if line.startswith(SENT):
        key = SENT
      elif line.startswith(MODALITY):
        key = MODALITY
      elif line.startswith(TEMPORAL):
        key = TEMPORAL
      elif line.startswith(COREF):
        key = COREF
      elif line.startswith(COMPREHENSIVE):
        results[COMPREHENSIVE].append(float(line.split('\t')[-1][:-1]))
        score_lines.append(f"\t{line}")
        continue
      else:
        if len(line) > 0:
          warnings.append(f"\t{line}")
        continue

      score_lines.append(f"\t{line}")

      line_scores = line.split('\t')[-3:]
      p = float(line_scores[0].split()[-1][:-1])
      r = float(line_scores[1].split()[-1][:-1])
      f = float(line_scores[2].split()[-1][:-1])
      results[key].append( (p,r,f) )

    # display results
    if len(warnings) > 0:
      logger.info("Warnings:\n%s", '\n'.join(warnings))
    logger.info("Scores:\n%s", '\n'.join(score_lines))

  # final MACRO f1
  logger.info("### MACRO F1 ###")
  for k, v in results.items():
    fscores = v[0] if k == COMPREHENSIVE else v[-1]
    avg = np.mean(fscores)
    results[k] = {MACRO_F1: avg, SCORES: v}
    logger.info(" %s: %.2f", k, avg)

  return results

@timer
def main(args):
  # set up golds
  gold_fpath_or_dir = args.gold
  assert os.path.exists(gold_fpath_or_dir)

  gold_fnames = set()
  gold_fpath_list = list()
  if os.path.isdir(gold_fpath_or_dir):
    for gold_fname in os.listdir(gold_fpath_or_dir):
      if gold_fname.endswith(C.TXT):
        gold_fnames.add(gold_fname)
        gold_fpath_list.append(os.path.join(gold_fpath_or_dir, gold_fname))
  else:
    gold_fnames.add(os.path.basename(gold_fpath_or_dir))
    gold_fpath_list = [gold_fpath_or_dir]

  print(gold_fnames)

  # set up preds by only including filenames that match with gold
  pred_fpath_list = list()
  pred_fpath_or_dir = output_dir = args.pred
  assert os.path.exists(pred_fpath_or_dir) and os.path.isdir(output_dir)
  if os.path.isdir(pred_fpath_or_dir):
    for pred_fname in os.listdir(pred_fpath_or_dir):
      if pred_fname.endswith(C.TXT) and pred_fname in gold_fnames:
        pred_fpath_list.append(os.path.join(pred_fpath_or_dir, pred_fname))
  else:
    pred_fname = os.path.basename(pred_fpath_or_dir)
    if pred_fname in gold_fnames:
      pred_fpath_list = [pred_fpath_or_dir]

  log_fpath = os.path.join(output_dir, D.EVAL_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin UMR Evaluation ===")
  logger.info("Pred files: `%s`", "`, `".join(pred_fpath_list))
  logger.info("Gold files: `%s`", "`, `".join(gold_fpath_list))
  logger.info("Output dir: %s", output_dir)

  run_eval(pred_fpath_list, gold_fpath_list, output_dir, args.eval_home)

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument("-p", "--pred", required=True, help="path to UMR prediction dir or file")
    argparser.add_argument("-g", "--gold", required=True, help="path to UMR gold dir or file")
    argparser.add_argument("--eval_home", help="path to UMR Inference Toolkit home")
  main(script_setup(add_args_fn=add_args))
