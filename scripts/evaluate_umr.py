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

def run_ancast(pred_fpath_list, gold_fpath_list, output_dir, ancast_home):
  ancast_home = os.path.abspath(ancast_home)
  assert os.path.exists(ancast_home)

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
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ancast_home)

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
        comp_score = float(line.split('\t')[-1][:-1])
        results[COMPREHENSIVE].append(comp_score)
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
    if k == COMPREHENSIVE:
      fscores = v
    else:
      fscores = [x[-1] for x in v]
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
    for gold_fname in sorted(os.listdir(gold_fpath_or_dir)):
      if gold_fname.endswith(C.TXT):
        gold_fnames.add(gold_fname)
        gold_fpath_list.append(os.path.join(gold_fpath_or_dir, gold_fname))
  else:
    gold_fnames.add(os.path.basename(gold_fpath_or_dir))
    gold_fpath_list = [gold_fpath_or_dir]

  # set up preds by only including filenames that match with gold
  pred_fpath_list = list()
  pred_fpath_or_dir = output_dir = args.pred
  assert os.path.exists(pred_fpath_or_dir) and os.path.isdir(output_dir)
  if os.path.isdir(pred_fpath_or_dir):
    for pred_fname in sorted(os.listdir(pred_fpath_or_dir)):
      if pred_fname.endswith(C.TXT) and pred_fname in gold_fnames:
        pred_fpath_list.append(os.path.join(pred_fpath_or_dir, pred_fname))
  else:
    pred_fname = os.path.basename(pred_fpath_or_dir)
    if pred_fname in gold_fnames:
      pred_fpath_list = [pred_fpath_or_dir]

  log_fpath = os.path.join(output_dir, D.ANCAST_UMR_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin UMR Evaluation ===")
  logger.info("Pred files: `%s`", "`, `".join(pred_fpath_list))
  logger.info("Gold files: `%s`", "`, `".join(gold_fpath_list))
  logger.info("Output dir: %s", output_dir)

  run_ancast(pred_fpath_list, gold_fpath_list, output_dir, args.ancast_home)

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument("-p", "--pred", required=True, help="path to UMR prediction dir or file")
    argparser.add_argument("-g", "--gold", required=True, help="path to UMR gold dir or file")
    argparser.add_argument("--ancast_home", help="path to UMR Inference Toolkit home")
  main(script_setup(add_args_fn=add_args))

###  MODAL BASELINE
# [2024-03-27 03:32:06,280][__main__][INFO]### MACRO F1 ###
# [2024-03-27 03:32:06,280][__main__][INFO] Sent: 66.71
# [2024-03-27 03:32:06,280][__main__][INFO] Modality: 44.75
# [2024-03-27 03:32:06,280][__main__][INFO] Temporal: 0.00
# [2024-03-27 03:32:06,280][__main__][INFO] Coref: 0.00
# [2024-03-27 03:32:06,280][__main__][INFO] Comprehensive: 57.04
# [2024-03-27 03:32:06,280][__main__][INFO]Done.

### MDP PROMPT 80
# [2024-03-27 03:32:40,273][__main__][INFO]### MACRO F1 ###
# [2024-03-27 03:32:40,273][__main__][INFO] Sent: 66.71
# [2024-03-27 03:32:40,273][__main__][INFO] Modality: 46.09
# [2024-03-27 03:32:40,273][__main__][INFO] Temporal: 0.00
# [2024-03-27 03:32:40,273][__main__][INFO] Coref: 0.00
# [2024-03-27 03:32:40,274][__main__][INFO] Comprehensive: 57.18
# [2024-03-27 03:32:40,274][__main__][INFO]Done.

### MDP PROMPT 30
# [2024-03-27 03:35:36,246][__main__][INFO]### MACRO F1 ###
# [2024-03-27 03:35:36,247][__main__][INFO] Sent: 66.71
# [2024-03-27 03:35:36,247][__main__][INFO] Modality: 46.23
# [2024-03-27 03:35:36,247][__main__][INFO] Temporal: 0.00
# [2024-03-27 03:35:36,247][__main__][INFO] Coref: 0.00
# [2024-03-27 03:35:36,248][__main__][INFO] Comprehensive: 57.20
# [2024-03-27 03:35:36,248][__main__][INFO]Done.

# [2024-03-27 04:33:48,322][__main__][INFO]### MACRO F1 ###
# [2024-03-27 04:33:48,322][__main__][INFO] Sent: 66.71
# [2024-03-27 04:33:48,322][__main__][INFO] Modality: 44.82
# [2024-03-27 04:33:48,322][__main__][INFO] Temporal: 0.00
# [2024-03-27 04:33:48,323][__main__][INFO] Coref: 17.93
# [2024-03-27 04:33:48,323][__main__][INFO] Comprehensive: 56.99
# [2024-03-27 04:33:48,323][__main__][INFO]Done.


### CDLM 80 0.5
# [2024-03-27 04:35:30,238][__main__][INFO]### MACRO F1 ###
# [2024-03-27 04:35:30,238][__main__][INFO] Sent: 66.71
# [2024-03-27 04:35:30,238][__main__][INFO] Modality: 46.27
# [2024-03-27 04:35:30,238][__main__][INFO] Temporal: 0.00
# [2024-03-27 04:35:30,238][__main__][INFO] Coref: 17.94
# [2024-03-27 04:35:30,238][__main__][INFO] Comprehensive: 57.14
# [2024-03-27 04:35:30,238][__main__][INFO]Done.
