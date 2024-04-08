#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 6/12/23 5:21 PM
import logging
import os

from evaluate import compute_scores
from utils import defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


@timer
def main(args):
  pred_fpath, gold_fpath = args.pred, args.gold
  assert os.path.isfile(pred_fpath)
  assert os.path.isfile(gold_fpath)

  output_dir = io_utils.get_dirname(pred_fpath)
  log_fpath = os.path.join(output_dir, D.SMATCH_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin Smatch Evaluation ===")
  logger.info("Pred: `%s`", pred_fpath)
  logger.info("Gold: `%s`", gold_fpath)

  compute_scores(pred_fpath, gold_fpath)

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument("-p", "--pred", required=True, help="path to UMR prediction dir or file")
    argparser.add_argument("-g", "--gold", required=True, help="path to UMR gold dir or file")
  main(script_setup(add_args_fn=add_args))
