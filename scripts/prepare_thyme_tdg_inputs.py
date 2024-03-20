#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/11/24 3:56 AM
"""porting from temporal baseline to thyme tdg

NOTE:
* abstract DCT ref node (-7_-7_-7) vs concrete DCT (typically 0_0_3), the first sentence
  * format: `month day , year` so (0_0_3)
  * but this doens't apply to Stage 1 output though
"""
import logging
import os

import tqdm

from utils import consts as C, defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def prepare_thyme_tdg_inputs(input_fpath, output_fpath):
  # stage 1 only has child_offset + mention_type (Timex vs Event)
  # so need to add any parent_offset (root: -1_-1_-1) + temporal label (Depend-on)
  # these are just placeholders, not actual predictions, to get by thyme_tdg's data loading
  texts = io_utils.load_txt(input_fpath, delimiter='\n\n\n')

  out = []
  for text in tqdm.tqdm(texts, desc=f'[Prepare Thyme-TDG Inputs]'):
    text_out = []
    edge_list_flag = False
    for line in text.split('\n'):
      if line == 'EDGE_LIST':
        edge_list_flag = True
      elif edge_list_flag:
        line_s = line.split('\t')
        if len(line_s) != 2:
          continue
        line_s.extend(['-1_-1_-1', 'Depend-on'])
        line = '\t'.join(line_s)
      text_out.append(line)
    out.append('\n'.join(text_out))
  out[-1] += '\n\n\n'

  io_utils.save_txt(out, output_fpath, delimiter='\n\n\n')

@timer
def main(args):
  # should be a file
  input_fpath = args.input
  assert os.path.exists(input_fpath), f"?? {input_fpath}"

  output_fpath = args.output
  output_dir, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_dir, f'{C.THYME_TDG}.{C.STAGE1_TXT}')
  log_fpath = os.path.join(output_dir, D.PREP_THYME_TDG_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin Preparing Thyme-TDG Inputs ===")
  logger.info("Input: %s", input_fpath)
  logger.info("Output: %s", output_fpath)

  prepare_thyme_tdg_inputs(input_fpath, output_fpath)

  logger.info("Done.")


if __name__ == '__main__':
  main(script_setup())
