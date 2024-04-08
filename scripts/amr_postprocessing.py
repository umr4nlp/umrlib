#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/24/24 3:45 AM
"""prepare for leamr + mbse

only keep `id` + `snt` + `tok` for meta, drop everything else
"""
import logging
import os
from typing import Dict

import tqdm

from utils import defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def postprocessing(input_fpath, output_fpath, doc2snt_mapping=None, snts_fpath=None):
  blocks = io_utils.load_txt(input_fpath, delimiter='\n\n')
  num_data = len(blocks)
  logger.info("Found %d data", num_data)

  sample = blocks[0]
  has_tok = '::tok' in sample
  has_snt = '::snt' in sample

  has_doc2snt_mapping = doc2snt_mapping is not None and len(doc2snt_mapping) > 0
  has_snts = snts_fpath is not None and len(snts_fpath) > 0

  # noinspection PyTypeChecker
  snt2doc_mapping = None
  if has_doc2snt_mapping:
    doc2snt_mapping = io_utils.load_json(doc2snt_mapping) # type: Dict[str, int]
    snt2doc_mapping = {v:k for k,v in doc2snt_mapping.items()} # type: Dict[int, str]
    assert len(snt2doc_mapping) == num_data

  snts = None
  if has_snts:
    snts = io_utils.load_txt(snts_fpath, delimiter='\n')
    assert len(snts) == num_data

  out = []
  for i, block in tqdm.tqdm(enumerate(blocks), desc='[AMR-POST]', total=num_data):
    lines = block.split('\n')

    # for output
    new_lines = []

    first_line = lines[0]
    if first_line.startswith("# ::id"):
      amr_id = first_line.split()[-1].strip()
      if amr_id not in snt2doc_mapping:
        assert amr_id.isdigit()
        first_line = f"# ::id {snt2doc_mapping[i]}"
      new_lines.append(first_line)
      lines = lines[1:]
    else:
      new_lines.append(f"# ::id {snt2doc_mapping[i]}")

    for line in lines:
      if line.startswith("#"):
        if line.startswith('# ::snt'):
          if has_snts:
            line = f"# ::snt {snts[i]}"
          new_lines.append(line)
          if not has_tok:
            new_lines.append(line.replace("::snt", "::tok"))
        elif line.startswith('# ::tok'):
          if has_snts:
            line = f"# ::tok {snts[i]}"
          new_lines.append(line)
          if not has_snt:
            new_lines.append(line.replace("::tok", "::snt"))
        # elif line.startswith('# ::node'): # keep ibm-style alignment
        #   new_lines.append(line)
      else:
        new_lines.append(line)
    out.append('\n'.join(new_lines))

  # mbse.py requires some new lines at the end to recognize the last amr entry
  out[-1] += '\n\n'

  io_utils.save_txt(out, output_fpath, delimiter='\n\n')

@timer
def main(args):
  input_fpath = args.input
  assert os.path.isfile(input_fpath)

  output_fpath = args.output
  dirname, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, get_is_dir_flag=True)
  assert not is_dir, "`output` should be a file, but a dir was given: `%s`" % output_fpath
  log_fpath = os.path.join(dirname, D.AMR_POST_LOG)
  add_log2file(log_fpath)

  doc2snt, snts = args.doc2snt, args.snts

  logger.info("=== Begin AMR Postprocessing ===")
  logger.info("Input: `%s`", args.input)
  logger.info("Output: `%s`", output_fpath)
  logger.info("Doc2Snt Mapping: %s", doc2snt)
  logger.info("Sentences: %s", snts)

  postprocessing(input_fpath, output_fpath, doc2snt, snts)

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument('--doc2snt', help='if set, will update `::id`')
    argparser.add_argument('--snts', help="if set, will update `::snt` and `::tok`")
  main(script_setup(add_args_fn=add_args))
