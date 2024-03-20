#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/8/24 8:19 PM
"""merge stage1 outputs

  1) Conceivers are omitted
  2) Timex only comes from TDP Stage 1
  3) Event come from both MDP and TDP Stage 1
"""
import logging
import os

from tqdm.contrib import tzip

from utils import consts as C, defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def merge_stage1s(stage1s, output_fpath, include_conceiver=False, include_timex=False, for_thyme_tdg=False):
  # load
  data_list, snts_list, doc_ids_list = [], [], []
  for stage1 in stage1s:
    data, snts, doc_ids = io_utils.readin_mtdp_tuples(stage1, from_file=True)
    data_list.append(data)
    snts_list.append(snts)
    doc_ids_list.append(doc_ids)

  # sanity check
  canonical_snts = snts_list[0]
  canonical_doc_ids = doc_ids_list[0]
  for i, (snts, doc_ids) in enumerate(zip(snts_list[1:], doc_ids_list[1:]),1):
    assert len(snts_list[i-1]) == len(snts), f"{len(snts_list[i-1])} vs {len(snts)}"
    assert doc_ids_list[i-1] == doc_ids, f"{doc_ids_list[i-1]} vs {doc_ids}"

  merged_docs = []
  for i, data_tuples in enumerate(tzip(*data_list, desc=f'[Merging Stage1s]')):
    cur_doc_id = canonical_doc_ids[i]
    cur_snts = canonical_snts[i]

    doc_inputs = [f"filename:<doc id={cur_doc_id}>:SNT_LIST"] + cur_snts + ['EDGE_LIST']

    conceivers, timexs, events = [], [], []
    for data_list in data_tuples:
      for data in data_list:
        data_type = data[1]
        if for_thyme_tdg and len(data) == 2:
          data.extend(['-1_-1_-1', 'Depend-on'])
        data_str = '\t'.join(data)
        if data_type == C.TIMEX:
          timexs.append(data_str)
        elif data_type == C.CONCEIVER:
          conceivers.append(data_str)
        elif data_str not in events:
          events.append(data_str)

    out_data = []
    if include_conceiver:
      out_data += conceivers
    if include_timex:
      out_data += timexs
    out_data += events

    # sort
    out_data.sort(key=lambda x: tuple(map(int, x.split('\t')[0].split('_'))))

    out_data = doc_inputs + out_data
    merged_docs.append('\n'.join(out_data))

  # export
  logger.info("Exporting Merged Stage1 at %s", output_fpath)
  io_utils.save_txt(merged_docs, output_fpath, delimiter='\n\n')

@timer
def main(args):
  # no `input`, but rather `stage1`, a list of input files
  stage1s = args.stage1s
  assert len(stage1s) > 1, "at least 2 stage1s necessary for merging"
  for stage1 in stage1s:
    assert os.path.exists(stage1)

  ### output should be a dir
  output_fpath = args.output
  output_dir, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_dir, f'{C.MERGED}.{C.STAGE1_TXT}')
  log_fpath = os.path.join(output_dir, D.MERGE_LOG)
  add_log2file(log_fpath)

  if args.for_thyme_tdg and not args.include_timex:
    logger.info("although not specified, `Timex` will still be included to prepare for thyme_tdg")
    args.include_timex = True

  logger.info("=== Merging Stage 1s ===")
  logger.info("Stage 1s: `%s`", "`, `".join(stage1s))
  logger.info("Output: %s", output_fpath)
  logger.info("Include Conceiver: %s", args.include_conceiver)
  logger.info("Include Timex: %s", args.include_timex)
  logger.info("For `thyme_tdg`: %s", args.for_thyme_tdg)

  merge_stage1s(
    stage1s=stage1s,
    output_fpath=output_fpath,
    include_conceiver=args.include_conceiver,
    include_timex=args.include_timex,
    for_thyme_tdg=args.for_thyme_tdg
  )

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument('--stage1s', nargs='+', required=True, help="list of stage1s to merge")
    argparser.add_argument('--include_conceiver', action='store_true', help='whether to include Conceivers')
    argparser.add_argument('--include_timex', action='store_true', help='whether to include Timex')
    argparser.add_argument('--for_thyme_tdg', action='store_true', help='whether the target input is being prepared for thyem_tdg')
  main(script_setup(add_args_fn=add_args))
