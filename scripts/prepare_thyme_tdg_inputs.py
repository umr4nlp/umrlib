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
from tqdm.contrib import tzip

from utils import consts as C, defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def map_dct(line, as_str=False):
  if isinstance(line, str):
    line = "\t".split(line)
  assert isinstance(line, list)

  if line[0].startswith('-7'):
    line[0] = '0_0_3'
  if len(line) == 4 and line[2].startswith('-7'):
    line[2] = '0_0_3'

  if as_str:
    return '\t'.join(line)
  return line

def prepare_thyme_tdg_inputs(input_fpath, output_fpath):
  # stage 1 only has child_offset + mention_type (Timex vs Event)
  # so need to add any parent_offset (root: -1_-1_-1) + temporal label (Depend-on)
  # these are just placeholders, not actual predictions, to get by thyme_tdg's data loading
  # also, map -7s to DCT (0_0_3)
  texts = io_utils.load_txt(input_fpath, delimiter='\n\n\n')

  out = []
  for text in tqdm.tqdm(texts, desc=f'[Prepare Thyme-TDG Inputs]'):
    text_out = []
    edge_list_flag = False
    for line in text.split('\n'):
      if line == 'EDGE_LIST':
        edge_list_flag = True
      elif edge_list_flag:
        line_ = line.strip()
        if not line_:
          continue
        line_s = line.strip().split('\t')
        if line_s[0].startswith('-7'):
          line_s[0] = '0_0_3'

        num_split = len(line_s)
        if num_split == 4:
          cidx, label_type, pidx, label = line_s
          if pidx.startswith('-7'):
            line_s[2] = '0_0_3'
        elif num_split == 2:
          line_s.extend(['-1_-1_-1', 'Depend-on'])
        line = '\t'.join(line_s)
        if line in text_out:
          continue
      text_out.append(line)
    out.append('\n'.join(text_out))
  out[-1] += '\n\n\n'

  io_utils.save_txt(out, output_fpath, delimiter='\n\n\n')

def merge_temporal_stage2s(temporal_tt_fpath, temporal_te_fpath, output_fpath):
  temporal_tt = io_utils.readin_mtdp_tuples(temporal_tt_fpath, from_file=True)
  temporal_te = io_utils.readin_mtdp_tuples(temporal_te_fpath, from_file=True)

  snts_list, doc_ids = temporal_tt[1], temporal_tt[2]
  assert snts_list == temporal_te[1], f"{len(snts_list)} snts list vs {len(temporal_te[1])} temporal_te[1]"
  assert doc_ids == temporal_te[2]

  out = []
  for snts, doc_id, tt_edges, te_edges in tzip(
          snts_list, doc_ids, temporal_tt[0], temporal_te[0]):
    cur_doc = [f"filename:<doc id={doc_id}>:SNT_LIST"] + snts + ['EDGE_LIST']

    edges = []

    # 1) temporal_time
    for edge in tt_edges:
      edge = map_dct(edge, as_str=True)
      if edge not in edges:
        edges.append(edge)

    # 2) temporal_event
    for edge in te_edges:
      edge = map_dct(edge)
      edge_str = "\t".join(edge)
      if edge[1] == 'Timex':
        if not edge[2].startswith('-'):
          logger.warning("Skipping TE Timex edge: %s", edge_str)
        continue
      if edge[2].startswith('-1'):
        logger.warning("Skipping TE Root edge: %s", edge_str)
        continue
      edges.append(edge_str)
    cur_doc.extend(edges)

    out.append("\n".join(cur_doc))

  # export
  logger.info("Exporting Merged Temporal Stage2s at %s", output_fpath)
  io_utils.save_txt(out, output_fpath, delimiter='\n\n')

@timer
def main(args):
  output_fpath = args.output
  output_dir, is_dir = io_utils.get_dirname(output_fpath, mkdir=True, get_is_dir_flag=True)
  log_fpath = os.path.join(output_dir, D.PREP_THYME_TDG_LOG)
  add_log2file(log_fpath)

  temporal_tt, temporal_te = args.temporal_tt, args.temporal_te
  if temporal_tt and temporal_te:
    if is_dir:
      canonicals = [io_utils.get_canonical_fname(temporal_tt, depth=2),
                    io_utils.get_canonical_fname(temporal_te, depth=2)]
      output_fpath = io_utils.get_unique_fpath(output_dir, f'{":".join(canonicals)}.{C.MERGED}.{C.STAGE2_TXT}')

    assert os.path.exists(temporal_tt)
    assert os.path.exists(temporal_te)

    logger.info("=== Begin Preparing Thyme-TDG Inputs ===")
    logger.info("Temporal-Time Stage 2: %s", temporal_tt)
    logger.info("Temporal-Event Stage 2: %s", temporal_te)
    logger.info("Output: %s", output_fpath)

    merge_temporal_stage2s(temporal_tt, temporal_te, output_fpath)
  else:
    if is_dir:
      output_fpath = io_utils.get_unique_fpath(output_dir, f'{C.THYME_TDG}.{C.STAGE1_TXT}')

    # should be a file
    input_fpath = args.input
    assert os.path.exists(input_fpath)

    logger.info("=== Begin Preparing Thyme-TDG Inputs ===")
    logger.info("Input: %s", input_fpath)
    logger.info("Output: %s", output_fpath)

    prepare_thyme_tdg_inputs(input_fpath, output_fpath)

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument('-tt', '--temporal_tt', help="temporal baseline `temporal_time` output")
    argparser.add_argument('-te', '--temporal_te', help="temporal baseline `temporal_event` output")
  main(script_setup(add_args_fn=add_args))
