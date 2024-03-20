#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/14/24 2:13 AM
"""need to run this after running event detection model

use `cdlm` virtulalenv
"""
import logging
import os

import spacy
from spacy.tokens import Doc
from tqdm.contrib import tzip

from utils import defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def prepare_cdlm_inputs(input_fpath, output_dir, split_mapping_fpath=None):
  # output data
  all_docs, all_events = dict(), []

  # input modal data (not really modal, but rather events)
  modal_data = io_utils.readin_mtdp_tuples(input_fpath, from_file=True)

  # split_mapping = None
  # has_split_mapping = split_mapping_fpath is not None and os.path.exists(split_mapping_fpath)
  # if has_split_mapping:
  #   split_mapping = io_utils.load_json(split_mapping_fpath)[C.DOC]

  # cdlm preprocessing (adapted from `get_ecb_data.py`)
  nlp = spacy.load('en_core_web_sm', disable=['textcat'])

  for modals, modal_snts, modal_doc_id in tzip(*modal_data, desc='[Prepare CDLM inputs]'):
    cur_doc_id = f'english_umr-000{modal_doc_id}'
    # if has_split_mapping:
    #   cur_doc_id = split_mapping[cur_doc_id]

    # modal_snts.pop(0) # pop artifiail date
    toks_list = []

    data_dict = dict()

    # 1. doc (snt_idx starts at 0; tok_idx starts at 1)
    tok_idx = 1
    for snt_idx, modal_snt in enumerate(modal_snts): # already tokenized
      modal_snt_split = modal_snt.split()
      tok_idx_list = []
      for j, tok in enumerate(modal_snt_split, 1):
        # cur_doc.append([0, j, tok, True]) # singleton flag probably doesn't matter
        # toks_list.append([snt_idx, j, tok, True]) # singleton flag probably doesn't matter
        toks_list.append([snt_idx, tok_idx, tok, True]) # singleton flag probably doesn't matter
        tok_idx_list.append(tok_idx)
        tok_idx += 1

      # cur_doc_id_full = cur_doc_id_template % snt_idx
      data_dict[snt_idx] = modal_snt_split, tok_idx_list, cur_doc_id

    all_docs[cur_doc_id] = toks_list

    # 2. mentions
    # for moffsets, mtype in modals:
    for modal_data in modals:
      if len(modal_data) == 4:
        moffsets, mtype, _, _ = modal_data
      else:
        moffsets, mtype = modal_data

      snt_idx, start_offset, end_offset = [int(ch) for ch in moffsets.split('_')]
      # snt_idx_adj = snt_idx - 1
      snt_idx_adj = snt_idx

      tokens_list, token_ids_list, cur_doc_id = data_dict[snt_idx_adj]

      tokens = tokens_list[start_offset:end_offset+1]
      token_ids = token_ids_list[start_offset:end_offset+1]

      lemmas, tags = [], []
      for tok in nlp(Doc(nlp.vocab, tokens)):
        lemmas.append(tok.lemma_)
        tags.append(tok.tag_)

      cur_mention = {
          'doc_id': cur_doc_id,
          'subtopic': cur_doc_id,
          'm_id': "",
          "sentence_id": snt_idx_adj,
          "tokens_ids": token_ids,
          "tokens": ' '.join(tokens),
          "tags": ' '.join(tags),
          "lemmas": ' '.join(lemmas),
          'cluster_id': 0,
          'topic': cur_doc_id,
          'singleton': "",
        }

      if mtype == 'Event':
        all_events.append(cur_mention)

  logger.info("Found %d events", len(all_events))

  logger.info("Exporting at `%s`", output_dir)
  io_utils.save_json(all_docs, output_dir, D.CDLM_JSON)
  io_utils.save_json(all_events, output_dir, D.CDLM_EVENTS_JSON)

@timer
def main(args):
  input_fpath = args.input
  assert os.path.exists(input_fpath), f"?? {input_fpath}"

  assert args.output
  output_dir = io_utils.get_dirname(args.output, mkdir=True)
  # if output_dir is None:
  #   output_dir = os.path.dirname(input_fpath)
  # os.makedirs(output_dir, exist_ok=True)
  log_fpath = os.path.join(output_dir, D.PREP_CDLM_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin Preparing CDLM Inputs ===")
  logger.info("Input: %s", input_fpath)
  logger.info("Output: %s", output_dir)
  logger.info("Split Mapping: %s", str(args.split_mapping))

  prepare_cdlm_inputs(input_fpath, output_dir, args.split_mapping)

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument('--split_mapping', help='fpath to optional split mapping file')
  main(script_setup(add_args_fn=add_args))
