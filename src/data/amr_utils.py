#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/6/24 3:23 PM
"""utils for processing AMR corpora"""
import logging
import os
from collections import defaultdict

import penman

from utils import consts as C, io_utils, regex_utils

logger = logging.getLogger(__name__)


def clean_text(text):
  """https://en.wikipedia.org/wiki/Windows-1252#Codepage_layout"""
  text = text.replace('\r', '')
  text = text.replace('\n\n\n', '\n\n')
  text = text.replace(u'\x80', "")
  text = text.replace(u'\x85', "...")
  text = text.replace(u'\x91', "'") # left quote
  text = text.replace(u'\x92', "'") # right quote
  text = text.replace(u'\x93', '"') # left double-quote
  text = text.replace(u'\x94', '"') # right double-quote
  text = text.replace(u'\x95', "·")
  text = text.replace(u'\x96', "-")
  text = text.replace(u'\x97', "-")
  text = text.replace(u'\x99', "")  # tm mark
  text = text.replace(u'\x9c', "")
  text = text.replace(u'\x9d', "")
  text = text.replace(u'\xa0', " ")
  return text

def split_meta_from_graphs(dataset, node_as_alignment=False):
  metas, graphs = [], []
  for data in dataset:
    alignment = defaultdict(tuple) # str -> (int,int)
    meta, graph = dict(), list()
    for line in data.split('\n'):
      if line.startswith('#'):
        if line.startswith('# ::preferred'):
          continue
        for k,v in regex_utils.parse_amr_meta(line):
          if k == 'node' and node_as_alignment:  # alignment from IBM parser
            try:
              _, alignment_var_or_gorn, alignment_label, alignment_span = v.split('\t')
            except ValueError:
              alignment_var_or_gorn, alignment_label, alignment_span = v.split('\t')
            # could be a gorn address, but in this case we most likely are using leamr whose output is stored in a separate file
            # alignment[alignment_var_or_gorn] = tuple(int(x) for x in alignment_span.split('-'))
            alignment[f'{alignment_var_or_gorn}_{alignment_label}'] = tuple(int(x) for x in alignment_span.split('-'))
          else:
            meta[k] = v
      else:
        graph.append(line)
    if node_as_alignment and len(alignment) > 0:
      meta[C.ALIGNMENT] = alignment
    metas.append(meta)
    graphs.append('\n'.join(graph))

  return metas, graphs

def load_amr_file_ibm(fpath_or_dir, fname=None, clean=False, doc2snt_mapping=None):
  fpath = io_utils.get_fpath(fpath_or_dir, fname)
  dataset = io_utils.load_txt(fpath)
  if clean:
    dataset = clean_text(dataset)
  dataset = list(filter(None, dataset.split('\n\n')))

  # drop `# AMR release;` first line
  if dataset[0].startswith('# AMR'):
    dataset.pop(0)

  # always split, b/c of need to check for duplicates
  metas, graphs = split_meta_from_graphs(dataset, node_as_alignment=True)

  if doc2snt_mapping is not None:
    logger.debug("Using Doc2Snt when loading IBM's AMR output file")
    if isinstance(doc2snt_mapping, str):
      doc2snt_mapping = io_utils.load_json(doc2snt_mapping)
    assert isinstance(doc2snt_mapping, dict)
    snt2doc_mapping = {v: k for k, v in doc2snt_mapping.items()}

    assert len(metas) == len(snt2doc_mapping)

    for i, meta in enumerate(metas):
      meta['id'] = snt2doc_mapping[i]
  else:
    logger.warning("[!] Loaded IBM's AMR output file without Doc2Snt mapping, without unique ID each graph may not be identifiable")

  logger.info("Loaded %d AMR data from file at `%s`", len(metas), fpath)
  return metas, graphs

def load_amr_file(fpath_or_dir, fname=None, clean=False, load_leamr=False, doc2snt_mapping=None):
  """single file containing AMR annotations

  for IBM, use another function, i.e. `node_as_alignment` is FALSE
  """
  fpath = io_utils.get_fpath(fpath_or_dir, fname)
  dataset = io_utils.load_txt(fpath)
  if clean:
    dataset = clean_text(dataset)
  dataset = list(filter(None, dataset.split('\n\n')))

  # drop `# AMR release;` first line
  if dataset[0].startswith('# AMR'):
    dataset.pop(0)

  # the AMR may be from IBM or LeakDistill, and here we load them differently depending on how they have been aligned
  # i.e., IBM natively produces alignment but provides no metadata except `::tok` so Doc2Snt mapping must be provied,
  # whereas LeakDistill (and SPRING) relies on LEAMR which produces additional files
  # BUT obviously they could both be False too, such as when reading AMR corpus
  is_ibm_aligner = doc2snt_mapping is not None

  metas, graphs = split_meta_from_graphs(dataset, node_as_alignment=is_ibm_aligner)
  if is_ibm_aligner:
    assert not load_leamr
    if isinstance(doc2snt_mapping, str):
      doc2snt_mapping = io_utils.load_json(doc2snt_mapping)
    assert isinstance(doc2snt_mapping, dict)
    # take inverse
    snt2doc_mapping = {v: k for k, v in doc2snt_mapping.items()}

    assert len(metas) == len(snt2doc_mapping)
    for i, meta in enumerate(metas):
      meta['id'] = snt2doc_mapping[i]

  elif load_leamr:
    logger.info("Loading LEAMR alignments")
    alignments = io_utils.load_json(f'{fpath}.mrged.subgraph_alignments.json')
    for meta in metas:
      cur_id = meta[C.ID]
      meta[C.ALIGNMENT] = alignments[cur_id]

  logger.info("Loaded %d AMR data from file at `%s`", len(metas), fpath)
  return metas, graphs

def load_amr_dir(dirpath, clean=False, load_leamr=False):
  """
  `prune_duplicates` only effective if `split_graph` is True
  """
  metas_list, graphs_list = [], []
  for fname in os.listdir(dirpath):
    if fname.endswith('.txt'):
      fpath = os.path.join(dirpath, fname)
      metas, graphs = load_amr_file(fpath, clean=clean, load_leamr=load_leamr)
      metas_list.extend(metas)
      graphs_list.extend(graphs)

  logger.info("Loaded %d AMR data from dir at `%s`", len(metas_list), dirpath)
  return metas_list, graphs_list

def save_amr_corpus(meta_list, graph_list, fpath_or_dir, fname=None, use_penman=False):
  fpath = io_utils.get_fpath(fpath_or_dir, fname)

  out = []
  for comment_dict, amr_str in zip(meta_list, graph_list):
    comment_str = '\n'.join([f'# ::{k} {v}' for k,v in comment_dict.items()])
    if use_penman:
      amr_str = penman.encode(penman.decode(amr_str))
    out.append(f'{comment_str}\n{amr_str}')

  io_utils.save_txt(out, fpath, delimiter='\n\n')
