#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/3/24 11:58 AM
"""utils for processing UMR corpora"""
import logging
import os
from typing import Dict, List, Optional, Union

import tqdm

from data import Document, Example, tokenize
from structure import DocGraph, SntGraph
from utils import consts as C, io_utils

logger = logging.getLogger(__name__)


def parse_umr_doc_id(doc_id: str, as_int=False):
  tmp = doc_id.split('.')
  try:
    int(tmp[-1])

    # e.g., `english_umr-xxxx.y`
    prefix = '.'.join(tmp[:-1])
    snt_idx = tmp[-1]
    if as_int:
      snt_idx = int(snt_idx)

    tmp = prefix.split('-')
    header = tmp[:-1]
    doc_idx = tmp[-1]
    if as_int:
      doc_idx = int(doc_idx)
    return header, doc_idx, snt_idx

  except ValueError:
    # e.g., `english_umr-xxxx`
    tmp = doc_id.split('-')
    header = tmp[:-1]
    doc_idx = tmp[-1]
    if as_int:
      doc_idx = int(doc_idx)
    return header, doc_idx

def build_umr_id(doc_idx: Union[str, int], snt_idx: Optional[Union[str, int]] = None):
  # ensure int type for 0-padding, e.g., 0004
  doc_id = None
  if isinstance(doc_idx, str):
    try:
      doc_idx = int(doc_idx)
    except ValueError:
      doc_id = doc_idx

  if not doc_id:
    doc_id = f'english_umr-{doc_idx:04}'

  if snt_idx:
    doc_id = f'{doc_id}.{snt_idx}'

  return doc_id

def merge_umr_docs(docs: Dict[Union[int, str], Document], split_mapping) -> Dict[str, Document]:
  if isinstance(split_mapping, str):
    split_mapping = io_utils.load_json(split_mapping)
  assert isinstance(split_mapping, dict)

  doc_split_mapping = split_mapping[C.DOC]
  # snt_split_mapping = split_mapping[C.SNT]

  merged_docs = dict()
  for split_doc_id, (ref_doc_id, doc_idx) in tqdm.tqdm(sorted(doc_split_mapping.items()), desc=f"[Merge Splits]"):
    cur_doc = docs[split_doc_id]
    if doc_idx == 0:
      merged_docs[ref_doc_id] = cur_doc
    else:
      merged_docs[ref_doc_id].merge(cur_doc)

  return merged_docs

def prepare_per_snt_doc_graph(docs: Dict[Union[int, str], Document], add_snt_prefix_to_vars=False):
  for doc_id, umr_doc in tqdm.tqdm(docs.items(), desc=f'[Prepare Per-Snt DocGraph]'):
    # init empty doc_graph per example
    for example in umr_doc:
      example.doc_graph = DocGraph(idx=example.idx)

    # this is the global doc_grpah, to be fragmented
    global_doc_graph = umr_doc.doc_graph

    # begin breakdown
    for name in [C.MODAL, C.TEMPORAL, C.COREF]:
      for triple in getattr(global_doc_graph, f'{name}_triples'):
        p,r,c = triple

        # these are 1-based ints (0 means abstract)
        # so when getting example from `umr_doc` must (1) subtract by 1 or (2) convert to str
        p_snt_idx, c_snt_idx = p.snt_idx, c.snt_idx
        p_abstract, c_abstract = p.is_abstract(), c.is_abstract()
        if p_abstract and c_abstract:
          continue

        elif p_abstract:
          # use c_snt_idx
          assert c_snt_idx > 0, f"??? {c_snt_idx} {c}"
          # assert c_snt_idx >= 0, f"??? {c_snt_idx} {c}"
          local_doc_graph = umr_doc[c_snt_idx-1].doc_graph
          getattr(local_doc_graph, f'add_{name}')(triple)

        elif c_abstract:
          breakpoint()

        else:
          # normal case
          assert p_snt_idx > 0 # not used since we add this to child, but jsut in case
          assert c_snt_idx > 0
          local_doc_graph = umr_doc[c_snt_idx-1].doc_graph
          getattr(local_doc_graph, f'add_{name}')(triple)

    for example in umr_doc:
      local_doc_graph = example.doc_graph
      if local_doc_graph.has_modals():
        local_doc_graph.add_root2author_modal()

      if add_snt_prefix_to_vars:
        for i, node in enumerate(example.snt_graph.node_list):
          if not node.is_attribute:
            node.set_var(f"s{example.idx}x{i}")

def export_umr_docs(docs: Dict[Union[int, str], Document], output_dir, with_alignment=False):
  assert os.path.isdir(output_dir)

  logger.info("Exporting UMRs at `%s`", output_dir)
  for doc_id, doc in docs.items():
    out = []
    for example in doc:
      out.append(example.encode(with_alignment=with_alignment))
    io_utils.save_txt(out, output_dir, fname=f'{doc_id}.txt', delimiter='\n\n\n')

def load_umr_file_aux(
        raw_txt,
        snt_to_tok=None,
        init_alignment=False,
        init_graphs=False,
) -> List[Example]:
  # here alignment needs not be init since it's not used at all
  examples = []

  segments = list(filter(None, raw_txt.split('\n\n\n')))
  for segment in segments:
    idx = snt = alignment = snt_graph = doc_graph = None

    for block in segment.strip().split('\n\n'):
      block = block.strip()

      if block.startswith("# :: snt"):
        if '\t' in block:
          a,b = block.split('\t')
        else:
          segment_s = block.split(' ')
          a = " ".join(segment_s[:3])
          b = " ".join(segment_s[3:])

        idx = int(a[8:])
        snt = b.strip()

      elif block.startswith('# alignment'):
        if init_alignment:
          alignment = dict()
          for cur_alignment in block.split('\n')[1:]:
            var, aligned = cur_alignment.split(':')
            var, aligned = var.strip(), aligned.strip()
            if aligned.startswith('-1'):
              aligned_a = aligned_b = -1
            else:
              aligned_a, aligned_b = aligned.split('-')
            alignment[var] = int(aligned_a), int(aligned_b)
        else:
          alignment = "\n".join(block.split('\n')[1:])

      elif block.startswith('# sentence level graph'):
        snt_graph = "\n".join(block.split('\n')[1:])
        if init_graphs:
          snt_graph = SntGraph.init_amr_graph(snt_graph, snt_idx=idx)

      elif block.startswith('# document level annotation'):
        doc_graph = "\n".join(block.split('\n')[1:])
        if init_graphs:
          doc_graph = DocGraph.init_from_string(doc_graph)

      else:
        logger.debug("Found a block with unconventional format: %s", block)

    toks = tokenize(snt, mode=snt_to_tok).split()

    # noinspection PyTypeChecker
    example = Example(
      idx=idx,
      snt=snt,
      toks=toks, # if `tokenizer` is None or "", this is just `snt`,
      snt_graph=snt_graph,
      alignment=alignment,
      doc_graph=doc_graph
    )
    examples.append(example)

  return examples

def load_umr_file(fpath_or_dir, fname=None, init_document=False, init_doc_var_nodes=False, **kwargs) -> Union[Document, List[Example]]:
  fpath = io_utils.get_fpath(fpath_or_dir, fname)

  fname = os.path.basename(fpath)
  doc_id = os.path.splitext(fname)[0]

  raw_txt = io_utils.load_txt(fpath)
  examples = load_umr_file_aux(raw_txt, **kwargs)
  logger.info("Found %d UMR annotations in `%s`", len(examples), doc_id)

  if init_document:
    return Document(doc_id, examples, init_var_nodes=init_doc_var_nodes)

  return examples

def load_umr_dir(umr_dir, exts=None, **kwargs) -> Union[List[Document], List[List[Example]]]:
  if exts is None:
    exts = ['.txt']

  count = 0
  docs_list = []
  # for fname in sorted(os.listdir(umr_dir)):
  for fname in os.listdir(umr_dir):
    ext = os.path.splitext(fname)[-1]
    if ext in exts:
      data = load_umr_file(os.path.join(umr_dir, fname), **kwargs)

      num_data = len(data)
      if num_data > 0:
        docs_list.append(data)
      count += num_data

  logger.info("UMR dir at `%s` contains %d UMR annotations in %d documents", umr_dir, count, len(docs_list))

  return docs_list

