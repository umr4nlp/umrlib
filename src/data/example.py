#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 1/23/24 8:31 PM
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from structure import Alignment, DocGraph, SntGraph

logger = logging.getLogger(__name__)


@dataclass
class Example:

  """single UMR annotation example

  1) sentence
  2) sentence-level graph
  3) alignment
  4) document-level graph

  see `document.py` for a document containing one or more examples
  see `UMR.py` for a corpus containing one or more documents
  """

  # unique snt-id in a document as int (`::snt1 ...`, `::snt34 ...`)
  snt_idx: int
  doc_id: str = None

  # text
  snt: str = None # sentence
  toks: List[str] = None # tokenized sentence

  # structures
  snt_graph: Union[str, SntGraph] = None
  dep_graph: Union[str, SntGraph] = None
  alignment: Union[str, Dict[str, Tuple[Union[int, str]]], Alignment] = None
  doc_graph: Union[str, DocGraph] = None

  def has_toks(self) -> bool:
    return self.toks is not None

  def has_snt_graph(self) -> bool:
    return self.snt_graph is not None

  def has_alignment(self) -> bool:
    return self.alignment is not None

  def has_doc_graph(self) -> bool:
    return self.doc_graph is not None

  def init_doc_graph(self, snt_idx=None):
    # initialize empty per-sentence document graph
    snt_idx = snt_idx if snt_idx else self.snt_idx # not that important
    self.doc_graph = DocGraph(snt_idx)
    return self.doc_graph

  def update_snt_idx(self, snt_idx: int):
    if isinstance(self.snt_graph, str):
      logger.warning("Updating Example's `snt_idx` has no effect since `snt_graph` is a string; this is not an error")
    else:
      self.snt_graph.set_idx(snt_idx, update_nodes=True)
    if isinstance(self.doc_graph, str):
      logger.warning("Updating Example's `snt_idx` has no effect since `doc_graph` is a string; this is not an error")
    self.snt_idx = snt_idx

  def add_snt_prefix_to_vars(self, snt_idx=None):
    snt_idx = snt_idx if snt_idx else self.snt_idx
    for i, node in enumerate(self.snt_graph.node_list):
      if not node.is_attribute:
        node.set_var(f"s{snt_idx}x{i}")

  @property
  def doc_snt_id(self):
    return  f"{self.doc_id}.{self.snt_idx}"

  @property
  def num_toks(self):
    return  len(self.toks)

  def encode(self, with_alignment=False):
    out = [f'# :: snt{self.snt_idx}\t{self.snt if self.snt is not None else " ".join(self.toks)}']

    snt_graph = self.snt_graph
    assert snt_graph is not None
    if isinstance(snt_graph, SntGraph):
      snt_graph = str(snt_graph)
    elif isinstance(snt_graph, list):
      snt_graph = '\n'.join(snt_graph)
    assert isinstance(snt_graph, str)
    out.append(f'# sentence level graph:\n{snt_graph}')

    alignment = self.alignment
    if with_alignment:
      if isinstance(alignment, dict):  # if gold
        alignment = "\n".join([f'{k}: {v[0]}-{v[1]}' for k,v in alignment.items()])
      elif isinstance(alignment, Alignment):
        breakpoint()
      if len(alignment.strip()) > 0:
          out.append(f'# alignment:\n{alignment}')
      else:
          out.append(f'# alignment:')
    else:
      out.append(f'# alignment:')

    doc_graph = self.doc_graph
    if isinstance(doc_graph, DocGraph):
      doc_graph = doc_graph.encode()
    elif isinstance(doc_graph, list):
      doc_graph = '\n'.join(doc_graph)
    out.append(f'# document level annotation:\n{doc_graph}')

    return "\n\n".join(out)

  def __repr__(self):
    return self.encode(with_alignment=True)
