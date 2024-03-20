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
  idx: int

  # text
  snt: str = None # sentence
  toks: List[str] = None # tokenized sentence; use `toks` for a list on white-space split

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

  def update_idx(self, idx: int):
    if isinstance(self.snt_graph, str):
      logger.warning("Updating Example's `idx` has no effect since `snt_graph` is a string; this may have been the intention though")
    else:
      self.snt_graph.set_idx(idx, update_nodes=True)
    if isinstance(self.doc_graph, str):
      logger.warning("Updating Example's `idx` has no effect since `doc_graph` is a string; this may have been the intention though")
    self.idx = idx

  # @property
  # def toks(self) -> List[str]:
  #   return  self.tok.split()

  @property
  def num_toks(self):
    return  len(self.toks)

  def encode(self, with_alignment=False):
    out = [f'# :: snt{self.idx}\t{self.snt if self.snt is not None else self.tok}']

    snt_graph = self.snt_graph
    if isinstance(snt_graph, SntGraph):
      snt_graph = str(snt_graph)
    elif isinstance(snt_graph, list):
      snt_graph = '\n'.join(snt_graph)
    out.append(f'# sentence level graph:\n{snt_graph}')

    alignment = self.alignment
    if with_alignment:
      if isinstance(alignment, dict):  # if gold
        alignment = "\n".join([f'{k}: {v[0]}-{v[1]}' for k,v in alignment.items()])
      elif isinstance(alignment, Alignment):
        breakpoint()
      out.append(f'# alignment:\n{alignment}')
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
