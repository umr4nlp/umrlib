#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/3/24 11:57 AM
import logging
from typing import List, Optional, Union

from data.example import Example
from structure import DocGraph, Node
from utils import consts as C, regex_utils

logger = logging.getLogger(__name__)


class Document:

  """single UMR document

  self.doc_id: full document identifier, e.g., `english_umr-0004`
  self.id: last 4 digits as int, e.g., `4`

  a single global DocGraph is useful when incrementally building as part of AMR2UMR conversion
  """

  def __init__(
          self,
          doc_id: str,
          examples: Optional[List[Example]] = None,
          init_global_doc_graph: bool = False,
          init_var_nodes: bool=False
  ):
    # `init_var_nodes`: replace every node string in doc_graph with actual Node pointers from snt_graph
    self.doc_id = doc_id

    if not examples:
      examples = []
    self.examples = examples

    if init_var_nodes:
      self.init_vars()

    self.global_doc_graph = None
    if init_global_doc_graph:
      self.global_doc_graph = DocGraph(self.id)

  def init_vars(self):
    logger.info("Initializing node variables in %s document triples", self.doc_id)
    for example in self.examples:
      doc_graph = example.doc_graph
      for key in [C.COREF, C.MODAL, C.TEMPORAL]:
        triples = getattr(doc_graph, f'{key}_triples')
        for i, triple in enumerate(triples[:]):
          s,r,t = triple

          replace = False
          if not isinstance(s, Node):
            snt_idx, var = regex_utils.parse_var(s)
            snt_graph = self[snt_idx].snt_graph # in this context, `snt_idx` is a str, like '12'
            s = snt_graph.get_node(snt_graph.var2idx[s])
            replace = True

          if not isinstance(t, Node):
            snt_idx, var = regex_utils.parse_var(t)
            snt_graph = self[snt_idx].snt_graph # in this context, `snt_idx` is a str, like '12'
            t = snt_graph.get_node(snt_graph.var2idx[t])
            replace = True

          if replace:
            doc_graph.update_triple_at_index(key, i, (s,r,t) )

  def merge(self, other: 'Document'):
    # merge examples
    start_snt_id_offset = self.examples[-1].idx+1
    for new_snt_idx, example in enumerate(other.examples, start_snt_id_offset):
      example.update_idx(new_snt_idx)
      self.examples.append(example)

    # merge graph doc
    this_doc_graph, other_doc_graph = self.doc_graph, other.doc_graph
    for modal_triple in other_doc_graph.modal_triples:
      if modal_triple[0] == other_doc_graph.root_node and modal_triple[2] == other_doc_graph.author_node:
        # skip `root2author` modal triple; there should only be one of this in the entire DocGraph
        continue
      this_doc_graph.add_modal(modal_triple)

    for temporal_triple in other_doc_graph.temporal_triples:
      this_doc_graph.add_temporal(temporal_triple)
    for coref_triple in other_doc_graph.coref_triples:
      this_doc_graph.add_coref(coref_triple)

  @property
  def id(self) -> int:
    return  int(self.doc_id.split('-')[-1])

  @property
  def doc_graph(self) -> Union[None, DocGraph]:
    return  self.global_doc_graph

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, item) -> Union[None, Example, List[Example]]:
    if isinstance(item, int):
      return self.examples[item]
    elif isinstance(item, str):
      if item.startswith(C.SNT):
        idx = int(item[3:]) - 1
      else:
        idx = int(item) - 1
      if idx == -1:
        return None
      return self.examples[idx]
    elif isinstance(item, slice):
      return self.examples[item]
    else:
      out = []
      for x in item:
        if isinstance(x, int):
          out.append(self.examples[x])
        else:
          if item.startswith(C.SNT):
            idx = int(item[3:]) - 1
          else:
            idx = int(item) - 1
          out.append(self.examples[idx])
      return out

  def __iter__(self):
    for x in self.examples:
      yield x

  def __repr__(self):
    return f"UMR Document `{self.doc_id}` with {len(self)} examples"
