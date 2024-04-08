#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/3/24 11:57 AM
import logging
from typing import Dict, List, Optional, Union

import tqdm

from data.example import Example
from structure import DocGraph, Node
from utils import consts as C, regex_utils

logger = logging.getLogger(__name__)


class Document:

  """single UMR document

  self.doc_id: full document identifier, e.g., `english_umr-0004`
  self.id: last 4 digits as int, e.g., `4`

  mostly a container for a list of examples
  """

  def __init__(
          self,
          doc_id: str,
          examples: Optional[List[Example]] = None,
          snt_idx_mapping: Optional[Dict[int, int]] = None,
          init_global_doc_graph: bool = False,
          init_var_nodes: bool=False
  ):
    # `init_var_nodes`: replace every node string in doc_graph with actual Node pointers from snt_graph
    self.doc_id = doc_id
    self.snt_idx_mapping = snt_idx_mapping
    self.has_snt_idx_mapping = snt_idx_mapping is not None

    if not examples:
      examples = dict()
    examples_dict = dict()
    for example in examples:
      snt_idx = self.snt_idx_mapping[example.snt_idx] if self.has_snt_idx_mapping else example.snt_idx
      examples_dict[snt_idx] = example
    self.examples = examples_dict  # type: Dict[int, Example]

    if init_var_nodes:
      self.init_vars()

    self.global_doc_graph = None
    if init_global_doc_graph:
      self.global_doc_graph = DocGraph(self.idx)

  def init_vars(self):
    logger.info("Initializing node variables in %s document triples", self.doc_id)
    for example in self:
      doc_graph = example.doc_graph
      for key in [C.COREF, C.MODAL, C.TEMPORAL]:
        triples = getattr(doc_graph, f'{key}_triples')
        for i, triple in enumerate(triples[:]):
          s,r,t = triple

          replace = False
          if not isinstance(s, Node):
            snt_idx, var = regex_utils.parse_var(s)
            snt_graph = self.get_example(snt_idx).snt_graph # in this context, `snt_idx` is a str, like '12'
            s = snt_graph.get_node(snt_graph.var2idx[s])
            replace = True

          if not isinstance(t, Node):
            snt_idx, var = regex_utils.parse_var(t)
            snt_graph = self.get_example(snt_idx).snt_graph # in this context, `snt_idx` is a str, like '12'
            t = snt_graph.get_node(snt_graph.var2idx[t])
            replace = True

          if replace:
            doc_graph.update_triple_at_index(key, i, (s,r,t) )

  def distribute_global_doc_graph_triples(self, add_root2author_modal=False):
    # if each example has no doc_graph yet, first init
    for snt_idx, example in self.examples.items():
      # sanity check
      example_snt_idx = self.snt_idx_mapping[example.snt_idx] if self.has_snt_idx_mapping else example.snt_idx
      assert int(snt_idx) == int(example_snt_idx)
      if not example.has_doc_graph():
        example.init_doc_graph(snt_idx)

    # distribute triples
    # snt idxs start at 1 (min)
    global_doc_graph = self.global_doc_graph
    for name in [C.MODAL, C.TEMPORAL, C.COREF]:
      cur_triples = getattr(global_doc_graph, f'{name}_triples')
      if len(cur_triples) > 0:
        for triple in tqdm.tqdm(cur_triples, desc=f'[Distributing {self.doc_id} {name.capitalize()} Triples]'):
          p,r,c = triple

          # these are 1-based ints (0 means abstract)
          # so when getting example from `umr_doc` must (1) subtract by 1 or (2) convert to str
          p_snt_idx, c_snt_idx = p.snt_idx, c.snt_idx
          p_abstract, c_abstract = p.is_abstract(), c.is_abstract()
          if p_abstract and c_abstract:
            continue

          elif p_abstract:
            # use c_snt_idx
            if self.has_snt_idx_mapping:
              try:
                c_snt_idx = self.snt_idx_mapping[c_snt_idx]
              except KeyError:
                # just skip
                logger.warning("Error  %s Triple: %s", name, triple)
                continue
            assert c_snt_idx > 0, f"??? {c_snt_idx} {c}"
            local_doc_graph = self.get_example(c_snt_idx).doc_graph
            getattr(local_doc_graph, f'add_{name}')(triple)

          elif c_abstract:
            logger.warning("Shouldn't reach here")
            breakpoint()

          else:
            # normal case
            if self.has_snt_idx_mapping:
              try:
                p_snt_idx = self.snt_idx_mapping[p_snt_idx]
                c_snt_idx = self.snt_idx_mapping[c_snt_idx]
              except KeyError:
                # just skip
                logger.warning("Error  %s Triple: %s", name, triple)
                continue
            assert p_snt_idx > 0 # not used since we add this to child, but jsut in case
            assert c_snt_idx > 0
            local_doc_graph = self.get_example(c_snt_idx).doc_graph
            getattr(local_doc_graph, f'add_{name}')(triple)

    # optional post-step
    if add_root2author_modal:
      for example in self:
        local_doc_graph = example.doc_graph
        if local_doc_graph.has_modals():
          # internally avoids duplicate if already exists
          local_doc_graph.add_root2author_modal()

    # at this point, `global_doc_graph` is no longer necessary
    self.global_doc_graph = None

  def add_snt_prefix_to_snt_vars(self):
    for snt_idx, example in self.examples.items():
      example.add_snt_prefix_to_vars(snt_idx)

  @property
  def idx(self) -> int:
    # `english_umr-0004` -> 4
    try:
      return  int(self.doc_id.split('-')[-1])
    except ValueError:
      return  1

  @property
  def doc_graph(self) -> Union[None, DocGraph]:
    return  self.global_doc_graph

  @property
  def examples_list(self):
    return  list(self.examples.values())

  def has_global_doc_graph(self):
    return self.global_doc_graph is not None

  def __len__(self):
    return len(self.examples)

  ### to avoid confusion, disable indexing behavior
  def __getitem__(self, item):
    raise Exception("Indexing behavior is disabled for Document; use `get_example` or `get_ith_example`")

  def get_example(self, snt_idx: int):
    if isinstance(snt_idx, str) and snt_idx.isdigit():
      snt_idx = int(snt_idx)
    assert isinstance(snt_idx, int)
    return self.examples[snt_idx]

  def get_ith_example(self, index: int):
    assert isinstance(index, int)
    return self.examples[index+1]

  def __iter__(self):
    for x in self.examples.values():
      yield x

  def __repr__(self):
    return f"UMR Document `{self.doc_id}` with {len(self)} examples"
