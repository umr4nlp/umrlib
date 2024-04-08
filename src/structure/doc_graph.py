#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 1/24/24 1:02 AM
import logging
from typing import List, Tuple, Union

from structure.components import Node
from utils import consts as C, regex_utils

logger = logging.getLogger(__name__)


class DocGraph:

  """single document graph object

  can be init in 2 ways, although the init mechanism is the same
  (1) single DocGraph object for an entire document
  (2) single DocGraph object per sentence in a document

  (1) recommended for prediction UMRs, whereas (2) is for gold UMRs:
  * (1) inits only a single instance of global nodes `root`, `author` and `dct`, and it may lead to some triples
    containing these abstract nodes not being properly matched with the correct sentence for the final output

  a triple: (
    parent: Union[str, Node],
    relation: str,
    child: Union[str, Node]
   )

  * typically, parent appears either before the child or at the same sentence
    * only rarely, in a coref `:subset-of` relation, would a child appear before parent
    * other than this case, action may be taken to inverse a triple, if possible
  * using Node is generally preferred when storing parent and child, but not always necessary
  * relation label string should be decorated with `:` prefix, e.g., `:same-entity`
  """

  # def __init__(self, idx: int = -1, num_snts=1, modal_triples=None, temporal_triples=None, coref_triples=None):
  def __init__(self, idx: int = -1, modal_triples=None, temporal_triples=None, coref_triples=None):
    ### identifier
    # `idx` could be an integer id for (1) a document or (2) a sentence
    # [!] depends on the context!! BUT has no practical use, just don't specify
    self.idx = idx
    # self.num_snts = num_snts

    ### global abstract nodes
    self.root_node = Node(idx=-1, label=C.ROOT)
    self.author_node = Node(idx=-1, label=C.AUTHOR)
    self.dct_node = Node(idx=-1, label=C.DCT_FULL)

    ### triples
    self.modal_triples = [] if modal_triples is None else modal_triples
    self.temporal_triples = [] if temporal_triples is None else temporal_triples
    self.coref_triples = [] if coref_triples is None else coref_triples

  def add_modal(self, *triple):
    if isinstance(triple, tuple) and len(triple) == 1:
      triple = triple[0] # type: Tuple[Union[str, Node], str, Union[str, Node]]
    p,r,c = triple
    if p == C.ROOT:
      p = self.root_node
    elif p == C.AUTHOR:
      p = self.author_node
    if c == C.AUTHOR:
      c = self.author_node

    triple = (p,r,c)
    if triple not in self.modal_triples:
      # if self.has_modal_node(c, as_tgt=True):
      #   org_modals = self.get_triples('modal', tgt=c)
      #   if len(org_modals) > 1:
      #     breakpoint()
      #   org_modal = org_modals[0]
      #
      #   if org_modal[0].get_label() == C.AUTHOR:
      #     # modal triples from sub-graph maneuvering during amr2umr conversion
      #     self.remove_triple('modal', org_modal)
      #     # pass
      #   else:
      #     # ??? shouldn't reach here
      #     breakpoint()
      self.modal_triples.append(triple)

  def add_root2author_modal(self):
    # always insert at the very beginning
    root2author_modal_triple = self.root_node, C.MODAL_EDGE, self.author_node
    if root2author_modal_triple not in self.modal_triples:
      self.modal_triples.insert(0, root2author_modal_triple)

  def add_modals(self, *triples):
    for triple in triples:
      self.add_modal(triple)

  def add_temporal(self, *triple):
    if isinstance(triple, tuple) and len(triple) == 1:
      triple = triple[0] # type: Tuple[Union[str, Node], str, Union[str, Node]]
    p,r,c = triple
    if p == C.DCT_FULL:
      p = self.dct_node
    elif p == C.ROOT:
      p = self.root_node
    elif c == C.DCT_FULL:
      logger.warning("Found DCT as child??? => %s", str(triple) )
      breakpoint()
    triple = (p,r,c)
    if triple not in self.temporal_triples:
      self.temporal_triples.append(triple)

  def add_coref(self, *triple):
    if isinstance(triple, tuple) and len(triple) == 1:
      triple = triple[0] # type: Tuple[Union[str, Node], str, Union[str, Node]]
    p, r, c = triple

    if isinstance(p, Node) and isinstance(c, Node):
      p_snt_idx, c_snt_idx = p.snt_idx, c.snt_idx
      assert p_snt_idx > 0 and c_snt_idx > 0

      if p_snt_idx > c_snt_idx and r in [C.SAME_EVENT_EDGE, C.SAME_ENTITY_EDGE]:
        p, c = c, p

    triple = (p,r,c)
    if triple not in self.coref_triples:
      self.coref_triples.append(triple)

  def get_triples(self, name=None, src=None, tgt=None, label=None):
    triples = getattr(self, f'{name}_triples') if name else self.triples

    has_src, has_tgt, has_label = src is not None, tgt is not None, label is not None

    out = []
    for triple in triples:
      src_match = triple[0] == src
      tgt_match = triple[2] == tgt
      if has_src and has_tgt:
        if triple[0] ==src and triple[-1] == tgt:
          out.append(triple)
      elif has_src:
        if src_match:
          out.append(triple)
      elif has_tgt:
        if tgt_match:
          out.append(triple)

    out_ = []
    if has_label:
      for triple in out:
        if triple[1] == label:
          out_.append(triple)
      out = out_

    return out

  def remove_triple(self, name, triple):
    triples = getattr(self, f'{name}_triples')
    if triple in triples:
      logger.info("Safely removed `%s` triple `%s` from `%s` doc-graph", name, str(triple), self.idx)
      triples.remove(triple)
    else:
      logger.warning("Tried to remove a triple `%s` that doesn't exist, skipping..", str(triple))

  def has_node(self, node: Node, name=None, as_src=False, as_tgt=False):
    if not isinstance(node, Node):
      return False

    triples = self.triples
    if name:
      triples = getattr(self, f'{name}_triples')
    for triple in triples:
      candidates = []
      if as_src:
        candidates.append(triple[0])
      if as_tgt:
        candidates.append(triple[-1])
      if node in candidates:
        return True
    return False

  def has_triple(self, triple, name):
    return getattr(self, f'has_{name}_triple')(triple)

  def has_modals(self):
    return len(self.modal_triples) > 0

  def has_modal_node(self, node: Node, **kwargs):
    return self.has_node(node, 'modal', **kwargs)

  def has_modal_triple(self, triple):
    if isinstance(triple, tuple) and len(triple) == 1:
      triple = triple[0] # type: Tuple[Union[str, Node], str, Union[str, Node]]
    triple = [triple[0], regex_utils.maybe_strip_edge(triple[1]), triple[2]]
    for modal_triple in self.modal_triples:
      modal_triple = [modal_triple[0], regex_utils.maybe_strip_edge(modal_triple[1]), modal_triple[2]]
      if triple == modal_triple:
        return True
    return False

  def has_temporals(self):
    return len(self.temporal_triples) > 0

  def has_temporal_triple(self, triple):
    if isinstance(triple, tuple) and len(triple) == 1:
      triple = triple[0] # type: Tuple[Union[str, Node], str, Union[str, Node]]
    label = regex_utils.maybe_strip_edge(triple[1])
    triples = [ [triple[0], label, triple[2]] ]

    if label == C.AFTER:
      triples.append( [triple[2], C.BEFORE, triple[0]] )
    elif label == C.BEFORE:
      triples.append([triple[2], C.AFTER, triple[0]])

    for temporal_triple in self.temporal_triples:
      temporal_triple = [temporal_triple[0], regex_utils.maybe_strip_edge(temporal_triple[1]), temporal_triple[2]]
      if temporal_triple in triples:
        return True
    return False

  def has_temporal_node(self, node: Node, **kwargs):
    return self.has_node(node, 'temporal', **kwargs)

  def has_corefs(self):
    return len(self.coref_triples) > 0

  def has_coref_node(self, node: Node, **kwargs):
    return self.has_node(node, 'coref', **kwargs)

  def has_coref_triple(self, triple):
    if isinstance(triple, tuple) and len(triple) == 1:
      triple = triple[0] # type: Tuple[Union[str, Node], str, Union[str, Node]]
    label = regex_utils.maybe_strip_edge(triple[1])
    triples = [ [triple[0], label, triple[2]] ]

    if label == C.SAME_EVENT:
      triples.append( [triple[2], C.SAME_EVENT, triple[0]] )
    elif label == C.SAME_ENTITY:
      triples.append([triple[2], C.SAME_ENTITY, triple[0]])

    for coref_triple in self.coref_triples:
      coref_triple = [coref_triple[0], regex_utils.maybe_strip_edge(coref_triple[1]), coref_triple[2]]
      if coref_triple in triples:
        return True
    return False

  def is_empty(self):
    return self.num_triples == 0

  @property
  def num_corefs(self):
    return  len(self.coref_triples)

  @property
  def num_modals(self):
    return  len(self.modal_triples)

  @property
  def num_temporals(self):
    return  len(self.temporal_triples)

  @property
  def triples(self):
    return  self.coref_triples + self.modal_triples + self.temporal_triples

  @property
  def num_triples(self):
    return  len(self.triples)

  @classmethod
  def init_from_string(cls, graph_string: str):
    # e.g., `s23s0` -> 23
    root_var = graph_string.split()[0][1:]
    snt_idx, _ = regex_utils.parse_var(root_var)
    graph = cls(idx=snt_idx)
    for key, triple in regex_utils.parse_doc_graph(graph_string):
      getattr(graph, f'add_{key}')(triple)
    return graph

  def update_triple_at_index(self, key, index, new_triple):
    assert key in ['coref', 'modal', 'temporal']
    triples = getattr(self, f'{key}_triples')
    old_triple = triples[index]
    triples[index] = new_triple
    msg = f"Updated {key} Triple at index {index} from `{old_triple}` to `{new_triple}`"
    logger.debug(msg)
    return msg

  # def split_per_sent(self, encode=False) -> List[Union[str, 'DocGraph']]:
  #   """split a global doc_graph into smaller doc_graphs, as seen with UMR v1.0
  #
  #   in practice, this means just spreading triples in appropriate bins
  #
  #   it's assumed that the doc ids start from `1`!!!
  #   """
  #   if self.num_snts == 1:
  #     if encode:
  #       return [self.encode()]
  #     else:
  #       return [self]
  #
  #   coref_triples = [[] for _ in range(self.num_snts)]
  #   for src, rel, tgt in self.coref_triples:
  #     src_snt_idx, tgt_snt_idx = src.snt_idx, tgt.snt_idx
  #     if rel != C.SUBSET_OF_EDGE and src_snt_idx > tgt_snt_idx:
  #       # flip src and tgt, but label needs no modifications
  #       src, tgt = tgt, src
  #       tgt_snt_idx = tgt.snt_idx
  #     # now prefer `tgt`
  #     coref_triples[tgt_snt_idx-1].append( (src, rel, tgt) )
  #
  #   temporal_triples = [[] for _ in range(self.num_snts)]
  #   for src, rel, tgt in self.coref_triples:
  #     src_snt_idx, tgt_snt_idx = src.snt_idx, tgt.snt_idx
  #     # only before and after can be flipped, if necessary
  #     if rel in [C.AFTER_EDGE, C.BEFORE_EDGE] and src_snt_idx > tgt_snt_idx:
  #       src,tgt = tgt, src
  #       rel = C.AFTER_EDGE if rel == C.BEFORE_EDGE else C.BEFORE_EDGE
  #       tgt_snt_idx = tgt.snt_idx
  #     # now prefer `tgt`
  #     temporal_triples[tgt_snt_idx-1].append( (src, rel, tgt) )
  #
  #   # presumably for modals, every triple can be flipped except for root2author, although
  #   # every triple in theory should contribute to a single top-down dependency graph
  #   modal_triples = [[] for _ in range(self.num_snts)]
  #   for src, rel, tgt in self.coref_triples:
  #     src_snt_idx, tgt_snt_idx = src.snt_idx, tgt.snt_idx
  #     if src_snt_idx > tgt_snt_idx:
  #       # modality label doesn't need any modifications?
  #       src, tgt = tgt, src
  #       tgt_snt_idx = tgt.snt_idx
  #     # now prefer `tgt`
  #     modal_triples[tgt_snt_idx-1].append( (src, rel, tgt) )
  #
  #   doc_graphs = []
  #   for split_doc_id, (corefs, modals, temporals) in enumerate(zip(coref_triples, modal_triples, temporal_triples), 1):
  #     doc_graph = DocGraph(split_doc_id, coref_triples=corefs, modal_triples=modals, temporal_triples=temporals)
  #     if encode:
  #       doc_graph = doc_graph.encode()
  #     doc_graphs.append(doc_graph)
  #
  #   return doc_graphs

  def triples2string(self, key, triples=None) -> str:
    if triples is None:
      triples = getattr(self, f'{key}_triples')

    prefix_ws_first = " " * 4
    prefix_ws = " " * (len(key)+len(prefix_ws_first)+3)  # `:`, ` `, `(`

    res = [f'{prefix_ws_first}:{key} (']
    for i, triple in enumerate(triples):
      p,r,c = triple
      if isinstance(p, Node):
        p = p.as_var()
      if isinstance(c, Node):
        c = c.as_var()

      triple_str = " ".join([p,r,c])
      if i == 0:
        res[-1] += f"({triple_str})"
      else:
        res.append(f'{prefix_ws}({triple_str})')
    res[-1] += ')'

    return "\n".join(res)

  def encode(self) -> str:
    out = [f'(s{self.idx}s0 / sentence']
    if self.has_corefs():
      out.append(self.triples2string("coref"))
    if self.has_modals():
      out.append(self.triples2string("modal"))
    if self.has_temporals():
      out.append(self.triples2string("temporal"))
    out[-1] += ')'
    return "\n".join(out)

  def __len__(self):
    return self.num_triples

  def __repr__(self):
    return self.encode()
