#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 7/29/23 03:33
import logging
from typing import Dict, List, Optional, Tuple, Union

try:
  import igraph as ig
except ModuleNotFoundError:
  pass
import penman
from penman import constant as penman_const
from penman.models.noop import NoOpModel

from structure import graph_utils
from structure.components import Edge, Node
from utils import consts as C, regex_utils

logger = logging.getLogger(__name__)


class SntGraph:

  """mostly structural queries

  AMRGraph + DepGraph

  for tree/graph traversals, see `graph_utils`
  """

  def __init__(self, idx: int = -1, var2idx=None):
    # identifier
    self.idx = idx # negative value always means un-anchored in the corpus
    self.var2idx = var2idx

    # structure
    self.root = -1  # type: int
    self.nodes = dict()  # type: Dict[int, Node]
    self.edges = list()  # type: List[Edge]
    self._node_idx = 0 # only used to init each new Node

  def set_idx(self, idx, update_nodes=False):
    if isinstance(idx, str):
      if idx.startswith(C.SNT):
        idx = idx[3:]
      idx = int(idx)
    self.idx = idx

    if update_nodes:
      self.set_node_snt_idx(idx)

  def set_node_snt_idx(self, idx):
    for node in self.node_list:
      node.set_snt_idx(idx)

  def set_var2idx(self, var2idx):
    self.var2idx = var2idx

  ### nodes
  def has_node(self, node: Union[int, Node]):
    node = self.get_node_idx(node)
    return node in self.nodes

  def is_root(self, node: Union[int, Node]):
    return self.root == self.get_node_idx(node)

  def is_terminal_node(self, node: Union[int, Node]):
    node = self.get_node_idx(node)
    outgoing_edges = self.get_edges(src=node)
    return len(outgoing_edges) == 0

  def get_node(self, node: Union[int, Node]) -> Node:
    if isinstance(node, int):
      assert node in self.nodes
      node = self.nodes[node]
    return node

  def get_nodes(self, *nodes, sort_by_depth=False) -> List[Node]:
    nodes = [self.get_node(x) for x in nodes]
    if sort_by_depth:
      nodes.sort(key=lambda x: x.depth)
    return nodes

  def get_nodes_from_span(self, span_range, sort_by_depth=False):
    nodes = [self.get_node(x) for x in range(*span_range)]
    if sort_by_depth:
      nodes.sort(key=lambda x: x.depth)
    return nodes

  # noinspection PyMethodMayBeStatic
  def get_node_idx(self, node: Union[int, Node]) -> int:
    if isinstance(node, Node):
      node = node.idx
    return node

  def get_node_idxs(self, *nodes) -> List[int]:
    return [self.get_node_idx(x) for x in nodes]

  def get_node_by_gorn(self, gorn) -> Node:
    node = self.root
    _nodes = [node] # for debugging

    gorn_ids = gorn.split('.')
    if gorn_ids[0] == '1':
      raise Exception('gorn address should start with 0 not 1')

    for level, gorn_id in enumerate(gorn_ids[1:]):
      # should skip reentrancies
      outgoing_edges = [
        edge for edge in self.edges if not edge.is_reentrancy and edge.src==node]
      gorn_id = int(gorn_id)

      num_outgoing_edges = len(outgoing_edges)
      if gorn_id < num_outgoing_edges :
        node = outgoing_edges[int(gorn_id)].tgt
        _nodes.append(node)

    return self.get_node(node)

  def set_root(self, node: Union[int, Node]):
    node = self.get_node_idx(node)
    # if self.root != -1:
    #   logger.debug('Overwriting existing root %d with %d', self.root, node)
    self.root = node

  def add_node(self, label=None, is_root=False, **kwargs) -> Node:
    # `is_root` records the node as root at the graph level
    assert self._node_idx not in self.nodes
    # node = Node(self.idx, self._node_idx, **kwargs)
    node = Node(self._node_idx, **kwargs)

    if label is not None:
      node.set_label(label)

    self.nodes[self._node_idx] = node
    self._node_idx += 1

    if is_root:
      self.set_root(node)

    return node

  def remove_node(self, node: Union[int, Node], remove_all_edges=False, find_new_root=False):
    node = self.get_node_idx(node)
    if node not in self.nodes:
      logger.debug("Attempted to remove node %d which doesn't exist; skipping..", node)
      return

    node_is_root = node == self.root_node

    # remove from node register
    node = self.get_node(node)

    logger.debug("Removing Node %s! Graph Before:\n%s", node, self)
    for k,v in self.nodes.copy().items():
      if node == v:
        self.nodes.pop(k)

    # remove all edges in which current `node` participates
    srcs, tgts, labels = [], [], []

    removed_edges = []
    for edge in self.get_edges(node, undirected=remove_all_edges):
      removed_edges.append(edge)
      self.remove_edge(edge)

      src, tgt = edge.src, edge.tgt
      if src != node.idx and edge.src not in srcs:
        srcs.append(edge.src)
        labels.append(edge.label)
      if tgt != node.idx and edge.tgt not in tgts:
        tgts.append(edge.tgt)

    if self.root_node == node:
      if len(srcs) == 0:
        if find_new_root:
          self.set_root(tgts[0])
      elif len(tgts) == 0:
        if find_new_root:
          self.set_root(srcs[0])
      else:
        src = srcs[0]
        tgt = tgts[0]
        label = labels[0] if len(labels) > 0 else ":ARG0"  # default fallback

        self.add_edge(src, tgt, label=label)
        if node_is_root and find_new_root:
          # self.root = src
          self.set_root(src)

    bfs_nodes, bfs_edges = graph_utils.bfs(self, undirected_edges=True)
    num_bfs_edges = len(bfs_edges)
    if num_bfs_edges != self.num_edges:
      raise Exception(f'Disconnected Graph: {num_bfs_edges} vs {self.num_edges}')

    logger.debug("Removed Node %s! Graph After:\n%s", node, self)

    return node, removed_edges

  ### edges
  def expand_edge(
          self,
          edge: Edge,
          get_nodes=False,
          **edge_label_kwargs
  ) -> Tuple[Union[int, Node], Union[int, Node], List[str]]:
    assert edge in self.edges

    src, tgt = edge.src, edge.tgt
    if get_nodes:
      src, tgt = self.get_nodes(edge.src, edge.tgt)
    edge_labels = [edge.get_label(**edge_label_kwargs)]
    return src, tgt, edge_labels

  def get_edge(
          self,
          src: Union[int, Node],
          tgt: Union[int, Node],
          undirected=False
  ) -> Tuple[bool, Union[None, Edge]]:
    """fetches an edge by its endpoints, can be used to check if such edge exists at all"""
    src, tgt = self.get_node_idx(src), self.get_node_idx(tgt)

    for edge in self.edges:
      edge_src, edge_tgt = edge.src, edge.tgt
      if edge_src == src and edge_tgt == tgt or (undirected and edge_src == tgt and edge_tgt == src):
        return True, edge
    return False, None

  def get_edges(
          self,
          src: Optional[Union[int, Node]] = None,
          tgt: Optional[Union[int, Node]] = None,
          label: Optional[str] = None,
          undirected=False,
          sort=False,
  ) -> List[Optional[Edge]]:
    """identifies edges that fit certain endpoint conditions
      ex1. all edges with `2` as src
      ex2. all edges with `3` as tgt having label `op`

    """
    has_src, has_tgt = src is not None, tgt is not None
    if has_src:
      src = self.get_node_idx(src)
    if has_tgt:
      tgt = self.get_node_idx(tgt)

    has_label = label is not None
    # if has_label:
    #   assert has_src or has_tgt, 'searching edges\ just by label not allowed currently'

    edges = []  # matching edges

    # first apply endpoints requirement(s) if any
    skip_anchor_filter = False
    if has_src and has_tgt:
      # NOTE: no explicit edge for self-loop
      # also, sorting has no effect here since at most a single edge case but including
      # 0 for compatibility
      has_edge_flag, edge = self.get_edge(src, tgt, undirected=undirected)
      if has_edge_flag:
        edges = [(edge, 0)]

    elif has_src:
      for edge in self.edges:
        if edge.src == src:
          edges.append((edge, self.get_node(edge.tgt).get_anchor()))
        elif undirected and edge.tgt == src:
          edges.append((edge, self.get_node(edge.src).get_anchor()))

    elif has_tgt:
      for edge in self.edges:
        if edge.tgt == tgt:
          edges.append((edge, self.get_node(edge.src).get_anchor()))
        elif undirected and edge.src == tgt:
          edges.append((edge, self.get_node(edge.tgt).get_anchor()))

    else:
      edges = self.edges
      skip_anchor_filter = True

    if not skip_anchor_filter:
      if sort: # use anchor here
        edges = [x for x in sorted(edges, key=lambda x: x[1])]

      # drop anchor now, only retain edges
      edges = [x[0] for x in edges]

    if has_label:
      check_labels = [label]
      if undirected:
        check_labels.append(regex_utils.invert_edge_label(label))
      edges = [edge for edge in edges if edge.get_label() in check_labels]

    return edges

  def invert_edge(self, edge: Edge):
    inverted_label = regex_utils.invert_edge_label(edge.get_label())
    self.reroute_edge(edge, new_src=edge.tgt, new_tgt=edge.src, new_label=inverted_label)

  def add_edge(
          self,
          src: Union[int, Node],
          tgt: Union[int, Node],
          label: str,
          **kwargs
  ) -> Edge:
    src, tgt = self.get_node_idx(src), self.get_node_idx(tgt)

    # UPDATE (March 2024): allow duplicate (i.e., src and tgt the same), but raise warning
    has_edge_flag, ref = self.get_edge(src, tgt, undirected=kwargs.get('undirected', True))
    if has_edge_flag:
      logger.debug("Add an edge (%s %s %s) that already exists (%s) --- this is not an error",
                   self.get_node(src), label, self.get_node(tgt), ref)

    edge = Edge(src, tgt, label, **kwargs)
    # BUT cannot allow everything (src, label, and tgt) to be the same
    assert edge not in self.edges
    self.edges.append(edge)

    return edge

  def remove_edge(self, edge: Edge):
    if edge in self.edges:
      self.edges.remove(edge)

  def reroute_edge(
          self,
          edge: Edge,
          new_src: Optional[Union[int, Node]] = None,
          new_tgt: Optional[Union[int, Node]] = None,
          new_label: Optional[str] = None,
  ):
    assert edge in self.edges

    has_src, has_tgt = new_src is not None, new_tgt is not None
    if has_src and isinstance(new_src, Node):
      new_src = new_src.idx
    if has_tgt and isinstance(new_tgt, Node):
      new_tgt = new_tgt.idx

    # each assert avoids self-loop
    if has_src and has_tgt:
      assert new_src != new_tgt
      logger.debug("Edge setting new src (%d -> %d) and new tgt (%d -> %d)", edge.src, new_src, edge.tgt, new_tgt)
      edge.src = new_src
      edge.tgt = new_tgt

    elif has_src:
      assert new_src != edge.tgt
      logger.debug("Edge setting new src (%d -> %d)", edge.src, new_src)
      edge.src = new_src

    elif has_tgt:
      assert edge.src != new_tgt
      logger.debug("Edge setting new tgt (%d -> %d)", edge.tgt, new_tgt)
      edge.tgt = new_tgt

    if new_label is not None:
      edge.set_label(new_label)

    return edge

  def plot(self, title=None, sugiyama=False, plot=False, fig=None, ax=None, ):
    # returns ig.Graph, but can also plot to ax
    root = -1
    mapping, node_labels = dict(), list()
    for i, node_idx in enumerate(self.nodes):
      mapping[node_idx] = i
      node = self.get_node(node_idx)
      if node.is_attribute:
        node_label = penman_const.evaluate(node.get_label())
        node_labels.append(penman_const.quote(node_label))
      else:
        node_label = node.get_label()
        if node_label not in [C.ROOT, C.AUTHOR, 'DCT']:
          # node_labels.append(f"{node.as_var()} /\n{node_label}")
          node_labels.append(node_label)
        else:
          node_labels.append(node_label)
      if node_idx == self.root:
        root = i

    edges, edge_labels = list(), list()
    for edge in self.edges:
      src, tgt = edge.src, edge.tgt
      edge_label = edge.get_label()
      edges.append( (mapping[src], mapping[tgt]) )
      edge_labels.append(edge_label)

    g = ig.Graph(self.num_nodes, edges, directed=True)
    g['title'] = title
    g.vs['label'] = node_labels
    g.es['label'] = edge_labels

    if sugiyama:
      layout = g.layout_sugiyama() # graph
    else:
      layout = g.layout_reingold_tilford(root=[root]) # tree
    layout.rotate(180)
    layout.mirror(0)

    if plot:
      ig.plot(
        g,
        target=ax,
        layout=layout,
        edge_label=g.es['label'],
        edge_background='white',
        edge_width=0.5,
        edge_label_size=8.,
        edge_arrow_size=8,
        vertex_color="white",
        vertex_frame_width=1.0,
        vertex_label=g.vs['label'],
        vertex_label_size=8.,
        vertex_size=55,
        vertex_shape="circle",
      )
      if title:
        ax.set_title(title)
      ax.axis('off')
      return g, fig, ax
    return g

  ### inits
  @classmethod
  def init_amr_graph(cls, graph_string: str, snt_idx: Optional[int] = -1, record_depth_directed=False):
    graph = cls(snt_idx)
    pg = penman.decode(graph_string, model=NoOpModel())

    # 1. collect variables (but not attributes) - only important during this init function
    var2idx = dict()  # type: Dict[str, int]
    for i in pg.instances():  # a) concepts `(a :instance aim-01)`
      var = i.source
      node = graph.add_node(label=i.target, snt_idx=snt_idx, var=var)
      var2idx[var] = node.idx

    # 2. handle root
    pg_top = pg.top
    pg_root = var2idx[pg_top]
    graph.set_root(pg_root)

    # 3. collect reentrancies
    # since JAMR's gorn description ignores reentrancy triples where by reentrancy triples
    # it's meant all edges whose target is a reentrancy variable per the corpus annotation
    reentrancies = list()
    reentrancy_order = dict()
    for r in pg.reentrancies():
      r_id = var2idx[r]
      reentrancies.append(r_id)

      spans = dict()
      # (t2 / talk-01 ...) but get rid of this if root
      instance_match_start = regex_utils.search_reentrancy_start(r, graph_string)
      if instance_match_start > 0:
        spans[0] = instance_match_start
      # noinspection PyTypeChecker
      for i, m in enumerate(regex_utils.finditer_reentrancy_span(r, graph_string), 1):
        spans[i] = m
      reentrancy_order[r_id] = [x[0] for x in sorted(spans.items(), key=lambda x: x[1])]

    # 4. collect edges
    attr_idx = 0
    variables = pg.variables()
    for t in pg.triples:
      edge_label = t[1]
      if edge_label == C.INSTANCE_EDGE:
        continue

      src, tgt = t[0], t[2]

      # source
      src_id = var2idx[src]

      # target
      is_reentrancy = False
      if tgt in variables:
        tgt = var2idx[tgt]
        # edge
        if tgt in reentrancy_order and len(reentrancy_order[tgt]) > 0:
          # leftmost -> if 0, valid edge; otherwise, reentrancy edge and so ignored by gorn
          if reentrancy_order[tgt].pop(0) > 0:
            is_reentrancy = True
      else:
        # attribute (var is meaningless)
        #### UPDATE: var is important; save as nth attribute from the top
        # tgt = graph.add_node(label=tgt, is_attribute=True, var=None)
        attr_idx_str = str(attr_idx)
        tgt = graph.add_node(label=tgt, is_attribute=True, var=attr_idx_str)
        var2idx[attr_idx_str] = tgt.idx
        attr_idx += 1

      graph.add_edge(src_id, tgt, label=edge_label, is_reentrancy=is_reentrancy)

    # 5. record depths for future reference
    graph_utils.record_depths(graph, undirected_edges=False if record_depth_directed else True)

    # # sanity check
    # for r_id, r_list in reentrancy_order.items():
    #   assert len(r_list) == 0, f'found non-empty reentrancy order list: {r_id}'
    graph.set_var2idx(var2idx)

    return graph

  @classmethod
  def init_dep_graph(cls, ud_conllu: List[dict], snt_idx: Optional[int] = -1):
    # technically a tree
    graph = cls(snt_idx)

    # 1. nodes
    dep_heads, dep_rels = [], []
    for ud_tok in ud_conllu:
      dep_heads.append(ud_tok.pop(C.DEP_HEAD))
      dep_rels.append(ud_tok.pop(C.DEP_REL))
      graph.add_node(**ud_tok)

    # 2. edges
    for cur_node, (dep_head, dep_rel) in enumerate(zip(dep_heads, dep_rels)):
      assert cur_node != dep_head

      if dep_head == -1:
        assert dep_rel == C.ROOT
        graph.set_root(cur_node)
      else:
        # replace any embedded colons (ex) `obl:npmod` -> `obl-npmod`
        graph.add_edge(dep_head, cur_node, label=dep_rel.replace(':', '-'))

    # distance from root
    graph_utils.record_depths(graph)

    return graph

  ### properties
  @property
  def root_node(self) -> Union[None, Node]:
    return self.get_node(self.root) if self.root != -1 else None

  @property
  def node_list(self):
    # used to be important when nodes could be merged, but not so much in current implementation
    node_list = []
    for x in self.nodes.values():
      if x not in node_list:
        node_list.append(x)
    return node_list

  @property
  def attributes(self):
    attribute_list = []
    for x in self.nodes.values():
      if x.is_attribute and x not in attribute_list:
        attribute_list.append(x)
    return  attribute_list

  @property
  def concepts(self):
    concepts = []
    for node in self.node_list:
      if regex_utils.is_concept(node.get_label()):
        concepts.append(node)
    return concepts

  @property
  def num_nodes(self):
    return len(self.node_list)

  @property
  def num_edges(self):
    return len(self.edges)

  @property
  def num_attributes(self):
    return  len(self.attributes)

  @property
  def num_concepts(self):
    return  len(self.concepts)

  def __len__(self):
    return self.num_nodes

  def __repr__(self):
    if len(self) == 0:
      return f"Empty Graph with idx: `{self.idx}`"
    top = self.root_node.as_var(self.idx) if self.idx != -1 else self.root_node.as_var()
    triples = graph_utils.collect_triples(self)
    # noinspection PyTypeChecker
    graph = penman.Graph(top=top, triples=triples)
    return penman.encode(graph, compact=False, indent=4)
