#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/1/23 16:14
"""tree/graph traversal utils"""
import logging
from typing import List, Optional, Tuple

from utils import consts as C

logger = logging.getLogger(__name__)


### traversals
def bfs(graph, root: int = -1, undirected_edges=False):
  # loop-impl.
  if root == -1:
    root = graph.root

  node_seq, edge_seq = [root], list()

  edge_kwargs = {'sort': True}
  if undirected_edges:
    edge_kwargs['undirected'] = True

  # 0: distance
  stack = [(root, 0)]
  while len(stack) > 0:
    node, dist = stack.pop(0)
    for edge in graph.get_edges(src=node, **edge_kwargs):
      if edge not in edge_seq:
        dst = edge.tgt
        if undirected_edges and dst == node:
          dst = edge.src

        if dst not in node_seq:
          node_seq.append(dst)
          stack.append((dst, dist+1))
        edge_seq.append(edge)

  return node_seq, edge_seq

def dfs(graph, root: int = -1, undirected_edges=False):
  # loop-impl.
  if root == -1:
    root = graph.root

  node_seq, edge_seq = list(), list()

  edge_kwargs = {'sort': True}
  if undirected_edges:
    edge_kwargs['undirected'] = True

  stack = [root]
  while len(stack) > 0:
    node = stack.pop(0)
    if node not in node_seq:
      node_seq.append(node)

    to_push = []
    for edge in graph.get_edges(src=node, **edge_kwargs):
      dst = edge.tgt
      if undirected_edges and dst == node:
        dst = edge.src

      if edge not in edge_seq:
        to_push.append(dst)
        edge_seq.append(edge)

    for x in reversed(to_push):
      stack.insert(0, x)

  return node_seq, edge_seq

def record_depths(graph, undirected_edges=False):
  # shortest number of steps from root, requiring a traversal
  graph.root_node.depth = 0

  stack = [(graph.root, 0)]
  visited = [graph.root]
  while len(stack) > 0:
    cur_node, cur_depth = stack.pop(0)

    child_depth = cur_depth + 1
    for edge in graph.get_edges(src=cur_node, undirected=undirected_edges, sort=True):
      target = edge.tgt
      if undirected_edges and target == cur_node:
        target = edge.src

      if target not in visited:
        visited.append(target)

        child = graph.get_node(target)
        if child.depth == -1:
          child.depth = child_depth
        else:
          child_depth = child.depth
        stack.append((target, child_depth))

def collect_triples(graph) -> List[Optional[Tuple[int, str, int]]]:
  graph_idx = graph.idx
  graph.set_node_snt_idx(graph_idx)
  placeholder_var = C.VAR_TEMPLATE % 0 if graph_idx == -1 else C.VAR_TEMPLATE_WITH_SNT_PREFIX % (graph_idx, 0)

  triples = []
  if graph.num_edges == 0:
    if graph.num_nodes > 0:
      # just get some single node, even if there are multiple nodes; after all, they are disconnected
      node = next(iter(graph.nodes.values()))
      triples.append((node.as_var(), C.INSTANCE_EDGE, node.get_label()))
    else: # last resort
      triples.append((placeholder_var, C.INSTANCE_EDGE, C.XCONCEPT))

  else:
    bfs_nodes, bfs_edges = bfs(graph, undirected_edges=True)
    num_bfs_edges = len(bfs_edges)
    if num_bfs_edges != graph.num_edges:
      raise Exception(f'Disconnected Graph: {num_bfs_edges} vs {graph.num_edges}')

    visited = set()
    for edge in bfs_edges:
      src, tgt = edge.src, edge.tgt
      src_node, tgt_node, edge_labels = graph.expand_edge(
        edge, get_nodes=True, decorate=True)

      # SRC
      if src not in visited:
        visited.add(src)
        if not src_node.is_attribute:
          triples.append((src_node.as_var(), C.INSTANCE_EDGE, src_node.get_label()))

      # SRC -> TGT
      for edge_label in edge_labels:
        if src_node.is_attribute and tgt_node.is_attribute:
          triples.append((src_node.get_label(), edge_label, tgt_node.get_label()))
        elif src_node.is_attribute:
          logger.warning("Found a triple with a source attribute: %s", edge)
          triples.append((src_node.get_label(), edge_label, tgt_node.as_var()))
        elif tgt_node.is_attribute:
          triples.append((src_node.as_var(), edge_label, tgt_node.get_label()))
        else:
          triples.append((src_node.as_var(), edge_label, tgt_node.as_var()))

      # TGT
      if tgt not in visited:
        visited.add(tgt)
        if not tgt_node.is_attribute:
          triples.append((tgt_node.as_var(), C.INSTANCE_EDGE, tgt_node.get_label()))

    if not visited:  # single node case for `ref_graph`
      if graph.num_nodes > 0:
        # just get a single node
        valid_node = [x for x in graph.nodes.values()][0]
        triples.append((valid_node.as_var(), C.INSTANCE_EDGE, valid_node.get_label()))
      else: # last resort
        triples.append((placeholder_var, C.INSTANCE_EDGE, C.XCONCEPT))

  return triples
