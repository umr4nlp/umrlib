#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 6/12/23 2:57 PM
import logging
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from typing import Dict, Tuple

from penman import constant

from structure.components import Node
from structure.snt_graph import SntGraph
from utils import consts as C

logger = logging.getLogger(__name__)


class Alignment:

  def __init__(self, meta: dict, graph: SntGraph, mode='leamr'):
    amr2tok = dict()
    tok2amr = defaultdict(list)
    graph_var2idx = graph.var2idx

    mode = mode.lower()
    if mode == 'leamr':
      # many (toks) -> many (node gorns)
      for alignment in meta[C.ALIGNMENT]:
        # get token ranges; https://stackoverflow.com/questions/2154249/identify-groups-of-consecutive-numbers-in-a-list
        tok_ranges = []
        for k, g in groupby(enumerate(alignment['tokens']), lambda x: x[0]-x[1]):
          group = list(map(itemgetter(1), g))
          tok_ranges.append((group[0], group[-1]+1))

        for node_gorn in alignment['nodes']:
          aligned_node = graph.get_node_by_gorn(node_gorn)
          aligned_node_idx = aligned_node.idx

          for tok_range in tok_ranges:
            amr2tok[aligned_node_idx] = tok_range
            for tok_idx in range(*tok_range):
              tok2amr[tok_idx].append(aligned_node)

    elif mode == 'ibm':
      # one (node var) -> many (toks)
      # just hold on to attributes, and process them later
      attributes = defaultdict(list)
      for amr_var_or_gorn_tok, tok_range in meta[C.ALIGNMENT].items():
        tmp = amr_var_or_gorn_tok.split('_')
        amr_var_or_gorn = tmp[0]
        alignment_node_label = "_".join(tmp[1:])
        try:
          # attribute case
          int(amr_var_or_gorn)
          attributes[alignment_node_label].append(tok_range)

        except ValueError:
          aligned_node_idx = graph_var2idx[amr_var_or_gorn]
          amr2tok[aligned_node_idx] = tok_range
          for tok_idx in range(*tok_range):
            # tok2amr[tok_idx].append(aligned_node_idx)
            tok2amr[tok_idx].append(graph.get_node(aligned_node_idx))

      # handle attributes again, sorted by the order of appearance from top-down in penman notation
      graph_attributes = sorted(graph.attributes, key=lambda x: x.var)
      for alignment_node_label, tok_ranges in attributes.items():
        attribute_candidates = sorted(
          [x for x in graph_attributes if alignment_node_label in [x.get_label(), constant.evaluate(x.get_label())]], key=lambda x: x.var)
        assert len(attribute_candidates) > 0
        if len(tok_ranges) == 1:
          if len(attribute_candidates) > 1:
            breakpoint()
          graph_attribute = attribute_candidates[0]

          tok_range = tok_ranges[0]
          amr2tok[graph_attribute.idx] = tok_range
          for tok_idx in range(*tok_range):
            tok2amr[tok_idx].append(graph_attribute)

        else:
          # the numbers may mismatch, but `zip` will take care of that and
          tok_ranges = sorted(tok_ranges, key=lambda x: x[0])
          for attribute_candidate, tok_range in zip(attribute_candidates, tok_ranges):
            amr2tok[attribute_candidate.idx] = tok_range
            for tok_idx in range(*tok_range):
              tok2amr[tok_idx].append(attribute_candidate)

    else:
      raise ValueError('Unknown alignment mode: %s' % mode)

    # for tok2amr, sort by depth (i.e., closest to root, or highest in structural depth); then only retain the closest (or highest)
    # this effectively ensures that a token is mapped to at most one AMR node
    # BUT an AMR node is mapped to a span of tokens, don't forget
    for k,v in tok2amr.items():
      if len(v) == 1:
        tok2amr[k] = v[0] # type: Node
      else:
        tok2amr[k] = sorted(v, key=lambda x: x.depth)[0] # type: Node

    # store vars
    self.amr2tok = amr2tok  # type: Dict[int, Tuple[int, int]]
    self.tok2amr = tok2amr  # type: Dict[int, Node]
    tok = meta[C.TOK] if C.TOK in meta else meta[C.SNT]
    self.toks = tok.split()

  def from_span(self, start, end, sort_by_depth=False):
    aligneds = []
    for tok_idx in range(start, end):
      aligned = self[tok_idx]
      if aligned is not None and aligned not in aligneds:
        aligneds.append(aligned)
    if sort_by_depth:
      aligneds = sorted(aligneds, key=lambda x: x.depth)
    return aligneds

  def pprint(self, toks=None):
    toks = toks if toks else self.toks
    out = [f'Node Alignments Mapping', 'TOK -> AMR']
    for i, (tok_idx, aligned_node) in enumerate(self.tok2amr.items()):
      out.append(f'{i}: {toks[tok_idx]} -> {aligned_node.var} ({aligned_node.idx}): {aligned_node.get_label()}')
    return "\n".join(out)

  def __getitem__(self, item):
    return self.tok2amr.get(item, None)

  def __repr__(self):
    return self.pprint()
