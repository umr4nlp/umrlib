#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 7/29/23 03:34
"""originally meant for AMR but since modified to fit UMR"""
import logging
from dataclasses import dataclass
from typing import Dict, Union

from structure import graph_utils
from utils import consts as C

logger = logging.getLogger(__name__)


@dataclass
class Node:

  ### identifiers
  # unique node identifier
  idx: int
  # which sentence-level graph this node belongs to (0 if not init, or global)
  snt_idx: int = 0
  # maybe maintain original var assignment
  var: str = None # for constant, this is the nth constant from top down, using Penman notation

  ### structural
  # id of anchored dep node
  anchor: int = -1
  # distance from root
  depth: int = -1

  ### linguistic features
  word: str = None
  lemma: str = None
  pos: str = None
  feats: Dict[str, str] = None

  ### core properties
  label: Union[int, float, str] = None

  # core
  is_attribute: bool = False

  def get_anchor(self) -> int:
    return self.anchor if self.anchor > -1 else self.idx

  def get_depth(self) -> int:
    if self.depth == -1:
      logger.warning("Fetching node's uninitialized depth (-1)..")
    return self.depth

  # these are umr-specific methods
  def as_var(self, snt_idx=None) -> str:
    """get node's variable, using its internal data

    note that it may be more advisable to do so using external data, in which case this method should be avoided
    """
    if self.var is not None:
      return self.var

    # account for global abstract nodes
    if self.is_abstract():
      return self.get_label()

    if snt_idx is not None:
      if self.snt_idx > 0:
        logger.warning("Getting Node's var with `snt_idx` (%s) which overrides the internal value (%s)", snt_idx, self.snt_idx)
      return C.VAR_TEMPLATE_WITH_SNT_PREFIX % (snt_idx, self.idx)
    elif self.snt_idx > 0:
      return C.VAR_TEMPLATE_WITH_SNT_PREFIX % (self.snt_idx, self.idx)
    else:
      return C.VAR_TEMPLATE % self.idx

  def set_var(self, new_var):
    if self.var is not None:
      logger.debug("[!] Overriding var from `%s` to `%s`", self.var, new_var)
    self.var = new_var

  def set_snt_idx(self, snt_idx):
    self.snt_idx = snt_idx

  def get_label(self):
    label = self.label if self.label is not None else self.word
    if not label:
      logger.warning("Node %s has no label; returning None..", self)
    return label

  def set_label(self, label):
    self.label = label

  def is_abstract(self):
    return self.idx == -1

  def __eq__(self, other):
    if not isinstance(other, Node):
      return False
    return self.idx == other.idx and self.snt_idx == other.snt_idx and self.label == other.label

  def __repr__(self):
    head = f'{self.snt_idx}:{self.idx}' if self.snt_idx > 0 else str(self.idx)
    return f'Node({head}):{self.get_label()}'

########################################################################################################################
@dataclass
class Edge:

  # no need for a unique identifier; use `src` and `tgt`
  src: int
  tgt: int
  label: str = None

  # only when reading from full graph annotations, useful if reentrancies exist
  is_reentrancy: bool = False

  def set_label(self, label: str):
    self.label = label

  def get_label(self, decorate=False) -> str:
    """any label set through `set_label` fn will have `:` prefix removed, but being safe here
    """
    label = self.label
    if decorate:
      label = graph_utils.maybe_decorate_edge(label)
    else:
      label = graph_utils.maybe_strip_edge(label)
    return label

  def __eq__(self, other: 'Edge'):
    if not isinstance(other, Edge):
      return False
    return self.src == other.src and self.label == other.label and self.tgt == other.tgt

  def __repr__(self):
    return f'<{self.src}:{self.get_label(decorate=False)}:{self.tgt}>'
