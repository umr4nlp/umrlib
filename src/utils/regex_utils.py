#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/4/24 5:17 PM
import re
from typing import Tuple, Union

from utils import consts as C


CONCEPT_PATTERN = re.compile(r'-[0-9]{2}$')
AMR_META_PATTERN = re.compile(r"::([^:\s]+)\s(((?!::).)*)")
SIDX_VAR_PATTERN = re.compile(r's([0-9]+)(.*)')
DOC_GRAPH_PATTERNS = {
  C.TEMPORAL: r':temporal\s*\(((\(.*\:.*\)\s*)*)\)',
  C.MODAL: r':modal\s*\(((\(.*\:.*\)\s*)*)\)',
  C.COREF: r':coref\s*\(((\(.*\:.*\)\s*)*)\)'
}
DOC_GRAPH_SUBPATTERN = r'\(((\S*)\s*(\:\S*)\s*(\S*?))\)'

def is_concept(label: str) -> bool:
  return re.search(CONCEPT_PATTERN, label) is not None

def parse_amr_meta(line: str) -> Tuple[str, str]:
  for x in re.finditer(AMR_META_PATTERN, line):
    yield x.group(1).strip(), x.group(2).strip()

def search_reentrancy_start(reentrancy_var: str, graph: str) -> int:
  return re.search(f'\({reentrancy_var}(\s|\))', graph).span()[0]

def finditer_reentrancy_span(reentrancy_var: str, graph: str) -> int:
  for x in re.finditer(f'[^\/]\s{reentrancy_var}(\s|\))', graph):
    yield x.span()[0]

def parse_var(var: str, prefix_snt_idx=None, as_int_snt_idx=False, offset_snt_idx=False) -> Tuple[Union[int, str], str]:
  match = SIDX_VAR_PATTERN.match(var)
  snt_idx, snt_var = match.group(1), match.group(2)
  if as_int_snt_idx:
    snt_idx = int(snt_idx)
    if offset_snt_idx:
      snt_idx -= 1
  elif prefix_snt_idx is not None:
    snt_idx = f'{prefix_snt_idx}{snt_idx}'
  return snt_idx, snt_var

def parse_doc_graph(doc_graph: str) -> Tuple[str, Tuple[str,str,str]]:
  for key, pattern in DOC_GRAPH_PATTERNS.items():
    match = re.search(pattern, doc_graph)
    if match:
      for quadruple in re.findall(DOC_GRAPH_SUBPATTERN, match.group(1)):
        p, r, c = quadruple[1], quadruple[2], quadruple[3]
        if not r.startswith(':'):
          r = f':{r}'
        yield key, (p, r, c)
