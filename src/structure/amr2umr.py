#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/15/24 10:45 PM
"""AMR2UMR Converter

based on https://aclanthology.org/2023.tlt-1.8/
"""
import logging
from copy import deepcopy
from typing import List, Tuple, Union

from structure import graph_utils
from structure.alignment import Alignment
from structure.components import Edge, Node
from structure.snt_graph import SntGraph
from utils import consts as C, regex_utils

logger = logging.getLogger(__name__)


class AMR2UMRConverter:

  PRONOUNS = {
    "i": (C.REF_PERSON_1ST, C.REF_NUMBER_SINGULAR),
    "we": (C.REF_PERSON_1ST, C.REF_NUMBER_PLURAL),
    "you": (C.REF_PERSON_2nd, C.REF_NUMBER_SINGULAR),
    "you all": (C.REF_PERSON_2nd, C.REF_NUMBER_PLURAL),
    "she": (C.REF_PERSON_3rd, C.REF_NUMBER_SINGULAR),
    "he": (C.REF_PERSON_3rd, C.REF_NUMBER_SINGULAR),
    "it": (C.REF_PERSON_3rd, C.REF_NUMBER_SINGULAR),
    "they": (C.REF_PERSON_3rd, C.REF_NUMBER_PLURAL)
  }
  CONCEPT_MAPPING = {
    'cite-01': 'cite-91',
    'infer-01': 'infer-91',  ### TODO: use discretion to use :reason where possible
    'mean-01': 'mean-91',
    'resemble-01': 'resemble-91',
  }
  NODE_MAPPING = { # these are simple cases only
    'amr-unknown': 'umr-unknown',
    'amr-choice': 'umr-choice',
    'amr-empty': 'umr-empty',
    'amr-unintelligible': 'umr-unintelligible',

    # just in case
    'location': 'place',

    # UMR removes some entities while adding others; these are hard to predict heuristically
    'political-party': 'political-organization',
    'military': 'armed-organization',
    'school': 'academic-organization',
    'university': 'academic-organization',
    'research-institute': 'academic-organization',
    'team': 'sports-organization',
    'league': 'sports-organization',
    'natural-object': 'plant',
    'program': 'computer-program'
  }
  RELATION_MAPPING = {
    ':accompanier': ':companion',
    ':beneficiary': ':affectee',
    ':cost': ':other-role',
    ':domain': ':mod',
    ':either': ':or',
    ':neither': ':or',
    ':location': ':place',
    ':time': ':temporal',
  }

  def __call__(self, *args, **kwargs):
    return self.convert_amr2umr(*args, **kwargs)

  @staticmethod
  def set_component_label(component: Union[Edge, Node], new_label: str):
    org_label = component.get_label()
    component.set_label(new_label)
    logger.debug("Re-labeled `%s` from `%s` to `%s`", component, org_label, new_label)

  @staticmethod
  def add_attr_edge(
          graph: SntGraph,
          src_node: Node,
          edge_label: str,
          tgt_node_label: str,
  ):
    tgt_attr = graph.add_node(label=tgt_node_label, is_attribute=True)
    edge = graph.add_edge(src=src_node, tgt=tgt_attr, label=edge_label)
    logger.debug("Added edge `%s` from `%s` to Attribute `%s`", edge, src_node, tgt_attr)
    return tgt_attr, edge

  def handle_pronouns(self, graph, node, node_label):
    ref_person_node_label, ref_number_node_label = self.PRONOUNS[node_label]
    self.add_attr_edge(graph, node, C.REF_PERSON_EDGE, ref_person_node_label)
    self.add_attr_edge(graph, node, C.REF_NUMBER_EDGE, ref_number_node_label)
    logger.debug("Mapped Pronoun `%s` to `person` with `%s` and `%s` attrs",
                 node_label, ref_person_node_label, ref_number_node_label)

  def offset_args(self, graph: SntGraph, subroot: Node, offset=1):
    # offset all ARG# by 1
    for edge in graph.get_edges(subroot, undirected=True):
      edge_label = edge.get_label()
      if edge_label.startswith("ARG"):
        has_of_suffix = graph_utils.has_of_suffix(edge_label)
        if has_of_suffix:
          edge_label = graph_utils.invert_edge_label(edge_label)
        edge_label = f'ARG{int(edge_label[-1]) + offset}'
        if has_of_suffix:
          edge_label = graph_utils.invert_edge_label(edge_label)
        self.set_component_label(edge, edge_label)

  def convert_amr2umr(
          self,
          amr_snt_graph: SntGraph,
          alignment: Alignment,
          dep_snt_graph: SntGraph
  ) -> Tuple[SntGraph, List[Tuple[Node, str, Node]]]:
    # since conversion takes place in-place, create a copy of the original for reference
    ref_snt_graph = deepcopy(amr_snt_graph)

    # any modal triples? rare, but still need them
    modals = []

    ################################## NODES ###################################
    for node in amr_snt_graph.node_list:
      node_label = node.get_label()

      if node.is_attribute or node_label in [C.NAME, ]:
        # do nothing
        continue

      ### Pronouns: no need for alignment
      elif node_label in self.PRONOUNS:
        self.handle_pronouns(amr_snt_graph, node, node_label)

      ### fine-grained control for concepts, with big focus on :aspect
      elif regex_utils.is_concept(node_label):
        # not all concepts get aspect annotation, but most do
        # `habitual` and `endeavor` are difficult to determine heuristically
        aspect_label = None

        ### ABSTRACT CONCEPTS
        if node_label.endswith('-91') or node_label.endswith('-92'):
          # some abstract nodes may not require aspect annotation, while others
          # typically prefer `state`

          #  OBJECT PREDICATION
          if node_label in ['have-rel-role-91', 'have-org-role-91']:
            # there is also `have-role-91`; it seems UMR prefers  `have-role-91` over 'have-rel-role-92' ???
            self.set_component_label(node, 'have-role-91' if 'rel' in node_label else node_label.replace('91', '92'))
            # offset all ARG# by 1
            self.offset_args(amr_snt_graph, node, offset=1)

          #  THETIC LOCATION (exist-91) vs PREDICATIVE LOCATION (have-place-91)
          # hard to tell heuristically
          elif node_label == 'be-located-at-91':
            new_label = 'have-place-91'
            # if node.idx in alignment.amr2tok:
            #   tok_range = alignment.amr2tok[node.idx]
            #   dep_nodes = dep_snt_graph.get_nodes_from_span(tok_range, sort_by_depth=True)
            #   if len(dep_nodes) > 0:
            #     dep_node = dep_nodes[0]
            #   breakpoint()
            self.set_component_label(node, new_label)

            for edge in amr_snt_graph.get_edges(src=node):
              if edge.get_label() == 'location':
                self.set_component_label(edge, 'place')

          elif node_label == 'be-temporally-at-91':
            self.set_component_label(node, 'have-temporal-91')
            for edge in amr_snt_graph.get_edges(src=node):
              if edge.get_label() == 'time':
                self.set_component_label(edge, 'temporal')

          elif node_label == 'have-li-91':
            self.set_component_label(node, 'have-list-item-91')
            for edge in amr_snt_graph.get_edges(src=node):
              if edge.get_label() == 'li':
                self.set_component_label(edge, 'list-item')
            aspect_label = C.STATE

          elif node_label == 'be-from-91':
            # `have-material-91` vs `have-source-91` vs `have-start-91`, hard to tell heuristically
            self.set_component_label(node, 'have-material-91')

          elif node_label == 'be-destined-for-91':
            # `have-goal-91` vs `have-recipient-91`, hard to tell heuristically
            self.set_component_label(node, 'have-goal-91')

          elif node_label in [
            'have-mod-91',
            'have-part-91',
            'have-quant-91',
            'have-concession-91',
            'have-quant-91'
            'have-polarity-91'
            'be-polite-91',
            'have-ord-91',
            'have-experience-91',
            'have-medium-91',
            'have-age-91',
            'have-91',
            'have-topic-91',
            'have-duration-91',
            'have-path-91',
            'have-direction-91',
            'have-degree-91',
            'have-degree-92',
            'have-frequency-92',
          ]:  # many of these don't even show up in AMR anyways
            aspect_label = C.STATE

          # else:
          #   # include-91, request-confirmation-91, ...
          #   breakpoint()

        ### CONCEPTS
        else:
          # any renamed roles
          if node_label == 'accompany-01':
            self.set_component_label(node, 'have-companion-91')
            for edge in amr_snt_graph.get_edges(src=node):
              if edge.get_label() == 'accompanier':
                self.set_component_label(edge, 'companion')

          elif node_label == 'age-01':
            self.set_component_label(node, 'have-age-91')
            aspect_label = C.STATE

          #  PREDICATIVE POSSESSION
          elif node_label == 'belong-01':
            self.set_component_label(node, 'belong-91')
            # offset all ARG# by 1
            self.offset_args(amr_snt_graph, node, offset=1)
            aspect_label = C.STATE

          elif node_label == 'benefit-01':
            self.set_component_label(node, 'have-affectee-91')
            for edge in amr_snt_graph.get_edges(src=node):
              if edge.get_label() == 'beneficiary':
                self.set_component_label(edge, 'affectee')

          elif node_label == 'concern-02':
            self.set_component_label(node, 'have-topic-91')

          elif node_label == 'cause-01':
            self.set_component_label(node, 'have-cause-91')

          elif node_label == 'consist-01':
            # don't show up in AMRS
            self.set_component_label(node, 'have-part-91')

          elif node_label == 'except-91':
            self.set_component_label(node, 'have-subtraction-91')

          elif node_label == 'exemplify-01':
            self.set_component_label(node, 'have-example-91')

          #  THETIC POSSESSION
          elif node_label in ['have-03', 'own-01']:
            self.set_component_label(node, 'have-91')
            # offset all ARG# by 1
            self.offset_args(amr_snt_graph, node, offset=1)
            aspect_label = C.STATE

          # elif node_label == 'instead-of-91':
          #   self.set_component_label(node, 'instead-of-91')

          elif node_label == 'last-01':
            self.set_component_label(node, 'have-duration-91')

          elif node_label == 'same-01':
            self.set_component_label(node, 'identity-91')
            aspect_label = C.STATE

          elif node_label == 'say-01':
            arg1_edge = None
            for edge in sorted(amr_snt_graph.get_edges(src=node), key=lambda x: x.get_label()):
              edge_label = edge.get_label()
              if edge_label == 'ARG1':
                # makes cycle with an inverse edge `:quote`
                amr_snt_graph.add_edge(src=edge.tgt, tgt=edge.src, label=':quote')
                arg1_edge = edge

              elif edge_label == 'ARG2':
                assert arg1_edge is not None
                amr_snt_graph.add_edge(src=arg1_edge.tgt, tgt=edge.tgt, label='vocative')

          # may not have been aligned; request this info first
          elif node.idx in alignment.amr2tok:
            tok_range = alignment.amr2tok[node.idx]
            dep_nodes = dep_snt_graph.get_nodes_from_span(tok_range, sort_by_depth=True)

            # if concept is aligned to a noun form, prefer `process`
            if len(dep_nodes) > 0:
              dep_node = dep_nodes[0]
              dep_node_pos = dep_node.pos

              if dep_node_pos == 'NOUN':
                aspect_label = C.PROCESS
              elif dep_node_pos == 'ADJ':
                aspect_label = C.STATE
              elif dep_node_pos == 'VERB':
                dep_node_feats = dep_node.feats

                tense = dep_node_feats.get('Tense', 'Pres')
                verb_form = dep_node_feats.get('VerbForm', 'Fin')
                if tense == 'Past':
                  if verb_form == 'Part':
                    try:
                      prev_dep_node = dep_snt_graph.get_node(dep_node.idx-1)
                      if prev_dep_node.pos == 'VERB':
                        aspect_label = C.PERFORMANCE
                      else:
                        aspect_label = C.STATE
                    except IndexError:
                      aspect_label = C.PERFORMANCE

                  else: ## finite
                  # elif verb_form == 'Fin':
                    aspect_label = C.PERFORMANCE

                # elif tense == 'Pres':
                else: ### {PRESENT
                  if verb_form == 'Ger':
                    aspect_label = C.ACTIVITY
                  else:
                    aspect_label = C.PERFORMANCE

              else:# ??? do nothing, not even add the default aspect
                # print(node, dep_node)
                pass

          else:
            # what can be said about non-aligned concepts>? do nothing
            pass

        if aspect_label:
          self.add_attr_edge(
            amr_snt_graph, src_node=node, edge_label=C.ASPECT_EDGE, tgt_node_label=aspect_label)

      else:
        # need to be mapped?
        if node_label in self.NODE_MAPPING:
          self.set_component_label(node, self.NODE_MAPPING[node_label])

        # check for plurality
        if node.idx in alignment.amr2tok and node_label != 'date-entity' and not node_label.endswith('-quantity'):
          tok_range = alignment.amr2tok[node.idx]
          dep_nodes = dep_snt_graph.get_nodes_from_span(tok_range, sort_by_depth=True)

          if len(dep_nodes) > 0:
            # always prefer the highest node
            dep_node = dep_nodes[0]

            dep_node_feats = dep_node.feats
            if dep_node.pos == 'NOUN' and 'Number' in dep_node_feats:
              skip_flag = False
              for edge in amr_snt_graph.get_edges(src=node):
                if edge.get_label() in [C.REF_NUMBER, C.NAME, C.WIKI]:
                  skip_flag = True
                  break
              if skip_flag:
                continue
              label = C.REF_NUMBER_SINGULAR if 'Sing' in dep_node_feats['Number'] else C.REF_NUMBER_PLURAL
              self.add_attr_edge(
                amr_snt_graph, src_node=node, edge_label=C.REF_NUMBER_EDGE, tgt_node_label=label)

    # now consider edges, pruning some while re-structuring others
    edges_to_prune = []
    for edge in amr_snt_graph.edges:
      edge_label = edge.get_label(decorate=True)
      if edge_label in self.RELATION_MAPPING:
        self.set_component_label(edge, self.RELATION_MAPPING[edge_label])

      # for date-entity sub-root, remove any `refer-number` plurals attached previously
      if amr_snt_graph.get_node(edge.src).get_label() == 'date-entity':
        for de_edge in amr_snt_graph.get_edges(src=edge.tgt):
          if de_edge.get_label() == C.REF_NUMBER:
            edges_to_prune.append(de_edge)

    for edge in edges_to_prune:
      amr_snt_graph.remove_edge(edge)

    # finally some in-graph modalities that require node removal
    for node in amr_snt_graph.node_list:
      node_label = node.get_label()

      if node_label in ['obligate-01','possible-01','recommend-01','permit-01','wish-01']:
        # there should be ARG1 edge inlcuded here
        edges = amr_snt_graph.get_edges(src=node)

        if node_label in ['oblitate-01', 'recommend-01']:
          mod_str = C.PART_AFF_EDGE
        else:
          mod_str = C.NEUT_AFF_EDGE

        canonical_edge = None
        for i, edge in enumerate(edges):
          if edge.get_label() == 'ARG1':
            canonical_edge = edge
            break
          elif edge.get_label() == 'polarity':
            if amr_snt_graph.get_node(edge.tgt).get_label() == '-':
              # make negative
              mod_str.replace('affirmative', 'negative')

        if canonical_edge is None:
          # if not found, just skip
          continue

        # isolate ARG1 from the rest
        edges.pop(i)

        canonical_child = amr_snt_graph.get_node(canonical_edge.tgt)
        if amr_snt_graph.is_root(node):
          amr_snt_graph.set_root(canonical_child)
        else:
          for incoming_edge in amr_snt_graph.get_edges(tgt=node):
            amr_snt_graph.reroute_edge(incoming_edge, new_tgt=canonical_child)

        for edge in edges:
          if edge.src == node.idx:
            amr_snt_graph.reroute_edge(edge, new_src=canonical_child)
          else:
            amr_snt_graph.reroute_edge(edge, new_tgt=canonical_child)

        # by now, `node` should be disconnected and safe to remove
        amr_snt_graph.remove_node(node)

        modals.append( (C.AUTHOR, mod_str, canonical_child) )

    logger.debug("Before:\n%s", ref_snt_graph)
    logger.debug("After:\n%s", amr_snt_graph)

    # final UMR snt-graph (technically unnecessary since the changes are in-place
    # and any modal triples that arise while handling modal predicates

    return amr_snt_graph, modals
