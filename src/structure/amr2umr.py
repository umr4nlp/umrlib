#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/15/24 10:45 PM
"""AMR2UMR Converter

based on https://aclanthology.org/2023.tlt-1.8/ and
  https://github.com/umr4nlp/umr-guidelines/blob/master/guidelines.md
"""
import logging
from copy import deepcopy
from typing import List, Tuple, Union

from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from structure import graph_utils
from structure.alignment import Alignment
from structure.components import Edge, Node
from structure.snt_graph import SntGraph
from utils import consts as C, resources as R, regex_utils

logger = logging.getLogger(__name__)


class AMR2UMRConverter:

  LMTZR = WordNetLemmatizer()
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
  REPORTING_CONCEPTS = ['say-01', 'report-01']
  # 1) posture verbs (sit, stand, lie, hang)
  # 2) perception verbs (see/look at, watch, hear/listen to, feel)
  # 3) some sensation verbs (ache), mental activity verbs (think, understand)
  # 4) verbs of operation/function (work in This washing machine works/is working)
  # For the UMR annotation, inactive actions in all constructions are annotated as state.
  # from https://github.com/umr4nlp/umr-guidelines/blob/master/guidelines.md#part-3-3-1-3-state
  INACTIVE_CONCEPTS = [
    'sit-01',
    'stand-01',
    'lie-07',
    'hang-01',
    'see-01',
    'look-01',
    'watch-01',
    'hear-01',
    'listen-01',
    'feel-01',
    'smell-01',
    'ache-01',
    'think-01',
    'understand-01',
    'know-01',
    'believe-01',
  ]
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
  }
  ENTITY_MAPPING = {
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
    logger.debug(
      "Re-labeled `%s` from `%s` to `%s`", component, org_label, new_label)

  @staticmethod
  def add_attr_edge(
          graph: SntGraph,
          src_node: Node,
          edge_label: str,
          tgt_node_label: str,
  ):
    tgt_attr = graph.add_node(label=tgt_node_label, is_attribute=True, snt_idx=graph.idx)
    edge = graph.add_edge(src=src_node, tgt=tgt_attr, label=edge_label)
    logger.debug(
      "Added edge `%s` from `%s` to Attribute `%s`", edge, src_node, tgt_attr)
    return tgt_attr, edge

  def handle_pronouns(self, graph, node, node_label):
    self.set_component_label(node, C.PERSON)
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
        has_of_suffix = regex_utils.has_of_suffix(edge_label)
        if has_of_suffix:
          edge_label = regex_utils.invert_edge_label(edge_label)
        edge_label = f'ARG{int(edge_label[-1]) + offset}'
        if has_of_suffix:
          edge_label = regex_utils.invert_edge_label(edge_label)
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

      if node.is_attribute or node_label == C.NAME:
        # do nothing cases
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
            # if :ARG1-of have-cause-91 :ARG0 -> shorthand for :cause
            incoming_edges = amr_snt_graph.get_edges(tgt=node)
            outgoing_edges = amr_snt_graph.get_edges(src=node)
            if len(incoming_edges)==1 and incoming_edges[0].get_label() == 'ARG1-of' and \
                    len(outgoing_edges)==1 and outgoing_edges[0].get_label() == 'ARG0':
              incoming_edge = incoming_edges[0]
              amr_snt_graph.remove_edge(incoming_edge)
              amr_snt_graph.reroute_edge(
                outgoing_edges[0], new_src=incoming_edge.src, new_label="reason")
              amr_snt_graph.remove_node(node)
              continue
            else:
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

          elif node_label == 'last-01':
            self.set_component_label(node, 'have-duration-91')

          elif node_label == 'same-01':
            self.set_component_label(node, 'identity-91')
            aspect_label = C.STATE

          elif node_label in self.INACTIVE_CONCEPTS:
            # `passive` action
            aspect_label = C.STATE

          # special treatment for reporting verbas (says nothing about aspect here)
          elif node_label in self.REPORTING_CONCEPTS:
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

          # aspect label still missing but the node fortunately is aligned
          if not aspect_label and alignment.is_aligned_amr2tok(node):
            tok_range = alignment.amr2tok[node.idx]
            dep_nodes = dep_snt_graph.get_nodes_from_span(tok_range, sort_by_depth=True)

            # if concept is aligned to a noun form, prefer `process`
            if len(dep_nodes) > 0:
              dep_node = dep_nodes[0]
              dep_node_label = dep_node.get_label()
              dep_node_feats = dep_node.feats
              dep_node_pos = dep_node.pos
              try:
                dep_edge = dep_snt_graph.get_edges(tgt=dep_node)[0]
                dep_rel_label = dep_edge.get_label()
              except IndexError:
                dep_edge = None
                dep_rel_label = None

              node_label_pred = "-".join(node_label.split('-')[:-1])
              if dep_node_pos == 'NOUN':
                ### hard to heuristically get right
                if node_label_pred in R.AMR_MORPH_VERB2NOUN and R.AMR_MORPH_VERB2NOUN[node_label_pred] == dep_node_label:
                  aspect_label = C.PROCESS
                elif dep_rel_label == 'compound':
                  aspect_label = C.HABITUAL

              elif dep_node_pos == 'ADJ':
                aspect_label = C.STATE
              elif dep_node_pos == 'VERB':
                tense = dep_node_feats.get('Tense', 'Pres')
                verb_form = dep_node_feats.get('VerbForm', 'Fin')
                if tense == 'Past':
                  if verb_form == 'Part':
                    try:
                      prev_dep_node = dep_snt_graph.get_node(dep_node.idx-1)
                      if prev_dep_node.pos in ['VERB', 'AUX']:
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
                  elif verb_form == 'Inf':
                    aspect_label = C.PERFORMANCE
                  # elif verb_form == 'Fin':
                  #   aspect_label = C.ENDEAVOR
                  # else:
                    # aspect_label = C.PERFORMANCE

              else:# ??? do nothing, not even add the default aspect
                # print(node, dep_node)
                pass

          else:
            # what can be said about non-aligned concepts>? do nothing
            pass

        if aspect_label:
          self.add_attr_edge(
            amr_snt_graph, src_node=node, edge_label=C.ASPECT_EDGE, tgt_node_label=aspect_label)

      ### non-concept nominals
      else:
        # need to be mapped?
        if node_label in self.NODE_MAPPING:
          self.set_component_label(node, self.NODE_MAPPING[node_label])

        elif node_label in self.ENTITY_MAPPING:
          # has to have `name` edge
          rename_flag = False
          for edge in amr_snt_graph.get_edges(src=node):
            if edge.get_label() == C.NAME:
              rename_flag = True
              break
          if rename_flag:
            self.set_component_label(node, self.ENTITY_MAPPING[node_label])

        # check for plurality
        if alignment.is_aligned_amr2tok(node) and node_label != 'date-entity' and not node_label.endswith('-quantity'):
          tok_range = alignment.amr2tok[node.idx]
          dep_nodes = dep_snt_graph.get_nodes_from_span(tok_range, sort_by_depth=True)

          if len(dep_nodes) > 0:
            # always prefer the highest node
            dep_node = dep_nodes[0]
            try:
              dep_edge = dep_snt_graph.get_edges(tgt=dep_node)[0]
              dep_rel_label = dep_edge.get_label()
            except IndexError:
              dep_edge = None
              dep_rel_label = None

            dep_node_feats = dep_node.feats
            if dep_node.pos == 'NOUN' and 'Number' in dep_node_feats:
              skip_flag = False
              # skip if a neighboring edge label is in..
              for edge in amr_snt_graph.get_edges(src=node):
                if edge.get_label() in [C.REF_NUMBER, C.NAME, C.WIKI, 'quant']:
                  skip_flag = True
                  break
              if not skip_flag:
                for edge in amr_snt_graph.get_edges(tgt=node):
                  if edge.get_label() in ['unit', 'quant']:
                    skip_flag = True
                    break
              if skip_flag or dep_rel_label in ['appos', 'compound', 'fixed', 'flat']:
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
      elif edge_label == ':consist-of':
        # either `:group` or `:material`; default material but look out for membership by using keywords
        node = amr_snt_graph.get_node(edge.tgt)
        node_label = node.get_label()
        node_lemma = self.LMTZR.lemmatize(node_label)

        new_edge = ':material'
        if node_lemma in ['group', 'committee', 'board', 'panel', 'jury', 'council']:
          new_edge = ':group'
        self.set_component_label(edge, new_edge)

      # for date-entity sub-root, remove any `refer-number` plurals attached previously
      if amr_snt_graph.get_node(edge.src).get_label() == 'date-entity':
        for de_edge in amr_snt_graph.get_edges(src=edge.tgt):
          if de_edge.get_label() == C.REF_NUMBER:
            edges_to_prune.append(de_edge)

    for edge in edges_to_prune:
      amr_snt_graph.remove_edge(edge)

    # some in-graph modalities that require node removal
    for node in amr_snt_graph.node_list:
      node_label = node.get_label()

      if node_label in ['obligate-01','possible-01','recommend-01','permit-01','wish-01']:
        mod_str = C.NEUT_AFF_EDGE
        if node_label in ['oblitate-01', 'recommend-01']:
          mod_str = C.PART_AFF_EDGE

        should_flip = False

        # there should be ARG1 edge inlcuded here
        edges = amr_snt_graph.get_edges(src=node)
        edge_labels = [edge.get_label() for edge in edges]
        if "ARG1" not in edge_labels:
          edge_invs = amr_snt_graph.get_edges(tgt=node)
          edge_inv_labels = [edge.get_label() for edge in edge_invs]
          if "ARG1-of" not in edge_inv_labels:
            # neither :ARG1 nor :ARG1-of, most likely a mistake then; just delete
            amr_snt_graph.remove_node(node, remove_all_edges=True)
            continue
          should_flip = True
          edges += edge_invs

        # remove :aspect if any
        for edge in edges[:]:
          if edge.get_label() == 'aspect':
            amr_snt_graph.remove_edge(edge)
            edges.remove(edge)

        canonical_edge = None
        for i, edge in enumerate(edges):
          edge_label = edge.get_label()
          if edge_label == 'ARG1' or (should_flip and edge_label == 'ARG1-of'):
            canonical_edge = edge
            break
          elif edge_label == 'polarity':
            if amr_snt_graph.get_node(edge.tgt).get_label() == '-':
              # make negative
              org_mod_str = mod_str
              mod_str = mod_str.replace('affirmative', 'negative')
              logger.debug("Flipped Modal Strength from `%s` to `%s`", org_mod_str, mod_str)

        if canonical_edge is None:
          # if not found, just skip
          continue

        # isolate ARG1 from the rest
        edges.pop(i)

        child_idx = canonical_edge.src if should_flip else canonical_edge.tgt
        canonical_child = amr_snt_graph.get_node(child_idx)
        if amr_snt_graph.is_root(node):
          amr_snt_graph.set_root(canonical_child)
        else:
          if should_flip:
            for incoming_edge in amr_snt_graph.get_edges(src=node):
              amr_snt_graph.reroute_edge(incoming_edge, new_src=canonical_child)
          else:
            for incoming_edge in amr_snt_graph.get_edges(tgt=node):
              amr_snt_graph.reroute_edge(incoming_edge, new_tgt=canonical_child)

        for edge in edges:
          if edge.src == node.idx:
            amr_snt_graph.reroute_edge(edge, new_src=canonical_child)
          else:
            amr_snt_graph.reroute_edge(edge, new_tgt=canonical_child)

        # by now, `node` should be disconnected and safe to remove
        amr_snt_graph.remove_node(node, remove_all_edges=True)

        # TODO:
        triple = ( C.AUTHOR, mod_str, canonical_child )
        modals.append(triple)

    # UMR being specifically against `and :op2` constructions for `And ...` sentences
    root = amr_snt_graph.root_node
    root_label = root.get_label()
    root_edges = amr_snt_graph.get_edges(src=root)
    # if root_label == 'and' and (len(root_edges) ==1 and root_edges[0].get_label() == 'op2'):
    if root_label == 'and':
      root_edge_labels = [x.get_label() for x in root_edges]
      if 'op2' in root_edge_labels and not 'op1' in root_edge_labels:
        op2_edge_idx = root_edge_labels.index('op2')
        op2_edge = root_edges.pop(op2_edge_idx)

        ### okay to not check
        # # sentence must begin with `and` or `And`
        # dep_first_node = dep_snt_graph.nodes[0]
        #
        # # remove any impurities
        # dep_first_node_label = "".join(
        #   [x for x in dep_first_node.get_label().lower() if x.isalpha()])
        # assert dep_first_node_label == 'and'

        # just proceed; no need to check for alignment
        canonical_child = amr_snt_graph.get_node(op2_edge.tgt)
        amr_snt_graph.set_root(canonical_child)
        amr_snt_graph.remove_edge(op2_edge)
        for edge in root_edges:
          tgt_node = amr_snt_graph.get_node(edge.tgt)
          if tgt_node.is_attribute:
            amr_snt_graph.remove_edge(edge)
          else:
            arg_edges = amr_snt_graph.get_edges(src=canonical_child)
            arg_edge_labels = [regex_utils.normalize_edge_lable(x.get_label()) for x in arg_edges]
            # simple fallback on some remaining ARGX
            for i in range(5):
              new_edge_label = f'ARG{i}'
              if new_edge_label not in arg_edge_labels:
                amr_snt_graph.reroute_edge(edge, new_src=canonical_child, new_label=new_edge_label)
                break
        amr_snt_graph.remove_node(root, remove_all_edges=False)

    elif root_label == 'multi-sentence':
      # UMR has no `multi-sentence` construct; then simply choose the subgraph with more nodes!
      # here `dfs` doesn't cleanly handle reentrancies, but should suffice for UMR v1.0
      root_edge_sg_size_dict = dict()
      for i, root_edge in enumerate(root_edges[:]):
        if not root_edge.get_label().startswith('snt'):
          amr_snt_graph.remove_node(root_edge.tgt, remove_all_edges=True)
          continue
        dfs_nodes, _ = graph_utils.dfs(amr_snt_graph, root_edge.tgt)
        root_edge_sg_size_dict[i] = len(dfs_nodes)
      root_edge_sg_size_dict_sorted = sorted(
        root_edge_sg_size_dict.items(), key=lambda x: x[1], reverse=True)

      # make largest subgraph new root
      root_edge_largest_sg = root_edges[root_edge_sg_size_dict_sorted.pop(0)[0]]
      largest_sg_child = amr_snt_graph.get_node(root_edge_largest_sg.tgt)

      for edge_idx, _ in root_edge_sg_size_dict_sorted:
        root_edge = root_edges[edge_idx]
        bfs_nodes, _ = graph_utils.bfs(amr_snt_graph, root_edge.tgt)
        for bfs_node in reversed(bfs_nodes):
          amr_snt_graph.remove_node(bfs_node, remove_all_edges=True)
        amr_snt_graph.remove_edge(root_edge)
      amr_snt_graph.set_root(largest_sg_child)
      amr_snt_graph.remove_edge(root_edge_largest_sg)
      amr_snt_graph.remove_node(root, remove_all_edges=True)

    logger.debug("Before:\n%s", ref_snt_graph)
    logger.debug("After:\n%s", amr_snt_graph)

    # final UMR snt-graph (technically unnecessary since the changes are in-place
    # and any modal triples that arise while handling modal predicates
    return amr_snt_graph, modals
