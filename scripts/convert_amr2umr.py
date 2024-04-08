#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/14/24 6:43 AM
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import tqdm
from tqdm.contrib import tzip

from data import Document, Example, amr_utils, umr_utils
from structure import AMR2UMRConverter, Alignment, Node, SntGraph
from utils import consts as C, defaults as D, io_utils, regex_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)

# omit generic concepts like perception/posture/sensation/operation verbs
CDLM_AVOID_EVENTS = [
  'have-'
  'do-',
  'say-',
  'believe-',
  'think-',
  'get-',
  'go-',
  'come-',
  'make-',
  'become-',
  'try-',
  'attempt-',
  'know-',
  'understand-',
  'realize-',
  'see-',
  'look-',
  'watch-',
  'hear-',
  'smell-',
  'listen-',
  'feel-',
  'ache-',
  'sit-',
  'stand-',
  'lie-',
  'hang-',
  'work-',
]

# unused
TOTAL_TRIPLES = TOTAL_UNALIGNED = 0

def setup_docs(
        examples: List[Example],
        mapping: Optional[Union[str, Union[Dict[str, str], List[str]]]] = None,
        prefer_int_doc_keys=False,
) -> Dict[Union[int, str], Document]:
  # init UMR Document with a single, global DocGraph per Document (only Root, Author + DCT nodes)
  mapping_is_list = False
  has_mapping = mapping is not None
  if has_mapping:
    if isinstance(mapping, str):
      mapping = io_utils.load_json(mapping)
    if isinstance(mapping, dict):
      # take inverse of snt dict
      mapping = {v:k for k,v in mapping[C.SNT].items()}
    else:
      mapping_is_list = True

  # aggregate by `doc_id`
  snt_idx_mapping = dict()
  tmp_docs = defaultdict(list)  # type: Dict[str, List[Example]]
  for i, example in tqdm.tqdm(enumerate(examples), desc='[Setup Docs]'): # type: int, Example
    key, ref_snt_idx = umr_utils.parse_umr_id(example.doc_snt_id, merge_doc_id=True, int_idx=True)
    if has_mapping:
      key, snt_idx = umr_utils.parse_umr_id(
        mapping[i] if mapping_is_list else mapping[example.doc_snt_id], merge_doc_id=True, int_idx=True)
      snt_idx_mapping[ref_snt_idx] = snt_idx
    tmp_docs[key].append(example)

  # init UMR Document with a single, global DocGraph per Document (only Root, Author + DCT nodes)
  docs = dict()  # type: Dict[Union[int, str], Document]
  for k, v in sorted(tmp_docs.items(), key=lambda x: x[0]): # sort by key (doc id)
    doc = Document(
      doc_id=k,
      examples=sorted(v, key=lambda x: x.snt_idx),
      snt_idx_mapping=snt_idx_mapping,
      init_global_doc_graph=True
    )

    if prefer_int_doc_keys:
      # str (`english_umr-0004`) -> int (4) -> Document
      _, doc_idx = umr_utils.parse_umr_id(k, int_idx=True)
      docs[doc_idx] = doc
    else:
      # str (`english_umr-0004`) -> Document
      docs[k] = doc

  return docs

def integrate_modals(examples, modal_fpath, split_mapping, add_root2author=False):
  global TOTAL_TRIPLES, TOTAL_UNALIGNED

  if not io_utils.exists(modal_fpath):
    logger.info("[!] MDP omitted")
    return

  data_list, snts_list, doc_ids = io_utils.readin_mtdp_tuples(modal_fpath, from_file=True)
  num_modal_docs = len(doc_ids)

  # set up docs
  if num_modal_docs > 5:
    assert split_mapping is not None

    # these splits are currently pre-defined and hard-coded
    if num_modal_docs == 6:
      split_mapping = split_mapping[80]
    else:
      split_mapping = split_mapping[30]
  docs = setup_docs(examples, split_mapping, prefer_int_doc_keys=True)
  num_docs = len(docs)

  assert num_modal_docs == num_docs == len(split_mapping[C.DOC])

  for data in tzip(data_list, snts_list, doc_ids, desc='[MDP Grounding]'):
    modals = data[0]  # type: List[List[str]]
    modal_snts = data[1]  # type: List[str]  # may be empty
    modal_doc_id = data[2]  # type: int

    cur_doc = docs[modal_doc_id]
    cur_doc_graph = cur_doc.doc_graph

    assert len(cur_doc) == len(modal_snts), f"{len(cur_doc)} vs {len(modal_snts)}"

    for modal in modals:  # type: List[str]
      child, modal_type, parent, modal_label = modal
      # for sidx:
      # (a) if int, must subtract by 1 to account for the artificial DATE sentence added at the very top (DEFAULT)
      # (b) if str, can use as is
      child_sidx, child_start, child_end = [int(ch) for ch in child.split('_')]
      parent_sidx, parent_start, parent_end = [int(ch) for ch in parent.split('_')]

      if child_sidx == -3:  # one per doc
        assert modal_type == "Conceiver"
        assert parent_sidx == -1
        assert modal_label == "Depend-on"

        ### this case means nothing; `root` to `author` modal edge is self-evident
        # # duplicate root2author triple not allowed
        # if add_root2author:
        #   cur_doc_graph.add_root2author_modal()
        continue

      else:
        TOTAL_TRIPLES += 1
        assert child_sidx >= 0
        # now child is definitely valid, but parent may not be
        # first, identify child's aligned node
        # child_example = cur_doc[child_sidx]
        child_example = cur_doc.get_ith_example(child_sidx)
        assert child_example.snt_idx > 0
        child_alignment = child_example.alignment

        # get aligned nodes from the span
        child_aligneds = child_alignment.from_span(
          child_start, child_end+1, sort_by_depth=True)

        # child may not have alignment; continue
        if len(child_aligneds) == 0:
          child_toks = child_example.toks
          child_span = " ".join(child_toks[child_start:child_end+1])
          logger.debug("Found a Modal triple (%s) whose child (%s) is unaligned; skipping..",  modal,child_span)
          continue

        # child is aligned to AMR node, select the highest
        child_aligned = child_aligneds[0]  # type: Node

        # now consider parent, which may be abstract
        if parent_sidx == -3: # author
          triple = ( C.AUTHOR, C.MODAL_EDGE_MAPPING[modal_label], child_aligned )
          logger.debug("NEW Modal Triple: %s", str(triple))
          cur_doc_graph.add_modal(triple)

        elif parent_sidx == -5: # null conceiver
          # here the construction is:
          # (a) author -> null-conceiver
          # (b) null-conceiver -> child
          # NOTE: null-conceiver is an abstract node that's local to each sentence
          #   so it has to be manually created
          null_conceiver_node = child_example.snt_graph.add_node(
            label=C.NULL_CONCEIVER,
            snt_idx=child_aligned.snt_idx,
          )
          triple_a = ( C.AUTHOR, C.MODAL_EDGE_MAPPING[modal_label], null_conceiver_node )
          triple_b = ( null_conceiver_node, C.FULL_AFF_EDGE, child_aligned )
          logger.debug("NEW Modal Null-Conceiver Triples: %s + %s", str(triple_a), str(triple_b))
          cur_doc_graph.add_modal(triple_a)
          cur_doc_graph.add_modal(triple_b)

        else:
          assert parent_sidx >= 0
          # here set up parent which must be valid
          # parent_example = cur_doc[parent_sidx]
          parent_example = cur_doc.get_ith_example(parent_sidx)
          assert parent_example.snt_idx > 0
          parent_alignment = parent_example.alignment

          # get aligned nodes from the span
          parent_aligneds = parent_alignment.from_span(
            parent_start, parent_end + 1, sort_by_depth=True)

          # parent may not have alignment; continue
          if len(parent_aligneds) == 0:
            parent_toks = parent_example.toks
            parent_span = " ".join(parent_toks[child_start:child_end + 1])
            logger.warning("Found a Modal triple (%s) whose parent (`%s`) is unaligned; skipping..", modal, parent_span)
            continue

          # parent is aligned to AMR node, select the highest
          parent_aligned = parent_aligneds[0]  # type: Node

          # ever the same? shouldn't be possible in theory but in practice, could happen due to misalignment
          if child_aligned == parent_aligned:
            logger.warning("Child and Parent are aligned to same node; continue..")
            continue

          # now finally add triple
          if parent_aligned.snt_idx > child_aligned.snt_idx:
            continue
          triple = ( parent_aligned, C.MODAL_EDGE_MAPPING[modal_label], child_aligned )
          logger.debug("NEW Modal Triple: %s", str(triple))
          cur_doc_graph.add_modal(triple)

  # distribute mdp triples to local doc graphs
  for doc in docs.values():
    doc.distribute_global_doc_graph_triples(add_root2author_modal=add_root2author)

def integrate_temporals(examples, temporal_fpath, split_mapping):
  global TOTAL_TRIPLES, TOTAL_UNALIGNED

  # -1 accounts for the DCT, which was artificially introduced during preprocessing
  # for some document inputs
  if not io_utils.exists(temporal_fpath):
    logger.info("[!] TDP omitted")
    return

  data_list, snts_list, doc_ids = io_utils.readin_mtdp_tuples(temporal_fpath, from_file=True)
  num_temporal_docs = len(doc_ids)

  # set up docs
  if num_temporal_docs > 5:
    assert split_mapping is not None

    # these splits are currently pre-defined and hard-coded
    if num_temporal_docs == 6:
      split_mapping = split_mapping[80]
    else:
      split_mapping = split_mapping[30]
  docs = setup_docs(examples, split_mapping, prefer_int_doc_keys=True)
  num_docs = len(docs)

  assert num_temporal_docs == num_docs == len(split_mapping[C.DOC])

  canonical_fname = io_utils.get_canonical_fname(temporal_fpath, depth=2)
  has_dct = 'dct' in canonical_fname or 'thyme_tdg' in canonical_fname
  for data in tzip(data_list, snts_list, doc_ids, desc='[TDP Grounding]'):
    temporals = data[0]  # type: List[List[str]]
    temporal_snts = data[1]  # type: List[str]
    temporal_doc_id = data[2]  # type: int

    cur_doc = docs[temporal_doc_id]
    cur_doc_graph = cur_doc.doc_graph
    num_cur_doc_snts = len(cur_doc)

    # thyme_tdg doesn't include sentences in its outputs
    num_temporal_snts = len(temporal_snts)
    if num_temporal_snts > 0:
      if has_dct:
        assert num_cur_doc_snts == num_temporal_snts - 1
      else:
        assert num_cur_doc_snts == num_temporal_snts

    # first pass -- setup
    temporal_types = {'-1_-1_-1': C.ROOT} # type: Dict[str, str]
    temporal_edges = defaultdict(list) # type: Dict[str, List[Tuple[str,str]]]
    temporal_aligneds = {'-1_-1_-1': C.ROOT, '0_0_3': C.DCT_FULL, '-7_-7_-7': C.DCT_FULL,}
    for temporal in temporals:
      child, temporal_type, parent, temporal_label = temporal
      if temporal_type == C.DCT:
        temporal_type = C.TIMEX
      temporal_types[child] = temporal_type

      # get alignment
      child_sidx, child_start, child_end = [int(ch) for ch in child.split('_')]

      if child_sidx == -7 or (has_dct and child_sidx == 0): # one per doc
        assert temporal_type in ['DCT', 'Timex']
        assert parent.startswith('-1')
        assert temporal_label == 'Depend-on'

        # UMR doesn't annotate temporal triple involving root, so skip
        continue

      if has_dct:
        child_sidx -= 1
      assert child_sidx >= 0

      child_example = cur_doc.get_ith_example(child_sidx)
      child_alignment = child_example.alignment

      # get aligned nodes from the span
      child_aligneds = child_alignment.from_span(
        child_start, child_end+1, sort_by_depth=True)
      num_aligneds = len(child_aligneds)
      if num_aligneds > 0:
        if num_aligneds > 1:
          logger.debug("All Aligneds: %s", str(child_aligneds))
          # 1. prefer `-quantity` or `-entity`
          for child_aligned in child_aligneds[:]:
            child_aligned_label = child_aligned.get_label()
            if child_aligned_label.endswith("-quantity") or child_aligned_label.endswith("-entity"):
              child_aligneds = [child_aligned]
              break
          # 2. sort by depth, which is already the case so nothing to be done
          pass

        if parent in temporal_edges and child in temporal_edges[parent]:
          # avoid direct back and forth
          logger.debug("Avoid direct cycle from TDG")
          continue
        temporal_aligneds[child] = child_aligneds[0]
        temporal_edges[child].append((parent, temporal_label))
      else:
        logger.debug(
          "Found unaligned temporal node: %s (%s), skipping..", child, temporal_type)

    # second pass -- establish triples (child is aligned already)
    for child, parents in sorted(
            temporal_edges.items(), key=lambda x: tuple(map(int, x[0].split('\t')[0].split('_')))):
      child_type = temporal_types[child]
      if child_type in [C.ROOT, C.DCT]:
        continue

      child_aligned = temporal_aligneds[child]
      assert not isinstance(child_aligned, str)

      num_parents = len(parents)
      if num_parents > 1:
        if num_parents > 2:
          # ??? shouldn't reach here
          breakpoint()
        assert child_type == C.EVENT

        # here we are not concerned with alignment
        # * although by defition events can and often do have (1) ref timex and (2)
        #  ref event anchor, UMR annotation is rather inconsistent in this aspect, sometimes
        #  using just one and sometimes both. Event coref. cluster may have an impact
        #  since same event needs not be linked to DCT multiple times
        # * at the end of the day, the temporal graph MUST BE CONNECTED (in theory at least)
        # * BUT alignment CAN get in the way here, but this complicates things too much
        #  so accept the trade-off by always preferring another event anchor first
        #  => why? this is much more in alignment with UMR style of annotations
        parents.sort(key=lambda x: temporal_types[x[0]]) # `Event` E comes before `Timex` T

      for parent in parents:
        target_parent, temporal_label = parent
        target_parent_type = temporal_types[target_parent]
        if target_parent not in temporal_aligneds:
          logger.debug("Found an edge with aligned child but unaligned parent %s (%s), skipping..",
                       target_parent, target_parent_type)
          # continue
          break

        parent_aligned = temporal_aligneds[target_parent]
        if isinstance(parent_aligned, str):
          if parent_aligned == C.ROOT:
            if child_type == C.EVENT:
              # event never depends on root, likely a remnant of baseline-to-temporal prep
              continue
            else:
              # rare cases; change root to DCT
              logger.debug("Mapped parent from `root` to `DCT`")
              parent_aligned = C.DCT_FULL
          assert parent_aligned == C.DCT_FULL
        else:
          if child_aligned == parent_aligned:
            # avoid self-loop, this is likely alignment error
            logger.debug("Avoid self-loop, with child (%s) and parent (%s) aligned to same node `%s`",
                         child, parent, child_aligned)
            # continue
            break

        triple = (parent_aligned, C.TEMPORAL_EDGE_MAPPING[temporal_label], child_aligned)
        logger.debug("NEW Temporal Triple: %s", str(triple))
        cur_doc_graph.add_temporal(triple)

        # break to create only one edge max
        break

  # distribute tdp triples to local doc graphs
  for doc in docs.values():
    doc.distribute_global_doc_graph_triples()

def integrate_cdlm(examples, cdlm_fpath, split_mapping):
  global TOTAL_TRIPLES, TOTAL_UNALIGNED

  if not io_utils.exists(cdlm_fpath):
    logger.info("[!] CDLM omitted")
    return

  # defaultdict with `cluster_id` as keys and values are also defaultdict with `doc_id` as keys and list of (snt_idx, token_idx, token_string) as value
  cdlms, doc_ids = io_utils.load_conll(cdlm_fpath, get_doc_ids=True) # type: Dict[int, Dict[str, List[Tuple[Union[int, str]]]]], List[str]
  num_cdlm_docs = len(doc_ids)

  # set up docs
  if num_cdlm_docs > 5:
    assert split_mapping is not None

    # these splits are currently pre-defined and hard-coded
    if num_cdlm_docs == 6:
      split_mapping = split_mapping[80]
    else:
      split_mapping = split_mapping[30]
  docs = setup_docs(examples, split_mapping)
  num_docs = len(docs)

  assert num_cdlm_docs == num_docs == len(split_mapping[C.DOC])

  for cluster_id, clusters in tqdm.tqdm(cdlms.items(), desc='[CDLM Grounding]'):
    for doc_id, per_doc_clusters in clusters.items():
      if len(per_doc_clusters) < 2:  # skip singletons
        logger.debug("Found a singleton event %s; skipping..", str(per_doc_clusters[0]))
        continue

      cur_doc = docs[doc_id]  # type: Document
      cur_doc_graph = cur_doc.doc_graph

      # identify alignments
      cluster_aligneds = list()  # type: List[Node]
      for snt_id, tok_id, tok in per_doc_clusters:
        TOTAL_TRIPLES += 1

        # cur_example = cur_doc[snt_id-1]
        cur_example = cur_doc.get_ith_example(snt_id)
        cur_snt_graph = cur_example.snt_graph
        cur_alignment = cur_example.alignment
        cur_snt_toks = cur_example.toks

        # sanity check
        assert cur_snt_toks[tok_id] == tok

        # get aligned node
        cur_aligned = cur_alignment[tok_id]  # type: Node

        # event may not have alignment; continue
        if not cur_aligned:
          logger.debug("Found an event (%s) in cluster (%d) which is unaligned; skipping..", tok, cluster_id)
          continue
        else:
          # maybe different events in same doc are aligned to same node; check for duplicate
          edges = cur_snt_graph.get_edges(src=cur_aligned)
          if cur_aligned in cluster_aligneds:
            logger.debug("Found a duplicate event (%s) aligned to Node (%s), which is already part of the aligned cluster", tok, str(cur_aligned))
          # elif any((x.get_label() == C.ASPECT and cur_snt_graph.get_node(x.tgt).get_label() == C.STATE) for x in edges):
          #   logger.debug("Found a STATE event (%s), skip..", x.get_la)
          else:
            cluster_aligneds.append(cur_aligned)

      # this cluster may not have any event which is aligned; continue
      if len(cluster_aligneds) == 0:
        logger.debug("Found a cluster (%s) with no aligned events; skipping..", cluster_id)
        continue

      # all generics?? see `CDLM_AVOID_EVENTS`
      avoid_flags= []
      for aligned in cluster_aligneds:
        aligned_label = aligned.get_label()
        avoid_flags.append(any(aligned_label.startswith(x) for x in CDLM_AVOID_EVENTS))
      if all(avoid_flags):
        logger.debug("Found a cluster with (%s) with only generic events; skipping..", cluster_id)
        continue

      # sort by snt_idx
      prev_aligned = None
      prev_snt_ids = set()
      for cur_aligned in sorted(cluster_aligneds, key=lambda x: x.snt_idx):
        cur_snt_idx = cur_aligned.snt_idx
        if cur_snt_idx in prev_snt_ids:
          logger.debug("This event is part of Sentence who has other canonical event which is part of this cluster; skipping..")
          continue

        if prev_aligned is not None and prev_aligned.snt_idx <= cur_aligned.snt_idx:
          # prev_aligned_label, cur_aligned_label = prev_aligned.get_label(), cur_aligned.get_label()
          # if not regex_utils.is_concept(prev_aligned_label) or not regex_utils.is_concept(cur_aligned_label):
          #   logger.info("Found an entity in CDLM triple (`%s` ; `%s`), skipping..", prev_aligned_label, cur_aligned_label)
          #   continue
          triple = (prev_aligned, C.SAME_EVENT_EDGE, cur_aligned)
          logger.info("NEW CDLM triple: %s", triple)
          cur_doc_graph.add_coref(triple)

        prev_snt_ids.add(cur_snt_idx)
        prev_aligned = cur_aligned

  # distribute cdlm triples to local doc graphs
  for doc in docs.values():
    doc.distribute_global_doc_graph_triples()

def integrate_coref(examples, coref_fpath, mapping):
  global TOTAL_TRIPLES, TOTAL_UNALIGNED

  if not io_utils.exists(coref_fpath):
    logger.info("[!] coref omitted")
    return

  all_coref_clusters = []

  # set up docs; by default, `coref` doesn't need to be split in the pipeline
  docs = setup_docs(examples, mapping)

  corefs = io_utils.load_jsonlines(coref_fpath) # type: List[Dict[str, Union[str, List[...]]]]
  for coref in tqdm.tqdm(corefs, desc='[Coref Grounding]'):
    # remove genre prefix
    doc_id = "_".join(coref[C.DOCUMENT_ID].split('_')[1:])

    # this `sent_ids` is 0-based, but it's also int so it evens out; can use as is
    sent_ids = coref[C.SENT_ID]
    cased_words = coref[C.CASED_WORDS]

    cur_doc = docs[doc_id]
    cur_doc_graph = cur_doc.doc_graph

    # get local token offset
    local_token_offsets = []
    local_token_offset = 0
    prev_snt_id = -1
    for sent_id in sent_ids:
      if sent_id > prev_snt_id:
        prev_snt_id = sent_id
        local_token_offset = 0
      local_token_offsets.append(local_token_offset)
      local_token_offset += 1

    # iterate through each cluster
    for word_cluster, span_cluster in zip(coref[C.WORD_CLUSTERS], coref[C.SPAN_CLUSTERS]):
      cluster_aligneds = list()  # type: List[Node]
      cluster_toks = list() # type: List[str]

      for word_offset, span in zip(word_cluster, span_cluster):
        snt_id = sent_ids[word_offset]
        cur_example = cur_doc.get_ith_example(snt_id)
        cur_alignment = cur_example.alignment
        cur_snt_toks = cur_example.toks

        # these are consecutive ints
        cur_tok_offsets = local_token_offsets[span[0]:span[1]]

        # sanity check
        cur_coref_toks = cased_words[span[0]:span[1]]
        cur_example_toks = [cur_snt_toks[x] for x in cur_tok_offsets]
        assert cur_coref_toks == cur_example_toks
        cur_tok = " ".join(cur_coref_toks)

        # # HANDLE THIS BY CHECKING IF ALIGNED NODE IS CONCEPT OR NOT, DEP PARSING UNRELIABLE
        # cur_dep_graph = cur_example.dep_graph
        # dep_span = cur_tok_offsets[:]
        # # if len(dep_span) == 1:
        # #   dep_span.append(max(dep_span))
        # dep_span = (min(dep_span), max(dep_span)+1)
        # cur_dep_node = cur_dep_graph.get_nodes_from_span(
        #   dep_span, sort_by_depth=True)[0]

        # get aligned nodes from the offsets
        aligneds = cur_alignment.from_span(
          start=min(cur_tok_offsets),
          end=max(cur_tok_offsets)+1,
          sort_by_depth=True
        )

        # entity may not have alignment; continue
        if len(aligneds) == 0:
          logger.debug("Found an Entity (`%s`) which is unaligned; skipping..",  cur_tok)
          continue

        # entity is aligned to AMR node, select the highest
        aligned = aligneds[0]  # type: Node
        if aligned.is_attribute:
          # but cannot be an attribute, so get its parent instead which is guaranteed to be not an attribute
          cur_snt_graph = cur_example.snt_graph
          aligned_edge = cur_snt_graph.get_edges(tgt=aligned)[0]
          aligned = cur_snt_graph.get_node(aligned_edge.src)
          if aligned.get_label() == C.NAME:
            # except if it's a name node, in which case get its parent one more time
            aligned_edge = cur_snt_graph.get_edges(tgt=aligned)[0]
            aligned = cur_snt_graph.get_node(aligned_edge.src)
          assert not aligned.is_attribute
        elif aligned.get_label() == 'date-entity':
          logger.info("Found a date-entity in coref cluster, skipping..")
          continue

        if aligned in cluster_aligneds:
          logger.debug("Found a duplicate entity (`%s`) aligned to Node (%s), which is already part of the aligned cluster", cur_tok, aligned)
        else:
          cluster_aligneds.append(aligned)
          cluster_toks.append(cur_tok)

      # this cluster may not have any event which is aligned; continue
      if len(cluster_aligneds) == 0:
        logger.debug("Found a cluster (%s) with no aligned events; skipping..", f'`{"`, `".join(cluster_toks)}`')
        continue

      # sort by snt_idx
      prev_aligned = None
      prev_snt_ids = set()
      for cur_aligned in sorted(cluster_aligneds, key=lambda x: x.snt_idx):
        cur_snt_idx = cur_aligned.snt_idx
        if cur_snt_idx in prev_snt_ids:
          logger.debug("This entity is part of Sentence who has other canonical entity which is part of this cluster; skipping..")
          continue

        if prev_aligned is not None and prev_aligned.snt_idx <= cur_aligned.snt_idx:
          prev_aligned_label, cur_aligned_label = prev_aligned.get_label(), cur_aligned.get_label()
          if regex_utils.is_concept(prev_aligned_label) or regex_utils.is_concept(cur_aligned_label):
            logger.debug("Found a concept in coref triple (`%s` ; `%s`), skipping..", prev_aligned_label, cur_aligned_label)
            continue
          triple = (prev_aligned, C.SAME_ENTITY_EDGE, cur_aligned)
          logger.info("NEW coref triple: %s", triple)
          cur_doc_graph.add_coref(triple)

        prev_snt_ids.add(cur_snt_idx)
        prev_aligned = cur_aligned

      all_coref_clusters.append(cluster_aligneds)

  # distribute coref triples to local doc graphs
  for doc in docs.values():
    doc.distribute_global_doc_graph_triples()

  return all_coref_clusters

def convert_amr2umr(
        input_fpath,
        aligner_type,
        udpipe_fpath,
        output_dir,
        modal_fpath=None,
        temporal_fpath=None,
        cdlm_fpath=None,
        coref_fpath=None,
        split_mapping_fpath=None,
):
  """AMR2UMR conversion + aggregate doc-level information"""
  global TOTAL_TRIPLES, TOTAL_UNALIGNED

  # AMR graphs
  if aligner_type == 'leamr':
    amrs = amr_utils.load_amr_file(input_fpath, load_leamr=True)
  else:
    amrs = amr_utils.load_amr_file(input_fpath, node_as_alignment=True)

  # UD graphs
  udpipe = io_utils.load_json(udpipe_fpath)

  ################################## AMR2UMR ###################################
  ### Conversion
  converter = AMR2UMRConverter()

  # companion to Snt UMRs
  snt_metas = []

  # original mapping
  snt2doc_mapping = list()

  # # { `doc_id` -> { `snt_id` -> Example } }
  # examples = defaultdict(list)  # type: Dict[str, List[Example]]
  examples = list()

  # assumed that AMR ids correspond to UMR ids
  for i, (amr_meta, amr_graph) in enumerate(tzip(*amrs, desc='[AMR2UMR-Conversion]')):
    doc_snt_id = amr_meta[C.ID]  # `english_umr-0008.28`
    snt2doc_mapping.append(doc_snt_id)

    doc_id, snt_idx = umr_utils.parse_umr_id(
      doc_snt_id, merge_doc_id=True, int_idx=True)

    # sentence; `snt` and `tok` may be (and often are) the same
    snt = amr_meta[C.SNT if C.SNT in amr_meta else C.TOK]
    tok = amr_meta[C.TOK if C.TOK in amr_meta else C.SNT]
    toks = tok.split()

    # init sentence-level structures (cannot init DocGraph at this point)
    amr_snt_graph = SntGraph.init_amr_graph(amr_graph, snt_idx=snt_idx)  # AMR Graph
    dep_snt_graph = SntGraph.init_dep_graph(udpipe[i])  # UD Dep. Tree
    alignment = Alignment(amr_meta, amr_snt_graph, mode=aligner_type)

    # sanity check; depending on the choice of tokenization during prp and for some AMR parsers, this may fail
    # BUT could be manually fixed through postprocessing
    assert len(toks) == len(dep_snt_graph)

    # convert
    umr_snt_graph, umr_snt_modals = \
      converter.convert_amr2umr(amr_snt_graph, alignment, dep_snt_graph)

    # main Example object
    example = Example(
      doc_id=doc_id,
      snt_idx=snt_idx,
      snt=snt,
      toks=toks,
      snt_graph=umr_snt_graph,
      dep_graph=dep_snt_graph,
      alignment=alignment,
    )
    examples.append(example)

    # init per-snt Doc Graph, and maybe adding modal triples if any
    doc_graph = example.init_doc_graph()
    doc_graph.add_modals(*umr_snt_modals)

    # for evaluating Snt UMRs; just keep bare essentials
    snt_metas.append({k: v for k, v in amr_meta.items() if k in [C.ID, C.SNT, C.TOK]})

  ################################## DocGraph ##################################
  # these steps don't change the content of UMR Snt Graph but relies on its alignment
  # to incrementally build a separate, global document graph as a list of triples
  # to be precise, each step inits a global Doc Graph which accumulatees triples
  # that are distributed to each Example (i.e. UMR Snt Graph) upon completion
  split_mapping = None
  if split_mapping_fpath:
    split_mapping = io_utils.load_json(split_mapping_fpath, as_int_keys=True)

  ### a) Modal Dependency Graph Triples
  integrate_modals(examples, modal_fpath, split_mapping, add_root2author=True)

  ### b) Temporal Dependency Graph Triples
  integrate_temporals(examples, temporal_fpath, split_mapping)

  ### c) CDLM :same-event Triples
  integrate_cdlm(examples, cdlm_fpath, split_mapping)

  ### d) Coref :same-entity Triples ; does not require split mapping
  integrate_coref(examples, coref_fpath, snt2doc_mapping)

  ################################### EXPORT ###################################
  # a) UMR Snt Graphs (eval w/ Smatch + AnCast-AMR), avoids overwrites
  snt_umrs_output_fdir = os.path.join(output_dir, C.SNT_UMR_GRAPH)
  os.makedirs(snt_umrs_output_fdir, exist_ok=True)
  snt_umrs_output_fpath = io_utils.get_unique_fpath(
    snt_umrs_output_fdir, f'{C.SNT_UMR_GRAPH}.{C.AMR_TXT}')
  amr_utils.save_amr_corpus(
    meta_list=snt_metas,
    graph_list=[x.snt_graph for x in examples],
    fpath_or_dir=snt_umrs_output_fpath
  )

  # b) UMRs (eval w/ AnCast-UMR), always overwrites
  umr_docs = setup_docs(examples, snt2doc_mapping)
  umr_utils.export_umr_docs(umr_docs, output_dir)

@timer
def main(args):
  """assume input is an AMR parse file and output is a dir"""
  input_fpath = args.input
  assert os.path.exists(input_fpath)

  # always dir
  output_dir, is_dir = io_utils.get_dirname(args.output, mkdir=True, get_is_dir_flag=True)
  assert is_dir
  log_fpath = os.path.join(output_dir, D.CONV_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin AMR2UMR Conversion ===")
  logger.info("Input AMR: %s", input_fpath)
  logger.info("Aligner Type: %s", args.aligner)
  logger.info("UDPipe: %s", args.udpipe)
  logger.info("Output: %s", output_dir)

  logger.info("MDP: %s", args.modal)
  logger.info("TDP: %s", args.temporal)
  logger.info("CDLM: %s", args.cdlm)
  logger.info("coref: %s", args.coref)
  logger.info("Split Mapping: %s", args.split_mapping)

  convert_amr2umr(
    input_fpath=input_fpath,
    aligner_type=args.aligner,
    udpipe_fpath=args.udpipe,
    output_dir=output_dir,
    modal_fpath=args.modal,
    temporal_fpath=args.temporal,
    cdlm_fpath=args.cdlm,
    coref_fpath=args.coref,
    split_mapping_fpath=args.split_mapping,
  )

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument('--aligner', default='leamr', type=str.lower, choices=['leamr', 'ibm'],
                           help='aligner used to align AMR parses to source sentences')
    argparser.add_argument('--udpipe', required=True, help='path to UDPipe output file')

    # optional doc graph inputs
    argparser.add_argument('--modal', help='path to MDP Stage 2 output file')
    argparser.add_argument('--temporal', help='path to TDP Stage 2 output file')
    argparser.add_argument('--cdlm', help='path to CDLM output file')
    argparser.add_argument('--coref', help='path to caw-coref or wl-coref output file')
    argparser.add_argument('--split_mapping', help='path to split mapping file (required if any document was split in the pipeline)')

  main(script_setup(add_args_fn=add_args))
