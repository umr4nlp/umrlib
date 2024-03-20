#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/14/24 6:43 AM
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import tqdm
from tqdm.contrib import tzip

from data import Document, Example, amr_utils, umr_utils
from structure import Alignment, AMR2UMRConverter, DocGraph, Node, SntGraph
from utils import consts as C, defaults as D, io_utils, regex_utils, subprocess_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


CDLM_AVOID_EVENTS = [
  # concepts that start with these
  'say', 'believe', 'think', 'see', 'tell', 'get', 'become', 'begin', 'start',
  'come', 'try', 'attempt', 'eat', 'walk',
]

def integrate_mdp(umr_docs, modal_fpath):
  if not io_utils.exists(modal_fpath):
    logger.info("[!] MDP omitted")
    return

  data_list, snts_list, doc_ids = io_utils.readin_mtdp_tuples(modal_fpath, from_file=True)
  for data in tzip(data_list, snts_list, doc_ids, desc='[2:MDP]'):
    modals = data[0]  # type: List[List[str]]
    modal_snts = data[1]  # type: List[str]  # may be empty
    modal_doc_id = data[2]  # type: int

    cur_umr_doc = umr_docs[modal_doc_id]
    cur_doc_graph = cur_umr_doc.doc_graph

    # -1 accounts for the DCT (which was artificially introduced) at the very beginning
    # assert len(cur_umr_doc) == len(modal_snts)-1
    assert len(cur_umr_doc) == len(modal_snts)
    # assert len(cur_umr_doc) == len(modal_snts), f"{len(cur_umr_doc)} vs {len(modal_snts)}"

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

        # duplicate root2author triple not allowed
        cur_doc_graph.add_root2author_modal()

      else:
        assert child_sidx >= 0
        # now child is definitely valid, but parent may not be
        # first, identify child's aligned node
        # child_example = cur_umr_doc[child_sidx-1]
        child_example = cur_umr_doc[child_sidx]
        child_alignment = child_example.alignment

        # get aligned nodes from the span
        child_aligneds = child_alignment.from_span(child_start, child_end+1, sort_by_depth=True)

        # child may not have alignment; continue
        if len(child_aligneds) == 0:
          child_toks = child_example.toks
          child_span = " ".join(child_toks[child_start:child_end+1])
          logger.warning("Found a Modal triple (%s) whose child (`%s`) is unaligned; skipping..",  modal,child_span)
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
          # parent_example = cur_umr_doc[parent_sidx - 1]
          parent_example = cur_umr_doc[parent_sidx]
          parent_alignment = parent_example.alignment

          # get aligned nodes from the span
          parent_aligneds = parent_alignment.from_span(parent_start, parent_end + 1, sort_by_depth=True)

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

def integrate_tdp(umr_docs, temporal_fpath):
  if not io_utils.exists(temporal_fpath):
    logger.info("[!] TDP omitted")
    return

  data_list, snts_list, doc_ids = io_utils.readin_mtdp_tuples(temporal_fpath, from_file=True)
  for data in tzip(data_list, snts_list, doc_ids, desc='[3:TDP]'):
    temporals = data[0]  # type: List[List[str]]
    temporal_snts = data[1]  # type: List[str]
    temporal_doc_id = data[2]  # type: int

    cur_umr_doc = umr_docs[temporal_doc_id]
    cur_doc_graph = cur_umr_doc.doc_graph

    # -1 accounts for the DCT (which was artificially introduced) at the very beginning
    # assert len(cur_umr_doc) == len(temporal_snts)-1
    assert len(cur_umr_doc) == len(temporal_snts)
    # assert len(cur_umr_doc) == len(temporal_snts)

    for temporal in temporals:  # type: List[str]
      child, temporal_type, parent, temporal_label = temporal
      # for sidx:
      # (a) if int, must subtract by 1 to account for the artificial DATE sentence added at the very top (DEFAULT)
      # (b) if str, can use as is
      child_sidx, child_start, child_end = [int(ch) for ch in child.split('_')]
      parent_sidx, parent_start, parent_end = [int(ch) for ch in parent.split('_')]

      if child_sidx == -7: # one per doc
        assert temporal_type == 'Timex'
        assert parent_sidx == -1
        assert temporal_label == 'Depend-on'

        # UMR doesn't annotate temporal triple involving root, so skip
        continue

      else:
        assert child_sidx >= 0
        # now child is definitely valid, but parent may not be
        # first, identify child's aligned node
        # child_example = cur_umr_doc[child_sidx-1]
        child_example = cur_umr_doc[child_sidx]
        child_alignment = child_example.alignment

        # get aligned nodes from the span
        child_aligneds = child_alignment.from_span(child_start, child_end+1, sort_by_depth=True)

        # child may not have alignment; continue
        if len(child_aligneds) == 0:
          child_toks = child_example.toks
          child_span = " ".join(child_toks[child_start:child_end+1])
          logger.warning("Found a Temporal triple (%s) whose child (`%s`) is unaligned; skipping..",  temporal, child_span)
          continue

        # child is aligned to AMR node, select the highest
        child_aligned = child_aligneds[0]  # type: Node

        # now consider parent, which may be DCT
        if parent_sidx == -7: # DCT
          triple = (C.DCT_FULL, C.TEMPORAL_EDGE_MAPPING[temporal_label], child_aligned)
          logger.debug("NEW Temporal Triple: %s", str(triple))
          cur_doc_graph.add_temporal(triple)

        elif parent_sidx == -1:
          triple = (C.ROOT, C.TEMPORAL_EDGE_MAPPING[temporal_label], child_aligned)
          logger.debug("NEW Temporal Triple: %s", str(triple))
          cur_doc_graph.add_temporal(triple)

        else:
          assert parent_sidx >= 0
          # here set up parent which must be valid
          parent_example = cur_umr_doc[parent_sidx - 1]
          parent_alignment = parent_example.alignment

          # get aligned nodes from the span
          parent_aligneds = parent_alignment.from_span(parent_start, parent_end + 1, sort_by_depth=True)

          # parent may not have alignment; continue
          if len(parent_aligneds) == 0:
            parent_toks = parent_example.toks
            parent_span = " ".join(parent_toks[child_start:child_end + 1])
            logger.warning("Found a Temporal triple (%s) whose parent (`%s`) is unaligned; skipping..", temporal, parent_span)
            continue

          # parent is aligned to AMR node, select the highest
          parent_aligned = parent_aligneds[0]  # type: Node

          # ever the same? shouldn't be possible in theory but in practice, could happen due to misalignment
          if child_aligned == parent_aligned:
            logger.warning("Child and Parent are aligned to same node; continue..")
            continue

          if parent_aligned.snt_idx > child_aligned.snt_idx:
            continue

          # now finally add triple
          triple = ( parent_aligned, C.TEMPORAL_EDGE_MAPPING[temporal_label], child_aligned )
          logger.debug("NEW Temporal Triple: %s", str(triple))
          cur_doc_graph.add_temporal(triple)

def integrate_cdlm(umr_docs, cdlm_fpath):
  if not io_utils.exists(cdlm_fpath):
    logger.info("[!] CDLM omitted")
    return

  # defaultdict with `cluster_id` as keys and values are also defaultdict with `doc_id` as keys and list of (snt_idx, token_idx, token_string) as value
  cdlms = io_utils.readin_conll(cdlm_fpath) # type: Dict[int, Dict[str, List[Tuple[Union[int, str]]]]]
  for cluster_id, clusters in tqdm.tqdm(cdlms.items(), desc='[4:CDLM]'):
    for doc_id, per_doc_clusters in clusters.items():
      if len(per_doc_clusters) < 2:  # skip singletons
        logger.warning("Found a singleton event %s; skipping..", str(per_doc_clusters[0]))
        continue

      cur_umr_doc = umr_docs[doc_id]  # type: Document
      cur_doc_graph = cur_umr_doc.doc_graph

      # identify alignments
      cluster_aligneds = list()  # type: List[Node]
      for snt_id, tok_id, tok in per_doc_clusters:
        cur_example = cur_umr_doc[snt_id-1]
        cur_alignment = cur_example.alignment
        cur_snt_toks = cur_example.toks

        # sanity check
        assert cur_snt_toks[tok_id] == tok

        # get aligned node
        cur_aligned = cur_alignment[tok_id]  # type: Node

        # event may not have alignment; continue
        if not cur_aligned:
          logger.warning("Found an event (%s) in cluster (%d) which is unaligned; skipping..", tok, cluster_id)
        else:
          # maybe different events in same doc are aligned to same node; check for duplicate
          if cur_aligned in cluster_aligneds:
            logger.debug("Found a duplicate event (%s) aligned to Node (%s), which is already part of the aligned cluster", tok, str(cur_aligned))
          else:
            cluster_aligneds.append(cur_aligned)

      # this cluster may not have any event which is aligned; continue
      if len(cluster_aligneds) == 0:
        logger.warning("Found a cluster (%d) with no aligned events; skipping..", cluster_id)
        continue

      # all generics?? see `CDLM_AVOID_EVENTS`
      avoid_flags= []
      for aligned in cluster_aligneds:
        aligned_label = aligned.get_label()
        avoid_flags.append(any(aligned_label.startswith(x) for x in CDLM_AVOID_EVENTS))
      if all(avoid_flags):
        logger.warning("Found a cluster with (%d) with only generic events; skipping..")
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
          triple = (prev_aligned, C.SAME_EVENT_EDGE, cur_aligned)
          logger.debug("NEW CDLM triple: %s", triple)
          cur_doc_graph.add_coref(triple)

        prev_snt_ids.add(cur_snt_idx)
        prev_aligned = cur_aligned

def integrate_coref(umr_docs, coref_fpath):
  if not io_utils.exists(coref_fpath):
    logger.info("[!] coref omitted")
    return

  corefs = io_utils.load_jsonlines(coref_fpath) # type: List[Dict[str, Union[str, List[...]]]]
  for coref in tqdm.tqdm(corefs, desc='[5:coref]'):
    # remove genre prefix
    doc_id = "_".join(coref[C.DOCUMENT_ID].split('_')[1:])

    # this `sent_ids` is 0-based, but it's also int so it evens out; can use as is
    sent_ids = coref[C.SENT_ID]
    cased_words = coref[C.CASED_WORDS]

    cur_umr_doc = umr_docs[doc_id]
    cur_doc_graph = cur_umr_doc.doc_graph

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
        cur_example = cur_umr_doc[snt_id]
        cur_alignment = cur_example.alignment
        cur_snt_toks = cur_example.toks

        # these are consecutive ints
        cur_tok_offsets = local_token_offsets[span[0]:span[1]]

        # sanity check
        cur_coref_toks = cased_words[span[0]:span[1]]
        cur_example_toks = [cur_snt_toks[x] for x in cur_tok_offsets]
        assert cur_coref_toks == cur_example_toks
        cur_tok = " ".join(cur_coref_toks)

        # get aligned nodes from the offsets
        aligneds = cur_alignment.from_span(min(cur_tok_offsets), max(cur_tok_offsets)+1, sort_by_depth=True)

        # entity may not have alignment; continue
        if len(aligneds) == 0:
          logger.warning("Found an Entity (`%s`) which is unaligned; skipping..",  cur_tok)
          continue

        # entity is aligned to AMR node, select the highest
        aligned = aligneds[0]  # type: Node
        if aligned.is_attribute:
          # but cannot be an attribute, so get its parent instead which is guaranteed to be not a constant
          cur_snt_graph = cur_example.snt_graph
          aligned_edge = cur_snt_graph.get_edges(tgt=aligned)[0]
          aligned = cur_snt_graph.get_node(aligned_edge.src)
          if aligned.get_label() == 'name':
            # except if it's a name node, in which case get its parent one more time
            aligned_edge = cur_snt_graph.get_edges(tgt=aligned)[0]
            aligned = cur_snt_graph.get_node(aligned_edge.src)
          assert not aligned.is_attribute

        if aligned in cluster_aligneds:
          logger.debug("Found a duplicate entity (`%s`) aligned to Node (%s), which is already part of the aligned cluster", cur_tok, aligned)
        else:
          cluster_aligneds.append(aligned)
          cluster_toks.append(cur_tok)

      # this cluster may not have any event which is aligned; continue
      if len(cluster_aligneds) == 0:
        logger.warning("Found a cluster (%s) with no aligned events; skipping..", f'`{"`, `".join(cluster_toks)}`')
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
          triple = (prev_aligned, C.SAME_ENTITY_EDGE, cur_aligned)
          logger.debug("NEW coref triple: %s", triple)
          cur_doc_graph.add_coref(triple)

        prev_snt_ids.add(cur_snt_idx)
        prev_aligned = cur_aligned

def convert_amr2umr(
        input_fpath,
        aligner_type,
        doc2snt_mapping_fpath,
        split_mapping_fpath,
        output_dir,
        modal_fpath=None,
        temporal_fpath=None,
        cdlm_fpath=None,
        coref_fpath=None,
        udpipe_fpath=None,
):
  """AMR2UMR conversion + aggregate doc-level information"""
  assert os.path.exists(input_fpath), "AMR parse is required for AMR2UMR conversion"

  # load here first, required anyways
  doc2snt_mapping = io_utils.load_json(doc2snt_mapping_fpath)
  split_mapping = io_utils.load_json(split_mapping_fpath)['80']
  split_mapping_inv = {v:k for k,v in split_mapping[C.SNT].items()}
  udpipe = io_utils.load_json(udpipe_fpath)

  # source AMR graphs
  if aligner_type == C.IBM:
    amrs = amr_utils.load_amr_file(input_fpath, doc2snt_mapping=doc2snt_mapping)
  else:
    amrs = amr_utils.load_amr_file(input_fpath, load_leamr=True)

  ################################## AMR2UMR ###################################
  ### Conversion
  converter = AMR2UMRConverter()

  # { `doc_id` -> { `snt_id` -> Example } }
  examples = defaultdict(list)  # type: Dict[str, List[Example]]
  # assumed that AMR ids correspond to UMR ids
  for i, (amr_meta, amr_graph) in enumerate(tzip(*amrs, desc='[AMR2UMR-Conversion]')):
    # IBM outputs only have `tok`, but with Doc2Snt mapping should also have `id`
    doc_snt_id = amr_meta[C.ID]  # `english_umr-0008.28`

    doc_snt_id_s = doc_snt_id.split('.')  # `english_umr-0008` , `28`
    doc_id = ".".join(doc_snt_id_s[:-1])  # `english_umr-0008`
    snt_id = int(doc_snt_id_s[-1])  # 28

    # sentence, may be the same
    snt = amr_meta[C.SNT if C.SNT in amr_meta else C.TOK]
    tok = amr_meta[C.TOK if C.TOK in amr_meta else C.SNT]
    toks = tok.split()

    # init sentence-level structures (cannot init DocGraph at this point)
    amr_snt_graph = SntGraph.init_amr_graph(amr_graph, snt_idx=snt_id)  # AMR Graph
    dep_snt_graph = SntGraph.init_dep_graph(udpipe[i])  # UD Dep. Tree
    alignment = Alignment(amr_meta, amr_snt_graph, mode=aligner_type)

    # sanity check
    assert len(toks) == len(dep_snt_graph)

    # convert
    umr_snt_graph, urm_snt_modals, = converter(amr_snt_graph, alignment, dep_snt_graph)

    example = Example(
      idx=snt_id,
      snt=snt,
      toks=toks,
      snt_graph=umr_snt_graph,
      dep_graph=dep_snt_graph,
      alignment=alignment,
    )
    examples[doc_id].append(example)

  ### now init UMR Document with a single, global DocGraph per Document (only Root, Author + DCT nodes)
  docs = dict()  # type: Dict[Union[int, str], Document]
  for k, v in tqdm.tqdm(sorted(examples.items(), key=lambda x: x[0]), desc='[1b:Init Documents]'):
    # str (`english_umr-0004`) -> Document
    docs[k] = doc = Document(k, sorted(v, key=lambda x: x.idx), init_global_doc_graph=True)
    # int (`4`) -> Document
    docs[doc.id] = doc

  ################################## DocGraph ##################################
  # these steps don't change the content of AMR but relies on its alignment to  the source sentence
  # to incrementally build a separate, global document graph (just a list of triples, really)
  ### MDP
  integrate_mdp(docs, modal_fpath)

  ### TDP
  integrate_tdp(docs, temporal_fpath)

  ### CDLM (events only)
  integrate_cdlm(docs, cdlm_fpath)

  ### coref (mostly pronoun resolution)
  integrate_coref(docs, coref_fpath)
  ##############################################################################

  # ### (optional) Merge splits
  # if split_mapping_fpath is not None:
  umr_docs = umr_utils.merge_umr_docs(docs, split_mapping_fpath)

  ### 7. from per-Doc DocGraph to per-Snt DocGraph
  umr_utils.prepare_per_snt_doc_graph(umr_docs, add_snt_prefix_to_vars=True)

  ### 8. export
  umr_utils.export_umr_docs(umr_docs, output_dir)

@timer
def main(args):
  """assume input is an AMR parse file and output is a dir"""
  input_fpath = args.input
  assert os.path.exists(input_fpath)

  # always dir
  output_dir = io_utils.get_dirname(args.output, mkdir=True)
  log_fpath = os.path.join(output_dir, D.CONV_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin AMR2UMR Conversion ===")
  logger.info("Input AMR: %s", input_fpath)
  logger.info("Aligner Type: %s", args.aligner)
  logger.info("Doc2Snt Mapping: %s", args.doc2snt_mapping)
  logger.info("Split Mapping: %s", args.split_mapping)

  logger.info("MDP: %s", args.modal)
  logger.info("TDP: %s", args.temporal)
  logger.info("CDLM: %s", args.cdlm)
  logger.info("coref: %s", args.coref)

  logger.info("UDPipe: %s", args.udpipe)
  logger.info("Output: %s", output_dir)

  convert_amr2umr(
    input_fpath=input_fpath,
    aligner_type=args.aligner,
    doc2snt_mapping_fpath=args.doc2snt_mapping,
    split_mapping_fpath=args.split_mapping,
    output_dir=output_dir,
    modal_fpath=args.modal,
    temporal_fpath=args.temporal,
    cdlm_fpath=args.cdlm,
    coref_fpath=args.coref,
    udpipe_fpath=args.udpipe,
  )

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    # in addition to `input` and `output`:
    argparser.add_argument('--aligner', default='leamr', type=str.lower, choices=['leamr', 'ibm'],
                           help='aligner used to align AMR parses to source sentences')
    argparser.add_argument('--doc2snt_mapping', help='path to doc2snt mapping file (required for IBM-style AMR parse)')
    argparser.add_argument('--split_mapping', help='path to split mapping file (required if any document was split in the pipeline)')

    argparser.add_argument('--modal', help='path to MDP Stage 2 output file')
    argparser.add_argument('--temporal', help='path to TDP Stage 2 output file')
    argparser.add_argument('--cdlm', help='path to CDLM output file')
    argparser.add_argument('--coref', help='path to caw-coref or wl-coref output file')

    argparser.add_argument('--udpipe', help='path to UDPipe output file')

  main(script_setup(add_args_fn=add_args))
