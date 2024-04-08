#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/12/24 5:45 AM
"""single entry point for preprocessing UMR v1.0 English

1) Cleanup
cleanup is first applied on textual level, without requiring any parsing in order to init SntGraph objs
BUT with high cleanup aggression, SntGraph is necessary for more detailed structural analysis,
which carries the inherent risk of altering content in the process (perhaps inadvertently)
=> exporting the SntGraphs without any fix yields 100% on the UMR Evaluation, SO SAFE TO USE

2) Pruning (spcific to the English portion)
all newswire sentences in 0001, 0003, and 0005 are taken straight from AMR R3
* so we prepare a smaller sub-folder containing 0002 and 0004 only

3) Split
0004 is too long (~140 sentences) for some models in our pipeline
* so it gets split into multiple fragments, each of which contains at most `max_per_doc` sentences
  * usually 30, but can go up to ~80 if `max_length` is 384
  * how to execute this partition depends on `--partition-strategy`, where
    1) `greedy`: early on, grab `max_per_doc` whenever possible; the last partition may contain fewer sentences
    2) `even`: try to distribute as evenly across all partitions as possible

4) Inputs Prep
here we prepare inputs that need to be consumed by the models in our pipeline
* Sapienza models (LeakDistill + Spring) require AMR format, whereas IBM simply needs sentences per line
* MDP + TDP has a specific input data format that needs to be followed
* CDLM inputs depends on stage1 output(s) from MDP and/or TDP
* caw-coref (and wl-coref) is a list of jsons, stored as a jsonlines
* naturally produced metadata include:
  1) split_mapping: in order to place together what has been split into fragments
  2) doc2snt_mapping: IBM parses only has ::tok as metadata, requiring an explicit mapping to inject ID back

"""
import logging
import os
from typing import Dict, List, Optional

import tqdm

from data import Document, Example, umr_utils
from utils import consts as C, defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def fix_umr_docs(input_dir, output_dir, aggression: int = 1):
  """fixes a single UMR file

  aggression: int representing aggressiveness when fixing annotation errors
    * 1: trivial annotation mistakes
    * 2: enforce labeling consistency

  """
  out_fpaths = []
  for fname in tqdm.tqdm(os.listdir(input_dir), desc='[Preprocessing]'):
    if fname.endswith('.txt'):
      logger.info("=> Preprocessing `%s`", fname)
      input_fpath = os.path.join(input_dir, fname)

      raw_text = io_utils.load_txt(input_fpath)

      if aggression == 0: # just copy
        out_fpath = io_utils.save_txt(raw_text, output_dir, fname=fname)
        out_fpaths.append(out_fpath)
        continue

      raw_text_s = raw_text.split('\n')

      ### text level cleanup
      if '0003' in fname:
        # fix some constant values not having closing quotation marks
        raw_text_s[45] = raw_text_s[45].replace('"Russia', '"Russia"')
        raw_text_s[300] = raw_text_s[300].replace('"cancer', '"cancer"')
        raw_text_s[308] = raw_text_s[308].replace('"Russia', '"Russia"')
        raw_text = "\n".join(raw_text_s)

      elif '0004' in fname:
        raw_text = raw_text.replace('# :: snt1155', '# :: snt115')

      elif '0005' in fname :
        assert raw_text_s.pop(320) == 'g'

        # these two don't matter in terms of internal representation
        raw_text_s[618] = f'{raw_text_s[618]})'
        raw_text_s[1715] = raw_text_s[1715].replace('(s5s0', '(s27s0')
        raw_text_s[1765] = raw_text_s[1765].replace('(s29s0', '(s28s0')
        raw_text_s[1773] = raw_text_s[1773][:raw_text_s[1773].rindex(")")+1]
        raw_text_s[1774] = raw_text_s[1774].replace(" marsha", "")
        raw_text_s[1804] = raw_text_s[1804][:raw_text_s[1804].rindex(")")+1]
        raw_text_s[1805] = raw_text_s[1805][:raw_text_s[1805].rindex(")")+1]
        raw_text = "\n".join(raw_text_s)

      # occurs in more than single file
      if aggression > 1:
        raw_text = raw_text.replace(':FULLfull-affirmative', C.FULL_AFF_EDGE)
        raw_text = raw_text.replace(":AFF", C.FULL_AFF_EDGE)
        raw_text = raw_text.replace(':UNSP', C.UNSP_EDGE)
        raw_text = raw_text.replace(':NEG', C.FULL_NEG_EDGE)

      # map `:modstr` to `:modal-strength`
      raw_text = raw_text.replace(':modstr', C.MODAL_STRENGTH_EDGE)

      ######################################################################################################################
      # if aggresive, init graphs for closer analysis
      if aggression > 1:
        # use penman
        examples = umr_utils.load_umr_file_aux(raw_text, init_graphs=True)

        # fix in-place then convert to str
        out = []
        for example in examples:
          doc_graph, snt_graph = example.doc_graph, example.snt_graph

          for edge in snt_graph.edges:
            edge_label = edge.get_label(decorate=True)

            tgt_node = snt_graph.get_node(edge.tgt)
            tgt_node_label = tgt_node.get_label()

            if edge_label == C.ASPECT_EDGE:
              if tgt_node_label == C.PART_AFF: # 'english_umr-0001.23'
                tgt_node.set_label(C.ACTIVITY)
            elif edge_label == C.REF_NUMBER_EDGE: # 'english_umr-0004.97'
              if tgt_node_label == C.REF_PERSON_3rd:
                edge.set_label(C.REF_PERSON_EDGE)
            elif edge_label == C.REF_PERSON_EDGE:
              if tgt_node_label == C.REF_NUMBER_SINGULAR:  # 'english_umr-0005.8'
                edge.set_label(C.REF_NUMBER_EDGE)
              elif tgt_node_label == C.REF_NUMBER_PLURAL:  # 'english_umr-0005.10'
                edge.set_label(C.REF_NUMBER_EDGE)

          # move modal annotations that are part of Sentence-Graph to Document Graph
          for node in snt_graph.node_list:
            node_edges = snt_graph.get_edges(src=node)

            # by default, parent is author UNLESS there is another is a sister `:modal-predicate` edge
            modpred_edge = modstr_edge = None
            for node_edge in node_edges:
              node_edge_label = node_edge.get_label()
              if node_edge_label == C.MODAL_PREDICATE:
                modpred_edge = node_edge
              elif node_edge_label == C.MODAL_STRENGTH:
                modstr_edge = node_edge

            has_modpred_edge = modpred_edge is not None
            has_modstr_edge = modstr_edge is not None
            if has_modpred_edge and has_modstr_edge:
              snt_graph.remove_edge(modstr_edge)
              snt_graph.remove_edge(modpred_edge)
              modal_strength = snt_graph.get_node(modstr_edge.tgt).get_label()
              assert modal_strength in [C.FULL_AFF, C.FULL_NEG, C.PART_AFF, C.PART_NEG, C.NEUT_AFF, C.NEUT_NEG]
              modal_triple = (snt_graph.get_node(modpred_edge.tgt), f':{modal_strength}', snt_graph.get_node(modpred_edge.src))
              doc_graph.add_modal(modal_triple)
              logger.info("Moved Snt edges `%s` and `%s` to DocGraph as `%s`", modstr_edge, modpred_edge, modal_triple)
            elif has_modstr_edge:
              snt_graph.remove_edge(modstr_edge)
              modal_strength = snt_graph.get_node(modstr_edge.tgt).get_label()
              assert modal_strength in [C.FULL_AFF, C.FULL_NEG, C.PART_AFF, C.PART_NEG, C.NEUT_AFF, C.NEUT_NEG]
              modal_triple = (C.AUTHOR, f':{modal_strength}', snt_graph.get_node(modstr_edge.src))
              # if not doc_graph.has_modals():
              #   doc_graph.add_root2author_modal()
              doc_graph.add_modal(modal_triple)
              logger.info("Moved Snt edge `%s` to DocGraph as `%s`", modstr_edge, modal_triple)
            elif has_modpred_edge:
              snt_graph.remove_edge(modpred_edge)
              # since no modal-strength is provided, full-affirmative is assumed
              modal_triple = (snt_graph.get_node(modpred_edge.tgt), C.FULL_AFF_EDGE, snt_graph.get_node(modpred_edge.src))
              doc_graph.add_modal(modal_triple)
              logger.info("Moved Snt edge `%s` to DocGraph as `%s`", modpred_edge, modal_triple)

          # if aggression == 3:
          for i, coref_triple in enumerate(doc_graph.coref_triples):
            cs, cr, ct = coref_triple
            if cr == ':subset':
              doc_graph.update_triple_at_index(C.COREF, index=i, new_triple=(cs, C.SUBSET_OF_EDGE, ct))
            elif cr == ':contains':
              doc_graph.update_triple_at_index(C.COREF, index=i, new_triple=(cs, C.SUBSET_OF_EDGE, ct))

          for k, temporal_triple in enumerate(doc_graph.temporal_triples):
            ts, tr, tt = temporal_triple
            if tr == ':contains':
              doc_graph.update_triple_at_index(C.TEMPORAL, index=k, new_triple=(ts, C.CONTAINED_EDGE, tt))

          out.append(example.encode(with_alignment=True))

        out_fpath = io_utils.save_txt(out, output_dir, fname=fname, delimiter='\n\n\n')
      else:
        out_fpath = io_utils.save_txt(raw_text, output_dir, fname=fname)
      out_fpaths.append(out_fpath)

      if '0002' in fname or '0004' in fname:
        io_utils.copy(out_fpath, os.path.join(output_dir, C.PRUNED, fname))

  return out_fpaths

def get_partition_offsets(data, max_per_doc: int = 0, strategy: str = "greedy") -> List[int]:
  num_data = len(data)
  if max_per_doc == 0:
    return [num_data]

  # a partition offset int means: `UP TO` but `NOT INCLUDING` (0-based)
  partition_offsets =  [] # type: List[int]
  if strategy == 'greedy':
    for i, example in enumerate(data, 1):
      if i % max_per_doc == 0:
        partition_offsets.append(i)
  else:  # more even distribution
    divider = (num_data // max_per_doc) + 1
    partition_size = num_data // divider
    offsets = list(range(partition_size, num_data, partition_size))
    if len(offsets) == 0:
      partition_offsets = [num_data]
    else:
      # just check the endpoint
      if partition_size + num_data - offsets[-1] < max_per_doc:
        offsets = offsets[:-1]
      partition_offsets = offsets

  return partition_offsets

def prepare_inputs(input_dir, output_dir, max_per_doc: List[int] = None,
                   partition_strategy: str = 'greedy', snt_to_tok: Optional[str] = None):
  ### FIST PASS: AMR + Coref
  snts = [] # for post-processing
  toks = [] # ibm-transition-parser
  amrs = [] # leak distill + spring
  amr_sents = [] # AMRBART, list of dicts
  corefs = [] # caw-coref + wl-coref
  # `document-id` to `nth` entry, 0-based
  doc2snt_mapping = dict() # type: Dict[str, int] # IBM + udpipe

  global_idx = 0

  # list of `Document`
  umr_docs = umr_utils.load_umr_dir(
    input_dir, snt_to_tok=snt_to_tok, init_document=True) # type: List[Document]
  num_docs = len(umr_docs)
  num_data = sum(len(x) for x in umr_docs)

  # first pass: AMR + coref inputs
  for umr_doc in tqdm.tqdm(umr_docs, desc='[PREP. 1st-Pass]'):
    doc_id = umr_doc.doc_id
    _, doc_idx = umr_utils.parse_umr_id(doc_id, merge_doc_id=True, int_idx=True)

    snt_graph_inputs = []

    genre_prefix = "wb" if doc_idx==4 else 'nw'
    wl_coref_inputs = {
      'document_id': f"{genre_prefix}_{doc_id}",
      'cased_words': list(),
      'sent_id': list()
    }

    for i, example in enumerate(umr_doc):  # type: int, Example
      cur_doc_snt_id = umr_utils.build_umr_id(doc_id, snt_idx=example.snt_idx)

      # snt graph for inputs to AMR Parsing
      snt, ex_toks = example.snt, example.toks
      tok = " ".join(ex_toks)

      # snt
      snts.append(snt)

      # tokens only
      toks.append(tok)
      amr_sents.append({"sent": tok, "amr": ""})

      # tokens as part of AMR
      snt_input = "\n".join([
        f'# ::id {cur_doc_snt_id}',
        f'# ::snt {snt}',
        f'# ::tok {tok}',
        example.snt_graph
      ])
      snt_graph_inputs.append(snt_input)

      # coref
      tok_split = tok.split()
      wl_coref_inputs['cased_words'].extend(tok_split)
      wl_coref_inputs['sent_id'].extend([i for _ in tok_split])

      # doc2snt mapping
      doc2snt_mapping[cur_doc_snt_id] = global_idx
      global_idx += 1

    amrs.extend(snt_graph_inputs)
    corefs.append(wl_coref_inputs)

  # sanity check
  assert num_data == len(toks) == len(amrs), f"{num_data} vs {len(toks)} vs {len(amrs)}"

  ### SECOND PASS: MDP + TDP (+CDLM)
  num_splits = len(max_per_doc)
  if num_splits == 0:
    num_splits = 1
    max_per_doc = [0]

  split_mappings = dict()
  with tqdm.tqdm(total=num_docs*num_splits, desc='[PREP. 2nd-Pass]') as pbar:
    for split_max_len in max_per_doc:
      docs = []
      docs_dct = []
      split_mappings[split_max_len] = cur_split_mappings = {C.DOC: dict(), C.SNT: dict()}

      num_snts = 0
      for umr_doc in umr_docs:
        doc_id = umr_doc.doc_id
        _, doc_idx = umr_utils.parse_umr_id(doc_id, merge_doc_id=True, int_idx=True)

        cur_split_mappings[C.DOC][doc_id] = doc_id

        doc_input_header = f"filename:<doc id={doc_idx}>:SNT_LIST"
        doc_inputs = [doc_input_header]
        doc_inputs_dct = [doc_input_header, "Januaray 01 , 2000"]

        fname_index_offset = num_docs + 1
        local_snt_idx = 1

        partition_offsets = get_partition_offsets(
          umr_doc, max_per_doc=split_max_len, strategy=partition_strategy)
        for j, example in enumerate(umr_doc): # type: int, Example
          ref_umr_id = example.doc_snt_id

          if j in partition_offsets:
            docs.append("\n".join(doc_inputs))
            docs_dct.append("\n".join(doc_inputs_dct))

            doc_idx = fname_index_offset
            doc_input_header = f"filename:<doc id={fname_index_offset}>:SNT_LIST"
            doc_inputs = [doc_input_header]
            doc_inputs_dct = [doc_input_header, "Januaray 01 , 2000"]

            cur_umr_doc_id = umr_utils.build_umr_id(doc_idx)
            cur_split_mappings[C.DOC][cur_umr_doc_id] = doc_id

            fname_index_offset += 1
            local_snt_idx = 1

          # prefer `tok` rather than `snt`, in case tokenization was previously applied
          tok = " ".join(example.toks)

          doc_inputs.append(tok)
          doc_inputs_dct.append(tok)

          cur_umr_doc_snt_id = umr_utils.build_umr_id(doc_idx, local_snt_idx)
          cur_split_mappings[C.SNT][cur_umr_doc_snt_id] = ref_umr_id

          local_snt_idx += 1

        if len(doc_inputs) > 1:
          docs.append("\n".join(doc_inputs))
          docs_dct.append("\n".join(doc_inputs_dct))

        num_snts += len(umr_doc)
        pbar.update(1)

      # sanity check (-1 for header, -2 for header + artificial dct)
      assert num_snts == sum(len(x.split('\n'))-1 for x in docs) == sum(len(x.split('\n'))-2 for x in docs_dct)

      docs_inputs_fpath = os.path.join(output_dir, D.DOCS_TXT_TEMP % str(split_max_len))
      logger.info("Exporting Doc inputs (Split Len: %d) at %s", split_max_len, docs_inputs_fpath)
      io_utils.save_txt(docs, docs_inputs_fpath, delimiter='\n\n')

      docs_dct_inputs_fpath = os.path.join(output_dir, D.DOCS_TXT_TEMP % f'{C.DCT}_{split_max_len}')
      logger.info("Exporting Doc-DCT inputs (Split Len: %d) at %s", split_max_len, docs_dct_inputs_fpath)
      io_utils.save_txt(docs_dct, docs_dct_inputs_fpath, delimiter='\n\n')

  ### EXPORT
  snts_fpath = os.path.join(output_dir, D.SNTS_TXT)
  logger.info("Exporting Sentences at %s", snts_fpath)
  io_utils.save_txt(snts, snts_fpath, delimiter='\n')

  toks_fpath = os.path.join(output_dir, D.TOKS_TXT)
  logger.info("Exporting Tokenized Sentences at %s", toks_fpath)
  io_utils.save_txt(toks, toks_fpath, delimiter='\n')

  doc2snt_mapping_fpath = os.path.join(output_dir, D.DOC2SNT_MAPPING_JSON)
  logger.info("Exporting Doc2Snt Mapping at %s", doc2snt_mapping_fpath)
  io_utils.save_json(doc2snt_mapping, doc2snt_mapping_fpath)

  amrs_fpath = os.path.join(output_dir, D.AMRS_TXT)
  logger.info("Exporting AMR inputs at %s", amrs_fpath)
  io_utils.save_txt(amrs, amrs_fpath, delimiter='\n\n')

  amrs_jsonl = os.path.join(output_dir, D.AMRS_JSONL)
  logger.info("Exporting AMR JSONL inputs at %s", amrs_jsonl)
  io_utils.save_jsonlines(amr_sents, amrs_jsonl)

  coref_inputs_fpath = os.path.join(output_dir, D.COREF_JSONLINES)
  logger.info("Exporting coref inputs at %s", coref_inputs_fpath)
  io_utils.save_jsonlines(corefs, coref_inputs_fpath)

  split_mapping_fpath = os.path.join(output_dir, D.SPLIT_MAPPING_JSON)
  logger.info("Exporting Split Mapping at %s", split_mapping_fpath)
  io_utils.save_json(split_mappings, split_mapping_fpath)

@timer
def main(args):
  """assume input and output are dirs"""
  input_dir = args.input if args.input is not None else D.UMR_EN
  assert os.path.isdir(input_dir)

  corpus_dir = io_utils.get_dirname(args.output)
  if os.path.exists(corpus_dir) and args.overwrite:
    logger.warning("Overwrite existing directory `%s`", corpus_dir)
    io_utils.remove(corpus_dir)
  os.makedirs(corpus_dir, exist_ok=True)
  pruned_output_dir = os.path.join(corpus_dir, C.PRUNED)
  os.makedirs(pruned_output_dir, exist_ok=True)
  prep_output_dir = os.path.join(corpus_dir, C.PREP)
  os.makedirs(prep_output_dir, exist_ok=True)

  log_fpath = os.path.join(corpus_dir, D.PRP_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin UMR v1.0 (ENG) Corpus Preprocessing ===")
  logger.info("Input dir: %s", input_dir)
  logger.info("Corpus dir: %s", corpus_dir)
  logger.info("Pruned dir: %s", pruned_output_dir)
  logger.info("Prep dir: %s", prep_output_dir)
  logger.info("Cleanup Aggression: %d", args.aggression)
  logger.info("Split Max Lengths: %s", str(args.max_per_doc))
  logger.info("Split Partition Strategy: %s", args.partition_strategy)
  logger.info("Snt-to-Tok: %s", args.snt_to_tok)

  # 1. cleanup + pruning
  fix_umr_docs(input_dir, corpus_dir, aggression=args.aggression)

  # 2. prepare inputs (may apply splitting, which is in fact the default behavior)
  prepare_inputs(
    input_dir=corpus_dir, # the fixed corpus should be the input
    output_dir=prep_output_dir,
    max_per_doc=args.max_per_doc,
    partition_strategy=args.partition_strategy,
    snt_to_tok=args.snt_to_tok,
  )

  logger.info("Done.")

if __name__ == '__main__':
  def add_args(argparser):
    argparser.add_argument('--aggression', type=int, default=1, choices=[0, 1, 2],
                           help='agressiveness when fixing annotation errors, from gentlest (1) to most aggressive (3)')
    argparser.add_argument('--max-per-doc', type=int, nargs='*', default=[0, 30, 80],
                           help="max number(s) of sentences per doc (required)")
    argparser.add_argument('--partition-strategy', choices=['greedy', 'even'], type=str.lower, default='greedy',
                           help="partition strategy when splitting long UMR document")
    argparser.add_argument('--snt-to-tok', choices=['spring', 'sapienza', 'leak_distill', 'ibm', 'jamr', 'none',], type=str.lower,
                           help="how to tokenize source sentences; set to None or simply leave unset for not tokenizing")
  main(script_setup(add_args_fn=add_args))
