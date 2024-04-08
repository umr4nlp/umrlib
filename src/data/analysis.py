#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/17/24 11:59 AM
"""meant to be used in Gradio's HighlightedText component"""
import logging
import os
from collections import defaultdict

import gradio as gr
import numpy as np
import tqdm

from data import amr_utils, umr_utils
from structure.alignment import Alignment
from structure.snt_graph import SntGraph
from utils import consts as C, io_utils, regex_utils

logger = logging.getLogger(__name__)


def collect_stats_amr(input_fpath, doc2snt_mapping=None, output_fpath_or_dir=None):
  if not doc2snt_mapping:
    mode = 'leamr'
    corpus = amr_utils.load_amr_file(input_fpath)
  else:
    mode = 'ibm'
    corpus = amr_utils.load_amr_file_ibm(input_fpath, doc2snt_mapping=doc2snt_mapping)

  stats = {
    'mode': mode,
    'num_snts': len(corpus[0]),
    'num_toks': 0,
    'num_nodes': 0,
    'num_concepts': 0,
    'num_attributes': 0,
    'num_edges': 0,
  }

  for meta, graph in zip(*corpus):
    graph = SntGraph.init_amr_graph(graph)

    tok = meta[C.TOK] if C.TOK in meta else meta[C.SNT]
    stats['num_toks'] += len(tok.split())
    stats['num_nodes'] += graph.num_nodes
    stats['num_concepts'] += graph.num_concepts
    stats['num_attributes'] += graph.num_attributes
    stats['num_edges'] += graph.num_edges

  # if output_fpath_or_dir is None:
  output_fpath = output_fpath_or_dir
  if output_fpath is not None:
    if os.path.isdir(output_fpath):
      output_fpath = io_utils.get_fpath(output_fpath, 'stats_amr.json')
    logger.info("Saving UMR stats to %s", output_fpath)
    io_utils.save_json(stats, output_fpath)

  return stats, output_fpath

def load_amr(input_fpath, doc2snt_mapping=None):
  if not doc2snt_mapping:
    corpus = amr_utils.load_amr_file(input_fpath)
  else:
    corpus = amr_utils.load_amr_file_ibm(input_fpath, doc2snt_mapping=doc2snt_mapping)

  out = []
  for meta, graph in zip(*corpus):
    for k,v in meta.items():
      out.append(f"# ::{k} {v}")
    out.append(graph + "\n")
  out_str = "\n".join(out)

  stats, _ = collect_stats_amr(input_fpath, doc2snt_mapping)

  text_out = {'value': [(out_str, None)], '__type__': 'update'}
  stats_out = {'value': stats, 'label': "Summary Statistics", '__type__': 'update'}

  return text_out, stats_out

def search_amr(input_fpath, amr_id, target, doc2snt_mapping=None):
  if not doc2snt_mapping:
    mode = 'leamr'
    corpus = amr_utils.load_amr_file(input_fpath, load_leamr=True)
  else:
    mode = 'ibm'
    corpus = amr_utils.load_amr_file(input_fpath, doc2snt_mapping=doc2snt_mapping)

  amr_id, target = amr_id.strip(), target.strip()

  found = False
  for meta, graph in zip(*corpus):
    if meta[C.ID] == amr_id:
      found = True
      break

  if not found:
    msg = f"Could not find `{target}` in `{amr_id}`"
    gr.Warning(msg)
    logger.warning(msg)
    return gr.Highlightedtext()

  tok = meta[C.TOK] if C.TOK in meta else meta[C.SNT]
  toks = tok.split()
  graph_obj = SntGraph.init_amr_graph(graph)
  alignment = Alignment(meta, graph_obj, mode=mode)

  out = [(f"# ::id {amr_id}\n", None), (f"# ::tok ", None)]
  if target in graph_obj.var2idx:
    color_map = {'found': 'blue'}

    # `target` is an AMR variable
    target_idx = graph_obj.var2idx[target]
    aligned_token_span = alignment.amr2tok[target_idx]

    snt_before = " ".join(toks[:aligned_token_span[0]])
    snt_found = " ".join(toks[aligned_token_span[0]:aligned_token_span[1]])
    snt_after = " ".join(toks[aligned_token_span[-1]:])
    out.append((snt_before, None))
    out.append((snt_found, 'found'))
    out.append((f'{snt_after}\n', None))

    targets = [f'({target} ', f' {target} ' f' {target})']
    for graph_line in graph.split('\n'):
      graph_line_ = graph_line.strip()
      if any(target in graph_line_ for target in targets) or graph_line_.endswith(target):
        out.append((f'{graph_line}\n', 'found'))
      else:
        out.append((f'{graph_line}\n', None))

  else:
    color_map = {}
    colors = ['red', 'yellow', 'blue', 'green']
    # AT MOST 4

    # start_offsets = [i for i in range(len(tok)) if tok.startswith(target, i)]
    # num_matches = len(start_offsets)
    # if num_matches > 4:
    #   msg = f"Found {num_matches} matches, only showing 5"
    #   gr.Warning(msg)
    #   logger.warning(msg)
    #   start_offsets = start_offsets[:4]
    #
    # for i, start_offset in enumerate(start_offsets):
    #   found_signal = f'found_{i+1}'
    #   color_map[found_signal] = colors[i-1]
    raise NotImplementedError()


    # `target` is a span of tokens

  return {'value': out, 'color_map': color_map, '__type__': 'update'}

def inspect_umr(input_fpath):
  org_input_fpath = input_fpath
  if os.path.isdir(input_fpath):
    input_fpath_list = [os.path.join(input_fpath, x) for x in os.listdir(input_fpath) if x.endswith('.txt')]
  else:
    input_fpath_list = [input_fpath]

  # 1. simple inspection
  inspections = []
  color_map = {'err': 'red', 'warn': 'yellow'}

  for input_fpath in tqdm.tqdm(input_fpath_list, desc=f'[UMR Inspection]'):
    cur_inspection = []

    raw_txt = io_utils.load_txt(input_fpath)
    segments = list(filter(None, raw_txt.split('\n\n\n')))
    for i, segment in enumerate(segments, 1):
      snt_idx = None

      for block in segment.strip().split('\n\n'):
        block = block.strip()

        if block.startswith("# ::"):
          if '\t' in block:
            a, b = block.split('\t')
          else:
            segment_s = block.split(' ')
            a = " ".join(segment_s[:3])
            b = " ".join(segment_s[3:])
          snt_idx = int(a[8:])
          if snt_idx != i:
            cur_inspection.append((f'{block}\n', 'err'))
          else:
            cur_inspection.append((f'{block}\n', None))

        elif block.startswith('# alignment'):
          for line in block.split('\n'):
            if line.startswith("#"):
              cur_inspection.append((f'{line}\n', None))
            else:
              var, aligned = line.split(':')
              if aligned.strip().startswith('-1'):
                cur_inspection.append((f'{line}\n', 'warn'))
              else:
                cur_inspection.append((f'{line}\n', None))

        elif block.startswith('# sentence'):
          for line in block.split('\n'):
            if line.count('"') % 2 == 1:
              cur_inspection.append((f'{line}\n', 'err'))
            else:
              cur_inspection.append((f'{line}\n', None))

        elif block.startswith('# document'):
          lines = block.split('\n')
          cur_inspection.append((f"{lines[0]}\n", None))

          root_line = lines[1]
          root_var = root_line.split()[0][1:]
          if snt_idx != regex_utils.parse_var(root_var, as_int_snt_idx=True)[0]:
            cur_inspection.append((f"{root_line}\n", 'err'))
          else:
            cur_inspection.append((f"{root_line}\n", None))
          # for doc_line in lines[2:]:
          #   try:
          #     close_idx = doc_line.strip().rindex(")")
          #     if close_idx + 1 != len(doc_line):
          #       cur_inspection.append((f"{doc_line}\n", 'err'))
          #     else:
          #       cur_inspection.append((f"{doc_line}\n", None))
          #   except ValueError:
          #     # print("???", snt_idx, doc_line)
          #     # raise Exception
          #     cur_inspection.append((f"{doc_line}\n", 'err'))
          cur_inspection.append(("\n".join(lines[2:]), None))

        cur_inspection.append(('\n', None))
      cur_inspection.append(('\n', None))
    inspections.extend(cur_inspection)

  # 2. summary statistics
  stats, _ = collect_stats_umr(org_input_fpath)

  inspect_out = {'value': inspections, 'color_map': color_map, '__type__': 'update'}
  stats_out = {'value': stats, 'label': "Summary Statistics", '__type__': 'update'}

  return inspect_out, stats_out

def search_var_umr(input_fpath, var):
  if os.path.isdir(input_fpath):
    input_fpath_list = [os.path.join(input_fpath, x) for x in os.listdir(input_fpath) if x.endswith('.txt')]
  else:
    input_fpath_list = [input_fpath]

  # sanity check
  try:
    snt_idx,snt_var = regex_utils.parse_var(var)
  except AttributeError:
    msg = f"A var ({var}) was given which doesn't look like one"
    logger.warning(msg)
    gr.Warning(msg)
    return gr.Highlightedtext()
  else:
    if len(snt_idx) == 0 or len(snt_var) == 0:
      msg = f"A var ({var}) was given which doesn't look like one"
      logger.warning(msg)
      gr.Warning(msg)
      return gr.Highlightedtext()

  snt_idx = int(snt_idx)

  out = []
  color_map = {'found': 'blue'}

  for input_fpath in tqdm.tqdm(input_fpath_list, desc=f'[UMR Var Search]'):
    raw_txt = io_utils.load_txt(input_fpath)
    for segment in list(filter(None, raw_txt.split('\n\n\n'))):
      blocks = segment.strip().split('\n\n')
      for block in blocks:
        block = block.strip()

        if block.startswith("# ::"):
          if '\t' in block:
            a, b = block.split('\t')
          else:
            segment_s = block.split(' ')
            a = " ".join(segment_s[:3])
            b = " ".join(segment_s[3:])
          if snt_idx == int(a[8:]):
            target = f'{var}:'
            for alignment in blocks[2].split('\n')[1:]:
              if target in alignment: # e.g., 's1l: 2-2'
                logger.info("Alignment: %s", alignment)
                start,end = alignment.split(':')[-1].split('-')
                start, end = int(start.strip()), int(end.strip())
                if start in [0,-1]:
                  out.append((f'{block}\n', None))
                else:
                  snt_tok = b.split()
                  out.append((" ".join(snt_tok[:start-1]), None))
                  out.append((" ".join(snt_tok[start-1:end]), 'found'))
                  out.append((" ".join(snt_tok[end:]) + "\n", None))
                break
          else:
            out.append((f'{block}\n', None))
        elif block.startswith('# alignment'):
          target = f'{var}:'
          for line in block.split('\n'):
            if target in line:
              out.append((f'{line}\n', 'found'))
            else:
              out.append((f'{line}\n', None))
        elif block.startswith('# sentence') or block.startswith('# document'):
          targets = [f'{var} ', f'{var})']
          for line in block.split('\n'):
            if any(target in line for target in targets) or line.endswith(var):
              out.append((f'{line}\n', 'found'))
            else:
              out.append((f'{line}\n', None))
        out.append(('\n', None))
      out.append(('\n', None))

  return {'value': out, 'color_map': color_map, '__type__': 'update'}

def collect_stats_umr(input_fpath_or_dir, output_fpath_or_dir=None):
  # 1. load
  assert os.path.exists(input_fpath_or_dir)

  if os.path.isdir(input_fpath_or_dir):
    corpus = umr_utils.load_umr_dir(input_fpath_or_dir, init_graphs=True)
  else:
    corpus = [umr_utils.load_umr_file(input_fpath_or_dir, init_graphs=True)]

  stats = {
    'num_docs': len(corpus),
    'num_snts': 0,
    'num_per_sent_docs': 0,
    'num_toks': 0,
    'num_empty_doc_graph': 0,
    'num_nodes': 0,
    'num_concepts': 0,
    'num_attributes': 0,
    'num_aspects': 0,
    'num_snt_edges': 0,
    'num_corefs': 0,
    'num_modals': 0,
    'num_temporals': 0,
    'snt_edges': defaultdict(int),
    'aspect_types': defaultdict(int),
    'refer_person_types': defaultdict(int),
    'refer_number_types': defaultdict(int),
    'coref_types': defaultdict(int),
    'modal_types': defaultdict(int),
    'temporal_types': defaultdict(int),
    'node2aspects': defaultdict(lambda: defaultdict(int)),
  }

  for doc_data_list in tqdm.tqdm(corpus, desc=f'[Collecting Statistics]'):
    stats['num_snts'] += len(doc_data_list)

    for data in tqdm.tqdm(doc_data_list):
      toks = data.toks
      stats['num_toks'] += len(toks)

      dg, sg = data.doc_graph, data.snt_graph
      if not dg.is_empty():
        stats['num_per_sent_docs'] += 1

      ### snt-graph stats
      stats['num_nodes'] += sg.num_nodes
      for node in sg.node_list:
        if regex_utils.is_concept(node.get_label()):
          stats['num_concepts'] += 1
        elif node.is_attribute:
          stats['num_attributes'] += 1

      stats['num_snt_edges'] += sg.num_edges
      for edge in sg.edges:
        edge_label = edge.get_label(decorate=True)  # no prefix `:`
        # if edge_label.startswith(":modal"):
        stats['snt_edges'][edge_label] += 1

        src_node_label = sg.get_node(edge.src).get_label()
        tgt_node_label = sg.get_node(edge.tgt).get_label()

        if edge_label == C.ASPECT_EDGE:
          stats['aspect_types'][tgt_node_label] += 1
          stats['node2aspects'][src_node_label][tgt_node_label] += 1
          stats['num_aspects'] += 1

        elif edge_label in [C.REF_PERSON_EDGE, C.REF_NUMBER_EDGE]:
          stats[f'{edge_label[1:].replace("-", "_")}_types'][tgt_node_label] += 1

        elif edge_label == C.MODAL_STRENGTH_EDGE:
          stats['modal_types'][tgt_node_label] += 1

      ### doc-graph stats
      if dg.is_empty():
        stats['num_empty_doc_graph'] += 1

      else:
        stats['num_corefs'] += dg.num_corefs
        stats['num_modals'] += dg.num_modals
        stats['num_temporals'] += dg.num_temporals

        for triple_name in ['coref', 'modal', 'temporal']:
          for triple in getattr(dg, f'{triple_name}_triples'):
            stats[f'{triple_name}_types'][triple[1]] += 1

  # if output_fpath_or_dir is None:
  output_fpath = output_fpath_or_dir
  if output_fpath is not None:
    if os.path.isdir(output_fpath):
      output_fpath = io_utils.get_fpath(output_fpath, 'stats_umr.json')
    logger.info("Saving UMR stats to %s", output_fpath)
    io_utils.save_json(stats, output_fpath)

  return stats, output_fpath

"""
def is_root_node(node):
  return node.snt_index_in_doc == -1
  
def is_author_node(node):
  return node.snt_index_in_doc == -3

def is_null_conceiver_node(node):
  return node.snt_index_in_doc == -5

def is_dct_node(node):
  return node.snt_index_in_doc == -7
"""

def collect_stats_mtdg(input_fpath, output_fpath_or_dir=None):
  # 1. load
  assert os.path.exists(input_fpath)

  data_list, snts_list, ids_list = io_utils.readin_mtdp_tuples(input_fpath, from_file=True)

  num_docs=  len(ids_list)
  stats = {
    'num_docs': num_docs,
    'num_snts': 0,
    "avg_num_snts_per_doc": 0,
    'num_toks': 0,
    C.ROOT: {'total': 0, 'src': 0, 'tgt': 0}, # -1_-1_-1
    C.AUTHOR: {'total': 0, 'src': 0, 'tgt': 0}, # -3_-3_-3
    C.NULL_CONCEIVER: {'total': 0, 'src': 0, 'tgt': 0}, # -5_-5_-5
    C.DCT_FULL: {'total': 0, 'src': 0, 'tgt': 0}, # -7_-7_-7
    'abstract': {'total': 0, 'src': 0, 'tgt': 0}, # abstract nodes: root, author, null_conc, dct
    'valid': {'total': 0, 'src': 0, 'tgt': 0}, # all other, valid nodes with refernce to tokens
    'total_anno': 0,
    'label_types': defaultdict(int),
    'labels': defaultdict(int),
  }
  # ACCOUNT FOR THE ARTIFICIAL DATE AT THE TOP??? NO, take it as is
  num_toks = 0
  num_snts = []
  for snts in snts_list:
    num_snts.append(len(snts))
    for snt in snts:
      num_toks += len(snt.split())
  stats['num_snts'] = total_num_snts =  sum(num_snts)
  stats['num_toks'] = num_toks
  if total_num_snts > 0:
    stats['avg_num_snts_per_doc'] = round(np.mean(num_snts), 2)

  for data in tqdm.tqdm(data_list, f'[Collecting Statistics]'):
    stats['total_anno'] += len(data)

    # eitjer add to total and that's it, or only add to srd and tgt
    for datum in data:
      if len(datum) == 2:
        child, label_type = datum

        if child.startswith('-1'):
          stats[C.ROOT]['total'] += 1
        elif child.startswith('-3'):
          stats[C.AUTHOR]['total'] += 1
        elif child.startswith('-5'):
          stats[C.NULL_CONCEIVER]['total'] += 1
        elif child.startswith('-7'):
          stats[C.DCT_FULL]['total'] += 1
        else:
          stats['valid']['total'] += 1
        stats['label_types'][label_type] += 1

      else:
        child, label_type, parent, label = datum

        if child.startswith('-1'):
          stats[C.ROOT]['tgt'] += 1
        elif child.startswith('-3'):
          stats[C.AUTHOR]['tgt'] += 1
        elif child.startswith('-5'):
          stats[C.NULL_CONCEIVER]['tgt'] += 1
        elif child.startswith('-7'):
          stats[C.DCT_FULL]['tgt'] += 1
        else:
          stats['valid']['tgt'] += 1

        if parent.startswith('-1'):
          stats[C.ROOT]['src'] += 1
        elif parent.startswith('-3'):
          stats[C.AUTHOR]['src'] += 1
        elif parent.startswith('-5'):
          stats[C.NULL_CONCEIVER]['src'] += 1
        elif parent.startswith('-7'):
          stats[C.DCT_FULL]['src'] += 1
        else:
          stats['valid']['src'] += 1

        stats['label_types'][label_type] += 1
        stats['labels'][label] += 1

  for k in [C.ROOT, C.AUTHOR, C.NULL_CONCEIVER, C.DCT_FULL]:
    v = stats[k]
    stats[k]['total'] += v['src'] + v['tgt']
    stats['abstract']['src'] += v['src']
    stats['abstract']['tgt'] += v['tgt']
  stats['abstract']['total'] += stats['abstract']['src'] + stats['abstract']['tgt']
  stats['valid']['total'] += stats['valid']['src'] + stats['valid']['tgt']

  # if output_fpath_or_dir is None:
  output_fpath = output_fpath_or_dir
  if output_fpath is not None:
    if os.path.isdir(output_fpath):
      output_fpath = io_utils.get_fpath(output_fpath, 'stats_mtdg.json')
    logger.info("Saving MTDG stats to %s", output_fpath)
    io_utils.save_json(stats, output_fpath)

  return stats, output_fpath

def load_mtdg(input_fpath):
  raw_text = io_utils.load_txt(input_fpath)
  stats, _ = collect_stats_mtdg(input_fpath)

  text_out = {'value': [(raw_text, None)], '__type__': 'update'}
  stats_out = {'value': stats, 'label': "Summary Statistics", '__type__': 'update'}

  return text_out, stats_out

def search_anno_mtdg(input_fpath, doc_id, var):
  assert os.path.exists(input_fpath)
  data_list, snts_list, ids_list = io_utils.readin_mtdp_tuples(input_fpath, from_file=True)

  out = []
  color_map = {'found': 'blue'}
  num_found = 0
  var = var.strip()

  for data, snts, cur_doc_id in zip(data_list, snts_list, ids_list):
    if cur_doc_id == doc_id:
      if len(snts) == 0:
        msg = "This MDTG file contains no sentences"
        logger.warning(msg)
        gr.Warning(msg)
        out.append(("\n".join(snts), None))
        out.append(("\n".join("\t".join(x) for x in data), None))

      else:
        if not var.startswith('-'):
          snt_idx, start, end = (int(ch) for ch in var.split('_'))
          snt_toks = snts[snt_idx].split()
          before = "\n".join(snts[:snt_idx]) + f"\n{' '.join(snt_toks[:start])}"
          found = " ".join(snt_toks[start:end+1])
          rest = " ".join(snt_toks[end+1:]) + "\n"
          after = rest + "\n".join(snts[snt_idx+1:]) + "\n"
          out.append((before, None))
          out.append((found, 'found'))
          out.append((after, None))
          num_found += 1

        else:
          out.append(("\n".join(snts) + "\n", None))

        for datum in tqdm.tqdm(data):
          datum_str = "\t".join(datum) + "\n"
          if len(datum) == 2:
            child, _ = datum
            if child == var:
              out.append((datum_str, 'found'))
              num_found += 1
            else:
              out.append((datum_str, None))
          else:
            child, _, parent, _ = datum
            if var in [child, parent]:
              out.append((datum_str, 'found'))
              num_found += 1
            else:
              out.append((datum_str, None))
      break

  if num_found == 0:
    msg = "Could not find `{}` in MDTG".format(var)
    logger.warning(msg)
    gr.Warning(msg)

  return {'value': out, 'color_map': color_map, '__type__': 'update'}

def collect_stats_cdlm(input_fpath, output_fpath_or_dir=None):
  assert os.path.exists(input_fpath)

  stats = {
    "num_docs": 0,
    "num_clusters": 0,
    "num_clusters_per_doc": defaultdict(int),
    "num_singletons": 0,
  }
  doc_ids = set()
  clusters = defaultdict(int)

  for line in tqdm.tqdm(io_utils.load_txt(input_fpath, delimiter='\n')[1:-1], desc=f'[Collecting Statistics]'):
    line_s = line.split('\t')
    doc_id = line_s[2]
    doc_ids.add(doc_id)
    cluster_id = line_s[-1]
    if cluster_id != '-':
      cluster_id = int(cluster_id[1:-1])
      clusters[cluster_id] += 1
      stats['num_clusters_per_doc'][doc_id] += 1

  stats['num_docs'] = len(doc_ids)
  stats['num_clusters'] = len(clusters)
  for x in clusters.values():
    if x == 1:
      stats['num_singletons'] += 1

  output_fpath = output_fpath_or_dir
  if output_fpath is not None:
    if os.path.isdir(output_fpath):
      output_fpath = io_utils.get_fpath(output_fpath, 'stats_cdlm.json')
    logger.info("Saving UMR stats to %s", output_fpath)
    io_utils.save_json(stats, output_fpath)

  return stats, output_fpath

def load_cdlm(input_fpath):
  raw_text = io_utils.load_txt(input_fpath)
  stats, _ = collect_stats_cdlm(input_fpath)

  text_out = {'value': [(raw_text, None)], '__type__': 'update'}
  stats_out = {'value': stats, 'label': "Summary Statistics", '__type__': 'update'}

  return text_out, stats_out

def search_cluster_cdlm(input_fpath, cluster_id):
  out = []
  color_map = {'found': 'blue'}
  num_found = 0

  found_doc_id = None
  for line in io_utils.load_txt(input_fpath, delimiter='\n'):
    if line.startswith("#"):
      out.append( (f'{line}\n', None) )
    else:
      line_s = line.split('\t')
      if line_s[-1] != '-':
        if cluster_id == int(line_s[-1][1:-1]):
          out.append((f'{line}\n', 'found'))
          num_found += 1
          found_doc_id = line_s[2]
        else:
          out.append((f'{line}\n', None))
      else:
        out.append((f'{line}\n', None))

  if num_found == 0:
    msg = "Could not find `{}` in MDTG".format(cluster_id)
    logger.warning(msg)
    gr.Warning(msg)

  else:
    pruned_out = [out[0]]
    for x in out[1:-1]:
      xs = x[0].split('\t')
      if xs[2] == found_doc_id:
        pruned_out.append(x)
    pruned_out.append(out[-1])
    out = pruned_out

  return {'value': out, 'color_map': color_map, '__type__': 'update'}

def collect_stats_coref(input_fpath, output_fpath_or_dir=None):
  assert os.path.exists(input_fpath)

  stats = {
    "num_docs": 0,
    "num_clusters": 0,
    "num_clusters_per_doc": defaultdict(int),
  }

  dicts = io_utils.load_jsonlines(input_fpath)
  for data in tqdm.tqdm(dicts, desc=f'[Collecting Statistics]'):
    stats['num_docs'] += 1

    word_clusters = data['word_clusters']
    num_word_clusters = len(word_clusters)
    stats['num_clusters'] += num_word_clusters
    stats['num_clusters_per_doc'][data['document_id']] += num_word_clusters

  output_fpath = output_fpath_or_dir
  if output_fpath is not None:
    if os.path.isdir(output_fpath):
      output_fpath = io_utils.get_fpath(output_fpath, 'stats_cdlm.json')
    logger.info("Saving UMR stats to %s", output_fpath)
    io_utils.save_json(stats, output_fpath)

  return stats, output_fpath

def load_coref(input_fpath):
  out = dict()
  dicts = io_utils.load_jsonlines(input_fpath)
  for data in dicts:
    out[data.pop('document_id')] = data

  stats, _ = collect_stats_coref(input_fpath)
  stats_out = {'value': stats, 'label': "Summary Statistics", '__type__': 'update'}
  return out, stats_out

def search_cluster_coref(input_fpath, doc_id, cluster_id):
  assert os.path.exists(input_fpath)
  dicts = io_utils.load_jsonlines(input_fpath)

  doc_id = doc_id.strip()
  color_map = {'found': 'blue'}

  data = None
  found = False
  for data in dicts:
    if data['document_id'] == doc_id:
      found = True
      break

  if not found:
    msg = "Could not find `{}` in COREF".format(doc_id)
    logger.warning(msg)
    gr.Warning(msg)

  else:
    words = data['cased_words']
    word_cluster = data['word_clusters'][cluster_id]
    span_cluster = data['span_clusters'][cluster_id]

    # int, List[int]
    ranges = []
    for word, span in zip(word_cluster, span_cluster):
      ranges.append(range(*span))

    out = []
    for idx, word in enumerate(words):
      in_range = False
      for cur_range in ranges:
        if idx in cur_range:
          in_range = True
          break

      if in_range:
        out.append( (word, 'found') )
      else:
        out.append( (word, None) )

    return {'value': out, 'color_map': color_map, '__type__': 'update'}
    # return gr.Highlightedtext(
    #   label="Coref Cluster Search Result",
    #   value=out,
    #   color_map=color_map,
    #   combine_adjacent=True,
    #   show_legend=True
    # )


if __name__ == '__main__':
  res = search_anno_mtdg('EXP/umr-v1.0-en_split/tmp/modals.txt', 1, '-3_-3_-3')
  # print(a['value'][0])
  # print(b['value'])
  print(res)


