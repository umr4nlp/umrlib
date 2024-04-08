#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/12/24 3:29 PM
import logging
import os

import igraph as ig
import matplotlib.pyplot as plt
from penman import constant as penman_const

from data.umr_utils import load_umr_file
from utils import consts as C, io_utils
from utils.misc_utils import script_setup

logger = logging.getLogger(__name__)


# FLAGS + CONST
FIGURE_SUBDIR = 'figures'
EXAMPLE_FPATH = 'example_abbr.txt'
if os.path.basename(os.getcwd()) != 'plot':
  FIGURE_SUBDIR = f'plot/{FIGURE_SUBDIR}'
  EXAMPLE_FPATH = f'plot/{EXAMPLE_FPATH}'
# TODO: may want to make this PDF ??>
FIGURE_FPATH = f'{FIGURE_SUBDIR}/doc_plot_v6_patch.png'
COORDS_PICKLE_FPATH = f'{FIGURE_SUBDIR}/coords_v5.pickle'

# when a good starting point fig coords have been found, set to True and apply manual changes
LOAD_COORDS_FROM_PICKLE = True

# 1) LOAD_COORDS_FROM_PICKLE = False and in LINES 19,20 make whitespace between `: `and `AFF`, `:` and `NEG`
# 2) save coords
# 3) LOAD_COORDS_FROM_PICKLE = True and re-attach LINES 19,20

# plotting
MODAL_REL_COLOR = '#d23952'
TEMPORAL_EDGE_COLOR = '#0e5801'
COREF_EDGE_COLOR = '#AAA'
ASPECT_EDGE_COLOR = 'purple'
REF_EDGE_COLOR = 'blue'
SUGIYAMA = False
WITH_TITLE = False
SAVE = True

script_setup()
# def main():
#   s1 = """(s1l / leave-02
# 	:ARG0 (s1p / person
# 		:name (s1n / name :op1 "Kim"))
# 	:aspect performance
# 	:purpose (s1j / join-04
# 		:ARG0 s1p
# 		:ARG1 (s1p2 / person
#                 :ARG1-of (s1o / other-01
#                 	:ARG2 s1p)
#                 :refer-number plural)))"""
#   # s1 = """(s1l / leave-02
#   # 	:ARG0 (s1p / person
#   # 		:name (s1n / name :op1 "Kim"))
#   # 	:aspect performance
#   # 	:purpose (s1j / join-04
#   # 		:ARG1 (s1p2 / person
#   #                 :ARG1-of (s1o / other-01)
#   #                 :refer-number plural)))"""
#   s2 = """(s2s / say-01
# 	:ARG0 (s2p / person
# 		:refer-person 3rd
# 		:refer-number singular)
# 	:ARG1 (s2e / eat-01
# 		:ARG0 (s2p2 / person
#                 :refer-person 3rd
#                 :refer-number plural)
#         :aspect activity
#         :quote s2s))
#     :aspect performance)"""
#   g1 = SntGraph.init_amr_graph(s1, snt_idx=1, record_depth_directed=True)
#   g2 = SntGraph.init_amr_graph(s2, snt_idx=2, record_depth_directed=True)
#   graphs = [g1, g2]
#   snts = [
#     "Kim left to join the others.",
#     '"They are probably eating," she said.'
#   ]
#   # print(snt1)
#   # print(snt2)
#
#   print("Plot")
#   plt.figure()
#   fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#   # fig, axes = plt.subplots(2, 1, figsize=(6, 10))
#   for i, (snt,graph,ax) in enumerate(zip(snts, graphs, axes), 1):
#     # graph.plot(title=f"Sent. {i}: {snt}", fig=fig, ax=ax)
#     graph.plot(fig=fig, ax=ax)
#
#   fig.tight_layout()
#   fig.savefig('./figures/snt_graphs.png')
#   plt.show()
doc = load_umr_file(EXAMPLE_FPATH, init_document=True, init_graphs=True, init_doc_var_nodes=True)
examples = doc.examples_list
print(doc)

plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

mapping = {
  C.ROOT: 0,
  C.AUTHOR: 1,
  C.DCT_FULL: 2,
}
node_labels = ['ROOT', 'AUTHOR', 'DCT']
node_colors = ['lightblue'] * 3
node_orders = [0, 1, 2]
edges = [(0, 1), (0, 2)]
edge_labels = [C.MODAL, C.TEMPORAL]
edge_colors = [MODAL_REL_COLOR, TEMPORAL_EDGE_COLOR]
edge_widths = [1] * 2

# doc_graph is init as part of example; collect triples for later
doc_triples = []
for example in examples:
  print(example)
  snt_graph = example.snt_graph

  snt_idx = snt_graph.idx
  # for i, node_idx in enumerate(snt_graph.nodes):
  for i, node in enumerate(sorted(snt_graph.node_list, key=lambda x: x.depth)):
    node_idx = node.idx
    key = f'{snt_idx}_{node_idx}'
    assert key not in mapping
    mapping[key] = len(mapping)
    # node = snt_graph.get_node(node_idx)
    if node.is_attribute:
      node_label = penman_const.evaluate(node.get_label())
      node_labels.append(penman_const.quote(node_label))
    else:
      node_labels.append(f"{node.as_var()}\n{node.get_label()}")
    node_colors.append('white') # default color
    node_orders.append(len(node_orders))

  for edge in snt_graph.edges:
    src, tgt = edge.src, edge.tgt
    edge_label = edge.get_label()
    src_key, tgt_key = f'{snt_idx}_{src}', f'{snt_idx}_{tgt}'
    src_mapped, tgt_mapped = mapping[src_key], mapping[tgt_key]
    edges.append((src_mapped, tgt_mapped))
    edge_labels.append(edge_label)
    if edge_label == C.ASPECT:
      edge_colors.append(ASPECT_EDGE_COLOR)
    elif edge_label.startswith('ref') or edge_label.startswith('REF'):
      edge_colors.append(REF_EDGE_COLOR)
    else:
      edge_colors.append("black") # default
    edge_widths.append(0.5)

  # doc graph
  doc_triples += example.doc_graph.triples

# add doc-level relations
for (src, rel, tgt) in doc_triples:
  if src.is_abstract():
    src_key = src.get_label()
  else:
    src_key = f'{src.snt_idx}_{src.idx}'
  tgt_key = f'{tgt.snt_idx}_{tgt.idx}'
  src_mapped, tgt_mapped = mapping[src_key], mapping[tgt_key]
  edges.append((src_mapped, tgt_mapped))
  edge_labels.append(rel[1:])  # without prefix `:`
  # edge_colors.append(DOC_REL_COLOR)
  if rel[1:] in C.TEMPORAL_EDGE_MAPPING:
    edge_colors.append(TEMPORAL_EDGE_COLOR)
    # edge_widths.append(1)
  elif rel == C.SAME_ENTITY_EDGE:
    edge_colors.append(COREF_EDGE_COLOR)
    # edge_widths.append(0.5)
  else:
    edge_colors.append(MODAL_REL_COLOR)
  edge_widths.append(1)

  if rel == C.SAME_ENTITY_EDGE:
    node_colors[mapping[src_key]] = node_colors[
      mapping[tgt_key]] = 'orange' if 'orange' not in node_colors else 'lightgreen'

g = ig.Graph(len(mapping), edges, directed=True)
if SUGIYAMA:
  layout = g.layout_sugiyama()  # graph
else:
  layout = g.layout_reingold_tilford(root=[0])  # tree
layout.rotate(180)
# layout.mirror(0)

if LOAD_COORDS_FROM_PICKLE:
  coords = io_utils.load_pickle(COORDS_PICKLE_FPATH)
else:
  coords = layout.coords
  # manual fix
  fix_snt_graph = doc.get_ith_example(0).snt_graph
  node = [x for x in fix_snt_graph.node_list if x.var == 's1p'][0]
  mapping_idx = mapping[f'1_{node.idx}']
  cur_coord = coords[mapping_idx]
  coords[mapping_idx] = [cur_coord[0], cur_coord[1] + 0.3]

  node = [x for x in fix_snt_graph.node_list if x.var == 's1n'][0]
  mapping_idx = mapping[f'1_{node.idx}']
  cur_coord = coords[mapping_idx]
  coords[mapping_idx] = [cur_coord[0], cur_coord[1] + 0.4]

  fix_snt_graph = doc.get_ith_example(1).snt_graph
  for node in fix_snt_graph.node_list:
    mapping_idx = mapping[f'2_{node.idx}']
    cur_coord = coords[mapping_idx]

    if node.var == 's2p':
      coords[mapping_idx] = [cur_coord[0] - 0.76, cur_coord[1] - 0.3]
      for outgoing_edge in fix_snt_graph.get_edges(src=node):
        fix_tgt = fix_snt_graph.get_node(outgoing_edge.tgt)
        mapping_idx = mapping[f'2_{fix_tgt.idx}']
        cur_coord = coords[mapping_idx]
        coords[mapping_idx] = [cur_coord[0] - 0.1, cur_coord[1] - 0.6]
    elif node.var == 's2s':
      coords[mapping_idx] = [cur_coord[0] - 0.33, cur_coord[-1]]
      for outgoing_edge in fix_snt_graph.get_edges(src=node):
        # if outgoing_edge.get_label() == C.ASPECT:
        fix_tgt = fix_snt_graph.get_node(outgoing_edge.tgt)
        mapping_idx = mapping[f'2_{fix_tgt.idx}']
        cur_coord = coords[mapping_idx]
        coords[mapping_idx] = [cur_coord[0] + 0.66, cur_coord[1]]

    elif node.var in ['s2e', 's2p2']:
      if node.var == 's2e':
        coords[mapping_idx] = [cur_coord[0] - 1.9, cur_coord[-1]]
      else:
        coords[mapping_idx] = [cur_coord[0] - 1.75, cur_coord[-1]]
      for outgoing_edge in fix_snt_graph.get_edges(src=node):
        out_label = outgoing_edge.get_label()
        if out_label == 'quote':
          continue
        fix_tgt = fix_snt_graph.get_node(outgoing_edge.tgt)
        mapping_idx = mapping[f'2_{fix_tgt.idx}']
        cur_coord = coords[mapping_idx]
        if node.var == 's2e':
          coords[mapping_idx] = [cur_coord[0] - .35, cur_coord[1]]
        else:
          coords[mapping_idx] = [cur_coord[0] - 2.1, cur_coord[1]]

    elif node.is_attribute:
      if node.get_label() == C.PERFORMANCE:
        coords[mapping_idx] = [cur_coord[0] + 0.33, cur_coord[-1]]

  coords[1] = [coords[1][0] - 0.33, coords[1][1]]
  coords[2] = [coords[2][0] + 0.33, coords[2][1]]
  io_utils.save_pickle(coords, COORDS_PICKLE_FPATH)

ig.plot(
  g,
  target=ax,
  # layout=layout,
  layout=coords,
  autocurve=True,
  edge_label=edge_labels,
  edge_background='white',
  edge_align_label=False,
  edge_color=edge_colors,
  edge_width=edge_widths,
  edge_label_size=12,
  edge_arrow_size=8,
  vertex_color=node_colors,
  vertex_frame_width=1.0,
  vertex_label=node_labels,
  vertex_label_size=18.,
  vertex_order=node_orders,
  vertex_size=110,
  vertex_shape="circle",
  margin=0,
)
if WITH_TITLE:
  ax.set_title(doc.doc_id)
ax.axis('off')
fig.tight_layout()
if SAVE:
  fig.savefig(FIGURE_FPATH)
plt.show()

