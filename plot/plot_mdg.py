#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/23/24 5:26 PM
import logging
import os

import igraph as ig
import matplotlib.pyplot as plt

from structure import SntGraph

logger = logging.getLogger(__name__)


# FLAGS + CONST
FIGURE_SUBDIR = 'figures'
if os.path.basename(os.getcwd()) != 'plot':
  FIGURE_SUBDIR = 'plot/figures'
# TODO: may want to make this PDF ??>
FIGURE_FPATH = f'{FIGURE_SUBDIR}/modal_plot_v3.png'

# plotting
SAVE = True

input_str = """
(r / root
  :modal (a / author
    :full-affirmative (s1l / left)
    :full-negative (s1j / join)
    :full-affirmative (s2s / said)
    :full-affirmative (s2p / she
      :partial-affirmative (s2e / eating)
    )))
"""
input_str = """
(r / root
  :modal (a / author
    :AFF (s1l / left)
    :NEG (s1j / join)
    :AFF (s2s / said)
    :AFF (s2p / she
      :PRT-AFF (s2e / eating)
    )))
"""

print(input_str)
graph = SntGraph.init_amr_graph(input_str)
print(graph)

plt.figure()
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
g = graph.plot()
layout = g.layout_sugiyama()  # graph
layout.rotate(180)
layout.mirror(0)

ig.plot(
  g,
  target=ax,
  layout=layout,
  vertex_color='white',
  edge_align_label=False,
  edge_label=g.es['label'],
  edge_background='white',
  edge_width=0.5,
  edge_label_size=16.,
  edge_arrow_size=8,
  vertex_shape="circle",
  vertex_label_size=18.,
  vertex_size=80,
  margin=0,
)
ax.axis('off')
fig.tight_layout()
if SAVE:
  fig.savefig(FIGURE_FPATH)
plt.show()
