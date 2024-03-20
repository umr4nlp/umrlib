#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/28/24 6:55 AM
import logging
import os

import tqdm

from data import analysis
from structure.snt_graph import SntGraph
from utils import io_utils

logger = logging.getLogger(__name__)


def main():
  # analysis.inspect_mdp('/Volumes/SamsungSSD/Research/B-MRP/UMRParsingPlatform/TEMP/EXP/umr-v1.0-en/tmp/modals.txt')
  # analysis.inspect_tdp('/Volumes/SamsungSSD/Research/B-MRP/UMRParsingPlatform/TEMP/EXP/umr-v1.0-en/tmp/temporals.txt')
  # analysis.inspect_mtdp('/Volumes/SamsungSSD/Research/B-MRP/UMRParsingPlatform/TEMP/sample_out.txt')
  fpath = "EXP/umr-v1.0-en_split/tmp/udpipe.json"
  assert os.path.exists(fpath)
  udpipes = io_utils.load_json(fpath)
  for udpipe in tqdm.tqdm(udpipes):
    snt_graph = SntGraph.init_dep_graph(udpipe)
    breakpoint()

if __name__ == '__main__':
  main()
