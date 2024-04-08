#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 4/5/24 9:18 PM
import json
import logging

from utils import defaults as D, io_utils

logger = logging.getLogger(__name__)


AMR_MORPH_VERB2NOUN = dict()
for line in io_utils.load_txt(
        D.RESOURCES, "morph-verbalization-v1.01.txt", delimiter='\n')[1:]:
  verb = noun = None
  for cur_line in line.split('::')[1:]:
    line_s = cur_line.split()
    key = line_s[0]
    if key.startswith("DERIV-"):
      key = key[6:]
    value = " ".join(line_s[1:])
    if value.startswith('"') and value.endswith('"'):
      value = json.loads(value)
    else:
      breakpoint()
    if key == 'VERB':
      verb = value
    elif key == 'NOUN':
      noun = value
  assert verb is not None
  if noun is not None:
    AMR_MORPH_VERB2NOUN[verb] = noun

AMR_NOUN2VERB = dict()
for line in io_utils.load_txt(
        D.RESOURCES, "verbalization-list-v1.06.txt", delimiter='\n')[1:]:
  line_s = line.split()
  noun, pred = line_s[1], line_s[-1]
  assert line_s[2] == 'TO'
  AMR_NOUN2VERB[noun] = pred
