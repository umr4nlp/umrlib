#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 12/6/23 1:12 AM
import json
import logging
import os
import subprocess
from collections import defaultdict

from utils import defaults as D, io_utils
from utils.misc_utils import add_log2file, script_setup, timer

logger = logging.getLogger(__name__)


def run_udpipe(input_fpath, output_fpath=None):
  cmd = f"curl -F data=@{input_fpath} -F model=english -F input=horizontal -F tagger= -F parser= http://lindat.mff.cuni.cz/services/udpipe/api/process"
  logger.info("Running UDPipe v2 with command `%s`", cmd)
  out = subprocess.check_output(cmd, shell=True, text=True)
  raw_ud_conllus = list(filter(None, json.loads(out)['result'].split('\n\n')))

  ud_conllus = []
  for i, raw_ud_conllu in enumerate(raw_ud_conllus):
    ud_conllu = list()
    for ud_conllu_line in raw_ud_conllu.split('\n'):
      if ud_conllu_line.startswith('#'):
        continue
      ud_conllu_line = ud_conllu_line.split('\t')

      feats_dict = defaultdict(str)

      feats = ud_conllu_line[5]
      if feats != '_':
        for x in feats.split('|'):
          k,v = x.split('=')
          feats_dict[k] = v

      ud_conllu.append({
        'word': ud_conllu_line[1],
        'lemma': ud_conllu_line[2],
        'pos': ud_conllu_line[3],
        'feats': feats_dict,
        'dep_head': int(ud_conllu_line[6])-1,
        'dep_rel': ud_conllu_line[7]
      })

    ud_conllus.append(ud_conllu)

  if output_fpath is not None:
    if os.path.isdir(output_fpath):
      output_fpath = io_utils.get_unique_fpath(output_fpath, D.UDPIPE_JSON)
    logger.info("Exporting UDPipe results at %s", output_fpath)
    io_utils.save_json(ud_conllus, output_fpath)

  return ud_conllus

@timer
def main(args):
  # input file with a single snt per line
  input_fpath = args.input
  assert input_fpath is not None and os.path.isfile(input_fpath)

  output_fpath = args.output
  output_dir, is_dir = io_utils.get_dirname(args.output, mkdir=True, return_is_dir_flag=True)
  if is_dir:
    output_fpath = io_utils.get_unique_fpath(output_fpath, D.UDPIPE_JSON)
  log_fpath = os.path.join(output_dir, D.UDPIPE_LOG)
  add_log2file(log_fpath)

  logger.info("=== Begin Running UDPipe V2 ===")
  logger.info("Input dir: %s", input_fpath)
  logger.info("Output dir: %s", output_dir)

  run_udpipe(input_fpath, output_fpath)

  logger.info("Done.")

if __name__ == '__main__':
  main(script_setup())
