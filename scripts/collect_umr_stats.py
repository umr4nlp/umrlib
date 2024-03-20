#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 2/12/24 5:51 AM
import logging

from data.analysis import collect_stats_umr
from utils.misc_utils import script_setup

logger = logging.getLogger(__name__)


if __name__ == '__main__':
  args = script_setup()

  stats, _ = collect_stats_umr(args.input, args.output)

  # sort also
  logger.info("*** Corpus Summary Statistics ***")
  for k, v in stats.items():
    if isinstance(v, dict) and len(v) > 0:
      sample = list(v.values())[0]

      if isinstance(sample, dict):
        for kk, vv in v.items():
          stats[k][kk] = dict(sorted(vv.items(), key=lambda x: x[1], reverse=True))
      else:
        stats[k] = dict(sorted(v.items(), key=lambda x: x[1], reverse=True))

      logger.info('== Nested Statistics for %s ==', k.upper())
      for kk, vv in stats[k].items():
        logger.info(" %s: %s", kk, vv)

    else:
      logger.info(k, v)

  logger.info("Done.")
