#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 7/28/23 16:15
import logging

from utils import io_utils

logger = logging.getLogger(__name__)


class Config:

  """supports getitem as a dict-like object"""

  def __init__(self, config):
    assert isinstance(config, dict), "Expects `dict` to init Config, but received %s" % type(config)

    # remove config param
    config.pop('config', None)

    self.__dict__.update(config)

  def __contains__(self, item):
    return item in self.__dict__

  def __getitem__(self, item):
    return self.__dict__[item]

  def __setitem__(self, key, value):
    self.__dict__[key] = value

  @classmethod
  def init_from_fpath(cls, fpath):
    json_config = io_utils.load_json(fpath)
    return cls(json_config)

  def save_to_fpath(self, fpath):
    io_utils.save_json(self.__dict__, fpath)
