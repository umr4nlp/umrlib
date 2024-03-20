#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 10/26/23 10:03 PM
import codecs
import json
import logging
import os
import pickle
import shutil
from collections import defaultdict
from typing import Dict, List, Union

import jsonlines

logger = logging.getLogger(__name__)


def exists(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  return fpath is not None and os.path.exists(fpath)

def get_fpath(fpath_or_dir, fname=None):
  fpath = fpath_or_dir
  if fname is not None:
    if not os.path.isdir(fpath_or_dir):
      fpath_or_dir = os.path.dirname(fpath_or_dir)
    fpath = os.path.join(fpath_or_dir, fname)
  return fpath

def get_unique_fpath(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  if fname is None:
    fname = os.path.split(fpath)[-1]

  idx = 1
  tail = ext = os.path.splitext(fname)[1]
  while os.path.exists(fpath):
    new_tail = f'_{idx}{ext}'
    fpath = fpath.replace(tail, new_tail)
    tail = new_tail
    idx += 1
  return fpath

def get_canonical_fname(fname):
  fname = os.path.basename(fname)
  return fname.split('.')[0]

def get_dirname(fpath, mkdir=False, return_is_dir_flag=False):
  # `fpath` may not yet exist
  is_dir = False
  if os.path.exists(fpath): # already exits;`mkdir` doesn't apply
    if os.path.isdir(fpath):
      dirname = fpath
      is_dir = True
    else:
      dirname = os.path.dirname(fpath)
  else:
    dirname = fpath
    splits = os.path.splitext(fpath)
    if len(splits[1]) > 0:
      prefix, ext = splits
      try:
        int(ext)
        is_dir = True
      except ValueError:
        if '-' in ext:
          is_dir = True
        else:
          dirname = os.path.dirname(fpath)
    else:
      is_dir = True
    if mkdir:
      os.makedirs(dirname, exist_ok=True)

  if return_is_dir_flag:
    return dirname, is_dir
  return dirname

def copy(src_fpath, tgt_fpath):
  return shutil.copy(src_fpath, tgt_fpath)

def move(src_fpath, tgt_fpath):
  return shutil.move(src_fpath, tgt_fpath)

def remove(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  if os.path.exists(fpath):
    shutil.rmtree(fpath, ignore_errors=True)

### json + jsonlines
def load_json(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath) as f:
    obj = json.load(f)
  return obj

def load_jsonlines(fpath_or_dir, fname=None) -> List[Dict]:
  fpath = get_fpath(fpath_or_dir, fname)
  with jsonlines.open(fpath, mode="r") as reader:
    data = [x for x in reader]
  return data

def save_json(obj, fpath_or_dir, fname=None, indent=4, append_if_exists=False):
  fpath = get_fpath(fpath_or_dir, fname)
  if os.path.exists(fpath) and append_if_exists:
    org_data = load_json(fpath)
    org_data.update(obj)
    obj = org_data

  with open(fpath, 'w') as f:
    json.dump(obj, f, indent=indent)
  return fpath

def save_jsonlines(obj, fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with jsonlines.open(fpath, mode='w') as writer:
    writer.write_all(obj)

# pickle
def load_pickle(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath, 'rb') as f:
    return pickle.load(f)

def save_pickle(obj, fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath, 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

### txt
def load_txt(fpath_or_dir, fname=None, encoding=None, delimiter=None) -> Union[str, List]:
  fpath = get_fpath(fpath_or_dir, fname)
  with codecs.open(fpath, encoding=encoding, errors='ignore') as f:
    data = f.read()
  if delimiter is not None:  # drop empty strings
    data = list(filter(None, data.split(delimiter)))
  return data

def save_txt(obj, fpath_or_dir, fname=None, delimiter=None):
  fpath = get_fpath(fpath_or_dir, fname)
  with open(fpath, 'w') as f:
    if delimiter is not None:
      obj = delimiter.join(obj)
    f.write(obj)
  return fpath

# custom
def readin_mtdp_tuples(lines, from_file=False):
  if from_file:
    lines = codecs.open(lines, 'r', 'utf-8').readlines()
  else:
    lines = lines.split('\n')

  num_data = 0

  snt_lists = []
  edge_tuples = []
  doc_ids = []

  mode = None
  for line in lines:
    line = line.strip()
    if line == '':
      continue
    elif line.endswith('LIST'):
      tmp = line.strip().split(':')

      mode = tmp[-1]
      if mode == 'SNT_LIST':
        edge_tuples.append([])
        snt_lists.append([])

        # may not be int but just for now
        doc_ids.append(int(tmp[-2][:-1].split('=')[-1]))
      else:
        num_data += len(snt_lists[-1])
    elif mode == 'EDGE_LIST':
      edge = line.strip().split('\t')
      # print(filename, edge)
      assert len(edge) in [2, 4]
      edge_tuples[-1].append(edge)
    else:
      assert mode == 'SNT_LIST'
      snt_lists[-1].append(line.strip())

  return edge_tuples, snt_lists, doc_ids

def readin_conll(fpath_or_dir, fname=None):
  fpath = get_fpath(fpath_or_dir, fname)
  lines = load_txt(fpath, delimiter='\n')[1:-1] # drop `#begin..` and `#end..` at both ends

  # {cluster_id -> {doc_id -> list} }
  prev_snt_id = local_tok_id = 0
  clusters_dict = defaultdict(lambda: defaultdict(list))
  for line in lines:
    _, _, doc_id, snt_id, tok_id, tok, _, cluster_id = line.split('\t')

    if prev_snt_id != snt_id:
      local_tok_id = 0
    else:
      local_tok_id += 1
    prev_snt_id = snt_id

    if cluster_id != '-':
      cluster_id = int(cluster_id[1:-1])
      if snt_id == 0:
        assert local_tok_id == tok_id-1
      clusters_dict[cluster_id][doc_id].append((int(snt_id)+1, local_tok_id, tok))

  # contains singletons per doc
  return clusters_dict
