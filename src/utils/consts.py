#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 1/2/24 11:29 PM
"""global constants"""
NOT_FOUND = 'NOT FOUND'

# keys
DATA = 'DATA'
EXP = 'EXP'
TMP = 'tmp'
PRP = "prp"
MODELS = 'models'
SCRIPTS = 'scripts'
CORPUS = 'corpus'
PRUNED = "pruned"
PRUNED_MAPPING = "pruned_mapping"
SPLIT_MAPPING = "split_mapping"
PREP = "prep"
DOC2SNT_MAPPING = 'doc2snt_mapping'
CONV = "conversion"
EVAL = 'eval'

# models
SAPIENZA = 'sapienza'
SPRING = 'Spring'
LEAK_DISTILL = 'LeakDistill'
IBM = 'ibm'
IBM_PARSER = 'transition-amr-parser'
AMRBART = 'AMRBART'
JAMR = 'jamr'
ISI = 'isi'
BLINK = 'BLINK'
LEAMR = 'LEAMR'
MBSE = 'mbse'
MODAL = 'modal'
TEMPORAL = 'temporal'
CDLM = 'cdlm'
COREF = 'coref'
WL_COREF = 'wl-coref'
CAW_COREF = 'caw-coref'
MDP = 'modal' # v1
MODAL_MULTI_TASK = 'modal_multi_task'
MDP_PROMPT = 'mdp_prompt'
MDP_PROMPT_END2END = 'mdp_prompt_end2end'
STAGE1 = 'stage1'
STAGE2 = 'stage2'
TDP = 'temporal' # v1
TEMPORAL_PIPELINE = 'temporal_pipeline'
THYME_TDG = 'thyme_tdg'
MDP_STAGE1 = f'{MDP}_{STAGE1}'
MDP_STAGE2 = f'{MDP}_{STAGE2}'
TDP_STAGE1 = f'{TDP}_{STAGE1}'
TDP_STAGE2 = f'{TDP}_{STAGE2}'
UDPIPE = 'udpipe'
MERGE = 'merge'
MERGED = 'merged'

# extension
LOG = 'log'
TXT = 'txt'
JSON = 'json'
JSONL = 'jsonl'
JSONLINES  = 'jsonlines'
AMR_TXT = 'amr.txt'
STAGE1_TXT = 'stage1.txt'
STAGE2_TXT = 'stage2.txt'
CDLM_CONLL = 'cdlm.conll'

# preprocessing
ID = 'id'
SNT = 'snt'
DOC = 'doc'
DOCS = 'docs'
TOK = 'tok'
TOKS = 'toks'
WORD = 'word'
LEMMA = 'lemma'
POS = 'pos'
FEATS = 'feats'
DEP_HEAD = 'dep_head'
DEP_REL = 'dep_rel'
ALIGNMENT = 'alignment'
SNT_GRAPH = 'snt_graph'
DOC_GRAPH = 'doc_graph'
AMR = 'amr'
AMRS = 'amrs'

################################## STRUCTURES ##################################
# nodes
XCONCEPT = 'xconcept'
VAR_TEMPLATE = 'x%d'
VAR_TEMPLATE_WITH_SNT_PREFIX = 's%dx%d'

# edges
INSTANCE = 'instance'
INSTANCE_EDGE = f':{INSTANCE}'
OP = 'op'
OP_edge = f':{OP}'
NAME = 'name'
NAME_EDGE = f':{NAME}'
OF_SUFFIX = '-of'
WIKI = 'wiki'
WIKI_EDGE = f':{WIKI}'

# single node across entire doc
ROOT = "root"
AUTHOR = "author"
DCT = 'dct'
DCT_FULL = "document-creation-time"

### corefs
# edges only
SAME_ENTITY = 'same-entity'
SAME_ENTITY_EDGE = f':{SAME_ENTITY}'
SAME_EVENT = 'same-event'
SAME_EVENT_EDGE = f':{SAME_EVENT}'
SUBSET_OF = 'subset-of'
SUBSET_OF_EDGE = f':{SUBSET_OF}'

EVENT = 'Event'
CONCEIVER = 'Conceiver'
TIMEX = 'Timex'

### modals
NULL_CONCEIVER = "null-conceiver"

# edges
MODAL_STRENGTH = 'modal-strength'
MODAL_STRENGTH_EDGE = f':{MODAL_STRENGTH}'
MODAL_PREDICATE = 'modal-predicate'
MODAL_PREDICATE_EDGE = f':{MODAL_PREDICATE}'
MODAL_EDGE = f':{MODAL}'
FULL_AFF = 'full-affirmative'
FULL_AFF_EDGE = f':{FULL_AFF}'
FULL_NEG = 'full-negative'
FULL_NEG_EDGE = f':{FULL_NEG}'
PART_AFF = 'partial-affirmative'
PART_AFF_EDGE = f':{PART_AFF}'
PART_NEG = 'partial-negative'
PART_NEG_EDGE = f':{PART_NEG}'
NEUT_AFF = 'neutral-affirmative'
NEUT_AFF_EDGE = f':{NEUT_AFF}'
NEUT_NEG = 'neutral-negative'
NEUT_NEG_EDGE = f':{NEUT_NEG}'
UNSP_EDGE = ':unspecified'  # part of UMR corpus but not predicted
MODAL_EDGE_MAPPING = {
  'pos': FULL_AFF_EDGE,
  'neg': FULL_NEG_EDGE,
  'pp': PART_AFF_EDGE,
  'pn': NEUT_AFF_EDGE,
}

### temporals
TEMPORAL_EDGE = ':temporal'
AFTER = "after"
AFTER_EDGE = f":{AFTER}"
BEFORE = "before"
BEFORE_EDGE = f":{BEFORE}"
DEPENDS_ON_EDGE = "depends-on"
DEPENDS_ON_EDGE = f":{DEPENDS_ON_EDGE}"
CONTAINED = 'contained'
CONTAINED_EDGE = f':{CONTAINED}'
# INCLUDES = ':includes'  # same as CONTAINED or OVERLAP?
OVERLAP = "overlap"
OVERLAP_EDGE = f":{OVERLAP}"
TEMPORAL_EDGE_MAPPING = {
  "after": AFTER_EDGE,
  "before": BEFORE_EDGE,
  "Depend-on": DEPENDS_ON_EDGE,
  # "Contained": CONTAINED,
  "included": CONTAINED_EDGE,
  "overlap": OVERLAP_EDGE,
}

### coref
DOCUMENT_ID = 'document_id'
CASED_WORDS = 'cased_words'
SENT_ID = 'sent_id'
SPEAKER = 'speaker'
WORD_CLUSTERS = 'word_clusters'
SPAN_CLUSTERS = 'span_clusters'

### entities
REF_PERSON = 'refer-person'
REF_PERSON_EDGE = f':{REF_PERSON}'
REF_NUMBER = "refer-number"
REF_NUMBER_EDGE = f":{REF_NUMBER}"
REF_PERSON_1ST = "1st"
REF_PERSON_2nd = "2nd"
REF_PERSON_3rd = "3rd"
REF_NUMBER_SINGULAR = "singular"
REF_NUMBER_PLURAL = "plural"

# aspects
ASPECT = 'aspect'
ASPECT_EDGE = f':{ASPECT}'
PERFORMANCE = 'performance'
STATE = 'state'
ACTIVITY = 'activity'
ENDEAVOR = 'endeavor'
PROCESS = 'process'
HABITUAL = 'habitual'
