#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: chunjy92
# Date: 3/4/24 8:12 AM
"""token-level tokenization

2 options:
  1) `jamr` or `ibm`
    * jamr-like tokenizer
    * modified from https://github.com/IBM/transition-amr-parser/blob/master/src/transition_amr_parser/amr.py#L1304

  2) `spring` or `leak_distill`
    * modified from https://github.com/SapienzaNLP/LeakDistill/blob/main/spring_amr/snt_to_tok.py
"""
import logging
import re

from spacy.attrs import ORTH
from spacy.lang.en import English

from utils import consts as C

logger = logging.getLogger(__name__)


# single interface
def tokenize(text: str, mode=C.JAMR) -> str:
  if not mode:
    # logger.debug("Empty tokenizer mode; returning `snt` as is")
    return text
  mode = mode.lower()
  if mode == 'none':
    return text
  if mode in [C.JAMR, C.IBM]:
    return tokenize_ibm(text, simple=False)
  else:
    return tokenize_spring(text)

################################################################################
nlp = English()
tokenizer = nlp.tokenizer
tokenizer.add_special_case("etc.", [{ORTH: "etc."}])
tokenizer.add_special_case("``", [{ORTH: "``"}])
tokenizer.add_special_case("no.", [{ORTH: "no."}])
tokenizer.add_special_case("No.", [{ORTH: "No."}])
tokenizer.add_special_case("%pw", [{ORTH: "%pw"}])
tokenizer.add_special_case("%PW", [{ORTH: "%PW"}])
tokenizer.add_special_case("mr.", [{ORTH: "mr."}])
tokenizer.add_special_case("goin'", [{ORTH: "goin"}, {ORTH: "'"}])
tokenizer.add_special_case("'cause", [{ORTH: "'"}, {ORTH: "cause"}])
tokenizer.add_special_case("'m", [{ORTH: "'m"}])
tokenizer.add_special_case("'ve", [{ORTH: "'ve"}])
'''
tokenizer.add_special_case("dont", [{ORTH: "dont"}])
tokenizer.add_special_case("doesnt", [{ORTH: "doesnt"}])
tokenizer.add_special_case("cant", [{ORTH: "cant"}])
tokenizer.add_special_case("havent", [{ORTH: "havent"}])
tokenizer.add_special_case("didnt", [{ORTH: "didnt"}])
tokenizer.add_special_case("youre", [{ORTH: "youre"}])
tokenizer.add_special_case("wont", [{ORTH: "wont"}])
tokenizer.add_special_case("im", [{ORTH: "im"}])
tokenizer.add_special_case("aint", [{ORTH: "aint"}])
'''

tokenizer.add_special_case("<<", [{ORTH: "<<"}])
tokenizer.add_special_case(">>", [{ORTH: ">>"}])


SPECIAL_CASES = {
  'wo': 'will',
  'Wo': 'Will',
  'ca': 'can',
  'Ca': 'Can'
}

def fix_special_cases(tokens, nltk=False):
  for i in range(len(tokens)):
    if tokens[i] == "n't" and tokens[i-1] in SPECIAL_CASES:
      tokens[i-1] = SPECIAL_CASES[tokens[i-1]]
    if nltk:
      if tokens[i] in {"``", "''"}:
        tokens[i] = '"'
  return tokens

def tokenize_spring(text):
  # Add spaces for some cases
  try:
    text = re.sub(r'([-:\ / $])', r' \1 ', text)
  except TypeError:
    print("TEXT:", repr(text))
    breakpoint()
  # Split dates, e.g. "11th" to "11 th"
  text = re.sub(r'(\d+)(st|nd|rd|th|s|am|pm)', r'\1 \2', text)
  # Remove double spaces
  text = re.sub(r'\s+', r' ', text)

  # Tokenize the text
  tokens = tokenizer(text)
  tokens = fix_special_cases([t.text for t in tokens])

  return ' '.join(tokens)

################################################################################
# def protected_tokenizer(sentence_string, simple=False):
def tokenize_ibm(sentence_string, simple=False):
  if simple:
    # simplest possible tokenizer
    # split by these symbols
    sep_re = re.compile(r'[\.,;:?!"\' \(\)\[\]\{\}]')
    toks, pos = simple_tokenizer(sentence_string, sep_re)
  else:
    # imitates JAMR (97% sentece acc on AMR2.0)
    # split by these symbols
    # TODO: Do we really need to split by - ?
    sep_re = re.compile(r'[/~\*%\.,;:?!"\' \(\)\[\]\{\}-]')
    toks, pos = jamr_like_tokenizer(sentence_string, sep_re)

  return " ".join(toks)

def jamr_like_tokenizer(sentence_string, sep_re):
  # quote normalization
  try:
    sentence_string = sentence_string.replace('``', '"')
  except AttributeError:
    breakpoint()
  sentence_string = sentence_string.replace("''", '"')
  sentence_string = sentence_string.replace("“", '"')

  # currency normalization
  # sentence_string = sentence_string.replace("£", 'GBP')

  # Do not split these strings
  protected_re = re.compile("|".join([
    # URLs (this conflicts with many other cases, we should normalize URLs
    # a priri both on text and AMR)
    # r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}
    # ([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
    #
    r'[0-9][0-9,\.:/-]+[0-9]',  # quantities, time, dates
    r'^[0-9][\.](?!\w)',  # enumerate
    r'\b[A-Za-z][\.](?!\w)',  # itemize
    r'\b([A-Z]\.)+[A-Z]?',  # acronym with periods (e.g. U.S.)
    r'!+|\?+|-+|\.+',  # emphatic
    r'etc\.|i\.e\.|e\.g\.|v\.s\.|p\.s\.|ex\.',  # latin abbreviations
    r'\b[Nn]o\.|\bUS\$|\b[Mm]r\.',  # ...
    r'\b[Mm]s\.|\bSt\.|\bsr\.|a\.m\.',  # other abbreviations
    r':\)|:\(',  # basic emoticons
    # contractions
    r'[A-Za-z]+\'[A-Za-z]{3,}',  # quotes inside words
    r'n\'t(?!\w)',  # negative contraction (needed?)
    r'\'m(?!\w)',  # other contractions
    r'\'ve(?!\w)',  # other contractions
    r'\'ll(?!\w)',  # other contractions
    r'\'d(?!\w)',  # other contractions
    # r'\'t(?!\w)'                     # other contractions
    r'\'re(?!\w)',  # other contractions
    r'\'s(?!\w)',  # saxon genitive
    #
    r'<<|>>',  # weird symbols
    #
    r'Al-[a-zA-z]+|al-[a-zA-z]+',  # Arabic article
    # months
    r'Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.'
  ]))

  # iterate over protected sequences, tokenize unprotected and append
  # protected strings
  tokens = []
  positions = []
  start = 0
  for point in protected_re.finditer(sentence_string):

    # extract preceeding and protected strings
    end = point.start()
    preceeding_str = sentence_string[start:end]
    protected_str = sentence_string[end:point.end()]

    if preceeding_str:
      # tokenize preceeding string keep protected string as is
      for token, (start2, end2) in zip(
              *simple_tokenizer(preceeding_str, sep_re)
      ):
        tokens.append(token)
        positions.append((start + start2, start + end2))
    tokens.append(protected_str)
    positions.append((end, point.end()))

    # move cursor
    start = point.end()

  # Termination
  end = len(sentence_string)
  if start < end:
    ending_str = sentence_string[start:end]
    if ending_str.strip():
      for token, (start2, end2) in zip(
              *simple_tokenizer(ending_str, sep_re)
      ):
        tokens.append(token)
        positions.append((start + start2, start + end2))

  return tokens, positions

def simple_tokenizer(sentence_string, separator_re):

  tokens = []
  positions = []
  start = 0
  for point in separator_re.finditer(sentence_string):

    end = point.start()
    token = sentence_string[start:end]
    separator = sentence_string[end:point.end()]

    # Add token if not empty
    if token.strip():
      tokens.append(token)
      positions.append((start, end))

    # Add separator
    if separator.strip():
      tokens.append(separator)
      positions.append((end, point.end()))

    # move cursor
    start = point.end()

  # Termination
  end = len(sentence_string)
  if start < end:
    token = sentence_string[start:end]
    if token.strip():
      tokens.append(token)
      positions.append((start, end))

  return tokens, positions
