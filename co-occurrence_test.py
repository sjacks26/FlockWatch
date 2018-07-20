import nltk
import pandas as pd
import time
import datetime
from scipy import sparse
import logging
import sys
import os
import json

import config as cfg

collection_name = 'ConspiracyTheories'
collection_terms = ['qanon', 'pizzagate', 'whoisq', 'sethrich']

text = pd.read_csv('/Users/samjackson/Google Drive/Projects/In Progress/Vox pol RMP/work/data/Flock_test_data.csv')

text = text['text'].str.replace('@','').str.replace('#','').str.lower()
text.reset_index(inplace=True, drop=True)
tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
tokens = [tknzr.tokenize(t) for t in text]
clean_texts = [w for t in tokens for w in t if (not w in stops_w_collection_terms and w.isalnum() and len(w) >= cfg.minimum_token_length)]
clean_tokens = list(set(clean_texts))
logging.info("Found {0} tokens that might occur with collection terms.".format(len(clean_tokens)))
# one_cooccurrence_value = 1/len(tokens)
'''
one_cooccurrence_value assumes that the cooccurrence rate we care about is relative to the whole body of messages, not to the number of times a particular term appears. I don't think that's right.
The next block changes this. Instead, the value per cooccurrence between a token and a collection term is 1 divided by the number of documents that contain that collection term.
'''
cooccurrence_values = {}
for c in collection_terms:
    count = text.str.contains(c).sum()
    if count > (.001 * len(text)) and 1/count < (cfg.co_occurrence_threshold/10):
        cooccurrence_values[c] = 1/count
        logging.info("Value for one co-occurrence with {0} is {1}.".format(c, cooccurrence_values[c]))
    elif count == 0:
        logging.info("No messages with {0} found. Not looking for co-occurrences with this term.".format(c))
    elif count > 0 and 1/count >= (cfg.co_occurrence_threshold/10):
        logging.info("Too few messages with {0} found. Not looking for co-occurrences with this term.".format(c))
good_collects = list(cooccurrence_values.keys())
good_collect_ngrams = [f for f in good_collects if ' ' in f]
co_matrix = sparse.lil_matrix((len(good_collects), len(clean_tokens)))
clean_messages = [[t for t in w if (not t in stops and t.isalnum() and len(t) >= cfg.minimum_token_length)] for w in tokens]
for i in range(0, len(clean_messages)):
    start_time = time.time()
    clean_message = clean_messages[i]
    present_collection_terms = list(set(good_collects).intersection(clean_message))
    present_collect_ngrams = [f for f in good_collect_ngrams if f in text[i]]
    present_collection_terms.extend(present_collect_ngrams)
    collection_term_indexes = {}
    for c in present_collection_terms:
        c_term_index = good_collects.index(c)
        collection_term_indexes[c] = c_term_index
        message = list(set(clean_message))
        if c in present_collect_ngrams:
            c_tokens = c.split(' ')
            for t in c_tokens:
                if t in message:
                    message.remove(t)
        '''
        for token in message:
            if token in clean_tokens:
                token_index = clean_tokens.index(token)
                co_matrix[c_term_index, token_index] += cooccurrence_values[c]
        '''
    token_indexes = []
    for token in message:
        if token in clean_tokens:
            token_index = clean_tokens.index(token)
            token_indexes.append(token_index)
    for c in collection_term_indexes:
        for token in token_indexes:
            co_matrix[collection_term_indexes[c], token] += cooccurrence_values[c]
co_matrix_df = pd.DataFrame(co_matrix.todense())
co_matrix_df.columns = clean_tokens
new_index = dict(enumerate(good_collects))
co_matrix_df = co_matrix_df.rename(index=new_index)
co_matrix_df = co_matrix_df.transpose()