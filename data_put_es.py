# coding: utf-8

import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import cPickle


# In[3]:

def read_json(line):
    d = json.loads(line)
    user_facts = []
    for f in d['facts']:
        fid = f['fid'] - 1
        ts = f['ts']
        if ts > 1000000000000000:
            user_facts.append((fid, ts / 1000))
        else:
            user_facts.append((fid, ts))
    return user_facts


# In[4]:

with open('tmp/df_train_folds.bin', 'rb') as f:
    df_train = cPickle.load(f)

train_users = set(df_train.user_1) | set(df_train.user_2) 
train_idx = sorted(train_users)

components = []
uid_to_others = {}
for _, group in tqdm(df_train.groupby('component')):
    users = set(group.user_1) | set(group.user_2)
    components.append(users)
    for uid in users:
        uid_to_others[uid] = users - {uid}


# In[5]:

train_users = set(df_train.user_1) | set(df_train.user_2) 
train_idx = sorted(train_users)

fold1 = df_train[df_train.fold == 1]
fold1_users = set(fold1.user_1) | set(fold1.user_2)

fold2 = df_train[df_train.fold == 2]
fold2_users = set(fold2.user_1) | set(fold2.user_2)


# In[6]:

TRAIN_1 = 1
TRAIN_2 = 2
TEST = 3

def user_fold(uid):
    if uid in fold1_users:
        return TRAIN_1
    if uid in fold2_users:
        return TRAIN_2
    return TEST

# In[7]:

with open('tmp/df_urls.bin', 'rb') as f:
    df_urls = cPickle.load(f)

# In[25]:

from elasticsearch import Elasticsearch, helpers
es_host = '172.17.0.2'
es = Elasticsearch(host=es_host)


# In[27]:

from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Mapping, String, Nested, Integer, Boolean
from elasticsearch_dsl import analyzer, tokenizer

whitespace_analyzer = analyzer('whitespace_analyzer', tokenizer=tokenizer('whitespace'))
con = connections.create_connection(host=es_host)

mapping = Mapping('user_log')

fact = Nested(multi=True, include_in_parent=True)
fact.field('domain', String(analyzer=whitespace_analyzer))
fact.field('address', String(analyzer=whitespace_analyzer))
fact.field('param', String(analyzer=whitespace_analyzer))
fact.field('title', String(analyzer=whitespace_analyzer))

mapping.field('fact', fact)
mapping.field('fold', Integer(index='not_analyzed'))

mapping.save('user')


# In[29]:

import itertools 

def chunk_iterator(iterator, size):
    while 1:
        batch = list(itertools.islice(iterator, size))
        if batch:
            yield batch
        else:
            break


# In[32]:

uid_idx = 0

with open('../data/facts.json', 'r') as fact_file:
    lines = iter(fact_file)

    for chunk in tqdm(chunk_iterator(lines, 100)):
        actions = []

        for line in chunk:
            log = read_json(line)
            facts, _ = zip(*log)
            facts = df_urls.iloc[list(facts)]
            user = {
                'fact': facts.to_dict(orient='records'), 
                'fold': user_fold(uid_idx),
            }
            action = {'_id': uid_idx, '_index': 'user', '_type': 'user_log', '_source': user}
            actions.append(action)
            uid_idx = uid_idx + 1

        helpers.bulk(es, actions)