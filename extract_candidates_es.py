# coding: utf-8

# In[1]:

import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import cPickle

from elasticsearch import Elasticsearch, helpers

# In[2]:

with open('tmp/df_train_folds.bin', 'rb') as f:
    df_train = cPickle.load(f)

train_users = set(df_train.user_1) | set(df_train.user_2) 
train_idx = sorted(train_users)

fold1 = df_train[df_train.fold == 1]
fold1_users = set(fold1.user_1) | set(fold1.user_2)

fold2 = df_train[df_train.fold == 2]
fold2_users = set(fold2.user_1) | set(fold2.user_2)


# In[11]:

components = []
uid_to_others = {}
for _, group in tqdm(df_train.groupby('component')):
    users = set(group.user_1) | set(group.user_2)
    components.append(users)
    for uid in users:
        uid_to_others[uid] = users - {uid}


# In[3]:

TRAIN_1 = 1
TRAIN_2 = 2
TEST = 3

def user_fold(uid):
    if uid in fold1_users:
        return TRAIN_1
    if uid in fold2_users:
        return TRAIN_2
    return TEST


# In[8]:

es_host = '172.17.0.2'
es = Elasticsearch(host=es_host)

# In[5]:

def find_similar(uid, limit=10):
    query = {
        'query': {
            'filtered': {
                'query': {
                    'more_like_this': {
                        'like': {
                            '_index': 'user',
                            '_type': 'user_log',
                            '_id': int(uid),
                        },
                        'max_query_terms': 10,
                        'fields': ['fact.domain', 'fact.address', 'fact.param', 'fact.title^2'],
                    }
                }
            }
        },
        'filter': {
            'bool': {
                'must': [{
                    'term': {
                        'fold': user_fold(uid),
                    },
                }],
                
            }
        },
        'fields': ['_id'],
        'size': limit,
    }

    res = es.search(index='user', doc_type='user_log', body=query)
    hits = res['hits']['hits']
    return [(int(d['_id']), d['_score']) for d in hits]


# In[ ]:

train_pairs = []

for uid in tqdm(train_users):
    similar = find_similar(uid, limit=70)
    others_truth = uid_to_others[uid]
    fold = user_fold(uid)
    train_pairs.extend((uid, u, score, u in others_truth, fold) for (u, score) in similar)


# In[ ]:

train_pairs = pd.DataFrame(train_pairs)
train_pairs.columns = ['user_1', 'user_2', 'es_score', 'target', 'fold']


# In[ ]:

train_pairs.user_1 = train_pairs.user_1.astype('uint32')
train_pairs.user_2 = train_pairs.user_2.astype('uint32')
train_pairs.target = train_pairs.target.astype('uint8')
train_pairs.fold = train_pairs.fold.astype('uint8')
train_pairs.es_score = train_pairs.es_score.astype('float32')


with open('tmp/es-retrieved-70.bin', 'wb') as f:
    cPickle.dump(train_pairs, f)


# In[ ]:

test_users = set(range(339405)) - train_users

test_pairs = []

for uid in tqdm(test_users):
    similar = find_similar(uid, limit=70)
    test_pairs.extend((uid, u, score, TEST) for (u, score) in similar)


# In[ ]:

test_pairs = pd.DataFrame(test_pairs)
test_pairs.columns = ['user_1', 'user_2', 'es_score', 'fold']
test_pairs.user_1 = test_pairs.user_1.astype('uint32')
test_pairs.user_2 = test_pairs.user_2.astype('uint32')
test_pairs.fold = test_pairs.fold.astype('uint8')
test_pairs.es_score = test_pairs.es_score.astype('float32')


# In[ ]:

with open('tmp/es-retrieved-70_test.bin', 'wb') as f:
    cPickle.dump(test_pairs, f)
