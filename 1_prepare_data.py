# coding: utf-8

# In[1]:

import json

import pandas as pd
import numpy as np
import networkx as nx

from tqdm import tqdm

import cPickle

from operator import itemgetter
itemgetter_1 = itemgetter(1)


# In[2]:

def process_facts(fact_dicts):
    global next_session_id
    facts = []

    for d in fact_dicts:
        fid = d['fid'] - 1
        ts = d['ts']
        if ts > 1000000000000000:
            facts.append((fid, ts / 1000))
        else:
            facts.append((fid, ts))

    facts = sorted(facts, key=itemgetter_1)
    return facts

# In[3]:

user_ids = []

with open('../data/facts.json', 'r') as fact_file:
    for line in tqdm(fact_file):
        d = json.loads(line)
        user_ids.append(d['uid'])

# In[5]:
uid_to_idx = {uid: i for (i, uid) in enumerate(user_ids)}

with open('tmp/uid_to_idx.bin', 'wb') as f:
    cPickle.dump(uid_to_idx, f)
with open('tmp/idx_to_uid.bin', 'wb') as f:
    cPickle.dump(user_ids, f)


# In[6]:

df_train = pd.read_csv('../data/train.csv', header=None)
df_train.columns = ['user_1', 'user_2']

df_train.user_1 = df_train.user_1.apply(uid_to_idx.get)
df_train.user_2 = df_train.user_2.apply(uid_to_idx.get)


train_users = set(df_train.user_1) | set(df_train.user_2) 
test_users = set(range(0, len(user_ids))) - train_users

# In[9]:

with open('tmp/train_test_users.bin', 'wb') as f:
    cPickle.dump((train_users, test_users), f)


# In[10]:

G = nx.Graph()

for u1, u2 in tqdm(zip(df_train.user_1, df_train.user_2)):
    G.add_edge(u1, u2)

components = nx.connected_components(G)
node_to_comp = {}
for idx, component in enumerate(components):
    for node in component:
        node_to_comp[node] = idx

df_train['component'] = df_train.user_1.apply(node_to_comp.get)


# In[]:

np.random.seed(2)

num_components = df_train.component.max()
component_idx = np.arange(0, num_components)
np.random.shuffle(component_idx)

split = num_components / 2
fold1_comps = set(component_idx[:split])

TRAIN_1 = 1
TRAIN_2 = 2

df_train['fold'] = TRAIN_2
is_fold1 = df_train.component.isin(fold1_comps)
df_train['fold'][is_fold1] = TRAIN_1
df_train.user_1 = df_train.user_1.astype('uint32')
df_train.user_2 = df_train.user_2.astype('uint32')
df_train.component = df_train.component.astype('uint32')
df_train.fold = df_train.fold.astype('uint8')

# In[11]:

with open('tmp/df_train_folds.bin', 'wb') as f:
    cPickle.dump(df_train, f)


# In[8]:
## Preparing facts

cnt = 0
code_dict = {}

def encode(token):
    if token is None:
        return ''

    if token in code_dict:
        code = code_dict[token]
        return np.base_repr(code, base=36)

    global cnt
    code = cnt
    code_dict[token] = code
    cnt = cnt + 1
    return np.base_repr(code, base=36)


# In[9]:

titles = {}

with open('../data/titles.csv', 'r') as f:
    for line in tqdm(f):
        uid, title = line.strip().split(',')
        tokens = title.split(' ')
        code_tokens = [encode(t) for t in tokens]
    
        titles[int(uid)] = ' '.join(code_tokens)


# In[10]:

url_dicts = []

with open('../data/urls.csv', 'r') as f:
    for line in tqdm(f):
        url_id, url = line.strip().split(',')
        url_id = int(url_id)

        param = None
        if '?' in url:
            url, param = url.split('?')

        url_tokens = url.split('/')
        url_tokens_code = [encode(t) for t in url_tokens]

        param = encode(param)
        domain = url_tokens_code[0]
        address = url_tokens_code[1:]

        row_dict = {'url_id': url_id, 'domain': domain, 
                    'address': ' '.join(address), 'param': param,
                    'title': titles.get(url_id, '') }
        url_dicts.append(row_dict)


# In[17]:

df_urls = pd.DataFrame(url_dicts, columns=['url_id', 'domain', 'address', 'param', 'title'])
df_urls.sort_values(by='url_id', inplace=1)
df_urls.reset_index(drop=True, inplace=1)
df_urls.param.fillna('', inplace=1)
df_urls.drop('url_id', axis=1, inplace=1)

# In[25]:

with open('tmp/df_urls.bin', 'wb') as f:
    cPickle.dump(df_urls, f)


