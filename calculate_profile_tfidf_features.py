
# coding: utf-8

# In[103]:

import json

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import normalize

import cPickle


# In[2]:

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


# In[3]:

print 'reading urls...'

with open('tmp/df_urls.bin', 'rb') as f:
    df_urls = cPickle.load(f)
df_urls.head()


# In[4]:

df_urls['url'] = df_urls.domain + ' ' + df_urls.address + ' ' + df_urls.param
del df_urls['address']
del df_urls['param']



# In[5]:

print 'reading json facts...'


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


# In[6]:

#users = []
user_urls = []
user_domains = []
user_titles = []

with open('../data/facts.json', 'r') as fact_file:
    for line in tqdm(fact_file):
        user_facts = read_json(line)
        # users.append(user_facts)

        fids = [f for (f, _) in user_facts]
        urls = df_urls.iloc[fids]

        user_urls.append(' '.join(urls.url))
        user_domains.append(' '.join(urls.domain))
        titles = urls[urls.title != ''].title
        user_titles.append(' '.join(titles))


# In[56]:

df_tokens = pd.DataFrame({'user_urls': user_urls,
                          'user_domains': user_domains,
                          'user_titles': user_titles})

# In[57]:

del user_urls, user_domains, user_titles
del df_urls

# In[37]:


# In[41]:
print 'reading profiles...'

df_profiles = pd.read_csv('./user_profiles.txt', dtype='float32')

del df_profiles['domain_cos_amax']
del df_profiles['domain_svd_cos_amax']
del df_profiles['title_cos_amax']
del df_profiles['title_svd_cos_amax']
del df_profiles['url_cos_amax']
del df_profiles['url_svd_cos_amax']


# In[51]:

print 'reading TFIDF and SVD models...'

tfidf_models_files = [
    'session_domain_tfidf.bin', 
    'session_domain_tfidf_svd.bin', 
    'session_title_tfidf.bin',
    'session_title_tfidf_svd.bin',
    'session_url_tfidf.bin',
    'session_url_tfidf_svd.bin']

tfidf_models = {}
for model_file in tfidf_models_files:
    name = model_file.strip('.bin')
    print 'loading %s...' % model_file

    with open('tmp/' + model_file, 'rb') as f:
        tfidf_models[name] = cPickle.load(f)


# In[52]:

domain_tfidf = tfidf_models['session_domain_tfidf']
domain_tfidf_svd = tfidf_models['session_domain_tfidf_svd']
title_tfidf = tfidf_models['session_title_tfidf']
title_tfidf_svd = tfidf_models['session_title_tfidf_svd']
url_tfidf = tfidf_models['session_url_tfidf']
url_tfidf_svd = tfidf_models['session_url_tfidf_svd']


# In[61]:

def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]


# In[65]:

import os

def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)      

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)


# In[117]:

def process_batch(batch):
    batch = batch.reset_index(drop=1)
    
    user1_prof = df_profiles.iloc[batch.user_1].reset_index(drop=1)
    user2_prof = df_profiles.iloc[batch.user_2].reset_index(drop=1)
    profile_diff = (user1_prof - user2_prof).abs()


    profile_tok = pd.DataFrame()

    tokens1 = df_tokens.iloc[batch.user_1].reset_index(drop=1)
    tokens2 = df_tokens.iloc[batch.user_2].reset_index(drop=1)

    dom1 = domain_tfidf.transform(tokens1.user_domains)
    dom2 = domain_tfidf.transform(tokens2.user_domains)

    dom_sim = dom1.multiply(dom2).sum(axis=1)
    profile_tok['domain_tfidf_sim'] = np.asarray(dom_sim).reshape(-1)

    dom1_svd = normalize(domain_tfidf_svd.transform(dom1))
    dom2_svd = normalize(domain_tfidf_svd.transform(dom2))

    profile_tok['domain_tfidf_svd_sim'] = (dom1_svd * dom2_svd).sum(axis=1)


    urls1 = url_tfidf.transform(tokens1.user_urls)
    urls2 = url_tfidf.transform(tokens2.user_urls)

    url_sim = urls1.multiply(urls2).sum(axis=1)
    profile_tok['url_tfidf_sim'] = np.asarray(url_sim).reshape(-1)

    urls1_svd = normalize(url_tfidf_svd.transform(urls1))
    urls2_svd = normalize(url_tfidf_svd.transform(urls2))

    profile_tok['url_tfidf_svd_sim'] = (urls1_svd * urls2_svd).sum(axis=1)


    titles1 = title_tfidf.transform(tokens1.user_titles)
    titles2 = title_tfidf.transform(tokens2.user_titles)

    title_sim = titles1.multiply(titles2).sum(axis=1)
    profile_tok['title_tfidf_sim'] = np.asarray(title_sim).reshape(-1)

    titles1_svd = normalize(title_tfidf_svd.transform(titles1))
    titles2_svd = normalize(title_tfidf_svd.transform(titles2))

    profile_tok['title_tfidf_svd_sim'] = (titles1_svd * titles2_svd).sum(axis=1)
    
    return pd.concat([batch, profile_diff, profile_tok], axis=1)



# In[121]:


print 'reading train from elastic search...'

with open('tmp/es-retrieved-nodup-50-more.bin', 'rb') as f:
    df_train_pairs = cPickle.load(f)

print 'calculating features for train...'
train_file = 'pair_features_tfidf_profiles_test.csv'
delete_file_if_exists(train_file)

df = df_train_pairs
batch_size = 10000

for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
    batch = process_batch(batch)
    append_to_csv(batch, train_file)

del df, df_train_pairs


print 'reading test from elastic search...'

with open('tmp/es-retrieved-nodup-50_test.bin', 'rb') as f:
    df_test_pairs = cPickle.load(f)

print 'calculating features for test...'
train_file = 'pair_features_tfidf_profiles_test.csv'

delete_file_if_exists(train_file)

df = df_test_pairs
batch_size = 10000

for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
    batch = process_batch(batch)
    append_to_csv(batch, train_file)