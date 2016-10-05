# coding: utf-8

# In[1]:

import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import cPickle

from sklearn.preprocessing import normalize

from concurrent.futures import ProcessPoolExecutor
import os


# In[2]:

print 'loading urls...'

with open('tmp/df_urls.bin', 'rb') as f:
    df_urls = cPickle.load(f)


# In[3]:

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


# In[6]:

domain_tfidf = tfidf_models['session_domain_tfidf']
domain_tfidf_svd = tfidf_models['session_domain_tfidf_svd']
title_tfidf = tfidf_models['session_title_tfidf']
title_tfidf_svd = tfidf_models['session_title_tfidf_svd']
url_tfidf = tfidf_models['session_url_tfidf']
url_tfidf_svd = tfidf_models['session_url_tfidf_svd']


# In[7]:

df_urls['url'] = df_urls.domain + ' ' + df_urls.address + ' ' + df_urls.param
del df_urls['address']
del df_urls['param']


# In[8]:

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


# In[9]:

half_hour = 30 * 60 * 1000
desc_functions = [np.min, np.max, np.mean, np.std]


# In[11]:

def lower_tri(X):
    return X[np.tril_indices_from(X), k=-1]


# In[25]:

def extract_profile(facts):
    res = {}

    df_facts = pd.DataFrame(facts)
    df_facts.columns = ['fid', 'ts']

    df_facts.sort_values(by='ts', inplace=1)
    df_facts['delta'] = df_facts.ts - df_facts.ts.shift()

    session_change = df_facts.delta >= half_hour

    res['sessions_no'] = 1 + session_change.sum()

    session_id = session_change.cumsum()
    session_cnt = session_id.value_counts() 
    
    for f in desc_functions:
        res['no_clicks_%s' % f.func_name]  = f(session_cnt)

    res['singletons_ratio'] = (session_cnt == 1).mean()
    res['singletons_num'] = (session_cnt == 1).sum()

    session_breaks = df_facts[session_change].delta
    for f in desc_functions:
        res['breaks_%s' % f.func_name]  = f(session_breaks)

    tail = df_facts.iloc[1:]
    small_deltas = tail.delta <= 1
    res['small_deltas_cnt'] = small_deltas.sum()
    res['small_deltas_ratio'] = small_deltas.mean()

    session_starts = []
    session_ends = []
    session_durations = []
    session_deltas = []

    session_domains_counts = []
    
    session_titles = []
    session_domains = []
    session_urls = []
    
    
    for _, session in df_facts.groupby(session_id):
        start = session.iloc[0].ts
        session_starts.append(start)

        end = session.iloc[-1].ts
        session_ends.append(end)

        session_durations.append(end - start)

        delta = session.iloc[1:].delta
        session_deltas.append(delta)

        fids = list(session.fid)
        urls = df_urls.iloc[fids]

        session_domains_counts.append(len(set(urls.domain)))

        titles = ' '.join(urls.title[urls.title != ''])
        session_titles.append(titles)
        domain = ' '.join(urls.domain)
        session_domains.append(domain)
        urls = ' '.join(urls.url)
        session_urls.append(urls)

    title_matrix = title_tfidf.transform(session_titles)
    title_cos = (title_matrix * title_matrix.T).toarray()
    title_cos = lower_tri(title_cos)

    title_matrix_svd = title_tfidf_svd.transform(title_matrix)
    title_matrix_svd = normalize(title_matrix_svd)
    title_svd_cos = title_matrix_svd.dot(title_matrix_svd.T)
    title_svd_cos = lower_tri(title_svd_cos)

    url_matrix = url_tfidf.transform(session_urls)
    url_cos = (url_matrix * url_matrix.T).toarray()
    url_cos = lower_tri(url_cos)

    url_matrix_svd = url_tfidf_svd.transform(url_matrix)
    url_matrix_svd = normalize(url_matrix_svd)
    url_svd_cos = url_matrix_svd.dot(url_matrix_svd.T)
    url_svd_cos = lower_tri(url_svd_cos)

    domain_matrix = domain_tfidf.transform(session_domains)
    domain_cos = (domain_matrix * domain_matrix.T).toarray()
    domain_cos = lower_tri(domain_cos)

    domain_matrix_svd = domain_tfidf_svd.transform(domain_matrix)
    domain_matrix_svd = normalize(domain_matrix_svd)
    domain_svd_cos = domain_matrix_svd.dot(domain_matrix_svd.T)
    domain_svd_cos = lower_tri(domain_svd_cos)
    

    session_starts = pd.to_datetime(session_starts, unit='ms')
    start_hours = [ts.hour for ts in session_starts]
    session_ends = pd.to_datetime(session_ends, unit='ms')
    end_hours = [ts.hour for ts in session_ends]

    for f in desc_functions:
        res['start_hour_%s' % f.func_name]= f(start_hours)
        res['num_domains_%s' % f.func_name]= f(session_domains_counts)
        res['end_hour_%s' % f.func_name]= f(end_hours)
        res['duration_%s' % f.func_name]= f(session_durations)
        
        res['title_cos_%s' % f.func_name]= f(title_cos)
        res['title_svd_cos_%s' % f.func_name]= f(title_svd_cos)
        res['url_cos_%s' % f.func_name]= f(url_cos)
        res['url_svd_cos_%s' % f.func_name]= f(url_svd_cos)
        res['domain_cos_%s' % f.func_name]= f(domain_cos)
        res['domain_svd_cos_%s' % f.func_name]= f(domain_svd_cos)

    return res

def extract_profile_from_json(line):
    raw_clickstream = read_json(line)
    return extract_profile(raw_clickstream)


# In[13]:

import itertools 

def chunk_iterator(iterator, size):
    while 1:
        batch = list(itertools.islice(iterator, size))
        if batch:
            yield batch
        else:
            break


# In[5]:


def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)


# In[34]:

delete_file_if_exists('user_profiles.txt')


# In[29]:

batch_size = 200

with ProcessPoolExecutor(max_workers=5) as pool:
    with open('../data/facts.json', 'r') as fact_file:
        lines = iter(fact_file)

        for chunk in tqdm(chunk_iterator(lines, batch_size)):
            df_profile_chunk = pool.map(extract_profile_from_json, chunk)
            df_profile_chunk = pd.DataFrame(df_profile_chunk)
            append_to_csv(df_profile_chunk, 'user_profiles.txt')