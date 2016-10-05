# coding: utf-8

import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import cPickle

from concurrent.futures import ProcessPoolExecutor
import itertools 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# TF-IDF features per user session

# In[2]:

with open('tmp/df_urls.bin', 'rb') as f:
    df_urls = cPickle.load(f)

df_urls.head()


# In[3]:

half_hour = 30 * 60 * 1000

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


def extract_facts(facts):
    df_facts = pd.DataFrame(facts)
    df_facts.columns = ['fid', 'ts']

    df_facts.sort_values(by='ts', inplace=1)
    df_facts['delta'] = df_facts.ts - df_facts.ts.shift()

    session_change = df_facts.delta >= half_hour
    df_facts['session_id'] = session_change.cumsum().astype(int)
    df_facts.delta.fillna(0, inplace=1)
    df_facts.delta = df_facts.delta.astype(int)
    df_facts.reset_index(drop=True, inplace=1)
    
    df_facts.fid = df_facts.fid.astype('uint32')
    df_facts.delta = df_facts.delta.astype('uint32')
    df_facts.session_id = df_facts.session_id.astype('uint16')
    return df_facts

def process(line):
    raw_clickstream = read_json(line)
    return extract_facts(raw_clickstream)


# In[4]:

def chunk_iterator(iterator, size):
    while 1:
        batch = list(itertools.islice(iterator, size))
        if batch:
            yield batch
        else:
            break


# In[5]:

def extract_session_info(df):
    session_urls = []
    session_domains = []
    session_titles = []

    for _, session in df.groupby('session_id'):
        ses_urls = df_urls.iloc[session.fid]
        urls = ses_urls.domain + ' ' + ses_urls.address + ' ' + ses_urls.param
        session_urls.append(' '.join(urls))

        titles = ses_urls.title[ses_urls.title != '']
        session_titles.append(' '.join(titles))

        session_domains.append(' '.join(ses_urls.domain))
    return session_urls, session_domains, session_titles


# In[6]:

session_urls = open('tmp/session-urls.txt', 'w')
session_domains = open('tmp/session-domains.txt', 'w')
session_titles = open('tmp/session-titles.txt', 'w')

with ProcessPoolExecutor(max_workers=8) as pool:
    with open('../data/facts.json', 'r') as fact_file:
        lines = iter(fact_file)

        for json_chunk in tqdm(chunk_iterator(lines, 200)):
            user_logs = pool.map(process, json_chunk)
            udt = pool.map(extract_session_info, user_logs)

            for urls, domains, titles in udt:
                for url in urls:
                    session_urls.write(url)
                    session_urls.write('\n')
            
                for domain in domains: 
                    session_domains.write(domain)
                    session_domains.write('\n')

                for title in titles:
                    if not title:
                        continue
                    session_titles.write(title)
                    session_titles.write('\n')

            session_urls.flush()
            session_domains.flush()            
            session_titles.flush()

session_urls.close()
session_domains.close()
session_titles.close()


# In[13]:

del json_chunk, urls, domains, titles, udt, lines, df_urls


# In[7]:

with open('tmp/session-urls.txt', 'r') as f:
    session_urls = []
    for line in tqdm(f):
        session_urls.append(line.strip())


# In[11]:

session_url_tfidf = TfidfVectorizer(min_df=5)
session_url_matrix = session_url_tfidf.fit_transform(session_urls)
del session_urls


# In[15]:

with open('tmp/session_url_tfidf.bin', 'wb') as f:
    cPickle.dump(session_url_tfidf, f)

del session_url_tfidf


# In[17]:

session_url_svd = TruncatedSVD(n_components=150, random_state=1)
session_url_svd.fit(session_url_matrix)

del session_url_matrix


# In[19]:

with open('tmp/session_url_tfidf_svd.bin', 'wb') as f:
    cPickle.dump(session_url_svd, f)

del session_url_svd


# In[21]:

with open('tmp/session-domains.txt', 'r') as f:
    session_domains = []
    for line in tqdm(f):
        session_domains.append(line.strip())


# In[23]:

session_domain_tfidf = TfidfVectorizer(min_df=5)
session_domain_matrix = session_domain_tfidf.fit_transform(session_domains)

del session_domains


# In[26]:

with open('tmp/session_domain_tfidf.bin', 'wb') as f:
    cPickle.dump(session_domain_tfidf, f)

del session_domain_tfidf


# In[28]:

session_domain_svd = TruncatedSVD(n_components=70, random_state=1)
session_domain_svd.fit(session_domain_matrix)


# In[29]:

with open('tmp/session_domain_tfidf_svd.bin', 'wb') as f:
    cPickle.dump(session_domain_svd, f)

del session_domain_matrix, session_domain_svd


# In[31]:

with open('tmp/session-titles.txt', 'r') as f:
    session_titles = []
    for line in tqdm(f):
        session_titles.append(line.strip())

session_title_tfidf = TfidfVectorizer(min_df=5)
session_title_matrix = session_title_tfidf.fit_transform(session_titles)


# In[33]:

with open('tmp/session_title_tfidf.bin', 'wb') as f:
    cPickle.dump(session_title_tfidf, f)

del session_title_tfidf


# In[35]:

session_title_svd = TruncatedSVD(n_components=70, random_state=1)
session_title_svd.fit(session_title_matrix)

del session_title_matrix


# In[37]:

with open('tmp/session_title_tfidf_svd.bin', 'wb') as f:
    cPickle.dump(session_title_svd, f)
