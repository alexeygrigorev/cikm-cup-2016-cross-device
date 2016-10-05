# coding: utf-8

import json

import pandas as pd
import numpy as np
from tqdm import tqdm

import xgboost as xgb


# In[ ]:

df_train = pd.read_csv('./pair_features_tfidf_profiles_train.csv', dtype='float32')
df_train.fold = df_train.fold.astype('uint8')
df_train.target = df_train.target.astype('uint8').values


# In[ ]:

users12 = set(zip(df_train.user_1, df_train.user_2))
pd_users21 = pd.Series(zip(df_train.user_2, df_train.user_1))
isin_21 = pd_users21.isin(users12)

users21 = set(zip(df_train.user_2, df_train.user_1))
pd_users12 = pd.Series(zip(df_train.user_1, df_train.user_2))
isin_12 = pd_users12.isin(users21)


df_train['is_duplicate'] = isin_12 | isin_21
df_train.is_duplicate = df_train.is_duplicate.astype('uint8')

del users12, pd_users21, users21, pd_users12
del isin_12, isin_21


# In[4]:

from sklearn.metrics import roc_auc_score
print 'baseline auc:', roc_auc_score(df_train.target, df_train.es_score)


# In[18]:

fold = df_train.fold
target = df_train.target

del df_train['user_1'], df_train['user_2']
del df_train['fold'], df_train['target']

X_all = df_train.values
feature_names = list(df_train.columns)
del df_train

# In[16]:

n_estimators = 1000

xgb_pars = {
    'eta': 0.15,
    'gamma': 0.5,
    'max_depth': 6,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 42,
    'silent': 1
}


dfull = xgb.DMatrix(X_all, target, feature_names=feature_names, missing=np.nan)
del X_all, target


# In[27]:

watchlist = [(dfull, 'train')]
model_full = xgb.train(xgb_pars, dfull, num_boost_round=n_estimators, 
    verbose_eval=5, evals=watchlist)

# In[29]:

df_test = pd.read_csv('./pair_features_tfidf_profiles_test.csv', dtype='float32')

users21 = set(zip(df_test.user_2, df_test.user_1))
pd_users12 = pd.Series(zip(df_test.user_1, df_test.user_2))
isin_12 = pd_users12.isin(users21)

users12 = set(zip(df_test.user_1, df_test.user_2))
pd_users21 = pd.Series(zip(df_test.user_2, df_test.user_1))
isin_21 = pd_users21.isin(users12)


# In[63]:

df_test['is_duplicate'] = isin_12 | isin_21
df_test.is_duplicate = df_test.is_duplicate.astype('uint8')
del users12, pd_users21, users21, pd_users12, isin_12, isin_21

df_test = df_test[feature_names].copy()
X_test = df_test.values
del df_test


# In[39]:
test_xgb = xgb.DMatrix(X_test, feature_names=feature_names, missing=np.nan)
del X_test

# In[42]:

df_perf_test = pd.read_csv('pair_features_tfidf_profiles_test.csv', 
                      dtype={'user_1': 'uint32', 'user_2': 'uint32'}, 
                      usecols=['user_1', 'user_2'])

pred_test = model_full.predict(test_xgb)
df_perf_test['model_score'] = pred_test
df_perf_test.model_score = df_perf_test.model_score.astype('float32')


## result

pairs = [sorted(p) for p in zip(df_perf_test.user_1, df_perf_test.user_2)]
pairs = np.array(pairs)
df_perf_test.user_1 = pairs[:, 0]
df_perf_test.user_2 = pairs[:, 1]
del pairs

df_perf_test.sort_values(by='model_score', ascending=0, inplace=1)

with open('tmp/idx_to_uid.bin', 'rb') as f:
    idx_to_uid = cPickle.load(f)


## graph completion


df_test_dedup = []
seen = set()
count = 215307 * 0.5

for _, row in df_perf_test.iterrows():
    uid1 = int(row.user_1)
    uid2 = int(row.user_2)
    if (uid1, uid2) in seen:
        continue

    seen.add((uid1, uid2))
    df_test_dedup.append((uid1, uid2, row.model_score))

    count = count - 1
    if count <= 0:
        break

df_test_dedup = pd.DataFrame(df_test_dedup)
df_test_dedup.columns = ['user_1', 'user_2', 'model_score']

df_certain = df_test_dedup[df_test_dedup.avg_score >= 0.8]



import networkx as nx
import itertools

G = nx.Graph()

for u1, u2 in tqdm(zip(df_certain.user_1, df_certain.user_2)):
    G.add_edge(u1, u2)

components = list(nx.connected_components(G))

certain_combinations = []

for cmb in tqdm(components):
    if len(cmb) >= 20:
        continue
    for (u1, u2) in list(itertools.combinations(cmb, r=2)):
        certain_combinations.append((u1, u2))


## final submission

seen = set()

with open('submission.txt', 'w') as f_out:
    for uid1, uid2 in certain_combinations:
        seen.add((uid1, uid2))

        uid1 = idx_to_uid[uid1]
        uid2 = idx_to_uid[uid2]
        if uid2 < uid1:
            uid2, uid1 = uid1, uid2

        f_out.write("%s,%s\n" % (uid1, uid2))

    cnt = len(df_test_dedup) - 5000

    for uid1, uid2 in tqdm(zip(df_test_dedup.user_1, df_test_dedup.user_2)):
        cnt = cnt - 1
        if cnt < 0:
            break
        if (uid1, uid2) in seen or (uid2, uid1) in seen:
            continue
        seen.add((uid1, uid2))

        uid1 = idx_to_uid[uid1]
        uid2 = idx_to_uid[uid2]

        if uid2 < uid1:
            uid2, uid1 = uid1, uid2

        f_out.write("%s,%s\n" % (uid1, uid2))
