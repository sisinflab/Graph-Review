# CREDITS: https://github.com/chenchongthu/NARRE/blob/master/pro_data/loaddata.py

'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import os
import json
import pandas as pd
import pickle
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Run load data.")
parser.add_argument('--dataset', type=str, default='music')
parser.add_argument('--filename', type=str, default='Digital_Music_5.json')
args = parser.parse_args()

TPS_DIR = f'../../../data/{args.dataset}'
TP_file = os.path.join(TPS_DIR, args.filename)

f= open(TP_file)
users_id=[]
items_id=[]
ratings=[]
reviews=[]
np.random.seed(2017)

for line in f:
    js=json.loads(line)
    if str(js['reviewerID'])=='unknown':
        print("unknown")
        continue
    if str(js['asin'])=='unknown':
        print("unknown2")
        continue
    try:
        reviews.append(js['reviewText'])
        users_id.append(str(js['reviewerID']) + ',')
        items_id.append(str(js['asin']) + ',')
        ratings.append(str(js['overall']))
    except:
        continue
data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]

data.drop_duplicates(subset=['user_id', 'item_id'], inplace=True)

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
unique_uid = usercount['user_id'].unique()
unique_sid = itemcount['item_id'].unique()
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
def numerize(tp):
    uid = map(lambda x: user2id[x], tp['user_id'])
    sid = map(lambda x: item2id[x], tp['item_id'])
    tp['user_id'] = list(uid)
    tp['item_id'] = list(sid)
    return tp

data=numerize(data)
tp_rating=data[['user_id','item_id','ratings']]


n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]  # test set
tp_train= tp_rating[~test_idx]  # training set

data2=data[test_idx]  # test set
data=data[~test_idx]  # training set


n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]

tp_train.to_csv(os.path.join(TPS_DIR, 'train.tsv'), index=False,header=None, sep='\t')
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.tsv'), index=False,header=None, sep='\t')
tp_test.to_csv(os.path.join(TPS_DIR, 'test.tsv'), index=False,header=None, sep='\t')

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}
for i in data.values:
    if i[0] in user_reviews.keys():
        user_reviews[i[0]].append(i[3])
        user_rid[i[0]].append(i[1])
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[i[3]]
    if i[1] in item_reviews.keys():
        item_reviews[i[1]].append(i[3])
        item_rid[i[1]].append(i[0])
    else:
        item_reviews[i[1]] = [i[3]]
        item_rid[i[1]]=[i[0]]

pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')


print(np.sort(np.array(usercount.values)))

print(np.sort(np.array(itemcount.values)))