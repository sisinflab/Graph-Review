from json import *
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import random
import argparse
import pandas as pd

random.seed(42)

nltk.download('punkt')

parser = argparse.ArgumentParser(description="Run preprocessing for RMG.")
parser.add_argument('--dataset', nargs='?', default='Digital_Music', help='dataset path')
parser.add_argument('--batch_size', default=128, help='batch size')
args = parser.parse_args()
dataset = args.dataset
batch_size = args.batch_size

with open(f'../../../data/{dataset}/{dataset}_5.json', 'r') as f:
    rawdata = [JSONDecoder().decode(x) for x in f.readlines()]

for i in range(len(rawdata)):
    rawdata[i]['text'] = [word_tokenize(x) for x in sent_tokenize(rawdata[i]['reviewText'].lower())]

word_dict = {'PADDING': [0, 999999]}
for i in rawdata:
    for k in i['text']:
        for j in k:
            if j in word_dict:
                word_dict[j][1] += 1
            else:
                word_dict[j] = [len(word_dict), 1]

word_dict_freq = {}
for x in word_dict:
    if word_dict[x][1] >= 10:
        word_dict_freq[x] = [len(word_dict_freq), word_dict[x][1]]
print(len(word_dict_freq), len(word_dict))

embdict={}
cnt=0
with open('../../../data/glove.840B.300d.txt','rb')as f:
    linenb=0
    while True:
        line=f.readline()
        if len(line)==0:
            break
        line = line.split()
        word=line[0].decode()
        linenb+=1
        if len(word) != 0:
            emb=[float(x) for x in line[1:]]
            if word in word_dict_freq:
                embdict[word]=emb
                if cnt%100==0:
                    print(cnt,linenb,word)
                cnt+=1


print(len(embdict),len(word_dict_freq))
print(len(word_dict_freq))


emb_mat=[0]*len(word_dict_freq)
xp=np.zeros(300,dtype='float32')

temp_emb=[]
for i in embdict.keys():
    emb_mat[word_dict_freq[i][0]]=np.array(embdict[i],dtype='float32')
    temp_emb.append(emb_mat[word_dict_freq[i][0]])
temp_emb=np.array(temp_emb,dtype='float32')

mu=np.mean(temp_emb, axis=0)
Sigma=np.cov(temp_emb.T)

norm=np.random.multivariate_normal(mu, Sigma, 1)

for i in range(len(emb_mat)):
    if type(emb_mat[i])==int:
        emb_mat[i]=np.reshape(norm, 300)
emb_mat[0]=np.zeros(300,dtype='float32')
emb_mat=np.array(emb_mat,dtype='float32')
print(emb_mat.shape)

np.save(f'../../../data/{dataset}/embed_vocabulary.npy', emb_mat)

uir_triples=[]
for i in rawdata:
    if i['reviewerID'] == 'unknown':
        print("unknown user id")
        continue
    if i['asin'] == 'unknown':
        print("unkown item id")
        continue
    if i['reviewText'] == '':
        continue
    try:
        temp = {}
        doc = []
        for y in i['text']:
            doc.append([word_dict_freq[x][0] for x in y if x in word_dict_freq])
        temp['text'] = doc
        temp['item'] = i['asin']
        temp['user'] = i['reviewerID']
        temp['label'] = i['overall']
        uir_triples.append(temp)
    except:
        continue

for i in range(len(uir_triples)):
    uir_triples[i]['id']=i

MAX_SENT_LENGTH = 20
MAX_SENTS = 10
MAX_REVIEW_USER = 13
MAX_REVIEW_ITEM = 24
MAX_NEIGHBOR = 75

data = pd.DataFrame(uir_triples)

uidList, iidList = data['user'].unique().tolist(), data['item'].unique().tolist()

user2id = dict((uid, i) for(i, uid) in enumerate(uidList))
item2id = dict((iid, i) for(i, iid) in enumerate(iidList))
uid = list(map(lambda x: user2id[x], data['user']))
iid = list(map(lambda x: item2id[x], data['item']))
data['user'] = uid
data['item'] = iid

train_np=np.load(f'../../../data/{dataset}/Train.npy')
val_np=np.load(f'../../../data/{dataset}/Val.npy')
test_np=np.load(f'../../../data/{dataset}/Test.npy')

train_uir = []
val_uir = []
test_uir = []

for row in train_np:
    train_uir.append({
        'text': data[(data['user'] == row[0]) & (data['item'] == row[1])]['text'].values,
        'item': row[1],
        'user': row[0],
        'id': data[(data['user'] == row[0]) & (data['item'] == row[1])]['id'].values[0],
        'label': data[(data['user'] == row[0]) & (data['item'] == row[1])]['label'].values[0]
    })

for row in val_np:
    val_uir.append({
        'text': data[(data['user'] == row[0]) & (data['item'] == row[1])]['text'].values,
        'item': row[1],
        'user': row[0],
        'id': data[(data['user'] == row[0]) & (data['item'] == row[1])]['id'].values[0],
        'label': data[(data['user'] == row[0]) & (data['item'] == row[1])]['label'].values[0]
    })

for row in test_np:
    test_uir.append({
        'text': data[(data['user'] == row[0]) & (data['item'] == row[1])]['text'].values,
        'item': row[1],
        'user': row[0],
        'id': data[(data['user'] == row[0]) & (data['item'] == row[1])]['id'].values[0],
        'label': data[(data['user'] == row[0]) & (data['item'] == row[1])]['label'].values[0]
    })

item_review_id = {}
user_review_id = {}

for i in train_uir:
    if i['item'] in item_review_id:

        item_review_id[i['item']].append(i['id'])
    else:
        item_review_id[i['item']] = [i['id']]

    if i['user'] in user_review_id:

        user_review_id[i['user']].append(i['id'])
    else:
        user_review_id[i['user']] = [i['id']]

all_user_texts = []
for i in user_review_id:
    pad_docs = []
    for j in user_review_id[i][:MAX_REVIEW_USER]:
        sents = [x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        pad_sents = [x + (MAX_SENT_LENGTH - len(x)) * [0] for x in sents]
        pad_docs.append(pad_sents + [[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(pad_sents)))
    all_user_texts.append(pad_docs + [[[0] * MAX_SENT_LENGTH] * MAX_SENTS] * (MAX_REVIEW_USER - len(pad_docs)))

all_item_texts = []
for i in item_review_id:
    pad_docs = []
    for j in item_review_id[i][:MAX_REVIEW_ITEM]:
        sents = [x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        pad_sents = [x + (MAX_SENT_LENGTH - len(x)) * [0] for x in sents]
        pad_docs.append(pad_sents + [[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(pad_sents)))
    all_item_texts.append(pad_docs + [[[0] * MAX_SENT_LENGTH] * MAX_SENTS] * (MAX_REVIEW_ITEM - len(pad_docs)))

all_user_texts = np.array(all_user_texts, dtype='int32')
all_item_texts = np.array(all_item_texts, dtype='int32')

item_id_dict = {}
cnt = 0
for i in item_review_id:
    item_id_dict[i] = cnt
    cnt += 1

user_id_dict = {}
cnt = 0
for i in user_review_id:
    user_id_dict[i] = cnt
    cnt += 1

item_to_user_id = {}

for i in train_uir:
    if item_id_dict[i['item']] in item_to_user_id:
        item_to_user_id[item_id_dict[i['item']]].append(user_id_dict[i['user']])
    else:
        item_to_user_id[item_id_dict[i['item']]] = [user_id_dict[i['user']]]

user_to_item_id = {}

for i in train_uir:
    if user_id_dict[i['user']] in user_to_item_id:
        user_to_item_id[user_id_dict[i['user']]].append(item_id_dict[i['item']])
    else:
        user_to_item_id[user_id_dict[i['user']]] = [item_id_dict[i['item']]]

user_to_item_to_user = []
user_to_item = []
for i in user_to_item_id:
    ids = []

    ui_ids = user_to_item_id[i][:MAX_NEIGHBOR]

    for j in user_to_item_id[i]:
        randids = random.sample(item_to_user_id[j], min(MAX_NEIGHBOR, len(item_to_user_id[j])))

        ids.append(randids + [len(user_to_item_id) + 1] * (MAX_NEIGHBOR - len(randids)))
    ids = ids[:MAX_NEIGHBOR]
    user_to_item_to_user.append(ids + [[len(user_to_item_id) + 1] * MAX_NEIGHBOR] * (MAX_NEIGHBOR - len(ids)))

    user_to_item.append(ui_ids + [len(item_to_user_id) + 1] * (MAX_NEIGHBOR - len(ui_ids)))

user_to_item_to_user = np.array(user_to_item_to_user, dtype='int32')

user_to_item = np.array(user_to_item, dtype='int32')

item_to_user_to_item = []
item_to_user = []
for i in item_to_user_id:
    ids = []

    iu_ids = item_to_user_id[i][:MAX_NEIGHBOR]

    for j in item_to_user_id[i]:
        randids = random.sample(user_to_item_id[j], min(MAX_NEIGHBOR, len(user_to_item_id[j])))

        ids.append(randids + [len(item_to_user_id) + 1] * (MAX_NEIGHBOR - len(randids)))
    ids = ids[:MAX_NEIGHBOR]
    item_to_user_to_item.append(ids + [[len(item_to_user_id) + 1] * MAX_NEIGHBOR] * (MAX_NEIGHBOR - len(ids)))

    item_to_user.append(iu_ids + [len(user_to_item_id) + 1] * (MAX_NEIGHBOR - len(iu_ids)))

item_to_user_to_item = np.array(item_to_user_to_item, dtype='int32')
item_to_user = np.array(item_to_user, dtype='int32')

train_item_id = []
train_user_id = []
train_label = []
for i in train_uir:
    train_item_id.append(item_id_dict[i['item']])
    train_user_id.append(user_id_dict[i['user']])
    train_label.append(i['label'])

val_item_id = []
val_user_id = []
val_label = []
for i in val_uir:
    val_item_id.append(item_id_dict[i['item']])
    val_user_id.append(user_id_dict[i['user']])
    val_label.append(i['label'])

test_item_id = []
test_user_id = []
test_label = []
for i in test_uir:
    test_item_id.append(item_id_dict[i['item']])
    test_user_id.append(user_id_dict[i['user']])
    test_label.append(i['label'])

train_label = np.array(train_label, dtype='float32')
train_item_id = np.array(train_item_id, dtype='int32')
train_user_id = np.array(train_user_id, dtype='int32')

val_label = np.array(val_label, dtype='float32')
val_item_id = np.array(val_item_id, dtype='int32')
val_user_id = np.array(val_user_id, dtype='int32')

test_label = np.array(test_label, dtype='float32')
test_item_id = np.array(test_item_id, dtype='int32')
test_user_id = np.array(test_user_id, dtype='int32')


def generate_batch_data_random(item,user,user_to_item_to_user,ui,item_to_user_to_item,iu,item_id,user_id, y, batch_size):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            yield ([item[item_id[i]],user[user_id[i]],user_to_item_to_user[user_id[i]],
                    user_to_item[user_id[i]],item_to_user_to_item,item_to_user[item_id[i]],
                    np.expand_dims(item_id[i],axis=1),np.expand_dims(user_id[i],axis=1)], y[i])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf

sentence_input = tf.keras.layers.Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedding_layer = tf.keras.layers.Embedding(len(word_dict_freq), 300, weights=[emb_mat],trainable=True)

embedded_sequences = tf.keras.layers.Dropout(0.2)(embedding_layer(sentence_input))

word_cnn_fea = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Convolution1D(filters=100, kernel_size=3,  padding='same', activation='relu', strides=1)(embedded_sequences))

word_att = tf.keras.layers.Dense(100,activation='tanh')(word_cnn_fea)
word_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(word_att))
word_att = tf.keras.layers.Activation('softmax')(word_att)
sent_emb=tf.keras.layers.Dot((1, 1))([word_cnn_fea, word_att])

sent_encoder = tf.keras.Model([sentence_input], sent_emb)

review_input = tf.keras.Input((MAX_SENTS,MAX_SENT_LENGTH,), dtype='int32')

review_encoder = tf.keras.layers.TimeDistributed(sent_encoder)(review_input)

sent_cnn_fea = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Convolution1D(filters=100, kernel_size=3, padding='same', activation='relu', strides=1)(review_encoder))

sent_att = tf.keras.layers.Dense(100,activation='tanh')(sent_cnn_fea)
sent_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(sent_att))
word_att = tf.keras.layers.Activation('softmax')(sent_att)
doc_emb=tf.keras.layers.Dot((1, 1))([sent_cnn_fea, word_att])

doc_encoder = tf.keras.Model([review_input], doc_emb)


reviews_input_item = tf.keras.Input((MAX_REVIEW_ITEM,MAX_SENTS,MAX_SENT_LENGTH,), dtype='int32')
reviews_input_user = tf.keras.Input((MAX_REVIEW_USER,MAX_SENTS,MAX_SENT_LENGTH,), dtype='int32')

reviews_emb_item = tf.keras.layers.TimeDistributed(doc_encoder)(reviews_input_item)
reviews_emb_user = tf.keras.layers.TimeDistributed(doc_encoder)(reviews_input_user)


doc_att = tf.keras.layers.Dense(100,activation='tanh')(reviews_emb_item)
doc_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(doc_att))
doc_att = tf.keras.layers.Activation('softmax')(doc_att)
item_emb=tf.keras.layers.Dot((1, 1))([reviews_emb_item, doc_att])

doc_att_u = tf.keras.layers.Dense(100,activation='tanh')(reviews_emb_user)
doc_att_u = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(doc_att_u))
doc_att_u = tf.keras.layers.Activation('softmax')(doc_att_u)
user_emb=tf.keras.layers.Dot((1, 1))([reviews_emb_user, doc_att_u])

user_id =tf.keras.layers. Input(shape=(1,), dtype='int32')
item_id = tf.keras.layers.Input(shape=(1,), dtype='int32')


user_embedding= tf.keras.layers.Embedding(len(user_review_id),100,trainable=True)
item_embedding = tf.keras.layers.Embedding(len(item_review_id), 100,trainable=True)

user_item_ids = tf.keras.Input((MAX_NEIGHBOR,), dtype='int32')
item_user_ids = tf.keras.Input((MAX_NEIGHBOR,), dtype='int32')

user_item_user_ids = tf.keras.Input((MAX_NEIGHBOR,MAX_NEIGHBOR), dtype='int32')
item_user_item_ids = tf.keras.Input((MAX_NEIGHBOR,MAX_NEIGHBOR), dtype='int32')

user_item_embedding= user_embedding(user_item_ids)
item_user_embedding= item_embedding(item_user_ids)

ui_att = tf.keras.layers.Dense(100,activation='tanh')(user_item_embedding)
ui_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(ui_att))
ui_att = tf.keras.layers.Activation('softmax')(ui_att)
ui_emb=tf.keras.layers.Dot((1, 1))([user_item_embedding, ui_att])

iu_att = tf.keras.layers.Dense(100,activation='tanh')(item_user_embedding)
iu_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(iu_att))
iu_att_weight = tf.keras.layers.Activation('softmax')(iu_att)
iu_emb=tf.keras.layers.Dot((1, 1))([item_user_embedding, iu_att_weight])

userencoder = tf.keras.Model([user_item_ids], ui_emb)
itemencoder = tf.keras.Model([item_user_ids], iu_emb)

user_encoder = tf.keras.layers.TimeDistributed(userencoder)(user_item_user_ids)
item_encoder = tf.keras.layers.TimeDistributed(itemencoder)(item_user_item_ids)

ufactor=tf.keras.layers.concatenate([user_item_embedding,user_encoder])
ifactor=tf.keras.layers.concatenate([item_user_embedding,item_encoder])

un_att = tf.keras.layers.Dense(100,activation='tanh')(ufactor)
un_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(un_att))
un_att = tf.keras.layers.Activation('softmax')(un_att)
user_emb_g=tf.keras.layers.Dot((1, 1))([ufactor, un_att])

in_att = tf.keras.layers.Dense(100,activation='tanh')(ifactor)
in_att = tf.keras.layers.Flatten()(tf.keras.layers.Dense(1)(in_att))
in_att = tf.keras.layers.Activation('softmax')(in_att)
item_emb_g=tf.keras.layers.Dot((1, 1))([ifactor, in_att])


user_embedding= tf.keras.layers.Flatten()(user_embedding(user_id))
item_embedding= tf.keras.layers.Flatten()(item_embedding(item_id))
factor_u=tf.keras.layers.concatenate([user_emb,user_embedding,user_emb_g])
factor_i=tf.keras.layers.concatenate([item_emb,item_embedding,item_emb_g])

preds=tf.keras.layers.Dense(1,activation='relu')(tf.multiply(factor_u,factor_i))

model = tf.keras.Model([reviews_input_item,reviews_input_user,user_item_user_ids,user_item_ids,item_user_item_ids,item_user_ids,item_id,user_id], preds)

model.compile(loss='mse', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['mse'])


for ep in range(1):
    traingen=generate_batch_data_random(all_item_texts,all_user_texts,user_to_item_to_user,user_to_item,item_to_user_to_item,item_to_user,train_item_id,train_user_id,train_label,batch_size)
    valgen=generate_batch_data_random(all_item_texts,all_user_texts,user_to_item_to_user,user_to_item,item_to_user_to_item,item_to_user,test_item_id,test_user_id,test_label,512)
    model.fit_generator(traingen, epochs=1,steps_per_epoch=len(train_item_id)//batch_size)
    cr = model.evaluate_generator(valgen, steps=len(test_item_id)//512)
    print(np.sqrt(cr))
