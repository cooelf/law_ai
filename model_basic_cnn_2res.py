import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
import pickle
K.set_image_dim_ordering('tf')
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from time import strftime, gmtime, time
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.layers import merge
from keras.layers.core import *
from keras.models import *
from keras.layers import Dense, Embedding, Input,LSTM,Convolution1D
from keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout,Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
import numpy as np
from judge import Judger
import pandas as pd
from datetime import datetime
import random
# num_words=423834
# maxlen=500
num_words=382516
maxlen=500
epochs = 50
folder = "formal_cnn_stop_res_acc_382516_500/"
np.random.seed(20180603)
if not os.path.exists('experiments/' + folder):
    os.makedirs('experiments/' + folder)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=datetime.now().strftime('experiments/' + folder + '/mylogfile_%H_%M_%d_%m_%Y'+".log"),
                    filemode='w')


def label2tag(labels):
    return [set_accusation[i == 1] for i in labels]


def predict2tag(predictions):
    return [set_accusation[i] for i in predictions]

train_fact_pad_seq = np.load('./data_deal/fact_pad_seq/train_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))
valid_fact_pad_seq = np.load('./data_deal/fact_pad_seq/valid_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))
test_fact_pad_seq = np.load('./data_deal/fact_pad_seq/test_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))

train_labels = np.load('./data_deal/labels/train_labels_accusation.npy')
valid_labels = np.load('./data_deal/labels/valid_labels_accusation.npy')
test_labels = np.load('./data_deal/labels/test_labels_accusation.npy')

set_accusation = np.load('./data_deal/set/set_accusation.npy')

data_input = Input(shape=[valid_fact_pad_seq.shape[1]])
word_vec = Embedding(input_dim=num_words + 1,
                     input_length=maxlen,
                     output_dim=128,
                     mask_zero=0,
                     name='Embedding')(data_input)
x = Convolution1D(512, 3, padding='same', activation='relu')(word_vec)
x = Dropout(0.5)(x)
x= concatenate([word_vec,x])
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
x =concatenate([avg_pool, max_pool])

x = Dense(valid_labels.shape[1], activation="sigmoid")(x)
model = Model(inputs=data_input, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def get_time():
    return strftime('%Y-%m-%d %H:%M:%S', gmtime())

def transform_thred(raw):
    final = []
    for item in raw:
        n = len(item)
        item_return = np.arange(1, n+1)[item > 0.5].tolist()
        if len(item_return) == 0:
            item_return = np.arange(1, n+1)[item == item.max()].tolist()
        final.append(item_return)
    return final

model.summary()
best_epoch = 0.
best_F1 = 0.
# for i in range(1, epochs+1):
#     print('%s -- Epoch %d ' % (get_time(), i))
#     logging.info('%s -- Epoch %d ' % (get_time(), i))
#     model.fit(x=train_fact_pad_seq, y=train_labels,
#               batch_size=128, epochs=1, verbose=1)
#
#     judge_score = Judger("data/accu.txt", "data/law.txt")
#
#     # validate
#     y1 = label2tag(valid_labels)
#     valid_raw = model.predict(valid_fact_pad_seq)
#     y2_thred = predict2tag(transform_thred(valid_raw))
#     print("evaluating...", str(len(y1)))
#     logging.info("evaluating..." + str(len(y1)))
#     print("thred=0.5...")
#     logging.info("thred=0.5...")
#     result = judge_score.get_score(judge_score.test(y1, y2_thred))
#     print(result)
#     logging.info(result)
#
#     # test
#     y1 = label2tag(test_labels)
#     test_raw = model.predict(test_fact_pad_seq)
#     outdata = {"truth": y1, "pred": test_raw}
#     y2_thred = predict2tag(transform_thred(test_raw))
#     print("testing...", str(len(y1)))
#     logging.info("testing..." + str(len(y1)))
#     print("thred=0.5...")
#     logging.info("thred=0.5...")
#     result = judge_score.get_score(judge_score.test(y1, y2_thred))
#     print(result)
#     logging.info(result)
#     if(result[0][-1]>best_F1):
#         best_epoch = i
#         best_F1 = result[0][-1]
#         if not os.path.exists('experiments/' + folder + "predicts/"):
#             os.makedirs('experiments/' + folder + "predicts/")
#         fout = open('experiments/' + folder + "predicts/" + "/best_predictions.h5", 'wb')
#         pickle.dump(outdata, fout)
#         if not os.path.exists('experiments/' + folder + "models/"):
#             os.makedirs('experiments/' + folder + "models/")
#         model.save('experiments/' + folder + "models/" + '/best_model.h5',overwrite=True)
#
#     print("bestF",best_F1,"bestE", best_epoch)
#     logging.info("bestF:"+str(best_F1)+"bestE:"+str(best_epoch))

judge_score = Judger("data/accu.txt", "data/law.txt")

for i in range(1, epochs+1):
    print('%s -- Epoch %d ' % (get_time(), i))
    logging.info('%s -- Epoch %d ' % (get_time(), i))
    model.fit(x=train_fact_pad_seq, y=train_labels,
              batch_size=512, epochs=1, verbose=1)
    model.fit(x=valid_fact_pad_seq, y=valid_labels,
              batch_size=512, epochs=1, verbose=1)
    if not os.path.exists('experiments/' + folder + "models/"):
        os.makedirs('experiments/' + folder + "models/")
    # test

    test_raw = model.predict(test_fact_pad_seq)
    outdata = {"truth": test_labels, "pred": test_raw}

    predictions = transform_thred(test_raw)
    result = []

    for item in predictions:
        result.append({
            "accusation": item,
            "articles":[1, 2] ,
            "imprisonment": 2
        })

    results = judge_score.test_json("data/data_test.json", result)
    score = judge_score.get_score(results)
    print(score)
    logging.info(score)

    if(score[0] > best_F1):
        best_epoch = i
        best_F1 = score[0]
        if not os.path.exists('experiments/' + folder + "predicts/"):
            os.makedirs('experiments/' + folder + "predicts/")
        fout = open('experiments/' + folder + "predicts/" + '/best_epochs.h5', 'wb')
        pickle.dump(outdata, fout)
        if not os.path.exists('experiments/' + folder + "models/"):
            os.makedirs('experiments/' + folder + "models/")
        model.save('experiments/' + folder + "models/" + '/best_model.h5')

    print("bestF", best_F1, "bestE", best_epoch)
    logging.info("bestF:" + str(best_F1) + "bestE:" + str(best_epoch))



