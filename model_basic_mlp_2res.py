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
from keras.layers import Dense, Embedding, Input,LSTM
from keras.layers import GRU, Bidirectional, GlobalMaxPool1D, Dropout,Conv1D,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
import numpy as np
from judger import Judger
import pandas as pd
from datetime import datetime
import random
# num_words=423834
# maxlen=500
num_words=382516
maxlen=500
epochs = 50
folder = "mlp_stop_res/"
np.random.seed(233)
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

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = 512
    timesteps = 300
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, timesteps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(timesteps, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

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
x = Dense(500, activation="relu")(word_vec)
x = Dropout(0.25)(x)
x= concatenate([word_vec,x])
x = GlobalMaxPool1D()(x)
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
        item_return = np.arange(0, n)[item > 0.5].tolist()
        if len(item_return) == 0:
            item_return = np.arange(0, n)[item == item.max()].tolist()
        final.append(item_return)
    return final

def transform_max(raw):
    final = []
    for item in raw:
        n = len(item)
        item_return = np.arange(0, n)[item == item.max()].tolist()
        final.append(item_return)
    return final
model.summary()
best_epoch = 0.
best_F1 = 0.
for i in range(1, epochs+1):
    print('%s -- Epoch %d ' % (get_time(), i))
    logging.info('%s -- Epoch %d ' % (get_time(), i))
    model.fit(x=train_fact_pad_seq, y=train_labels,
              batch_size=128, epochs=1, verbose=1)

    if not os.path.exists('experiments/' + folder + "models/"):
        os.makedirs('experiments/' + folder + "models/")
    model.save('experiments/' + folder + "models/" +'/epochs_%d.h5' % i)

    judge_score = Judger("data/accu.txt", "data/law.txt")

    # validate
    y1 = label2tag(valid_labels)
    valid_raw = model.predict(valid_fact_pad_seq)

    outdata = {"truth": y1, "pred": valid_raw}

    if not os.path.exists('experiments/' + folder + "predicts/"):
        os.makedirs('experiments/' + folder + "predicts/")
    fout = open('experiments/' + folder + "predicts/" + "/valid_epoch%d.pkl" % i, 'wb')

    pickle.dump(outdata, fout)

    y2_thred = predict2tag(transform_thred(valid_raw))
    y2_max = predict2tag(transform_max(valid_raw))
    print("evaluating...", str(len(y1)))
    logging.info("evaluating..." + str(len(y1)))
    print("thred=0.5...")
    logging.info("thred=0.5...")
    result = judge_score.get_score(judge_score.test(y1, y2_thred))
    print(result)
    logging.info(result)
    print("max...")
    result = judge_score.get_score(judge_score.test(y1, y2_max))
    print(result)
    logging.info(result)
    # test
    y1 = label2tag(test_labels)
    test_raw = model.predict(test_fact_pad_seq)

    outdata = {"truth": y1, "pred": test_raw}
    fout = open('experiments/' + folder + "predicts/" + "/test_epoch%d.pkl" % i, 'wb')
    pickle.dump(outdata, fout)

    y2_thred = predict2tag(transform_thred(test_raw))
    y2_max = predict2tag(transform_max(test_raw))
    print("testing...", str(len(y1)))
    logging.info("testing..." + str(len(y1)))
    print("thred=0.5...")
    logging.info("thred=0.5...")
    result = judge_score.get_score(judge_score.test(y1, y2_thred))
    print(result)
    logging.info(result)
    if(result[0][-1]>best_F1):
        best_epoch = i
        best_F1 = result[0][-1]
    print("bestF",best_F1,"bestE", best_epoch)
    logging.info("bestF:"+str(best_F1)+"bestE:"+str(best_epoch))
    print("max...")
    logging.info("max...")
    result = judge_score.get_score(judge_score.test(y1, y2_max))
    print(result)
    logging.info(result)
    if(result[0][-1]>best_F1):
        best_epoch = i
        best_F1 = result[0][-1]
    print("bestF", best_F1, "bestE", best_epoch)
    logging.info("bestF:" + str(best_F1) + "bestE:" + str(best_epoch))




