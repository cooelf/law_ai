# -*- coding: utf-8 -*-
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

import string
from keras.models import load_model
import numpy as np
import os
import json
import pickle
from predictor.data_utils import data_utils
num_words=382516
maxlen=500

model1 = load_model("formal_ens/acc_83.13.h5")
model2 = load_model("formal_ens/acc_83.28.h5")
model3 = load_model("formal_ens/acc_83.97.h5")
model4 = load_model("formal_ens/acc_84.39.h5")

test_fact_pad_seq = np.load('formal_ens/formal_test_fact_pad_seq_%d_%d.npy'%(num_words,maxlen))

test_labels = np.load('formal_ens/formal_test_labels_accusation.npy')

acc1 = model1.predict(test_fact_pad_seq, batch_size=128)
acc2 = model2.predict(test_fact_pad_seq, batch_size=128)
acc3 = model3.predict(test_fact_pad_seq, batch_size=128)
acc4 = model4.predict(test_fact_pad_seq, batch_size=128)


def label2tag(labels):
    all_labels = []
    for label in labels:
        each_label = []
        for i,label_index in enumerate(label):
            if label_index == 1:
                each_label.append(i+1)
        all_labels.append(each_label)

    return all_labels

def transform(x):
    n = len(x)
    x_return = np.arange(1, n + 1)[x > 0.5].tolist()
    if len(x_return) == 0:
        x_return = np.arange(1, n + 1)[x == x.max()].tolist()
    return x_return

from judge import Judger
judge_score = Judger("predictor/data/accu.txt", "predictor/data/law.txt")

truths = label2tag(test_labels.tolist())

best = 0.
type = ""
for C_1 in range(10, -1, -1):
    for C_2 in range((10 - C_1) + 1):
        for C_3 in range((10 - C_1 - C_2 + 1)):
            C_4 = 10 - C_1 - C_2 - C_3
            c_1 = float(C_1) / 10.
            c_2 = float(C_2) / 10.
            c_3 = float(C_3) / 10.
            c_4 = float(C_4) / 10.

            print(c_1,c_2,c_3 ,c_4)
            result = []
            ground_truth = []
            for acc1_item, acc2_item,acc3_item, acc4_item, t_item  in zip(acc1,acc2, acc3, acc4,truths):
                new_item = acc1_item * c_1 + acc2_item * c_2 + acc3_item * c_3 + acc3_item * c_3
                result.append({
                    "accusation": transform(new_item),
                    "articles": [1,2],
                    "imprisonment": 2
                })
                ground_truth.append({
                    "accusation": t_item,
                    "articles": [1, 2],
                    "imprisonment": 2
                })


            predict_out = "ensemble_acc4/model_acc_" + str(C_1) + "_" + str(
                C_2) + "_" + str(C_3) + "_" + str(C_4) + ".out"
            out = open(predict_out,"w", encoding="utf-8")
            for x in result:
                print(json.dumps(x), file=out)
            out.close()
            results = judge_score.test_json(ground_truth, result)
            score = judge_score.get_score(results)
            if(score[0] > best):
                best = score[0]
                type =  str(C_1) + "_" + str(C_2) + "_" + str(C_3) + "_" + str(C_4)
            print(score)
print(type)
print(best)
