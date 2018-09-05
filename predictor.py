import os
import pickle
from keras.models import load_model
from .data_utils import data_utils
import numpy as np
import os
import jieba
jieba.setLogLevel('WARN')

class Predictor:
    def __init__(self, maxlen=500,
                 accusation_path1='predictor/model/acc_83.28.h5',
                 accusation_path2='predictor/model/acc_83.97.h5',
                 tokenizer_path='predictor/model/word_index.pkl'):
        self.maxlen = maxlen
        self.batch_size = 32
        self.content_handle = data_utils()
        self.tokenizer_path = tokenizer_path
        self.accusation_model1 = load_model(accusation_path1)
        self.accusation_model2 = load_model(accusation_path2)

    def get_imps_time(self,y):
        max = 360
        min = 0
        y_imp = float(y * (max - min) + min)
        imprison_month = int(y_imp + 0.5)
        if imprison_month >= 320 and imprison_month <= 360:
            imprison_month = -2
        elif imprison_month >= 300 and imprison_month <= 320:
            imprison_month = -1
        return imprison_month

    def predict(self, content):
        maxlen = self.maxlen
        content_handle = self.content_handle
        tokenizer_path = self.tokenizer_path
        content_cut = content_handle.seg(texts=content)
        with open(tokenizer_path, mode='rb') as f:
            tokenizer_fact = pickle.load(f)
        content_handle.pad_seq(texts_cut=content_cut, tokenizer_fact=tokenizer_fact,
                                    maxlen=maxlen)
        content_fact_pad_seq = np.array(content_handle.fact_pad_seq)

        accusation_model = self.accusation_model1
        accusation1 = accusation_model.predict(content_fact_pad_seq, batch_size=self.batch_size)
        accusation_model = self.accusation_model2
        accusation2 = accusation_model.predict(content_fact_pad_seq, batch_size=self.batch_size)
        accusation = 0.7 * accusation1 + 0.3 * accusation2

        def transform(x):
            n = len(x)
            x_return = np.arange(1, n + 1)[x > 0.5].tolist()
            if len(x_return) == 0:
                x_return = np.arange(1, n + 1)[x == x.max()].tolist()
            return x_return

        result = []
        for i in range(0, len(content)):
            result.append({
                "accusation": transform(accusation[i]),
                "articles": [1,2],
                "imprisonment": 2
            })
        return result


