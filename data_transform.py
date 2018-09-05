import json
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import jieba
import re
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
jieba.setLogLevel('WARN')

print("initial jieba seg...")
jieba.load_userdict("./data/userdict.txt")
def load_stopwords(stopwords_fname):
    """ load stopwords into set from local file """
    stopwords = set()
    with open(stopwords_fname, 'r',encoding="utf-8") as f:
        for line in f.readlines():
            stopwords.add(line.strip())
    return stopwords
stopwords = load_stopwords("./data/stopwords.txt")

class data_transform():
    def __init__(self):
        self.data_path = None
        self.data = None
        self.texts_cut = None
        self.tokenizer = None
        self.label_set = {}
        self.extraction = {}
        self.imprison_set = list(range(0,304))
        self.tokenizer_fact = None

    def filter_info(self,line):
        rst = re.findall(u"\d{4}年\d{1,2}月", line)
        if (len(rst) > 0):
            pos = line.find(rst[0])
            while (pos < len(line) and line[pos] != u"，"):
                pos += 1
            if pos < len(line):
                line = line[pos + 1:]
        return line

    def read_data(self, path=None):
        '''
        读取json文件,必须readlines，否则中间有格式会报错
        :param path: 文件路径
        :return:json数据
        eg. data_valid = data_transform.read_data(path='./data/data_valid.json')
        '''
        self.data_path = path
        f = open(path, 'r', encoding='utf8')
        data_raw = f.readlines()
        data = []
        for num, data_one in enumerate(data_raw):
            try:
                data_json = json.loads(data_one)
                data_json["fact"] = self.filter_info(data_json["fact"])
                data.append(data_json)
            except Exception as e:
                print('num: %d', '\n',
                      'error: %s', '\n',
                      'data: %s' % (num, e, data_one))
        self.data = data

    def extract_data(self, name='accusation'):
        '''
        提取需要的信息，以字典形式存储
        :param name: 提取内容
        :return: 事实描述,罪名,相关法条
        eg. data_valid_accusations = data_transform.extract_data(name='accusation')
        '''
        data = self.data
        if name == 'fact':
            extraction = list(map(lambda x: x['fact'], data))
        elif name in ['accusation', 'relevant_articles']:
            extraction = list(map(lambda x: x['meta'][name], data))
        elif name == "imprisonment":
            extraction = list(map(lambda x: self.gettime(x['meta']['term_of_imprisonment']), data))
        self.extraction.update({name: extraction})

    def cut_texts(self, texts=None, need_cut=True,word_len=1, texts_cut_savepath=None):
        '''
        文本分词剔除停用词
        :param texts:文本列表
        :param need_cut:是否需要分词
        :param word_len:保留词语长度
        :param texts_cut_savepath:保存路径
        :return:
        '''

        texts_cut = [[word for word in jieba.lcut(one_text) if word not in stopwords and len(word) >= word_len] for one_text in texts] #去除停用词

        if texts_cut_savepath is not None:
            with open(texts_cut_savepath, 'w') as f:
                json.dump(texts_cut, f)
        return texts_cut

    def text2seq(self, texts_cut=None, tokenizer_fact=None, num_words=2000, maxlen=30):
        '''
        文本转序列，训练集过大全部转换会内存溢出，每次放5000个样本
        :param texts_cut: 分词后的文本列表
        :param tokenizer:转换字典
        :param num_words:字典词数量
        :param maxlen:保留长度
        :return:向量列表
        eg. ata_transform.text2seq(texts_cut=train_fact_cut,num_words=2000, maxlen=500)
        '''
        texts_cut_len = len(texts_cut)
        #兼容低版本keras
        text4fit = [" ".join(tt) for tt in texts_cut]
        texts_cut = text4fit
        #避免不支持里面有list
        if tokenizer_fact is None:
            # tokenizer_fact = Tokenizer(num_words=num_words)
            tokenizer_fact = Tokenizer()
            if texts_cut_len > 10000:
                print('文本过多，分批转换')
            n = 0
            # 分批训练
            while n < texts_cut_len:
                #*******************************************************
                tokenizer_fact.fit_on_texts(texts=texts_cut[n:n + 10000])

                n += 10000
                if n < texts_cut_len:
                    print('tokenizer finish fit %d samples' % n)
                else:
                    print('tokenizer finish fit %d samples' % texts_cut_len)
            self.tokenizer_fact = tokenizer_fact

        print("word num", len(tokenizer_fact.word_index))

        # 全部转为数字序列
        fact_seq = tokenizer_fact.texts_to_sequences(texts=texts_cut)
        print('finish texts to sequences')

        # indicator考虑两种，一种是在每个词后面加标记，一种是在句子最后点明几个指示词
        # 读取罪名清单
        accu_list = []
        f = open("data/accu.txt", "r", encoding="utf-8")
        for line in f:
            accu_list.append(line[:-1])
        accu_cut = self.cut_texts(texts=accu_list, word_len=2, need_cut=True)

        acc_ids = []
        for accs in accu_cut:
            word_list = []
            for word in accs:
                if word in tokenizer_fact.word_index:
                    word_list.append(tokenizer_fact.word_index[word])
            acc_ids.append(word_list)

        em_seq = calculate_exact_match(fact_seq,acc_ids)
        #acc_ids = [[tokenizer_fact.word_index[word] if word in tokenizer_fact.word_index for word in accs] for accs in accu_cut]
        # print(acc_ids)
        # 内存不够，删除
        del texts_cut

        fact_pad_seq = list(pad_sequences(fact_seq, maxlen=maxlen,
                                               padding='post', value=0, dtype='int'))
        em_pad_seq = list(pad_sequences(em_seq, maxlen=maxlen,
                                          padding='post', value=0, dtype='int'))
        self.fact_pad_seq = fact_pad_seq
        self.em_pad_seq = em_pad_seq

    def creat_label_set(self, name):
        '''
        获取标签集合，用于one-hot
        :param name: 待创建集合的标签名称
        :return:
        '''
        if  name == "imprisonment":
            label_set = self.imprison_set
        else:
            if name == 'accusation':
                name_f = 'accu'
            elif name == 'relevant_articles':
                name_f = 'law'
            with open('./data/%s.txt' % name_f, encoding='utf-8') as f:
                label_set = f.readlines()
            label_set = [i[:-1] for i in label_set]
            if name_f == 'law':
                label_set = [int(i) for i in label_set]  #让法条用int表示而不是str，保证跟json读取后的法条格式一致
        self.label_set.update({name: np.array(label_set)})

    def creat_label(self, label, label_set):
        '''
        构建标签one-hot
        :param label: 原始标签
        :param label_set: 标签集合
        :return: 标签one-hot
        eg. creat_label(label=data_valid_accusations[12], label_set=accusations_set)
        '''
        label_zero = np.zeros(len(label_set))
        label_zero[np.in1d(label_set, label)] = 1
        return label_zero

    def creat_labels(self, label_set=None, labels=None, name='accusation'):
        '''
        调用creat_label遍历标签列表生成one-hot二维数组
        :param label_set: 标签集合,数组
        :param labels: 标签数据，二维列表，没有则调用extract_data函数提取
        :param name:
        :return:
        '''
        if label_set is None:
            label_set = self.label_set[name]
        if labels is None:
            labels = self.extraction[name]
        labels_one_hot = list(map(lambda x: self.creat_label(label=x, label_set=label_set), labels))
        return labels_one_hot

    def gettime(self,time):
        # 将刑期用分类模型来做
        v = int(time['imprisonment'])
        if time['death_penalty']:
            v = 302
        if time['life_imprisonment']:
            v = 303
        return v

def reverse_vocab(vocab):
    reverse_vocab = dict((v, k) for k, v in vocab.items())
    return reverse_vocab

def calculate_exact_match(source_sentences, target_labels):
    new_sourse = []
    # rev_vocab = reverse_vocab(tokenizer.word_index)
    for s_sentent in source_sentences:
        new_sent = [] #分析每个句子
        for word in s_sentent:
            isEM = False  #判断句子里面的每个词
            for t_label in target_labels: #遍历标签看有没有match的
                if word in t_label:
                    isEM = True
                    # print(rev_vocab[word])
                    break # find EM then break to save calculation
            new_sent.append(isEM)
        new_sourse.append(new_sent)
    em_np = np.array(new_sourse, copy=False)
    return em_np

def load_word2vec_embeddings(dictionary, vocab_embed_file):
    fp = open(vocab_embed_file,encoding="u")
    info = fp.readline().split()
    embed_dim = int(info[1])

    vocab_embed = {}
    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(map(float, line[1:]), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.iteritems():
        if w in vocab_embed:
            W[i,:] = vocab_embed[w]
            n += 1
    print("%d/%d vocabs are initialized with word2vec embeddings." % (n, vocab_size))
    return W, embed_dim