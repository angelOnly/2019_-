#! -*- coding:utf-8 -*-

import codecs
import json
import math
import datetime


# 词在文章中出现次数 / 该文章总词数
def tf(word, doc):
    count = sum(1 for w in doc if w.strip()==word.strip())
    return count / len(doc)


# log(总文档数 / (包含该词文档数+1))
def idf(word, docs):
    count = sum(1 for doc in docs if word in doc)
    return math.log(len(docs) / (1 + count))


# tf*idf
def tfidf(word, doc, docs):
    return tf(word, doc) * idf(word, docs)


# tfidf矩阵
def tfidf_metrix(docs=[]):
    s = set(word for doc in docs for word in doc)
    res = list([tfidf(word,doc,docs) for word in s] for doc in docs)
    return res


def train_tfidf_dic():
    start_time = datetime.datetime.now()
    f = codecs.open('../runs/train_word_dic.json', 'r', 'utf-8')
    final_train_test_dic = json.load(f)
    docs = final_train_test_dic.items()
    tfidf_dics = {}
    idf_dics = {}
    tf_dics = {}
    i = 0
    for key,doc in final_train_test_dic.items():
        tfidf_dic = {}
        for word in doc:
            tfidf_dic[word] = tfidf(word, doc, docs)
            idf_dics[word] = idf(word, docs)
            tf_dics[word] = tf(word, doc)
        if i%500 == 0:
            print(i, ' doc finish')
        i += 1
        tfidf_dics[key] = tfidf_dic

    with codecs.open('../runs/train_tfidf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(tfidf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/train_idf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(idf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/train_tf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(tf_dics, f, indent=4, ensure_ascii=False)

    end_time = datetime.datetime.now()
    print('train:', (end_time - start_time).seconds)


def test_tfidf_dic():
    start_time = datetime.datetime.now()
    f = codecs.open('../runs/test_word_dic.json', 'r', 'utf-8')
    final_train_test_dic = json.load(f)
    docs = final_train_test_dic.items()
    tfidf_dics = {}
    idf_dics  ={}
    tf_dics = {}
    i = 0
    for key,doc in final_train_test_dic.items():
        tfidf_dic = {}
        for word in doc:
            tfidf_dic[word] = tfidf(word, doc, docs)
            idf_dics[word] = idf(word, docs)
            tf_dics[word] = tf(word, doc)
        if i%20 == 0:
            print(i, ' doc finish')
        i += 1
        tfidf_dics[key] = tfidf_dic

    with codecs.open('../runs/test_tfidf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(tfidf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/test_idf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(idf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/test_tf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(tf_dics, f, indent=4, ensure_ascii=False)

    end_time = datetime.datetime.now()
    print('test:', (end_time - start_time).seconds)


if __name__ == '__main__':
    train_tfidf_dic()
    test_tfidf_dic()