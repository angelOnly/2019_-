#! -*- coding:utf-8 -*-

import codecs
import json
import numpy as np
import math
import re
import pandas as pd
from fasttext_embedding import load_model
import datetime


def topn_tfidf_dics(tfidf_dics, n=10):
    top_tfidf_dics = {}
    for newsId,tfidf_doc in tfidf_dics.items():
        top_tfidf_dic = {}
        sorted_dict = sorted(tfidf_doc.items(), key=lambda item: item[1], reverse=True)
        for word,tfidf in sorted_dict[:n]:
            top_tfidf_dic[word] = tfidf
        top_tfidf_dics[newsId] = top_tfidf_dic
    return top_tfidf_dics


def load_word_flag():
    f = codecs.open('../runs/train_dic.json', 'r', 'utf-8')
    word_flag_dic = json.load(f)
    return word_flag_dic


def topn_flag_filter(tfidf_dics, word_flag_dic):
    top10_tfidf_dics = topn_tfidf_dics(tfidf_dics, n=10)
    flag_filter = ['n', 'nr', 'ns', 'nt', 'nz', 't', 'x']
    for newsId, tfidf_dic in top10_tfidf_dics.items():
        common_words = top10_tfidf_dics[newsId].keys() & word_flag_dic.keys()
        dif_words = top10_tfidf_dics[newsId].keys() - word_flag_dic.keys()
        for dif in dif_words:
            top10_tfidf_dics[newsId].pop(dif)
        for word in common_words:
            if word_flag_dic[word] not in flag_filter:
                top10_tfidf_dics[newsId].pop(word)
    return top10_tfidf_dics


def save(top10_test_tfidf_dics, top15_test_tfidf_dics, top10_tfidf_dics, top15_tfidf_dics):
    with codecs.open('../runs/top10_test_tfidf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(top10_test_tfidf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/top15_test_tfidf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(top15_test_tfidf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/top10_tfidf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(top10_tfidf_dics, f, indent=4, ensure_ascii=False)
    with codecs.open('../runs/top15_tfidf_dics.json', 'w', encoding='utf-8') as f:
        json.dump(top15_tfidf_dics, f, indent=4, ensure_ascii=False)



def simlar_calcute(x, y, model_words, model):
    diff_x = x - model_words
    diff_y = y - model_words
    if len(diff_x)>=1:
        print('diff_x:',diff_x)
    if len(diff_y)>=1:
        print('diff_y:',diff_y)
    for dif in diff_x:
        if dif in x:
            x.remove(dif)
    for dif in diff_y:
        if dif in y:
            y.pop(dif)
    x_mean = sum(np.array([model[x1] for x1 in x]))/len(x)
    y_mean = sum(np.array([model[y1] for y1 in y]))/len(y)
    res = np.dot(x_mean, y_mean) / (np.linalg.norm(x_mean) * np.linalg.norm(y_mean))
    return round(res, 8)


def subject_freq_count(sub_list, title, content):
    '''
    统计每个主体出现的次数
    '''
    subject_count_dict = {}
    for sub in sub_list:
        # 文中出现的主体
        if sub not in subject_count_dict:
            # 初始化出现次数都为1
            subject_count_dict[sub] = 1
            if len(set([sub]) & set(title))>=1:
                # title中，权重设置为3
                subject_count_dict[sub] += 3
            if len(set([sub]) & set(content))>=1:
                # 文本中，权重设置为1
                subject_count_dict[sub] += 1
    return subject_count_dict


def left_right_entropy(sub, title, content):
    title_content = "".join(title+content)
    stop_word = ['【', '】', ')', '(', '、', '，', '“', '”', '。', '>', '<', '\n', '《', '》', ' ', '-', '！', '？', '.',
                 '\'', '[', ']', '：', '/', '.', '"', '\u3000', '’', '．', ',', '…', '?']
    for sw in stop_word:
        title_content = title_content.replace(sw, "")
    lr = re.findall('(.)%s(.)' % sub, title_content)

    from collections import Counter
    def entropy(alist):
        f = dict(Counter(alist))
        ent = (-1) * sum([i / len(alist) * math.log(i / len(alist)) for i in f.values()])
        return ent

    if lr:
        left_entropy = entropy([w[0] for w in lr])
        right_entropy = entropy([w[1] for w in lr])
        if left_entropy == -0.0:
            left_entropy = 0.5
        if right_entropy == -0.0:
            right_entropy = 0.5
    else:
        left_entropy = 0.4
        right_entropy = 0.4
    return left_entropy,right_entropy


def first_index(sub, title,content):
    if len(set([sub]) & set(title))>=1:
        subject_start = title.index(sub)
        first_offset = 1-subject_start/len(title)
    else:
        try:
            subject_start = content.index(sub)
            first_offset = 1-subject_start/len(content)
        except:
            first_offset = 0.5
    return first_offset


# 1、主体 和 title 关键词的相似度
# # 2、主体 和 tfidf的topN词相似度
# 3、主体 和 text_rank词[:15]的相似度
# 4、主体是否在title中
# 5、title中是否有其他主体
# 6、主体的TF/IDF/IF-IDF
# 7、文章中是否包含其他主体
# 8、主体最高出现次数/该主体出现的次数
# 9、1-主体首次出现位置占全文的比例
# 10、文章的长度
# 11、主体个数
# 12、主体的左右熵
def generate_test_features(top10_tfidf_dics, top15_tfidf_dics):
    import random
    f = codecs.open('../runs/test_word_dic.json', 'r', 'utf-8')
    test_word_dic = json.load(f)
    f1 = codecs.open('../runs/test_tf_dics.json', 'r', 'utf-8')
    test_tf_dics = json.load(f1)
    f1 = codecs.open('../runs/test_idf_dics.json', 'r', 'utf-8')
    test_idf_dics = json.load(f1)
    label_dic = []
    ids = top10_tfidf_dics.keys()
    model = load_model()
    model_words = model.wv.vocab.keys()
    for newsId in ids:
        keys = top10_tfidf_dics[newsId].keys()
        temp = []
        sub_title_sim = 0.5
        sub_content_sim = 0.5
        sub_tfidf_top_sim = 0.5
        sub_is_in_title = 0
        title_has_other_sub = 0
        title_has_other_sub = 1
        title = test_word_dic[newsId][0]
        content = test_word_dic[newsId][1]
        title_content = title+content
        # title中是否有其他主体
        title_has_other_sub = 1 if len(keys & set(title)) >= 2 else 0
        # 文章中是否包含其他主体
        content_has_other_sub = 1 if len(keys & set(content)) >= 2 else 0
        subject_count_dict = subject_freq_count(keys, title, content)
        # 文章的长度
        len_doc = (len(title)+len(content))/100
        # 每篇文章中主体总数
        sub_in_doc_sum = (len(keys))/10
        label = 0
        for sub in keys:
            # 主体和title相似度
            sub_title_sim = simlar_calcute([sub], title, model_words)
            # 主体和content相似度
            sub_content_sim = simlar_calcute([sub], content, model_words)
            # 主体 和 tfidf的topN词相似度
            sub_tfidf_top_sim = simlar_calcute([sub], top15_tfidf_dics[newsId], model_words)
            # 主体是否在title中
            sub_is_in_title = 1 if len(set([sub]) & set(title))>=1 else 0
            # 主体在文章中出现的次数
            count = subject_count_dict[sub]
            # 主体首次出现位置占全文的比例
            first_offset = first_index(sub, title, content)
            # 主体的左右熵
            left_entropy,right_entropy = left_right_entropy(sub, title, content)
            temp.append([newsId,# id
                         sub, # sub
                         test_tf_dics[sub], # tf
                         test_idf_dics[sub], # idf
                         top10_tfidf_dics[newsId][sub], # tfidf
                         title_has_other_sub, #标题中是否有其它主体
                         content_has_other_sub, # 文章中是否包含其他主体
                         len_doc, #文章的长度
                         sub_in_doc_sum, # 每篇文章中主体总数
                         sub_title_sim, # 主体和title相似度
                         sub_content_sim, # 主体和content相似度
                         sub_tfidf_top_sim, # 主体 和 tfidf的topN词相似度
                         sub_is_in_title, # 主体是否在title中
                         count, # 主体在文章中出现的次数
                         first_offset, # 主体首次出现位置占全文的比例
                         left_entropy, # 主体的左熵
                         right_entropy])# 主体的右熵
        label_dic.append(temp)
    random.shuffle(label_dic)
    return label_dic

# 1、主体 和 title 关键词的相似度
# # 2、主体 和 tfidf的topN词相似度
# 3、主体 和 text_rank词[:15]的相似度
# 4、主体是否在title中
# 5、title中是否有其他主体
# 6、主体的TF/IDF/IF-IDF
# 7、文章中是否包含其他主体
# 8、主体最高出现次数/该主体出现的次数
# 9、1-主体首次出现位置占全文的比例
# 10、文章的长度
# 11、主体个数
# 12、主体的左右熵
def generate_features(top10_tfidf_dics, top15_tfidf_dics):
    import random
    f = codecs.open('../runs/train_entity_dic.json', 'r', 'utf-8')
    train_entity_dic = json.load(f)
    f = codecs.open('../runs/train_word_dic.json', 'r', 'utf-8')
    train_word_dic = json.load(f)
    f1 = codecs.open('../runs/train_tf_dics.json', 'r', 'utf-8')
    train_tf_dics = json.load(f1)
    f1 = codecs.open('../runs/train_idf_dics.json', 'r', 'utf-8')
    train_idf_dics = json.load(f1)
    label_dic = []
    common_ids = top10_tfidf_dics.keys() & train_entity_dic.keys()
    model = load_model()
    model_words = model.wv.vocab.keys()
    for newsId in common_ids:
        common_keys = top10_tfidf_dics[newsId].keys() & train_entity_dic[newsId]
        diff_keys = top10_tfidf_dics[newsId].keys() - train_entity_dic[newsId]
        keys = common_keys|diff_keys
        temp = []
        sub_title_sim = 0.5
        sub_content_sim = 0.5
        sub_tfidf_top_sim = 0.5
        sub_is_in_title = 0
        title_has_other_sub = 0
        title_has_other_sub = 1
        title = train_word_dic[newsId][0]
        content = train_word_dic[newsId][1]
        title_content = title+content
        # title中是否有其他主体
        title_has_other_sub = 1 if len(keys & set(title)) >= 2 else 0
        # 文章中是否包含其他主体
        content_has_other_sub = 1 if len(keys & set(content)) >= 2 else 0
        subject_count_dict = subject_freq_count(keys, title, content)
        # 文章的长度
        len_doc = (len(title)+len(content))/100
        # 每篇文章中主体总数
        sub_in_doc_sum = (len(keys))/10
        label = 0
        for sub in keys:
            if len(set([sub]) & set(common_keys))>=1:
                label = 1
            if len(set([sub]) & set(diff_keys))>=1:
                label = 0
            # 主体和title相似度
            sub_title_sim = simlar_calcute([sub], title, model_words)
            # 主体和content相似度
            sub_content_sim = simlar_calcute([sub], content, model_words)
            # 主体 和 tfidf的topN词相似度
            sub_tfidf_top_sim = simlar_calcute([sub], top15_tfidf_dics[newsId], model_words)
            # 主体是否在title中
            sub_is_in_title = 1 if len(set([sub]) & set(title))>=1 else 0
            # 主体在文章中出现的次数
            count = subject_count_dict[sub]
            # 主体首次出现位置占全文的比例
            first_offset = first_index(sub, title, content)
            # 主体的左右熵
            left_entropy,right_entropy = left_right_entropy(sub, title, content)
            temp.append([newsId,# id
                         sub, # sub
                         train_tf_dics[sub], # tf
                         train_idf_dics[sub], # idf
                         top10_tfidf_dics[newsId][sub], # tfidf
                         title_has_other_sub, #标题中是否有其它主体
                         content_has_other_sub, # 文章中是否包含其他主体
                         len_doc, #文章的长度
                         sub_in_doc_sum, # 每篇文章中主体总数
                         sub_title_sim, # 主体和title相似度
                         sub_content_sim, # 主体和content相似度
                         sub_tfidf_top_sim, # 主体 和 tfidf的topN词相似度
                         sub_is_in_title, # 主体是否在title中
                         count, # 主体在文章中出现的次数
                         first_offset, # 主体首次出现位置占全文的比例
                         left_entropy, # 主体的左熵
                         right_entropy, # 主体的右熵
                         label])#label
        label_dic.append(temp)
    random.shuffle(label_dic)
    return label_dic

def load_train_tfidf_dics():
    f = codecs.open('../runs/train_tfidf_dics.json', 'r', 'utf-8')
    tfidf_dics = json.load(f)
    return tfidf_dics

def load_test_tfidf_dics():
    f = codecs.open('../runs/test_tfidf_dics.json', 'r', 'utf-8')
    tfidf_dics = json.load(f)
    return tfidf_dics

def save_train_data(train_data):
    with codecs.open('../runs/train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

def save_test_data(test_data):
    with codecs.open('../runs/test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

def transe_data(data):
    label_data = []
    for item in data:
        for line in item:
            label_data.append(line)
    return label_data


if __name__ == '__main__':
    train_tfidf_dics = load_train_tfidf_dics()
    test_tfidf_dics = load_test_tfidf_dics()
    word_flag_dic = load_word_flag()
    # top10 tfidf 关键词，经过词性筛选
    top10_tfidf_dics = topn_flag_filter(train_tfidf_dics, word_flag_dic)
    # top15 tfidf 关键词
    top15_tfidf_dics = topn_tfidf_dics(tfidf_dics=train_tfidf_dics, n=15)

    # top10 tfidf 关键词，经过词性筛选
    top10_test_tfidf_dics = topn_flag_filter(test_tfidf_dics, word_flag_dic)
    # top15 tfidf 关键词
    top15_test_tfidf_dics = topn_tfidf_dics(tfidf_dics=test_tfidf_dics, n=15)
    # 保存
    save(top10_test_tfidf_dics, top15_test_tfidf_dics, top10_tfidf_dics, top15_tfidf_dics)

    train_data = generate_features(top10_tfidf_dics, top15_tfidf_dics)
    save_train_data(train_data)

    train_data = transe_data(train_data)


    df = pd.DataFrame(train_data, columns=['newsId', 'word', 'tf', 'idf', 'tfidf',
                                            'title_has_other_sub', 'content_has_other_sub',
                                            'len_doc', 'sub_in_doc_count', 'sub_title_sim',
                                            'sub_content_sim', 'sub_tfidf_top_sim',
                                            'sub_is_in_title', 'count', 'first_offset',
                                            'left_entropy', 'right_entropy',
                                            'label'])
    df.to_csv("../runs/label_train_data.csv", index=False, sep='\t')


    test_data = generate_test_features(top10_test_tfidf_dics, top15_test_tfidf_dics)
    save_test_data(test_data)
    test_data = transe_data(test_data)
    df = pd.DataFrame(train_data, columns=['newsId', 'word', 'tf', 'idf', 'tfidf',
                                          'title_has_other_sub', 'content_has_other_sub',
                                           'len_doc', 'sub_in_doc_count', 'sub_title_sim',
                                           'sub_content_sim', 'sub_tfidf_top_sim',
                                           'sub_is_in_title', 'count', 'first_offset',
                                           'left_entropy', 'right_entropy'])
    df.to_csv("../runs/label_test_data.csv", index=False, sep='\t')

