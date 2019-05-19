#! -*- coding:utf-8 -*-

import re
from tqdm import tqdm
import jieba
import codecs
import json
import os


def filter_text(text):
    re_tag0 = re.compile('</?\w+[^>]*>')  # HTML标签
    re_tag1 = re.compile(r'http://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.S)
    re_tag2 = re.compile(r'https://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',re.S)
    re_tag3 = re.compile('(?<=\>).*?(?=\<)')
    re_tag4 = re.compile('购买链接')
    re_tag5 = re.compile('京东：')
    re_tag6 = re.compile('淘宝：')
    re_tag7 = re.compile(r'\d.*?w|\d.*?v|\d.*?a|\d.*?亿元|\d.*?元|\d.*?plus')
    new_text = re.sub(re_tag0,"",text)
    new_text = re.sub(re_tag1,"",new_text)
    new_text = re.sub(re_tag2,"",new_text)
    new_text = re.sub(re_tag3,"",new_text)
    new_text = re.sub(re_tag4,"",new_text)
    new_text = re.sub(re_tag5,"",new_text)
    new_text = re.sub(re_tag6,"",new_text)
    new_text = re.sub(re_tag7, "", new_text)
    new_text = re.sub("-+", "-", new_text)  # 合并-
    new_text = re.sub("———+", "——", new_text)  # 合并-
    return new_text


def filter_cut_data():
    files = os.listdir('../字典')
    for file_name in files:
        jieba.load_userdict('../字典/' + file_name)
    jieba.load_userdict('../runs/entity.txt')

    train_data = []
    with open('../data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
        for l in tqdm(f):
            a = json.loads(l.strip())
            train_data.append(
                {
                    'newsId': a['newsId'],
                    'title': a['title'],
                    'content': a['content']
                }
            )

    test_data = []
    with open('../data/coreEntityEmotion_test_stage1.txt', encoding='utf-8') as f:
        for l in tqdm(f):
            a = json.loads(l.strip())
            test_data.append(
                {
                    'newsId': a['newsId'],
                    'title': a['title'],
                    'content': a['content']
                }
            )

    stop_words = []
    with open('../data/stop_words.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            stop_words.append(line)

    train_word_dic = {}
    train_docs = []
    i = 0
    for data in train_data:
        sentence = []
        line = filter_text(data['title']) + filter_text(data['content'])
        line = line.replace("\n", "")
        words = jieba.cut(line)
        words = filter(lambda x: x not in stop_words, words)
        sentence.extend(list(words))
        train_word_dic[data['newsId']] = sentence
        train_docs.append(sentence)
        if i % 500 == 0:
            print(i, ' train data finish')
        i += 1

    with codecs.open('../runs/train_word_dic.json', 'w', encoding='utf-8') as f:
        json.dump(train_word_dic, f, indent=4, ensure_ascii=False)

    with codecs.open('../runs/train_docs.json', 'w', encoding='utf-8') as f:
        json.dump(train_docs, f, indent=4, ensure_ascii=False)

    test_word_dic = {}
    test_docs = []
    i = 0
    for data in test_data:
        sentence = []
        line = filter_text(data['title']) + filter_text(data['content'])
        line = line.replace("\n", "")
        words = jieba.cut(line)
        words = filter(lambda x: x not in stop_words, words)
        sentence.extend(list(words))
        test_word_dic[data['newsId']] = sentence
        test_docs.append(sentence)
        if i % 500 == 0:
            print(i, ' test data finish')
        i += 1

    with codecs.open('../runs/test_word_dic.json', 'w', encoding='utf-8') as f:
        json.dump(test_word_dic, f, indent=4, ensure_ascii=False)

    with codecs.open('../runs/test_docs.json', 'w', encoding='utf-8') as f:
        json.dump(test_docs, f, indent=4, ensure_ascii=False)


def generate_embedding_data():
    f = codecs.open('../runs/test_docs.json', 'r', 'utf-8')
    test_docs = json.load(f)
    f = codecs.open('../runs/train_docs.json', 'r', 'utf-8')
    train_docs = json.load(f)
    all_docs = []
    all_docs.extend(test_docs)
    all_docs.extend(train_docs)
    with codecs.open('../runs/all_docs.json', 'w', encoding='utf-8') as f:
        json.dump(all_docs, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    filter_cut_data()
    generate_embedding_data()


