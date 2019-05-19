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
    new_text = re.sub(re_tag7,"",new_text)
    new_text = re.sub("-+", "-", new_text)  # 合并-
    new_text = re.sub("———+", "——", new_text)  # 合并-
    return new_text


def cut_data():
    files = os.listdir('../字典')
    for file_name in files:
        jieba.load_userdict('../字典/' + file_name)
    jieba.load_userdict('../runs/entity.txt')

    orig_data = []
    k = 0
    with open('../data/coreEntityEmotion_train.txt', encoding='utf-8') as f:
        for l in tqdm(f):
            a = json.loads(l.strip())
            orig_data.append(
                {
                    'newsId': a['newsId'],
                    'title': a['title'],
                    'content': a['content']
                }
            )
            k += 1
            if k%20==0:
                print('k', k)
            if k==100:
                break

    stop_words = []
    with open('../data/stop_words.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            stop_words.append(line)

    small_word_dic = {}
    i = 0
    for data in orig_data:
        sentence = []
        title = filter_text(data['title'])
        title = title.replace("\n", "")
        content = filter_text(data['content'])
        content = content.replace("\n", "")
        titles = jieba.cut(title)
        contents = jieba.cut(content)
        titles = filter(lambda x: x not in stop_words, titles)
        contents = filter(lambda x: x not in stop_words, contents)
        sentence.append(list(titles))
        sentence.append(list(contents))
        small_word_dic[data['newsId']] = sentence
        if i % 500 == 0:
            print(i, ' data finish')
        i += 1

    with codecs.open('../runs/small_word_dic.json', 'w', encoding='utf-8') as f:
        json.dump(small_word_dic, f, indent=4, ensure_ascii=False)
    return small_word_dic

