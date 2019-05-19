#! -*- coding:utf-8 -*-

import jieba
import jieba.posseg as psg
import jieba.analyse
import codecs
import json
from tqdm import tqdm
import os

test_path = '../data/coreEntityEmotion_test_stage1.txt'
data_path = '../runs/all_train_data.json'
cut_train_path = '../runs/all_train_cut.json'
cut_test_path = '../runs/all_test_cut.json'
tran_dic_path = '../runs/train_dic.json'
test_dic_path = '../runs/test_dic.json'


def jieba_config():
    files = os.listdir('../dic')
    for file_name in files:
        jieba.load_userdict('../dic/'+file_name)
    jieba.load_userdict('../runs/entity.txt')
    jieba.analyse.set_stop_words('../data/stop_words.txt')

def load_train_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
        return data

def stop_words():
    stop_words = []
    with open('../data/stop_words.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            stop_words.append(line)
    return stop_words

def cut_train(data_path):
    final_train_data = []
    # word_flag_dic = {}
    train_data = load_train_data(data_path)
    i = 0
    for x in train_data:
        if len(x.items()) == 4:
            try:
                title = x['title']
                content = x['content']

                # title_words = jieba.cut(title)
                # content_words = jieba.cut(content)

                # title_filter = [word for word in title_words if word not in stop_words()]
                # content_filter = [word for word in content_words if word not in stop_words()]

                title_filter = filter(lambda x: x not in stop_words and len(x.strip()) > 0, psg.cut(title))
                content_filter = filter(lambda x: x not in stop_words and len(x.strip())>0, psg.cut(content))

                # word_flag_dic = dict((word,flag) for (word,flag) in [title_filter,content_filter])
                # for word, flag in title_filter:
                #     word_flag_dic[word] = flag
                # for word, flag in content_filter:
                #     word_flag_dic[word] = flag

                # temp = {}
                # temp['newsId'] = x['newsId']
                # temp['title'] = title_filter
                # temp['content'] = content_filter
                # entitys = [c[0] for c in x['coreEntityEmotions']]
                # temp['entities'] = entitys
                # final_train_data.append(temp)

                if i % 500 == 0:
                    print(i,' data finish')
                i+=1
            except KeyError:
                print(x)
                pass
    # with codecs.open(cut_train_path, 'w', encoding='utf-8') as f:
    #     json.dump(final_train_data, f, indent=4, ensure_ascii=False)
    # with codecs.open(tran_dic_path, 'w', encoding='utf-8') as f:
    #     json.dump(word_flag_dic, f, indent=4, ensure_ascii=False)
    return final_train_data

def load_test():
    test_chars = {}
    test_data = []
    min_count = 2

    with open(test_path, encoding='utf-8') as f:
        for l in tqdm(f):
            a = json.loads(l.strip())
            test_data.append(
                {
                    'newsId': a['newsId'],
                    'title': a['title'],
                    'content': a['content'],
                }
            )
            for c in a['content']:
                test_chars[c] = test_chars.get(c, 0) + 1
            for c in a['title']:
                test_chars[c] = test_chars.get(c, 0) + 1

    with codecs.open('../runs/test_chars.json', 'w', encoding='utf-8') as f:
        chars = {i: j for i, j in test_chars.items() if j >= min_count}
        id2char = {i + 2: j for i, j in enumerate(chars)}  # padding: 0, unk: 1
        char2id = {j: i for i, j in id2char.items()}
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
    return test_data

def cut_test():
    # word_flag_dic = {}
    final_test_data = []
    test_data = load_test()
    i = 0
    for x in test_data:
        if len(x.items()) == 3:
            try:
                title = x['title']
                content = x['content']

                title_words = jieba.cut(title)
                content_words = jieba.cut(content)

                # title_filter = [word for word in title_words if word not in stop_words()]
                # content_filter = [word for word in content_words if word not in stop_words()]

                title_filter = filter(lambda x: x not in stop_words and len(x.strip()) > 0, psg.cut(title))
                content_filter = filter(lambda x: x not in stop_words and len(x.strip()) > 0, psg.cut(content))

                # for word, flag in title_filter:
                #     word_flag_dic[word] = flag
                # for word, flag in content_filter:
                #     word_flag_dic[word] = flag

                # temp = {}
                # temp['newsId'] = x['newsId']
                # temp['title'] = title_filter
                # temp['content'] = content_filter
                # final_test_data.append(temp)
                if i%500 == 0:
                    print(i,' data finish')
                i+=1
            except KeyError:
                print(x)
                pass
    # with codecs.open(cut_test_path, 'w', encoding='utf-8') as f:
    #     json.dump(final_test_data, f, indent=4, ensure_ascii=False)
    # with codecs.open(test_dic_path, 'w', encoding='utf-8') as f:
    #     json.dump(final_test_data, f, indent=4, ensure_ascii=False)
    return final_test_data

if __name__ == '__main__':
    cut_train(data_path)
    cut_test()
