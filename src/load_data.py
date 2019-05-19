#! -*- coding:utf-8 -*-

import codecs
import json
from tqdm import tqdm

entity = set()
emotion = set()
chars = {}
data = []
min_count = 2

train_data_path = '../data/coreEntityEmotion_train.txt'

entity_path = '../runs/entity.txt'
emotion_path = '../runs/emotion.json'
char_path = '../runs/all_chars.json'
data_path = '../runs/all_train_data.json'

def load_train_data(train_data_path):
    with open(train_data_path, encoding='utf-8') as f:
        for l in tqdm(f):
            a = json.loads(l.strip())
            data.append(
                {
                    'newsId':a['newsId'],
                    'title': a['title'],
                    'content': a['content'],
                    'coreEntityEmotions': [(i['entity'], i['emotion']) for i in a['coreEntityEmotions']],
                }
            )
            for c in a['content']:
                chars[c] = chars.get(c, 0) + 1
            for c in a['title']:
                chars[c] = chars.get(c, 0) + 1
            for c in a['coreEntityEmotions']:
                entity.add(c['entity'])
                emotion.add(c['emotion'])


def save_entity(entity_path, entity):
    with open(entity_path, 'w', encoding='utf-8') as f:
        for v in entity:
            f.write(v+'\n')


def save_emotions(emotion_path, emotion):
    id2emotion = {i: j for i, j in enumerate(emotion)}
    emotion2id = {j: i for i, j in id2emotion.items()}

    with open(emotion_path, 'w', encoding='utf-8') as f:
        json.dump([id2emotion, emotion2id], f, indent=4, ensure_ascii=False)


def save_chars(char_path, chars):
    with codecs.open(char_path, 'w', encoding='utf-8') as f:
        chars = {i: j for i, j in chars.items() if j >= min_count}
        id2char = {i + 2: j for i, j in enumerate(chars)}  # padding: 0, unk: 1
        char2id = {j: i for i, j in id2char.items()}
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

def save_train_data(data_path, data):
    with codecs.open(data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    load_train_data(train_data_path)
    save_entity(entity_path, entity)
    save_chars(char_path, chars)
    save_emotions(emotion_path, emotion)
    save_train_data(data_path, data)