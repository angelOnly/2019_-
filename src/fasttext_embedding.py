#! -*- coding:utf-8 -*-

import codecs
import json
from gensim.models import FastText
import datetime


def train_model():
    start_time = datetime.datetime.now()
    f = codecs.open('../runs/all_docs.json', 'r', 'utf-8')
    sentences = json.load(f)
    model = FastText(sentences, size=200, window=3, min_count=2, iter=10, min_n=3, max_n=6, word_ngrams=0)
    model.save('../runs/fasttext_model')
    end_time = datetime.datetime.now()
    print((end_time - start_time).seconds)


def load_model():
    model = FastText.load('../runs/fasttext_model')
    return model


if __name__ == '__main__':
    train_model()