# -*- coding: UTF-8 -*-
import os
import json
import warnings
from collections import Counter
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# 配置参数
class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 200

    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 1  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout

    dropoutKeepProb = 0.5  # 全连接层的dropout
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 1500  # 取了所有序列长度的均值
    batchSize = 128

    dataSource = "../output/train_df.csv"

    numClasses = 2

    rate = 0.8  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()


# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """

        df = pd.read_csv(filePath, sep='\t', names=['newsId', 'content', 'entity', 'emotion'])
        review = df["content"].tolist()

        def label_dic(label):
            if label == 'POS':
                return 1
            elif label == 'NORM':
                return 0
            elif label == 'NEG':
                return -1

        df["emotion"] = df["emotion"].apply(label_dic)
        labels = df["emotion"].tolist()

        import jieba
        files = os.listdir('../字典')
        for file_name in files:
            jieba.load_userdict('../字典/' + file_name)
        jieba.load_userdict('../runs/entity.txt')

        stop_words = []
        with open('../data/stop_words.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                stop_words.append(line)

        all_docs = []
        i = 0
        for data in review:
            words = jieba.cut(data)
            words = filter(lambda x: x not in stop_words, words)
            all_docs.append(list(words))
            if i % 500 == 0:
                print(i, ' data finish')
            i += 1

        return all_docs, labels

    def _reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """

        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]

        return reviewVec

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """

        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._wordToIndex)
            reviews.append(reviewVec)

            labels.append([y[i]])

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 2]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../output/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)

        with open("../output/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        from gensim.models import FastText
        wordVec = FastText.load('../runs/fasttext_model')
        vocab = []
        wordEmbedding = []

        # 添加 "pad" 和 "UNK",
        vocab.append("pad")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

if __name__ == '__main__':
    # 实例化配置参数对象
    config = Config()

    data = Dataset(config)
    data.dataGen()