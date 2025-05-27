import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words


def get_text(filename):
    """读取文本内容并过滤无效字符"""
    with open(filename, 'r', encoding='utf-8') as fr:
        text = fr.read()
        # 过滤无效字符
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
        return text


def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


def extract_features(method='freq', top_num=100):
    """特征提取方法，可选择高频词特征或TF-IDF加权特征"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]

    if method == 'freq':
        # 高频词特征提取方法
        all_words = []
        for filename in filename_list:
            all_words.append(get_words(filename))

        top_words = get_top_words(top_num)
        vector = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)
        return np.array(vector)

    elif method == 'tfidf':
        # TF-IDF特征提取方法
        corpus = []
        for filename in filename_list:
            text = get_text(filename)
            # 使用jieba分词并用空格连接
            words = cut(text)
            words = [word for word in words if len(word) > 1]
            corpus.append(' '.join(words))

        vectorizer = TfidfVectorizer(max_features=top_num)
        X = vectorizer.fit_transform(corpus)
        return X.toarray()

    else:
        raise ValueError("Invalid method. Choose either 'freq' or 'tfidf'.")


def train_and_predict(feature_method='freq'):
    """训练模型并进行预测"""
    # 提取特征
    vector = extract_features(method=feature_method)

    # 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
    labels = np.array([1] * 127 + [0] * 24)

    # 训练模型
    model = MultinomialNB()
    model.fit(vector, labels)

    def predict(filename):
        """对未知邮件分类"""
        if feature_method == 'freq':
            # 高频词特征预测
            words = get_words(filename)
            current_vector = np.array(
                tuple(map(lambda word: words.count(word), get_top_words(100))))
        else:
            # TF-IDF特征预测
            text = get_text(filename)
            words = cut(text)
            words = [word for word in words if len(word) > 1]
            corpus = [' '.join(words)]
            vectorizer = TfidfVectorizer(max_features=100)
            # 需要先拟合训练数据，这里简化处理，实际应用中应该保存训练时的vectorizer
            train_corpus = []
            for i in range(151):
                text = get_text(f'邮件_files/{i}.txt')
                words = cut(text)
                words = [word for word in words if len(word) > 1]
                train_corpus.append(' '.join(words))
            vectorizer.fit(train_corpus)
            current_vector = vectorizer.transform(corpus).toarray()[0]

        # 预测结果
        result = model.predict(current_vector.reshape(1, -1))
        return '垃圾邮件' if result == 1 else '普通邮件'

    # 打印预测结果
    print(f'使用{feature_method}特征提取方法:')
    print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
    print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
    print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
    print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
    print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))


# 使用高频词特征
train_and_predict(feature_method='freq')

# 使用TF-IDF特征
train_and_predict(feature_method='tfidf')