import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------
# 预处理模块
# --------------------------
def get_words(filename):
    """读取整个文本并过滤无效字符和长度为1的词"""
    with open(filename, 'r', encoding='utf-8') as fr:
        text = fr.read().strip()
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
        words = cut(text)
        return [word for word in words if len(word) > 1]


def custom_tokenizer(text):
    """TF-IDF专用的分词函数（保持与get_words逻辑一致）"""
    text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
    return [word for word in cut(text) if len(word) > 1]


# --------------------------
# 特征工程模块
# --------------------------
def get_feature_extractor(feature_type, filenames, top_num=100):
    """特征提取工厂函数"""
    if feature_type == 'frequency':
        # 高频词特征模式
        all_words = [get_words(fname) for fname in filenames]
        top_words = [w[0] for w in Counter(chain(*all_words)).most_common(top_num)]
        return {
            'type': 'frequency',
            'features': [[words.count(w) for w in top_words] for words in all_words],
            'extractor': top_words
        }

    elif feature_type == 'tfidf':
        # TF-IDF特征模式
        corpus = [open(fname, 'r', encoding='utf-8').read().strip() for fname in filenames]
        vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer,
            token_pattern=None,
            max_features=top_num
        )
        return {
            'type': 'tfidf',
            'features': vectorizer.fit_transform(corpus).toarray(),
            'extractor': vectorizer
        }


# --------------------------
# 模型训练模块
# --------------------------
def train_model(feature_type='frequency', top_num=100):
    """训练带特征选择的分类器"""
    # 准备训练数据路径
    train_files = [f'邮件_files/{i}.txt' for i in range(151)]

    # 获取特征矩阵和提取器
    feature_data = get_feature_extractor(feature_type, train_files, top_num)

    # 准备标签
    labels = np.array([1] * 127 + [0] * 24)

    # 训练模型
    model = MultinomialNB()
    model.fit(feature_data['features'], labels)

    return model, feature_data['extractor']


# --------------------------
# 预测模块
# --------------------------
def predict_file(filename, model, extractor, feature_type):
    """预测单个文件"""
    if feature_type == 'frequency':
        # 高频词模式特征生成
        words = get_words(filename)
        features = np.array([words.count(w) for w in extractor])
    elif feature_type == 'tfidf':
        # TF-IDF模式特征生成
        text = open(filename, 'r', encoding='utf-8').read().strip()
        features = extractor.transform([text]).toarray()[0]

    # 执行预测
    pred = model.predict(features.reshape(1, -1))[0]
    return '垃圾邮件' if pred == 1 else '普通邮件'


# --------------------------
# 执行示例
# --------------------------
if __name__ == "__main__":
    # 配置参数
    FEATURE_TYPE = 'tfidf'  # 可切换为 frequency
    TOP_NUM = 100

    # 训练模型
    model, extractor = train_model(feature_type=FEATURE_TYPE, top_num=TOP_NUM)

    # 预测测试文件
    test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]
    for file in test_files:
        if os.path.exists(file):
            result = predict_file(file, model, extractor, FEATURE_TYPE)
            print(f'{os.path.basename(file)} 分类结果: {result}')
        else:
            print(f'文件 {file} 不存在')