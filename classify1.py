import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.utils.validation import validate_data


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    try:
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
    except FileNotFoundError:
        print(f"警告: 文件 {filename} 未找到")
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")
    return words


def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    if top_num <= 0:
        raise ValueError("top_num 必须为正整数")

    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


def extract_features(selection_method, top_num=100):
    if selection_method not in ['high_frequency', 'tf_idf']:
        raise ValueError("不支持的特征选择方法，请选择 'high_frequency' 或 'tf_idf'")

    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]

    if selection_method == 'high_frequency':
        if top_num <= 0:
            raise ValueError("top_num 必须为正整数")
        top_words = get_top_words(top_num)
        all_words = []
        for filename in filename_list:
            all_words.append(get_words(filename))
        vector = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)
        vector = np.array(vector)
        return vector
    else:  # tf_idf
        corpus = []
        for filename in filename_list:
            words = get_words(filename)
            corpus.append(' '.join(words))
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform(corpus)
        return vector.toarray()


def main():
    selection_method = 'tf_idf'  # 可以切换为 'high_frequency'
    try:
        vector = extract_features(selection_method)
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return

    # 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
    labels = np.array([1] * 127 + [0] * 24)

    # 使用SMOTE进行过采样
    try:
        smote = SMOTE(random_state=42)
        vector_resampled, labels_resampled = smote.fit_resample(vector, labels)
    except Exception as e:
        print(f"过采样失败: {str(e)}")
        return

    model = MultinomialNB()
    try:
        model.fit(vector_resampled, labels_resampled)
    except Exception as e:
        print(f"模型训练失败: {str(e)}")
        return

    # 假设测试集文件范围是151 - 155
    test_files = ['邮件_files/{}.txt'.format(i) for i in range(151, 156)]
    true_labels = [0] * 5  # 这里假设测试集全是普通邮件，实际使用时需要根据真实情况修改
    predicted_labels = []

    def predict(filename, selection_method=selection_method):
        """对未知邮件分类"""
        try:
            if selection_method == 'high_frequency':
                top_words = get_top_words(100)  # 使用默认top_num=100
                # 构建未知邮件的词向量
                words = get_words(filename)
                current_vector = np.array(
                    tuple(map(lambda word: words.count(word), top_words)))
            elif selection_method == 'tf_idf':
                filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
                corpus = []
                for f in filename_list:
                    ws = get_words(f)
                    corpus.append(' '.join(ws))
                vectorizer = TfidfVectorizer()
                vectorizer.fit(corpus)
                words = get_words(filename)
                current_vector = vectorizer.transform([' '.join(words)]).toarray()
            else:
                raise ValueError("不支持的特征选择方法")

            # 预测结果
            result = model.predict(current_vector.reshape(1, -1))
            return '垃圾邮件' if result == 1 else '普通邮件'
        except Exception as e:
            print(f"预测文件 {filename} 时出错: {str(e)}")
            return '未知'

    for filename in test_files:
        result = predict(filename, selection_method=selection_method)
        predicted_labels.append(1 if result == '垃圾邮件' else 0)
        print(f'{filename} 分类情况: {result}')

    # 输出分类评估报告
    print("\n分类评估报告:")
    try:
        # 确保评估报告中有所有可能的标签
        labels_to_show = [0, 1] if 1 in true_labels or 1 in predicted_labels else [0]
        print(classification_report(
            true_labels,
            predicted_labels,
            labels=labels_to_show,
            zero_division=0
        ))
    except Exception as e:
        print(f"生成评估报告失败: {str(e)}")


if __name__ == '__main__':
    print("Building prefix dict from the default dictionary ...")
    main()