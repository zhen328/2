# 基于朴素贝叶斯的中文邮件分类系统

## 核心功能

本系统实现了一个高效的中文邮件自动分类解决方案，能够准确区分垃圾邮件和普通邮件。系统采用多项式朴素贝叶斯作为核心分类算法，通过分析邮件文本内容中的词汇特征来进行智能判断。

## 算法说明

系统采用多项式朴素贝叶斯分类器，该算法具有以下特点：

1. **特征独立性假设**：假设每个特征（单词）在给定类别条件下相互独立，这大大简化了概率计算过程。
2. **概率计算机制**：通过计算单词在不同类别中出现的条件概率来进行分类预测。
3. **多项式分布**：专门处理离散型特征（如词频），适合文本分类场景。

## 数据处理流程

### 1. 文本清洗

系统首先对原始邮件文本进行深度清洗：
```python
# 使用正则表达式去除标点、数字等干扰字符
line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
```

### 2. 中文分词处理

采用jieba分词工具进行精准的中文分词：
```python
# 执行分词并过滤无效词汇
line = cut(line)  # jieba分词
line = filter(lambda word: len(word) > 1, line)  # 过滤单字词
```

### 3. 样本平衡处理

针对数据不平衡问题，系统集成了SMOTE过采样技术：
```python
# 使用SMOTE算法平衡样本分布
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
vector_resampled, labels_resampled = smote.fit_resample(vector, labels)
```

## 特征工程

### 高频词特征模式（默认）

系统默认采用高频词特征提取方式：
```python
# 统计特征词出现频次构建特征向量
word_map = list(map(lambda word: words.count(word), top_words))
vector.append(word_map)
```

### TF-IDF特征模式（可选）

也可切换至TF-IDF加权特征模式：
```python
# 使用TF-IDF加权构建特征向量
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=top_words)
vector = vectorizer.fit_transform([" ".join(words) for words in all_words])
```

## 模型评估体系

系统建立了完善的评估机制：
```python
# 数据划分保持原始分布
X_train, X_test, y_train, y_test = train_test_split(
    vector, labels, test_size=0.2, random_state=42, stratify=labels)

# 生成详细分类报告
print(classification_report(y_test, y_pred, target_names=['普通邮件', '垃圾邮件']))
```

# 代码运行结果

## 默认分类模式，对应代码classify.py
<img src="images/nlp_classify.png" width="800" alt="classify">

## 灵活切换方式
### 局部切换 对应代码classify_local.py
<img src="images/nlp_local.png" width="800" alt="local">

### 局部切换 对应代码classify_global.py
<img src="images/nlp_global.png" width="800" alt="global">

## 样本平衡处理
<img src="images/nlp_balance.png" width="800" alt="sample_balancing">

## 最终版_添加全局方法选择/样本平衡处理/模型评估指标
<img src="images/nlp_all.png" width="800" alt="classify_all">
