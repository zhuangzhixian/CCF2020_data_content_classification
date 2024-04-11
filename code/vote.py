import os
import pandas as pd
import numpy as np

directory = './result/'
files = [file for _, _, file in os.walk(directory)][0]

# 初始化结果存储列表
res = []

# 读取所有预测结果文件
for file in files:
    tmp = pd.read_csv(directory + file)
    # 假设class_label列的每个元素是以逗号分隔的多个类别
    labels = tmp["class_label"].apply(lambda x: x.split(','))
    res.append(labels)

# 获取类别的总数
num_categories = len(set([label for sublist in res for item in sublist for label in item]))
# 初始化一个全0的数组，用于存放每个样本每个类别出现的次数
vote_counts = np.zeros((len(res[0]), num_categories), dtype=int)

# 类别到索引的映射
category_to_index = {category: index for index, category in enumerate(sorted(set([label for sublist in res for item in sublist for label in item])))}

# 累计每个类别的出现次数
for labels_list in res:
    for i, labels in enumerate(labels_list):
        for label in labels:
            index = category_to_index[label]
            vote_counts[i, index] += 1

# 设置阈值为模型数量的一半，表示如果超过半数模型预测为正类，则认为该样本属于该类
threshold = len(files) // 2
predicted_labels = []

# 根据阈值判断最终的类别集合
for i in range(len(vote_counts)):
    sample_labels = [category for category, index in category_to_index.items() if vote_counts[i, index] > threshold]
    predicted_labels.append(','.join(sample_labels))

# 读取提交文件模板
sub = pd.read_csv('data/submit_example.csv')
# 将预测结果赋值给class_label列
sub['class_label'] = predicted_labels
# 保存最终的预测结果
sub.to_csv("./result_vote.csv", index=False)
