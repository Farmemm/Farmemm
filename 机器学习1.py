# 导入必要的库
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
data = load_breast_cancer()

# 将数据转换为 DataFrame，方便分析
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 分析数据集的结构
print(df.head())  # 查看前几行数据
print(df.describe())  # 查看数据的统计描述

# 观察目标变量的分布情况
print(df['target'].value_counts())

# 输出每个类别的样本数量
print("Number of malignant tumors:", df[df['target'] == 0].shape[0])
print("Number of benign tumors:", df[df['target'] == 1].shape[0])

# 导入数据预处理相关的库
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 对数据集进行归一化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('target', axis=1))

# 将归一化后的数据转换回 DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=data.feature_names)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, df['target'], test_size=0.2, random_state=42)

# 导入逻辑回归模型
from sklearn.linear_model import LogisticRegression

# 使用逻辑回归模型进行训练
# 选择合适的超参数，例如惩罚项类型（L1 或 L2），并使用不同的正则化强度（C 参数）进行实验
logreg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')  # L2 正则化
logreg.fit(X_train, y_train)

# 可以选择不同的 C 值进行实验，例如：
# C_values = [0.001, 0.01, 0.1, 1, 10, 100]
# best_score = 0
# best_model = None
# for C in C_values:
#     model = LogisticRegression(penalty='l2', C=C, solver='liblinear')
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)
#     if score > best_score:
#         best_score = score
#         best_model = model
# print(f"Best C: {best_model.C}, Best Score: {best_score}")

# 计算并输出模型在训练集和测试集上的准确率
train_accuracy = logreg.score(X_train, y_train)
test_accuracy = logreg.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 使用混淆矩阵、精确率、召回率、F1分数等指标对模型进行评估
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

y_pred = logreg.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 绘制 ROC 曲线和 PR 曲线，并计算 AUC 值
from sklearn.metrics import roc_curve, auc, precision_recall_curve

y_pred_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('Rcc')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")

plt.show()