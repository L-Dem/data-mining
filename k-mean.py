import scipy.stats as sp
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt


path = "output/"
data = pd.read_csv(path + "AllK-mean-3.csv", encoding="gb18030")
'''求每一个label的聚类均值'''
data = data[data["state"] == 2]
# data = data.drop("userID")
name = data.columns.values.tolist()
print(name)
r = len(name)
for i in range(1, r):
    data = data.loc[data[name[i]] != 0]
data = data.as_matrix()
n_clusters = 1
estimator = KMeans(n_clusters)
res = estimator.fit_predict(data)
labelPredict = estimator.labels_  # 预测类别标签结果
centroids = estimator.cluster_centers_  # 各个类别的聚类中心值
inertia = estimator.inertia_  # 聚类中心均值向量的总和
# print res
print("label")
print(labelPredict)
print("centroids")
print(centroids)
out1 = pd.DataFrame(centroids)
out1.to_csv(path + "centroids3.csv", encoding="gb18030")
print("inertia:")
print(inertia)
# count = data.shape[0]
# line = inertia / count
# print("count" + str(count))
# print("line" + str(int(line)))

# for j in range(1, 2):
#     for i in range(len(data)):
#         plt.figure(j)
#         if int(labelPredict[i]) == 0:
#             plt.scatter(data[i][4], data[i][5], color='red')
#         if int(labelPredict[i]) == 1:
#             plt.scatter(data[i][4], data[i][5], color='black')
#         plt.xlabel(name[4])
#         plt.ylabel(name[5])
#         # if int(lablePredict[i]) == 2:
#         #     plt.scatter(data[i][0], data[i][1], color='blue')
#
#     print("start drawing")
#     plt.show()
