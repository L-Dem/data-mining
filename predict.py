import scipy.stats as sp
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt


path = "output/"
pathOut = "predict/"
data = pd.read_csv(path + "AllK-mean-use.csv", encoding="gb18030")
data["activity"] = data["activity"] / data["activity"].max()
data["stayTime"] = data["stayTime"] / data["stayTime"].max()
data["major"] = data["major"] / data["major"].max()


data["sex"] = data["sex"] / data["sex"].max()
data["age"] = data["age"] / data["age"].max()
data["education"] = data["education"] / data["education"].max()
data["websiteName"] = data["websiteName"] / data["education"].max()
# data["stayTime"].apply(lambda x: x/x.max())
# data["major"].apply(lambda x: x/x.max())
# data["age"].apply(lambda x: x/x.max())
# data["education"].apply(lambda x: x/x.max())
# data["websiteName"].apply(lambda x: x/x.max())
new = data
'''求每一个label的聚类均值'''
data = data[data["state"] == 2]
data = data.drop("userID", 1)
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
# print("label")
# print(labelPredict)
print("centroids")
print(centroids)
out1 = pd.DataFrame(centroids)
out1.to_csv(path + "centroids3.csv", encoding="gb18030")
print("inertia:")
print(inertia)
count = data.shape[0]
line = inertia / count
print("count:" + str(count))
print("line:" + str(int(line)))
k = 1
new["predict"] = 0
# h = centroids.as_matrix()
for i in range(0, r):
    # print(name[i])
    # print(k)
    # print(centroids[0][i])
    # print(new["predict"])
    new["predict"] = k * (new[name[i]] - centroids[0][i]) + new["predict"]
# new["predict"] = new["predict"] / new["predict"].max()
new.to_csv(pathOut + "predict.csv", encoding="gb18030")

