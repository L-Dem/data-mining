import scipy.stats as sp
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt


path = "output/"
data = pd.read_csv(path + "All-stateIn0123.csv", encoding="gb18030")
# data = data.loc[data["state"] != 0]
# data = data.loc[data["state"] != 0]
# data.to_csv(path + "testInvalid.csv", encoding="gb18030")

'''label相关系数'''
row = 0
column = 0

# index1 = ["state", "age", "sex", "education", "major"]  # 2
index1 = ["state", "stayTime", "rate", "activity", "webShop", "onlineVideo", "ph-time", "TV-time", "websiteName"]

lengthIndex = len(index1)
print(lengthIndex)
rho = np.zeros((lengthIndex, lengthIndex))
pval = np.zeros((lengthIndex, lengthIndex))
# # index2 = [data["state"], data["age"], data["sex"], data["education"], data["major"], data["model"],
#           data["rate"], data["activity"], data["webShop"], data["onlineVideo"]]
#
# i = 0
# j = 5
# dataIn = data
# dataIn = dataIn.loc[data[index1[i]] != 0]
# dataIn = dataIn.loc[data[index1[j]] != 0]
# dataIn.to_csv(path + "testInvalid.csv", encoding="gb18030")
# rho[i][j], pval[i][j] = sp.spearmanr(dataIn[index1[i]], dataIn[index1[j]])

for i in range(lengthIndex):
    for j in range(lengthIndex):
        dataIn = data
        dataIn = dataIn.loc[data[index1[i]] != 0]
        dataIn = dataIn.loc[data[index1[j]] != 0]
        rho[i][j], pval[i][j] = sp.spearmanr(dataIn[index1[i]], dataIn[index1[j]])
result1 = pd.DataFrame(rho)
result2 = pd.DataFrame(pval)
result1.to_csv(path + "spearman3-result1.csv", encoding="gb18030")
result2.to_csv(path + "spearman3-result2.csv", encoding="gb18030")
print("file done")
