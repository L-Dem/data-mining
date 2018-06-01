#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np
from sklearn import preprocessing
from datetime import datetime

path = "input/"
pathOut = "output/"
data = pd.read_csv(path + "2question.csv",  encoding="gb18030")
AllData = pd.read_csv(pathOut + "All-stateIn012.csv", encoding="gb18030")
name = data.columns.values.tolist()
print(name)
lengthIn = len(name)
userID = data["userID"].as_matrix()
AllList = []
for i in userID:
    find = AllData.ix[AllData["userID"] == i]
    AllList.append(find)
result = pd.concat(AllList)
# print(result)
result.to_csv(pathOut + "extract.csv", encoding="gb18030")
print("haveAll0123.csv is done")

