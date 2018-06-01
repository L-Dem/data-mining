#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np
from sklearn import preprocessing
from datetime import datetime

path = "input/"
pathOut = "output/"
data = pd.read_csv(path + "2question.csv",  encoding="gb18030")
AllData = pd.read_csv(pathOut + "haveAllIn12.csv", encoding="gb18030")
name = data.columns.values.tolist()
print(name)
lengthIn = len(name)
userID = data["userID"].as_matrix()
AllList = []
for i in userID:
    AllData = AllData[AllData["userID"] != i]
# result = pd.concat(AllList)
# print(result)
AllData.to_csv(pathOut + "haveAllIn12_drop.csv", encoding="gb18030")
print("haveAll0123.csv is done")

