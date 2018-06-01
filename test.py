#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np
from sklearn import preprocessing
from datetime import datetime

path = "input/"
pathOut = "output/"
data = pd.read_csv(pathOut + "All-stateIn012.csv",  encoding="gb18030")
# name = data.columns.values.tolist()
# print(name)
# lengthIn = len(name)
# for i in range(1, lengthIn):
#     data = data.loc[data[name[i]] != 0]
data = data[data["state"] != 0]
data.sort_values("userID", ascending=True)
data.to_csv(pathOut + "haveAllIn12.csv", encoding="gb18030")
print("haveAll.csv is done")

