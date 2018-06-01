#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np
from sklearn import preprocessing
from datetime import datetime

path = "input/"
pathOut = "output/"
data = pd.read_csv(path + "2question.csv",  encoding="gb18030")
allData = pd.read_csv(pathOut + "All-stateIn012.csv", encoding="gb18030")
name = data.columns.values.tolist()
print(name)
lengthIn = len(name)
result = allData[allData["model"] == 0]
result2 = allData[allData["model"] == 1]
result.to_csv(pathOut + "newIdea0.csv", encoding="gb18030")
result2.to_csv(pathOut + "newIdea1.csv", encoding="gb18030")
print("haveAll0123.csv is done")

