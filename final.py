#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np
from sklearn import preprocessing
from datetime import datetime

path = "input/"
path2 = "predict/"
data = pd.read_csv(path + "2question.csv",  encoding="gb18030")
allData = pd.read_csv(path2 + "predict.csv", encoding="gb18030")
allData = allData.sort_values("predict", ascending=False)
allData = allData.loc[allData["state"] != 2]

allData.to_csv(path2 + "final.csv", encoding="gb18030")
print("all is done")

