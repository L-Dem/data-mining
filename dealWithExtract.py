#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np
from sklearn import preprocessing
from datetime import datetime

path = "input/"
pathOut = "output/"
data = pd.read_csv(path + "2question.csv",  encoding="gb18030")
name = data.columns.values.tolist()
print(name)
lengthIn = len(name)
userID = data["userID"].as_matrix()
userStayTime = data.as_matrix()
userShoppingTime = data
userShoppingTime["stayTime"] = userShoppingTime["stayTime"].str.extract("([0-9]+:[0-9][0-9])", expand=False)
userShoppingTime["shopping-minute"] = userShoppingTime["stayTime"].str.extract("([0-9]+)", expand=False)
userShoppingTime["shopping-minute"] = pd.to_numeric(userShoppingTime["shopping-minute"])
userShoppingTime["shopping-second"] = userShoppingTime["stayTime"].str.replace("([0-9]+:)", "")
userShoppingTime["shopping-second"] = pd.to_numeric(userShoppingTime["shopping-second"])
userShoppingTime["stayTime"] = userShoppingTime["shopping-minute"] * 60 + userShoppingTime["shopping-second"]

data["stayTime"] = userShoppingTime["stayTime"]
data.to_csv("extractNew.csv", encoding="gb18030")