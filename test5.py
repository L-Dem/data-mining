#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np


path = "input/"
pathOut = "output/"
userBasic = pd.read_csv(path + "test1.csv", encoding="gb18030")
# userBasic["state"] = np.where(userBasic["state"].str.)
# userAll = dataFunction.feat_value(userBasic, userNew, ["userID"], "state", "state")
print(userBasic)

