#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np


path = "input/"
pathOut = "output/"
userTVTime = pd.read_csv(path + "test4.csv", encoding="gb18030")
userTVTime = userTVTime.drop_duplicates()
userTVTime["TVTime"] = userTVTime["TVTime"].str.extract("([0-9]+:[0-9][0-9])")
userTVTime["TV-minute"] = userTVTime["TVTime"].str.extract("([0-9]+)")
userTVTime["TV-minute"] = pd.to_numeric(userTVTime["TV-minute"])
userTVTime["TV-second"] = userTVTime["TVTime"].str.replace("([0-9]+:)", "")
userTVTime["TV-second"] = pd.to_numeric(userTVTime["TV-second"])
userResultTV = userTVTime.groupby(["userID"])["TV-minute"].sum() * 60 + \
               userTVTime.groupby(["userID"])["TV-second"].sum()
userResultTV = pd.DataFrame(userResultTV)
userResultTV = userResultTV.reset_index()
userResultTV.columns = ["userID"] + ["TV-time"]
userResultTV = dataFunction.merge_sum(userResultTV, ["userID"], "TV-time", "sumTV-Time")
print(userResultTV)
userResultTV.to_csv(pathOut + "1-5tvTime.csv", encoding="gb18030")
