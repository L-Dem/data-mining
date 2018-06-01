# encoding=utf8
import pandas as pd
import dataFunction
import numpy as np


path = "input/"
pathOut = "output/"
userPhoneTime = pd.read_csv(path + "test3.csv", encoding="gb18030")

userPhoneTime = userPhoneTime.drop_duplicates()
'''删除换行负、字符串前后空格'''
userPhoneTime["model"] = userPhoneTime["model"].str.replace('\n', '')
userPhoneTime["model"] = userPhoneTime["model"].map(str.strip)
userPhoneTime["model"] = np.where(userPhoneTime["model"].str.contains("Surpass"), "Surpass", "other")
userPhoneTime = dataFunction.merge_count(userPhoneTime, ["phoneTime"], "userID", "allTime")
userPhoneTime["phoneTime"] = userPhoneTime["phoneTime"].str.extract("([0-9]+:[0-9][0-9])")
userPhoneTime["ph-minute"] = userPhoneTime["phoneTime"].str.extract("([0-9]+)")
userPhoneTime["ph-minute"] = pd.to_numeric(userPhoneTime["ph-minute"])
userPhoneTime["ph-second"] = userPhoneTime["phoneTime"].str.replace("([0-9]+:)", "")
userPhoneTime["ph-second"] = pd.to_numeric(userPhoneTime["ph-second"])
userResultPhone = userPhoneTime.groupby(["userID", "model"])["ph-minute"].sum() * 60 + \
                  userPhoneTime.groupby(["userID", "model"])["ph-second"].sum()
userResultPhone = pd.DataFrame(userResultPhone)
userResultPhone = userResultPhone.reset_index()
userResultPhone.columns = ["userID"] + ["model"] + ["ph-time"]
userResultPhone = dataFunction.merge_sum(userResultPhone, ["userID"], "ph-time", "ph-sumTime")
userResultPhone["rate"] = userResultPhone["ph-time"] / userResultPhone["ph-sumTime"]
userResultPhone = userResultPhone.loc[userResultPhone["model"] == "Surpass"]
userResultPhone.to_csv(pathOut + "1-4phoneTime.csv", encoding="gb18030")
