#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np


path = "input/"
pathOut = "output/"
userActionFeature = pd.read_csv(path + "testAction.csv", encoding="gb18030")
userActionFeature = userActionFeature.drop_duplicates()
# user = dataFunction.feat_max(userShoppingTime, userActionFeature, ["userID"], "action", "basic-action")

userActionFeature["activity"] = np.where(userActionFeature["action"].str.contains("网络活跃"),
                                         userActionFeature["action"].str.extract("网络活跃指数:([0-9]+)"), "0")
userActionFeature["webShop"] = np.where(userActionFeature["action"].str.contains("网络购物"),
                                        userActionFeature["action"].str.extract("网络购物指数:([0-9]+)"), "0")
userActionFeature["onlineVideo"] = np.where(userActionFeature["action"].str.contains("在线视频指数"),
                                            userActionFeature["action"].str.extract("在线视频指数:([0-9]+)"), "0")

# userActivity = pd.DataFrame(userActionFeature["action"].str.split(",", expand=True)).reset_index()
# userActionFeature = userActionFeature.drop(columns="action")
# userActionFeature = pd.concat([userActionFeature, userActivity], axis=1)

# print(userActionFeature)
userActionFeature.to_csv(pathOut + "test1-3userAction.csv", encoding="gb2312")
