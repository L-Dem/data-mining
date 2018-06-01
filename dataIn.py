#  encoding=utf8
import pandas as pd
import dataFunction
import numpy as np

path = "input/"
pathOut = "output/"
userBasic = pd.read_csv(path + "basic.csv",  encoding="gb18030")
userShoppingTime = pd.read_csv(path + "1-1shoppingTime.csv",  encoding="gb18030")
# test1 = pd.read_csv(path + "test1.csv",  encoding="gb2312")
userBasicFeature = pd.read_csv(path + "1-2userBasicFeature.csv", encoding="gb18030")
userActionFeature = pd.read_csv(path + "1-3userActionFeature.csv", encoding="gb18030")
userPhoneTime = pd.read_csv(path + "1-4userPhoneTime.csv", encoding="gb18030")
userTVTime = pd.read_csv(path + "1-5userTVTime.csv", encoding="gb18030")
userWebsite = pd.read_csv(path + "1-6userWebsite.csv", encoding="gb18030")


'''1.筛选出购买的用户id--选择“购买”的用户'''

userShoppingTime = userShoppingTime.sort_values("state", ascending=False)  # 排序
userShoppingTime = userShoppingTime.drop_duplicates()
userShoppingTime["state"] = userShoppingTime["state"].str.replace("搜索", "1")
userShoppingTime["state"] = userShoppingTime["state"].str.replace("购买", "2")
userShoppingTime["state"] = userShoppingTime["state"].str.replace("浏览", "1")
userShoppingTime["state"] = pd.to_numeric(userShoppingTime["state"])
userShoppingTime["stayTime"] = userShoppingTime["stayTime"].str.extract("([0-9]+:[0-9][0-9])", expand=False)
userShoppingTime["shopping-minute"] = userShoppingTime["stayTime"].str.extract("([0-9]+)", expand=False)
userShoppingTime["shopping-minute"] = pd.to_numeric(userShoppingTime["shopping-minute"])
userShoppingTime["shopping-second"] = userShoppingTime["stayTime"].str.replace("([0-9]+:)", "")
userShoppingTime["shopping-second"] = pd.to_numeric(userShoppingTime["shopping-second"])
userShoppingTime["stayTime"] = userShoppingTime["shopping-minute"] * 60 + userShoppingTime["shopping-second"]
userShoppingTime.to_csv(pathOut + "1-1shoppingFeature.csv", encoding="gb18030")
print("1-done")

'''2.用户数据--去重---done'''
userBasicFeature = userBasicFeature.drop_duplicates()
userBasicFeature.to_csv(pathOut + "1-2resultBasicFeature.csv", encoding="gb2312")
userBasicFeature["age"] = userBasicFeature["age"].str.replace("18-24", "1")
userBasicFeature["age"] = userBasicFeature["age"].str.replace("25-34", "2")
userBasicFeature["age"] = userBasicFeature["age"].str.replace("35-44", "3")
userBasicFeature["age"] = userBasicFeature["age"].str.replace("45-54", "4")
userBasicFeature["age"] = pd.to_numeric(userBasicFeature["age"])
userBasicFeature["sex"] = userBasicFeature["sex"].str.replace("男", "1")
userBasicFeature["sex"] = userBasicFeature["sex"].str.replace("女", "2")
userBasicFeature["sex"] = pd.to_numeric(userBasicFeature["sex"])
userBasicFeature["education"] = userBasicFeature["education"].str.replace("大学本科", "3")
userBasicFeature["education"] = userBasicFeature["education"].str.replace("高中及以下", "1")
userBasicFeature["education"] = userBasicFeature["education"].str.replace("硕士及以上", "4")
userBasicFeature["education"] = userBasicFeature["education"].str.replace("大学专科", "2")
userBasicFeature["education"] = userBasicFeature["education"].str.replace("其它", "0")
userBasicFeature["education"] = pd.to_numeric(userBasicFeature["education"])
userBasicFeature["major"] = userBasicFeature["major"].str.replace("服务业", "11")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("公务员_翻译_其他", "1")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("广告_市场_媒体_艺术", "2")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("会计_金融_银行_保险", "3")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("计算机_互联网_通信_电子", "4")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("建筑_房地产", "5")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("人事_行政_高级管理", "6")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("生产_运营_采购_物流", "7")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("生物_制药_医疗_护理", "8")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("销售_客服_技术支持", "9")
userBasicFeature["major"] = userBasicFeature["major"].str.replace("咨询_法律_教育_科研", "10")
userBasicFeature["major"] = pd.to_numeric(userBasicFeature["major"])
print("2-done")

'''3.'''
userActionFeature = userActionFeature.drop_duplicates()
# user = dataFunction.feat_max(userShoppingTime, userActionFeature, ["userID"], "action", "basic-action")
userActionFeature["activity"] = np.where(userActionFeature["action"].str.contains("网络活跃"),
                                         userActionFeature["action"].str.extract("网络活跃指数:([0-9]+)", expand=False), "0")
userActionFeature["webShop"] = np.where(userActionFeature["action"].str.contains("网络购物"),
                                        userActionFeature["action"].str.extract("网络购物指数:([0-9]+)", expand=False), "0")
userActionFeature["onlineVideo"] = np.where(userActionFeature["action"].str.contains("在线视频指数"),
                                            userActionFeature["action"].str.extract("在线视频指数:([0-9]+)", expand=False), "0")
userActionFeature.to_csv(pathOut + "1-3userAction.csv", encoding="gb18030")
print("3-done")

'''4.'''
userPhoneTime = userPhoneTime.drop_duplicates()
userPhoneTime["model"] = userPhoneTime["model"].str.replace('\n', '')   # 删除换行负、字符串前后空格
userPhoneTime["model"] = userPhoneTime["model"].map(str.strip)
userPhoneTime["model"] = np.where(userPhoneTime["model"].str.contains("Surpass"), "Surpass", "other")
userPhoneTime = dataFunction.merge_count(userPhoneTime, ["phoneTime"], "userID", "allTime")
userPhoneTime["phoneTime"] = userPhoneTime["phoneTime"].str.extract("([0-9]+:[0-9][0-9])",expand=False)
userPhoneTime["ph-minute"] = userPhoneTime["phoneTime"].str.extract("([0-9]+)", expand=False)
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
userResultPhone["model"] = userResultPhone["model"].str.replace("Surpass", "1")
userResultPhone["model"] = userResultPhone["model"].str.replace("other", "2")
userResultPhone["model"] = pd.to_numeric(userResultPhone["model"])
# userResultPhone["ph-time"] = pd.to_numeric(userResultPhone["ph-time"])
# userResultPhone["ph-sumTime"] = pd.to_numeric(userResultPhone["ph-sumTime"])
# userResultPhone["rate"] = pd.to_numeric(userResultPhone["rate"])
userResultPhone.to_csv(pathOut + "1-4phoneTime.csv", encoding="gb18030")
print("4-done")


'''5'''
userTVTime = userTVTime.drop_duplicates()
userTVTime["TVTime"] = userTVTime["TVTime"].str.extract("([0-9]+:[0-9][0-9])", expand=False)
userTVTime["TV-minute"] = userTVTime["TVTime"].str.extract("([0-9]+)", expand=False)
userTVTime["TV-minute"] = pd.to_numeric(userTVTime["TV-minute"])
userTVTime["TV-second"] = userTVTime["TVTime"].str.replace("([0-9]+:)", "")
userTVTime["TV-second"] = pd.to_numeric(userTVTime["TV-second"])
userResultTV = userTVTime.groupby(["userID"])["TV-minute"].sum() * 60 + \
               userTVTime.groupby(["userID"])["TV-second"].sum()
userResultTV = pd.DataFrame(userResultTV)
userResultTV = userResultTV.reset_index()
userResultTV.columns = ["userID"] + ["TV-time"]

userResultTV.to_csv(pathOut + "1-5tvTime.csv", encoding="gb18030")
print("5-done")

'''6'''
userWebsite = userWebsite.drop_duplicates()
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("IT数码", "21")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("电子商务", "1")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("房产家居", "2")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("交通旅游", "3")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("汽车", "4")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("女性时尚", "5")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("教学及考试", "6")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("人才招聘", "7")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("社交网络和在线社区", "8")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("生活服务", "9")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("搜索服务", "10")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("投资金融", "11")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("网络服务应用", "12")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("网址导航", "13")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("新闻媒体", "14")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("休闲娱乐", "15")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("音乐", "16")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("医疗保健", "17")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("游戏", "18")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("在线视频", "19")
userWebsite["websiteName"] = userWebsite["websiteName"].str.replace("垂直行业", "20")
userWebsite["websiteName"] = pd.to_numeric(userWebsite["websiteName"])
userWebsite.to_csv(pathOut + "1-6website.csv", encoding="gb18030")
print("6-done")

'''编码与合并'''
userAll = dataFunction.feat_max(userBasic, userShoppingTime, ["userID"], "state", "state")
userAll = dataFunction.feat_sum(userAll, userShoppingTime, ["userID"], "stayTime", "stayTime")
userAll = dataFunction.feat_mean_age(userAll, userBasicFeature, ["userID"], "age", "age")
userAll = dataFunction.feat_mean_sex(userAll, userBasicFeature, ["userID"], "sex", "sex")
userAll = dataFunction.feat_mean_education(userAll, userBasicFeature, ["userID"], "education", "education")
userAll = dataFunction.feat_mean_major(userAll, userBasicFeature, ["userID"], "major", "major")
# *****userAll = dataFunction.feat_mean(userAll, userActionFeature, ["userID"], "action", "action")
userAll = dataFunction.feat_mean_model(userAll, userResultPhone, ["userID"], "model", "model")
userAll = dataFunction.feat_sum(userAll, userResultPhone, ["userID"], "ph-time", "ph-time")
userAll = dataFunction.feat_sum(userAll, userResultPhone, ["userID"], "ph-sumTime", "ph-sumTime")
userAll = dataFunction.feat_max(userAll, userResultPhone, ["userID"], "rate", "rate")
userAll = dataFunction.feat_max_tv(userAll, userResultTV, ["userID"], "TV-time", "TV-time")
userAll = dataFunction.feat_max_website(userAll, userWebsite, ["userID"], "websiteName", "websiteName")
userAll = dataFunction.feat_max_activity(userAll, userActionFeature, ["userID"], "activity", "activity")
userAll = dataFunction.feat_max_webshop(userAll, userActionFeature, ["userID"], "webShop", "webShop")
userAll = dataFunction.feat_max_online(userAll, userActionFeature, ["userID"], "onlineVideo", "onlineVideo")
# userAll = dataFunction.feat_mean(userAll, userTVTime, ["userID"], "w-class", "w-class")
# userAll = dataFunction.feat_mean(userAll, userTVTime, ["userID"], "website", "website")
userAll.to_csv(pathOut + "All-stateIn012.csv", encoding="gb18030")
print("all-done")
