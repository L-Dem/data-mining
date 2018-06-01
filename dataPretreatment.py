import dataFunction
import pandas as pd
path = "input/"
pathOut = "output/"

'''
    筛选出购买的用户id--选择“购买”的用户
    '''
def do_data1():
    userShoppingTime = pd.read_csv(path + "1-1shoppingTime.csv", encoding="gb2312")
    userUseful = userShoppingTime[userShoppingTime["state"] == "购买"]
    userUseful.to_csv("1-1resultBasicFeature.csv", encoding="gb2312")
