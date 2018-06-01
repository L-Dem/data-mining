#  encoding=utf8
import pandas as pd
from sklearn import preprocessing
import numpy as np
import time
import math


def encode_onehot(df, column_name):  # 按照column进行数据分类（一个特征的不同value值,分成很多列）
    feature_df = pd.get_dummies(df[column_name], prefix=column_name)  # 用int表示string，
    allin = pd.concat([df.drop([column_name], axis=1), feature_df], axis=1)  # 用分好类的表进行替换
    return allin


def encode_count(df, column_name):  # 表中column的种类和每种的个数
    lbl = preprocessing.LabelEncoder()  # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
    lbl.fit(list(df[column_name].values))  # 对表中column进行处理
    df[column_name] = lbl.transform(list(df[column_name].values))  # 将原始的feature变成用id表示的结果“yellow--1 red--2”
    return df


'''
# 给每一列在左侧增加一列id值
'''


def merge_count(df, columns, value, cname):  # 使用一个column值对另一个特征value进行分类
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()  # 又在左边增加了新的一列index
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_nunique(df, columns, value, cname):  # 返回组中column唯一元素的值
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_median(df, columns, value, cname):  # 返回组中column的中位数
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_mean(df, columns, value, cname):  # 返回组中column的均值
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_sum(df, columns, value, cname):  # 返回组中column的和
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_max(df, columns, value, cname):  # 返回组中column的最大值
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_min(df, columns, value, cname):  # 返回组中column的最小值
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def merge_std(df, columns, value, cname):  # 返回组中column的标准偏差
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns = columns+[cname]
    df = df.merge(add, on=columns, how="left")
    return df


def feat_count(df, df_feature, fe, value, name=""):  # 特征（行）
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)  # 空值填充为0
    return df


def feat_nunique(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_mean(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_mean_age(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(2.05)
    return df


def feat_mean_sex(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(1.52)
    return df


def feat_mean_education(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(2.3)  # 6.09
    return df


def feat_mean_major(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)  # 6.09
    return df


def feat_mean_model(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_std(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_median(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_max(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_max_tv(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(193383)
    return df


def feat_max_website(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)  # 21
    return df


def feat_max_activity(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(5357)
    return df


def feat_max_webshop(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(3368)
    return df


def feat_max_online(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(1556)
    return df


def feat_min(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_sum(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_var(df, df_feature, fe, value, name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_quantile(df, df_feature, fe, value, n, name=""):  # 扩展分位数
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_skew(df, df_feature, fe, value, name=""):  # 无偏的扩展偏度
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].skew()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_skew" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_value(df, df_feature, fe, value, name=""):  # 无偏的扩展偏度
    df_count = pd.DataFrame(df_feature.groupby(fe)[value]).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_skew" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def action_feats(df, df_features, fe="userid"):
    a = pd.get_dummies(df_features, columns=['actionType']).groupby(fe).sum()
    a = a[[i for i in a.columns if 'actionType' in i]].reset_index()
    df = df.merge(a, on=fe, how='left')
    return df


# 分组排序
def rank(data, feat1, feat2, ascending):  # 按照feat1 feat2进行排序 ascending为升降序
    data.sort_values([feat1, feat2], inplace=True, ascending=ascending)  # 在原地执行操作
    data['rank'] = range(data.shape[0])
    min_rank = data.groupby(feat1, as_index=False)['rank'].agg({'min_rank': 'min'})  # 对min_rank进行最小值聚合
    data = pd.merge(data, min_rank, on=feat1, how='left')
    data['rank'] = data['rank'] - data['min_rank']
    del data['min_rank']
    return data


# 对日期做一些处理
def get_date(timestamp):
    time_local = time.localtime(timestamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt


# 计算特征和类的平均值
def calcMean(x, y):
   sum_x = sum(x)
   sum_y = sum(y)
   n = len(x)
   x_mean = float(sum_x+0.0)/n
   y_mean = float(sum_y+0.0)/n
   return x_mean, y_mean


# 计算Pearson系数
def calcPearson(x, y):
    x_mean, y_mean = calcMean(x, y)   # 计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean, 2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean, 2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p


# # 计算每个特征的spearman系数
# def calcSpearman(x, y):
#     x_mean, y_mean = calcMean(x, y)  # 计算x,y向量平均值
#     n = len(x)
#     df = pd.DataFrame(x)
#     df["before"] = df.reset_index()
#     df.columns = ["data"] + ["beforeIndex"]
#     dfSort = df.sort_values(ascending=True)
#     dfSort.columns = [] + []
#     sumTop = 0.0
#     sumBottom = n*(n*n - 1)
#     x_pow = 0.0
#     y_pow = 0.0
#     for i in range(n):
#         sumTop += math.pow(d[i], 2)
#     sumTop = sumTop * 6
#     sp = 1 - sumTop / sumBottom


# 计算每个特征的spearman系数，返回数组
def calcAttribute(dataSet):
    prr = []
    n, m = np.shape(dataSet)    # 获取数据集行数和列数
    x = [0] * n                # 初始化特征x和类别y向量
    y = [0] * n
    for i in range(n):      # 得到类向量
        y[i] = dataSet[i][m-1]
    for j in range(m-1):    # 获取每个特征的向量，并计算Pearson系数，存入到列表中
        for k in range(n):
            x[k] = dataSet[k][j]
        prr.append(calcPearson(x, y))
    return prr

