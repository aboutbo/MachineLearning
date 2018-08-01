# __*__ coding: utf-8 _*_
__author__ = 'xb'
__date__ = '2018.5.15 10:19'

import csv
import os
import pandas
import numpy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
from sklearn.feature_extraction.text import CountVectorizer


def extract_feature(file):
    feature_list = []
    target_list = []
    with open(file) as f:
        f_csv = csv.reader(f)
        for line in f_csv:
            feature = []
            target = []
            # ods_cust_media_web_login_form_event.login_page_type:0,1,2
            feature.append(line[21])
            # ods_cust_media_web_login_form_event.event_type:0,1,2
            feature.append(line[24])
            # ods_cust_media_web_login_form_event.cookie_enable:0,1,2
            if line[27] == 'true':
                feature.append('1')
            elif line[27] == 'false':
                feature.append('2')
            else:
                feature.append('0')
            # ods_cust_media_web_login_form_event.java_enable:0,1,2
            if line[28] == 'true':
                feature.append('1')
            elif line[27] == 'false':
                feature.append('2')
            else:
                feature.append('0')
            # ods_cust_media_web_login_form_event.mouse & ods_cust_media_web_login_form_event.screen:0,1
            mouse_x = int(float(line[29]))
            mouse_y = int(float(line[30]))
            screen_w = int(float(line[31]))
            screen_h = int(float(line[32]))
            if not (mouse_x or mouse_y or screen_w or screen_h):
                feature.append('0')
            else:
                feature.append('1')
            # white IP:1    black IP:0
            target.append(line[36])
            feature_list.append(feature)
            target_list.append(target)
    return feature_list, target_list

# 描述性特征转换为数字特征
def text_to_number(file):
    data = pandas.read_csv(file)
    columns = ['event_action', 'java_enable', 'cookie_enable']
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(categorical_features = [0], sparse = False)
    # 创建行数相同的空矩阵
    results = numpy.empty(shape = [data.shape[0], 0])
    for col in columns:
        # 描述性类别转化成数字
        data[col] = label_encoder.fit_transform(data[col])
        # 进一步进行one hot编码
        x = onehot_encoder.fit_transform(data[col].values.reshape(-1, 1))
        results = numpy.concatenate((results, x), axis = 1)
    return results

# 数据标准化
def data_normalization(file):
    data = pandas.read_csv(file)
    results = scale(data[['mouse_x', 'mouse_y']])
    return results

# user-agent使用n-gram特征化
def n_gram_v2(file, n):
    data = pandas.read_csv(file)
    ua = data.user_agent
    vectorizer = CountVectorizer(ngram_range = (n, n), decode_error = "ignore",
                                 token_pattern = r'\b[a-zA-Z0-9\.\/]+\b', min_df=1)
    #vectorizer = CountVectorizer(ngram_range = (n, n), decode_error = "ignore",
    #                             token_pattern = r'\b\w+\b', min_df=1)
    #x = vectorizer.fit_transform(ua)
    x = vectorizer.fit_transform(ua)
    vect_df = pandas.DataFrame(x.todense(), columns = vectorizer.get_feature_names())
    return vect_df



if __name__ == '__main__':
    results = text_to_number('../data/test_labeled.csv')
    print(results[:, 0])
