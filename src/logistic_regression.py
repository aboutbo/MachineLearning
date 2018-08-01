# __*__ coding: utf-8 __*__
__author__ = 'xb'
__date__ = '2018.5.16 10:25'

import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics
from data_process import text_to_number, data_normalization, n_gram_v2


def cross_validate():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 3)
    class_features = text_to_number(filename)
    mouse_features = data_normalization(filename)
    target = data.label
    # 矩阵降维（n, 1）--> (n,)
    target = target.ravel()
    all_features = numpy.concatenate((class_features, mouse_features), axis = 1)
    # 样本随机
    all_features, target = shuffle(all_features, target, random_state = 0)
    #x_train, x_test, y_train, y_test = train_test_split(all_features, target, test_size = 0.4, random_state = 0)
    classifier = LogisticRegression()
    scores = cross_val_score(classifier, all_features, target, cv = 10)
    print(scores)
    print(numpy.mean(scores))

def train_and_predict():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 1)
    class_features = text_to_number(filename)
    mouse_features = data_normalization(filename)
    target = data.label
    # 矩阵降维（n, 1）--> (n,)
    target = target.ravel()
    all_features = numpy.concatenate((class_features, mouse_features, user_agent), axis = 1)
    # 样本随机
    all_features, target = shuffle(all_features, target, random_state = 0)
    x_train, x_test, y_train, y_test = train_test_split(all_features, target, test_size = 0.4, random_state = 0)
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, '../models/LogisticRegression.m')
    y_predict = classifier.predict(x_test)
    y_predict_proba = classifier.predict_proba(x_test)
    do_metrics(y_test, y_predict, y_predict_proba[:, 1])

# 计算recall, accuracy, precision, confusion matrix
def do_metrics(y_test, y_pred, y_predict_proba):
    print("accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("f1_score:")
    print(metrics.f1_score(y_test, y_pred))
    print('AUC:')
    print(metrics.roc_auc_score(y_test, y_predict_proba))
    print("confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    #cross_validate()
    train_and_predict()
