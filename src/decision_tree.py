# __*__ coding: utf-8 __*__
__author__ = 'xb'
__date__ = '2018.5.16 14:38'

import numpy
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas
from data_process import text_to_number, data_normalization, n_gram_v2
from logistic_regression import do_metrics



def cross_validate():
    filename = '../data/6.5w_labeled.csv'
    user_agent = n_gram(filename)
    other_features, target = extract_feature(filename)
    other_features = numpy.array(other_features).astype(numpy.int32)
    target = numpy.array(target).astype(numpy.int32)
    # 矩阵降维（n, 1）--> (n,)
    target = target.ravel()
    all_features = numpy.concatenate((user_agent, other_features), axis = 1)
    # 样本随机
    all_features, target = shuffle(all_features, target, random_state = 0)
    #x_train, x_test, y_train, y_test = train_test_split(all_features, target, test_size = 0.4, random_state = 0)
    classifier = tree.DecisionTreeClassifier()
    scores = cross_val_score(classifier, all_features, target, cv = 10)
    print(scores)
    print(numpy.mean(scores))


def train_and_predict():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 1)
    class_features = text_to_number(filename)
    mouse_features = data_normalization(filename)
    #mouse_features = data[['mouse_x', 'mouse_y']]
    target = data.label
    # 矩阵降维（n, 1）--> (n,)
    target = target.ravel()
    all_features = numpy.concatenate((class_features, mouse_features, user_agent), axis = 1)
    # 样本随机
    all_features, target = shuffle(all_features, target, random_state = 0)
    x_train, x_test, y_train, y_test = train_test_split(all_features, target, test_size = 0.4, random_state = 0)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, '../models/DecisionTree.m')
    y_predict = classifier.predict(x_test)
    y_predict_proba = classifier.predict_proba(x_test)
    do_metrics(y_test, y_predict, y_predict_proba[:, 1])

if __name__ == '__main__':
    #cross_validate()
    train_and_predict()
