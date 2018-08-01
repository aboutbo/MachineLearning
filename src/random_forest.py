# __*__ coding: utf-8 __*__
__author__ = 'xb'
__date__ = '2018.5.16 14:47'

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas
from data_process import text_to_number, data_normalization, n_gram_v2
from logistic_regression import do_metrics


def cross_validate():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    target = data.label
    # 矩阵降维（n, 1）--> (n,)
    target = target.ravel()
    all_features = pandas.read_csv('../data/6.5w_distinct1_features.csv')
    # 样本随机
    all_features, target = shuffle(all_features, target, random_state = 0)
    x_train, x_test, y_train, y_test = train_test_split(all_features, target, test_size = 0.4, random_state = 0)
    param_test1 = {'max_features':range(1,33,10)}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_jobs = -1,
                                                               max_depth = 47,
                                                               min_samples_split = 2,
                                                               min_samples_leaf = 1,
                                                               max_features = 31,
                                                               n_estimators = 180),
                            param_grid = param_test1,
                            scoring='roc_auc',
                            cv=5)
    gsearch1.fit(x_train, y_train)
    print(gsearch1.best_estimator_, gsearch1.best_params_, gsearch1.best_score_)



def train_and_predict():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 2)
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
    classifier = RandomForestClassifier(n_jobs = -1,
                                        max_depth = 47,
                                        min_samples_split = 2,
                                        min_samples_leaf = 1,
                                        max_features = 31,
                                        n_estimators = 180)
    '''
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, '../models/RandomForest.m')
    y_predict = classifier.predict(x_test)
    y_predict_proba = classifier.predict_proba(x_test)
    do_metrics(y_test, y_predict, y_predict_proba[:, 1])
    '''
    scores = cross_val_score(classifier, all_features, target, cv = 10)
    print(scores)
    print(numpy.mean(scores))


if __name__ == '__main__':
    #cross_validate()
    train_and_predict()
