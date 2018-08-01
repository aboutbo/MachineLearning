# __*__ coding: utf-8 __*__
__author__ = 'xb'
__date__ = '2018.5.16 15:00'

import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas
from data_process import text_to_number, data_normalization, n_gram_v2
from logistic_regression import do_metrics


def cross_validate():
    file_black = '../data/6.5w_labeled_black.csv'
    file_white = '../data/6.5w_labeled_white.csv'
    # black IP对应的user-agent vocabulary
    user_agent_black = load_file(file_black)
    vectorizer_black = CountVectorizer(ngram_range = (2, 2), decode_error = "ignore",
                                        token_pattern = r'\b\w+\b', min_df = 1)
    x_black = vectorizer_black.fit_transform(user_agent_black).toarray()
    x_black_vocabulary = vectorizer_black.vocabulary_
    y_black = [0] * len(x_black)
    # 将black IP vocabulary用在white IP上
    user_agent_white = load_file(file_white)
    vectorizer_white = CountVectorizer(ngram_range = (2, 2), decode_error = 'ignore',
                                        token_pattern = r'\b\w+\b', min_df = 1, vocabulary = x_black_vocabulary)
    x_white = vectorizer_white.fit_transform(user_agent_white).toarray()
    y_white = [1] * len(x_white)
    x = numpy.concatenate((x_white, x_black), axis = 0)
    y = numpy.concatenate((y_white, y_black), axis = 0)
    # 样本随机
    #all_features, target = shuffle(all_features, target, random_state = 0)
    #x_train, x_test, y_train, y_test = train_test_split(all_features, target, test_size = 0.4, random_state = 0)
    classifier = GaussianNB()
    scores = cross_val_score(classifier, x, y, cv = 10)
    print(scores)
    print(numpy.mean(scores))

def train_and_predict():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 4)
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
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    joblib.dump(classifier, '../models/NativeBayes.m')
    y_predict = classifier.predict(x_test)
    y_predict_proba = classifier.predict_proba(x_test)
    do_metrics(y_test, y_predict, y_predict_proba[:, 1])

if __name__ == '__main__':
    #cross_validate()
    train_and_predict()
