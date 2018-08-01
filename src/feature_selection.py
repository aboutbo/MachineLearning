import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import pandas
from data_process import text_to_number, data_normalization, n_gram_v2
from logistic_regression import do_metrics


def random_forest():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 1)
    # 所有特征放入DataFrame
    all_features = user_agent
    class_features = text_to_number(filename)
    class_features_names = ['event_action_account_input',
                            'event_action_pwd_input',
                            'event_action_login_click',
                            'java_enable',
                            'java_disable',
                            'cookie_enable',
                            'cookie_disenable']
    count = 0
    for name in class_features_names:
        all_features[name] = class_features[:, count]
        count += 1
    #mouse_features = data[['mouse_x', 'mouse_y']]
    mouse_features = data_normalization(filename)
    all_features['mouse_x'] = mouse_features[:, 0]
    all_features['mouse_y'] = mouse_features[:, 1]
    #all_features['mouse_x'] = data['mouse_x']
    #all_features['mouse_y'] = data['mouse_y']
    target = data.label
    # 矩阵降维（n, 1）--> (n,)
    target = target.ravel()
    #all_features = numpy.concatenate((class_features, mouse_features, user_agent), axis = 1)
    # 样本随机
    #all_features, target = shuffle(all_features, target, random_state = 0)

    classifier = RandomForestClassifier()
    classifier.fit(all_features, target)
    feature_importances_indices = numpy.argsort(classifier.feature_importances_)
    for num in range(numpy.shape(all_features)[1]):
        print('%d. %s  %f' % (num, all_features.columns.values[feature_importances_indices[num]], classifier.feature_importances_[feature_importances_indices[num]]))
    #final_features = SelectFromModel(RandomForestClassifier()).fit_transform(all_features, target)
    #print(numpy.shape(final_features))
    #return final_features, target

def low_variance():
    all_features = pandas.read_csv('../data/6.5w_distinct1_features.csv')
    #selector = VarianceThreshold()
    #x = selector.fit_transform(all_features)
    #print(all_features.ix[:, 1072])
    var_list = []
    for num in range(numpy.shape(all_features)[1]):
        if num == 0:
            continue
        var_list.append(numpy.var(all_features.ix[:, num].values))
    indices = numpy.argsort(var_list)
    for num in range(numpy.shape(indices)[0]):
        print('%d. %s  %f' % (num + 1, all_features.columns.values[indices[num] + 1], var_list[indices[num]]))


def write_features_to_file():
    filename = '../data/6.5w_distinct1_labeled.csv'
    data = pandas.read_csv(filename)
    user_agent = n_gram_v2(filename, 2)
    # 所有特征放入DataFrame
    all_features = user_agent
    class_features = text_to_number(filename)
    class_features_names = ['event_action_account_input',
                            'event_action_pwd_input',
                            'event_action_login_click',
                            'java_enable',
                            'java_disable',
                            'cookie_enable',
                            'cookie_disenable']
    count = 0
    for name in class_features_names:
        all_features[name] = class_features[:, count]
        count += 1
    #mouse_features = data[['mouse_x', 'mouse_y']]
    mouse_features = data_normalization(filename)
    all_features['mouse_x'] = mouse_features[:, 0]
    all_features['mouse_y'] = mouse_features[:, 1]
    all_features.to_csv('../data/6.5w_distinct1_features.csv')

if __name__ == '__main__':
    #write_features_to_file()
    low_variance()
    #random_forest()
