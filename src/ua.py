import csv
from sklearn.feature_extraction.text import CountVectorizer
import pandas

# 提取user-agent列
def load_file(file):
    ua = []
    with open(file) as f:
        f_csv = csv.reader(f)
        #header = next(f_csv)
        for line in f_csv:
            #print(line[14])
            #str = str + line[14]
            ua.append(line[14])
        return ua
# 适用2-gram提取词频，生成特征向量x
def n_gram(file):
    ua = load_file(file)
    #print(ua)
    vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                token_pattern = r'\b\w+\b', min_df=1)
    x = vectorizer.fit_transform(ua).toarray()
    #print(vectorizer.vocabulary_)
    #print(x)
    return x

# 使用n-gram提取词频，包括一些符号
def n_gram_v2(file):
    data = pandas.read_csv(file)
    ua = data.user_agent
    vectorizer = CountVectorizer(ngram_range = (1, 1), decode_error = "ignore",
                                 token_pattern = r'\b[a-zA-Z0-9\.\/]+\b', min_df=1)
    #x = vectorizer.fit_transform(ua).toarray()
    x = vectorizer.fit_transform(ua)
    vect_df = pandas.DataFrame(x.todense(), columns = vectorizer.get_feature_names())
    #print(vectorizer.get_feature_names())
    print(vect_df.head())
    return x

if __name__ == '__main__':
    n_gram_v2('../data/test_labeled.csv')
