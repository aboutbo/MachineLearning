# __*__ coding: utf-8 __*__


import csv
import os

def separate_data(file):
    white_data = []
    black_data = []
    with open(file) as f:
        f_csv = csv.reader(f)
        for line in f_csv:
            # white IP
            if line[36] == '1':
                white_data.append(line)
            else:
                black_data.append(line)

    return white_data, black_data

if __name__ == '__main__':
    file = '../data/6.5w_labeled.csv'
    white_data, black_data = separate_data(file)
    # 写入新文件
    dir_path = os.path.dirname(file)
    filename = os.path.basename(file)
    white_filename = os.path.join(dir_path, os.path.splitext(filename)[0] + '_white' + os.path.splitext(filename)[1])
    with open(white_filename, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(white_data)
    black_filename = os.path.join(dir_path, os.path.splitext(filename)[0] + '_black' + os.path.splitext(filename)[1])
    with open(black_filename, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(black_data)
