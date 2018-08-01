# _*_ coding: utf-8 _*_

__author__ = 'xb'
__date__ = '2018.5.14 15:45'

import csv
import os
import time
import pandas
from vb_api import VB_API


def label(file):
    vb_api = VB_API()
    # 打标后的文件名file_labeled.csv
    dir_path = os.path.dirname(file)
    filename = os.path.basename(file)
    labeled_filename = os.path.join(dir_path, os.path.splitext(filename)[0] + '_labeled' + os.path.splitext(filename)[1])

    with open(filename) as f:
        f_csv = csv.reader(f)
        with open(labeled_filename, 'w') as f_labeled:
            f_labeled_csv = csv.writer(f_labeled)
            #header = next(f_csv)
            # 添加label列
            #header.append('label')
            #print(header)
            #f_labeled_csv.writerow(header)
            white_sum = 0
            black_sum = 0
            #access_limit = 1500
            for line in f_csv:
                # 1:白IP
                if not vb_api.is_web_login_brute(line[7]):
                    white_sum += 1
                    print('white_sum %d' %white_sum)
                    line.append('1')
                    f_labeled_csv.writerow(line)
                # 0:黑IP
                else:
                    black_sum += 1
                    print('black_sum %d' %black_sum)
                    line.append('0')
                    f_labeled_csv.writerow(line)
                #access_limit -= 1
                #print(access_limit)
                #if access_limit < 1:
                #    sleep(1800)
                #    access_limit = 1500

# 使用DataFrame处理csv
def label_v2(file):
    vb_api = VB_API()
    data = pandas.read_csv(file)
    label = []
    ip_dict = {}
    white_sum = 0
    black_sum = 0
    for ip in data.IP:
        if ip in ip_dict:
            label.append(ip_dict[ip])
        else:
            # 0:白IP
            if not vb_api.is_web_login_brute(ip):
                white_sum += 1
                print('white_sum %d' %white_sum)
                ip_dict[ip] = 0
                label.append(0)
            # 1:黑IP
            else:
                black_sum += 1
                print('black_sum %d' %black_sum)
                ip_dict[ip] = 1
                label.append(1)

    data['label'] = label
    dir_path = os.path.dirname(file)
    filename = os.path.basename(file)
    labeled_filename = os.path.join(dir_path, os.path.splitext(filename)[0] + '_labeled' + os.path.splitext(filename)[1])
    data.to_csv(labeled_filename)


if __name__ == '__main__':
    label_v2('../data/6.5w_distinct1.csv')
