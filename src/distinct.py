# __*__ coding: utf-8 __*__
__author__ = 'xb'
__date__ = '2018.5.21 9:57'


import csv
import pandas

def data_distinct(file):
    with open(file) as f:
        f_csv = csv.reader(f)
        user_agent = []
        IP = []
        event_action = []
        java_enable = []
        cookie_enable = []
        mouse_x = []
        mouse_y = []
        #label = []
        for line in f_csv:
            user_agent.append(line[14])
            IP.append(line[7])
            event_action.append(line[25])
            java_enable.append(line[28])
            cookie_enable.append(line[27])
            mouse_x.append(line[29])
            mouse_y.append(line[30])
            #label.append(line[36])
        data = pandas.DataFrame({'IP': IP,
                                 'user_agent': user_agent,
                                 'event_action': event_action,
                                 'cookie_enable': cookie_enable,
                                 'java_enable': java_enable,
                                 'mouse_x': mouse_x,
                                 'mouse_y': mouse_y})
    data = data.drop_duplicates()
    return data


if __name__ == '__main__':
    data = data_distinct('../data/6.5w.csv')
    #print(data.describe())
    data.to_csv('../data/6.5w_distinct1.csv')
