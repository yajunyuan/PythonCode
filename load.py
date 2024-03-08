import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging
import pandas as pd
import pymysql


logging.basicConfig(format='%(asctime)s : %(name)s : %(message)s', level=logging.INFO)
min_max_scaler = preprocessing.MinMaxScaler()
class GruRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(hidden_size, 16)  # 全连接层
        self.linear2 = nn.Linear(16, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(s, b, -1)
        return x



def predict(datax, datay):
    INPUT_FEATURES_NUM = datax.shape[1]
    OUTPUT_FEATURES_NUM = datay[:,[-2]].shape[1]
    model = GruRNN(INPUT_FEATURES_NUM, 32, output_size=OUTPUT_FEATURES_NUM, num_layers=1)
    model.load_state_dict(torch.load('gru_model.pt'))
    model.eval()
    torch.no_grad()
    #data = [[2,3,4,5,6,7], [3,4,5,6,7,8],[1,2,3,4,5,6],[6,7,8,9,10,11]]

    # data = preprocessing.StandardScaler().fit_transform(np.array(data).reshape(1,-1))
    input = torch.from_numpy(datax.reshape(-1,1,INPUT_FEATURES_NUM)).to(torch.float32)
    outputs = model(input).tolist()
    # print(outputs)
    index = outputs.index(max(outputs))
    np.array(outputs).reshape(-1,OUTPUT_FEATURES_NUM)
    logging.info("out:")
    logging.info(outputs)
    logging.info(min_max_scaler.inverse_transform(datax[index].reshape(1,-1)))

def data_read(file):

    data = pd.read_csv(file, encoding='gbk')
    data.drop(columns=['Id','remark', 'spcTime'], inplace=True)
    data.dropna(axis=1, how='all',inplace = True)
    logging.info("原始数据的尺寸： ")
    logging.info(data.shape)
    data1 = data.loc[:, (data != data.iloc[0]).any()]
    logging.info("原始数据（去除不变的数据）的尺寸： ")
    logging.info(data1.shape)
    data_x = data1.iloc[:, :-3]
    data_y = data1.iloc[:, -3:]
    return  min_max_scaler.fit_transform(np.array(data_y)), min_max_scaler.fit_transform(np.array(data_x))

import random
class randomval:
    def __init__(self, rand_val, rand_num, influence_coeff, input):
        self.rand_val = rand_val
        self.rand_num = rand_num
        self.influence_coeff = influence_coeff
        self.input = input
        self.group_list = []

    def liststogroup(self, lists, jude=True):
        if jude: lists = [[[i] for i in lists[0]]] + lists[1:]
        if len(lists) > 2:
            for i in lists[0]:
                for j in lists[1]:
                    self.liststogroup([[i + [j]]] + lists[2:], False)
        elif len(lists) == 2:
            for i in lists[0]:
                for j in lists[1]:
                    self.group_list.append(i + [j])

    def create_random_list(self):
        random_list = []
        for i in range(len(self.input)):
            low = self.input[i] - self.rand_val * self.influence_coeff[i] / 2.0
            high = self.input[i] + self.rand_val * self.influence_coeff[i] / 2.0
            randomlisttmp = []
            if round(self.rand_num * self.influence_coeff[i]) == 0:
                randomlisttmp.append(self.input[i])
            else:
                for i in range(round(self.rand_num * self.influence_coeff[i])):
                    randomlisttmp.append(random.uniform(low, high))
            random_list.append(randomlisttmp)
        print(random_list)
        self.liststogroup(random_list)
        print(len(self.group_list), self.group_list)
        return self.group_list

class Train(object):
    def __init__(self, data_x, data_y):
        self.x_features = data_x
        self.y_labels = data_y[:,:]*100
        self.pearindex =[]

    def train_model(self):
        # x_train, x_test, y_train, y_test = train_test_split(self.x_features, self.y_labels, test_size=0.2, random_state=0, stratify = self.y_labels)
        importance = list()
        forest = RandomForestRegressor(n_estimators=300, random_state=1)
        # f, ax = plt.subplots(figsize=(48, 4), ncols=1, nrows=3)
        for i in range(self.y_labels.shape[1]):
            train_y = self.y_labels[:, [i]]
            train_x = self.x_features
            # train_x = self.x_features.take(self.pearindex[i],axis=1)
            # x_field = np.array(self.x_fields).take(self.pearindex[i])
            print(train_x.shape)
            # train_x = np.delete(self.x_features, i, axis=1)
            forest.fit(train_x, train_y)
            y_pred = forest.predict(train_x)
            logging.info("randomforest mean_squared_error: %.6f" % mean_squared_error(train_y, y_pred))
            importancetmp = forest.feature_importances_
            #importancetmpnew=np.around(np.insert(importancetmp, i, 1), decimals=2)
            importancetmpnew = np.around(importancetmp, decimals=2)

            # f, ax = plt.subplots(figsize=(48, 4))
            # sns.set(font_scale=0.8)
            # print(importancetmpnew.shape)
            # ax = sns.heatmap(np.array(importancetmpnew).reshape(1, -1), ax=ax, xticklabels=x_field,
            #                  yticklabels=self.x_fields[-3+i], linewidths=.1, cmap="OrRd", annot=True)
            # ax.set_title('sensitivity')
            # plt.rcParams['font.sans-serif'] = 'SimHei'
            # plt.rcParams['axes.unicode_minus'] = False
            # label_x = ax.get_xticklabels()
            # plt.setp(label_x, rotation=45, horizontalalignment='right')
            # string = "sensitivity"+str(i+2)+".svg"
            # plt.savefig(string, bbox_inches='tight')
            importance.append(importancetmpnew)
        print(np.array(importance))
        return np.array(importance)

if __name__ == '__main__':
    # sql_cmd = "SELECT * FROM table"
    # con = pymysql.connect(host=localhost, user='root', password='123456', database='', charset='utf8',
    #                       use_unicode=True)
    # df = pd.read_sql(sql_cmd, con)
    # logging.info(df)

    data_y, data_x = data_read("D:\dataananlysis\Train\Train\data_spc.csv")
    importarray = Train(data_x, data_y).train_model()
    grouplist = randomval(0.5, 20, importarray[1, :], data_x[-1,:]).create_random_list()
    predict(np.array(grouplist), np.array(data_y))