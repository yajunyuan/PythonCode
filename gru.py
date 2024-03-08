import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
import math
from scipy import stats
from sklearn import preprocessing
import logging
import csv

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
class GruTrain(object):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.overthreshold = 1.2
        self.overflag = False
        #self.datapre_x = DataPreprocess()
        #self.datapre_y = DataPreprocess()

    def data_pre_process(self):
        # self.datapre_x.pearson(self.data_x)
        print(self.data_x.shape, self.data_y.shape)
        # data_x = self.datapre_x.standard(self.data_x)
        # data_y = self.datapre_y.standard(self.data_y.reshape(-1,3))
        data_x = np.array(self.data_x)
        data_y = np.array(self.data_y)
        where_are_nan = np.isnan(data_x)
        where_are_inf = np.isinf(data_x)
        data_x[where_are_nan] = 0
        data_x[where_are_inf] = 0
        where_are_nan = np.isnan(data_y)
        where_are_inf = np.isinf(data_y)
        data_y[where_are_nan] = 0
        data_y[where_are_inf] = 0
        print(where_are_nan, where_are_inf)
        # 数据集分割
        data_len = len(data_y)
        t = np.linspace(0, data_len, data_len)

        train_data_ratio = 0.8  # Choose 80% of the data for training
        train_data_len = int(data_len * train_data_ratio)

        self.train_x = data_x[0:24100]
        self.train_y = data_y[0:24100]
        self.t_for_training = t[0:24100]

        self.test_x = data_x[24100:30100]
        self.test_y = data_y[24100:30100]
        self.t_for_testing = t[24100:30100]

        # ----------------- train -------------------
        INPUT_FEATURES_NUM = self.data_x.shape[1]
        OUTPUT_FEATURES_NUM = self.data_y.shape[1]
        train_x_tensor = self.train_x.reshape(-1, 100, INPUT_FEATURES_NUM)  # set batch size to 1
        train_y_tensor = self.train_y.reshape(-1, 100, OUTPUT_FEATURES_NUM)  # set batch size to 1

        # transfer data to pytorch tensor
        train_x_tensor = torch.from_numpy(train_x_tensor).to(torch.float32)
        train_y_tensor = torch.from_numpy(train_y_tensor).to(torch.float32)
        self.train_x_tensor = train_x_tensor.cuda()
        self.train_y_tensor = train_y_tensor.cuda()

        # prediction on test dataset
        test_x_tensor = self.test_x.reshape(-1, 1, INPUT_FEATURES_NUM)
        test_x_tensor = torch.from_numpy(test_x_tensor).to(torch.float32)  # 变为tensor
        self.test_x_tensor = test_x_tensor.cuda()

    def train_gru_model(self, hidden_size):
        OUTPUT_FEATURES_NUM = self.data_y.shape[1]
        gru_model = GruRNN(self.data_x.shape[1], hidden_size, output_size=OUTPUT_FEATURES_NUM, num_layers=1).to("cuda:0")  # 30 hidden units
        # print('GRU model:', gru_model)
        # gru_model = Net(1, output_size=OUTPUT_FEATURES_NUM).to("cuda:0")
        print('train x tensor dimension:', Variable(self.train_x_tensor).size())
        print('GRU model:', gru_model)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-2)

        prev_loss = 1
        max_epochs = 500


        for epoch in range(max_epochs):
            output = gru_model(self.train_x_tensor).to(device)
            loss = criterion(output, self.train_y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < prev_loss:
                torch.save(gru_model.state_dict(), 'gru_model.pt')  # save model parameters to files
                prev_loss = loss

            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
                print("The loss value is reached")
                break
            if (epoch + 1) % 10 == 0:
                print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            if (epoch + 1) % 20 == 0:
                pred_y_for_test = gru_model(self.test_x_tensor).to(device)
                pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
                # pred_y_for_test = self.datapre_y.re_standard(pred_y_for_test)
                # test_y = self.datapre_y.re_standard(self.test_y)

                test_loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(self.test_y))
                print("test loss：", test_loss.item())
                if test_loss.item() / loss.item() > self.overthreshold:
                    self.overflag = True
                    break
        # prediction on training dataset
        pred_y_for_train = gru_model(self.train_x_tensor).to(device)
        pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        # pred_y_for_train = self.datapre_y.re_standard(pred_y_for_train)
        # train_y =self.datapre_y.re_standard(self.train_y)



        pred_y_for_test = gru_model(self.test_x_tensor).to(device)
        pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        # pred_y_for_test = self.datapre_y.re_standard(pred_y_for_test)
        # test_y = self.datapre_y.re_standard(self.test_y)

        loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(self.test_y))
        print("test loss：", loss.item())

        # ----------------- plot -------------------
        plt.figure()
        plt.plot(self.t_for_training, self.train_y, 'b', label='y_trn')
        plt.plot(self.t_for_training, pred_y_for_train, 'y--', label='pre_trn')

        plt.plot(self.t_for_testing, self.test_y, 'k', label='y_tst')
        plt.plot(self.t_for_testing, pred_y_for_test, 'm--', label='pre_tst')

        plt.xlabel('t')
        plt.ylabel('Vce')
        plt.draw()
        plt.pause(5)
        plt.close()

    def check_overfit(self):
        for i in range(100, 10, -5):
            self.train_gru_model(i)
            if self.overflag:
                continue
            else:
                break

class DataRead:
    def __init__(self):
        pass
        #self.datapre = DataPreprocess()
    def data_read(self, file):
        data = pd.read_csv(file,nrows=1000)
        data.to_csv("traindata1.csv")
        print(data.shape)
        print(data.columns)
        #data = shuffle(data)
        data.drop(columns=['ID','fnlwgt'], inplace=True)
        # data.drop(columns=['Time'], inplace=True)
        data = pd.get_dummies(data)
        # data = self.data_smote(data)
        data_x = data.drop(columns=['capital_gain','capital_loss','income_bracket'], inplace=False)
        #data_x = data.iloc[:, :-2]
        data_y = data.loc[:, ['income_bracket']]
        #data_y = data.iloc[:, [-2,-1]]
        # data_y.loc[:, ['Amount']] = self.datapre.minmax(data_y.loc[:, ['Amount']])
        data_col = list(data_x.columns.values)
        print(data_x.shape)
        print(data_col)
        # data_y = DataPreprocess().standard(data_y)
        #self.datapre.pearson(data)
        #logging.info(data_y.head())
        return np.array(data_x), np.array(data_y), data_col
    def data_smote(self, data):
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        groupby_data_orginal = data.groupby('Class').count()
        logging.info("class distribution:\n")
        logging.info(groupby_data_orginal)
        model_smote = SMOTE()  # 建立smote模型对象
        x_smote_resampled, y_smote_resampled = model_smote.fit_resample(x, y)
        x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=list(x.columns.values))
        y_smote_resampled = pd.DataFrame(y_smote_resampled, columns=['Class'])
        smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled], axis=1)
        groupby_data_smote = smote_resampled.groupby('Class').count()
        logging.info("smote class distribution:\n")
        logging.info(groupby_data_smote)
        logging.info(smote_resampled.shape)
        return smote_resampled

if __name__ == '__main__':

    # checking if GPU is available
    # device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    data_x, data_y, data_col = DataRead().data_read("E:\dataset//train_date.csv")
    #data_x, data_y, data_col = DataRead().data_read("E:\dataset\income_census_train.csv")
    # data_x, data_y, data_col = DataRead().data_read("E:\dataset\creditcard.csv")
    gru = GruTrain(data_x, data_y)
    gru.data_pre_process()
    gru.check_overfit()