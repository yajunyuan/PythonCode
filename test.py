
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
import mat73
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import RegressorChain
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import seaborn as sns
import time
from itertools import combinations
import os
import scipy.stats as stats
import io
from cryptography.fernet import Fernet
import random
from sklearn.preprocessing import LabelEncoder
import pickle as pkl


logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='## %Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import sys
sys.setrecursionlimit(10000000)
class orthogonal_table():
    def __init__(self, levels, factor, dic):
        self.levels = levels
        self.factor = factor
        self.dic = dic

    def independnum(self, num, numvaluelist, numflaglist):
        if num % self.levels != 0 and int(num / self.levels) != 0:
            numflaglist[0]=1
        if int(num / self.levels) != 0:
            numvaluelist[0] +=1
        else:
            return
        self.independnum(int(num / self.levels), numvaluelist, numflaglist)

    def getable(self, independvalue, num, matb, matgroup):
        nu = int(num / self.levels)
        for i in range(nu):
            for j in range(self.levels):
                for dp in range(independvalue):
                    for k in range(int(num/pow(self.levels,(independvalue - dp)))):
                        matb[int(j*num/pow(self.levels, dp+1)+i/pow(self.levels,dp)+k*num/pow(self.levels,dp)), dp] = j
        for i in range(independvalue, self.factor):
            for j in range(num):
                sum = 0
                for k in range(len(matgroup[i - independvalue])):
                    sum += matb[j, matgroup[i-independvalue][k]]
                matb[j,i]=sum%self.levels
        # logging.info("正交表为：")
        for j in range(num):
            for i in range(self.factor):
                matb[j,i] = list(self.dic[list(self.dic.keys())[i]])[int(matb[j,i])]
        # logging.info(matb)

    def combine_increase(self, start, result, count, NUM, independvalue, matgroup):
        if len(matgroup) >= self.factor - independvalue:
            return
        i = 0
        for i in range(start, independvalue +1 -count):
            result[count -1]=i
            if count-1 == 0:
                grouptmp = []
                for j in range(NUM-1, -1, -1):
                    grouptmp.append(result[j])
                matgroup.append(grouptmp)
            else:
                self.combine_increase(i+1, result, count-1,NUM,independvalue, matgroup)

    def fac(self, n):
        sum = 0
        if n==0 or n==1:
            sum=1
        if n>=2:
            sum = n*self.fac(n-1)
        return sum

    def groupvecadd(self, independvalue, matgroup):
        if self.levels>2 and len(matgroup)<self.factor- independvalue:
            tolnum =0
            for i in range(2, independvalue+1):
                tolnum += int(self.fac(independvalue) / self.fac(i) / self.fac(independvalue - i))
            grouppre = 0
            for i in range(self.levels -2 , 0 ,-1):
                for j in range(1, independvalue):
                    for k in range(independvalue + grouppre + j - 1, independvalue + grouppre + j - 1 + tolnum):
                        grouptmp = []
                        grouptmp.append(j)
                        grouptmp.append(k)
                        matgroup.append(grouptmp)
                        if len(matgroup) >=self.factor-independvalue:
                            return
                grouppre = len(matgroup) - 1

    def table(self):
        numvalue = [0]
        numflag = [0]
        self.independnum(((self.levels - 1) * self.factor + 1), numvalue, numflag)
        independvalue = numvalue[0] + numflag[0]
        num = pow(self.levels, independvalue)
        logging.info("total num: %d", num)
        a = np.array(np.zeros((num, self.factor)))
        group_vec = []
        for i in range(2, independvalue+1):
            result = [0]*i
            # logging.info(len(result))
            self.combine_increase(0, result, i, i, independvalue, group_vec)
            if len(group_vec)>= self.factor-independvalue:
                break
        self.groupvecadd(independvalue, group_vec)
        if len(group_vec) < self.factor - independvalue:
            logging.info("add group vector size error: %d", len(group_vec))
            return
        # logging.info(group_vec)

        self.getable(independvalue, num, a, group_vec)
        return a



class Net(nn.Module):
    #构造函数
    def __init__(self, input_size, output_size=1):
        super(Net, self).__init__()
        #卷积层三个参数：in_channel, out_channels, 5*5 kernal
        self.con1 = nn.Conv1d(input_size, 1, 5)
        self.con2 = nn.Conv1d(1, 1, 4)
        #全连接层两个参数：in_channels, out_channels
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(3, output_size)

    #前向传播
    def forward(self, input):
        #卷积 --> 激活函数（Relu) --> 池化
        x = self.con1(input)
        x = nn.functional.relu(x)
        # x = nn.functional.max_pool1d(x, 2)
        # print(x.shape)

        #重复上述过程
        x = self.con2(x)
        x = nn.functional.relu(x)
        # x = nn.functional.max_pool1d(x, 2)
        # print(x.shape)

        #展平
        x = x.view(-1, self.num_flat_features(x))

        #全连接层
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, -1)

        return x


    #展平
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features = num_features * i
        return num_features

# Define LSTM Neural Networks
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

class DataPreprocess:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()

        self.min_max_scaler = preprocessing.MinMaxScaler()

    def normal_distribution(self, data):
        for i in range(data.shape[1]):
            mean = data.iloc[:, i].mean()
            std = data.iloc[:, i].std()
            static, pvalue = stats.kstest(data[:, i], 'norm', (mean, std))
            if pvalue > 0.05:
                print("data columns %s is norm", data.columns.tolist()[i])
            else:
                print("data columns %s is norm", data.columns.tolist()[i])
    def pearson(self, data, outputdim, savename):
        res = data.corr()

        # mask = np.zeros_like(res)
        # mask[np.triu_indices_from(mask)] = True
        # f, ax = plt.subplots(figsize=(48, 4))
        # sns.set(font_scale=0.5)
        # ax = sns.heatmap(np.array(res.iloc[-3:, :-3]), vmin=-1, vmax=1, ax=ax, xticklabels=np.array(res.columns)[:-3], yticklabels=np.array(res.columns)[-3:],
        #                  linewidths=.1, cmap="RdBu_r", annot=True)
        #
        # ax.set_title('correlation')
        # # 汉字字体，优先使用楷体，找不到则使用黑体
        # plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei', 'Microsoft YaHei']
        #
        # # 正常显示负号
        # plt.rcParams['axes.unicode_minus'] = False
        # label_x = ax.get_xticklabels()
        # plt.setp(label_x, rotation=45, horizontalalignment='right')
        # plt.savefig(savename)
        # plt.close()

        restmp = res.iloc[outputdim:, :outputdim]
        logging.info(restmp)

        cols = [x for i, x in enumerate(restmp.columns) if np.fabs(restmp.iat[0, i]) > 0.2]
        logging.info(cols)
        # restmp = restmp[cols] #去除影响小的要素

        return np.array(restmp), list(res.columns), cols
        #print(res['age'].sort_values(ascending=False))

    def adjust(self, pear, expectedvalue, errorratio):
        # pear=pearin[:, pearin[0, :].argsort()[::-1]]
        # logging.info(pear)
        for selectnum in range(pear.shape[1]):
            for selectlist in combinations(range(1, pear.shape[1]+1), selectnum):
                pearsum=[]
                for i in range(pear.shape[0]):
                    pearsumtmp=0
                    for k in range(len(selectlist)):
                        pearsumtmp+=pear[i,selectlist[k]-1]
                    pearsum.append(pearsumtmp)
                if self.sumerror(pearsum, expectedvalue, errorratio):
                    logging.info(pearsum)
                    logging.info(selectlist)
                    return
        logging.info("select failed")

    def sumerror(self, pearsum, expectedvalue, errorratio):
        if pearsum[expectedvalue[0]-1] == 0:
            return False
        ratio=expectedvalue[1]/pearsum[expectedvalue[0]-1]
        for i in range(len(pearsum)):
            if i == expectedvalue[0]-1:
                continue
            if np.fabs(pearsum[i] * ratio)>errorratio:
                return False
        logging.info(ratio)
        return True




    def standard(self, data):
        return self.scaler.fit_transform(data)

    def re_standard(self, data):
        return self.scaler.inverse_transform(data)

    def minmax_fittf(self, data):
        return self.min_max_scaler.fit_transform(data)

    def minmax_tf(self, data):
        return self.min_max_scaler.transform(data)

    def re_minmax(self, data):
        return self.min_max_scaler.inverse_transform(data)

# read original data
class DataRead:
    def __init__(self):
        self.datapre_x = DataPreprocess()
        self.datapre_y = DataPreprocess()

    def wgn(self, sequence, snr):
        Ps = np.sum(abs(sequence) ** 2) / len(sequence)
        Pn = Ps / (10 ** ((snr / 10)))
        noise = np.random.randn(len(sequence)) * np.sqrt(Pn)
        signal_add_noise = sequence + noise
        return signal_add_noise

    def quartile(self, datatmp):
        # .iloc[:,[-6,-3,-2,-1]]
        print(np.any(datatmp.isnull()))
        Q1 = np.quantile(a=datatmp, q=0.25, axis=0)
        Q3 = np.quantile(a=datatmp, q=0.75, axis=0)
        # 计算 四分位差
        QR = Q3 - Q1
        # 下限 与 上线
        low_limit = Q1 - 100 * QR
        up_limit = Q3 + 100 * QR
        low_dataframe = pd.DataFrame(low_limit.reshape(1, -1), columns=datatmp.columns.tolist())
        up_dataframe = pd.DataFrame(up_limit.reshape(1, -1), columns=datatmp.columns.tolist())
        logging.info('下限为：')
        logging.info(low_dataframe)
        logging.info('上限为：')
        logging.info(up_dataframe)
        data1 = datatmp
        for index, row in datatmp.iteritems():
            if low_dataframe[index].values[0] == up_dataframe[index].values[0]:
                continue
            data1 = data1[
                (data1[index] >= low_dataframe[index].values[0]) & (data1[index] <= up_dataframe[index].values[0])]
        logging.info(data1.shape)
        data1.to_csv('E:\\dataset\\data\\clear\\oridata.csv', index=False)
        return data1

    def data_readtmp(self, file1, file2):
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M:%S')
        data1 = pd.read_csv(file1, parse_dates=['sync_time'], index_col='sync_time', date_parser=dateparse)
        data1.drop(columns=['id', 'status', 'create_by', 'create_time', 'update_by', 'update_time'], inplace=True)
        data2 = pd.read_csv(file2, parse_dates=['sync_time'], index_col='sync_time', date_parser=dateparse)
        data2=data2[['value']]
        logging.info(data1.shape)
        logging.info(data2.shape)
        data = data2.join(data1.reindex(data2.index, method='nearest'))
        # data = data2.set_index("sync_time").join(data1.set_index("sync_time"))
        logging.info(data.shape)
        datavalue = data['value'].apply(lambda x: list(map(float, x.split(',')))[0])
        logging.info(datavalue.shape)
        data.insert(data.shape[1], 'valuenew', datavalue)
        data.drop(columns=['value'], inplace=True)
        logging.info("原始数据的尺寸： ")
        logging.info(data.shape)
        logging.info(data.head())
        logging.info(data.columns.is_unique)
        data.dropna(axis=1, how='all',inplace = True)
        data.dropna(axis=0, how='all', inplace=True)
        datatmp = data.loc[:, (data != data.iloc[0]).any()]
        logging.info("原始数据（去除不变的数据）的尺寸： ")
        logging.info(datatmp.shape)

        # from sklearn.decomposition import PCA
        # pca = PCA(n_components=40)
        # pca.fit(datatmp)  # 训练
        # datatmp = pca.fit_transform(datatmp)
        # dataDf = pd.DataFrame(datatmp)
        # self.data_doe(datatmp)
        data_x = datatmp.iloc[:, :-1]
        data_y = datatmp.iloc[:, [-1]]
        logging.info("数据x的尺寸： ")
        logging.info(data_x.shape)
        pear, data_col = self.datapre_x.pearson(datatmp, data_y.shape[1], "correlation_ori_real.svg")
        logging.info("数据列： ")
        logging.info(data_col)

        diffvalue = np.ptp(np.array(data_x), axis=0)
        logging.info(len(diffvalue))

        # n, bins, patches = plt.hist(datatmp['valuenew'][0:10000], bins=20, color="g", histtype="bar")
        # for i in range(len(n)):
        #     plt.text(bins[i] + (bins[1] - bins[0]) / 2, n[i] * 1.01, int(n[i]), ha='center', va='bottom')
        # min_1 = datatmp['valuenew'][0:10000].min()
        # max_1 = datatmp['valuenew'][0:10000].max()
        # t1 = np.linspace(min_1, max_1, num=21)
        # plt.xticks(t1)
        # plt.show()

        # self.data_plt(np.array(datatmp))
        # logging.info(data1.iloc[-2].split(","))
        # self.data_plt(np.array(data1.iloc[-2].split(",")))
        return (np.array(data_x), (np.array(data_y)), data_col, pear, diffvalue)

    def data_readfile(self, filepath):
        if "xls" in filepath or "xlsx" in filepath:
            xdata = pd.read_excel(filepath, sheet_name=1)
            ydata = pd.read_excel(filepath, sheet_name=2)
        else:
            xdata = pd.read_csv(filepath, delim_whitespace=True)
        # allfile = os.listdir(filepath)
        # data = pd.DataFrame()
        # for file in allfile:
        #     datafiletmp = pd.read_csv(filepath + file)
        #     data = data.append(datafiletmp)

        # data = pd.read_csv(file)
        # data = pd.read_csv(file, encoding='gbk')
        # data = shuffle(data)
        # data.drop(columns=['Date','Time'], inplace=True)

        #data.drop(columns=['Id','remark', 'spcTime'], inplace=True)
        # xdata.dropna(axis=1, how='all',inplace = True)
        # xdata.dropna(axis=0, how='all', inplace=True)
        # xdata.fillna(0)
        ydata = ydata[ydata['工位'] == 3]
        logging.info("原始数据的尺寸： ")
        logging.info(xdata.shape)
        data_x = xdata.iloc[:, [1] + list(range(8, xdata.shape[1]))]
        data_col = list(data_x.columns.values)
        print(data_col)
        data_y = ydata.iloc[:, [2, 5]]
        #data_y = ydata.iloc[:, [2] + list(range(5, 17))]
        data_merge = pd.merge(data_x, data_y, left_on='生产批号', right_on='源批支号')
        data_merge.drop(columns=['源批支号'], inplace=True)
        # data_merge.to_csv('E:\\dataset\\data\\clear\\data.csv', index=False)

        #data_merge.drop(data_merge[(data_merge['针强度Ave'] > 1000.0) | (data_merge['针强度R'] > 1000.0)].index, inplace=True)
        data_merge.drop(columns=['生产批号'], inplace=True)
        data_xlen = data_x.shape[1] - 1
        pear, data_col, cols_index = self.datapre_x.pearson(data_merge, data_xlen, "correlation_ori.svg")
        logging.info("数据列： ")
        logging.info(data_col)

        #data_x = data_merge.iloc[:, list(range(0, data_xlen))]
        data_x = data_merge[cols_index]
        data_y = data_merge.iloc[:, list(range(data_xlen, data_merge.shape[1]))]
        data_y_field = list(data_y.columns.values)
        # data_x = data_x[cols_index]  #去除影响小的要素
        logging.info("数据x的尺寸： ")
        logging.info(data_x.shape)
        data_x.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        diffvalue = np.ptp(np.array(data_x), axis=0)
        logging.info(diffvalue)
        difflen = data_x.nunique()
        logging.info(difflen)
        data_x = self.datapre_x.minmax_fittf(np.array(data_x))
        data_y = self.datapre_y.minmax_fittf(np.array(data_y))
        datamed = np.median(data_x, axis=0)
        np.savez('dataparam.npz', diffvalue=diffvalue, difflen=difflen, pear=pear[0, :], datamed=datamed)
        # Save scaler
        with open("scalerx.pkl", "wb") as outfile:
            pkl.dump(self.datapre_x.min_max_scaler, outfile)
        with open("scalery.pkl", "wb") as outfile:
            pkl.dump(self.datapre_y.min_max_scaler, outfile)
        #self.data_plt(np.array(datatmp))
        # logging.info(data1.iloc[-2].split(","))
        # self.data_plt(np.array(data1.iloc[-2].split(",")))
        return (np.array(data_x), (np.array(data_y)), data_col, pear, diffvalue, difflen, self.datapre_x, self.datapre_y, data_y_field)

    def data_read(self, data, dataxdim):
        data_merge = data
        data_xlen = dataxdim
        pear, data_col, cols_index = self.datapre_x.pearson(data_merge, data_xlen, "correlation_ori.svg")
        logging.info("数据列： ")
        logging.info(data_col)

        data_x = data_merge.iloc[:, list(range(0, data_xlen))]
        # data_x = data_merge[cols_index]
        data_y = data_merge.iloc[:, list(range(data_xlen, data_merge.shape[1]))]
        data_y_field = list(data_y.columns.values)
        # data_x = data_x[cols_index]  #去除影响小的要素
        logging.info("数据x的尺寸： ")
        logging.info(data_x.shape)
        data_x.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
        diffvalue = np.ptp(np.array(data_x), axis=0)
        logging.info(diffvalue)
        difflen = data_x.nunique()
        logging.info(difflen)
        data_x = self.datapre_x.minmax_fittf(np.array(data_x))
        data_y = self.datapre_y.minmax_fittf(np.array(data_y))
        datamed = np.median(data_x, axis=0)
        np.savez('dataparam.npz', diffvalue=diffvalue, difflen=difflen, pear=pear[0, :], datamed=datamed)
        # Save scaler
        with open("scalerx.pkl", "wb") as outfile:
            pkl.dump(self.datapre_x.min_max_scaler, outfile)
        with open("scalery.pkl", "wb") as outfile:
            pkl.dump(self.datapre_y.min_max_scaler, outfile)
        # self.data_plt(np.array(datatmp))
        # logging.info(data1.iloc[-2].split(","))
        # self.data_plt(np.array(data1.iloc[-2].split(",")))
        return (
        np.array(data_x), (np.array(data_y)), data_col, pear, diffvalue, difflen, self.datapre_x, self.datapre_y,
        data_y_field)

    def data_doe(self, datatmp):
        # 正交表
        dic = {'AP': [1040, 1050], 'AH': [110, 120], 'AFDP': [8, 9]}
        table = orthogonal_table(2, 3, dic).table()
        logging.info(table)
        for j in range(10000, 12500):
            temp_df = datatmp.iloc[j]
            for i in range(len(table)):
                datatmp.loc[datatmp.shape[0]] = temp_df
                datatmp.loc[datatmp.index[-1], list(dic.keys())] = table[i]
                datatmp.iloc[-1, -3:] = self.wgn(datatmp.iloc[-1, -3:], 85)
        logging.info(datatmp.shape)
        df1 = datatmp.iloc[:(datatmp.shape[0]-5000), :]
        df2 = datatmp.iloc[(datatmp.shape[0]-5000):, :]
        df_new = pd.concat([df2, df1], ignore_index=True)
        return df_new

    def data_plt(self, data):
        logging.info(data.shape)
        # data=list(map(float, data))
        plt.subplot(3, 3, 1)
        plt.scatter(range(len(data[:, [-1]])), data[:, [-1]], s=0.1,c="r", marker='.', alpha=1, label="size1")

        plt.subplot(3, 3, 2)
        plt.scatter(data[:, [-3]], data[:, [-1]], s=0.1,c="g", marker='.', alpha=1, label="size2")

        plt.subplot(3, 3, 3)
        plt.scatter(data[:, [-5]], data[:, [-1]], s=0.1,c="b", marker='.', alpha=1, label="size3")



        #plt.subplot(3, 3, 4)
        # plt.scatter(range(len(data[[-2], :])), data[[-2], :].reshape(-1,1), s=0.1,c="r", marker='.', alpha=1, label="size2")

        # plt.subplot(3, 3, 5)
        # plt.scatter(range(len(data[[-2], :])), data[[-2], :].reshape(-1,1), s=0.1,c="g", marker='.', alpha=1, label="size2")
        #
        # plt.subplot(3, 3, 6)
        # plt.scatter(range(len(data[[-3], :])), data[[-3], :].reshape(-1,1), s=0.1,c="b", marker='.', alpha=1, label="size3")
        #
        # plt.subplot(3, 3, 7)
        # plt.scatter(data[:, [-6]], data[:, [-3]], s=0.1,c="r", marker='.', alpha=1, label="size1")
        #
        # plt.subplot(3, 3, 8)
        # plt.scatter(data[:, [-6]], data[:, [-2]], s=0.1,c="g", marker='.', alpha=1, label="size2")
        #
        # plt.subplot(3, 3, 9)
        # plt.scatter(data[:, [-6]], data[:, [-1]], s=0.1,c="b", marker='.', alpha=1, label="size3")
        plt.show()
        #-19

        # ax = plt.axes(projection='3d')
        # ax.scatter3D(data[:, [-3]], data[:, [-2]], data[:, [-1]], c=data[:, [-1]], cmap='Greens')
        # plt.show()

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

# Plot training deviance
def plot_training_deviance(clf, n_estimators, X_test, y_test):
    # compute test set deviance
    test_score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    train_score = clf.train_score_
    logging.info("len(train_score): %s" % len(train_score))
    logging.info(train_score)
    logging.info("len(test_score): %s" % len(test_score))
    logging.info(test_score)
    plt.plot(np.arange(n_estimators) + 1, train_score, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r*', label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()


# Plot feature importance
def plot_feature_importance(clf, feature_names):
    feature_importance = clf.feature_importances_
    print(feature_importance)
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    # plt.yticks(pos, feature_names[sorted_idx])
    plt.yticks(pos, [feature_names[idx] for idx in sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

class Train(object):
    def __init__(self, data_x, data_y, data_col, pear):
        self.x_features = data_x
        self.y_labels = data_y
        self.x_fields = data_col
        self.pear = pear
        self.pearindex =[]

    def preprocess(self):
        pear = np.argwhere(abs(self.pear) > 0.1)
        for i in range(self.y_labels.shape[1]):
            pear_index=[]
            for j in range(pear.shape[0]):
                if pear[j,0] == i:
                    pear_index.append(pear[j,1])
            self.pearindex.append(pear_index)
        logging.info(self.pearindex)


    def train_model(self):
        self.preprocess()
        # x_train, x_test, y_train, y_test = train_test_split(self.x_features, self.y_labels, test_size=0.2, random_state=0, stratify = self.y_labels)
        parameters = {'n_estimators': range(1, 200, 10), 'learning_rate': [0.00001, 0.0001, 0.001, 0.01, 0.1]}
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.001, max_depth=15, random_state=0, loss='ls')
        # regress = RegressorChain(model)
        # clf = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=3)
        # model.fit(x_train, y_train)
        # cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        # with open('gbdt_cv_result.csv', 'w') as f:
        #     cv_result.to_csv(f)
        # y_pred = model.predict(x_test)
        # logging.info("GBDT mean_squared_error: %.6f" % mean_squared_error(y_test, y_pred))
        # importances = model.feature_importances_
        # indices = np.argsort(importances)[::-1]
        # threshold = 0.02
        # x_selected = x_train[:, importances > threshold]
        # for f in range(x_selected.shape[1]):
        #     logging.info("%2d) %-*s %f" % (f + 1, 30, self.x_fields[indices[f]], importances[indices[f]]))
        # logging.info(clf.best_params_)
        # logging.info(clf.best_estimator_)
        # logging.info(clf.best_score_)

        # plot_training_deviance(clf=model, n_estimators=model.get_params()["n_estimators"], X_test=self.x_features,
        #                         y_test=self.y_labels)



        # plot_feature_importance(clf=model, feature_names=self.x_fields)
        parameters = {'n_estimators': range(300, 400, 10)}
        forest = RandomForestRegressor(n_estimators=300, random_state=1)
        clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=3)
        importance = list()
        # f, ax = plt.subplots(figsize=(48, 4), ncols=1, nrows=3)
        for i in range(self.y_labels.shape[1]):
            train_y = self.y_labels[:, [i]]
            train_x = self.x_features
            # train_x = self.x_features.take(self.pearindex[i],axis=1)
            # x_field = np.array(self.x_fields).take(self.pearindex[i])
            logging.info(train_x.shape)
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
        #return np.array(importance)

        # y_pred = forest.predict(x_test)
        # logging.info("randomforest mean_squared_error: %.6f" % mean_squared_error(y_test, y_pred))

        f, ax = plt.subplots(figsize=(48, 4))
        sns.set(font_scale=0.8)
        ax = sns.heatmap(np.array(importance), ax=ax, xticklabels=self.x_fields[:-1], yticklabels=self.x_fields[-1], linewidths=.1, cmap="OrRd", annot=True)
        ax.set_title('sensitivity')
        plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        label_x = ax.get_xticklabels()
        plt.setp(label_x, rotation=45, horizontalalignment='right')
        plt.savefig("sensitivityreal.svg", bbox_inches='tight')
        return [i for i, x in enumerate(importance[0]) if x >= 0.01]

        # bar_width = 0.2  # 条形宽度
        # index_male = np.arange(len(self.x_fields))
        # index_female = index_male + bar_width
        #
        # # 使用两次 bar 函数画出两组条形图
        # plt.bar(index_male, height=importances, width=bar_width, color='r', label='有功功率')
        # plt.bar(index_female, height=importances1, width=bar_width, color='b', label='无功功率')
        #
        # plt.legend()  # 显示图例
        # plt.xticks(index_male + bar_width / 2, self.x_fields, rotation ='vertical')  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
        # plt.ylabel('灵敏度')  # 纵坐标轴标题
        # plt.title('sensitivity')  # 图形标题
        # plt.show()


        # indices = np.argsort(importances)[::-1]
        # threshold = 0.02
        # x_selected = x_train[:, importances > threshold]
        # for f in range(x_selected.shape[1]):
        #     logging.info("%2d) %-*s %f" % (f + 1, 30, self.x_fields[indices[f]], importances[indices[f]]))
        # logging.info(clf.best_params_)
        # logging.info(clf.best_estimator_)
        # logging.info(clf.best_score_)

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class GruTrain(object):
    def __init__(self, data_x, data_y, batch_size, dataprex, dataprey):
        self.data_x = data_x
        self.data_y = data_y  # [:, 2:]
        self.time_step = 1
        self.batch_size = batch_size
        self.datapre_x = dataprex
        self.datapre_y = dataprey

    def data_pre_process(self):

        logging.info("input size: {}  output size: {}".format(self.data_x.shape, self.data_y.shape))
        # data_x = self.datapre_x.minmax(self.data_x)
        # data_x = self.data_x
        # data_y = self.data_y
        # data_y = self.datapre_y.minmax(self.data_y)

        # where_are_nan = np.isnan(self.data_x)
        # where_are_inf = np.isinf(self.data_x)
        # self.data_x[where_are_nan] = 0
        # self.data_x[where_are_inf] = 0
        # where_are_nan = np.isnan(self.data_y)
        # where_are_inf = np.isinf(self.data_y)
        # self.data_y[where_are_nan] = 0
        # self.data_y[where_are_inf] = 0
        # 数据集分割
        data_len = len(self.data_y)
        t = np.linspace(0, data_len, data_len)

        train_data_ratio = 0.8  # Choose 80% of the data for training
        train_data_len = int(data_len * train_data_ratio)
        test_data_len = data_len - train_data_len
        if train_data_len % self.time_step != 0:
            train_data_len = int(train_data_len/self.time_step)*self.time_step
        if test_data_len % self.time_step != 0:
            test_data_len = int(test_data_len / self.time_step) * self.time_step
        #test_data_len=1000
        logging.info("train data len : %f",train_data_len)
        logging.info("test data len : %f", test_data_len)
        self.train_x = self.data_x[0:train_data_len]
        self.train_y = self.data_y[0:train_data_len]
        self.t_for_training = t[0:train_data_len]

        self.test_x = self.data_x[train_data_len:train_data_len+test_data_len]
        self.test_y = self.data_y[train_data_len:train_data_len+test_data_len]
        self.t_for_testing = t[train_data_len:train_data_len+test_data_len]

    def train_gru_model(self, data_y_field, device=torch.device("cpu")):
        self.data_pre_process()
        # ----------------- train -------------------
        INPUT_FEATURES_NUM = self.train_x.shape[1]
        OUTPUT_FEATURES_NUM = self.train_y.shape[1]

        train_x_tensor = self.train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  # set batch size to 1
        train_y_tensor = self.train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)  # set batch size to 1

        # transfer data to pytorch tensor
        train_x_tensor = torch.from_numpy(train_x_tensor).to(torch.float32)
        train_y_tensor = torch.from_numpy(train_y_tensor).to(torch.float32)
        train_x_tensor = train_x_tensor.cuda()
        train_y_tensor = train_y_tensor.cuda()

        # # prediction on test dataset
        # test_x_tensor = self.test_x.reshape(-1, 1, INPUT_FEATURES_NUM)
        # test_y_tensor = self.test_y.reshape(-1, 1, OUTPUT_FEATURES_NUM)
        # test_x_tensor = torch.from_numpy(test_x_tensor).to(torch.float32)  # 变为tensor
        # test_y_tensor = torch.from_numpy(test_y_tensor).to(torch.float32)
        # test_x_tensor = test_x_tensor.to(device)
        # test_y_tensor = test_y_tensor.to(device)

        gru_model = GruRNN(INPUT_FEATURES_NUM, 128, output_size=OUTPUT_FEATURES_NUM, num_layers=1).to(device)  # 30 hidden units
        # print('GRU model:', gru_model)
        # gru_model = Net(1, output_size=OUTPUT_FEATURES_NUM).to("cuda:0")
        logging.info('train x tensor dimension: {}'.format(Variable(train_x_tensor).size()))
        logging.info('GRU model:{}'.format(gru_model))

        # awl = AutomaticWeightedLoss(2)
        # # learnable parameters
        # optimizer = torch.optim.Adam([
        #     {'params': gru_model.parameters()},
        #     {'params': awl.parameters(), 'weight_decay': 0.01}
        # ], lr=1e-1)

        criterion = nn.SmoothL1Loss()
        #criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-4)

        prev_loss = 50
        max_epochs = 300
        test_loss = 999
        train_x_tensor = train_x_tensor.to(device)

        test_x_tensor = self.test_x.reshape(-1, 1, INPUT_FEATURES_NUM)
        test_x_tensor = torch.from_numpy(test_x_tensor).to(torch.float32)  # 变为tensor
        test_x_tensor = test_x_tensor.to(device)

        epochlist=[]
        trainloss = []
        valloss = []
        plt.ion()  # 开启interactive mode 成功的关键函数
        plt.figure(figsize=(20, 20), dpi=200)
        plt.tight_layout()
        for epoch in range(max_epochs):
            output = gru_model(train_x_tensor).to(device)
            # loss0 = criterion(output[:, :, 0], train_y_tensor[:, :, 0])
            # loss1 = criterion(output[:, :, 1], train_y_tensor[:, :, 1])
            loss = criterion(output, train_y_tensor)
            # loss = awl(loss0,loss1)

            pred_y_for_test = gru_model(test_x_tensor).to(device)
            pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
            # pred_y_for_test = self.datapre_y.re_standard(pred_y_for_test)
            # test_y = self.datapre_y.re_standard(self.test_y)
            test_loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(self.test_y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epochlist.append(epoch)
            trainloss.append(loss.item())
            valloss.append(test_loss)
            plt.plot(epochlist, trainloss, 'b', lw=1)  # lw为曲线宽度
            plt.plot(epochlist, valloss, 'r', lw=1)
            plt.title("loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(["train_loss", "val_loss"])
            plt.pause(0.05)
            plt.ioff()
            if test_loss < prev_loss:
                torch.save(gru_model.state_dict(), 'gru_model-3.pt')  # save model parameters to files
                prev_loss = test_loss

            if loss.item() < 1e-4:
                logging.info('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
                logging.info("The loss value is reached")
                break
            if (epoch + 1) % 10 == 0:
                logging.info('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            if (epoch + 1) % 20 == 0:
                logging.info("test loss：%f", test_loss.item())

        plt.savefig("D:\dataananlysis\loss.svg")
        # prediction on training dataset
        pred_y_for_train = gru_model(train_x_tensor).to(device)
        pred_y_for_train = pred_y_for_train.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        pred_y_for_train = self.datapre_y.re_minmax(pred_y_for_train)
        self.train_y =self.datapre_y.re_minmax(self.train_y)

        # prediction on test dataset
        test_x_tensor = self.test_x.reshape(-1, 1, INPUT_FEATURES_NUM)
        test_x_tensor = torch.from_numpy(test_x_tensor).to(torch.float32)  # 变为tensor
        test_x_tensor = test_x_tensor.to(device)

        pred_y_for_test = gru_model(test_x_tensor).to(device)
        pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        pred_y_for_test = self.datapre_y.re_minmax(pred_y_for_test)
        self.test_y = self.datapre_y.re_minmax(self.test_y)
        pred_data_df = pd.DataFrame(np.concatenate((self.test_x, pred_y_for_test), axis=1))
        self.datapre_y.pearson(pred_data_df, OUTPUT_FEATURES_NUM, "correlation_pred.svg")

        loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(self.test_y))
        logging.info("test loss： %f", loss.item())
        logging.info("prev_loss loss： %f", prev_loss)

        # data_c=np.concatenate((self.train_x[:, [0]],np.array(list(np.delete(self.train_x[[0],:],0,axis=1))* len(self.train_x)).reshape(len(self.train_x), -1)),axis=1)
        # data_c_tensor = torch.from_numpy(data_c.reshape(-1, self.batch_size, INPUT_FEATURES_NUM)).to(torch.float32).cuda()
        # logging.info("simulated data size:{}".format(data_c.shape))
        # simu_y_for_test = gru_model(data_c_tensor).to(device)
        # simu_y_for_test = simu_y_for_test.view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        # logging.info("simu y : {}".format(simu_y_for_test.shape))
        # print(self.train_x[:, [0]].reshape(1,-1))
        # print(simu_y_for_test.reshape(1,-1))
        # logging.info("simu pearsonr {}".format(stats.pearsonr(self.train_x[:, [0]].reshape(1,-1).flatten(), simu_y_for_test.reshape(1,-1).flatten())[0]))

        # ----------------- plot -------------------
        for i in range(OUTPUT_FEATURES_NUM):
            plt.figure(figsize=(20, 10), dpi=200)
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # 用来正常显示中文标签SimHei
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            p1=plt.scatter(self.t_for_training, self.train_y[:,i], s=5,color='b', alpha=1, label='y_trn')
            p2=plt.scatter(self.t_for_training, pred_y_for_train[:,i], s=5,color='y',alpha=0.5, label='pre_trn')

            p3=plt.scatter(self.t_for_testing, self.test_y[:,i], s=5,color='g', alpha=1, label='y_tst')
            p4=plt.scatter(self.t_for_testing, pred_y_for_test[:,i],s=5, color='m', alpha=0.5, label='pre_tst')
            plt.legend([p1, p2, p3, p4], ['Training real value', 'Training predictive value', 'Test real value', 'Test predictive value'], loc=1)
            plt.xlabel(data_y_field[i]+"-data number")
            plt.ylabel(data_y_field[i])
            plt.savefig("D:/dataananlysis/"+data_y_field[i]+".svg")

import random
class randomval:
    def __init__(self, rand_val, difflen, rand_num, influence_coeff, datapre):
        self.rand_val = rand_val
        self.difflen = difflen
        self.rand_num = rand_num
        self.influence_coeff = influence_coeff
        self.group_list = []
        # self.datapre_x = DataPreprocess()
        self.datapre_x =datapre

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

    def create_random_list(self, datainput):
        random_list = []
        for i in range(len(datainput)):
            low = datainput[i] - self.rand_val[i] * self.influence_coeff[i] / 2.0
            high = datainput[i] + self.rand_val[i] * self.influence_coeff[i] / 2.0
            randomlisttmp = []
            if round(abs(self.rand_num * self.influence_coeff[i])) == 0:
                randomlisttmp.append(datainput[i])
            else:
                for i in range(round(abs(self.rand_num * self.influence_coeff[i]))):
                    randomlisttmp.append(random.uniform(low, high))
            random_list.append(randomlisttmp)
        self.liststogroup(random_list)
        return self.group_list

    def create_total_list(self, datainput):
        total_list = []
        for i in range(len(datainput)):
            totallisttmp = []
            for j in range(-5,5):
                totallisttmp.append(datainput[i]+j*(self.rand_val[i]/10))
            total_list.append(totallisttmp)
        self.liststogroup(total_list)
        return self.group_list

    """
    data orthogonal table extension
    if mode is doe, it is orthogonal table;
    else total extension.
    index is uesd in the total extension.
    """
    def doe_expand(self, datainput, mode='doe', index=0):
        # valuelen = min(np.min(abs(np.multiply(self.influence_coeff, self.difflen))),100)
        # if valuelen%2 != 0:
        #     valuelen +=1
        valuelen =4
        dic = dict()
        for i in range(len(datainput)):
            dic[i] = []
            for j in range(-int(valuelen/2), int(valuelen/2)):
                dic[i].append(datainput[i] + j / self.difflen[i])
                # dic[i].append(datainput[i] + j * self.rand_val[i] / self.difflen[i])
        if mode == 'doe':
            table = orthogonal_table(int(valuelen), len(datainput), dic).table()
        else:
            self.group_list=[]
            self.liststogroup(list(np.array(list(dic.values()))[:,index*10:(index+1)*10]))
            table = np.array(self.group_list)
        return table

    def predict(self, batch_size, datainput, expectedoutput, errorratio, mode='doe'):
        logging.info(datainput)
        # if mode != 'doe':
        #     self.create_total_list(datainput)
        #     data_x = np.array(self.group_list)
        # else:
        #     data_x = self.doe_expand(datainput)
        data_x = self.doe_expand(datainput, mode)
        datax = data_x
        # datax=self.datapre_x.minmax_tf(data_x)
        # datax1 = self.datapre_x.re_minmax(data_x)

        global count
        logging.info(len(datax))
        INPUT_FEATURES_NUM = datax.shape[1]
        OUTPUT_FEATURES_NUM = len(expectedoutput)
        model = GruRNN(INPUT_FEATURES_NUM, 128, output_size=OUTPUT_FEATURES_NUM, num_layers=1).to("cuda:0")
        # model.load_state_dict(model_decryption('encryt_file1', 'license1'))
        model.load_state_dict(torch.load('gru_model-3.pt'))
        model.eval()
        torch.no_grad()

        # data1 = [
        #     [6.8211, 1030.755, 72.0425, 5.23496, 14.649, 1043.55, 529.935, 9.5604],
        #     [27.7, 920.8, 112.8, 6.9, 19.2, 468.9, 474.3, 5.4],
        #     [28.49, 1017.925, 72.0425, 5.23496, 14.649, 1043.55, 529.935, 10.62184]]
        # input = torch.from_numpy(np.array(data1).reshape(-1, 1, INPUT_FEATURES_NUM)).to(torch.float32).to(
        #     device)
        # outputs = model(input).to(device).view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        # logging.info(outputs)

        # data = [[2,3,4,5,6,7], [3,4,5,6,7,8],[1,2,3,4,5,6],[6,7,8,9,10,11]]

        if mode != 'doe':
            dataxlen = len(datax)

            for i in range(500):
                datax_input = datax[i * int(dataxlen / 500):(i + 1) * int(dataxlen / 500)]
                input = torch.from_numpy(datax_input.reshape(1, -1, INPUT_FEATURES_NUM)).to(
                    torch.float32).to(
                    device)
                outputs = model(input).to(device).view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
                # logging.info(datax_input[10000:10003])
                # logging.info(outputs[10000:10003])
                flag, losslist = self.sumerror(outputs, expectedoutput, errorratio)
                logging.info(losslist)
                if flag == 1:
                    logging.info(errorratio)
                    logging.info(datax[losslist[0]].reshape(1, -1))
                    errorratio -= 0.01
        else:
            # data = preprocessing.StandardScaler().fit_transform(np.array(data).reshape(1,-1))
            datax = datax[:int(len(datax) / batch_size) * batch_size, :]
            input = torch.from_numpy(datax.reshape(-1, 1, INPUT_FEATURES_NUM)).to(torch.float32).to(device)
            outputs = model(input).to(device).view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
            # logging.info(outputs[:2])
            # logging.info(outputs[94:96])
            # logging.info(outputs[4418:4420])
            # logging.info(outputs[4512:4514])
            # print(outputs)
            flag, losslist = self.sumerror(outputs, expectedoutput, errorratio)
            # flag, losslist = self.sumerrorSimulatedAnnealing(outputs, expectedoutput, errorratio)
            logging.info(losslist)
            if flag == 0:
                logging.info("select failed")
                logging.info(losslist[0])
                if count > 100:
                    logging.info("Number of iterations over max")
                    logging.info(datax[losslist[0]].reshape(1, -1))
                    return datax[losslist[0]].reshape(1, -1)
                if losslist[0] == len(datax)/2:
                    losslist[0] = random.randint(0,len(datax))
                count = count + 1
                logging.info("Number of iterations is {}".format(count))
                return self.predict(batch_size, data_x[losslist[0]], expectedoutput, errorratio, mode)
            # index = outputs.index(max(outputs))
            # np.array(outputs).reshape(-1, OUTPUT_FEATURES_NUM)
            # logging.info("out:")
            # logging.info(outputs)
            else:
                logging.info(datax[losslist[0]].reshape(1, -1))
                return datax[losslist[0]].reshape(1, -1)

    """ 
    Comparison of model output and expected output;
    losslist is next index ,totalloss and lossratio
    """
    def sumerror(self, outputs, expectedoutput, errorratio):
        losslist = [0,999]
        for i in range(len(outputs[0])):
            losslist.append(100)
        for i in range(len(outputs)):
            flag=1
            lossratio=[0]*len(outputs[0])
            losstmp = 0
            for j in range(len(outputs[0])):
                losstmp += np.fabs(outputs[i][j] - expectedoutput[j])
                lossratio[j] = np.fabs(outputs[i][j] - expectedoutput[j]) / expectedoutput[j]
                if lossratio[j] > errorratio:
                    flag=0
            # lossratioflag=[0,0,0]
            # for j in range(len(lossratio)):
            #     if lossratio[j] < losslist[j+1] or lossratio[j] < errorratio:
            #         lossratioflag[j]=1
            # if lossratioflag == [1,1,1]:
            #     losslist[0] = i
            if losslist[1] >= losstmp:
                losslist[0] = i
                losslist[1] = losstmp
                for j in range(len(lossratio)):
                    losslist[j+2] = lossratio[j]
            # else:
            #     expvalue = math.exp((losslist[1] - losstmp) * i / 1000 )
            #     randomvalue = random.uniform(0, 1)
            #
            #     if expvalue > randomvalue :
            #         logging.info("expvlaue is {}; random value is {}".format(expvalue, randomvalue))
            #         losslist[0] = i
            #         losslist[1] = losstmp
            #         for j in range(len(lossratio)):
            #             losslist[j + 2] = lossratio[j]
            if flag == 1:
                logging.info(outputs[i])
                return flag, losslist
        return flag, losslist

def write_license(license_file):
    key = Fernet.generate_key()
    with open(license_file, 'wb') as fw:
        fw.write(key)

def read_license(license_file):
    with open(license_file, 'rb') as fr:
        key = fr.read()
    return key


def model_encryption(pth_file, encryp_file, license):
    write_license(license)
    model = torch.load(pth_file)
    b = io.BytesIO()
    torch.save(model, b)
    b.seek(0)
    pth_bytes = b.read()
    key = read_license(license)
    encrypted_data = Fernet(key).encrypt(pth_bytes)
    with open(encryp_file, 'wb') as fw:
        fw.write(encrypted_data)


def model_decryption(encryt_file, license):
    with open(encryt_file, 'rb') as fr:
        encrypted_data = fr.read()
    key = read_license(license)
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    model = torch.load(b)
    return model


def gru_testmodel(data_x, OUTPUT_FEATURES_NUM):
    INPUT_FEATURES_NUM = data_x.shape[1]
    with open("scalerx.pkl", "rb") as infile:
        logging.info(data_x)
        scalerx = pkl.load(infile)
        data_x = scalerx.fit_transform(data_x)

        # OUTPUT_FEATURES_NUM = val_y.shape[1]

        model = GruRNN(INPUT_FEATURES_NUM, 128, output_size=OUTPUT_FEATURES_NUM, num_layers=1).to("cuda:0")
        model.load_state_dict(torch.load('gru_model.pt'))
        model.eval()
        torch.no_grad()
        input = torch.from_numpy(data_x.reshape(-1, 1, INPUT_FEATURES_NUM)).to(torch.float32).to(device)
        outputs = model(input).to(device).view(-1, OUTPUT_FEATURES_NUM).data.cpu().numpy()
        with open("scalery.pkl", "rb") as infile:
            scalery = pkl.load(infile)
            outputs1 = scalery.inverse_transform(outputs)
            # print(self.datapre_y.re_standard(val_y))
            logging.info(outputs1)

def auto_testmode(data_y):
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    batch_size = 1
    errorratio = 0.1
    count = 1
    dataparam = np.load('dataparam.npz')
    diffvalue = dataparam['diffvalue']
    difflen = dataparam['difflen']
    pear = dataparam['pear']
    datamed = dataparam['datamed']
    with open("scalerx.pkl", "rb") as infile:
        scalerx = pkl.load(infile)
        with open("scalery.pkl", "rb") as infile:
            logging.info(data_y)
            scalery = pkl.load(infile)
            data_y = scalery.transform(data_y.reshape(-1, 1))
            dataout_pre = randomval(diffvalue, difflen, 30, pear, scalerx).predict(batch_size, datamed,
                                                                                   data_y,
                                                                                   errorratio, 'doe')
            dataout_x = scalerx.inverse_transform(dataout_pre)

            datatmp = np.around(dataout_x, decimals=4)
            datatmp = datatmp.reshape(-1)
            logging.info(datatmp)
            # datatmp = [round(sub_lst, 4) for sub_lst in dataout_x]
            dataintindex = [8, 9, 11,12,13,14,15,16,17,18]
            for index in dataintindex:
                datatmp[index] = int(math.ceil(datatmp[index]))
            datatmp2 = [datat for datat in datatmp]
            logging.info(datatmp2)
            return dataout_x


def train_mode(data):
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
    batch_size = 1
    # data = pd.read_csv("boston.csv")
    data_x, data_y, data_col, pear, diffvalue, difflen, dataprex, dataprey, data_y_field = DataRead().data_read(data)
    gru = GruTrain(data_x, data_y, batch_size, dataprex, dataprey)
    gru.train_gru_model(data_y_field, device)


if __name__ == '__main__':
    from read_config import getConfig

    # xdata = pd.read_csv("E:\dataset\data\clear\data.csv")
    # xdata.drop(columns=['生产批号'], inplace=True)
    # # xdata.replace([np.inf, -np.inf], np.nan, inplace=True)
    # xdata.fillna(method='ffill', inplace=True)
    # xdata.replace([np.nan], 0, inplace=True)
    # DataRead().quartile(xdata)


    # checking if GPU is available
    # device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # data = pd.read_excel("E:\dataset\data\萃取数据.xlsx")
    # data = data.iloc[:-3,:]
    # a=data.dtypes
    # data=data.astype('float')
    # train_mode(data, data.shape[1]-1)
    datay = np.array([6.5])
    auto_testmode(datay)
    # confname = "conf.config"
    # model = getConfig(confname, "mode", "modevalue")
    # batch_size = int(getConfig(confname, "train", "batchsize"))
    # modelname = getConfig(confname, "predict", "modelname")
    # errorratio = float(getConfig(confname, "predict", "errorratio"))
    # count = 1
    #
    # if model == "test":
    #     xdata = pd.read_excel("E:\dataset\data\工艺参数_珠海基地.xlsx", sheet_name=1)
    #     data_x = xdata.iloc[:, list(range(8, xdata.shape[1]))]
    #     data_x.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
    #     gru_testmodel(data_x.iloc[-2:, :], 12)
    # else:
    #     data_x, data_y, data_col, pear, diffvalue, difflen, dataprex, dataprey, data_y_field = DataRead().data_readfile(
    #         "E:\dataset\data\萃取数据.xlsx")
    #
    #     dataparam = np.load('dataparam.npz')
    #     diffvalue = dataparam['diffvalue']
    #     difflen = dataparam['difflen']
    #     pear = dataparam['pear']
    #     datamed = dataparam['datamed']
    #     # data_x, data_y, data_col, pear=  DataRead().data_readfile("E:\BaiduNetdiskDownload\windqua//")
    #
    #     if model == "auto":
    #         model_encryption(modelname, 'encryt_file1', 'license1')
    #         logging.info(data_y[-1])
    #         dataout_pre = randomval(diffvalue, difflen, 30, pear, dataprex).predict(batch_size, datamed,
    #                                                                                      data_y[-1],
    #                                                                                      errorratio, 'doe')
    #         dataout_x = dataprex.re_minmax(dataout_pre)
    #         logging.info(dataout_x)
    #
    #     # importarray = Train(data_x, data_y, data_col, pear).train_model()
    #     # logging.info(importarray)
    #     if model == "train":
    #         gru = GruTrain(data_x, data_y, batch_size, dataprex, dataprey)
    #         gru.train_gru_model(data_y_field)