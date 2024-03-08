import numpy as np
import pandas as pd
# read original data
class DataRead:
    def __init__(self):
        pass
    def data_read(self, file1, file2):
        dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y %H:%M:%S')
        data1 = pd.read_csv(file1, parse_dates=['sync_time'], index_col='sync_time', date_parser=dateparse)
        data1.drop(columns=['id', 'status', 'create_by', 'create_time', 'update_by', 'update_time'], inplace=True)
        data2 = pd.read_csv(file2, parse_dates=['sync_time'], index_col='sync_time', date_parser=dateparse)
        data2=data2[['value']]
        print(data1.shape)
        print(data2.shape)
        data = data2.join(data1.reindex(data2.index, method='nearest'))
        # data = data2.set_index("sync_time").join(data1.set_index("sync_time"))
        print("原始数据的尺寸： ")
        print(data.shape)
        print(data.head(), data.columns.is_unique)
        data.dropna(axis=1, how='all',inplace = True)
        data.dropna(axis=0, how='all', inplace=True)
        print(data.shape)


if __name__ == '__main__':
    DataRead().data_read("E:\BaiduNetdiskDownload//real\CSV\CSV\data_line.csv", "E:\BaiduNetdiskDownload//real\CSV\CSV//new.csv")