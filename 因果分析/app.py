# 导入Flask模块
import csv
from test import gru_testmodel, auto_testmode, train_mode, getCurTrainEpoch
import xlrd
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from flask import Flask, request, render_template
import math
# 导入xlrd模块，用于读取Excel文件
# 创建一个Flask应用对象
app = Flask(__name__)

current_step = 0
@app.route('/get-current-step')
def get_current_step():
    current_step = getCurTrainEpoch()
    return str(current_step)  # 将current_step转换为字符串返回
# 定义一个路由，用于处理根目录的请求
@app.route('/')
def index():
    # 渲染HTML模板，返回给客户端
    return render_template('train.html')
@app.route('/train', methods=['POST'])
def train_process():
    training_method = request.form.get('training-method')
    if training_method == 'manual':
        # 获取用户输入的数据
        manual_data = request.form.get('manual-data')
        try:
            datatmp = [float(x) for x in manual_data.split(',')]
        except ValueError:
            return '请输入有效的数字，用逗号分隔'
    # 如果用户选择了从Excel文件中读取
    elif training_method == 'excel':
        # 获取用户上传的Excel文件
        excel_file = request.files.get('excel-data')
        # 如果没有上传文件，则返回错误信息
        if excel_file is None:
            return '请选择一个Excel文件'
        # 尝试打开Excel文件，如果失败则返回错误信息
        try:
            exceldata = pd.read_excel(excel_file)
            data = exceldata.iloc[:-3, :]
            data=data.astype('float')
            train_mode(data, data.shape[1]-1)
        except xlrd.XLRDError:
            return '请选择一个有效的Excel文件'
    elif training_method == 'csv':
        # 获取用户上传的csv文件
        csv_file = request.files.get('csv-data')
        # 如果没有上传文件，则返回错误信息
        if csv_file is None:
            return '请选择一个csv文件'
        # 尝试打开csv文件，如果失败则返回错误信息
        try:
            csvdata = pd.read_csv(csv_file)
            train_mode(csvdata, csvdata.shape[1]-1)
        except csv.Error:
            return '请选择一个有效的csv文件'
    else:
        return '请选择输入数据的方式'
    return render_template('train.html')

# 定义一个路由，用于处理/process的请求，指定请求方法为POST
@app.route('/test')
def test_index():
    return  render_template('result-old.html')
@app.route('/test/process', methods=['POST'])
def test_process():
    input_mode = request.form.get('input-mode')
    # 获取用户选择的输入数据的方式
    input_method = request.form.get('input-method')
    # 初始化一个空列表，用于存储数据
    data = []
    # 如果用户选择了手动输入
    if input_method == 'manual':
        # 获取用户输入的数据
        manual_data = request.form.get('manual-data')
        # 尝试将数据转换为浮点数列表，如果失败则返回错误信息
        try:
            datatmp = [float(x) for x in manual_data.split(',')]
            if input_mode == 'reverse':
                data = auto_testmode(np.array(datatmp))

        except ValueError:
            return '请输入有效的数字，用逗号分隔'
    # 如果用户选择了从Excel文件中读取
    elif input_method == 'excel':
        # 获取用户上传的Excel文件
        excel_file = request.files.get('excel-data')
        # 如果没有上传文件，则返回错误信息
        if excel_file is None:
            return '请选择一个Excel文件'
        # 尝试打开Excel文件，如果失败则返回错误信息
        try:
            xdata = pd.read_excel(excel_file)

            data_x = xdata.iloc[:-3, :xdata.shape[1]-1]
            data_y = xdata.iloc[:-3, [xdata.shape[1]-1]]
            xfield = list(data_x.columns.values)
            yfield = list(data_y.columns.values)
            data_x = data_x.astype('float')
            data_x.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
            data = gru_testmodel(data_x.iloc[:2, :], 1)
            # workbook = xlrd.open_workbook(file_contents=excel_file.read())
        except xlrd.XLRDError:
            return '请选择一个有效的Excel文件'
    elif input_method == 'csv':
        # 获取用户上传的csv文件
        csv_file = request.files.get('csv-data')
        # 如果没有上传文件，则返回错误信息
        if csv_file is None:
            return '请选择一个csv文件'
        # 尝试打开csv文件，如果失败则返回错误信息
        try:
            if input_mode == 'forward':
                csvdata = pd.read_csv(csv_file)
                data_x = csvdata.iloc[:3, :csvdata.shape[1]-1]
                data_y = csvdata.iloc[:3, [csvdata.shape[1]-1]]
                xfield = list(data_x.columns.values)
                yfield = list(data_y.columns.values)
                data_x.replace([np.nan, np.inf, -np.inf], 0, inplace=True)
                data = gru_testmodel(data_x, 1)
        except csv.Error:
            return '请选择一个有效的csv文件'
    else:
        return '请选择输入数据的方式'
    # 如果数据列表为空，则返回错误信息
    if not data.any():
        return '没有找到任何有效的数据'
    # # 计算数据的平均值
    # if(data.shape[0]>1):
    #     datatmp = [[round(x, 4) for x in sub_lst] for sub_lst in data]
    # else:
    #     datatmp = [round(x, 4) for x in data]
    datatmp = [[round(x, 4) for x in sub_lst] for sub_lst in data]

    ytable = PrettyTable()
    if input_mode == 'reverse':

        # 设置表头
        ytable.field_names = ['萃取1槽温度℃','萃取2槽温度℃','萃取3槽温度℃','萃取4槽温度℃','萃取5槽温度℃','萃取6槽温度℃','萃取7槽温度℃','萃取8槽温度℃',
                              '膜厚um', 'TD2入口幅宽mm', '线速度m / min','萃取1槽跳辊','萃取2槽跳辊','萃取3槽跳辊','萃取4槽跳辊','萃取5槽跳辊','萃取6槽跳辊',
                              '萃取7槽跳辊','萃取8槽跳辊','隔膜残油率']

        # ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PIRATIO', 'B', 'LSTAT']
        # ['厚度Ave', '厚度R', '面密度Ave','面密度R','孔隙率Ave','孔隙率R','透气率Ave','透气率R','针强度Ave','针强度R','105M收MD','105T收TD']
        # 添加数据
        for datatmpi in datatmp:
            ytable.add_row(datatmpi)

        result = ytable.get_html_string()
    else:
        ytable.field_names = yfield
        for datatmpi in datatmp:
            ytable.add_row(datatmpi)
        xtable = PrettyTable()
        xtable.field_names = xfield
        for index in range(0, len(datatmp)):
            xtable.add_row(np.array(data_x.iloc[[index], :]).reshape(-1, 1).tolist())
        # xtable.add_row(np.array(data_x.iloc[[-1], :]).reshape(-1,1).tolist())
        result = xtable.get_html_string() + "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" + ytable.get_html_string()

    return f'输出结果\n{result}'
if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0')