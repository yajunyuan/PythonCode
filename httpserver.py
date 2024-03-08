from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing
import json

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
            random_list.append([random.uniform(low, high) for i in range(round(self.rand_num * self.influence_coeff[i]))])
        print(random_list)
        self.liststogroup(random_list)
        print(len(self.group_list), self.group_list)

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

model = GruRNN(6, 30, output_size=1, num_layers=1)
model.load_state_dict(torch.load('lstm_model.pt'))
model.eval()
torch.no_grad()
group_key=['val1','val2','val3','val4','val5','val6']

def predict(data):
    random_value = randomval(5, 10, [0.1, 0.1, 0.2, 0.4, 0.1, 0.1], data)
    random_value.create_random_list()
    # data = [[2,3,4,5,6,7], [3,4,5,6,7,8],[1,2,3,4,5,6],[6,7,8,9,10,11]]
    # data = preprocessing.StandardScaler().fit_transform(np.array(data).reshape(1,-1))
    print(np.array(random_value.group_list).shape)
    input = torch.from_numpy(np.array(random_value.group_list).reshape(-1,1,6)).to(torch.float32)
    outputs = model(input).tolist()
    print(outputs)
    index = outputs.index(max(outputs))
    np.array(outputs).reshape(-1,1)
    # data_out = list(map(lambda x: str(x), random_value.group_list[index]))
    # data_str = ','.join(data_out)
    data_dict = dict(zip(group_key, random_value.group_list[index]))
    return data_dict

class RequestHandler(BaseHTTPRequestHandler):
    # 处理一个GET请求
    def do_POST(self):
        data = self.receive_data()
        self.send_content(data)
    def receive_data(self):
        req_datas = self.rfile.read(int(self.headers['content-length']))
        # print("--------------------接受client发送的数据----------------")
        # res1 = req_datas.decode('utf-8')
        res = json.loads(req_datas)
        print(res)
        data1 = predict(res["data"])
        data2 = {
            "seq": 1,
            "predict_data": data1,
            "time": self.date_time_string()
        }
        data = json.dumps(data2)
        return data

    def send_content(self, data):
        self.send_response(200)
        self.send_header("Content-type", 'application/json')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data.encode('utf-8'))


if __name__ == '__main__':
    serverAddress = ('192.168.13.21', 8000)
    server = HTTPServer(serverAddress, RequestHandler)
    server.serve_forever()

# import win32event
# import win32service
# import win32serviceutil
# 
# 
# class PythonService(win32serviceutil.ServiceFramework):
#     _svc_name_ = "AIweb"  # 服务名
#     _svc_display_name_ = "AIweb"  # 服务在windows系统中显示的名称
#     _svc_description_ = "This code is a Python service Test"  # 服务的描述
# 
#     def __init__(self, args):
#         # __init__的写法基本固定，可以参考帮助文档中的任意一种
#         # https://www.programcreek.com/python/example/99659/win32serviceutil.ServiceFramework
#         win32serviceutil.ServiceFramework.__init__(self, args)
#         self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
# 
#     def SvcDoRun(self):
#         # 把自己的代码放到这里，就OK
#         serverAddress = ('localhost', 8080)
#         server = HTTPServer(serverAddress, RequestHandler)
#         server.serve_forever()
#         # 等待服务被停止
#         win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
# 
#     def SvcStop(self):
#         # 先告诉SCM停止这个过程
#         self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
#         # 设置事件
#         win32event.SetEvent(self.hWaitStop)
# 
# 
# if __name__ == '__main__':
#     win32serviceutil.HandleCommandLine(PythonService)
#     # 括号里参数可以改成其他名字，但是必须与class类名一致