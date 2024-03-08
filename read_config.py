import os
import configparser


# 读取配置文件
def getConfig(filename, section, option):
    """
    :param filename 文件名称
    :param section: 服务
    :param option: 配置参数
    :return:返回配置信息
    """
    proDir = os.path.split(os.path.realpath(__file__))[0]
    configPath = os.path.join(proDir, filename)
    conf = configparser.ConfigParser()
    conf.read(configPath)
    config = conf.get(section, option)
    return config
