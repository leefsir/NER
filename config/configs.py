#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/4 9:23
# ide： PyCharm


# 根目录及日志配置信息
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 敏感词词库信息
SEN_FILE = root_path + '/utils/sensitive_words.txt'
# entity_dict path
ENTITY_DICT = root_path + '/utils/dynamic_data_cache/entity.csv'

# 日志文件信息
LOG_DIR = os.path.join(root_path, 'logs')
LOG_NAME = 'search_service.log'

# es 配置信息
ES_URL = ['124.70.1.30']
ES_USERNAME = 'elastic'
ES_PASSWORD = 'kms@2020'
ES_PORT = "5601"
INDEX = 'doc_section'

# mysql配置
mysql_host = "192.168.1.97"
mysql_port = "3306"
mysql_user = "root"
mysql_password = "asdfghjkl"
mysql_database = "kg_platform"
mysql_dbchar = "utf8"


# gunicorn配置信息
WORKS = 2
FLASK_URL = '0.0.0.0'
SERVICE_PORT = 7000

# 训练数据目录
CORPUS_ROOT_PATH = os.path.join(root_path,'corpus')

# bert模型目录
# BERT_DIR = 'E:/lwf_work/beikaizw/data_pro/re_ext_bert/model/'
BERT_DIR = '/home/hemei/ap/nas/kg/model/word_embedding/bert/'
MODEL_DIR = root_path + '/models/'