#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/4 10:10 
# ide： PyCharm
import json
import random


def success_result(result):
    res = {'code': 1, 'success': True, 'message': 'success', 'data': result}
    return json.dumps(res, ensure_ascii=False)


def fail_result(result):
    res = {'code': -1, 'success': False, 'message': 'fail', 'data': result}
    return json.dumps(res, ensure_ascii=False)


def save_json(jsons, json_path):
    """
      保存json，
    :param json_: json
    :param path: str
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False))
    fj.close()


def load_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.loads(fj.readlines()[0])
    return model_json

def split(train_data,sep=0.8):
    data_len = len(train_data)
    indexs = list(range(data_len))
    random.shuffle(indexs)
    sep = int(data_len * sep)
    train_data, valid_data = [train_data[i] for i in indexs[:sep]], [train_data[i] for i in
                                                                     indexs[sep:]]
    # self.train_data ,self.valid_data = [self.train_data[i] for i in indexs[:sep]],[self.train_data[i] for i in indexs[sep:]]

    return train_data,valid_data