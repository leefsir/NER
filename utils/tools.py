#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/4 10:10 
# ide： PyCharm
import json


def success_result(result):
    res = {'code': 1, 'success': True, 'message': 'success', 'data': result}
    return json.dumps(res, ensure_ascii=False)


def fail_result(result):
    res = {'code': -1, 'success': False, 'message': 'fail', 'data': result}
    return json.dumps(res, ensure_ascii=False)


def myiter(d, cols=None):
    if cols is None:
        v = d.values.tolist()
        cols = d.columns.values.tolist()
    else:
        j = [d.columns.get_loc(c) for c in cols]
        v = d.values[:, j].tolist()

    n = namedtuple('MyTuple', cols)

    for line in iter(v):
        yield n(*line)
