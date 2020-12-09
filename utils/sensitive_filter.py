#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/4 9:23
# ide： PyCharm

import time

from config.configs import SEN_FILE


class DFAFilter(object):
    def __init__(self, sensitive_file):
        # super(DFAFilter, self).__init__()
        self.keyword_chains = {}
        self.delimit = '\x00'
        self.parse_sensitive_words(sensitive_file)

    # 读取解析敏感词
    def parse_sensitive_words(self, path):
        with open(path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for keyword in lines:
                keyword = str(keyword).strip().replace(',', '')
                if '' is not keyword:
                    self.add_sensitivewords(keyword)

    # 生成敏感词树
    def add_sensitivewords(self, keyword):
        keyword = keyword.lower()
        chars = keyword.strip()
        if not chars:
            return
        level = self.keyword_chains
        for i in range(len(chars)):
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}

                    last_level, last_char = level, chars[j]

                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
            if i == len(chars) - 1:
                level[self.delimit] = 0

    def detect_sensitive_words(self, message):
        """
        检测敏感词
        :param message:
        :param repl:
        :return:
        """
        message = message.lower()
        ret = []
        start = 0
        senwords = []
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            message_chars = message[start:]
            for char in message_chars:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        words = message[start:start+step_ins]
                        # ret.append(repl * step_ins)
                        start += step_ins - 1
                        senwords.append(''.join(words))
                        break
                else:
                    ret.append(message[start])
                    break
            start += 1
        # return True if len(senwords) > 0 else False
        return senwords

    def filter_sensitive_words(self, message, repl="*"):
        """
        过滤并替换敏感词
        :param message:
        :param repl:
        :return:
        """
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            message_chars = message[start:]
            for char in message_chars:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                else:
                    ret.append(message[start])
                    break
            start += 1

        return ''.join(ret)

gfw = DFAFilter(SEN_FILE)
if __name__ == "__main__":
    time1 = time.time()
    gfw = DFAFilter(SEN_FILE)
    text = "小明是个王八蛋"
    res = gfw.detect_sensitive_words(text)
    print(res)
    time2 = time.time()
    print('总共耗时:' + str(time2 - time1) + 's')

