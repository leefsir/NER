#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/10 9:28 
# ide： PyCharm
import json
import os
import random
from bert4keras.backend import K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import open
from bert4keras.tokenizers import Tokenizer
from keras.layers import Dense
from keras.models import Model

from config.configs import  BERT_DIR, CORPUS_ROOT_PATH, NER_MODEL_DIR
from utils.data_process import data_process, NerDataGenerator, NamedEntityRecognizer, Evaluator


class NerHandler():
    def __init__(self, params, Train=False):
        self.is_training = params.get('is_training', False)
        self.bert_config_path = BERT_DIR + "chinese_L-12_H-768_A-12/bert_config.json"
        self.bert_checkpoint_path = BERT_DIR + "chinese_L-12_H-768_A-12/bert_model.ckpt"
        self.bert_vocab_path = BERT_DIR + "chinese_L-12_H-768_A-12/vocab.txt"
        self.model_path = NER_MODEL_DIR + "ner_best_model.weights"
        self.params_path = NER_MODEL_DIR + "ner_params.json'"
        self.maxlen = params.get('maxlen', 128)
        self.batch_size = params.get('batch_size', 32)
        print(self.bert_vocab_path)
        self.tokenizer = Tokenizer(self.bert_vocab_path, do_lower_case=True)
        self.train_data_path = params.get('train_data_path')
        self.valid_data_path = params.get('valid_data_path')
        self.test_data_path = params.get('test_data_path')
        self.maxlen = params.get('maxlen', 128)
        self.batch_size = params.get('batch_size', 32)
        self.epoch = params.get('epoch', 20)
        self.learing_rate = params.get('learing_rate', 1e-5)  # bert_layers越小，学习率应该要越大
        self.bert_layers = params.get('bert_layers', 12)
        self.crf_lr_multiplier = params.get('crf_lr_multiplier', 1000)  # 必要时扩大CRF层的学习率
        self.params = params
        gpu_id = params.get("gpu_id", None)
        self._set_gpu_id(gpu_id)  # 设置训练的GPU_ID
        if Train:
            self.data_process()
            self.build_model()
            self.compile_model()
        else:
            load_params = json.load(open(self.params_path, encoding='utf-8'))
            self.maxlen = load_params.get('maxlen', 128)
            self.num_classes = load_params.get('num_classes')
            self.label2index = load_params.get('label2index')
            self.index2label = load_params.get('index2label')
            self.build_model()
            self.load_model()
        print('init nerhandler done')

    def _set_gpu_id(self, gpu_id):
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def build_model(self):
        model = build_transformer_model(
            self.bert_config_path,
            self.bert_checkpoint_path,
        )
        output_layer = 'Transformer-%s-FeedForward-Norm' % (self.bert_layers - 1)
        output = model.get_layer(output_layer).output
        output = Dense(self.num_classes)(output)
        self.CRF = ConditionalRandomField(lr_multiplier=self.crf_lr_multiplier)
        output = self.CRF(output)
        self.model = Model(model.input, output)
        self.model.summary(120)
        self.NER = NamedEntityRecognizer(trans=K.eval(self.CRF.trans), tokenizer=self.tokenizer, model=self.model,id2label=self.index2label,
                                         starts=[0], ends=[0])
        print('build model done')

    def data_process(self):
        labels, self.train_data = data_process(self.train_data_path)
        if self.valid_data_path:
            _, self.valid_data = data_process(self.valid_data_path)
        else:
            data_len = len(self.train_data)
            indexs = list(range(data_len))
            random.shuffle(indexs)
            sep = int(data_len * 0.8)
            # self.train_data ,self.valid_data = [self.train_data[i] for i in indexs[:sep]],[self.train_data[i] for i in indexs[sep:]]
            self.valid_data = [self.train_data[i] for i in indexs[sep:]]  # 训练集全量训练，分出0.2做验证
        if self.test_data_path: _, self.test_data = data_process(self.test_data_path)
        self.index2label = dict(enumerate(labels))
        self.label2index = {j: i for i, j in self.index2label.items()}
        self.num_classes = len(labels) * 2 + 1
        self.params['num_classes'] = self.num_classes
        self.params['index2label'] = self.index2label
        self.params['label2index'] = self.label2index
        self.train_generator = NerDataGenerator(self.train_data, self.batch_size, self.tokenizer, self.label2index,
                                                self.maxlen)
        print('data process done')

    def load_model(self):
        self.model.load_weights(self.model_path)
        print('load model done')

    def compile_model(self):
        self.model.compile(
            loss=self.CRF.sparse_loss,
            optimizer=Adam(self.learing_rate),
            metrics=[self.CRF.sparse_accuracy]
        )
        print('compile model done')
    def recognize(self, text):
        tokens = self.NER.recognize(text)
        return tokens
    def predict(self, sentences):
        """
        :param sentences:
        :return:
        """
        res00 = []
        text = [t for t in sentences if t]
        if text:
            tmp_res = self.NER.batch_recognize(sentences)
            for res in tmp_res:
                entities = []
                for item in res:
                    dics = {}
                    dics['word'] = item[0]
                    dics['start_pos'] = item[2]
                    dics['end_pos'] = item[2]+len(item[0])
                    dics['entity_type'] = item[1]
                    entities.append(dics)
                entities = sorted(entities, key=lambda x: x['start_pos'])
                res00.append({'entities': entities})
        return res00

    def train(self):
        evaluator = Evaluator(self.model, self.model_path, self.CRF, self.NER, self.recognize,self.label2index,
                              self.valid_data, self.test_data)

        self.model.fit_generator(self.train_generator.forfit(),
                                 steps_per_epoch=len(self.train_generator),
                                 epochs=self.epoch,
                                 callbacks=[evaluator])

if __name__ == '__main__':
    params = {
        'train_data_path':CORPUS_ROOT_PATH + '/28_baidu/train.txt',
        'valid_data_path':CORPUS_ROOT_PATH + '/28_baidu/dev.txt',
        'test_data_path':CORPUS_ROOT_PATH + '/28_baidu/test.txt',
        'epoch':5,
        'batch_size':32,
    }
    nerModel = NerHandler(params,Train=True)
    nerModel.train()
    texts = ['这次海钓的地点在厦门和深圳之间的海域,中国建设银行金融科技中心在这里举办活动', '日俄两国国内政局都充满了变数']
    res = nerModel.predict(texts)
    print(res)