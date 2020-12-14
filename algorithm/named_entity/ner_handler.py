#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/10 9:28
# ide： PyCharm
from bert4keras.backend import K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Model

from algorithm.basis_graph.basis_graph import BasisGraph
from config.configs import CORPUS_ROOT_PATH
from utils.data_process import data_process, NerDataGenerator, NamedEntityRecognizer, Evaluator
from utils.logger import logger
from utils.tools import split


class NerHandler(BasisGraph):
    def __init__(self, params={}, Train=False):
        if not params.get('model_code'):
            params['model_code']= 'ner'
        super().__init__(params, Train)
        self.build_ViterbiDecoder()
        logger.info('init ner_handler done')

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
        logger.info('build model done')

    def data_process(self):
        labels, train_data = data_process(self.train_data_path)
        if self.valid_data_path:
            _, self.valid_data = data_process(self.valid_data_path)
        else:
            train_data, self.valid_data = split(train_data, self.split)
            # self.train_data ,self.valid_data = [self.train_data[i] for i in indexs[:sep]],[self.train_data[i] for i in indexs[sep:]]
        if self.test_data_path: _, self.test_data = data_process(self.test_data_path)
        self.index2label = dict(enumerate(labels))
        self.label2index = {j: i for i, j in self.index2label.items()}
        self.num_classes = len(labels) * 2 + 1
        self.train_generator = NerDataGenerator(train_data, self.batch_size, self.tokenizer, self.label2index,
                                                self.max_len)
        logger.info('data process done')

    def build_ViterbiDecoder(self):
        self.NER = NamedEntityRecognizer(trans=K.eval(self.CRF.trans), tokenizer=self.tokenizer, model=self.model,
                                         id2label=self.index2label,
                                         starts=[0], ends=[0])

    def compile_model(self):
        self.model.compile(
            loss=self.CRF.sparse_loss,
            optimizer=Adam(self.learning_rate),
            metrics=[self.CRF.sparse_accuracy]
        )
        logger.info('compile model done')

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
                    dics['end_pos'] = item[2] + len(item[0])
                    dics['entity_type'] = item[1]
                    entities.append(dics)
                entities = sorted(entities, key=lambda x: x['start_pos'])
                res00.append({'entities': entities})
        return res00

    def train(self):
        evaluator = Evaluator(self.model, self.model_path, self.CRF, self.NER, self.recognize, self.label2index,
                              self.valid_data, self.test_data)

        self.model.fit_generator(self.train_generator.forfit(),
                                 steps_per_epoch=len(self.train_generator),
                                 epochs=self.epoch,
                                 callbacks=[evaluator])


if __name__ == '__main__':
    params = {
        'train_data_path': CORPUS_ROOT_PATH + '/28_baidu/train.txt',
        'valid_data_path': CORPUS_ROOT_PATH + '/28_baidu/dev.txt',
        'test_data_path': CORPUS_ROOT_PATH + '/28_baidu/test.txt',
        'epoch': 6,
        'batch_size': 32,
        'gpu_id': 0,
    }
    # nerModel = NerHandler(params, Train=True)
    # nerModel.train()
    nerModel = NerHandler()
    texts = ['这次海钓的地点在厦门和深圳之间的海域,中国建设银行金融科技中心在这里举办活动', '《舞梦成真》是Darren Grant执导的剧情片，玛丽·伊丽莎白·文斯蒂德、瑞利·史密斯的参加演出']
    res = nerModel.predict(texts)
    print(res)
else:
    nerModel = NerHandler()
