# ! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 实测验证集的F1可以到96.18%，测试集的F1可以到95.35%

from bert4keras.backend import keras, K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import open, ViterbiDecoder
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

# from configs.constant import model_root_path
from config.configs import CORPUS_ROOT_PATH, BERT_DIR

maxlen = 256
epochs = 20
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
# config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

# bert配置E:\lwf_work\beikaizw\data_pro\re_ext_bert\model\chinese_L-12_H-768_A-12
config_path = BERT_DIR + '/bert_config.json'
checkpoint_path = BERT_DIR + '/bert_model.ckpt'
dict_path = BERT_DIR + '/vocab.txt'

label_file = CORPUS_ROOT_PATH + '/28_baidu/baidu_label.txt'


def load_data(filename):
    D = []
    flags = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                c = c.strip()
                if len(c.split(' ')) == 1:
                    continue
                char, this_flag = c.split(' ')
                flags.append(this_flag)
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    flags = list(set(flags))
    flags = list(set([item.split('-')[1] for item in flags if len(item.split('-'))>1]))
    with open(CORPUS_ROOT_PATH + '/28_baidu/baidu_label.txt', encoding='utf-8', mode='w') as w:
        for flag in flags:
            w.write(flag)
            w.write('\n')
            w.flush()
    return D


labels = []
with open(label_file, mode='r', encoding='utf-8') as f:
    for line in f.read().split('\n'):
        line = str(line).strip()
        if line:
            labels.append(line)
#
# # 标注数据
valid_data = load_data(CORPUS_ROOT_PATH + '/28_baidu/dev.txt')
test_data = load_data(CORPUS_ROOT_PATH + '/28_baidu/test.txt')
train_data = load_data(CORPUS_ROOT_PATH + '/28_baidu/train.txt')

# train_data = train_data[:1000]
# valid_data = valid_data[:200]
# test_data = test_data[:500]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 类别映射

# labels = ['PER', 'LOC', 'ORG']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]  # 只获取token_ids 不包含cls和sep
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1  # 防止当某个标签为0时和‘O’的标签id冲突故而使除O外的标签id>0
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]  # ['[CLS]']+[ids] +['[SEP]']
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                print(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """

    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids], [segment_ids]])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])# [[B_index],label]
                elif starting:
                    entities[-1][0].append(i)  # [[B_index,I_index,...I_index],label]
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)  # [ ('string',label),...]
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def predict(text):
    """
    实体识别预测
    :param sentence:
    :return:
    """
    tokens = NER.recognize(text)
    return tokens


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            # model.save_weights(model_root_path + '/model/named_entity/recognition/baidu_28/best_model_baidu.h5')
            model.save_weights(model_root_path + '/model/named_entity/recognition/baidu_28/best_model_baidu_full.h5')
            # model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    evaluator = Evaluate()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    # model.load_weights(model_root_path + '/model/named_entity/recognition/baidu_28/best_model_baidu.h5')
    model.load_weights(model_root_path + '/model/named_entity/recognition/baidu_28/best_model_baidu_full.h5')


