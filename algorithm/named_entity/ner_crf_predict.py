# coding=utf-8
# __project__ = " "
# __author__ = "Nicksphere"
# __time__ = "2020/5/30 下午8:25"


from bert4keras.backend import K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import open, ViterbiDecoder, sequence_padding
from bert4keras.tokenizers import Tokenizer
from keras.layers import Dense
from keras.models import Model

from configs.constant import model_root_path


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


maxlen = 256
epochs = 10
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
config_path = model_root_path+'/model/word_embedding/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = model_root_path+'/model/word_embedding/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = model_root_path+'/model/word_embedding/bert/chinese_L-12_H-768_A-12/vocab.txt'

label_file = model_root_path + '/corpus/ner/datas/28_baidu/baidu_label.txt'

labels = []
with open(label_file, mode='r', encoding='utf-8') as f:
    for line in f.read().split('\n'):
        line = str(line).strip()
        if line:
            labels.append(line)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 类别映射
# labels = ['PER', 'LOC', 'ORG']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class Bert4kerascrfHandler(object):

    def __init__(self, model_path):
        self.model, self.crf = self.load_model(model_path)
        self.verterbi = ViterbiDecoder(trans=K.eval(self.crf.trans), starts=[0], ends=[0])

    def build_model(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

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
        # model.summary()
        model.compile(
            loss=CRF.sparse_loss,
            optimizer=Adam(learing_rate),
            metrics=[CRF.sparse_accuracy]
        )
        return model, CRF

    def load_model(self, model_path):
        model, crf = self.build_model()
        model.load_weights(model_path)
        return model, crf

    def predict_(self, text):
        """
        实体识别预测
        :param sentence:
        :return:
        """
        ret = []
        batch_token_ids, batch_segment_ids, batch_token = [], [], []
        for sentence in text:
            tokens = tokenizer.tokenize(sentence, max_length=maxlen)
            while len(tokens) > 510:
                tokens.pop(-2)
            batch_token.append(tokens)
            token_ids, segment_ids = tokenizer.encode(sentence, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        batch_token_ids = sequence_padding(batch_token_ids)
        batch_segment_ids = sequence_padding(batch_segment_ids)
        nodes = self.model.predict([batch_token_ids, batch_segment_ids])
        for index, node in enumerate(nodes):
            pre_dict = []
            labels = self.verterbi.decode(node)
            # labels = viterbi_decode(num_labels, node, trans)
            arguments, starting = [], False
            for i, label in enumerate(labels):
                if label > 0:
                    if label % 2 == 1:
                        starting = True
                        arguments.append([[i], id2label[(label - 1) // 2]])
                    elif starting:
                        arguments[-1][0].append(i)
                    else:
                        starting = False
                else:
                    starting = False
            pre_ = [(tokenizer.decode(batch_token_ids[index, w[0]:w[-1] + 1], batch_token[index][w[0]:w[-1] + 1]), l,
                     search(tokenizer.decode(batch_token_ids[index, w[0]:w[-1] + 1], batch_token[index][w[0]:w[-1] + 1]),text[index]))
                    for w, l in arguments]

            ret.append(pre_)
        return ret

    def predict(self, sentences):
        """

        :param sentences:
        :return:
        """
        res00 = []
        text = [t for t in sentences if t]
        if text:
            tmp_res = self.predict_(sentences)
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


if __name__ == '__main__':
    # m_path = './best_model.weights'
    # m_path = model_root_path + '/model/named_entity/recognition/baidu_28/best_model_baidu.h5'
    m_path = model_root_path + '/model/named_entity/recognition/baidu_28/best_model_baidu_full.h5'
    handler = Bert4kerascrfHandler(m_path)
    texts = ['这次海钓的地点在厦门和深圳之间的海域,中国建设银行金融科技中心在这里举办活动', '日俄两国国内政局都充满了变数']
    # res = handler.model_predict(texts)
    res = handler.predict(texts)
    print(res)

