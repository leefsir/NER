import re
import copy
from utils.dynamic_data_cache.keyword_dao import keywordinital


def entity_rule(sentence:str)->[]:
    """
    关系实体规则识别
    :param sentence: 规则识别文本
    :return: [{'word': 'XXX', 'start': 1, 'end': 3, 'type': 'relation_entity'}]
    """
    entity_list = []
    if sentence:
        for entity_type, entity_tree in keywordinital.entity_trees.items():
            tranfer_list = entity_tree.extract_keyword(sentence)
            if tranfer_list:
                tranfer_list.sort(key=lambda i: len(i), reverse=True)
                print(tranfer_list)
                for entity_single in tranfer_list:
                    position = re.finditer(entity_single, sentence)
                    for posi in position:
                        entity_dict = {'word': entity_single, 'start_pos': posi.start(), 'end_pos': posi.end(),
                                       'entity_type': entity_type}
                        if entity_dict not in entity_list:
                            entity_list.append(entity_dict)
    return entity_list


def ner_by_rule(sentence_list):
    entities = []
    for sentence in sentence_list:
        entities_list = entity_rule(sentence)
        entities_list = [i for i in entities_list if i != {}]
        entities_list.sort(key=lambda i: i['start_pos'])
        entities.append({'entities': entities_list})
    return entities


def ner_model_rule_syn(ner_rule, ner_model):
    """
    规则模型识别融合
    :param text: 原始文本
    :param ner_model_id:模型id
    :return: 实体list []
    """
    tranfer_name_list = []
    ner_model_list = []
    entity_rule_list = ner_rule
    if ner_model:
        ner_model_list = ner_model[0].get('entities')
        # ner_model_list = [elem for elem in ner_model.get('entities') if elem.get('entity_type')]
    _ner_model_list = copy.deepcopy(ner_model_list)
    for elem_rule in entity_rule_list:
        tranfer_name_list.append(elem_rule)
        for elem_model in ner_model_list:
            if elem_rule.get('start_pos') <= elem_model.get('start_pos') and elem_rule.get(
                    'end_pos') >= elem_model.get('start_pos'):
                if elem_model in _ner_model_list:
                    _ner_model_list.remove(elem_model)
            elif elem_rule.get('start_pos') <= elem_model.get('end_pos') and elem_rule.get(
                    'end_pos') >= elem_model.get(
                'end_pos'):
                if elem_model in _ner_model_list:
                    _ner_model_list.remove(elem_model)
    if _ner_model_list:
        for elem in _ner_model_list:
            tranfer_name_list.append(elem)
    tranfer_name_list.sort(key=lambda i: i['start_pos'])
    return tranfer_name_list


if __name__ == '__main__':
    while True:
        print('input: ')
        sentence = input()
        ret = ner_by_rule([sentence])
        print(ret)
