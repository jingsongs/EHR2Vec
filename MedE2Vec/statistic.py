#encoding:utf8
#@File:statistic.py

#encoding:utf8
#@File:statistic.py

import pickle
import argparse

def initParamaters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_types', default="./example/dict_types.pkl", help='dict_types path')
    parser.add_argument('--example_train_path', default='./example/train_data_example.pkl',
                             help="example train data path")
    return parser

def get_types_number_maxlen(exapmple_dict_types,example_train_path):
    types = pickle.load(open(exapmple_dict_types, 'rb'))
    train_data = pickle.load(open(example_train_path, 'rb'))
    return len(types),max([len(visit) for visit in train_data])

hparams=initParamaters()
hp=hparams.parse_args()
types_number,maxlen=get_types_number_maxlen(hp.dict_types,hp.example_train_path)
print('The number of dict_types:{0},max length of input event:{1}'.format(types_number,maxlen))
