#encoding:utf8
#@File:statistic.py

import pickle
dict_types = "./example/dict_types.pkl"
types = pickle.load(open(dict_types,'rb'))
print('types',len(types))

train_path='./example/train_data_example.pkl'
test_data = pickle.load(open(train_path,'rb'))
maxlen=max([len(visit) for visit in test_data])
print('maxlen',maxlen)