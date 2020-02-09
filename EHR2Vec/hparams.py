#encoding:utf8
#@File:hparams.py

import argparse

class Hparams:
    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('--n_input', default=15, type=int)
    parser.add_argument('--maxlen_seqs', default=6, type=int,help="max length of input event")
    #model
    parser.add_argument('--d_model', default=8, type=int,help="dimension of entity embedding")
    parser.add_argument('--d_ff', default=32, type=int, help="number of neurons of feedword network")
    parser.add_argument('--num_blocks', default=1, type=int, help="number of blocks")
    parser.add_argument('--num_heads', default=8, type=int, help="head number of the multi-head attenion")
    parser.add_argument('--dropout_rate', default=0.1, type=float, help="dropout rate")
    #train
    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--display_step', default=1, type=int,help='display frequency of the training process')
    #path
    parser.add_argument('--data_path', default='./example/train_data_example.pkl', help='path of the train data')
    parser.add_argument('--dict_types_path', default='./example/dict_types.pkl', help='path of the dict_types')
    parser.add_argument('--save_model_path', default='./example/MedE/', help='save model path')
    parser.add_argument('--entity_embedding_path', default='./example/entity_embedding.pkl', help='save entity_embedding path')


