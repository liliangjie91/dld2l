#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys, os,torch
import mydata_RNN as mdata
import mymodel_RNN as mmodel
from d2l import torch as d2l

a=torch.randn([3,4])

if __name__ == '__main__':
    # print('python dir:', sys.executable)
    train_iter, src_vocab, tgt_vocab = mdata.load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break
