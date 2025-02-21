#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys, random, os
import numpy as np
import torch

# print('python dir:', sys.executable)
# print('is torch gpu ok:', torch.cuda.is_available())
# print('torch version:', torch.__version__)
import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt','090b5e7e70c295757f55df93cb0a180b9691891a')
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')



def read_time_machine():  # @save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine', folder='../data'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# doc -> words lists
def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print("err token type")


# word count
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:  # @save
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1, token_type='char'):  # @save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, token_type)
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):  # @save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield np.array(X), np.array(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):  # @save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset: offset + num_tokens])
    Ys = np.array(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

class SeqDataLoader:  # @save
    """加载序列数据的迭代器"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens, token_type):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens, token_type)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# my_seq = list(range(35))
# for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)

def load_data_time_machine(batch_size, num_steps,  # @save
                           use_random_iter=False, max_tokens=10000, token_type='char'):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens, token_type)
    return data_iter, data_iter.vocab


####################################### frn-eng nmt #############################

def get_processed_data_nmt(processed_data_path='default',raw_data_path='deafult'):
    # 直接获取预处理后的fra-eng 翻译数据集
    if processed_data_path == 'default':
        processed_data_path = '../data/fra-eng/fra_preprocessed.txt'
        raw_data_path = '../data/fra-eng/fra.txt'
    if os.path.exists(processed_data_path):
        print('{} exists, loading...'.format(processed_data_path))
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print('{} not exist'.format(processed_data_path))
        if not os.path.exists(raw_data_path):
            print('{} not exist, downloading'.format(raw_data_path))
            raw_text = d2l.read_data_nmt()
        else:
            print('{} exists, loading...'.format(raw_data_path))
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        text = d2l.preprocess_nmt(raw_text)
        print('saving text to {}'.format(processed_data_path))
        with open(processed_data_path, 'wb') as f:
            f.write(text.encode('utf-8'))
        return text

def nmt_split_train_eval(text,eval_num=5000):
    random.seed(0)
    text_lines = [line for line in text.split('\n')]
    random.shuffle(text_lines)
    eval_set = text_lines[:eval_num]
    eval_eng_list, eval_fra_list = [], []
    for l in eval_set:
        eng,fra = l.split('\t')
        eval_eng_list.append(eng)
        eval_fra_list.append(fra)
    train_set = text_lines[eval_num:]
    # print("train set:{}\ntest set{}".format())
    return '\n'.join(train_set), eval_eng_list, eval_fra_list

def load_data_nmt(batch_size, num_steps, num_examples=600, is_split = False):
    """返回翻译数据集的迭代器和词表"""
    text = get_processed_data_nmt()
    eval_eng_list, eval_fra_list = [], []
    if is_split:
        print('using dataset split mode!!!\ntrain set:{}\ntest set:{}'.format(num_examples,5000))
        text, eval_eng_list, eval_fra_list = nmt_split_train_eval(text)
    source, target = d2l.tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = d2l.build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list