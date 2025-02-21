#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from jinja2.optimizer import optimize

import py_mydata_RNN as mdata
import py_mymodel_RNN as mmodel
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math, torch, collections
from torch import nn
from py_mydata_RNN import tokenize

def predict_my(test_input, num_preds, model, vocab, token_type='char'):  # @save
    """在prefix后面生成新字符"""
    state = model.begin_state(batch_size=1)
    sep = ''
    if token_type == 'word':
        test_input = test_input.split(' ')
        sep = ' '
    outputs = [vocab[test_input[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=model.device).reshape((1, 1))
    for y in test_input[1:]:  # 预热期
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return sep.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(model, theta):  # @save
    """裁剪梯度"""
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# @save
def train_epoch_my(model, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = model.begin_state(model.batch_size)
        else:
            if isinstance(model, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        # X, y = X.to(device), y.to(device)
        X, y = torch.Tensor(X).to(device).long(), torch.Tensor(y).to(device)
        # X = X.long()
        y_hat, state = model(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(model, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(model, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    try:
        result = math.exp(metric[0] / metric[1])
    except OverflowError:
        result = math.inf
    return result, metric[1] / timer.stop()


# @save
def train_my(model, train_iter, vocab, lr, num_epochs, device, use_random_iter=False, token_type='char'):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(model, nn.Module):
        updater = torch.optim.SGD(model.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    predict = lambda prefix: predict_my(prefix, 20, model, vocab, token_type)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_my(model, train_iter, loss, updater, device, use_random_iter)
        if (epoch) % 50 == 0:
            print("epo:{}/{} loss:{:.1f} {:.0f} tokens/sec ".format(epoch+1, num_epochs, ppl, speed) + predict('time traveller'))
            # print(predict('time traveller'))
            # animator.add(epoch + 1, [ppl])
    print(f'===================== loss = {ppl:.1f}, {speed:.1f} tokens/sec on device: {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    return ppl, speed


if __name__ == '__main__':

    # batch_size, num_steps = 32, 32
    # train_iter, vocab = mdata.load_data_time_machine(batch_size, num_steps, max_tokens=-1, token_type='word')
    # # freqs = [freq for token, freq in vocab.token_freqs]
    # # d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='linear', yscale='log')
    # # plt.show()
    # # print('data sets len:', len(train_iter.corpus))
    # # # train_iter.vocab._token_freqs
    # # print('token:', len(vocab))
    # # print(vocab.token_freqs[:10])
    # # print(vocab.token_freqs[-10:])
    # dim_emb, dim_hiddens = 32, 32
    # device = d2l.try_gpu()
    # # net = mmodel.RNNModelScratch(batch_size, len(vocab), dim_emb, dim_hiddens, device)
    # net = mmodel.RNNModelSimple(batch_size,len(vocab), dim_hiddens, device)
    # net.to(device)
    # num_epochs, lr = 300, 1
    # train_my(net, train_iter, vocab, lr, num_epochs, device, token_type='word')
    #
    # prefix = 'time'
    # predict_res = predict_my(prefix, 10, net, vocab, token_type='word')
    # print(predict_res)
    plt.show()

