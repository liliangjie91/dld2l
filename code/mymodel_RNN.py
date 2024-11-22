#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from d2l import torch as d2l
import matplotlib.pyplot as plt
from torch.nn.functional import dropout


############################### modelling ##################################

class RNNModelScratch:  # @save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, batch_size, vocab_size, dim_emb, num_hiddens, device):
        self.batch_size, self.vocab_size, self.dim_hiddens = batch_size, vocab_size, num_hiddens
        self.dim_emb, self.device = dim_emb, device
        self.params = self.get_params()

    def __call__(self, X, state):
        # print(X)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state)

    def begin_state(self, batch_size):
        return (torch.zeros((batch_size, self.dim_hiddens), device=self.device),)

    def get_params(self):
        vocab_size = num_outputs = self.vocab_size
        dim_emb, dim_hiddens, device = self.dim_emb, self.dim_hiddens, self.device
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        W_xh = normal((vocab_size, dim_emb))
        W_hh = normal((dim_emb + dim_hiddens, dim_hiddens))
        b_h = torch.zeros(dim_hiddens, device=device)

        W_hq = normal((dim_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def forward_fn(self, inputs, state):
        # inputs的形状：(时间步数量，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = self.params
        H, = state
        outputs = []
        for X in inputs:
            # H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            X = torch.mm(X, W_xh)
            H = torch.relu(torch.mm(torch.concat((X, H), dim=-1), W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

class RNNModelSimple(nn.Module):
    def __init__(self, batch_size, vocab_size, dim_hiddens, device, bidirectional=False):
        super().__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dim_hiddens = dim_hiddens
        self.device = device
        # self.rnn = nn.RNN(self.vocab_size, self.dim_hiddens, bidirectional=bidirectional)
        # self.rnn = nn.GRU(self.vocab_size, self.dim_hiddens, bidirectional = bidirectional)
        self.rnn = nn.LSTM(self.vocab_size, self.dim_hiddens, bidirectional = bidirectional)
        self.linear = nn.Linear(self.dim_hiddens, self.vocab_size)
        self.num_directions = 2 if self.rnn.bidirectional else 1

    def forward(self, X, state):
        X = F.one_hot(X.T.long(),self.vocab_size).to(torch.float32) # 把batch维度移到中间
        Y, state = self.rnn(X,state)
        Y = self.linear(Y.reshape(-1, Y.shape[-1]))
        return Y, state

    def begin_state(self, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.dim_hiddens),device=self.device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.dim_hiddens), device=self.device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.dim_hiddens), device=self.device))

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, dim_emb, dim_hiddens, num_layers=1, dropout=0, cell_type='gru'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim_emb)
        self.rnn = nn.GRU(dim_emb, dim_hiddens, num_layers, dropout=dropout)
        if cell_type == 'rnn':
            self.rnn = nn.RNN(dim_emb, dim_hiddens, num_layers, dropout=dropout)
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(dim_emb, dim_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        # X 原始维度[batch_size, seq_len]
        # embedding 之后的X [batch_size, seq_len, dim_emb]
        X = self.emb(X)
        # 把X转换成[seq_len, batch_size, dim_emb] 格式
        X = X.permute(1, 0, 2)
        output, state = self.rnn(X)
        # output 维度 [seq_len, batch_size, dim_hiddens] 其实是各个时间点的隐层h_t的拼接
        # state 维度 [num_layer *  num_direction, batch_size, dim_hiddens]
        return output, state

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, dim_emb, dim_hiddens, num_layers=1, dropout=0, cell_type='gru'):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim_emb)
        self.rnn = nn.GRU(dim_emb + dim_hiddens, dim_hiddens, num_layers, dropout=dropout)
        if cell_type == 'rnn':
            self.rnn = nn.RNN(dim_emb + dim_hiddens, dim_hiddens, num_layers, dropout=dropout)
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(dim_emb + dim_hiddens, dim_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(dim_hiddens, vocab_size)

    def init_state(self, en_hiddens):
        return en_hiddens

    def forward(self, X, en_state, state):
        # input arg: X, size = [batch_size, seq_len]
        # input arg: en_state, size = [num_layer *  num_direction, batch_size, dim_hiddens]
        # input arg: state, size = [num_layer *  num_direction, batch_size, dim_hiddens]
        # emb并调整维度顺序后 [seq_len, batch_size, dim_emb]
        X = self.emb(X).permute(1,0,2)
        # 原始en_state[-1] size = [batch_size, dim_hiddens]
        # repeat 之后 size = [seq_len, batch_size, dim_hiddens]
        context = en_state[-1].repeat(X.shape[0], 1, 1)
        # after cat size = [seq_len, batch_size, dim_hiddens + dim_emb]
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size, num_steps, vocab_size)
        # state的形状: (num_layers*  num_direction, batch_size, num_hiddens)
        return output, state


class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, dim_emb, dim_hiddens, device, num_layers=1, dropout=0, cell_type='gru'):
        super().__init__()
        self.device = device
        self.emb = nn.Embedding(vocab_size, dim_emb)
        self.rnn = nn.GRU(dim_emb + dim_hiddens, dim_hiddens, num_layers, dropout=dropout)
        if cell_type == 'rnn':
            self.rnn = nn.RNN(dim_emb + dim_hiddens, dim_hiddens, num_layers, dropout=dropout)
        if cell_type == 'lstm':
            self.rnn = nn.LSTM(dim_emb + dim_hiddens, dim_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(dim_hiddens, vocab_size)

    def init_state(self, en_hiddens):
        return en_hiddens

    def forward(self, X, en_outputs, state, x_vare_len):
        # input arg: X, size = [batch_size, seq_len]
        # input arg: en_outputs, size = [seq_len, batch_size, dim_hiddens]  K V
        # input arg: state, size = [num_layer *  num_direction, batch_size, dim_hiddens] Q
        # input arg: x_vare_len, size = [batch_size]
        X = self.emb(X).permute(1, 0, 2) # emb并调整维度顺序后 [seq_len, batch_size, dim_emb]

        ### attention
        # Q = state; K = en_outputs; V = en_outputs
        # res = F.softmax(Q.permute(1,0,2).matmul(K.permute(1,2,0))/math.sqrt(K.shape[-1])).matmul(K.permute(1,0,2))
        # [batch_size, 1, dim_hiddens] @ [batch_size, dim_hiddens, seq_len]  = (batch_size, 1, seq_len)
        atte_scores = state.permute(1,0,2).matmul(en_outputs.permute(1,2,0))
        atte_scores = masked_softmax_my(atte_scores/math.sqrt(en_outputs.shape[-1]), x_vare_len, device=self.device, dim=-1)
        # (batch_size, 1, seq_len) @ [batch_size, seq_len, dim_hiddens] = [batch_size, 1, dim_hiddens]
        atten_res = atte_scores.matmul(en_outputs.permute(1,0,2))[:,-1,:]
        context = atten_res.repeat(X.shape[0], 1, 1)
        # after cat size = [seq_len, batch_size, dim_hiddens + dim_emb]
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size, num_steps, vocab_size)
        # state的形状: (num_layers * num_direction, batch_size, num_hiddens)
        return output, state

def masked_softmax_my(score, valid_lens, device, dim=-1):
    # score: (batch_size, 1, seq_len)
    # valid_lens: (batch_size)
    if len(valid_lens)<=0:
        return F.softmax(score, dim=dim)
    batch_size, num_layers, seq_len = score.shape
    mat_valid = valid_lens.repeat(seq_len,num_layers,1).permute(2,1,0)
    mat_arrage = torch.arange(seq_len).repeat(batch_size,num_layers,1).to(device)
    score[mat_arrage>=mat_valid]=-1e6
    score = F.softmax(score, dim=dim)
    return score

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        _, enc_state = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_state)
        return self.decoder(dec_X, enc_state, dec_state)

class EncoderDecoderAttention(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, x_vare_len):
        en_outputs, enc_state = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_state)
        return self.decoder(dec_X, en_outputs, dec_state, x_vare_len)

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = d2l.sequence_mask(weights, valid_len)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0,2,1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss





if __name__ == '__main__':
    encoder = Seq2SeqEncoder(10, 8,8)
    encoder.eval()
    X=torch.zeros((4,8),dtype=torch.long)
    y,h = encoder(X)
    print(y.shape)

















