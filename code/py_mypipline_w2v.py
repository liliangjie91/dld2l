#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import py_mydata_RNN as mdata
import py_mymodel_RNN as mmodel
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math, torch, collections
from torch import nn

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)



def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')

d2l.Image

if __name__ == '__main__':
    # loss = SigmoidBCELoss()
    # batch_size, max_window_size, num_noise_words = 512, 5, 5
    # data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,num_noise_words)
    #
    # embed_size = 100
    # net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
    #                                  embedding_dim=embed_size),
    #                     nn.Embedding(num_embeddings=len(vocab),
    #                                  embedding_dim=embed_size))
    #
    # lr, num_epochs = 0.002, 5
    # train(net, data_iter, lr, num_epochs)
    # vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    # norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    # encoder = d2l.BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
    #                           ffn_num_hiddens, num_heads, num_layers, dropout)
    # tokens = torch.randint(0, vocab_size, (2, 8))
    # segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    # encoded_X = encoder(tokens, segments, None)
    # encoded_X.shape
    batch_size, max_len = 512, 64
    train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)