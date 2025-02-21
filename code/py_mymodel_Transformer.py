#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from d2l import torch as d2l

class ScaledDotProductAttentionMy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, tgt_mask=None):

        # size qkv = [batch_size, num_heads, seq_len, qkv_dim]
        # size scores = [batch_size, num_heads, seq_len, seq_len]
        # print(q.shape)
        scores = q.matmul(k.permute(0,1,3,2))/math.sqrt(q.shape[-1])
        # print("score before mask")
        # print('scores: \n',scores[0,:,0,:])
        if tgt_mask is not None:
             # 添加调试信息
            # print(f"scores shape: {scores.shape}")
            # print(f"mask shape: {tgt_mask.shape}")
            # 检查mask的值
            # print(f"mask unique values: {tgt_mask[0][0]}")

            assert tgt_mask.dim()==4
            # if tgt_mask.shape[0]==1 and tgt_mask.shape[1]==1:
            scores = scores.masked_fill(tgt_mask, -1e6)
            # print("score before mask")
            # print(scores[0][0][0])
            # else:
            #     scores[tgt_mask] = -1e6
        scores = F.softmax(scores, dim=-1)
        # print('scores: \n', scores[0, :, 0, :])
        res = scores.matmul(v)
        # print('res: \n', res[0, :, 0, :])
        return res

def attention_encoder_mask(valid_lens, num_heads, query_len, key_len, device):
    """
    valid_lens: (batch_size)
    query_len: 当前查询序列长度
    key_len: 键值序列长度（通常是源序列长度）
    res: [batch_size, num_heads, query_len, key_len]
    """
    batch_size = len(valid_lens)
    mat_valid = valid_lens.repeat(num_heads, query_len, key_len, 1).permute(3, 0, 1, 2)
    mat_arrage = torch.arange(key_len, device=device).repeat(batch_size, num_heads, query_len, 1)
    mask = mat_arrage >= mat_valid
    return mask

def attention_decoder_mask(seq_len, device):
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()#.to(device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

class MultiHeadAttentionMy(nn.Module):
    def __init__(self, dim_input, num_heads, dim_attention):
        super().__init__()
        assert dim_input == num_heads * dim_attention, "dim_input != num_heads * dim_attention"
        self.dim_input = dim_input
        self.num_heads, self.dim_attention = num_heads, dim_attention
        self.attention = ScaledDotProductAttentionMy()
        self.w_q = nn.Linear(dim_input, num_heads * dim_attention)
        self.w_k = nn.Linear(dim_input, num_heads * dim_attention)
        self.w_v = nn.Linear(dim_input, num_heads * dim_attention)
        self.w_o = nn.Linear(num_heads * dim_attention, dim_input)

    def forward(self, Q, K, V, tgt_mask=None):
        # X size = [batch_size, seq_len, dim_input]
        batch_size, seq_len, dim_input = Q.shape
        assert dim_input == self.dim_input
        # print('Q before\n',Q[0][0][:5])
        Q = self.w_q(Q)
        K = self.w_k(K)
        V = self.w_v(V)

        # 把QKV转为多头形式[batch_size, seq_len, dim_input]-->[batch_size, num_heads, seq_len, dim_attention]
        Q = Q.reshape([Q.shape[0], Q.shape[1], self.num_heads, self.dim_attention]).permute(0,2,1,3)
        K = K.reshape([K.shape[0], K.shape[1], self.num_heads, self.dim_attention]).permute(0,2,1,3)
        V = V.reshape([V.shape[0], V.shape[1], self.num_heads, self.dim_attention]).permute(0,2,1,3)
        # print('K after\n', K.shape)
        # print('K after\n', K[0,:,0,:])
        # [batch_size, num_heads, seq_len, qkv_dim] --> [batch_size, seq_len, dim_input]
        res_attention = self.attention(Q, K, V, tgt_mask).permute(0,2,1,3).reshape([batch_size, seq_len, self.dim_input])
        # print('res_attention:', res_attention.shape)
        # print('res_attention\n', res_attention[0][0][:])
        outputs = self.w_o(res_attention)
        # print('outputs:', outputs.shape)
        # print('outputs\n', outputs[0][0][:10])
        # w = self.w_o.weight.t()
        # b = self.w_o.bias
        # manual_output = torch.matmul(res_attention, w) + b
        # print('manual_output\n', manual_output[0][0][:10])

        return outputs

class TransformerEncoderCellMy(nn.Module):
    def __init__(self, dim_input, num_heads, dim_attention):
        super().__init__()
        # self.emb = nn.Embedding(vocab_size, dim_input)
        self.ln1 = nn.LayerNorm(dim_input)
        self.ln2 = nn.LayerNorm(dim_input)
        self.self_attention = MultiHeadAttentionMy(dim_input, num_heads, dim_attention)
        self.ffn = nn.Sequential(
            nn.Linear(dim_input, 4*dim_input),
            nn.ReLU(),
            nn.Linear(dim_input * 4, dim_input)
        )

    def forward(self, X, tgt_mask=None):
        # X = self.emb(X) # [batch_size, seq_len, dim_input]
        x1 = self.ln1(X)
        res_atten = X + self.self_attention(x1, x1, x1, tgt_mask)

        res_atten1 = self.ln2(res_atten)
        outputs = res_atten + self.ffn(res_atten1)
        return outputs

class TransformerDecoderCellMy(nn.Module):
    def __init__(self, dim_input, num_heads, dim_attention):
        super().__init__()
        # self.emb = nn.Embedding(vocab_size, dim_input)
        self.ln1 = nn.LayerNorm(dim_input)
        self.ln2 = nn.LayerNorm(dim_input)
        self.ln3 = nn.LayerNorm(dim_input)
        self.masked_self_attention = MultiHeadAttentionMy(dim_input, num_heads, dim_attention)
        self.cross_attention = MultiHeadAttentionMy(dim_input, num_heads, dim_attention)
        self.ffn = nn.Sequential(
            nn.Linear(dim_input, 4 * dim_input),
            nn.ReLU(),
            nn.Linear(dim_input * 4, dim_input)
        )

    def forward(self, X, enc_outputs, en_tgt_mask=None, de_tgt_mask=None):
        # X = self.emb(X)  # [batch_size, seq_len, dim_input]
        x1 = self.ln1(X)
        # print('---------------------------decoder masked_self_attention')
        # print('X \n', X[0][0][:])
        x1 = self.masked_self_attention(x1, x1, x1, de_tgt_mask)
        res_atten = X + x1
        # print('----------------------------------------------------------------------------')
        # print('res_atten masked_self_attention\n', x1[0][0][:5])

        res_atten1 = self.ln2(res_atten)
        # print('-------------------------decoder cross_attention')
        res_atten2 = res_atten + self.cross_attention(res_atten1, enc_outputs, enc_outputs,en_tgt_mask)

        res_atten3 = self.ln3(res_atten2)
        outputs = res_atten2 + self.ffn(res_atten3)
        return outputs

class TransformerMy(nn.Module):
    def __init__(self, input_dim, num_heads, qkv_dim, num_layers=2, vocab_size_en=None, vocab_size_de = None):
        super().__init__()
        self.num_heads = num_heads
        self.emb_en = nn.Embedding(vocab_size_en, input_dim)
        self.emb_de = nn.Embedding(vocab_size_de, input_dim)

        # 使用 ModuleList 创建多层 encoder 和 decoder
        self.encoders = nn.ModuleList([
            TransformerEncoderCellMy(input_dim, num_heads, qkv_dim)
            for _ in range(num_layers)
        ])

        self.decoders = nn.ModuleList([
            TransformerDecoderCellMy(input_dim, num_heads, qkv_dim)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

        # 添加线性层和 softmax 层
        self.linear = nn.Linear(input_dim, vocab_size_de)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_X, dec_X, en_tgt_mask=None, de_tgt_mask=None):
        # 依次通过每个 encoder 层
        enc_outputs = self.emb_en(enc_X)
        for encoder in self.encoders:
            enc_outputs = encoder(enc_outputs, en_tgt_mask)

        # 依次通过每个 decoder 层
        dec_outputs = self.emb_de(dec_X)
        for decoder in self.decoders:
            dec_outputs = decoder(dec_outputs, enc_outputs, en_tgt_mask, de_tgt_mask)

        # 通过线性层和 softmax 层
        dec_outputs = self.linear(dec_outputs)
        # output = self.softmax(logits)

        return dec_outputs


if __name__ == '__main__':
    print('hello')
    embed_size, num_heads, dim_attention, num_layers = 32, 4, 8, 2
    src_vocab_len, tgt_vocab_len = 2000, 3000
    model = TransformerMy(embed_size, num_heads, dim_attention, num_layers, src_vocab_len, tgt_vocab_len)
    print(model)
    # encoder = Seq2SeqEncoder(10, 8,8)
    # encoder.eval()
    # X=torch.zeros((4,8),dtype=torch.long)
    # y,h = encoder(X)
    # print(y.shape)

















