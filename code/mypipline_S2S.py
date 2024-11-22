#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import mydata_RNN as mdata
import mymodel_RNN as mmodel
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math, torch, collections

def train_s2s_my(model, data_iter, lr, num_epochs, tgt_vocab, device, model_path, incellplot = True, is_saving_best = True):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = mmodel.MaskedSoftmaxCELoss()
    best_loss = 1000.0
    model.train()
    if incellplot:
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs], figsize=[5, 4])
    for epo in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 输出序列第一个元素设置为<bos>字符
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:,:-1]], -1)
            Y_hat, _ = model(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len).sum()
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l, num_tokens)
        if epo % 10 == 0:
            if incellplot:
                animator.add(epo + 1, (metric[0] / metric[1],))
            loss_cur = metric[0] / metric[1]
            print(f'epo: {epo:d}/{num_epochs:d} loss {loss_cur:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
            if loss_cur<best_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = loss_cur
                print("saved model to {}\n current best loss={:.3f}".format(model_path, best_loss))
    print(f'epochs {num_epochs:d} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


def predict_s2s_my(model, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将model设置为评估模式
    model.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    _, enc_state = model.encoder(enc_X) #enc_valid_len
    dec_state = model.decoder.init_state(enc_state) # , enc_valid_len
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, enc_state, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(model.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    if len(output_seq) <= 1:
        output_seq += [0,0]
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def predict_s2s_atten_my(model, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将model设置为评估模式
    model.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs, enc_state = model.encoder(enc_X) #enc_valid_len
    dec_state = model.decoder.init_state(enc_state) # , enc_valid_len
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, enc_outputs, dec_state, enc_valid_len)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(model.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    if len(output_seq) <= 1:
        output_seq += [0,0]
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def train_pipline(save_path, batch_size, num_steps,embed_size,
                  num_hiddens, num_layers, dropout, num_epochs = 200, lr=0.01, num_examples=10000, incellplot=True):
    device = d2l.try_gpu()
    # train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    train_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, num_examples=num_examples, is_split=True)
    encoder = mmodel.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = mmodel.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    model = mmodel.EncoderDecoder(encoder, decoder)

    model_prefix = "_bz{}_ns{}_dime{}_dimh{}_nl{}_epo{}_lr{}_{}w.pkl".format(batch_size, num_steps, embed_size,
                                                                    num_hiddens, num_layers, num_epochs, lr, num_examples//10000)
    train_s2s_my(model, train_iter, lr, num_epochs, tgt_vocab, device,
                 model_path= save_path + model_prefix, incellplot=incellplot, is_saving_best=True)
    # if save_path:
    #     save_path = save_path + model_prefix
    #     torch.save(model.state_dict(), save_path)
    #     print("saved final model to ", save_path)
    plt.show()
    return save_path

def pridict_pipline(model_path, batch_size, num_steps, embed_size,
                    num_hiddens, num_layers, dropout, device, num_examples=10000):
    train_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, num_examples=num_examples, is_split=True)
    model = mmodel.EncoderDecoder(mmodel.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                                    dropout), 
                                mmodel.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                                    dropout)
                                )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # _,_,_, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, is_split=True)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    # engs=['the baby was crying to be fed .']
    # fras=['the baby was crying to be fed .']
    if len(eval_eng_list) > 0:
        engs, fras = eval_eng_list, eval_fra_list
    blues, k = [], 0
    # print(len(engs))
    for eng, fra in zip(engs, fras):
        # print(eng)
        translation, attention_weight_seq = predict_s2s_my(model, eng, src_vocab, tgt_vocab, num_steps, device)
        blue = bleu(translation, fra, k=2)
        if k%500==0:
            print(f'{eng} => {translation} \n correct trans: {fra}, bleu={blue:.3f}')
        k+=1
        blues.append(blue)
    print('##########################\navg blue:{:.3f}, test num:{}'.format(sum(blues)/len(blues), len(blues)))

def train_s2s_attention_pipline(save_path, batch_size, num_steps,embed_size,
                  num_hiddens, num_layers, dropout, num_epochs = 200, lr=0.01, num_examples=10000, incellplot=True):
    device = d2l.try_gpu()
    # train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    train_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, num_examples=num_examples, is_split=True)
    encoder = mmodel.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = mmodel.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, device, num_layers, dropout)
    model = mmodel.EncoderDecoderAttention(encoder, decoder)

    model_prefix = "_bz{}_ns{}_dime{}_dimh{}_nl{}_epo{}_lr{}_{}w.pkl".format(batch_size, num_steps, embed_size,
                                                                    num_hiddens, num_layers, num_epochs, lr, num_examples//10000)
    train_s2s_my(model, train_iter, lr, num_epochs, tgt_vocab, device,
                 model_path= save_path + model_prefix, incellplot=incellplot, is_saving_best=True)
    # if save_path:
    #     save_path = save_path + model_prefix
    #     torch.save(model.state_dict(), save_path)
    #     print("saved final model to ", save_path)
    plt.show()
    return save_path


def pridict_s2s_attention_pipline(model_path, batch_size, num_steps, embed_size,
                    num_hiddens, num_layers, dropout, device, num_examples=10000):
    train_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, num_examples=num_examples, is_split=True)
    model = mmodel.EncoderDecoder(mmodel.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                                    dropout),
                                mmodel.Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, device, num_layers,
                                                               dropout)
                                )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # _,_,_, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, is_split=True)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    # engs=['the baby was crying to be fed .']
    # fras=['the baby was crying to be fed .']
    if len(eval_eng_list) > 0:
        engs, fras = eval_eng_list, eval_fra_list
    blues, k = [], 0
    # print(len(engs))
    for eng, fra in zip(engs, fras):
        # print(eng)
        translation, attention_weight_seq = predict_s2s_atten_my(model, eng, src_vocab, tgt_vocab, num_steps, device)
        blue = bleu(translation, fra, k=2)
        if k%500==0:
            print(f'{eng} => {translation} \n correct trans: {fra}, bleu={blue:.3f}')
        k+=1
        blues.append(blue)
    print('##########################\navg blue:{:.3f}, test num:{}'.format(sum(blues)/len(blues), len(blues)))

if __name__ == '__main__':
    model_path = './saved_models/nmt_eng2fra_s2s_atten'
    batch_size, num_steps = 64, 16
    embed_size, num_hiddens, num_layers, dropout = 128, 128, 2, 0.2
    incellplot = False
    epoch = 200
    num_examples = 10000
    device = d2l.try_gpu()
    # model_path = train_s2s_attention_pipline(model_path, batch_size, num_steps,
    #                            embed_size, num_hiddens, num_layers,
    #                            dropout,num_epochs=epoch, num_examples=num_examples, incellplot=incellplot)
    model_path = './saved_models/nmt_eng2fra_s2s_atten_bz64_ns16_dime128_dimh128_nl2_epo200_lr0.01_1w.pkl'
    pridict_s2s_attention_pipline(model_path, batch_size, num_steps, embed_size, num_hiddens, num_layers, dropout, device, num_examples=num_examples)
    # 0.110 atten 32 32 10000
    # avg blue:0.281, test num:5000 nmt_eng2fra_s2s_atten_bz64_ns16_dime128_dimh128_nl1_epo200_lr0.01_10w.pkl
