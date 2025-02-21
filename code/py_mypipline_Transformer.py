#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import py_mydata_RNN as mdata
import py_mymodel_RNN as mmodel
import py_mymodel_Transformer as mTrans
from d2l import torch as d2l
import matplotlib.pyplot as plt
import math, torch, collections

def train_transformer_my(model, data_iter,batch_size, num_steps, lr, num_epochs, tgt_vocab, device, model_path, incellplot = True, is_saving_best=True):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = mmodel.MaskedSoftmaxCELoss().to(device)
    best_loss = 1000.0
    model.train()
    if incellplot:
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs], figsize=[5, 4])
    # data_iter = data_iter.to(device)
    # bos = torch.tensor([tgt_vocab['<bos>']] * batch_size, device=device).reshape(-1, 1)

    for epo in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            # batch = batch.to(device)
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            _, seq_len = X.shape
            # 输出序列第一个元素设置为<bos>字符
            # if Y.shape[0] != batch_size:
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            mask_en = mTrans.attention_encoder_mask(X_valid_len, model.num_heads, seq_len, seq_len, device=device)
            mask_de = mTrans.attention_decoder_mask(num_steps, device)
            dec_input = torch.cat([bos, Y[:, :-1]], -1)
            Y_hat = model(X, dec_input, mask_en, mask_de)
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
            print(f'epo: {epo:d}/{num_epochs:d} loss {loss_cur:.8f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
            if loss_cur<best_loss:
                torch.save(model.state_dict(), model_path)
                best_loss = loss_cur
                print("saved model to {}\n current best loss={:.3f}".format(model_path, best_loss))
    print(f'epochs {num_epochs:d} loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


def predict_transformer_my(model, src_sentence, src_vocab, tgt_vocab, seq_len, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    model.eval()
    with torch.no_grad():  # 添加这行来避免梯度计算
        # 处理源序列
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
        enc_valid_len = torch.tensor([len(src_tokens)], device=device)
        src_tokens = d2l.truncate_pad(src_tokens, seq_len, src_vocab['<pad>'])
        enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
        # 编码器前向传播
        mask_en = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, seq_len, seq_len, device=device)
        enc_outputs = model.emb_en(enc_X)
        for encoder in model.encoders:
            enc_outputs = encoder(enc_outputs, mask_en)

        # 初始化目标序列（只包含BOS）
        dec_X = torch.full((1, seq_len), tgt_vocab['<pad>'], device=device)
        dec_X[0, 0] = tgt_vocab['<bos>']

        output_seq = []

        for i in range(seq_len - 1):  # -1 是因为已经有了BOS
            # decoder mask

            mask_en = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, i + 1, seq_len, device=device)
            dec_mask = mTrans.attention_decoder_mask(i+1, device)
            # 只使用到当前位置的输入
            current_dec_input = dec_X[:, :i+1]

            # decoder前向传播
            dec_outputs = model.emb_de(current_dec_input)
            for decoder in model.decoders:
                dec_outputs = decoder(dec_outputs, enc_outputs, mask_en, dec_mask)

            # 预测下一个token
            logits = model.linear(dec_outputs)
            # prob = model.softmax(logits[:, -1])  # 只取最后一个位置的预测
            next_word = logits[:, -1].argmax(dim=-1)  # [batch_size]

            # 如果生成了EOS，提前停止
            if next_word.item() == tgt_vocab['<eos>']:
                break

            output_seq.append(next_word.item())
            # 将预测结果填入下一个位置
            dec_X[:, i + 1] = next_word

    if len(output_seq) <= 1:
        output_seq += [0, 0]
    return ' '.join(tgt_vocab.to_tokens(output_seq))

def predict_transformer_my2(model, src_sentence, src_vocab, tgt_vocab, seq_len, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    model.eval()
    with torch.no_grad():  # 添加这行来避免梯度计算
        # 处理源序列
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
        enc_valid_len = torch.tensor([len(src_tokens)], device=device)
        src_tokens = d2l.truncate_pad(src_tokens, seq_len, src_vocab['<pad>'])
        enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
        # 编码器前向传播
        mask_en = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, seq_len, seq_len, device=device)
        enc_outputs = model.emb_en(enc_X)
        for encoder in model.encoders:
            enc_outputs = encoder(enc_outputs, mask_en)
            
        # 初始化目标序列（只包含BOS）
        dec_X = torch.full((1, seq_len), tgt_vocab['<pad>'], device=device)
        dec_X[0, 0] = tgt_vocab['<bos>']
        
        output_seq = []
        dec_mask = mTrans.attention_decoder_mask(seq_len, device)
        for i in range(seq_len-1):  # -1 是因为已经有了BOS
             # decoder mask


            # mask_en = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, i + 1, seq_len, device=device)
            # 只使用到当前位置的输入
            current_dec_input = dec_X#[:, :i+1]
            
            # decoder前向传播
            dec_outputs = model.emb_de(current_dec_input)
            for decoder in model.decoders:
                dec_outputs = decoder(dec_outputs, enc_outputs, mask_en, dec_mask)
            
            # 预测下一个token
            logits = model.linear(dec_outputs)
            # prob = model.softmax(logits[:, i])  # 只取最后一个位置的预测
            next_word = logits.argmax(dim=-1)[0][i]  # [batch_size]
            
            # 如果生成了EOS，提前停止
            if next_word.item() == tgt_vocab['<eos>']:
                break
            output_seq.append(next_word.item())
            # 将预测结果填入下一个位置
            dec_X[:, i+1] = next_word

    if len(output_seq) <= 1:
        output_seq += [0,0]
    return ' '.join(tgt_vocab.to_tokens(output_seq))

def predict_transformer_my3(model, src_sentence, src_vocab, tgt_vocab, seq_len, device, save_attention_weights=False):
    """序列到序列模型的预测"""
    model.eval()
    with torch.no_grad():  # 添加这行来避免梯度计算
        # 处理源序列
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
        enc_valid_len = torch.tensor([len(src_tokens)], device=device)
        src_tokens = d2l.truncate_pad(src_tokens, seq_len, src_vocab['<pad>'])
        enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
        # 编码器前向传播
        mask_en = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, seq_len, seq_len, device=device)
        enc_outputs = model.emb_en(enc_X)
        for encoder in model.encoders:
            enc_outputs = encoder(enc_outputs, mask_en)
            
        # 初始化目标序列（只包含BOS）
        dec_X = torch.full((1, seq_len), tgt_vocab['<pad>'], device=device)
        dec_X[0, 0] = tgt_vocab['<bos>']
        
        output_seq1 = []
        output_seq2 = []
        dec_mask = mTrans.attention_decoder_mask(seq_len, device)
        for i in range(seq_len-1):  # -1 是因为已经有了BOS
                     # 方法1的结果
            mask_en1 = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, i + 1, seq_len, device)
            dec_mask1 = mTrans.attention_decoder_mask(i+1, device)
            current_dec_input1 = dec_X[:, :i+1]
            
            # 方法2的结果
            mask_en2 = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, seq_len, seq_len, device)
            dec_mask2 = mTrans.attention_decoder_mask(seq_len, device)
            current_dec_input2 = dec_X
            # mask_en = mTrans.attention_encoder_mask(enc_valid_len, model.num_heads, i + 1, seq_len, device=device)
            # 只使用到当前位置的输入
            # current_dec_input = dec_X#[:, :i+1]
            
            # decoder前向传播
            # print('############################ 1111111111111111')
            dec_outputs1 = model.emb_de(current_dec_input1)
            for decoder in model.decoders:
                dec_outputs1 = decoder(dec_outputs1, enc_outputs, mask_en1, dec_mask1)
            # print('############################ 2222222222222222')
            dec_outputs2 = model.emb_de(current_dec_input2)
            for decoder in model.decoders:
                dec_outputs2 = decoder(dec_outputs2, enc_outputs, mask_en2, dec_mask2)

            # 预测下一个token
            logits1 = model.linear(dec_outputs1)
            logits2 = model.linear(dec_outputs2)
            # prob = model.softmax(logits[:, i])  # 只取最后一个位置的预测
            next_word1 = logits1.argmax(dim=-1)[0][i]  # [batch_size]
            next_word2 = logits2.argmax(dim=-1)[0][i]  # [batch_size]
            # 如果生成了EOS，提前停止
            if next_word1.item() == tgt_vocab['<eos>']:
                break
            output_seq1.append(next_word1.item())
            output_seq2.append(next_word2.item())
            # 将预测结果填入下一个位置
            dec_X[:, i+1] = next_word1
            
            # print(f"Step {i}:")
            # print(f"Y: {logits1[0][-1]} vs {logits2[0][-1]}")
            # print(f"Mask shapes: {mask_en1.shape} vs {mask_en2.shape}")
            # print(f"Dec input shapes: {current_dec_input1.shape} vs {current_dec_input2.shape}")

    if len(output_seq1) <= 1:
        output_seq1 += [0,0]
        output_seq2 += [0,0]
    print(' '.join(tgt_vocab.to_tokens(output_seq1)), ' '.join(tgt_vocab.to_tokens(output_seq2)))
    return ' '.join(tgt_vocab.to_tokens(output_seq1))

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

def train_transformer_pipline(save_path, batch_size, num_steps, embed_size, num_heads, dim_attention, num_layers=2,
                              num_epochs = 200, lr=0.01, num_examples=10000, incellplot=True):
    device = d2l.try_gpu()
    # train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    train_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, num_examples=num_examples, is_split=True)
    print('train src vocab {}, tgt vocab {}'.format(len(src_vocab), len(tgt_vocab)))
    model = mTrans.TransformerMy(embed_size, num_heads, dim_attention, num_layers, len(src_vocab), len(tgt_vocab))
    # print(model)
    model_prefix = "_bz{}_ns{}_dime{}_heads{}_nl{}_epo{}_lr{}_{}w.pkl".format(batch_size, num_steps, embed_size,
                                                                    num_heads, num_layers, num_epochs, lr, num_examples//10000)
    train_transformer_my(model, train_iter, batch_size, num_steps, lr, num_epochs, tgt_vocab, device,
                         model_path= save_path + model_prefix, incellplot=incellplot, is_saving_best=True)

    return save_path


def pridict_transformer_pipline(model_path, batch_size, num_steps, embed_size, num_heads, dim_attention, device,
                                num_layers=2, num_examples=10000):
    train_iter, src_vocab, tgt_vocab, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, num_examples=num_examples, is_split=True)

    model = mTrans.TransformerMy(embed_size, num_heads, dim_attention, num_layers, len(src_vocab), len(tgt_vocab))
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # _,_,_, eval_eng_list, eval_fra_list = mdata.load_data_nmt(batch_size, num_steps, is_split=True)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    # engs = ['he\'s calm .']
    # fras = ['il est calme .']
    # engs=['the baby was crying to be fed .']
    # fras=['the baby was crying to be fed .']
    # if len(eval_eng_list) > 0:
    #     engs, fras = eval_eng_list, eval_fra_list
    blues, k = [], 0
    # print(len(engs))
    num_to_show = 500
    if len(engs)<=10:
        num_to_show = 1
    for eng, fra in zip(engs, fras):
        # print(eng)
        translation = predict_transformer_my(model, eng, src_vocab, tgt_vocab, num_steps, device)
        blue = bleu(translation, fra, k=2)
        if k%num_to_show==0:
            print(f'{eng} => {translation} \n correct trans: {fra}, bleu={blue:.3f}')
        k+=1
        blues.append(blue)
    print('##########################\navg blue:{:.3f}, test num:{}'.format(sum(blues)/len(blues), len(blues)))

if __name__ == '__main__':
    model_path = './saved_models/nmt_eng2fra_transformer'
    batch_size, num_steps = 64, 16
    # embed_size, num_hiddens, num_layers, dropout = 128, 128, 2, 0.2
    embed_size, num_heads, dim_attention, num_layers = 64, 4, 16, 4 # transformer
    incellplot = False
    epoch = 200
    num_examples = 100000
    device = d2l.try_gpu()
    # model_path = train_transformer_pipline(model_path, batch_size, num_steps, embed_size, num_heads, dim_attention,
    #                                        num_layers, num_epochs=epoch, lr=0.01, num_examples=num_examples, incellplot=incellplot)
    model_path = './saved_models/nmt_eng2fra_transformer_bz64_ns16_dime64_heads4_nl4_epo200_lr0.01_10w.pkl'
    pridict_transformer_pipline(model_path, batch_size, num_steps, embed_size, num_heads, dim_attention, device,
                                num_layers, num_examples=num_examples)
    # 0.110 atten 32 32 10000
    # avg blue:0.281, test num:5000 nmt_eng2fra_s2s_atten_bz64_ns16_dime128_dimh128_nl1_epo200_lr0.01_10w.pkl
    # avg blue:0.231, test num:5000 nmt_eng2fra_transformer_bz32_ns16_dime32_heads4_nl2_epo200_lr0.01_1w.pkl
    # avg blue:0.307, test num:5000 nmt_eng2fra_transformer_bz64_ns16_dime64_heads4_nl4_epo200_lr0.01_5w.pkl
    # avg blue:0.431, test num:5000 nmt_eng2fra_transformer_bz64_ns16_dime64_heads4_nl4_epo200_lr0.01_10w.pkl
    # d2l.predict_seq2seq()
