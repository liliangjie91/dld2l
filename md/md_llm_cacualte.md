
# 大模型算算术
[大模型计算量](https://blog.csdn.net/sinat_39620217/article/details/136910771)

[英伟达显卡](https://blog.csdn.net/sinat_39620217/article/details/135916437)

https://cloud.tencent.com/developer/article/2317901

## 0. 基础
|参数格式|||
|--|--|--|
全精度| float32| 4字节=4Byte
半精度| float16| 2字节=2Byte
||int8 | 1字节=1Byte
||int4 | 半字节=0.5Byte

## 1. 参数量
### 1.1 传统Transformer Decoder
对于多层堆叠的transformer Decoder，模型参数总量即
decoder层数L x 单层参数量 + embedding参数
```math
N_{model} = LN_{deocder} + 2N_{emb}=L(4n^2 + 2mn) + N_{vocab}·n
```
以上是一个近似，还有bias以及layerNorm参数忽略了  
对于基础transformer decoder 类的大模型，  
参数计算可以进一步简化：
一般m=4n，公式进一步简化为$N_{model} = 12Ln^2 + N_{vocab}·n$  
后一项远小于前一项，故进一步简单估计：
$N_{model} = 12Ln^2$  
下面分段解释
#### 1.1.1 单个Decoder
实际大模型训练时使用的decoder与原版transformer的decoder不同
即大模型的decoder只有1个自注意力层
每个Decoder = 1个自注意力层，1个FFN层
$$N_{decoder} = 4n^2 + 2mn$$

1. **多头自注意力层**  
$N_{attention} = 4n^2$  
emb_input=n, heads=h, emb_QKV = n/h  
注意力层: $3 \times emb_{QKV} \times heads = 3n^2$  
拼接后的投影层：$n^2$

2. **FFN层**  
$N_{ffn} = 2mn$   
emb_input=n，emb_hidden=m

#### 1.1.2 embedding层  
该参数与最后的输出层共享权重，故只计算1次$N_{emb} = N_{vocab}·n$

### 1.2 改进Transformer Decoder
#### 1.2.1 Attention阶段-GQA技术
使用G个组可以减少存储每个头的键和值所需的内存开销，特别是在具有大的上下文窗口或批次大小的情况下。GQA提供了对模型质量和效率的细致控制，让KV矩阵的heads少于Q矩阵的heads，一般kv_heads=8 q_heads可以从32涨到128等，而QKV矩阵维度是相同的，少的那些KVhead会通过复制补充。此时，计算多头注意力参数量会有点变化
$$
N_{attention}=2*n^2 + 2*n^2*\frac{heads_{kv}}{heads_{q}}= 2*n^2*(1+\frac{heads_{kv}}{heads_{q}})
$$

#### 1.2.2 FFN阶段-GLU门控
通常加一个gate，所以多了一个参数矩阵，故$N_{ffn}=3mn$

## 2. 参数量与模型大小关系--B与G的关系
[原文链接](https://blog.csdn.net/u012939880/article/details/136911525)
对同一个模型，虽然模型不变，但是训练过程与推理过程使用的空间却差距很大，主要原因是，推理时，不需要计算梯度，不需要储存中间激活结果，不需要储存优化器参数

**推理过程:**  
计算的公式是 
```math
G = \frac{B}{(1.024)^3}\times sizeof(参数类型) 
```
比如6B全精度, 那就是
$G= \frac{6}{(1.024)^3}\times sizeof(float)=\frac{6}{(1.024)^3}\times 4\approx22.35GB$  
而如果使用量化技术，INT4类型下
$G= \frac{6}{(1.024)^3}\times 0.5\approx2.8GB$

**训练过程:**  
训练过程要保存梯度等计算过程，需要内存空间一般是推理的4倍,  
上面全精度6B需要$G\times4\approx22.35\times4=89GB$  
而且训练过程要求至少半精度，不能使用量化，所以训练过程使用的显存要远大于推理
## 3. 计算量
### 3.1 FLOPs-矩阵相乘计算次数
FLOPs：floating point operations 指的是浮点运算次数，一般特指乘加运算次数，理解为计算量，可以用来衡量算法/模型时间的复杂度  
对于矩阵相乘$A\in R^{k\times m} * B\in R^{m\times n}$  
$FLOPs = 2kmn$  
当k=1时，即向量与矩阵相乘，最终得到$1\times n$的向量，其中计算了$m\times n$次乘法，以及$m\times n$次加法
$$\begin{bmatrix}1 & 2 & 3\end{bmatrix}
\begin{bmatrix}
 1 & 1\\
 2 & 2\\
 3 & 3
\end{bmatrix}=\begin{bmatrix}1*1+2*2+3*3 & ···\end{bmatrix}=
\begin{bmatrix} 1+4+9 & ···\end{bmatrix}=\begin{bmatrix}14 & 14\end{bmatrix}$$  
以上例子，最终结果每个元素计算了3次乘法，3次加法(3个元素相加)  
总计算次数$=2nm=2*(3+3)=12$  

### 3.2 大模型计算量-前向传播
- 在简单MLP网络前向传播中，对于**1个样本(1个token)**$X\in R^{1\times n}$ 在经过1个3层的网络$\left \{W_1^{n\times m},W_2^{m\times k},W_3^{k\times o}  \right \} $时，对于bias，其数量级较小可以忽略，则
$$FLOPs=2*(nm+mk+ko)=2*(模型参数量)$$
- 在self-Attention前向传播中，对于**1个seq(设有m个token)**，则输入$X\in R^{m\times n}$  
    * 投影层
        * 8头投影$\left \{W_{Q1}^{n\times \frac{n}{8}},W_{K1}^{n\times \frac{n}{8}},W_{V1}^{n\times \frac{n}{8}},...,W_{V8}^{n\times \frac{n}{8}}  \right \} $
        * $FLOPs=8*3*(2*m*n*\frac{n}{8})=m*2*3*n^2=m*2*(该层参数量)$
        * 所以对于每一个token，投影层$FLOPs=2*(该层参数量)=6n^2$
    * self_Attetion  
        * 8头自注意力$\left \{{Q_1}^{m\times \frac{n}{8}},{K_1}^{m\times \frac{n}{8}},{V_1}^{m\times \frac{n}{8}},...,{V_8}^{m\times \frac{n}{8}}  \right \} $
        * 计算注意力分数$score_{atten}=QK^T$ 则 $FLOPs=8*2*(m*\frac{n}{8}*m)=m*2*(m*n)$
        * 计算输出结果$res=score_{atten} \cdot V $ 则 $FLOPs = 8*2*(m*m*\frac{n}{8})=m*2*(m*n)$
        * 所以对于每一个token，$FLOPs=4*(m*n)$
    * Attention 输出投影层
        * 单层前向 $W_{O}^{n\times n}$
        * $FLOPs = 2*m*n*n = m*2*(该层参数量)$
        * 所以对于每一个token，$FLOPs=2*(该层参数量)=2n^2$
    * FFN层
        * 双层前向 $\left \{W_{ffn1}^{n\times 4n},W_{ffn2}^{n\times 4n}  \right \} $
        * $FLOPs = 2*m*(n*4n+n*4n) = m*2*(该层参数量)$
        * 所以对于每一个token，$FLOPs=2*(该层参数量)=16n^2$
    * 小计-**对于每一个token**
    $$FLOPs=6n^2+2n^2+16n^2+4nm=24n^2+4nm=2*(该层参数量)+4nm $$  
    * 当m小于n或不是远大于n时，$4nm$可以忽略
    $$FLOPs\approx 2*(该层参数量)$$
- 对于L层decoder的模型**对于每一个token** $FLOPs=L*(24n^2+4nm)\approx 2*(参数量)$
- 对于最终的输出层即$W_{final}^{n\times Vocab}$ 则**对于每一个token** $FLOPs=2*(n*V)=2*(参数量)$
- **合计-对于每一个token-前向传播阶段**  
$$FLOPs_{one-token}\approx2*(模型参数量)$$

可以看出，因为模型中几乎所有运算都是矩阵相乘，而对于**一个向量输入**模型到得到结果，计算量就是2倍参数量，所以对于大模型，1个token计算量就是2倍参数量，则1个epoch的计算量就是
$$FLOPs_{前向1epo}=2*(模型参数量)*(Token数)$$

### 3.2 大模型计算量-反向传播
反向传播计算量是前向传播的2倍即(**为什么是2倍？**)
$$FLOPs_{反向1epo}=2*FLOPs_{前向1epo}=4*(模型参数量)*(Token数)$$
所以整体训练即前向+反向过程
$$FLOPs_{训练1epo}=6*(模型参数量)*(Token数)$$


### 3.3 大模型计算量-推理
推理阶段，假设prompt长度=p，最终输出长度=o
整体过程是  
第1轮：$FLOPs_{step1}=2*(模型参数量)*(p)$  
第2轮：$FLOPs_{step2}=2*(模型参数量)*(p+1)$  
...  
第o轮：$FLOPs_{stepo}=2*(模型参数量)*(p+o-1)$
$$FLOPs_{all}=2*(模型参数量)*\sum_{i=0}^{o-1}(p+i)\approx o*(2p+o)*(模型参数量)\approx o^2*(参数量) $$

可以看到，随着输出越多，计算量是平方上升的。  
这里面是有很多计算浪费的
例如，每次计算得到的QKV矩阵，除了最后一行是因为新token的emb投影得到的新内容，其他行和上步骤的矩阵是一样的。所以，每一次计算QKV大部分计算是重复计算，而且，随着输出越来越长，重复计算率几乎是100%  
基于此，就有了[**KV_cache优化技术:**](https://github.com/harleyszhang/llm_note/blob/main/1-transformer_model/kv-cache%E5%8E%9F%E7%90%86%E5%8F%8A%E4%BC%98%E5%8C%96%E6%A6%82%E8%BF%B0.md)
- 对KV矩阵做缓存，这样每次就只计算最新的1行并拼接到KV中即可。  
- 那Q呢？实际上因为我们每一轮推理的目的，是得到当前序列中最后1个token(即注意力结果矩阵的最后一行)对应的输出层的Logit输出，并以此得出下一个token，并将此token当作下一轮的输入。那么真正需要的就是Q的最后一行。
- 整体流程分为
    - prefill：对于用户输入的prompt，按照训练时的流程喂入模型并得到KV缓存以及输出的第1个token
    - decode：对于上一轮输出的token，通过投影计算得到$V_Q,V_K,V_V$3个向量，$V_K,V_V$分别拼接到KV缓存，$V_Q$对K矩阵相乘得到当前token的注意力分数，并通过Q得到最终注意力结果**向量**$V_{res}$
- 这样计算量就计算量就是线性的了：
$$FLOPs = 2*(p+o)*(模型参数量)$$
节省计算，提高速度

### 3.4 decoder-only类型的大模型的mask-attention
考虑一个问题，当前大模型预训练任务都是“预测下一个词”,那么为什么还要用mask-attention呢？  
当初transformer的decoder第一个attention是mask attention,因为当时的任务是翻译任务，训练时decoder的输入就是‘bos’字符拼接正确的翻译结果，然后使用mask，防止训练时第i个输入会使用到i之后token的信息。  
而在预测下一个词任务时，输入时abcd预测输出e。那么训练时，输入之间就是要互相看的。  
实际上，下一词任务的数据格式时这样的$X=(a,b,c,d,e),Y=(b,c,d,e,end)$  
也只有这种数据格式才能充分发挥attention的并行优势。可以看到，任务虽然是预测下一词，但实际上数据格式像是序列翻译。  
所以这种数据格式和模型搭配，可以充分发挥模型优势，所以mask就成了必须的。而且，这也与推理场景符合。

## 4. 为什么反向传播计算量是正向传播2倍
反向传播时，因为梯度的链式法则，并不是直接计算$ \frac{\partial L}{\partial W}$ 而是
$$ \frac{\partial L}{\partial w^{l-1}}= \frac{\partial L}{\partial z^{l}}\cdot  \frac{\partial z^{l}}{\partial z^{l-1}}\cdot  \frac{\partial z^{l-1}}{\partial w^{l-1}}$$
这里面涉及到的矩阵乘法比单纯的前向传播矩阵乘法多一倍