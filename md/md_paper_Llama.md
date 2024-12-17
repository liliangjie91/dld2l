## 简介
模型：
405B参数， dense transformer
window：128k
||8B|70B|405B|
|---|---|---|---|
|layers|32|80|126|
|dim_model|4096|8192|16384|
|dim_FFN|14336|28672|53248|
|attention_heads|32|64|128|
|kv_heads|8|8|8|
|peak_lr|3e-4|1.5e-4|8e-5|
|Acti Fun|swiGLU|
|Vocab size|128000|
|Position embedding|RoPE(500k)|

Llama2 使用了2T tokens 即2万亿个token
Llama3 使用了超过15T tokens

## 1. introduction
**大模型训练流程：**
1. pre-training: 大数据集上的下一词训练，类似GPT123 等
2. post-training: 微调，对齐，专业能力

**Llama 3 突出能力**
1. multilinguality
2. coding
3. reasoning
4. tool usage

**获取高质量大模型的3个关键点**
1. Data：相比之前的模型，llama3使用的数据质和量均有提高，1.8T-->15T
2. Scale：相比llama2，模型大了50倍
3. Managing complexity：管理复杂度：使用标准Transformer而非MOE；使用简单强化学习方法而非复杂方法

## 2. General Overview
**训练过程**
### pre-training
`
we pre-train a model with 405B parameters on 15.6T tokens using a
context window of 8K tokens.
`

next stage 8k-->128k
### post-training

1. 多轮 SFT+DPO
2. 结合其他能力
3. 安全措施

### 模型多模态能力训练
`we add image, video, and speech capabilities to Llama 3`

#### Multi-modal encoder pre-training
各个模态数据encoder分开训练各自的encoder。
- image: 使用image-text pairs
- speech: 自监督方法，使用mask

#### Vision adapter training
训练adapter，将image encoder 和 language model 连接起来
- video: 使用video-text pairs数据
- image: 使用image-text pairs数据

第7章详解

#### Speech adapter training
同样训练adapter，将speech encoder 和 language model 连接起来

## 3.Pre-training
- 数据的处理
- 确定模型结构与规模
- 大数据集上的高效预训练
- 预训练手册

### 3.1 Data

**数据获取与预处理**
- 在网上爬取巨量数据，处理PII,成人数据等，去重，基于模型的文本质量判断，特殊数据（code，reasoning 等），多语言数据

**数据配比**
- 50% 通用知识
- 25% 数学，reasoning
- 17% 代码
- 8% 多语言

**数据退火**

Annealing Data 的背后逻辑是模仿人类学习过程——从简单到复杂逐步进阶。对于大模型，直接使用全部数据进行训练可能导致：

- 过早过拟合：模型在初期对数据的复杂模式学习不足。
- 优化不稳定：尤其是在模型权重初始化阶段，大量复杂样本可能使梯度不稳定。 通过数据退火，可以让模型逐步适应复杂任务，从而实现更稳定的优化。

在小模型使用数据退火可以取得很好提升，但是随着模型变大，数据退火效果减弱。在405B模型上，数据退火效果不明显。
### 3.2 Model

模型基础架构没变，提升来自于数据加大和模型加大
2点小变化
- GQA
- attention mask 用在一个sequence内而不同的document,即一个sequence包含了2个document时，我们需要确保attention不会跨越document

#### Scaling Laws
预测最佳模型规模
预估模型在下游任务的表现
挑战：
- 当前的scaling laws 仅预测下一词损失，没有预测其他指标
- 当前的scaling laws 是基于小模型，对于大模型，需要新的scaling laws

解决方法
- 建立`training FLOPs`和`下游任务损失`之间的相关性
- 建立`下游任务损失`和`任务精度`之间的相关性

### 3.3 Infrastructure, Scaling, and Efficiency

### 3.4 Training recipes
- initial pre-training
渐进式增加batch size
- long context pre-training
预训练结束阶段，大幅增加context window到128K。意思是这是个渐进过程，分6次完成。
- annealing
预训练最后阶段lr退火

## 4. Post-training
多轮后训练
### 4.1 Modeling
1. 使用人工注释数据在最终预训练模型上，训练Reward Model
2. 使用SFT
3. 使用DPO 方法
### 4.2 Post-training Data

#### 偏好数据
- 后训练数据级别：edited > chosen > rejected
- 注释者为chosen选项评级，共4级，相对于rejected的等级:
    - significantly better
    - better
    - slightly better
    - marginally better
- 对于edited数据，注释者可以在chosen的基础上做编辑修改，也可以通过与模型多次反馈对话获得好的输出

#### SFT数据
#### Data Processing and Quality Control
因为后训练数据也大都来自模型，所以需要数据处理和质量控制

### 4.3 Capabilities
#### Code
训练code expert
后训练数据生成：
- 使用code expert 生成数据：提问，生成，评估，修正
- 代码翻译

#### Multilinguality

#### Math and Reasoning

#### Long Context

#### Tool Use

#### Factuality
处理幻觉

