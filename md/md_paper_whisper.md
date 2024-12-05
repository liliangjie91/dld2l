[Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
## intoduction
1. 微调有弱点，可能会导致过拟合？ 所以我的模型是预训练不需要微调的
2. 对于高质量的数据集例如imagenet，可能泛化不够好，所以使用更大的数据集，  
但更大的数据集未必清洗的好，所以在这种情况下，就要求这中较弱监督的数据集的体量要更大，例如大出一个数量级  
所以作者使用了长达**68万小时**的语音数据集，这个数据集是有监督的，但可能label不会非常精准
3. 多语言68万小时中，11.7万涵盖了96中语言，还有12.5万其他语言翻译到英语的数据

总结：简单拓展弱监督数据集，就可以训练的不错--大力出奇迹

## 2 approach
### 2.1数据处理
1. 数据清洗：网上获取的语音-文本对，有的本身就是机器识别出来的文本，这会降低模型质量，需要剔除
2. 把语音切成**30s的片段**，无语音的音频也会用于训练，正好用于训练判断**是否有语音**
3. 训练过程中，对于准确率低的数据集，人工检查是否是低质数据

### 2.2模型
原版 encoder-decoder transformer
#### 输入
1. 所有音频重采样到 16000Hz
2. 每25ms一个窗口，窗口以10ms为步长
3. 对每个窗口的数据，计算80通道的对数幅度梅尔频谱图(an 80-channel log-magnitude Mel spectrogram representation)  
*梅尔频谱图是一种常用的音频特征表示方法，它模拟了人耳对不同频率的感知特性。具体来说，80通道的梅尔频谱图意味着将音频信号转换为80个不同的频率带宽上的能量分布表示。*
4. 数值归一化-均值0，方差1
5. 上一步输出是一张图，所以用卷积，经过2层卷积层，卷积核3，激活函数GLUE 
6. 上述输出加上位置编码，送入transformer的encoder
7. BPE分词
#### 80通道对数频谱图
1. 傅里叶变换与短时傅里叶变换  
声音信号是**一维信号**，直观上只能看到时域信息，不能看到频域信息。通过傅里叶变换(FT)可以变换到频域，但是丢失了时域信息，无法看到时频关系,即**不知道这个频率信号是在那个时候出现的，只记录了整个声音信号该频率下总的信息**。为了解决这个问题--短时傅里叶变换(STFT)，就是对短时的信号做傅里叶变换。原理如下：对一段长语音信号，分帧、加窗，再对每一帧做傅里叶变换，之后把每一帧的结果沿另一维度堆叠，得到一张图（类似于二维信号）-声谱图  
![alt text](img_whisper02.png)

2. 梅尔频率  
人耳能听到的频率范围是20-20000HZ，但是人耳对HZ单位不是线性敏感，而是对低HZ敏感，对高HZ不敏感，将HZ频率转化为梅尔频率，则人耳对频率的感知度就变为线性。*例如如果我们适应了1000Hz的音调，如果把音调频率提高到2000Hz，我们的耳朵只能觉察到频率提高了一点点，根本察觉不到频率提高了一倍*  
**梅尔频率转换公式**   
$M = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$  
特点：低频时，变化随f敏锐变化;高频时，变化随f缓慢变化

3. 梅尔频谱  
**梅尔频率刻度**: 将线性频率刻度转换为梅尔频率刻度。梅尔频率刻度更接近人耳对频率的感知特性，特别是在低频区域。  
**滤波器组**: 设计一组梅尔滤波器，这些滤波器在梅尔频率刻度上是均匀分布的。每个滤波器是一个三角形滤波器，用于对功率谱进行加权求和。  
**加权求和**: 对每个滤波器，计算其对应的加权求和结果。这一步将功率谱从线性频率刻度转换为梅尔频率刻度。  
**对数幅度:** 对每个梅尔滤波器的加权求和结果取对数，得到对数幅度梅尔频谱图。对数变换可以将幅度的动态范围压缩，使得小幅度的差异也能被有效捕捉。  
**生成梅尔频谱图:**
将所有帧的对数幅度梅尔频谱图堆叠在一起。这个矩阵的每一列对应一个时间帧，每一行对应一个梅尔频率通道。对于Whisper模型，这个矩阵的大小是80通道（即80个梅尔频率通道=80个滤波器）。下图是40个通道
![alt text](img_whisper03.png)

### 2.3 多任务结构
语音转文本 (Transcription): 将语音信号转换为文本。  
翻译 (Translation): 将一种语言的文本翻译成另一种语言。  
语音活动检测 (Voice Activity Detection): 检测音频中是否有语音活动，即区分语音和非语音部分。  
对齐 (Alignment): 将语音和文本对齐，通常用于训练语音识别模型。  
语言识别 (Language Identification): 确定语音信号的语言类型。  

![整体模型结构](img_whisper01.png)
#### 特殊token
1. <|startoftranscript|>  
转文本开始token
2. unique token for each language (99 total)
类似<|en|>, <|zh|>等表示某种语言
3. <|nospeech|>  
判断有无语音的label
4. <|transcribe|>  
转成文本
5. <|translate|>  
翻译成英文文本
6. <|notimestamps|>  
有的训练文本时带有时间戳的，这样可以训练语音与文本时间对齐  
7. <|endoftranscript|>   
解释标志
#### 多任务就一定好吗？
1. 杀鸡焉用牛刀？  
如果仅为了判断有无人声，不需要这么大的模型
2. 如果某一个任务不好怎么办？  


### 2.4 训练细节
1. AdamW激活，梯度裁剪
2. warm up - 在前2048批次warm up
3. 正常训练 - 学习率线性递减到0
4. batch_size=256
5. 总结训练了$2^{20}$次更新，大于2-3个epoch
6. 学习率根据模型规模从小到大范围是1.5e-3到2.0e-4