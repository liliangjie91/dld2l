Hierarchical Text-Conditional Image Generation with CLIP Latents
## 论文信息
- 论文标题：Hierarchical Text-Conditional Image Generation with CLIP Latents
- 论文作者：Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen
- 论文来源：2022, OpenAI
- 主要基础模型：CLIP, 扩散模型

## abstract
对比学习模型CLIP可以同时学习到语义和视觉信息。在图像生成任务中，就可以利用CLIP的这种表达能力。
具体而言分为两个阶段：
1. prior: 给CLIP文本输入，获得图像embedding
2. decoder: 以上一阶段的图像embedding为基础，生成图像

这种分层生成的方式，尤其是中间CLIP的图像embedding，可以有效提高生成图像的质量。

decoder 使用的是扩散模型GLIDE。

**GAN 与 扩散模型**
GAN更逼真，扩散模型更具多样性，后期，扩散模型的保真度得到了提升

## 1. Introduction
![整体框架](image.png)
虚线上是一般CLIP模型的架构即文本输入text encoder 得到文本表示，图像输入image encoder 得到图像表示。然后配对的表示应该是相似的。

虚线下是本文提出的模型架构。
- prior阶段，就是使用CLIP的text encoder 得到文本表示，然后输入到prior模型中，得到图像的embedding。而这个embedding目标是要和CLIP的image encoder 得到的图像表示尽可能相似。所以在训练时，CLIP的image encoder 的输出就可以当作prior的label。
- decoder阶段，就是使用扩散模型，以prior阶段得到的图像embedding为条件，生成图像。

注意：整个过程，CLIP的text encoder 和 image encoder 是冻结的，只负责输出，参数不会变。

### 图片生成模型演化
#### GAN
生成器G生成图像，判别器D判断图像真伪。
- G就是通过噪声生成图像，具体使用的是逆卷积
- D是一个二分类器，用于判断图像真伪。
- 两者在对抗中互相提升，最终使得G生成的图像越来越逼真。

优点：
- 生成图像真实度很高

问题：
- 训练不稳定，可能坍缩到某些点
- 生成多样性不足

#### AE autoencoder
**AE**
encoder-decoder 结构
encoder 将图像X压缩为低维表示-bottleneck，decoder 将低维表示解码为图像X1。目的是最小化X和X1之间的差异。

**DAE**
在AE的基础上，添加了噪声-把图像X打乱为X2
encoder 将图像X2压缩为低维表示-bottleneck，decoder 将低维表示解码为图像X1。目的是最小化X和X1之间的差异（而不是X1和X2）。
这种改变让模型变得十分稳健

**MAE**
在DAE的基础上，MAE的M即mask，即把图像mask掉一部分

**VAE**
上述中的bottleneck，目的是为了重建原来的图片。
VAE中，bottleneck的目的是为了学习到数据的隐变量。
虽然结构上和AE类似，但是目的不同。
具体来说，在encoder后加入FFN，将bottleneck的表示映射到隐变量，目的是拟合这个图像的分布，然后采样一个bottleneck的表示，然后decoder解码为图像。

**VQ-VAE**
在VAE的基础上，使用VQ-VAE的VQ即vector quantization，即使用一个码本来表示一个图像。

**VQ-VAE2**

**Diffusion**
对于一个图片X，逐步加噪声，最终会变成一个纯噪声图片，反之，逐步移除噪声，最终会还原为原图。
forward process: 逐步添加噪声
reverse process: 逐步移除噪声
因为是逐步去噪，所以diffusion模型比较慢
U-net

**DDPM**
如果直接预测下一步的图像很难，那么就预测当前图与下一图的差异r,这个差异是高斯分布的。所以只要预测均值和方差就可以。而且进一步发现，只要预测均值就可以了。

**Improved DDPM**
openai 的改进版
- 也预测了方差
- 使用scheduler来控制噪声的添加,类似学习率的scheduler,从线性到余弦
- 把模型放大可以得到更好的效果

**Diffusion model beats GAN**
- classifier guided diffusion model

**GLIDE**
- 使用classifier guided diffusion model
- classifier free guided diffusion model

