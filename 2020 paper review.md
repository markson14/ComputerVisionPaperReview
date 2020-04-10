### Data Augmentation

#### GridMask Data Augmenation

**3 Categories of Augmentation**

1. Spatial transformation
2. Color Distortion
3. Information Dropping

#### Cutmix



**Cutmix/cutout/mixup**

![Screen Shot 2020-03-10 at 4.22.42 pm](assets/Screen%20Shot%202020-03-10%20at%204.22.42%20pm-3828736.png)

![Screen Shot 2020-03-10 at 4.22.33 pm](assets/Screen%20Shot%202020-03-10%20at%204.22.33%20pm-3828736.png)

- 作用

![Screen Shot 2020-03-10 at 4.23.21 pm](assets/Screen%20Shot%202020-03-10%20at%204.23.21%20pm-3828736.png)

**Experiment**

![Screen Shot 2020-03-10 at 4.23.37 pm](assets/Screen%20Shot%202020-03-10%20at%204.23.37%20pm-3828736.png)

![Screen Shot 2020-03-10 at 4.23.50 pm](assets/Screen%20Shot%202020-03-10%20at%204.23.50%20pm-3828736.png)

#### Mixup

$$
\hat{x}= \lambda{x_i}+(1-\lambda)x_j \ \ where \ x_i,x_j \ are \ raw \ input\ vectors  \\
\hat{y} = \lambda{y_i}+(1-\lambda)y_j \ \ where \ y_i,y_j \ are \ onehot \ label\ encoding
$$

#### Cutout

---


### Attention is all you need (Transformer)

![Screen Shot 2019-04-15 at 10.00.38 am](assets/Screen%20Shot%202019-04-15%20at%2010.00.38%20am.png)

### Attention 

> ```
> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
> ```

- 对于某个时刻输出 `y` ，它在输入`x` 上各个部分的注意力。可以理解为**权重**。 

- 不同机制下的 **attention** 计算方法

  ![Screen Shot 2019-04-15 at 10.45.23 am](assets/Screen%20Shot%202019-04-15%20at%2010.45.23%20am.png)

- 其中，$S_t$ 指的是输出序列的隐藏状态 ,$h_i$ 为输入序列的隐藏状态 

#### Self-Attention

- **输出序列** 为 **输入序列** 

#### Transformer中的Multi-head Self-Attention（Dot-product）

$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其中，$d_k$ 表示k的维度，paper里默认 `64`

当k很大时，得到的点积结果很大，使得结果处softmax梯度很小，不利于bp。

- 在encoder的self-attention中，Q，K，V是上一层的encoder输出。对于第一层，它们是word-embedding和position-embedding相加得到的输出。

- 在decoder的self-attention中，Q、K、V都来自于同一个地方（相等），它们是上一层decoder的输出。对于第一层decoder，它们就是word embedding和positional encoding相加得到的输入。但是对于decoder，我们不希望它能获得下一个time step（即将来的信息），因此我们需要进行**sequence masking**。

- 在encoder-decoder attention中，Q来自于decoder的上一层的输出，K和V来自于encoder的输出，K和V是一样的。

- Q、K、V三者的维度一样，即 $d_q = d_k = d_v$

  ```python
  import torch
  import torch.nn as nn
  
  class ScaledDotProductAttention(nn.Module):
     """Scaled dot-product attention mechanism."""
  def __init__(self, attention_dropout=0.0):
      super(ScaledDotProductAttention, self).__init__()
      self.dropout = nn.Dropout(attention_dropout)
      self.softmax = nn.Softmax(dim=2)
      
  def forward(self, q, k, v, scale=None, attn_mask=None):
      """前向传播.
  
      Args:
      	q: Queries张量，形状为[B, L_q, D_q]
      	k: Keys张量，形状为[B, L_k, D_k]
      	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
      	scale: 缩放因子，一个浮点标量
      	attn_mask: Masking张量，形状为[B, L_q, L_k]
  
      Returns:
      	上下文张量和attetention张量
      """
      attention = torch.bmm(q, k.transpose(1, 2))
      if scale:
      	attention = attention * scale
      if attn_mask:
      	# 给需要mask的地方设置一个负无穷
      	attention = attention.masked_fill_(attn_mask, -np.inf)
  	# 计算softmax
      attention = self.softmax(attention)
  	# 添加dropout
      attention = self.dropout(attention)
  	# 和V做点积
      context = torch.bmm(attention, v)
      return context, attention
  ```

![Screen Shot 2019-04-15 at 10.55.17 am](assets/Screen%20Shot%202019-04-15%20at%2010.55.17%20am.png)

- 输入的时候Q，K，V在维度上切分，默认`8` ，输入`512` 。所以 `h` 为`64`。输出的时候再将结果concat，实验结果比不切割直接通过要优。

### Layer Normalization

- 作用于**同一个样本**，计算每一个样本上的均值和方差。

### Feed forward Network

$$
FNN(x)=W^T(max(0, W^Tx + b)) +b
$$

---

### Bag of tricks for Image Classification with CNN

### Large-batch training

1. **Linear scaling learning rate**
   - e.g. ResNet-50 SGD 256 batch size 0.1 learning rate
   - init learning $rate = 0.1 * b/256$. where b is the batch size
2. **Learning rate warm up**
   - at the beginning, paras are far from the final solution
   - e.g. we use first $m$ batch to warm up, the init learning rate is $η$, at the $i$ batch where $1 ≤ i ≤ m$, set the learning rate to be $iη/m$
3. **Zero $ γ $**
   - Batch Normalization: $γxˆ + β$  **Normally**, both elements $γ$ and $β$ are initialized to 1s and 0s
   - Instead of setting them in a normal way, it set it as $γ = 0$ to all BN layers that **sit at the end of the residual block (最后一层residual block的BN层)**.
   - **easy to train at the initial stage**
4. **No bias decay**
   - Weight decay will apply to both weight and bias
   - it recommended that only apply to weight regularization to avoid overfitting. BN parameters are left unregularized

### Low-precision training (降低位数)

1. Normal setting: 32-bit floating point (FP32) precision
2. Trick switching it to larger batch size (1024) with FP16 and get higher accuracy

### Model Tweaks

**ResNet Architecture**

![Screen Shot 2019-02-01 at 3.32.42 PM](.\assets\Screen%20Shot%202019-02-01%20at%203.32.42%20PM.png)

1. ResNet-B
   - 为了避免 1x1 conv stride=2 带来的information loss
2. ResNet-C
   - 为了避免计算量，使用两个3x3 conv代替一个7x7 conv
3. ResNet-D
   - ResNet-B中path B中的1x1 conv stride=2还是会带来信息丢失，在之前加一个avgpool stride=2 能够有效避免信息丢失

![Screen Shot 2019-02-01 at 3.29.10 PM](./assets/Screen%20Shot%202019-02-01%20at%203.29.10%20PM.png)

![Screen Shot 2019-05-26 at 4.58.20 pm](assets/Screen%20Shot%202019-05-26%20at%204.58.20%20pm.png)

### Training Refinement

1. **Cosine Learning Rate Decay**

   - $η_t =\frac{1}{2} (1 + cos (\frac{tπ}{T}))η$
     - where $T$ is the total number of batches (ignore warmup stage)
     - t is the current batch
     - $η$ is the init learning rate
   - **potentially improve the training progress**

   ![Screen Shot 2019-02-01 at 3.54.09 PM](./assets/Screen%20Shot%202019-02-01%20at%203.54.09%20PM.png)

2. **Label Smoothing**

   - 正则化方法，对于ground truth的分布进行混合。原始gt分布记为$q_i$，经过label smoothing之后

   $$
   q^`_i = \begin{cases}
   &(1-\epsilon)q_i & if \ i=y,\\
   &\frac{\epsilon}{K-1} & otherwise,
   \end{cases}
   $$

   - $\epsilon$ 为常量，**K**为分类类别。可以减少模型对于标签的过度信赖，对于标签不够精准有较好的帮助。

3. **Knowledge Distillation**

   1. 训练一个复杂的网络（N1）
   2. 使用数据train N1网络并得到（M1）
   3. 根据复杂网络设计一个简单网络 （N0）
   4. 将M1 softmax 设T=20 预测数据得到 soft target
   5. soft target 和 hard target加权得出Target （推荐0.1:0.9）
   6. 使用 $label = Target$ 的数据集训练N0（T=20）得到 M0
   7. 设T=1，M0 模型为我们得到的训练好的精简模型

4. **Mixup Training**

   ![Screen Shot 2019-05-23 at 6.27.49 pm](assets/Screen%20Shot%202019-05-23%20at%206.27.49%20pm.png)

   - Data Augmentation,数据进行插值扩充

   - Weighted linear interpolation (双线性插值)

     - $ x = λx_i + (1 - λ)x_j$
     - $ y = λy_i + (1 - λ)y_j$

   - ###### $λ ∈ [0, 1]$ In mixup training, we only use $(x, y)$

5. **Result of Image Classification**

   ![Screen Shot 2019-02-02 at 8.03.51 pm](./assets/Screen%20Shot%202019-02-02%20at%208.03.51%20pm.png)

6. **Result of Object Detection **

![Screen Shot 2019-05-23 at 9.43.04 pm](assets/Screen%20Shot%202019-05-23%20at%209.43.04%20pm-8619091.png)

---

### Distill the Knowledge in a Neural Network

**Proposal**

1. 提出 **知识蒸馏 （Knowledge Distillation）** 方法，从大模型中学习到的知识中学习有用信息来训练小模型，在保证性能的前提下进行 **模型压缩** 
2. 提出一种新的 **集成模型** 方法，包括通用模型和多个专用模型，其中，专用模型用来对通用模型无法区分的细粒度（Fine-grained）类别图像进行分类

**Knowledge Distilling**

![Screen Shot 2019-04-17 at 3.05.26 pm](./assets/Screen%20Shot%202019-04-17%20at%203.05.26%20pm.png)



- cumbersome model：复杂的大模型

- distilled model：蒸馏得到的小模型

- hard target：输入数据所对应的label [0,0,1,0]

- soft target：softmax层得到的输出 [0.01,0.02,0.98,0.17]

- Softmax in distillation:
  $$
  q_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
  $$
  其中 **温度系数 T** 表示输出概率的soft程度

**实验流程**

1. 使用一个较大的 **T （例如 T=1）** 和 **Hard target** 训练一个大模型，生产 **Soft target** 

2. 使用 **Soft target** 训练一个简单的小模型（distilled model）

3. Distilled model 的Cost Function由以下两项加权平均组成： 

   - Soft target和小模型的输出数据的交叉熵 （保证小模型和大模型的结果一致性）

   - Hard target和大模型的输出数据的交叉熵 （保证小模型的结果与实际类别标签一致性）
     $$
     \frac{\delta C }{\delta z_i} = \frac{1}{T}(q_i - p_i) = \frac{1}{T}(\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}})
     $$

**Training Ensemble Model**

当数据集非常巨大以及模型非常复杂时，训练多个模型所需要的资源是难以想象的，因此提出一种新的集成模型方法，包括：

- 一个 Generalist model ：使用全部数据进行训练
- 多个 Specialist models ：对某些易混淆的类别进行专门训练的专有模型

Specialist models 的训练集中，一半是初始训练集中某些特定类别的子集（special subset），另一半由剩余初始训练集中随机采样组成。

在该方法中，只有 generalist model 耗时较长，剩余的 specialist model 由于训练数据较少，且相互独立，可以并行训练，因此整体运算量少了非常多。

但是，specialist model由于只使用特定类别的数据进行训练，因此模型对别的类别的判断能力几乎为0，导致非常容易过拟合，我们可以采用如下方法来解决：

当 specialist model 通过 hard targets 训练完成后，再使用由 generalist model 生成的 soft targets 进行 fine-tune，这样做是因为 soft targets 保留了一些对于其他类别数据的信息，因此模型可以在原来基础上学到更多知识，有效避免了过拟合
$$
KL(p^g,q) = \sum _{m \in A_k} KL(p^m,q)
$$

**实验流程**

1. 训练一个复杂的网络 `N1`
2. 使用数据train `N1`网络并得到 `M1`
3. 根据复杂网络设计一个简单网络  `N0`
4. 将M1 softmax 设 `T=20` 预测数据得到 soft target
5. soft target 和 hard target加权得出Target  `推荐0.1:0.9`
6. 使用 `label = Target` 的数据集训练 `N0 T=20`得到  `M0`
7. 设 `T=1`，`M0` 模型为我们得到的训练好的精简模型

![Screen Shot 2019-11-11 at 1.31.52 pm](assets/Screen%20Shot%202019-11-11%20at%201.31.52%20pm.png)

---

### Localization-aware Channel Pruning for Object Detection

**Abstract**

- Few of research on object detection model pruning
- Requires both semantic information and location information

**Conclusion**

- propose a localisation-aware auxiliary network with channel pruning for obj det
  - Obtain precise localisation info of default boxes by pixels alignment and enlarge the receptive fields of the default boxes when pruning shallow layers
  - Construct loss for obj det task to keep channels that contains both cls and reg

---

### DropBlock: A regularization method for convolutional networks

![Screen Shot 2020-03-09 at 1.56.58 pm](./assets/Screen%20Shot%202020-03-09%20at%201.56.58%20pm.png)

**Experiment**

![Screen Shot 2020-03-09 at 2.12.47 pm](assets/Screen%20Shot%202020-03-09%20at%202.12.47%20pm-3734759.png)

**Comparison**

- DropBlock vs. dropout & SpatialDropout

![Screen Shot 2020-03-09 at 2.24.15 pm](assets/Screen%20Shot%202020-03-09%20at%202.24.15%20pm.png)

---

### Learning from Web Data with Memory Module

**Abstract**

- label noise：由多级标签或众包导致的
- background noise：由于背景或主体不明确导致的

![Screen Shot 2020-03-10 at 4.21.27 pm](assets/Screen%20Shot%202020-03-10%20at%204.21.27%20pm.png)

---

### Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)

**Abstract**

- 两种基本范式，分别是使用标签类别和使用正负样本对标签进行学习
- 使用类标签时，一般需要用分类损失函数（比如 softmax + cross entropy）优化样本和权重向量之间的相似度；使用样本对标签时，通常用度量损失函数（比如 triplet 损失）来优化样本之间的相似度。
- 这两种学习方法之间并无本质区别，其目标都是最大化类内相似度（$s_p$）和最小化类间相似度（$s_n$）。从这个角度看，很多常用的损失函数（如 triplet 损失、softmax 损失及其变体）有着相似的优化模式: 它们会将$s_n$和$s_p$组合成相似度对 (similarity pair)来优化，并试图减小（$s_n - s_p$）。在（$s_n - s_p$）中，增大 等效于降低 。这种对称式的优化方法容易出现以下两个问题，如图 1 (a) 所示。

![Screen Shot 2020-03-27 at 4.35.42 pm](assets/Screen%20Shot%202020-03-27%20at%204.35.42%20pm.png)

---

### Designing Network Design Spaces

**Abstract**

- 

