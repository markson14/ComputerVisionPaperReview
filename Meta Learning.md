# Meta Learning

$$
\underset{\theta}{min}\sum_{task\ i}\mathcal{L}(\phi_i^*, \mathcal{D}_i^{ts}) \\
\phi_i^* = \theta-\alpha\nabla_{\theta}\mathcal{L}(\theta, \mathcal{D}_i^{tr})
$$

1. 定义Task $\mathcal{T}_i$
2. 将先验数据集拆分成训练 $\mathcal{D}_i^{tr}$ 和测试集 $\mathcal{D}_i^{ts}$
3. 优化 $\phi_i^* \leftarrow \theta-\alpha\nabla_{\theta}\mathcal{L}(\theta, \mathcal{D}_i^{tr})$
4. 使用 $\nabla_{\theta}\mathcal{L}(\phi_i, \mathcal{D}_i^{ts})$ 更新 $\theta$ 

解决问题：Long tail data problem; Small Dataset; General-purpose AI System

###   Supervised Learning Problem Definitions

已知数据集 $D$，求模型参数 $\phi$ 使模型最大程度拟合数据集分布：
$$
\underset{\phi}{argmax}\log p(\phi|\mathcal{D}) 
= \underset{\phi}{argmax} \log p(\mathcal{D}|\phi) + \log p(\phi)\\ \mathcal{D}=\{(x1,y1),(x2,y2)...\}
$$

1. 需要**大量已标注数据**用于训练
2. 某些场景标注数据**十分稀有**

### Meta Learning Problem Definitions

$\mathcal{D_{meta-train}}$ 是额外的训练数据，可能来自于同分布的**不同task的数据集集合**(先验数据集，先验知识)
$$
\underset{\phi}{argmax}\log p(\phi|\mathcal{D},\mathcal{D_{meta-train}}) \\ 
\begin{align}
\mathcal{D_{meta-train}}=&\{\mathcal{D_1},\mathcal{D_2}...\} \\ 
\mathcal{D_i} =& \{(x_1^i, y_1^i), (x_2^i, y_2^i)...\}
\end{align}
$$
$\mathcal{D_{meta-train}}$ 是额外的训练数据无法长期持有，我们需要学习meta- parameter $\theta$：$p(\theta | \mathcal{D}_{meta-train})$ 

- $\theta$ 便是我们从meta-train中学习到的先验知识，用于解决new task

$$
\begin{align}
\log p(\phi|\mathcal{D},\mathcal{D_{meta-train}}) =& \log \int_\theta p(\phi|\mathcal,\theta)p(\theta|\mathcal{D}_{meta-train})d\theta\\
\approx& \log p(\phi|\mathcal{D},\theta^*) + \log(\theta^*|\mathcal{D}_{meta-train})\\
\approx& \log p(\phi|\mathcal{D},\theta^*)
\end{align}
$$

目的：希望能从先验知识中汲取足够的经验，使其能够用于学习新的任务

# Few-shot Learning

目的：如何训练一个模型使其基于少量的训练样本能在目标任务中获得好的性能

少量样本训练模型必然导致过拟合，因此必须引入先验或者外部知识来提高模型泛化能力。而这在few-shot的问题设定里没有进行假设。所以通常会借鉴Meta learning的方法

### General ideas

1. Re-sampling Methods（采样方法）: 

   - Oversampling
     - High potential risk of overfitting
   - Undersampling
     - Infleasible in extreme long-tailed datasets

   - Class-balanced sampling

   - decoupling training schema: first learns the representations and classifier jointly, then obtains a balanced classifier by re-training the classifier with class-balanced sampling

2. Re-weighting Methods: Assign weights for different training samples （设置class权重）

   - Online hard example mining

3. Feature Manipulation: （特征重构）

   - Range loss enlarges inter-classes distance and reduces intra classes variations simultaneously（增加inter-class距离，减少intra-class距离）
   - Augments the feature space of tail classes by transferring the feature variance of regular classes that have sufficient training samples（通过传递具有足够训练样本的常规类别的特征方差来扩展尾部类别的特征空间）

## Few-shot Object Detection via Feature Reweighting

### Abstract

- Meta feature learner和reweighting module within one-stage detector

### Architecture

![Screen Shot 2020-07-06 at 10.47.34 am](assets/Screen%20Shot%202020-07-06%20at%2010.47.34%20am.png)



## Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax (CVPR2020)

![Screen Shot 2020-09-17 at 3.42.51 pm](assets/Screen%20Shot%202020-09-17%20at%203.42.51%20pm.png)

### Abstract

- 由于long tail尾部classes的分类的weight norm较小，原因是在训练class j的时候，会提升class j的weight norm，降低others。所以导致tail classes仅有的少量数据训练，weight norm远不及head classes。
- 提出Balanced Group Softmax来缓解目前问题

### Introduction

```markdown
1. Through comprehensive analysis, we reveal the rea- son why existing models perform not well for long-tail detection, i.e. their classifiers are imbalanced and not trained equally well, reflected by the observed imbal- anced classifier weight norms.

2. We propose a simple yet effective balanced group soft- max module to address the problem. It can be easily combined with object detection and instance segmen- tation frameworks to improve their long-tail recogni- tion performance.

3. We conduct extensive evaluations with state-of-the-art long-tail classification methods for object detection. Such benchmarking not only deepens our understand- ings of these methods as well as the unique challenges of long-tail detection, but also provides reliable and strong baselines for future research in this direction.
```

### Long-tail classification

- Re-sampling: 过采样，降采样，类均衡采样。
  - 缺点：多余的训练时间，对tail classes过拟合
- Cost-sensitive：平均每个classes的权重或增大tail classes权重
  - 缺点：参数过多难以调试

### Balanced Group Softmax

![Screen Shot 2020-09-17 at 3.32.03 pm](assets/Screen%20Shot%202020-09-17%20at%203.32.03%20pm.png)

#### Group Softmax

#### sCalbration via category “others”

## Equalization Loss for Long-Tailed Object Recognition (CVPR 2020)

### Abstract

- 该文中，将positive sample of one category视作另外category的negative sample，使其获得更多的负梯度。
- 提出equalization loss，其能从梯度反向传播与网络更新中保护学习rare categories

### Introduction

先前的工作主要从1. Sampling的方法解决long-tail class的问题。2. loss的方法主要是解决前后景类别不均衡而提出的(focal loss，面临问题是前后景数量差距过大，前景object的classes number分类还是均衡的)，在解决不同前景的长尾数据集（前景object的classes number十分不均衡）中依旧是个难题

我们主要的贡献有

1. 提出创新的角度处理long tail问题：抑制rare categories的之间的竞争，这个导致poor performance的重要原因
2. 在图像识别和图像分割中也做了消融实验证明方法的可行性和鲁棒性

### Equalization Loss

$$
L_{EQL} = -\sum_{j=1}^Cw_j\log(\hat {p_j}) \\
w_j = 1-E(r)T_\lambda(f_j)(1-y_j) \\
TR(\lambda) = \frac{\sum_j^CT_\lambda(f_j)N_j}{\sum_j^CN_j}
$$

1. 对于小于threshold的rare categories忽略作为负样本的抑制梯度更新
2. 不忽略背景图片的梯度更新