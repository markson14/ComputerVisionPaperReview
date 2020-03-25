## Visual Object Tracking

![Screen Shot 2020-03-25 at 1.09.44 pm](assets/Screen%20Shot%202020-03-25%20at%201.09.44%20pm.png)

### DeepSORT: Deep Simple Online Realtime Tracker

:star2: **Highlight** :star2:：

- 减少SORT中的ID Switch

**DeepSORT**：

- Kalman Filter: 根据测量值和预估值推测出当前近似值
  
- State: $u,v,a,h,u',v',a',h'$ ，(u,v) 是bbox 中心，(a,h) 是 bbox 缩放尺度和image高度，另外四个参数代表他们的速度/导数(velocities)
  
- Assignment Problem

  - Motion Metric

    - Mahalanobis distance: between predicted Kalman states and newly arrived measurements
    - 表示第j个detection和第i条轨迹之间的运动匹配度，Si是kalman滤波器预测得到当前状态时刻观测时空的协方差矩阵，yi是轨迹在当前时刻的预测管测量，dj是第j个detection的状态(u, v, r, h)

    $$
    d^{(1)}(i,j)=(d_j-y_i)^TS_I^-1(d_j-y_i)
    \\ where \ \ \ b_{i,j}^{(1)} = threshold(d^{(1)}(i,j) \le t^{(1)})
    $$

  - Appearance Metric

    - 目标运动不确定性较低时，mahalanobis distance是一个很好的关联度量，实际中，相机运动会造成mahalanobis distance大量不能匹配。因此，对每一个BBox检测狂dj计算一个表面特征描述子rj，创建gallary存放最新的Lk=100 个的轨迹描述子，即 $R_k = \{r_k^{(i)}\}_{k=1}^{L_k}$ ，然后使用第i个轨迹和第j个轨迹的最小余弦距离作为第二个度量尺度
      $$
      d^{(2)}(i,j)=min\{1-r_j^Tr_k^{(i)} | \ r_k^{(i)} \in R_i \}
      \\ where \ \ \ b_{i,j}^{(2)} = threshold(d^{(2)}(i,j) \le t^{(2)})
      $$

  - Fusion Metric

    - 融合上述两种metric

    - $$
      c_{i,j} = \lambda d^{(1)}(i,j) + (1-\lambda)d^{(2)}(i,j)
      \\ where \ \ \ b_{i,j} = \prod_{m=1}^2b_{i,j}^{(m)}
      $$

- Matching Cascade：缓解部分遮挡导致的tracking问题，缓解kalman滤波的不确定性

  ![Screen Shot 2019-08-01 at 3.25.15 pm](assets/Screen%20Shot%202019-08-01%20at%203.25.15%20pm.png)

- Deep Appearance Descriptor：特征提取网络

  ![Screen Shot 2019-08-01 at 3.25.22 pm](./assets/Screen%20Shot%202019-08-01%20at%203.25.22%20pm.png)

---

### Fully-Convolutional Siamese Networks for Object Tracking （SiamsFC, ECCV2016）

**Abstract**

- 不对称物体检测通常使用在线学习的模式，约束了模型的泛化能力
- 提出端到端网络，使用全卷积层，实时侦测的SiamsFC网络
- 使用**无padding的CNN**，为了更好的保留位置信息
- （基于中心关键点的追踪，训练的时候将正样本放在中心）

**Structure**：

![Screen Shot 2019-08-01 at 6.21.07 pm](./assets/Screen%20Shot%202019-08-01%20at%206.21.07%20pm.png)

- X is for search, z is for exemplar

- $\varphi$ is Fully Conv Network which sharing the weight — siamese network

- 卷积，返回相似度。最后相似度高的地方被确定为bbox位置
  $$
  f(z,x) = \varphi(z) * \varphi(x) + b \mathbb{1}
  $$
  
- B1表示在的得分图中每个位置的取值

- 跟踪时，以上一帧目标位置为中心的搜索图像来计算响应得分图，将得分最大的位置乘以步长即可得到当前目标位置

**正负样本定义**

在输入搜索图上，只要和目标半径距离不超过R，就算正样本，否则为负样本。这样会导致网络对图像中心位置产生高响应，而对边缘位置为随缘。

1. K为网络总步长
2. C为目标的中心
3. U为score map的所有位置
4. R定义为半径

$$
y[u]= \left\{
\begin{array}
+1 & if \ k||u-c||\le R\\ 
-1 & otherwise\\
\end{array}
\right.
$$



**Head**:

```python
class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
```



---

### High Performance Visual Tracking with Siamese Region Proposal Network  (SiamRPN, CVPR2018)

**Abstract**

- 其他网络无法做到高精度且实时
- 提出了Siamese-RPN能做到端到端训练
- siamese network作为特征提取，region proposal subnetwork包括classification和regression
- One-shot detection task
- （基于目标范围的精确追踪）

**Network**

![Screen Shot 2020-03-23 at 6.21.30 pm](assets/Screen%20Shot%202020-03-23%20at%206.21.30%20pm.png)

上面是第一帧的bbox，下面是待检测帧。前半部分和Siamese FC一样。RPN分支分别得到cls(用来负责前后景)和reg(用来负责边框信息回归)最后的feature map，做卷积操作

1. Feature extraction: 使用无padding alexnet(remove conv2&4)提取特征
2. RPN: 生成2k cls conv和4k reg conv
3. **Correlation**: 相当于卷积操作，**template**分支提取的特征作为**卷积核**，detection分支提取的特征作为卷积的input，输出结果为分类/回归结果

**Tracking as one-shot detection**

![Screen Shot 2020-03-23 at 6.25.20 pm](assets/Screen%20Shot%202020-03-23%20at%206.25.20%20pm.png)

---

### Distractor-aware Siamese Networks for Visual Object Tracking (DaSiamRPN)

**Abstract**

- Siamese tracking网络的方法通常只是区分前景和无语义信息的背景，语义背景通常被考虑为siamese trakcer的一种障碍
- 提出一种能够使siamese tracker for accurate long-term tracking

改进

- 由于VID和YouTube-BB训练集类别少，难以胜任现实任务。这里使用COCO和ImageNet Det检测数据集训练，分别有80和200类。siamese network可以使用图像对训练，不需要完整视频。

- 增加有语义信息的负样本对，增强tracker的distraction能力

![Screen Shot 2020-03-24 at 3.56.09 pm](assets/Screen%20Shot%202020-03-24%20at%203.56.09%20pm.png)

- 训练过程中，不再让template和search region是相同目标；是让网络学习判别能力，去search region里面寻找template相似物体，而不是一个简单的有语义的物体。这样改动可以让DaSiamRPN从短时间跟踪拓展到长时间跟踪。

![Screen Shot 2020-03-24 at 3.55.55 pm](assets/Screen%20Shot%202020-03-24%20at%203.55.55%20pm.png)

---

### SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks (CVPR2019)

**Abstract**

- Siamese tracker无法使用深层神经网络
- New arch to perform layer-wise and depth-wise aggregations

**Siamese Tracking with Very Deep Network**

1. 深层神经网络的Padding会导致严格的位置误差

   - Padding在深层神经网络是无法避免的，这也是损失位置信息的重要原因之一
   - 尝试设计将正样本从只放在中心到均匀分布在某个范围内，距中心距离为shift，可以发现效果如下

   ![Screen Shot 2020-03-24 at 6.28.36 pm](assets/Screen%20Shot%202020-03-24%20at%206.28.36%20pm.png)

   

2. RPN需要来学习cls和reg的feature不一致

**Framework**

![Screen Shot 2020-03-24 at 5.27.12 pm](assets/Screen%20Shot%202020-03-24%20at%205.27.12%20pm-5043049.png)

- 现代网络一般stride为32，siamese为了精确的追踪，stride一般为8
- 这里将ResNet最后两个block的stride去掉，同时增加dialated conv，一是增加感受野，二是为了能利用上pretrained weight。改动后，后面3个block参数一致

**Layer-wise Aggregation**

- 最后三个block的输出进行融合，类似FPN，这里是输出的线性加权

$$
S_{all}=\sum^5_{l=3}\alpha_iSl, \ \ \ \ \ \ \ \ \ \ B_{all}=\sum^5_{l=3}\beta_iBl
$$

**Depth-wise Aggregation**

![Screen Shot 2020-03-24 at 6.36.12 pm](assets/Screen%20Shot%202020-03-24%20at%206.36.12%20pm.png)

- **Cross Correlation**: (a) 用于SiamFC，template在search region上面做滑动窗口获得不同位置的响应
- **Up-Channel Cross Correlation**: (b)用于SiamRPN，跟(a)不同的是在correlation之前多了两个卷积层，一个提升维度(channel)，另一个保持不变。通过卷积的方式得到最终输出。通过控制升维的卷积实现最终输出特征图的通道数
- **Depth-wise Cross Correlation**: (c) 和上面一样，但不需要提升维度，这里只是为了提供一个非siamese的特征（SiamRPN中与SiamFC不同，比如回归分支，是非对称的，因为输出不是一个响应值；需要模版分支和搜索分支关注不同的内容）。在这之后，通过类似depthwise卷积的方法，逐通道计算correlation结果，这样的好处是可以得到一个通道数非1的输出，可以在后面添加一个普通的`1x1`卷积就可以得到分类和回归的结果。
  - 这里的改进主要源自于upchannel的方法中，升维卷积参数量极大, `256 x (256 x 2k) x 3 x 3`, 分类分支参数就有接近6M的参数，回归分支12M。其次升维造成了两支参数量的极度不平衡，模版分支是搜索支参数量的`2k / 4k`倍，也造成整体难以优化，训练困难。
  - 改为Depthwise版本以后，参数量能够急剧下降；同时整体训练也更为稳定，整体性能也得到了加强。

---

### Fast Online Object Tracking and Segmentation: A Unifying Approach             (SiamMask, CVPR2019)

**Motivation**

1. VOT2015提出用旋转矩形框作为label，实际上是一种mask的近似
2. SiamFC是用score map来得到物体的位置，SiamRPN是用网络预测bbox长宽比来获得更精确的bbox
3. SiamMask是为了提升Visual Object Segmentation的效率，以及减少给定第一帧mask这种人机交互成本提出的一个统一框架

**Abstract**

- SiamMask在初始只依赖于单个bbox然后能够自己推理出mask和最小外接矩形
- 在Video Object Segmentation数据集上面达到SOTA精度且 55fps 实时效果

**Pipeline**

![Screen Shot 2020-03-25 at 1.43.30 pm](assets/Screen%20Shot%202020-03-25%20at%201.43.30%20pm.png)

这里是用的是SiamRPN++里面的depth-wise correlation

- $f_\theta$

![Screen Shot 2020-03-25 at 1.58.52 pm](assets/Screen%20Shot%202020-03-25%20at%201.58.52%20pm.png)

- **Two Variants**
  1. 基于mask，bbox和score map (Table 9.)
  2. 基于mask和score map (Table 10.)

![Screen Shot 2020-03-25 at 2.03.06 pm](assets/Screen%20Shot%202020-03-25%20at%202.03.06%20pm.png)

- **Mask Refinement Module**

![Screen Shot 2020-03-25 at 2.13.30 pm](assets/Screen%20Shot%202020-03-25%20at%202.13.30%20pm.png)

![Screen Shot 2020-03-25 at 2.13.59 pm](assets/Screen%20Shot%202020-03-25%20at%202.13.59%20pm.png)