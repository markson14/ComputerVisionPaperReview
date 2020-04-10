## Anchor Free

### CornerNet: Detecting Objects as Paired Key-points (ECCV 2018)

**Abstract**

- 提出使用左上，右下角点作为一对关键点确定bbox的网络
- 解决了需要设计anchor boxes的麻烦
- 提出corner pooling帮助网络获取更好的角点，获得42.2% AP 在MS COCO上

![Screen Shot 2020-03-31 at 2.20.22 pm](assets/Screen Shot 2020-03-31 at 2.20.22 pm.png)

**Network**

![Screen Shot 2020-03-31 at 2.25.04 pm](assets/Screen%20Shot%202020-03-31%20at%202.25.04%20pm.png)

- **Grouping Coners**
  - Embeddings的距离，距离近的一组corner为作用一同一个instance

- **Corner Pooling & Prediction Module** 

![Screen Shot 2020-03-31 at 2.27.00 pm](assets/Screen%20Shot%202020-03-31%20at%202.27.00%20pm.png)

---

### Feature Selective Anchor-Free Module for Single-Shot Object Detection 					(FSAF, CVPR 2019)

:star:**Structure**:

![Screen Shot 2019-03-26 at 7.18.46 PM](./assets/Screen%20Shot%202019-03-26%20at%207.18.46%20PM.png)

- 内含anchor-based和anchor-free模块（A为anchor个数，K为class num）

监督训练信号（supervision signal）:

<img src="./assets/Screen%20Shot%202019-03-26%20at%207.21.50%20PM.png" alt="Screen Shot 2019-03-26 at 7.21.50 PM" style="zoom:50%;" />

1. ground truth box：k
2. ground truth box 坐标:$b = [x,y,w,h]$
3. ground truth box 在第$l$层上的投影:$b_p^l=[x_p^l,y_p^l,w_p^l,h_p^l]$
4. effective box：$b_e^l=[x_e^l,y_e^l,w_e^l,h_e^l]$，他表示$b_p^l$的一部分，缩放系数比例 $\epsilon_e = 0.2$
5. ignoring box：$b_i^l=[x_i^l,y_i^l,w_i^l,h_i^l]$ ，他表示$b_p^l$的一部分，缩放系数比例 $\epsilon_i = 0.5$

- Classification Output

  - effective box 表示positive区域，如图白色部分所示。$b_i^l - b_e^l$ 这个部分ignoring区域不参与分类任务，如图灰色部分所示。剩余黑色部分为negative。分类任务是对每个像素做分类，考虑到正负样本不均衡，采用Focal Loss

- Box Regression Output

  - 输出4个offset map。这里取的是像素（i，j）与$b_p^l$ 四个边框的距离。gt bbox 只影响了$b_e^l$区域，所以这里（i，j）是该区域的所有像素。回归分支采用的是IOU Loss

    ![Screen Shot 2019-03-26 at 7.16.00 PM](./assets/Screen%20Shot%202019-03-26%20at%207.16.00%20PM.png)

- Online Feature Selection

  ![Screen Shot 2019-05-25 at 10.23.05 am](assets/Screen%20Shot%202019-05-25%20at%2010.23.05%20am.png)

  1. 实例输入到金字塔所有层，求得IOU loss和focal loss的和
  2. 选取loss和最小的层来学习实例得到feature map
  3. 训练时，特征根据安排的实例进行更新。
  4. 推理时，不需要进行特征更新，因为最适合学习的金字塔层自然输出最高置信分数。

---

### FCOS: Fully Convolutional One-Stage Object Detection

**Abstract**

- Anchor Free和Proposal Free
- 逐点像素回归预测，多尺度特征和centerness

**Network**

![Screen Shot 2020-03-31 at 2.50.49 pm](assets/Screen%20Shot%202020-03-31%20at%202.50.49%20pm.png)

- Output:

  - 80 channel tensor of classification labels on MS COCO
  - 4 channel tensor of `(l, t, r, b)` bboxes coordinates
  - Center-ness: $centerness^*=\sqrt{\frac{min(l^*,r^*)}{max(l^*,r^*)}\times\frac{min(t^*,b^*)}{max(t^*,b^*)}}$ 

  ![Screen Shot 2020-03-31 at 3.06.56pm](assets/Screen%20Shot%202020-03-31%20at%203.06.56%20pm-5638447.png)

---

### <span id="keyp">Object as Points (CenterNet) </span>

**特点:**

1. anchor 仅基于位置，而不是IOU

   ![youz](assets/Screen%20Shot%202019-06-13%20at%201.57.36%20pm.png)

2. 无需进行NMS操作，每个object只有一个positive anchor (from local peak in the keypoints heatmap)

3. 输出的是大分辨率的图像来与原图做对比 (stride = 4)

**Network**

- ground truth使用高斯核将关键点分布到特征图上供给模型学习

- Backbone选择Hourglass，有助于预测heatmap

- 3个header分别是类别，长宽，中心偏置。其中每个像素点预测`(1, 80, 128, 128), (1, 2, 128, 128), (1, 2, 128, 128)`
- 采用focal loss训练类别，L1损失训练回归
- $L_{det} = L_k + \lambda_{size}(0.1)L_{size} + \lambda_{off}(1)L_{off}$

**Inference**

1. 针对一张图像做下采样后进行预测，对每个类的下采样特征图预测中心点
2. 热点提取通过检测当前热点的八邻近点都小于中心点，采集100个热点（使用nn.MaxPool(3x3)达到目的）假设$\hat{P}_c$ 为检测到点， $\hat{P}_c = (\hat{x}_i, \hat{y}_i)^n_{i=1}$
3. 随后，标定框为 $(\hat{x}_i+\delta\hat{x}_i-\hat{w}_i/2, \hat{y}_i+\delta\hat{y}_i-\hat{h}_i/2, \hat{x}_i+\delta\hat{x}_i+\hat{w}_i/2, \hat{y}_i+\delta\hat{y}_i+\hat{h}_i/2)$
4. 最后根据模型预测的中心点存在物体的概率值（heatmap分支），设置阈值0.3，将上面100个热点结果中高于阈值的作为最终结果。

**Result:**

![Screen Shot 2019-10-28 at 5.28.17 pm](assets/Screen%20Shot%202019-10-28%20at%205.28.17%20pm.png)

![Screen Shot 2019-08-08 at 6.14.03 pm](assets/Screen%20Shot%202019-08-08%20at%206.14.03%20pm.png)

---

### Region Proposal by Guided Anchoring

**Abstract**

- 提出使用图像语义特征来知道anchor的生成
- 性能和速度远超传统RPN，可以直接替换

**Background**

- Two Stage：anchor —> proposal —> detection bbox
- One Stage: anchor —> detection bbox

**Motivation**

- 尺度和长宽比需要预设，对性能影响大。也无法满足长宽比特殊的物体
- 过多的anchors影响性能和速度
- 大部分anchor分布在背景区域，对proposal或检测没有任何正面影响
- 希望制作稀疏，形状根据位置可变的anchor

**Formulation**

- $P(x,y,w,h | I) = p(x,y|I)p(w,h|x,y,I)$
- 从上式可以看出anchor概率分布可以分解为两个条件概率分布，中心点概率分布和形状概率分布
- Sliding Window(RPN)可以看作均匀分布的$p(x,y|I)$和冲击函数$p(w,h|x,y,I)$

![Screen Shot 2020-03-31 at 3.44.35 pm](assets/Screen%20Shot%202020-03-31%20at%203.44.35%20pm.png)

1. **位置预测**
- 这里并不是预测每个点是前后景，而是预测是否中心点
  
- 设计时距离中心一定范围对应正样本，这样可以筛选出一小部分区域作为anchor的候选中心点，使得anchor数量大大降低。
  
2. **形状预测**

   - 使用sample的9组w和h去做与匹配得到和gt IoU最大的anchor，这里发现sample超参并不敏感

3. **Feature Adaption**

   - anchor形状信息融入feature map中输出新的特征图，利用3x3 deformable conv来修正原始特征图。offset通过anchor的w和h经过一个1x1 conv得到的。

   - 使feature的有效范围和anchor的形状更加接近



---

### CornerNet-Lite: Efficient Key-points Based Object Detection 

**Abstract**

- CornerNet-Saccade提升AP且加快速度

**Saccade for detection**

- 以顺序或者并行的方式有选择的裁剪和处理图片patch，能有效提升inference速度
- 扫视在R-CNN，Fast R-CNN以及Faster R-CNN的体现是selective search和RPN。在处理后，每个crop image patch会被带上label box继续做分类或者回归或者丢弃。
- Cascade R-CNN在Faster R-CNN的基础上增加了分类起和回归期进一步的筛选增强region proposal
- saccade在R-CNN的变体中处理的都是单类型单目标。然而CornerNet-Saccade中，是单类型多目标。因此，通过CornerNet-Saccade产生的crop数量会比目标数量少很多

**Network**

![Screen Shot 2020-04-02 at 5.53.44 pm](assets/Screen%20Shot%202020-04-02%20at%205.53.44%20pm.png)

1. 在downsized full image通过attention map生成物体的bbox
2. zoom in和crop bbox附近的区域成image patch，然后在image patch上做detect
3. 利用top-k ranking控制生成image patch数量还有NMS避免image patch生成overlap冗余的patch

![Screen Shot 2020-04-02 at 6.00.00 pm](assets/Screen%20Shot%202020-04-02%20at%206.00.00%20pm.png)

**Image Patch处理**

![Screen Shot 2020-04-02 at 5.59.56 pm](assets/Screen%20Shot%202020-04-02%20at%205.59.56%20pm.png)

---

---

## Rotation Detector

### SCRDet: Towards More Robust Detection for Small, Cluttered and Rotated Objects （ICCV 2019）

**Abstract**

- 小目标，任意方向，分布密集的目标仍是挑战
- 使用多层融合feature来提高对小目标的感知
- 有监督pixel attention和channel attention能够抑制噪声和更好地感知小而密集的目标
- 使用添加了IoU参数的smooth L1来解决rotation bbox问题

![Screen Shot 2020-04-10 at 9.06.56 am](assets/Screen%20Shot%202020-04-10%20at%209.06.56%20am.png)

**SF-Net**

![Screen Shot 2020-04-10 at 9.33.41 am](assets/Screen%20Shot%202020-04-10%20at%209.33.41%20am.png)

- Inception Module用于增加感受野和语义信息

**Multi-Dimensional Attention Network**

![Screen Shot 2020-04-10 at 9.33.35 am](assets/Screen%20Shot%202020-04-10%20at%209.33.35%20am.png)

- 有监督注意力机制能够指导网络更好的学习
- Pixel Attention: 使用saliency map+softmax作为像素注意力机制，排除噪声。双通道saliency map代表前景和背景的概率
- Channel Attention：使用SE-module类似的方法，作为通道注意力机制

![Screen Shot 2020-04-10 at 9.47.41 am](assets/Screen%20Shot%202020-04-10%20at%209.47.41%20am.png)

**Rotation-Branch**

![Screen Shot 2020-04-10 at 10.58.59 am](assets/Screen%20Shot%202020-04-10%20at%2010.58.59%20am.png)

- Angle: $[\pi/2, 0)$，对齐OpenCV，一边是x轴，另一边是w
- rotation-NMS用来做后处理

---

### R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection		 (CVPR 2017)

**Abstract**

- Axis-aligned bboxes

![Screen Shot 2020-04-10 at 1.25.53 pm](assets/Screen%20Shot%202020-04-10%20at%201.25.53%20pm.png)

---

### Arbitrary-Oriented Scene Text Detection via Rotation Proposals (CVPR 2019)

**Abstract**

- 提出了RRPN和RROI概念来处理旋转矩形框问题

![Screen Shot 2020-04-10 at 2.30.02 pm](assets/Screen%20Shot%202020-04-10%20at%202.30.02%20pm.png)

**Rotation RPN**

![Screen Shot 2020-04-10 at 2.30.08 pm](assets/Screen%20Shot%202020-04-10%20at%202.30.08%20pm.png)

- ground truth：$(x,y,w,h,\theta)$，$x,y$是中心，$w$是长边，$h$是短边，$\theta$为长边旋转角$[\frac{\pi}{4}, \frac{3\pi}{4})$
- 一共3x3x6=54个anchor

**Rotation RoI Pooling**

- 仿射变换之后再做RoI Pooling