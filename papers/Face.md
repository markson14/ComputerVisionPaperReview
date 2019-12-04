## Face Detection

### Approaching Human Level Facial Landmark Localization by Deep Learning

##### **难点**

1. 遮挡
2. 低分辨率
3. 光线
4. 模糊

##### **方法对比**

- 常用landmark方法：将landmark看成是一个回归问题 (regression-based)
- 本文方法：CNN Cascade (deeplearning-based)

##### **Coarse-to-fine**

1. 侦测人脸，粗略初始化landmark
2. 根据landmark进一步观测周遭area，refine landmark

![Screen Shot 2019-03-04 at 5.28.59 PM](assets/Screen%20Shot%202019-03-04%20at%205.28.59%20PM.png)



------

### FacePoseNet：Making a Case for Landmark-free Face Alignment

##### **Introduction**

- Facial landmark主要用作pipeline里面对齐，方便后续操作。但是更好的landmark并不意味更好的face alignment

##### **Critique of facial landmark detection**

- Landmark detection accracy measures

  - inter-ocular distance, $L$= { $p_i$ } 属于set of m 2D facial landmark坐标，$L$ = {$\hat{p_i}$ 属于ground truth坐标，$\hat{p_l}, \hat{p_r}$ 为左右眼outer corner position。

    $$e(L, \hat{L}) = \frac{1}{m||\hat{p_l} - \hat{p_r}||_2} \sum_{i=1}^m ||p_i - \hat{p_i}||_2$$

- Landmark detection speed

  - 实时landmark detection准确率相对低
  - 目前没有landmark detector是基于GPU的，如今串行的优化方式难以convert to并行。

- Effects of facial expression and shape on alignment



------

### Deep Convolitional Network Cascade for Facial Point Detection (CVPR 2013)

##### **Cascaded Convolutional networks**

![Screen Shot 2019-03-11 at 5.08.34 PM](assets/Screen%20Shot%202019-03-11%20at%205.08.34%20PM.png)

- Level 1: 3个CNN网络级联

  - F1: 回归5个点（左眼，右眼，鼻子，左嘴角，右嘴角）

  ![Screen Shot 2019-03-11 at 5.17.58 PM](assets/Screen%20Shot%202019-03-11%20at%205.17.58%20PM.png)

  - EN1: 回归3个点（左眼，右眼，鼻子）**same structure with different size**
  - NM1: 回归3个点（左嘴角，右嘴角，鼻子）**same structure with different size**
  - output：将上面得到的11个点回归成5个点

- Level 2: 10个CNN网络级联

  - 输入位level1输出关键点附近(15x15)局部裁剪的图像，每两个CNN负责回归上一级输出的一个关键点

- Level 3：10个CNN网络级联

  - 输入为level2输出关键点附近的裁剪图像，每两个CNN负责回归上一级输出的一个关键点

##### **Locally-share Conv**

- 全局权值共享（Fully Connected Layer）
  - 假设object出现在图片的任意地方
  - 防止bp的时候出现梯度消失，因为权值的梯度是aggregrated的。
  - 对具有fixed spatial layout（固定空间分布）的图片难以学习
- 局部权值共享（Conv）
  - 对于分割后的图片，卷积层更好的学习high level feature



---

### CFAN (ECCV 2014)

##### **introduction**

- AAM无法照顾到复杂的人脸外观，归咎于single linear model 无法cover 所有non-linear variations in facial appearance



---

### MTCNN (ECCV 2016)

##### **Inference Pipeline**

- Stage 1: Proposal Network 获得候选框和回归向量，之后再用**NMS**对多余候选框进行删减
  - Input：原图resize至 `12 x 12` 的过程中得到的图片金字塔，e.g. `原图，原图*factor，原图*factor^2`
  - Ongoing: 将图片stack输入Pnet，得到输出形状 `(m, n, 16)` ；根据得分筛选掉一大部分候选，再根据4个偏移量对bbox进行校准后得到bbox左上右下的坐标；根据IOU再进行NMS筛选，得到 `(nms_left, 16)` 的tensor。
  - Output：Face Classification`(nms_left, 2)`，Bounding box regression`(nms_left, 4)`，Facial landmark location`(nms_left, 10)`
- Stage 2: Refine Network 候选框feed in CNN，进一步拒绝false candidates，校正候选框回归，进行**NMS**
  - Input：以**Pnet bbox** 作为基础，将原图沿bbox截取再resize至 `24 x 24`，无需`image pyramid`
  - Ongoing: 重复PNet的操作，唯一不同的只有输入的size固定为`24 x 24`
  - Output：Face Classification`(nms_left, 2)`，Bounding box regression`(nms_left, 4)`，Facial landmark location`(nms_left, 10)`
- Stage 3: Output Network 定位5个人脸关键点
  - Input：以**Rnet**信息作为基础，将原图沿bbox截取再resize至 `48 x 48`
  - Ongoing: 重复PNet的操作，唯一不同的只有输入的size固定为`48 x 48`
  - Output：Face Classification`(nms_left, 2)`，Bounding box regression`(nms_left, 4)`，Facial landmark location`(nms_left, 10)`

![Screen Shot 2019-03-12 at 2.25.00 PM](assets/Screen%20Shot%202019-03-12%20at%202.25.00%20PM.png)

##### **Training**

- 通过随机生成候选框将数据集分成四类：positive(IOU > 0.65)，negative（IOU < 0.3），partfaces（0.4<IOU<0.65），landmark
- positive和negative参与到分类任务，positive和partface参与回归任务，landmark参与关键点回归任务
  - Cross Entropy Loss for facial classification
  - L2 Loss for bounding box regression
  - L2 Loss for landmark localization
  - Multi-source training：$ min \sum^N_{i=1} \sum_{j\in (det,box,landmark) \alpha_j \beta^j_i L^j_i } $ 

![Screen Shot 2019-03-12 at 3.32.26 PM](assets/Screen%20Shot%202019-03-12%20at%203.32.26%20PM.png)

##### 偏移量offset标记

- 截图在原图绝对坐标：`(x1, y1),  (x2,y2)`
- 原图上label：`(xlu1, ylu1),  (xrd2, yrd2)`
- Offset: `[(x1-xlu1)/x2-x1, (y1-ylu1)/y2-y1], [(x2-xrd2)/x2-x1, (y2-yrd2)/y2-y1]` 归一化之后的offset
  - 能够更好的收敛，抗resize

##### OHEM(Online hard example mining)

- 在每一个mini-batch中，对forward propagation的loss排序（高到低），选择前70%的用作反向传播。这样的作用是忽略easy sample对于模型的训练作用。

**缺点**

- 由于通过threshold筛选proposal，所以当人脸多了之后，很多时间会消耗在PNet上面

---

### Single Shot Headless Detector (SSH)

##### **模型特点**

- 在前向网络传播中**同时**处理多尺度faces
- 使用**skip connections**从early conv layers提取尺度偏小的face (目前常用做法)
- 将所有尺度的proposal分配给3个Module：`m1 ; m2 ; m3` 每个module负责处理相对应range size的proposal
- Low memory foot-print；quick inference time
- Context module：通过增大感受野达到增加上下文信息目的，为了减少conv参数，感受野的增大转为使用叠层conv的思路，figure 4说明

##### **General Architecture**

![Screen Shot 2019-07-08 at 5.38.30 pm](./assets/Screen%20Shot%202019-07-08%20at%205.38.30%20pm.png)

![Screen Shot 2019-07-08 at 5.50.37 pm](./assets/Screen%20Shot%202019-07-08%20at%205.50.37%20pm.png)

##### **Loss**

- $L_c$ = face classification loss — standard multinomial logistic loss
- $K$ 代表module，$A_k$ 代表anchors in module

$$
\sum \frac{1}{N_k^c} \sum_{i \in A_k} l_c(p_i,g_i) +\lambda\sum \frac{1}{N_k^r} \sum_{i \in A_k} I(g_i =1)l_r(b_i,t_i)
$$



---

### RetinaFace（CVPR 2019）

##### 背景

- Face localisation 的精度和稳定性相对较差。

##### 模型特点

- Single Stage
- 增加self-supervised分支来做像素级别3D预测
- 在CPU上达到**VGA(640 x 480)**级别实时侦测

##### Multitasking Loss

$$
L= L_{cls}(p_i,\hat p_i) + \lambda_1 \hat p_iL_{box}(t_i,\hat t_i) + \lambda_2 \hat p_iL_{pts}(l_i,\hat l_i)+\lambda_3 \hat p_iL_{pixel} \\ where \ \ \{ \lambda_1 = 0.25, \lambda_2=0.1,\lambda_3 = 0.01 \}
$$

1. $L_{cls}$ 是人脸分类softmax loss for binary classification (face/not face)，pi是predicted人脸的概率(模型输出)，$\hat p_i$ 是GT
2. $L_{box}$ 是人脸bbox的Smooth-L1 Loss，$t_i = \{ t_x,t_y,t_w,t_h\}$ 代表预测box和GT box。坐标均使用相对坐标
3. $L_{pts}$ 是landmark regression。与2类似
4. $L_{pixel}$ 是Dense regression loss

##### Dense Regression Branch (self-supervised)

- 将2D的人脸映射到3D模型上，再将3D模型解码为2D图片(Mesh Decoder)，然后计算经过编解码的图片和原始图片的差别
  - 普通卷积参数：$Kernel_H \times Kernel_w \times Channel_{in} \times Channel_{out}$
  - GCN：$K \times Channels_{in} \times Channels_{out}$

- `neighbour distance is calculated on the graph by counting the minimum number of edges connecting two vertices. `

- 3D解码至2D通过$face\R(D_{PST} , P_{cam}, P_{ill})$ , 分别是色彩范围，相机参数和灯光参数 

##### Implement Details

- **FPN: **from P2~P6，P2~P5是从Backbone output的feature map通过FPN计算得到，P6是从C5通过3x3 conv, stride=2，计算得到的

  ![Screen Shot 2019-05-22 at 4.10.58 pm](./assets/Screen%20Shot%202019-05-22%20at%204.10.58%20pm-8512753.png)

- **Context Module (语义信息采集): **Inspired by SSH，在FPN中使用deformable conv代替conv

- **Loss Head: **对于negative anchors，只计算 $L_{cls}$ ，postive anchors参与全部计算

- **Anchor Settings: **P2用来capture小脸。anchor大小涵盖`16x16 ~ 406x406`(default input: 640x480)。OHEM在训练中会使用，保证正负样本比例接近 1:3

  ![Screen Shot 2019-10-31 at 6.26.00 pm](assets/Screen%20Shot%202019-10-31%20at%206.26.00%20pm.png)

- **Data Augmentation: **使用随机裁剪来保证小脸的检测(根据短边 [0.3, 1])，同时使用random horizontal flip 和 色彩扭曲(colour distortion)

##### Ablation Study

- **WIDER FACE face detection challenge**

  ![Screen Shot 2019-11-01 at 9.55.27 am](assets/Screen%20Shot%202019-11-01%20at%209.55.27%20am.png)

---

##### Inference Efficiency

![Screen Shot 2019-11-01 at 10.16.20 am](assets/Screen%20Shot%202019-11-01%20at%2010.16.20%20am.png)



---

### RefineFace: Refinement Neural Network for High Performance Face Detection 

##### 背景



---

---

## Face Recognition

### MobileFaceNet

**Contribution**

1. Use global depth wise convolution layer rather than a global average pooling layer or a fc layer to output a discriminative feature vector
2. A class of face feature embedding CNNs

**Weakness of Common Mobile Networks for Face Verification**

- 全局平均池化对FMap-end(output feature map of the last conv)采用平等权重。在实际中，脸部更偏向中心，边角权重可以弱化
- Flatten FMap-end后的维度太高，所以采用了GAP。GAP**在人脸识别项目中**又会导致infer的准确率下降 (原因可能是人脸识别项目更加喜欢model overfiting，因为输入的图片非常干净规范，希望能够过拟合人和对应的标签)

**Global Depthwise Convolution**

- Replace it with a global depth wise conv(GDConv) + 1x1 Conv for channel reducing

![Screen Shot 2019-12-03 at 4.59.12 pm](assets/Screen%20Shot%202019-12-03%20at%204.59.12%20pm.png)

### ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR2019)

##### 背景

- **Center Loss** penalise deep feature 与相对应class中心，获得较好的类内紧凑性

- **SphereFace** 假设最后一层FC feature能够用来在球面空间做表征学习，于是penalise对deep feature与其对应权重的角度进行多种penalise

- Four kinds of Loss to enhance intra-class compactness and inter-class discrepancy

  - X是random sample，W是classes center
  
  ![Screen Shot 2019-11-01 at 3.15.11 pm](assets/Screen%20Shot%202019-11-01%20at%203.15.11%20pm.png)

##### Loss Comparison

- **Softmax**: (1) size of matrix 随着class数量线性增加 (2) 在人脸检测效果不行
  $$
  Softmax = -\ln \frac{e^{W^T_ix}}{\sum_{j=1}^n e^{W^T_jx}}
  $$
  
- **Triplet**：(1) 需要更多iteration的迭代  (2) semi-hard sample mining 效率不高

- **SphereFace**：训练unstable，softmax损失决定了训练过程，因为基于整数的多角度余量使目标logit曲线非常陡峭，从而阻碍了收敛。
- **CosineFace**：直接将余弦余量损失加到目标logit上，与SphereFace相比，可以获得更好的性能，但允许更轻松地实现，并且减轻了softmax损失的联合监管的需求。

- **ArcFace**

  ![Screen Shot 2019-11-01 at 3.38.02 pm](assets/Screen%20Shot%202019-11-01%20at%203.38.02%20pm.png)

![Screen Shot 2019-11-01 at 4.33.13 pm](assets/Screen%20Shot%202019-11-01%20at%204.33.13%20pm.png)

