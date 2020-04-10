## Optical Character Recognition

### Scene Text Detection and Recognition: The Deep Learning Era

<img src="assets/Screen%20Shot%202020-03-26%20at%2011.19.29%20am-5192849.png" alt="Screen Shot 2020-03-26 at 11.19.29 am" style="zoom:200%;" />

**Abstract**

1. 提出新的想法
2. 分析最新技术和benchmark
3. 展望未来

**Introduction**

- Main Difficulty
  1. 自然场景的多变性
  2. 背景的复杂性和干扰
  3. 有缺陷的成像条件
- 解决方案
  1. 深度学习
  2. 针对性的数据集和算法
  3. 先进的辅助技术

**传统方法**

- 文字检测：CCA(connected components analysis)，滑动窗口，手工设计特征，条件岁机场等
- 文字识别：基于特征字符片段算法，利用label embedding来匹配字符串和图像

**深度学习**

![Screen Shot 2020-03-26 at 11.19.07 am](assets/Screen%20Shot%202020-03-26%20at%2011.19.07%20am.png)

1. 文本检测系统
   1. 流程简化
      - 上图a，b是多阶段方法代表；c，d是简化流程。c只包含检测分支，因此与另一个单独的识别模型一起使用。d是同事训练一个检测和识别模型
      - ![Screen Shot 2020-03-26 at 11.51.01 am](assets/Screen%20Shot%202020-03-26%20at%2011.51.01%20am.png)
      - ![Screen Shot 2020-03-26 at 11.56.41 am](assets/Screen%20Shot%202020-03-26%20at%2011.56.41%20am-5195043.png)
2. 文本识别系统

<img src="assets/Screen%20Shot%202020-03-26%20at%2012.01.25%20pm.png" alt="Screen Shot 2020-03-26 at 12.01.25 pm" style="zoom:150%;" />

- 传统：图像预处理 —> 字符分割 —> 字符识别
- a：RNN + CNN
- b：FCN
- c：attention-based
- d：添加有监督模块在attention模块
- e：使用attention提升alignment问题

3. 端到端系统

![Screen Shot 2020-03-26 at 12.01.11 pm](assets/Screen%20Shot%202020-03-26%20at%2012.01.11%20pm.png)



4. 辅助方法

### An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

**Abstract**

1. End-to-end trainable (端到端可学习)
2. Naturally handle sequences in arbitrary lengths, no character segmentation or horizontal scale normalisation (无需切割，可以处理自然场景不对称长度的字符串)
3. It is not confined to any lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (没有基于任何词典，可以超越基于词典的模型)
4. Effective yet much smaller model (模型非常小且非常有效)

### Connectionist Temporal Classification(CTC)

**Abstract**

- Ordinary sequence learners require pre-segmented training data and post-processing to transform their outputs into label sequences, applicability is limited(输入需要裁剪，和后处理才能得到序列标签，用途有限)
- Novel method for training RNNs to label unsegmented sequences directly. (可以直接训练无需切割输入的RNN序列)

****