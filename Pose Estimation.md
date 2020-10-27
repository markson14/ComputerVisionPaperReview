# Pose Estimation

#### 2D

- 【Accurate】top-down: 先找出所有人，对单个人进行姿态估计。直接对关键点回归；每个关键点预测一个heatmap，对于gt，以关键点为中心的二位高斯分布

- - CPM
  - Hourglass
  - CPN
  - Simple Baselines
  - HRNet
  - MSPN

- 【Fast】bottom-up：找出所有关键点，对关键点进行分组

- - openpose
  - Hourglass+Associative Embedding

#### 3D

#### Metric

- PCK - Percentage of Correct Keypoints
- PDJ - Percentage of Detected Joints

$$
PDJ=\frac{\sum_{i=1}^nbool(d_i < 0.05 * diagonal)}{n}
$$



- OKS - Object Keypoint Similarity

$$
OKS = \exp(-\frac{d_i^2}{2s^2k^2_i})
$$



## Convolutional Pose Machine