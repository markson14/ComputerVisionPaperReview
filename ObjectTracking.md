## Object Tracking

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

### Fully-Convolutional Siamese Networks for Object Tracking （SiamsFC）

**Abstract**

- 不对称物体检测通常使用在线学习的模式，约束了模型的泛化能力
- 提出端到端网络，使用全卷积层，实时侦测的SiamsFC网络

**Highlight** ：

- High performance, Real-time
- Leanring strong embeddings in offline phase

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

在输入搜索图上，只要和目标半径距离不超过R，就算正样本，否则为负样本

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

### SiamRPN

### SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks (CVPR2019)

### Fast Online Object Tracking and Segmentation: A Unifying Approach (CVPR2019)