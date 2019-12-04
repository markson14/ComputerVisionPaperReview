## Bag of tricks for Image Classification with CNN

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