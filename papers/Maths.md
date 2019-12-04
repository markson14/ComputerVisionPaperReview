# Math

## Loss

### Information (信息量)

发生概率越小，信息量越大。假设 X 是一个离散的随机变量
$$
p(x) = Pr(X=x), x \in \chi \\ 
\ \\
I(x_0) = -\log(p(x_0))
$$


### Entropy (熵)

信息量的 **期望** 就是熵，
$$
H(X) = -\sum_{i=1}^{n} p(x_i)\log(p(x_i))
$$
其中，二项分布的熵可以简化成，
$$
H(X) = -p(x)\log(p(x))-(1-p(x))\log(1-p(x))
$$


### Cross-Entropy (相对熵/KL散度[Kullback-Leibler Divergence])

衡量两个分布的差异，**P** 代表真实分布，**Q** 代表模型预测分布，两个分布的差异，也就是Loss，
$$
D_{KL}(p||q) = \sum_{i=1}^{n}p(x_i)\log\frac{p(x_i)}{q(x_i)}
$$
从公示看，Q分布越接近P，散度越小，损失越小。

因为对数函数是凸函数，所以KL散度的值为非负值。



### 交叉熵

$$
\begin{equation}
\begin{aligned}
KL散度 \ \ D_{KL}(p||q) &= \sum_{i=1}^{n}p(x_i)\log\frac{p(x_i)}{q(x_i)} \\&=\sum_{i=1}^{n}p(x_i)\log(p(x_i)) - \sum_{i=1}^{n}p(x_i)\log(q(x_i)) \\
&= -H(p(x)) + [- \sum_{i=1}^{n}p(x_i)\log(q(x_i))]
\end{aligned}
\end{equation}
$$

等式的前半部分是P的熵，后半部分就是交叉熵
$$
交叉熵 \ \ H(p,q) = - \sum_{i=1}^{n}p(x_i)\log(q(x_i))
$$
由于在评估label和predict之间的差距时，前半部分不变，故在优化过程中，只关注交叉熵



### JS散度

解决了KL散度非对称问题
$$
JS(p||q) = \frac{1}{2}KL(p||\frac{p+q}{2})+\frac{1}{2}KL(q||\frac{p+q}{2})
$$


### Wasserstein距离

如果P，Q分布完全没有重叠时，KL散度将没有意义，而JS散度值是一个常数($\log2$)。这样在学习算法中，意味着梯度一只是0，无法学习。

**Wasserstein距离**的度量两个概率分布之间的距离，brew
$$
W(P,Q) = \inf_{\gamma \backsim\prod(P,Q)}\mathbb{E}{(x,y)\backsim\gamma}[||x-y||]
$$
$\prod(P,Q)$ 是P，Q联合分布的集合。从中取(x,y)并计算这对样本距离，样本对距离的**期望值**取到的**下界**就是wasserstein距离





## Bilinear Interpolation

### Linear Interpolation

- 已知两点A B，要在AB中插入一点C，即在AB的连线上插入。可套用一次线性公式：

### 双线性

- 已知Q11, Q12, Q21, Q22，要在其中插入一点P。得到P点，需要两次线性插值，即双线性插值

![Screen Shot 2019-02-19 at 11.42.40 AM](./assets/Screen%20Shot%202019-02-19%20at%2011.42.40%20AM.png)

1. 首先求出插值R1,	R2

![Screen Shot 2019-02-19 at 11.39.40 AM](./assets/Screen%20Shot%202019-02-19%20at%2011.39.40%20AM.png)

2. 由插值R1,  R2求出插值P

![Screen Shot 2019-02-19 at 11.39.42 AM](./assets/Screen%20Shot%202019-02-19%20at%2011.39.42%20AM.png)

3. 式子扩展化简即为

![Screen Shot 2019-02-19 at 11.39.46 AM](./assets/Screen%20Shot%202019-02-19%20at%2011.39.46%20AM.png)

4. 若选择一个坐标系使得Q点为(0, 0), (0, 1), (1, 0), (1, 1)， 那么插值公式可以化简为矩阵形式

## KL Divergence

**度量概率分布P与概率分布Q的相似性**:

- PQ的交叉熵 - P的熵

$$
H(Q,P) = E_{x～P}[-\log Q(x)] \\
H(P) = E_{x～P}[-\log P(x)] \\
$$

$$
\begin {eqnarray}D_{KL}(P||Q) 
&=& H(Q,P) - H(P) \\
&=& E_{x～P}[-\log Q(x)] - E_{x～P}[-\log P(x)] \\
&=& E_{x～P}[-\log Q(x) + \log P(x)] \\ 
&=& E_{x～P}[\log \frac{P(x)}{Q(x)}]
\end{eqnarray}
$$

- 离散形式：

$$
D_{KL}(P||Q) = \sum_i P(i)\log \frac{P(x)}{Q(x)}
$$



- 连续形式：

$$
D_{KL}(P||Q) = \int P(i)\log \frac{P(x)}{Q(x)}dx
$$

- 性质：
  - 非负性质
  - 不对称性质
- 应用：
  - VAE - Variational Autoencoder

## Levenberg-Maquardt 算法

### 前置知识

#### 1. 牛顿法

作用：1. 求根 2.求极值

1. 求根

   目标: 求解 $f(y)=0$ 的根

   计算穿过初始点(x_0,f(x_0)) 并且斜率为 f'(x) 的直线与x轴的交点可得 

   ​			
   $$
   0=(x-x_0)f'(x_0)+f(x_0)
   $$
   迭代公式：	
   $$
   x_{n+1}=x_n-\frac{f(x_n)}{f'(x_{n})}
   $$

2. 求解一维无约束最小值

   目标: 求解 $min f(x) , x\in R$ 的根

   牛顿法也可用来求解函数的极值。极值点是导数为0，可用牛顿法求导数的零点。

   $f(x+\Delta)$ 的二阶泰勒展开为 	
   $$
   f(x+\Delta)=f(x)+f'(x)\Delta+\frac{1}{2}f''(x)\Delta^2
   $$

   求解	 
$$
   \frac{\partial f(x+\Delta)}{ \partial \Delta}=0
$$
   可得	
$$
   f'(x)+f''(x)\Delta=0 \\ \Delta=-\frac{f'(x_n)}{f''(x_{n})}
$$
   迭代公式： 
$$
   x_{n+1}=x_n-\frac{f'(x_n)}{f''(x_{n})}
$$

3. 求解高维无约束最小值

   高维情况下泰勒二阶展开为 	
   $$
   f(\mathbf x+\mathbf  \Delta)=f(\mathbf x)+\nabla f(\mathbf x)\Delta+\frac{1}{2}\Delta ^TH(f(\mathbf x))\Delta
   $$
   因此迭代公式：
   $$
   x_{n+1}=x_n-[H(f(x_n))]^{-1}\nabla f(x)
   $$

**优点**

- 牛顿法是二阶收敛，比一般梯度下降法更快收敛，特别是当初始点距离目标足够靠近。

**缺点**

- 应用求极值的时候需要目标函数二次可微，而梯度下降法只需要可微
- 需要 $Hessian$ 矩阵正定，遇到 $f$ 的极值点，或者初始点距离目标远时候可能无法收敛
- 每次迭代需要求 $Hessian$ 的逆矩阵，运算量非常大

#### 2. 高斯-牛顿法

作用：降低牛顿法的计算量，提高计算效率

**最小二乘法问题**

对于向量函数 	${\bf f}:R^m\to R^n,m\ge n$

最小化 $||f(x)||$ 或者找到 	$x^*=argmin_x\{F(x)\}$

这里 	$$F(x)=\frac{1}{2}\sum_{i=1}^m(f_i(x))^2=\frac{1}{2}{\bf f}({\bf x})^T{\bf f}({\bf x})$$

**牛顿法推导**

已知牛顿法迭代公式：	$$x_{n+1}=x_n-[H(f(x_n))]^{-1}\nabla f(x)$$

$F(x)$ 的梯度 		
$$
\nabla F(x)=\begin{bmatrix}\frac{\partial (\frac{1}{2}\sum_{i=1}^m(f_i(x))^2)}{\partial x_1}\\\vdots\\\frac{\partial (\frac{1}{2}\sum_{i=1}^m(f_i(x))^2)}{\partial x_n}\end{bmatrix}=\begin{bmatrix}\sum_{i=1}^{m}f_i\frac{\partial f_i}{\partial x_1}\\\vdots\\\sum_{i=1}^{m}f_i\frac{\partial f_i}{\partial x_n} \end{bmatrix}= \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}  \end{bmatrix}^T\begin{bmatrix}f_1 \\ \vdots \\ f_m  \end{bmatrix}
$$
即 	$$\nabla F(x)=J_f^Tf$$

$Hessian$ 矩阵有 	
$$
H_{jk}=\sum_{i=1}^m(\frac{\partial f_i}{\partial x_j}\frac{\partial f_i}{\partial x_k}+f_i\frac{\partial ^2f_i}{\partial x_j\partial x_k})
$$
忽略二阶导数项有：	$$H_{jk}\approx \sum_{i=1}^mJ_{ij}J_{ik}$$

所以： 	$$H\approx J_f^TJ_f$$

高斯-牛顿迭代公式：	$$x_{n+1}=x_n-[J_f^TJ_f]^{-1}J_f^Tf(x_n)$$		$$s.t. |\frac{\partial f_i}{\partial x_j}\frac{\partial f_i}{\partial x_k}|\gg  |f_i\frac{\partial ^2f_i}{\partial x_j\partial x_k}|$$

 **优点**

- $J_f$满秩，此时二次项 $|f_i\frac{\partial ^2f_i}{\partial x_j\partial x_k}|$ 可以忽略，高斯牛顿和牛顿法都会收敛
- 无需计算 $Hessian$ 矩阵

**缺点**

- 若 $|fi|$ 或  $|f_i\frac{\partial ^2f_i}{\partial x_j\partial x_k}|$ 比较大，会导致难以忽略该二次项，高斯牛顿法的收敛速度会很慢，甚至无法收敛。

### Levenberg-Maquardt 算法

 根据目标函数$F(x)$ 二阶近似得到：	
$$
F(x+h)\approx F( x)+\nabla F(x)h+\frac{1}{2}h^TH_Fh\approx F( x)+J_f^Tfh+\frac{1}{2}h^TJ_f^TJ_fh
$$
我们定义下降方向为 $L(h)$ 
$$
L(h)\equiv F( x)+J_f^Tfh+\frac{1}{2}h^TJ_f^TJ_fh  \\s.t. \ \ h = x_{n+1} - x_n= \Delta
$$
高斯牛顿法迭代公式：
$$
x_{n+1}=x_n-[J_f^TJ_f]^{-1}J_f^Tf(x_n)
$$
**LM迭代公式：** 	$$x_{n+1}=x_n-[J_f^TJ_f+\mu I]^{-1}J_f^Tf(x_n)$$  
​							**or**
​				$$[J_f^TJ_f+\mu I]h=-J_f^Tf(x_n)$$

**作用**：结合了高斯牛顿法与梯度下降法的特点，引入阻尼因子来调节算法特性。

**因子作用**：

- 当 $\mu\gt 0$ 时保证系数矩阵正定，从而确保迭代的下降方向
- 当 $\mu$ 很大时，退化为梯度下降法：  $x_{n+1}=x_n-\frac{1}{\mu}J_f^Tf(x_n)$
- 当 $\mu$ 很小时，退化为高斯牛顿法：  $x_{n+1}=x_n-[J_f^TJ_f]^{-1}J_f^Tf(x_n)$

$\mu$**的计算**

- 初始取值：$\mu$ 的初始值$\mu_0$ 与 $J(x_0)^TJ(x_0)$ 矩阵的元素个数有关：
  $$
  \mu_0=\tau*max_i\{a_{ii}^{(0)}\}
  $$

- 更新： 由系数 $\varrho$ 来控制，这里：
  $$
  \varrho=\frac{F(x)-F(x+h)}{L(0)-L(h)}
  $$



  	分子的目标函数在步长 $h$ 下的实际变化，分母为目标函数二阶近似的变化：
$$
L(0)-L(h)=(F(x))-(F(x)+h^TJ^Tf+\frac{1}{2}h^TJ^TJh)=-h^TJ^Tf-\frac{1}{2}h^TJ^TJh
$$
  可以看出

- $\varrho$ 越大，表明 $L$ 对 $F$ 的效果越好，可以缩小 $\mu$ 以使得 $LM$ 算法接近高斯牛顿法

- $\varrho$ 越小，表明 $L$ 对 $F$ 的效果越差，所以增大 $\mu$ 以使得 $LM$ 算法接近梯度下降法并减少步长 $h$
  
  
  $$
  if\,\varrho>0\,\mu =\mu * max\{\frac{1}{3},1-(2\varrho-1)^3\} \\ else\,\mu =\mu * v;v=2
  $$



![Screen Shot 2019-04-17 at 3.27.43 pm](assets/Screen%20Shot%202019-04-17%20at%203.27.43%20pm.png)

**LM算法流程图**

![Screen Shot 2019-04-17 at 3.27.38 pm](assets/Screen%20Shot%202019-04-17%20at%203.27.38%20pm.png)

**Reference**

- [boksic非线性优化整理-1.牛顿法](http://blog.csdn.net/boksic/article/details/79130509)
- [boksic非线性优化整理-2.高斯牛顿法](https://blog.csdn.net/boksic/article/details/79055298)
- [boksic 非线性优化整理-3.Levenberg-Marquardt法(LM法)](https://blog.csdn.net/boksic/article/details/79177055#)
- [Timmy_Y训练数据常用算法之Levenberg–Marquardt（LM）](https://blog.csdn.net/mingtian715/article/details/53579379)
- [Miroslav Balda 的Methods for non-linear least square problems](http://download.csdn.net/detail/mingtian715/9708842)



---

---

# Bi-Tempered Loss

**Background**

- Real world dataset typically contain some amount of noise that introduces challenges for ML models

**Logistic regression shortcut**

- Outliers far away can dominate the overall loss: 单个bad example (原理decision boundary)对loss有非常大的penalize，导致model会为了妥协而牺牲good examples
- Mislabeled examples nearby can stretch the decision boundary: training process will tend to stretch the boundary closer to a mislabeled example in order to compensate for its small margin. 

**New Loss**

- Implementing Bi-Tempered Loss with 2 parameters t1 & t2

  ![Screen Shot 2019-11-25 at 2.32.45 pm](assets/Screen%20Shot%202019-11-25%20at%202.32.45%20pm.png)

- Dealing with 4 kinds of Cases

  ![Screen Shot 2019-11-25 at 2.39.08 pm](assets/Screen%20Shot%202019-11-25%20at%202.39.08%20pm-4663989.png)
  1. Noise Free Case: 无噪音
  2. Small-Margin Case: 噪音靠近decision boundary
  3. Large-Margin Case: 噪音原理decision boundary
  4. Random Noise: real world noise