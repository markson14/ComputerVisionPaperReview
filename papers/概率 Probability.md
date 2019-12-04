## 概率

#### **随机变量 random variable**

1. 离散随机变量
2. 连续随机变量

#### **联合概率 joint probability distribution**

![Screen Shot 2019-10-08 at 10.38.47 am](assets/Screen%20Shot%202019-10-08%20at%2010.38.47%20am-0502343.png)

#### **边缘化 Marginal Distribution**

- 任意单位变量都可以通过联合概率分布上求其他变量的和(离散变量)或积分(连续变量)得到
  $$
  已知 \ Pr(x,y) \\
  Pr(x) = \int Pr(x,y) dy \\
  Pr(y) = \int Pr(x,y) dx \\
  $$
  
  $$
  已知 w 是离散变量，z是连续变量 \\
  Pr(x,y) = \sum_w \int Pr(w,x,y,z)dz
  $$
  

#### 条件概率 Conditional Probability

- 给定y时x的条件概率 $Pr(x|y)$ ，“|” 可以理解为“给定”

$$
Pr(x|y) = \frac{Pr(x,y)}{\int Pr(x,y)dx} = \frac{Pr(x,y)}{Pr(y)}
$$

$$
Pr(x,y) = Pr(x|y)Pr(y)
$$

#### 贝叶斯公式 Bayesian

$$
Pr(y|x)Pr(x) = Pr(x|y)Pr(y)
$$

整理得到：
$$
\begin{equation}
\begin{aligned}
Pr(y|x) &= \frac{Pr(x|y)Pr(y)}{Pr(x)} \\ 
&= \frac{Pr(x|y)Pr(y)}{\int Pr(x,y)dy} \\ 
&=\frac{Pr(x|y)Pr(y)}{\int Pr(x|y)Pr(y)dy}
\end{aligned}
\end{equation}
$$

- $Pr(y|x)$：后验概率
- $Pr(y)$：先验概率
- $Pr(x|y)$：似然
- $Pr(x)$：证据

#### 独立性 Independency

- 若从变量x不能获得变量y，则x和y是独立的
  $$
  Pr(x|y) = Pr(x) \\
  Pr(y|x) = Pr(y) \\
  $$

$$
Pr(x,y) = Pr(x|y)Pr(y) = Pr(x)Pr(y)
$$

#### 期望 Mean

