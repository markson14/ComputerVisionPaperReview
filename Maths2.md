# Calculus

## 微分

设函数y = f(x)在x的[邻域](http://baike.baidu.com/view/348547.htm)内有定义，x及x + Δx在此[区间](http://baike.baidu.com/view/70334.htm)内。如果函数的增量Δy = f(x + Δx) - f(x)可表示为 Δy = AΔx + o(Δx)（其中A是不依赖于Δx的[常数](http://baike.baidu.com/view/122755.htm)），而o(Δx)是比Δx高阶的[无穷小](http://baike.baidu.com/view/454622.htm)（注：o读作奥密克戎，希腊字母）那么称函数f(x)在点x是[可微](http://baike.baidu.com/view/1962558.htm)的，且AΔx称作函数在点x相应于[自变量](http://baike.baidu.com/view/379564.htm)增量Δx的微分，记作dy，即dy = AΔx。函数的微分是函数增量的主要部分，且是Δx的[线性函数](http://baike.baidu.com/view/2169890.htm)，故说函数的微分是函数增量的[线性主部](http://baike.baidu.com/view/3780520.htm)（△x→0）。

此时当Δx无穷小时，下图的dy是近似等于Δy 的，

![Screen Shot 2020-11-18 at 10.39.22 am](assets/Screen%20Shot%202020-11-18%20at%2010.39.22%20am.png)

解读：当Δx无穷小时，dy是无限近似等于Δy 的，而dy是和dx有一定关系的，dy/dx就是函数在x点处切线的斜率，也就是上边所说的常数A，Δy和dy的差就是o(Δx)，是比Δx高阶的[无穷小](http://baike.baidu.com/view/454622.htm)，就是上图中的MN部分。下面我们来正着推：Δy = dy + o(Δx)=dx*k(k为斜率)+o(Δx)；Δx无穷小时，Δy = dy=dx*k，dy/dx=k,k也该函数的导数。

**可以看出，微分的提出其实就是在近似函数在自变量变化无限小时函数的变化。这是针对连续性函数，那么怎么求离散型函数的微分的。**

## 差分

**差分的目的是去近似离散型函数的导数，导数其实就是变化率**

一阶差分就是[离散函数](http://baike.baidu.com/item/离散函数)中连续相邻两项之差，

定义X(k)，则Y(k)=X(k+1)-X(k)就是此函数的一阶差分，X在k处的变化率就是{X(k+1)-X(k)}/(k+1-k)=X(k+1)-X(k)

同理，Y(k)一阶差分的[二阶差分](http://baike.baidu.com/item/二阶差分)为Z(k)=Y(k+1)-Y(k)=X(k+2)-2*X(k+1)+X(k)，Z(k)为此函数的二阶差分

- 前向差分
- 后向差分
- 中心差分