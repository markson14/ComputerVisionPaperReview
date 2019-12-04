## Model  Quantization (模型量化)

### FP16, FP32, FP64

Sign：0表示正数，1表示负数

exponent：用于存储科学计算中的指数部分，采用移位（127+exponent）的二进制方式

fraction：用于存储尾数部分

#### FP32 & FP64

- 主要用于计算

**FP32** 

Positive取值范围 $[2^{-149},infinity]$

![Screen Shot 2019-08-28 at 5.53.01 pm](assets/Screen%20Shot%202019-08-28%20at%205.53.01%20pm.png)

**FP64**

Positive取值范围 $[2^{-1074},infinity]$

![Screen Shot 2019-08-28 at 5.52.49 pm](assets/Screen%20Shot%202019-08-28%20at%205.52.49%20pm.png)

#### FP16

Positive取值范围 $[2^{-24},infinity]$

- 主要用于存储和通信

![Screen Shot 2019-08-28 at 5.52.58 pm](assets/Screen%20Shot%202019-08-28%20at%205.52.58%20pm-6986085.png)

## Model Deployment (模型部署)

### TVM

![Screen Shot 2019-10-17 at 10.51.53 am](assets/Screen%20Shot%202019-10-17%20at%2010.51.53%20am.png)

#### **TVM模型部署流程：**

1. TF、Pytorch、MXNet等frontend深度学习框架的模型到计算图IR的转换；
2. 对原始计算图IR进行**graph优化**，得到Optimized Computational Graph；
3. 对计算图中的每个op获取用**tensor expression language**描述的Tensor计算表达式，并针对所需要硬件平台，选择primitives生成具体的schedule；
4. 使用某种机遇机器学习的Automated Optimizer生成经过优化的Low Level Loop Program；
5. 生成特定于硬件的二进制程序**(.so)**
6. 生成可以部署的module**(.json/.params)**

#### **Graph优化:**

深度神经网络本质上是一个计算图

![Screen Shot 2019-10-17 at 11.17.58 am](assets/Screen%20Shot%202019-10-17%20at%2011.17.58%20am.png)

早期大家主要在优化operator上(各种conv的实现)，现在发现graph层面可以做的改进非常多，TVM实现了以下几种：

- operator fusion：把多个独立的operator融合成一个；

  - 例如conv-bn-relu，`x1 = conv(x), x2 = bn(x1), y = relu(x2)`计算流程是3步分开的，进行3次函数调用，储存中间两次结果 `x1, x2`；op fusion后，过程会变成`y = conv_bn_relu(x)`
  - Operator分类：
    - injective (one-to-one map, e.g., add)
    - reduction (e.g., sum)
    - complex-out-fusable (can fuse element-wise map to output, e.g., conv2d)
    - opaque (cannot be fused, e.g., sort)
  - 原则：
    - 任意多个(连续)injective op可以被合并成一个op；
    - injective op如果作为一个reduction op输入，则可以被合并；
    - 如果complex-out op (例如conv2d) 的输出接的是element-wise，则可以被合并；

  ![Screen Shot 2019-10-17 at 11.34.51 am](assets/Screen%20Shot%202019-10-17%20at%2011.34.51%20am.png)

- constant-folding：把一些可以静态计算出来的常量提前计算；
- static memory planning pass：预先把需要的存储空间申请下来，避免动态分配；
- data layout transformations：有些特殊的计算设备可能会对不同的data layout (i.e. NCHW, NHWC, HWCN)有不同的性能，TVM可以根据实际需求在graph层面就做好这些转换。
  

#### **Auto-Tuning:**

TVM为了解决高效计算的问题，提出了利用机器学习来进行自动优化的方案：

- Schedule explorer：用于不断搜寻新的schedule配置；

  - 定义搜索空间

- Cost Model：用于预测每一种schedule配置的计算耗时；

  - black box optimizer，尽量少遍历各种schedule方案的情况下找到最优点

  ![Screen Shot 2019-10-17 at 11.52.53 am](assets/Screen%20Shot%202019-10-17%20at%2011.52.53%20am.png)

### TensorRT

- **简介**

   NVIDIA推出的基于CUDA和cudnn的神经网络推理加速引擎，相比于一般的深度学习框架，能有10X至100X的加速。

- 加速原理

  - 支持INT8和FP16的Inference，在减少计算量和保持精度之间达到一个trade-off

  - 对网络进行了重构和优化

    ![Screen Shot 2019-10-25 at 2.20.35 pm](assets/Screen%20Shot%202019-10-25%20at%202.20.35%20pm.png)

    (1) 通过解析网络模型将网络中的无用层消除以减少计算

    (2) 网络垂直整合，将主流的神经网络conv，BN，Relu三个层融合为一个(类似TVM合并operator)

    ![Screen Shot 2019-10-25 at 2.20.41 pm](assets/Screen%20Shot%202019-10-25%20at%202.20.41%20pm.png)

    (3) 网络水平整合，即指将输入的为相同张量和执行相同操作的层融合

    ![Screen Shot 2019-10-25 at 2.20.46 pm](assets/Screen%20Shot%202019-10-25%20at%202.20.46%20pm.png)

    (4) 对于concat层，将concat层的输入直接送入下面操作中，不单独进行concat后的输入再计算，相当于减少一次IO

## Model Convert (模型转换)

### ONNX (Open Neural Network Exchange)

- 通用模型转换的中间件。提供定义可延展的计算图模型，定义操作符和标准数据格式(dtypes)
- 每一个数据流计算图中定义了一系列的nodes来组成DAG。Node拥有多个input多个output。每个Node都可以称为一个操作符。

### MMDNN (model converter)

#### 1. From MXNet to TensorFlowLite

1. **convert the model to intermediate representation format.**
   - `python3 -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath resnet100.pb --IRWeightPath resnet100.npy --dstModelPath tf_resnet100.py`
2. **convert to tensorflow code**
   - `mmtocode -f tensorflow --IRModelPath resnet100.pb --IRWeightPath resnet100.npy --dstModelPath tf_resnet100.py`
3. **convert to tensorflow file**
   - `mmtomodel -f tensorflow -in tf_resnet100.py -iw resnet100.npy -o tf_resnet1001 --dump_tag SERVING`
4. **convert to tensorflowlite file**

Problems：

- set `allow_picle=True` can fix issue.

