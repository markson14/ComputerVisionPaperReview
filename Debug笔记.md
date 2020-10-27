# Debug笔记

### **Janus-gateway**

- Docker for Mac 配置：`docker_ip` 设置成docker subnet ip

- `docker-compose.yml`文件

  ```
  version: '3'
     2   │ 
     3   │ services:
     4   │   janus-gateway:
     5   │     image: xiaolin/janus-gateway
     6 ~ │     # network_mode: host
     7 ~ │     ports:
     8 ~ │         - "80:80"
     9 ~ │         - "7088:7088"
    10 ~ │         - "8088:8088"
    11 ~ │         - "8188:8188"
    12 ~ │         - "8089:8089"
    13 ~ │         - "10000-10200:10000-10200/udp"
    14   │     environment:
    15   │       - DOCKER_IP=${DOCKER_IP}
    16   │     volumes:
    17   │       - ./conf/:/opt/janus/etc/janus/
    18   │       - ./nginx/:/etc/nginx/
    19   │       - ./janus-gateway-html:/root/janus-gateway/html
  ```

### 服务器conda环境搭建：

```bash
 https://repo.continuum.io/archive/Anaconda3-2018.12-Linux-x86_64.sh
bash Anaconda3-2018.12-Linux-x86_64.sh
export PATH="/home/zhangziwei/anaconda3/bin:$PATH"

#ignore channel
conda config --append channels conda-forge

#国内源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls ye
```

- `tar`打包本地conda下envs再用`scp` 上传到服务器
- Case insensitive  `echo "bind 'set completion-ignore-case on'" >> ~/.bashrc`

### 服务器端IDE的使用 (使用VS code sftp插件)

   ```json
   ## sftp.json
   {
       "host": "172.20.1.32",
       "port": 22,
       "username": "zhangziwei",
       "password": "Zhangziwei2019!",
       "protocol": "sftp", 
       "agent": null,
       "privateKeyPath": null, 
       "passphrase": null, 
       "passive": false, 
       "interactiveAuth": true,
       "remotePath": "/home/zhangziwei/",
       "uploadOnSave": true,
       "syncMode": "update",
       "ignore": [
           "**/.vscode/**",
           "**/.git/**",
           "**/.DS_Store"
       ],
       "watcher": {
           "files": "glob",
           "autoUpload": true,
           "autoDelete": true
       }
   
   }
   ```

### **部署与训练face-recognition**（insightface）

- Training set 部署：
  - Create `pairs.txt` - like in example `pairs_label.txt` ✅
  - Get `.rec, .idx` files by running `dir2rec.py ` 用来制作训练集 ✅
  - Get ` .bin ` file by running ` build_eval_pack.py` 用来制作测试集 ✅
  - MegaFace Test：
    - 对齐FaceScrub和MegaFace distractors
    - 生成对应feature(`src/megaface/gen_megaface.py`)
    - 运行megaface development kit.

- 遇到问题
  1. Encoding问题（解决方法：在pickle内加上`encoding="bytes"`) ✅
  2. `property` 包含了`<num_of_classes>,width,height `✅
  3. `config.py` 的作用：最新版本的训练，都在目录 `recognition/train.py` 下面 ✅
  4. **out of memory： **需要释放显卡缓存 ✅
  5. 训练时 **out of memory：**bathch_size 过大，随机到的batch超过显存。 ✅
  6. 训练速度慢 (解决方法：换小模型，使用`train_parall.py`) ✅
  7. **RetinaFace 人脸侦测模块使用：需要先 `make` 才能使用** :exclamation::exclamation::exclamation: ✅

### 服务器密钥对

1. `ssh-keygen -t rsa` 生成密钥对
2. 将`scp /home/zc/.ssh/id_rsa.pub 用户名@你的服务器的ip:/home/chrisd/.ssh/authorized_keys`

### OpenCV环境安装 

1. 下载`xcode` 

2. 执行下列命令

   ```shell
   brew install opencv
   brew link opencv
   brew install pkg-config
   pkg-config --cflags --libs /usr/local/Cellar/opencv/<version_number>/lib/pkgconfig/opencv.pc
   ```

3. 调试`OpenCV` 

#### Linux

```shell
# pre-requirements

[compiler] sudo apt-get install build-essential
[required] sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
[optional] sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

# repo

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

#compile

cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8
```

### RetinaNet-mobilenet.25的输出的指代 :exclamation:

```shell
## RetinaFace-TVM
(1, 4, 10, 8)           #score
(1, 1, 10, 8, 8)				#bbox					height,width = 8,10
(1, 4, 10, 8, 5)				#landmark
(1, 4, 20, 15)					#score
(1, 1, 20, 15, 8)				#bbox					height,width = 15,20
(1, 4, 20, 15, 5)				#landmark
(1, 4, 40, 30)					#score
(1, 1, 40, 30, 8)				#bbox					height,width = 30,40
(1, 4, 40, 30, 5)				#landmark

## RetinaFace
sym size: 9
im_scale 1.0
(1, 4, 8, 10)  -->(160,1)			x.transpose((0,2,3,1)).reshape((-1,1)) 
(1, 8, 8, 10)		--> (160,4)		x.transpose((0,2,3,1)).reshape((-1,bbox_pred_len)) 	 
(1, 20, 8, 10)  --> (160,5,2)	x.transpose((0,2,3,1)).reshape((-1,5, landmark_len//5)) 
(1, 4, 15, 20) -- (600,1)
(1, 8, 15, 20)  --> (600,4)
(1, 20, 15, 20) --> (600,5,2)
(1, 4, 30, 40) --> (2400,1)
(1, 8, 30, 40) --> (2400,4)
(1, 20, 30, 40) --> (2400,5,2)

x.transpose((0,2,3,1)).reshape((-1,1)) #reshape
```

- TVM Cpp部署问题 `fatal error: '../../src/runtime/c_runtime_api.cc' file not found`: `tvm_runtime_package.cc` 路径错误，把`../../src` 改成`../src `✅

### 人脸识别部署问题

```markdown
1. 在计算人脸相似度时候，需不需要对模型输出的feature做什么操作？比如归一化正则化 `需要使用归一化`
2. 度量人脸相似度距离除了余弦相似，欧式距离以外，还有没有其他度量方法？	`一般跟模型训练所使用的Loss相关`
3. 在人脸做相似度检测是，人脸距离摄像头的距离也会影响相似度的计算，具体原因是什么？（分辨率？）
4. Incompatible attr in node at 0-th output: expected [128,512], got [128,25088]            `output size` 问题：因为`net_output` 没有同意，如果是`mobilefacenet` 应该选择`GDC` ✅ 
5. Cpp文件读图片路径问题：使用绝对路径可以解决 ✅
6. Opencv Cpp Mat格式每个像素值有取值范围(0~255) ✅
7. Opencv Python numpy**无法输出图像**：因为np的格式是`int64` opencv要求`uint8` ✅
8. FacePreprocess.h用法：`similartransform得到的结果是一个透视变换矩阵，使用opencv透视变换来align两张图`
9. WrapPerspective 图片总是随着脸出现的位置而改变位置：`因为landmark是根据全局size定下的坐标`
10. Finetune的时候loss不下降：
	1.1 `降低learning rate` ✅
	1.2 `使用Adam或其他优化器`
11. Finetune出来的模型无法直接使用TVM压缩：`需要在deploy里使用model_slim.py将最后面的fc7层以后的去除`
12. fuser -v /dev/nvidia `查看gpu僵尸进程`
13. `gstreamer-1.0`如何使用？: 直接放在/opt 目录下就可以用
14. 确认iPhone & iPad无法识别问题：iOS 逆时针旋转图片90度，所以检测不到人脸 ✅
```

### C++问题

~~~markdown
1. module not found: `brew install pkg-config`
2. Could not find PROTOBUF Compiler: `brew install protobuf`
3. struct无输出？: `全局变量问题，在使用前定义变量`
4. struct输出越界？：`没有分配内存，该用vector<array<double,15>>` 的形式，以保证内存分配。数组在界内
5. 输入进function的是那个参数，`filter or inframe`: frame是输入进去的图片矩阵
6. `#include array.h no such dir or file`：因为c不支持cpp的库，将`.c`改成`.cpp`，										并用`extern "C"{...}` 将整个文件框起来
7. `undefined symbol: gst_opencv_video_filter_get_type`: library没有link进去
8. C 与 cpp or 操作符问题：`cpp需要自己定义operator` ✅
   ```c++
   inline GParamFlags operator|(GParamFlags a, GParamFlags b) {
       return static_cast<GParamFlags>(static_cast<int>(a) | static_cast<int>(b));
   }
   ```
9. 解决memory leak问题：`定位为bug用二分法` ✅
	1. **JsonBuilder封装成JsonNode之后，其生命周期交给JsonNode管理。无需单独释放。**
  2. **JsonNode添加进JsonObject里面之后，其生命周期交给JsonObject管理。无需单独释放。**
  3. 原因：
     1. :exclamation::exclamation:GValue event_body 需要被释放，否则为`NULL` ✅
     2. `g_value_unset(&event_body)` 释放GValue
     
10. 可以使用 `leaks` 调试内存泄漏问题 
11. build的时候无法link到opt的环境：`source /opt/gstreamer-1.0/gst-env.sh`，由于env路径先后顺序问题，没有首先加载opt下的lib ✅
12. tensorflowlite对比tvm部署
	- Command `onnx-tf convert -i y1-arcface-emore.onnx -o arcface.pb`
  - Problems1: `ValueError: '_minusscalar0' is not a valid scope name`：onnx-tf pull#348
  - Problems2: `ValueError: Dimensions must be equal`: p_relu.py `slope = BroadcastMixin.explicit_broadcast([x,tensor_dict[node.inputs[1]]], axis=1)`
  

~~~

### CUDA & CUDNN


```shell
# CUDA
cat /usr/local/cuda/version.txt
#CUDNN
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -
```

### maskrcnn-benchmark

```markdown
1. demo的bbox位置和inference的不一致，导致IOU很低
  - [x] transformer的问题，train和test公用了一套。【角度】
  - [x] 解决上面问题后，出现的bbox还是过大？因为有个RRPN_margin，这个是为了增加框取的面积，防止rrpn生成的时候会截取部分字符
2. Add IoU to training step ：IOU通常不会再train step计算，通常在inference时候计算
3. Two-stage object detector 通常会fix BN，由于batch太小的缘故。MegDet paper：Large mini-batch training
```

### vscode remote无法连接时：`pgrep -f "vscode" | xargs kill`

