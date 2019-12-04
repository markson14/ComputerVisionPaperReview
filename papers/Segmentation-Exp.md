# Image Segmentation Experiment

#### ADE20K

Scene-centric containing 20k images annotated with 150 object categories.

#### MS-COCO

330k images, 1.5m object instances, 171 categories, 5 captions per image

#### Pascal VOC 2007

20 classes, 9963images containing 24,640 annotated objects (2.47 objects/image)

### Semantic Segmentation

|       Model Name       |  Method   |  Dataset   | pixAcc | mIoU |
| :--------------------: | :-------: | :--------: | :----: | :--: |
|    fcn_resnet50_ade    |    FCN    |   ADE20K   |  79.0  | 39.5 |
|   fcn_resnet101_ade    |    FCN    |   ADE20K   |  80.6  | 41.6 |
|    psp_resnet50_ade    |    PSP    |   ADE20K   |  80.1  | 41.5 |
|   psp_resnet101_ade    |    PSP    |   ADE20K   |  80.8  | 43.3 |
|  deeplab_resnet50_ade  | DeepLabV3 |   ADE20K   |  80.5  | 42.5 |
| deeplab_resnet101_ade  | DeepLabV3 |   ADE20K   |  81.1  | 44.1 |
|   fcn_resnet101_coco   |    FCN    |  MS-COCO   |  92.2  | 66.2 |
|   psp_resnet101_coco   |    PSP    |  MS-COCO   |  92.4  | 70.4 |
| deeplab_resnet101_coco | DeepLabV3 |  MS-COCO   |  92.5  | 70.4 |
|   fcn_resnet101_voc    |    FCN    | Pascal VOC |  N/A   | 83.6 |
|   psp_resnet101_voc    |    PSP    | Pascal VOC |  N/A   | 85.1 |
| deeplab_resnet101_voc  | DeepLabV3 | Pascal VOC |  N/A   | 86.2 |
| deeplab_resnet152_voc  | DeepLabV3 | Pascal VOC |  N/A   | 86.7 |

#### Performance

|      Model Name       |  Method   |  Time CPU/GPU  |
| :-------------------: | :-------: | :------------: |
|   fcn_resnet50_ade    |    FCN    | 26.785s/1.421s |
|   fcn_resnet101_ade   |    FCN    | 38.456s/2.247s |
|   psp_resnet50_ade    |    PSP    | 29.071s/1.734  |
|   psp_resnet101_ade   |    PSP    | 41.186s/2.314s |
| deeplab_resnet50_ade  | DeepLabV3 | 26.630s/1.415s |
| deeplab_resnet101_ade | DeepLabV3 | 36.627s/2.105s |
| deeplab_resnet152_voc | DeepLabV3 | 44.906s/2.953s |



### Instance Segmentation

|            Model Name            | Method  | Box AP/AP@0.5/AP@0.75 | Seg AP/AP@0.5/AP@0.75 |  Time CPU/GPU  |
| :------------------------------: | :-----: | :-------------------: | :-------------------: | :------------: |
|   mask_rcnn_resnet18_v1b_coco    |   v1b   |    31.2/51.1/33.1     |    28.4/48.1/29.8     | 24.259s/0.794s |
| mask_rcnn_fpn_resnet18_v1b_coco  | v1b/FPN |    34.9/56.4/37.4     |    30.4/52.2/31.4     | 9.139s/0.435s  |
|   mask_rcnn_resnet50_v1b_coco    |   v1b   |    38.3/58.7/41.4     |    33.1/54.8/35.0     | 61.053s/2.877s |
| mask_rcnn_fpn_resnet50_v1b_coco  | v1b/FPN |    39.2/61.2/42.2     |    35.4/57.5/37.3     | 30.237s/1.853s |
|   mask_rcnn_resnet101_v1d_coco   |   v1d   |    41.3/61.7/44.4     |    35.2/57.8/36.9     | 65.629s/2.887s |
| mask_rcnn_fpn_resnet101_v1d_coco | v1d/FPN |    42.3/63.9/46.2     |    37.7/60.5/40.0     | 31.425s/2.039s |



### Conclusion

- Model with FPN in Instance segmentation is able to reduce processing time and return better results