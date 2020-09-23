## Maskrcnn-benchmark (Facebook)

### Backbone

```python
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        #1 图片通过backbone获得features
        features = self.backbone(images.tensors)
        
				if self.fp4p_on:
            # get you C4 only
            proposals, proposal_losses = self.rpn(images, (features[-1],), targets)
            
        #2 图片+features+targets通过rpn获得proposal和loss
        else:
            proposals, proposal_losses = self.rpn(images, features, targets)
            
        if self.roi_heads:
            #3 features+proposals+targets通过roi_heads得到x(cls)，result(reg)，losses
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
```

- 通用RCNN的框架：
  1. backbone
     - ResNet50，VGG...
  2. rpn
     - Anchor generator: 根据设定生成anchor [[x,y,w,h,theta], … ]
     - RPN head: 简单的classification (num_anchors输出，判断是不是属于某个anchor)和regression heads
     - Box selector: 后处理box选择，通常是NMS操作
     - Loss evaluator: 
       - Smooth L1
       - CE
  3. roi_head (box)
     - Feature extractor: ResNet50Conv5ROIFeatureExtractor
     - Roi box predictor: 
     - Post processor
     - Loss evaluator 

### Dataset

从config.paths_catelog.py查找model对应的Dataset

**Custom Dataset**

1. 构造解析代码，并保存成im_info格式

   ```python
   im_info = {
               'gt_classes': gt_classes,  #[class1, class2, class3, ...]
               'max_classes': max_classes,  # overlaps.argmax(axis=1)
               'image': impath,  # image_name
               'boxes': gt_boxes,  # (x,y,w,h,theta)
               'flipped': False,
               'gt_overlaps': overlaps,  # gt set 1.0
               'seg_areas': seg_areas,  # w*h
               'height': im.shape[0],  # image height
               'width': im.shape[1],  # image width
               'max_overlaps': max_overlaps,  # overlaps.max(axis=1)
               'rotated': True
           }
   ```

   

2. 使用对应的Dataset解读

|      Model       |       Dataset       |                      Features                      |
| :--------------: | :-----------------: | :------------------------------------------------: |
|                  |     COCODataset     |                                                    |
|                  |  PascalVOCDataset   |                                                    |
|                  |    ConcatDataset    |                                                    |
| ICDAR2013Dataset |  ICDAR2013Dataset   |                      Vertices                      |
|    RRPN_Train    |   RotationDataset   | (x,y,w,h,$\theta$) $\theta$为x轴与长边夹角[0, 180) |
|     RRPN_E2E     |   SpottingDataset   |                                                    |
|    MASK_RRPN     | RotationMaskDataset |                                                    |

**Augmentation**

- dataset/transform