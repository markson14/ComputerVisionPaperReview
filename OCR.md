## Optical Character Recognition

### An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

**Abstract**

1. End-to-end trainable (端到端可学习)
2. Naturally handle sequences in arbitrary lengths, no character segmentation or horizontal scale normalisation (无需切割，可以处理自然场景不对称长度的字符串)
3. It is not confined to any lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (没有基于任何词典，可以超越基于词典的模型)
4. Effective yet much smaller model (模型非常小且非常有效)

### Connectionist Temporal Classification(CTC)

**Abstract**

- Ordinary sequence learners require pre-segmented training data and post-processing to transform their outputs into label sequences, applicability is limited(输入需要裁剪，和后处理才能得到序列标签，用途有限)
- Novel method for training RNNs to label unsegmented sequences directly. (可以直接训练无需切割输入的RNN序列)

****