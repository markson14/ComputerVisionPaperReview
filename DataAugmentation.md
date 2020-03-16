# Data Augmentation

### GridMask Data Augmenation

**3 Categories of Augmentation**

1. Spatial transformation
2. Color Distortion
3. Information Dropping

### Cutmix



**Cutmix/cutout/mixup**

![Screen Shot 2020-03-10 at 4.22.42 pm](assets/Screen%20Shot%202020-03-10%20at%204.22.42%20pm-3828736.png)

![Screen Shot 2020-03-10 at 4.22.33 pm](assets/Screen%20Shot%202020-03-10%20at%204.22.33%20pm-3828736.png)

- 作用

![Screen Shot 2020-03-10 at 4.23.21 pm](assets/Screen%20Shot%202020-03-10%20at%204.23.21%20pm-3828736.png)

**Experiment**

![Screen Shot 2020-03-10 at 4.23.37 pm](assets/Screen%20Shot%202020-03-10%20at%204.23.37%20pm-3828736.png)

![Screen Shot 2020-03-10 at 4.23.50 pm](assets/Screen%20Shot%202020-03-10%20at%204.23.50%20pm-3828736.png)

### Mixup

$$
\hat{x}= \lambda{x_i}+(1-\lambda)x_j \ \ where \ x_i,x_j \ are \ raw \ input\ vectors  \\
\hat{y} = \lambda{y_i}+(1-\lambda)y_j \ \ where \ y_i,y_j \ are \ onehot \ label\ encoding
$$

### Cutout

