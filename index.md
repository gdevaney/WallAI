---
layout: default
title: WallAI - CS6473 Computer Vision Project
description: Image Segmentation and Recoloration
---


# Introduction/Problem Definition
ProjectColor is an application made by Home Depot that allows users to visualize different paint colors after they take a picture of their room. The Home Depot app has some difficulty detecting edges and assigning appropriate pixels to what it defines as “walls”. The application also has difficulty detecting light exposure and applying appropriate color transformations (the example below likely does not translate to what the color would actually look like on the wall). We will attempt to solve these problems using Machine Learning and Computer Vision techniques.

<img src="{{site.baseurl}}/assets/images/hd_example.png" width="100%"/> \
Home Depot's ProjectColor Example

In order to train and test our models, we used the Large-scale Scene Understanding (LSUN) bedroom image dataset from Kaggle. After downloading the repository, we scraped the images by hand to exclude bedroom scenes without walls. An example of the kinds of images used in our project are below: 

<img src="{{site.baseurl}}/assets/images/bedroom_example.jpg" width="50%"/>

# Related Works
In computer vision, the utilization of large-scale datasets like ImageNet [1] and Pascal [2] has been instrumental in propelling the field forward.
ImageNet consists of millions of labeled images across thousands of categories, pivotal in advancing computer vision research by providing benchmarks for evaluating algorithms in tasks such as image classification and object segmentation [1].
Pascal Visual Object Classes dataset, while smaller compared to ImageNet, has played a significant role in the development of object detection and segmentation algorithms [2].
Pascal offers a diverse set of images annotated with object bounding boxes and segmentation masks across 20 different object categories.
However, it is noteworthy that these datasets predominantly contain annotations at the image level or offer bounding box delineations around objects, rendering them somewhat less conducive to segmentation tasks.

In response to this limitation, Zhou constructs an ADE20K: a dataset with pictures that are labeled on every pixel by one expert annotator [3].
This meticulous annotation scheme allowed for the diversity of the labels within the image while maintaining the consistency of the annotation.
On average, the annotator labeled 29 different segments per image, labeling discrete objects with well-defined shapes, background regions, or object parts.
Zhou also establishes a benchmark for scene parsing tasks by utilizing ADE20K.

There were previous attempts at scene parsing that could be applied to detecting walls: pyramid scheme parsing network [4]. This network uses ResNet to get features and then uses a pyramid pooling module as a decoder. The pyramid pooling module fuses features under four
different pyramid scales, where the highest level captures the global context, while the lowest level captures more fine-grained context. This context-aware model takes into consideration what objects are associated with which, e.g. boat is the object usually associated with water, not a car. Similarly, PSPNet could be used to take into consideration that a wall is an object that is to be associated indoors.

# Methods/Approach

### Home Depot Model
The Home Depot model is proprietary, so we are unable to understand what is used under the hood. However, we will use this as a baseline for scoring our models. Quantitatively, we will use Mean Intersection over Union to compare our outputs to Home Depot's using a similarity metric. Then, we will use human interpretation to score the best model against Home Depot's model qualitatively on three metrics: segmentation, edge accuracy, and coloration. 

### OpenCV Module
OpenCV has a number of edge detection and masking modules that can be used for filtering objects in images [5]. We used techniques outlined in Garga's work to input an image and a color of choice, apply a masking technique using interpolation, a Canny edge detector, and several OpenCV modules to identify the wall, and edit the HSV color space to recolor wall segments while preserving natural light. The OpenCV approach performed well at handling various light intensities, but struggled to identify and segment walls with deep shadows and fine details. Therefore, we wanted to explore techniques using semantic segmentation and potentially combine the two approaches to create our final product. 

### Semantic Segmentation CNN Models
Semantic Segmentation is a computer vision task that assigns a semantic label to each partitioned segment.
Unlike instance segmentation, where the goal is to differentiate between individual object instances, semantic segmentation focuses on categorizing each pixel in the image into meaningful classes, such as road, sky, person, car, or wall.
The primary objective of semantic segmentation is to understand the content of the image at the pixel level, enabling machines to interpret the scene with a higher level of understanding.

We propose to utilize semantic segmentation models to distinguish walls within our dataset and subsequently color the walls.
We integrate a pre-trained model from Zhou's work [3]: ResNET50Dilated [6] as the encoder and PPM-Deepsup as the decoder of the semantic segmentation model, which is widely used as a starting point for evaluating deep learning semantic segmentation models.

The next model we try is using PSPNet to identify just the walls.
The model would use ResNet50, a 50-layer convolutional neural network (CNN), for encoding and pyramid scheme parsing network for decoding, which exploits global context information by different-region-based context aggregation.
We decided on this model because ResNet is great for semantic segmentation, and pyramid scheme parsing network is great for identifying the object at different scale: taking in a global context of the image at a highest level to deduce the object relation to one another within an image, and also taking in more fine-grained context at continuously lower levels.
We believed this model would be great for our use case since walls are only present within an indoor image, and there most likely is a piece of furniture or a bed next to a wall.
Lastly, we propose to make use of only a subset of ADE20K indoor images to fine-tune smaller semantic segmentation models and test the Mean Intersection of the Union area (IoU) using the ground truth from ADE20K images.


# Experiments/Results

- Result using pre-trained model from Zhou\
<img src="{{site.baseurl}}/assets/images/Screenshot 2024-03-26 at 00.04.52.png" width="50%"/>

- Result using OpenCV modules\
<img src="{{site.baseurl}}/assets/images/opencv_test.png" width="50%"/>


# What's Next

- We want to investigate whether conducting fine-tuning with a subset of the dataset from ADE20K, with only wall annotation, can improve the performance of the models (April 10th)

- We plan to explore unsupervised semantic segmentation models like Hamilton's work [6], to see if this model can be modified to identify walls more efficiently and accurately. First, we would have to read the paper to evaluate the feasibility of this model (April 20th).


# Team Member Contributions

Make a checklist

# References

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[2] M. Everingham, L. Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” Int. J. Comput. Vision, vol. 88, no. 2, p. 303–338, jun 2010. [Online]. Available: https://doi.org/10.1007/s11263-009-0275-4

[3] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” 2018.

[4] X. Q. X. W. J. J. Hengshuang Zhao, Jianping Shi, “Pyramid scene parsing network,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[5] S. Garga, “UC HACK 20: Detect Wall from the Image and Change Its Colour or Apply Various Patterns,” GitHub, 2020. 

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition, ser. CVPR ’16. IEEE, Jun. 2016, pp. 770–778. [Online]. Available: http://ieeexplore.ieee.org/document/7780459

[7] M. Hamilton, Z. Zhang, B. Hariharan, N. Snavely, and W. T. Freeman, “Unsupervised semantic segmentation by distilling feature correspondences,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=SaKO6z6Hl0c
