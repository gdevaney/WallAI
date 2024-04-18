---
layout: default
title: WallAI - CS6476 Computer Vision Project
description: Image Segmentation and Recoloration
---


# Introduction/Problem Definition
ProjectColor is an application made by Home Depot that allows users to visualize different paint colors after they take a picture of their room. The Home Depot app has some difficulty detecting edges and assigning appropriate pixels to what it defines as “walls”. The application also has difficulty detecting light exposure and applying appropriate color transformations (the example below likely does not translate to what the color would actually look like on the wall). We will attempt to solve these problems using Machine Learning and Computer Vision techniques.

<img src="{{site.baseurl}}/assets/images/hd_example.png" width="100%"/> \
Home Depot's ProjectColor Example

In order to train and test our models, we used the ADE20K bedroom segmented image dataset. After downloading the repository, we scraped the images by hand to exclude bedroom scenes with impurities (text, log cabins, etc.) and aggregate the human generated wall segments into a mask. An example of the kinds of images used in our project are below: 

<img src="{{site.baseurl}}/assets/images/bedroom_mask_example.png" width="100%"/>

Using the training data, image segmentation models, and edge detection techniques, we will craft a solution that will hopefully outperform Home Depot in wall detection and recoloration.

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

There were previous attempts at scene parsing that could be applied to detecting walls: pyramid scheme parsing network [4].
This network uses ResNet to get features and then uses a pyramid pooling module as a decoder.
The pyramid pooling module fuses features under four different pyramid scales, where the highest level captures the global context, while the lowest level captures more fine-grained context.
This context-aware model takes into consideration what objects are associated with which, e.g. boat is the object usually associated with water, not a car.
Similarly, PSPNet could be used to take into consideration that a wall is an object that is to be associated indoors.
Our work plans to leverage the human-annotated dataset and pretrained models trained on lots of data to create models that specifically detects and paints walls.
Painting the walls is another challenge after segmenting the image correctly as there are multiple aspects to consider to color image naturally.

# Methods/Approach

## TODO: Add the diagram!

Given the limited hand annotated resource regarding wall segmentation task, we plan to conduct multiple rounds consisting of qualitative and quantitative analysis on our models and come up with the best model to compare against Home Depot's Project Color.
The qualitative metrics that we utilize to assess the quality of the wall painted outputs from different models are coloration, edge detection, and segmenatation, with each score ranging from 1 to 5 [^1].

Coloration quantifies how similar the lighting of new painted walls are when compared to the original image.
Painting the walls considering different lighting is essential especially because walls are adjacent to multiple light sources within our dataset such as sunlight from the windows or lamps by the bedside.
Edge detection score estimates how well the painted walls draw out the original edges within the original image.
If the paint covers up the edges of the original room, the image is going to look awkward.
Segmentation score evaluates how well different models segment different objects within the image such as ceiling, bed frames, window, and etc.
We determined that these qualitative metrics can be subjective, so in order to maintain consistency througout the entire experiments, we only had one annotator conducting the evaluation.

For the quantitative analysis, we leverage the annotations from ADE20K dataset and calculate precision, recall, F1, and Intersection over Union score for two of our semantic segmentation models [3]. After conducting error analysis on the segmentation, we ensemble edge detection algorithm with two of our segmentation models and run the experiments again on a different dataset, 'tensorflow/lsun/bedroom'. With the two new models and dataset, we finally compare the results quantitatively and qualitatively with Home Depot's result. We concluded, through qualitative and quantitative analysis, that our ensembled model best paints the walls given an indoor image.

[^1]: Score 1 indicates very bad, Score 2 indicates major errors, Score 3 indicates minor errors, Score 4 inidcates minimal errors that cannot be detected easily, and Score 5 indicates almost perfect coloration, edge detection, and segementation score respectively

### OpenCV Module
OpenCV has a number of edge detection and masking modules that can be used for filtering objects in images [5]
We used techniques outlined in Garga's work to input an image and a color of choice, apply a masking technique using interpolation, a Canny edge detector, and several OpenCV modules to identify the wall, and edit the HSV color space to recolor wall segments while preserving natural light.
The OpenCV approach performed well at handling various light intensities, but struggled to identify and segment walls with deep shadows and fine details.
Therefore, we wanted to explore techniques using semantic segmentation and potentially combine the two approaches to create our final product. 

Notes: Utilized Canny edge detector to refine edges predicted by models. Used model outputs to sample points from predicted wall masks for FloodFill function to properly fill out wall. Combined Canny edge detector from original image and edges of model predictions to refine points. 

<img src="{{site.baseurl}}/assets/images/masking_example.png" width="100%"/> \
Blue Area: Semantic Segmentation prediction; Lines: Canny edge detector, Dots: Sampled points for recoloration

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

### Home Depot Model
The Home Depot model is proprietary, so we are unable to understand what is used under the hood.
However, we will use this as a baseline for scoring our models.
Qualitatively, we will use human interpretation on three metrics mentioned above and conduct error analysis.
Then, we will use quantitave evaluation using Mean Intersection over Union to compare outputs to Home Depot's using similarity metrics.


# Experiments/Results
During experimentation, we attempted 3 methods and used the Home Depot output as a baseline.
Using the 3 qualitative metrics outlined above on one particular instance shown below, Zhou's model perfoms best in segmentation, and OpenCV performs best at edge detection and coloration.
While our baseline didn't perform best at any one category, it combined accurate coloration with segmentation and object detection, unlike the OpenCV.
We experimented with thresholding the Canny edge detector and found that a large threshold range (45,225) outperformed a smaller range.
The quality of images taken by consumers varies, and using a large threshold range forces the model to be picky choosing pixels in the edge map, yet generous to pixels connected to those in the edge map.

## Experiment 1: Semantic Segmentation vs. PSPNet 

### Qualitative Analysis
We conduct human evaluation on 100 images from ADE20K to score how well each model detects different objects within the images.
Among the three qualitative analysis, only the segmentation score is relevant as these models at this stage just filled the pixels with the same color.
Looking into how semantic segmentation model by Zhou was performing bettyer than PSPNet, we noticed that Zhou's model handels different depths of walls within the same image very well.
In contrast, PSPNet had issues when there were doors open within the image and walls behind the door.
Furthermore, PSPNet exhibited limitations in discerning patterned walls and exhibited suboptimal performance in scenarios where disparate walls within an image exhibited varying colors- a challenge not encountered by the initial model.
A drawback of Zhou's model was its propensity to over-paint, resulting in false positives.
Additionally, neither model demonstrated robustness in detecting walls at various angles.


| Method              | Coloration | Edge Detection | Segmentation
| Semantic Segmentation (Zhou)|   1   | 1   | 3.65  |
| PSPNet    |  1   | 1   | 2.97   |

<img src="{{site.baseurl}}/assets/images/Screenshot 2024-03-26 at 00.04.52.png" width="40%"/> \
Semantic Segmentation Output


<img src="{{site.baseurl}}/assets/images/bedroom_pspnet.png" width="40%"/> \
PSPNet Output


### Quantitative Analysis
We sought to validate our qualitative findings through quantitative analysis.
To this end, we computed precision, recall, F1 score, and Intersection over Union (IOU) metrics for both models' wall segmentation performance using the ADE20K dataset.
The results show that Semantic Segmentation model has a higher score for precision, F1, and IOU.
These quantitative metrics corroborate our qualitative observations, affirming the efficacy of Zhou's Semantic Segmentation model in accurately delineating walls within images when compared to PSPNet.


| Method              | Precision | Recall | F1 | IOU
| Semantic Segmentation (Zhou) | 0.87 | 0.83 | 0.85 | 0.74 | 
| PSPNet | 0.77 | 0.86 | 0.81 | 0.67 | 

## Experiment 2: OpenCV infused Semantic Segmentation vs. Home Depot

### Qualitative Analysis
Next, we analyze our newly proposed ensembled model, which combies the Canny Edge Detector with a pretrained Semantic Segmentation network, and the model deployed by Home Depot.
To evaluate their performance, we ran the experiments on 100 unseen data from the 'tensorflow/lsun/bedroom' dataset.
For each model, we assessed the models across multiple dimension: coloration, edge detection, and segmentation, utilizing a rating scale ranging from 1 to 5.
Our qualitative analysis reveal distinct strengths and weaknesses inherent to each model.

Home Depot's model exhibits remarkable proficiency in identifying minute objects within the background, consistently achieving segmentation scores of 4 or 5.
Instances where the model received lower scores typically occurred when confronted with low-resolution input images or encountered challenges arising from image distortion induced by wall painting processes.
Notably, in one instance among the evaluated images, the Home Depot model produced an output featuring jagged artifacts due to a wall segmentation error.
Conversely, our ensemble model demonstrates suboptimal performance when presented with predominantly white backgorunds.
Furthermore, it occasionally misidentifies the ceiling and exhibitis a tendency to over-paint, resulting in false positives- a recurring issue observed in previous experiment.

Our model outperforms Home Depot's model in terms of color fidelity and edge detection capabilities.
A notable distinction lies in the treatment of multiple light sources within the scene.
While Home Depot's model overlooks the presence of multiple light sources in the majority of its outputs, our ensemble model accurately captures and reflects all light sources present in the original image
Moreover, the integration of the Canny Edge Detector significantly enhances the clarity of edges in our model's outputs, contributing to improved overall performance in edge detection tasks.
In addition, the runtime of Home Depot's model is 5 times longer than our ensemble method, which can lead to latency issue when deployed in a real-world application scenario.

| Method              | Coloration | Edge Detection | Segmentation |
| OpenCV infused Semantic Segmentation    |  3.71   | 3.72   | 4.09   |
| Home Depot |   3.24   | 3.74   | 4.4   |

<img src="{{site.baseurl}}/assets/images/home_depot_bad.png" width="100%"/> \
Home Depot outperforming our model

<img src="{{site.baseurl}}/assets/images/home_depot_good.png" width="100%"/> \
Our model outperforming Home Depot

### Quantitative Analysis

When conducting quantitative analysis, we utilize different reference image used for IOU ground truth.
By merging precision-focused coloration strategies with ground truth masks, we hoped that our ensemble model prioritizes accurate wall segmentation while minimizing coloration errors in non-wall regions.
We could not calculate metrics for comparison against Home Depots model other than IOU because Home Depot does not export masks used to guide recoloration.
This study not only advances wall segmentation accuracy but also underscores the significance of precision in coloration techniques as we did not want to color over areas which were not walls, just to improve recall.
Integrating OpenCV allowed us to do that without sacrificing accuracy.

| Method              | Accuracy | Precision | Recall | F1 | IOU
| Segmentation | 0.91 | 0.82 | 0.92 | 0.87 | NA |
| Segmentation + OpenCV| 0.91 | 0.88 | 0.83 | 0.85 | 0.988 | 
| HomeDepot | NA | NA | NA | NA | 0.985 |

# Conclusion

## Discussion
Before embarking on this project, our collective assumption rested on the notion that the advances in Computer Vision stemmed primarily from the upscaling of Deep Convolutional Neural Network and the intricate architecture's capacity to extract richer insights from images.
However, we learned the pivotal role of quality data.
We came to understand that the mere abundance of data is insufficient; rather it is the quality of data that truly matters.
By "good data," we refer to datasets meticulously annotated by domain experts, rather than relying solely on crowdsourced annotations.
This emphasiss on expert-driven annotation ensures incorporation of meaningful information into the dataset.
Armed with such high-quality data, we made a really simple model to perform pretty well against one made by Home Depot.

## Challenges Encountered
We explored if a recent unsupervised semantic segmentation model, like Hamilton's work [6], can be integrated to our problem.
Because the paper had promising results, we delve into it before evaluating the feasibility of their model to our task.
Our task of coloring a wall mandates some type of classification for wall, and the only way to make unsupervised model detect for walls is to manually annotate for walls given their output, and train again.
We did not have enough time nor resources to pursue this, so subsequently made sweeping assumptions about the dataset to bypass this step.
We told the model that walls should be the biggest segment among all the segments or the segment should only straight lines, but these assumptions made the model perform worse. 
If we were to start this project again, we would explore other options that can make our supervised semantic segmentation models better.

# References

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[2] M. Everingham, L. Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” Int. J. Comput. Vision, vol. 88, no. 2, p. 303–338, jun 2010. [Online]. Available: https://doi.org/10.1007/s11263-009-0275-4

[3] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” 2018.

[4] X. Q. X. W. J. J. Hengshuang Zhao, Jianping Shi, “Pyramid scene parsing network,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[5] S. Garga, “UC HACK 20: Detect Wall from the Image and Change Its Colour or Apply Various Patterns,” GitHub, 2020. 

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition, ser. CVPR ’16. IEEE, Jun. 2016, pp. 770–778. [Online]. Available: http://ieeexplore.ieee.org/document/7780459

[7] M. Hamilton, Z. Zhang, B. Hariharan, N. Snavely, and W. T. Freeman, “Unsupervised semantic segmentation by distilling feature correspondences,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=SaKO6z6Hl0c
