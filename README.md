# WallAI
Image segmentation and masked recoloration using Computer Vision techniques to enhance state-of-the-art semantic segmentation models.

## Introduction
ProjectColor is an application made by Home Depot that allows users to visualize different paint colors after they take a picture of their room. The Home Depot app has some difficulty detecting edges and assigning appropriate pixels to what it defines as “walls”. The application also has difficulty detecting light exposure and applying appropriate color transformations (the example below likely does not translate to what the color would actually look like on the wall). I attempted to solve these problems using Machine Learning and Computer Vision techniques.

<img src="assets/images/hd_example.png" width="100%"/> \
Home Depot's ProjectColor Example

In order to train and test the models, I used the ADE20K bedroom segmented image dataset. After downloading the repository, I scraped the images by hand to exclude bedroom scenes with impurities (text, log cabins, etc.) and aggregate the human generated wall segments into a mask. An example of the kinds of images used in this project are below: 

<img src="assets/images/bedroom_mask_example.png" width="100%"/>

Using the training data, pretrained image segmentation models, and edge detection techniques, I demonstrate a solution that competes with Home Depot in wall detection and recoloration.

## Related Works
In computer vision, the utilization of large-scale datasets like ImageNet [1] and Pascal [2] has been instrumental in propelling the field forward.
However, it is noteworthy that these datasets predominantly contain annotations at the image level or offer bounding box delineations around objects, rendering them somewhat less conducive to segmentation tasks.

In response to this limitation, Zhou constructs an ADE20K: a dataset with pictures that are labeled on every pixel by one expert annotator [3].
This meticulous annotation scheme allowed for the diversity of the labels within the image while maintaining the consistency of the annotation.
On average, the annotator labeled 29 different segments per image, labeling discrete objects with well-defined shapes, background regions, or object parts.
Zhou also establishes a benchmark for scene parsing tasks by utilizing ADE20K.

A previous technique, Pyramid Scheme Parsing network (PSPNet), uses ResNet to get features and a pyramid pooling module as a decoder [4].
This context-aware model takes into consideration what objects are associated with which, e.g. boat is the object usually associated with water, not a car.
Similarly, PSPNet could be used to take into consideration that a wall is an object that is to be associated indoors.
This project leverages the human-annotated dataset, pretrained models trained on lots of data, and computer vision techniques to create models that specifically detect and paint walls.
Wall recoloration is another challenge after segmenting the image correctly, as there are multiple aspects to consider when naturally coloring images.

## Methods

Given the limited hand annotated resource regarding wall segmentation task, multiple rounds of qualitative and quantitative analysis were performed to determine the best model to compare against Home Depot's Project Color.
The qualitative metrics used to assess the quality of the wall painted outputs from different models are coloration, edge detection, and segmenatation, with each score ranging from 1 to 5 [^1].

Coloration quantifies how similar the lighting of newly painted walls compares to the original image.
Lighting considerations are essential because walls are adjacent to multiple light sources within the dataset, such as sunlight from the windows or lamps by the bedside.
Edge detection score estimates how well the painted walls draw out the edges within the original image.
Segmentation score evaluates how well models segment different objects within the image, such as ceilings, bed frames, windows, and etc.

For the quantitative analysis, I used the annotations from ADE20K dataset and calculate precision, recall, F1, and Intersection over Union score for two semantic segmentation models [3].
After conducting error analysis on the segmentation, I ensembled my edge detection algorithm with two segmentation models and ran the experiments again.
With the two new models, I compared the results quantitatively and qualitatively with Home Depot's result.
Through qualitative and quantitative analysis, it is evident that my ensembled model best paints the walls given bedroom images.

[^1]: Score 1 indicates very bad, Score 2 indicates major errors, Score 3 indicates minor errors, Score 4 inidcates minimal errors that cannot be detected easily, and Score 5 indicates almost perfect coloration, edge detection, and segementation score respectively

### Semantic Segmentation CNN Models
Semantic Segmentation is a computer vision task that assigns a semantic label to each partitioned segment.
Unlike instance segmentation, where the goal is to differentiate between individual object instances, semantic segmentation focuses on categorizing each pixel in the image into meaningful classes, such as road, sky, person, car, or wall.
The primary objective of semantic segmentation is to understand the content of the image at the pixel level, enabling machines to interpret the scene with a higher level of understanding.

Semantic segmentation models were used to distinguish walls within the dataset for subsequent coloration.
Zhous's Semantic Segmentation [3]: ResNET50Dilated [6] is the encoder and PPM-Deepsup is the decoder of the semantic segmentation model, which is widely used as a starting point for evaluating deep learning semantic segmentation models.

PSPNet: ResNet50, a 50-layer convolutional neural network (CNN), for encoding and pyramid scheme parsing network for decoding, which exploits global context information by different-region-based context aggregation.

### OpenCV Module
OpenCV has a number of edge detection and masking modules that can be used for filtering objects in images [5].
I used techniques outlined in Garga's work to input an image and a color of choice, apply a masking technique using interpolation, a Canny edge detector, and several OpenCV modules to identify the wall, and edit the HSV color space to recolor wall segments while preserving natural light.
I've highlighted a few key functions below for use of OpenCV during image manipulation:

**1. getColoredImage()**
input the bedroom image and target color, converted the image and target color to HSV, replaced the hue and saturation values of the original image with the target values, then converted the image back to RGB. 

**2. getOutlineImage()**
uses a gaussian blur and Canny edge detector to find and mark edges within the image. I used this for both the original image and the Semantic Segmentation output mask to refine edges and provide boundaries for color filling. 

**3. getSamples()**
sampled predicted wall pixels from Semantic Segmentation model outputs to use in follow on methods for color-filling operations. I put barriers in place to filter sampled pixels that were located in central points of predicted walls. 

**4. selectWall()**
creates the mask used for color-filling by leveraging OpenCV's floodFill function. By combining the outline image with the pixel samples, floodFill creates a mask by filling in bounded regions of the sampled pixels with an opposing color to other parts of the image.

**5. mergeImages()**
uses bitwise operations to combine original image with the recolored image at regions specified by the wall mask created in function 4.

Per the example below, you can see how the edges defined by OpenCV techniques snap the segmentation model pixel predictions to actual edges within the photo for a more precise solution. 
<img src="assets/images/masking_example.png" width="100%"/> \
(**Left** Blue Region: Semantic Segmentation prediction; Lines: Canny edge detector, Dots: Sampled points for recoloration) (**Right** Image with wall color replaced by Hidden Sea Glass)

### Home Depot Model
The Home Depot model is proprietary, so I was unable to understand what is used under the hood.
However, I used this to compete with mymodel through human interpretation on three metrics defined above and quantitave evaluation using Mean Intersection over Union.

# Experiments/Results
Using the 3 qualitative metrics outlined above on one particular instance shown below, Zhou's model perfoms best in segmentation, and OpenCV performs best at edge detection and coloration.
I experimented with thresholding the Canny edge detector and found that a large threshold range (45,225) outperformed a smaller range.
The quality of images taken by consumers varies, and using a large threshold range forces the model to be picky choosing pixels in the edge map, yet generous to pixels connected to those in the edge map.

## Experiment 1: Semantic Segmentation vs. PSPNet 

### Qualitative Analysis
A third pary conducted human evaluation on 100 images from ADE20K to score how well each model detects different objects within the images.
Zhou's model handles different depths of walls within the same image very well.
In contrast, PSPNet had issues when there were doors open within the image and walls behind the door.
Furthermore, PSPNet exhibited limitations in discerning patterned walls and exhibited suboptimal performance in scenarios where disparate walls within an image exhibited varying colors- a challenge not encountered by the initial model.
A drawback of Zhou's model was its propensity to over-paint, resulting in false positives.
Additionally, neither model demonstrated robustness in detecting walls at various angles.


| **Method**                   | **Coloration** | **Edge Detection** | **Segmentation** |
|------------------------------|----------------|--------------------|------------------|
| Semantic Segmentation (Zhou) | 1              | 1                  | 3.65             |
| PSPNet                       | 1              | 1                  | 2.97             |


<img src="assets/images/Screenshot 2024-03-26 at 00.04.52.png" width="40%"/> \
Semantic Segmentation Output

<img src="assets/images/bedroom_pspnet.png" width="40%"/> \
PSPNet Output


### Quantitative Analysis
I sought to validate qualitative findings through quantitative analysis.
To this end, I computed precision, recall, F1 score, and Intersection over Union (IOU) metrics for both models' wall segmentation performance using the ADE20K dataset.
The results show that Semantic Segmentation model has a higher score for precision, F1, and IOU.
These quantitative metrics corroborate qualitative observations, affirming the efficacy of Zhou's Semantic Segmentation model in accurately delineating walls within images when compared to PSPNet.


| **Method**                   | **Precision** | **Recall** | **F1** | **IOU** |
|------------------------------|---------------|------------|--------|---------|
| Semantic Segmentation (Zhou) | 0.87          | 0.83       | 0.85   | 0.74    |
| PSPNet                       | 0.77          | 0.86       | 0.81   | 0.67    |


## Experiment 2: OpenCV infused Semantic Segmentation vs. Home Depot

### Qualitative Analysis
Next, a third party analyzed the ensembled model, which combines the Canny Edge Detector with a pretrained Semantic Segmentation network, and the model deployed by Home Depot.
For each model, a third party assessed the models across multiple dimensions: coloration, edge detection, and segmentation, utilizing a rating scale ranging from 1 to 5.
The qualitative analysis revealed distinct strengths and weaknesses inherent to each model.

Home Depot's model exhibits remarkable proficiency in identifying minute objects within the background, consistently achieving segmentation scores of 4 or 5.
Instances where the model received lower scores typically occurred when confronted with low-resolution input images or encountered challenges arising from image distortion induced by wall painting processes.
Notably, in one instance among the evaluated images, the Home Depot model produced an output featuring jagged artifacts due to a wall segmentation error.
Conversely, the ensemble model demonstrates suboptimal performance when presented with predominantly white backgorunds.
Furthermore, it occasionally misidentifies the ceiling and exhibitis a tendency to over-paint, resulting in false positives- a recurring issue observed in previous experiment.

My model outperforms Home Depot's model in terms of color fidelity and edge detection capabilities.
A notable distinction lies in the treatment of multiple light sources within the scene.
While Home Depot's model overlooks the presence of multiple light sources in the majority of its outputs, my ensemble model accurately captures and reflects all light sources present in the original image.
Moreover, the integration of the Canny Edge Detector significantly enhances the clarity of edges in the model's outputs, contributing to improved overall performance in edge detection tasks.
In addition, the runtime of Home Depot's model is 5 times longer than my ensemble method, which can lead to latency issue when deployed in a real-world application scenario.

| **Method**                       | **Coloration** | **Edge Detection** | **Segmentation** |
|----------------------------------|----------------|--------------------|------------------|
| OpenCV infused Semantic Segmentation | 3.71           | 3.72               | 4.09             |
| Home Depot                       | 3.24           | 3.74               | 4.4              |



<img src="assets/images/home_depot_bad.png" width="100%"/> \
Home Depot (left) outperforming my model (right)


<img src="assets/images/home_depot_good.png" width="100%"/> \
My model (right) outperforming Home Depot (left)

### Quantitative Analysis

When conducting quantitative analysis, I used the ADE20K data and it's annotations to compute Accuracy, Precision, Recall, and F1 scores.
I could not calculate metrics for comparison against Home Depots model other than IOU because Home Depot does not export masks used to guide recoloration.
In order to evaluate IOU, I used getColoredImage() and the annotated ADE20K masks to fill the image with the new color and create a "ground truth".
By merging precision-focused coloration strategies with ground truth masks, I hoped that my ensemble model prioritized accurate wall segmentation while minimizing coloration errors in non-wall regions. 

This study not only advances wall segmentation accuracy but also underscores the significance of precision in coloration techniques. I did not want to color over areas which were not walls just to improve recall, and integration of OpenCV techniques achieved this goal without sacrificing accuracy.

| **Method**                   | **Accuracy** | **Precision** | **Recall** | **F1** | **IOU** |
|------------------------------|--------------|---------------|------------|--------|---------|
| Segmentation                 | 0.91         | 0.82          | 0.92       | 0.87   | NA      |
| Segmentation + OpenCV        | 0.92         | 0.88          | 0.85       | 0.86   | 0.988   |
| HomeDepot                    | NA           | NA            | NA         | NA     | 0.985   |



# References

[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE Conference on Computer Vision and Pattern Recognition, 2009, pp. 248–255.

[2] M. Everingham, L. Gool, C. K. Williams, J. Winn, and A. Zisserman, “The pascal visual object classes (voc) challenge,” Int. J. Comput. Vision, vol. 88, no. 2, p. 303–338, jun 2010. [Online]. Available: https://doi.org/10.1007/s11263-009-0275-4

[3] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, “Semantic understanding of scenes through the ade20k dataset,” 2018.

[4] X. Q. X. W. J. J. Hengshuang Zhao, Jianping Shi, “Pyramid scene parsing network,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[5] S. Garga, “UC HACK 20: Detect Wall from the Image and Change Its Colour or Apply Various Patterns,” GitHub, 2020. 

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” in Proceedings of 2016 IEEE Conference on Computer Vision and Pattern Recognition, ser. CVPR ’16. IEEE, Jun. 2016, pp. 770–778. [Online]. Available: http://ieeexplore.ieee.org/document/7780459

[7] M. Hamilton, Z. Zhang, B. Hariharan, N. Snavely, and W. T. Freeman, “Unsupervised semantic segmentation by distilling feature correspondences,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=SaKO6z6Hl0c

