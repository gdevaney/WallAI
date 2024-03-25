---
layout: default
title: WallAI - CS6473 Computer Vision Project
description: Image Segmentation and Recoloration
---


# Introduction/Problem Definition
ProjectColor is an application made by Home Depot that allows users to visualize different paint colors after they take a picture of their room. The Home Depot app has some difficulty detecting edges and assigning appropriate pixels to what it defines as “walls”. The application also has difficulty detecting light exposure and applying appropriate color transformations (the example below likely does not translate to what the color would actually look like on the wall). We will attempt to solve these problems using Machine Learning and Computer Vision techniques.

# Related Works
- “Home Staging Using Machine Learning Techniques” by Marti Grau Gasulla
  - This research paper discussed using style transfer and other deep learning techniques to modify the style of different bathroom images while retaining objects in the room.
- “SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation” by Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla
  - This research paper details a deep, fully convolutional neural network called SegNet that performs pixel-wise segmentation of images.
- “Computer vision based room interior design” by Nasir Ahmad
  - This paper is the first attempt at using computer vision for interior design. The paper approaches the problem through image segmentation, color assignment, and post-processing.
- “A transfer learning approach for indoor object identification” by Mouna Afif
  - This paper discusses using transfer learning techniques and deep convolutional neural networks for detecting with a big performance three categories of indoor objects (door, stair, and sign). This paper may aid us in how indoor objects/furniture might be different from regular object recognition, and help us with our project.
- “What’s in my Room? Object Recognition on Indoor Panoramic Images” by Julia Guerrero-Viu
  - This research paper proposes deep learning-based methods for conducting object recognition in images that may contain distortions from 360-degree panoramic images.
- “Fast R-CNN” by Ross Girshick
  - This research paper proposes a Fast Region-based Convolutional Network method for object detection. While CNN has been known to perform well in object detection, Fast R-CNN’s speed and accuracy stand out from previous methods like R-CNN.

# Methods/Approach
The following sections describe each method and our particular implementation.

# Experiments/Results

<img src="{{site.baseurl}}/assets/images/bedroom1.png" width="48%"/>

# What's Next

# Team Member Contributions

Make a checklist
