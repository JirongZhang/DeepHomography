### Introduction
![](./images/teaser.jpg)

Homography estimation is a basic image alignment method in many applications. It is usually done by extracting and matching sparse feature points, which are error-prone in low-light and low-texture images. On the other hand, previous deep homography approaches use either synthetic images for supervised learning or aerial images for unsupervised learning, both ignoring the importance of handling depth disparities and moving objects in real world applications. To overcome these problems, in this work we propose an unsupervised deep homography method with a new architecture design. In the spirit of the RANSAC procedure in traditional methods, we specifically learn an outlier mask to only select reliable regions for homography estimation. We calculate loss with respect to our learned deep features instead of directly comparing image content as did previously. To achieve the unsupervised training, we also formulate a novel triplet loss customized for our network. We valid our method by conducting comprehensive comparisons on a new dataset that covers a wide range of scenes with varying degrees of difficulties for the task. Experimental results reveal that our method outperforms the state-of-the-art including deep solutions and feature-based solutions.

![](./images/model.png)

This repository contains the required models and scripts for the paper  ["Content-Aware Unsupervised Deep Homography Estimation"](https://arxiv.org/pdf/1909.05983.pdf).



### Results of Quantitative comparison
| Comparison with previous DNN based methods|
|:-------------------------:|
|![](./images/err_1.jpg)  |

| Comparison with feature-based methods |
| :-----------------------------------: |
|        ![](./images/fig7.jpg)         |

### Results of Qualitative comparison
Input |Ours |SIFT+RANSAC |Unsupervised[1] |Supervised[2]
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](images/images/0_input.gif)  |  ![](images/images/0_Ours.gif)  |  ![](images/images/0_SIFT_RANSAC.gif)  |  ![](images/images/0_Unsupervised.gif)  |  ![](images/images/0_Supervised.gif)
![](images/images/17_input.gif)  |  ![](images/images/17_Ours.gif)  |  ![](images/images/17_SIFT_RANSAC.gif)  |  ![](images/images/17_Unsupervised.gif)  |  ![](images/images/17_Supervised.gif)
![](images/images/12_input.gif)  |  ![](images/images/12_Ours.gif)  |  ![](images/images/12_SIFT_RANSAC.gif)  |  ![](images/images/12_Unsupervised.gif)  |  ![](images/images/12_Supervised.gif)
![](images/images/14_input.gif)  |  ![](images/images/14_Ours.gif)  |  ![](images/images/14_SIFT_RANSAC.gif)  |  ![](images/images/14_Unsupervised.gif)  |  ![](images/images/14_Supervised.gif)
![](images/images/5_input.gif)  |  ![](images/images/5_Ours.gif)  |  ![](images/images/5_SIFT_RANSAC.gif)  |  ![](images/images/5_Unsupervised.gif)  |  ![](images/images/5_Supervised.gif)
![](images/images/1_input.gif)  |  ![](images/images/1_Ours.gif)  |  ![](images/images/1_SIFT_RANSAC.gif)  |  ![](images/images/1_Unsupervised.gif)  |  ![](images/images/1_Supervised.gif)
![](images/images/13_input.gif)  |  ![](images/images/13_Ours.gif)  |  ![](images/images/13_SIFT_RANSAC.gif)  |  ![](images/images/13_Unsupervised.gif)  |  ![](images/images/13_Supervised.gif)
![](images/images/2_input.gif)  |  ![](images/images/2_Ours.gif)  |  ![](images/images/2_SIFT_RANSAC.gif)  |  ![](images/images/2_Unsupervised.gif)  |  ![](images/images/2_Supervised.gif)
![](images/images/19_input.gif)  |  ![](images/images/19_Ours.gif)  |  ![](images/images/19_SIFT_RANSAC.gif)  |  ![](images/images/19_Unsupervised.gif)  |  ![](images/images/19_Supervised.gif)
![](images/images/8_input.gif)  |  ![](images/images/8_Ours.gif)  |  ![](images/images/8_SIFT_RANSAC.gif)  |  ![](images/images/8_Unsupervised.gif)  |  ![](images/images/8_Supervised.gif)
![](images/images/3_input.gif)  |  ![](images/images/3_Ours.gif)  |  ![](images/images/3_SIFT_RANSAC.gif)  |  ![](images/images/3_Unsupervised.gif)  |  ![](images/images/3_Supervised.gif)
![](images/images/4_input.gif)  |  ![](images/images/4_Ours.gif)  |  ![](images/images/4_SIFT_RANSAC.gif)  |  ![](images/images/4_Unsupervised.gif)  |  ![](images/images/4_Supervised.gif)
![](images/images/11_input.gif)  |  ![](images/images/11_Ours.gif)  |  ![](images/images/11_SIFT_RANSAC.gif)  |  ![](images/images/11_Unsupervised.gif)  |  ![](images/images/11_Supervised.gif)
![](images/images/6_input.gif)  |  ![](images/images/6_Ours.gif)  |  ![](images/images/6_SIFT_RANSAC.gif)  |  ![](images/images/6_Unsupervised.gif)  |  ![](images/images/6_Supervised.gif)
![](images/images/9_input.gif)  |  ![](images/images/9_Ours.gif)  |  ![](images/images/9_SIFT_RANSAC.gif)  |  ![](images/images/9_Unsupervised.gif)  |  ![](images/images/9_Supervised.gif)
![](images/images/16_input.gif)  |  ![](images/images/16_Ours.gif)  |  ![](images/images/16_SIFT_RANSAC.gif)  |  ![](images/images/16_Unsupervised.gif)  |  ![](images/images/16_Supervised.gif)
![](images/images/7_input.gif)  |  ![](images/images/7_Ours.gif)  |  ![](images/images/7_SIFT_RANSAC.gif)  |  ![](images/images/7_Unsupervised.gif)  |  ![](images/images/7_Supervised.gif)
![](images/images/10_input.gif)  |  ![](images/images/10_Ours.gif)  |  ![](images/images/10_SIFT_RANSAC.gif)  |  ![](images/images/10_Unsupervised.gif)  |  ![](images/images/10_Supervised.gif)
