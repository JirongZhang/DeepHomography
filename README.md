
## Code and models of [paper](https://arxiv.org/pdf/1909.05983.pdf). "Content-Aware Unsupervised Deep Homography Estimation"
 By Jirong Zhang, Chuan Wang, Shuaicheng Liu, Lanpeng Jia, Jue Wang, Ji Zhou




### Update
- **2019.11.22**: We will upload the code&model after this paper has been received.
- **2019.9.12**: Repository for ["Content-Aware Unsupervised Deep Homography Estimation"](https://arxiv.org/pdf/1909.05983.pdf).



### Introduction
This repository will contains all the required models and scripts for the paper  ["Content-Aware Unsupervised Deep Homography Estimation"](https://arxiv.org/pdf/1909.05983.pdf).

![](./images/model.png)


In this work, we propose an unsupervised approach with a new architecture for content awareness learning. In particular, we learn a content mask to reject outlier regions to mimic the traditional RANSAC procedure. To realize this, we introduce a novel triple loss for the effective optimization. Moreover, instead of comparing intensity values directly, we calculate loss with respect to our learned deep features, which is more effective. In addition, we introduce a comprehensive homography dataset, within which the testing set contains manually labeled ground-truth point matches for the purpose of quantitative comparison. The dataset consisted of 5 categories, including regular(RE), low-texture(LT), low-light(LL), small foreground(SF), and large-foreground(LF) of scenes. We show the advantages of our method over both traditional feature-based approaches and previous deep-based solutions. 



### Results of Quantitative comparison
 Comparison with previous DNN based methods|  Comparison with feature-based methods
:-------------------------:|:-------------------------:
![](./images/err1.png)  |  ![](./images/err2.png)



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


### TODO
1. Data
2. Code
3. Requirements
4. Usage
5. Training


### License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{zhang2019content,
  title={Content-Aware Unsupervised Deep Homography Estimation},
  author={Zhang, Jirong and Wang, Chuan and Liu, Shuaicheng and Jia, Lanpeng and Wang, Jue and Zhou, Ji},
  journal={arXiv preprint arXiv:1909.05983},
  year={2019}
}
```

### References
  [1] T. Nguyen, S. W. Chen, S. S. Shivakumar, C. J. Taylor, and V. Kumar. Unsupervised deep homography: A fast and robust homography estimation model. IEEE Robotics and Automation Letters, 3(3):2346–2353, 2018  
  [2] D. DeTone, T. Malisiewicz, and A. Rabinovich. Deep image homography estimation. arXiv preprint arXiv:1606.03798, 2016
  
### Contact

  Questions can be left as issues in the repository. We will be happy to answer them.
