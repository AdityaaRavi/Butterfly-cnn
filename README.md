# Butterfly-cnn
Final project for the ECS 171 (FQ 22) class at UC Davis.

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/main/Butteryfly_cnn_main.ipynb)
        

# Team Members
| Name                | github username
| ----                | ---
| Adityaa Ravi        | adityaaravi
| Yash Inani          | yinani24
| Aadarsh Subramaniam | aadsub
| Akhileshwar Shriram | AkhilTheBoss 
| Japman Singh Kohli  | buzzshocker
| Lakshitha Anand     | Laks

# Project Abstract
Butterflies are elegant creatures that magnify the beauty of our environment and help in pollination. They serve a greater purpose in this world but many are going endangered and some are even on the verge of extinction. So, it is crucial to identify these butterflies for the safety of our ecosystem. For that reason, we have decided to develop a machine-learning algorithm to help us easily identify these species. Our approach would involve multiple parts such as image enhancement, image segmentation, region analysis, and more computer vision and deep learning techniques. Our dataset includes over 800 images of 10 unique classes of butterflies. ([Data-Set](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)) We want to train a Convolutional Neural Network to classify the butterflies based on their features. Furthermore, it is possible to deploy cameras to recreate a butterfly object detection model like YOLO.

# Data Exploration Milestone 

##Q4 - Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, size of images, are sizes standardized? do they need to be cropped? normalized? etc.

The dataset has 100 classes of butterfly or moth species. 

Number of images: There are 13639 images. Out of these, we use 500 for our test data and another 500 for validation purposes with each class having 5 images. This makes our train dataset equal to 12639 images. This should be sufficient to train the model and run tests to validate its results. 

Size of images: 224 x 224 - The size of all the images is standardized to this image resolution. We do not need to crop the images since the data is already standardized. Note: The images are colored thus it has three channels(Red, Green, and Blue).

The column values represent the name of the class, the filename, and the image.

The data has 3 channels so each channel will have values ranging from 0-255. Due to this, the data needs to be normalized. That way we can guarantee the accuracy of the results, having removed all inconsistencies from it. This ensures that the data has a similar distribution which enables a faster convergence while training the neural network.


##Q6 - How will you preprocess your data? You should explain this in your Readme.MD file and link your jupyter notebook to it. Your jupyter notebook should be uploaded to your repo.

For preprocessing our data we will perform certain filtration on our dataset.

The gaussian filter should be applied to the dataset to blur out the images since every image might have random noises. These random noises could be because different devices have different lenses with which these images were captured. All those lenses have different characteristics that can result in varying types of image noise (the artifacting that is present in the image, generated from the device used to get the image and not present at the original source) disrupting the image. Using a gaussian filter to blur the images will remove these artifacts and only preserve the most important features.
 
After the gaussian filter, we add the laplace filter that helps out in highlighting the edges. This helps us remove inconsistencies and inaccuracies from the result of the machine learning model, making it more accurate. 

We then perform rotations/transformations on the data set to help the model improve the neural network with each image being viewed from a different angle and zoom. This is to improve the usability of our model in a real-life scenario as there could be images that may be clicked at an angle that is not perfectly straight. By doing these steps, we can allow the model to be more applicable in real-life scenarios where we won’t always have the ideal data.



