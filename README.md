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

## Q4 - Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, size of images, are sizes standardized? do they need to be cropped? normalized? etc.

The dataset has 100 classes of butterfly or moth species. 

Number of images: There are 13639 images. Out of these, we use 500 for our test data and another 500 for validation purposes with each class having 5 images. This makes our train dataset equal to 12639 images. This should be sufficient to train the model and run tests to validate its results. 

Size of images: 224 x 224 - The size of all the images is standardized to this image resolution. We do not need to crop the images since the data is already standardized. Note: The images are colored thus it has three channels(Red, Green, and Blue).

The column values represent the name of the class, the filename, and the image.

The data has 3 channels so each channel will have values ranging from 0-255. Due to this, the data needs to be normalized. That way we can guarantee the accuracy of the results, having removed all inconsistencies from it. This ensures that the data has a similar distribution which enables a faster convergence while training the neural network.


## Q6 - How will you preprocess your data? You should explain this in your Readme.MD file and link your jupyter notebook to it. Your jupyter notebook should be uploaded to your repo.

For preprocessing our data we will perform certain filtration on our dataset.

The gaussian filter should be applied to the dataset to blur out the images since every image might have random noises. These random noises could be because different devices have different lenses with which these images were captured. All those lenses have different characteristics that can result in varying types of image noise (the artifacting that is present in the image, generated from the device used to get the image and not present at the original source) disrupting the image. Using a gaussian filter to blur the images will remove these artifacts and only preserve the most important features.
 
After the gaussian filter, we add the laplace filter that helps out in highlighting the edges. This helps us remove inconsistencies and inaccuracies from the result of the machine learning model, making it more accurate. 

We then perform rotations/transformations on the data set to help the model improve the neural network with each image being viewed from a different angle and zoom. This is to improve the usability of our model in a real-life scenario as there could be images that may be clicked at an angle that is not perfectly straight. By doing these steps, we can allow the model to be more applicable in real-life scenarios where we won’t always have the ideal data.


# Preprocessing & First Model building and evaluation Milestone
The model seems to be underfitting when comparing the loss and val_loss during the training of the model.

<img width="838" alt="Screen Shot 2022-11-27 at 3 54 30 PM" src="https://user-images.githubusercontent.com/63729973/204166597-8e74b7a0-7087-4e6c-8ebb-b46baa3fa048.png">

As can be seen from the above image, the loss (loss from the training dataset) is bigger than the val_loss (loss from the validation dataset). Moreover, the model had a lower accuracy on the training dataset than on the validation and testing datasets. It shows that there are more features to be captured from the training data, and so our model is underfitting on the dataset. As a result, the model seems to be inefficient in classifying the butterflies and moths.

<img width="429" alt="Screen Shot 2022-11-27 at 3 55 47 PM" src="https://user-images.githubusercontent.com/63729973/204166646-e2d7b9f2-3913-4024-8861-5246e92ff8f7.png">
 
To improve this model, extra layers can be added to increase the amount of information the model captures from the training dataset, the model can be run for more epochs, and image augmentation can be toned down.


# Introduction - 

Introduction of your project. Why was it chosen? Why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?

We started with the idea that our model should be based on Computer Vision. It was a concept that intrigued us from the start and something we didn't have much experience with. We believed that it would be a great learning opportunity to improve our foundation of an important skill that would help us in the future. While searching for an ideal dataset we came across this butterfly dataset. The dataset is interesting since butterflies serve a greater purpose to this world and are on the verge of extinction. Due to which, it is crucial to identify these butterflies for the safety of our ecosystem. For that reason, we have decided to develop a machine-learning algorithm to help us easily identify these species. Further, we concluded that the dataset would be compatible with our planned machine learning model. It would be ideal as it consists of over 13000 images of 100 unique classes of butterflies. Our approach would involve multiple parts such as image enhancement, image segmentation, region analysis, and more computer vision and deep learning techniques. We believe that the combination of this dataset and our implementation of machine learning techniques will make a great machine learning model that would be usable in real world scenarios, similar to other models such as YOLO which has gained traction in the data science field. As we see the results in the end, we get an accuracy of over 70 percent in the finalized model. 


# Methods:

Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods

## Data Exploration:

```
BATCH_SIZE = 128
CHANNELS = 3
IMAGE_SIZE = 224
```	
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=AUO_mxTF1Bm4&line=3&uniqifier=1
``` 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import models, layers, preprocessing, Sequential
import tensorflow.keras
 
print("Training Data:\n---------------")
training_data = preprocessing.image_dataset_from_directory(
    "/content/butterfly-data/train", 
    batch_size=BATCH_SIZE, 
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True
)
 
print("\n\nTesting Data:\n---------------")
testing_data = preprocessing.image_dataset_from_directory(
    "/content/butterfly-data/test", 
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True
)
print("\n\nValidation Data:\n---------------")
validation_data = preprocessing.image_dataset_from_directory(
    "/content/butterfly-data/valid", 
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True
)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=n0nnVL-xo4Tm&line=8&uniqifier=1
``` 
classes = training_data.class_names
print(
    "Num classes: ", len(classes),
    "\nClass Names: ", classes
)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=n0nnVL-xo4Tm&line=8&uniqifier=1
``` 
from tensorflow.image import rgb_to_grayscale
import cv2
mean_array = []
#laplacian_array = []
for images, labels in training_data.as_numpy_iterator():
    for image in images:
        mean_array.append(np.asarray(rgb_to_grayscale(image)).reshape(-1, 1).mean())
        #laplacian_array.append(np.asarray(np.random.laplace(image)).reshape(-1, 1).var())
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=qtI1vBvx6NHP&line=4&uniqifier=1
``` 
len(mean_array)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=XiKVsOkjBaw3&line=1&uniqifier=1
``` 
mean_array[:5]
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=oTKVlbVTB7qr&line=1&uniqifier=1

``` 
mean_array = np.asarray(mean_array)
median = np.median(mean_array)
std_dev = np.std(mean_array)
min = mean_array.min()
max = mean_array.max()
mean = np.mean(mean_array)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=nIo_VlNnCAYl&line=7&uniqifier=1
``` 
print(
    "\nImage Grayscale Values:",
    "\n----------", 
    "\n\tMean: ", mean,
    "\n\tMedian: ", median,
    "\n\tStandard Deviation: ", std_dev,
    "\n\tMinimum: ", min,
    "\n\tMaximum: ", max
)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=PiBiGk1kCvxp&line=6&uniqifier=1
``` 
plt.figure(figsize=(30,30))
for images, labels in training_data.take(1):
    #print(labels)
    for j in range(100):
        axis = plt.subplot(10, 10, j + 1)
        plt.imshow(np.asarray(images[j]).astype("uint8"))
        plt.title(classes[labels[j]])
        plt.axis("off")
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=vNpd5Gbt4unq&line=7&uniqifier=1

As seen above, the dataset has 100 classes of butterfly or moth species.
Number of images: There are 13639 images. Out of these, we can see that 500 are for our test data and another 500 are for our validation purposes with each class having 5 images. This makes our train dataset equal to 12639 images.  
Size of images: 224 x 224 - The size of all the images is standardized to this image resolution. Note: The images are colored thus it has three channels(Red, Green, and Blue).
 
The column values represent the name of the class, the filename, and the image.
 
The data has 3 channels so each channel will have values ranging from 0-255. Since some classes might have brighter images than others, we are converting the images to grayscale and finding the average pixel value of each image to check for variance in brightness between those images as CNNs tend to be biased toward classes with higher values.

## Preprocessing
### Model 1
``` 
data_preprocessing_1 = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.GaussianNoise(0.1),
    #tfio.experimental.filter.laplacian()
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomCrop(IMAGE_SIZE, IMAGE_SIZE)
])
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=FENvgs2hHr5t&line=5&uniqifier=1
``` 
training_data = training_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
testing_data = testing_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=wCKXgNFSlCyH&line=1&uniqifier=1

### Model 2	
```
data_preprocessing_2 = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.GaussianNoise(0.05),
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomCrop(IMAGE_SIZE, IMAGE_SIZE)
])
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=MDwITcrX3a-t&line=1&uniqifier=1

```
training_data = training_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
testing_data = testing_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=EjTZZ9XWFURF&line=3&uniqifier=1

For preprocessing our data (further details in discussion):
 
We performed rescaling on our images where the images are converted to a scale between 0 and 1
	
Then, Gaussian noise is applied to the dataset
 
Then random flips are performed on the image where the images are flipped either horizontally or vertically through RandomFlip. Then we do rotation on the image where the images are rotated by a boundary value. Then we moved on with cropping our image to have certain transformations. 
Then we do lazy reading on the training dataset using prefetch.


## Model 1
```
INPUT_SHAPE = (128, IMAGE_SIZE, IMAGE_SIZE, 3)
model_1 = models.Sequential([
    data_preprocessing_1,
    layers.Conv2D(128, (3,3), activation = 'relu', input_shape = INPUT_SHAPE),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(16, (3,3), activation = 'relu'),
    layers.Flatten(),
    layers.Dense(units = 128, activation = 'relu'),
    #layers.Dense(units = 500, activation = 'relu'),
    layers.Dense(units = 100, activation = 'sigmoid')
])
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=bQPyhIhkSDRM&line=6&uniqifier=1
```
model_1.build(input_shape = INPUT_SHAPE)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=cDmVC1RZiE8B&line=1&uniqifier=1
```
model_1.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=EumIabMJlec8&line=4&uniqifier=1
```
history_1 = model_1.fit(training_data, batch_size = BATCH_SIZE, validation_data = validation_data, epochs = 10)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=bityvc1WfQUZ&line=1&uniqifier=1

We decided to build the first model with 3 2D Convolutional layers having relu activation function, applied Max Pooling on our pre-processed data and flattened the model before applying Dense with relu and sigmoid. Then we build the model by compiling it with Adam optimizer, SparseCategoricalCrossentropy to compute the losses, and use the accuracy metric to evaluate our model.

## Model 2

```
INPUT_SHAPE = (128, IMAGE_SIZE, IMAGE_SIZE, 3)
model_2 = models.Sequential([
    data_preprocessing_2,
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape = INPUT_SHAPE),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    layers.Dropout(0.1),
    layers.Dense(units = 500, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(units = 100, activation = 'sigmoid')
])

model_2.build(input_shape = INPUT_SHAPE)

model_2.compile(
    optimizer = tf.keras.optimizers.Adam(), 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

history_2 = model_2.fit(training_data, batch_size = BATCH_SIZE, validation_data = validation_data, epochs = 40)
```

https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=f8aYHDWWNbg2&line=6&uniqifier=1

In the second model, we applied 2 2D Convolutional layers having relu activation function, applied Max Pooling on our pre-processed data and flattened the model before applying Dense with relu. We also use an Average Pooling layer to get the average value of the filter size. Then we applied Dropout to our model to set the input units to 0 with a frequency of 0.1 before finally applying Dense with the sigmoid activation function. Then we build the model by compiling it with Adam optimizer, SparseCategoricalCrossentropy to compute the losses, and use the accuracy metric to evaluate our model.

## getTestingAccuracy 

### Model 1
```
def getTestingAccuracy(model, testing_data, history):
    # getting the predictions from the model
    predicted_classes = model.predict(testing_data).argmax(axis=1)
    
    # converting the numerical predictions to the class names for ease of life
    predicted_class_names = [classes[x] for x in predicted_classes]
    
    # getting the actual class names from the testing dataset
    actual_classes = []
    for images, labels in testing_data:
        for label in labels:
            actual_classes.append(classes[label])

    # put the actual labels and the predicted data into a single dataframe to evaluate the results
    testing_df = pd.DataFrame()
    testing_df['actual'] = actual_classes
    testing_df['predictions'] = predicted_class_names

    # defining a function to provide to the lamba function to check for accuracy
    def isCorrect(actual, target):
        return actual == target

    # creating a column and storing if the actual labels and the predicted labels match
    testing_df['correct'] = testing_df.apply(lambda x: isCorrect(x.actual, x.predictions), axis=1)

    plt.figure(figsize=(20, 10))
    # plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    
    # plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    

    plt.show()
                                             
    # returning the accuracy
    print("Accuracy on testing data:", (sum(testing_df.correct)/len(predicted_class_names))*100, "%")                                          
    return (sum(testing_df.correct)/len(predicted_class_names))
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=9Q0XrF4A1j44&line=19&uniqifier=1
```
getTestingAccuracy(model_1, testing_data, history_1)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=MDwITcrX3a-t&line=1&uniqifier=1

### Model 2

```
getTestingAccuracy(model_2, testing_data, history_2)
```
https://colab.research.google.com/github/AdityaaRavi/Butterfly-cnn/blob/model-2/Butteryfly_cnn_main.ipynb#scrollTo=f5ORbUw4NpVN&line=1&uniqifier=1
This was a helper function we created to get the training accuracy of our models. 
	
# Results - 

## Data Exploration:

In the dataset, we have three different distributions - 

Training Data -  Training data is a dataset that is used to fit the ML model.
Testing Data - Testing data is a dataset that is used to evaluate the final ML model on the training data.
Validation Data - Validation data is a dataset that is used to evaluate the ML model on the training data while tuning model hyperparameters(parameters that can be adjusted to obtain the best performing model)

```
Training Data:
---------------
Found 12639 files belonging to 100 classes.


Testing Data:
---------------
Found 500 files belonging to 100 classes.


Validation Data:
---------------
Found 500 files belonging to 100 classes.
```

Number of Classes - 100 

```
Num classes:  100 
Class Names:  ['ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ARCIGERA FLOWER MOTH', 'ATALA', 'ATLAS MOTH', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BANDED TIGER MOTH', 'BECKERS WHITE', 'BIRD CHERRY ERMINE MOTH', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROOKES BIRDWING', 'BROWN ARGUS', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHALK HILL BLUE', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CINNABAR MOTH', 'CLEARWING MOTH', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMET MOTH', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'EMPEROR GUM MOTH', 'GARDEN TIGER MOTH', 'GIANT LEOPARD MOTH', 'GLITTERING SAPPHIRE', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREEN HAIRSTREAK', 'GREY HAIRSTREAK', 'HERCULES MOTH', 'HUMMING BIRD HAWK MOTH', 'INDRA SWALLOW', 'IO MOTH', 'Iphiclus sister', 'JULIA', 'LARGE MARBLE', 'LUNA MOTH', 'MADAGASCAN SUNSET MOTH', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'OLEANDER HAWK MOTH', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POLYPHEMUS MOTH', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'ROSY MAPLE MOTH', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SIXSPOT BURNET MOTH', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WHITE LINED SPHINX MOTH', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING']
```

After the RGB values to grayscale conversion, we got the mean array. The length of which matches the length of the training data, as expected.
```
12639
```
To verify that the values in the array were converted to grayscale - 
```
[91.20976, 108.2275, 149.88997, 94.101555, 116.68662]
```
The values we could use for further training and computation from the mean array - 
```
Image Grayscale Values: 
---------- 
	Mean:  116.32708 
	Median:  114.14225 
	Standard Deviation:  29.468502 
	Minimum:  13.593116 
	Maximum:  242.44257
```
Model 1 and 2 - 

The accuracy of our first model 

![download](https://user-images.githubusercontent.com/93968740/206358206-6b7679bd-b56a-4178-83e9-88923a2765db.png)

```
Accuracy on testing data: 58.599999999999994 %
0.586
```

The accuracy of our second model

![download](https://user-images.githubusercontent.com/93968740/206358251-ab0bc005-299d-4f64-8ea7-df3231167a1e.png)

```
Accuracy on testing data: 76.4 %
0.764
```

# Discussion - 

## Data Exploration - 

Taking the values from the testing and training array, we convert these values into the mean array. We convert all values to grayscale since butterflies tend to be more colorful, having a high RGB value as compared to moths. The neural network will favor higher values of RGB/brighter images compared to the lower values of RGB/darker images. This can be seen in butterflies which tend to be brighter than moths, so the network would prefer butterfly images over moth images. High RGB → higher grayscale value. Similarly the other way around. To avoid skewing the ML model, we convert them to grayscale, and to make our model run efficiently and effectively we add them up and average them out. We then calculate important values such as mean, median etc. 
These values are the important values we computed, also present in the results above. As we can see, the range of color values is quite wide--with the average grayscale values of images ranging anywhere from 13.60 to 242.44. Furthermore, the standard deviation of 29.47 shows that there is a significant difference in the "colorfulness" between our darkest and brightest classes. Therefore, we can conclude that we need to rescale our image values to be between 0 and 1 to reduce the bias that color values would cause in our neural network.


## Preprocessing - 
The preprocessing methods we used were done to make the model better for the actual scenarios. The images wouldn’t always be in the ideal shape and size, being an image to the spec that is straight. For this, we do rescaling, flipping, rotations, and cropping. This would help us make the model better and more accurate for real world scenarios. 

Another thing we do is using the GaussianNoise filter to reduce overfitting. Every image might have random noises. These random noises could be because different devices have different lenses with which these images were captured. All those lenses have different characteristics that can result in varying types of image noise (the artifacting that is present in the image, generated from the device used to get the image and not present at the original source) disrupting the image. By adding gaussian noise to our training images, we can prevent overfitting and thus train the model to work better in a real world scenario. 

Our training, testing, and validation datasets are pretty big. This could be an issue if we load it into RAM. The most clear problem - loading too much data and not having enough RAM for the same. To counteract this, we do something called Lazy Loading. To put it simply, Lazy Loading takes smaller chunks of data from the large dataset and loads those smaller chunks into the RAM. Once one particular chunk is processed, it is removed from the RAM and then another chunk is loaded into the RAM, continuing till all the data is processed. This helps us to save RAM and makes the process more efficient, while avoiding any insufficient memory issues. 


## Model 1- 

In the pre-processing step, we applied gaussian noise of 0.1 std deviation to mitigate overfitting before flipping, rotating and cropping the images. While training the data, we decided to use a CNN model with 2D Convolutional layers going from high to low where we apply relu function as it was the most widely used activation function for CNN models. We apply MaxPooling to calculate the maximum value for each channel of the input along with the 2D Convolutional layers before flattening the dataset and applying Dense to create a dense NN layer. The model is then built and compiled using the Adam optimizer which is appropriate for problems with very noisy and/or sparse gradients while being ideal for very large datasets such as ours, SparseCategoricalCrossentropy to compute the loss between the labels and the predictions, and the accuracy is the metric to be evaluated by the model during training, testing and validation. Then we calculate the accuracy and loss on the training and validation data for 10 total epochs, which observes our average total accuracy of our first model to be 59.6% (0.596). 


## Model 2 - 

For model 2, we continue on with the same preprocessing steps of flipping, rotations, cropping, and gaussian noise of 0.05 std deviation. The steps done here were similar to what we did for model 1 but with the application of the dropout layer. This is done to prevent overfitting. It does so by changing the input units to 0 on a random basis at the rate provided (0.1 and 0.2 here), while scaling up the input units that were not changed to 0 by 1/(1-rate) to make the sum over all the inputs change. Alongside that, we changed the Kernel sizes and number of layers, where we had 3 layers in the first model that went from high to low Kernel sizes and 2 layers in the second one that went from low to high Kernel sizes. With the final layer having a smaller kernel size than the maximum of the first model, our model drops fewer important details that a larger kernel would, making model 2 more accurate and even faster. Finally, we tested it for more epochs (40 compared to 10) than the model and ended up with a higher accuracy than our previous model (72.6% for model 2 vs 59.6% for model 1). We believe that the higher number of epochs and the further processing of the dropout layer resulted in the higher accuracy of this model. 

# Conclusion - 

Thoughts Dump:

Our data exploration steps greatly improved our learning and understanding of the dataset. By dividing out the data into usable datasets, we made the computation cleaner for our model. Along with that, computing the useful values such as the mean, median, standard deviation etc, helped us in further steps and also for making sure that the data wasn’t imbalanced since it would need over/under sampling otherwise. 

Our preprocessing steps made it so that we could use the model in a real world scenario. While this is not intended to be a production model, using these preprocessing steps helped us improve its usefulness in the real world. 

Though our first model and second model were constructed similarly, the addition of a dropout layer in our second model increased the accuracy by 13%. Our layers that we set the model upon in model 1 gave a strong foundation for the second model where we changed the number of layers and kernel size, added the dropout layer, and even more epochs, leading to the accuracy increase that we reported above. 

# Possibilities for the future - 

In the future, we could think of adding a laplacian filter. The laplacian filter helps in providing edge enhancements to the image. For instance, we might have animage of two different classes, let’s say Viceroy and Monarch. Both of them look very similar to each other. While our model would be able to distinguish between the two in it’s current state, based on how it was trained, if we get an image of the two that has a bokeh effect on it for example which may have blurred out the edges of the same, that would be helped by the laplacian filter which would make the image clearer and allow our model to detect the image better. 

We would like to train our model on even more images of different classes. However, for the purposes of this version 1.0 of the model, 100 classes is a good sample size. 

Lastly, adding more layers would be our final improvement attempt. It would lead to a potential improvement in the accuracy of one of, if not both of the models. The accuracy of the current model is great, but there are ways to improve. 


# Collaboration Statement - 

Everyone: Helped with coding and debugging 

Adityaa Ravi (Leader): Leader of the group who came up with the idea for the project and the plan to execute the idea

Aadarsh Subramanian: Helped with documentation and added ideas to data exploration and preprocessing

Akhileshwar Shriram: Contacted everyone and organized meetings and helped with documentation in the introduction and methods section

Lakshitha Anand: She set up the deadlines and helped in documentation in discussion and methods 

Japman Singh Kohli: Helped specifically in model 1 and model 2 and documentation of introduction 

Yash Inani: Helped with documentation of results, discussion and conclusion and in documentation during data exploration and preprocessing
