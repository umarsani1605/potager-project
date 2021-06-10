# Start and Grow with Potager

## Clarification on 10 June 2021

We apologize to the Bangkit 2021 team, there are some accidentally technical issue on our repository. 

Yesterday, we have push all the files and source code to the repository before 23.59. We have checked it right, the Github has been updated. But, in the morning we checked the repository and all the commits pushed last night are gone. We can't identify the real cause, because the commit logs is also gone. We have thought, maybe one of us acciddentaly reset the repository to the early commit.

So we decided to push all files again, because we thought our work will be needed by Bangkit team in assessing and checking our project. 
Again, we apologize for updating the repository after the capstone project deadline.

## Android Development

The development steps of the android app are explained below:

1. The first step is, making a wireframe and prototype for UI design,
2. Implementing wireframe into the android development,
3. Focusing on creating camera and make it works, and in this deployment, there are many bugs and some activities & fragments are still under development and not yet perfect,
4. Creating a simple database and CRUD on home fragment for adding a new plant.

The steps above are some of the steps we took for the development of this potager application, there were several obstacles that we encountered when doing development, such as the connection between the frontend and the backend.

## Machine Learning

The image classification model are build using Tensorflow. The following are the steps to build the model:

1. Download or clone the datasets, the extract to the defined folder. We are using three dataset, such as PlatVillage, iBean, and Cucumber plant dataset,
2. Preprocessing all of the three datasets, by combine all the three dataset into one folder,
3. Then remove some unsuited class and rename some class to the same naming format,
4. Plot the number of images distribution, so that we know the proportion of each class,
5. Plot some sample images of each class, so that we know its shapes and dimensions,
6. Split the dataset into training set and validation set,
7. Define the ImageDataGenerator parameter, then pass the training and validation set folder into the corresponding ImageDataGenerator,
8. Build the model using convolutional neural network, then compile it using Adam optimizer and sparse categorical crossentropy loss function,
9. Train and validate the model using 50 epoch. We use early stop callback, so that the training process will stop if the model achieve the desired accuracy,
10. Plot the accuracy and  loss of the model, so that we know the performance and to know how to tune the model,
11. Test the model with unseen plant disease images,
12. Save and export the model using Keras model format, so that the model can be deployed to the Google Cloud Platform.
