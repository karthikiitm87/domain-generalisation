# Domain generalisation for Object Detection

This repository contains all the codes needed to replicate the results in the paper '[Domain Generalisation for Object Detection'](https://arxiv.org/abs/2203.05294).  

We used the following datasets for our experiments. 

1. [Global Wheat Head Detection](GWHD)(https://zenodo.org/record/5092309#.YjeR1zynzJU) 
2. [Cityscapes](https://www.cityscapes-dataset.com/) 
3. [Sim10K](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) 
4. [Berkely Deep Drive 100K](BDD100K)(https://bdd-data.berkeley.edu/) 

Our code expects the input data format to be in csv format and we provide necessary helper functions to convert the annotations from json (Cityscapes, BDD100K), xml (Sim10K) formats into csv format for all the datasets. But we expect the users to download the datasets from respective websites and follow the file structure mentioned below so that the code can directly access the datasets. The file requirements.txt has all the dependencies needed to run this code. 


# Directory structure

We recommend everyone who would like to use this code follow this directory structure to run with fewer commands which we recommend in the next. 

# Using Faster-RCNN detector

Even though we do not restrict our approach to improve the generalisation to a specific detector, we have currently used Faster-RCNN as an example detector to demonstrate our approach. In future, we will include the codes using other popular detectors as well. In this code, we use torchvision's Faster-RCNN detector with COCO pretrained weights for experimenting with Cityscapes, Sim10K, BDD100K while we obtained better accuracy for GWHD using ImageNet pretrained weights. 

Dataset details/Folder Structure!
Command for training
Command for testing
