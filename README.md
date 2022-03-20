# Domain generalisation for Object Detection

This repository contains all the codes needed to replicate the results in the paper '[Domain Generalisation for Object Detection'](https://arxiv.org/abs/2203.05294).  

We used the following datasets for our experiments. 

1. [Global Wheat Head Detection](https://zenodo.org/record/5092309#.YjeR1zynzJU) 
2. [Cityscapes](https://www.cityscapes-dataset.com/) 
3. [Sim10K](https://fcav.engin.umich.edu/projects/driving-in-the-matrix) 
4. [Berkely Deep Drive](https://bdd-data.berkeley.edu/) 

Our code expects the input data format to be in csv format and we provide necessary helper functions to convert the annotations in json, xml formats into csv format for all the datasets. But we expect the users to download the datasets from respective websites and follow the file structure mentioned below so that the code can directly access the datasets. The file requirements.txt has all the dependencies needed to run this code. 

Dataset details/Folder Structure!
Command for training
Command for testing
